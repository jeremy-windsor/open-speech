"""Wyoming TTS handler — bridges Wyoming Synthesize events to Open Speech TTS pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event

from src.config import settings
from src.tts.pipeline import float32_to_int16, encode_pcm
from src.cache.tts_cache import PCM_SAMPLE_RATE, get_shared_tts_cache
from src.audio.postprocessing import process_tts_chunks
from src.audio.resample import StreamingResampler, resample
from src.pronunciation.dictionary import PronunciationDictionary
from src.services.tts import run_on_provider, validate_tts_input_limit
from src.utils.redact import redact_paths

if TYPE_CHECKING:
    from src.tts.router import TTSRouter

logger = logging.getLogger(__name__)

# Wyoming standard: 16kHz, 16-bit, mono
WYOMING_RATE = 16000
WYOMING_WIDTH = 2
WYOMING_CHANNELS = 1

# Used only when a backend cannot report its native rate.
DEFAULT_TTS_SAMPLE_RATE = 24000

# The same instance as /v1/audio/speech: one lock orders every write and eviction.
_tts_cache = get_shared_tts_cache(settings.tts_cache_dir, settings.tts_cache_max_mb, settings.tts_cache_enabled)
_pronunciation_dict = PronunciationDictionary(settings.tts_pronunciation_dict or None)


class InputTooLongError(ValueError):
    """Text over TTS_MAX_INPUT_LENGTH or the model's own limit."""


def _sample_rate_for_model(tts_router: TTSRouter, model_id: str) -> int:
    sample_rate_for = getattr(tts_router, "sample_rate_for", None)
    if callable(sample_rate_for):
        try:
            sample_rate = sample_rate_for(model_id)
            if isinstance(sample_rate, (int, float)) and sample_rate > 0:
                return int(sample_rate)
        except Exception:
            logger.warning("Could not determine sample rate for %s", model_id, exc_info=True)
    return DEFAULT_TTS_SAMPLE_RATE


def _encode_cache_pcm(chunks: list[np.ndarray], source_rate: int) -> bytes:
    """Encode audio as cache PCM, resampled the way /v1/audio/speech does before caching."""
    return encode_pcm(resample(np.concatenate(chunks), source_rate, PCM_SAMPLE_RATE))


def _to_wyoming_pcm(chunks: list[np.ndarray], source_rate: int) -> list[bytes]:
    """Convert the utterance to Wyoming's 16 kHz PCM16, one piece per chunk with audio.

    One anti-aliased StreamingResampler runs across the chunks, so content above the
    8 kHz Nyquist limit is filtered out instead of folding back, and chunk boundaries
    leave no seams. The filter's tail goes out with the last piece.
    """
    resampler = StreamingResampler(source_rate, WYOMING_RATE)
    pieces: list[np.ndarray] = []
    for chunk in chunks:
        if hasattr(chunk, "numpy"):
            chunk = chunk.numpy()
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size:
            pieces.append(resampler.process(samples))
    if pieces:
        pieces[-1] = np.concatenate((pieces[-1], resampler.flush()))
    return [float32_to_int16(piece).tobytes() for piece in pieces if piece.size]


def _prepare(text: str, voice_id: str, model_id: str, tts_router: TTSRouter) -> tuple[str, dict, list[bytes] | None]:
    """Return the spoken text, its cache key, and the cached utterance as Wyoming PCM, if any.

    Blocking (dictionary, cache files): runs in a worker thread.
    """
    text = _pronunciation_dict.apply(text)
    if not text.strip():
        # Nothing is left to say, as when the dictionary removes every word: an utterance with no audio, since the
        # models refuse an empty text.
        return text, {}, []
    limit_error = validate_tts_input_limit(tts_router=tts_router, model_id=model_id, text=text)
    if limit_error:
        raise InputTooLongError(limit_error)

    # /v1/audio/speech shares these entries, so they follow its key and 24 kHz PCM format.
    cache_key = {
        "text": text,
        "voice": voice_id,
        "speed": 1.0,
        "fmt": "pcm",
        "model": model_id,
        "trim": settings.tts_trim_silence,
        "normalize": settings.tts_normalize_output,
    }
    cached_pcm = _tts_cache.get(**cache_key) if settings.tts_cache_enabled else None
    if cached_pcm:
        pcm = np.frombuffer(cached_pcm, dtype=np.int16)
        return text, cache_key, _to_wyoming_pcm([pcm.astype(np.float32) / 32767.0], PCM_SAMPLE_RATE)
    return text, cache_key, None


def _synthesize(text: str, voice_id: str, model_id: str, tts_router: TTSRouter, cache_key: dict) -> list[bytes]:
    """Synthesize the utterance as Wyoming PCM pieces and cache it; runs on the provider's thread."""
    source_rate = _sample_rate_for_model(tts_router, model_id)
    chunks = list(
        process_tts_chunks(
            tts_router.synthesize(text=text, model=model_id, voice=voice_id),
            trim=settings.tts_trim_silence,
            normalize=settings.tts_normalize_output,
        )
    )
    if settings.tts_cache_enabled and chunks:
        _tts_cache.set(**cache_key, audio=_encode_cache_pcm(chunks, source_rate))
    return _to_wyoming_pcm(chunks, source_rate)


async def handle_synthesize(
    text: str,
    voice: str | None,
    tts_router: TTSRouter,
    write_event: Callable[[Event], Awaitable[None]],
) -> bool:
    """Synthesize text and stream audio chunks back via Wyoming events.

    Every request ends in audio-stop or an error event; the client waits for one of them.
    Returns False after an error event, so the caller can end the connection.
    """
    if not settings.tts_enabled:
        logger.warning("TTS disabled, cannot synthesize via Wyoming")
        await write_event(Error(text="TTS is disabled").event())
        return False

    voice_id = voice or settings.tts_voice
    model_id = settings.tts_model
    try:
        # The unauthenticated port gets the limit /v1/audio/speech enforces.
        if len(text) > settings.tts_max_input_length:
            raise InputTooLongError(f"Input too long. Max: {settings.tts_max_input_length} characters")
        spoken, cache_key, pcm_pieces = await asyncio.to_thread(
            _prepare, text, voice_id, model_id, tts_router
        )
        if pcm_pieces is None:
            # Waits for the provider's synthesis lock on the provider's own thread, so a queue
            # behind a busy provider holds no thread of the shared default executor.
            pcm_pieces = await run_on_provider(
                tts_router, model_id, _synthesize, spoken, voice_id, model_id, tts_router, cache_key
            )
    except InputTooLongError as exc:
        await write_event(Error(text=str(exc), code="input_too_long").event())
        return False
    except Exception as exc:
        logger.exception("Wyoming TTS synthesis failed")
        await write_event(Error(text=f"TTS synthesis failed: {redact_paths(str(exc))}", code=type(exc).__name__).event())
        return False

    await write_event(
        AudioStart(
            rate=WYOMING_RATE,
            width=WYOMING_WIDTH,
            channels=WYOMING_CHANNELS,
        ).event()
    )
    for pcm in pcm_pieces:
        await write_event(
            AudioChunk(
                rate=WYOMING_RATE,
                width=WYOMING_WIDTH,
                channels=WYOMING_CHANNELS,
                audio=pcm,
            ).event()
        )
    await write_event(AudioStop().event())
    return True
