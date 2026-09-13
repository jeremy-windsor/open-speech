"""Wyoming TTS handler — bridges Wyoming Synthesize events to Open Speech TTS pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

from src.config import settings
from src.tts.pipeline import float32_to_int16, encode_audio
from src.cache.tts_cache import TTSCache
from src.audio.postprocessing import process_tts_chunks
from src.pronunciation.dictionary import PronunciationDictionary

if TYPE_CHECKING:
    from src.tts.router import TTSRouter

logger = logging.getLogger(__name__)

# Wyoming standard: 16kHz, 16-bit, mono
WYOMING_RATE = 16000
WYOMING_WIDTH = 2
WYOMING_CHANNELS = 1

# Used only when a backend cannot report its native rate.
DEFAULT_TTS_SAMPLE_RATE = 24000

_tts_cache = TTSCache(settings.tts_cache_dir, settings.tts_cache_max_mb, settings.tts_cache_enabled)
_pronunciation_dict = PronunciationDictionary(settings.tts_pronunciation_dict or None)


def _resample_to_16k(audio: np.ndarray, source_rate: int = DEFAULT_TTS_SAMPLE_RATE) -> np.ndarray:
    """Resample audio from source_rate to 16kHz using linear interpolation."""
    if source_rate == WYOMING_RATE:
        return audio
    ratio = WYOMING_RATE / source_rate
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


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


async def handle_synthesize(
    text: str,
    voice: str | None,
    tts_router: TTSRouter,
    write_event: Callable[[Event], Awaitable[None]],
) -> None:
    """Synthesize text and stream audio chunks back via Wyoming events."""
    if not settings.tts_enabled:
        logger.warning("TTS disabled, cannot synthesize via Wyoming")
        return

    voice_id = voice or settings.tts_voice
    model_id = settings.tts_model
    source_rate = _sample_rate_for_model(tts_router, model_id)
    text = _pronunciation_dict.apply(text)

    cache_fmt = "pcm"
    cached_pcm = _tts_cache.get(text=text, voice=voice_id, speed=1.0, fmt=cache_fmt, model=model_id) if settings.tts_cache_enabled else None
    if cached_pcm:
        pcm = np.frombuffer(cached_pcm, dtype=np.int16)
        chunks = [pcm.astype(np.float32) / 32767.0]
    else:
        try:
            def _synthesize_chunks() -> list[np.ndarray]:
                raw_chunks = tts_router.synthesize(
                    text=text,
                    model=model_id,
                    voice=voice_id,
                )
                return list(
                    process_tts_chunks(
                        raw_chunks,
                        trim=settings.tts_trim_silence,
                        normalize=settings.tts_normalize_output,
                    )
                )

            chunks = await asyncio.to_thread(_synthesize_chunks)
            if settings.tts_cache_enabled:
                pcm_bytes = await asyncio.to_thread(
                    lambda: encode_audio(iter(chunks), fmt="pcm", sample_rate=source_rate)
                )
                await asyncio.to_thread(
                    lambda: _tts_cache.set(
                        text=text,
                        voice=voice_id,
                        speed=1.0,
                        fmt=cache_fmt,
                        model=model_id,
                        audio=pcm_bytes,
                    )
                )
        except Exception:
            logger.exception("Wyoming TTS synthesis failed")
            return

    if not chunks:
        return

    # Send AudioStart
    await write_event(
        AudioStart(
            rate=WYOMING_RATE,
            width=WYOMING_WIDTH,
            channels=WYOMING_CHANNELS,
        ).event()
    )

    # Send audio chunks
    for chunk_array in chunks:
        if hasattr(chunk_array, "numpy"):
            chunk_array = chunk_array.numpy()
        chunk_array = chunk_array.flatten()

        # Resample to 16kHz
        resampled = _resample_to_16k(chunk_array, source_rate)

        # Convert to int16 PCM
        pcm = float32_to_int16(resampled)

        await write_event(
            AudioChunk(
                rate=WYOMING_RATE,
                width=WYOMING_WIDTH,
                channels=WYOMING_CHANNELS,
                audio=pcm.tobytes(),
            ).event()
        )

    # Send AudioStop
    await write_event(AudioStop().event())
