"""Wyoming STT handler — bridges Wyoming audio events to Open Speech STT pipeline."""

from __future__ import annotations

import asyncio
import io
import logging
import struct
from typing import TYPE_CHECKING

import numpy as np

from src.audio.preprocessing import preprocess_stt_audio
from src.config import settings
from src.services.stt import await_stt_fetch, run_stt_inference
from src.utils.audio import MAX_INPUT_SAMPLE_RATE, MIN_INPUT_SAMPLE_RATE
from src.vad.silero import get_vad_model

if TYPE_CHECKING:
    from src.router import BackendRouter

logger = logging.getLogger(__name__)

# Bytes per sample of the PCM a WAV header can describe (8-, 16-, 24- and 32-bit).
PCM_SAMPLE_WIDTHS = (1, 2, 3, 4)


def audio_format_error(rate: object, width: object, channels: object) -> str | None:
    """Return why a client's PCM format cannot be transcribed, or None if it can.

    The decoder resamples to 16 kHz, so a rate far below that multiplies the work
    (200 samples at 1 Hz decode to 3.2M); HTTP uploads get the same bounds.
    """
    if not (isinstance(rate, int) and MIN_INPUT_SAMPLE_RATE <= rate <= MAX_INPUT_SAMPLE_RATE):
        return (
            f"Unsupported sample rate {rate} Hz; audio must be "
            f"{MIN_INPUT_SAMPLE_RATE}-{MAX_INPUT_SAMPLE_RATE} Hz"
        )
    if not (isinstance(width, int) and width in PCM_SAMPLE_WIDTHS):
        return f"Unsupported sample width {width}; audio must be 8-, 16-, 24- or 32-bit PCM"
    if not (isinstance(channels, int) and channels >= 1):
        return f"Unsupported channel count {channels}"
    return None


def _pcm_to_wav(audio_bytes: bytes, rate: int, width: int, channels: int) -> bytes:
    """Convert raw PCM audio to WAV format."""
    data_size = len(audio_bytes)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", rate))
    buf.write(struct.pack("<I", rate * channels * width))  # byte rate
    buf.write(struct.pack("<H", channels * width))  # block align
    buf.write(struct.pack("<H", width * 8))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_bytes)
    return buf.getvalue()


def _pcm_to_mono16(pcm_data: bytes, width: int, channels: int) -> bytes:
    """Convert PCM of any accepted width and channel count to 16-bit mono.

    VAD trimming, STT_NORMALIZE and STT_NOISE_REDUCE read 16-bit mono, as HTTP uploads are
    decoded. As in a WAV file, 8-bit samples are unsigned and wider ones signed little-endian;
    a wider sample keeps its top 16 bits. The channels are averaged, and a partial frame at
    the end is dropped.
    """
    frame_bytes = width * channels
    pcm_data = pcm_data[: len(pcm_data) - len(pcm_data) % frame_bytes]
    if width == 2 and channels == 1:
        return pcm_data
    raw = np.frombuffer(pcm_data, dtype=np.uint8)
    if width == 1:
        samples = (raw.astype(np.int16) - 128) << 8
    elif width == 2:
        samples = raw.view("<i2")
    elif width == 3:
        samples = np.ascontiguousarray(raw.reshape(-1, 3)[:, 1:]).view("<i2").reshape(-1)
    else:
        samples = raw.view("<i4") >> 16
    if channels > 1:
        samples = np.rint(samples.reshape(-1, channels).mean(axis=1))
    return samples.astype("<i2").tobytes()


def _extract_speech_segments(
    pcm_data: bytes,
    rate: int,
    width: int,
    channels: int,
) -> bytes:
    """Use VAD to extract only speech segments from PCM audio.

    Returns PCM bytes containing only the speech portions.
    If VAD is disabled or fails, returns the original audio.
    """
    if not settings.stt_vad_enabled:
        return pcm_data

    if not pcm_data or width != 2 or channels != 1:
        return pcm_data

    try:
        from src.streaming import VAD_PREROLL_MS

        # handle_transcribe loaded the singleton (get_vad_model) before this thread started.
        from src.vad.silero import VAD_SAMPLE_RATE, SileroVAD, _vad_model
        if _vad_model is None:
            return pcm_data  # Silero could not be loaded: keep the audio as it is

        vad = SileroVAD(
            _vad_model.session,
            threshold=settings.stt_vad_threshold,
        )

        # Resample to 16kHz if needed
        if rate != VAD_SAMPLE_RATE:
            from src.streaming import resample_pcm16
            pcm_16k = resample_pcm16(pcm_data, rate, VAD_SAMPLE_RATE)
        else:
            pcm_16k = pcm_data

        segments = vad.get_speech_segments(
            pcm_16k,
            min_speech_ms=settings.stt_vad_min_speech_ms,
            silence_ms=settings.stt_vad_silence_ms,
        )

        if not segments:
            logger.debug("Wyoming VAD: no speech segments found")
            return pcm_data  # Return original to let STT decide

        # Extract speech segments from original-rate audio. Exact sample positions: at
        # rate // 1000 samples per ms (22 at 22.05 kHz) the cuts drifted early and lost
        # the end of the speech. The end rounds up, so no final sample is dropped.
        # Each segment starts VAD_PREROLL_MS early, since Silero scores a word's onset below the
        # threshold, but not before the audio or the previous segment, which it then joins.
        audio_samples = np.frombuffer(pcm_data, dtype=np.int16)
        stretches: list[list[int]] = []
        for seg in segments:
            start_sample = max(0, seg.start_ms - VAD_PREROLL_MS) * rate // 1000
            end_sample = min(-(-seg.end_ms * rate // 1000), len(audio_samples))
            if start_sample >= len(audio_samples):
                continue
            if stretches and start_sample <= stretches[-1][1]:
                stretches[-1][1] = max(stretches[-1][1], end_sample)
            else:
                stretches.append([start_sample, end_sample])
        speech_parts = [audio_samples[start:end] for start, end in stretches]

        if speech_parts:
            combined = np.concatenate(speech_parts)
            logger.debug(
                "Wyoming VAD: extracted %d speech segments (%.1fs → %.1fs)",
                len(segments),
                len(audio_samples) / rate,
                len(combined) / rate,
            )
            return combined.tobytes()

    except Exception:
        logger.exception("Wyoming VAD filtering failed, using original audio")

    return pcm_data


def _prepare_wav(audio_chunks: list[bytes], rate: int, width: int, channels: int) -> bytes:
    """Join the chunks as 16-bit mono, keep the speech (VAD), wrap it as WAV and apply STT preprocessing.

    CPU-bound (Silero over the whole utterance, STT_NORMALIZE, STT_NOISE_REDUCE), so it
    runs in a worker thread, as it does for HTTP uploads.
    """
    pcm_data = _pcm_to_mono16(b"".join(audio_chunks), width, channels)
    pcm_data = _extract_speech_segments(pcm_data, rate, 2, 1)
    wav_bytes = _pcm_to_wav(pcm_data, rate, 2, 1)
    return preprocess_stt_audio(
        wav_bytes,
        noise_reduce=settings.stt_noise_reduce,
        normalize=settings.stt_normalize,
    )


async def handle_transcribe(
    audio_chunks: list[bytes],
    rate: int,
    width: int,
    channels: int,
    stt_router: BackendRouter,
    model: str | None = None,
    language: str | None = None,
) -> str:
    """Transcribe assembled audio chunks using the STT pipeline.

    Returns the transcribed text, "" when no audio arrived. A failure to prepare or
    transcribe the audio is raised for the caller to report to the client.
    """
    if not any(audio_chunks):
        return ""
    model_id = model or settings.stt_model
    if settings.stt_vad_enabled:
        # The shared, bounded loader of streaming and realtime: every voice command is trimmed
        # the same way, whether or not another session loaded Silero first.
        try:
            await get_vad_model()
        except Exception as exc:
            logger.warning("Silero VAD is unavailable; Wyoming audio is not trimmed: %s", exc)

    def _transcribe(wav_bytes: bytes) -> str:
        result = stt_router.transcribe(
            audio=wav_bytes,
            model=model_id,
            language=language,
            temperature_fallback=False,  # a voice command waits for its one decode
        )
        return result.get("text", "")

    # The app's HTTP, WebSocket and realtime clients share this event loop. The decode waits
    # for the STT inference lock on the STT thread, not in the shared default executor.
    wav_bytes = await asyncio.to_thread(_prepare_wav, audio_chunks, rate, width, channels)
    # A model that isn't loaded: wait for its files here, holding no STT pool thread, as the HTTP routes do; a failed
    # or refused fetch raises for the caller's error event.
    await await_stt_fetch(stt_router, model_id)
    return await run_stt_inference(_transcribe, wav_bytes)
