"""Wyoming STT handler — bridges Wyoming audio events to Open Speech STT pipeline."""

from __future__ import annotations

import asyncio
import io
import logging
import struct
from typing import TYPE_CHECKING

import numpy as np

from src.config import settings
from src.audio.preprocessing import preprocess_stt_audio

if TYPE_CHECKING:
    from src.router import STTRouter

logger = logging.getLogger(__name__)


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
        from src.vad.silero import SileroVAD, VAD_SAMPLE_RATE

        # We need a synchronous VAD model load for Wyoming
        # Import the module-level singleton
        from src.vad.silero import _vad_model
        if _vad_model is None:
            # VAD not loaded yet — skip filtering
            return pcm_data

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

        # Extract speech segments from original-rate audio
        samples_per_ms = rate // 1000
        audio_samples = np.frombuffer(pcm_data, dtype=np.int16)
        speech_parts = []
        for seg in segments:
            start_sample = seg.start_ms * samples_per_ms
            end_sample = min(seg.end_ms * samples_per_ms, len(audio_samples))
            if start_sample < len(audio_samples):
                speech_parts.append(audio_samples[start_sample:end_sample])

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


async def handle_transcribe(
    audio_chunks: list[bytes],
    rate: int,
    width: int,
    channels: int,
    stt_router: STTRouter,
    model: str | None = None,
    language: str | None = None,
) -> str:
    """Transcribe assembled audio chunks using the STT pipeline.

    Returns the transcribed text.
    """
    pcm_data = b"".join(audio_chunks)
    if not pcm_data:
        return ""

    # Apply VAD filtering to extract speech segments
    pcm_data = _extract_speech_segments(pcm_data, rate, width, channels)

    wav_bytes = _pcm_to_wav(pcm_data, rate, width, channels)
    wav_bytes = preprocess_stt_audio(
        wav_bytes,
        noise_reduce=settings.stt_noise_reduce,
        normalize=settings.stt_normalize,
    )
    model_id = model or settings.stt_model

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: stt_router.transcribe(
                audio=wav_bytes,
                model=model_id,
                language=language,
            ),
        )
        return result.get("text", "")
    except Exception:
        logger.exception("Wyoming STT transcription failed")
        return ""
