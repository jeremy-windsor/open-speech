"""Wyoming STT handler — bridges Wyoming audio events to Open Speech STT pipeline."""

from __future__ import annotations

import asyncio
import io
import logging
import struct
from typing import TYPE_CHECKING

from src.config import settings

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

    wav_bytes = _pcm_to_wav(pcm_data, rate, width, channels)
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
