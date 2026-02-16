"""Audio encoding pipeline — converts raw numpy audio to output formats."""

from __future__ import annotations

import io
import logging
import struct
import subprocess
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Content-Type mapping
FORMAT_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def get_content_type(fmt: str) -> str:
    return FORMAT_CONTENT_TYPES.get(fmt, "application/octet-stream")


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def encode_wav(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Encode float32 numpy array to WAV bytes."""
    pcm = float32_to_int16(audio)
    buf = io.BytesIO()
    num_samples = len(pcm)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    # Write WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))   # PCM format
    buf.write(struct.pack("<H", 1))   # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))   # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


def encode_pcm(audio: np.ndarray) -> bytes:
    """Encode to raw 24kHz 16-bit little-endian mono PCM."""
    return float32_to_int16(audio).tobytes()


def encode_with_ffmpeg(audio: np.ndarray, fmt: str, sample_rate: int = 24000) -> bytes:
    """Encode audio using ffmpeg subprocess."""
    pcm_data = float32_to_int16(audio).tobytes()

    fmt_args: dict[str, list[str]] = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "128k"],
        "opus": ["-f", "opus", "-codec:a", "libopus", "-b:a", "64k"],
        "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
    }

    if fmt not in fmt_args:
        raise ValueError(f"Unsupported ffmpeg format: {fmt}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        *fmt_args[fmt],
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=pcm_data,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            logger.error("ffmpeg error: %s", proc.stderr.decode(errors="replace"))
            raise RuntimeError(f"ffmpeg failed with return code {proc.returncode}")
        return proc.stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg for mp3/opus/aac/flac support.")


def encode_audio(
    chunks: Iterator[np.ndarray],
    fmt: str = "mp3",
    sample_rate: int = 24000,
) -> bytes:
    """Collect all audio chunks and encode to the requested format.
    
    Returns complete encoded audio bytes.
    """
    # Collect all chunks into one array
    all_chunks = list(chunks)
    if not all_chunks:
        return b""
    audio = np.concatenate(all_chunks)

    if fmt == "wav":
        return encode_wav(audio, sample_rate)
    elif fmt == "pcm":
        return encode_pcm(audio)
    else:
        return encode_with_ffmpeg(audio, fmt, sample_rate)


def encode_audio_streaming(
    chunks: Iterator[np.ndarray],
    fmt: str = "mp3",
    sample_rate: int = 24000,
) -> Iterator[bytes]:
    """Encode audio chunks one at a time for streaming response.
    
    For wav/pcm, yields raw data per chunk.
    For compressed formats, each chunk is independently encoded via ffmpeg.
    """
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        if fmt == "pcm":
            yield encode_pcm(chunk)
        elif fmt == "wav":
            yield encode_wav(chunk, sample_rate)
        else:
            yield encode_with_ffmpeg(chunk, fmt, sample_rate)
