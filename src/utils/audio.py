"""Audio conversion utilities using ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def convert_to_wav(audio_bytes: bytes, suffix: str = ".ogg") -> bytes:
    """Convert audio bytes to 16kHz mono WAV using ffmpeg.
    
    If ffmpeg is not available, returns the original bytes and lets
    the backend handle format detection.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as infile:
        infile.write(audio_bytes)
        infile.flush()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as outfile:
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", infile.name,
                        "-ar", "16000",
                        "-ac", "1",
                        "-f", "wav",
                        outfile.name,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                return Path(outfile.name).read_bytes()
            except (subprocess.CalledProcessError, FileNotFoundError):
                # ffmpeg not available or conversion failed; return original
                return audio_bytes


def get_suffix_from_content_type(content_type: str | None) -> str:
    """Map content type to file suffix."""
    mapping = {
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/flac": ".flac",
        "audio/x-flac": ".flac",
        "audio/mp4": ".m4a",
        "audio/m4a": ".m4a",
        "audio/webm": ".webm",
        "video/webm": ".webm",
    }
    return mapping.get(content_type or "", ".ogg")
