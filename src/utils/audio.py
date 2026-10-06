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
    # ffmpeg must reopen these paths. Open NamedTemporaryFiles are locked on
    # Windows, so close all Python handles before starting the subprocess.
    with tempfile.TemporaryDirectory(prefix="open-speech-convert-") as temp_dir:
        infile = Path(temp_dir) / f"input{suffix}"
        outfile = Path(temp_dir) / "output.wav"
        infile.write_bytes(audio_bytes)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(infile),
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "wav",
                    str(outfile),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
            return outfile.read_bytes()
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
