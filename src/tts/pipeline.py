"""Audio encoding pipeline — converts raw numpy audio to output formats."""

from __future__ import annotations

import io
import logging
import struct
import subprocess
import threading
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


FFMPEG_FORMAT_ARGS: dict[str, list[str]] = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "128k"],
    "opus": ["-f", "opus", "-codec:a", "libopus", "-b:a", "64k"],
    "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "flac", "-codec:a", "flac"],
}


def encode_with_ffmpeg(audio: np.ndarray, fmt: str, sample_rate: int = 24000) -> bytes:
    """Encode audio using ffmpeg subprocess."""
    pcm_data = float32_to_int16(audio).tobytes()

    if fmt not in FFMPEG_FORMAT_ARGS:
        raise ValueError(f"Unsupported ffmpeg format: {fmt}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        *FFMPEG_FORMAT_ARGS[fmt],
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


class StreamingFFmpegEncoder:
    """Persistent ffmpeg subprocess for streaming compressed audio encoding.
    
    Keeps a single ffmpeg process open, writes PCM chunks to stdin,
    reads encoded bytes from stdout. Produces a single valid stream
    (one header, continuous encoding).
    """

    def __init__(self, fmt: str, sample_rate: int = 24000) -> None:
        if fmt not in FFMPEG_FORMAT_ARGS:
            raise ValueError(f"Unsupported ffmpeg format: {fmt}")
        self._fmt = fmt
        self._sample_rate = sample_rate
        self._proc: subprocess.Popen | None = None
        self._output_chunks: list[bytes] = []
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _start(self) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(self._sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            *FFMPEG_FORMAT_ARGS[self._fmt],
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install ffmpeg for mp3/opus/aac/flac support.")

        # Background thread to read stdout without blocking
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        """Read encoded bytes from ffmpeg stdout in a background thread."""
        assert self._proc is not None
        assert self._proc.stdout is not None
        while True:
            data = self._proc.stdout.read(4096)
            if not data:
                break
            with self._lock:
                self._output_chunks.append(data)

    def _drain(self) -> bytes:
        """Drain any available encoded output."""
        with self._lock:
            if not self._output_chunks:
                return b""
            result = b"".join(self._output_chunks)
            self._output_chunks.clear()
            return result

    def write_chunk(self, audio: np.ndarray) -> None:
        """Write a PCM chunk to ffmpeg's stdin."""
        if self._proc is None:
            self._start()
        assert self._proc is not None
        assert self._proc.stdin is not None
        pcm_data = float32_to_int16(audio).tobytes()
        self._proc.stdin.write(pcm_data)
        self._proc.stdin.flush()

    def finish(self) -> bytes:
        """Close stdin and read remaining output."""
        if self._proc is None:
            return b""
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        self._proc.wait(timeout=30)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=10)
        # Drain remaining
        return self._drain()

    def close(self) -> None:
        """Kill the ffmpeg process if still running."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


def encode_audio_streaming(
    chunks: Iterator[np.ndarray],
    fmt: str = "mp3",
    sample_rate: int = 24000,
) -> Iterator[bytes]:
    """Encode audio chunks for streaming response.
    
    For wav/pcm, yields encoded data per chunk.
    For compressed formats (mp3/opus/aac/flac), uses a persistent ffmpeg
    subprocess so the output is a single valid stream.
    """
    if fmt in ("pcm", "wav"):
        # Uncompressed: each chunk is independent
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            if fmt == "pcm":
                yield encode_pcm(chunk)
            else:
                yield encode_wav(chunk, sample_rate)
        return

    # Compressed: use persistent ffmpeg pipe
    encoder = StreamingFFmpegEncoder(fmt, sample_rate)
    try:
        import time
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            encoder.write_chunk(chunk)
            # Give ffmpeg a moment to produce output
            time.sleep(0.01)
            data = encoder._drain()
            if data:
                yield data
        # Finish and yield remaining
        remaining = encoder.finish()
        if remaining:
            yield remaining
    finally:
        encoder.close()
