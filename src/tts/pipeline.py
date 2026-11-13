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
    "m4a": "audio/mp4",
}


def get_content_type(fmt: str) -> str:
    return FORMAT_CONTENT_TYPES.get(fmt, "application/octet-stream")


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16."""
    if hasattr(audio, 'numpy'):
        audio = audio.numpy()
    # Casting NaN to an integer is undefined, so map it as Live Reader does.
    audio = np.clip(np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
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
    """Encode to raw 16-bit little-endian mono PCM at the caller's sample rate."""
    return float32_to_int16(audio).tobytes()


# Only errors reach stderr: the banner and progress stats would otherwise fill the pipe.
FFMPEG_QUIET_ARGS = ["-hide_banner", "-nostats", "-loglevel", "error"]

# How much of ffmpeg's stderr a streaming encoder keeps for its error message.
_STDERR_TAIL_BYTES = 8192

FFMPEG_FORMAT_ARGS: dict[str, list[str]] = {
    "pcm": ["-f", "s16le", "-codec:a", "pcm_s16le"],
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "128k"],
    "opus": ["-f", "opus", "-codec:a", "libopus", "-b:a", "64k"],
    "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "flac", "-codec:a", "flac"],
    # MP4 normally needs seekable output so it can rewrite container metadata.
    # Fragmented MP4 keeps M4A output valid when ffmpeg writes to stdout.
    "m4a": [
        "-f", "mp4",
        "-movflags", "+frag_keyframe+empty_moov",
        "-codec:a", "aac",
        "-b:a", "128k",
    ],
}


def encode_with_ffmpeg(audio: np.ndarray, fmt: str, sample_rate: int = 24000) -> bytes:
    """Encode audio using ffmpeg subprocess."""
    pcm_data = float32_to_int16(audio).tobytes()

    if fmt not in FFMPEG_FORMAT_ARGS:
        raise ValueError(f"Unsupported ffmpeg format: {fmt}")

    cmd = [
        "ffmpeg", "-y", *FFMPEG_QUIET_ARGS,
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
    # Collect all chunks into one array; no chunks is a clip with no samples, so a WAV
    # is still a header-only RIFF file.
    all_chunks = list(chunks)
    audio = np.concatenate(all_chunks) if all_chunks else np.zeros(0, dtype=np.float32)

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

    def __init__(self, fmt: str, sample_rate: int = 24000, output_sample_rate: int | None = None) -> None:
        if fmt not in FFMPEG_FORMAT_ARGS:
            raise ValueError(f"Unsupported ffmpeg format: {fmt}")
        self._fmt = fmt
        self._sample_rate = sample_rate
        self._output_sample_rate = output_sample_rate
        self._proc: subprocess.Popen | None = None
        self._output_chunks: list[bytes] = []
        self._stderr_tail = b""
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _start(self) -> None:
        cmd = [
            "ffmpeg", "-y", *FFMPEG_QUIET_ARGS,
            "-f", "s16le",
            "-ar", str(self._sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            *(["-ar", str(self._output_sample_rate)] if self._output_sample_rate else []),
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

        # Background threads read stdout and stderr, so ffmpeg never blocks on a full pipe
        # while write_chunk() waits for it to read stdin.
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_errors, daemon=True)
        self._stderr_thread.start()

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

    def _read_errors(self) -> None:
        """Keep the end of ffmpeg's stderr for the error message in _failure()."""
        assert self._proc is not None
        assert self._proc.stderr is not None
        while True:
            data = self._proc.stderr.read(4096)
            if not data:
                break
            with self._lock:
                self._stderr_tail = (self._stderr_tail + data)[-_STDERR_TAIL_BYTES:]

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
        try:
            self._proc.stdin.write(pcm_data)
            self._proc.stdin.flush()
        except OSError as exc:
            # ffmpeg exited before reading its input, for example a build without the encoder.
            raise self._failure() from exc

    def _failure(self) -> RuntimeError:
        """Wait for ffmpeg to exit and describe the failure with the end of its stderr."""
        assert self._proc is not None
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.close()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        with self._lock:
            stderr = self._stderr_tail.decode(errors="replace").strip()
        return RuntimeError(f"ffmpeg failed to encode streamed {self._fmt}: {stderr}")

    def finish(self) -> bytes:
        """Close stdin and read remaining output."""
        if self._proc is None:
            return b""
        assert self._proc.stdin is not None
        try:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                self.close()
                raise RuntimeError("ffmpeg timed out while finalizing streamed audio") from exc
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=10)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=10)
            if self._proc.returncode:
                raise self._failure()
            # Drain remaining
            return self._drain()
        finally:
            self._release_pipes()

    def close(self) -> None:
        """Kill the ffmpeg process if still running, and close its pipes."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg process did not exit after kill")
        self._release_pipes()

    def _release_pipes(self) -> None:
        """Close ffmpeg's stdin, stdout and stderr: a Popen leaves that to its owner.

        stdout and stderr close only once their reader threads are done with them, since
        closing a pipe another thread is blocked reading can block as well (Windows).
        """
        proc = self._proc
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:  # BrokenPipeError: ffmpeg exited before taking the last bytes
                pass
        for pipe, reader in ((proc.stdout, self._reader_thread), (proc.stderr, self._stderr_thread)):
            if reader is not None:
                reader.join(timeout=5)
                if reader.is_alive():
                    logger.warning("ffmpeg output pipe still being read; leaving it open")
                    continue
            if pipe is not None:
                pipe.close()


def encode_audio_streaming(
    chunks: Iterator[np.ndarray],
    fmt: str = "mp3",
    sample_rate: int = 24000,
    output_sample_rate: int | None = None,
) -> Iterator[bytes]:
    """Encode audio chunks for streaming response.
    
    For pcm at the source rate, yields each chunk's samples as they arrive.
    For wav, yields nothing until the chunks run out, then one complete RIFF
    file, the same bytes as encode_audio: its one header must hold the true
    length. A WAV stream therefore starts only when synthesis ends; pcm is the
    format for the lowest latency.
    For compressed formats (mp3/opus/aac/flac), and pcm resampled to
    output_sample_rate, uses a persistent ffmpeg subprocess so the output is a
    single valid stream.
    """
    if fmt == "pcm" and output_sample_rate in {None, sample_rate}:
        for chunk in chunks:
            if len(chunk) > 0:
                yield encode_pcm(chunk)
        return

    if fmt == "wav":
        all_chunks = [chunk for chunk in chunks if len(chunk) > 0]
        # A streamed WAV response must contain one RIFF header, not one
        # complete WAV file per chunk. Buffer uncompressed WAV output so
        # clients receive a valid file, header-only when there are no samples.
        audio = np.concatenate(all_chunks) if all_chunks else np.zeros(0, dtype=np.float32)
        yield encode_wav(audio, sample_rate)
        return

    # Compressed: use persistent ffmpeg pipe
    encoder = StreamingFFmpegEncoder(fmt, sample_rate, output_sample_rate)
    try:
        import time
        wrote_samples = False
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            encoder.write_chunk(chunk)
            wrote_samples = True
            # Give ffmpeg a moment to produce output
            time.sleep(0.01)
            data = encoder._drain()
            if data:
                yield data
        if not wrote_samples and fmt != "pcm":
            # Like encode_audio, let ffmpeg write its file for a clip with no samples.
            encoder.write_chunk(np.zeros(0, dtype=np.float32))
        # Finish and yield remaining
        remaining = encoder.finish()
        if remaining:
            yield remaining
    finally:
        encoder.close()
