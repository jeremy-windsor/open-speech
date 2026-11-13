"""Audio conversion utilities using ffmpeg."""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path

# Decoding resamples to 16 kHz, so a declared rate far below that turns a few KB into
# gigabytes (200 samples at 1 Hz become 3.2M samples). Real speech audio is 8-192 kHz.
MIN_INPUT_SAMPLE_RATE = 8000
MAX_INPUT_SAMPLE_RATE = 192000
# Whisper decodes the whole input into one 16 kHz float32 array, and a few KB of FLAC
# silence can declare 100 hours at a legal rate. Four hours is about 1 GB decoded.
MAX_INPUT_AUDIO_SECONDS = 4 * 60 * 60
CONVERTED_BYTES_PER_SECOND = 16000 * 2  # convert_to_wav's 16 kHz mono 16-bit PCM


class UnsupportedAudioError(ValueError):
    """Audio the server refuses to decode; the message is safe to show the client."""


# A 0-byte upload's message (src/services/stt.py), and the same for audio that decodes to no samples.
EMPTY_AUDIO_MESSAGE = "Empty audio file"


class EmptyAudioError(UnsupportedAudioError):
    """Audio that decodes to zero samples, such as a WAV header with no frames."""

    def __init__(self) -> None:
        super().__init__(EMPTY_AUDIO_MESSAGE)


def _check_rate(rate: int) -> None:
    # 0 means the header does not say; the decoder rejects such input itself.
    if rate and not MIN_INPUT_SAMPLE_RATE <= rate <= MAX_INPUT_SAMPLE_RATE:
        raise UnsupportedAudioError(
            f"Unsupported sample rate {rate} Hz; audio must be "
            f"{MIN_INPUT_SAMPLE_RATE}-{MAX_INPUT_SAMPLE_RATE} Hz"
        )


def _too_long(max_seconds: float) -> UnsupportedAudioError:
    return UnsupportedAudioError(
        f"Audio is longer than {max_seconds:g} seconds; split it into shorter files"
    )


def check_input_sample_rate(audio_bytes: bytes) -> None:
    """Reject audio whose declared sample rate is outside 8-192 kHz, before decoding it.

    Only the container header is read. Input PyAV cannot open is left for the decoder
    to accept or reject, as before.
    """
    try:
        import av
    except ImportError:
        return
    try:
        # The same open options as faster-whisper's decoder, so it cannot read past us.
        with av.open(io.BytesIO(audio_bytes), mode="r", metadata_errors="ignore") as container:
            rates = [stream.codec_context.sample_rate for stream in container.streams.audio]
    except Exception:
        return
    for rate in rates:
        _check_rate(rate)


def check_decoded_length(audio_bytes: bytes, max_seconds: float) -> None:
    """Decode the first audio stream without keeping it, rejecting it past max_seconds.

    Headers can omit or misstate the length and the rate, so samples are counted frame
    by frame. Frames are read as faster-whisper's decode_audio reads them (invalid ones
    skipped); input it cannot open or decode is left for it to reject. A stream that
    decodes to the end with no samples raises EmptyAudioError.
    """
    try:
        import av
    except ImportError:
        return
    try:
        container = av.open(io.BytesIO(audio_bytes), mode="r", metadata_errors="ignore")
    except Exception:
        return
    with container:
        if not container.streams.audio:
            return
        frames = container.decode(audio=0)
        seconds = 0.0
        samples = 0
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                if not samples:
                    raise EmptyAudioError() from None
                return
            except av.error.InvalidDataError:
                continue
            except Exception:
                return
            _check_rate(frame.sample_rate)
            samples += frame.samples
            if frame.sample_rate:
                seconds += frame.samples / frame.sample_rate
            if seconds > max_seconds:
                raise _too_long(max_seconds)


def _wav_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        if wav_file.getnframes() == 0:
            raise EmptyAudioError()  # ffmpeg decoded no samples
        return wav_file.getnframes() / wav_file.getframerate()


def convert_to_wav(audio_bytes: bytes, suffix: str = ".ogg", max_seconds: float | None = None) -> bytes:
    """Convert audio bytes to 16kHz mono WAV using ffmpeg.

    If ffmpeg is not available, returns the original bytes and lets
    the backend handle format detection.

    Either way, audio longer than max_seconds (by default MAX_INPUT_AUDIO_SECONDS)
    raises UnsupportedAudioError.
    """
    if max_seconds is None:
        max_seconds = MAX_INPUT_AUDIO_SECONDS
    # ffmpeg stops one second past the limit, so the output shows whether the input was
    # longer. -fs also bounds the file if odd timestamps keep -t from triggering.
    stop_at = max_seconds + 1
    max_output_bytes = int(stop_at * CONVERTED_BYTES_PER_SECOND) + 4096
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
                    "-t", f"{stop_at:.3f}",
                    "-fs", str(max_output_bytes),
                    "-f", "wav",
                    str(outfile),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
            converted = outfile.read_bytes()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # ffmpeg not available, failed or too slow; return original. The backend
            # decodes all of it, so count its length first.
            check_decoded_length(audio_bytes, max_seconds)
            return audio_bytes
    try:
        seconds = _wav_seconds(converted)
    except (EOFError, wave.Error, ZeroDivisionError):
        return converted  # -t and -fs bound it; the backend accepts or rejects it
    if seconds > max_seconds:
        raise _too_long(max_seconds)
    return converted


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
