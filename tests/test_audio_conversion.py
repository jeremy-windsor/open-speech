"""Subprocess file ownership and real ffmpeg conversion regressions."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import audio as audio_module


def test_converter_closes_temporary_files_before_ffmpeg(monkeypatch):
    # Model Windows refusing ffmpeg access to an open NamedTemporaryFile.
    temporary_files = []
    subprocess_paths = []
    original_tempfile = audio_module.tempfile.NamedTemporaryFile

    def tracked_tempfile(*args, **kwargs):
        handle = original_tempfile(*args, **kwargs)
        temporary_files.append(handle)
        return handle

    def convert(command, **kwargs):
        source = Path(command[command.index("-i") + 1])
        output = Path(command[-1])
        subprocess_paths.extend([source, output])
        if any(not handle.closed for handle in temporary_files):
            raise subprocess.CalledProcessError(1, command, stderr=b"Permission denied")
        assert source.read_bytes() == b"source audio"
        output.write_bytes(b"converted wav")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_module.tempfile, "NamedTemporaryFile", tracked_tempfile)
    monkeypatch.setattr(audio_module.subprocess, "run", convert)

    assert audio_module.convert_to_wav(b"source audio") == b"converted wav"
    assert all(not path.exists() for path in subprocess_paths)


@pytest.mark.parametrize(
    "failure", [FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired]
)
def test_conversion_failure_returns_original_and_cleans_files(monkeypatch, failure):
    subprocess_paths = []

    def fail(command, **kwargs):
        subprocess_paths.extend([Path(command[command.index("-i") + 1]), Path(command[-1])])
        if failure is FileNotFoundError:
            raise FileNotFoundError("ffmpeg unavailable")
        if failure is subprocess.TimeoutExpired:
            # Long uploads can outlast the ffmpeg timeout; the backend can still decode them.
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(audio_module.subprocess, "run", fail)

    assert audio_module.convert_to_wav(b"source audio") == b"source audio"
    assert all(not path.exists() for path in subprocess_paths)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_conversion_resamples_stereo_wav():
    source = io.BytesIO()
    with wave.open(source, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x10\x00\x10" * 2400)

    converted = audio_module.convert_to_wav(source.getvalue(), suffix=".wav")

    with wave.open(io.BytesIO(converted), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 1600


@pytest.mark.parametrize("consumer", ["whisper", "diarization"])
@pytest.mark.parametrize("dependency_fails", [False, True])
def test_inference_closes_input_handles_and_cleans_files(monkeypatch, consumer, dependency_fails):
    from src.backends.faster_whisper import FasterWhisperBackend
    from src.diarization.pyannote_diarizer import PyannoteDiarizer

    temporary_files = []
    input_paths = []
    original_tempfile = audio_module.tempfile.NamedTemporaryFile

    def tracked_tempfile(*args, **kwargs):
        handle = original_tempfile(*args, **kwargs)
        temporary_files.append(handle)
        return handle

    def read_audio(path):
        path = Path(path)
        input_paths.append(path)
        if any(not handle.closed for handle in temporary_files):
            raise PermissionError("Windows input file is still open")
        assert path.read_bytes() == b"input audio"
        if dependency_fails:
            raise RuntimeError("inference failed")

    monkeypatch.setattr(audio_module.tempfile, "NamedTemporaryFile", tracked_tempfile)
    if consumer == "whisper":
        def transcribe(path, **kwargs):
            read_audio(path)

            def segments():
                # faster-whisper consumes the path lazily while iterating.
                assert Path(path).read_bytes() == b"input audio"
                yield SimpleNamespace(text="heard")

            return segments(), SimpleNamespace(language="en", duration=0.1)

        backend = FasterWhisperBackend()
        backend._models["test"] = SimpleNamespace(transcribe=transcribe)
        def run():
            return backend.transcribe(b"input audio", "test")
    else:
        def pipeline(path):
            read_audio(path)
            turn = SimpleNamespace(start=0.0, end=0.1)
            return SimpleNamespace(itertracks=lambda **kwargs: [(turn, None, "speaker")])

        diarizer = object.__new__(PyannoteDiarizer)
        diarizer._pipeline = pipeline
        def run():
            return diarizer.diarize(b"input audio")

    if dependency_fails:
        with pytest.raises(RuntimeError, match="inference failed"):
            run()
    else:
        assert run()
    assert input_paths
    assert all(not path.exists() for path in input_paths)


# ── Declared sample rate ────────────────────────────────────────────────────


def _wav(rate: int, frames: int = 200) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x10\x00" * frames)
    return buf.getvalue()


@pytest.mark.parametrize("rate", [1, 100, 7999, 192001, 384000])
def test_implausible_declared_sample_rate_is_rejected_before_decoding(rate):
    # Decoding resamples to 16 kHz: 200 samples declared at 1 Hz become 3.2M samples.
    with pytest.raises(audio_module.UnsupportedAudioError, match=f"sample rate {rate} Hz"):
        audio_module.check_input_sample_rate(_wav(rate))


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 192000])
def test_real_speech_sample_rates_are_accepted(rate):
    audio_module.check_input_sample_rate(_wav(rate))


def test_input_the_header_reader_cannot_open_is_left_to_the_decoder():
    audio_module.check_input_sample_rate(b"not audio at all")


# ── Decoded length ──────────────────────────────────────────────────────────


def _crc(data: bytes, poly: int, bits: int) -> int:
    crc, top, mask = 0, 1 << (bits - 1), (1 << bits) - 1
    for byte in data:
        crc ^= byte << (bits - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & mask if crc & top else (crc << 1) & mask
    return crc


def _silent_flac(seconds: float, *, frames_at_1hz: int = 0) -> bytes:
    """8 kHz mono FLAC of digital silence, optionally ending in frames declared at 1 Hz.

    A CONSTANT subframe codes 65535 samples in 3 bytes, so a few KB declare hours of
    audio at a legitimate sample rate.
    """
    block = 65535
    frames = max(1, math.ceil(seconds * 8000 / block))
    streaminfo = (
        block.to_bytes(2, "big") * 2 + bytes(6)
        + ((8000 << 44) | (15 << 36) | (frames + frames_at_1hz) * block).to_bytes(8, "big")
        + bytes(16)
    )
    out = bytearray(b"fLaC\x80" + len(streaminfo).to_bytes(3, "big") + streaminfo)
    for index in range(frames + frames_at_1hz):
        at_1hz = index >= frames
        # 16-bit block size; an 8 kHz rate code, or a 16-bit rate in Hz after the header.
        header = bytearray(b"\xff\xf8") + bytes([0x70 | (0b1101 if at_1hz else 0b0100), 0b1000])
        header += chr(index).encode("utf-8")  # FLAC codes frame numbers like UTF-8
        header += (block - 1).to_bytes(2, "big")
        if at_1hz:
            header += (1).to_bytes(2, "big")
        header.append(_crc(header, 0x07, 8))
        frame = header + b"\x00\x00\x00"  # CONSTANT subframe holding sample value 0
        out += frame + _crc(frame, 0x8005, 16).to_bytes(2, "big")
    return bytes(out)


def _fail_ffmpeg(monkeypatch, failure) -> None:
    def fail(command, **kwargs):
        if failure is FileNotFoundError:
            raise FileNotFoundError("ffmpeg unavailable")
        if failure is subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(audio_module.subprocess, "run", fail)


@pytest.mark.parametrize(
    "failure", [FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired]
)
def test_audio_longer_than_the_limit_is_rejected_when_ffmpeg_cannot_convert_it(monkeypatch, failure):
    # The original bytes then reach faster-whisper, which decodes all of them into one
    # 16 kHz float32 array: 100 hours of silent FLAC is 657 KB, but 23 GB decoded.
    _fail_ffmpeg(monkeypatch, failure)

    with pytest.raises(audio_module.UnsupportedAudioError, match="longer than 60 seconds"):
        audio_module.convert_to_wav(_silent_flac(10 * 60), suffix=".flac", max_seconds=60)

    short = _silent_flac(30)
    assert audio_module.convert_to_wav(short, suffix=".flac", max_seconds=60) == short


def test_default_limit_accepts_an_hour_and_rejects_hours_of_audio(monkeypatch):
    _fail_ffmpeg(monkeypatch, FileNotFoundError)
    hour = _silent_flac(60 * 60)

    assert audio_module.convert_to_wav(hour, suffix=".flac") == hour
    with pytest.raises(audio_module.UnsupportedAudioError, match="longer than"):
        audio_module.convert_to_wav(_silent_flac(5 * 60 * 60), suffix=".flac")


def test_frames_declaring_an_implausible_rate_mid_stream_are_rejected(monkeypatch):
    # The header says 8 kHz, so only decoding finds the 1 Hz frames that would resample
    # to 16000 times as many samples.
    _fail_ffmpeg(monkeypatch, FileNotFoundError)

    with pytest.raises(audio_module.UnsupportedAudioError, match="sample rate 1 Hz"):
        audio_module.convert_to_wav(_silent_flac(10, frames_at_1hz=2), suffix=".flac", max_seconds=60)


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(seconds * 16000))


@pytest.mark.parametrize(("input_seconds", "rejected"), [(1.5, False), (10.0, True)])
def test_ffmpeg_output_is_bounded_and_rejected_past_the_limit(monkeypatch, input_seconds, rejected):
    # ffmpeg itself must stop shortly past the limit: hours of 16 kHz WAV would fill the
    # temporary directory and then memory.
    commands = []

    def ffmpeg(command, **kwargs):
        commands.append(command)
        stop_at = float(command[command.index("-t") + 1])
        _write_wav(Path(command[-1]), min(input_seconds, stop_at))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_module.subprocess, "run", ffmpeg)

    if rejected:
        with pytest.raises(audio_module.UnsupportedAudioError, match="longer than 2 seconds"):
            audio_module.convert_to_wav(b"source audio", max_seconds=2)
    else:
        converted = audio_module.convert_to_wav(b"source audio", max_seconds=2)
        with wave.open(io.BytesIO(converted), "rb") as wav:
            assert wav.getnframes() == 24000
    command = commands[0]
    assert 2 < float(command[command.index("-t") + 1]) <= 3
    assert int(command[command.index("-fs") + 1]) <= 3 * 16000 * 2 + 64 * 1024


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_ffmpeg_rejects_silent_flac_past_the_limit():
    with pytest.raises(audio_module.UnsupportedAudioError, match="longer than 60 seconds"):
        audio_module.convert_to_wav(_silent_flac(60 * 60), suffix=".flac", max_seconds=60)
