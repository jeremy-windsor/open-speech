"""Subprocess file ownership and real ffmpeg conversion regressions."""

from __future__ import annotations

import io
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


@pytest.mark.parametrize("failure", [FileNotFoundError, subprocess.CalledProcessError])
def test_conversion_failure_returns_original_and_cleans_files(monkeypatch, failure):
    subprocess_paths = []

    def fail(command, **kwargs):
        subprocess_paths.extend([Path(command[command.index("-i") + 1]), Path(command[-1])])
        if failure is FileNotFoundError:
            raise FileNotFoundError("ffmpeg unavailable")
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
