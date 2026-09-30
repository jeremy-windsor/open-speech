"""Subprocess file ownership and real ffmpeg conversion regressions."""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path

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
