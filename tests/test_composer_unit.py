from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from src.composer import MultiTrackComposer
from src.config import settings


def _sine(freq=440.0, sr=24000, sec=0.1, amp=0.25):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mix_two_in_phase_sines_doubles_amplitude_before_clip():
    c = MultiTrackComposer()
    s = _sine(amp=0.2)
    mixed = c._mix_prepared([
        {"samples": s, "offset_s": 0.0},
        {"samples": s, "offset_s": 0.0},
    ], sample_rate=24000)
    assert np.isclose(np.max(np.abs(mixed)), np.max(np.abs(s)) * 2, rtol=0.05)


def test_offset_positions_track_in_output_buffer():
    c = MultiTrackComposer()
    pulse = np.zeros(100, dtype=np.float32)
    pulse[0] = 1.0
    mixed = c._mix_prepared([
        {"samples": pulse, "offset_s": 0.1},
    ], sample_rate=1000)
    assert len(mixed) == 200
    assert mixed[100] == 1.0
    assert np.allclose(mixed[:100], 0.0)


def test_volume_half_reduces_amplitude():
    c = MultiTrackComposer()
    s = _sine(amp=0.6)
    mixed = c._mix_prepared([
        {"samples": s * 0.5, "offset_s": 0.0},
    ], sample_rate=24000)
    assert np.isclose(np.max(np.abs(mixed)), np.max(np.abs(s)) * 0.5, rtol=0.05)


@pytest.mark.parametrize("dtype,channels", [
    (np.int16, 1),
    (np.int16, 2),
    (np.int32, 2),
    (np.uint8, 1),
    (np.uint8, 2),
    (np.float32, 2),
])
def test_load_wav_preserves_pcm_amplitude_when_downmixing(tmp_path, dtype, channels):
    expected = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    if dtype == np.uint8:
        pcm = (expected * 128 + 128).astype(dtype)
    elif np.issubdtype(dtype, np.signedinteger):
        pcm = (expected * (2 ** (np.iinfo(dtype).bits - 1))).astype(dtype)
    else:
        pcm = expected.astype(dtype)
    if channels > 1:
        pcm = np.column_stack([pcm] * channels)
    source = tmp_path / "source.wav"
    wavfile.write(source, 24000, pcm)

    sample_rate, audio = MultiTrackComposer()._load_audio(source)

    assert sample_rate == 24000
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, expected, atol=2e-5)


@pytest.mark.parametrize("setting", [
    "os_voice_library_path", "os_conversations_dir", "os_composer_dir",
])
def test_composer_accepts_audio_in_configured_storage(tmp_path, monkeypatch, setting):
    root = tmp_path / "custom-storage"
    root.mkdir()
    source = root / "source.wav"
    wavfile.write(source, 24000, np.zeros(20, dtype=np.int16))
    monkeypatch.setattr(settings, setting, str(root))

    assert MultiTrackComposer()._validate_source_path(str(source)) == source.resolve()


def test_composer_rejects_symlink_escaping_configured_storage(tmp_path, monkeypatch):
    root = tmp_path / "custom-storage"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    wavfile.write(outside, 24000, np.zeros(20, dtype=np.int16))
    source = root / "source.wav"
    source.symlink_to(outside)
    monkeypatch.setattr(settings, "os_voice_library_path", str(root))

    with pytest.raises(PermissionError, match="outside allowed roots"):
        MultiTrackComposer()._validate_source_path(str(source))
