from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from src.composer import MultiTrackComposer
from src.config import settings


def _sine(freq=440.0, sr=24000, sec=0.1, amp=0.25):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _render_mix(tmp_path, monkeypatch, sources, sample_rate=24000):
    """Render each (samples, track fields) pair as a WAV track and return the mixed output."""
    monkeypatch.setattr(settings, "os_voice_library_path", str(tmp_path))
    monkeypatch.setattr(settings, "os_composer_dir", str(tmp_path / "renders"))
    composer = MultiTrackComposer()
    monkeypatch.setattr(composer, "_save_composition", lambda **_kwargs: None)
    tracks = []
    for n, (samples, fields) in enumerate(sources):
        path = tmp_path / f"track_{n}.wav"
        wavfile.write(path, sample_rate, samples)
        tracks.append({"source_path": str(path), **fields})
    result = composer.render(tracks=tracks, sample_rate=sample_rate)
    rate, mixed = wavfile.read(composer._resolve_repo_path(result["output_path"]))
    assert rate == sample_rate
    return mixed.astype(np.float32) / 32767


def test_mix_two_in_phase_sines_doubles_amplitude_before_clip(tmp_path, monkeypatch):
    s = _sine(amp=0.2)
    mixed = _render_mix(tmp_path, monkeypatch, [(s, {}), (s, {})])
    assert np.isclose(np.max(np.abs(mixed)), np.max(np.abs(s)) * 2, rtol=0.05)


def test_offset_positions_track_in_output_buffer(tmp_path, monkeypatch):
    pulse = np.zeros(100, dtype=np.float32)
    pulse[0] = 1.0
    mixed = _render_mix(tmp_path, monkeypatch, [(pulse, {"offset_s": 0.1})], sample_rate=1000)
    assert len(mixed) == 200
    assert np.isclose(mixed[100], 1.0, atol=1e-4)
    assert np.allclose(mixed[:100], 0.0)


def test_volume_half_reduces_amplitude(tmp_path, monkeypatch):
    s = _sine(amp=0.6)
    mixed = _render_mix(tmp_path, monkeypatch, [(s, {"volume": 0.5})])
    assert np.isclose(np.max(np.abs(mixed)), np.max(np.abs(s)) * 0.5, rtol=0.05)


def test_mix_is_sized_from_headers_to_the_resampled_track_length(tmp_path, monkeypatch):
    # The mix is allocated before any source is decoded, so the header plan must match resample_poly.
    s = _sine(sr=22050, sec=0.1)
    source = tmp_path / "source.wav"
    wavfile.write(source, 22050, s)
    composer = MultiTrackComposer()
    rate, samples = composer._load_audio(source)

    planned = composer._resampled_length(len(s), 22050, 48000)

    assert planned == len(composer._resample(samples, rate, 48000))
    mixed = _render_mix(tmp_path, monkeypatch, [(s, {"offset_s": 0.5})], sample_rate=22050)
    assert len(mixed) == round(0.5 * 22050) + len(s)


@pytest.mark.parametrize("sample_width,channels", [(1, 1), (2, 2), (3, 1), (3, 2), (4, 1)])
def test_wav_header_gives_frame_count_without_decoding(tmp_path, sample_width, channels):
    import wave

    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as wav_file:  # 24-bit samples cannot be memory-mapped
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(16000)
        wav_file.writeframes(bytes(sample_width * channels * 1234))

    assert MultiTrackComposer()._read_wav_header(source) == (16000, 1234)


def test_wav_header_of_a_float_wav_and_of_a_non_wav(tmp_path):
    float_wav, not_wav = tmp_path / "float.wav", tmp_path / "clip.mp3"
    wavfile.write(float_wav, 24000, np.zeros((321, 2), dtype=np.float32))
    not_wav.write_bytes(b"ID3" + bytes(64))

    assert MultiTrackComposer()._read_wav_header(float_wav) == (24000, 321)
    with pytest.raises(ValueError):
        MultiTrackComposer()._read_wav_header(not_wav)


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


def test_composer_rejects_symlink_escaping_configured_storage(tmp_path, monkeypatch, create_symlink):
    root = tmp_path / "custom-storage"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    wavfile.write(outside, 24000, np.zeros(20, dtype=np.int16))
    source = root / "source.wav"
    create_symlink(source, outside)
    monkeypatch.setattr(settings, "os_voice_library_path", str(root))

    with pytest.raises(PermissionError, match="outside allowed roots"):
        MultiTrackComposer()._validate_source_path(str(source))
