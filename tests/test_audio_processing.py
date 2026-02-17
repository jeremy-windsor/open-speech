from __future__ import annotations

import numpy as np

from src.audio.preprocessing import (
    float32_mono_to_wav_bytes,
    normalize_gain,
    preprocess_stt_audio,
    wav_bytes_to_float32_mono,
)
from src.audio.postprocessing import normalize_output, process_tts_chunks, trim_silence


def _wav(sr=16000):
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    return float32_mono_to_wav_bytes(audio, sr)


def test_wav_roundtrip():
    w = _wav()
    arr, sr = wav_bytes_to_float32_mono(w)
    assert sr == 16000
    assert len(arr) == 16000


def test_normalize_gain_increases_level():
    x = np.ones(1000, dtype=np.float32) * 0.01
    y = normalize_gain(x)
    assert float(np.mean(np.abs(y))) > float(np.mean(np.abs(x)))


def test_preprocess_normalize_only():
    w = _wav()
    out = preprocess_stt_audio(w, noise_reduce=False, normalize=True)
    assert out[:4] == b"RIFF"


def test_trim_silence_removes_edges():
    x = np.concatenate([np.zeros(100), np.ones(200) * 0.5, np.zeros(100)]).astype(np.float32)
    y = trim_silence(x, threshold=0.1)
    assert len(y) == 200


def test_trim_silence_all_silence_returns_input():
    x = np.zeros(100, dtype=np.float32)
    y = trim_silence(x)
    assert len(y) == 100


def test_normalize_output_peaks_to_target():
    x = np.array([0.1, -0.2, 0.4], dtype=np.float32)
    y = normalize_output(x, peak=0.9)
    assert abs(float(np.max(np.abs(y))) - 0.9) < 1e-3


def test_process_tts_chunks_empty():
    out = list(process_tts_chunks(iter(())))
    assert out == []


def test_process_tts_chunks_combines():
    out = list(process_tts_chunks(iter([np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32)]), trim=False, normalize=False))
    assert len(out) == 1
    assert len(out[0]) == 10


def test_process_tts_chunks_trim_and_normalize():
    x = np.concatenate([np.zeros(5), np.ones(5) * 0.2, np.zeros(5)]).astype(np.float32)
    out = list(process_tts_chunks(iter([x]), trim=True, normalize=True))[0]
    assert len(out) == 5
    assert float(np.max(np.abs(out))) > 0.9


def test_normalize_output_zero_safe():
    x = np.zeros(10, dtype=np.float32)
    y = normalize_output(x)
    assert np.allclose(x, y)
