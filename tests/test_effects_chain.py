from __future__ import annotations

import numpy as np

from src.effects.chain import _reverb, apply_chain


def test_normalize_reduces_peak_level():
    samples = np.ones(1000, dtype=np.float32) * 0.95
    out = apply_chain(samples, 24000, [{"type": "normalize", "target_lufs": -20}])
    assert np.max(np.abs(out)) < np.max(np.abs(samples))


def test_pitch_shift_preserves_array_length():
    samples = np.sin(2 * np.pi * 220 * np.arange(24000) / 24000).astype(np.float32)
    out = apply_chain(samples, 24000, [{"type": "pitch", "semitones": 4}])
    assert len(out) == len(samples)
    assert not np.allclose(out, samples)


def test_reverb_increases_output_mix():
    samples = np.zeros(24000, dtype=np.float32)
    samples[0] = 1.0
    out = _reverb(samples, 24000, room="medium", mix=0.5)
    assert np.sum(np.abs(out[1:])) > 0


def test_apply_chain_multiple_effects_in_order():
    samples = np.sin(2 * np.pi * 220 * np.arange(24000) / 24000).astype(np.float32)
    out = apply_chain(samples, 24000, [{"type": "robot"}, {"type": "normalize", "target_lufs": -18}])
    assert len(out) == len(samples)
    assert not np.allclose(out, samples)


def test_apply_chain_empty_returns_unchanged_audio():
    samples = np.random.randn(1000).astype(np.float32) * 0.1
    out = apply_chain(samples, 24000, [])
    assert np.allclose(out, samples)
