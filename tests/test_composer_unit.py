from __future__ import annotations

import numpy as np

from src.composer import MultiTrackComposer


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
