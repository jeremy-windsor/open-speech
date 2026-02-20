"""
Effects chain — scipy-only, no pedalboard dependency.
Each effect is a pure function: (samples: np.ndarray, sample_rate: int, **params) -> np.ndarray
"""

from __future__ import annotations

import numpy as np
from scipy import signal

SUPPORTED_EFFECTS = {"normalize", "pitch", "reverb", "podcast_eq", "robot"}


def apply_chain(samples: np.ndarray, sample_rate: int, effects: list[dict] | None) -> np.ndarray:
    """Apply ordered list of effects. Each dict: {type: str, ...params}"""
    for fx in (effects or []):
        fx_type = fx.get("type")
        if fx_type == "normalize":
            samples = _normalize(samples, fx.get("target_lufs", -16))
        elif fx_type == "pitch":
            samples = _pitch_shift(samples, sample_rate, fx.get("semitones", 0))
        elif fx_type == "reverb":
            samples = _reverb(samples, sample_rate, fx.get("room", "small"), fx.get("mix", 0.2))
        elif fx_type == "podcast_eq":
            samples = _podcast_eq(samples, sample_rate)
        elif fx_type == "robot":
            samples = _robot(samples, sample_rate)
    return samples.astype(np.float32, copy=False)


def _normalize(samples: np.ndarray, target_lufs: float = -16) -> np.ndarray:
    """Simple peak + RMS normalization toward target level."""
    rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 1.0
    if rms < 1e-8:
        return samples
    target_rms = 10 ** (target_lufs / 20)
    return samples * (target_rms / rms)


def _pitch_shift(samples: np.ndarray, sample_rate: int, semitones: float = 0) -> np.ndarray:
    """Pitch shift via resampling (changes duration — simple approach)."""
    if semitones == 0:
        return samples
    factor = 2 ** (semitones / 12)
    resampled_len = max(1, int(len(samples) / factor))
    return signal.resample(samples, resampled_len)


def _reverb(samples: np.ndarray, sample_rate: int, room: str = "small", mix: float = 0.2) -> np.ndarray:
    """Simple FIR reverb via convolution."""
    room_ms = {"small": 50, "medium": 120, "large": 300}.get(room, 50)
    ir_len = max(1, int(sample_rate * room_ms / 1000))
    ir = np.exp(-np.linspace(0, 6, ir_len))
    ir /= ir.sum()
    wet = signal.fftconvolve(samples, ir, mode="full")[: len(samples)]
    return (1 - mix) * samples + mix * wet


def _podcast_eq(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """High-pass at 80Hz + slight presence boost at 3kHz."""
    nyquist = sample_rate / 2
    b_hp, a_hp = signal.butter(2, 80 / nyquist, btype="high")
    samples = signal.lfilter(b_hp, a_hp, samples)
    b_pk, a_pk = signal.iirpeak(3000 / nyquist, Q=2)
    return signal.lfilter(b_pk, a_pk, samples)


def _robot(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Vocoder-style ring modulator at 100Hz."""
    t = np.arange(len(samples)) / sample_rate
    carrier = np.sin(2 * np.pi * 100 * t)
    return samples * carrier
