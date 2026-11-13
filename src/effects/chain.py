"""
Effects chain — scipy-only, no pedalboard dependency.
Each effect is a pure function: (samples: np.ndarray, sample_rate: int, **params) -> np.ndarray
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy import signal

SUPPORTED_EFFECTS = {"normalize", "pitch", "reverb", "podcast_eq", "robot"}

# Values outside these ranges either fail inside the effect or wreck the signal.
_NUMERIC_PARAMETERS = {
    "normalize": {"target_lufs": (-60.0, 0.0)},
    "pitch": {"semitones": (-24.0, 24.0)},
    "reverb": {"mix": (0.0, 1.0)},
}


def effect_parameter_error(effects: list[dict] | None) -> str | None:
    """Explain the first effect parameter apply_chain cannot use, or return None."""
    for fx in effects or []:
        fx_type = fx.get("type")
        if fx_type == "reverb" and not isinstance(fx.get("room", "small"), str):
            return "reverb room must be a string"
        ranges = _NUMERIC_PARAMETERS.get(fx_type, {}) if isinstance(fx_type, str) else {}
        for name, (low, high) in ranges.items():
            if name not in fx:
                continue
            value = fx[name]
            # bool is an int subclass, and NaN fails every comparison.
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                return f"{fx_type} {name} must be a number from {low:g} to {high:g}"
    return None


def apply_chain(samples: np.ndarray, sample_rate: int, effects: list[dict] | None) -> np.ndarray:
    """Apply ordered list of effects. Each dict: {type: str, ...params}"""
    for fx in (effects or []):
        fx_type = fx.get("type")
        if fx_type == "normalize":
            samples = _normalize(samples, fx.get("target_lufs", -16))
        elif fx_type == "pitch":
            samples = _pitch_shift(samples, sample_rate, fx.get("semitones", 0))
        elif fx_type == "reverb":
            room = fx.get("room", "small")
            mix_map = {"small": 0.25, "medium": 0.4, "large": 0.55}
            mix = fx.get("mix", mix_map.get(room, 0.3))
            samples = _reverb(samples, sample_rate, room, mix)
        elif fx_type == "podcast_eq":
            samples = _podcast_eq(samples, sample_rate)
        elif fx_type == "robot":
            samples = _robot(samples, sample_rate)
    return samples.astype(np.float32, copy=False)


def _normalize(samples: np.ndarray, target_lufs: float = -16, peak: float = 0.95) -> np.ndarray:
    """Simple peak + RMS normalization toward target level."""
    if len(samples) == 0:
        return samples
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-8:
        return samples
    target_rms = 10 ** (target_lufs / 20)
    # Speech with pauses has a high crest factor, so RMS gain alone would clip its peaks.
    gain = min(target_rms / rms, peak / float(np.max(np.abs(samples))))
    return samples * gain


def _pitch_shift(samples: np.ndarray, sample_rate: int, semitones: float = 0) -> np.ndarray:
    """Duration-preserving pitch shift."""
    if semitones == 0:
        return samples
    return librosa.effects.pitch_shift(samples.astype(np.float32), sr=sample_rate, n_steps=semitones)


def _reverb(samples: np.ndarray, sample_rate: int, room: str = "small", mix: float = 0.2) -> np.ndarray:
    """Simple FIR reverb via convolution."""
    room_ms = {"small": 50, "medium": 120, "large": 300}.get(room, 50)
    ir_len = max(1, int(sample_rate * room_ms / 1000))
    ir = np.exp(-np.linspace(0, 6, ir_len))
    ir /= ir.sum()
    wet = signal.fftconvolve(samples, ir, mode="full")[: len(samples)]
    return (1 - mix) * samples + mix * wet


def _podcast_eq(samples: np.ndarray, sample_rate: int, peak: float = 0.95) -> np.ndarray:
    """High-pass at 80Hz + slight presence boost at 3kHz, keeping the peak under its ceiling."""
    if len(samples) == 0:
        return samples
    peak_in = float(np.max(np.abs(samples)))
    nyquist = sample_rate / 2
    b_hp, a_hp = signal.butter(2, 80 / nyquist, btype="high")
    samples = signal.lfilter(b_hp, a_hp, samples)
    # iirpeak is a band-pass (0 dB at 3 kHz, falling off on both sides), so add its band
    # to the signal instead of replacing it: about +3 dB at 3 kHz, 0 dB through the mids.
    b_bp, a_bp = signal.iirpeak(3000 / nyquist, Q=2)
    presence_gain = 10 ** (3 / 20) - 1
    samples = samples + presence_gain * signal.lfilter(b_bp, a_bp, samples)
    # Output is often normalized to 0.95 before the effects run, so the boost has no headroom:
    # a fixed make-down gain (not a per-sample clip) keeps the peak at the input's or 0.95.
    ceiling = max(peak_in, peak)
    peak_out = float(np.max(np.abs(samples)))
    if peak_out > ceiling:
        samples = samples * (ceiling / peak_out)
    return samples


def _robot(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Vocoder-style ring modulator at 100Hz."""
    t = np.arange(len(samples)) / sample_rate
    carrier = np.sin(2 * np.pi * 100 * t)
    return samples * carrier
