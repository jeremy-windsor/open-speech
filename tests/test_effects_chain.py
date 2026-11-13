from __future__ import annotations

import numpy as np
import pytest

from src.effects.chain import _reverb, apply_chain, effect_parameter_error


def test_normalize_reduces_peak_level():
    samples = np.ones(1000, dtype=np.float32) * 0.95
    out = apply_chain(samples, 24000, [{"type": "normalize", "target_lufs": -20}])
    assert np.max(np.abs(out)) < np.max(np.abs(samples))


def test_normalize_does_not_push_speech_with_pauses_past_full_scale():
    rng = np.random.default_rng(0)
    speech = rng.laplace(scale=1.0, size=24000)  # the classic speech amplitude model
    speech[12000:] = 0.0  # a pause, as in any sentence
    speech = (0.95 * speech / np.max(np.abs(speech))).astype(np.float32)

    out = apply_chain(speech, 24000, [{"type": "normalize"}])

    assert float(np.max(np.abs(out))) <= 1.0


def test_normalize_leaves_empty_audio_empty():
    # A backend can yield no samples (punctuation-only text); without effects that request
    # returns empty audio, so the effect must not fail on it either.
    out = apply_chain(np.zeros(0, dtype=np.float32), 24000, [{"type": "normalize"}])

    assert out.size == 0


def test_normalize_still_raises_quiet_audio_to_the_target_level():
    samples = (0.01 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)).astype(np.float32)

    out = apply_chain(samples, 24000, [{"type": "normalize", "target_lufs": -16}])

    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(10 ** (-16 / 20), rel=1e-3)


def test_effect_parameter_error_accepts_web_ui_effects():
    effects = [
        {"type": "reverb", "room": "medium"},
        {"type": "pitch", "semitones": -12},
        {"type": "normalize"},
        {"type": "podcast_eq"},
        {"type": "robot"},
    ]

    assert effect_parameter_error(effects) is None


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


def _podcast_eq_gain_db(freq: float, sample_rate: int = 24000) -> float:
    """Steady-state gain of a sine through podcast_eq, after the filters settle."""
    t = np.arange(2 * sample_rate) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    out = apply_chain(tone, sample_rate, [{"type": "podcast_eq"}])
    assert out.shape == tone.shape
    settled = slice(sample_rate, None)
    rms_in = np.sqrt(np.mean(np.square(tone[settled], dtype=np.float64)))
    rms_out = np.sqrt(np.mean(np.square(out[settled], dtype=np.float64)))
    return float(20 * np.log10(rms_out / rms_in))


@pytest.mark.parametrize("freq", [150, 300, 500, 1000])
def test_podcast_eq_keeps_the_speech_band_above_its_high_pass(freq):
    """It was a band-pass around 3 kHz: fundamentals and first formants lost 14-32 dB."""
    assert _podcast_eq_gain_db(freq) > -3.0


@pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
def test_podcast_eq_gives_a_slight_presence_boost_at_3_khz(sample_rate):
    gain = _podcast_eq_gain_db(3000, sample_rate)
    assert 1.0 < gain < 6.0
    assert gain - _podcast_eq_gain_db(500, sample_rate) > 1.0


def test_podcast_eq_cuts_rumble_below_80_hz():
    assert _podcast_eq_gain_db(30) < -12.0


def _tone_level(samples: np.ndarray, freq: float, sample_rate: int) -> float:
    """Amplitude of one tone in a whole number of its cycles (a one-bin DFT)."""
    t = np.arange(len(samples)) / sample_rate
    return float(2 * np.abs(np.mean(samples.astype(np.float64) * np.exp(-2j * np.pi * freq * t))))


@pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
def test_podcast_eq_keeps_speech_normalized_to_095_under_full_scale(sample_rate):
    """TTS_NORMALIZE_OUTPUT leaves speech at peak 0.95 before the effects run,
    and podcast_eq's +3 dB presence boost took 3 kHz-rich peaks past 1.0, where float32_to_int16 clips them.
    The EQ must change the tone, not the ceiling: the peak stays at 0.95 and 3 kHz still stands out from
    the mids by about 3 dB."""
    t = np.arange(2 * sample_rate) / sample_rate
    mix = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 3000 * t)
    clip = (0.95 * mix / np.max(np.abs(mix))).astype(np.float32)
    out = apply_chain(clip, sample_rate, [{"type": "podcast_eq"}])
    assert float(np.max(np.abs(out))) <= 0.95 + 1e-6
    settled = slice(sample_rate, None)
    tilt_in = _tone_level(clip[settled], 3000, sample_rate) / _tone_level(clip[settled], 500, sample_rate)
    tilt_out = _tone_level(out[settled], 3000, sample_rate) / _tone_level(out[settled], 500, sample_rate)
    assert 2.0 < 20 * np.log10(tilt_out / tilt_in) < 4.0


def test_podcast_eq_leaves_a_quiet_clip_its_boost():
    """The limit is a ceiling, not a make-down for every clip."""
    sample_rate = 24000
    t = np.arange(2 * sample_rate) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    out = apply_chain(tone, sample_rate, [{"type": "podcast_eq"}])
    assert _tone_level(out[sample_rate:], 3000, sample_rate) > 1.2 * 0.3


def test_apply_chain_multiple_effects_in_order():
    samples = np.sin(2 * np.pi * 220 * np.arange(24000) / 24000).astype(np.float32)
    out = apply_chain(samples, 24000, [{"type": "robot"}, {"type": "normalize", "target_lufs": -18}])
    assert len(out) == len(samples)
    assert not np.allclose(out, samples)


def test_apply_chain_empty_returns_unchanged_audio():
    samples = np.random.randn(1000).astype(np.float32) * 0.1
    out = apply_chain(samples, 24000, [])
    assert np.allclose(out, samples)
