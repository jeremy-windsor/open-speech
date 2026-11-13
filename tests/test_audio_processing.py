from __future__ import annotations

import numpy as np
import pytest

from src.audio.preprocessing import (
    float32_mono_to_wav_bytes,
    normalize_gain,
    preprocess_stt_audio,
    wav_bytes_to_float32_mono,
)
from src.audio.postprocessing import (
    StreamingEdgeTrimmer,
    normalize_output,
    process_tts_chunks,
    trim_silence,
)


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


def test_normalize_gain_caps_the_boost_for_near_silent_input():
    # Room hiss at -80 dBFS would need +62 dB to reach -18 dBFS, and Whisper hallucinates
    # text on amplified noise; STT_NORMALIZE boosts by at most +30 dB.
    x = np.full(1000, 1e-4, dtype=np.float32)
    y = normalize_gain(x)
    assert np.allclose(y, x * 10 ** (30 / 20), rtol=1e-4)


def _speech_with_a_pause(sr=16000):
    """Laplacian 'speech' with a 4 Hz syllable envelope and -1 dBFS peaks, then as long a pause."""
    rng = np.random.default_rng(1234)
    t = np.arange(5 * sr) / sr
    speech = rng.laplace(0.0, 1.0, len(t)) * 0.5 * (1 - np.cos(2 * np.pi * 4 * t))
    speech *= 10 ** (-1 / 20) / np.max(np.abs(speech))
    return np.concatenate([speech, np.zeros(5 * sr)]).astype(np.float32)


def test_normalize_gain_is_one_gain_for_speech_with_pauses():
    # Its RMS is low, so the RMS gain (+8 dB) would push the peaks past full scale; clipping
    # them distorted the speech Whisper decodes. The boost stops where the peak reaches 0.95.
    x = _speech_with_a_pause()
    y = normalize_gain(x)
    voiced = np.abs(x) > 1e-6
    ratios = y[voiced] / x[voiced]
    assert float(np.ptp(ratios)) < 1e-4
    assert float(np.max(np.abs(y))) == pytest.approx(0.95, abs=1e-4)


def test_preprocess_normalize_does_not_clip_a_short_loud_word_before_silence():
    sr = 16000
    t = np.arange(sr // 2) / sr
    audio = np.concatenate([0.9 * np.sin(2 * np.pi * 440 * t), np.zeros(int(29.5 * sr))])
    wav_in = float32_mono_to_wav_bytes(audio.astype(np.float32), sr)

    x, _ = wav_bytes_to_float32_mono(wav_in)
    y, _ = wav_bytes_to_float32_mono(preprocess_stt_audio(wav_in, noise_reduce=False, normalize=True))

    assert len(y) == len(x)
    assert float(np.max(np.abs(y))) < 0.96  # nothing at full scale
    gain = float(np.dot(x, y) / np.dot(x, x))
    assert gain > 1.0  # still boosted, up to the peak limit
    np.testing.assert_allclose(y, gain * x, atol=4 / 32768)  # one gain, to int16 rounding


def test_preprocess_normalize_keeps_near_silent_audio_quiet():
    t = np.linspace(0, 1, 16000, endpoint=False)
    hiss = (3e-4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # about -73 dBFS RMS
    out, _ = wav_bytes_to_float32_mono(
        preprocess_stt_audio(float32_mono_to_wav_bytes(hiss, 16000), noise_reduce=False, normalize=True)
    )
    rms = float(np.sqrt(np.mean(np.square(out))))
    assert rms < 0.01  # +30 dB from -73 dBFS, far below the -18 dBFS target (0.126)


def test_preprocess_normalize_only():
    w = _wav()
    out = preprocess_stt_audio(w, noise_reduce=False, normalize=True)
    assert out[:4] == b"RIFF"


def test_normalize_gain_returns_an_empty_clip_unchanged():
    # A clip with no samples raised numpy's "zero-size array to
    # reduction operation maximum". Any gain of an empty clip is an empty clip, as in
    # postprocessing.normalize_output and the `normalize` effect.
    out = normalize_gain(np.zeros(0, dtype=np.float32))

    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_preprocess_normalize_keeps_a_header_only_wav():
    # A header-only WAV (0 frames, as /v1/audio/speech answers a clip
    # with no samples) passes the 0-byte "Empty audio file" check, and STT_NORMALIZE (on by default)
    # turned it into a 500 for HTTP and batch, and an error event for Wyoming.
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"")

    out = preprocess_stt_audio(buf.getvalue(), noise_reduce=False, normalize=True)

    with wave.open(io.BytesIO(out), "rb") as wf:
        assert (wf.getnframes(), wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (0, 1, 2, 16000)


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


@pytest.mark.parametrize("bad_sample", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_process_tts_chunks_keeps_the_clip_around_a_non_finite_sample(bad_sample):
    # Live Reader and the streaming trimmer map NaN to 0 and inf to full scale, so one bad
    # backend sample costs one sample there. Peak normalization must not spread it to the clip,
    # neither as silence nor as a gain set by the bad sample.
    t = np.arange(24000) / 24000
    # Voiced at both ends, so trimming keeps every sample and indices line up.
    clean = (0.5 * np.cos(2 * np.pi * 220 * t)).astype(np.float32)
    clip = clean.copy()
    clip[1000] = bad_sample

    expected = list(process_tts_chunks(iter([clean]), trim=True, normalize=True))[0]
    out = list(process_tts_chunks(iter([clip]), trim=True, normalize=True))[0]

    assert np.isfinite(out).all()
    assert len(out) == len(expected)
    rest = np.arange(len(out)) != 1000
    np.testing.assert_allclose(out[rest], expected[rest], atol=1e-6)


def test_streaming_edge_trimmer_preserves_interior_silence_across_chunks():
    trimmer = StreamingEdgeTrimmer(
        1000,
        threshold=0.1,
        lead_padding_ms=2,
        tail_padding_ms=3,
    )

    outputs = [
        trimmer.push(np.array([0, 0, 0, 0.5, 0, 0], dtype=np.float32)),
        trimmer.push(np.array([0, 0.5, 0, 0, 0, 0], dtype=np.float32)),
        trimmer.finish(),
    ]

    assert np.allclose(np.concatenate(outputs), [0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0])


def test_streaming_edge_trimmer_emits_voice_without_waiting_for_finish():
    trimmer = StreamingEdgeTrimmer(1000, threshold=0.1, lead_padding_ms=1, tail_padding_ms=2)

    output = trimmer.push(np.array([0, 0, 0.5, 0], dtype=np.float32))

    assert np.allclose(output, [0, 0.5])


def test_streaming_edge_trimmer_drops_all_silent_input():
    trimmer = StreamingEdgeTrimmer(1000)

    assert trimmer.push(np.zeros(100, dtype=np.float32)).size == 0
    assert trimmer.finish().size == 0


def test_normalize_output_zero_safe():
    x = np.zeros(10, dtype=np.float32)
    y = normalize_output(x)
    assert np.allclose(x, y)
