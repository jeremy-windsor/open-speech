"""Tests for the VAD module (src/vad/silero.py).

Tests the SileroVAD wrapper, is_speech, get_speech_segments,
and configuration behavior without requiring the real ONNX model.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vad.silero import SileroVAD, Segment, VAD_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Mock ONNX session
# ---------------------------------------------------------------------------

class MockOrtSession:
    """Mock ONNX session matching Silero VAD signature."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob
        self.call_count = 0

    def run(self, output_names, inputs):
        self.call_count += 1
        state = inputs["state"]
        return [np.array([[self.prob]], dtype=np.float32), state]


class SequenceSession:
    """Mock that returns a sequence of probabilities."""

    def __init__(self, probs: list[float]):
        self.probs = probs
        self.idx = 0

    def run(self, output_names, inputs):
        prob = self.probs[self.idx % len(self.probs)]
        self.idx += 1
        state = inputs["state"]
        return [np.array([[prob]], dtype=np.float32), state]


# ---------------------------------------------------------------------------
# SileroVAD.__call__
# ---------------------------------------------------------------------------

class TestSileroVADCall:
    def test_returns_probability(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        audio = np.zeros(512, dtype=np.float32)
        assert vad(audio) == pytest.approx(0.9)

    def test_empty_audio(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad(np.array([], dtype=np.float32)) == 0.0

    def test_short_audio(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad(np.zeros(100, dtype=np.float32)) == 0.0

    def test_multiple_windows_max(self):
        vad = SileroVAD(SequenceSession([0.1, 0.5, 0.3]))
        audio = np.zeros(1536, dtype=np.float32)  # 3 windows
        assert vad(audio) == pytest.approx(0.5)

    def test_reset(self):
        vad = SileroVAD(MockOrtSession())
        vad._state = np.ones((2, 1, 128), dtype=np.float32)
        vad.reset()
        assert np.all(vad._state == 0)

    def test_custom_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.3), threshold=0.4)
        assert vad.threshold == 0.4


# ---------------------------------------------------------------------------
# is_speech
# ---------------------------------------------------------------------------

class TestIsSpeech:
    def test_speech_above_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.8), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is True

    def test_silence_below_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.2), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is False

    def test_empty_bytes(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad.is_speech(b"") is False

    def test_custom_threshold_override(self):
        vad = SileroVAD(MockOrtSession(prob=0.6), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm, threshold=0.7) is False
        assert vad.is_speech(pcm, threshold=0.5) is True

    def test_at_exact_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.5), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is True

    def test_short_pcm(self):
        """PCM shorter than 512 samples → no windows → not speech."""
        vad = SileroVAD(MockOrtSession(prob=0.9), threshold=0.5)
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is False


# ---------------------------------------------------------------------------
# get_speech_segments
# ---------------------------------------------------------------------------

class TestGetSpeechSegments:
    def test_empty_audio(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad.get_speech_segments(b"") == []

    def test_all_speech(self):
        """All windows above threshold → one segment."""
        vad = SileroVAD(MockOrtSession(prob=0.9), threshold=0.5)
        # 2 seconds at 16kHz = 32000 samples
        pcm = np.zeros(32000, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=800)
        assert len(segments) >= 1
        assert segments[0].start_ms == 0

    def test_all_silence(self):
        """All windows below threshold → no segments."""
        vad = SileroVAD(MockOrtSession(prob=0.1), threshold=0.5)
        pcm = np.zeros(32000, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=100)
        assert segments == []

    def test_speech_then_silence(self):
        """Speech followed by enough silence to close segment."""
        # 10 windows speech, then 30 windows silence
        probs = [0.9] * 10 + [0.1] * 30
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        # 40 windows × 512 samples = 20480 samples
        pcm = np.zeros(40 * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=100)
        assert len(segments) == 1
        assert segments[0].start_ms == 0

    def test_min_speech_duration_filters(self):
        """Short speech burst below min_speech_ms is filtered."""
        # 2 windows speech (64ms), then silence
        probs = [0.9] * 2 + [0.1] * 30
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        pcm = np.zeros(32 * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=250, silence_ms=100)
        assert segments == []

    def test_segment_dataclass(self):
        seg = Segment(start_ms=100, end_ms=500)
        assert seg.start_ms == 100
        assert seg.end_ms == 500

    def test_two_speech_segments(self):
        """Two speech bursts with silence gap between them."""
        # 10 windows speech, 30 silence, 10 speech, 10 silence
        probs = [0.9] * 10 + [0.1] * 30 + [0.9] * 10 + [0.1] * 30
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        pcm = np.zeros(80 * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=100)
        assert len(segments) == 2


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestVADConfig:
    def test_default_config_values(self):
        from src.config import settings
        assert settings.stt_vad_enabled is True
        assert settings.stt_vad_threshold == 0.5
        assert settings.stt_vad_min_speech_ms == 250
        assert settings.stt_vad_silence_ms == 800

    def test_disabled_config(self, monkeypatch):
        """STT_VAD_ENABLED=false should be accessible."""
        monkeypatch.setenv("STT_VAD_ENABLED", "false")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_enabled is False

    def test_custom_threshold(self, monkeypatch):
        monkeypatch.setenv("STT_VAD_THRESHOLD", "0.7")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_threshold == 0.7

    def test_custom_min_speech_ms(self, monkeypatch):
        monkeypatch.setenv("STT_VAD_MIN_SPEECH_MS", "500")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_min_speech_ms == 500

    def test_custom_silence_ms(self, monkeypatch):
        monkeypatch.setenv("STT_VAD_SILENCE_MS", "1200")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_silence_ms == 1200


# ---------------------------------------------------------------------------
# Streaming VAD events
# ---------------------------------------------------------------------------

class TestStreamingVADEvents:
    """Test that streaming sessions include VAD state events."""

    def test_session_accepts_vad_param(self):
        """StreamingSession accepts vad_enabled parameter."""
        from src.streaming import StreamingSession
        # Can't fully construct without a WS, but verify the class signature
        import inspect
        sig = inspect.signature(StreamingSession.__init__)
        assert "vad_enabled" in sig.parameters

    def test_streaming_endpoint_accepts_vad(self):
        """streaming_endpoint accepts vad parameter."""
        from src.streaming import streaming_endpoint
        import inspect
        sig = inspect.signature(streaming_endpoint)
        assert "vad" in sig.parameters
