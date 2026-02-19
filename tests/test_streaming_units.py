"""Unit tests for streaming module — pure functions and testable logic.

Tests the non-WebSocket, non-model-dependent parts of streaming.py:
- resample_pcm16 (resampling)
- LocalAgreement2 (word stabilization)
- StreamingSession._pcm_to_wav (WAV header generation)
- SileroVAD interface contract
- Session constants and limits
"""

from __future__ import annotations

import struct
import wave
import io

import numpy as np
import pytest

from src.streaming import (
    LocalAgreement2,
    StreamingSession,
    resample_pcm16,
    INTERNAL_SAMPLE_RATE,
    MAX_UTTERANCE_SECONDS,
    MAX_UTTERANCE_BYTES,
    MIN_SAMPLE_RATE,
    MAX_SAMPLE_RATE,
)
from src.vad.silero import SileroVAD


# ---------------------------------------------------------------------------
# resample_pcm16
# ---------------------------------------------------------------------------

class TestResamplePCM16:
    """Tests for PCM16 resampling function."""

    def test_same_rate_passthrough(self):
        """No resampling when rates match."""
        pcm = np.array([100, -200, 300], dtype=np.int16).tobytes()
        result = resample_pcm16(pcm, 16000, 16000)
        assert result == pcm

    def test_upsample_doubles_length(self):
        """Upsampling 16kHz → 32kHz roughly doubles sample count."""
        samples = np.arange(100, dtype=np.int16)
        pcm = samples.tobytes()
        result = resample_pcm16(pcm, 16000, 32000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        assert len(out_samples) == 200  # exactly 2x

    def test_downsample_halves_length(self):
        """Downsampling 48kHz → 16kHz reduces by 3x."""
        samples = np.arange(300, dtype=np.int16)
        pcm = samples.tobytes()
        result = resample_pcm16(pcm, 48000, 16000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        assert len(out_samples) == 100

    def test_empty_input(self):
        """Empty bytes returns empty bytes."""
        result = resample_pcm16(b"", 16000, 48000)
        assert result == b""

    def test_single_sample(self):
        """Single sample edge case."""
        pcm = np.array([1000], dtype=np.int16).tobytes()
        result = resample_pcm16(pcm, 16000, 32000)
        out = np.frombuffer(result, dtype=np.int16)
        assert len(out) == 2

    def test_output_is_valid_pcm16(self):
        """Output is valid int16 range."""
        samples = np.array([32767, -32768, 0, 16000, -16000], dtype=np.int16)
        result = resample_pcm16(samples.tobytes(), 8000, 16000)
        out = np.frombuffer(result, dtype=np.int16)
        assert out.min() >= -32768
        assert out.max() <= 32767

    def test_non_integer_ratio(self):
        """Non-integer ratio like 44100→16000 produces correct length."""
        samples = np.arange(441, dtype=np.int16)  # 10ms at 44.1kHz
        result = resample_pcm16(samples.tobytes(), 44100, 16000)
        out = np.frombuffer(result, dtype=np.int16)
        expected_len = int(441 * 16000 / 44100)
        assert len(out) == expected_len

    def test_preserves_dc_offset(self):
        """Constant signal stays constant after resampling."""
        samples = np.full(100, 5000, dtype=np.int16)
        result = resample_pcm16(samples.tobytes(), 16000, 48000)
        out = np.frombuffer(result, dtype=np.int16)
        # Polyphase resampling keeps DC nearly constant (allow minor FIR ripple/rounding)
        assert np.allclose(out, 5000, atol=4)


# ---------------------------------------------------------------------------
# LocalAgreement2
# ---------------------------------------------------------------------------

class TestLocalAgreement2:
    """Tests for the word stabilization algorithm."""

    def test_first_input_no_confirmed(self):
        """First transcription has nothing to agree with."""
        la = LocalAgreement2()
        confirmed, pending = la.process("hello world")
        assert confirmed == []
        assert pending == ["hello", "world"]

    def test_agreement_on_prefix(self):
        """Matching prefix words become confirmed."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("hello world foo")
        assert confirmed == ["hello", "world"]
        assert pending == ["foo"]

    def test_disagreement_resets(self):
        """When words diverge, nothing new is confirmed."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("hello earth")
        assert confirmed == ["hello"]
        assert pending == ["earth"]

    def test_complete_disagreement(self):
        """Completely different text = no new confirmation."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("goodbye earth")
        assert confirmed == []
        assert pending == ["goodbye", "earth"]

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        la = LocalAgreement2()
        la.process("Hello World")
        confirmed, pending = la.process("hello world more")
        assert confirmed == ["hello", "world"]

    def test_flush_returns_remaining(self):
        """Flush emits unconfirmed words."""
        la = LocalAgreement2()
        la.process("the quick brown fox")
        la.process("the quick brown fox jumps")
        # "the quick brown fox" confirmed, "jumps" pending
        remaining = la.flush()
        assert remaining == ["jumps"]

    def test_flush_empty_when_all_confirmed(self):
        """Flush returns empty if all confirmed."""
        la = LocalAgreement2()
        la.process("hello")
        la.process("hello")
        remaining = la.flush()
        assert remaining == []

    def test_reset(self):
        """Reset clears all state."""
        la = LocalAgreement2()
        la.process("hello world")
        la.process("hello world more")
        la.reset()
        confirmed, pending = la.process("new text")
        assert confirmed == []
        assert pending == ["new", "text"]

    def test_empty_input(self):
        """Empty string produces no words."""
        la = LocalAgreement2()
        confirmed, pending = la.process("")
        assert confirmed == []
        assert pending == []

    def test_whitespace_only(self):
        """Whitespace-only input = empty."""
        la = LocalAgreement2()
        confirmed, pending = la.process("   ")
        assert confirmed == []
        assert pending == []

    def test_progressive_agreement(self):
        """Three rounds of progressive agreement."""
        la = LocalAgreement2()
        la.process("the")
        c1, p1 = la.process("the quick")
        assert c1 == ["the"]
        assert p1 == ["quick"]

        c2, p2 = la.process("the quick brown")
        assert c2 == ["quick"]
        assert p2 == ["brown"]

    def test_no_double_confirm(self):
        """Already-confirmed words aren't re-emitted."""
        la = LocalAgreement2()
        la.process("a b c")
        la.process("a b c d")  # confirms a, b, c
        confirmed, pending = la.process("a b c d e")  # confirms d
        assert confirmed == ["d"]
        assert pending == ["e"]

    def test_single_word_stability(self):
        """Single repeated word gets confirmed."""
        la = LocalAgreement2()
        la.process("hello")
        confirmed, pending = la.process("hello")
        assert confirmed == ["hello"]
        assert pending == []

    def test_shorter_input_still_confirms_prefix(self):
        """If second input is shorter but shares prefix, prefix is confirmed."""
        la = LocalAgreement2()
        la.process("hello world foo")
        confirmed, pending = la.process("hello world")
        assert confirmed == ["hello", "world"]
        assert pending == []

    def test_empty_after_nonempty(self):
        """Empty input after non-empty confirms nothing."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("")
        assert confirmed == []
        assert pending == []

    def test_flush_after_empty_input(self):
        """Flush after empty input returns nothing (previous_words cleared)."""
        la = LocalAgreement2()
        la.process("hello world")
        la.process("")
        remaining = la.flush()
        assert remaining == []

    def test_retraction_after_confirmation_is_ignored(self):
        """Once words are confirmed, retraction in next input doesn't undo them.

        This is by design — confirmed words have already been sent to the client
        as is_final=True. The algorithm sacrifices retraction for stability.
        """
        la = LocalAgreement2()
        la.process("a b")
        la.process("a b")  # confirms 'a', 'b'
        assert la.confirmed_words == ["a", "b"]
        # Now ASR retracts 'b' → 'c'. common_len=1, but confirmed=2
        confirmed, pending = la.process("a c")
        assert confirmed == []  # No new confirmations
        assert pending == []    # 'c' is lost — by design
        assert la.confirmed_words == ["a", "b"]  # Stale but intentional

    def test_punctuation_affects_matching(self):
        """Punctuation in ASR output affects word matching."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("hello, world")
        # "hello" != "hello," so only no common prefix
        assert confirmed == []
        assert pending == ["hello,", "world"]

    def test_confirmed_preserves_current_casing(self):
        """Confirmed words use the casing from the current (second) input."""
        la = LocalAgreement2()
        la.process("Hello World")
        confirmed, pending = la.process("hello world more")
        # Case-insensitive match, but stores current casing
        assert confirmed == ["hello", "world"]


# ---------------------------------------------------------------------------
# StreamingSession._pcm_to_wav
# ---------------------------------------------------------------------------

class TestPcmToWav:
    """Tests for WAV header generation."""

    def test_valid_wav_header(self):
        """Generated WAV is parseable by the wave module."""
        pcm = np.zeros(1600, dtype=np.int16).tobytes()
        wav_data = StreamingSession._pcm_to_wav(pcm, 16000)

        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 1600

    def test_different_sample_rates(self):
        """WAV encodes the correct sample rate."""
        for rate in [8000, 16000, 22050, 44100, 48000]:
            pcm = np.zeros(100, dtype=np.int16).tobytes()
            wav_data = StreamingSession._pcm_to_wav(pcm, rate)
            with wave.open(io.BytesIO(wav_data), "rb") as wf:
                assert wf.getframerate() == rate

    def test_empty_audio(self):
        """Empty PCM produces valid (empty) WAV."""
        wav_data = StreamingSession._pcm_to_wav(b"", 16000)
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            assert wf.getnframes() == 0

    def test_round_trip_data(self):
        """PCM data survives the WAV round-trip."""
        original = np.array([100, -200, 32767, -32768, 0], dtype=np.int16)
        wav_data = StreamingSession._pcm_to_wav(original.tobytes(), 16000)
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            frames = wf.readframes(5)
        recovered = np.frombuffer(frames, dtype=np.int16)
        np.testing.assert_array_equal(original, recovered)


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify streaming constants are sane."""

    def test_internal_sample_rate(self):
        assert INTERNAL_SAMPLE_RATE == 16000

    def test_max_utterance_bytes_matches_seconds(self):
        expected = MAX_UTTERANCE_SECONDS * INTERNAL_SAMPLE_RATE * 2
        assert MAX_UTTERANCE_BYTES == expected

    def test_max_utterance_reasonable(self):
        assert 10 <= MAX_UTTERANCE_SECONDS <= 120

    def test_sample_rate_range(self):
        assert MIN_SAMPLE_RATE == 8000
        assert MAX_SAMPLE_RATE == 192000


# ---------------------------------------------------------------------------
# SileroVAD interface (mocked ONNX session)
# ---------------------------------------------------------------------------

class MockOrtSession:
    """Mock ONNX session matching real Silero VAD return signature.
    
    Real ort.InferenceSession.run() returns a flat list of output arrays.
    Silero VAD returns [output_prob, new_state] as a 2-element list.
    The SileroVAD wrapper unpacks: out, self._state = session.run(...)
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def run(self, output_names, inputs):
        state = inputs["state"]
        # Real signature: returns [prob_array, state_array] as a flat list
        # prob_array shape is (1, 1) so out[0][0] yields a scalar
        return [np.array([[self.prob]], dtype=np.float32), state]


class TestSileroVAD:
    """Tests for the Silero VAD wrapper (with mock session)."""

    def test_returns_probability(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        audio = np.zeros(512, dtype=np.float32)
        prob = vad(audio)
        assert prob == pytest.approx(0.9)

    def test_empty_audio_returns_zero(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        prob = vad(np.array([], dtype=np.float32))
        assert prob == 0.0

    def test_short_audio_under_window(self):
        """Audio shorter than 512 samples returns 0."""
        vad = SileroVAD(MockOrtSession(prob=0.9))
        audio = np.zeros(100, dtype=np.float32)
        prob = vad(audio)
        assert prob == 0.0

    def test_multiple_windows_returns_max(self):
        """Multiple 512-sample windows return max probability."""
        # Session returns different probs based on call count
        class IncreasingSession:
            def __init__(self):
                self.call = 0
            def run(self, output_names, inputs):
                self.call += 1
                state = inputs["state"]
                prob = 0.1 * self.call
                return [np.array([[prob]], dtype=np.float32), state]

        vad = SileroVAD(IncreasingSession())
        audio = np.zeros(1536, dtype=np.float32)  # 3 windows
        prob = vad(audio)
        assert prob == pytest.approx(0.3)  # max of 0.1, 0.2, 0.3

    def test_reset_clears_state(self):
        vad = SileroVAD(MockOrtSession())
        vad._state = np.ones((2, 1, 128), dtype=np.float32)
        vad.reset()
        assert np.all(vad._state == 0)
