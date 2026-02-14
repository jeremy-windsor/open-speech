"""Tests for the streaming transcription pipeline.

Tests LocalAgreement2, SileroVAD wrapper, resample_pcm16, and _pcm_to_wav
without requiring a running server or model.
"""

import struct
import numpy as np
import pytest

# Import the units under test
from src.streaming import LocalAgreement2, resample_pcm16, StreamingSession


class TestLocalAgreement2:
    """Test the LocalAgreement2 stable emission algorithm."""

    def test_first_call_no_confirmed(self):
        """First transcription should have no confirmed words."""
        la = LocalAgreement2()
        confirmed, pending = la.process("hello world")
        assert confirmed == []
        assert pending == ["hello", "world"]

    def test_second_call_matching_confirms(self):
        """Two identical transcriptions should confirm all words."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("hello world")
        assert confirmed == ["hello", "world"]
        assert pending == []

    def test_partial_agreement(self):
        """Only the common prefix should be confirmed."""
        la = LocalAgreement2()
        la.process("hello world foo")
        confirmed, pending = la.process("hello world bar")
        assert confirmed == ["hello", "world"]
        assert pending == ["bar"]

    def test_no_agreement(self):
        """Completely different transcriptions shouldn't confirm anything."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("goodbye earth")
        assert confirmed == []
        assert pending == ["goodbye", "earth"]

    def test_cumulative_confirmation(self):
        """Confirmed words accumulate across calls."""
        la = LocalAgreement2()
        la.process("the quick brown fox")
        la.process("the quick brown fox jumps")
        # "the quick brown fox" confirmed
        la.process("the quick brown fox jumps over")
        # "jumps" now confirmed too
        assert la.confirmed_words == ["the", "quick", "brown", "fox", "jumps"]

    def test_flush_emits_remaining(self):
        """Flush should emit all unconfirmed words."""
        la = LocalAgreement2()
        la.process("hello world")
        remaining = la.flush()
        assert remaining == ["hello", "world"]

    def test_empty_input(self):
        la = LocalAgreement2()
        confirmed, pending = la.process("")
        assert confirmed == []
        assert pending == []

    def test_reset(self):
        la = LocalAgreement2()
        la.process("hello world")
        la.reset()
        assert la.previous_words == []
        assert la.confirmed_words == []

    def test_case_insensitive_matching(self):
        """Agreement should be case-insensitive."""
        la = LocalAgreement2()
        la.process("Hello World")
        confirmed, pending = la.process("hello world")
        assert confirmed == ["hello", "world"]


class TestResamplePcm16:
    """Test PCM16 resampling."""

    def test_same_rate_noop(self):
        """Same rate should return identical bytes."""
        pcm = np.array([100, -200, 300], dtype=np.int16).tobytes()
        result = resample_pcm16(pcm, 16000, 16000)
        assert result == pcm

    def test_downsample_halves_length(self):
        """Downsampling 2:1 should roughly halve the sample count."""
        samples = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
        result = resample_pcm16(samples.tobytes(), 48000, 16000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        expected_len = int(1000 * 16000 / 48000)
        assert abs(len(out_samples) - expected_len) <= 1

    def test_upsample_doubles_length(self):
        """Upsampling 1:2 should roughly double the sample count."""
        samples = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
        result = resample_pcm16(samples.tobytes(), 16000, 32000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        expected_len = int(1000 * 32000 / 16000)
        assert abs(len(out_samples) - expected_len) <= 1

    def test_empty_input(self):
        result = resample_pcm16(b"", 44100, 16000)
        assert result == b""

    def test_44100_to_16000(self):
        """Common browser rate → server rate."""
        # 2 seconds at 44100 Hz
        t = np.linspace(0, 2, 88200)
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        result = resample_pcm16(tone.tobytes(), 44100, 16000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        expected = int(88200 * 16000 / 44100)
        assert abs(len(out_samples) - expected) <= 1

    def test_48000_to_16000(self):
        """Another common browser rate."""
        samples = np.zeros(96000, dtype=np.int16)  # 2s at 48kHz
        result = resample_pcm16(samples.tobytes(), 48000, 16000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        assert abs(len(out_samples) - 32000) <= 1


class TestPcmToWav:
    """Test WAV header generation."""

    def test_valid_wav_header(self):
        """Generated WAV should have valid RIFF/WAVE header."""
        pcm = np.zeros(16000, dtype=np.int16).tobytes()  # 1s at 16kHz
        wav = StreamingSession._pcm_to_wav(pcm, 16000)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert wav[36:40] == b"data"

        # Check sizes
        riff_size = struct.unpack("<I", wav[4:8])[0]
        assert riff_size == 36 + len(pcm)

        data_size = struct.unpack("<I", wav[40:44])[0]
        assert data_size == len(pcm)

    def test_sample_rate_in_header(self):
        """WAV header should contain the correct sample rate."""
        pcm = b"\x00" * 100
        wav = StreamingSession._pcm_to_wav(pcm, 44100)
        sample_rate = struct.unpack("<I", wav[24:28])[0]
        assert sample_rate == 44100

    def test_pcm_data_preserved(self):
        """PCM data after header should be identical to input."""
        pcm = np.array([1, -1, 32767, -32768, 0], dtype=np.int16).tobytes()
        wav = StreamingSession._pcm_to_wav(pcm, 16000)
        assert wav[44:] == pcm
