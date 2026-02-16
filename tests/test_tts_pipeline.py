"""Tests for the audio encoding pipeline."""

import struct

import numpy as np
import pytest

from src.tts.pipeline import (
    encode_wav,
    encode_pcm,
    float32_to_int16,
    get_content_type,
    encode_audio,
)


class TestFloat32ToInt16:
    def test_silence(self):
        audio = np.zeros(100, dtype=np.float32)
        result = float32_to_int16(audio)
        assert result.dtype == np.int16
        assert np.all(result == 0)

    def test_max_value(self):
        audio = np.array([1.0], dtype=np.float32)
        result = float32_to_int16(audio)
        assert result[0] == 32767

    def test_min_value(self):
        audio = np.array([-1.0], dtype=np.float32)
        result = float32_to_int16(audio)
        assert result[0] == -32767

    def test_clipping(self):
        audio = np.array([2.0, -2.0], dtype=np.float32)
        result = float32_to_int16(audio)
        assert result[0] == 32767
        assert result[1] == -32767


class TestEncodeWav:
    def test_valid_wav_header(self):
        audio = np.zeros(24000, dtype=np.float32)  # 1 second
        wav = encode_wav(audio, sample_rate=24000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "

    def test_sample_rate_in_header(self):
        audio = np.zeros(100, dtype=np.float32)
        wav = encode_wav(audio, sample_rate=24000)
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 24000

    def test_mono_channel(self):
        audio = np.zeros(100, dtype=np.float32)
        wav = encode_wav(audio)
        channels = struct.unpack_from("<H", wav, 22)[0]
        assert channels == 1

    def test_data_length(self):
        audio = np.zeros(100, dtype=np.float32)
        wav = encode_wav(audio)
        data_size = struct.unpack_from("<I", wav, 40)[0]
        assert data_size == 200  # 100 samples * 2 bytes


class TestEncodePcm:
    def test_correct_length(self):
        audio = np.zeros(100, dtype=np.float32)
        pcm = encode_pcm(audio)
        assert len(pcm) == 200  # 100 samples * 2 bytes

    def test_silence(self):
        audio = np.zeros(100, dtype=np.float32)
        pcm = encode_pcm(audio)
        assert pcm == b"\x00" * 200


class TestEncodeAudio:
    def test_wav_format(self):
        chunks = iter([np.zeros(100, dtype=np.float32)])
        result = encode_audio(chunks, fmt="wav")
        assert result[:4] == b"RIFF"

    def test_pcm_format(self):
        chunks = iter([np.zeros(100, dtype=np.float32)])
        result = encode_audio(chunks, fmt="pcm")
        assert len(result) == 200

    def test_empty_chunks(self):
        result = encode_audio(iter([]), fmt="wav")
        assert result == b""

    def test_multiple_chunks_concatenated(self):
        chunks = iter([
            np.ones(50, dtype=np.float32) * 0.5,
            np.ones(50, dtype=np.float32) * -0.5,
        ])
        result = encode_audio(chunks, fmt="pcm")
        assert len(result) == 200


class TestContentType:
    def test_known_formats(self):
        assert get_content_type("mp3") == "audio/mpeg"
        assert get_content_type("wav") == "audio/wav"
        assert get_content_type("opus") == "audio/opus"
        assert get_content_type("flac") == "audio/flac"
        assert get_content_type("aac") == "audio/aac"
        assert get_content_type("pcm") == "audio/pcm"

    def test_unknown_format(self):
        assert get_content_type("xyz") == "application/octet-stream"
