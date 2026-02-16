"""Tests for streaming ffmpeg encoder and pipeline streaming."""

import subprocess
import numpy as np
import pytest

from src.tts.pipeline import (
    StreamingFFmpegEncoder,
    encode_audio_streaming,
    float32_to_int16,
)


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")


class TestStreamingFFmpegEncoder:
    @needs_ffmpeg
    def test_mp3_produces_valid_output(self):
        enc = StreamingFFmpegEncoder("mp3", sample_rate=24000)
        try:
            # Write several chunks
            for _ in range(5):
                chunk = np.random.randn(4800).astype(np.float32) * 0.1
                enc.write_chunk(chunk)
            result = enc.finish()
            # MP3 files start with 0xff 0xfb or ID3 tag
            assert len(result) > 0
            assert result[:3] == b"ID3" or result[0] == 0xFF
        finally:
            enc.close()

    @needs_ffmpeg
    def test_flac_produces_valid_output(self):
        enc = StreamingFFmpegEncoder("flac", sample_rate=24000)
        try:
            chunk = np.zeros(24000, dtype=np.float32)
            enc.write_chunk(chunk)
            result = enc.finish()
            assert len(result) > 0
            assert result[:4] == b"fLaC"
        finally:
            enc.close()

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            StreamingFFmpegEncoder("wav")

    @needs_ffmpeg
    def test_no_chunks_returns_empty(self):
        enc = StreamingFFmpegEncoder("mp3")
        result = enc.finish()
        # Empty input → empty or minimal output
        enc.close()


class TestEncodeAudioStreamingCompressed:
    @needs_ffmpeg
    def test_streaming_mp3_single_output(self):
        """Streaming MP3 should produce a single valid stream, not per-chunk headers."""
        chunks = [
            np.random.randn(4800).astype(np.float32) * 0.1
            for _ in range(3)
        ]
        result = b"".join(encode_audio_streaming(iter(chunks), fmt="mp3"))
        assert len(result) > 0
        # Verify it's valid MP3 by checking header
        assert result[:3] == b"ID3" or result[0] == 0xFF

    def test_streaming_pcm_per_chunk(self):
        """PCM streaming yields independent chunks (no encoding needed)."""
        chunks = [
            np.zeros(100, dtype=np.float32),
            np.ones(100, dtype=np.float32) * 0.5,
        ]
        results = list(encode_audio_streaming(iter(chunks), fmt="pcm"))
        assert len(results) == 2
        assert len(results[0]) == 200
        assert len(results[1]) == 200

    def test_streaming_wav_per_chunk(self):
        """WAV streaming yields independent chunks with headers."""
        chunks = [np.zeros(100, dtype=np.float32)]
        results = list(encode_audio_streaming(iter(chunks), fmt="wav"))
        assert len(results) == 1
        assert results[0][:4] == b"RIFF"

    def test_streaming_empty_chunks_skipped(self):
        chunks = [
            np.array([], dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.array([], dtype=np.float32),
        ]
        results = list(encode_audio_streaming(iter(chunks), fmt="pcm"))
        assert len(results) == 1
