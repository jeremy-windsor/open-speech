"""Tests for SRT/VTT/text formatters."""

import pytest
from src.formatters import format_as_text, format_as_srt, format_as_vtt, format_transcription


class TestFormatAsText:
    def test_extracts_text(self):
        assert format_as_text({"text": "Hello world"}) == "Hello world"

    def test_strips_whitespace(self):
        assert format_as_text({"text": "  hello  "}) == "hello"

    def test_empty(self):
        assert format_as_text({}) == ""


class TestFormatAsSrt:
    def test_with_segments(self):
        result = {
            "text": "Hello world",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello"},
                {"start": 1.5, "end": 3.0, "text": "world"},
            ],
        }
        srt = format_as_srt(result)
        assert "1\n00:00:00,000 --> 00:00:01,500\nHello" in srt
        assert "2\n00:00:01,500 --> 00:00:03,000\nworld" in srt

    def test_without_segments(self):
        result = {"text": "Hello world", "duration": 2.5}
        srt = format_as_srt(result)
        assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in srt

    def test_empty(self):
        assert format_as_srt({"text": ""}) == ""

    def test_hours(self):
        result = {"text": "test", "segments": [{"start": 3661.5, "end": 3663.0, "text": "test"}]}
        srt = format_as_srt(result)
        assert "01:01:01,500" in srt


class TestFormatAsVtt:
    def test_starts_with_header(self):
        result = {"text": "Hello", "segments": [{"start": 0, "end": 1, "text": "Hello"}]}
        vtt = format_as_vtt(result)
        assert vtt.startswith("WEBVTT")

    def test_uses_dot_separator(self):
        result = {"text": "Hello", "segments": [{"start": 0, "end": 1.5, "text": "Hello"}]}
        vtt = format_as_vtt(result)
        assert "00:00:00.000 --> 00:00:01.500" in vtt

    def test_without_segments(self):
        result = {"text": "Hello", "duration": 2.0}
        vtt = format_as_vtt(result)
        assert "WEBVTT" in vtt
        assert "Hello" in vtt

    def test_empty(self):
        assert format_as_vtt({"text": ""}) == "WEBVTT\n"


class TestFormatTranscription:
    def test_text_format(self):
        content, ct = format_transcription({"text": "hello"}, "text")
        assert content == "hello"
        assert ct == "text/plain"

    def test_srt_format(self):
        content, ct = format_transcription({"text": "hello", "duration": 1.0}, "srt")
        assert "00:00:00,000" in content
        assert ct == "text/plain"

    def test_vtt_format(self):
        content, ct = format_transcription({"text": "hello", "duration": 1.0}, "vtt")
        assert "WEBVTT" in content
        assert ct == "text/vtt"

    def test_json_format(self):
        content, ct = format_transcription({"text": "hello"}, "json")
        assert content == ""
        assert ct == "application/json"
