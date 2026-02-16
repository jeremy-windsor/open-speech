"""Tests for audio file format detection."""

from src.utils.audio import get_suffix_from_content_type
from src.main import _suffix_from_filename


class TestContentTypeDetection:
    def test_wav(self):
        assert get_suffix_from_content_type("audio/wav") == ".wav"
        assert get_suffix_from_content_type("audio/x-wav") == ".wav"
        assert get_suffix_from_content_type("audio/wave") == ".wav"

    def test_mp3(self):
        assert get_suffix_from_content_type("audio/mpeg") == ".mp3"
        assert get_suffix_from_content_type("audio/mp3") == ".mp3"

    def test_ogg(self):
        assert get_suffix_from_content_type("audio/ogg") == ".ogg"

    def test_flac(self):
        assert get_suffix_from_content_type("audio/flac") == ".flac"
        assert get_suffix_from_content_type("audio/x-flac") == ".flac"

    def test_m4a(self):
        assert get_suffix_from_content_type("audio/mp4") == ".m4a"
        assert get_suffix_from_content_type("audio/m4a") == ".m4a"

    def test_webm(self):
        assert get_suffix_from_content_type("audio/webm") == ".webm"
        assert get_suffix_from_content_type("video/webm") == ".webm"

    def test_unknown_defaults_ogg(self):
        assert get_suffix_from_content_type(None) == ".ogg"
        assert get_suffix_from_content_type("application/octet-stream") == ".ogg"


class TestFilenameDetection:
    def test_common_formats(self):
        assert _suffix_from_filename("test.wav") == ".wav"
        assert _suffix_from_filename("test.mp3") == ".mp3"
        assert _suffix_from_filename("test.flac") == ".flac"
        assert _suffix_from_filename("test.m4a") == ".m4a"
        assert _suffix_from_filename("test.webm") == ".webm"
        assert _suffix_from_filename("test.ogg") == ".ogg"

    def test_case_insensitive(self):
        assert _suffix_from_filename("test.WAV") == ".wav"
        assert _suffix_from_filename("test.MP3") == ".mp3"

    def test_unknown(self):
        assert _suffix_from_filename("test.xyz") is None
        assert _suffix_from_filename("noext") is None
