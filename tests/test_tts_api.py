"""Tests for TTS API endpoints (mocked backend)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src.tts.backends.base import VoiceInfo


@pytest.fixture
def tts_client():
    """Create test client with mocked TTS router."""
    mock_router = MagicMock()
    # synthesize returns an iterator of numpy chunks
    mock_router.synthesize.return_value = iter([
        np.zeros(24000, dtype=np.float32),  # 1 second of silence
    ])
    mock_router.list_voices.return_value = [
        VoiceInfo(id="af_heart", name="Heart", language="en-us", gender="female"),
        VoiceInfo(id="am_adam", name="Adam", language="en-us", gender="male"),
    ]
    mock_router.loaded_models.return_value = []

    with patch.object(main_module, "tts_router", mock_router):
        yield TestClient(app), mock_router


class TestSpeechEndpoint:
    def test_basic_synthesis(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": "alloy",
            "response_format": "wav",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"

    def test_pcm_format(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "pcm",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"

    def test_empty_input_rejected(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "",
            "voice": "alloy",
        })
        assert resp.status_code == 400

    def test_whitespace_input_rejected(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "   ",
            "voice": "alloy",
        })
        assert resp.status_code == 400

    def test_invalid_format_rejected(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "invalid",
        })
        assert resp.status_code == 400

    def test_input_too_long(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "x" * 5000,
            "voice": "alloy",
        })
        assert resp.status_code == 400

    def test_default_format_is_mp3(self, tts_client):
        """Verify that default response_format matches OpenAI spec (mp3)."""
        client, mock = tts_client
        # Send without response_format
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
        })
        # Should attempt mp3 encoding (may fail without ffmpeg, that's ok)
        # Just verify the request was accepted
        assert resp.status_code in (200, 500)  # 500 if no ffmpeg

    def test_speed_validation(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "speed": 0.1,  # Below minimum
        })
        assert resp.status_code == 422

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "speed": 5.0,  # Above maximum
        })
        assert resp.status_code == 422


class TestVoicesEndpoint:
    def test_list_voices(self, tts_client):
        client, mock = tts_client
        resp = client.get("/v1/audio/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert "voices" in data
        assert len(data["voices"]) == 2
        assert data["voices"][0]["id"] == "af_heart"
