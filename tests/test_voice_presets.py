"""Tests for voice presets API."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app, _load_voice_presets, DEFAULT_VOICE_PRESETS
from src import main as main_module


@pytest.fixture
def client():
    mock_router = MagicMock()
    mock_router.loaded_models.return_value = []
    with patch.object(main_module, "tts_router", mock_router):
        yield TestClient(app)


class TestVoicePresetsAPI:
    def test_get_default_presets(self, client):
        resp = client.get("/api/voice-presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "presets" in data
        assert len(data["presets"]) == 3
        names = [p["name"] for p in data["presets"]]
        assert "Will" in names
        assert "Female" in names
        assert "British Butler" in names

    def test_preset_has_required_fields(self, client):
        resp = client.get("/api/voice-presets")
        for p in resp.json()["presets"]:
            assert "name" in p
            assert "voice" in p
            assert "speed" in p

    def test_load_presets_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("presets:\n  - name: Test\n    voice: af_heart\n    speed: 1.2\n    description: Test preset\n")
            f.flush()
            with patch.dict(os.environ, {"TTS_VOICES_CONFIG": f.name}):
                presets = _load_voice_presets()
                assert len(presets) == 1
                assert presets[0]["name"] == "Test"
        os.unlink(f.name)

    def test_load_presets_missing_file(self):
        with patch.dict(os.environ, {"TTS_VOICES_CONFIG": "/nonexistent/path.yml"}):
            presets = _load_voice_presets()
            assert presets == DEFAULT_VOICE_PRESETS


class TestVersionEndpoint:
    def test_health_version(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["version"] == "0.5.0"


class TestExtendedTTSAPI:
    def test_speech_with_voice_design(self):
        """voice_design field is accepted in request body."""
        import numpy as np
        mock_router = MagicMock()
        mock_router.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])
        mock_router.get_backend.return_value = mock_router
        mock_router.capabilities = {"voice_design": True, "voice_clone": True}
        # Make synthesize signature accept voice_design
        mock_router.synthesize.__wrapped__ = None

        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post("/v1/audio/speech", json={
                "model": "qwen3-tts-0.6b",
                "input": "Hello world",
                "voice": "Chelsie",
                "response_format": "wav",
                "voice_design": "warm female voice",
            })
            assert resp.status_code == 200

    def test_speech_with_reference_audio(self):
        """reference_audio field is accepted in request body."""
        import numpy as np
        mock_router = MagicMock()
        mock_router.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])
        mock_router.get_backend.return_value = mock_router
        mock_router.capabilities = {"voice_design": True, "voice_clone": True}

        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post("/v1/audio/speech", json={
                "model": "qwen3-tts-0.6b",
                "input": "Hello world",
                "voice": "default",
                "response_format": "wav",
                "reference_audio": "AAAA",  # base64
            })
            assert resp.status_code == 200

    def test_clone_endpoint(self):
        """Multipart clone endpoint works."""
        import io
        import numpy as np
        mock_router = MagicMock()
        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])
        mock_backend.capabilities = {"voice_design": True, "voice_clone": True}
        mock_router.get_backend.return_value = mock_backend

        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post(
                "/v1/audio/speech/clone",
                data={"input": "Hello", "model": "qwen3-tts-0.6b", "response_format": "wav"},
                files={"reference_audio": ("ref.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            )
            assert resp.status_code == 200

    def test_clone_endpoint_empty_text(self):
        import numpy as np
        mock_router = MagicMock()
        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post(
                "/v1/audio/speech/clone",
                data={"input": "  ", "model": "kokoro", "response_format": "wav"},
            )
            assert resp.status_code == 400
