"""Tests for TTS model management endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src.tts.backends.base import TTSLoadedModelInfo


@pytest.fixture
def tts_client():
    mock_router = MagicMock()
    mock_router.loaded_models.return_value = []
    mock_router.is_model_loaded.return_value = False
    mock_router.list_backends.return_value = ["kokoro"]

    with patch.object(main_module, "tts_router", mock_router):
        yield TestClient(app), mock_router


class TestTTSModelLoad:
    def test_load_model(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/models/load", json={"model": "kokoro"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"
        mock.load_model.assert_called_once_with("kokoro")

    def test_load_default(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/models/load", json={})
        assert resp.status_code == 200
        mock.load_model.assert_called_once()


class TestTTSModelUnload:
    def test_unload_model(self, tts_client):
        client, mock = tts_client
        mock.is_model_loaded.return_value = True
        resp = client.post("/v1/audio/models/unload", json={"model": "kokoro"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

    def test_unload_not_loaded(self, tts_client):
        client, mock = tts_client
        mock.is_model_loaded.return_value = False
        resp = client.post("/v1/audio/models/unload", json={"model": "kokoro"})
        assert resp.status_code == 404


class TestTTSModelList:
    def test_list_empty(self, tts_client):
        client, mock = tts_client
        resp = client.get("/v1/audio/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        # Default model should appear as not_loaded
        assert any(m["status"] == "not_loaded" for m in data["models"])

    def test_list_with_loaded(self, tts_client):
        client, mock = tts_client
        mock.loaded_models.return_value = [
            TTSLoadedModelInfo(
                model="kokoro", backend="kokoro",
                device="cpu", loaded_at=1000.0,
            )
        ]
        resp = client.get("/v1/audio/models")
        data = resp.json()
        loaded = [m for m in data["models"] if m["status"] == "loaded"]
        assert len(loaded) == 1
        assert loaded[0]["model"] == "kokoro"
