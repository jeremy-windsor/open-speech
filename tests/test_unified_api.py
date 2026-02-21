"""Tests for unified /api/models endpoints."""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app, model_manager
from src.router import router as backend_router
from src.config import settings


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestGetModels:
    def test_returns_models_list(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_includes_default_models(self, client):
        resp = client.get("/api/models")
        ids = [m["id"] for m in resp.json()["models"]]
        assert settings.stt_model in ids

    def test_model_has_required_fields(self, client):
        resp = client.get("/api/models")
        for m in resp.json()["models"]:
            assert "id" in m
            assert "type" in m
            assert "provider" in m
            assert "state" in m
            assert "is_default" in m


class TestGetModelStatus:
    def test_status_of_known_model(self, client):
        resp = client.get(f"/api/models/{settings.stt_model}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == settings.stt_model
        assert data["type"] == "stt"

    def test_status_of_unknown_model(self, client):
        resp = client.get("/api/models/nonexistent-model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] in ("available", "provider_installed")


class TestLoadModel:
    def test_load_model_endpoint(self, client):
        """POST /api/models/{id}/load loads a model."""
        with patch.object(model_manager, "load") as mock_load:
            from src.model_manager import ModelInfo, ModelState
            mock_load.return_value = ModelInfo(
                id="test-model", type="stt", provider="faster-whisper",
                state=ModelState.LOADED,
            )
            resp = client.post("/api/models/test-model/load")
            assert resp.status_code == 200
            assert resp.json()["state"] == "loaded"
            mock_load.assert_called_once_with("test-model")

    def test_load_model_error(self, client):
        with patch.object(model_manager, "load", side_effect=RuntimeError("fail")):
            resp = client.post("/api/models/test-model/load")
            assert resp.status_code == 500


class TestPrefetchModel:
    def test_prefetch_model_endpoint(self, client):
        with patch.object(model_manager, "download") as mock_download:
            from src.model_manager import ModelInfo, ModelState
            mock_download.return_value = ModelInfo(
                id="test-model", type="tts", provider="piper",
                state=ModelState.DOWNLOADED,
            )
            resp = client.post("/api/models/test-model/prefetch")
            assert resp.status_code == 200
            assert resp.json()["state"] == "downloaded"
            mock_download.assert_called_once_with("test-model")

    def test_prefetch_model_error(self, client):
        with patch.object(model_manager, "download", side_effect=RuntimeError("fail")):
            resp = client.post("/api/models/test-model/prefetch")
            assert resp.status_code == 500


class TestUnloadModel:
    def test_unload_not_loaded(self, client):
        resp = client.delete("/api/models/not-loaded-model")
        assert resp.status_code == 404

    def test_unload_default_allowed(self, client):
        """Default model can be unloaded."""
        with patch.object(model_manager, "status") as mock_status, \
             patch.object(model_manager, "unload") as mock_unload:
            from src.model_manager import ModelInfo, ModelState
            mock_status.return_value = ModelInfo(
                id=settings.stt_model, type="stt", provider="faster-whisper",
                state=ModelState.LOADED, is_default=True,
            )
            resp = client.delete(f"/api/models/{settings.stt_model}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "unloaded"
            mock_unload.assert_called_once_with(settings.stt_model)
