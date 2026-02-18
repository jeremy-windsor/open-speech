"""Tests for model CRUD operations (GET /api/models, DELETE /api/models/{model})."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.router import router as backend_router
from src.config import settings


@pytest.fixture
def fake_cache(tmp_path):
    """Create a fake HuggingFace cache with model dirs."""
    m1 = tmp_path / "models--Systran--faster-whisper-base"
    m1.mkdir()
    (m1 / "model.bin").write_bytes(b"x" * 1024 * 300)

    m2 = tmp_path / "models--Systran--faster-whisper-tiny"
    m2.mkdir()
    (m2 / "model.bin").write_bytes(b"x" * 1024 * 150)

    return tmp_path


@pytest.fixture
def client(fake_cache):
    """TestClient with patched cache dir and default model."""
    backend = backend_router._default_backend
    original_get_cache_dir = backend._get_cache_dir.__func__
    original_default = settings.stt_model

    # Patch cache dir and default model
    backend._get_cache_dir = lambda: fake_cache
    settings.stt_model = "Systran/faster-whisper-base"

    c = TestClient(app, raise_server_exceptions=False)
    yield c, backend, fake_cache

    # Restore
    import types
    backend._get_cache_dir = types.MethodType(original_get_cache_dir, backend)
    settings.stt_model = original_default


class TestListModels:
    def test_list_models_unified(self, client):
        """GET /api/models returns unified model list with id, type, state, etc."""
        c, backend, cache = client
        resp = c.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        model_ids = [m["id"] for m in data["models"]]
        # Default STT model should always appear
        assert "Systran/faster-whisper-base" in model_ids

    def test_list_shows_state(self, client):
        """Models show correct state (downloaded, loaded, available)."""
        c, backend, cache = client
        resp = c.get("/api/models")
        models = resp.json()["models"]
        for m in models:
            assert "state" in m
            assert m["state"] in ("available", "provider_missing", "provider_installed", "downloading", "downloaded", "loaded")

    def test_list_shows_default_flag(self, client):
        c, backend, cache = client
        resp = c.get("/api/models")
        models = resp.json()["models"]
        base = next((m for m in models if m["id"] == "Systran/faster-whisper-base"), None)
        assert base is not None
        assert base["is_default"] is True

    def test_list_shows_type(self, client):
        c, backend, cache = client
        resp = c.get("/api/models")
        models = resp.json()["models"]
        for m in models:
            assert "type" in m
            assert m["type"] in ("stt", "tts")

    def test_list_shows_provider(self, client):
        c, backend, cache = client
        resp = c.get("/api/models")
        models = resp.json()["models"]
        for m in models:
            assert "provider" in m


class TestDeleteModel:
    def test_unload_not_loaded_returns_404(self, client):
        """DELETE /api/models/{id} returns 404 if model not loaded."""
        c, backend, cache = client
        resp = c.delete("/api/models/Systran/faster-whisper-tiny")
        assert resp.status_code == 404

    def test_unload_default_returns_409(self, client):
        """DELETE /api/models/{id} returns 409 for default model."""
        c, backend, cache = client
        # Fake-load the default model
        backend._models["Systran/faster-whisper-base"] = "fake"
        backend._loaded_at["Systran/faster-whisper-base"] = time.time()
        backend._last_used["Systran/faster-whisper-base"] = time.time()

        resp = c.delete("/api/models/Systran/faster-whisper-base")
        assert resp.status_code == 409

        # Cleanup
        del backend._models["Systran/faster-whisper-base"]

    def test_unload_loaded_model(self, client):
        """DELETE /api/models/{id} unloads a loaded model."""
        c, backend, cache = client
        # Fake-load a non-default model
        backend._models["Systran/faster-whisper-tiny"] = "fake"
        backend._loaded_at["Systran/faster-whisper-tiny"] = time.time()
        backend._last_used["Systran/faster-whisper-tiny"] = time.time()

        assert backend.is_model_loaded("Systran/faster-whisper-tiny")

        resp = c.delete("/api/models/Systran/faster-whisper-tiny")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

        assert not backend.is_model_loaded("Systran/faster-whisper-tiny")
