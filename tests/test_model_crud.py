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
    original_default = settings.stt_default_model

    # Patch cache dir and default model
    backend._get_cache_dir = lambda: fake_cache
    settings.stt_default_model = "Systran/faster-whisper-base"

    c = TestClient(app, raise_server_exceptions=False)
    yield c, backend, fake_cache

    # Restore
    import types
    backend._get_cache_dir = types.MethodType(original_get_cache_dir, backend)
    settings.stt_default_model = original_default


class TestListModels:
    def test_list_cached_models(self, client):
        c, backend, cache = client
        resp = c.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        model_names = [m["model"] for m in data["models"]]
        assert "Systran/faster-whisper-base" in model_names
        assert "Systran/faster-whisper-tiny" in model_names

    def test_list_shows_loaded_status(self, client):
        c, backend, cache = client
        resp = c.get("/api/models")
        models = resp.json()["models"]
        for m in models:
            assert m["loaded"] is False

    def test_list_shows_default_flag(self, client):
        c, backend, cache = client
        resp = c.get("/api/models")
        models = resp.json()["models"]
        base = next(m for m in models if m["model"] == "Systran/faster-whisper-base")
        tiny = next(m for m in models if m["model"] == "Systran/faster-whisper-tiny")
        assert base["is_default"] is True
        assert tiny["is_default"] is False

    def test_list_shows_size(self, client):
        c, backend, cache = client
        resp = c.get("/api/models")
        models = resp.json()["models"]
        for m in models:
            assert "size_mb" in m
            assert m["size_mb"] >= 0


class TestDeleteModel:
    def test_delete_nonexistent_returns_404(self, client):
        c, backend, cache = client
        resp = c.delete("/api/models/Systran/nonexistent-model")
        assert resp.status_code == 404

    def test_delete_default_returns_409(self, client):
        c, backend, cache = client
        resp = c.delete("/api/models/Systran/faster-whisper-base")
        assert resp.status_code == 409

    def test_delete_cached_model(self, client):
        c, backend, cache = client
        # Verify it exists
        resp = c.get("/api/models")
        model_names = [m["model"] for m in resp.json()["models"]]
        assert "Systran/faster-whisper-tiny" in model_names

        # Delete it
        resp = c.delete("/api/models/Systran/faster-whisper-tiny")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp = c.get("/api/models")
        model_names = [m["model"] for m in resp.json()["models"]]
        assert "Systran/faster-whisper-tiny" not in model_names

    def test_delete_unloads_first(self, client):
        c, backend, cache = client
        # Fake-load the model
        backend._models["Systran/faster-whisper-tiny"] = "fake"
        backend._loaded_at["Systran/faster-whisper-tiny"] = time.time()
        backend._last_used["Systran/faster-whisper-tiny"] = time.time()

        assert backend.is_model_loaded("Systran/faster-whisper-tiny")

        resp = c.delete("/api/models/Systran/faster-whisper-tiny")
        assert resp.status_code == 200

        assert not backend.is_model_loaded("Systran/faster-whisper-tiny")
        assert not (cache / "models--Systran--faster-whisper-tiny").exists()
