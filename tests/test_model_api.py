"""Integration tests for model management API."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.models import LoadedModelInfo
from src import router as router_module


def _make_mock_backend(**overrides):
    mock = MagicMock()
    mock.name = "faster-whisper"
    mock.loaded_models.return_value = overrides.get("loaded_models", [])
    mock.is_model_loaded.return_value = overrides.get("is_model_loaded", False)
    mock.transcribe.return_value = {"text": "hello"}
    mock.translate.return_value = {"text": "hello"}
    return mock


@pytest.fixture
def client():
    mock_backend = _make_mock_backend()
    with patch.object(router_module.router, "_default_backend", mock_backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}):
        yield TestClient(app), mock_backend


# 11. Load via POST /api/ps/{model}
def test_load_model(client):
    c, mock = client
    resp = c.post("/api/ps/test-model")
    assert resp.status_code == 200
    assert resp.json()["status"] == "loaded"
    mock.load_model.assert_called_with("test-model")


# 12. Unload via DELETE /api/ps/{model}
def test_unload_model(client):
    c, mock = client
    mock.is_model_loaded.return_value = True
    with patch.object(settings, "stt_default_model", "default"):
        resp = c.delete("/api/ps/test-model")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unloaded"


# 13. Transcribe triggers auto-load
def test_transcribe_autoload(client):
    c, mock = client
    audio = b"RIFF" + b"\x00" * 100
    resp = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio, "audio/wav")},
        data={"model": "new-model"},
    )
    assert resp.status_code == 200


# 14. GET /api/ps response has new fields
def test_ps_response_fields():
    now = time.time()
    mock = _make_mock_backend(loaded_models=[
        LoadedModelInfo(
            model="test-model",
            backend="faster-whisper",
            device="cpu",
            compute_type="int8",
            loaded_at=now,
            last_used_at=now,
            is_default=True,
            ttl_remaining=None,
        )
    ])
    with patch.object(router_module.router, "_default_backend", mock), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock}):
        c = TestClient(app)
        resp = c.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        m = data["models"][0]
        assert "last_used_at" in m
        assert "is_default" in m
        assert "ttl_remaining" in m
        assert m["is_default"] is True


# 15. Preload on startup works
def test_preload_on_startup():
    mock = _make_mock_backend()
    with patch.object(router_module.router, "_default_backend", mock), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock}), \
         patch.object(settings, "stt_default_model", "base-model"), \
         patch.object(settings, "stt_preload_models", "base-model,extra-model"):
        # TestClient triggers lifespan
        with TestClient(app):
            calls = [c[0][0] for c in mock.load_model.call_args_list]
            assert "base-model" in calls
            assert "extra-model" in calls
