"""Tests for API endpoints (mocked backend)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import router as router_module


@pytest.fixture
def client():
    """Create test client with mocked backend router."""
    mock_backend = MagicMock()
    mock_backend.name = "faster-whisper"
    mock_backend.loaded_models.return_value = []
    mock_backend.is_model_loaded.return_value = False
    mock_backend.transcribe.return_value = {"text": "hello world"}
    mock_backend.translate.return_value = {"text": "hello world"}

    with patch.object(router_module.router, "_default_backend", mock_backend):
        with patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}):
            yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_get_model(client):
    resp = client.get("/v1/models/some-model")
    assert resp.status_code == 200
    assert resp.json()["id"] == "some-model"


def test_transcribe(client):
    audio = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio, "audio/wav")},
        data={"model": "test-model"},
    )
    assert resp.status_code == 200
    assert "text" in resp.json()


def test_translate(client):
    audio = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", audio, "audio/wav")},
        data={"model": "test-model"},
    )
    assert resp.status_code == 200
    assert "text" in resp.json()


def test_loaded_models(client):
    resp = client.get("/api/ps")
    assert resp.status_code == 200
    assert "models" in resp.json()
