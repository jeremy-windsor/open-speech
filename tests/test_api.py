"""Tests for API endpoints (mocked backend)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
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
    assert data["version"] == main_module.get_runtime_version()


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_get_model(client):
    resp = client.get("/v1/models/some-model")
    assert resp.status_code == 404


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


def test_web_page_and_static_files_are_revalidated(client):
    # Without this, Safari can keep running an old reader.js after a deploy (heuristic caching).
    page = client.get("/web")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-cache"
    script = client.get("/static/reader.js")
    assert script.status_code == 200
    assert script.headers["cache-control"] == "no-cache"
    unchanged = client.get("/static/reader.js", headers={"If-None-Match": script.headers["etag"]})
    assert unchanged.status_code == 304
    assert unchanged.headers["cache-control"] == "no-cache"
