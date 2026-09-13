"""Integration tests for model management API."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.model_manager import ModelState
from src.models import LoadedModelInfo
from src import router as router_module
from src.services.models import ModelProgressService


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
    with patch.object(settings, "stt_model", "default"):
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
         patch.object(settings, "stt_model", "base-model"), \
         patch.object(settings, "stt_preload_models", "base-model,extra-model"):
        # TestClient triggers lifespan
        with TestClient(app):
            calls = [c[0][0] for c in mock.load_model.call_args_list]
            assert "base-model" in calls
            assert "extra-model" in calls


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ["load", "download"])
async def test_model_progress_operation_does_not_block_event_loop(operation_name):
    service = ModelProgressService()
    model_manager = MagicMock()
    operation_release = threading.Event()
    operation_result = MagicMock()
    operation_result.to_dict.return_value = {"id": "test-model"}

    def blocking_operation(_model_id):
        if not operation_release.wait(timeout=1):
            raise TimeoutError("test did not release the fake model operation")
        return operation_result

    getattr(model_manager, operation_name).side_effect = blocking_operation

    heartbeat_ran = asyncio.Event()

    async def heartbeat():
        await asyncio.sleep(0)
        heartbeat_ran.set()

    heartbeat_task = asyncio.create_task(heartbeat())
    release_timer = threading.Timer(0.05, operation_release.set)
    release_timer.start()

    try:
        result = await getattr(service, operation_name)(
            model_id="test-model",
            model_manager=model_manager,
        )
        loop_remained_responsive = heartbeat_ran.is_set()
    finally:
        operation_release.set()
        release_timer.cancel()
        release_timer.join(timeout=1)
        await heartbeat_task

    assert result == {"id": "test-model"}
    assert loop_remained_responsive, (
        f"ModelProgressService.{operation_name} blocked the asyncio event loop"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ["get_status", "get_progress"])
async def test_model_status_read_does_not_block_event_loop(operation_name):
    service = ModelProgressService()
    model_manager = MagicMock()
    operation_release = threading.Event()
    model_info = MagicMock()
    model_info.to_dict.return_value = {"id": "test-model", "state": "available"}
    model_info.state = ModelState.AVAILABLE

    def blocking_status(_model_id):
        if not operation_release.wait(timeout=1):
            raise TimeoutError("test did not release the fake status read")
        return model_info

    model_manager.status.side_effect = blocking_status
    heartbeat_ran = asyncio.Event()

    async def heartbeat():
        await asyncio.sleep(0)
        heartbeat_ran.set()

    heartbeat_task = asyncio.create_task(heartbeat())
    release_timer = threading.Timer(0.05, operation_release.set)
    release_timer.start()
    try:
        await getattr(service, operation_name)(
            model_id="test-model",
            model_manager=model_manager,
        )
        loop_remained_responsive = heartbeat_ran.is_set()
    finally:
        operation_release.set()
        release_timer.cancel()
        release_timer.join(timeout=1)
        await heartbeat_task

    assert loop_remained_responsive, (
        f"ModelProgressService.{operation_name} blocked the asyncio event loop"
    )
