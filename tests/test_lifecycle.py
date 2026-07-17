"""Unit tests for model lifecycle management."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.backends.faster_whisper import FasterWhisperBackend
from src.lifecycle import ModelLifecycleManager
from src.model_manager import ModelInfo, ModelState


class FakeModelManager:
    def __init__(self, loaded_count: int = 0) -> None:
        self.check_ttl_calls = 0
        self.evict_lru_calls = 0
        self._loaded = [
            ModelInfo(
                id=f"model-{i}",
                type="stt",
                provider="faster-whisper",
                state=ModelState.LOADED,
            )
            for i in range(loaded_count)
        ]

    def check_ttl(self) -> None:
        self.check_ttl_calls += 1

    def list_loaded(self) -> list[ModelInfo]:
        return list(self._loaded)

    def evict_lru(self) -> None:
        self.evict_lru_calls += 1
        if self._loaded:
            self._loaded.pop(0)


@pytest.mark.asyncio
async def test_eviction_pass_checks_ttl():
    with patch.object(settings, "os_max_loaded_models", 0):
        manager = FakeModelManager(loaded_count=2)
        lm = ModelLifecycleManager(manager)
        await lm._evict()

        assert manager.check_ttl_calls == 1
        assert manager.evict_lru_calls == 0


@pytest.mark.asyncio
async def test_eviction_pass_repeats_lru_until_under_limit():
    with patch.object(settings, "os_max_loaded_models", 1):
        manager = FakeModelManager(loaded_count=3)
        lm = ModelLifecycleManager(manager)
        await lm._evict()

        assert manager.check_ttl_calls == 1
        assert manager.evict_lru_calls == 2
        assert len(manager.list_loaded()) == 1


@pytest.mark.asyncio
async def test_eviction_pass_skips_lru_when_limit_disabled():
    with patch.object(settings, "os_max_loaded_models", 0):
        manager = FakeModelManager(loaded_count=3)
        lm = ModelLifecycleManager(manager)
        await lm._evict()

        assert manager.check_ttl_calls == 1
        assert manager.evict_lru_calls == 0
        assert len(manager.list_loaded()) == 3


@pytest.mark.asyncio
async def test_eviction_pass_can_evict_default_model():
    with patch.object(settings, "os_model_ttl", 0), \
         patch.object(settings, "os_max_loaded_models", 1), \
         patch.object(settings, "stt_model", "default-model"):
        manager = FakeModelManager(loaded_count=2)
        manager._loaded[0].id = "default-model"
        manager._loaded[0].is_default = True
        lm = ModelLifecycleManager(manager)
        await lm._evict()

        assert manager.evict_lru_calls == 1
        assert [m.id for m in manager.list_loaded()] == ["model-1"]


def test_manual_unload_200():
    from src.main import app
    from src import router as router_module

    mock_backend = MagicMock()
    mock_backend.name = "faster-whisper"
    mock_backend.loaded_models.return_value = []
    mock_backend.is_model_loaded.return_value = True

    with patch.object(router_module.router, "_default_backend", mock_backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}), \
         patch.object(settings, "stt_model", "default-model"):
        client = TestClient(app)
        resp = client.delete("/api/ps/some-other-model")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"


def test_unload_default_200():
    from src.main import app
    from src import router as router_module

    mock_backend = MagicMock()
    mock_backend.name = "faster-whisper"
    mock_backend.loaded_models.return_value = []
    mock_backend.is_model_loaded.return_value = True

    with patch.object(router_module.router, "_default_backend", mock_backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}), \
         patch.object(settings, "stt_model", "my-default"):
        client = TestClient(app)
        resp = client.delete("/api/ps/my-default")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"


def test_unload_nonexistent_404():
    from src.main import app
    from src import router as router_module

    mock_backend = MagicMock()
    mock_backend.name = "faster-whisper"
    mock_backend.loaded_models.return_value = []
    mock_backend.is_model_loaded.return_value = False

    with patch.object(router_module.router, "_default_backend", mock_backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}), \
         patch.object(settings, "stt_model", "default-model"):
        client = TestClient(app)
        resp = client.delete("/api/ps/nonexistent-model")
        assert resp.status_code == 404


def test_concurrent_backend_load_constructs_model_once():
    worker_count = 10
    start_barrier = threading.Barrier(worker_count)
    whisper_model = MagicMock(return_value=object())
    faster_whisper_module = ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = whisper_model
    backend = FasterWhisperBackend()

    def load_model() -> None:
        start_barrier.wait(timeout=10)
        backend.load_model("test-model")

    with (
        patch.dict(sys.modules, {"faster_whisper": faster_whisper_module}),
        ThreadPoolExecutor(max_workers=worker_count) as executor,
    ):
        futures = [executor.submit(load_model) for _ in range(worker_count)]
        for future in futures:
            future.result(timeout=10)

    whisper_model.assert_called_once_with(
        "test-model",
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        download_root=settings.stt_model_dir,
    )
    assert backend.is_model_loaded("test-model") is True
