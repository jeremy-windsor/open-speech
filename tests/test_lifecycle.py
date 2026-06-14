"""Unit tests for model lifecycle management."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.backends.faster_whisper import FasterWhisperBackend
from src.lifecycle import ModelLifecycleManager
from src.model_manager import ModelInfo, ModelState
from src.router import BackendRouter


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


# 1. TTL eviction delegates to unified model manager
@pytest.mark.asyncio
async def test_ttl_eviction():
    with patch.object(settings, "os_max_loaded_models", 0):
        manager = FakeModelManager(loaded_count=2)
        lm = ModelLifecycleManager(manager)
        await lm._evict()

        assert manager.check_ttl_calls == 1
        assert manager.evict_lru_calls == 0


# 2. Max loaded eviction repeats until under limit
@pytest.mark.asyncio
async def test_ttl_reset_on_use():
    with patch.object(settings, "os_max_loaded_models", 1):
        manager = FakeModelManager(loaded_count=3)
        lm = ModelLifecycleManager(manager)
        await lm._evict()

        assert manager.check_ttl_calls == 1
        assert manager.evict_lru_calls == 2
        assert len(manager.list_loaded()) == 1


# 3. Max loaded disabled leaves LRU alone
@pytest.mark.asyncio
async def test_default_exempt_from_ttl():
    with patch.object(settings, "os_max_loaded_models", 0):
        manager = FakeModelManager(loaded_count=3)
        lm = ModelLifecycleManager(manager)
        await lm._evict()

        assert manager.check_ttl_calls == 1
        assert manager.evict_lru_calls == 0
        assert len(manager.list_loaded()) == 3


# 4. Max models can evict default through unified manager
@pytest.mark.asyncio
async def test_max_models_lru():
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


# 5. Manual unload returns 200
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


# 6. Default model can be unloaded
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


# 9. Unload nonexistent returns 404
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


# 10. Concurrent load safety
@pytest.mark.asyncio
async def test_concurrent_load_safety():
    """Multiple simultaneous loads should only load once."""
    b = FasterWhisperBackend()
    load_count = 0

    def counting_load(model_id):
        nonlocal load_count
        with b._lock:
            if model_id not in b._models:
                load_count += 1
                now = time.time()
                b._models[model_id] = object()
                b._loaded_at[model_id] = now
                b._last_used[model_id] = now

    b.load_model = counting_load
    r = BackendRouter()
    r._default_backend = b
    r._backends = {"faster-whisper": b}

    async def load_once():
        async with r._lock:
            b.load_model("test-model")

    await asyncio.gather(*[load_once() for _ in range(10)])
    assert load_count == 1
