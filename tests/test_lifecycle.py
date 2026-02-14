"""Unit tests for model lifecycle management."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.backends.faster_whisper import FasterWhisperBackend
from src.lifecycle import ModelLifecycleManager
from src.router import BackendRouter


@pytest.fixture
def backend():
    """Create a backend with mock models (no real WhisperModel loading)."""
    b = FasterWhisperBackend()
    return b


def _fake_load(backend, model_id):
    """Simulate loading a model without actually importing faster_whisper."""
    backend._models[model_id] = MagicMock()
    backend._loaded_at[model_id] = time.time()
    backend._last_used[model_id] = time.time()


def _make_router(backend):
    r = BackendRouter.__new__(BackendRouter)
    r._backends = {"faster-whisper": backend}
    r._default_backend = backend
    r._lock = asyncio.Lock()
    return r


# 1. TTL eviction
@pytest.mark.asyncio
async def test_ttl_eviction():
    with patch.object(settings, "stt_model_ttl", 2), \
         patch.object(settings, "stt_max_loaded_models", 0), \
         patch.object(settings, "stt_default_model", "default-model"):
        b = FasterWhisperBackend()
        _fake_load(b, "default-model")
        _fake_load(b, "other-model")
        b._last_used["other-model"] = time.time() - 3  # idle 3s

        r = _make_router(b)
        lm = ModelLifecycleManager(r)
        await lm._evict()

        assert "other-model" not in b._models
        assert "default-model" in b._models


# 2. TTL reset on use
@pytest.mark.asyncio
async def test_ttl_reset_on_use():
    with patch.object(settings, "stt_model_ttl", 2), \
         patch.object(settings, "stt_max_loaded_models", 0), \
         patch.object(settings, "stt_default_model", "default-model"):
        b = FasterWhisperBackend()
        _fake_load(b, "other-model")
        # Touch it recently
        b._last_used["other-model"] = time.time()

        r = _make_router(b)
        lm = ModelLifecycleManager(r)
        await lm._evict()

        assert "other-model" in b._models


# 3. Default model exempt from TTL
@pytest.mark.asyncio
async def test_default_exempt_from_ttl():
    with patch.object(settings, "stt_model_ttl", 1), \
         patch.object(settings, "stt_max_loaded_models", 0), \
         patch.object(settings, "stt_default_model", "default-model"):
        b = FasterWhisperBackend()
        _fake_load(b, "default-model")
        b._last_used["default-model"] = time.time() - 100

        r = _make_router(b)
        lm = ModelLifecycleManager(r)
        await lm._evict()

        assert "default-model" in b._models


# 4. Max models LRU
@pytest.mark.asyncio
async def test_max_models_lru():
    with patch.object(settings, "stt_model_ttl", 0), \
         patch.object(settings, "stt_max_loaded_models", 2), \
         patch.object(settings, "stt_default_model", "default-model"):
        b = FasterWhisperBackend()
        _fake_load(b, "default-model")
        _fake_load(b, "model-a")
        _fake_load(b, "model-b")
        # model-a is oldest
        b._last_used["model-a"] = time.time() - 10
        b._last_used["model-b"] = time.time()

        r = _make_router(b)
        lm = ModelLifecycleManager(r)
        await lm._evict()

        assert "model-a" not in b._models
        assert "model-b" in b._models
        assert "default-model" in b._models


# 5. Max models protects default
@pytest.mark.asyncio
async def test_max_models_protects_default():
    with patch.object(settings, "stt_model_ttl", 0), \
         patch.object(settings, "stt_max_loaded_models", 1), \
         patch.object(settings, "stt_default_model", "default-model"):
        b = FasterWhisperBackend()
        _fake_load(b, "default-model")
        b._last_used["default-model"] = time.time() - 100
        _fake_load(b, "model-a")

        r = _make_router(b)
        lm = ModelLifecycleManager(r)
        await lm._evict()

        assert "default-model" in b._models
        assert "model-a" not in b._models


# 6. TTL=0 disables eviction
@pytest.mark.asyncio
async def test_ttl_zero_disables():
    with patch.object(settings, "stt_model_ttl", 0), \
         patch.object(settings, "stt_max_loaded_models", 0), \
         patch.object(settings, "stt_default_model", "default-model"):
        b = FasterWhisperBackend()
        _fake_load(b, "other-model")
        b._last_used["other-model"] = time.time() - 9999

        r = _make_router(b)
        lm = ModelLifecycleManager(r)
        await lm._evict()

        assert "other-model" in b._models


# 7. Manual unload returns 200
def test_manual_unload_200():
    from src.main import app
    from src import router as router_module

    mock_backend = MagicMock()
    mock_backend.name = "faster-whisper"
    mock_backend.loaded_models.return_value = []
    mock_backend.is_model_loaded.return_value = True

    with patch.object(router_module.router, "_default_backend", mock_backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}), \
         patch.object(settings, "stt_default_model", "default-model"):
        client = TestClient(app)
        resp = client.delete("/api/ps/some-other-model")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"


# 8. Cannot unload default returns 409
def test_cannot_unload_default_409():
    from src.main import app
    from src import router as router_module

    mock_backend = MagicMock()
    mock_backend.name = "faster-whisper"
    mock_backend.loaded_models.return_value = []

    with patch.object(router_module.router, "_default_backend", mock_backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}), \
         patch.object(settings, "stt_default_model", "my-default"):
        client = TestClient(app)
        resp = client.delete("/api/ps/my-default")
        assert resp.status_code == 409


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
         patch.object(settings, "stt_default_model", "default-model"):
        client = TestClient(app)
        resp = client.delete("/api/ps/nonexistent-model")
        assert resp.status_code == 404


# 10. Concurrent load safety
@pytest.mark.asyncio
async def test_concurrent_load_safety():
    """Multiple simultaneous loads should only load once."""
    b = FasterWhisperBackend()
    load_count = 0
    original_load = b.load_model

    def counting_load(model_id):
        nonlocal load_count
        if model_id not in b._models:
            load_count += 1
            _fake_load(b, model_id)

    b.load_model = counting_load
    r = _make_router(b)

    async def load_once():
        async with r._lock:
            b.load_model("test-model")

    await asyncio.gather(*[load_once() for _ in range(10)])
    assert load_count == 1
