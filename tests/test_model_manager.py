"""Tests for the unified ModelManager."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.model_manager import ModelInfo, ModelLifecycleError, ModelManager, ModelState


class FakeSTTBackend:
    name = "faster-whisper"

    def __init__(self):
        self._models = {}
        self._loaded_at = {}
        self._last_used = {}

    def load_model(self, model_id):
        self._models[model_id] = True
        self._loaded_at[model_id] = time.time()
        self._last_used[model_id] = time.time()

    def unload_model(self, model_id):
        self._models.pop(model_id, None)
        self._loaded_at.pop(model_id, None)
        self._last_used.pop(model_id, None)

    def is_model_loaded(self, model_id):
        return model_id in self._models

    def loaded_models(self):
        from src.models import LoadedModelInfo
        return [
            LoadedModelInfo(
                model=mid, backend="faster-whisper", device="cpu",
                compute_type="int8", loaded_at=self._loaded_at[mid],
                last_used_at=self._last_used.get(mid),
            )
            for mid in self._models
        ]

    def list_cached_models(self):
        return []


class FakeSTTRouter:
    def __init__(self):
        self._default_backend = FakeSTTBackend()
        self._backends = {"faster-whisper": self._default_backend}

    def get_backend(self, model_id):
        return self._default_backend

    def load_model(self, model_id):
        self._default_backend.load_model(model_id)

    def unload_model(self, model_id):
        self._default_backend.unload_model(model_id)

    def is_model_loaded(self, model_id):
        return self._default_backend.is_model_loaded(model_id)

    def loaded_models(self):
        return self._default_backend.loaded_models()

    def list_cached_models(self):
        return self._default_backend.list_cached_models()


class FakeTTSBackend:
    name = "kokoro"

    def __init__(self):
        self._models = {}
        self._loaded_at = {}
        self._last_used = {}

    def load_model(self, model_id):
        self._models[model_id] = True
        self._loaded_at[model_id] = time.time()
        self._last_used[model_id] = time.time()

    def unload_model(self, model_id):
        self._models.pop(model_id, None)

    def is_model_loaded(self, model_id):
        return model_id in self._models

    def loaded_models(self):
        from src.tts.backends.base import TTSLoadedModelInfo
        return [
            TTSLoadedModelInfo(
                model=mid, backend="kokoro", device="cpu",
                loaded_at=self._loaded_at[mid],
                last_used_at=self._last_used.get(mid),
            )
            for mid in self._models
        ]


class FakeTTSRouter:
    def __init__(self):
        self._backends = {"kokoro": FakeTTSBackend()}
        self._default = self._backends["kokoro"]

    def get_backend(self, model_id):
        return self._default

    def load_model(self, model_id):
        self._default.load_model(model_id)

    def unload_model(self, model_id):
        self._default.unload_model(model_id)

    def is_model_loaded(self, model_id):
        return self._default.is_model_loaded(model_id)

    def loaded_models(self):
        return self._default.loaded_models()


@pytest.fixture
def manager():
    stt = FakeSTTRouter()
    tts = FakeTTSRouter()
    return ModelManager(stt_router=stt, tts_router=tts)


class TestModelManagerProviderResolution:
    def test_resolve_provider_pocket_tts(self, manager):
        assert manager.resolve_provider("pocket-tts") == "pocket-tts"


class TestModelManagerLoad:
    def test_load_stt_model(self, manager):
        info = manager.load("Systran/faster-whisper-base")
        assert info.type == "stt"
        assert info.state == ModelState.LOADED
        assert info.provider == "faster-whisper"

    def test_load_tts_model(self, manager):
        info = manager.load("kokoro")
        assert info.type == "tts"
        assert info.state == ModelState.LOADED

    def test_load_returns_model_info(self, manager):
        info = manager.load("Systran/faster-whisper-base")
        assert isinstance(info, ModelInfo)
        assert info.id == "Systran/faster-whisper-base"
        assert info.loaded_at is not None

    def test_load_missing_tts_provider_raises_before_evict(self, manager):
        manager.load("kokoro")

        with pytest.raises(ModelLifecycleError) as exc:
            manager.load("piper/en_US-lessac-medium")

        assert exc.value.code == "provider_missing"
        loaded_ids = [m.id for m in manager.list_loaded()]
        assert "kokoro" in loaded_ids


class TestModelManagerUnload:
    def test_unload_stt_model(self, manager):
        manager.load("Systran/faster-whisper-base")
        manager.unload("Systran/faster-whisper-base")
        loaded = manager.list_loaded()
        assert not any(m.id == "Systran/faster-whisper-base" for m in loaded)

    def test_unload_tts_model(self, manager):
        manager.load("kokoro")
        manager.unload("kokoro")
        loaded = manager.list_loaded()
        assert not any(m.id == "kokoro" for m in loaded)


class TestModelManagerList:
    def test_list_loaded_empty(self, manager):
        assert manager.list_loaded() == []

    def test_list_loaded_after_load(self, manager):
        manager.load("Systran/faster-whisper-base")
        loaded = manager.list_loaded()
        assert len(loaded) == 1
        assert loaded[0].id == "Systran/faster-whisper-base"

    def test_list_all_includes_defaults(self, manager):
        all_models = manager.list_all()
        ids = [m.id for m in all_models]
        assert settings.stt_model in ids
        assert settings.tts_model in ids

    def test_list_all_includes_loaded(self, manager):
        manager.load("Systran/faster-whisper-base")
        all_models = manager.list_all()
        loaded_model = next(m for m in all_models if m.id == "Systran/faster-whisper-base")
        assert loaded_model.state == ModelState.LOADED

    def test_list_all_marks_unregistered_tts_backends_provider_missing(self, manager):
        all_models = manager.list_all()
        piper = next(m for m in all_models if m.id == "piper/en_US-lessac-medium")
        pocket = next(m for m in all_models if m.id == "pocket-tts")

        assert piper.state == ModelState.PROVIDER_MISSING
        assert piper.provider_available is False
        assert pocket.state == ModelState.PROVIDER_MISSING
        assert pocket.provider_available is False


class TestModelManagerStatus:
    def test_status_loaded(self, manager):
        manager.load("Systran/faster-whisper-base")
        info = manager.status("Systran/faster-whisper-base")
        assert info.state == ModelState.LOADED

    def test_status_not_loaded(self, manager):
        info = manager.status("Systran/faster-whisper-base")
        assert info.state in (ModelState.AVAILABLE, ModelState.PROVIDER_INSTALLED)

    def test_status_default_flag(self, manager):
        with patch.object(settings, "stt_model", "Systran/faster-whisper-base"):
            info = manager.status("Systran/faster-whisper-base")
            assert info.is_default is True

    def test_status_marks_unregistered_tts_provider_missing(self, manager):
        info = manager.status("pocket-tts")
        assert info.state == ModelState.PROVIDER_MISSING
        assert info.provider_available is False


class TestModelManagerEviction:
    def test_evict_lru(self, manager):
        manager.load("model-a")
        time.sleep(0.01)
        manager.load("model-b")
        manager.evict_lru()
        loaded = manager.list_loaded()
        ids = [m.id for m in loaded]
        assert "model-a" not in ids
        assert "model-b" in ids

    def test_evict_lru_skips_default(self, manager):
        with patch.object(settings, "stt_model", "model-a"):
            manager.load("model-a")
            time.sleep(0.01)
            manager.load("model-b")
            manager.evict_lru()
            loaded = manager.list_loaded()
            ids = [m.id for m in loaded]
            assert "model-a" in ids
            assert "model-b" not in ids

    def test_check_ttl(self, manager):
        with patch.object(settings, "os_model_ttl", 0):
            manager.load("model-a")
            # TTL=0 means never evict
            manager.check_ttl()
            assert len(manager.list_loaded()) == 1

    def test_check_ttl_evicts_old(self, manager):
        with patch.object(settings, "os_model_ttl", 1):
            manager.load("model-a")
            # Fake old last_used
            for m in manager._stt.loaded_models():
                if m.model == "model-a":
                    manager._stt._default_backend._last_used["model-a"] = time.time() - 10
            manager.check_ttl()
            assert len(manager.list_loaded()) == 0


class TestModelInfoDict:
    def test_to_dict(self):
        info = ModelInfo(
            id="test", type="stt", provider="faster-whisper",
            state=ModelState.LOADED, device="cpu",
        )
        d = info.to_dict()
        assert d["id"] == "test"
        assert d["state"] == "loaded"
        assert d["type"] == "stt"
