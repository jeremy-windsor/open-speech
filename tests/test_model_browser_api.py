"""Tests for model browser API — list available, load, download states."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.model_manager import ModelManager, ModelState, ModelInfo
from src.model_registry import KNOWN_MODELS


class FakeSTTRouter:
    def __init__(self):
        self._loaded = {}
        self._backends = {"faster-whisper": MagicMock(name="faster-whisper")}

    def get_backend(self, model_id):
        b = MagicMock()
        b.name = "faster-whisper"
        return b

    def load_model(self, model_id):
        self._loaded[model_id] = time.time()

    def unload_model(self, model_id):
        self._loaded.pop(model_id, None)

    def loaded_models(self):
        from src.models import LoadedModelInfo
        return [
            LoadedModelInfo(model=m, backend="faster-whisper", device="cpu",
                          compute_type="int8", loaded_at=t, last_used_at=t)
            for m, t in self._loaded.items()
        ]

    def list_cached_models(self):
        return []


class FakeTTSRouter:
    def __init__(self):
        self._loaded = {}
        self._backends = {"kokoro": MagicMock(name="kokoro"), "piper": MagicMock(name="piper")}

    def get_backend(self, model_id):
        if model_id.startswith("piper/"):
            b = MagicMock()
            b.name = "piper"
            return b
        b = MagicMock()
        b.name = "kokoro"
        return b

    def load_model(self, model_id):
        self._loaded[model_id] = time.time()

    def unload_model(self, model_id):
        self._loaded.pop(model_id, None)

    def loaded_models(self):
        from src.tts.backends.base import TTSLoadedModelInfo
        return [
            TTSLoadedModelInfo(model=m, backend="piper" if m.startswith("piper/") else "kokoro",
                              device="cpu", loaded_at=t, last_used_at=t)
            for m, t in self._loaded.items()
        ]


class TestListAllIncludesRegistry:
    def test_list_all_includes_piper_models(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        models = mm.list_all()
        ids = {m.id for m in models}
        assert "piper/en_US-lessac-medium" in ids
        assert "piper/en_GB-alan-medium" in ids
        assert "kokoro" in ids

    def test_list_all_includes_stt_models(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        models = mm.list_all()
        ids = {m.id for m in models}
        assert "Systran/faster-whisper-tiny" in ids
        assert "Systran/faster-whisper-base" in ids

    def test_available_models_have_size(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        models = mm.list_all()
        piper_m = next(m for m in models if m.id == "piper/en_US-lessac-medium")
        assert piper_m.size_mb == 35
        assert piper_m.state in (ModelState.AVAILABLE, ModelState.PROVIDER_INSTALLED, ModelState.PROVIDER_MISSING)

    def test_list_all_ignores_unknown_cached_hf_repos(self):
        stt = FakeSTTRouter()
        stt.list_cached_models = lambda: [{"model": "kyutai/pocket-tts-without-voice-cloning", "backend": "faster-whisper"}]
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        models = mm.list_all()
        ids = {m.id for m in models}
        assert "kyutai/pocket-tts-without-voice-cloning" not in ids

    def test_loaded_model_overrides_registry(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        with patch("src.model_manager._check_provider", return_value=True):
            mm.load("piper/en_US-lessac-medium")
        models = mm.list_all()
        piper_m = next(m for m in models if m.id == "piper/en_US-lessac-medium")
        assert piper_m.state == ModelState.LOADED

    def test_description_in_to_dict(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        models = mm.list_all()
        piper_m = next(m for m in models if m.id == "piper/en_US-lessac-medium")
        d = piper_m.to_dict()
        assert "description" in d
        assert d["description"] == "US English - Lessac voice"


class TestProviderResolution:
    def test_piper_prefix_resolves(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        assert mm._resolve_type("piper/en_US-lessac-medium") == "tts"

    def test_kokoro_resolves(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        assert mm._resolve_type("kokoro") == "tts"

    def test_whisper_resolves_stt(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        assert mm._resolve_type("Systran/faster-whisper-tiny") == "stt"

    def test_base_resolves_stt(self):
        stt = FakeSTTRouter()
        tts = FakeTTSRouter()
        mm = ModelManager(stt_router=stt, tts_router=tts)
        assert mm._resolve_type("Systran/faster-whisper-base") == "stt"
