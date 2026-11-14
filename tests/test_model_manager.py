"""Tests for the unified ModelManager."""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import urllib.error
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.config import settings
from src.model_manager import ModelInfo, ModelLifecycleError, ModelManager, ModelState
from src.services.models import ModelProgressService, list_all_models
from src.tts.external import ExternalProviderError


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


class FakeKokoroBackend(FakeTTSBackend):
    @staticmethod
    def supports_model(model_id):
        return model_id == "kokoro"  # KokoroBackend.supports_model


class FakePiperBackend(FakeTTSBackend):
    name = "piper"

    @staticmethod
    def supports_model(model_id):
        from src.tts.backends.piper_backend import PIPER_MODELS
        return model_id in PIPER_MODELS  # PiperBackend.supports_model


class FakeOfflineWorker(FakeTTSBackend):
    """Isolated provider whose worker is down, so it cannot tell which of its IDs are real."""

    name = "qwen3"

    def supports_model(self, model_id):
        return model_id.startswith(f"{self.name}/")  # ExternalTTSBackend.supports_model, worker down

    def is_provider_available(self):
        return False


def _tts_router(**backends):
    """Real TTSRouter (its prefix routing) holding only the given fake backends."""
    from src.tts.router import TTSRouter

    with patch("src.tts.router._discover_backends", return_value={}):
        tts = TTSRouter(device="cpu", external_providers="")
    for name, backend in backends.items():
        tts.register_backend(name, backend)
    return tts


@pytest.fixture
def manager():
    stt = FakeSTTRouter()
    tts = FakeTTSRouter()
    return ModelManager(stt_router=stt, tts_router=tts)


@pytest.fixture
def hf_cache(manager, tmp_path):
    """Point the manager's artifact scan at an empty cache instead of this machine's."""
    with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
        yield tmp_path


class TestModelManagerProviderResolution:
    def test_resolve_provider_pocket_tts(self, manager):
        assert manager.resolve_provider("pocket-tts") == "pocket-tts"

    def test_qwen_manifest_models_are_tts_even_without_worker(self, manager):
        assert manager._resolve_type("qwen3/0.6b-custom-voice") == "tts"
        assert manager.resolve_provider("qwen3/0.6b-custom-voice") == "qwen3"

    def test_unknown_model_is_not_classified_as_stt(self, manager):
        with pytest.raises(ModelLifecycleError) as caught:
            manager._resolve_type("unregistered/new-voice")
        assert caught.value.code == "unknown_model"
        assert "Unknown model" in caught.value.message


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
        assert "Rebuild with BAKED_PROVIDERS including 'piper'" in exc.value.message

    # fish/s2-pro: its registry row.
    @pytest.mark.parametrize("model_id", ["qwen3/0.6b-custom-voice", "chatterbox/turbo", "fish/s2-pro"])
    @pytest.mark.parametrize("operation", ["load", "download"])
    def test_unconfigured_worker_model_says_to_run_its_isolated_worker(self, manager, model_id, operation):
        # The Dockerfile rejects the cloning providers' names in BAKED_PROVIDERS: they run as
        # isolated workers (docker-compose.voice-models.yml) listed in TTS_EXTERNAL_PROVIDERS.
        with pytest.raises(ModelLifecycleError) as caught:
            getattr(manager, operation)(model_id)

        provider = model_id.split("/", 1)[0]
        assert caught.value.code == "provider_missing"
        assert caught.value.provider == provider
        assert "BAKED_PROVIDERS" not in caught.value.message
        assert "docker-compose.voice-models.yml" in caught.value.message
        assert "TTS_EXTERNAL_PROVIDERS" in caught.value.message
        assert f"'{provider}'" in caught.value.message

    def test_unadvertised_model_does_not_evict_kokoro(self, manager):
        manager.load("kokoro")
        manager._tts._backends["qwen3"] = object()
        manager._tts.provider_is_available = lambda _provider: True
        manager._tts.get_capabilities = lambda _model: (_ for _ in ()).throw(ValueError("Unknown qwen3 TTS model"))

        with pytest.raises(ModelLifecycleError) as caught:
            manager.load("qwen3/0.6b-base")

        assert caught.value.code == "unknown_model"
        assert any(model.id == "kokoro" for model in manager.list_loaded())

    def test_unknown_external_model_does_not_route_to_stt_or_evict_kokoro(self, manager):
        manager.load("kokoro")
        manager._tts._backends["qwen3"] = object()
        manager._tts.provider_is_available = lambda _provider: True
        manager._tts.get_capabilities = lambda _model: (_ for _ in ()).throw(ValueError("Not advertised"))

        with pytest.raises(ModelLifecycleError) as caught:
            manager.load("qwen3/not-advertised")

        assert caught.value.code == "unknown_model"
        assert manager._stt.loaded_models() == []
        assert any(model.id == "kokoro" for model in manager.list_loaded())

    def test_failed_unload_prevents_second_tts_model_from_loading(self, manager):
        manager.load("kokoro")
        manager._tts._backends["qwen3"] = object()
        manager._tts.provider_is_available = lambda _provider: True
        manager._tts.get_capabilities = lambda _model: {}
        original_unload = manager._tts.unload_model

        def fail_kokoro_unload(model_id):
            if model_id == "kokoro":
                raise RuntimeError("GPU model could not unload")
            original_unload(model_id)

        manager._tts.unload_model = fail_kokoro_unload
        # Unpinned: a TTS load leaves a pinned Kokoro loaded.
        with patch.object(settings, "os_pin_default_models", False), pytest.raises(ModelLifecycleError) as caught:
            manager.load("qwen3/0.6b-custom-voice")

        assert caught.value.code == "unload_failed"
        assert [model.id for model in manager.list_loaded() if model.type == "tts"] == ["kokoro"]


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


class TestModelManagerDownload:
    def test_download_restores_active_model_for_single_resident_provider(self, manager):
        backend = manager._tts._default
        manager._tts._backends["chatterbox"] = backend
        manager._tts.provider_is_available = lambda provider: provider == "chatterbox"
        manager._tts.get_capabilities = lambda _model: {}

        def load_single_model(model_id):
            backend._models.clear()
            backend.load_model(model_id)

        manager._tts.load_model = load_single_model
        active_model = "chatterbox/regular"
        download_model = "chatterbox/turbo"
        manager.load(active_model)

        manager.download(download_model)

        loaded_ids = {model.id for model in manager.list_loaded()}
        assert active_model in loaded_ids
        assert download_model not in loaded_ids

    def test_stt_download_fetches_the_files_without_loading_the_model(self, tmp_path, monkeypatch):
        # Download in Models caches weights; Load to GPU activates them. A load next to the active
        # STT model held both in memory (about 3 GB of RAM for large-v3 on CPU).
        import faster_whisper
        import faster_whisper.utils

        from src.backends.faster_whisper import FasterWhisperBackend
        from src.router import BackendRouter

        fetched, constructed = [], []
        monkeypatch.setattr(
            faster_whisper.utils, "download_model",
            lambda model_id, cache_dir=None: fetched.append((model_id, cache_dir)),
        )
        monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *args, **kwargs: constructed.append(args))
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(settings, "stt_model_dir", "")  # as Compose passes STT_MODEL_DIR when unset
        stt = BackendRouter()
        backend: FasterWhisperBackend = stt.get_backend("Systran/faster-whisper-base")
        backend._models["Systran/faster-whisper-base"] = object()  # the active STT model
        backend._loaded_at["Systran/faster-whisper-base"] = time.time()
        manager = ModelManager(stt_router=stt, tts_router=FakeTTSRouter())

        manager.download("Systran/faster-whisper-small")

        assert fetched == [("Systran/faster-whisper-small", None)]
        assert constructed == []
        assert [m.id for m in manager.list_loaded()] == ["Systran/faster-whisper-base"]

    def test_kokoro_download_fetches_weights_and_voices_without_loading(self, tmp_path, monkeypatch):
        import sys
        from types import ModuleType

        import huggingface_hub

        from src.tts.backends.kokoro import KokoroBackend

        fetched, pipelines = [], []
        monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: fetched.append(kwargs))
        kokoro_package = ModuleType("kokoro")
        kokoro_package.KPipeline = lambda **kwargs: pipelines.append(kwargs)
        monkeypatch.setitem(sys.modules, "kokoro", kokoro_package)
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(kokoro=KokoroBackend(device="cpu")))

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            manager.download("kokoro")

        assert fetched == [{
            "repo_id": "hexgrad/Kokoro-82M",
            "allow_patterns": ["config.json", "kokoro-v1_0.pth", "voices/*.pt"],
        }]
        assert pipelines == [], "the download built a Kokoro pipeline"
        assert manager.list_loaded() == []

    def test_piper_download_fetches_the_voice_files_without_loading(self, tmp_path, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        import huggingface_hub

        from src.tts.backends.piper_backend import PiperBackend

        fetched, voices = [], []

        def hf_hub_download(repo_id, filename):
            fetched.append((repo_id, filename))
            return str(tmp_path / filename)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
        piper_package = ModuleType("piper")
        piper_package.PiperVoice = SimpleNamespace(load=lambda *args, **kwargs: voices.append(args))
        monkeypatch.setitem(sys.modules, "piper", piper_package)
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(piper=PiperBackend(device="cpu")))

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            manager.download("piper/en_US-lessac-medium")

        assert fetched == [
            ("rhasspy/piper-voices", "en/en_US/lessac/medium/en_US-lessac-medium.onnx"),
            ("rhasspy/piper-voices", "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"),
        ]
        assert voices == [], "the download loaded the Piper voice"
        assert manager.list_loaded() == []

    def test_download_uses_the_backends_download_hook_instead_of_a_load(self, tmp_path):
        # The optional TTSBackend.download_model hook, not a check for Kokoro or Piper, decides
        # whether a download can skip the load (a load next to the active model, then an unload).
        class DownloadingBackend(FakePiperBackend):
            def __init__(self):
                super().__init__()
                self.downloaded, self.loads = [], []

            def load_model(self, model_id):
                self.loads.append(model_id)
                super().load_model(model_id)

            def download_model(self, model_id):
                self.downloaded.append(model_id)

        backend = DownloadingBackend()
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(piper=backend))

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            info = manager.download("piper/en_US-ryan-medium")

        assert backend.downloaded == ["piper/en_US-ryan-medium"]
        assert backend.loads == [], "the download loaded the model to fetch it"
        assert info.provider == "piper"
        assert manager.list_loaded() == []

    def test_download_skips_the_hook_for_a_loaded_model(self, tmp_path):
        class DownloadingBackend(FakeKokoroBackend):
            def __init__(self):
                super().__init__()
                self.downloaded = []

            def download_model(self, model_id):
                self.downloaded.append(model_id)

        backend = DownloadingBackend()
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(kokoro=backend))
        manager.load("kokoro")

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            manager.download("kokoro")

        assert backend.downloaded == []  # its weights are in the cache already
        assert [m.id for m in manager.list_loaded()] == ["kokoro"]

    def test_download_without_a_hook_loads_once_and_unloads(self, tmp_path):
        # Pocket-TTS picks its repository while it loads and the isolated workers have no
        # download-only call, so a backend without download_model is fetched by a load.
        class LoadOnlyBackend(FakePiperBackend):
            def __init__(self):
                super().__init__()
                self.loads = []

            def load_model(self, model_id):
                self.loads.append(model_id)
                super().load_model(model_id)

        backend = LoadOnlyBackend()
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(piper=backend))

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            manager.download("piper/en_US-ryan-medium")

        assert backend.loads == ["piper/en_US-ryan-medium"]
        assert manager.list_loaded() == []


class TestModelManagerDefaultPreload:
    """TTS_PRELOAD_DEFAULT: the first speech request should find TTS_MODEL loaded."""

    def test_installed_default_tts_model_is_loaded(self, manager, hf_cache):
        with patch.object(settings, "tts_model", "kokoro"):
            assert manager.warm_default_tts() is True

        assert [m.id for m in manager.list_loaded()] == ["kokoro"]

    def test_default_of_a_missing_provider_is_not_loaded(self, manager, hf_cache):
        with patch.object(settings, "tts_model", "pocket-tts"):  # only Kokoro is installed
            assert manager.warm_default_tts() is False

        assert manager.list_loaded() == []

    def test_loaded_default_is_left_as_it_is(self, manager, hf_cache):
        with patch.object(settings, "tts_model", "kokoro"):
            manager.load("kokoro")
            loaded_at = manager.list_loaded()[0].loaded_at
            assert manager.warm_default_tts() is False

        assert [(m.id, m.loaded_at) for m in manager.list_loaded()] == [("kokoro", loaded_at)]

    def test_default_of_an_offline_worker_is_left_for_a_later_try(self, tmp_path):
        # Workers start on their own, often after this server: None tells startup to try again.
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(qwen3=FakeOfflineWorker()))

        with patch.object(settings, "tts_model", "qwen3/0.6b-custom-voice"), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            assert manager.warm_default_tts() is None

        assert manager.list_loaded() == []

    def test_first_start_download_does_not_hold_up_a_tts_switch(self, tmp_path, monkeypatch):
        # On a first start the warm-up downloads Kokoro's weights, which can take minutes. A TTS
        # switch meanwhile (UI Load, /v1/audio/models/load) must not wait for that download.
        import sys
        from types import ModuleType

        import huggingface_hub

        from src.tts.backends.kokoro import KokoroBackend

        downloading, release = threading.Event(), threading.Event()

        def first_fetch_downloads(*args, **kwargs):
            # snapshot_download, or KPipeline when its load finds no weights in the cache
            if not downloading.is_set():
                downloading.set()
                release.wait(timeout=5)
            return object()

        monkeypatch.setattr(huggingface_hub, "snapshot_download", first_fetch_downloads)
        kokoro_package = ModuleType("kokoro")
        kokoro_package.KPipeline = first_fetch_downloads
        monkeypatch.setitem(sys.modules, "kokoro", kokoro_package)
        tts = _tts_router(kokoro=KokoroBackend(device="cpu"), piper=FakePiperBackend())
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=tts)

        with patch.object(settings, "tts_model", "kokoro"), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            warm_up = threading.Thread(target=manager.warm_default_tts)
            warm_up.start()
            assert downloading.wait(timeout=5)
            switch = threading.Thread(target=manager.load, args=("piper/en_US-lessac-medium",))
            switch.start()
            switch.join(timeout=2)
            switched_during_download = not switch.is_alive()
            release.set()
            warm_up.join(timeout=5)
            switch.join(timeout=5)
            loaded = {m.id for m in manager.list_loaded()}

        assert switched_during_download, "the TTS switch waited for the default model's download"
        assert "kokoro" in loaded, "the warm-up did not load the default after its download"


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

    def test_inventory_advertises_configured_reading_default(self, manager):
        inventory = list_all_models(
            model_manager=manager,
            tts_capabilities_for=lambda _model: {},
            default_stt_model=settings.stt_model,
            default_tts_model=settings.tts_model,
        )
        assert inventory["default_tts_model"] == "kokoro"

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

    def test_optional_qwen_models_are_hidden_until_worker_is_configured(self, manager):
        ids = {model.id for model in manager.list_all()}
        assert "qwen3/0.6b-custom-voice" not in ids
        assert "qwen3/0.6b-base" not in ids

    def test_configured_but_unhealthy_worker_is_explicit(self, manager):
        manager._tts._backends["qwen3"] = object()
        manager._tts.provider_is_available = lambda provider: provider != "qwen3"

        qwen = next(
            model
            for model in manager.list_all()
            if model.id == "qwen3/0.6b-custom-voice"
        )

        assert qwen.state == ModelState.PROVIDER_UNAVAILABLE
        assert qwen.provider_available is False

    def test_optional_catalog_requires_exact_worker_model(self, manager):
        class Worker:
            def advertises_model(self, model_id):
                return model_id == "qwen3/0.6b-custom-voice"

        manager._tts._backends["qwen3"] = Worker()
        manager._tts.provider_is_available = lambda _provider: True

        ids = {model.id for model in manager.list_all()}
        assert "kokoro" in ids
        assert "qwen3/0.6b-custom-voice" in ids
        assert "qwen3/0.6b-base" not in ids
        assert manager.status("qwen3/0.6b-base").state == ModelState.PROVIDER_UNAVAILABLE

        with patch.object(settings, "tts_model", "qwen3/0.6b-base"):
            base = next(model for model in manager.list_all() if model.id == "qwen3/0.6b-base")
            assert base.state == ModelState.PROVIDER_UNAVAILABLE
            assert base.provider_available is False

    def test_inventory_and_status_agree_on_cached_custom_default_stt_model(self, manager):
        # Any faster-whisper repo can be STT_MODEL, inside the curated registry or not.
        manager._stt._default_backend.list_cached_models = lambda: [
            {"model": "acme/whisper-custom-ct2", "backend": "faster-whisper"},
        ]
        with patch.object(settings, "stt_model", "acme/whisper-custom-ct2"):
            status_state = manager.status("acme/whisper-custom-ct2").state
            listed = next(m for m in manager.list_all() if m.id == "acme/whisper-custom-ct2")

        assert status_state == ModelState.DOWNLOADED
        assert listed.state == ModelState.DOWNLOADED


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

    def test_unknown_cached_stt_model_is_not_offered_as_ready(self, manager):
        manager._stt._default_backend.list_cached_models = lambda: [{"model": "unknown/model"}]
        with pytest.raises(ModelLifecycleError) as caught:
            manager.status("unknown/model")
        assert caught.value.code == "unknown_model"

    def test_unified_status_and_load_reject_unknown_model(self, manager):
        service = ModelProgressService()

        async def check():
            with pytest.raises(HTTPException) as status_error:
                await service.get_status(model_id="unknown/model", model_manager=manager)
            assert status_error.value.status_code == 404
            assert status_error.value.detail["code"] == "unknown_model"

            with pytest.raises(HTTPException) as load_error:
                await service.load(model_id="unknown/model", model_manager=manager)
            assert load_error.value.status_code == 404
            assert load_error.value.detail["code"] == "unknown_model"
            assert "unknown/model" not in service.download_progress

        asyncio.run(check())

    def test_stt_model_cached_under_hf_home_is_downloaded(self, manager, tmp_path, monkeypatch):
        # huggingface_hub caches under $HF_HOME/hub when HF_HUB_CACHE is unset.
        from src.backends.faster_whisper import FasterWhisperBackend

        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        # Keep this machine's own ~/.cache/huggingface out of the answer.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        monkeypatch.setattr(settings, "stt_model_dir", None)
        snapshot = tmp_path / "hf" / "hub" / "models--Systran--faster-whisper-small" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"x" * 1024)
        manager._stt._default_backend = FasterWhisperBackend()

        assert manager.status("Systran/faster-whisper-small").state == ModelState.DOWNLOADED

    @pytest.mark.parametrize("layout", ["hf_home_with_tilde", "xdg_cache_home"])
    def test_models_are_found_where_huggingface_hub_downloads_them(self, tmp_path, monkeypatch, layout):
        # huggingface_hub.constants: HF_HOME = expandvars(expanduser(getenv("HF_HOME",
        # join(getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")))), the hub cache HF_HOME/hub.
        # faster-whisper (without STT_MODEL_DIR), Kokoro and Piper download there.
        from src.backends.faster_whisper import FasterWhisperBackend

        home = tmp_path / "home"
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setattr(settings, "stt_model_dir", None)
        if layout == "hf_home_with_tilde":
            monkeypatch.setenv("HF_HOME", "~/hf")  # systemd Environment= and .env files keep the '~'
            hub = home / "hf" / "hub"
        else:
            monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
            hub = tmp_path / "xdg" / "huggingface" / "hub"
        whisper = hub / "models--Systran--faster-whisper-small" / "snapshots" / "abc"
        kokoro = hub / "models--hexgrad--Kokoro-82M" / "snapshots" / "abc"
        for snapshot in (whisper, kokoro):
            snapshot.mkdir(parents=True)
            (snapshot / "weights.bin").write_bytes(b"x")
        stt = FakeSTTRouter()
        stt._default_backend = FasterWhisperBackend()
        manager = ModelManager(stt_router=stt, tts_router=_tts_router(kokoro=FakeKokoroBackend()))

        assert manager.status("Systran/faster-whisper-small").state == ModelState.DOWNLOADED
        assert manager.status("kokoro").state == ModelState.DOWNLOADED
        assert stt._default_backend.delete_cached_model("Systran/faster-whisper-small") is True
        assert manager.delete_artifacts("kokoro")["status"] == "deleted"
        assert not whisper.exists() and not kokoro.exists()

    def test_cached_pocket_tts_weights_are_downloaded_and_deletable(self, tmp_path):
        # pocket-tts downloads kyutai/pocket-tts-without-voice-cloning (or kyutai/pocket-tts once
        # its terms are accepted); no download creates a models--pocket-tts directory.
        class FakePocketBackend(FakeTTSBackend):
            name = "pocket-tts"

        snapshot = tmp_path / "models--kyutai--pocket-tts-without-voice-cloning" / "snapshots" / "d4fdd22"
        snapshot.mkdir(parents=True)
        (snapshot / "tts_b6369a24.safetensors").write_bytes(b"weights")
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(**{"pocket-tts": FakePocketBackend()}))

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            state = manager.status("pocket-tts").state
            result = manager.delete_artifacts("pocket-tts")

        assert state == ModelState.DOWNLOADED
        assert result["status"] == "deleted"
        assert not snapshot.exists()


class TestModelProgressStatus:
    """A finished load or download must not outrank the live model state."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("unload_via", ["service", "manager"])
    async def test_progress_is_not_ready_after_loaded_model_is_unloaded(self, manager, hf_cache, unload_via):
        # TTL and LRU eviction unload through the manager, past the service.
        service = ModelProgressService()
        await service.load(model_id="kokoro", model_manager=manager)
        if unload_via == "service":
            await service.unload(model_id="kokoro", model_manager=manager)
        else:
            manager.unload("kokoro")

        progress = await service.get_progress(model_id="kokoro", model_manager=manager)

        assert progress["status"] == "downloaded"

    @pytest.mark.asyncio
    async def test_status_is_not_downloaded_after_artifacts_are_deleted(self, manager, hf_cache):
        weights = hf_cache / "models--hexgrad--Kokoro-82M"
        weights.mkdir()
        (weights / "model.pth").write_bytes(b"x")
        service = ModelProgressService()

        await service.download(model_id="kokoro", model_manager=manager)
        result = await service.delete_artifacts(model_id="kokoro", model_manager=manager)
        status = await service.get_status(model_id="kokoro", model_manager=manager)

        assert result["status"] == "deleted"
        assert status["state"] == "provider_installed"

    @pytest.mark.asyncio
    async def test_status_trusts_a_finished_download_the_cache_scan_cannot_see(self, manager, hf_cache):
        # Isolated workers keep weights in their own volume; the UI download poll waits for "downloaded".
        service = ModelProgressService()
        await service.download(model_id="kokoro", model_manager=manager)
        assert manager.status("kokoro").state == ModelState.PROVIDER_INSTALLED

        status = await service.get_status(model_id="kokoro", model_manager=manager)

        assert status["state"] == "downloaded"

    @pytest.mark.asyncio
    async def test_status_reports_provider_unavailable_after_worker_goes_offline(self, manager, hf_cache):
        # The UI shows "provider is offline" only for state provider_unavailable.
        service = ModelProgressService()
        await service.load(model_id="kokoro", model_manager=manager)
        manager._tts._default._models.clear()  # the worker crashed and lost its model
        manager._tts.provider_is_available = lambda _provider: False

        status = await service.get_status(model_id="kokoro", model_manager=manager)

        assert status["state"] == "provider_unavailable"

    @pytest.mark.asyncio
    async def test_status_reports_loaded_model_as_loaded_while_a_reload_is_queued(self, manager, hf_cache):
        # The UI treats only "loaded" as ready to generate.
        service = ModelProgressService()
        await service.load(model_id="kokoro", model_manager=manager)

        tts_lock = service.operation_lock("tts")
        await tts_lock.acquire()  # another TTS model operation is in flight
        queued = asyncio.create_task(service.load(model_id="kokoro", model_manager=manager))
        try:
            for _ in range(200):
                if service.download_progress.get("kokoro", {}).get("status") == "queued":
                    break
                await asyncio.sleep(0.01)
            assert service.download_progress["kokoro"]["status"] == "queued"
            status = await service.get_status(model_id="kokoro", model_manager=manager)
        finally:
            tts_lock.release()
            await asyncio.wait_for(queued, timeout=10)

        assert status["state"] == "loaded"

    @pytest.mark.asyncio
    async def test_failed_load_keeps_the_downloaded_state_of_a_finished_download(self, manager, hf_cache):
        # A failed load (CUDA out of memory) deletes no weights, so the Models tab must not offer
        # Download again for weights the artifact scan cannot see (an isolated worker's volume).
        service = ModelProgressService()
        await service.download(model_id="kokoro", model_manager=manager)
        manager._tts.load_model = _raise(RuntimeError("CUDA out of memory"))

        with pytest.raises(HTTPException) as failed:
            await service.load(model_id="kokoro", model_manager=manager)
        after_failed_load = (
            (await service.get_status(model_id="kokoro", model_manager=manager))["state"],
            (await service.get_progress(model_id="kokoro", model_manager=manager))["status"],
        )
        await service.delete_artifacts(model_id="kokoro", model_manager=manager)
        with pytest.raises(HTTPException):
            await service.load(model_id="kokoro", model_manager=manager)
        after_deletion = (await service.get_status(model_id="kokoro", model_manager=manager))["state"]

        assert failed.value.status_code == 500
        assert after_failed_load == ("downloaded", "downloaded")
        assert after_deletion == "provider_installed"


class TestModelProgressTypeLocks:
    """The unified routes queue an operation only behind operations of its own model type.

    ModelManager keeps one lock per type, and runs a download that needs no load outside them,
    so the route layer must not serialize STT and TTS again.
    """

    STT_ID = "Systran/faster-whisper-small"

    @staticmethod
    def _blocking(entered: threading.Event, release: threading.Event, then):
        def run(model_id):
            entered.set()
            release.wait(10)  # a running decode, or a multi-gigabyte download, holds this thread
            return then(model_id)
        return run

    async def _tts_load_while(self, service, manager, stt_operation, entered, release):
        blocked = asyncio.create_task(stt_operation())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=10)
            return await asyncio.wait_for(
                service.load(model_id="kokoro", model_manager=manager), timeout=3
            )
        finally:
            release.set()
            await asyncio.wait_for(blocked, timeout=10)

    @pytest.mark.asyncio
    async def test_tts_load_does_not_wait_behind_an_stt_unload_held_by_a_decode(self, manager, hf_cache):
        # faster-whisper's unload waits for the inference lock a long transcription holds.
        stt = manager._stt
        stt.load_model(self.STT_ID)
        entered, release = threading.Event(), threading.Event()
        stt.unload_model = self._blocking(entered, release, stt._default_backend.unload_model)
        service = ModelProgressService()

        loaded = await self._tts_load_while(
            service, manager,
            lambda: service.unload(model_id=self.STT_ID, model_manager=manager),
            entered, release,
        )

        assert loaded["state"] == "loaded"
        assert not stt.is_model_loaded(self.STT_ID)  # the unload still finished afterwards

    @pytest.mark.asyncio
    async def test_tts_load_does_not_wait_behind_an_stt_download(self, manager, hf_cache):
        stt = manager._stt
        entered, release = threading.Event(), threading.Event()
        stt.download_model = self._blocking(entered, release, lambda _model_id: True)
        service = ModelProgressService()

        loaded = await self._tts_load_while(
            service, manager,
            lambda: service.download(model_id="Systran/faster-whisper-large-v3", model_manager=manager),
            entered, release,
        )

        assert loaded["state"] == "loaded"
        assert service.download_progress["Systran/faster-whisper-large-v3"]["status"] == "downloaded"

    @pytest.mark.asyncio
    async def test_unknown_model_operation_is_404_before_any_lock(self, manager, hf_cache):
        service = ModelProgressService()
        for operation in (service.load, service.download, service.delete_artifacts):
            with pytest.raises(HTTPException) as caught:
                await asyncio.wait_for(operation(model_id="no-such/model", model_manager=manager), timeout=5)
            assert caught.value.status_code == 404
            assert caught.value.detail["code"] == "unknown_model"
        assert "no-such/model" not in service.download_progress


def _raise(error):
    def fail(_model_id):
        raise error
    return fail


class TestModelProgressErrors:
    """Server and provider failures keep their status instead of becoming client errors."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "code", "status_code"),
        [
            (RuntimeError("CUDA out of memory"), "load_failed", 500),
            (ExternalProviderError("worker is gone", code="provider_unavailable", status_code=503),
             "provider_unavailable", 503),
        ],
    )
    async def test_unified_load_failure_keeps_its_status_code(self, manager, error, code, status_code):
        manager._tts.load_model = _raise(error)

        with pytest.raises(HTTPException) as caught:
            await ModelProgressService().load(model_id="kokoro", model_manager=manager)

        assert caught.value.detail["code"] == code
        assert caught.value.status_code == status_code

    @pytest.mark.asyncio
    async def test_unified_load_reports_offline_provider_as_503(self, manager):
        manager._tts.provider_is_available = lambda _provider: False

        with pytest.raises(HTTPException) as caught:
            await ModelProgressService().load(model_id="kokoro", model_manager=manager)

        assert caught.value.detail["code"] == "provider_unavailable"
        assert caught.value.status_code == 503

    @pytest.mark.asyncio
    async def test_unified_download_keeps_provider_status_code_from_its_unload(self, manager, hf_cache):
        manager._tts.unload_model = _raise(
            ExternalProviderError("worker is gone", code="provider_unavailable", status_code=503),
        )
        service = ModelProgressService()

        with pytest.raises(HTTPException) as caught:
            await service.download(model_id="kokoro", model_manager=manager)

        assert caught.value.detail["code"] == "provider_unavailable"
        assert caught.value.status_code == 503
        assert "kokoro" not in service.download_progress

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "code", "status_code"),
        [
            (ExternalProviderError("worker is gone", code="provider_unavailable", status_code=503),
             "provider_unavailable", 503),
            (RuntimeError("CUDA error"), "unload_failed", 500),
        ],
    )
    async def test_unified_unload_maps_failure_to_http_error(self, manager, error, code, status_code):
        service = ModelProgressService()
        await service.load(model_id="kokoro", model_manager=manager)
        manager._tts.unload_model = _raise(error)

        with pytest.raises(HTTPException) as caught:
            await service.unload(model_id="kokoro", model_manager=manager)

        assert caught.value.detail["code"] == code
        assert caught.value.status_code == status_code


class TestSttFirstLoadDownloadsOutsideTheLock:
    """An STT model's first load downloads before the STT lock.

    CHANGELOG ("Models API: one queue per type ... no load or unload waits for a download"). Load in the Models view
    on a Whisper model that is not cached held ``operation_lock("stt")`` for the whole 1.5 or 3 GB download, after it
    had already unloaded the STT model in use.
    """

    ACTIVE = "deepdml/faster-whisper-large-v3-turbo-ct2"
    NEW = "Systran/faster-whisper-medium"

    @staticmethod
    def _install_whisper(monkeypatch, download_model):
        """A fake faster_whisper whose WhisperModel downloads a hub id first, as faster-whisper 1.2.1's does."""
        import sys
        from types import ModuleType

        utils = ModuleType("faster_whisper.utils")
        utils.download_model = download_model

        class WhisperModel:
            def __init__(self, model_size_or_path, *args, **kwargs):
                if not model_size_or_path.startswith("/cache/"):
                    utils.download_model(model_size_or_path)

        module = ModuleType("faster_whisper")
        module.utils, module.WhisperModel = utils, WhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", module)
        monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)

    def test_the_download_holds_no_stt_lock_and_the_model_in_use_keeps_serving(self, tmp_path, monkeypatch):
        from src.router import BackendRouter

        started, release, fetched = threading.Event(), threading.Event(), []

        def download_model(model_id, cache_dir=None, **kwargs):
            fetched.append(model_id)
            started.set()
            release.wait(20)  # a 1.5 GB hub download
            return "/cache/" + model_id

        self._install_whisper(monkeypatch, download_model)
        monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))  # an empty cache
        stt = BackendRouter()
        backend = stt.get_backend(self.ACTIVE)
        backend._models[self.ACTIVE] = object()
        backend._loaded_at[self.ACTIVE] = time.time()
        manager = ModelManager(stt_router=stt, tts_router=FakeTTSRouter())
        errors: list = []

        def load():
            try:
                manager.load(self.NEW)
            except Exception as exc:  # reported below
                errors.append(exc)

        loader = threading.Thread(target=load, daemon=True)
        loader.start()
        try:
            assert started.wait(5), "the load never started its download"
            lock = manager.operation_lock("stt")
            lock_free = lock.acquire(timeout=2)
            if lock_free:
                lock.release()
            serving_during_the_download = stt.is_model_loaded(self.ACTIVE)
        finally:
            release.set()
            loader.join(timeout=10)

        assert not loader.is_alive() and errors == []
        assert lock_free, "the STT model lock was held for the whole download"
        assert serving_during_the_download, "the STT model in use was unloaded before the download"
        assert [m.id for m in manager.list_loaded() if m.type == "stt"] == [self.NEW]
        assert fetched[0] == self.NEW

    def test_a_failed_download_is_a_load_failure_and_keeps_the_model_in_use(self, tmp_path, monkeypatch):
        from src.router import BackendRouter

        def download_model(model_id, cache_dir=None, **kwargs):
            raise OSError("hub unreachable")

        self._install_whisper(monkeypatch, download_model)
        monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))
        stt = BackendRouter()
        backend = stt.get_backend(self.ACTIVE)
        backend._models[self.ACTIVE] = object()
        backend._loaded_at[self.ACTIVE] = time.time()
        manager = ModelManager(stt_router=stt, tts_router=FakeTTSRouter())

        with pytest.raises(ModelLifecycleError) as caught:
            manager.load(self.NEW)

        assert (caught.value.code, caught.value.status_code) == ("load_failed", 500)
        assert "hub unreachable" in caught.value.message
        assert stt.is_model_loaded(self.ACTIVE)


class TestModelsRouteLoadFetchesBeforeItsQueue:
    """``POST /api/models/{id}/load`` fetches a Whisper model before the route's STT queue.

    Load in Models on a model that isn't downloaded fetches before taking the STT locks, so other STT loads and
    unloads go on meanwhile, and the model in use stays loaded. ``ModelProgressService.load`` held
    its ``operation_lock("stt")`` around the whole ``ModelManager.load``, download included, so an unload or artifact
    deletion through the API waited for the 1.5 or 3 GB download. And a download that stopped half way (or one still
    running) leaves ``models--<id>`` behind, which read as cached, so the load skipped its fetch, unloaded the model in
    use and finished the download under both locks.
    """

    ACTIVE = "deepdml/faster-whisper-large-v3-turbo-ct2"
    NEW = "Systran/faster-whisper-medium"
    OTHER = "Systran/faster-whisper-small"
    WAIT_S = 8.0  # the unload's first `import torch` (its CUDA cache release) alone takes about 1.6 s

    def _setup(self, tmp_path, monkeypatch, *, fail: bool = False):
        from src.router import BackendRouter

        started, release, fetched = threading.Event(), threading.Event(), []

        def download_model(model_id, cache_dir=None, **kwargs):
            fetched.append(model_id)
            if model_id == self.NEW:
                started.set()
                release.wait(20)  # a 1.5 GB hub download
                if fail:
                    raise OSError("hub unreachable")
            return "/cache/" + model_id

        TestSttFirstLoadDownloadsOutsideTheLock._install_whisper(monkeypatch, download_model)
        monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))  # NEW isn't downloaded
        (tmp_path / ("models--" + self.OTHER.replace("/", "--"))).mkdir()  # OTHER is downloaded, not loaded
        stt = BackendRouter()
        backend = stt.get_backend(self.ACTIVE)
        backend._models[self.ACTIVE] = object()
        backend._loaded_at[self.ACTIVE] = time.time()
        manager = ModelManager(stt_router=stt, tts_router=FakeTTSRouter())
        return stt, manager, ModelProgressService(), started, release, fetched

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["unload", "delete_artifacts"])
    async def test_an_stt_unload_or_artifact_deletion_goes_on_during_another_models_first_load(
        self, tmp_path, monkeypatch, operation,
    ):
        stt, manager, service, started, release, fetched = self._setup(tmp_path, monkeypatch)
        target = {"unload": self.ACTIVE, "delete_artifacts": self.OTHER}[operation]
        load = asyncio.create_task(service.load(model_id=self.NEW, model_manager=manager))
        other = None
        try:
            assert await asyncio.to_thread(started.wait, 5), "the load never started its download"
            # The load writes "downloading" once its start_stt_fetch thread has handed the
            # future back to the loop, which can be after this test's own wake-up; poll (bounded) while the download
            # is still held instead of reading once.
            statuses = []
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                statuses.append((await service.get_progress(model_id=self.NEW, model_manager=manager))["status"])
                if statuses[-1] == "downloading":
                    break
                await asyncio.sleep(0.01)
            read_during_the_download = not release.is_set()
            other = asyncio.create_task(getattr(service, operation)(model_id=target, model_manager=manager))
            done, _ = await asyncio.wait({other}, timeout=self.WAIT_S)
            finished_during_the_download = other in done
        finally:
            release.set()
        info = await asyncio.wait_for(load, 10)
        await asyncio.wait_for(other, 10)

        assert finished_during_the_download, (
            f"the STT {operation} of {target} waited for {self.NEW}'s first download: the route's STT queue was held"
        )
        assert statuses[-1] == "downloading" and read_during_the_download, statuses
        assert info["state"] == "loaded"
        assert [m.id for m in manager.list_loaded() if m.type == "stt"] == [self.NEW]

    @pytest.mark.asyncio
    async def test_a_download_that_stopped_half_way_is_finished_before_the_model_in_use_is_unloaded(
        self, tmp_path, monkeypatch,
    ):
        stt, manager, service, started, release, fetched = self._setup(tmp_path, monkeypatch)
        # huggingface_hub writes refs/main before it fetches any file, so a stopped download reads as cached.
        refs = tmp_path / ("models--" + self.NEW.replace("/", "--")) / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text("0" * 40)
        assert stt.is_model_cached(self.NEW)
        load = asyncio.create_task(service.load(model_id=self.NEW, model_manager=manager))
        try:
            assert await asyncio.to_thread(started.wait, 5), "the load never finished the stopped download"
            serving_during_the_download = stt.is_model_loaded(self.ACTIVE)
            queue_free = not service.operation_lock("stt").locked()
        finally:
            release.set()
        await asyncio.wait_for(load, 10)

        assert serving_during_the_download, "the model in use was unloaded before the stopped download was finished"
        assert queue_free, "the stopped download was finished inside the route's STT queue"
        assert [m.id for m in manager.list_loaded() if m.type == "stt"] == [self.NEW]

    @pytest.mark.asyncio
    async def test_a_failed_first_download_is_a_load_failure_and_keeps_the_model_in_use(self, tmp_path, monkeypatch):
        stt, manager, service, started, release, fetched = self._setup(tmp_path, monkeypatch, fail=True)
        release.set()

        with pytest.raises(HTTPException) as caught:
            await asyncio.wait_for(service.load(model_id=self.NEW, model_manager=manager), 10)

        assert caught.value.status_code == 500
        assert caught.value.detail["code"] == "load_failed"
        assert "hub unreachable" in caught.value.detail["message"]
        assert stt.is_model_loaded(self.ACTIVE)
        assert self.NEW not in service.download_progress


class TestModelManagerUnknownPrefixedIds:
    """A registered TTS prefix does not make every ID under it a model."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["get_status", "delete_artifacts"])
    async def test_unknown_piper_model_returns_404_not_server_error(self, operation, tmp_path):
        tts = _tts_router(kokoro=FakeKokoroBackend(), piper=FakePiperBackend())
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=tts)

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            with pytest.raises(HTTPException) as caught:
                await getattr(ModelProgressService(), operation)(
                    model_id="piper/en_US-not-a-voice-medium", model_manager=manager,
                )

        assert caught.value.status_code == 404
        assert caught.value.detail["code"] == "unknown_model"

    @pytest.mark.asyncio
    async def test_deleting_unknown_kokoro_prefixed_id_keeps_kokoro_weights(self, tmp_path):
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(kokoro=FakeKokoroBackend()))
        weights = tmp_path / "models--hexgrad--Kokoro-82M"
        weights.mkdir()
        (weights / "kokoro-v1_0.pth").write_bytes(b"weights")

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            with pytest.raises(HTTPException) as caught:
                await ModelProgressService().delete_artifacts(model_id="kokoro/typo", model_manager=manager)

        assert caught.value.detail["code"] == "unknown_model"
        assert weights.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["get_status", "delete_artifacts"])
    async def test_tts_model_setting_the_backend_rejects_is_unknown_not_a_server_error(self, operation, tmp_path):
        # Being TTS_MODEL does not make an ID a model: an .env typo (quality suffix missing) was
        # resolved as TTS and then raised KeyError looking up Piper's voice files (500).
        typo = "piper/en_US-lessac"
        tts = _tts_router(kokoro=FakeKokoroBackend(), piper=FakePiperBackend())
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=tts)

        with patch.object(settings, "tts_model", typo), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            with pytest.raises(HTTPException) as caught:
                await getattr(ModelProgressService(), operation)(model_id=typo, model_manager=manager)

        assert caught.value.status_code == 404
        assert caught.value.detail["code"] == "unknown_model"

    @pytest.mark.asyncio
    async def test_deleting_a_kokoro_prefixed_tts_model_setting_keeps_kokoro_weights(self, tmp_path):
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(kokoro=FakeKokoroBackend()))
        weights = tmp_path / "models--hexgrad--Kokoro-82M"
        weights.mkdir()
        (weights / "kokoro-v1_0.pth").write_bytes(b"weights")

        with patch.object(settings, "tts_model", "kokoro/v1.1-zh"), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            with pytest.raises(HTTPException) as caught:
                await ModelProgressService().delete_artifacts(model_id="kokoro/v1.1-zh", model_manager=manager)

        assert caught.value.detail["code"] == "unknown_model"
        assert weights.exists()

    def test_mistyped_tts_model_of_an_uninstalled_provider_reports_provider_missing(self, tmp_path):
        # No Piper backend can be asked about the ID, so it keeps resolving, as provider_missing.
        typo = "piper/en_US-lessac"
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(kokoro=FakeKokoroBackend()))

        with patch.object(settings, "tts_model", typo), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            state = manager.status(typo).state
            deleted = manager.delete_artifacts(typo)

        assert state == ModelState.PROVIDER_MISSING
        assert deleted["status"] == "not_found"

    def test_inventory_leaves_out_a_tts_model_setting_the_backend_rejects(self, tmp_path):
        # status() and every operation answer unknown_model for it, so the Models tab must not
        # offer it as an installed model to load.
        typo = "piper/en_US-lessac"
        tts = _tts_router(kokoro=FakeKokoroBackend(), piper=FakePiperBackend())
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=tts)

        with patch.object(settings, "tts_model", typo), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            listed = {m.id for m in manager.list_all()}

        assert typo not in listed
        assert "piper/en_US-lessac-medium" in listed

    def test_offline_worker_model_still_reports_provider_unavailable(self, tmp_path):
        # The worker cannot be asked which IDs it serves, so its IDs must keep resolving.
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(qwen3=FakeOfflineWorker()))

        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            assert manager.status("qwen3/custom-voice").state == ModelState.PROVIDER_UNAVAILABLE


class TestModelManagerArtifactContainment:
    """Artifact deletion stays inside the model's own directory under a managed cache root."""

    def test_cache_root_itself_is_never_removed(self, manager, tmp_path):
        (tmp_path / "models--Systran--faster-whisper-base").mkdir()

        assert manager._safe_remove_dir(tmp_path, [tmp_path]) is False
        assert (tmp_path / "models--Systran--faster-whisper-base").exists()

    @pytest.mark.skipif(os.name != "nt", reason="a backslash is a path separator only on Windows")
    @pytest.mark.parametrize("escape", ["..\\..", "..\\..\\models--Systran--faster-whisper-large-v3"])
    def test_backslash_id_cannot_delete_cache_root_or_other_models(self, tmp_path, escape):
        # An offline worker accepts any "qwen3/..." ID, so only path containment stops this one.
        root = tmp_path / "hub"
        other = root / "models--Systran--faster-whisper-large-v3"
        other.mkdir(parents=True)
        (other / "model.bin").write_bytes(b"weights")
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(qwen3=FakeOfflineWorker()))

        with patch.object(manager, "_hf_cache_roots", return_value=[root]):
            result = manager.delete_artifacts(f"qwen3/{escape}")

        assert result["deleted_paths"] == []
        assert other.exists()

    @pytest.mark.parametrize("model_id", [
        "acme/..",
        pytest.param("acme/..\\other-model", marks=pytest.mark.skipif(
            os.name != "nt", reason="a backslash is a path separator only on Windows")),
    ])
    def test_stt_cache_deletion_stays_inside_the_model_dir(self, tmp_path, monkeypatch, model_id):
        from src.backends.faster_whisper import FasterWhisperBackend

        cache = tmp_path / "outer" / "stt-models"
        other = tmp_path / "outer" / "other-model"
        cache.mkdir(parents=True)
        other.mkdir()
        monkeypatch.setattr(settings, "stt_model_dir", str(cache))

        assert FasterWhisperBackend().delete_cached_model(model_id) is False
        assert cache.exists()
        assert other.exists()


class TestModelManagerEviction:
    def test_evict_lru(self, manager):
        # Load two STT models (disable auto-eviction to keep both loaded), neither of them STT_MODEL
        with patch.object(settings, "stt_model", "Systran/faster-whisper-large-v3"):
            manager.load("Systran/faster-whisper-base", _evict_others=False)
            time.sleep(0.01)
            manager.load("Systran/faster-whisper-small", _evict_others=False)
            manager.evict_lru()
            loaded = manager.list_loaded()
        ids = [m.id for m in loaded]
        assert "Systran/faster-whisper-base" not in ids
        assert "Systran/faster-whisper-small" in ids

    def test_evict_lru_can_evict_default_when_defaults_are_not_pinned(self, manager):
        with patch.object(settings, "os_pin_default_models", False), \
             patch.object(settings, "stt_model", "Systran/faster-whisper-base"):
            manager.load("Systran/faster-whisper-base", _evict_others=False)
            time.sleep(0.01)
            manager.load("Systran/faster-whisper-small", _evict_others=False)
            manager.evict_lru()
            loaded = manager.list_loaded()
            ids = [m.id for m in loaded]
            assert "Systran/faster-whisper-base" not in ids
            assert "Systran/faster-whisper-small" in ids

    def test_evict_lru_never_picks_a_pinned_default(self, manager):
        # OS_PIN_DEFAULT_MODELS (default true): LRU eviction leaves STT_MODEL and TTS_MODEL loaded.
        assert settings.os_pin_default_models is True
        with patch.object(settings, "stt_model", "Systran/faster-whisper-base"), \
             patch.object(settings, "tts_model", "kokoro"):
            manager.load("Systran/faster-whisper-base", _evict_others=False)
            manager.load("kokoro")
            time.sleep(0.01)
            manager.load("Systran/faster-whisper-small", _evict_others=False)
            evicted_first = manager.evict_lru()  # the defaults were used longer ago
            after_first_pass = {m.id for m in manager.list_loaded()}
            evicted_second = manager.evict_lru()  # only defaults are left
            after_second_pass = {m.id for m in manager.list_loaded()}

        assert after_first_pass == after_second_pass == {"Systran/faster-whisper-base", "kokoro"}
        assert (evicted_first, evicted_second) == (True, False)

    def test_check_ttl(self, manager):
        with patch.object(settings, "os_model_ttl", 0):
            manager.load("Systran/faster-whisper-base")
            # TTL=0 means never evict
            manager.check_ttl()
            assert len(manager.list_loaded()) == 1

    def test_check_ttl_evicts_old(self, manager):
        with patch.object(settings, "os_model_ttl", 1), \
             patch.object(settings, "stt_model", "Systran/faster-whisper-large-v3"):
            manager.load("Systran/faster-whisper-base")
            # Fake old last_used
            for m in manager._stt.loaded_models():
                if m.model == "Systran/faster-whisper-base":
                    manager._stt._default_backend._last_used[m.model] = time.time() - 10
            manager.check_ttl()
            assert len(manager.list_loaded()) == 0

    def test_check_ttl_evicts_default_stt_when_defaults_are_not_pinned(self, manager):
        with patch.object(settings, "os_model_ttl", 1), \
             patch.object(settings, "os_pin_default_models", False), \
             patch.object(settings, "stt_model", "Systran/faster-whisper-base"):
            manager.load("Systran/faster-whisper-base")
            manager._stt._default_backend._last_used["Systran/faster-whisper-base"] = time.time() - 10
            manager.check_ttl()
            assert len(manager.list_loaded()) == 0

    def test_check_ttl_evicts_tts_when_defaults_are_not_pinned(self, manager):
        with patch.object(settings, "os_model_ttl", 1), \
             patch.object(settings, "os_pin_default_models", False), \
             patch.object(settings, "tts_model", "kokoro"):
            manager.load("kokoro")
            manager._tts._default._last_used["kokoro"] = time.time() - 10
            manager.check_ttl()
            assert len(manager.list_loaded()) == 0

    def test_check_ttl_keeps_pinned_defaults_and_evicts_the_rest(self, manager):
        # The user's priority is instant speech: an idle TTS_MODEL (and STT_MODEL) stays warm.
        with patch.object(settings, "os_model_ttl", 1), \
             patch.object(settings, "stt_model", "Systran/faster-whisper-base"), \
             patch.object(settings, "tts_model", "kokoro"):
            manager.load("Systran/faster-whisper-base", _evict_others=False)
            manager.load("Systran/faster-whisper-small", _evict_others=False)
            manager.load("kokoro")
            for backend in (manager._stt._default_backend, manager._tts._default):
                for model_id in backend._models:
                    backend._last_used[model_id] = time.time() - 10
            manager.check_ttl()
            loaded = {m.id for m in manager.list_loaded()}

        assert loaded == {"Systran/faster-whisper-base", "kokoro"}

    def test_ttl_pass_does_not_wait_for_a_stream_of_the_same_provider(self):
        # A Piper stream holds Piper's synthesis lock while its client reads. Unloading another,
        # idle, Piper voice needs that lock, so the pass leaves it for later instead of waiting
        # (and holding the TTS operation lock) until the stream ends.
        import numpy as np

        class StreamingPiperBackend(FakePiperBackend):
            def synthesize(self, text, voice, speed=1.0, lang_code=None, **kwargs):
                yield np.zeros(8, dtype=np.float32)
                yield np.zeros(8, dtype=np.float32)

        piper = StreamingPiperBackend()
        tts = _tts_router(piper=piper)
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=tts)
        streaming, idle = "piper/en_US-lessac-medium", "piper/en_US-amy-medium"

        with patch.object(settings, "os_model_ttl", 300):
            tts.load_model(streaming)
            tts.load_model(idle)
            piper._last_used[idle] = time.time() - 600
            stream = tts.synthesize(text="A long chapter.", model=streaming, voice=streaming)
            next(stream)  # the client has read the first chunk
            ttl_pass = threading.Thread(target=manager.check_ttl)
            ttl_pass.start()
            ttl_pass.join(timeout=2)
            pass_waited = ttl_pass.is_alive()
            idle_kept_during_stream = tts.is_model_loaded(idle)
            stream.close()
            ttl_pass.join(timeout=5)
            manager.check_ttl()  # the next pass
            idle_unloaded_after = not tts.is_model_loaded(idle)

        assert not pass_waited, "the TTL pass waited for a stream to end"
        assert idle_kept_during_stream
        assert idle_unloaded_after
        assert tts.is_model_loaded(streaming)

    def test_explicit_unload_still_unloads_a_pinned_default(self, manager):
        with patch.object(settings, "tts_model", "kokoro"):
            manager.load("kokoro")
            manager.unload("kokoro")

        assert manager.list_loaded() == []


# A worker whose torch build has no code for the GPU
# answers /health with status "failed" and failure {code, message}, the detail its load's 503 gives. The core read
# only the status: a load answered the generic "configured but unavailable" and /api/models gave no reason. The
# reason is passed on now: in the provider_unavailable answer (the same code), and as provider_failure on the rows.
ARCH_FAILURE = {
    "code": "cuda_arch_unsupported",
    "message": "Torch wheel has no code that runs on sm_120 (built for: sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90)",
}
CHATTERBOX_IDS = ("chatterbox/regular", "chatterbox/turbo")
CHATTERBOX_MANIFEST = {
    "schema_version": 1, "provider": "chatterbox",
    "models": [
        {"id": model_id, "provider": "chatterbox", "sample_rate": 24000, "capabilities": {}, "voices": []}
        for model_id in CHATTERBOX_IDS
    ],
}


def _worker_health(status: str, failure=None) -> dict:
    return {"status": status, "provider": "chatterbox", "device": "cuda:0", "loaded_model": None, "failure": failure}


class _WorkerAnswer(io.BytesIO):
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _chatterbox_worker(health: dict | None, extra_models: tuple[str, ...] = ()):
    """A real ExternalTTSBackend whose stubbed worker serves its manifest and `health` (None: the worker is down).

    `extra_models` adds models to the worker's manifest, for a worker model the registry doesn't list."""
    from src.tts.external import ExternalTTSBackend

    backend = ExternalTTSBackend(provider="chatterbox", base_url="http://chatterbox:8200")
    manifest = {**CHATTERBOX_MANIFEST, "models": [
        *CHATTERBOX_MANIFEST["models"],
        *({"id": model_id, "provider": "chatterbox", "sample_rate": 24000, "capabilities": {}, "voices": []}
          for model_id in extra_models),
    ]}

    def open_fn(request, **_kwargs):
        if request.full_url.endswith("/v1/manifest"):
            return _WorkerAnswer(json.dumps(manifest).encode())
        if request.full_url.endswith("/health"):
            if health is None:
                raise urllib.error.URLError("connection refused")
            return _WorkerAnswer(json.dumps(health).encode())
        raise AssertionError(f"only the manifest and /health are read, not {request.full_url}")

    backend._opener.open = open_fn
    return backend


class TestModelManagerWorkerFailure:
    @staticmethod
    def _rows(health, tmp_path) -> dict:
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(chatterbox=_chatterbox_worker(health)))
        with patch.object(settings, "tts_enabled", True), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            listed = list_all_models(
                model_manager=manager, tts_capabilities_for=lambda _model_id: {},
                default_stt_model=settings.stt_model, default_tts_model=settings.tts_model,
            )["models"]
            status = manager.status("chatterbox/regular").to_dict()
        rows = {
            m["id"]: (m["state"], m["provider_available"], m.get("provider_failure", "no provider_failure field"))
            for m in listed if m["provider"] == "chatterbox"
        }
        others = {m["id"]: m.get("provider_failure", "no provider_failure field") for m in listed if m["provider"] != "chatterbox"}
        return {"rows": rows, "others": others,
                "status": (status["state"], status["provider_available"], status.get("provider_failure", "no field"))}

    def test_the_rows_and_the_status_give_the_workers_reason(self, tmp_path):
        seen = self._rows(_worker_health("failed", ARCH_FAILURE), tmp_path)

        assert seen["rows"] == {model_id: ("provider_unavailable", False, ARCH_FAILURE) for model_id in CHATTERBOX_IDS}
        assert seen["status"] == ("provider_unavailable", False, ARCH_FAILURE)
        assert seen["others"] and set(seen["others"].values()) == {None}, "every other row has provider_failure null"

    @pytest.mark.parametrize(
        ("health", "state"),
        [
            pytest.param(_worker_health("ok"), "provider_installed", id="ok"),
            pytest.param(_worker_health("unavailable"), "provider_unavailable", id="unavailable"),
            pytest.param(None, "provider_unavailable", id="worker-down"),
            pytest.param(_worker_health("failed"), "provider_unavailable", id="failed-without-a-failure"),
        ],
    )
    def test_without_a_reason_provider_failure_is_null(self, tmp_path, health, state):
        seen = self._rows(health, tmp_path)

        available = state != "provider_unavailable"
        assert seen["rows"] == {model_id: (state, available, None) for model_id in CHATTERBOX_IDS}
        assert seen["status"] == (state, available, None)

    @pytest.mark.parametrize(
        ("failure", "expected"),
        [
            pytest.param(
                {"code": "cuda_arch_unsupported", "message": "Torch wheel\nhas no code\x00 for sm_120 " + "sm_90 " * 200},
                ("cuda_arch_unsupported", "Torch wheel has no code for sm_120 sm_90"), id="long-and-multiline",
            ),
            pytest.param({"code": "<b>bad code</b>", "message": "Torch wheel has no code"}, None, id="not-a-code"),
            pytest.param({"code": "x" * 65, "message": "Torch wheel has no code"}, None, id="code-too-long"),
            pytest.param({"code": "cuda_arch_unsupported", "message": 503}, None, id="message-not-text"),
            pytest.param({"code": "cuda_arch_unsupported", "message": " \n "}, None, id="message-empty"),
        ],
    )
    def test_the_reason_is_one_bounded_line_with_a_plain_code(self, tmp_path, failure, expected):
        seen = self._rows(_worker_health("failed", failure), tmp_path)

        given = seen["rows"]["chatterbox/regular"][2]
        if expected is None:
            assert given is None, given
            return
        assert isinstance(given, dict) and set(given) == {"code", "message"}, given
        code, start = expected
        assert given["code"] == code
        assert given["message"].startswith(start), given["message"]
        assert len(given["message"]) <= 300 and given["message"].isprintable(), given["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("health", "message"),
        [
            pytest.param(
                _worker_health("failed", ARCH_FAILURE),
                f"Provider 'chatterbox' is configured but its worker can't run: {ARCH_FAILURE['message']}.",
                id="failed",
            ),
            pytest.param(None, "Provider 'chatterbox' is configured but unavailable.", id="worker-down"),
        ],
    )
    async def test_a_load_says_the_workers_reason_and_keeps_the_code(self, tmp_path, health, message):
        manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=_tts_router(chatterbox=_chatterbox_worker(health)))

        with patch.object(settings, "tts_enabled", True), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]), \
             pytest.raises(HTTPException) as caught:
            await ModelProgressService().load(model_id="chatterbox/regular", model_manager=manager)

        assert caught.value.status_code == 503
        assert caught.value.detail == {"message": message, "code": "provider_unavailable"}

    @pytest.mark.parametrize(
        ("health", "expected"),
        [
            pytest.param(
                _worker_health("failed", ARCH_FAILURE), ("provider_unavailable", False, ARCH_FAILURE), id="failed",
            ),
            pytest.param(None, ("provider_unavailable", False, None), id="worker-down"),
            pytest.param(_worker_health("unavailable"), ("provider_unavailable", False, None), id="unavailable"),
            pytest.param(_worker_health("ok"), ("provider_installed", True, None), id="ok"),
        ],
    )
    def test_a_tts_model_outside_the_registry_gets_the_same_runtime_check(self, tmp_path, health, expected):
        # list_all() built the row for a TTS_MODEL that is not in the registry
        # from the manifest alone, so it said provider_installed and available, with no provider_failure, for a
        # worker that answers it can't run (or doesn't answer), while status() of the same model said why.
        from src.model_registry import get_known_model

        custom = "chatterbox/custom"
        assert get_known_model(custom) is None, "the test needs a worker model the registry doesn't list"
        manager = ModelManager(
            stt_router=FakeSTTRouter(), tts_router=_tts_router(chatterbox=_chatterbox_worker(health, (custom,))),
        )
        with patch.object(settings, "tts_enabled", True), patch.object(settings, "tts_model", custom), \
             patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            listed = list_all_models(
                model_manager=manager, tts_capabilities_for=lambda _model_id: {},
                default_stt_model=settings.stt_model, default_tts_model=settings.tts_model,
            )["models"]
            status = manager.status(custom).to_dict()

        rows = [m for m in listed if m["id"] == custom]
        assert len(rows) == 1, rows
        seen = (rows[0]["state"], rows[0]["provider_available"], rows[0].get("provider_failure", "no field"))
        assert seen == expected
        assert (status["state"], status["provider_available"], status.get("provider_failure")) == expected
        assert rows[0]["is_default"] is True


PARAKEET = "nvidia/parakeet-tdt-0.6b-v2"


class FakeParakeetBackend(FakeSTTBackend):
    """Parakeet's files differ by device: fp16 on a GPU (about 3.7 GB with its fp32 source), int8 on a CPU."""

    name = "parakeet"

    def __init__(self, device, cached_mb=None):
        super().__init__()
        self.device = device
        self.cached_mb = cached_mb  # None: no files in the cache; 0: a loaded model whose folder is gone
        self.runtime_calls = 0

    def runtime(self):
        self.runtime_calls += 1
        return (self.device, "float16" if self.device == "cuda" else "int8")

    def loaded_models(self):
        from src.models import LoadedModelInfo
        return [
            LoadedModelInfo(model=mid, backend="parakeet", device=self.device, compute_type="x",
                            loaded_at=self._loaded_at[mid], last_used_at=None)
            for mid in self._models
        ]

    def list_cached_models(self):
        if self.cached_mb is None and PARAKEET not in self._models:
            return []
        return [{"model": PARAKEET, "backend": "parakeet", "size_mb": self.cached_mb or 0}]


def _parakeet_manager(device, cached_mb=None, loaded=False):
    stt = FakeSTTRouter()
    parakeet = FakeParakeetBackend(device, cached_mb)
    stt._backends["parakeet"] = parakeet
    stt.loaded_models = lambda: stt._default_backend.loaded_models() + parakeet.loaded_models()
    stt.list_cached_models = lambda: parakeet.list_cached_models()
    if loaded:
        parakeet.load_model(PARAKEET)
    return ModelManager(stt_router=stt, tts_router=FakeTTSRouter()), parakeet


class TestParakeetSizes:
    """The Parakeet row's size: its files on this device, the same whether it is downloaded, loaded or not yet
    downloaded (the registry's 1.3 GB was the fp16 encoder alone), and the measured size once files exist."""

    @pytest.mark.parametrize(("device", "expected"), [("cuda", 3700), ("cpu", 700)])
    def test_not_downloaded_and_loaded_without_files_show_the_devices_size(self, device, expected):
        manager, _ = _parakeet_manager(device)
        assert next(m for m in manager.list_all() if m.id == PARAKEET).size_mb == expected
        loaded, _ = _parakeet_manager(device, cached_mb=0, loaded=True)
        row = next(m for m in loaded.list_all() if m.id == PARAKEET)
        assert (row.state, row.size_mb) == (ModelState.LOADED, expected)

    @pytest.mark.parametrize("loaded", [False, True], ids=["downloaded", "loaded"])
    def test_files_on_disk_show_their_measured_size(self, loaded):
        manager, _ = _parakeet_manager("cuda", cached_mb=3712.4, loaded=loaded)
        row = next(m for m in manager.list_all() if m.id == PARAKEET)
        assert row.state == (ModelState.LOADED if loaded else ModelState.DOWNLOADED)
        assert row.size_mb == 3712.4

    def test_the_backend_is_asked_for_its_device_once(self):
        manager, parakeet = _parakeet_manager("cpu")
        manager.list_all()
        manager.list_all()
        assert parakeet.runtime_calls == 1

    def test_the_registry_names_both_devices(self):
        from src.model_registry import get_known_model, known_model_size_mb

        known = get_known_model(PARAKEET)
        assert known_model_size_mb(known, "cuda:0") == 3700
        assert known_model_size_mb(known, "cpu") == 700
        assert known_model_size_mb(get_known_model("Systran/faster-whisper-base"), "cuda") == 150


class TestPinnedFlag:
    """Each row says whether it is a pinned default, so the web app can say what a load leaves loaded."""

    def test_the_defaults_are_pinned_while_os_pin_default_models_is_on(self, manager):
        with patch.object(settings, "stt_model", PARAKEET), patch.object(settings, "os_pin_default_models", True):
            rows = {m.id: m.to_dict() for m in manager.list_all()}
        assert rows[PARAKEET]["pinned"] is True
        assert rows["kokoro"]["pinned"] is True
        assert rows["Systran/faster-whisper-small"]["pinned"] is False

    def test_nothing_is_pinned_while_it_is_off(self, manager):
        with patch.object(settings, "stt_model", PARAKEET), patch.object(settings, "os_pin_default_models", False):
            rows = {m.id: m.to_dict() for m in manager.list_all()}
        assert rows[PARAKEET]["pinned"] is False
        assert rows["kokoro"]["pinned"] is False


class TestDeleteDuringFetch:
    """Delete files while Parakeet's first-use fetch writes them: refused with 409 model_fetching, files kept."""

    @staticmethod
    def _fetching_manager(tmp_path):
        from src.backends.parakeet import ModelFetchRunningError

        manager, parakeet = _parakeet_manager("cpu")
        # The folder sweep that follows the backend's own deletion (models--<model id>) must not run either.
        repo = tmp_path / "models--nvidia--parakeet-tdt-0.6b-v2"
        repo.mkdir()
        (repo / "config.json").write_bytes(b"{}")

        def refuse(model_id):
            raise ModelFetchRunningError(f"{model_id}'s files are being downloaded; delete them once the download has finished.")

        manager._stt.delete_cached_model = refuse
        return manager, repo

    def test_the_manager_refuses_with_model_fetching(self, tmp_path):
        manager, repo = self._fetching_manager(tmp_path)
        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            with pytest.raises(ModelLifecycleError) as caught:
                manager.delete_artifacts(PARAKEET)
        assert (caught.value.code, caught.value.status_code) == ("model_fetching", 409)
        assert "being downloaded" in caught.value.message
        assert repo.exists()

    @pytest.mark.asyncio
    async def test_the_route_answers_409(self, tmp_path):
        manager, repo = self._fetching_manager(tmp_path)
        with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
            with pytest.raises(HTTPException) as caught:
                await ModelProgressService().delete_artifacts(model_id=PARAKEET, model_manager=manager)
        assert caught.value.status_code == 409
        assert caught.value.detail["code"] == "model_fetching"
        assert "delete them once the download has finished" in caught.value.detail["message"]
        assert repo.exists()


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


def test_the_web_uis_rebuild_command_can_succeed_and_keeps_the_default_image_providers():
    # Ported from the old web UI's contract tests. The Dockerfile bakes in only its own extras and
    # rejects any other name in BAKED_PROVIDERS; the cloning providers run as isolated workers. The Models
    # view's "rebuild your image" command (src/static/js/core/models.js BAKED_PROVIDERS) must therefore name
    # only bakeable providers, every one of them, and keep what the default image already has.
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    bakeable = set(json.loads(re.search(r"extras = (\{[^}]*\})", dockerfile).group(1)))
    image_default = set(re.search(r'ARG BAKED_PROVIDERS="([^"]*)"', dockerfile).group(1).split(","))
    models_js = (root / "src" / "static" / "js" / "core" / "models.js").read_text(encoding="utf-8")
    listed = re.search(r"BAKED_PROVIDERS = Object\.freeze\(\[([^\]]*)\]\)", models_js)
    assert listed, "core/models.js no longer lists BAKED_PROVIDERS"
    ui = re.findall(r"'([\w-]+)'", listed.group(1))

    assert ui and len(ui) == len(set(ui))
    assert set(ui) == bakeable
    assert image_default <= set(ui)
    assert bakeable.isdisjoint({"qwen3", "chatterbox", "fish"})
