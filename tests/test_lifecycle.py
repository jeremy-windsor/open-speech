"""Unit tests for model lifecycle management."""

from __future__ import annotations

import inspect
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.backends.faster_whisper import FasterWhisperBackend
from src.lifecycle import ModelLifecycleManager
from src.model_manager import ModelInfo, ModelState
from src.tts.backends.base import TTSLoadedModelInfo


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


class NothingToEvictModelManager(FakeModelManager):
    """Every model beyond the cap is a pinned default or busy, so evict_lru unloads none."""

    def evict_lru(self) -> bool:
        self.evict_lru_calls += 1
        if self.evict_lru_calls > 5:  # bounds a pass that would retry forever
            raise AssertionError("the pass kept calling evict_lru with nothing to evict")
        return False


class NoProgressModelManager(FakeModelManager):
    def evict_lru(self) -> bool:
        self.evict_lru_calls += 1
        return True  # claims an unload, but the loaded count does not drop


@pytest.mark.asyncio
async def test_eviction_pass_with_nothing_to_evict_stops_without_a_warning(caplog):
    # Pinned defaults do not count against OS_MAX_LOADED_MODELS and busy models are skipped,
    # so evict_lru finding nothing to unload is normal; a warning would repeat every 30 s pass.
    with patch.object(settings, "os_max_loaded_models", 1), \
         caplog.at_level(logging.DEBUG, logger="src.lifecycle"):
        manager = NothingToEvictModelManager(loaded_count=3)
        await ModelLifecycleManager(manager)._evict()

    assert manager.evict_lru_calls == 1
    assert len(manager.list_loaded()) == 3
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("No idle model to evict" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_eviction_pass_warns_when_an_eviction_unloads_nothing(caplog):
    with patch.object(settings, "os_max_loaded_models", 1), \
         caplog.at_level(logging.DEBUG, logger="src.lifecycle"):
        manager = NoProgressModelManager(loaded_count=3)
        await ModelLifecycleManager(manager)._evict()

    assert manager.evict_lru_calls == 1
    assert any(
        r.levelno == logging.WARNING and "did not reduce loaded model count" in r.getMessage()
        for r in caplog.records
    )


def test_stt_backend_protocol_declares_the_transcribe_options_callers_pass():
    # Live paths (streaming, realtime, Wyoming) pass temperature_fallback=False through
    # BackendRouter to whichever STTBackend serves the model, so every backend must take it.
    from src.backends.base import STTBackend

    protocol = inspect.signature(STTBackend.transcribe).parameters
    implemented = inspect.signature(FasterWhisperBackend.transcribe).parameters

    assert protocol["temperature_fallback"].default is True
    assert {name: p.default for name, p in protocol.items()} == {
        name: p.default for name, p in implemented.items()
    }


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


def _with_cached_files(module: ModuleType) -> ModuleType:
    """Give a fake faster_whisper the ``utils.download_model`` that a first load calls before it builds the model.

    The app moves the download out of WhisperModel.__init__, before the inference lock.
    Here the files are cached already and their path is the model id. Install ``module.utils`` in ``sys.modules``
    as ``faster_whisper.utils`` beside the module.
    """
    utils = ModuleType("faster_whisper.utils")
    utils.download_model = lambda model_id, cache_dir=None, **kwargs: model_id
    module.utils = utils
    return module


def test_concurrent_backend_load_constructs_model_once():
    worker_count = 10
    start_barrier = threading.Barrier(worker_count)
    whisper_model = MagicMock(return_value=object())
    faster_whisper_module = _with_cached_files(ModuleType("faster_whisper"))
    faster_whisper_module.WhisperModel = whisper_model
    backend = FasterWhisperBackend()

    def load_model() -> None:
        start_barrier.wait(timeout=10)
        backend.load_model("test-model")

    with (
        patch.dict(sys.modules, {"faster_whisper": faster_whisper_module,
                                 "faster_whisper.utils": faster_whisper_module.utils}),
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


def test_ttl_does_not_unload_stt_model_during_or_right_after_long_transcription():
    # OS_MODEL_TTL counts idle seconds. The lifecycle pass runs on its own thread, so it can
    # look while a request longer than the TTL is still decoding.
    from src.model_manager import ModelManager
    from src.router import BackendRouter

    clock = {"now": 1_000.0}
    fake_time = SimpleNamespace(time=lambda: clock["now"])
    stt = BackendRouter()
    manager = ModelManager(stt_router=stt, tts_router=SimpleNamespace(loaded_models=lambda: []))
    passes: list[threading.Thread] = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            clock["now"] += 600.0  # a 10-minute transcription
            ttl_pass = threading.Thread(target=manager.check_ttl)
            passes.append(ttl_pass)
            ttl_pass.start()
            ttl_pass.join(timeout=2)
            return [], SimpleNamespace(language="en", duration=600.0)

    fake_module = _with_cached_files(ModuleType("faster_whisper"))
    fake_module.WhisperModel = FakeWhisperModel
    model_id = "Systran/faster-whisper-base"

    # The model is the default STT_MODEL; unpinned, so the TTL check really looks at its idle time.
    with patch.dict(sys.modules, {"faster_whisper": fake_module, "faster_whisper.utils": fake_module.utils}), \
         patch("src.backends.faster_whisper.time", fake_time), \
         patch("src.model_manager.time", fake_time), \
         patch.object(settings, "os_model_ttl", 300), \
         patch.object(settings, "os_pin_default_models", False):
        manager.load(model_id)
        stt.transcribe(b"RIFF" + b"\x00" * 64, model_id)
        passes[0].join(timeout=10)  # an unload it decided on runs once the request ends
        loaded_after_mid_request_pass = stt.is_model_loaded(model_id)
        clock["now"] += 1.0  # the model finished serving one second ago
        manager.check_ttl()
        loaded_after_next_pass = stt.is_model_loaded(model_id)

    assert loaded_after_mid_request_pass, "the TTL pass during a request unloaded the model"
    assert loaded_after_next_pass, "TTL unloaded a model that finished a request 1s ago"


class _OneSTTModelRouter:
    """An STT model loaded and last used at a fixed time, at the router boundary."""

    def __init__(self, model_id: str, used_at: float) -> None:
        self._backends = {"faster-whisper": object()}
        self.models = {model_id: used_at}

    def is_model_loaded(self, model_id):
        return model_id in self.models

    def unload_model(self, model_id):
        self.models.pop(model_id, None)

    def loaded_models(self):
        from src.models import LoadedModelInfo

        return [
            LoadedModelInfo(model=m, backend="faster-whisper", device="cpu", compute_type="int8",
                            loaded_at=at, last_used_at=at)
            for m, at in self.models.items()
        ]

    def list_cached_models(self):
        return []


@pytest.mark.parametrize("eviction", ["ttl", "lru"])
def test_idle_eviction_counts_an_in_process_tts_model_idle_from_the_end_of_its_synthesis(eviction):
    # OS_MODEL_TTL counts idle seconds, and LRU picks the model used longest ago. Kokoro stamps its
    # last use when a synthesis starts; without the router's end-of-request stamp a 400 s synthesis
    # (a long chapter on CPU) looked idle for 430 s, 30 s after it ended: TTL unloaded it, and LRU
    # picked it over a model used during it.
    from src.model_manager import ModelManager
    from src.tts.backends import kokoro as kokoro_module
    from src.tts.router import TTSRouter

    clock = {"now": 900.0}
    fake_time = SimpleNamespace(time=lambda: clock["now"], monotonic=time.monotonic)

    class KPipeline:
        def __init__(self, lang_code, device, repo_id):
            pass

        def __call__(self, text, voice, speed):
            for index in range(3):
                if index == 2:
                    clock["now"] = 1400.0  # the synthesis started at 1000 and ran for 400 s
                yield text, "", np.zeros(240, dtype=np.float32)

    fake_kokoro = ModuleType("kokoro")
    fake_kokoro.KPipeline = KPipeline
    tts = TTSRouter(device="cpu", external_providers="")
    tts._backends.clear()
    tts._default_backend = None
    tts.register_backend("kokoro", kokoro_module.KokoroBackend(device="cpu"))
    stt = _OneSTTModelRouter("Systran/faster-whisper-small", used_at=1300.0)  # used while the TTS ran
    manager = ModelManager(stt_router=stt, tts_router=tts)

    # Unpinned, so eviction really looks at the default TTS model's idle time.
    with patch.dict(sys.modules, {"kokoro": fake_kokoro}), \
         patch.object(kokoro_module, "time", fake_time), \
         patch("src.tts.router.time", fake_time), \
         patch("src.model_manager.time", fake_time), \
         patch.object(settings, "os_model_ttl", 300), \
         patch.object(settings, "os_max_loaded_models", 1), \
         patch.object(settings, "os_pin_default_models", False):
        tts.load_model("kokoro")
        clock["now"] = 1000.0
        list(tts.synthesize(text="A long chapter.", model="kokoro", voice="af_heart"))
        clock["now"] = 1430.0  # 30 s after the synthesis ended
        if eviction == "ttl":
            manager.check_ttl()
        else:
            manager.evict_lru()

    assert tts.is_model_loaded("kokoro"), f"{eviction} unloaded Kokoro 30 s after its synthesis ended"
    if eviction == "lru":
        assert not stt.is_model_loaded("Systran/faster-whisper-small"), "LRU unloaded nothing"


class _IdleTTSRouter:
    """TTS models loaded ten minutes ago and idle since, as TTS_MODEL warmed at startup."""

    def __init__(self, *model_ids):
        self._backends = {"kokoro": object()}
        self.loaded = dict.fromkeys(model_ids, time.time() - 600)

    def loaded_models(self):
        return [
            TTSLoadedModelInfo(model=model_id, backend="kokoro", device="cpu", loaded_at=at, last_used_at=at)
            for model_id, at in self.loaded.items()
        ]

    def unload_model(self, model_id):
        self.loaded.pop(model_id, None)


def _whisper_module_decoding_until(decoding: threading.Event, finish: threading.Event) -> ModuleType:
    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            decoding.set()
            finish.wait(timeout=5)
            return [], SimpleNamespace(language="en", duration=1.0)

    module = _with_cached_files(ModuleType("faster_whisper"))
    module.WhisperModel = FakeWhisperModel
    return module


def _install_whisper(monkeypatch, module: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", module.utils)


def _start_decode(stt, model_id: str) -> threading.Thread:
    window = threading.Thread(
        target=stt.transcribe, args=(b"RIFF" + b"\x00" * 64, model_id),
        kwargs={"temperature_fallback": False},
    )
    window.start()
    return window


@pytest.mark.parametrize("moment", ["mid_decode", "between_windows"])
def test_lru_pass_keeps_the_model_a_session_uses_beside_the_pinned_defaults(monkeypatch, moment):
    # The image ships OS_MAX_LOADED_MODELS=2, and OS_PIN_DEFAULT_MODELS keeps the idle STT_MODEL
    # and TTS_MODEL (warmed at startup) loaded. Counted against the cap, they would make every
    # 30 s pass unload a live session's STT model, mid-decode or between two of its windows,
    # and the next window reload it.
    from src.model_manager import ModelManager
    from src.router import BackendRouter

    decoding, finish = threading.Event(), threading.Event()
    stt = BackendRouter()
    manager = ModelManager(stt_router=stt, tts_router=_IdleTTSRouter("kokoro"))
    live_model = "Systran/faster-whisper-large-v3"
    # Not patch.dict: an unload imports torch, which must stay imported after the test.
    _install_whisper(monkeypatch, _whisper_module_decoding_until(decoding, finish))

    with patch.object(settings, "os_model_ttl", 0), \
         patch.object(settings, "os_max_loaded_models", 2), \
         patch.object(settings, "stt_model", "Systran/faster-whisper-base"), \
         patch.object(settings, "tts_model", "kokoro"):
        stt.load_model("Systran/faster-whisper-base")
        stt.load_model(live_model)
        if moment == "between_windows":
            finish.set()
        window = _start_decode(stt, live_model)
        assert decoding.wait(timeout=5)
        if moment == "between_windows":
            window.join(timeout=5)
        eviction_pass = threading.Thread(target=ModelLifecycleManager(manager)._evict_sync)
        eviction_pass.start()
        eviction_pass.join(timeout=2)
        pass_waited = eviction_pass.is_alive()
        finish.set()
        window.join(timeout=5)
        eviction_pass.join(timeout=5)
        loaded = {m.id for m in manager.list_loaded()}

    assert not pass_waited, "the eviction pass waited for the running decode"
    assert loaded == {"Systran/faster-whisper-base", live_model, "kokoro"}


def test_lru_pass_skips_busy_models_and_evicts_beyond_the_cap_once_they_are_idle(monkeypatch):
    # Beyond the cap, LRU unloads the least recently used model, but never waits for a decode to
    # do so: faster-whisper has one inference lock for all its models, so the pass would sit on
    # the model operation lock for the rest of the transcription and then unload the next model.
    from src.model_manager import ModelManager
    from src.router import BackendRouter

    decoding, finish = threading.Event(), threading.Event()
    stt = BackendRouter()
    manager = ModelManager(stt_router=stt, tts_router=_IdleTTSRouter("kokoro"))
    lifecycle = ModelLifecycleManager(manager)
    idle_model, live_model = "Systran/faster-whisper-small", "Systran/faster-whisper-large-v3"
    _install_whisper(monkeypatch, _whisper_module_decoding_until(decoding, finish))

    with patch.object(settings, "os_model_ttl", 0), \
         patch.object(settings, "os_max_loaded_models", 1), \
         patch.object(settings, "stt_model", "Systran/faster-whisper-base"), \
         patch.object(settings, "tts_model", "kokoro"):
        for model_id in ("Systran/faster-whisper-base", idle_model, live_model):
            stt.load_model(model_id)
        stt.get_backend(idle_model)._last_used[idle_model] = time.time() - 600
        window = _start_decode(stt, live_model)
        assert decoding.wait(timeout=5)
        eviction_pass = threading.Thread(target=lifecycle._evict_sync)
        eviction_pass.start()
        eviction_pass.join(timeout=2)
        pass_waited = eviction_pass.is_alive()
        loaded_during_decode = {m.id for m in manager.list_loaded()}
        finish.set()
        window.join(timeout=5)
        eviction_pass.join(timeout=5)
        lifecycle._evict_sync()  # the next pass, after the decode
        loaded_after_decode = {m.id for m in manager.list_loaded()}

    assert not pass_waited, "the eviction pass waited for the running decode"
    assert loaded_during_decode == {"Systran/faster-whisper-base", idle_model, live_model, "kokoro"}
    assert loaded_after_decode == {"Systran/faster-whisper-base", live_model, "kokoro"}


def test_ttl_pass_leaves_an_idle_model_for_later_rather_than_wait_for_a_decode(monkeypatch):
    # Unloading the idle model needs faster-whisper's inference lock, which another model's long
    # transcription holds. Waiting for it would hold the model operation lock just as long.
    from src.model_manager import ModelManager
    from src.router import BackendRouter

    decoding, finish = threading.Event(), threading.Event()
    stt = BackendRouter()
    manager = ModelManager(stt_router=stt, tts_router=_IdleTTSRouter())
    idle_model, live_model = "Systran/faster-whisper-small", "Systran/faster-whisper-large-v3"
    _install_whisper(monkeypatch, _whisper_module_decoding_until(decoding, finish))

    with patch.object(settings, "os_model_ttl", 300), \
         patch.object(settings, "stt_model", "Systran/faster-whisper-base"):
        stt.load_model(idle_model)
        stt.load_model(live_model)
        stt.get_backend(idle_model)._last_used[idle_model] = time.time() - 600
        window = _start_decode(stt, live_model)
        assert decoding.wait(timeout=5)
        ttl_pass = threading.Thread(target=manager.check_ttl)
        ttl_pass.start()
        ttl_pass.join(timeout=2)
        pass_waited = ttl_pass.is_alive()
        idle_model_kept_during_decode = stt.is_model_loaded(idle_model)
        finish.set()
        window.join(timeout=5)
        ttl_pass.join(timeout=5)
        manager.check_ttl()  # the next pass
        idle_model_unloaded_after = not stt.is_model_loaded(idle_model)

    assert not pass_waited, "the TTL pass waited for another model's decode"
    assert idle_model_kept_during_decode
    assert idle_model_unloaded_after
    assert stt.is_model_loaded(live_model)
