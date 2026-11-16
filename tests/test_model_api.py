"""Integration tests for model management API."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import main as main_module
from src import router as router_module
from src.config import settings
from src.main import app
from src.model_manager import ModelState
from src.models import LoadedModelInfo
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
        # TestClient triggers lifespan; the preload runs in the background once the server has started.
        with TestClient(app):
            deadline = time.monotonic() + 10
            while mock.load_model.call_count < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            calls = [c[0][0] for c in mock.load_model.call_args_list]
            assert calls == ["base-model", "extra-model"]


def test_startup_preloads_the_default_tts_model_in_the_background():
    # Keep the default warm (TTS_PRELOAD_DEFAULT): startup must not wait for the load, and the
    # first speech request should find TTS_MODEL loaded.
    from src.model_manager import ModelManager

    loading, release, done, loaded = threading.Event(), threading.Event(), threading.Event(), []

    class TTSRouter:
        def __init__(self):
            self._backends = {"kokoro": object()}

        def load_model(self, model_id):
            loading.set()
            release.wait(timeout=5)
            loaded.append(model_id)
            done.set()

        def loaded_models(self):
            return []

    stt = MagicMock(_backends={})
    stt.loaded_models.return_value = []
    stt.list_cached_models.return_value = []
    manager = ModelManager(stt_router=stt, tts_router=TTSRouter())
    try:
        with patch.object(main_module, "model_manager", manager), \
             patch.object(manager, "_hf_cache_roots", return_value=[]), \
             patch.object(settings, "tts_model", "kokoro"), \
             patch.object(settings, "tts_preload_default", True):
            with TestClient(app) as client:
                # The app serves requests while the default model is still loading.
                loading_after_startup = loading.wait(timeout=5) and not loaded
                health = client.get("/health").status_code
                release.set()
                done.wait(timeout=5)
    finally:
        release.set()

    assert loading_after_startup, "startup did not preload TTS_MODEL, or waited for its load"
    assert health == 200
    assert loaded == ["kokoro"]


def test_app_startup_in_tests_does_not_warm_the_default_tts_model():
    # tests/conftest.py sets TTS_PRELOAD_DEFAULT=false: on a machine with Kokoro installed, every
    # lifespan in the suite would otherwise load, or first download, TTS_MODEL in the background.
    warm_ups = threading.Event()

    async def warm_default_tts():
        warm_ups.set()

    with patch.object(main_module, "_warm_default_tts", warm_default_tts):
        with TestClient(app) as client:
            health = client.get("/health").status_code  # the loop has run any task startup made

    assert health == 200
    assert not warm_ups.is_set(), "a test's app startup began warming TTS_MODEL"
    assert settings.tts_preload_default is False


@pytest.mark.asyncio
async def test_default_tts_warm_up_waits_for_an_isolated_worker_that_starts_later():
    # Workers boot on their own, often after this server, and load a model only when asked: a
    # default worker model found offline at startup would otherwise stay cold until first use.
    from types import SimpleNamespace

    from src.model_manager import ModelManager

    class TTSRouter:
        def __init__(self):
            self.up = False
            self.health_checks = 0
            self.loaded = []
            worker = SimpleNamespace(advertises_model=lambda model_id: True if self.up else None)
            self._backends = {"chatterbox": worker}

        def provider_is_available(self, provider):
            self.health_checks += 1
            self.up = self.health_checks >= 3  # the worker answers from its third check on
            return self.up

        def load_model(self, model_id):
            self.loaded.append(model_id)

        def loaded_models(self):
            return []

    stt = MagicMock(_backends={})
    stt.loaded_models.return_value = []
    stt.list_cached_models.return_value = []
    tts = TTSRouter()
    manager = ModelManager(stt_router=stt, tts_router=tts)

    with patch.object(main_module, "model_manager", manager), \
         patch.object(main_module, "TTS_WARM_RETRY_S", 0.01), \
         patch.object(manager, "_hf_cache_roots", return_value=[]), \
         patch.object(settings, "tts_model", "chatterbox/regular"):
        await asyncio.wait_for(main_module._warm_default_tts(), timeout=10)

    assert tts.loaded == ["chatterbox/regular"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ["load", "download"])
async def test_model_progress_operation_does_not_block_event_loop(operation_name):
    service = ModelProgressService()
    model_manager = MagicMock()
    model_manager._resolve_type.return_value = "tts"  # picks the operation's type lock
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


class _BlockingSTTBackend:
    """Backend whose model I/O waits until the event loop proves it is still running."""

    name = "faster-whisper"

    def __init__(self, blocking_op):
        self.blocking_op = blocking_op
        self.started = threading.Event()
        self.release = threading.Event()
        self.released_while_blocked: bool | None = None

    def _maybe_block(self, op):
        if op == self.blocking_op:
            self.started.set()
            self.released_while_blocked = self.release.wait(timeout=2)

    def load_model(self, model_id):
        self._maybe_block("load")

    def unload_model(self, model_id):
        self._maybe_block("unload")

    def loaded_models(self):
        return []

    def is_model_loaded(self, model_id):
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "blocking_op"),
    [
        ("POST", "/api/ps/Systran/faster-whisper-base", "load"),
        ("DELETE", "/api/ps/Systran/faster-whisper-base", "unload"),
        ("POST", "/api/pull/Systran/faster-whisper-base", "load"),
    ],
)
async def test_legacy_model_endpoints_do_not_block_event_loop(method, path, blocking_op):
    backend = _BlockingSTTBackend(blocking_op)

    async def heartbeat():
        # Only completes if the event loop keeps running while the model op is in flight.
        if await asyncio.to_thread(backend.started.wait, 5):
            backend.release.set()

    with patch.object(router_module.router, "_default_backend", backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": backend}):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            beat = asyncio.create_task(heartbeat())
            try:
                response = await asyncio.wait_for(client.request(method, path), timeout=10)
            finally:
                backend.release.set()
                await asyncio.wait_for(beat, timeout=10)

    assert response.status_code == 200
    assert backend.released_while_blocked is True, (
        f"{method} {path} ran blocking model I/O on the asyncio event loop"
    )


class _OverlappingLoadSTTBackend:
    """Fake faster-whisper whose first load waits (bounded) for a second request's listing.

    Two loads allowed to overlap both find no other STT model to unload and both stay loaded.
    """

    name = "faster-whisper"

    def __init__(self):
        self._state = threading.Lock()
        self.loaded: set[str] = set()
        self.listings = 0
        self.second_listing = threading.Event()

    def loaded_models(self):
        with self._state:
            self.listings += 1
            if self.listings >= 2:
                self.second_listing.set()
            ids = sorted(self.loaded)
        return [
            LoadedModelInfo(model=m, backend=self.name, device="cpu", compute_type="int8", loaded_at=time.time())
            for m in ids
        ]

    def load_model(self, model_id):
        self.second_listing.wait(timeout=1.0)
        with self._state:
            self.loaded.add(model_id)

    def unload_model(self, model_id):
        with self._state:
            self.loaded.discard(model_id)

    def is_model_loaded(self, model_id):
        with self._state:
            return model_id in self.loaded

    def list_cached_models(self):
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("first_route", ["/api/ps/", "/api/models/"])
async def test_concurrent_stt_loads_leave_one_stt_model_loaded(first_route):
    # A legacy load unloads every other STT model first, as the unified load does; both take the
    # one model-operation lock, so two loads cannot both find nothing to unload.
    backend = _OverlappingLoadSTTBackend()
    requests = [
        ("/api/ps/Systran/faster-whisper-base" if first_route == "/api/ps/"
         else "/api/models/Systran/faster-whisper-base/load"),
        "/api/ps/Systran/faster-whisper-small",
    ]

    with patch.object(router_module.router, "_default_backend", backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": backend}), \
         patch.object(main_module, "progress_service", ModelProgressService()):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.wait_for(
                asyncio.gather(*(client.post(path) for path in requests)), timeout=20,
            )

    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert len(backend.loaded) == 1, f"two STT models stayed loaded: {sorted(backend.loaded)}"


def test_legacy_pull_of_the_loaded_stt_model_keeps_it_loaded():
    # CHANGELOG: "Model downloads preserve the provider's previously active model". A backend that
    # can fetch only by loading must not unload the model that serves transcriptions.
    backend = _OverlappingLoadSTTBackend()
    backend.second_listing.set()
    backend.loaded.add("Systran/faster-whisper-base")

    with patch.object(router_module.router, "_default_backend", backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": backend}):
        resp = TestClient(app).post("/api/pull/Systran/faster-whisper-base")

    assert resp.status_code == 200, resp.text
    assert backend.loaded == {"Systran/faster-whisper-base"}


class _SwapRouter:
    """An STT router whose load of the "broken" or an unknown model fails, as a missing or corrupt model does."""

    def __init__(self, loaded):
        self.loaded = set(loaded)
        self.loads: list[str] = []

    def loaded_models(self):
        return [SimpleNamespace(model=model_id) for model_id in sorted(self.loaded)]

    def unload_model(self, model_id):
        self.loaded.discard(model_id)

    def load_model(self, model_id):
        self.loads.append(model_id)
        if model_id == "Systran/faster-whisper-broken":
            raise RuntimeError("corrupt model files")
        if model_id == "no-such-model":
            raise ValueError("Unknown model 'no-such-model'")
        self.loaded.add(model_id)


@pytest.mark.parametrize(("model", "status"), [("Systran/faster-whisper-broken", 500), ("no-such-model", 404)])
def test_a_failed_legacy_load_loads_the_pinned_default_again(model, status):
    # POST /api/ps/{model} unloads the other STT models first; a load that then failed left none loaded, the
    # pinned STT_MODEL among them, until a request loaded it again. The unified load restores them.
    from fastapi import HTTPException

    from src.services.models import load_legacy_model

    parakeet = "nvidia/parakeet-tdt-0.6b-v2"
    router = _SwapRouter({parakeet})
    with pytest.raises(HTTPException) as caught:
        load_legacy_model(model=model, backend_router=router)

    assert caught.value.status_code == status
    assert router.loaded == {parakeet}
    assert router.loads == [model, parakeet]


def test_legacy_pull_downloads_without_loading(tmp_path, monkeypatch):
    import faster_whisper
    import faster_whisper.utils

    from src.backends.faster_whisper import FasterWhisperBackend

    fetched, constructed = [], []
    monkeypatch.setattr(
        faster_whisper.utils, "download_model",
        lambda model_id, cache_dir=None: fetched.append((model_id, cache_dir)),
    )
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *args, **kwargs: constructed.append(args))
    monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))
    backend = FasterWhisperBackend()

    with patch.object(router_module.router, "_default_backend", backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": backend}):
        resp = TestClient(app).post("/api/pull/Systran/faster-whisper-small")

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "downloaded"
    assert fetched == [("Systran/faster-whisper-small", str(tmp_path))]
    assert constructed == []
    assert backend.loaded_models() == []


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


# Ported from the old web UI's flow tests: the answers the Models view shows.


def test_unloading_an_stt_model_that_is_not_loaded_answers_404_not_loaded(client):
    # Models › Unload uses DELETE /api/models/{id}. When OS_MODEL_TTL already unloaded the STT model behind
    # a stale "Loaded" row, the UI shows this answer and refreshes the row.
    c, _backend = client
    resp = c.delete("/api/models/Systran/faster-whisper-base")

    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "not_loaded" and "not loaded" in error["message"]


def test_a_worker_model_without_its_worker_reads_provider_missing(monkeypatch):
    # TTS_MODEL names an isolated worker model (docker-compose.voice-models.yml) whose worker is not in
    # TTS_EXTERNAL_PROVIDERS: the list and the status route both say provider_missing, which the Models view
    # turns into "Worker not configured" with the worker compose file as the hint.
    monkeypatch.setattr(settings, "tts_model", "chatterbox/turbo")
    client = TestClient(app)

    listed = client.get("/api/models")
    status = client.get("/api/models/chatterbox/turbo/status")

    assert listed.status_code == 200 and status.status_code == 200
    row = next(m for m in listed.json()["models"] if m["id"] == "chatterbox/turbo")
    assert row["state"] == "provider_missing" and row["provider"] == "chatterbox"
    assert status.json()["state"] == "provider_missing"


# /v1/audio/models/load and /unload ran in asyncio.to_thread, and waited there for
# the TTS operation lock and then the provider's synthesis lock. Behind a busy provider each queued call held a
# thread of asyncio's default executor, which every other request and to_thread call shares.
class _Counts:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.values: dict[str, int] = {}

    def add(self, key: str) -> None:
        with self._lock:
            self.values[key] = self.values.get(key, 0) + 1

    def get(self, key: str) -> int:
        with self._lock:
            return self.values.get(key, 0)


class _CountingOperationLock:
    """A real RLock standing in for ModelManager.operation_lock("tts"), counting each call that waits for it."""

    def __init__(self, counts: _Counts) -> None:
        self._lock = threading.RLock()
        self._counts = counts

    def __enter__(self):
        self._counts.add("at_operation_lock")
        self._lock.acquire(timeout=30)
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


class _GatedTTSProvider:
    """A TTS provider whose synthesis, once started, waits until the test opens its gate."""

    sample_rate = 24000
    capabilities = {"speed_control": True}

    def __init__(self, name: str, counts: _Counts, gate: threading.Event | None = None) -> None:
        self.name, self._counts, self._gate = name, counts, gate

    def synthesize(self, text, voice, speed=1.0, lang_code=None, **_options):
        self._counts.add(f"{self.name}_synthesizing")
        if self._gate is not None:
            self._gate.wait(timeout=30)
        yield np.zeros(2400, dtype=np.float32)

    def load_model(self, model_id):
        return None

    def unload_model(self, model_id):
        return None

    def is_model_loaded(self, model_id):
        return True

    def loaded_models(self):
        return []

    def validate_voice(self, voice):
        return None

    def list_voices(self):
        return []


def _tts_operations_app(tmp_path, counts: _Counts, gate: threading.Event) -> FastAPI:
    from src.cache.tts_cache import TTSCache
    from src.routes.tts import create_router
    from src.tts.router import TTSRouter

    with patch("src.tts.router._discover_backends", return_value={}):
        tts_router = TTSRouter(device="cpu", external_providers="")
    tts_router.register_backend("slow", _GatedTTSProvider("slow", counts, gate))
    tts_router.register_backend("fast", _GatedTTSProvider("fast", counts))
    test_settings = SimpleNamespace(
        tts_enabled=True, tts_model="slow", tts_voice="v", tts_speed=1.0, tts_default_format="pcm",
        tts_max_input_length=4096, os_history_enabled=False, tts_trim_silence=False,
        tts_normalize_output=False, tts_cache_enabled=False, os_effects_enabled=False,
    )
    operation_lock = _CountingOperationLock(counts)

    def model_manager():
        counts.add("dispatched")  # a load or unload route asks for the lock before it runs the operation
        return SimpleNamespace(operation_lock=lambda _model_type: operation_lock, list_all=lambda: [])

    cache = TTSCache(str(tmp_path / "cache"), enabled=False)
    app = FastAPI()
    app.include_router(create_router(
        get_settings=lambda: test_settings, get_tts_router=lambda: tts_router, get_tts_cache=lambda: cache,
        get_pronunciation_dict=lambda: SimpleNamespace(apply=lambda text: text),
        get_history_manager=MagicMock, get_voice_library=MagicMock, get_model_manager=model_manager,
    ))
    return app


async def _answer_within(awaitable, timeout: float):
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError:
        return f"no answer within {timeout:g} s"


@pytest.mark.parametrize("operation", ["load", "unload"])
async def test_tts_model_operations_queued_behind_a_busy_provider_leave_the_shared_executor_free(tmp_path, operation):
    workers = 4
    queued = workers + 2
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=workers))  # this test's loop only
    counts, gate = _Counts(), threading.Event()
    app = _tts_operations_app(tmp_path, counts, gate)
    speech = {"input": "Hello", "voice": "v", "response_format": "pcm"}

    def settled() -> bool:
        # Either `workers` calls wait at the operation lock, or every call was dispatched and one waits there.
        arrived = counts.get("at_operation_lock")
        return arrived >= workers or (counts.get("dispatched") >= queued and arrived >= 1)

    async def until(condition, what: str) -> None:
        for _ in range(1000):
            if condition():
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"never reached: {what} ({counts.values})")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

        async def speak(model: str):
            response = await client.post("/v1/audio/speech", json={**speech, "model": model})
            return response.status_code, len(response.content)

        async def model_operation():
            response = await client.post(f"/v1/audio/models/{operation}", json={"model": "slow"})
            return response.status_code, response.json()

        async def listing():
            return (await client.get("/v1/audio/models")).status_code

        busy = asyncio.create_task(speak("slow"))  # 'slow' synthesizes behind the gate, holding its lock
        operations: list[asyncio.Task] = []
        try:
            await until(lambda: counts.get("slow_synthesizing") >= 1, "the slow provider synthesizing")
            operations = [asyncio.create_task(model_operation()) for _ in range(queued)]
            await until(settled, "the model operations at the TTS operation lock")
            await asyncio.sleep(0.3)  # the last ones dispatch their work
            fast = await _answer_within(speak("fast"), 5)
            listed = await _answer_within(listing(), 3)
            bare = await _answer_within(asyncio.to_thread(lambda: "ran"), 2)
        finally:
            gate.set()
            results = await asyncio.wait_for(asyncio.gather(busy, *operations, return_exceptions=True), timeout=30)

    assert fast == (200, 4800), (
        f"a speech request for the idle 'fast' provider: {fast} while {queued} /v1/audio/models/{operation} calls "
        f"waited behind the busy 'slow' provider, with {workers} default-executor threads ({counts.values})"
    )
    assert listed == 200, f"GET /v1/audio/models: {listed}"
    assert bare == "ran", f"a bare asyncio.to_thread: {bare}"
    status = "loaded" if operation == "load" else "unloaded"
    assert results == [(200, 4800)] + [(200, {"status": status, "model": "slow"})] * queued, results
