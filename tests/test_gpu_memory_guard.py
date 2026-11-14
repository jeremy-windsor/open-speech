"""The GPU memory guard (src/gpu_memory.py): every load onto the GPU is checked against the card's free memory.

The card, the core backends and the workers are fakes: a FakeCard is the guard's probe, and each fake model takes its
size from it when it loads and gives it back when it unloads. The workers are real ExternalTTSBackends talking to a
fake worker's HTTP endpoints. No GPU is needed.
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.parse
from unittest.mock import patch

import numpy as np
import pytest

from src.config import settings
from src.gpu_memory import (
    BIG_MODEL_MB,
    GpuMemory,
    GpuMemoryGuard,
    GpuMemoryProbe,
    InsufficientGpuMemoryError,
    max_big_models,
)
from src.model_manager import ModelLifecycleError, ModelManager
from src.model_registry import get_known_model
from src.models import LoadedModelInfo
from src.router import BackendRouter
from src.tts.backends.base import TTSLoadedModelInfo, TTSModelManifest
from src.tts.router import TTSRouter

PARAKEET = "nvidia/parakeet-tdt-0.6b-v2"
WHISPER_SMALL = "Systran/faster-whisper-small"
WHISPER_MEDIUM = "Systran/faster-whisper-medium"
WHISPER_TURBO = "deepdml/faster-whisper-large-v3-turbo-ct2"
QWEN3 = "qwen3/0.6b-custom-voice"
CHATTERBOX_REGULAR = "chatterbox/regular"
CHATTERBOX_TURBO = "chatterbox/turbo"
FISH = "fish/s2-pro"
GB8, GB16, GB24 = 8192, 16384, 24576


def gpu_mb(model_id: str) -> int:
    return get_known_model(model_id)["gpu_mb"]


class FakeCard:
    """The GPU: what each model and other program holds on it. It is the guard's probe."""

    def __init__(self, total_mb: int, other_mb: int = 400) -> None:
        self.total_mb = total_mb
        self.held: dict[str, int] = {"other programs": other_mb}
        # Memory an unloaded worker model still holds until its process ends: key -> reads left.
        self._releasing: dict[str, int] = {}
        self.reads = 0

    def read(self) -> GpuMemory:
        self.reads += 1
        for key in list(self._releasing):
            self._releasing[key] -= 1
            if self._releasing[key] <= 0:
                del self._releasing[key]
                self.held.pop(key, None)
        return GpuMemory(free_mb=self.total_mb - sum(self.held.values()), total_mb=self.total_mb)

    def take(self, key: str, mb: int) -> None:
        self.held[key] = mb

    def give_back(self, key: str, *, after_reads: int = 0) -> None:
        if after_reads:
            self._releasing[key] = after_reads
        else:
            self.held.pop(key, None)


class FakeSttBackend:
    """A core STT backend on the GPU (faster-whisper or Parakeet): it loads a model on its first decode."""

    def __init__(self, name: str, card: FakeCard) -> None:
        self.name = name
        self.card = card
        self.loaded: dict[str, float] = {}
        self.busy: set[str] = set()
        self.loads: list[str] = []

    def load_model(self, model_id: str, _model_path: str | None = None) -> None:
        if model_id not in self.loaded:
            self.card.take(model_id, gpu_mb(model_id))
            self.loaded[model_id] = time.time()
            self.loads.append(model_id)

    def unload_model(self, model_id: str) -> None:
        self.loaded.pop(model_id, None)
        self.card.give_back(model_id)

    def unload_if_idle(self, model_id: str) -> bool:
        if model_id in self.busy:
            return False
        self.unload_model(model_id)
        return True

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self.loaded

    def loaded_models(self) -> list[LoadedModelInfo]:
        return [LoadedModelInfo(model=m, backend=self.name, device="cuda", compute_type="float16", loaded_at=t,
                                last_used_at=t) for m, t in self.loaded.items()]

    def list_cached_models(self) -> list:
        return []

    def transcribe(self, audio: bytes, model: str, **_kwargs) -> dict:
        self.load_model(model)  # as faster-whisper does on a first use
        return {"text": "ok"}


class FakeKokoro:
    """Kokoro on the GPU: it loads on its first synthesis."""

    name = "kokoro"
    sample_rate = 24000
    capabilities: dict = {}

    def __init__(self, card: FakeCard, device: str = "cuda") -> None:
        self.card = card
        self.device = device
        self.loaded_at: float | None = None

    def _get_device(self) -> str:
        return self.device

    def load_model(self, model_id: str = "kokoro") -> None:
        if self.loaded_at is None:
            self.card.take("kokoro", gpu_mb("kokoro"))
            self.loaded_at = time.time()

    def unload_model(self, model_id: str = "kokoro") -> None:
        self.loaded_at = None
        self.card.give_back("kokoro")

    def is_model_loaded(self, model_id: str) -> bool:
        return self.loaded_at is not None

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        if self.loaded_at is None:
            return []
        return [TTSLoadedModelInfo(model="kokoro", backend="kokoro", device=self.device, loaded_at=self.loaded_at,
                                   last_used_at=self.loaded_at)]

    def list_voices(self) -> list:
        return []

    def synthesize(self, text, voice, speed=1.0, lang_code=None, **_kwargs):
        self.load_model()
        yield np.zeros(4, dtype=np.float32)


class _Response(io.BytesIO):
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class FakeWorker:
    """A GPU worker's HTTP endpoints (worker contract v1). ``reports`` False is a worker from before the memory fields:
    no gpu_memory_mb in its manifest and no gpu_memory in its /health."""

    def __init__(self, provider: str, card: FakeCard, sizes: dict[str, int], *, reports: bool = True,
                 release_after_reads: int = 0) -> None:
        self.provider = provider
        self.card = card
        self.sizes = sizes
        self.reports = reports
        self.release_after_reads = release_after_reads
        self.loaded: str | None = None
        self.loaded_at: float | None = None
        self.loads: list[str] = []
        self.unloads: list[str] = []

    def _manifest(self) -> dict:
        models = []
        for model_id, mb in self.sizes.items():
            model = {"id": model_id, "provider": self.provider, "sample_rate": 24000, "capabilities": {},
                     "voices": []}
            if self.reports:
                model["gpu_memory_mb"] = mb
            models.append(model)
        return {"schema_version": 1, "provider": self.provider, "models": models}

    def _health(self) -> dict:
        health = {"status": "ok", "provider": self.provider, "loaded_model": self.loaded, "device": "cuda:0",
                  "loaded_at": self.loaded_at, "last_used_at": self.loaded_at, "busy": False}
        if self.reports:
            held = self.card.held.get(self.provider) if self.loaded else None
            need = self.sizes.get(self.loaded) if self.loaded else None
            health["gpu_memory"] = {"model_mb": held, "need_mb": need}
        return health

    def _unload(self) -> None:
        if self.loaded:
            self.unloads.append(self.loaded)
            self.loaded = None
            self.card.give_back(self.provider, after_reads=self.release_after_reads)

    def open(self, request, timeout=None):
        path = urllib.parse.urlsplit(request.full_url).path
        if path == "/v1/manifest":
            body = self._manifest()
        elif path == "/health":
            body = self._health()
        elif path == "/v1/models/load":
            model_id = json.loads(request.data)["model"]
            if self.loaded != model_id:
                self._unload()  # a worker holds one model
                self.card.take(self.provider, self.sizes[model_id])
                self.loaded, self.loaded_at = model_id, time.time()
                self.loads.append(model_id)
            body = {"status": "loaded", "model": model_id}
        elif path == "/v1/models/unload":
            self._unload()
            body = {"status": "unloaded"}
        else:  # pragma: no cover - the core asks nothing else here
            raise AssertionError(path)
        return _Response(json.dumps(body).encode())


class Server:
    """A core with Parakeet and Whisper, Kokoro, and the three GPU workers, on one fake card."""

    def __init__(self, card: FakeCard, *, workers_report: bool = True, release_after_reads: int = 0) -> None:
        self.card = card
        self.stt = BackendRouter()
        self.whisper = FakeSttBackend("faster-whisper", card)
        self.parakeet = FakeSttBackend("parakeet", card)
        self.stt._backends = {"faster-whisper": self.whisper, "parakeet": self.parakeet}
        self.stt._default_backend = self.whisper
        self.workers = {
            "qwen3": FakeWorker("qwen3", card, {QWEN3: 3500}, reports=workers_report,
                                release_after_reads=release_after_reads),
            "chatterbox": FakeWorker("chatterbox", card, {CHATTERBOX_REGULAR: 3000, CHATTERBOX_TURBO: 2500},
                                     reports=workers_report, release_after_reads=release_after_reads),
            "fish": FakeWorker("fish", card, {FISH: 3300}, reports=workers_report,
                               release_after_reads=release_after_reads),
        }
        urls = {name: f"http://{name}.test:8200" for name in self.workers}
        with patch("src.tts.router._discover_backends", return_value={}):
            self.tts = TTSRouter(device="cuda", external_providers=json.dumps(urls))
        self.kokoro = FakeKokoro(card)
        self.tts.register_backend("kokoro", self.kokoro)
        for name, worker in self.workers.items():
            self.tts._backends[name]._opener.open = worker.open
        self.manager = ModelManager(self.stt, self.tts)
        self.sleeps: list[float] = []
        self.manager.gpu_guard = GpuMemoryGuard(card, sleep=self.sleeps.append)

    def start_pinned_pair(self) -> None:
        """Parakeet and Kokoro, loaded at startup."""
        self.parakeet.load_model(PARAKEET)
        self.kokoro.load_model()

    def loaded(self) -> set[str]:
        return {m.id for m in self.manager.list_loaded()}


@pytest.fixture(autouse=True)
def guard_settings(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_memory_guard", True)
    monkeypatch.setattr(settings, "os_gpu_headroom_mb", 600)
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "auto")
    monkeypatch.setattr(settings, "os_pin_default_models", True)
    monkeypatch.setattr(settings, "stt_model", PARAKEET)
    monkeypatch.setattr(settings, "tts_model", "kokoro")
    monkeypatch.setattr(settings, "tts_enabled", True)
    monkeypatch.setattr(settings, "stt_device", "cuda")
    monkeypatch.setattr(settings, "tts_device", "cuda")


def _since(server: Server, model_id: str, seconds_ago: float) -> None:
    """Mark a loaded core STT model as last used this long ago."""
    backend = server.parakeet if model_id == PARAKEET else server.whisper
    backend.loaded[model_id] = time.time() - seconds_ago


# ── Fits, makes room, or refuses ─────────────────────────────────────────────────────────────────────────────────


def test_a_model_that_fits_loads_and_nothing_is_unloaded():
    server = Server(FakeCard(GB8))
    server.start_pinned_pair()

    server.manager.load(QWEN3, keep_pinned=True)

    assert server.loaded() == {PARAKEET, "kokoro", QWEN3}
    assert server.workers["qwen3"].loads == [QWEN3]
    assert server.card.reads > 0, "the guard read the card before the load"


def test_a_model_that_does_not_fit_unloads_the_least_recently_used_idle_model_first(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "0")  # free memory only
    server = Server(FakeCard(GB8))
    server.start_pinned_pair()
    server.whisper.load_model(WHISPER_MEDIUM)  # 2000 MiB, used an hour ago
    server.whisper.load_model(WHISPER_SMALL)  # 900 MiB, used a minute ago
    _since(server, WHISPER_MEDIUM, 3600)
    _since(server, WHISPER_SMALL, 60)
    _since(server, PARAKEET, 7200)  # the oldest, but pinned
    # 8192 - 400 - 1500 - 1230 - 2000 - 900 = 2162 MiB free; Chatterbox Turbo needs 2500 + 600.

    server.manager.load(CHATTERBOX_TURBO, keep_pinned=True)

    assert server.loaded() == {PARAKEET, "kokoro", WHISPER_SMALL, CHATTERBOX_TURBO}
    assert WHISPER_MEDIUM not in server.whisper.loaded


def test_a_model_that_cannot_fit_is_refused_naming_the_need_what_is_free_and_the_pinned_pair_stays():
    card = FakeCard(GB8, other_mb=3000)  # a game holds 2.6 GB besides the desktop
    server = Server(card)
    server.start_pinned_pair()
    # 8192 - 3000 - 1500 - 1230 = 2462 MiB free; Qwen3 needs 3500 + 600.

    with pytest.raises(ModelLifecycleError) as caught:
        server.manager.load(QWEN3, keep_pinned=True)

    error = caught.value
    assert (error.code, error.status_code) == ("insufficient_gpu_memory", 503)
    assert error.message.startswith(
        f"Not enough GPU memory to load {QWEN3}: it needs about 3.4 GiB plus 600 MiB of headroom, and 2.4 GiB of the "
        "card's 8.0 GiB is free."
    ), error.message
    assert f"The pinned defaults ({PARAKEET}, kokoro) stay loaded" in error.message
    assert error.details["gpu"] | {"unloadable": None} == {
        "model": QWEN3, "need_mb": 3500, "headroom_mb": 600, "free_mb": 2462, "total_mb": GB8, "unloadable": None,
    }
    assert server.loaded() == {PARAKEET, "kokoro"}
    assert server.workers["qwen3"].loads == [], "the worker was never asked to load"


def test_nothing_is_unloaded_when_unloading_every_idle_model_would_still_not_make_room(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "0")
    server = Server(FakeCard(GB8, other_mb=3000))
    server.start_pinned_pair()
    server.whisper.load_model(WHISPER_SMALL)
    # 8192 - 3000 - 1500 - 1230 - 900 = 1562 MiB free, 2462 with Whisper small unloaded; Qwen3 needs 4100.

    with pytest.raises(ModelLifecycleError) as caught:
        server.manager.load(QWEN3, keep_pinned=True)

    assert f"Even with {WHISPER_SMALL} (0.9 GiB) unloaded it wouldn't fit: close other programs" in caught.value.message
    assert server.loaded() == {PARAKEET, "kokoro", WHISPER_SMALL}


def test_a_busy_model_is_not_unloaded_and_the_refusal_says_it_is_busy(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "0")
    server = Server(FakeCard(GB8, other_mb=1300))
    server.start_pinned_pair()
    server.whisper.load_model(WHISPER_MEDIUM)
    server.whisper.busy.add(WHISPER_MEDIUM)  # transcribing a long file
    # 8192 - 1300 - 1500 - 1230 - 2000 = 2162 MiB free: unloading Whisper medium would make room (4162 for 4100), but
    # it is busy.

    with pytest.raises(ModelLifecycleError) as caught:
        server.manager.load(QWEN3, keep_pinned=True)

    assert caught.value.code == "insufficient_gpu_memory"
    assert f"Unload a model you aren't using ({WHISPER_MEDIUM} (2.0 GiB, busy))" in caught.value.message
    assert caught.value.details["gpu"]["unloadable"] == [{"model": WHISPER_MEDIUM, "gpu_mb": 2000, "busy": True}]
    assert server.loaded() == {PARAKEET, "kokoro", WHISPER_MEDIUM}


def test_the_refusal_reaches_the_models_view_as_a_503_with_the_message_and_the_figures():
    from src.services.models import _model_http_error

    server = Server(FakeCard(GB8, other_mb=3000))
    server.start_pinned_pair()
    with pytest.raises(ModelLifecycleError) as caught:
        server.manager.load(QWEN3, keep_pinned=True)

    http = _model_http_error(caught.value)

    assert http.status_code == 503
    assert http.detail["code"] == "insufficient_gpu_memory"
    assert http.detail["message"] == caught.value.message
    assert http.detail["gpu"]["need_mb"] == 3500


def test_an_unload_a_worker_finishes_later_is_waited_for_before_the_load():
    # Chatterbox and Fish give an unloaded model's memory back when their process restarts, seconds later.
    server = Server(FakeCard(GB8), release_after_reads=12)
    server.start_pinned_pair()
    server.manager.load(FISH, keep_pinned=True)  # 3300 MiB; 1762 MiB left

    server.manager.load(QWEN3, keep_pinned=True)

    assert server.workers["fish"].unloads == [FISH]
    assert server.loaded() == {PARAKEET, "kokoro", QWEN3}
    assert server.sleeps, "the guard waited for the worker to give the memory back"


def test_switching_a_workers_model_counts_the_model_it_replaces_as_free(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "0")
    server = Server(FakeCard(GB8))
    server.start_pinned_pair()
    server.whisper.load_model(WHISPER_SMALL)
    server.manager.load(CHATTERBOX_REGULAR, keep_pinned=True)
    # 8192 - 400 - 1500 - 1230 - 900 - 3000 = 1162 MiB free; Turbo needs 3100, and replaces regular's 3000. The
    # router's load is what a load that unloads nothing first (a download by loading) runs.

    server.tts.load_model(CHATTERBOX_TURBO)

    assert server.loaded() == {PARAKEET, "kokoro", WHISPER_SMALL, CHATTERBOX_TURBO}


# ── Every load path goes through it ──────────────────────────────────────────────────────────────────────────────


def test_a_transcription_that_loads_its_model_on_first_use_is_checked_first(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "0")
    server = Server(FakeCard(GB8, other_mb=2700))
    server.start_pinned_pair()
    # 8192 - 2700 - 1500 - 1230 = 2762 MiB free; Whisper turbo needs 2100 + 600 = 2700: fits.
    assert server.stt.transcribe(b"audio", WHISPER_TURBO) == {"text": "ok"}
    assert WHISPER_TURBO in server.whisper.loaded

    server.whisper.unload_model(WHISPER_TURBO)
    server.card.take("another program", 500)
    with pytest.raises(InsufficientGpuMemoryError) as caught:
        server.stt.transcribe(b"audio", WHISPER_TURBO)
    assert WHISPER_TURBO not in server.whisper.loaded, "the decode never loaded it"
    assert WHISPER_TURBO in str(caught.value)

    from src.services.stt import _client_error

    http = _client_error(caught.value)
    assert (http.status_code, http.detail["code"]) == (503, "insufficient_gpu_memory")
    assert http.detail["message"] == str(caught.value)


def test_a_speech_request_that_loads_kokoro_on_first_use_is_checked_first(monkeypatch):
    monkeypatch.setattr(settings, "tts_model", "piper/en_US-amy-medium")  # Kokoro isn't a pinned default here
    server = Server(FakeCard(GB8, other_mb=6000))
    server.parakeet.load_model(PARAKEET)
    # 8192 - 6000 - 1500 = 692 MiB free; Kokoro needs 1230 + 600.

    with pytest.raises(InsufficientGpuMemoryError):
        list(server.tts.synthesize(text="Hello.", model="kokoro", voice="af_heart"))
    assert server.kokoro.loaded_at is None

    server.card.give_back("other programs")
    assert len(list(server.tts.synthesize(text="Hello.", model="kokoro", voice="af_heart"))) == 1
    assert server.kokoro.loaded_at is not None


def test_a_cpu_only_server_loads_without_any_check(monkeypatch):
    class NoGpu:
        reads = 0

        def read(self):
            NoGpu.reads += 1
            return None

    server = Server(FakeCard(GB8, other_mb=GB8))  # a "full" card that no probe can see
    server.manager.gpu_guard = GpuMemoryGuard(NoGpu())
    server.start_pinned_pair()

    server.manager.load(QWEN3, keep_pinned=True)
    server.stt.transcribe(b"audio", WHISPER_TURBO)

    assert {QWEN3, WHISPER_TURBO} <= server.loaded()
    assert server.manager.gpu_admission(QWEN3, "tts") is None


def test_a_model_that_runs_on_the_cpu_is_not_checked(monkeypatch):
    server = Server(FakeCard(GB8, other_mb=GB8))  # nothing free on the GPU
    monkeypatch.setattr(settings, "stt_device", "cpu")
    server.kokoro.device = "cpu"

    assert server.manager.gpu_need_mb(WHISPER_TURBO, "stt") is None
    assert server.manager.gpu_need_mb("kokoro", "tts") is None
    assert server.manager.gpu_need_mb("piper/en_US-amy-medium", "tts") is None  # no gpu_mb: Piper is CPU-only
    assert server.manager.gpu_admission(WHISPER_TURBO, "stt") is None


def test_the_guard_turned_off_checks_nothing(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_memory_guard", False)
    server = Server(FakeCard(GB8, other_mb=GB8))
    server.manager.load(QWEN3)
    assert QWEN3 in server.loaded()
    assert server.card.reads == 0


# ── What the workers report ──────────────────────────────────────────────────────────────────────────────────────


def test_the_workers_own_figures_are_used_for_the_need_and_for_what_a_loaded_model_holds(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "0")
    server = Server(FakeCard(GB8, other_mb=2200))
    server.workers["qwen3"].sizes[QWEN3] = 2600  # bfloat16 on a newer card: less than the registry's 3500
    server.workers["chatterbox"].sizes[CHATTERBOX_TURBO] = 1800
    server.start_pinned_pair()
    server.manager.load(CHATTERBOX_TURBO, keep_pinned=True)
    busy = threading.Event()
    done = threading.Event()

    def speak():  # a synthesis holds the worker's lock: it is busy
        with server.tts._synthesis_lock_for(server.tts._backends["chatterbox"], CHATTERBOX_TURBO):
            busy.set()
            done.wait(5)

    thread = threading.Thread(target=speak)
    thread.start()
    busy.wait(5)
    try:
        # 8192 - 2200 - 1500 - 1230 - 1800 = 1462 MiB free, 3262 with Turbo unloaded; Qwen3 needs 2600 + 600.
        with pytest.raises(InsufficientGpuMemoryError) as caught:
            server.tts.load_model(QWEN3)
    finally:
        done.set()
        thread.join()

    figures = caught.value.details
    assert figures["need_mb"] == 2600, "the manifest's gpu_memory_mb, not the registry's"
    assert figures["unloadable"] == [{"model": CHATTERBOX_TURBO, "gpu_mb": 1800, "busy": True}], (
        "what the worker's /health says its model holds"
    )


def test_a_worker_from_before_the_memory_fields_is_sized_from_the_registry(monkeypatch):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", "0")
    server = Server(FakeCard(GB8, other_mb=1000), workers_report=False)
    server.start_pinned_pair()
    server.manager.load(FISH, keep_pinned=True)
    fish = server.tts._backends["fish"]
    assert fish.gpu_memory_need_mb(FISH) is None
    assert [m.gpu_memory_mb for m in fish.loaded_models()] == [None]
    lock = server.tts._synthesis_lock_for(fish, FISH)
    held = threading.Event()
    release = threading.Event()
    thread = threading.Thread(target=lambda: (lock.acquire(), held.set(), release.wait(5), lock.release()))
    thread.start()
    held.wait(5)
    try:
        with pytest.raises(InsufficientGpuMemoryError) as caught:
            server.tts.load_model(QWEN3)
    finally:
        release.set()
        thread.join()

    figures = caught.value.details
    assert figures["need_mb"] == gpu_mb(QWEN3) == 3500
    assert figures["unloadable"] == [{"model": FISH, "gpu_mb": gpu_mb(FISH), "busy": True}]


def test_a_manifest_reads_gpu_memory_mb_and_takes_a_missing_or_unusable_one_as_unknown():
    base = {"id": QWEN3, "provider": "qwen3", "sample_rate": 24000, "capabilities": {}}
    assert TTSModelManifest.from_dict({**base, "gpu_memory_mb": 2600}).gpu_memory_mb == 2600
    assert TTSModelManifest.from_dict({**base, "gpu_memory_mb": 2600}).to_dict()["gpu_memory_mb"] == 2600
    for value in (None, 0, -5, "2600", True, 2.5):
        manifest = TTSModelManifest.from_dict({**base, "gpu_memory_mb": value})
        assert manifest.gpu_memory_mb is None, value
        assert "gpu_memory_mb" not in manifest.to_dict()
    assert TTSModelManifest.from_dict(base).gpu_memory_mb is None


# ── Card-size defaults ───────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("total_mb", "setting", "cap"),
    [
        (7800, "auto", 1), (GB8, "auto", 1), (12288, "auto", 2), (15900, "auto", 2), (GB16, "auto", 2),
        (23600, "auto", None), (GB24, "auto", None), (49152, "auto", None),
        (GB8, "3", 3), (GB24, "1", 1), (GB8, "0", None),
    ],
)
def test_the_cap_on_big_models_comes_from_the_card_size_unless_it_is_set(monkeypatch, total_mb, setting, cap):
    monkeypatch.setattr(settings, "os_gpu_max_big_models", setting)
    assert max_big_models(total_mb) == cap


@pytest.mark.parametrize(
    ("total_mb", "unloaded"),
    [
        pytest.param(GB8, {QWEN3, FISH}, id="8-GB-one-big-model"),
        pytest.param(GB16, {QWEN3}, id="16-GB-two-big-models"),
        pytest.param(GB24, set(), id="24-GB-no-cap"),
    ],
)
def test_big_models_loaded_at_once_follow_the_card_size(total_mb, unloaded):
    server = Server(FakeCard(total_mb))
    server.start_pinned_pair()

    # 3500, 3300 and 2500 MiB, loaded in turn as requests would, each next to the models loaded before
    for model_id in (QWEN3, FISH, CHATTERBOX_TURBO):
        server.tts.load_model(model_id)
        time.sleep(0.01)

    assert server.loaded() == {PARAKEET, "kokoro", QWEN3, FISH, CHATTERBOX_TURBO} - unloaded


def test_the_cap_refuses_when_the_big_model_holding_its_place_is_busy():
    server = Server(FakeCard(GB16))
    server.start_pinned_pair()
    server.whisper.load_model(WHISPER_TURBO)  # 2100 MiB: a big model too
    server.whisper.busy.add(WHISPER_TURBO)
    server.manager.load(FISH, keep_pinned=True)
    lock = server.tts._synthesis_lock_for(server.tts._backends["fish"], FISH)
    held = threading.Event()
    release = threading.Event()
    thread = threading.Thread(target=lambda: (lock.acquire(), held.set(), release.wait(5), lock.release()))
    thread.start()
    held.wait(5)
    try:
        with pytest.raises(InsufficientGpuMemoryError) as caught:
            server.tts.load_model(QWEN3)
    finally:
        release.set()
        thread.join()

    assert caught.value.code == "insufficient_gpu_memory"
    assert "holds at most 2 big models next to the pinned defaults (OS_GPU_MAX_BIG_MODELS)" in caught.value.message
    assert server.loaded() == {PARAKEET, "kokoro", WHISPER_TURBO, FISH}


def test_a_pinned_default_does_not_count_against_the_cap(monkeypatch):
    monkeypatch.setattr(settings, "tts_model", QWEN3)  # a big pinned default
    server = Server(FakeCard(12288))  # a 12 GB card: two big models, and the pinned one on top
    server.parakeet.load_model(PARAKEET)
    server.manager.load(QWEN3)
    server.tts.load_model(FISH)
    server.tts.load_model(CHATTERBOX_TURBO)
    assert {QWEN3, FISH, CHATTERBOX_TURBO} <= server.loaded()


def test_the_startup_line_names_the_card_and_its_cap():
    guard = GpuMemoryGuard(FakeCard(GB8))
    assert guard.describe() == (
        "GPU memory guard: 8.0 GiB card, 7.6 GiB free; at most 1 big model(s) next to the pinned defaults; "
        "600 MiB headroom"
    )
    assert "no NVIDIA GPU" in GpuMemoryGuard(type("NoGpu", (), {"read": lambda self: None})()).describe()


# ── The probe ────────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_probe_falls_back_to_nvidia_smi_and_to_nothing_without_a_driver(monkeypatch):
    import src.gpu_memory as gpu_memory

    monkeypatch.setattr(gpu_memory, "_nvml_reader", lambda: None)
    monkeypatch.setattr(gpu_memory.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    runs = []

    def run(command, **kwargs):
        runs.append(command)
        return type("Done", (), {"stdout": "5321, 8192\n"})()

    monkeypatch.setattr(gpu_memory.subprocess, "run", run)
    assert GpuMemoryProbe().read() == GpuMemory(free_mb=5321, total_mb=8192)
    assert "--query-gpu=memory.free,memory.total" in runs[0]

    monkeypatch.setattr(gpu_memory.shutil, "which", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert GpuMemoryProbe().read() is None


def test_the_probe_never_starts_a_cuda_context_through_torch(monkeypatch):
    import sys

    import src.gpu_memory as gpu_memory

    monkeypatch.setattr(gpu_memory, "_nvml_reader", lambda: None)
    monkeypatch.setattr(gpu_memory, "_nvidia_smi_reader", lambda: None)
    asked = []
    cuda = type("Cuda", (), {
        "is_initialized": staticmethod(lambda: False),
        "mem_get_info": staticmethod(lambda index: asked.append(index) or (1 * 2**30, 8 * 2**30)),
    })
    monkeypatch.setitem(sys.modules, "torch", type("Torch", (), {"cuda": cuda}))
    probe = GpuMemoryProbe()
    assert probe.read() is None and asked == []

    cuda.is_initialized = staticmethod(lambda: True)  # Kokoro loaded on the GPU: the context exists
    assert probe.read() == GpuMemory(free_mb=1024, total_mb=8192)


def test_big_means_two_gib():
    assert BIG_MODEL_MB == 2048
    assert gpu_mb(WHISPER_TURBO) >= BIG_MODEL_MB > gpu_mb(WHISPER_MEDIUM)
