"""faster-whisper backend: locking, temperature fallback and subtitle timestamps."""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import wave
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import router as router_module
from src.backends.faster_whisper import FasterWhisperBackend
from src.main import app

MODEL_ID = "test-model"


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _segment(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        seek=0, start=start, end=end, text=text, tokens=[], temperature=0.0,
        avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
    )


def _backend_with_blocking_model(started: threading.Event, release: threading.Event):
    """Backend whose model holds inference open until ``release`` is set."""
    def transcribe(path, **kwargs):
        def segments():
            # faster-whisper decodes lazily, so inference runs while segments are consumed.
            started.set()
            release.wait(10)
            yield _segment(0.0, 1.0, " hi")

        return segments(), SimpleNamespace(language="en", duration=1.0)

    backend = FasterWhisperBackend()
    backend._models[MODEL_ID] = SimpleNamespace(transcribe=transcribe)
    backend._loaded_at[MODEL_ID] = time.time()
    return backend


class _ContendedLock:
    """RLock that records when another thread has to wait for it."""

    def __init__(self):
        self._lock = threading.RLock()
        self.contended = threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def test_health_responds_while_a_transcription_is_running():
    # /health is the Docker HEALTHCHECK liveness probe (3 s timeout); it must not wait
    # for an unrelated transcription to finish.
    started, release = threading.Event(), threading.Event()
    backend = _backend_with_blocking_model(started, release)
    health = {}

    with patch.object(router_module.router, "_default_backend", backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": backend}):
        transcriber = threading.Thread(
            target=backend.transcribe, args=(_wav_bytes(), MODEL_ID), daemon=True
        )
        prober = threading.Thread(
            target=lambda: health.update(resp=TestClient(app).get("/health")), daemon=True
        )
        transcriber.start()
        try:
            assert started.wait(5)
            prober.start()
            prober.join(timeout=5)
            answered_during_inference = not prober.is_alive()
        finally:
            release.set()
            transcriber.join(timeout=10)
            if prober.ident is not None:
                prober.join(timeout=10)

    assert answered_during_inference, "/health waited for the transcription to finish"
    assert health["resp"].status_code == 200
    assert health["resp"].json()["models_loaded"] == 1


def _whisper_modules(whisper_model, download_model=None) -> dict:
    """A fake ``faster_whisper`` and ``faster_whisper.utils``, for ``patch.dict(sys.modules, ...)``.

    load_model fetches with ``faster_whisper.utils.download_model`` before it builds the model. The default fake
    download finds the files cached and returns the model id as their path.
    """
    utils = ModuleType("faster_whisper.utils")
    utils.download_model = download_model or (lambda model_id, cache_dir=None, **kwargs: model_id)
    module = ModuleType("faster_whisper")
    module.utils = utils
    module.WhisperModel = whisper_model
    return {"faster_whisper": module, "faster_whisper.utils": utils}


def test_status_reads_do_not_wait_for_a_model_load():
    # First use downloads the model inside load_model; status reads must not queue behind it.
    started, release = threading.Event(), threading.Event()

    class SlowWhisperModel:
        def __init__(self, *args, **kwargs):
            started.set()
            release.wait(10)

    backend = FasterWhisperBackend()
    status = {}

    def read_status():
        status["loaded"] = backend.loaded_models()
        status["is_loaded"] = backend.is_model_loaded(MODEL_ID)

    with patch.dict(sys.modules, _whisper_modules(SlowWhisperModel)):
        loader = threading.Thread(target=backend.load_model, args=(MODEL_ID,), daemon=True)
        reader = threading.Thread(target=read_status, daemon=True)
        loader.start()
        try:
            assert started.wait(5)
            reader.start()
            reader.join(timeout=5)
            answered_during_load = not reader.is_alive()
        finally:
            release.set()
            loader.join(timeout=10)
            if reader.ident is not None:
                reader.join(timeout=10)

    assert answered_during_load, "status reads waited for the model load to finish"
    assert status == {"loaded": [], "is_loaded": False}
    assert backend.is_model_loaded(MODEL_ID)


def test_unload_waits_for_an_in_flight_transcription():
    # An idle-TTL unload must not free the model mid-request, even though status reads
    # no longer wait for inference.
    started, release = threading.Event(), threading.Event()
    backend = _backend_with_blocking_model(started, release)
    backend._inference_lock = lock = _ContendedLock()
    result = {}

    transcriber = threading.Thread(
        target=lambda: result.update(backend.transcribe(_wav_bytes(), MODEL_ID)), daemon=True
    )
    unloader = threading.Thread(target=backend.unload_model, args=(MODEL_ID,), daemon=True)
    transcriber.start()
    try:
        assert started.wait(5)
        unloader.start()
        assert lock.contended.wait(5)
        assert backend.is_model_loaded(MODEL_ID)
    finally:
        release.set()
        transcriber.join(timeout=10)
        if unloader.ident is not None:
            unloader.join(timeout=10)

    assert result["text"] == "hi"
    assert not backend.is_model_loaded(MODEL_ID)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0.0, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)), (0.4, 0.4)],
)
def test_temperature_zero_keeps_whisper_fallback_schedule(requested, expected):
    # OpenAI: at temperature 0 "the model will use log probability to automatically increase
    # the temperature until certain thresholds are hit"; an explicit temperature stays fixed.
    calls = []

    def transcribe(path, **kwargs):
        calls.append(kwargs)
        return iter([_segment(0.0, 1.0, " hi")]), SimpleNamespace(language="en", duration=1.0)

    backend = FasterWhisperBackend()
    backend._models[MODEL_ID] = SimpleNamespace(transcribe=transcribe)

    backend.transcribe(_wav_bytes(), MODEL_ID, temperature=requested)

    assert calls[0]["temperature"] == expected


def test_live_decode_without_fallback_runs_once_at_temperature_zero():
    # Live paths pass temperature_fallback=False, so a window cut mid-word that
    # fails the thresholds is not decoded up to six times; file transcription keeps the schedule.
    calls = []

    def transcribe(path, **kwargs):
        calls.append(kwargs)
        return iter([_segment(0.0, 1.0, " hi")]), SimpleNamespace(language="en", duration=1.0)

    backend = FasterWhisperBackend()
    backend._models[MODEL_ID] = SimpleNamespace(transcribe=transcribe)

    backend.transcribe(_wav_bytes(), MODEL_ID, temperature=0.0, temperature_fallback=False)
    backend.transcribe(_wav_bytes(), MODEL_ID, temperature=0.4, temperature_fallback=False)

    assert [call["temperature"] for call in calls] == [0.0, 0.4]


@pytest.mark.parametrize(("pinned", "expected_ttl"), [(True, None), (False, 300.0)])
def test_pinned_default_model_reports_no_idle_timeout(monkeypatch, pinned, expected_ttl):
    # OS_PIN_DEFAULT_MODELS: TTL eviction never unloads STT_MODEL, so /api/ps shows no countdown.
    from src.config import settings

    monkeypatch.setattr(settings, "os_model_ttl", 300)
    monkeypatch.setattr(settings, "os_pin_default_models", pinned)
    monkeypatch.setattr(settings, "stt_model", MODEL_ID)
    backend = FasterWhisperBackend()
    backend._models[MODEL_ID] = object()
    backend._loaded_at[MODEL_ID] = backend._last_used[MODEL_ID] = time.time()

    ttl_remaining = backend.loaded_models()[0].ttl_remaining

    assert ttl_remaining == (None if expected_ttl is None else pytest.approx(expected_ttl, abs=5))


def test_empty_stt_model_dir_downloads_into_the_scanned_hugging_face_cache(tmp_path, monkeypatch):
    # .env.example documents "STT_MODEL_DIR=" as optional and Compose passes it through empty.
    # snapshot_download(cache_dir="") writes models--<id> into the working directory instead.
    from src.config import settings

    captured = {}

    class WhisperModel:
        def __init__(self, model_id, device, compute_type, download_root=None):
            captured["download_root"] = download_root

    def download_model(model_id, cache_dir=None, **kwargs):
        captured["cache_dir"] = cache_dir
        return model_id

    for name, module in _whisper_modules(WhisperModel, download_model).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(settings, "stt_model_dir", "")
    backend = FasterWhisperBackend()

    backend.load_model(MODEL_ID)

    assert captured["download_root"] is None  # huggingface_hub's own cache, which is scanned
    assert captured["cache_dir"] is None  # the download before the load goes there too
    assert backend._get_cache_dir() == tmp_path / "hub"


# faster-whisper 1.2.1's WhisperModel.__init__ downloads a hub id before it builds
# the model (transcribe.py:678-687), and load_model built it under _inference_lock. So the first translation on
# STT_TRANSLATE_MODEL (1.5 or 3 GB, "downloaded on the first translation", README) held every decode, load and unload
# of the other models for the whole download; CHANGELOG: "no load or unload waits for a download". The fake below
# downloads in WhisperModel.__init__ as the real one does, unless it is given the downloaded files' path.
_LOADED_STT = "deepdml/faster-whisper-large-v3-turbo-ct2"
_FIRST_USE = "Systran/faster-whisper-medium"
_CACHED = "/cache/"


class _LoadedWhisper:
    def transcribe(self, path, **kwargs):
        return iter([_segment(0.0, 1.0, " hello")]), SimpleNamespace(language="en", duration=1.0)


def _blocking_download_modules(download_started: threading.Event, release_download: threading.Event):
    built: list[str] = []

    def download_model(model_id, cache_dir=None, **kwargs):
        download_started.set()
        release_download.wait(20)  # a 1.5 GB hub download
        return _CACHED + model_id

    class WhisperModel:
        def __init__(self, model_size_or_path, device="auto", compute_type="default", download_root=None, **kwargs):
            if not model_size_or_path.startswith(_CACHED):  # as the real one: a hub id is downloaded first
                model_size_or_path = modules["faster_whisper.utils"].download_model(model_size_or_path)
            built.append(model_size_or_path)

        def transcribe(self, path, **kwargs):
            return iter([_segment(0.0, 1.0, " hola")]), SimpleNamespace(language="es", duration=1.0)

    modules = _whisper_modules(WhisperModel, download_model)
    return modules, built


def _import_what_unload_imports() -> None:
    """Import torch, as FasterWhisperBackend.unload_model does, where it is installed."""
    try:
        import torch  # noqa: F401
    except Exception:  # not installed (CI), or unusable: unload_model skips it too
        pass


_FIRST_USES = {
    "translation": lambda backend: backend.translate(_wav_bytes(), _FIRST_USE)["text"],
    "load": lambda backend: backend.load_model(_FIRST_USE),
}
_DURING_THE_DOWNLOAD = {
    "transcription": lambda backend: backend.transcribe(_wav_bytes(), _LOADED_STT)["text"],
    "unload": lambda backend: backend.unload_model(_LOADED_STT),
}


@pytest.mark.parametrize("other", sorted(_DURING_THE_DOWNLOAD))
@pytest.mark.parametrize("first_use", sorted(_FIRST_USES))
def test_a_first_use_download_holds_no_decode_load_or_unload_of_another_model(monkeypatch, first_use, other):
    download_started, release_download = threading.Event(), threading.Event()
    modules, built = _blocking_download_modules(download_started, release_download)
    backend = FasterWhisperBackend()
    backend._models[_LOADED_STT] = _LoadedWhisper()
    backend._loaded_at[_LOADED_STT] = time.time()
    results: dict = {}
    # unload_model imports torch to release the CUDA cache, about 1.8 s the first time.
    # Import it before the 2 s window below, so the window times the lock, not the import; and install the fake
    # faster_whisper with monkeypatch, not patch.dict(sys.modules), which would drop torch again on exit and break
    # every later torch import in the process.
    if other == "unload":
        _import_what_unload_imports()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    first = threading.Thread(target=lambda: results.update(first=_FIRST_USES[first_use](backend)), daemon=True)
    during = threading.Thread(target=lambda: results.update(other=_DURING_THE_DOWNLOAD[other](backend)), daemon=True)
    first.start()
    try:
        assert download_started.wait(5), "the first use never started its download"
        during.start()
        during.join(timeout=2)
        finished_during_the_download = not during.is_alive()
    finally:
        release_download.set()
        first.join(timeout=10)
        if during.ident is not None:
            during.join(timeout=10)

    assert not first.is_alive() and not during.is_alive()
    assert finished_during_the_download, f"the {other} on {_LOADED_STT} waited for the download of {_FIRST_USE}"
    assert results["other"] == {"transcription": "hello", "unload": None}[other]
    assert results["first"] == {"translation": "hola", "load": None}[first_use]
    # The model is built once, from the downloaded files, and kept under its id.
    assert built == [_CACHED + _FIRST_USE]
    assert backend.is_model_loaded(_FIRST_USE)


def test_a_failed_first_use_download_raises_as_before_and_loads_nothing():
    # The download moved before the lock; its error still reaches the caller.
    def download_model(model_id, cache_dir=None, **kwargs):
        raise OSError("hub unreachable")

    class WhisperModel:
        def __init__(self, model_size_or_path, *args, **kwargs):
            download_model(model_size_or_path)  # as the real one does for a hub id

    backend = FasterWhisperBackend()
    with patch.dict(sys.modules, _whisper_modules(WhisperModel, download_model)):
        with pytest.raises(OSError, match="hub unreachable"):
            backend.translate(_wav_bytes(), _FIRST_USE)

    assert not backend.is_model_loaded(_FIRST_USE)
    assert backend._inference_lock.acquire(blocking=False)
    backend._inference_lock.release()


def test_a_fetch_before_the_load_is_not_repeated_by_the_load(tmp_path, monkeypatch):
    # A load or a decode now fetches before it takes a lock or a
    # decode thread, and its own fetch then finds those files rather than asking the hub again. A model unloaded
    # since, or whose files were deleted, asks the hub again.
    files = tmp_path / "snapshot"
    files.mkdir()
    fetched: list[str] = []

    def download_model(model_id, cache_dir=None, **kwargs):
        fetched.append(model_id)
        return str(files)

    class WhisperModel:
        def __init__(self, model_size_or_path, *args, **kwargs):
            assert model_size_or_path == str(files)

    backend = FasterWhisperBackend()
    # Not patch.dict: unload_model imports torch, which must stay imported after the test.
    for name, module in _whisper_modules(WhisperModel, download_model).items():
        monkeypatch.setitem(sys.modules, name, module)

    assert backend.start_fetch(_FIRST_USE).result(timeout=5) == str(files)
    backend.load_model(_FIRST_USE)
    assert fetched == [_FIRST_USE]

    backend.unload_model(_FIRST_USE)
    backend.load_model(_FIRST_USE)
    assert fetched == [_FIRST_USE] * 2

    backend.unload_model(_FIRST_USE)
    assert backend.start_fetch(_FIRST_USE).result(timeout=5) == str(files)
    backend.delete_cached_model(_FIRST_USE)
    backend.load_model(_FIRST_USE)
    assert fetched == [_FIRST_USE] * 4


def test_backend_subtitles_round_timestamps_to_nearest_millisecond():
    # whisper.utils.format_timestamp rounds; truncating renders 2.3 s as 00:00:02,299.
    segments = [_segment(1.0, 2.3, " Hello")]

    assert "00:00:01,000 --> 00:00:02,300" in FasterWhisperBackend._to_srt(segments)
    assert "00:00:01.000 --> 00:00:02.300" in FasterWhisperBackend._to_vtt(segments)


# The Hugging Face cache stores each file once in blobs/ and links it from every
# snapshots/<revision>/ (huggingface_hub "Understand caching"); summing every path counted a file once per link, so
# whisper tiny showed 149 MB and large-v3-turbo 4.6 GB. A hard link stands in for the symlink on a host that can't
# make one; both are another path to the same file.
_MIB = 1024 * 1024
_TINY = "Systran/faster-whisper-tiny"


def _hf_model(root, link, revisions=("r1",)):
    model = root / "models--Systran--faster-whisper-tiny"
    (model / "blobs").mkdir(parents=True)
    blob = model / "blobs" / "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9"
    blob.write_bytes(b"\0" * _MIB)
    for rev in revisions:
        snapshot = model / "snapshots" / rev
        snapshot.mkdir(parents=True)
        link(blob, snapshot / "model.bin")
    return model


def _cached_size(monkeypatch, cache_dir, *, custom_dir):
    from src.config import settings

    if custom_dir:
        monkeypatch.setattr(settings, "stt_model_dir", str(cache_dir))
    else:
        monkeypatch.setattr(settings, "stt_model_dir", "")
        monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
    return {row["model"]: row["size_mb"] for row in FasterWhisperBackend().list_cached_models()}.get(_TINY)


@pytest.mark.parametrize("revisions", [("r1",), ("r1", "r2")], ids=["one-snapshot", "two-snapshots"])
@pytest.mark.parametrize("custom_dir", [True, False], ids=["STT_MODEL_DIR", "hf-cache"])
def test_a_cached_model_counts_each_file_once(monkeypatch, tmp_path, custom_dir, revisions):
    _hf_model(tmp_path / "hub", os.link, revisions)
    (tmp_path / "hub" / "models--Systran--faster-whisper-tiny" / "refs").mkdir()
    (tmp_path / "hub" / "models--Systran--faster-whisper-tiny" / "refs" / "main").write_bytes(b"x" * 40)

    assert _cached_size(monkeypatch, tmp_path / "hub", custom_dir=custom_dir) == 1.0


def test_a_symlinked_snapshot_counts_its_blob_once(monkeypatch, tmp_path):
    probe = tmp_path / "probe"
    probe.write_bytes(b"x")
    try:
        os.symlink(probe, tmp_path / "probe-link")
    except (OSError, NotImplementedError):
        pytest.skip("this host can't create symbolic links without a privilege")

    _hf_model(tmp_path / "hub", lambda blob, path: os.symlink(os.path.relpath(blob, path.parent), path), ("r1", "r2"))

    assert _cached_size(monkeypatch, tmp_path / "hub", custom_dir=True) == 1.0
