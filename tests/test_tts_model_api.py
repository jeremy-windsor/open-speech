"""Tests for TTS model management endpoints."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src.tts.backends.base import TTSLoadedModelInfo


@pytest.fixture
def tts_client():
    mock_router = MagicMock()
    mock_router.loaded_models.return_value = []
    mock_router.is_model_loaded.return_value = False
    mock_router.list_backends.return_value = ["kokoro"]
    # The listing's inventory (ModelManager.list_all) is empty: these tests see only what the router reports.
    with patch.object(main_module, "tts_router", mock_router), \
         patch.object(main_module.model_manager, "list_all", return_value=[]):
        yield TestClient(app), mock_router


class TestTTSModelLoad:
    def test_load_model(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/models/load", json={"model": "kokoro"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"
        mock.load_model.assert_called_once_with("kokoro")

    def test_load_default(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/models/load", json={})
        assert resp.status_code == 200
        mock.load_model.assert_called_once()

    def test_unknown_model_does_not_unload_working_model(self, tts_client):
        client, mock = tts_client
        mock.loaded_models.return_value = [
            TTSLoadedModelInfo(
                model="kokoro", backend="kokoro", device="cpu", loaded_at=1000.0,
            )
        ]
        mock.get_backend.side_effect = ValueError("Unknown TTS model or backend: missing")

        resp = client.post("/v1/audio/models/load", json={"model": "missing"})

        assert resp.status_code == 400
        mock.unload_model.assert_not_called()
        mock.load_model.assert_not_called()


class TestTTSModelUnload:
    def test_unload_model(self, tts_client):
        client, mock = tts_client
        mock.is_model_loaded.return_value = True
        resp = client.post("/v1/audio/models/unload", json={"model": "kokoro"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

    def test_unload_not_loaded(self, tts_client):
        client, mock = tts_client
        mock.is_model_loaded.return_value = False
        resp = client.post("/v1/audio/models/unload", json={"model": "kokoro"})
        assert resp.status_code == 404


@pytest.mark.parametrize("action", ["load", "unload"])
def test_request_without_model_uses_configured_tts_model(tts_client, action):
    client, mock = tts_client
    mock.is_model_loaded.return_value = True

    with patch.object(main_module.settings, "tts_model", "piper/en_US-lessac-medium"):
        resp = client.post(f"/v1/audio/models/{action}", json={})

    assert resp.status_code == 200
    called = mock.load_model if action == "load" else mock.unload_model
    called.assert_called_once_with("piper/en_US-lessac-medium")


class TestTTSModelList:
    def test_list_empty(self, tts_client):
        client, mock = tts_client
        resp = client.get("/v1/audio/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        # Default model should appear as not_loaded
        assert any(m["status"] == "not_loaded" for m in data["models"])

    def test_list_with_loaded(self, tts_client):
        client, mock = tts_client
        mock.loaded_models.return_value = [
            TTSLoadedModelInfo(
                model="kokoro", backend="kokoro",
                device="cpu", loaded_at=1000.0,
            )
        ]
        resp = client.get("/v1/audio/models")
        data = resp.json()
        loaded = [m for m in data["models"] if m["status"] == "loaded"]
        assert len(loaded) == 1
        assert loaded[0]["model"] == "kokoro"

    def test_unloaded_default_reports_its_backend(self, tts_client):
        client, mock = tts_client
        mock.get_backend.return_value = SimpleNamespace(name="piper")

        with patch.object(main_module.settings, "tts_model", "piper/en_US-lessac-medium"):
            resp = client.get("/v1/audio/models")

        [row] = resp.json()["models"]
        assert {k: row[k] for k in ("id", "model", "backend", "status")} == {
            "id": "piper/en_US-lessac-medium", "model": "piper/en_US-lessac-medium", "backend": "piper",
            "status": "not_loaded",
        }

    def test_unresolvable_default_reports_its_model_prefix(self, tts_client):
        client, mock = tts_client
        mock.get_backend.side_effect = ValueError("Unknown TTS model or backend")

        with patch.object(main_module.settings, "tts_model", "piper/en_US-missing"):
            resp = client.get("/v1/audio/models")

        assert resp.status_code == 200
        assert resp.json()["models"][0]["backend"] == "piper"


class _OverlappingLoadTTSRouter:
    """Two providers. A load waits (bounded) until a second request has listed loaded models.

    As with isolated workers, whose listing asks /health without waiting for another worker's
    long load, a load allowed to overlap another finds nothing to evict.
    """

    def __init__(self):
        self._backends = {"kokoro": object(), "pocket-tts": object()}
        self._state = threading.Lock()
        self.loaded: set[str] = set()
        self.listings = 0
        self.second_listing = threading.Event()

    def get_backend(self, model_id):
        if model_id not in self._backends:
            raise ValueError(f"Unknown TTS model or backend: {model_id}")
        return self._backends[model_id]

    def get_capabilities(self, model_id):
        self.get_backend(model_id)
        return {}

    def provider_is_available(self, provider):
        return provider in self._backends

    def loaded_models(self):
        with self._state:
            self.listings += 1
            if self.listings >= 2:
                self.second_listing.set()
            ids = sorted(self.loaded)
        return [TTSLoadedModelInfo(model=m, backend=m, device="cpu", loaded_at=time.time()) for m in ids]

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


_LOAD_REQUESTS = {
    "unified": lambda model: ("POST", f"/api/models/{model}/load", None),
    "tts_route": lambda model: ("POST", "/v1/audio/models/load", {"model": model}),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), [("unified", "tts_route"), ("tts_route", "tts_route")])
async def test_concurrent_tts_loads_leave_one_tts_model_loaded(first, second):
    # README: "Loading a TTS model always unloads the previously loaded one first, so only one TTS
    # model is loaded at a time." Both load routes take ModelManager's TTS operation lock.
    from src.model_manager import ModelManager
    from src.services.models import ModelProgressService

    router = _OverlappingLoadTTSRouter()
    stt = MagicMock(_backends={})
    stt.loaded_models.return_value = []
    requests = [_LOAD_REQUESTS[first]("kokoro"), _LOAD_REQUESTS[second]("pocket-tts")]

    with patch.object(main_module, "tts_router", router), \
         patch.object(main_module, "model_manager", ModelManager(stt_router=stt, tts_router=router)), \
         patch.object(main_module, "progress_service", ModelProgressService()):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.wait_for(
                asyncio.gather(*(client.request(m, url, json=body) for m, url, body in requests)),
                timeout=20,
            )

    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert len(router.loaded) == 1, f"two TTS models stayed loaded: {sorted(router.loaded)}"


def test_tts_switch_does_not_wait_behind_an_stt_unload_that_waits_for_a_decode(monkeypatch):
    # An STT unload waits for the decode that holds faster-whisper's inference lock, under the
    # lock that serializes STT model operations. A TTS switch needs no STT model, so it must not
    # wait for that transcription to end too.
    from src.model_manager import ModelManager
    from src.router import BackendRouter

    decoding, finish, unloading = threading.Event(), threading.Event(), threading.Event()

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            decoding.set()
            finish.wait(timeout=5)
            return [], SimpleNamespace(language="en", duration=1.0)

    whisper_module = ModuleType("faster_whisper")
    whisper_module.WhisperModel = FakeWhisperModel
    # A first load fetches with faster_whisper.utils before it builds the model;
    # the files are cached already.
    whisper_utils = ModuleType("faster_whisper.utils")
    whisper_utils.download_model = lambda model_id, cache_dir=None, **kwargs: model_id
    stt = BackendRouter()
    backend = stt.get_backend("any")
    unload_model = backend.unload_model

    def announced_unload(model_id):
        unloading.set()
        unload_model(model_id)

    tts = MagicMock()
    tts.loaded_models.return_value = []
    manager = ModelManager(stt_router=stt, tts_router=tts)
    response = {}

    def switch_tts():
        response["status"] = TestClient(app).post("/v1/audio/models/load", json={"model": "kokoro"}).status_code

    # Not patch.dict: the unload imports torch, which must stay imported after the test.
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper_module)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", whisper_utils)
    with patch.object(backend, "unload_model", announced_unload), \
         patch.object(main_module, "tts_router", tts), \
         patch.object(main_module, "model_manager", manager):
        stt.load_model("Systran/faster-whisper-small")
        stt.load_model("Systran/faster-whisper-large-v3")
        threads = [
            threading.Thread(target=stt.transcribe, args=(b"RIFF" + b"\x00" * 64, "Systran/faster-whisper-large-v3")),
            threading.Thread(target=manager.unload, args=("Systran/faster-whisper-small",)),
        ]
        threads[0].start()
        assert decoding.wait(timeout=5)
        threads[1].start()
        assert unloading.wait(timeout=5)
        switch = threading.Thread(target=switch_tts)
        switch.start()
        switch.join(timeout=3)
        switched_during_decode = not switch.is_alive()
        finish.set()
        for thread in (*threads, switch):
            thread.join(timeout=5)

    assert switched_during_decode, "the TTS load waited for an STT transcription"
    assert response == {"status": 200}
    tts.load_model.assert_called_once_with("kokoro")
