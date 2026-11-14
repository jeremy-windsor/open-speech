"""STT_PRELOAD_MODELS loads in the background after the server has started (src/main.py lifespan).

The preload ran before the server answered, so a first GPU start of Parakeet (about 7 minutes of download and
conversion) served nothing and Docker's healthcheck could mark the container unhealthy meanwhile.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from src import router as router_module
from src.backends import parakeet as parakeet_module
from src.backends.parakeet import ParakeetBackend
from src.config import settings
from src.main import app
from tests.test_parakeet import (  # noqa: F401 - fixtures
    MODEL,
    _snapshot,
    _wav,
    cache,
    fake_onnx_asr,
)


@pytest.fixture
def slow_first_fetch(fake_onnx_asr, cache, monkeypatch):  # noqa: F811
    """Parakeet whose first download waits for .release; .downloads records each download."""
    state = type("State", (), {})()
    state.started, state.release, state.downloads = threading.Event(), threading.Event(), []
    state.loads = fake_onnx_asr.loads

    def download_files(self, model_id, device):
        state.downloads.append(model_id)
        state.started.set()
        state.release.wait(10)
        return _snapshot(cache, parakeet_module.INT8_FILES)

    class Backend(ParakeetBackend):
        def transcribe(self, audio, model, **kwargs):
            self._ensure_model(model)
            return {"text": "hello"}

    monkeypatch.setattr(ParakeetBackend, "_download_files", download_files)
    monkeypatch.setitem(router_module.router._backends, "parakeet", Backend(score_speech_fn=lambda samples: []))
    monkeypatch.setattr(settings, "stt_preload_models", f" {MODEL} ,")
    try:
        yield state
    finally:
        state.release.set()
        router_module.router._backends["parakeet"].unload_model(MODEL)


def test_the_server_answers_while_the_preload_downloads_and_a_request_joins_it(slow_first_fetch, caplog):
    state = slow_first_fetch
    caplog.set_level(logging.INFO, logger="open-speech")
    answers = []
    with TestClient(app) as client:
        assert state.started.wait(10), "the preload did not start"
        health = client.get("/health").status_code  # startup did not wait for the download
        request = threading.Thread(target=lambda: answers.append(client.post(
            "/v1/audio/transcriptions", files={"file": ("a.wav", _wav(1.0), "audio/wav")}, data={"model": MODEL},
        )))
        request.start()
        time.sleep(0.2)
        waiting = request.is_alive()  # the request waits for the preload's download
        state.release.set()
        request.join(15)
        deadline = time.monotonic() + 10
        while "Preloaded STT model" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)

    assert health == 200
    assert waiting
    assert [(answer.status_code, answer.json()) for answer in answers] == [(200, {"text": "hello"})]
    assert state.downloads == [MODEL], "the request started a second download"
    assert len(state.loads) == 1, "the request loaded the model a second time"
    assert f"Preloading STT model {MODEL} in the background" in caplog.text
    assert f"Preloaded STT model {MODEL} in" in caplog.text


def test_shutdown_does_not_wait_for_a_preload_still_downloading(slow_first_fetch):
    state = slow_first_fetch
    with TestClient(app):
        assert state.started.wait(10)
        stopping = time.monotonic()
    stopped_in = time.monotonic() - stopping
    state.release.set()

    assert stopped_in < 5, f"shutdown waited {stopped_in:.1f} s for the preload"


def test_a_failed_preload_is_logged_and_the_server_keeps_serving(fake_onnx_asr, cache, monkeypatch, caplog):  # noqa: F811
    def download_files(self, model_id, device):
        raise OSError("the hub could not be reached")

    monkeypatch.setattr(ParakeetBackend, "_download_files", download_files)
    monkeypatch.setitem(router_module.router._backends, "parakeet", ParakeetBackend(score_speech_fn=lambda s: []))
    monkeypatch.setattr(settings, "stt_preload_models", MODEL)
    caplog.set_level(logging.INFO, logger="open-speech")
    with TestClient(app) as client:
        deadline = time.monotonic() + 10
        while "Failed to preload STT model" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)
        health = client.get("/health").status_code

    assert health == 200
    assert f"Failed to preload STT model {MODEL}: the hub could not be reached" in caplog.text
