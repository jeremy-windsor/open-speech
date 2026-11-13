"""TTS_ENABLED=false turns off every TTS path, not only the /v1/audio/speech family.

README: `TTS_ENABLED | true | Enable TTS endpoints`. A silent Kokoro on the app's own router
records every load and every synthesis, so each test checks that nothing was loaded or spoken.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _silent_kokoro(spoken: list, loads: list):
    from src.tts.backends import kokoro as kokoro_module

    class SilentKokoro(kokoro_module.KokoroBackend):
        def download_model(self, model_id):
            loads.append(f"download {model_id}")  # never the Hugging Face hub

        def load_model(self, model_id):
            loads.append(model_id)
            with self._state_lock:
                self._pipeline = object()
                self._pipelines = {"a": self._pipeline}
                self._model_id = model_id
                self._loaded_at = time.time()

        def unload_model(self, model_id):
            with self._state_lock:
                self._pipeline = None
                self._pipelines = {}
                self._model_id = None
                self._loaded_at = None

        def synthesize(self, text, voice, speed=1.0, lang_code=None, **_kwargs):
            spoken.append(text)
            yield np.zeros(2400, dtype=np.float32)

    return SilentKokoro(device="cpu")


@pytest.fixture
def tts_off(monkeypatch, tmp_path):
    from src import main as main_module
    from src import storage as storage_module
    from src.tts.backends import kokoro as kokoro_module

    monkeypatch.setattr(kokoro_module, "_discover_voices_from_package", lambda: None)
    spoken: list[str] = []
    loads: list[str] = []
    backend = _silent_kokoro(spoken, loads)
    router = main_module.tts_router  # model_manager holds this same instance
    monkeypatch.setattr(router, "_backends", {"kokoro": backend})
    monkeypatch.setattr(router, "_default_backend", backend)
    monkeypatch.setattr(router, "_synthesis_locks", {"kokoro": threading.RLock()})

    settings = main_module.settings
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "tts_cache_enabled", False)
    monkeypatch.setattr(settings, "tts_model", "kokoro")
    monkeypatch.setattr(settings, "tts_voice", "af_heart")
    monkeypatch.setattr(settings, "os_studio_db_path", str(tmp_path / "studio.db"))
    monkeypatch.setattr(settings, "os_conversations_dir", str(tmp_path / "conversations"))
    previous_conn = storage_module._conn
    storage_module._conn = None
    storage_module.init_db()
    try:
        yield SimpleNamespace(main=main_module, spoken=spoken, loads=loads)
    finally:
        if storage_module._conn is not None:
            storage_module._conn.close()
        storage_module._conn = previous_conn


def test_speech_and_the_tts_load_route_answer_tts_disabled(tts_off):
    client = TestClient(tts_off.main.app)
    speech = client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "Hello", "voice": "af_heart", "response_format": "wav"},
    )
    load = client.post("/v1/audio/models/load", json={"model": "kokoro"})

    assert (speech.status_code, load.status_code) == (404, 404)
    assert tts_off.spoken == [] and tts_off.loads == []


def test_conversation_render_is_refused(tts_off):
    client = TestClient(tts_off.main.app)
    created = client.post(
        "/api/conversations",
        json={"name": "x", "turns": [{"speaker": "A", "text": "Spoken anyway", "profile_id": None, "effects": None}]},
    )
    assert created.status_code == 201, created.text

    render = client.post(
        f"/api/conversations/{created.json()['id']}/render", json={"format": "wav", "sample_rate": 24000}
    )

    assert render.status_code == 404, render.text
    assert render.json()["error"]["message"] == "TTS is disabled"
    assert tts_off.spoken == []


def test_studio_synthesis_refuses_whatever_route_calls_it(tts_off):
    from src.services.tts import TTSDisabledError

    with pytest.raises(TTSDisabledError):
        tts_off.main._synthesize_array(text="Spoken anyway", model="kokoro", voice="af_heart", speed=1.0)
    assert tts_off.spoken == []


def test_reader_render_is_refused_before_any_part_is_queued(tts_off, monkeypatch, tmp_path):
    from src.reader.renderer import ReaderRenderer

    main = tts_off.main
    renderer = ReaderRenderer(
        store=main.reader_store,
        synthesize_fn=main._synthesize_array,  # the app's real synthesis path
        encode_fn=lambda audio, sample_rate: b"ID3" + bytes(64),
        pronounce_fn=lambda text: text,
        busy_fn=lambda: False,
        signer=main.reader_signer,
        cache_dir=str(tmp_path / "reader"),
    )
    monkeypatch.setattr(main, "reader_renderer", renderer)
    try:
        client = TestClient(main.app)
        imported = client.post("/v1/reader/documents", json={"type": "text", "text": "Read aloud anyway."})
        assert imported.status_code in (200, 201), imported.text

        started = client.post(
            f"/v1/reader/documents/{imported.json()['id']}/render",
            json={"from_block": 0, "model": "kokoro", "voice": "af_heart"},
        )
    finally:
        renderer.stop()

    assert started.status_code == 404, started.text
    assert started.json()["error"]["message"] == "TTS is disabled"
    assert tts_off.spoken == []


@pytest.mark.parametrize("operation", ["load", "download"])
def test_unified_model_routes_refuse_tts_models(tts_off, operation):
    client = TestClient(tts_off.main.app)

    resp = client.post(f"/api/models/kokoro/{operation}")

    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "tts_disabled"
    assert tts_off.loads == [] and not tts_off.main.tts_router.is_model_loaded("kokoro")


@pytest.mark.parametrize("read", ["status", "progress"])
def test_unified_model_reads_refuse_tts_models(tts_off, monkeypatch, tmp_path, read):
    """/api/models leaves Kokoro out and its Load answers 404 tts_disabled, yet
    /api/models/kokoro/status answered 200 downloaded (provider_available true) and /progress 200 downloaded, so
    Speak and the Reader showed a "Loading" job before the load's refusal. Both answer as the load does now."""
    hf_root = tmp_path / "hf-hub"
    (hf_root / "models--hexgrad--Kokoro-82M").mkdir(parents=True)  # Kokoro's weights "on disk"
    monkeypatch.setattr(tts_off.main.settings, "stt_model_dir", str(hf_root))
    monkeypatch.delitem(tts_off.main.progress_service.download_progress, "kokoro", raising=False)
    client = TestClient(tts_off.main.app)

    resp = client.get(f"/api/models/kokoro/{read}")

    assert resp.status_code == 404, f"TTS_ENABLED=false, yet GET /api/models/kokoro/{read} answers {resp.text}"
    assert resp.json()["error"] == {"message": "TTS is disabled", "code": "tts_disabled"}
    assert tts_off.loads == []
    # Only TTS models: an STT model is read as before, and an unknown model is still unknown_model.
    stt = client.get(f"/api/models/{tts_off.main.settings.stt_model}/{read}")
    assert stt.status_code == 200, stt.text
    unknown = client.get(f"/api/models/no-such-model/{read}")
    assert (unknown.status_code, unknown.json()["error"]["code"]) == (404, "unknown_model"), unknown.text


def test_the_model_list_offers_no_tts_model(tts_off):
    """/api/models listed Kokoro as downloaded and usable, though its Load
    answers 404 tts_disabled. /v1/models already left it out."""
    client = TestClient(tts_off.main.app)

    resp = client.get("/api/models")

    assert resp.status_code == 200, resp.text
    models = resp.json()["models"]
    # The web UI's rule for a usable TTS model (ttsModels() in src/static/js/core/models.js)
    offered = {
        m["id"]: (m.get("state"), m.get("provider_available")) for m in models
        if m.get("type") == "tts" and m.get("state") != "provider_missing" and m.get("provider_available") is not False
    }
    assert offered == {}, f"TTS_ENABLED=false, yet /api/models offers TTS models (state, provider_available): {offered}"
    assert any(m.get("type") == "stt" for m in models), "STT models are still listed"
    assert tts_off.loads == []


async def test_startup_does_not_preload_tts_models(tts_off, monkeypatch):
    main = tts_off.main
    monkeypatch.setattr(main.settings, "stt_preload_models", "")
    monkeypatch.setattr(main.settings, "tts_preload_models", "kokoro")
    monkeypatch.setattr(main.settings, "tts_preload_default", False)

    async def run():
        async with main.lifespan(main.app):
            pass

    await asyncio.wait_for(run(), timeout=60)

    assert tts_off.loads == []


async def test_realtime_reply_is_an_error_event(tts_off):
    from src.realtime.server import RealtimeSession

    class RecordingWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

        async def close(self, code=1000, reason=""):
            pass

    websocket = RecordingWebSocket()
    session = RealtimeSession(websocket, tts_off.main.tts_router, model="kokoro", ga=True)
    await asyncio.wait_for(
        session._handle_response_create({"response": {"instructions": "Hello from realtime"}}),
        timeout=30,
    )

    assert tts_off.spoken == []
    assert [event["type"] for event in websocket.sent] == ["error"]
    assert websocket.sent[0]["error"]["code"] == "tts_disabled"
