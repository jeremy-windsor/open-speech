"""Voice design through the core, against the Breeze TTS 2 worker with its server faked (tests/test_breeze_worker.py):
POST /api/voices/design, saving the sample as a recording, and the OpenAI-compatible endpoints with the voice made
from it, as the OpenAI SDK and Open WebUI call them."""

from __future__ import annotations

import io
import json
import sqlite3
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src import storage
from src.model_manager import ModelManager
from src.tts.router import TTSRouter
from src.voice_library import VoiceLibraryManager
from tests.test_breeze_worker import DESCRIPTION, MODEL, breeze, form_fields, speech  # noqa: F401 (fixture)
from tests.test_voice_model_workers import _bridge_to_worker

SAMPLE = "Hello! This is the voice you described."


class PlainBackend:
    """An in-process model that can't design a voice."""

    name = "plain"
    sample_rate = 24000
    capabilities = {"speed_control": True, "voice_design": False}

    def load_model(self, model_id): pass
    def unload_model(self, model_id): pass
    def is_model_loaded(self, model_id): return True
    def loaded_models(self): return []
    def list_voices(self): return []

    def synthesize(self, text, voice, speed=1.0, lang_code=None, **options):
        yield np.zeros(2400, dtype=np.float32)


@pytest.fixture
def core(breeze, tmp_path, monkeypatch):  # noqa: F811
    worker, worker_client, server, _started, _state = breeze
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    monkeypatch.setattr(storage, "_conn", connection)
    storage.init_db()
    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu", external_providers=json.dumps({"breeze": "http://breeze.voice-workers.internal.:8240"}))
    router.register_backend("plain", PlainBackend())
    router._backends["breeze"]._opener.open = _bridge_to_worker(worker_client)
    stt = MagicMock(_backends={})
    stt.loaded_models.return_value = []
    stt.list_cached_models.return_value = []
    manager = ModelManager(stt_router=stt, tts_router=router)
    server.response = lambda: speech(samples=np.full(4800, 8000, dtype="<i2"))
    with (
        patch.object(main_module, "tts_router", router),
        patch.object(main_module, "model_manager", manager),
        patch.object(main_module, "voice_library", VoiceLibraryManager(tmp_path / "voices")),
        patch.object(main_module.settings, "tts_cache_enabled", False),
        patch.object(main_module.settings, "tts_trim_silence", False),
        patch("src.tts.pipeline.encode_with_ffmpeg",
              side_effect=lambda audio, fmt, sample_rate=24000: f"{fmt}:{sample_rate}".encode()),
    ):
        yield TestClient(main_module.app), worker_client, server
    connection.close()


def load_breeze(client):
    response = client.post(f"/api/models/{MODEL}/load")
    assert response.status_code == 200, response.text


def sent(server) -> dict:
    request = server.requests[-1]
    return form_fields(request["body"], request["headers"]["Content-Type"])


def test_a_design_request_speaks_the_sample_in_the_described_voice_as_wav(core):
    client, _worker, server = core
    load_breeze(client)

    response = client.post("/api/voices/design", json={"model": MODEL, "description": f"  {DESCRIPTION}  ",
                                                         "text": SAMPLE})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(response.content)) as clip:
        assert (clip.getframerate(), clip.getnchannels(), clip.getnframes()) == (24000, 1, 4800)
    assert sent(server) == {"text": SAMPLE, "seed": "42", "instruction": DESCRIPTION, "cfg_scale": "3"}


def test_the_sample_text_has_a_default_and_both_fields_have_limits(core):
    client, _worker, server = core
    load_breeze(client)
    response = client.post("/api/voices/design", json={"model": MODEL, "description": DESCRIPTION})
    assert response.status_code == 200
    assert sent(server)["text"].startswith("Hello! This is the voice you described.")
    for body in ({"model": MODEL}, {"model": MODEL, "description": ""},
                 {"model": MODEL, "description": "x" * 1001}, {"model": MODEL, "description": "x", "text": "y" * 501},
                 {"model": MODEL, "description": "x", "unknown": 1}):
        assert client.post("/api/voices/design", json=body).status_code == 422, body
    assert client.post("/api/voices/design", json={"model": MODEL, "description": "   "}).status_code == 400


def test_a_model_that_cannot_design_a_voice_is_refused(core):
    client, _worker, server = core
    response = client.post("/api/voices/design", json={"model": "plain", "description": DESCRIPTION})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "voice_design is not supported by the plain backend."
    response = client.post("/api/voices/design", json={"model": "nobody/x", "description": DESCRIPTION})
    assert response.status_code == 400
    assert server.requests == []


def test_a_worker_that_is_not_loaded_or_refuses_answers_with_its_code(core):
    client, _worker, server = core
    response = client.post("/api/voices/design", json={"model": MODEL, "description": DESCRIPTION})
    assert response.status_code == 409 and response.json()["error"]["code"] == "model_not_loaded"
    load_breeze(client)
    response = client.post("/api/voices/design", json={"model": MODEL, "description": DESCRIPTION, "language": "fr"})
    assert response.status_code == 400
    assert "Unsupported language 'fr'" in json.dumps(response.json())


def designed_voice(client) -> str:
    """Design a sample, save it as a recording with its words, and make a named voice of it for Breeze."""
    sample = client.post("/api/voices/design", json={"model": MODEL, "description": DESCRIPTION, "text": SAMPLE})
    assert sample.status_code == 200
    saved = client.post("/api/voices/library", data={"name": "warm_narrator", "transcript": SAMPLE},
                        files={"audio": ("warm_narrator.wav", sample.content, "audio/wav")})
    assert saved.status_code == 201, saved.text
    identity = client.post("/api/voices/identities", json={"name": "Warm narrator"}).json()
    realization = client.post(f"/api/voices/identities/{identity['id']}/realizations",
                              json={"model": MODEL, "voice": "reference", "reference_audio_id": "warm_narrator"})
    assert realization.status_code == 201, realization.text
    return identity["voice"]


def test_a_designed_voice_is_a_recording_any_openai_client_can_name(core):
    client, _worker, server = core
    load_breeze(client)
    voice = designed_voice(client)

    # Open WebUI's voice list: {id: name} from /audio/voices.
    voices = client.get(f"/v1/audio/voices?model={MODEL}").json()["voices"]
    assert {v["id"]: v["name"] for v in voices} == {voice: "Warm narrator"}

    # openai-python's audio.speech.create(model=..., voice=..., input=..., response_format="wav").
    response = client.post("/v1/audio/speech", json={
        "model": MODEL, "voice": voice, "input": "Reading in the voice I designed.", "response_format": "wav",
    }, headers={"Authorization": "Bearer sk-anything", "User-Agent": "OpenAI/Python 1.99.0"})
    assert response.status_code == 200, response.text
    with wave.open(io.BytesIO(response.content)) as clip:
        assert clip.getframerate() == 24000
    form = sent(server)
    assert (form["text"], form["ref_text"], form["cfg_scale"]) == ("Reading in the voice I designed.", SAMPLE, "1")
    assert "instruction" not in form, "a cloned voice needs no description"
    with wave.open(io.BytesIO(form["ref_audio"])) as clip:
        assert clip.getnframes() == 4800, "the saved sample is the reference"

    # Open WebUI posts {input, model, voice} and plays the default format; a plain speed of 1.0 is fine.
    response = client.post("/v1/audio/speech", json={"input": "Open WebUI says hi.", "model": MODEL, "voice": voice})
    assert response.status_code == 200
    assert response.content == f"{main_module.settings.tts_default_format}:24000".encode()

    # The recording by name, as an extra field.
    response = client.post("/v1/audio/speech", json={"input": "By its recording.", "model": MODEL, "voice": "x",
                                                     "voice_library_ref": "warm_narrator", "response_format": "wav"})
    assert response.status_code == 200
    assert sent(server)["ref_text"] == SAMPLE


def test_a_voice_can_be_designed_on_the_fly_through_the_openai_endpoint(core):
    client, _worker, server = core
    load_breeze(client)
    response = client.post("/v1/audio/speech?stream=true", json={
        "model": MODEL, "voice": "designed", "input": "Hello (laugh) there.", "voice_design": DESCRIPTION,
        "response_format": "pcm",
    })
    assert response.status_code == 200, response.text
    assert len(response.content) == 4800 * 2
    assert sent(server) == {"text": "Hello (laugh) there.", "seed": "42", "instruction": DESCRIPTION, "cfg_scale": "3"}
    # A request with no voice at all says what it needs.
    response = client.post("/v1/audio/speech", json={"model": MODEL, "voice": "alloy", "input": "Hi."})
    assert response.status_code == 400 and response.json()["error"]["code"] == "voice_required"


def test_the_model_is_listed_for_openai_clients_with_its_name(core):
    client, _worker, _server = core
    load_breeze(client)
    rows = {row["id"]: row for row in client.get("/v1/audio/models").json()["models"]}
    assert rows[MODEL]["name"] == "Breeze TTS 2"
    assert (rows[MODEL]["status"], rows[MODEL]["backend"]) == ("loaded", "breeze")
    capabilities = client.get(f"/api/tts/capabilities?model={MODEL}").json()["capabilities"]
    assert capabilities["voice_design"] is True and capabilities["voice_clone"] is True


def test_a_cloning_model_lists_only_the_named_voices_it_has_a_version_for(core):
    client, _worker, _server = core
    load_breeze(client)
    designed_voice(client)
    other = client.post("/api/voices/identities", json={"name": "Kokoro only"}).json()
    client.post(f"/api/voices/identities/{other['id']}/realizations", json={"model": "plain", "voice": "v"})
    voices = client.get(f"/v1/audio/voices?model={MODEL}").json()["voices"]
    assert [v["name"] for v in voices] == ["Warm narrator"]
    assert voices[0]["model"] == MODEL
    # A model with no cloning keeps its own (here empty) list.
    assert client.get("/v1/audio/voices?model=plain").json()["voices"] == []
