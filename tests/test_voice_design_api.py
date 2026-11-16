"""Voice design through the core, against the Breeze TTS 2 worker with its server faked (tests/test_breeze_worker.py):
POST /api/voices/design, saving the sample as a designed voice, and the OpenAI-compatible endpoints with the voice
made from it, as the OpenAI SDK and Open WebUI call them."""

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
from src.services import voice_design
from src.tts.router import TTSRouter
from src.voice_library import VoiceLibraryManager
from tests.test_breeze_worker import (  # noqa: F401 (fixture)
    DESCRIPTION,
    MODEL,
    breeze,
    form_fields,
    speech,
)
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
    monkeypatch.setattr(voice_design, "random_seed", lambda: 777)
    monkeypatch.setattr(worker, "_random_seed", lambda: 777)
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
    assert sent(server) == {"text": SAMPLE, "seed": "777", "instruction": DESCRIPTION, "cfg_scale": "3"}
    assert response.headers["x-voice-seed"] == "777"


def test_generate_again_gives_a_new_voice_and_a_seed_gives_one_back(core, monkeypatch):
    client, _worker, server = core
    load_breeze(client)
    seeds = iter([1234, 98765])
    monkeypatch.setattr(voice_design, "random_seed", lambda: next(seeds))
    body = {"model": MODEL, "description": DESCRIPTION, "text": SAMPLE}
    for seed in ("1234", "98765"):
        response = client.post("/api/voices/design", json=body)
        assert response.status_code == 200
        assert sent(server)["seed"] == seed == response.headers["x-voice-seed"], "a new seed for each sample"
    response = client.post("/api/voices/design", json={**body, "seed": 123})
    assert response.status_code == 200
    assert sent(server)["seed"] == "123" == response.headers["x-voice-seed"]
    assert client.post("/api/voices/design", json={**body, "seed": -1}).status_code == 422


def test_a_design_request_sends_the_options_the_model_declares(core):
    client, _worker, server = core
    load_breeze(client)
    body = {"model": MODEL, "description": DESCRIPTION, "text": SAMPLE}
    response = client.post("/api/voices/design", json={**body, "options": {"cfg_scale": 2, "temperature": 0.7,
                                                                            "steps": 4}})
    assert response.status_code == 200, response.text
    assert sent(server) == {"text": SAMPLE, "seed": "777", "instruction": DESCRIPTION, "cfg_scale": "2",
                            "temperature": "0.7"}, "steps is not a Breeze option, so it is left out"
    response = client.post("/api/voices/design", json={**body, "options": {"cfg_scale": 6}})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "cfg_scale must be from 1.0 to 5.0 for the breeze model."
    for options in ({"cfg_scale": "2"}, {"cfg_scale": True}, {"Bad Key": 1}, [1]):
        assert client.post("/api/voices/design", json={**body, "options": options}).status_code == 422, options


def test_speech_requests_carry_guidance_and_variation_to_breeze_and_others_ignore_them(core):
    client, _worker, server = core
    load_breeze(client)
    response = client.post("/v1/audio/speech", json={
        "model": MODEL, "voice": "designed", "input": "Hello.", "voice_design": DESCRIPTION, "response_format": "pcm",
        "cfg_scale": 2.5, "temperature": 1.1, "steps": 4,
    })
    assert response.status_code == 200, response.text
    form = sent(server)
    assert (form["cfg_scale"], form["temperature"]) == ("2.5", "1.1")
    assert "steps" not in form
    out_of_range = client.post("/v1/audio/speech", json={
        "model": MODEL, "voice": "designed", "input": "Hello.", "voice_design": DESCRIPTION, "temperature": 2,
    })
    assert out_of_range.status_code == 400
    assert "temperature must be from 0.1 to 1.3" in out_of_range.json()["error"]["message"]
    for bad in ({"cfg_scale": "2"}, {"temperature": True}):
        assert client.post("/v1/audio/speech", json={"model": MODEL, "input": "Hi.", **bad}).status_code == 422
    plain = client.post("/v1/audio/speech", json={"model": "plain", "voice": "v", "input": "Hi.",
                                                  "response_format": "wav", "cfg_scale": 2, "temperature": 0.5})
    assert plain.status_code == 200


def test_a_sample_longer_than_a_recording_to_clone_from_can_be_is_refused(core):
    client, _worker, server = core
    load_breeze(client)
    server.response = lambda: speech(samples=np.full(31 * 24000, 8000, dtype="<i2"))
    response = client.post("/api/voices/design", json={"model": MODEL, "description": DESCRIPTION, "text": SAMPLE})
    assert response.status_code == 400
    assert "at most 30" in response.json()["error"]["message"]
    assert "shorter sample text" in response.json()["error"]["message"]


def test_a_first_load_has_the_worker_download_its_weights_before_it_loads(core):
    client, _worker, _server = core
    backend = main_module.tts_router._backends["breeze"]
    paths = []
    forward = backend._opener.open

    def recording(request, timeout=None):
        if request.get_method() == "POST":
            paths.append(request.full_url.rsplit(":8240", 1)[1])
        return forward(request, timeout)

    backend._opener.open = recording
    load_breeze(client)
    assert paths == ["/v1/models/download", "/v1/models/load"]


def test_the_sample_text_has_a_default_and_both_fields_have_limits(core):
    client, _worker, server = core
    load_breeze(client)
    response = client.post("/api/voices/design", json={"model": MODEL, "description": DESCRIPTION})
    assert response.status_code == 200
    assert sent(server)["text"].startswith("Hello! This is the voice you described.")
    for body in ({"model": MODEL}, {"model": MODEL, "description": ""},
                 {"model": MODEL, "description": "x" * 1001}, {"model": MODEL, "description": "x", "text": "y" * 301},
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
    """Design a sample and save it as a designed voice, which Breeze speaks by cloning the sample."""
    sample = client.post("/api/voices/design", json={"model": MODEL, "description": DESCRIPTION, "text": SAMPLE,
                                                     "options": {"temperature": 0.7}})
    assert sample.status_code == 200
    saved = client.post("/api/voices/designed", data={
        "name": "Warm narrator", "model": MODEL, "description": DESCRIPTION, "transcript": SAMPLE,
        "seed": sample.headers["x-voice-seed"], "options": json.dumps({"temperature": 0.7}),
    }, files={"audio": ("sample.wav", sample.content, "audio/wav")})
    assert saved.status_code == 201, saved.text
    identity = saved.json()
    sample_name = identity["design"]["sample"]
    assert identity["design"] == {"description": DESCRIPTION, "model": MODEL, "seed": 777,
                                  "options": {"temperature": 0.7}, "sample": sample_name, "transcript": SAMPLE,
                                  "duration_s": 0.2}
    assert [(r["model"], r["voice"], r["reference_audio_id"]) for r in identity["realizations"]] == [
        (MODEL, sample_name, sample_name)], "a design model that clones speaks the voice itself"
    assert client.get("/api/voices/library").json() == [], "a designed sample is not a recording"
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

    # Its sample by name, as an extra field.
    sample = client.get("/api/voices/identities").json()["voices"][0]["design"]["sample"]
    response = client.post("/v1/audio/speech", json={"input": "By its sample.", "model": MODEL, "voice": "x",
                                                     "voice_library_ref": sample, "response_format": "wav"})
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
    assert sent(server) == {"text": "Hello (laugh) there.", "seed": "777", "instruction": DESCRIPTION, "cfg_scale": "3"}
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
    assert voices[0]["kind"] == "named_voice", "the web app keeps these out of a model's built-in voices"
    # A model with no cloning keeps its own (here empty) list.
    assert client.get("/v1/audio/voices?model=plain").json()["voices"] == []
