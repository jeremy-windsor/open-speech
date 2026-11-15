"""Designed voices: POST /api/voices/designed and PUT /api/voices/designed/{id}, a named voice's `design`, and its
hidden sample in the voice library, with a design model that does not clone (Qwen3 VoiceDesign's shape) and the
cloning models of its provider."""

from __future__ import annotations

import io
import json
import sqlite3
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main, storage
from src.voice_library import VoiceLibraryManager

DESIGN = "qwen3/1.7b-voice-design"
BASE = "qwen3/1.7b-base"
SMALL_BASE = "qwen3/0.6b-base"
DESCRIPTION = "a bright, quick young woman with a smile in her voice"
TRANSCRIPT = "Hello! This is the voice you described."

CAPABILITIES = {
    DESIGN: {"voice_design": True, "voice_clone": False, "options": {
        "temperature": {"type": "number", "label": "Variation", "min": 0.1, "max": 1.3, "step": 0.05, "default": 0.9},
    }},
    BASE: {"voice_clone": True, "clone_transcript_required": True},
    SMALL_BASE: {"voice_clone": True, "clone_transcript_required": True},
    "kokoro": {"speed_control": True},
}


class Router:
    def get_backend(self, model):
        if model not in CAPABILITIES:
            raise ValueError(f"Unknown model {model}")
        return SimpleNamespace(name=model.split("/")[0], capabilities=CAPABILITIES[model])

    def get_capabilities(self, model):
        return self.get_backend(model).capabilities

    def validate_voice(self, model, voice):
        self.get_backend(model)
        if model == "kokoro" and voice != "af_heart":
            raise ValueError(f"Unknown Kokoro voice '{voice}'")

    def list_voices(self, model):
        self.get_backend(model)
        return ["af_heart"] if model == "kokoro" else []

    def sample_rate_for(self, model):
        return 24000


class Manager:
    """ModelManager.list_all(): the 1.7B base model is downloaded, the 0.6B one is not."""

    def list_all(self):
        def row(model_id, state):
            return SimpleNamespace(id=model_id, type="tts", provider=model_id.split("/")[0], state=state)

        return [row(DESIGN, "loaded"), row(BASE, "downloaded"), row(SMALL_BASE, "provider_installed"),
                row("kokoro", "loaded"), SimpleNamespace(id="qwen3/asr", type="stt", provider="qwen3", state="loaded")]


def wav(seconds: float = 1.0, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(np.full(int(seconds * rate), 1000, dtype="<i2").tobytes())
    return buffer.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    monkeypatch.setattr(storage, "_conn", connection)
    storage.init_db()
    library = VoiceLibraryManager(tmp_path / "voices")
    monkeypatch.setattr(main, "tts_router", Router())
    monkeypatch.setattr(main, "model_manager", Manager())
    monkeypatch.setattr(main, "voice_library", library)
    yield TestClient(main.app), library
    connection.close()


def create(client, *, name="Bright", audio=None, speak_with=None, **fields):
    data = {"name": name, "model": DESIGN, "description": DESCRIPTION, "transcript": TRANSCRIPT, "seed": "4242",
            "options": json.dumps({"temperature": 0.6}), **fields}
    if speak_with is not None:
        data["speak_with"] = speak_with
    return client.post("/api/voices/designed", data=data,
                       files={"audio": ("sample.wav", audio if audio is not None else wav(), "audio/wav")})


def leftovers(client, library) -> tuple[list, list]:
    """Named voices and library files: what a failed request must not leave behind."""
    return client.get("/api/voices/identities?check_availability=false").json()["voices"], sorted(
        path.name for path in library.library_path.iterdir())


def test_a_designed_voice_is_a_named_voice_with_its_design_and_a_hidden_sample(env):
    client, library = env
    response = create(client)

    assert response.status_code == 201, response.text
    voice = response.json()
    sample = "designed_" + voice["id"].replace("-", "_")
    assert voice["name"] == "Bright" and voice["voice"] == f"voice:{voice['id']}"
    assert voice["design"] == {"description": DESCRIPTION, "model": DESIGN, "seed": 4242,
                               "options": {"temperature": 0.6}, "sample": sample, "transcript": TRANSCRIPT,
                               "duration_s": 1.0}
    assert [(r["model"], r["voice"], r["reference_audio_id"]) for r in voice["realizations"]] == [
        (BASE, sample, sample)], "the provider's installed cloning model, not the 0.6B one that isn't downloaded"

    assert client.get("/api/voices/library").json() == [], "Recordings leave it out"
    metadata = client.get(f"/api/voices/library/{sample}").json()
    assert metadata["designed"]["identity_id"] == voice["id"]
    assert {k: metadata["designed"][k] for k in ("description", "model", "seed", "options")} == {
        "description": DESCRIPTION, "model": DESIGN, "seed": 4242, "options": {"temperature": 0.6}}
    assert client.get(f"/api/voices/library/{sample}/audio").content == wav(), "its card plays the sample"

    listed = client.get("/api/voices/identities").json()["voices"]
    assert listed[0]["design"] == voice["design"] and listed[0]["available"] is True
    unchecked = client.get("/api/voices/identities?check_availability=false").json()["voices"]
    assert unchecked[0]["design"] == voice["design"]
    renamed = client.patch(f"/api/voices/identities/{voice['id']}", json={"name": "Brighter"}).json()
    assert renamed["design"] == voice["design"]
    plain = client.post("/api/voices/identities", json={"name": "Plain"}).json()
    assert plain["design"] is None


def test_speak_with_names_the_versions_repeated_or_as_a_comma_list(env):
    client, _library = env
    response = create(client, speak_with=[f"{BASE}, {SMALL_BASE}", BASE])
    assert response.status_code == 201, response.text
    assert [r["model"] for r in response.json()["realizations"]] == [SMALL_BASE, BASE]
    response = create(client, name="None", speak_with="")
    assert response.status_code == 201, response.text
    assert response.json()["realizations"] == [], "an empty speak_with makes a voice with no versions yet"


def test_seed_and_options_are_optional_and_only_declared_options_are_kept(env):
    client, _library = env
    response = create(client, seed="", options=json.dumps({"temperature": 0.75, "steps": 4}))
    assert response.status_code == 201, response.text
    assert (response.json()["design"]["seed"], response.json()["design"]["options"]) == (None, {"temperature": 0.75})
    data = {"name": "Bare", "model": DESIGN, "description": DESCRIPTION, "transcript": TRANSCRIPT}
    response = client.post("/api/voices/designed", data=data, files={"audio": ("s.wav", wav(), "audio/wav")})
    assert response.status_code == 201, response.text
    assert (response.json()["design"]["seed"], response.json()["design"]["options"]) == (None, {})


@pytest.mark.parametrize(
    ("fields", "status", "says"),
    [
        pytest.param({"name": "Taken"}, 409, "already exists", id="name-in-use"),
        pytest.param({"name": "taken"}, 409, "already exists", id="name-in-use-any-case"),
        pytest.param({"audio": wav(30.5)}, 422, "at most 30 s", id="over-30-seconds"),
        pytest.param({"audio": b"not a wav"}, 422, "WAV", id="not-a-wav"),
        pytest.param({"model": "kokoro"}, 400, "voice_design is not supported", id="model-cannot-design"),
        pytest.param({"model": "nobody/x"}, 400, "Unknown model", id="unknown-model"),
        pytest.param({"speak_with": "kokoro"}, 400, "kokoro:", id="speak-with-a-model-that-cannot-clone"),
        pytest.param({"speak_with": "nobody/x"}, 400, "nobody/x:", id="speak-with-an-unknown-model"),
        pytest.param({"options": json.dumps({"temperature": 2})}, 400, "temperature must be from", id="option-range"),
        pytest.param({"options": "{not json"}, 422, "options", id="options-not-json"),
        pytest.param({"options": json.dumps({"temperature": "0.5"})}, 422, "options", id="option-not-a-number"),
        pytest.param({"options": json.dumps([0.5])}, 422, "options", id="options-not-an-object"),
        pytest.param({"seed": "-1"}, 422, None, id="seed-below-0"),
        pytest.param({"description": "   "}, 422, "description is required", id="no-description"),
        pytest.param({"transcript": ""}, 422, "transcript", id="no-transcript"),
        pytest.param({"name": " "}, 422, "name is required", id="no-name"),
    ],
)
def test_a_refused_request_leaves_nothing_behind(env, fields, status, says):
    client, library = env
    client.post("/api/voices/identities", json={"name": "Taken"})
    before = leftovers(client, library)
    response = create(client, **fields)
    assert response.status_code == status, response.text
    if says:
        assert says in json.dumps(response.json())
    assert leftovers(client, library) == before


def test_a_sample_that_cannot_be_stored_leaves_no_named_voice(env, monkeypatch):
    client, library = env

    def full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(library, "save", full)
    with pytest.raises(OSError):
        create(client)
    assert leftovers(client, library) == ([], [])


def test_a_failure_after_the_sample_is_stored_removes_it(env, monkeypatch):
    client, library = env
    real_save = library.save

    def save_then_fail(*args, **kwargs):
        real_save(*args, **kwargs)
        raise OSError("lost the disk after writing")

    monkeypatch.setattr(library, "save", save_then_fail)
    with pytest.raises(OSError):
        create(client)
    assert leftovers(client, library) == ([], [])


def test_replacing_a_designed_voice_keeps_its_versions_on_the_same_sample(env):
    client, library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    new_audio = wav(2.0)

    response = client.put(f"/api/voices/designed/{voice['id']}", data={
        "model": DESIGN, "description": "an older, slower voice", "transcript": "New words for the sample.",
        "seed": "7", "options": json.dumps({"temperature": 1.2}),
    }, files={"audio": ("sample.wav", new_audio, "audio/wav")})

    assert response.status_code == 200, response.text
    replaced = response.json()
    assert replaced["design"] == {"description": "an older, slower voice", "model": DESIGN, "seed": 7,
                                  "options": {"temperature": 1.2}, "sample": sample,
                                  "transcript": "New words for the sample.", "duration_s": 2.0}
    assert replaced["realizations"] == voice["realizations"]
    assert client.get(f"/api/voices/library/{sample}/audio").content == new_audio
    assert client.get("/api/voices/library").json() == []

    put = {"model": DESIGN, "description": DESCRIPTION, "transcript": TRANSCRIPT}
    audio = {"audio": ("sample.wav", wav(), "audio/wav")}
    plain = client.post("/api/voices/identities", json={"name": "Plain"}).json()
    response = client.put(f"/api/voices/designed/{plain['id']}", data=put, files=audio)
    assert response.status_code == 404 and "not a designed voice" in response.json()["error"]["message"]
    assert client.put("/api/voices/designed/no-such-voice", data=put, files=audio).status_code == 404
    too_long = client.put(f"/api/voices/designed/{voice['id']}", data=put,
                          files={"audio": ("sample.wav", wav(31), "audio/wav")})
    assert too_long.status_code == 422
    assert client.get(f"/api/voices/library/{sample}/audio").content == new_audio, "a refused replace changes nothing"
    assert client.put(f"/api/voices/designed/{voice['id']}", data={**put, "name": "Renamed"},
                      files=audio).json()["name"] == "Bright", "a replace keeps the name"


def test_the_sample_belongs_to_its_voice_and_goes_with_it(env):
    client, library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    upload = client.post("/api/voices/library", data={"name": sample},
                         files={"audio": ("x.wav", wav(), "audio/wav")})
    assert upload.status_code == 409, "a recording can't overwrite it"
    assert client.patch(f"/api/voices/library/{sample}", json={"transcript": "x"}).status_code == 409
    assert client.delete(f"/api/voices/library/{sample}").status_code == 409, "it can't be deleted alone"
    assert library.metadata(sample)["transcript"] == TRANSCRIPT

    assert client.delete(f"/api/voices/identities/{voice['id']}").status_code == 204
    assert leftovers(client, library) == ([], [])
    assert client.get(f"/api/voices/library/{sample}").status_code == 404


def test_a_named_voice_that_is_not_designed_deletes_as_before(env):
    client, library = env
    library.save("own_recording", wav(), transcript="My own words.")
    voice = client.post("/api/voices/identities", json={"name": "Mine"}).json()
    client.post(f"/api/voices/identities/{voice['id']}/realizations",
                json={"model": BASE, "voice": "own_recording", "reference_audio_id": "own_recording"})
    assert client.delete(f"/api/voices/identities/{voice['id']}").status_code == 204
    assert [item["name"] for item in client.get("/api/voices/library").json()] == ["own_recording"]
