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
from src.profiles import ProfileManager
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

    response = client.delete(f"/api/voices/identities/{voice['id']}")
    assert (response.status_code, response.json()) == (200, {"moved_to": None})
    assert leftovers(client, library) == ([], [])
    assert client.get(f"/api/voices/library/{sample}").status_code == 404


def test_a_named_voice_that_is_not_designed_deletes_as_before(env):
    client, library = env
    library.save("own_recording", wav(), transcript="My own words.")
    voice = client.post("/api/voices/identities", json={"name": "Mine"}).json()
    client.post(f"/api/voices/identities/{voice['id']}/realizations",
                json={"model": BASE, "voice": "own_recording", "reference_audio_id": "own_recording"})
    response = client.delete(f"/api/voices/identities/{voice['id']}")
    assert (response.status_code, response.json()) == (200, {"moved_to": None})
    assert [item["name"] for item in client.get("/api/voices/library").json()] == ["own_recording"]
    assert client.delete(f"/api/voices/identities/{voice['id']}").status_code == 404


def preset_for(client, voice, name="Bright reading"):
    response = client.post("/api/profiles", json={"name": name, "backend": "qwen3", "model": BASE, "voice": "x",
                                                  "voice_identity_id": voice["id"]})
    assert response.status_code == 201, response.text
    return response.json()


def test_a_sample_a_preset_still_uses_becomes_a_recording_named_after_the_voice(env):
    client, library = env
    library.save("bright", wav(0.5), transcript="Someone else's recording.")
    voice = create(client).json()
    sample = voice["design"]["sample"]
    preset = preset_for(client, voice)
    assert preset["reference_audio_id"] == sample

    response = client.delete(f"/api/voices/identities/{voice['id']}")

    assert (response.status_code, response.json()) == (200, {"moved_to": "bright_2"})
    recordings = {item["name"]: item for item in client.get("/api/voices/library").json()}
    assert set(recordings) == {"bright", "bright_2"}
    assert recordings["bright_2"]["transcript"] == TRANSCRIPT and "designed" not in recordings["bright_2"]
    assert client.get("/api/voices/library/bright_2/audio").content == wav()
    assert client.get(f"/api/voices/library/{sample}").status_code == 404
    kept = client.get(f"/api/profiles/{preset['id']}").json()
    assert (kept["voice_identity_id"], kept["voice"], kept["reference_audio_id"]) == (None, "bright_2", "bright_2")
    assert client.delete("/api/voices/library/bright_2").status_code == 204, "it is an ordinary recording now"


def test_a_sample_another_voices_version_uses_becomes_a_recording_and_the_version_follows(env):
    client, library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    other = client.post("/api/voices/identities", json={"name": "Other"}).json()
    # A version written before other named voices' samples were refused.
    storage.get_db().execute(
        "INSERT INTO voice_realizations VALUES (?, ?, ?, ?, ?, ?)",
        ("r-old", other["id"], SMALL_BASE, sample, sample, "2026-11-01T00:00:00+00:00"),
    )
    storage.get_db().commit()

    response = client.delete(f"/api/voices/identities/{voice['id']}")

    assert response.json() == {"moved_to": "bright"}
    [version] = client.get("/api/voices/identities?check_availability=false").json()["voices"][0]["realizations"]
    assert (version["voice"], version["reference_audio_id"]) == ("bright", "bright")


def test_a_failed_move_leaves_the_voice_its_sample_and_its_presets_as_they_were(env, monkeypatch):
    client, library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    preset = preset_for(client, voice)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(library, "move", broken)
    with pytest.raises(OSError):
        client.delete(f"/api/voices/identities/{voice['id']}")
    listed = client.get("/api/voices/identities?check_availability=false").json()["voices"]
    assert [item["id"] for item in listed] == [voice["id"]] and listed[0]["design"] == voice["design"]
    assert client.get(f"/api/profiles/{preset['id']}").json()["voice_identity_id"] == voice["id"]
    assert client.get("/api/voices/library").json() == []
    assert library.metadata(sample)["designed"]["identity_id"] == voice["id"]


def test_a_commit_that_fails_after_the_move_moves_the_sample_back(env, monkeypatch):
    client, library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    preset_for(client, voice)

    def failing_commit(self):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(storage.StudioConnection, "commit", failing_commit)
    with pytest.raises(RuntimeError):
        client.delete(f"/api/voices/identities/{voice['id']}")
    assert library.metadata(sample)["designed"]["identity_id"] == voice["id"]
    assert sorted(path.name for path in library.library_path.iterdir()) == [
        f"{sample}.audio.wav", f"{sample}.meta.json"]


def test_a_version_can_clone_only_its_own_voices_designed_sample(env):
    client, library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    other = client.post("/api/voices/identities", json={"name": "Other"}).json()
    body = {"model": SMALL_BASE, "voice": sample, "reference_audio_id": sample}

    refused = client.post(f"/api/voices/identities/{other['id']}/realizations", json=body)
    assert refused.status_code == 400 and "another named voice's designed sample" in refused.json()["error"]["message"]
    upper = {**body, "reference_audio_id": sample.upper()}
    assert client.post(f"/api/voices/identities/{other['id']}/realizations", json=upper).status_code == 400
    library.save("own", wav(), transcript="Mine.")
    assert client.post(f"/api/voices/identities/{other['id']}/realizations",
                       json={"model": BASE, "voice": "own", "reference_audio_id": "own"}).status_code == 201
    refused = client.put(f"/api/voices/identities/{other['id']}/realizations/{BASE}",
                         json={**body, "model": BASE})
    assert refused.status_code == 400

    own = client.post(f"/api/voices/identities/{voice['id']}/realizations", json=body)
    assert own.status_code == 201, own.text
    assert own.json()["reference_audio_id"] == sample


def test_an_imported_preset_on_another_voices_sample_stays_a_standalone_preset(env):
    client, _library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    ProfileManager().import_presets([{"name": "Imported", "backend": "qwen3", "model": BASE, "voice": sample,
                                      "reference_audio_id": sample}])
    names = [item["name"] for item in client.get("/api/voices/identities?check_availability=false").json()["voices"]]
    assert names == ["Bright"]
    [imported] = [p for p in client.get("/api/profiles").json()["profiles"] if p["name"] == "Imported"]
    assert (imported["voice_identity_id"], imported["reference_audio_id"]) == (None, sample)


def test_the_recording_limit_counts_recordings_only(tmp_path, monkeypatch, env):
    client, _library = env
    library = VoiceLibraryManager(tmp_path / "limited", max_count=1)
    monkeypatch.setattr(main, "voice_library", library)
    assert create(client).status_code == 201, "a designed sample is not a recording"
    assert create(client, name="Second").status_code == 201
    upload = client.post("/api/voices/library", data={"name": "mine"}, files={"audio": ("m.wav", wav(), "audio/wav")})
    assert upload.status_code == 201, upload.text
    full = client.post("/api/voices/library", data={"name": "more"}, files={"audio": ("m.wav", wav(), "audio/wav")})
    assert full.status_code == 422 and "1 voices max" in full.json()["error"]["message"]


def test_a_designed_voices_versions_speak_only_with_its_own_sample(env):
    client, library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    library.save("own", wav(), transcript="Mine.")
    base = f"/api/voices/identities/{voice['id']}/realizations"

    stock = client.put(f"{base}/{BASE}", json={"model": BASE, "voice": "ryan", "reference_audio_id": None})
    assert stock.status_code == 400 and "its own sample" in stock.json()["error"]["message"]
    recording = client.post(base, json={"model": SMALL_BASE, "voice": "own", "reference_audio_id": "own"})
    assert recording.status_code == 400
    [version] = client.get("/api/voices/identities?check_availability=false").json()["voices"][0]["realizations"]
    assert (version["voice"], version["reference_audio_id"]) == (sample, sample)


def test_a_version_cannot_name_another_voices_sample_as_its_voice(env):
    client, library = env
    sample = create(client).json()["design"]["sample"]
    library.save("own", wav(), transcript="Mine.")
    other = client.post("/api/voices/identities", json={"name": "Other"}).json()
    refused = client.post(f"/api/voices/identities/{other['id']}/realizations",
                          json={"model": SMALL_BASE, "voice": sample, "reference_audio_id": "own"})
    assert refused.status_code == 400 and "another named voice's designed sample" in refused.json()["error"]["message"]


def test_a_full_recordings_list_keeps_a_designed_voice_a_preset_uses(tmp_path, monkeypatch, env):
    client, _library = env
    library = VoiceLibraryManager(tmp_path / "limited", max_count=1)
    monkeypatch.setattr(main, "voice_library", library)
    voice = create(client).json()
    preset = preset_for(client, voice)
    library.save("mine", wav(), transcript="Mine.")

    refused = client.delete(f"/api/voices/identities/{voice['id']}")

    assert refused.status_code == 409 and "1 voices max" in refused.json()["error"]["message"]
    assert [item["id"] for item in client.get("/api/voices/identities?check_availability=false").json()["voices"]] \
        == [voice["id"]]
    assert client.get(f"/api/profiles/{preset['id']}").json()["voice_identity_id"] == voice["id"]
    assert [item["name"] for item in client.get("/api/voices/library").json()] == ["mine"]


def test_history_takes_of_a_deleted_designed_voice_repeat_with_its_recording(env):
    client, _library = env
    voice = create(client).json()
    sample = voice["design"]["sample"]
    preset_for(client, voice)
    entry_id = main.history_manager.log_tts(
        BASE, sample, 1.0, "mp3", "Hello.", None, None, streamed=True,
        meta={"named_voice": voice["voice"], "reference_voice": sample, "instructions": ""},
    )
    kept_id = main.history_manager.log_tts(BASE, "ryan", 1.0, "mp3", "Other.", None, None, streamed=True,
                                           meta={"reference_voice": None})

    assert client.delete(f"/api/voices/identities/{voice['id']}").json() == {"moved_to": "bright"}

    entries = {item["id"]: item for item in client.get("/api/history").json()["items"]}
    assert entries[entry_id]["voice"] == "bright"
    assert entries[entry_id]["meta"] == {"reference_voice": "bright", "instructions": ""}
    assert (entries[kept_id]["voice"], entries[kept_id]["meta"]) == ("ryan", {"reference_voice": None})


def test_a_failed_move_back_is_logged(env, monkeypatch, caplog):
    client, library = env
    voice = create(client).json()
    preset_for(client, voice)
    move = library.move

    def move_once(*args, **kwargs):
        monkeypatch.setattr(library, "move", broken)
        return move(*args, **kwargs)

    def broken(*args, **kwargs):
        raise OSError("file locked")

    def failing_commit(self):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(library, "move", move_once)
    monkeypatch.setattr(storage.StudioConnection, "commit", failing_commit)
    with pytest.raises(RuntimeError):
        client.delete(f"/api/voices/identities/{voice['id']}")
    assert "Could not move recording bright back" in caplog.text
