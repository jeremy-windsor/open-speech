"""Named voices resolve to exact models while legacy recipes remain usable."""

from __future__ import annotations

import io
import sqlite3
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main, storage
from src.conversation import ConversationManager
from src.profiles import ProfileManager
from src.services.tts import load_voice_presets
from src.tts.external import ExternalProviderError
from src.voice_identities import VoiceIdentityManager, resolve_named_voice
from src.voice_library import VoiceLibraryManager


class Router:
    def __init__(self):
        self.requests = []
        self.offline = False

    def get_backend(self, model):
        if model == "clone/model" and self.offline:
            raise ExternalProviderError("Worker offline", status_code=503)
        if model not in ("kokoro", "clone/model"):
            raise ValueError("Unknown model")
        capabilities = (
            {
                "voice_clone": True,
                "clone_transcript_required": True,
                "instructions": True,
                "live_reader": True,
            }
            if model == "clone/model"
            else {"speed_control": True}
        )
        return SimpleNamespace(name=model.split("/")[0], capabilities=capabilities)

    def get_capabilities(self, model):
        return self.get_backend(model).capabilities

    def validate_voice(self, model, voice):
        self.get_backend(model)
        if model == "kokoro" and voice not in ("af_heart", "af_bella(2)+af_sky(1)"):
            raise ValueError("Unknown Kokoro voice")

    def list_voices(self, model):
        self.get_backend(model)
        return [] if model == "clone/model" else ["af_heart"]

    def sample_rate_for(self, model):
        self.get_backend(model)
        return 24000

    def synthesize(self, **kwargs):
        self.requests.append(kwargs)
        yield np.full(2400, 0.2, dtype=np.float32)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    monkeypatch.setattr(storage, "_conn", connection)
    storage.init_db()
    router = Router()
    library = VoiceLibraryManager(tmp_path / "voices")
    monkeypatch.setattr(main, "tts_router", router)
    monkeypatch.setattr(main, "voice_library", library)
    monkeypatch.setattr(main.settings, "tts_cache_enabled", False)
    monkeypatch.setattr(main.settings, "tts_live_enabled", True)
    monkeypatch.setattr(main.settings, "os_conversations_dir", str(tmp_path / "conversations"))
    yield TestClient(main.app), router, library
    connection.close()


def named_voice(model="kokoro", voice="af_bella(2)+af_sky(1)", reference=None):
    manager = VoiceIdentityManager()
    identity = manager.create("Will")
    manager.add_realization(identity["id"], model=model, voice=voice, reference_audio_id=reference)
    return identity


def preset(identity=None, **changes):
    return ProfileManager().create(
        **{
            "name": "Reading",
            "backend": "kokoro",
            "model": "kokoro",
            "voice": "af_heart",
            "speed": 1.2,
            "format": "wav",
            "blend": None,
            "reference_audio_id": None,
            "effects": [],
            "voice_identity_id": identity["id"] if identity else None,
            **changes,
        }
    )


def save_reference(library):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        recording.writeframes(np.full(2400, 1000, dtype="<i2").tobytes())
    library.save("reference", buffer.getvalue(), transcript="The exact recorded words.")


def test_identity_keeps_distinct_model_realizations_and_rejects_substitution(environment):
    identity = named_voice()
    manager = VoiceIdentityManager()
    manager.add_realization(
        identity["id"], model="clone/model", voice="reference", reference_audio_id="reference"
    )
    assert resolve_named_voice("kokoro", identity["voice"])[0] == "af_bella(2)+af_sky(1)"
    assert resolve_named_voice("clone/model", identity["voice"]) == ("reference", "reference")
    with pytest.raises(ValueError, match="no realization"):
        resolve_named_voice("other/model", identity["voice"])
    with pytest.raises(ValueError, match="cannot be overridden"):
        resolve_named_voice("clone/model", identity["voice"], "different-recording")
    with pytest.raises(ValueError, match="already has"):
        manager.add_realization(identity["id"], model="kokoro", voice="af_heart")
    with pytest.raises(ValueError, match="already exists"):
        manager.create("will")


def test_api_creates_explicit_realization_and_marks_provider_unavailable(environment):
    client, router, library = environment
    save_reference(library)
    identity = client.post("/api/voices/identities", json={"name": "Narrator"}).json()
    url = f"/api/voices/identities/{identity['id']}/realizations"
    assert client.post(url, json={"model": "clone/model", "voice": "reference"}).status_code == 400
    assert (
        client.post(
            url,
            json={"model": "clone/model", "voice": "reference", "reference_audio_id": "reference"},
        ).status_code
        == 201
    )
    assert client.post(url, json={"model": "kokoro", "voice": "invented"}).status_code == 400
    assert (
        client.get("/api/voices/identities?model=kokoro").json()["voices"][0]["available"] is False
    )
    assert (
        client.get("/api/voices/identities?model=clone/model").json()["voices"][0]["available"]
        is True
    )
    router.offline = True
    unavailable = client.get("/api/voices/identities?model=clone/model").json()["voices"][0]
    assert unavailable["id"] == identity["id"]
    assert unavailable["available"] is False
    assert unavailable["realizations"][0]["reference_audio_id"] == "reference"
    router.offline = False
    library.delete("reference")
    assert (
        client.get("/api/voices/identities?model=clone/model").json()["voices"][0]["available"]
        is False
    )


def test_legacy_migration_preserves_profile_recipe_and_is_idempotent(environment):
    connection = storage.get_db()
    old_schema = storage.SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS profiles (")[1].split(
        ");", 1
    )[0]
    old_schema = old_schema.replace(
        "  voice_identity_id TEXT REFERENCES voice_identities(id),\n", ""
    )
    old_schema = old_schema.replace("  instructions TEXT,\n", "")
    connection.execute("DROP TABLE profiles")
    connection.execute("DELETE FROM schema_migrations")
    connection.execute("CREATE TABLE profiles (" + old_schema + ")")
    connection.execute(
        "INSERT INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-id",
            "Will",
            "kokoro",
            "kokoro",
            "af_heart",
            1.2,
            "wav",
            "af_bella(2)+af_sky(1)",
            None,
            '[{"type":"reverb"}]',
            1,
            "old",
            "old",
        ),
    )
    connection.commit()
    storage.init_db()
    manager = ProfileManager()
    migrated = manager.get("legacy-id")
    assert migrated["is_default"] is True
    assert migrated["blend"] == "af_bella(2)+af_sky(1)"
    assert migrated["speed"] == 1.2
    assert migrated["effects"] == [{"type": "reverb"}]
    assert manager.resolve("legacy-id")["voice"] == migrated["blend"]
    identity_id = migrated["voice_identity_id"]
    storage.init_db()
    assert manager.get("legacy-id")["voice_identity_id"] == identity_id
    assert len(VoiceIdentityManager().list_all()) == 1
    manager.update("legacy-id", voice="af_heart", blend=None)
    storage.init_db()
    assert manager.get("legacy-id")["voice_identity_id"] is None


def test_yaml_templates_and_saved_presets_share_one_catalog_without_reimport(
    environment, tmp_path, monkeypatch
):
    client, _, _ = environment
    yaml_path = tmp_path / "voices.yml"
    yaml_path.write_text('presets:\n  narrator:\n    voice: "af_heart"\n    speed: 1.1\n')
    monkeypatch.setenv("TTS_VOICES_CONFIG", str(yaml_path))
    assert load_voice_presets()[0]["name"] == "narrator"
    ProfileManager().import_presets(load_voice_presets())
    preset(name="My own preset")
    profiles = client.get("/api/profiles").json()["profiles"]
    assert profiles == client.get("/api/voice-presets").json()["presets"]
    imported = next(item for item in profiles if item["name"] == "narrator")
    assert imported["voice_identity_id"]
    client.delete(f"/api/profiles/{imported['id']}")
    assert [p["name"] for p in client.get("/api/voice-presets").json()["presets"]] == [
        "My own preset"
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_named_voice_http_returns_audio_and_exact_blend(environment, stream):
    client, router, _ = environment
    identity = named_voice()
    response = client.post(
        f"/v1/audio/speech?stream={str(stream).lower()}",
        json={
            "model": "kokoro",
            "voice": identity["voice"],
            "input": "Hello",
            "response_format": "wav",
        },
    )
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content)) as audio:
        assert audio.getnchannels() == 1
        assert audio.getnframes() > 0
    assert router.requests[0]["voice"] == "af_bella(2)+af_sky(1)"


def test_named_clone_resolves_private_reference_without_accepting_override(environment):
    client, router, library = environment
    save_reference(library)
    identity = named_voice("clone/model", "reference", "reference")
    payload = {
        "model": "clone/model",
        "voice": identity["voice"],
        "input": "Hello",
        "response_format": "wav",
    }
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert router.requests[0]["reference_audio"] == library.get("reference")[0]
    assert router.requests[0]["clone_transcript"] == "The exact recorded words."
    assert "X-Cache" not in response.headers
    assert (
        client.post("/v1/audio/speech", json={**payload, "reference_audio": "AAAA"}).status_code
        == 400
    )
    library.delete("reference")
    assert client.post("/v1/audio/speech", json=payload).status_code == 404


def test_missing_realization_is_an_error_before_synthesis(environment):
    client, router, _ = environment
    identity = named_voice()
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "clone/model",
            "voice": identity["voice"],
            "input": "Hello",
        },
    )
    assert response.status_code == 400
    assert "no realization" in response.text
    assert router.requests == []


def test_live_reader_resolves_named_voice_and_rejects_unavailable_model(environment):
    client, router, _ = environment
    identity = named_voice()
    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "clone/model",
                    "voice": identity["voice"],
                },
            }
        )
        assert websocket.receive_json()["type"] == "error"
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "kokoro",
                    "voice": identity["voice"],
                },
            }
        )
        assert websocket.receive_json()["session"]["voice"] == "af_bella(2)+af_sky(1)"
        websocket.send_json({"type": "input_text.append", "text": "Hello."})
        websocket.send_json({"type": "input_text.commit"})
        while websocket.receive_json()["type"] != "input_text.done":
            pass
    assert router.requests[0]["voice"] == "af_bella(2)+af_sky(1)"


def test_conversation_uses_named_voice_and_keeps_preset_settings(environment, monkeypatch):
    identity = named_voice()
    profile = preset(identity, effects=[{"type": "reverb"}])
    requests, applied_effects = [], []
    manager = ConversationManager(
        ProfileManager(),
        lambda **kwargs: requests.append(kwargs) or np.full(2400, 0.2, dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.conversation.apply_chain",
        lambda audio, rate, effects: applied_effects.extend(effects) or audio,
    )
    conversation = manager.create(
        "Reading", [{"text": "Hello", "profile_id": profile["id"], "effects": [{"type": "robot"}]}]
    )
    manager.render(conversation["id"])
    assert requests[0]["voice"] == "af_bella(2)+af_sky(1)"
    assert requests[0]["speed"] == 1.2
    assert applied_effects == [{"type": "reverb"}, {"type": "robot"}]
    ProfileManager().delete(profile["id"])
    with pytest.raises(ValueError, match="preset is unavailable"):
        manager.render(conversation["id"])
    assert len(requests) == 1


def test_preset_resolve_rejects_missing_realization_and_updates_keep_recipe_semantics(environment):
    client, _, _ = environment
    identity = named_voice()
    profile = preset(identity)
    manager = ProfileManager()
    manager.update(profile["id"], speed=1.5, instructions="Quietly")
    resolved = client.get(f"/api/profiles/{profile['id']}/resolve").json()
    assert resolved["voice_identity_id"] == identity["id"]
    assert resolved["instructions"] == "Quietly"
    assert resolved["speed"] == 1.5
    with pytest.raises(ValueError, match="no realization"):
        manager.update(profile["id"], voice_identity_id=identity["id"], model="clone/model")
    storage.get_db().execute(
        "DELETE FROM voice_realizations WHERE voice_identity_id = ?", (identity["id"],)
    )
    response = client.get(f"/api/profiles/{profile['id']}/resolve")
    assert response.status_code == 409


def test_standalone_conversation_preset_uses_blend_and_instructions(environment):
    profile = preset(blend="af_bella(2)+af_sky(1)", instructions="Read softly.")
    requests = []
    manager = ConversationManager(ProfileManager(), lambda **kwargs: (
        requests.append(kwargs) or np.full(2400, 0.2, dtype=np.float32)
    ))
    conversation = manager.create("Legacy reading", [{"text": "Hello", "profile_id": profile["id"]}])
    manager.render(conversation["id"])
    assert requests[0]["voice"] == "af_bella(2)+af_sky(1)"
    assert requests[0]["instructions"] == "Read softly."
