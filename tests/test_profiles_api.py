from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src import storage as storage_module
from src.main import app


def _reset_db(tmp_path):
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    storage_module._conn = None
    storage_module.init_db()


def test_profiles_crud_lifecycle(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)

    payload = {
        "name": "Narrator",
        "backend": "kokoro",
        "model": "kokoro",
        "voice": "af_heart",
        "speed": 1.0,
        "format": "mp3",
        "blend": None,
        "reference_audio_id": None,
        "effects": [],
    }

    create = client.post("/api/profiles", json=payload)
    assert create.status_code == 201
    profile_id = create.json()["id"]

    listed = client.get("/api/profiles")
    assert listed.status_code == 200
    assert any(profile["id"] == profile_id for profile in listed.json()["profiles"])

    got = client.get(f"/api/profiles/{profile_id}")
    assert got.status_code == 200
    assert got.json()["name"] == "Narrator"

    updated = client.put(f"/api/profiles/{profile_id}", json={**payload, "name": "Narrator2", "speed": 1.2})
    assert updated.status_code == 200
    assert updated.json()["name"] == "Narrator2"
    assert updated.json()["speed"] == 1.2

    deleted = client.delete(f"/api/profiles/{profile_id}")
    assert deleted.status_code == 204


def test_profiles_duplicate_default_and_404(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)
    payload = {
        "name": "Narrator",
        "backend": "kokoro",
        "model": "kokoro",
        "voice": "af_heart",
        "speed": 1.0,
        "format": "mp3",
        "blend": None,
        "reference_audio_id": None,
        "effects": [],
    }
    first = client.post("/api/profiles", json=payload)
    assert first.status_code == 201
    profile_id = first.json()["id"]

    dup = client.post("/api/profiles", json=payload)
    assert dup.status_code == 409

    set_default = client.post(f"/api/profiles/{profile_id}/default")
    assert set_default.status_code == 200
    assert set_default.json()["default_profile_id"] == profile_id
    listed = client.get("/api/profiles").json()
    assert listed["default_profile_id"] == profile_id
    assert [profile["is_default"] for profile in listed["profiles"]] == [True]

    missing_get = client.get("/api/profiles/missing")
    assert missing_get.status_code == 404
    missing_put = client.put("/api/profiles/missing", json=payload)
    assert missing_put.status_code == 404
    missing_delete = client.delete("/api/profiles/missing")
    assert missing_delete.status_code == 404


def test_profile_round_trips_reference_audio_id(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)
    payload = {
        "name": "Cloned narrator",
        "backend": "qwen3",
        "model": "qwen3/0.6b-base",
        "voice": "alex_reference",
        "speed": 1.0,
        "format": "wav",
        "blend": None,
        "reference_audio_id": "alex_reference",
        "effects": [],
    }

    created = client.post("/api/profiles", json=payload)

    assert created.status_code == 201
    profile_id = created.json()["id"]
    assert client.get(f"/api/profiles/{profile_id}").json()["reference_audio_id"] == "alex_reference"


def test_updating_preset_with_unknown_named_voice_does_not_report_profile_not_found(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)
    payload = {"name": "Narrator", "backend": "kokoro", "model": "kokoro", "voice": "af_heart"}
    preset = client.post("/api/profiles", json=payload).json()

    response = client.put(
        f"/api/profiles/{preset['id']}", json={**payload, "voice_identity_id": "no-such-voice"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Named voice not found"
    assert client.get(f"/api/profiles/{preset['id']}").status_code == 200
    missing = client.put("/api/profiles/missing", json=payload)
    assert missing.json()["error"]["message"] == "Profile not found"


def test_an_empty_named_voice_id_is_an_unknown_named_voice_not_a_duplicate_name(tmp_path):
    # "voice_identity_id": "" skipped the named-voice lookup (it is
    # falsy) and failed studio.db's foreign key, which create and update reported as 409 "Profile name
    # already exists" for a free name, or for the preset's own. No named voice has the id "", so it is
    # answered as any unknown named voice is (docs/VOICE-IDENTITIES.md: the field is optional, so null or
    # leaving it out means none).
    _reset_db(tmp_path)
    client = TestClient(app)
    payload = {"name": "Narrator", "backend": "kokoro", "model": "kokoro", "voice": "af_heart"}

    created = client.post("/api/profiles", json={**payload, "voice_identity_id": ""})

    assert (created.status_code, created.json()["error"]["message"]) == (404, "Named voice not found")
    assert client.get("/api/profiles").json()["profiles"] == []

    preset = client.post("/api/profiles", json={**payload, "voice_identity_id": None})
    assert (preset.status_code, preset.json()["voice_identity_id"]) == (201, None)
    updated = client.put(f"/api/profiles/{preset.json()['id']}", json={**payload, "voice_identity_id": ""})

    assert (updated.status_code, updated.json()["error"]["message"]) == (404, "Named voice not found")
    assert client.get(f"/api/profiles/{preset.json()['id']}").json() == preset.json()


def test_preset_with_an_invalid_effect_parameter_is_refused(tmp_path):
    # As for /v1/audio/speech: an effect value apply_chain cannot use is a client error, not a 500
    # in every later render that uses the preset.
    _reset_db(tmp_path)
    client = TestClient(app)
    payload = {"name": "Narrator", "backend": "kokoro", "model": "kokoro", "voice": "af_heart"}
    bad_effects = [{"type": "reverb", "mix": "heavy"}]

    refused = client.post("/api/profiles", json={**payload, "effects": bad_effects})
    preset = client.post("/api/profiles", json={**payload, "effects": ["robot"]})
    update = client.put(f"/api/profiles/{preset.json()['id']}", json={**payload, "effects": bad_effects})

    assert refused.status_code == 400
    assert refused.json()["error"]["message"] == "reverb mix must be a number from 0 to 1"
    assert preset.status_code == 201
    assert update.status_code == 400
    assert client.get(f"/api/profiles/{preset.json()['id']}").json()["effects"] == [{"type": "robot"}]
    assert [p["name"] for p in client.get("/api/profiles").json()["profiles"]] == ["Narrator"]


def test_preset_import_skips_an_incomplete_entry_and_keeps_the_rest(tmp_path, caplog):
    # One YAML preset without a voice (or a name) used to raise KeyError and stop startup.
    from src.profiles import ProfileManager

    _reset_db(tmp_path)
    manager = ProfileManager()

    manager.import_presets([
        {"name": "Narrator", "model": "piper/en_US-lessac-medium"},
        {"voice": "af_bella"},
        "not a mapping",
        {"name": "Butler", "voice": "bm_george"},
    ])

    assert [p["name"] for p in manager.list_all()] == ["Butler"]
    assert "Skipping voice preset" in caplog.text and "Narrator" in caplog.text


def test_app_starts_when_a_yaml_preset_entry_is_incomplete(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    config = tmp_path / "presets.yml"
    config.write_text(
        "presets:\n"
        "  narrator:\n"
        "    model: piper/en_US-lessac-medium\n"
        "  butler:\n"
        "    voice: bm_george\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TTS_VOICES_CONFIG", str(config))

    with TestClient(app) as client:
        response = client.get("/api/profiles")

    assert response.status_code == 200
    assert "narrator" not in [p["name"] for p in response.json()["profiles"]]


def _yaml_presets(tmp_path, monkeypatch, entries: dict[str, str]) -> None:
    """TTS_VOICES_CONFIG with one preset per entry: its name and the YAML lines under it."""
    _reset_db(tmp_path)
    config = tmp_path / "presets.yml"
    lines = ["presets:"]
    for name, fields in entries.items():
        lines += [f"  {name}:", *(f"    {line}" for line in fields.splitlines())]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("TTS_VOICES_CONFIG", str(config))


def test_app_starts_and_imports_the_rest_when_a_yaml_preset_field_cannot_be_stored(tmp_path, monkeypatch, caplog):
    # A field SQLite refuses (a NOT NULL column left blank, a mapping
    # where text belongs) raised out of the one import transaction, so the server did not start and no
    # preset was imported.
    _yaml_presets(tmp_path, monkeypatch, {
        "blank-speed": "voice: bm_george\nspeed:",
        "nan-speed": "voice: bm_george\nspeed: .nan",
        "null-format": "voice: bm_george\nformat: ~",
        "null-backend": "voice: bm_george\nbackend: ~",
        "mapping-instructions": "voice: bm_george\ninstructions: {a: b}",
        "dated-effect": "voice: bm_george\neffects: [{type: robot, at: 2020-01-01}]",
        "butler": "voice: bm_george",
    })

    with TestClient(app) as client:
        response = client.get("/api/profiles")

    assert response.status_code == 200
    assert [p["name"] for p in response.json()["profiles"]] == ["butler"]
    for name, field in [
        ("blank-speed", "speed"), ("nan-speed", "speed"), ("null-format", "format"),
        ("null-backend", "backend"), ("mapping-instructions", "instructions"), ("dated-effect", "effects"),
    ]:
        assert f"Skipping voice preset '{name}': {field}" in caplog.text, name


def test_a_yaml_preset_with_a_value_the_api_refuses_is_skipped_and_the_catalog_answers(tmp_path, monkeypatch):
    # These were stored as they were. An infinite speed made
    # GET /api/voice-presets answer 500, "effects: 5" made GET /api/profiles answer 500, and a text speed
    # failed every conversation render. Later YAML edits never overwrite an imported preset, so the
    # catalog stayed broken. POST /api/profiles refuses each of them.
    _yaml_presets(tmp_path, monkeypatch, {
        "infinite": "voice: bm_george\nspeed: .inf",
        "backwards": "voice: bm_george\nspeed: -.inf",
        "text-speed": "voice: bm_george\nspeed: fast",
        "boolean-speed": "voice: bm_george\nspeed: true",
        "too-slow": "voice: bm_george\nspeed: 0.1",
        "too-fast": "voice: bm_george\nspeed: 5.0",
        "counted-effects": "voice: bm_george\neffects: 5",
        "bad-reverb": "voice: bm_george\neffects: [{type: reverb, mix: heavy}]",
        "steady": "voice: bm_george\nspeed: 2\neffects: [robot]",
        "butler": "voice: bm_george",
    })

    with TestClient(app, raise_server_exceptions=False) as client:
        catalog = client.get("/api/voice-presets")
        listed = client.get("/api/profiles")

    assert catalog.status_code == 200, catalog.text
    assert listed.status_code == 200, listed.text
    assert [(p["name"], p["speed"], p["effects"]) for p in catalog.json()["presets"]] == [
        ("butler", 1.0, []),
        ("steady", 2.0, [{"type": "robot"}]),
    ]


def test_a_preset_speed_outside_the_speech_range_is_refused_before_it_is_stored(tmp_path):
    # JSON has no Infinity or NaN, but Starlette parses them (and 1e309 overflows to inf): an
    # infinite speed was stored and then every answer naming it was a 500; NaN read as a duplicate.
    _reset_db(tmp_path)
    client = TestClient(app)
    body = '{"name": "Fast", "backend": "kokoro", "model": "kokoro", "voice": "af_heart", "speed": %s}'
    headers = {"content-type": "application/json"}

    for speed in ("1e309", "Infinity", "-Infinity", "NaN", "0.1", "5.0"):
        response = client.post("/api/profiles", content=body % speed, headers=headers)
        assert response.status_code == 422, (speed, response.text)
    assert client.get("/api/profiles").json()["profiles"] == []

    preset = client.post("/api/profiles", content=body % "1.5", headers=headers).json()
    assert client.put(f"/api/profiles/{preset['id']}", content=body % "1e309", headers=headers).status_code == 422
    assert client.get(f"/api/profiles/{preset['id']}").json()["speed"] == 1.5
    assert client.get(f"/api/profiles/{preset['id']}/resolve").status_code == 200
    assert [entry["speed"] for entry in client.get("/api/voice-presets").json()["presets"]] == [1.5]


def test_a_lone_surrogate_in_a_preset_is_refused_naming_the_field_not_a_duplicate(tmp_path):
    # Half of an emoji's UTF-16 pair (the JSON escape "\ud83d") reached
    # studio.db, which can't encode it, and the UnicodeEncodeError was answered 409 (a duplicate name)
    # with Python's codec message. It is refused with 422 naming the field, and nothing is stored.
    _reset_db(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"content-type": "application/json"}
    body = '{"name": "%s", "backend": "kokoro", "voice": "%s", "instructions": "%s"}'
    lone = "\\ud83d"  # the JSON escape, which json.loads decodes to a lone surrogate

    for field, values in (("name", ("Narrator " + lone, "af_heart", "")),
                          ("voice", ("Narrator", "af_heart" + lone, "")),
                          ("instructions", ("Narrator", "af_heart", "Calm " + lone))):
        response = client.post("/api/profiles", content=body % values, headers=headers)
        assert response.status_code == 422, (field, response.text)
        assert field in response.text and "surrogate" in response.text, (field, response.text)
    assert client.get("/api/profiles").json()["profiles"] == []

    preset = client.post("/api/profiles", content=body % ("Narrator", "af_heart", ""), headers=headers).json()
    update = client.put(f"/api/profiles/{preset['id']}", content=body % ("Narrator " + lone, "af_heart", ""),
                        headers=headers)
    assert update.status_code == 422, update.text
    assert "name" in update.text and "surrogate" in update.text
    assert client.get(f"/api/profiles/{preset['id']}").json()["name"] == "Narrator"


def _presets_with_an_escaped_emoji(field: str) -> str:
    """TTS_VOICES_CONFIG as json.dumps writes it: an emoji in ``field`` of the first preset becomes the UTF-16 escape
    pair "\\ud83d\\ude00", which PyYAML reads as two lone surrogates; the second preset is valid."""
    bad = {"name": "Narrator", "voice": "bm_george", field: "Calm \U0001F600"}
    text = json.dumps({"presets": [bad, {"name": "butler", "voice": "bm_george"}]})
    assert "\\ud83d\\ude00" in text
    return text + "\n"


@pytest.mark.parametrize("field", ["name", "voice", "instructions", "model", "backend", "format",
                                   "reference_audio_id"])
def test_a_yaml_preset_holding_an_escaped_emoji_pair_is_skipped_naming_the_field(tmp_path, monkeypatch, caplog,
                                                                                 field):
    # PyYAML reads the pair as two lone surrogates, and the import bound the name
    # (or the INSERT bound the field) before any check, so SQLite's UTF-8 encode raised out of the one import
    # transaction: the server did not start and no preset of the file was imported. POST /api/profiles refuses a
    # lone surrogate, so the entry is skipped with a warning naming the field, and the rest imported.
    from src.profiles import ProfileManager
    from src.services import tts as tts_service

    _reset_db(tmp_path)
    config = tmp_path / "presets.yml"
    config.write_text(_presets_with_an_escaped_emoji(field), encoding="utf-8")
    monkeypatch.setenv("TTS_VOICES_CONFIG", str(config))
    presets = tts_service.load_voice_presets()
    assert [hex(ord(c)) for c in presets[0][field][-2:]] == ["0xd83d", "0xde00"]

    manager = ProfileManager()
    with caplog.at_level(logging.WARNING, logger="src.profiles"):
        manager.import_presets(presets)

    assert [p["name"] for p in manager.list_all()] == ["butler"]
    skipped = [r.getMessage() for r in caplog.records if "Skipping voice preset" in r.getMessage()]
    assert len(skipped) == 1 and f": {field} has a lone surrogate" in skipped[0], skipped


def test_the_app_starts_when_a_yaml_preset_holds_an_escaped_emoji_pair(tmp_path, monkeypatch):
    # The UnicodeEncodeError left lifespan, so the server never started.
    _reset_db(tmp_path)
    config = tmp_path / "presets.yml"
    config.write_text(_presets_with_an_escaped_emoji("name"), encoding="utf-8")
    monkeypatch.setenv("TTS_VOICES_CONFIG", str(config))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/profiles")

    assert response.status_code == 200, response.text
    assert [p["name"] for p in response.json()["profiles"]] == ["butler"]
