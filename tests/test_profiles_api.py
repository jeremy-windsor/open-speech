from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src import storage as storage_module


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
    assert listed.json()["profiles"][0]["id"] == profile_id

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

    missing_get = client.get("/api/profiles/missing")
    assert missing_get.status_code == 404
    missing_put = client.put("/api/profiles/missing", json=payload)
    assert missing_put.status_code == 404
    missing_delete = client.delete("/api/profiles/missing")
    assert missing_delete.status_code == 404
