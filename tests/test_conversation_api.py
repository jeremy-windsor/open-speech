from __future__ import annotations

from fastapi.testclient import TestClient
import numpy as np

from src.main import app
from src import main as main_module
from src import storage as storage_module


def _reset_db(tmp_path):
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    main_module.settings.os_conversations_dir = str(tmp_path / "conversations")
    storage_module._conn = None
    storage_module.init_db()


def test_conversation_crud_and_render(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)

    main_module.conversation_manager.synthesize_fn = lambda **kwargs: np.zeros(24000, dtype=np.float32)

    create = client.post(
        "/api/conversations",
        json={
            "name": "Demo",
            "turns": [
                {"speaker": "Alice", "text": "Hello", "profile_id": None, "effects": None},
                {"speaker": "Bob", "text": "Hi", "profile_id": None, "effects": [{"type": "robot"}]},
            ],
        },
    )
    assert create.status_code == 201
    cid = create.json()["id"]
    assert len(create.json()["turns"]) == 2

    listing = client.get("/api/conversations")
    assert listing.status_code == 200
    assert listing.json()["total"] >= 1

    got = client.get(f"/api/conversations/{cid}")
    assert got.status_code == 200
    assert got.json()["id"] == cid
    assert len(got.json()["turns"]) == 2

    add = client.post(f"/api/conversations/{cid}/turns", json={"speaker": "Host", "text": "Welcome", "profile_id": None, "effects": None})
    assert add.status_code == 201
    turn_id = add.json()["id"]

    delete_turn = client.delete(f"/api/conversations/{cid}/turns/{turn_id}")
    assert delete_turn.status_code == 204

    render = client.post(f"/api/conversations/{cid}/render", json={"format": "wav", "sample_rate": 24000, "save_turn_audio": True})
    assert render.status_code == 200
    assert render.json()["output_path"].endswith(".wav")

    audio = client.get(f"/api/conversations/{cid}/audio")
    assert audio.status_code == 200

    deleted = client.delete(f"/api/conversations/{cid}")
    assert deleted.status_code == 204
