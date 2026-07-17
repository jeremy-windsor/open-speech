from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

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
    artifact_dir = Path(main_module.settings.os_conversations_dir) / cid
    assert (artifact_dir / "render.wav").is_file()
    assert (artifact_dir / "turn_1.wav").is_file()

    audio = client.get(f"/api/conversations/{cid}/audio")
    assert audio.status_code == 200

    deleted = client.delete(f"/api/conversations/{cid}")
    assert deleted.status_code == 204
    assert not artifact_dir.exists()


def _insert_conversation(conversation_id: str) -> None:
    db = storage_module.get_db()
    db.execute(
        """
        INSERT INTO conversations
            (id, name, created_at, updated_at, render_output_path, meta_json)
        VALUES (?, 'Test', '2026-01-01', '2026-01-01', NULL, '{}')
        """,
        (conversation_id,),
    )
    db.commit()


def test_delete_rejects_traversal_artifact_id(tmp_path):
    _reset_db(tmp_path)
    root = Path(main_module.settings.os_conversations_dir)
    victim_id = str(uuid4())
    victim_dir = root / victim_id
    victim_dir.mkdir(parents=True)
    sentinel = victim_dir / "keep.wav"
    sentinel.write_bytes(b"keep")
    crafted_id = f"unused/../{victim_id}"
    _insert_conversation(crafted_id)

    assert main_module.conversation_manager.delete(crafted_id) is True

    assert sentinel.read_bytes() == b"keep"


def test_delete_rejects_noncanonical_uuid_artifact_id(tmp_path):
    _reset_db(tmp_path)
    root = Path(main_module.settings.os_conversations_dir)
    conversation_id = str(uuid4()).upper()
    artifact_dir = root / conversation_id
    artifact_dir.mkdir(parents=True)
    sentinel = artifact_dir / "keep.wav"
    sentinel.write_bytes(b"keep")
    _insert_conversation(conversation_id)

    assert main_module.conversation_manager.delete(conversation_id) is True

    assert sentinel.read_bytes() == b"keep"


def test_delete_unlinks_direct_symlink_without_touching_target(tmp_path):
    _reset_db(tmp_path)
    root = Path(main_module.settings.os_conversations_dir)
    root.mkdir(parents=True)
    target = tmp_path / "external-conversation-audio"
    target.mkdir()
    sentinel = target / "keep.wav"
    sentinel.write_bytes(b"keep")
    conversation_id = str(uuid4())
    artifact_link = root / conversation_id
    artifact_link.symlink_to(target, target_is_directory=True)
    _insert_conversation(conversation_id)

    assert main_module.conversation_manager.delete(conversation_id) is True

    assert not artifact_link.is_symlink()
    assert sentinel.read_bytes() == b"keep"


def test_delete_does_not_touch_artifacts_when_database_commit_fails(tmp_path):
    conversation_id = str(uuid4())
    db = MagicMock()
    db.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value={"id": conversation_id})),
        MagicMock(rowcount=1),
    ]
    db.commit.side_effect = RuntimeError("database is locked")

    with (
        patch("src.conversation.get_db", return_value=db),
        patch.object(main_module.conversation_manager, "_delete_owned_artifacts") as cleanup,
        pytest.raises(RuntimeError, match="database is locked"),
    ):
        main_module.conversation_manager.delete(conversation_id)

    db.rollback.assert_called_once_with()
    cleanup.assert_not_called()


def test_delete_commits_even_when_artifact_cleanup_fails(tmp_path, caplog):
    _reset_db(tmp_path)
    conversation = main_module.conversation_manager.create("Demo", [])

    with patch.object(
        main_module.conversation_manager,
        "_delete_owned_artifacts",
        side_effect=OSError("permission denied"),
    ):
        assert main_module.conversation_manager.delete(conversation["id"]) is True

    assert main_module.conversation_manager.get(conversation["id"]) is None
    assert "artifacts could not be fully removed" in caplog.text
