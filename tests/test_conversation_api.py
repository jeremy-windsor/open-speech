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


def test_conversation_crud_and_render(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    client = TestClient(app)

    monkeypatch.setattr(
        main_module.conversation_manager, "synthesize_fn", lambda **kwargs: np.zeros(24000, dtype=np.float32)
    )

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


def test_delete_unlinks_direct_symlink_without_touching_target(tmp_path, create_symlink):
    _reset_db(tmp_path)
    root = Path(main_module.settings.os_conversations_dir)
    root.mkdir(parents=True)
    target = tmp_path / "external-conversation-audio"
    target.mkdir()
    sentinel = target / "keep.wav"
    sentinel.write_bytes(b"keep")
    conversation_id = str(uuid4())
    artifact_link = root / conversation_id
    create_symlink(artifact_link, target, target_is_directory=True)
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
        patch("src.storage.get_db", return_value=db),
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


def _assert_studio_db_writable() -> None:
    # The Reader keeps its own connection to studio.db; it must not find it write-locked.
    import sqlite3

    other = sqlite3.connect(main_module.settings.os_studio_db_path, timeout=0)
    try:
        other.execute("INSERT INTO preset_imports VALUES (?)", (str(uuid4()),))
        other.commit()
    finally:
        other.close()


def _two_turn_manager(synthesize_fn):
    from src.conversation import ConversationManager

    manager = ConversationManager(profile_manager=None, synthesize_fn=synthesize_fn)
    conversation = manager.create(
        "Demo", [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]
    )
    return manager, conversation["id"]


def test_conversation_render_does_not_hold_studio_write_lock_while_synthesizing(tmp_path):
    _reset_db(tmp_path)
    lock_errors: list[str] = []
    calls = {"n": 0}

    def synthesize(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            try:
                _assert_studio_db_writable()
            except Exception as exc:
                lock_errors.append(str(exc))
        return np.zeros(240, dtype=np.float32)

    manager, cid = _two_turn_manager(synthesize)
    manager.render(cid, format="wav", sample_rate=24000)

    assert lock_errors == []
    assert all(turn["duration_ms"] == 10 for turn in manager.get(cid)["turns"])


def test_conversation_render_file_is_written_while_other_threads_can_read_studio_db(tmp_path, monkeypatch):
    # Every statement on studio.db's shared connection waits for another thread's transaction,
    # and a long conversation's render is tens of MB: it is written before the transaction,
    # which only moves it into place.
    import threading

    from src.voice_identities import VoiceIdentityManager

    _reset_db(tmp_path)
    manager, cid = _two_turn_manager(lambda **_kwargs: np.zeros(240, dtype=np.float32))
    write_bytes = Path.write_bytes
    readers: list[threading.Thread] = []
    reader_waited: list[bool] = []

    def write_while_another_thread_reads(path, data):
        if "render" in path.name:
            reader = threading.Thread(target=VoiceIdentityManager().list_all, daemon=True)
            reader.start()
            reader.join(0.5)
            reader_waited.append(reader.is_alive())
            readers.append(reader)
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", write_while_another_thread_reads)
    manager.render(cid, format="wav", sample_rate=24000)
    for reader in readers:
        reader.join(10)

    assert reader_waited == [False]
    assert (Path(main_module.settings.os_conversations_dir) / cid / "render.wav").is_file()


def test_conversation_render_replaces_a_render_that_is_being_downloaded(tmp_path):
    # Windows refuses to move a file over one that another handle (a download) has open.
    from scipy.io import wavfile

    _reset_db(tmp_path)
    samples = {"n": 240}
    manager, cid = _two_turn_manager(lambda **_kwargs: np.zeros(samples["n"], dtype=np.float32))
    first = Path(manager.render(cid, format="wav", sample_rate=24000)["output_path"])
    samples["n"] = 480
    with first.open("rb"):  # a download of the first render, still reading
        second = manager.render(cid, format="wav", sample_rate=24000)

    _, rendered = wavfile.read(second["output_path"])
    assert second["output_path"] == str(first)
    assert len(rendered) == 480 + 12000 + 480  # two turns and the half second between them
    assert sorted(path.name for path in first.parent.iterdir()) == ["render.wav", "turn_1.wav", "turn_2.wav"]


def test_conversation_render_replaces_turn_audio_that_is_being_read(tmp_path):
    # Turn files are staged and moved into place at the end, and
    # Windows refuses that move while another handle (a composer render reading a source) has one open.
    from scipy.io import wavfile

    _reset_db(tmp_path)
    samples = {"n": 240}
    manager, cid = _two_turn_manager(lambda **_kwargs: np.zeros(samples["n"], dtype=np.float32))
    manager.render(cid, format="wav", sample_rate=24000)
    first_turn = Path(manager.get(cid)["turns"][0]["audio_path"])
    samples["n"] = 480
    with first_turn.open("rb"):  # a reader of the first render's turn audio
        manager.render(cid, format="wav", sample_rate=24000)

    _, spoken = wavfile.read(first_turn)
    assert len(spoken) == 480
    assert manager.get(cid)["turns"][0]["duration_ms"] == 20
    assert sorted(path.name for path in first_turn.parent.iterdir()) == ["render.wav", "turn_1.wav", "turn_2.wav"]


def test_failed_conversation_render_releases_studio_write_lock(tmp_path):
    _reset_db(tmp_path)
    calls = {"n": 0}

    def synthesize(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("provider rejected the second turn")
        return np.zeros(240, dtype=np.float32)

    manager, cid = _two_turn_manager(synthesize)
    with pytest.raises(ValueError):
        manager.render(cid, format="wav", sample_rate=24000)

    _assert_studio_db_writable()


# Each turn text speaks its own length at its own level, so a turn
# file can be traced back to the turn that spoke it.
_TURN_SPEECH = {"one": (1000, 0.1), "two": (2000, 0.2), "three": (3000, 0.3), "four": (4000, 0.4)}


def _speak_turn_text(**kwargs):
    length, level = _TURN_SPEECH[kwargs["text"]]
    return np.full(length, level, dtype=np.float32)


def _assert_each_turn_names_its_own_audio(turns: list[dict]) -> None:
    from scipy.io import wavfile

    for turn in turns:
        rate, pcm = wavfile.read(turn["audio_path"])
        length, level = _TURN_SPEECH[turn["text"]]
        spoken = (len(pcm), int(pcm[0]))
        assert spoken == (length, int(np.float32(level) * 32767)), (
            f"turn {turn['text']!r}: {Path(turn['audio_path']).name} holds {spoken[0]} frames at level {spoken[1]}"
        )
        assert turn["duration_ms"] == int(1000 * len(pcm) / rate), turn["text"]


def test_failed_conversation_render_changes_no_turns_audio(tmp_path, monkeypatch):
    # After turn one is deleted the others still name turn_2.wav to
    # turn_4.wav. A render that then fails on its third turn has spoken two and three already, and must not
    # have written them over turn_1.wav and turn_2.wav: turn two's path would name turn three's speech.
    _reset_db(tmp_path)
    fail_on = {"text": None}

    def synthesize(**kwargs):
        if kwargs["text"] == fail_on["text"]:
            raise ValueError("provider rejected this turn")
        return _speak_turn_text(**kwargs)

    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app)
    cid = client.post(
        "/api/conversations",
        json={"name": "Demo", "turns": [{"speaker": "A", "text": text} for text in _TURN_SPEECH]},
    ).json()["id"]
    render = {"format": "wav", "sample_rate": 8000}
    assert client.post(f"/api/conversations/{cid}/render", json=render).status_code == 200
    first_turn = client.get(f"/api/conversations/{cid}").json()["turns"][0]
    assert client.delete(f"/api/conversations/{cid}/turns/{first_turn['id']}").status_code == 204
    before = client.get(f"/api/conversations/{cid}").json()["turns"]
    _assert_each_turn_names_its_own_audio(before)

    fail_on["text"] = "four"
    failed = client.post(f"/api/conversations/{cid}/render", json=render)

    assert failed.status_code == 400
    after = client.get(f"/api/conversations/{cid}").json()["turns"]
    _assert_each_turn_names_its_own_audio(after)
    assert after == before
    out_dir = Path(main_module.settings.os_conversations_dir) / cid
    assert sorted(path.name for path in out_dir.iterdir()) == [  # no staged turn file left behind
        "render.wav", "turn_1.wav", "turn_2.wav", "turn_3.wav", "turn_4.wav"
    ]


class _OsWithAFailingMove:
    """Stands in for the os module src.conversation sees: records where each staged file is moved, and
    the first move of a staged file onto ``name`` raises EIO, a disk error. With name None none fails."""

    def __init__(self, real_os, name: str | None) -> None:
        self._os = real_os
        self.name = name
        self.staged_moves: list[str] = []

    def __getattr__(self, attr):
        return getattr(self._os, attr)

    def replace(self, src, dst, *args, **kwargs):
        import errno

        if Path(src).name.startswith((".render-", ".turn-")):
            if Path(dst).name == self.name:
                self.name = None
                raise OSError(errno.EIO, "Input/output error (injected)", str(dst))
            self.staged_moves.append(Path(dst).name)
        return self._os.replace(src, dst, *args, **kwargs)


def _four_turns_rendered_then_turn_one_deleted(manager) -> tuple[str, list[dict], Path]:
    """Turns two to four then name turn_2.wav to turn_4.wav; the next render gives them turn_1.wav to turn_3.wav."""
    cid = manager.create("Demo", [{"speaker": "A", "text": text} for text in _TURN_SPEECH])["id"]
    manager.render(cid, format="wav", sample_rate=8000)
    assert manager.delete_turn(cid, manager.get(cid)["turns"][0]["id"])
    before = manager.get(cid)["turns"]
    assert [Path(turn["audio_path"]).name for turn in before] == ["turn_2.wav", "turn_3.wav", "turn_4.wav"]
    _assert_each_turn_names_its_own_audio(before)
    return cid, before, Path(main_module.settings.os_conversations_dir) / cid


def _folder(out_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}


@pytest.mark.parametrize("held_open", [False, True], ids=["moved", "turn_2.wav open"])
def test_a_render_whose_turn_file_move_fails_half_way_puts_the_moved_files_back(tmp_path, monkeypatch, held_open):
    # In the final transaction render.wav and each staged turn file were moved
    # into place before the UPDATEs. An OSError other than PermissionError part way rolled the rows back but
    # left the files already moved, so turn two's turn_2.wav held turn three's speech (CHANGELOG "Studio: a
    # conversation render that fails": no turn's audio file changes). With turn_2.wav open (a composer render
    # reading it), Windows refuses to move it, so it is written in place and its old bytes are written back.
    import contextlib

    from src import conversation as conversation_module
    from src.conversation import ConversationManager

    _reset_db(tmp_path)
    manager = ConversationManager(synthesize_fn=_speak_turn_text)
    cid, before, out_dir = _four_turns_rendered_then_turn_one_deleted(manager)
    files = _folder(out_dir)
    moves = _OsWithAFailingMove(conversation_module.os, "turn_3.wav")
    monkeypatch.setattr(conversation_module, "os", moves)

    with contextlib.ExitStack() as stack:
        if held_open:
            stack.enter_context((out_dir / "turn_2.wav").open("rb"))
        with pytest.raises(OSError, match="injected"):
            manager.render(cid, format="wav", sample_rate=8000)

    assert moves.staged_moves == ["render.wav", "turn_1.wav", "turn_2.wav"]   # then the move onto turn_3.wav failed
    after = manager.get(cid)
    assert (after["turns"], after["render_output_path"]) == (before, None)   # the rows rolled back
    _assert_each_turn_names_its_own_audio(after["turns"])
    assert _folder(out_dir) == files   # render.wav too, and nothing staged or kept aside is left


def test_a_render_whose_turn_update_waits_out_the_busy_timeout_puts_the_moved_files_back(tmp_path, monkeypatch):
    # SQLite's write lock is asked for only at the first UPDATE, after every file
    # has moved. When another connection holds it past the busy timeout, as a long Reader import does on the
    # Reader's own connection, the UPDATE fails and the rows roll back, so the files must go back too.
    import concurrent.futures
    import sqlite3

    from src import conversation as conversation_module
    from src.conversation import ConversationManager

    monkeypatch.setattr(storage_module, "BUSY_TIMEOUT_S", 0.5)   # 30 s; read when the connection opens
    _reset_db(tmp_path)
    manager = ConversationManager(synthesize_fn=_speak_turn_text)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    blocker = sqlite3.connect(main_module.settings.os_studio_db_path, timeout=0, isolation_level=None)
    try:
        cid, before, out_dir = _four_turns_rendered_then_turn_one_deleted(manager)
        files = _folder(out_dir)
        moves = _OsWithAFailingMove(conversation_module.os, None)   # records the moves; none fails
        monkeypatch.setattr(conversation_module, "os", moves)
        blocker.execute("BEGIN IMMEDIATE")   # WAL lets the render's reads through, not its UPDATE
        render = pool.submit(manager.render, cid, format="wav", sample_rate=8000)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            render.result(timeout=60)
    finally:
        blocker.rollback()
        blocker.close()
        pool.shutdown(wait=False, cancel_futures=True)

    try:
        assert moves.staged_moves == ["render.wav", "turn_1.wav", "turn_2.wav", "turn_3.wav"]   # all before the UPDATE
        after = manager.get(cid)
        assert (after["turns"], after["render_output_path"]) == (before, None)
        _assert_each_turn_names_its_own_audio(after["turns"])
        assert _folder(out_dir) == files
    finally:
        storage_module._conn.close()   # its busy timeout is 0.5 s
        storage_module._conn = None


def test_a_render_retired_by_a_turn_edit_leaves_the_newer_renders_turn_audio(tmp_path, monkeypatch):
    # A render answered 409 spoke the turns at positions they no longer
    # have, so it must neither move its turn files into place nor save their paths. Here a newer render
    # has already given turn_1.wav and turn_2.wav to the turns as they are now.
    _reset_db(tmp_path)
    manager = main_module.conversation_manager
    client = TestClient(app)
    conversation = client.post(
        "/api/conversations",
        json={"name": "Demo", "turns": [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]},
    ).json()
    cid = conversation["id"]
    edited = {"done": False}
    newer: dict = {}

    def synthesize(**kwargs):
        if not edited["done"]:  # while the first render speaks turn one, the turns change and render again
            edited["done"] = True
            assert manager.delete_turn(cid, conversation["turns"][0]["id"])
            manager.add_turn(cid, "C", "three")
            manager.render(cid, format="wav", sample_rate=8000)
            newer["turns"] = manager.get(cid)["turns"]
        return _speak_turn_text(**kwargs)

    monkeypatch.setattr(manager, "synthesize_fn", synthesize)

    stale = client.post(f"/api/conversations/{cid}/render", json={"format": "wav", "sample_rate": 8000})

    assert stale.status_code == 409
    turns = client.get(f"/api/conversations/{cid}").json()["turns"]
    _assert_each_turn_names_its_own_audio(turns)
    assert turns == newer["turns"]
    out_dir = Path(main_module.settings.os_conversations_dir) / cid
    assert sorted(path.name for path in out_dir.iterdir()) == ["render.wav", "turn_1.wav", "turn_2.wav"]


def test_profile_name_conflict_during_render_keeps_rendered_turn_audio(tmp_path):
    from src.profiles import ProfileManager

    _reset_db(tmp_path)
    profiles = ProfileManager()
    profiles.create("Taken", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None, [])
    calls = {"n": 0}

    def synthesize(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            # A concurrent duplicate-name POST /api/profiles rolls back the shared connection.
            with pytest.raises(ValueError):
                profiles.create("Taken", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None, [])
        return np.zeros(240, dtype=np.float32)

    manager, cid = _two_turn_manager(synthesize)
    manager.render(cid, format="wav", sample_rate=24000)

    turns = manager.get(cid)["turns"]
    assert all(turn["audio_path"] for turn in turns)
    assert all(turn["duration_ms"] == 10 for turn in turns)


def test_deleting_unknown_conversation_turn_releases_studio_write_lock(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)
    conversation = client.post("/api/conversations", json={"name": "Demo", "turns": []}).json()

    response = client.delete(f"/api/conversations/{conversation['id']}/turns/not-a-turn")

    assert response.status_code == 404
    _assert_studio_db_writable()


def test_render_with_missing_reference_recording_names_the_recording(tmp_path, monkeypatch):
    from src.voice_library import VoiceLibraryManager

    _reset_db(tmp_path)
    monkeypatch.setattr(main_module, "voice_library", VoiceLibraryManager(tmp_path / "voices"))
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", main_module._synthesize_array)
    client = TestClient(app)
    preset = client.post(
        "/api/profiles",
        json={"name": f"Clone {uuid4()}", "backend": "kokoro", "model": "kokoro", "voice": "af_heart",
              "reference_audio_id": "deleted_recording"},
    ).json()
    conversation = client.post(
        "/api/conversations",
        json={"name": "Demo", "turns": [{"speaker": "A", "text": "hi", "profile_id": preset["id"]}]},
    ).json()

    response = client.post(f"/api/conversations/{conversation['id']}/render", json={})

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Voice library entry 'deleted_recording' not found"
    assert client.get(f"/api/conversations/{conversation['id']}").status_code == 200


def test_render_unknown_conversation_is_not_found(tmp_path):
    _reset_db(tmp_path)
    response = TestClient(app).post(f"/api/conversations/{uuid4()}/render", json={})

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Conversation not found"


def test_turn_with_an_invalid_effect_parameter_is_refused_before_it_is_stored(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    synthesize = MagicMock(return_value=np.zeros(240, dtype=np.float32))
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app)
    bad_turn = {"speaker": "A", "text": "Hello", "effects": [{"type": "reverb", "mix": "heavy"}]}

    created = client.post("/api/conversations", json={"name": "Demo", "turns": [bad_turn]})
    conversation = client.post("/api/conversations", json={"name": "Demo", "turns": []}).json()
    added = client.post(f"/api/conversations/{conversation['id']}/turns", json=bad_turn)

    assert created.status_code == 400
    assert created.json()["error"]["message"] == "Turn 1: reverb mix must be a number from 0 to 1"
    assert added.status_code == 400
    assert client.get("/api/conversations").json()["total"] == 1
    assert client.get(f"/api/conversations/{conversation['id']}").json()["turns"] == []


def test_invalid_stored_preset_or_turn_effect_is_a_client_error_at_render(tmp_path, monkeypatch):
    # Rows written before effects were validated (or through the managers) must not become a 500.
    from src.profiles import ProfileManager

    _reset_db(tmp_path)
    synthesize = MagicMock(return_value=np.zeros(240, dtype=np.float32))
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    preset = ProfileManager().create("Echo", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None,
                                     [{"type": "pitch", "semitones": "up"}])
    manager = main_module.conversation_manager
    by_preset = manager.create("Demo", [{"speaker": "A", "text": "Hi", "profile_id": preset["id"]}])
    by_turn = manager.create("Demo", [{"speaker": "A", "text": "Hi"},
                                      {"speaker": "B", "text": "Yo", "effects": [{"type": "reverb", "mix": 2}]}])
    client = TestClient(app, raise_server_exceptions=False)

    preset_render = client.post(f"/api/conversations/{by_preset['id']}/render", json={})
    turn_render = client.post(f"/api/conversations/{by_turn['id']}/render", json={})

    assert preset_render.status_code == 400
    assert preset_render.json()["error"]["message"] == "Turn 1: pitch semitones must be a number from -24 to 24"
    assert turn_render.status_code == 400
    assert turn_render.json()["error"]["message"] == "Turn 2: reverb mix must be a number from 0 to 1"
    synthesize.assert_not_called()


def test_render_without_any_tts_provider_reports_provider_missing(tmp_path, monkeypatch):
    from src.tts.router import TTSRouter

    _reset_db(tmp_path)
    with patch("src.tts.router._discover_backends", return_value={}):
        empty_router = TTSRouter(device="cpu", external_providers="")
    monkeypatch.setattr(main_module, "tts_router", empty_router)
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", main_module._synthesize_array)
    client = TestClient(app, raise_server_exceptions=False)
    conversation = client.post(
        "/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "Hello"}]}
    ).json()

    response = client.post(f"/api/conversations/{conversation['id']}/render", json={})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_missing"


def test_render_reports_an_offline_provider_with_its_status(tmp_path, monkeypatch):
    from src.tts.external import ExternalProviderError

    _reset_db(tmp_path)

    def offline(**_kwargs):
        raise ExternalProviderError("Worker offline", status_code=503)

    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", offline)
    client = TestClient(app, raise_server_exceptions=False)
    conversation = client.post(
        "/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "Hello"}]}
    ).json()

    response = client.post(f"/api/conversations/{conversation['id']}/render", json={})

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "Worker offline"


def test_conversation_speaks_the_pronunciation_dictionary(tmp_path):
    # TTS_PRONUNCIATION_DICT is server-wide: /v1/audio/speech, Live Reader, Reader and Wyoming use it.
    import json

    from src.conversation import ConversationManager
    from src.pronunciation.dictionary import PronunciationDictionary

    _reset_db(tmp_path)
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps({"Nguyen": "Win"}), encoding="utf-8")
    spoken: list[str] = []

    def synthesize(**kwargs):
        spoken.append(kwargs["text"])
        return np.zeros(240, dtype=np.float32)

    manager = ConversationManager(synthesize_fn=synthesize, pronounce_fn=PronunciationDictionary(str(path)).apply)
    conversation = manager.create("Demo", [{"speaker": "A", "text": "Hello Nguyen"}])
    manager.render(conversation["id"], format="wav")
    assert spoken == ["Hello Win"]


def test_dictionary_expansions_answer_only_to_the_models_own_limit(tmp_path, monkeypatch):
    # /v1/audio/speech holds TTS_MAX_INPUT_LENGTH to the client's text: the operator's dictionary
    # is not part of it, so its expansions answer only to the model's own limit, which
    # synthesize_array checks. A turn typed within the limit must not be refused for them.
    import json

    from src.conversation import ConversationManager
    from src.pronunciation.dictionary import PronunciationDictionary

    _reset_db(tmp_path)
    path = tmp_path / "dictionary.json"
    expansion = "National Aeronautics and Space Administration"
    path.write_text(json.dumps({"NASA": expansion}), encoding="utf-8")
    spoken: list[str] = []

    def synthesize(**kwargs):
        spoken.append(kwargs["text"])
        return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(main_module.settings, "tts_max_input_length", 20)
    manager = ConversationManager(synthesize_fn=synthesize, pronounce_fn=PronunciationDictionary(str(path)).apply)
    conversation = manager.create("Demo", [{"speaker": "A", "text": "Hello NASA"}])

    manager.render(conversation["id"], format="wav")

    assert spoken == [f"Hello {expansion}"]


def test_rendering_through_the_api_speaks_the_servers_pronunciation_dictionary(tmp_path, monkeypatch):
    # The app's manager was built without the dictionary, so studio renders spoke "Nguyen".
    # It reads the dictionary when it speaks, so the one swapped in here takes effect.
    import json

    from src.pronunciation.dictionary import PronunciationDictionary

    _reset_db(tmp_path)
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps({"Nguyen": "Win"}), encoding="utf-8")
    monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(str(path)))
    synthesize = MagicMock(return_value=np.zeros(240, dtype=np.float32))
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app)
    conversation = client.post(
        "/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "Hello Nguyen"}]}
    ).json()

    rendered = client.post(f"/api/conversations/{conversation['id']}/render", json={"format": "wav"})

    assert rendered.status_code == 200, rendered.text
    assert [call.kwargs["text"] for call in synthesize.call_args_list] == ["Hello Win"]
    # What the conversation stores is still the text the user wrote.
    assert client.get(f"/api/conversations/{conversation['id']}").json()["turns"][0]["text"] == "Hello Nguyen"


def test_a_turn_the_pronunciation_dictionary_empties_is_refused_before_any_synthesis(tmp_path, monkeypatch):
    # A turn the dictionary removes entirely went to the
    # model as "", which a GPU worker answers with "rejected the request: text: String should have at least 1
    # character" after the turns before it were synthesized. Like /v1/audio/speech, the render answers 400 "Input
    # text is empty", naming the turn, before synthesizing any.
    import json

    from src.pronunciation.dictionary import PronunciationDictionary

    _reset_db(tmp_path)
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps({"[Illustration]": ""}), encoding="utf-8")
    monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(str(path)))
    synthesize = MagicMock(return_value=np.zeros(240, dtype=np.float32))
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app)
    conversation = client.post("/api/conversations", json={"name": "Demo", "turns": [
        {"speaker": "A", "text": "Hello there."}, {"speaker": "B", "text": "[Illustration]"},
    ]}).json()

    rendered = client.post(f"/api/conversations/{conversation['id']}/render", json={"format": "wav"})

    assert rendered.status_code == 400, rendered.text
    assert rendered.json()["error"]["message"] == "Turn 2: Input text is empty"
    assert synthesize.call_count == 0


def test_turn_longer_than_the_input_limit_is_never_synthesized(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(main_module.settings, "tts_max_input_length", 50)
    synthesize = MagicMock(return_value=np.zeros(240, dtype=np.float32))
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app)

    created = client.post("/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "x" * 51}]})
    conversation = client.post(
        "/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "x" * 50}]}
    ).json()
    added = client.post(f"/api/conversations/{conversation['id']}/turns", json={"speaker": "B", "text": "y" * 51})

    assert created.status_code == 400
    assert created.json()["error"]["message"] == "Turn 1 is too long. Max: 50 characters"
    assert added.status_code == 400
    assert len(client.get(f"/api/conversations/{conversation['id']}").json()["turns"]) == 1

    # A turn saved under a higher limit is refused at render, before any synthesis.
    monkeypatch.setattr(main_module.settings, "tts_max_input_length", 40)
    rendered = client.post(f"/api/conversations/{conversation['id']}/render", json={})

    assert rendered.status_code == 400
    assert rendered.json()["error"]["message"] == "Turn 1 is too long. Max: 40 characters"
    synthesize.assert_not_called()


def test_effects_disabled_setting_also_disables_conversation_effects(tmp_path, monkeypatch):
    from scipy.io import wavfile

    _reset_db(tmp_path)
    monkeypatch.setattr(main_module.settings, "os_effects_enabled", False)
    monkeypatch.setattr(
        main_module.conversation_manager, "synthesize_fn", lambda **_kwargs: np.full(2400, 0.5, dtype=np.float32)
    )
    client = TestClient(app)
    conversation = client.post(
        "/api/conversations",
        json={"name": "Demo", "turns": [{"speaker": "A", "text": "Hello", "effects": [{"type": "robot"}]}]},
    ).json()

    response = client.post(f"/api/conversations/{conversation['id']}/render", json={"format": "wav"})

    assert response.status_code == 200
    _, rendered = wavfile.read(response.json()["output_path"])
    assert np.all(rendered == 16383), rendered[:8]


@pytest.mark.parametrize("edit", ["add", "delete"])
def test_editing_turns_after_a_render_retires_the_stale_render(tmp_path, monkeypatch, edit):
    # GET /audio serves render_output_path; after an edit it must not serve audio without that edit.
    _reset_db(tmp_path)
    monkeypatch.setattr(
        main_module.conversation_manager, "synthesize_fn", lambda **_kwargs: np.zeros(240, dtype=np.float32)
    )
    client = TestClient(app)
    conversation = client.post(
        "/api/conversations",
        json={"name": "Demo", "turns": [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]},
    ).json()
    assert client.post(f"/api/conversations/{conversation['id']}/render", json={"format": "wav"}).status_code == 200
    assert client.get(f"/api/conversations/{conversation['id']}/audio").status_code == 200

    if edit == "add":
        changed = client.post(f"/api/conversations/{conversation['id']}/turns", json={"speaker": "C", "text": "three"})
        assert changed.status_code == 201
    else:
        turn_id = conversation["turns"][1]["id"]
        assert client.delete(f"/api/conversations/{conversation['id']}/turns/{turn_id}").status_code == 204

    assert client.get(f"/api/conversations/{conversation['id']}").json()["render_output_path"] is None
    stale = client.get(f"/api/conversations/{conversation['id']}/audio")
    assert stale.status_code == 404
    assert stale.json()["error"]["message"] == "Conversation has no rendered output"


def test_concurrent_preset_conflict_keeps_a_finished_renders_turn_updates(tmp_path, monkeypatch):
    # The render writes its turn rows and its render row in one transaction on the shared
    # connection; a duplicate preset name saved meanwhile must roll back only its own insert.
    import threading

    from src import conversation as conversation_module
    from src.profiles import ProfileManager

    _reset_db(tmp_path)
    profiles = ProfileManager()
    profiles.create("Taken", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None, [])
    manager, cid = _two_turn_manager(lambda **_kwargs: np.zeros(240, dtype=np.float32))
    between_statements, resume = threading.Event(), threading.Event()
    real_now = conversation_module._now_iso

    def now_between_updates():  # the render calls it between its turn UPDATEs and its render UPDATE
        if threading.current_thread().name == "render":
            between_statements.set()
            resume.wait(timeout=10)
        return real_now()

    monkeypatch.setattr(conversation_module, "_now_iso", now_between_updates)
    outcome: dict = {}

    def render():
        outcome["result"] = manager.render(cid, format="wav", sample_rate=24000)

    def save_duplicate_preset():
        try:
            profiles.create("Taken", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None, [])
        except ValueError as exc:
            outcome["conflict"] = exc

    worker = threading.Thread(target=render, name="render")
    conflict = threading.Thread(target=save_duplicate_preset, name="request")
    worker.start()
    try:
        assert between_statements.wait(timeout=5)
        conflict.start()
        conflict.join(timeout=0.5)  # serialized writes wait for the render's transaction
    finally:
        resume.set()
        worker.join(timeout=10)
        if conflict.ident is not None:
            conflict.join(timeout=10)

    assert "result" in outcome and "conflict" in outcome, outcome
    turns = manager.get(cid)["turns"]
    assert all(turn["audio_path"] for turn in turns)
    assert all(turn["duration_ms"] == 10 for turn in turns)


@pytest.mark.parametrize("edit", ["add", "delete"])
def test_turn_edit_during_a_render_keeps_that_render_from_being_served(tmp_path, monkeypatch, edit):
    # Add Turn and Delete stay usable while Render All synthesizes. The finished render lacks
    # that edit, so it must not come back as what GET /audio serves.
    _reset_db(tmp_path)
    manager = main_module.conversation_manager
    client = TestClient(app)
    conversation = client.post(
        "/api/conversations",
        json={"name": "Demo", "turns": [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]},
    ).json()
    cid = conversation["id"]
    calls = {"n": 0}

    def synthesize(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            if edit == "add":
                manager.add_turn(cid, "C", "three")
            else:
                assert manager.delete_turn(cid, conversation["turns"][1]["id"])
        return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(manager, "synthesize_fn", synthesize)

    stale = client.post(f"/api/conversations/{cid}/render", json={"format": "wav"})

    assert stale.status_code == 409
    assert stale.json()["error"]["message"] == "Conversation turns changed during the render. Render it again."
    assert client.get(f"/api/conversations/{cid}").json()["render_output_path"] is None
    assert client.get(f"/api/conversations/{cid}/audio").status_code == 404
    assert not (Path(main_module.settings.os_conversations_dir) / cid / "render.wav").exists()

    fresh = client.post(f"/api/conversations/{cid}/render", json={"format": "wav"})
    assert fresh.status_code == 200
    assert fresh.json()["turn_count"] == (3 if edit == "add" else 1)
    assert client.get(f"/api/conversations/{cid}/audio").status_code == 200


def test_conversation_deleted_before_its_render_is_saved_is_not_found(tmp_path, monkeypatch):
    from src import conversation as conversation_module

    _reset_db(tmp_path)
    manager = main_module.conversation_manager
    monkeypatch.setattr(manager, "synthesize_fn", lambda **_kwargs: np.zeros(240, dtype=np.float32))
    client = TestClient(app)
    cid = client.post("/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "one"}]}).json()["id"]
    real_encode = conversation_module.encode_audio

    def encode_then_delete(*args, **kwargs):
        encoded = real_encode(*args, **kwargs)
        assert manager.delete(cid)
        return encoded

    monkeypatch.setattr(conversation_module, "encode_audio", encode_then_delete)

    response = client.post(f"/api/conversations/{cid}/render", json={"format": "wav"})

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Conversation not found"
    assert not (Path(main_module.settings.os_conversations_dir) / cid).exists()


def test_render_maps_a_missing_named_voice_to_not_found_but_not_any_key_error(tmp_path, monkeypatch):
    from src.voice_identities import NamedVoiceNotFoundError, resolve_named_voice

    _reset_db(tmp_path)
    with pytest.raises(NamedVoiceNotFoundError):  # what a turn naming a deleted voice raises
        resolve_named_voice("kokoro", f"voice:{uuid4()}")
    errors = iter([NamedVoiceNotFoundError("Named voice not found"), KeyError("speaker")])

    def synthesize(**_kwargs):
        raise next(errors)

    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app, raise_server_exceptions=False)
    cid = client.post("/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "one"}]}).json()["id"]

    missing_voice = client.post(f"/api/conversations/{cid}/render", json={})
    programming_error = client.post(f"/api/conversations/{cid}/render", json={})

    assert missing_voice.status_code == 404
    assert missing_voice.json()["error"]["message"] == "Named voice not found"
    assert programming_error.status_code == 500


# Ported from the old web UI's flow tests: the answers Studio's
# Render, Download and conversation list rely on.


def test_an_unknown_conversation_is_not_found_for_open_and_delete(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)
    missing = f"/api/conversations/{uuid4()}"

    for response in (client.get(missing), client.delete(missing)):
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "Conversation not found"


def test_rendering_a_conversation_without_turns_is_a_client_error(tmp_path):
    # Studio reports the reason instead of rendering forever.
    _reset_db(tmp_path)
    client = TestClient(app)
    empty = client.post("/api/conversations", json={"name": "Empty", "turns": []}).json()

    failed = client.post(f"/api/conversations/{empty['id']}/render", json={"format": "mp3", "sample_rate": 24000})

    assert failed.status_code == 400
    assert failed.json()["error"]["message"]


def test_the_audio_route_serves_the_latest_render_in_its_format(tmp_path, monkeypatch):
    # Download names the file for the render it gets: a WAV render followed by an MP3 one serves MP3.
    import src.conversation as conversation_module

    _reset_db(tmp_path)
    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", lambda **_kw: np.zeros(2400, dtype=np.float32))
    real_encode = conversation_module.encode_audio
    monkeypatch.setattr(   # no ffmpeg here: stand in for the MP3 encoder only
        conversation_module, "encode_audio",
        lambda chunks, fmt, sample_rate: b"ID3" + bytes(64) if fmt == "mp3" else real_encode(chunks, fmt=fmt, sample_rate=sample_rate),
    )
    client = TestClient(app)
    cid = client.post("/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "Hello"}]}).json()["id"]
    render = f"/api/conversations/{cid}/render"

    assert client.post(render, json={"format": "wav"}).status_code == 200
    wav = client.get(f"/api/conversations/{cid}/audio")
    assert client.post(render, json={"format": "mp3"}).status_code == 200
    mp3 = client.get(f"/api/conversations/{cid}/audio")

    assert wav.status_code == 200 and wav.headers["content-type"].startswith("audio/wav") and wav.content[:4] == b"RIFF"
    assert mp3.status_code == 200 and mp3.headers["content-type"].startswith("audio/mpeg") and mp3.content[:3] == b"ID3"


def test_a_render_after_adding_a_turn_speaks_the_current_turns(tmp_path, monkeypatch):
    # Download after an edit renders again (the edit retired the old render), and that render has the new turn.
    _reset_db(tmp_path)
    spoken: list[str] = []

    def synthesize(**kwargs):
        spoken.append(kwargs["text"])
        return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app)
    cid = client.post("/api/conversations", json={"name": "Chat", "turns": [{"speaker": "Ann", "text": "Hello."}]}).json()["id"]
    render = f"/api/conversations/{cid}/render"
    assert client.post(render, json={"format": "wav"}).status_code == 200
    added = client.post(f"/api/conversations/{cid}/turns", json={"speaker": "Ben", "text": "A line added after rendering."})
    assert added.status_code == 201
    spoken.clear()

    rerendered = client.post(render, json={"format": "wav"})

    assert rerendered.status_code == 200 and rerendered.json()["output_path"].endswith(".wav")
    assert spoken == ["Hello.", "A line added after rendering."]
    conversation = client.get(f"/api/conversations/{cid}").json()
    assert len(conversation["turns"]) == 2 and conversation["render_output_path"].endswith(".wav")
    assert client.get(f"/api/conversations/{cid}/audio").status_code == 200


def test_conversation_deleted_while_a_turn_synthesizes_is_not_found(tmp_path, monkeypatch):
    # The delete removes the folder the next turn's audio goes to: that was a FileNotFoundError 500.
    _reset_db(tmp_path)
    manager = main_module.conversation_manager
    client = TestClient(app, raise_server_exceptions=False)
    cid = client.post(
        "/api/conversations",
        json={"name": "Demo", "turns": [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]},
    ).json()["id"]
    deleted: list[bool] = []

    def synthesize_then_delete(**_kwargs):
        if not deleted:
            deleted.append(manager.delete(cid))
        return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(manager, "synthesize_fn", synthesize_then_delete)

    response = client.post(f"/api/conversations/{cid}/render", json={"format": "wav"})

    assert deleted == [True]
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Conversation not found"
    assert not (Path(main_module.settings.os_conversations_dir) / cid).exists()


def test_conversation_deleted_before_its_render_makes_a_folder_leaves_no_files(tmp_path, monkeypatch):
    # Deleted before the render made its folder: the render made it again and left turn audio there.
    _reset_db(tmp_path)
    manager = main_module.conversation_manager
    monkeypatch.setattr(manager, "synthesize_fn", lambda **_kwargs: np.zeros(240, dtype=np.float32))
    client = TestClient(app)
    cid = client.post("/api/conversations", json={"name": "Demo", "turns": [{"speaker": "A", "text": "one"}]}).json()["id"]

    def pronounce_then_delete(text):
        manager.delete(cid)
        return text

    monkeypatch.setattr(manager, "pronounce_fn", pronounce_then_delete)

    response = client.post(f"/api/conversations/{cid}/render", json={"format": "wav"})

    assert response.status_code == 404
    assert not (Path(main_module.settings.os_conversations_dir) / cid).exists()


def _render_one_turn_with_preset(monkeypatch, client, profile_id) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def synthesize(**kwargs):
        calls.append((kwargs["model"], kwargs["voice"]))
        return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    cid = client.post(
        "/api/conversations", json={"name": "C", "turns": [{"speaker": "A", "text": "Hi", "profile_id": profile_id}]}
    ).json()["id"]
    response = client.post(f"/api/conversations/{cid}/render", json={"format": "wav"})
    assert response.status_code == 200, response.text
    return calls


def test_a_preset_without_a_model_is_spoken_by_its_backend_not_tts_model(tmp_path, monkeypatch):
    # A preset's model is "model or backend" everywhere else (resolve, Speak); TTS_MODEL is no stand-in.
    _reset_db(tmp_path)
    monkeypatch.setattr(main_module.settings, "tts_model", "piper/en_US-lessac-medium")
    client = TestClient(app)
    preset = client.post("/api/profiles", json={"name": f"Heart {uuid4()}", "backend": "kokoro", "voice": "af_heart"}).json()
    assert preset["model"] is None

    assert _render_one_turn_with_preset(monkeypatch, client, preset["id"]) == [("kokoro", "af_heart")]


def test_a_named_voice_preset_without_a_model_speaks_its_backends_realization_with_it(tmp_path, monkeypatch):
    from src.voice_identities import VoiceIdentityManager

    _reset_db(tmp_path)
    monkeypatch.setattr(main_module.settings, "tts_model", "piper/en_US-lessac-medium")
    client = TestClient(app)
    identity_id = client.post("/api/voices/identities", json={"name": "Narrator"}).json()["id"]
    VoiceIdentityManager().add_realization(identity_id, model="kokoro", voice="af_bella")
    preset = client.post(
        "/api/profiles",
        json={"name": f"Narrated {uuid4()}", "backend": "kokoro", "voice": "x", "voice_identity_id": identity_id},
    ).json()
    assert preset["voice"] == "af_bella"   # its kokoro realization

    assert _render_one_turn_with_preset(monkeypatch, client, preset["id"]) == [("kokoro", "af_bella")]


def test_renders_wait_in_their_own_pool_not_the_shared_default_executor(tmp_path, monkeypatch):
    # A render waits for each turn's provider; a queue of renders behind a busy provider must
    # not hold the default executor every other request and to_thread call shares.
    import threading

    _reset_db(tmp_path)
    threads: list[str] = []

    def synthesize(**_kwargs):
        threads.append(threading.current_thread().name)
        return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(main_module.conversation_manager, "synthesize_fn", synthesize)
    client = TestClient(app)
    cid = client.post("/api/conversations", json={"name": "Chat", "turns": [{"speaker": "Ann", "text": "Hello."}]}).json()["id"]

    assert client.post(f"/api/conversations/{cid}/render", json={"format": "wav"}).status_code == 200
    assert len(threads) == 1 and threads[0].startswith("conversation-render")


@pytest.mark.parametrize("parameter", ["limit", "offset"])
def test_conversation_paging_past_sqlites_integer_range_is_a_client_error(tmp_path, parameter):
    # SQLite binds at most 2**63 - 1; a larger value was an OverflowError 500.
    _reset_db(tmp_path)
    client = TestClient(app)

    assert client.get("/api/conversations", params={parameter: 2**63}).status_code == 422
    assert client.get("/api/conversations", params={parameter: 2**63 - 1}).status_code == 200


# Half of an emoji's UTF-16 pair (the JSON escape "\ud83d") in a
# conversation's name or a turn's speaker or text reached studio.db, which can't encode it: a 500.
_LONE_SURROGATE = "\\ud83d"  # the JSON escape, which json.loads decodes to a lone surrogate


@pytest.mark.parametrize("field, body", [
    ("name", '{"name": "Chat %s", "turns": []}' % _LONE_SURROGATE),
    ("speaker", '{"name": "Chat", "turns": [{"speaker": "Ann %s", "text": "Hi"}]}' % _LONE_SURROGATE),
    ("text", '{"name": "Chat", "turns": [{"speaker": "Ann", "text": "Hi %s"}]}' % _LONE_SURROGATE),
])
def test_a_lone_surrogate_in_a_new_conversation_is_refused_naming_the_field(tmp_path, field, body):
    _reset_db(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/conversations", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422, response.text
    assert field in response.text and "surrogate" in response.text
    assert client.get("/api/conversations").json()["total"] == 0


@pytest.mark.parametrize("field, body", [
    ("speaker", '{"speaker": "Ann %s", "text": "Hi"}' % _LONE_SURROGATE),
    ("text", '{"speaker": "Ann", "text": "Hi %s"}' % _LONE_SURROGATE),
])
def test_a_lone_surrogate_in_an_added_turn_is_refused_naming_the_field(tmp_path, field, body):
    _reset_db(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    cid = client.post("/api/conversations", json={"name": "Chat", "turns": []}).json()["id"]

    response = client.post(f"/api/conversations/{cid}/turns", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422, response.text
    assert field in response.text and "surrogate" in response.text
    assert client.get(f"/api/conversations/{cid}").json()["turns"] == []
    # A whole emoji (both halves of the pair) is text like any other.
    whole = client.post(f"/api/conversations/{cid}/turns", json={"speaker": "Ann", "text": "Hi \U0001F600"})
    assert whole.status_code == 201, whole.text
