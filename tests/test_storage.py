"""studio.db's one shared connection, used at once by many worker threads."""

import sqlite3
import threading
import time
from contextlib import contextmanager

import pytest

from src import storage
from src.composer import MultiTrackComposer
from src.conversation import ConversationManager
from src.history import HistoryManager
from src.profiles import ProfileManager
from src.reader import store as reader_store
from src.reader.document import Block, Chapter, Document
from src.voice_identities import VoiceIdentityManager

# How long a test lets another thread try to get past a held connection. Old code gets past
# in a few milliseconds; correct code never does, so each such test costs this long once.
PASS_THROUGH_S = 0.5
JOIN_S = 10
# sqlite3 waits 5 s for another connection's write lock by default (Python docs,
# sqlite3.connect). A large Reader import holds studio.db's lock longer than that.
READER_IMPORT_HOLD_S = 6.0


@pytest.fixture
def studio_db(tmp_path, monkeypatch):
    """Point the shared connection at a fresh file database opened as production opens it."""

    def open_db():
        connection = sqlite3.connect(str(tmp_path / "studio.db"), check_same_thread=False,
                                     timeout=storage.BUSY_TIMEOUT_S)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        monkeypatch.setattr(storage, "_conn", connection)
        storage.init_db()
        opened.append(connection)
        return connection

    opened: list[sqlite3.Connection] = []
    # The Reader opens its own connection to the same file.
    monkeypatch.setattr(storage.settings, "os_studio_db_path", str(tmp_path / "studio.db"))
    monkeypatch.setattr(storage.settings, "os_history_max_entries", 1000)
    yield open_db
    reader_store.close()
    for connection in opened:
        connection.close()


class _Worker(threading.Thread):
    """Runs one call in its own thread and keeps its result or exception for the test."""

    def __init__(self, target, stop: threading.Event | None = None, name: str | None = None):
        super().__init__(daemon=True, name=name)
        self._target_call = target
        self._stop_others = stop
        self.result = None
        self.error: Exception | None = None

    def run(self):
        try:
            self.result = self._target_call()
        except Exception as exc:  # noqa: BLE001 - reported by _finish()
            self.error = exc
            if self._stop_others is not None:
                self._stop_others.set()


def _start(target, stop: threading.Event | None = None, name: str | None = None) -> _Worker:
    worker = _Worker(target, stop, name)
    worker.start()
    return worker


def _finish(*workers: _Worker) -> list:
    """Join every worker before reporting any failure: none may outlive the test's database."""
    for worker in workers:
        worker.join(JOIN_S)
    assert not any(worker.is_alive() for worker in workers), "a worker is stuck on studio.db"
    for worker in workers:
        if worker.error is not None:
            raise worker.error
    return [worker.result for worker in workers]


def _join_together(workers: list[_Worker], seconds: float) -> None:
    """Give every worker the same few seconds to finish, without waiting that long for each."""
    deadline = time.monotonic() + seconds
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))


def _pause_after(monkeypatch, owner, name: str, thread_name: str) -> tuple[threading.Event, threading.Event]:
    """Hold the thread named thread_name just after owner.name first returns in it."""
    paused, resume = threading.Event(), threading.Event()
    original = getattr(owner, name)

    def pausing(*args, **kwargs):
        result = original(*args, **kwargs)
        if threading.current_thread().name == thread_name and not paused.is_set():
            paused.set()
            resume.wait(PASS_THROUGH_S)
        return result

    monkeypatch.setattr(owner, name, pausing)
    return paused, resume


def test_a_read_in_another_thread_never_sees_an_open_transaction(studio_db):
    studio_db()
    in_transaction = threading.Event()
    finish_transaction = threading.Event()

    def write_then_roll_back():
        try:
            with storage.transaction() as db:
                db.execute("INSERT INTO voice_identities VALUES ('open-id', 'Uncommitted', '2026-01-01')")
                db.execute(
                    "INSERT INTO profiles (id, name, backend, voice, created_at, updated_at) "
                    "VALUES ('open-preset', 'Uncommitted', 'kokoro', 'af_heart', '2026-01-01', '2026-01-01')"
                )
                db.execute(
                    "INSERT INTO history_entries (id, type, created_at, full_text) "
                    "VALUES ('open-entry', 'tts', '2026-01-01', 'Uncommitted')"
                )
                in_transaction.set()
                finish_transaction.wait(PASS_THROUGH_S)
                raise RuntimeError("roll back")
        except RuntimeError:
            return "rolled back"

    writer = _start(write_then_roll_back)
    assert in_transaction.wait(JOIN_S)
    # Each read in its own thread, so each meets the open transaction: single statements
    # (presets, a bare execute) and reads of several statements (voices, history).
    readers = [
        _start(lambda: [voice["name"] for voice in VoiceIdentityManager().list_all()]),
        _start(lambda: [preset["name"] for preset in ProfileManager().list_all()]),
        _start(lambda: HistoryManager().list_entries()["total"]),
        _start(lambda: storage.get_db().execute("SELECT COUNT(*) FROM voice_identities").fetchone()[0]),
    ]
    _join_together(readers, PASS_THROUGH_S)
    finish_transaction.set()

    assert _finish(writer, *readers) == ["rolled back", [], [], 0, 0]


def test_named_voice_list_never_sees_a_voice_added_between_its_reads(studio_db, monkeypatch):
    # GET /api/voices/identities answered 500 (KeyError) when a realization of a voice it had
    # not listed yet showed up in its second query. Each statement holds the connection only
    # while it runs, so list_all is paused between its two statements, where only its
    # snapshot() keeps another thread from writing.
    studio_db()
    voices = VoiceIdentityManager()
    existing = voices.create("Existing")
    voices.add_realization(existing["id"], model="kokoro", voice="af_heart")
    paused, resume = threading.Event(), threading.Event()
    run_statement = storage.StudioConnection.execute

    def execute(self, sql, parameters=()):
        result = run_statement(self, sql, parameters)
        if threading.current_thread().name == "lister" and "FROM voice_identities v" in sql:
            paused.set()
            resume.wait(PASS_THROUGH_S)
        return result

    monkeypatch.setattr(storage.StudioConnection, "execute", execute)
    lister = _start(voices.list_all, name="lister")
    assert paused.wait(JOIN_S)

    def add_voice():
        added = voices.create("Added")
        voices.add_realization(added["id"], model="kokoro", voice="af_sky")

    adder = _start(add_voice)
    adder.join(PASS_THROUGH_S)
    resume.set()

    listed, _ = _finish(lister, adder)
    assert [voice["name"] for voice in listed] == ["Existing"]
    for voice in listed:
        assert [r["voice_identity_id"] for r in voice["realizations"]] == [voice["id"]]
    assert [voice["name"] for voice in voices.list_all()] == ["Added", "Existing"]


def test_concurrent_studio_reads_and_writes_keep_every_row_where_it_belongs(studio_db):
    studio_db()
    voices, presets, history = VoiceIdentityManager(), ProfileManager(), HistoryManager()
    conversations, composer = ConversationManager(), MultiTrackComposer()
    for n in range(4):
        voice = voices.create(f"Voice {n}")
        for model in ("kokoro", "piper"):
            voices.add_realization(voice["id"], model=model, voice=f"{model}-{n}")
        presets.create(f"Preset {n}", "kokoro", "kokoro", f"af_{n}", 1.0, "mp3", None, None, [])
        history.log_tts("kokoro", f"af_{n}", 1.0, "mp3", f"Entry {n}", None, 10)
        conversations.create(f"Talk {n}", [{"speaker": "A", "text": f"Line {n}"}])
    deadline = time.monotonic() + 1.0
    stop = threading.Event()

    def running():
        return time.monotonic() < deadline and not stop.is_set()

    def read():
        while running():
            for voice in voices.list_all():
                assert voice["name"].startswith(("Voice ", "Extra "))
                assert {r["voice_identity_id"] for r in voice["realizations"]} <= {voice["id"]}
                assert voice["voice"] == "voice:" + voice["id"]
            for preset in presets.list_all():
                assert preset["name"].startswith(("Preset ", "Extra ")) and preset["voice"]
            listed = history.list_entries(limit=100)
            assert len(listed["items"]) == listed["total"] >= 4
            assert all(item["full_text"].startswith(("Entry ", "Extra ")) for item in listed["items"])
            talks = conversations.list_all()
            assert len(talks["items"]) == talks["total"] >= 4
            for talk in talks["items"]:
                full = conversations.get(talk["id"]) or {"turns": []}  # None: deleted since
                assert all(turn["conversation_id"] == talk["id"] for turn in full["turns"])
            assert composer.list_renders()["total"] == 0
        return "read"

    def write():
        n = 0
        while running():
            n += 1
            voice = voices.create(f"Extra {n}")
            voices.add_realization(voice["id"], model="kokoro", voice="extra")
            preset = presets.create(f"Extra {n}", "kokoro", "kokoro", "af_extra", 1.0, "mp3", None, None, [])
            entry = history.log_tts("kokoro", "af_extra", 1.0, "mp3", f"Extra {n}", None, 10)
            talk = conversations.create(f"Extra {n}", [{"speaker": "B", "text": "Extra"}])
            conversations.add_turn(talk["id"], "C", "Extra turn")
            voices.delete(voice["id"])
            presets.delete(preset["id"])
            history.delete_entry(entry)
            conversations.delete(talk["id"])
        return "wrote"

    workers = [_start(read, stop) for _ in range(6)] + [_start(write, stop)]
    assert _finish(*workers) == ["read"] * 6 + ["wrote"]


def test_a_model_version_added_to_a_voice_being_deleted_is_not_called_a_duplicate(studio_db, monkeypatch):
    # add_realization checked that the voice exists before its INSERT's transaction. A delete
    # in between failed the INSERT's foreign key, reported as 409 "already has a realization".
    studio_db()
    voices = VoiceIdentityManager()
    voice = voices.create("Leaving")
    paused, resume = _pause_after(monkeypatch, VoiceIdentityManager, "get", "adder")
    adder = _start(lambda: voices.add_realization(voice["id"], model="kokoro", voice="af_sky"), name="adder")
    assert paused.wait(JOIN_S)
    deleter = _start(lambda: voices.delete(voice["id"]))
    deleter.join(PASS_THROUGH_S)
    resume.set()

    added, _ = _finish(adder, deleter)
    assert (added["voice"], added["name"]) == ("af_sky", "Leaving")
    assert voices.list_all() == []


@pytest.mark.parametrize("save", ["create", "update"])
def test_a_preset_saved_with_a_voice_being_deleted_is_not_called_a_duplicate_name(studio_db, monkeypatch, save):
    # Saving a preset resolved its named voice before the write's transaction. A delete in
    # between failed the write's foreign key, reported as 409 "Profile name already exists".
    studio_db()
    voices, presets = VoiceIdentityManager(), ProfileManager()
    voice = voices.create("Leaving")
    voices.add_realization(voice["id"], model="kokoro", voice="af_sky")
    plain = presets.create("Linked", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None, [])
    if save == "create":
        presets.delete(plain["id"])
    paused, resume = _pause_after(monkeypatch, VoiceIdentityManager, "resolve", "saver")

    def save_preset():
        if save == "create":
            return presets.create("Linked", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None, [],
                                  voice_identity_id=voice["id"])
        return presets.update(plain["id"], voice_identity_id=voice["id"])

    saver = _start(save_preset, name="saver")
    assert paused.wait(JOIN_S)
    deleter = _start(lambda: voices.delete(voice["id"]))
    deleter.join(PASS_THROUGH_S)
    resume.set()

    saved, _ = _finish(saver, deleter)
    # Saved linked to the voice, then detached by its delete, keeping the voice it resolved to.
    [kept] = presets.list_all()
    assert (kept["id"], kept["name"]) == (saved["id"], "Linked")
    assert (kept["voice_identity_id"], kept["voice"]) == (None, "af_sky")


PRESET = dict(name="Fresh", backend="kokoro", model="kokoro", voice="af_heart", speed=1.0, format="mp3",
              blend=None, reference_audio_id=None, effects=[])


@pytest.mark.parametrize("column", ["voice", "backend"])
def test_a_preset_save_refused_by_another_constraint_is_not_called_a_duplicate_name(studio_db, column):
    # Create and update turned every IntegrityError into
    # ValueError("Profile name already exists"), the routes' 409, so a NOT NULL failure called a free
    # name (or the preset's own) taken. Only the name's UNIQUE constraint is a duplicate name.
    studio_db()
    presets = ProfileManager()
    refused = f"NOT NULL constraint failed: profiles.{column}"

    with pytest.raises(sqlite3.IntegrityError, match=refused):
        presets.create(**{**PRESET, column: None})
    assert presets.list_all() == []

    kept = presets.create(**PRESET)
    with pytest.raises(sqlite3.IntegrityError, match=refused):
        presets.update(kept["id"], **{column: None})
    assert presets.list_all() == [kept]


def test_a_preset_saved_under_a_taken_name_is_still_a_duplicate_name(studio_db):
    # A control: narrowing the mapping keeps the name's own conflict.
    studio_db()
    presets = ProfileManager()
    presets.create(**PRESET)
    other = presets.create(**{**PRESET, "name": "Other"})

    with pytest.raises(ValueError, match="^Profile name already exists$"):
        presets.create(**PRESET)
    with pytest.raises(ValueError, match="^Profile name already exists$"):
        presets.update(other["id"], name="Fresh")
    assert [preset["name"] for preset in presets.list_all()] == ["Fresh", "Other"]


def test_a_statements_rows_are_fixed_when_it_returns(studio_db):
    # A cursor read its rows lazily, after the connection was let go, so another thread's
    # statements on the shared connection showed up in a result already returned.
    studio_db()
    for name in ("One", "Two"):
        VoiceIdentityManager().create(name)
    result = storage.get_db().execute("SELECT * FROM voice_identities")
    in_transaction, finish_transaction = threading.Event(), threading.Event()

    def write_then_roll_back():
        try:
            with storage.transaction() as db:
                db.execute("INSERT INTO voice_identities VALUES ('open-id', 'Uncommitted', '2026-01-01')")
                in_transaction.set()
                finish_transaction.wait(JOIN_S)
                raise RuntimeError("roll back")
        except RuntimeError:
            return "rolled back"

    writer = _start(write_then_roll_back)
    try:
        assert in_transaction.wait(JOIN_S)
        names = [row["name"] for row in result]
    finally:
        finish_transaction.set()
    assert _finish(writer) == ["rolled back"]
    assert names == ["One", "Two"]


def test_a_write_outside_a_transaction_is_not_undone_by_another_threads_rollback(studio_db):
    studio_db()
    storage.get_db().execute("INSERT INTO voice_identities VALUES ('stray-id', 'Stray', '2026-01-01')")

    def roll_back():
        try:
            with storage.transaction() as db:
                db.execute("INSERT INTO voice_identities VALUES ('other-id', 'Other', '2026-01-01')")
                raise RuntimeError("roll back")
        except RuntimeError:
            return "rolled back"

    assert _finish(_start(roll_back)) == ["rolled back"]
    assert [voice["name"] for voice in VoiceIdentityManager().list_all()] == ["Stray"]


def _try_to_write(path) -> None:
    """Write to studio.db from another connection, refusing to wait for anyone's write lock."""
    other = sqlite3.connect(str(path), timeout=0)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.rollback()
    finally:
        other.close()


def test_a_failed_reader_write_is_rolled_back_not_left_for_the_next_commit(studio_db, tmp_path):
    # The Reader's own connection to studio.db: a write that failed halfway stayed open,
    # holding SQLite's write lock, and the Reader's next commit saved half a document.
    studio_db()
    library = reader_store.ReaderStore()
    reader_store.get_db().execute(
        "CREATE TEMP TRIGGER refuse_blocks BEFORE INSERT ON reader_blocks "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="disk full"):
        library.save_document(Document(title="Half", source_type="text", chapters=[Chapter("One", [Block("Lost.")])]))

    _try_to_write(tmp_path / "studio.db")
    reader_store.get_db().execute("DROP TRIGGER temp.refuse_blocks")
    library.save_document(Document(title="Whole", source_type="text", chapters=[Chapter("One", [Block("Kept.")])]))
    assert [document["title"] for document in library.list_documents()] == ["Whole"]


@pytest.fixture
def production_studio_db(tmp_path, monkeypatch):
    """A fresh studio.db whose shared connection storage.get_db() opens itself, as at startup."""
    monkeypatch.setattr(storage.settings, "os_studio_db_path", str(tmp_path / "studio.db"))
    monkeypatch.setattr(storage.settings, "os_history_max_entries", 1000)
    monkeypatch.setattr(storage, "_conn", None)
    monkeypatch.setattr(storage, "_shared", None)
    reader_store.close()
    storage.init_db()
    yield
    reader_store.close()
    if storage._conn is not None:
        storage._conn.close()


def _save_history_entry():
    entry_id = HistoryManager().log_tts("kokoro", "af_heart", 1.0, "mp3", "Spoken while a book imports.", None, None,
                                        streamed=True)
    row = storage.get_db().execute("SELECT full_text FROM history_entries WHERE id = ?", (entry_id,)).fetchone()
    return row["full_text"] if row else None


def _save_preset():
    return ProfileManager().create("Narrator", "kokoro", "kokoro", "af_heart", 1.0, "mp3", None, None, [])["name"]


@pytest.mark.parametrize("save, saved", [(_save_history_entry, "Spoken while a book imports."),
                                         (_save_preset, "Narrator")], ids=["history", "preset"])
def test_a_save_waits_out_a_long_reader_import(production_studio_db, monkeypatch, save, saved):
    # The Reader writes a whole imported document in one
    # transaction on its own connection, and a large import holds studio.db's write lock for
    # many seconds. The shared connection waited only sqlite3's default 5 s for it, so a save
    # meanwhile failed with "database is locked": a history entry was lost, a preset save 500.
    import_holds_lock, save_started, save_done = threading.Event(), threading.Event(), threading.Event()
    real_writing = reader_store._writing

    @contextmanager
    def long_import_writing():
        # The real Reader transaction, with save_document's inserts in it, kept open as a large
        # import keeps it open: until the save has waited READER_IMPORT_HOLD_S for the lock.
        with real_writing() as db:
            yield db
            assert db.in_transaction, "the import's inserts hold studio.db's write lock"
            import_holds_lock.set()
            assert save_started.wait(JOIN_S)
            save_done.wait(READER_IMPORT_HOLD_S)

    def save_meanwhile():
        assert import_holds_lock.wait(JOIN_S)
        save_started.set()
        try:
            return save()
        finally:
            save_done.set()

    library = reader_store.ReaderStore()
    document = Document(title="Long book", source_type="text",
                        chapters=[Chapter("One", [Block("First paragraph."), Block("Second paragraph.")])])
    monkeypatch.setattr(reader_store, "_writing", long_import_writing)
    saver = _start(save_meanwhile, name="saver")
    try:
        document_id, created = library.save_document(document)
    finally:
        import_holds_lock.set()  # a failed import must not leave the saver waiting
        monkeypatch.setattr(reader_store, "_writing", real_writing)

    assert _finish(saver) == [saved]
    assert created and library.get_document(document_id, include_blocks=False)["title"] == "Long book"
