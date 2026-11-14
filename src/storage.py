"""Shared storage helpers for studio features."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

# How long a studio.db write waits for another connection's write lock before it fails with
# "database is locked". The Reader's own connection (src/reader/store.py) writes a whole
# imported document in one transaction, which holds the lock for seconds; sqlite3's default
# 5 s made other saves fail meanwhile. Both connections wait this long.
BUSY_TIMEOUT_S = 30.0
# The error of a book job that a restart cut off (src/reader/assistant_jobs.py, code "interrupted").
INTERRUPTED_MESSAGE = "The server stopped while the job was running; start it again to finish it"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
# Every thread shares one connection, so they share its open transaction and its prepared
# statements too. Two threads' cursors on it interleave (one gets the other's rows, or rows
# reset under it), and a read from one thread sees, or a commit from it ends, another
# thread's half-written transaction. So every statement, and every transaction() and
# snapshot() block as a whole, runs holding this lock.
_db_lock = threading.RLock()
_transaction_depth = threading.local()
_shared: StudioConnection | None = None


class StudioResult:
    """A statement's rows, fetched in full before the connection was let go, and its counts.

    Reads like the sqlite3 cursor it replaces: fetchone, fetchmany, fetchall and iteration
    consume the rows in order; rowcount, lastrowid and description are the cursor's.
    """

    __slots__ = ("_next", "_rows", "description", "lastrowid", "rowcount")

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        try:
            self._rows = cursor.fetchall()
            self.rowcount = cursor.rowcount
            self.lastrowid = cursor.lastrowid
            self.description = cursor.description
        finally:
            cursor.close()
        self._next = 0

    def fetchone(self) -> Any:
        if self._next >= len(self._rows):
            return None
        self._next += 1
        return self._rows[self._next - 1]

    def fetchmany(self, size: int = 1) -> list:
        rows = self._rows[self._next : self._next + size]
        self._next += len(rows)
        return rows

    def fetchall(self) -> list:
        rows = self._rows[self._next :]
        self._next = len(self._rows)
        return rows

    def __iter__(self) -> StudioResult:
        return self

    def __next__(self) -> Any:
        if self._next >= len(self._rows):
            raise StopIteration
        return self.fetchone()


class StudioConnection:
    """The shared studio.db connection, used by one thread and one statement at a time.

    Outside transaction(), a statement that writes is committed (or, failing, rolled back) at
    once, so the connection is never left mid-transaction for another thread to read or end.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def _run(self, statement: Callable[..., sqlite3.Cursor], *args: Any) -> StudioResult:
        with _db_lock:
            try:
                result = StudioResult(statement(*args))
            except BaseException:
                self._end_stray_transaction(commit=False)
                raise
            self._end_stray_transaction(commit=True)
            return result

    def _end_stray_transaction(self, *, commit: bool) -> None:
        if getattr(_transaction_depth, "value", 0) or not self.connection.in_transaction:
            return
        if not commit:
            self.connection.rollback()
            return
        try:
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def execute(self, sql: str, parameters: Any = ()) -> StudioResult:
        return self._run(self.connection.execute, sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Any) -> StudioResult:
        return self._run(self.connection.executemany, sql, seq_of_parameters)

    def executescript(self, script: str) -> StudioResult:
        return self._run(self.connection.executescript, script)

    def create_function(self, *args: Any, **kwargs: Any) -> None:
        with _db_lock:
            self.connection.create_function(*args, **kwargs)

    def commit(self) -> None:
        with _db_lock:
            self.connection.commit()

    def rollback(self) -> None:
        with _db_lock:
            self.connection.rollback()

    @property
    def in_transaction(self) -> bool:
        return self.connection.in_transaction


def get_db() -> StudioConnection:
    """Return the shared studio.db connection (thread-safe; see StudioConnection)."""
    global _conn, _shared
    with _lock:
        if _conn is None:
            db_path = Path(settings.os_studio_db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=BUSY_TIMEOUT_S)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA foreign_keys=ON")
            _register_functions(_conn)
        # A test may install its own connection in _conn.
        if _shared is None or _shared.connection is not _conn:
            _shared = StudioConnection(_conn)
        return _shared


@contextmanager
def transaction() -> Iterator[StudioConnection]:
    """Run one write transaction on the shared connection, alone; commit it or roll it back.

    Every write to studio.db goes through here. No other thread's statement runs until the
    block ends. A nested call joins the outer transaction.
    """
    with _db_lock:
        db = get_db()
        depth = getattr(_transaction_depth, "value", 0)
        _transaction_depth.value = depth + 1
        try:
            yield db
            if depth == 0:
                db.commit()
        except BaseException:
            if depth == 0:
                db.rollback()
            raise
        finally:
            _transaction_depth.value = depth


@contextmanager
def snapshot() -> Iterator[StudioConnection]:
    """Hold the shared connection for several reads, so together they see one state of studio.db."""
    with _db_lock:
        yield get_db()


def _contains_casefold(text: str | None, needle: str) -> bool:
    return text is not None and needle in text.casefold()


def _register_functions(db: sqlite3.Connection | StudioConnection) -> None:
    # History search folds case in Python: SQLite's LIKE and lower() fold only ASCII letters.
    db.create_function("os_contains_casefold", 2, _contains_casefold, deterministic=True)


def init_db() -> None:
    """Run all CREATE TABLE IF NOT EXISTS DDL. Called at app startup."""
    with transaction() as db:
        _register_functions(db)  # also for a connection a caller installed itself
        db.executescript(SCHEMA_SQL)
        columns = {row[1] for row in db.execute("PRAGMA table_info(profiles)")}
        if "voice_identity_id" not in columns:
            db.execute("ALTER TABLE profiles ADD COLUMN voice_identity_id TEXT "
                       "REFERENCES voice_identities(id)")
        if "instructions" not in columns:
            db.execute("ALTER TABLE profiles ADD COLUMN instructions TEXT")
        if "options_json" not in columns:
            db.execute("ALTER TABLE profiles ADD COLUMN options_json TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS idx_profiles_voice_identity ON profiles(voice_identity_id)")
        from src.voice_identities import migrate_legacy_profiles

        if not db.execute("SELECT 1 FROM schema_migrations WHERE version = 6").fetchone():
            migrate_legacy_profiles(db)
            db.execute("INSERT INTO schema_migrations VALUES (6)")
        # 7: the AI assistant's book jobs (src/reader/assistant_jobs.py). Their tables are in SCHEMA_SQL, so an
        # older studio.db got them above; the version records that it has them.
        if not db.execute("SELECT 1 FROM schema_migrations WHERE version = 7").fetchone():
            db.execute("INSERT INTO schema_migrations VALUES (7)")
        # 8: a book job's block range, for a part of a document sent through POST /api/assistant/ask. A studio.db
        # from before has assistant_jobs without the two columns; its jobs keep NULL.
        job_columns = {row[1] for row in db.execute("PRAGMA table_info(assistant_jobs)")}
        for column in ("first_block", "last_block"):
            if column not in job_columns:
                db.execute(f"ALTER TABLE assistant_jobs ADD COLUMN {column} INTEGER")
        if not db.execute("SELECT 1 FROM schema_migrations WHERE version = 8").fetchone():
            db.execute("INSERT INTO schema_migrations VALUES (8)")
        # A book job still queued or running was cut off by a crash: it fails with the code "interrupted", and
        # starting it again sends only the pieces it hadn't finished. (A clean shutdown does this itself.)
        db.execute(
            "UPDATE assistant_jobs SET status = 'failed', error = 'interrupted', error_message = ?, updated = ? "
            "WHERE status IN ('queued', 'running')",
            (INTERRUPTED_MESSAGE, datetime.now(timezone.utc).isoformat()),
        )


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY);

CREATE TABLE IF NOT EXISTS voice_identities (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_realizations (
  id TEXT PRIMARY KEY,
  voice_identity_id TEXT NOT NULL REFERENCES voice_identities(id),
  model TEXT NOT NULL,
  voice TEXT NOT NULL,
  reference_audio_id TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(voice_identity_id, model)
);

CREATE TABLE IF NOT EXISTS preset_imports (
  name TEXT PRIMARY KEY COLLATE NOCASE
);

CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  backend TEXT NOT NULL,
  model TEXT,
  voice TEXT NOT NULL,
  speed REAL NOT NULL DEFAULT 1.0,
  format TEXT NOT NULL DEFAULT 'mp3',
  blend TEXT,
  reference_audio_id TEXT,
  voice_identity_id TEXT REFERENCES voice_identities(id),
  instructions TEXT,
  effects_json TEXT,
  options_json TEXT,
  is_default INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS history_entries (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK(type IN ('tts','stt')),
  created_at TEXT NOT NULL,
  model TEXT,
  voice TEXT,
  speed REAL,
  format TEXT,
  text_preview TEXT,
  full_text TEXT,
  input_filename TEXT,
  output_path TEXT,
  output_bytes INTEGER,
  streamed INTEGER NOT NULL DEFAULT 0,
  meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_type_created ON history_entries(type, created_at DESC);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  name TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  render_output_path TEXT,
  meta_json TEXT
);

CREATE TABLE IF NOT EXISTS conversation_turns (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  speaker TEXT NOT NULL,
  profile_id TEXT,
  text TEXT NOT NULL,
  audio_path TEXT,
  duration_ms INTEGER,
  effects_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_turns_conv ON conversation_turns(conversation_id, turn_index);

CREATE TABLE IF NOT EXISTS reader_documents (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  author TEXT,
  source_type TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  char_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  meta_json TEXT
);

CREATE TABLE IF NOT EXISTS reader_chapters (
  document_id TEXT NOT NULL REFERENCES reader_documents(id) ON DELETE CASCADE,
  chapter_index INTEGER NOT NULL,
  title TEXT NOT NULL,
  first_block INTEGER NOT NULL,
  block_count INTEGER NOT NULL,
  PRIMARY KEY (document_id, chapter_index)
);

CREATE TABLE IF NOT EXISTS reader_blocks (
  document_id TEXT NOT NULL REFERENCES reader_documents(id) ON DELETE CASCADE,
  block_index INTEGER NOT NULL,
  chapter_index INTEGER NOT NULL,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  sentences_json TEXT NOT NULL,
  PRIMARY KEY (document_id, block_index)
);

CREATE TABLE IF NOT EXISTS reader_progress (
  document_id TEXT PRIMARY KEY REFERENCES reader_documents(id) ON DELETE CASCADE,
  block_index INTEGER NOT NULL,
  offset_ms INTEGER NOT NULL DEFAULT 0,
  model TEXT,
  voice TEXT,
  speed REAL NOT NULL DEFAULT 1.0,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reader_parts (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES reader_documents(id) ON DELETE CASCADE,
  first_block INTEGER NOT NULL,
  last_block INTEGER NOT NULL,
  model TEXT NOT NULL,
  voice TEXT NOT NULL,
  path TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  sentences_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_access REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reader_parts_start ON reader_parts(document_id, model, voice, first_block);

-- The AI assistant's book jobs (src/reader/assistant_jobs.py). Deleting the source document deletes its jobs and
-- their pieces; the result is a document of its own and stays.
CREATE TABLE IF NOT EXISTS assistant_jobs (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES reader_documents(id) ON DELETE CASCADE,
  preset_id TEXT NOT NULL,
  settings_json TEXT NOT NULL,
  first_chapter INTEGER,
  last_chapter INTEGER,
  title TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','done','failed','cancelled')),
  error TEXT,
  error_message TEXT,
  result_document_id TEXT REFERENCES reader_documents(id) ON DELETE SET NULL,
  created TEXT NOT NULL,
  updated TEXT NOT NULL,
  -- A part of the document by block index (migration 8): set instead of the chapters, or neither for all of it.
  first_block INTEGER,
  last_block INTEGER
);
CREATE INDEX IF NOT EXISTS idx_assistant_jobs_document ON assistant_jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_assistant_jobs_result ON assistant_jobs(result_document_id);

-- One saved answer per piece, keyed by a hash of the settings that reach the AI and the piece's text and blocks.
CREATE TABLE IF NOT EXISTS assistant_pieces (
  key TEXT PRIMARY KEY,
  job_id TEXT NOT NULL REFERENCES assistant_jobs(id) ON DELETE CASCADE,
  chapter_index INTEGER NOT NULL,
  first_block INTEGER NOT NULL,
  last_block INTEGER NOT NULL,
  text TEXT NOT NULL,
  thinking TEXT,
  finish_reason TEXT,
  created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assistant_pieces_job ON assistant_pieces(job_id);

CREATE TABLE IF NOT EXISTS compositions (
  id TEXT PRIMARY KEY,
  name TEXT,
  sample_rate INTEGER NOT NULL DEFAULT 24000,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  render_output_path TEXT,
  tracks_json TEXT,
  meta_json TEXT
);
"""
