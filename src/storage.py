"""Shared storage helpers for studio features."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from src.config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    """Return the shared studio.db connection (thread-safe)."""
    global _conn
    with _lock:
        if _conn is None:
            db_path = Path(settings.os_studio_db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(str(db_path), check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA foreign_keys=ON")
        return _conn


def init_db() -> None:
    """Run all CREATE TABLE IF NOT EXISTS DDL. Called at app startup."""
    db = get_db()
    with _lock:
        db.executescript(SCHEMA_SQL)
        columns = {row[1] for row in db.execute("PRAGMA table_info(profiles)")}
        if "voice_identity_id" not in columns:
            db.execute("ALTER TABLE profiles ADD COLUMN voice_identity_id TEXT "
                       "REFERENCES voice_identities(id)")
        if "instructions" not in columns:
            db.execute("ALTER TABLE profiles ADD COLUMN instructions TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS idx_profiles_voice_identity ON profiles(voice_identity_id)")
        from src.voice_identities import migrate_legacy_profiles

        if not db.execute("SELECT 1 FROM schema_migrations WHERE version = 6").fetchone():
            migrate_legacy_profiles(db)
            db.execute("INSERT INTO schema_migrations VALUES (6)")
        db.commit()


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
