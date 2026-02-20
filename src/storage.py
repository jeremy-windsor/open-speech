"""Shared storage helpers for Phase 8 studio features."""

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
        db.commit()


SCHEMA_SQL = """
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
"""
