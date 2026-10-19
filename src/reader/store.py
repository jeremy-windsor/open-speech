"""SQLite persistence for Reader documents, positions and rendered audio parts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

from src import storage
from src.live_tts.segmenter import split_sentences
from src.reader.document import Document

SENTENCE_MAX_CHARS = 400

# The Reader has its own connection to studio.db: the renderer thread commits
# often, and sharing the studio connection would mix its commits into other
# features' open transactions. The lock serialises the renderer and handlers.
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None
RECENTLY_PLAYED_S = 300


def get_db() -> sqlite3.Connection:
    global _conn, _conn_path
    with _lock:
        path = storage.settings.os_studio_db_path
        if _conn is None or _conn_path != path:
            close()
            _conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA foreign_keys=ON")
            _conn_path = path
        return _conn


def close() -> None:
    global _conn, _conn_path
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _conn_path = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(document: Document) -> str:
    payload = json.dumps(
        [[chapter.title, [[block.kind, block.text] for block in chapter.blocks]] for chapter in document.chapters],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{document.source_type}\n{payload}".encode("utf-8")).hexdigest()


def _progress_row(row) -> dict | None:
    if row is None:
        return None
    return {
        "block_index": row["block_index"],
        "offset_ms": row["offset_ms"],
        "model": row["model"],
        "voice": row["voice"],
        "speed": row["speed"],
        "updated_at": row["updated_at"],
    }


class ReaderStore:
    def save_document(self, document: Document) -> tuple[str, bool]:
        """Store a parsed document. Returns ``(id, created)``; identical content is reused."""
        digest = content_hash(document)
        with _lock:
            db = get_db()
            existing = db.execute("SELECT id FROM reader_documents WHERE content_hash = ?", (digest,)).fetchone()
            if existing:
                return existing["id"], False
            document_id = uuid4().hex
            now = _now_iso()
            db.execute(
                "INSERT INTO reader_documents (id, title, author, source_type, content_hash, char_count, created_at, updated_at, meta_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (document_id, document.title, document.author, document.source_type, digest, document.char_count, now, now, "{}"),
            )
            block_index = 0
            for chapter_index, chapter in enumerate(document.chapters):
                db.execute(
                    "INSERT INTO reader_chapters (document_id, chapter_index, title, first_block, block_count) VALUES (?, ?, ?, ?, ?)",
                    (document_id, chapter_index, chapter.title, block_index, len(chapter.blocks)),
                )
                rows = []
                for block in chapter.blocks:
                    sentences = split_sentences(block.text, SENTENCE_MAX_CHARS)
                    rows.append((document_id, block_index, chapter_index, block.kind, block.text, json.dumps(sentences)))
                    block_index += 1
                db.executemany(
                    "INSERT INTO reader_blocks (document_id, block_index, chapter_index, kind, text, sentences_json) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
            db.commit()
            return document_id, True

    def _summary(self, db, row) -> dict:
        counts = db.execute(
            "SELECT COUNT(*) AS chapters, COALESCE(SUM(block_count), 0) AS blocks FROM reader_chapters WHERE document_id = ?",
            (row["id"],),
        ).fetchone()
        progress = db.execute("SELECT * FROM reader_progress WHERE document_id = ?", (row["id"],)).fetchone()
        return {
            "id": row["id"],
            "title": row["title"],
            "author": row["author"],
            "source_type": row["source_type"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "chapter_count": counts["chapters"],
            "block_count": counts["blocks"],
            "char_count": row["char_count"],
            "progress": _progress_row(progress),
        }

    def list_documents(self) -> list[dict]:
        with _lock:
            db = get_db()
            rows = db.execute("SELECT * FROM reader_documents ORDER BY updated_at DESC").fetchall()
            return [self._summary(db, row) for row in rows]

    def get_document(self, document_id: str, *, include_blocks: bool = True) -> dict | None:
        with _lock:
            db = get_db()
            row = db.execute("SELECT * FROM reader_documents WHERE id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            document = self._summary(db, row)
            document["chapters"] = [
                {"index": r["chapter_index"], "title": r["title"], "first_block": r["first_block"], "block_count": r["block_count"]}
                for r in db.execute(
                    "SELECT * FROM reader_chapters WHERE document_id = ? ORDER BY chapter_index", (document_id,)
                )
            ]
            if include_blocks:
                document["blocks"] = [
                    {
                        "index": r["block_index"],
                        "chapter_index": r["chapter_index"],
                        "kind": r["kind"],
                        "text": r["text"],
                        "sentences": json.loads(r["sentences_json"]),
                    }
                    for r in db.execute(
                        "SELECT * FROM reader_blocks WHERE document_id = ? ORDER BY block_index", (document_id,)
                    )
                ]
            return document

    def get_blocks(self, document_id: str, first: int, last: int) -> list[dict]:
        with _lock:
            rows = get_db().execute(
                "SELECT * FROM reader_blocks WHERE document_id = ? AND block_index BETWEEN ? AND ? ORDER BY block_index",
                (document_id, first, last),
            ).fetchall()
        return [
            {
                "index": r["block_index"],
                "chapter_index": r["chapter_index"],
                "kind": r["kind"],
                "text": r["text"],
                "sentences": json.loads(r["sentences_json"]),
            }
            for r in rows
        ]

    def outline(self, document_id: str) -> list[tuple[int, int, int]] | None:
        """Return ``(block_index, chapter_index, char_count)`` per block, or None if missing."""
        with _lock:
            db = get_db()
            if db.execute("SELECT 1 FROM reader_documents WHERE id = ?", (document_id,)).fetchone() is None:
                return None
            rows = db.execute(
                "SELECT block_index, chapter_index, LENGTH(text) AS chars FROM reader_blocks WHERE document_id = ? ORDER BY block_index",
                (document_id,),
            ).fetchall()
        return [(r["block_index"], r["chapter_index"], r["chars"]) for r in rows]

    def delete_document(self, document_id: str) -> list[str] | None:
        """Delete a document and its rows. Returns the audio paths to remove, or None if missing."""
        with _lock:
            db = get_db()
            if db.execute("SELECT 1 FROM reader_documents WHERE id = ?", (document_id,)).fetchone() is None:
                return None
            paths = [r["path"] for r in db.execute("SELECT path FROM reader_parts WHERE document_id = ?", (document_id,))]
            db.execute("DELETE FROM reader_documents WHERE id = ?", (document_id,))
            db.commit()
            return paths

    def set_progress(self, document_id: str, *, block_index: int, offset_ms: int, model: str | None, voice: str | None, speed: float) -> dict | None:
        with _lock:
            db = get_db()
            if db.execute("SELECT 1 FROM reader_documents WHERE id = ?", (document_id,)).fetchone() is None:
                return None
            blocks = db.execute("SELECT COUNT(*) FROM reader_blocks WHERE document_id = ?", (document_id,)).fetchone()[0]
            if block_index >= blocks:
                raise ValueError(f"block_index must be below {blocks}")
            now = _now_iso()
            db.execute(
                "INSERT INTO reader_progress (document_id, block_index, offset_ms, model, voice, speed, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(document_id) DO UPDATE SET "
                "block_index = excluded.block_index, offset_ms = excluded.offset_ms, model = excluded.model, "
                "voice = excluded.voice, speed = excluded.speed, updated_at = excluded.updated_at",
                (document_id, block_index, offset_ms, model, voice, speed, now),
            )
            db.execute("UPDATE reader_documents SET updated_at = ? WHERE id = ?", (now, document_id))
            db.commit()
            return _progress_row(db.execute("SELECT * FROM reader_progress WHERE document_id = ?", (document_id,)).fetchone())

    # ── Rendered audio parts ────────────────────────────────────────────────

    def ready_parts(self, document_id: str, model: str, voice: str) -> dict[int, dict]:
        """Ready parts keyed by first block (the longest one wins when several start there)."""
        with _lock:
            rows = get_db().execute(
                "SELECT * FROM reader_parts WHERE document_id = ? AND model = ? AND voice = ? ORDER BY first_block, last_block",
                (document_id, model, voice),
            ).fetchall()
        return {r["first_block"]: dict(r) for r in rows}

    def get_part(self, part_id: str) -> dict | None:
        with _lock:
            row = get_db().execute("SELECT * FROM reader_parts WHERE id = ?", (part_id,)).fetchone()
        return dict(row) if row else None

    def save_part(self, *, part_id: str, document_id: str, first_block: int, last_block: int, model: str, voice: str,
                  path: str, size: int, duration_ms: int, sentences: list[dict]) -> bool:
        """Record a rendered part; returns False if its document was deleted meanwhile."""
        with _lock:
            db = get_db()
            if db.execute("SELECT 1 FROM reader_documents WHERE id = ?", (document_id,)).fetchone() is None:
                return False
            db.execute(
                "INSERT OR REPLACE INTO reader_parts (id, document_id, first_block, last_block, model, voice, path, bytes, "
                "duration_ms, sentences_json, created_at, last_access) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (part_id, document_id, first_block, last_block, model, voice, path, size, duration_ms,
                 json.dumps(sentences), _now_iso(), time.time()),
            )
            db.commit()
            return True

    def forget_part(self, part_id: str) -> None:
        with _lock:
            db = get_db()
            db.execute("DELETE FROM reader_parts WHERE id = ?", (part_id,))
            db.commit()

    def part_paths(self) -> dict[str, str]:
        """Every recorded part as ``{id: path}``, for reconciling the cache folder."""
        with _lock:
            return {r["id"]: r["path"] for r in get_db().execute("SELECT id, path FROM reader_parts")}

    def touch_part(self, part_id: str) -> None:
        with _lock:
            db = get_db()
            db.execute("UPDATE reader_parts SET last_access = ? WHERE id = ?", (time.time(), part_id))
            db.commit()

    def prune_parts(self, max_bytes: int, keep: set[str]) -> list[str]:
        """Forget least recently used parts until the total fits; returns their paths.

        Parts in ``keep`` and parts played in the last few minutes (the one a
        listener is on right now) are never removed.
        """
        with _lock:
            db = get_db()
            total = db.execute("SELECT COALESCE(SUM(bytes), 0) FROM reader_parts").fetchone()[0]
            if total <= max_bytes:
                return []
            removed: list[str] = []
            recent = time.time() - RECENTLY_PLAYED_S
            rows = db.execute("SELECT id, path, bytes, last_access FROM reader_parts ORDER BY last_access").fetchall()
            for row in rows:
                if total <= max_bytes or row["last_access"] >= recent:
                    break
                if row["id"] in keep:
                    continue
                db.execute("DELETE FROM reader_parts WHERE id = ?", (row["id"],))
                removed.append(row["path"])
                total -= row["bytes"]
            db.commit()
            return removed
