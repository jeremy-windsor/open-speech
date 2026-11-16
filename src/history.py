"""History manager for TTS/STT records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import settings
from src.storage import get_db, snapshot, transaction

# created_at ties at clock resolution (about 16 ms on Windows); rowid keeps insertion order.
_NEWEST_FIRST = "created_at DESC, rowid DESC"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(text: str, n: int = 180) -> str:
    t = (text or "").strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def _parse_meta(meta_json: str | None) -> dict:
    try:
        meta = json.loads(meta_json) if meta_json else {}
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


class HistoryManager:
    def log_tts(self, model, voice, speed, format, text, output_path, output_bytes, streamed=False, meta: dict | None = None) -> str:
        entry_id = str(uuid4())
        path_value = None if streamed else output_path
        bytes_value = None if streamed else output_bytes
        with transaction() as db:
            db.execute(
                """
                INSERT INTO history_entries (id, type, created_at, model, voice, speed, format, text_preview, full_text, output_path, output_bytes, streamed, meta_json)
                VALUES (?, 'tts', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    _now_iso(),
                    model,
                    voice,
                    speed,
                    format,
                    _preview(text),
                    text,
                    path_value,
                    bytes_value,
                    1 if streamed else 0,
                    json.dumps(meta or {}),
                ),
            )
        self.prune()
        return entry_id

    def log_stt(self, model, input_filename, result_text) -> str:
        entry_id = str(uuid4())
        with transaction() as db:
            db.execute(
                """
                INSERT INTO history_entries (id, type, created_at, model, text_preview, full_text, input_filename, streamed, meta_json)
                VALUES (?, 'stt', ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    entry_id,
                    _now_iso(),
                    model,
                    _preview(result_text),
                    result_text,
                    input_filename,
                    json.dumps({}),
                ),
            )
        self.prune()
        return entry_id

    def list_entries(self, type_filter=None, limit=50, offset=0, search=None) -> dict:
        conditions: list[str] = []
        params: list = []
        if type_filter in {"tts", "stt"}:
            conditions.append("type = ?")
            params.append(type_filter)
        search = (search or "").strip()
        if search:
            # A literal, case-insensitive substring of the whole text or file name, not the preview.
            conditions.append("(os_contains_casefold(full_text, ?) OR os_contains_casefold(input_filename, ?))")
            params.extend([search.casefold()] * 2)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with snapshot() as db:  # the total counts the same entries the page is cut from
            total = db.execute(f"SELECT COUNT(*) FROM history_entries {where}", tuple(params)).fetchone()[0]
            rows = db.execute(
                f"SELECT * FROM history_entries {where} ORDER BY {_NEWEST_FIRST} LIMIT ? OFFSET ?",
                tuple([*params, int(limit), int(offset)]),
            ).fetchall()
        items = [dict(r) for r in rows]
        for item in items:
            item["streamed"] = bool(item.get("streamed"))
            item["meta"] = _parse_meta(item.pop("meta_json", None))
        return {"items": items, "total": total, "limit": int(limit), "offset": int(offset)}

    def delete_entry(self, entry_id: str) -> bool:
        with transaction() as db:
            row = db.execute("SELECT output_path FROM history_entries WHERE id = ?", (entry_id,)).fetchone()
            if not row:
                return False
            output_path = row["output_path"]
            db.execute("DELETE FROM history_entries WHERE id = ?", (entry_id,))
        if output_path:
            self._delete_file_if_exists(output_path)
        return True

    def clear_all(self) -> int:
        with transaction() as db:
            rows = db.execute("SELECT output_path FROM history_entries WHERE output_path IS NOT NULL").fetchall()
            count = db.execute("SELECT COUNT(*) FROM history_entries").fetchone()[0]
            db.execute("DELETE FROM history_entries")
        for row in rows:
            self._delete_file_if_exists(row["output_path"])
        return count

    def prune(self) -> int:
        deleted = 0
        db = get_db()

        max_entries = max(0, int(settings.os_history_max_entries))
        if max_entries > 0:
            overflow = db.execute(
                f"SELECT id FROM history_entries ORDER BY {_NEWEST_FIRST} LIMIT -1 OFFSET ?",
                (max_entries,),
            ).fetchall()
            for row in overflow:
                if self.delete_entry(row["id"]):
                    deleted += 1

        return deleted

    def _delete_file_if_exists(self, path: str) -> None:
        try:
            p = Path(path)
            if p.exists() and p.is_file():
                p.unlink()
        except OSError:
            pass
