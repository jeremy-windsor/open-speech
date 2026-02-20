"""History manager for TTS/STT records."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import settings
from src.storage import get_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(text: str, n: int = 180) -> str:
    t = (text or "").strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


class HistoryManager:
    def log_tts(self, model, voice, speed, format, text, output_path, output_bytes, streamed=False) -> str:
        entry_id = str(uuid4())
        path_value = None if streamed or not settings.os_history_retain_audio else output_path
        bytes_value = None if streamed else output_bytes
        db = get_db()
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
                json.dumps({}),
            ),
        )
        db.commit()
        self.prune()
        return entry_id

    def log_stt(self, model, input_filename, result_text) -> str:
        entry_id = str(uuid4())
        db = get_db()
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
        db.commit()
        self.prune()
        return entry_id

    def list_entries(self, type_filter=None, limit=50, offset=0) -> dict:
        db = get_db()
        where = ""
        params: list = []
        if type_filter in {"tts", "stt"}:
            where = "WHERE type = ?"
            params.append(type_filter)

        total = db.execute(f"SELECT COUNT(*) FROM history_entries {where}", tuple(params)).fetchone()[0]
        rows = db.execute(
            f"SELECT * FROM history_entries {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            tuple([*params, int(limit), int(offset)]),
        ).fetchall()
        items = [dict(r) for r in rows]
        for item in items:
            item["streamed"] = bool(item.get("streamed"))
        return {"items": items, "total": total, "limit": int(limit), "offset": int(offset)}

    def delete_entry(self, entry_id: str) -> bool:
        db = get_db()
        row = db.execute("SELECT output_path FROM history_entries WHERE id = ?", (entry_id,)).fetchone()
        if not row:
            return False
        output_path = row["output_path"]
        db.execute("DELETE FROM history_entries WHERE id = ?", (entry_id,))
        db.commit()
        if output_path:
            self._delete_file_if_exists(output_path)
        return True

    def clear_all(self) -> int:
        db = get_db()
        rows = db.execute("SELECT output_path FROM history_entries WHERE output_path IS NOT NULL").fetchall()
        count = db.execute("SELECT COUNT(*) FROM history_entries").fetchone()[0]
        db.execute("DELETE FROM history_entries")
        db.commit()
        for row in rows:
            self._delete_file_if_exists(row["output_path"])
        return count

    def prune(self) -> int:
        deleted = 0
        db = get_db()

        max_entries = max(0, int(settings.os_history_max_entries))
        if max_entries > 0:
            overflow = db.execute(
                "SELECT id FROM history_entries ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                (max_entries,),
            ).fetchall()
            for row in overflow:
                if self.delete_entry(row["id"]):
                    deleted += 1

        max_bytes = max(0, int(settings.os_history_max_mb)) * 1024 * 1024
        if max_bytes > 0:
            while True:
                rows = db.execute(
                    "SELECT id, output_path FROM history_entries WHERE output_path IS NOT NULL ORDER BY created_at DESC"
                ).fetchall()
                total = 0
                sizes: list[tuple[str, str, int]] = []
                for r in rows:
                    p = r["output_path"]
                    if not p:
                        continue
                    size = self._file_size(p)
                    total += size
                    sizes.append((r["id"], p, size))
                if total <= max_bytes:
                    break
                oldest_with_audio = next(((eid, p, s) for eid, p, s in reversed(sizes)), None)
                if not oldest_with_audio:
                    break
                if self.delete_entry(oldest_with_audio[0]):
                    deleted += 1
                else:
                    break

        return deleted

    def _file_size(self, path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def _delete_file_if_exists(self, path: str) -> None:
        try:
            p = Path(path)
            if p.exists() and p.is_file():
                p.unlink()
        except OSError:
            pass
