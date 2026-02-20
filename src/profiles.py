"""Voice profile storage manager."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from src.storage import get_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_profile(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["is_default"] = bool(data.get("is_default"))
    effects_json = data.pop("effects_json", None)
    data["effects"] = json.loads(effects_json) if effects_json else []
    return data


class ProfileManager:
    def create(self, name, backend, model, voice, speed, format, blend, reference_audio_id, effects) -> dict:
        db = get_db()
        profile_id = str(uuid4())
        now = _now_iso()
        try:
            db.execute(
                """
                INSERT INTO profiles (id, name, backend, model, voice, speed, format, blend, reference_audio_id, effects_json, is_default, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    profile_id,
                    name,
                    backend,
                    model,
                    voice,
                    speed,
                    format,
                    blend,
                    reference_audio_id,
                    json.dumps(effects or []),
                    now,
                    now,
                ),
            )
            db.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError("Profile name already exists") from e
        return self.get(profile_id) or {}

    def list_all(self) -> list[dict]:
        db = get_db()
        rows = db.execute("SELECT * FROM profiles ORDER BY name COLLATE NOCASE ASC").fetchall()
        return [_row_to_profile(r) for r in rows]

    def get(self, profile_id: str) -> dict | None:
        db = get_db()
        row = db.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        if not row:
            return None
        return _row_to_profile(row)

    def update(self, profile_id: str, **fields) -> dict:
        allowed = {"name", "backend", "model", "voice", "speed", "format", "blend", "reference_audio_id", "effects"}
        changes = {k: v for k, v in fields.items() if k in allowed}
        if not changes:
            existing = self.get(profile_id)
            if not existing:
                raise KeyError(profile_id)
            return existing

        params = []
        sets = []
        for key, value in changes.items():
            column = "effects_json" if key == "effects" else key
            if key == "effects":
                value = json.dumps(value or [])
            sets.append(f"{column} = ?")
            params.append(value)
        sets.append("updated_at = ?")
        params.append(_now_iso())
        params.append(profile_id)

        db = get_db()
        try:
            cur = db.execute(f"UPDATE profiles SET {', '.join(sets)} WHERE id = ?", tuple(params))
            db.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError("Profile name already exists") from e
        if cur.rowcount == 0:
            raise KeyError(profile_id)
        return self.get(profile_id) or {}

    def delete(self, profile_id: str) -> bool:
        db = get_db()
        cur = db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        db.commit()
        return cur.rowcount > 0

    def set_default(self, profile_id: str) -> None:
        db = get_db()
        cur = db.execute("SELECT id FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        if not cur:
            raise KeyError(profile_id)
        db.execute("UPDATE profiles SET is_default = 0")
        db.execute("UPDATE profiles SET is_default = 1, updated_at = ? WHERE id = ?", (_now_iso(), profile_id))
        db.commit()

    def get_default(self) -> dict | None:
        db = get_db()
        row = db.execute("SELECT * FROM profiles WHERE is_default = 1 LIMIT 1").fetchone()
        if not row:
            return None
        return _row_to_profile(row)
