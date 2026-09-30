"""Voice profile storage manager."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from src.storage import get_db
from src.voice_identities import VoiceIdentityManager, migrate_legacy_profiles


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_profile(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["is_default"] = bool(data.get("is_default"))
    effects_json = data.pop("effects_json", None)
    effects = json.loads(effects_json) if effects_json else []
    data["effects"] = [{"type": effect} if isinstance(effect, str) else effect for effect in effects]
    return data


class ProfileManager:
    def create(self, name, backend, model, voice, speed, format, blend, reference_audio_id, effects, voice_identity_id=None, instructions=None) -> dict:
        db = get_db()
        if voice_identity_id:
            realization = VoiceIdentityManager().resolve(voice_identity_id, model or backend)
            voice, blend = realization["voice"], None
            reference_audio_id = realization["reference_audio_id"]
        profile_id = str(uuid4())
        now = _now_iso()
        try:
            db.execute(
                """
                INSERT INTO profiles (id, name, backend, model, voice, speed, format, blend, reference_audio_id, effects_json, is_default, created_at, updated_at, voice_identity_id, instructions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
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
                    voice_identity_id,
                    instructions,
                ),
            )
            db.commit()
        except sqlite3.IntegrityError as e:
            db.rollback()
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
        allowed = {"name", "backend", "model", "voice", "speed", "format", "blend", "reference_audio_id", "effects", "voice_identity_id", "instructions"}
        changes = {k: v for k, v in fields.items() if k in allowed}
        existing = self.get(profile_id)
        if not existing:
            raise KeyError(profile_id)
        merged = {**existing, **changes}
        if "voice_identity_id" not in changes and any(
            key in changes and changes[key] != existing.get(key)
            for key in ("model", "voice", "blend", "reference_audio_id")
        ):
            changes["voice_identity_id"] = None
            merged["voice_identity_id"] = None
        if merged.get("voice_identity_id"):
            realization = VoiceIdentityManager().resolve(
                merged["voice_identity_id"], merged.get("model") or merged["backend"]
            )
            changes.update(voice=realization["voice"], blend=None,
                           reference_audio_id=realization["reference_audio_id"])
        if not changes:
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
            db.rollback()
            raise ValueError("Profile name already exists") from e
        if cur.rowcount == 0:
            raise KeyError(profile_id)
        return self.get(profile_id) or {}

    def resolve(self, profile_id: str) -> dict:
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError("Preset not found")
        if profile.get("voice_identity_id"):
            realization = VoiceIdentityManager().resolve(
                profile["voice_identity_id"], profile.get("model") or profile["backend"]
            )
            profile.update(voice=realization["voice"], blend=None,
                           reference_audio_id=realization["reference_audio_id"],
                           voice_identity_name=realization["name"])
        else:
            profile["voice"] = profile.get("blend") or profile["voice"]
        return profile

    def import_presets(self, presets: list[dict]) -> None:
        """Import legacy YAML/default recipes once into the shared preset catalog."""
        db = get_db()
        imported_ids = []
        with db:
            for preset in presets:
                name = preset["name"]
                if db.execute("SELECT 1 FROM preset_imports WHERE name = ?", (name,)).fetchone():
                    continue
                if not db.execute("SELECT 1 FROM profiles WHERE name = ? COLLATE NOCASE", (name,)).fetchone():
                    profile_id = str(uuid4())
                    now = _now_iso()
                    db.execute(
                        """INSERT INTO profiles
                        (id, name, backend, model, voice, speed, format, reference_audio_id,
                         effects_json, instructions, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (profile_id, name, preset.get("backend", "kokoro"),
                         preset.get("model", "kokoro"), preset["voice"],
                         preset.get("speed", 1.0), preset.get("format", "mp3"),
                         preset.get("reference_audio_id"), json.dumps(preset.get("effects", [])),
                         preset.get("instructions"), now, now),
                    )
                    imported_ids.append(profile_id)
                db.execute("INSERT INTO preset_imports VALUES (?)", (name,))
            migrate_legacy_profiles(db, imported_ids)

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
