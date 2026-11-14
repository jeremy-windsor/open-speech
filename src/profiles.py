"""Voice profile storage manager."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from src.assistant.presets import has_lone_surrogate
from src.effects.chain import effect_parameter_error
from src.storage import get_db, transaction
from src.voice_identities import VoiceIdentityManager, migrate_legacy_profiles

logger = logging.getLogger(__name__)


class ProfileNotFoundError(KeyError):
    """Raised when the addressed preset does not exist."""


def _is_name_conflict(error: sqlite3.IntegrityError) -> bool:
    """Whether the write failed only because another preset already has the name."""
    return str(error) == "UNIQUE constraint failed: profiles.name"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_complete_preset(preset) -> bool:
    return isinstance(preset, dict) and all(
        isinstance(preset.get(key), str) and preset[key].strip() for key in ("name", "voice")
    )


def _preset_problem(preset: dict) -> str | None:
    """Why POST /api/profiles would refuse this imported preset (ProfilePayload's rules), or None."""
    speed = preset.get("speed", 1.0)
    # bool is an int subclass, and NaN fails every comparison.
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4.0:
        return "speed must be a number from 0.25 to 4.0"
    for field in ("name", "voice", "backend", "format", "model", "instructions", "reference_audio_id"):
        if field not in preset:
            continue
        value = preset[field]
        nullable = field not in ("name", "voice", "backend", "format")   # backend and format have defaults, no null
        if not isinstance(value, str) and not (nullable and value is None):
            return f"{field} must be text"
        # YAML reads an emoji's JSON escape pair as two lone surrogates, which studio.db can't encode (ProfilePayload
        # refuses them).
        if isinstance(value, str) and has_lone_surrogate(value):
            return f"{field} has a lone surrogate (half of a UTF-16 pair, such as half an emoji), which can't be stored"
    effects = preset.get("effects", [])
    if not isinstance(effects, list) or not all(isinstance(effect, (dict, str)) for effect in effects):
        return "effects must be a list of effects"
    error = effect_parameter_error([{"type": fx} if isinstance(fx, str) else fx for fx in effects])
    if error:
        return f"effects: {error}"
    try:
        json.dumps(effects, allow_nan=False)
    except (TypeError, ValueError):
        return "effects must hold only text, finite numbers, lists and mappings"
    return None


def _row_to_profile(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["is_default"] = bool(data.get("is_default"))
    effects_json = data.pop("effects_json", None)
    effects = json.loads(effects_json) if effects_json else []
    data["effects"] = [{"type": effect} if isinstance(effect, str) else effect for effect in effects]
    options_json = data.pop("options_json", None)
    data["options"] = json.loads(options_json) if options_json else {}
    return data


class ProfileManager:
    def create(self, name, backend, model, voice, speed, format, blend, reference_audio_id, effects, voice_identity_id=None, instructions=None, options=None) -> dict:
        profile_id = str(uuid4())
        now = _now_iso()
        try:
            with transaction() as db:
                if voice_identity_id is not None:
                    # In the INSERT's transaction: a named voice deleted in between would fail
                    # its foreign key. Any id names a voice, so "" is an unknown one (404).
                    realization = VoiceIdentityManager().resolve(voice_identity_id, model or backend)
                    voice, blend = realization["voice"], None
                    reference_audio_id = realization["reference_audio_id"]
                db.execute(
                    """
                    INSERT INTO profiles (id, name, backend, model, voice, speed, format, blend, reference_audio_id, effects_json, is_default, created_at, updated_at, voice_identity_id, instructions, options_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
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
                        json.dumps(options or {}),
                    ),
                )
        except sqlite3.IntegrityError as e:
            if _is_name_conflict(e):
                raise ValueError("Profile name already exists") from e
            raise
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
        allowed = {"name", "backend", "model", "voice", "speed", "format", "blend", "reference_audio_id", "effects", "voice_identity_id", "instructions", "options"}
        changes = {k: v for k, v in fields.items() if k in allowed}
        try:
            # One transaction from the read to the UPDATE: a named voice deleted in between
            # would fail the UPDATE's foreign key.
            with transaction() as db:
                existing = self.get(profile_id)
                if not existing:
                    raise ProfileNotFoundError(profile_id)
                merged = {**existing, **changes}
                if "voice_identity_id" not in changes and any(
                    key in changes and changes[key] != existing.get(key)
                    for key in ("model", "voice", "blend", "reference_audio_id")
                ):
                    changes["voice_identity_id"] = None
                    merged["voice_identity_id"] = None
                if merged.get("voice_identity_id") is not None:
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
                    column = f"{key}_json" if key in ("effects", "options") else key
                    if key == "effects":
                        value = json.dumps(value or [])
                    elif key == "options":
                        value = json.dumps(value or {})
                    sets.append(f"{column} = ?")
                    params.append(value)
                sets.append("updated_at = ?")
                params.append(_now_iso())
                params.append(profile_id)
                cur = db.execute(f"UPDATE profiles SET {', '.join(sets)} WHERE id = ?", tuple(params))
        except sqlite3.IntegrityError as e:
            if _is_name_conflict(e):
                raise ValueError("Profile name already exists") from e
            raise
        if cur.rowcount == 0:
            raise ProfileNotFoundError(profile_id)
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
        imported_ids = []
        with transaction() as db:
            for preset in presets:
                if not _is_complete_preset(preset):
                    logger.warning("Skipping voice preset %r: it needs a name and a voice", preset)
                    continue
                name = preset["name"]
                # Checked before the first statement, which binds the name.
                problem = _preset_problem(preset)
                if problem:
                    # Not recorded as imported, so the entry is imported once the file is fixed.
                    logger.warning("Skipping voice preset %r: %s", name, problem)
                    continue
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
                         float(preset.get("speed", 1.0)), preset.get("format", "mp3"),
                         preset.get("reference_audio_id"), json.dumps(preset.get("effects", [])),
                         preset.get("instructions"), now, now),
                    )
                    imported_ids.append(profile_id)
                db.execute("INSERT INTO preset_imports VALUES (?)", (name,))
            migrate_legacy_profiles(db, imported_ids)

    def delete(self, profile_id: str) -> bool:
        with transaction() as db:
            cur = db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        return cur.rowcount > 0

    def set_default(self, profile_id: str) -> None:
        with transaction() as db:
            cur = db.execute("SELECT id FROM profiles WHERE id = ?", (profile_id,)).fetchone()
            if not cur:
                raise KeyError(profile_id)
            db.execute("UPDATE profiles SET is_default = 0")
            db.execute("UPDATE profiles SET is_default = 1, updated_at = ? WHERE id = ?", (_now_iso(), profile_id))

    def get_default(self) -> dict | None:
        db = get_db()
        row = db.execute("SELECT * FROM profiles WHERE is_default = 1 LIMIT 1").fetchone()
        if not row:
            return None
        return _row_to_profile(row)
