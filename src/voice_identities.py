"""Named voices and their explicit, model-specific implementations."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

from src.storage import get_db

VOICE_PREFIX = "voice:"


class VoiceIdentityManager:
    def create(self, name: str) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("Voice name is required")
        db = get_db()
        identity_id = str(uuid4())
        try:
            db.execute(
                "INSERT INTO voice_identities (id, name, created_at) VALUES (?, ?, ?)",
                (identity_id, name, datetime.now(UTC).isoformat()),
            )
            db.commit()
        except sqlite3.IntegrityError as exc:
            db.rollback()
            raise ValueError("Voice name already exists") from exc
        return self.get(identity_id)

    def get(self, identity_id: str) -> dict:
        db = get_db()
        row = db.execute("SELECT * FROM voice_identities WHERE id = ?", (identity_id,)).fetchone()
        if row is None:
            raise KeyError("Named voice not found")
        identity = dict(row)
        identity["voice"] = VOICE_PREFIX + identity_id
        identity["realizations"] = [
            dict(item)
            for item in db.execute(
                "SELECT * FROM voice_realizations WHERE voice_identity_id = ? ORDER BY model",
                (identity_id,),
            )
        ]
        return identity

    def list_all(self) -> list[dict]:
        return [
            self.get(row[0])
            for row in get_db()
            .execute("SELECT id FROM voice_identities ORDER BY name COLLATE NOCASE")
            .fetchall()
        ]

    def add_realization(
        self, identity_id: str, *, model: str, voice: str, reference_audio_id: str | None = None
    ) -> dict:
        self.get(identity_id)
        if not model.strip() or not voice.strip() or voice.strip().startswith(VOICE_PREFIX):
            raise ValueError("A realization requires an exact model and a provider voice")
        db = get_db()
        realization_id = str(uuid4())
        try:
            db.execute(
                """INSERT INTO voice_realizations
                (id, voice_identity_id, model, voice, reference_audio_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    realization_id,
                    identity_id,
                    model.strip(),
                    voice.strip(),
                    reference_audio_id,
                    datetime.now(UTC).isoformat(),
                ),
            )
            db.commit()
        except sqlite3.IntegrityError as exc:
            db.rollback()
            raise ValueError("This voice already has a realization for that model") from exc
        return self.resolve(identity_id, model.strip())

    def resolve(self, identity_id: str, model: str) -> dict:
        identity = self.get(identity_id)
        for realization in identity["realizations"]:
            if realization["model"] == model:
                return {**realization, "name": identity["name"]}
        raise ValueError(f"Named voice '{identity['name']}' has no realization for {model}")


def resolve_named_voice(
    model: str, voice: str, reference_audio_id: str | None = None
) -> tuple[str, str | None]:
    """Resolve only explicit named-voice IDs; provider voice strings stay unchanged."""
    if not voice.startswith(VOICE_PREFIX):
        return voice, reference_audio_id
    realization = VoiceIdentityManager().resolve(voice[len(VOICE_PREFIX) :], model)
    saved_reference = realization["reference_audio_id"]
    if reference_audio_id and reference_audio_id != saved_reference:
        raise ValueError("Named voice reference cannot be overridden")
    return realization["voice"], saved_reference


def migrate_legacy_profiles(db: sqlite3.Connection, profile_ids: list[str] | None = None) -> None:
    """Link legacy recipes without changing their IDs, audio, blends, or settings."""
    rows = db.execute("SELECT * FROM profiles WHERE voice_identity_id IS NULL").fetchall()
    for row in rows:
        if profile_ids is not None and row["id"] not in profile_ids:
            continue
        model = row["model"] or row["backend"]
        if not model:
            continue
        identity = db.execute(
            "SELECT id FROM voice_identities WHERE name = ? COLLATE NOCASE", (row["name"],)
        ).fetchone()
        identity_id = identity[0] if identity else str(uuid4())
        if identity is None:
            db.execute(
                "INSERT INTO voice_identities VALUES (?, ?, ?)",
                (identity_id, row["name"], row["created_at"]),
            )
        recipe = row["blend"] or row["voice"]
        existing = db.execute(
            "SELECT voice, reference_audio_id FROM voice_realizations "
            "WHERE voice_identity_id = ? AND model = ?",
            (identity_id, model),
        ).fetchone()
        if existing is not None and tuple(existing) != (recipe, row["reference_audio_id"]):
            # A user recipe with the same label may differ. Keep it standalone.
            continue
        if existing is None:
            db.execute(
                "INSERT INTO voice_realizations VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    identity_id,
                    model,
                    recipe,
                    row["reference_audio_id"],
                    row["created_at"],
                ),
            )
        db.execute(
            "UPDATE profiles SET voice_identity_id = ? WHERE id = ?", (identity_id, row["id"])
        )
