"""Conversation manager for Studio conversation mode."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.config import settings
from src.effects.chain import apply_chain
from src.storage import get_db
from src.tts.pipeline import encode_audio, encode_wav


SILENCE_MS = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationManager:
    def __init__(self, profile_manager=None, synthesize_fn=None):
        self.profile_manager = profile_manager
        self.synthesize_fn = synthesize_fn

    def create(self, name: str, turns: list[dict]) -> dict:
        db = get_db()
        cid = str(uuid4())
        now = _now_iso()
        db.execute(
            "INSERT INTO conversations (id, name, created_at, updated_at, render_output_path, meta_json) VALUES (?, ?, ?, ?, NULL, ?)",
            (cid, name, now, now, json.dumps({})),
        )
        for idx, turn in enumerate(turns or []):
            self._insert_turn(db, cid, idx, turn.get("speaker") or "Speaker", turn.get("text") or "", turn.get("profile_id"), turn.get("effects"))
        db.commit()
        return self.get(cid) or {}

    def list_all(self, limit=50, offset=0) -> dict:
        db = get_db()
        total = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        rows = db.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (int(limit), int(offset)),
        ).fetchall()
        items = [self._conversation_row(r) for r in rows]
        return {"items": items, "total": total}

    def get(self, conversation_id: str) -> dict | None:
        db = get_db()
        row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not row:
            return None
        conv = self._conversation_row(row)
        turns = db.execute(
            "SELECT * FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index ASC",
            (conversation_id,),
        ).fetchall()
        conv["turns"] = [self._turn_row(r) for r in turns]
        return conv

    def add_turn(self, conversation_id: str, speaker: str, text: str, profile_id=None, effects=None) -> dict:
        db = get_db()
        exists = db.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not exists:
            raise KeyError(conversation_id)
        idx = db.execute("SELECT COALESCE(MAX(turn_index), -1) + 1 FROM conversation_turns WHERE conversation_id = ?", (conversation_id,)).fetchone()[0]
        turn_id = self._insert_turn(db, conversation_id, idx, speaker, text, profile_id, effects)
        db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now_iso(), conversation_id))
        db.commit()
        row = db.execute("SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)).fetchone()
        return self._turn_row(row)

    def delete_turn(self, conversation_id: str, turn_id: str) -> bool:
        db = get_db()
        cur = db.execute("DELETE FROM conversation_turns WHERE id = ? AND conversation_id = ?", (turn_id, conversation_id))
        if cur.rowcount <= 0:
            return False
        turns = db.execute("SELECT id FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index ASC", (conversation_id,)).fetchall()
        for idx, row in enumerate(turns):
            db.execute("UPDATE conversation_turns SET turn_index = ? WHERE id = ?", (idx, row["id"]))
        db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now_iso(), conversation_id))
        db.commit()
        return True

    def delete(self, conversation_id: str) -> bool:
        db = get_db()
        cur = db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        db.commit()
        return cur.rowcount > 0

    def render(self, conversation_id: str, format="wav", sample_rate=24000, save_turn_audio=True) -> dict:
        db = get_db()
        conversation = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not conversation:
            raise KeyError(conversation_id)

        turns = db.execute(
            "SELECT * FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index ASC",
            (conversation_id,),
        ).fetchall()
        if not turns:
            raise ValueError("Conversation has no turns")

        out_dir = Path(settings.os_conversations_dir) / conversation_id
        out_dir.mkdir(parents=True, exist_ok=True)

        all_turn_audio: list[np.ndarray] = []
        silence = np.zeros(int(sample_rate * SILENCE_MS / 1000), dtype=np.float32)

        for n, row in enumerate(turns, start=1):
            turn = self._turn_row(row)
            profile = self.profile_manager.get(turn["profile_id"]) if (self.profile_manager and turn.get("profile_id")) else None
            model = (profile or {}).get("model") or settings.tts_model
            voice = (profile or {}).get("voice") or settings.tts_voice
            speed = float((profile or {}).get("speed") or 1.0)
            effects = turn.get("effects") or []

            samples = self._synthesize_turn(text=turn["text"], model=model, voice=voice, speed=speed, sample_rate=sample_rate)
            if effects:
                samples = apply_chain(samples, sample_rate, effects)

            duration_ms = int(1000 * len(samples) / sample_rate) if len(samples) else 0
            turn_audio_path = str(out_dir / f"turn_{n}.wav") if save_turn_audio else None
            if save_turn_audio:
                Path(turn_audio_path).write_bytes(encode_wav(samples, sample_rate=sample_rate))

            db.execute(
                "UPDATE conversation_turns SET audio_path = ?, duration_ms = ? WHERE id = ?",
                (turn_audio_path, duration_ms, turn["id"]),
            )
            all_turn_audio.append(samples)
            if n < len(turns):
                all_turn_audio.append(silence)

        merged = np.concatenate(all_turn_audio) if all_turn_audio else np.zeros(0, dtype=np.float32)
        out_ext = format.lower()
        output_path = out_dir / f"render.{out_ext}"
        output_bytes = encode_audio(iter([merged]), fmt=out_ext, sample_rate=sample_rate)
        output_path.write_bytes(output_bytes)

        db.execute(
            "UPDATE conversations SET render_output_path = ?, updated_at = ? WHERE id = ?",
            (str(output_path), _now_iso(), conversation_id),
        )
        db.commit()

        return {
            "conversation_id": conversation_id,
            "output_path": str(output_path),
            "download_url": f"/api/conversations/{conversation_id}/audio",
            "duration_ms": int(1000 * len(merged) / sample_rate) if len(merged) else 0,
            "turn_count": len(turns),
        }

    def _synthesize_turn(self, text: str, model: str, voice: str, speed: float, sample_rate: int) -> np.ndarray:
        if self.synthesize_fn is None:
            raise RuntimeError("No synthesis function configured")
        audio = self.synthesize_fn(text=text, model=model, voice=voice, speed=speed, sample_rate=sample_rate)
        return np.asarray(audio, dtype=np.float32)

    def _insert_turn(self, db, conversation_id: str, idx: int, speaker: str, text: str, profile_id=None, effects=None) -> str:
        turn_id = str(uuid4())
        db.execute(
            """
            INSERT INTO conversation_turns (id, conversation_id, turn_index, speaker, profile_id, text, audio_path, duration_ms, effects_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (turn_id, conversation_id, idx, speaker, profile_id, text, json.dumps(effects or []), _now_iso()),
        )
        return turn_id

    def _conversation_row(self, row) -> dict:
        return dict(row)

    def _turn_row(self, row) -> dict:
        data = dict(row)
        effects_json = data.pop("effects_json", None)
        data["effects"] = json.loads(effects_json) if effects_json else []
        return data
