"""Conversation manager for Studio conversation mode."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from src.config import settings
from src.effects.chain import apply_chain, effect_parameter_error
from src.storage import snapshot, transaction
from src.tts.pipeline import encode_audio, encode_wav

logger = logging.getLogger(__name__)


SILENCE_MS = 500


class ConversationNotFoundError(KeyError):
    """Raised when the addressed conversation does not exist."""


class ConversationChangedError(Exception):
    """Raised when turns were added or deleted while the conversation was rendering."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _put_in_place(staged: Path | None, target: Path, data: bytes | None = None) -> None:
    """Move the staged file over target; write data to target if it was not staged or cannot move.

    Without data, the staged file's own bytes are written.
    """
    if staged is not None:
        try:
            os.replace(staged, target)
            return
        except PermissionError:
            pass  # Windows: a download or a composer render has the old file open. Overwrite it in place.
    target.write_bytes(staged.read_bytes() if data is None else data)


class _FileMoves:
    """A render's final moves into place, each replaced file kept until the database commits.

    A replaced file is moved aside to ``folder/.prev-<hex>.tmp``, or its bytes are kept where Windows
    refuses to move a file another handle has open (_put_in_place then writes in place). restore()
    puts every old file back and removes the new ones, so a failure after the first move (a disk
    error, or an UPDATE that waits out the busy timeout) changes no file; discard() deletes what was
    kept once the moves stand.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._moved: list[tuple[Path, Path | bytes | None]] = []   # target, its old file or bytes, None if new
        self._kept: set[Path] = set()   # old files a failed restore left aside: better kept than lost

    def put_in_place(self, staged: Path | None, target: Path, data: bytes | None = None) -> None:
        old: Path | bytes | None = None
        if target.exists():
            old = self.folder / f".prev-{uuid4().hex}.tmp"
            try:
                os.replace(target, old)
            except PermissionError:
                old = target.read_bytes()
        self._moved.append((target, old))
        _put_in_place(staged, target, data)

    def restore(self) -> None:
        for target, old in reversed(self._moved):
            try:
                if old is None:
                    target.unlink(missing_ok=True)
                elif isinstance(old, bytes):
                    target.write_bytes(old)
                else:
                    _put_in_place(old, target)
            except Exception:  # noqa: BLE001 - the render's own error is the one to raise
                logger.exception("Could not put %s back after a failed conversation render", target)
                if isinstance(old, Path):
                    self._kept.add(old)

    def discard(self) -> None:
        for _target, old in self._moved:
            if isinstance(old, Path) and old not in self._kept:
                with contextlib.suppress(OSError):
                    old.unlink(missing_ok=True)


class ConversationManager:
    def __init__(self, profile_manager=None, synthesize_fn=None, pronounce_fn=None):
        self.profile_manager = profile_manager
        self.synthesize_fn = synthesize_fn
        # The server-wide pronunciation dictionary, applied to each turn's text before synthesis.
        self.pronounce_fn = pronounce_fn

    def create(self, name: str, turns: list[dict]) -> dict:
        cid = str(uuid4())
        now = _now_iso()
        with transaction() as db:
            db.execute(
                "INSERT INTO conversations (id, name, created_at, updated_at, render_output_path, meta_json) VALUES (?, ?, ?, ?, NULL, ?)",
                (cid, name, now, now, json.dumps({})),
            )
            for idx, turn in enumerate(turns or []):
                self._insert_turn(db, cid, idx, turn.get("speaker") or "Speaker", turn.get("text") or "", turn.get("profile_id"), turn.get("effects"))
        return self.get(cid) or {}

    def list_all(self, limit=50, offset=0) -> dict:
        with snapshot() as db:
            total = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            rows = db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
        items = [self._conversation_row(r) for r in rows]
        return {"items": items, "total": total}

    def get(self, conversation_id: str) -> dict | None:
        with snapshot() as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if not row:
                return None
            turns = db.execute(
                "SELECT * FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index ASC",
                (conversation_id,),
            ).fetchall()
        conv = self._conversation_row(row)
        conv["turns"] = [self._turn_row(r) for r in turns]
        return conv

    def add_turn(self, conversation_id: str, speaker: str, text: str, profile_id=None, effects=None) -> dict:
        with transaction() as db:
            exists = db.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if not exists:
                raise ConversationNotFoundError(conversation_id)
            idx = db.execute("SELECT COALESCE(MAX(turn_index), -1) + 1 FROM conversation_turns WHERE conversation_id = ?", (conversation_id,)).fetchone()[0]
            turn_id = self._insert_turn(db, conversation_id, idx, speaker, text, profile_id, effects)
            self._turns_changed(db, conversation_id)
            row = db.execute("SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)).fetchone()
        return self._turn_row(row)

    def delete_turn(self, conversation_id: str, turn_id: str) -> bool:
        with transaction() as db:
            cur = db.execute("DELETE FROM conversation_turns WHERE id = ? AND conversation_id = ?", (turn_id, conversation_id))
            if cur.rowcount <= 0:
                return False
            turns = db.execute("SELECT id FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index ASC", (conversation_id,)).fetchall()
            for idx, row in enumerate(turns):
                db.execute("UPDATE conversation_turns SET turn_index = ? WHERE id = ?", (idx, row["id"]))
            self._turns_changed(db, conversation_id)
        return True

    def _turns_changed(self, db, conversation_id: str) -> None:
        # The saved render no longer matches the turns, so /audio must not serve it.
        db.execute(
            "UPDATE conversations SET updated_at = ?, render_output_path = NULL WHERE id = ?",
            (_now_iso(), conversation_id),
        )

    def delete(self, conversation_id: str) -> bool:
        with transaction() as db:
            exists = db.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not exists:
                return False
            cur = db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        try:
            self._delete_owned_artifacts(conversation_id)
        except OSError:
            logger.exception(
                "Conversation %s was deleted but its artifacts could not be fully removed",
                conversation_id,
            )
        return cur.rowcount > 0

    def _exists(self, conversation_id: str) -> bool:
        with snapshot() as db:
            return db.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is not None

    def _delete_owned_artifacts(self, conversation_id: str) -> None:
        try:
            parsed_id = UUID(conversation_id)
        except (AttributeError, ValueError):
            return
        if parsed_id.version != 4 or str(parsed_id) != conversation_id:
            return

        root = Path(settings.os_conversations_dir).resolve()
        artifact_dir = root / conversation_id
        if artifact_dir.parent != root:
            return
        if artifact_dir.is_symlink() or artifact_dir.is_file():
            artifact_dir.unlink()
        elif artifact_dir.is_dir():
            shutil.rmtree(artifact_dir)

    def render(self, conversation_id: str, format="wav", sample_rate=24000, save_turn_audio=True) -> dict:
        with snapshot() as db:
            conversation = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if not conversation:
                raise ConversationNotFoundError(conversation_id)
            turns = db.execute(
                "SELECT * FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index ASC",
                (conversation_id,),
            ).fetchall()
        if not turns:
            raise ValueError("Conversation has no turns")

        # Check every turn before synthesizing any, so a bad turn costs no synthesis.
        planned = [self._plan_turn(n, self._turn_row(row)) for n, row in enumerate(turns, start=1)]

        out_dir = Path(settings.os_conversations_dir) / conversation_id
        out_dir.mkdir(parents=True, exist_ok=True)

        all_turn_audio: list[np.ndarray] = []
        turn_updates: list[tuple[str | None, int, str]] = []
        silence = np.zeros(int(sample_rate * SILENCE_MS / 1000), dtype=np.float32)
        # Each turn's file is staged like the render and moved over turn_N.wav only when the render
        # is saved. A saved audio_path can name another position's file (a deleted turn shifts the
        # rest), so a render that fails or is retired must leave every turn file as it was.
        staged_turns: list[tuple[Path, Path]] = []
        staged: Path | None = None
        moves = _FileMoves(out_dir)
        try:
            for n, (turn, profile, text, effects) in enumerate(planned, start=1):
                # A preset is spoken by its model or else its backend, as resolve() reads it; only a
                # turn without one uses the configured model.
                model = (profile.get("model") or profile["backend"]) if profile else settings.tts_model
                voice = (profile or {}).get("voice") or settings.tts_voice
                speed = float((profile or {}).get("speed") or 1.0)
                voice_library_ref = (profile or {}).get("reference_audio_id")

                samples = self._synthesize_turn(
                    text=text,
                    model=model,
                    voice=voice,
                    speed=speed,
                    sample_rate=sample_rate,
                    voice_library_ref=voice_library_ref,
                    instructions=(profile or {}).get("instructions"),
                )
                if effects and settings.os_effects_enabled:
                    samples = apply_chain(samples, sample_rate, effects)

                duration_ms = int(1000 * len(samples) / sample_rate) if len(samples) else 0
                turn_audio_path = str(out_dir / f"turn_{n}.wav") if save_turn_audio else None
                if save_turn_audio:
                    stage = out_dir / f".turn-{uuid4().hex}.tmp"
                    staged_turns.append((stage, Path(turn_audio_path)))
                    try:
                        stage.write_bytes(encode_wav(samples, sample_rate=sample_rate))
                    except FileNotFoundError:
                        if self._exists(conversation_id):
                            raise
                        raise ConversationNotFoundError(conversation_id) from None   # its folder went with it

                turn_updates.append((turn_audio_path, duration_ms, turn["id"]))
                all_turn_audio.append(samples)
                if n < len(turns):
                    all_turn_audio.append(silence)

            merged = np.concatenate(all_turn_audio) if all_turn_audio else np.zeros(0, dtype=np.float32)
            out_ext = format.lower()
            output_path = out_dir / f"render.{out_ext}"
            output_bytes = encode_audio(iter([merged]), fmt=out_ext, sample_rate=sample_rate)

            # Saved only after synthesis, in one short transaction: an open write would lock the
            # Reader's own connection out of studio.db for the whole render. Every other thread's
            # studio.db statement waits for that transaction too, so the file is written before it
            # and only moved into place inside it.
            staged = out_dir / f".render-{uuid4().hex}.tmp"
            try:
                staged.write_bytes(output_bytes)
            except FileNotFoundError:
                staged = None  # the conversation was deleted meanwhile; the transaction says so
            with transaction() as db:
                if not db.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone():
                    raise ConversationNotFoundError(conversation_id)
                # A turn added or deleted meanwhile retired this render, so it must neither be
                # served nor overwrite a newer render of the edited turns. Its turn files and paths
                # are kept out too: its positions are no longer the turns' positions, and a newer
                # render may already have given turn_N.wav to another turn.
                current = db.execute(
                    "SELECT id FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index ASC",
                    (conversation_id,),
                ).fetchall()
                unchanged = [row["id"] for row in current] == [turn["id"] for turn, *_ in planned]
                if unchanged:
                    try:
                        # Before the first UPDATE, so SQLite's write lock is not held meanwhile.
                        moves.put_in_place(staged, output_path, output_bytes)
                        for stage, target in staged_turns:
                            moves.put_in_place(stage, target)
                        db.executemany(
                            "UPDATE conversation_turns SET audio_path = ?, duration_ms = ? WHERE id = ?",
                            turn_updates,
                        )
                        db.execute(
                            "UPDATE conversations SET render_output_path = ?, updated_at = ? WHERE id = ?",
                            (str(output_path), _now_iso(), conversation_id),
                        )
                        db.commit()   # here, so a failed commit puts the old files back too
                    except BaseException:
                        # The rows roll back, so the files do too, while no other render can move its own.
                        moves.restore()
                        raise
        except ConversationNotFoundError:
            # Deleted during the render, perhaps before this render made its folder again: what
            # the render wrote there belongs to no conversation.
            try:
                self._delete_owned_artifacts(conversation_id)
            except OSError:
                logger.warning("Could not remove the render files of deleted conversation %s", conversation_id)
            raise
        finally:
            moves.discard()
            for leftover in [staged, *(stage for stage, _ in staged_turns)]:
                if leftover is not None:
                    with contextlib.suppress(OSError):
                        leftover.unlink(missing_ok=True)
        if not unchanged:
            raise ConversationChangedError("Conversation turns changed during the render. Render it again.")

        return {
            "conversation_id": conversation_id,
            "output_path": str(output_path),
            "download_url": f"/api/conversations/{conversation_id}/audio",
            "duration_ms": int(1000 * len(merged) / sample_rate) if len(merged) else 0,
            "turn_count": len(turns),
        }

    def _plan_turn(self, n: int, turn: dict) -> tuple[dict, dict | None, str, list[dict]]:
        """Resolve a turn's preset and spoken text, rejecting what synthesis or effects cannot use."""
        profile = None
        if turn.get("profile_id"):
            if self.profile_manager is None:
                raise ValueError("Preset manager is unavailable")
            try:
                profile = self.profile_manager.resolve(turn["profile_id"])
            except KeyError as exc:
                raise ValueError("Saved conversation preset is unavailable") from exc
        # Same limits as /v1/audio/speech: TTS_MAX_INPUT_LENGTH covers the typed text. The
        # operator's dictionary is not part of it, so its expansions answer only to the model's
        # own limit, which synthesize_array checks.
        max_length = settings.tts_max_input_length
        text = turn["text"]
        if len(text) > max_length:
            raise ValueError(f"Turn {n} is too long. Max: {max_length} characters")
        if self.pronounce_fn is not None:
            text = self.pronounce_fn(text)
        if not text.strip():
            # As /v1/audio/speech answers after the dictionary: a GPU worker refuses an empty text, and only after the
            # turns before it were synthesized.
            raise ValueError(f"Turn {n}: Input text is empty")
        effects = (profile or {}).get("effects", []) + (turn.get("effects") or [])
        effect_error = effect_parameter_error(effects)
        if effect_error:
            raise ValueError(f"Turn {n}: {effect_error}")
        return turn, profile, text, effects

    def _synthesize_turn(self, text: str, model: str, voice: str, speed: float, sample_rate: int, voice_library_ref: str | None = None, instructions: str | None = None) -> np.ndarray:
        if self.synthesize_fn is None:
            raise RuntimeError("No synthesis function configured")
        kwargs = {
            "text": text,
            "model": model,
            "voice": voice,
            "speed": speed,
            "sample_rate": sample_rate,
        }
        if voice_library_ref:
            kwargs["voice_library_ref"] = voice_library_ref
        if instructions:
            kwargs["instructions"] = instructions
        audio = self.synthesize_fn(**kwargs)
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
