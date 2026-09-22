"""Persistent voice reference library for cloning."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import threading
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)
MAX_TRANSCRIPT_CHARS = 10_000


class VoiceNotFoundError(KeyError):
    """Raised when a named voice entry does not exist."""


@dataclass(frozen=True)
class WavInfo:
    frames: int
    sample_rate: int
    channels: int
    sample_width: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def _wav_info(data: bytes) -> WavInfo | None:
    """Inspect a complete PCM WAV containing at least one audio frame."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frame_size = wav_file.getnchannels() * wav_file.getsampwidth()
            remaining_frames = wav_file.getnframes()
            if frame_size <= 0 or remaining_frames <= 0 or wav_file.getframerate() <= 0:
                return None

            info = WavInfo(
                frames=remaining_frames,
                sample_rate=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                sample_width=wav_file.getsampwidth(),
            )

            while remaining_frames > 0:
                requested_frames = min(remaining_frames, 8192)
                frame_bytes = wav_file.readframes(requested_frames)
                if len(frame_bytes) != requested_frames * frame_size:
                    return None
                remaining_frames -= requested_frames
    except (EOFError, OSError, wave.Error):
        return None
    return info


class VoiceLibraryManager:
    def __init__(
        self,
        library_path: str | Path,
        max_count: int = 0,
        max_seconds: int = 60,
    ) -> None:
        self.library_path = Path(library_path)
        self.max_count = max_count  # 0 = unlimited
        self.max_seconds = max_seconds  # 0 = unlimited
        self._lock = threading.RLock()
        with self._lock:
            self.library_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        transcript: str | None = None,
    ) -> dict:
        """Store validated PCM WAV bytes; the supplied media type is discarded."""
        safe_name = self._sanitize_name(name)
        if not audio_bytes:
            raise ValueError("Audio data is empty")
        wav_info = _wav_info(audio_bytes)
        if wav_info is None:
            raise ValueError(
                "Reference audio must be valid WAV format with at least one complete audio frame. "
                "Convert MP3/OGG/FLAC to WAV before uploading."
            )
        if self.max_seconds > 0 and wav_info.duration_seconds > self.max_seconds:
            raise ValueError(
                "Reference audio is too long "
                f"({wav_info.duration_seconds:.2f}s). Max: {self.max_seconds}s"
            )
        # The library accepts only validated PCM WAV bytes. Never persist a
        # caller-supplied media type because the preview route serves this
        # content inline on the application origin.
        content_type = "audio/wav"
        ext = self._extension_for_content_type(content_type)
        created_at = datetime.now(timezone.utc).isoformat()
        normalized_transcript = self._normalize_transcript(transcript)
        metadata = {
            "name": safe_name,
            "size_bytes": len(audio_bytes),
            "content_type": content_type,
            "sha256": hashlib.sha256(audio_bytes).hexdigest(),
            "created_at": created_at,
            "duration_s": round(wav_info.duration_seconds, 2),
            "sample_rate": wav_info.sample_rate,
            "channels": wav_info.channels,
        }
        if normalized_transcript:
            metadata["transcript"] = normalized_transcript

        meta_path = self._meta_path(safe_name)
        audio_path = self.library_path / f"{safe_name}.audio.{ext}"

        with self._lock:
            self.library_path.mkdir(parents=True, exist_ok=True)
            # Enforce max voice count (0 = unlimited)
            if self.max_count > 0 and not meta_path.exists():
                existing_count = sum(1 for _ in self.library_path.glob("*.meta.json"))
                if existing_count >= self.max_count:
                    raise ValueError(
                        f"Voice library is full ({self.max_count} voices max). "
                        "Delete a voice before adding more."
                    )

            transaction_id = uuid4().hex
            staged_audio_path = self.library_path / f".{safe_name}.{transaction_id}.audio.tmp"
            staged_meta_path = self.library_path / f".{safe_name}.{transaction_id}.meta.tmp"
            backup_audio_path = self.library_path / f".{safe_name}.{transaction_id}.audio.backup"
            audio_backed_up = False
            audio_installed = False
            committed = False
            try:
                staged_audio_path.write_bytes(audio_bytes)
                staged_meta_path.write_text(json.dumps(metadata), encoding="utf-8")

                if audio_path.exists():
                    audio_path.replace(backup_audio_path)
                    audio_backed_up = True
                staged_audio_path.replace(audio_path)
                audio_installed = True
                staged_meta_path.replace(meta_path)
                committed = True
            except Exception:
                if audio_installed:
                    audio_path.unlink(missing_ok=True)
                if audio_backed_up:
                    backup_audio_path.replace(audio_path)
                raise
            finally:
                cleanup_paths = [staged_audio_path, staged_meta_path]
                if committed:
                    cleanup_paths.append(backup_audio_path)
                for cleanup_path in cleanup_paths:
                    try:
                        cleanup_path.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Could not remove voice-library staging file %s", cleanup_path)

            # Metadata now points at the committed WAV. Remove any obsolete
            # legacy extension only after the new pair is safely installed.
            for existing in self.library_path.glob(f"{safe_name}.audio.*"):
                if existing != audio_path:
                    try:
                        existing.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Could not remove obsolete voice audio %s", existing)

        return metadata

    def list_voices(self) -> list[dict]:
        with self._lock:
            voices: list[dict] = []
            for meta_path in self.library_path.glob("*.meta.json"):
                try:
                    item = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not isinstance(item, dict):
                        continue
                    # Skip entries whose audio file is missing (corrupted state)
                    ct = item.get("content_type", "audio/wav")
                    ext = self._extension_for_content_type(ct)
                    safe_name = item.get("name", "")
                    audio_path = self.library_path / f"{safe_name}.audio.{ext}"
                    if not audio_path.exists():
                        logger.warning("Voice library: audio file missing for '%s' — skipping", safe_name)
                        continue
                    voices.append(item)
                except Exception as exc:
                    logger.warning("Voice library: skipping corrupted metadata %s (%s)", meta_path, exc)
                    continue
            voices.sort(key=lambda x: x.get("name", ""))
            return voices

    def get(self, name: str) -> tuple[bytes, dict]:
        safe_name = self._sanitize_name(name)
        with self._lock:
            meta_path = self._meta_path(safe_name)
            if not meta_path.exists():
                raise VoiceNotFoundError(name)

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            content_type = metadata.get("content_type", "audio/wav")
            ext = self._extension_for_content_type(content_type)
            audio_path = self.library_path / f"{safe_name}.audio.{ext}"
            if not audio_path.exists():
                raise VoiceNotFoundError(name)

            return audio_path.read_bytes(), metadata

    def delete(self, name: str) -> None:
        safe_name = self._sanitize_name(name)
        with self._lock:
            meta_path = self._meta_path(safe_name)
            matched_audio = list(self.library_path.glob(f"{safe_name}.audio.*"))
            if not meta_path.exists() and not matched_audio:
                raise VoiceNotFoundError(name)

            meta_path.unlink(missing_ok=True)
            for p in matched_audio:
                p.unlink(missing_ok=True)

    def set_transcript(self, name: str, transcript: str | None) -> dict:
        """Update only a saved reference transcript and return its metadata."""
        safe_name = self._sanitize_name(name)
        with self._lock:
            meta_path = self._meta_path(safe_name)
            if not meta_path.exists():
                raise VoiceNotFoundError(name)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            normalized_transcript = self._normalize_transcript(transcript)
            if normalized_transcript:
                metadata["transcript"] = normalized_transcript
            else:
                metadata.pop("transcript", None)
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

    def exists(self, name: str) -> bool:
        safe_name = self._sanitize_name(name)
        with self._lock:
            return self._meta_path(safe_name).exists()

    def _meta_path(self, safe_name: str) -> Path:
        return self.library_path / f"{safe_name}.meta.json"

    def _normalize_transcript(self, transcript: str | None) -> str | None:
        if not transcript or not transcript.strip():
            return None
        normalized = transcript.strip()
        if len(normalized) > MAX_TRANSCRIPT_CHARS:
            raise ValueError(
                f"Reference transcript is too long. Max: {MAX_TRANSCRIPT_CHARS} characters"
            )
        return normalized

    def _sanitize_name(self, name: str) -> str:
        safe = name.strip().lower()
        safe = safe.replace(" ", "_").replace("-", "_")
        safe = re.sub(r"[^a-z0-9_]", "", safe)
        safe = safe[:64]
        if not safe:
            raise ValueError("Voice name must contain at least one alphanumeric character")
        return safe

    def _extension_for_content_type(self, content_type: str) -> str:
        ct = content_type.lower().strip()
        mapping = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/flac": "flac",
        }
        return mapping.get(ct, "wav")
