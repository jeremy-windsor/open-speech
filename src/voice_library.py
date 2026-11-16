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


class DesignedSampleError(ValueError):
    """Raised when a library write would change a designed voice's sample, which only its named voice changes."""


DESIGNED_PREFIX = "designed_"


def designed_sample_name(identity_id: str) -> str:
    """The library name of a designed named voice's sample."""
    return DESIGNED_PREFIX + identity_id.lower().replace("-", "_")


_DESIGNED_NAME = re.compile(r"designed_([0-9a-f]{8})_([0-9a-f]{4})_([0-9a-f]{4})_([0-9a-f]{4})_([0-9a-f]{12})")


def designed_sample_owner(name: str | None) -> str | None:
    """The named voice (its id) whose designed sample ``name`` is, by the name designed_sample_name gives."""
    if not name:
        return None
    match = _DESIGNED_NAME.fullmatch(name.strip().lower().replace("-", "_"))
    return "-".join(match.groups()) if match else None


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
        *,
        designed: dict | None = None,
        max_seconds: float | None = None,
    ) -> dict:
        """Store validated PCM WAV bytes; the supplied media type is discarded.

        ``designed`` marks a designed voice's sample (its metadata's ``designed`` block): it is left out of
        list_voices(), and only a save with ``designed`` replaces it. ``max_seconds`` is a stricter limit than the
        library's own.
        """
        safe_name = self._sanitize_name(name)
        if not audio_bytes:
            raise ValueError("Audio data is empty")
        wav_info = _wav_info(audio_bytes)
        if wav_info is None:
            raise ValueError(
                "Reference audio must be valid WAV format with at least one complete audio frame. "
                "Convert MP3/OGG/FLAC to WAV before uploading."
            )
        limits = [limit for limit in (self.max_seconds, max_seconds) if limit is not None and limit > 0]
        if limits and wav_info.duration_seconds > min(limits):
            raise ValueError(
                "Reference audio is too long "
                f"({wav_info.duration_seconds:.2f}s). Max: {min(limits):g}s"
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
        if designed is not None:
            metadata["designed"] = dict(designed)

        meta_path = self._meta_path(safe_name)
        audio_path = self.library_path / f"{safe_name}.audio.{ext}"

        with self._lock:
            self.library_path.mkdir(parents=True, exist_ok=True)
            if designed is None and self._designed(meta_path):
                raise DesignedSampleError(f"'{safe_name}' is a designed voice's sample")
            if designed is None and not meta_path.exists():
                self._check_room()

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
        """The recordings: every entry but designed voices' samples."""
        with self._lock:
            voices: list[dict] = []
            for meta_path in self.library_path.glob("*.meta.json"):
                try:
                    item = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not isinstance(item, dict) or "designed" in item:
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
        safe_name = self._lookup_name(name)
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

    def metadata(self, name: str) -> dict:
        """What get() returns besides the audio, without reading the audio file."""
        safe_name = self._lookup_name(name)
        with self._lock:
            meta_path = self._meta_path(safe_name)
            if not meta_path.exists():
                raise VoiceNotFoundError(name)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            ext = self._extension_for_content_type(metadata.get("content_type", "audio/wav"))
            if not (self.library_path / f"{safe_name}.audio.{ext}").exists():
                raise VoiceNotFoundError(name)
            return metadata

    def delete(self, name: str, *, designed: bool = False) -> None:
        """Delete an entry; a designed voice's sample only with ``designed``."""
        safe_name = self._lookup_name(name)
        with self._lock:
            meta_path = self._meta_path(safe_name)
            matched_audio = list(self.library_path.glob(f"{safe_name}.audio.*"))
            if not meta_path.exists() and not matched_audio:
                raise VoiceNotFoundError(name)
            if not designed and self._designed(meta_path):
                raise DesignedSampleError(f"'{safe_name}' is a designed voice's sample")

            meta_path.unlink(missing_ok=True)
            for p in matched_audio:
                p.unlink(missing_ok=True)

    def set_transcript(self, name: str, transcript: str | None) -> dict:
        """Update only a saved reference transcript and return its metadata."""
        safe_name = self._lookup_name(name)
        with self._lock:
            meta_path = self._meta_path(safe_name)
            if not meta_path.exists():
                raise VoiceNotFoundError(name)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if "designed" in metadata:
                raise DesignedSampleError(f"'{safe_name}' is a designed voice's sample")
            normalized_transcript = self._normalize_transcript(transcript)
            if normalized_transcript:
                metadata["transcript"] = normalized_transcript
            else:
                metadata.pop("transcript", None)
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

    def free_name(self, name: str) -> str:
        """A name no entry has, from ``name`` by the safe-name rules: name, name_2, name_3, ..."""
        try:
            base = self._sanitize_name(name)
        except ValueError:
            base = "voice"
        with self._lock:
            for n in range(1, 10_000):
                suffix = "" if n == 1 else f"_{n}"
                candidate = base[: 64 - len(suffix)] + suffix
                if not self._taken(candidate):
                    return candidate
        raise ValueError("No free voice name")

    def move(self, name: str, new_name: str, *, designed: dict | None) -> dict:
        """Rename an entry to ``new_name`` (which must be free), with ``designed`` as its designed block; None makes
        it an ordinary recording."""
        safe_name = self._lookup_name(name)
        new_safe = self._sanitize_name(new_name)
        with self._lock:
            meta_path = self._meta_path(safe_name)
            if not meta_path.exists():
                raise VoiceNotFoundError(name)
            if self._taken(new_safe):
                raise ValueError(f"Voice '{new_safe}' already exists")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if designed is None and "designed" in metadata:
                self._check_room()
            ext = self._extension_for_content_type(metadata.get("content_type", "audio/wav"))
            audio_path = self.library_path / f"{safe_name}.audio.{ext}"
            if not audio_path.exists():
                raise VoiceNotFoundError(name)
            metadata["name"] = new_safe
            metadata.pop("designed", None)
            if designed is not None:
                metadata["designed"] = dict(designed)
            new_audio = self.library_path / f"{new_safe}.audio.{ext}"
            new_meta = self._meta_path(new_safe)
            staged_meta = self.library_path / f".{new_safe}.{uuid4().hex}.meta.tmp"
            audio_path.replace(new_audio)
            try:
                staged_meta.write_text(json.dumps(metadata), encoding="utf-8")
                staged_meta.replace(new_meta)
            except Exception:
                new_audio.replace(audio_path)
                staged_meta.unlink(missing_ok=True)
                raise
            meta_path.unlink(missing_ok=True)
            return metadata

    def _check_room(self) -> None:
        """Room for one more recording (max_count 0 = unlimited); designed voices' samples don't count."""
        if self.max_count <= 0:
            return
        if sum(1 for path in self.library_path.glob("*.meta.json") if not self._designed(path)) >= self.max_count:
            raise ValueError(
                f"Voice library is full ({self.max_count} voices max). Delete a voice before adding more."
            )

    def _taken(self, safe_name: str) -> bool:
        return self._meta_path(safe_name).exists() or any(self.library_path.glob(f"{safe_name}.audio.*"))

    def _designed(self, meta_path: Path) -> bool:
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(metadata, dict) and "designed" in metadata

    def exists(self, name: str) -> bool:
        try:
            safe_name = self._lookup_name(name)
        except VoiceNotFoundError:
            return False
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

    def _lookup_name(self, name: str) -> str:
        # save() rejects a name with no usable characters, so no entry can exist under it.
        try:
            return self._sanitize_name(name)
        except ValueError:
            raise VoiceNotFoundError(name) from None

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
