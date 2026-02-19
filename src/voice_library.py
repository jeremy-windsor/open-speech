"""Persistent voice reference library for cloning."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


class VoiceNotFoundError(KeyError):
    """Raised when a named voice entry does not exist."""


class VoiceLibraryManager:
    def __init__(self, library_path: str | Path) -> None:
        self.library_path = Path(library_path)
        self._lock = threading.RLock()
        with self._lock:
            self.library_path.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, audio_bytes: bytes, content_type: str = "audio/wav") -> dict:
        safe_name = self._sanitize_name(name)
        ext = self._extension_for_content_type(content_type)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "name": safe_name,
            "size_bytes": len(audio_bytes),
            "content_type": content_type,
            "created_at": created_at,
        }

        meta_path = self._meta_path(safe_name)
        audio_path = self.library_path / f"{safe_name}.audio.{ext}"

        with self._lock:
            self.library_path.mkdir(parents=True, exist_ok=True)
            for existing in self.library_path.glob(f"{safe_name}.audio.*"):
                if existing != audio_path:
                    existing.unlink(missing_ok=True)
            audio_path.write_bytes(audio_bytes)
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")

        return metadata

    def list_voices(self) -> list[dict]:
        with self._lock:
            voices: list[dict] = []
            for meta_path in self.library_path.glob("*.meta.json"):
                try:
                    item = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(item, dict):
                        voices.append(item)
                except Exception:
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

    def exists(self, name: str) -> bool:
        safe_name = self._sanitize_name(name)
        with self._lock:
            return self._meta_path(safe_name).exists()

    def _meta_path(self, safe_name: str) -> Path:
        return self.library_path / f"{safe_name}.meta.json"

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
