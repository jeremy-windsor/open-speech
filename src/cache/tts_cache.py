from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSCache:
    """File-backed TTS response cache with LRU eviction."""

    def __init__(self, cache_dir: str, max_size_mb: int = 500, enabled: bool = False) -> None:
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(
        text: str,
        voice: str,
        speed: float,
        fmt: str,
        model: str,
        language: str | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "format": fmt,
                "language": language,
                "model": model,
                "speed": speed,
                "text": text,
                "voice": voice,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path_for(self, key: str, fmt: str) -> Path:
        return self.cache_dir / f"{key}.{fmt}"

    def get(
        self,
        *,
        text: str,
        voice: str,
        speed: float,
        fmt: str,
        model: str,
        language: str | None = None,
    ) -> bytes | None:
        if not self.enabled:
            return None
        key = self.make_key(text, voice, speed, fmt, model, language)
        path = self._path_for(key, fmt)
        with self._lock:
            if not path.exists():
                return None
            try:
                data = path.read_bytes()
            except OSError:
                return None
            now = time.time()
            try:
                os.utime(path, (now, now))
            except OSError:
                pass
            return data

    def set(
        self,
        *,
        text: str,
        voice: str,
        speed: float,
        fmt: str,
        model: str,
        audio: bytes,
        language: str | None = None,
    ) -> None:
        if not self.enabled or not audio:
            return
        key = self.make_key(text, voice, speed, fmt, model, language)
        path = self._path_for(key, fmt)
        temp_path = self.cache_dir / (
            f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with self._lock:
            try:
                temp_path.write_bytes(audio)
                temp_path.replace(path)
            except OSError:
                logger.warning("Failed to write TTS cache entry %s", path, exc_info=True)
                return
            finally:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            try:
                self.evict_if_needed()
            except OSError:
                logger.warning("Failed to evict TTS cache entries", exc_info=True)

    def size_bytes(self) -> int:
        with self._lock:
            if not self.cache_dir.exists():
                return 0
            return sum(p.stat().st_size for p in self.cache_dir.glob("*.*") if p.is_file())

    def evict_if_needed(self) -> int:
        with self._lock:
            if not self.enabled or not self.cache_dir.exists():
                return 0
            files = [p for p in self.cache_dir.glob("*.*") if p.is_file()]
            total = sum(p.stat().st_size for p in files)
            if total <= self.max_size_bytes:
                return 0
            files.sort(key=lambda p: p.stat().st_mtime)  # oldest first
            removed = 0
            for p in files:
                if total <= self.max_size_bytes:
                    break
                try:
                    size = p.stat().st_size
                except FileNotFoundError:
                    continue
                p.unlink(missing_ok=True)
                total -= size
                removed += 1
            return removed


def cache_cleanup_loop(cache: TTSCache, interval_s: int = 30):
    """Background loop helper for periodic cache cleanup."""
    while True:
        cache.evict_if_needed()
        time.sleep(interval_s)
