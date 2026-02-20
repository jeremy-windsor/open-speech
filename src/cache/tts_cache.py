from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path


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
    def make_key(text: str, voice: str, speed: float, fmt: str, model: str) -> str:
        payload = f"{text}{voice}{speed}{fmt}{model}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path_for(self, key: str, fmt: str) -> Path:
        return self.cache_dir / f"{key}.{fmt}"

    def get(self, *, text: str, voice: str, speed: float, fmt: str, model: str) -> bytes | None:
        if not self.enabled:
            return None
        key = self.make_key(text, voice, speed, fmt, model)
        path = self._path_for(key, fmt)
        if not path.exists():
            return None
        with self._lock:
            data = path.read_bytes()
            now = time.time()
            os.utime(path, (now, now))
            return data

    def set(self, *, text: str, voice: str, speed: float, fmt: str, model: str, audio: bytes) -> None:
        if not self.enabled or not audio:
            return
        key = self.make_key(text, voice, speed, fmt, model)
        path = self._path_for(key, fmt)
        with self._lock:
            path.write_bytes(audio)
            self.evict_if_needed()

    def size_bytes(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self.cache_dir.glob("*.*") if p.is_file())

    def evict_if_needed(self) -> int:
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
            size = p.stat().st_size
            p.unlink(missing_ok=True)
            total -= size
            removed += 1
        return removed


def cache_cleanup_loop(cache: TTSCache, interval_s: int = 30):
    """Background loop helper for periodic cache cleanup."""
    while True:
        cache.evict_if_needed()
        time.sleep(interval_s)
