from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from stat import S_ISREG

logger = logging.getLogger(__name__)

# PCM entries hold 24 kHz audio whatever the model's native rate, because every
# transport that shares the cache directory reads them back at this rate.
PCM_SAMPLE_RATE = 24000

# A writer renames its temp file into place within one write; a temp file this old was
# left by a process that died in between, and eviction removes it.
STALE_TEMP_FILE_S = 600


class TTSCache:
    """File-backed TTS response cache with LRU eviction."""

    def __init__(self, cache_dir: str, max_size_mb: int = 500, enabled: bool = False) -> None:
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
        # Bytes stored at the last directory scan plus every write since. A write rescans,
        # under the lock every lookup waits on, only once this passes the cap.
        self._size_estimate: int | None = None
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
        *,
        trim: bool = True,
        normalize: bool = True,
    ) -> str:
        # Entries outlive restarts, so the post-processing settings that shaped them are keyed too.
        payload = json.dumps(
            {
                "format": f"pcm_{PCM_SAMPLE_RATE}" if fmt == "pcm" else fmt,
                "language": language,
                "model": model,
                "normalize": normalize,
                "speed": speed,
                "text": text,
                "trim": trim,
                "voice": voice,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", "surrogatepass")  # JSON text may carry a lone UTF-16 surrogate
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
        trim: bool = True,
        normalize: bool = True,
    ) -> bytes | None:
        if not self.enabled:
            return None
        key = self.make_key(text, voice, speed, fmt, model, language, trim=trim, normalize=normalize)
        path = self._path_for(key, fmt)
        with self._lock:
            try:
                data = path.read_bytes()
            except OSError:  # missing, or unreadable: either way a miss, never a failed request
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
        trim: bool = True,
        normalize: bool = True,
    ) -> None:
        if not self.enabled or not audio:
            return
        key = self.make_key(text, voice, speed, fmt, model, language, trim=trim, normalize=normalize)
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
            if self._size_estimate is not None:
                self._size_estimate += len(audio)
            if self._size_estimate is not None and self._size_estimate <= self.max_size_bytes:
                return
            try:
                self.evict_if_needed()
            except OSError:
                logger.warning("Failed to evict TTS cache entries", exc_info=True)

    def _entries(self, *, remove_stale_temp_files: bool = False) -> list[tuple[Path, os.stat_result]]:
        """Stored entries with their stat; another writer's in-flight ".tmp" file is not one."""
        entries = []
        stale_before = time.time() - STALE_TEMP_FILE_S
        for path in self.cache_dir.glob("*.*"):
            try:
                info = path.stat()
            except OSError:  # removed by another writer since the listing
                continue
            if not S_ISREG(info.st_mode):
                continue
            if path.name.startswith("."):
                if remove_stale_temp_files and path.suffix == ".tmp" and info.st_mtime < stale_before:
                    with suppress(OSError):
                        path.unlink()
                continue
            entries.append((path, info))
        return entries

    def size_bytes(self) -> int:
        with self._lock:
            if not self.cache_dir.exists():
                return 0
            return sum(info.st_size for _path, info in self._entries())

    def evict_if_needed(self) -> int:
        with self._lock:
            if not self.enabled or not self.cache_dir.exists():
                return 0
            entries = self._entries(remove_stale_temp_files=True)
            total = sum(info.st_size for _path, info in entries)
            if total <= self.max_size_bytes:
                self._size_estimate = total
                return 0
            entries.sort(key=lambda entry: entry[1].st_mtime)  # oldest first
            removed = 0
            for path, info in entries:
                if total <= self.max_size_bytes:
                    break
                path.unlink(missing_ok=True)
                total -= info.st_size
                removed += 1
            self._size_estimate = total
            return removed


_shared_caches: dict[str, TTSCache] = {}
_shared_caches_lock = threading.Lock()


def get_shared_tts_cache(cache_dir: str, max_size_mb: int = 500, enabled: bool = False) -> TTSCache:
    """Return the process-wide cache for ``cache_dir``, creating it on the first call.

    HTTP speech and Wyoming share cache entries, so they must share one instance: its lock
    orders every write and eviction in the directory. The first call's settings apply.
    """
    key = os.path.normcase(os.path.abspath(cache_dir))
    with _shared_caches_lock:
        cache = _shared_caches.get(key)
        if cache is None:
            cache = _shared_caches[key] = TTSCache(cache_dir, max_size_mb, enabled)
        return cache


def cache_cleanup_loop(cache: TTSCache, interval_s: int = 30):
    """Background loop helper for periodic cache cleanup."""
    while True:
        cache.evict_if_needed()
        time.sleep(interval_s)
