from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path

from src.cache.tts_cache import TTSCache


def _entry_path(cache: TTSCache, text: str) -> Path:
    key = cache.make_key(text, "v", 1.0, "wav", "kokoro")
    return cache._path_for(key, "wav")


def test_make_key_stable():
    k1 = TTSCache.make_key("hello", "alloy", 1.0, "mp3", "kokoro")
    k2 = TTSCache.make_key("hello", "alloy", 1.0, "mp3", "kokoro")
    assert k1 == k2


def test_make_key_changes_with_inputs():
    base = TTSCache.make_key("hello", "alloy", 1.0, "mp3", "kokoro")
    assert TTSCache.make_key("hello!", "alloy", 1.0, "mp3", "kokoro") != base
    assert TTSCache.make_key("hello", "nova", 1.0, "mp3", "kokoro") != base
    assert TTSCache.make_key("hello", "alloy", 1.1, "mp3", "kokoro") != base
    assert TTSCache.make_key("hello", "alloy", 1.0, "wav", "kokoro") != base
    assert TTSCache.make_key("hello", "alloy", 1.0, "mp3", "piper/en_US-ryan-medium") != base


def test_make_key_distinguishes_adjacent_field_boundaries():
    first = TTSCache.make_key("ab", "c", 1.0, "wav", "kokoro")
    second = TTSCache.make_key("a", "bc", 1.0, "wav", "kokoro")

    assert first != second


def test_make_key_distinguishes_language():
    english = TTSCache.make_key("hello", "alloy", 1.0, "wav", "kokoro", "en")
    french = TTSCache.make_key("hello", "alloy", 1.0, "wav", "kokoro", "fr")

    assert english != french


def test_make_key_distinguishes_post_processing():
    base = TTSCache.make_key("hello", "alloy", 1.0, "wav", "kokoro")

    # Callers that pass no post-processing settings keep the defaults' (trimmed, normalized) key.
    assert TTSCache.make_key("hello", "alloy", 1.0, "wav", "kokoro", trim=True, normalize=True) == base
    assert TTSCache.make_key("hello", "alloy", 1.0, "wav", "kokoro", trim=False) != base
    assert TTSCache.make_key("hello", "alloy", 1.0, "wav", "kokoro", normalize=False) != base


def test_disabled_cache_returns_none(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=False)
    assert cache.get(text="a", voice="b", speed=1.0, fmt="wav", model="kokoro") is None


def test_set_and_get(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="hello", voice="alloy", speed=1.0, fmt="wav", audio=b"abc", model="kokoro")
    got = cache.get(text="hello", voice="alloy", speed=1.0, fmt="wav", model="kokoro")
    assert got == b"abc"


def test_size_bytes(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="a", voice="v", speed=1.0, fmt="pcm", audio=b"12345", model="kokoro")
    assert cache.size_bytes() >= 5


def test_lru_evicts_oldest(tmp_path):
    cache = TTSCache(str(tmp_path), max_size_mb=0, enabled=True)
    cache.max_size_bytes = 12
    cache.set(text="1", voice="v", speed=1.0, fmt="wav", audio=b"11111111", model="kokoro")
    os.utime(_entry_path(cache, "1"), (1, 1))
    cache.set(text="2", voice="v", speed=1.0, fmt="wav", audio=b"22222222", model="kokoro")
    assert cache.get(text="1", voice="v", speed=1.0, fmt="wav", model="kokoro") is None
    assert cache.get(text="2", voice="v", speed=1.0, fmt="wav", model="kokoro") == b"22222222"


def test_get_updates_lru(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.max_size_bytes = 16
    cache.set(text="1", voice="v", speed=1.0, fmt="wav", audio=b"11111111", model="kokoro")
    cache.set(text="2", voice="v", speed=1.0, fmt="wav", audio=b"22222222", model="kokoro")
    os.utime(_entry_path(cache, "1"), (1, 1))
    os.utime(_entry_path(cache, "2"), (2, 2))
    assert cache.get(text="1", voice="v", speed=1.0, fmt="wav", model="kokoro") == b"11111111"
    # force eviction by adding another
    cache.set(text="3", voice="v", speed=1.0, fmt="wav", audio=b"33333333", model="kokoro")
    assert cache.get(text="1", voice="v", speed=1.0, fmt="wav", model="kokoro") == b"11111111"


def test_evict_if_needed_noop(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    removed = cache.evict_if_needed()
    assert removed == 0


def test_set_ignores_empty_audio(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="x", voice="v", speed=1.0, fmt="wav", audio=b"", model="kokoro")
    assert cache.size_bytes() == 0


def test_get_missing_file(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    assert cache.get(text="missing", voice="v", speed=1.0, fmt="wav", model="kokoro") is None


def test_get_treats_concurrent_file_removal_as_cache_miss(tmp_path, monkeypatch):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="hello", voice="v", speed=1.0, fmt="wav", audio=b"audio", model="kokoro")

    def remove_before_read(path):
        path.unlink()
        raise FileNotFoundError(path)

    monkeypatch.setattr(type(tmp_path), "read_bytes", remove_before_read)

    assert cache.get(text="hello", voice="v", speed=1.0, fmt="wav", model="kokoro") is None


def test_get_treats_unreadable_entry_as_cache_miss(tmp_path, monkeypatch):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="hello", voice="v", speed=1.0, fmt="wav", audio=b"audio", model="kokoro")

    def fail_read(_path):
        raise PermissionError("unreadable cache entry")

    monkeypatch.setattr(type(tmp_path), "read_bytes", fail_read)

    assert cache.get(text="hello", voice="v", speed=1.0, fmt="wav", model="kokoro") is None


def test_failed_cache_write_preserves_existing_entry(tmp_path, monkeypatch):
    cache = TTSCache(str(tmp_path), enabled=True)
    kwargs = dict(text="hello", voice="v", speed=1.0, fmt="wav", model="kokoro")
    cache.set(**kwargs, audio=b"complete audio")
    original_write_bytes = type(tmp_path).write_bytes

    def fail_temp_write(path, data):
        if path.suffix == ".tmp":
            with path.open("wb") as temp_file:
                temp_file.write(data[:3])
            raise OSError("simulated interrupted write")
        return original_write_bytes(path, data)

    monkeypatch.setattr(type(tmp_path), "write_bytes", fail_temp_write)

    cache.set(**kwargs, audio=b"replacement audio")

    assert cache.get(**kwargs) == b"complete audio"
    assert list(tmp_path.glob("*.tmp")) == []


def test_eviction_failure_does_not_fail_cache_write(tmp_path, monkeypatch):
    cache = TTSCache(str(tmp_path), enabled=True)

    def fail_eviction():
        raise PermissionError("cache cleanup denied")

    monkeypatch.setattr(cache, "evict_if_needed", fail_eviction)

    cache.set(
        text="hello",
        voice="v",
        speed=1.0,
        fmt="wav",
        model="kokoro",
        audio=b"audio",
    )

    assert cache.get(text="hello", voice="v", speed=1.0, fmt="wav", model="kokoro") == b"audio"


def test_get_treats_a_directory_it_cannot_search_as_cache_miss(tmp_path, monkeypatch):
    # A TTS_CACHE_DIR the service user cannot search fails stat() with EACCES, which
    # Path.exists re-raises on Python 3.12; the request must synthesize, not fail.
    cache = TTSCache(str(tmp_path), enabled=True)

    def denied(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(type(tmp_path), "stat", denied)
    monkeypatch.setattr(type(tmp_path), "read_bytes", denied)

    assert cache.get(text="hello", voice="v", speed=1.0, fmt="wav", model="kokoro") is None


def test_eviction_leaves_another_writers_temp_file_alone(tmp_path):
    # set() writes ".<key>.<pid>.<thread>.tmp" and renames it into place; eviction in
    # another process sharing the directory must not count or delete it meanwhile.
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.max_size_bytes = 0
    in_flight = tmp_path / ".abc.123.456.tmp"
    in_flight.write_bytes(b"partial audio")
    (tmp_path / "old.wav").write_bytes(b"cached audio")

    assert cache.evict_if_needed() == 1
    assert in_flight.read_bytes() == b"partial audio"
    assert not (tmp_path / "old.wav").exists()
    assert cache.size_bytes() == 0


def test_eviction_removes_a_temp_file_left_by_a_writer_that_died(tmp_path):
    # A writer killed between write and rename leaves its temp file behind. It is not an
    # entry, so the cap would never count it and nothing else would remove it.
    cache = TTSCache(str(tmp_path), enabled=True)
    orphan = tmp_path / ".abc.123.456.tmp"
    orphan.write_bytes(b"partial audio")
    day_ago = time.time() - 24 * 3600
    os.utime(orphan, (day_ago, day_ago))
    in_flight = tmp_path / ".def.123.789.tmp"
    in_flight.write_bytes(b"partial audio")

    cache.evict_if_needed()

    assert not orphan.exists()
    assert in_flight.read_bytes() == b"partial audio"


def test_write_under_the_cap_does_not_rescan_the_directory(tmp_path, monkeypatch):
    # Every lookup waits on the lock a write holds, so a write that leaves the cache under
    # its cap must not hold it for a scan of every entry.
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.max_size_bytes = 20
    kwargs = dict(voice="v", speed=1.0, fmt="wav", model="kokoro")
    cache.set(text="1", **kwargs, audio=b"11111111")
    scans = []
    real_entries = cache._entries

    def counted_entries(**entry_kwargs):
        scans.append(1)
        return real_entries(**entry_kwargs)

    monkeypatch.setattr(cache, "_entries", counted_entries)

    cache.set(text="2", **kwargs, audio=b"22222222")
    assert scans == []

    os.utime(_entry_path(cache, "1"), (1, 1))
    cache.set(text="3", **kwargs, audio=b"33333333")
    assert len(scans) == 1
    assert cache.get(text="1", **kwargs) is None
    assert cache.get(text="2", **kwargs) == b"22222222"
    assert cache.get(text="3", **kwargs) == b"33333333"


def test_shared_cache_is_one_instance_per_directory(tmp_path):
    # HTTP speech and Wyoming share cache entries, so they share one instance and its lock.
    from src.cache.tts_cache import get_shared_tts_cache

    shared = get_shared_tts_cache(str(tmp_path / "a"), 10, True)
    kwargs = dict(text="hello", voice="v", speed=1.0, fmt="wav", model="kokoro")
    shared.set(**kwargs, audio=b"audio")

    same = get_shared_tts_cache(os.path.join(str(tmp_path), "b", "..", "a"), 10, True)
    assert same is shared
    assert same.get(**kwargs) == b"audio"
    assert get_shared_tts_cache(str(tmp_path / "b"), 10, True) is not shared


async def test_tts_cache_cleanup_loop_survives_a_failed_pass(monkeypatch):
    # The periodic eviction (TTS_CACHE_CLEANUP_INTERVAL_S) used to end on its first error,
    # leaving the cache to grow past TTS_CACHE_MAX_MB for the rest of the process.
    import src.main as main_mod

    calls = []

    def evict_if_needed():
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError("removed by another cache instance")

    monkeypatch.setattr(main_mod, "TTS_CACHE_CLEANUP_INTERVAL_S", 0)
    monkeypatch.setattr(main_mod.tts_cache, "evict_if_needed", evict_if_needed)
    task = asyncio.create_task(main_mod._tts_cache_cleanup_loop())
    try:
        async with asyncio.timeout(5):
            while len(calls) < 2:
                await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
