from __future__ import annotations

import os
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
