from __future__ import annotations

import time

from src.cache.tts_cache import TTSCache


def test_make_key_stable():
    k1 = TTSCache.make_key("hello", "alloy", 1.0, "mp3")
    k2 = TTSCache.make_key("hello", "alloy", 1.0, "mp3")
    assert k1 == k2


def test_make_key_changes_with_inputs():
    base = TTSCache.make_key("hello", "alloy", 1.0, "mp3")
    assert TTSCache.make_key("hello!", "alloy", 1.0, "mp3") != base
    assert TTSCache.make_key("hello", "nova", 1.0, "mp3") != base
    assert TTSCache.make_key("hello", "alloy", 1.1, "mp3") != base
    assert TTSCache.make_key("hello", "alloy", 1.0, "wav") != base


def test_disabled_cache_returns_none(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=False)
    assert cache.get(text="a", voice="b", speed=1.0, fmt="wav") is None


def test_set_and_get(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="hello", voice="alloy", speed=1.0, fmt="wav", audio=b"abc")
    got = cache.get(text="hello", voice="alloy", speed=1.0, fmt="wav")
    assert got == b"abc"


def test_size_bytes(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="a", voice="v", speed=1.0, fmt="pcm", audio=b"12345")
    assert cache.size_bytes() >= 5


def test_lru_evicts_oldest(tmp_path):
    cache = TTSCache(str(tmp_path), max_size_mb=0, enabled=True)
    cache.max_size_bytes = 12
    cache.set(text="1", voice="v", speed=1.0, fmt="wav", audio=b"11111111")
    time.sleep(0.01)
    cache.set(text="2", voice="v", speed=1.0, fmt="wav", audio=b"22222222")
    assert cache.get(text="1", voice="v", speed=1.0, fmt="wav") is None
    assert cache.get(text="2", voice="v", speed=1.0, fmt="wav") == b"22222222"


def test_get_updates_lru(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.max_size_bytes = 16
    cache.set(text="1", voice="v", speed=1.0, fmt="wav", audio=b"11111111")
    cache.set(text="2", voice="v", speed=1.0, fmt="wav", audio=b"22222222")
    assert cache.get(text="1", voice="v", speed=1.0, fmt="wav") == b"11111111"
    # force eviction by adding another
    cache.set(text="3", voice="v", speed=1.0, fmt="wav", audio=b"33333333")
    assert cache.get(text="1", voice="v", speed=1.0, fmt="wav") == b"11111111"


def test_evict_if_needed_noop(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    removed = cache.evict_if_needed()
    assert removed == 0


def test_set_ignores_empty_audio(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    cache.set(text="x", voice="v", speed=1.0, fmt="wav", audio=b"")
    assert cache.size_bytes() == 0


def test_get_missing_file(tmp_path):
    cache = TTSCache(str(tmp_path), enabled=True)
    assert cache.get(text="missing", voice="v", speed=1.0, fmt="wav") is None
