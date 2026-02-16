"""Tests for TTSRouter register_backend and auto-discovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo
from src.tts.router import TTSRouter, _discover_backends


class FakeBackend:
    """Minimal TTSBackend implementation for testing."""
    name: str = "fake"
    sample_rate: int = 16000

    def __init__(self, device: str = "cpu") -> None:
        self._loaded = False

    def load_model(self, model_id: str) -> None:
        self._loaded = True

    def unload_model(self, model_id: str) -> None:
        self._loaded = False

    def is_model_loaded(self, model_id: str) -> bool:
        return self._loaded

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        return []

    def synthesize(self, text, voice, speed=1.0, lang_code=None):
        yield np.zeros(100, dtype=np.float32)

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="fake_voice", name="Fake")]


class TestRegisterBackend:
    def test_register_and_use(self):
        router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        assert "fake" in router.list_backends()
        backend = router.get_backend("fake")
        assert backend is fake

    def test_register_sets_default_when_empty(self):
        """If no backends, registering one sets it as default."""
        router = TTSRouter.__new__(TTSRouter)
        router._backends = {}
        router._default_backend = None
        router._device = "cpu"

        fake = FakeBackend()
        router.register_backend("fake", fake)
        # Should be usable as default
        backend = router.get_backend("nonexistent")
        assert backend is fake

    def test_list_backends(self):
        router = TTSRouter(device="cpu")
        names = router.list_backends()
        assert "kokoro" in names

    def test_voices_from_registered_backend(self):
        router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        voices = router.list_voices("fake")
        assert any(v.id == "fake_voice" for v in voices)

    def test_aggregate_voices(self):
        router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        all_voices = router.list_voices()
        ids = [v.id for v in all_voices]
        assert "fake_voice" in ids
        # Should also have kokoro voices
        assert any(v.startswith("af_") for v in ids)


class TestAutoDiscovery:
    def test_discovers_kokoro(self):
        """Auto-discovery should find the kokoro backend."""
        discovered = _discover_backends()
        assert "kokoro" in discovered
