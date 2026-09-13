"""Tests for TTSRouter register_backend and auto-discovery."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo
from src.tts.backends.kokoro import KokoroBackend
from src.tts.router import TTSRouter, _discover_backends


class FakeBackend:
    """Minimal TTSBackend implementation for testing."""
    name: str = "fake"
    sample_rate: int = 16000

    @classmethod
    def is_available(cls) -> bool:
        return True

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


class ModelRateBackend(FakeBackend):
    def get_sample_rate(self, model_id: str) -> int:
        return 16000 if model_id == "fake/low-rate" else 22050


class UnavailableBackend(FakeBackend):
    name = "optional"

    @classmethod
    def is_available(cls) -> bool:
        return False


class AvailableBackend(FakeBackend):
    name = "available"


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
        with patch(
            "src.tts.router._discover_backends",
            return_value={"available": AvailableBackend},
        ):
            router = TTSRouter(device="cpu")

        assert router.list_backends() == ["available"]

    def test_voices_from_registered_backend(self):
        router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        voices = router.list_voices("fake")
        assert any(v.id == "fake_voice" for v in voices)

    def test_aggregate_voices(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        second = FakeBackend()
        second.list_voices = lambda: [VoiceInfo(id="second_voice", name="Second")]
        router.register_backend("second", second)

        all_voices = router.list_voices()
        ids = [v.id for v in all_voices]
        assert "fake_voice" in ids
        assert "second_voice" in ids

    def test_sample_rate_uses_model_specific_backend_lookup(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        router.register_backend("fake", ModelRateBackend())

        assert router.sample_rate_for("fake/low-rate") == 16000
        assert router.sample_rate_for("fake/medium-rate") == 22050

    def test_sample_rate_falls_back_to_backend_attribute(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        router.register_backend("fake", FakeBackend())

        assert router.sample_rate_for("fake") == 16000


class TestBackendAvailability:
    def test_skips_unavailable_backends(self, caplog):
        caplog.set_level("INFO")
        with patch(
            "src.tts.router._discover_backends",
            return_value={"optional": UnavailableBackend, "available": AvailableBackend},
        ):
            router = TTSRouter(device="cpu")

        assert "available" in router.list_backends()
        assert "optional" not in router.list_backends()
        assert any("Skipping TTS backend optional" in rec.message for rec in caplog.records)

    def test_kokoro_reports_unavailable_when_optional_package_is_missing(self):
        with patch.dict("sys.modules", {"kokoro": None}):
            assert KokoroBackend.is_available() is False


class TestAutoDiscovery:
    def test_discovers_kokoro(self):
        """Auto-discovery should find the kokoro backend."""
        discovered = _discover_backends()
        assert "kokoro" in discovered

    def test_discovers_pocket_tts(self):
        """Auto-discovery should find the pocket-tts backend module."""
        discovered = _discover_backends()
        assert "pocket-tts" in discovered
