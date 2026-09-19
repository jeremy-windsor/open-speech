"""Tests for TTSRouter register_backend and auto-discovery."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest

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


class ConcurrentBackend(FakeBackend):
    def __init__(self, name: str, barrier: threading.Barrier) -> None:
        super().__init__()
        self.name = name
        self._barrier = barrier

    def supports_model(self, model_id: str) -> bool:
        return model_id == self.name

    def synthesize(self, text, voice, speed=1.0, lang_code=None):
        self._barrier.wait(timeout=2)
        yield np.zeros(10, dtype=np.float32)


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

    def test_unknown_model_does_not_fall_back_to_registered_default(self):
        router = TTSRouter.__new__(TTSRouter)
        router._backends = {}
        router._default_backend = None
        router._device = "cpu"

        fake = FakeBackend()
        router.register_backend("fake", fake)
        with pytest.raises(ValueError, match="Unknown TTS model or backend"):
            router.get_backend("nonexistent")

    def test_unknown_prefixed_model_is_rejected_when_backend_declines_it(self):
        router = TTSRouter.__new__(TTSRouter)
        router._backends = {}
        router._default_backend = None
        router._device = "cpu"

        fake = FakeBackend()
        fake.supports_model = lambda model_id: model_id == "fake/known"
        router.register_backend("fake", fake)

        assert router.get_backend("fake/known") is fake
        with pytest.raises(ValueError, match="fake/unknown"):
            router.get_backend("fake/unknown")

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

    def test_input_limit_is_model_specific_and_optional(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)
        assert router.max_input_chars_for("fake") is None

        fake.get_model_manifest = lambda _model_id: type("Manifest", (), {"max_input_chars": 8})()
        assert router.max_input_chars_for("fake") == 8

    def test_different_providers_can_synthesize_concurrently(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        barrier = threading.Barrier(2)
        router.register_backend("alpha", ConcurrentBackend("alpha", barrier))
        router.register_backend("beta", ConcurrentBackend("beta", barrier))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(lambda: list(router.synthesize("hello", "alpha", "voice"))),
                pool.submit(lambda: list(router.synthesize("hello", "beta", "voice"))),
            ]
            for future in futures:
                assert len(future.result(timeout=3)) == 1


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
