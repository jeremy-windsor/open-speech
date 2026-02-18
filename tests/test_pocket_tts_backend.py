"""Tests for Pocket TTS backend."""

from __future__ import annotations

import sys
from importlib import reload
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class _FakeChunk:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeTTSModel:
    sample_rate = 24000
    device = "cpu"

    def __init__(self):
        self._state_calls = []

    @classmethod
    def load_model(cls):
        return cls()

    def get_state_for_audio_prompt(self, voice: str):
        self._state_calls.append(voice)
        return {"voice": voice}

    def generate_audio_stream(self, model_state, text: str):
        assert model_state["voice"]
        assert text
        yield _FakeChunk(np.zeros(400, dtype=np.float32))
        yield _FakeChunk(np.ones(200, dtype=np.float32) * 0.25)


@pytest.fixture
def backend_module_with_mock_pocket():
    fake_mod = MagicMock()
    fake_mod.TTSModel = _FakeTTSModel

    with patch.dict(sys.modules, {"pocket_tts": fake_mod}):
        import src.tts.backends.pocket_tts_backend as mod

        reload(mod)
        yield mod


def test_capabilities_declared(backend_module_with_mock_pocket):
    mod = backend_module_with_mock_pocket
    caps = mod.PocketTTSBackend.capabilities

    assert caps["voice_blend"] is False
    assert caps["voice_clone"] is False
    assert caps["voice_design"] is False
    assert caps["streaming"] is True
    assert caps["speed_control"] is False
    assert caps["languages"] == ["en"]
    assert len(caps["speakers"]) == 8


def test_provider_missing_behavior():
    with patch.dict(sys.modules, {"pocket_tts": None}):
        import src.tts.backends.pocket_tts_backend as mod

        reload(mod)
        backend = mod.PocketTTSBackend(device="cpu")
        with pytest.raises(RuntimeError, match="pocket-tts"):
            backend.load_model("pocket-tts")


def test_load_unload_generate_flow(backend_module_with_mock_pocket):
    mod = backend_module_with_mock_pocket
    backend = mod.PocketTTSBackend(device="cpu")

    assert backend.loaded_models() == []
    assert not backend.is_model_loaded("pocket-tts")

    backend.load_model("pocket-tts")
    assert backend.is_model_loaded("pocket-tts")

    chunks = list(backend.synthesize("Hello from Pocket", "alba"))
    assert len(chunks) == 2
    assert all(isinstance(c, np.ndarray) for c in chunks)
    assert all(c.dtype == np.float32 for c in chunks)

    loaded = backend.loaded_models()
    assert len(loaded) == 1
    assert loaded[0].model == "pocket-tts"
    assert loaded[0].backend == "pocket-tts"

    backend.unload_model("pocket-tts")
    assert not backend.is_model_loaded("pocket-tts")


def test_lazy_load_on_synthesize(backend_module_with_mock_pocket):
    mod = backend_module_with_mock_pocket
    backend = mod.PocketTTSBackend(device="cpu")

    chunks = list(backend.synthesize("Hello", "unknown-voice"))
    assert len(chunks) == 2
    # unknown voice should fall back to alba
    model_info = backend._models["pocket-tts"]
    fake_model = model_info["model"]
    assert fake_model._state_calls[0] == "alba"
