"""Tests for Fish Speech backend (mocked package)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_fish_speech():
    """Mock fish_speech package."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_fish = MagicMock()
    mock_engine = MagicMock()
    mock_engine.synthesize.return_value = np.random.randn(24000).astype(np.float32)
    mock_fish.inference.TTSInference.return_value = mock_engine

    with patch.dict(sys.modules, {
        "torch": mock_torch,
        "fish_speech": mock_fish,
        "fish_speech.inference": mock_fish.inference,
    }):
        yield mock_engine


class TestFishSpeechBackend:
    def test_init(self):
        from src.tts.backends.fish_speech_backend import FishSpeechBackend
        backend = FishSpeechBackend(device="cpu")
        assert backend.name == "fish-speech"
        assert backend.sample_rate == 24000

    def test_load_unload(self):
        from src.tts.backends.fish_speech_backend import FishSpeechBackend
        backend = FishSpeechBackend(device="cpu")
        backend.load_model("fish-speech-1.5")
        assert backend.is_model_loaded("fish-speech-1.5")
        assert len(backend.loaded_models()) == 1
        backend.unload_model("fish-speech-1.5")
        assert not backend.is_model_loaded("fish-speech-1.5")

    def test_load_unknown_model(self):
        from src.tts.backends.fish_speech_backend import FishSpeechBackend
        backend = FishSpeechBackend(device="cpu")
        with pytest.raises(ValueError, match="Unknown Fish Speech"):
            backend.load_model("fish-speech-99")

    def test_synthesize_not_loaded(self):
        from src.tts.backends.fish_speech_backend import FishSpeechBackend
        backend = FishSpeechBackend(device="cpu")
        with pytest.raises(RuntimeError, match="No Fish Speech model loaded"):
            list(backend.synthesize("hello", "default"))

    def test_synthesize(self, mock_fish_speech):
        from src.tts.backends.fish_speech_backend import FishSpeechBackend
        backend = FishSpeechBackend(device="cpu")
        backend.load_model("fish-speech-1.5")
        chunks = list(backend.synthesize("hello world", "default"))
        assert len(chunks) == 1
        assert chunks[0].dtype == np.float32

    def test_list_voices(self):
        from src.tts.backends.fish_speech_backend import FishSpeechBackend
        backend = FishSpeechBackend(device="cpu")
        voices = backend.list_voices()
        assert len(voices) >= 1
        assert voices[0].id == "default"

    def test_loaded_models_info(self):
        from src.tts.backends.fish_speech_backend import FishSpeechBackend
        backend = FishSpeechBackend(device="cpu")
        backend.load_model("fish-speech-1.5")
        info = backend.loaded_models()
        assert info[0].backend == "fish-speech"
