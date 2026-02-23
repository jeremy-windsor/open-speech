"""Tests for Piper TTS backend — mocked piper-tts package."""

from __future__ import annotations

import io
import struct
import sys
import wave
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

# Mock the piper package before importing the backend
_mock_piper = MagicMock()
_mock_piper_voice = MagicMock()
_mock_piper.PiperVoice = _mock_piper_voice
sys.modules.setdefault("piper", _mock_piper)

# Mock piper.config.SynthesisConfig so the backend can import it without
# the real piper package being present.
@dataclass
class _MockSynthesisConfig:
    speaker_id: Optional[int] = None
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w_scale: Optional[float] = None
    normalize_audio: bool = True
    volume: float = 1.0

_mock_piper_config = MagicMock()
_mock_piper_config.SynthesisConfig = _MockSynthesisConfig
sys.modules.setdefault("piper.config", _mock_piper_config)


@dataclass
class _MockAudioChunk:
    """Minimal AudioChunk stand-in for unit tests."""
    audio_float_array: np.ndarray
    sample_rate: int = 22050
    sample_width: int = 2
    sample_channels: int = 1
    phonemes: list = field(default_factory=list)
    phoneme_ids: list = field(default_factory=list)

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo
from src.tts.backends.piper_backend import (
    PiperBackend,
    PIPER_MODELS,
    _hf_path_for_model,
)


class TestHFPathForModel:
    def test_lessac_medium(self):
        onnx, json_ = _hf_path_for_model("en_US-lessac-medium")
        assert onnx == "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        assert json_ == "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

    def test_alan_medium(self):
        onnx, json_ = _hf_path_for_model("en_GB-alan-medium")
        assert onnx == "en/en_GB/alan/medium/en_GB-alan-medium.onnx"

    def test_lessac_high(self):
        onnx, _ = _hf_path_for_model("en_US-lessac-high")
        assert "high" in onnx


class TestPiperModelsRegistry:
    def test_all_models_have_required_fields(self):
        for model_id, meta in PIPER_MODELS.items():
            assert model_id.startswith("piper/")
            assert "name" in meta
            assert "lang" in meta
            assert "sample_rate" in meta
            assert meta["sample_rate"] > 0

    def test_known_models_count(self):
        assert len(PIPER_MODELS) == 30


class TestPiperBackendInterface:
    """Test that PiperBackend satisfies the TTSBackend protocol shape."""

    def test_has_required_attributes(self):
        backend = PiperBackend()
        assert hasattr(backend, "name")
        assert hasattr(backend, "sample_rate")
        assert backend.name == "piper"
        assert backend.sample_rate == 22050

    def test_no_models_loaded_initially(self):
        backend = PiperBackend()
        assert backend.loaded_models() == []
        assert not backend.is_model_loaded("piper/en_US-lessac-medium")

    def test_list_voices_empty_when_no_models(self):
        backend = PiperBackend()
        assert backend.list_voices() == []


class TestPiperBackendLoadUnload:
    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_load_model(self, mock_download):
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")
        _mock_piper_voice.load.return_value = MagicMock()

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-medium")

        assert backend.is_model_loaded("piper/en_US-lessac-medium")
        loaded = backend.loaded_models()
        assert len(loaded) == 1
        assert loaded[0].model == "piper/en_US-lessac-medium"
        assert loaded[0].backend == "piper"

    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_unload_model(self, mock_download):
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")
        _mock_piper_voice.load.return_value = MagicMock()

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-medium")
        assert backend.is_model_loaded("piper/en_US-lessac-medium")

        backend.unload_model("piper/en_US-lessac-medium")
        assert not backend.is_model_loaded("piper/en_US-lessac-medium")
        assert backend.loaded_models() == []

    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_load_idempotent(self, mock_download):
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")
        _mock_piper_voice.load.return_value = MagicMock()

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-medium")
        backend.load_model("piper/en_US-lessac-medium")
        assert len(backend.loaded_models()) == 1


class TestPiperBackendSynthesize:
    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_synthesize_yields_float32(self, mock_download):
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")

        # Create a mock voice using the new API:
        # synthesize(text, syn_config) → Iterable[AudioChunk]
        def fake_synthesize(text, syn_config=None):
            sr = 22050
            samples = int(sr * 0.1)
            audio_float = np.zeros(samples, dtype=np.float32)
            yield _MockAudioChunk(audio_float_array=audio_float, sample_rate=sr)

        mock_voice = MagicMock()
        mock_voice.synthesize = fake_synthesize
        _mock_piper_voice.load.return_value = mock_voice

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-medium")

        chunks = list(backend.synthesize("Hello", "piper/en_US-lessac-medium"))
        assert len(chunks) == 1
        assert chunks[0].dtype == np.float32
        assert len(chunks[0]) > 0

    @patch("src.tts.backends.piper_backend.PIPER_MODELS", {})
    def test_synthesize_no_model_raises(self):
        backend = PiperBackend()
        with pytest.raises(RuntimeError, match="No Piper model loaded"):
            list(backend.synthesize("Hello", "unknown/model"))


class TestPiperBackendVoices:
    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_list_voices_single_speaker(self, mock_download):
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")
        _mock_piper_voice.load.return_value = MagicMock()

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-medium")
        voices = backend.list_voices()
        assert len(voices) >= 1
        assert voices[0].id == "piper/en_US-lessac-medium"

    def test_get_sample_rate_known_model(self):
        backend = PiperBackend()
        assert backend.get_sample_rate("piper/en_US-lessac-medium") == 22050

    def test_get_sample_rate_unknown(self):
        backend = PiperBackend()
        assert backend.get_sample_rate("piper/unknown") == 22050
