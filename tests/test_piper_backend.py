"""Tests for Piper TTS backend — mocked piper-tts package."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_mock_piper = MagicMock()
_mock_piper_voice = MagicMock()
_mock_piper.PiperVoice = _mock_piper_voice

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


@pytest.fixture(autouse=True)
def mock_piper_package(monkeypatch):
    """Keep Piper tests isolated even if app discovery imported the real package."""
    _mock_piper_voice.reset_mock()
    monkeypatch.setitem(sys.modules, "piper", _mock_piper)
    monkeypatch.setitem(sys.modules, "piper.config", _mock_piper_config)


@dataclass
class _MockAudioChunk:
    """Minimal AudioChunk stand-in for unit tests."""
    audio_float_array: np.ndarray
    sample_rate: int = 22050
    sample_width: int = 2
    sample_channels: int = 1
    phonemes: list = field(default_factory=list)
    phoneme_ids: list = field(default_factory=list)

from src.tts.backends.piper_backend import (  # noqa: E402
    PIPER_MODELS,
    PiperBackend,
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
        assert len(PIPER_MODELS) == 28


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

    @pytest.mark.parametrize(
        ("configured", "speed", "expected"),
        [
            (0.8, 1.0, 0.8), (0.8, 2.0, 0.4), (0.8, 0.5, 1.6),
            (1.0, 1.0, 1.0), (1.0, 2.0, 0.5),
            (None, 2.0, 0.5),   # a mock voice or config with no number keeps Piper's default of 1
        ],
    )
    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_speed_scales_the_voices_own_length_scale(self, mock_download, configured, speed, expected):
        # piper-tts uses the voice's inference.length_scale (0.8 for
        # en_US-hfc_male-medium) only when SynthesisConfig.length_scale is None. Sending 1 / speed spoke that
        # voice 25% slower than published at every speed; speed 1.0 is the voice's own pace.
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")
        sent = []

        def fake_synthesize(text, syn_config=None):
            sent.append(syn_config.length_scale)
            yield _MockAudioChunk(audio_float_array=np.zeros(10, dtype=np.float32))

        mock_voice = MagicMock()
        mock_voice.synthesize = fake_synthesize
        if configured is not None:
            mock_voice.config.length_scale = configured
        _mock_piper_voice.load.return_value = mock_voice

        backend = PiperBackend()
        backend.load_model("piper/en_US-hfc_male-medium")
        list(backend.synthesize("Hello", "piper/en_US-hfc_male-medium", speed=speed))

        assert sent == [pytest.approx(expected)]

    def test_unknown_model_is_rejected(self):
        backend = PiperBackend()
        with pytest.raises(ValueError, match="Unknown Piper model"):
            list(backend.synthesize("Hello", "unknown/model"))

    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_unknown_model_does_not_reuse_loaded_voice(self, mock_download):
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")
        mock_voice = MagicMock()
        _mock_piper_voice.load.return_value = mock_voice

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-medium")

        with pytest.raises(ValueError, match="Unknown Piper model"):
            list(backend.synthesize("Hello", "piper/not-a-real-voice"))
        mock_voice.synthesize.assert_not_called()


class TestPiperBackendVoices:
    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_list_voices_single_speaker(self, mock_download, tmp_path):
        # A real config file keeps list_voices from fetching one from Hugging Face.
        config = tmp_path / "model.onnx.json"
        config.write_text('{"speaker_id_map": {}}', encoding="utf-8")
        mock_download.return_value = (str(tmp_path / "model.onnx"), str(config))
        _mock_piper_voice.load.return_value = MagicMock()

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-medium")
        voices = backend.list_voices()
        assert len(voices) >= 1
        assert voices[0].id == "piper/en_US-lessac-medium"

    def test_get_sample_rate_known_model(self):
        backend = PiperBackend()
        assert backend.get_sample_rate("piper/en_US-lessac-medium") == 22050
        assert backend.get_sample_rate("piper/en_US-lessac-low") == 16000

    def test_get_sample_rate_unknown(self):
        backend = PiperBackend()
        with pytest.raises(ValueError, match="Unknown Piper model"):
            backend.get_sample_rate("piper/unknown")

    @patch("src.tts.backends.piper_backend.PiperBackend._download_model")
    def test_loading_low_rate_model_does_not_mutate_backend_default(self, mock_download):
        mock_download.return_value = ("/tmp/model.onnx", "/tmp/model.onnx.json")
        _mock_piper_voice.load.return_value = MagicMock()

        backend = PiperBackend()
        backend.load_model("piper/en_US-lessac-low")

        assert backend.sample_rate == 22050
        assert backend.get_sample_rate("piper/en_US-lessac-low") == 16000
