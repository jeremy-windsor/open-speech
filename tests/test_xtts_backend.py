"""Tests for XTTS backend (mocked TTS package)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_engine(sample_rate=24000):
    """Create a mock XTTS engine that returns realistic audio."""
    engine = MagicMock()
    wav = np.random.randn(sample_rate * 2).astype(np.float32) * 0.5
    engine.tts.return_value = wav
    engine.synthesizer.output_sample_rate = sample_rate
    engine.is_multi_speaker = True
    engine.is_multi_lingual = True
    return engine


@pytest.fixture(autouse=True)
def mock_xtts():
    """Mock TTS package so tests don't need GPU or model downloads."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    engine = _make_mock_engine()

    mock_tts_api = MagicMock()
    mock_tts_api.TTS.return_value = engine

    mock_tts_pkg = MagicMock()
    mock_tts_pkg.api = mock_tts_api

    with patch.dict(
        sys.modules,
        {
            "torch": mock_torch,
            "torch.backends": mock_torch.backends,
            "TTS": mock_tts_pkg,
            "TTS.api": mock_tts_api,
        },
    ):
        yield engine, mock_tts_api


class TestXTTSBackendInit:
    def test_init_defaults(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend()
        assert backend.name == "xtts"
        assert backend.sample_rate == 24000
        assert backend.capabilities["voice_clone"] is True
        assert backend.capabilities["streaming"] is False
        assert backend.capabilities["speed_control"] is False

    def test_init_explicit_device(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        assert backend._device == "cpu"

    def test_capabilities_languages(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend()
        for lang in ["en", "fr", "de", "es", "it", "pt", "nl", "tr", "ru", "pl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]:
            assert lang in backend.capabilities["languages"]


class TestXTTSDeviceResolution:
    def test_cpu_explicit(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        assert backend._resolve_device() == "cpu"

    def test_auto_no_gpu(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="auto")
        assert backend._resolve_device() == "cpu"

    def test_auto_with_cuda(self):
        import torch as mock_torch
        mock_torch.cuda.is_available.return_value = True
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="auto")
        result = backend._resolve_device()
        mock_torch.cuda.is_available.return_value = False
        assert result == "cuda"


class TestXTTSModelLifecycle:
    def test_load_model(self, mock_xtts):
        _engine, mock_api = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        assert backend.is_model_loaded("xtts/v2")
        mock_api.TTS.assert_called_once_with(
            model_name_or_path="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False,
        )

    def test_load_model_idempotent(self, mock_xtts):
        _engine, mock_api = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        backend.load_model("xtts/v2")
        assert mock_api.TTS.call_count == 1

    def test_load_unknown_model(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        with pytest.raises(ValueError, match="Unknown XTTS model"):
            backend.load_model("xtts/unknown")

    def test_unload_model(self, mock_xtts):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        backend.unload_model("xtts/v2")
        assert not backend.is_model_loaded("xtts/v2")
        assert len(backend.loaded_models()) == 0

    def test_unload_nonexistent(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.unload_model("xtts/v2")

    def test_loaded_models_info(self, mock_xtts):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        models = backend.loaded_models()
        assert len(models) == 1
        assert models[0].model == "xtts/v2"
        assert models[0].backend == "xtts"
        assert models[0].device == "cpu"
        assert models[0].loaded_at > 0
        assert models[0].last_used_at is None

    def test_load_failure_propagates(self, mock_xtts):
        _engine, mock_api = mock_xtts
        mock_api.TTS.side_effect = Exception("CUDA OOM")
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        with pytest.raises(RuntimeError, match="Failed to load XTTS"):
            backend.load_model("xtts/v2")
        assert not backend.is_model_loaded("xtts/v2")


class TestXTTSMissingPackage:
    def test_import_error(self):
        with patch.dict(sys.modules, {"TTS": None, "TTS.api": None}):
            from importlib import reload
            import src.tts.backends.xtts_backend as mod

            reload(mod)
            backend = mod.XTTSBackend(device="cpu")
            with pytest.raises(RuntimeError, match="TTS"):
                backend.load_model("xtts/v2")


class TestXTTSSynthesize:
    def test_synthesize_no_model(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        with pytest.raises(RuntimeError, match="No XTTS model loaded"):
            list(backend.synthesize("hello", "default", reference_audio=b"a"))

    def test_synthesize_empty_text(self, mock_xtts):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        with pytest.raises(ValueError, match="Text must not be empty"):
            list(backend.synthesize("", "default", reference_audio=b"a"))

    def test_synthesize_whitespace_text(self, mock_xtts):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        with pytest.raises(ValueError, match="Text must not be empty"):
            list(backend.synthesize("   ", "default", reference_audio=b"a"))

    def test_synthesize_missing_reference_audio(self, mock_xtts):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        with pytest.raises(RuntimeError, match="requires reference audio"):
            list(backend.synthesize("test", "default"))

    def test_synthesize_default_voice(self, mock_xtts):
        engine, _ = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        chunks = list(backend.synthesize("Hello world", "default", reference_audio=b"\x00" * 64))

        assert len(chunks) == 1
        assert isinstance(chunks[0], np.ndarray)
        assert chunks[0].dtype == np.float32
        engine.tts.assert_called_once()
        assert engine.tts.call_args.kwargs["text"] == "Hello world"
        assert engine.tts.call_args.kwargs["language"] == "en"

    def test_synthesize_with_reference_audio(self, mock_xtts):
        engine, _ = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")

        ref_audio = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100
        chunks = list(backend.synthesize("Generate this text", "clone", reference_audio=ref_audio))

        assert len(chunks) == 1
        call_kwargs = engine.tts.call_args.kwargs
        assert call_kwargs["text"] == "Generate this text"
        assert isinstance(call_kwargs["speaker_wav"], str)

    def test_synthesize_with_lang_code(self, mock_xtts):
        engine, _ = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")

        list(backend.synthesize("hola", "clone", lang_code="es", reference_audio=b"\x00" * 100))
        assert engine.tts.call_args.kwargs["language"] == "es"

    def test_synthesize_without_lang_code_defaults_en(self, mock_xtts):
        engine, _ = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")

        list(backend.synthesize("hello", "clone", lang_code=None, reference_audio=b"\x00" * 100))
        assert engine.tts.call_args.kwargs["language"] == "en"

    def test_synthesize_speed_ignored(self, mock_xtts):
        engine, _ = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")

        list(backend.synthesize("hello", "clone", speed=2.0, reference_audio=b"\x00" * 100))
        assert "speaker_wav" in engine.tts.call_args.kwargs

    def test_synthesize_updates_last_used(self, mock_xtts):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        assert backend.loaded_models()[0].last_used_at is None
        list(backend.synthesize("test", "default", reference_audio=b"\x00" * 100))
        assert backend.loaded_models()[0].last_used_at is not None

    def test_synthesize_normalizes_loud_audio(self, mock_xtts):
        engine, _ = mock_xtts
        loud_wav = np.ones(24000, dtype=np.float32) * 2.0
        engine.tts.return_value = loud_wav

        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        chunks = list(backend.synthesize("test", "default", reference_audio=b"\x00" * 100))

        assert np.abs(chunks[0]).max() <= 1.0

    def test_synthesize_engine_failure(self, mock_xtts):
        engine, _ = mock_xtts
        engine.tts.side_effect = Exception("Model inference crashed")

        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        with pytest.raises(RuntimeError, match="synthesis failed"):
            list(backend.synthesize("test", "default", reference_audio=b"\x00" * 100))

    def test_synthesize_cleans_up_temp_file(self, mock_xtts):
        engine, _ = mock_xtts
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")

        list(backend.synthesize("test", "clone", reference_audio=b"\x00" * 100))
        ref_file = engine.tts.call_args.kwargs["speaker_wav"]
        assert not Path(ref_file).exists()

    def test_synthesize_cleans_up_on_error(self, mock_xtts):
        engine, _ = mock_xtts
        engine.tts.side_effect = Exception("boom")

        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")

        with pytest.raises(RuntimeError):
            list(backend.synthesize("test", "clone", reference_audio=b"\x00" * 100))

        ref_file = engine.tts.call_args.kwargs["speaker_wav"]
        assert not Path(ref_file).exists()

    def test_synthesize_non_numpy_output(self, mock_xtts):
        engine, _ = mock_xtts
        engine.tts.return_value = [0.1, 0.2, 0.3]

        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        chunks = list(backend.synthesize("test", "default", reference_audio=b"\x00" * 100))

        assert isinstance(chunks[0], np.ndarray)
        assert chunks[0].dtype == np.float32

    def test_synthesize_empty_audio_returned(self, mock_xtts):
        engine, _ = mock_xtts
        engine.tts.return_value = np.array([], dtype=np.float32)

        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        with pytest.raises(RuntimeError, match="empty audio"):
            list(backend.synthesize("test", "default", reference_audio=b"\x00" * 100))

    def test_synthesize_none_returned(self, mock_xtts):
        """engine.tts() returning None should raise RuntimeError, not crash in np.array."""
        engine, _ = mock_xtts
        engine.tts.return_value = None

        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        backend.load_model("xtts/v2")
        with pytest.raises(RuntimeError, match="None"):
            list(backend.synthesize("test", "default", reference_audio=b"\x00" * 100))


class TestXTTSListVoices:
    def test_list_voices(self):
        from src.tts.backends.xtts_backend import XTTSBackend

        backend = XTTSBackend(device="cpu")
        voices = backend.list_voices()
        assert len(voices) == 1
        assert voices[0].id == "default"


class TestXTTSModels:
    def test_model_registry(self):
        from src.tts.backends.xtts_backend import XTTS_MODELS

        assert "xtts/v2" in XTTS_MODELS
        meta = XTTS_MODELS["xtts/v2"]
        assert meta["model_name"] == "tts_models/multilingual/multi-dataset/xtts_v2"
        assert meta["sample_rate"] == 24000
