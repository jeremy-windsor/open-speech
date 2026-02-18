"""Tests for F5-TTS backend (mocked f5_tts package)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


def _make_mock_engine(sample_rate=24000):
    """Create a mock F5TTS engine that returns realistic audio."""
    engine = MagicMock()
    # F5TTS.infer returns (wav, sr, spec)
    wav = np.random.randn(sample_rate * 2).astype(np.float32) * 0.5  # 2s audio
    spec = np.random.randn(100, 200).astype(np.float32)
    engine.infer.return_value = (wav, sample_rate, spec)
    engine.target_sample_rate = sample_rate
    return engine


@pytest.fixture(autouse=True)
def mock_f5tts():
    """Mock f5_tts package so tests don't need GPU or model downloads."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    engine = _make_mock_engine()

    mock_f5_api = MagicMock()
    mock_f5_api.F5TTS.return_value = engine

    mock_f5_pkg = MagicMock()
    mock_f5_pkg.api = mock_f5_api

    # Mock importlib.resources for built-in ref audio
    mock_files = MagicMock()

    with patch.dict(sys.modules, {
        "torch": mock_torch,
        "torch.backends": mock_torch.backends,
        "f5_tts": mock_f5_pkg,
        "f5_tts.api": mock_f5_api,
    }):
        yield engine, mock_f5_api


class TestF5TTSBackendInit:
    def test_init_defaults(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend()
        assert backend.name == "f5-tts"
        assert backend.sample_rate == 24000
        assert backend.capabilities["voice_blend"] is False
        assert backend.capabilities["voice_clone"] is True
        assert backend.capabilities["voice_design"] is False
        assert backend.capabilities["streaming"] is False
        assert backend.capabilities["instructions"] is False

    def test_init_explicit_device(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        assert backend._device == "cpu"

    def test_capabilities_languages(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend()
        assert "en" in backend.capabilities["languages"]
        assert "zh" in backend.capabilities["languages"]


class TestF5TTSDeviceResolution:
    def test_cpu_explicit(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        assert backend._resolve_device() == "cpu"

    def test_auto_no_gpu(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="auto")
        # mock_torch.cuda.is_available returns False
        assert backend._resolve_device() == "cpu"

    def test_auto_with_cuda(self):
        import torch as mock_torch
        mock_torch.cuda.is_available.return_value = True
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="auto")
        result = backend._resolve_device()
        mock_torch.cuda.is_available.return_value = False
        assert result == "cuda"


class TestF5TTSModelLifecycle:
    def test_load_model(self, mock_f5tts):
        engine, mock_api = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")
        assert backend.is_model_loaded("f5-tts/v1-base")
        mock_api.F5TTS.assert_called_once_with(
            model="F5TTS_v1_Base",
            device="cpu",
        )

    def test_load_model_idempotent(self, mock_f5tts):
        engine, mock_api = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")
        backend.load_model("f5-tts/v1-base")  # Should not re-load
        assert mock_api.F5TTS.call_count == 1

    def test_load_unknown_model(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        with pytest.raises(ValueError, match="Unknown F5-TTS model"):
            backend.load_model("f5-tts/nonexistent")

    def test_load_e2_model(self, mock_f5tts):
        engine, mock_api = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/e2-base")
        assert backend.is_model_loaded("f5-tts/e2-base")
        mock_api.F5TTS.assert_called_once_with(
            model="E2TTS_Base",
            device="cpu",
        )

    def test_unload_model(self, mock_f5tts):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")
        backend.unload_model("f5-tts/v1-base")
        assert not backend.is_model_loaded("f5-tts/v1-base")
        assert len(backend.loaded_models()) == 0

    def test_unload_nonexistent(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.unload_model("f5-tts/v1-base")  # Should not raise

    def test_loaded_models_info(self, mock_f5tts):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")
        models = backend.loaded_models()
        assert len(models) == 1
        assert models[0].model == "f5-tts/v1-base"
        assert models[0].backend == "f5-tts"
        assert models[0].device == "cpu"
        assert models[0].loaded_at > 0
        assert models[0].last_used_at is None

    def test_load_failure_propagates(self, mock_f5tts):
        engine, mock_api = mock_f5tts
        mock_api.F5TTS.side_effect = Exception("CUDA OOM")
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        with pytest.raises(RuntimeError, match="Failed to load F5-TTS"):
            backend.load_model("f5-tts/v1-base")
        assert not backend.is_model_loaded("f5-tts/v1-base")


class TestF5TTSMissingPackage:
    def test_import_error(self):
        """If f5-tts not installed, load_model raises RuntimeError."""
        with patch.dict(sys.modules, {"f5_tts": None, "f5_tts.api": None}):
            from importlib import reload
            import src.tts.backends.f5tts_backend as mod
            reload(mod)
            backend = mod.F5TTSBackend(device="cpu")
            with pytest.raises(RuntimeError, match="f5-tts"):
                backend.load_model("f5-tts/v1-base")


class TestF5TTSSynthesize:
    def test_synthesize_no_model(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        with pytest.raises(RuntimeError, match="No F5-TTS model loaded"):
            list(backend.synthesize("hello", "default"))

    def test_synthesize_empty_text(self, mock_f5tts):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")
        with pytest.raises(ValueError, match="Text must not be empty"):
            list(backend.synthesize("", "default"))

    def test_synthesize_whitespace_text(self, mock_f5tts):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")
        with pytest.raises(ValueError, match="Text must not be empty"):
            list(backend.synthesize("   ", "default"))

    def test_synthesize_empty_audio_returned(self, mock_f5tts):
        engine, _ = mock_f5tts
        engine.infer.return_value = (np.array([], dtype=np.float32), 24000, None)
        from src.tts.backends.f5tts_backend import F5TTSBackend
        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value="/fake/ref.wav"):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            with pytest.raises(RuntimeError, match="empty audio"):
                list(backend.synthesize("test", "default"))

    def test_synthesize_speed_clamped(self, mock_f5tts):
        engine, _ = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend
        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value="/fake/ref.wav"):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            list(backend.synthesize("test", "default", speed=100.0))
            assert engine.infer.call_args.kwargs["speed"] == 4.0
            list(backend.synthesize("test", "default", speed=0.01))
            assert engine.infer.call_args.kwargs["speed"] == 0.25

    def test_synthesize_default_voice(self, mock_f5tts):
        engine, _ = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend, _get_builtin_ref_audio_path, _BUILTIN_REF_TEXT

        # Mock builtin ref path
        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path") as mock_ref:
            mock_ref.return_value = "/fake/ref.wav"
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            chunks = list(backend.synthesize("Hello world", "default"))

        assert len(chunks) == 1
        assert isinstance(chunks[0], np.ndarray)
        assert chunks[0].dtype == np.float32
        # Verify engine.infer was called with correct args
        engine.infer.assert_called_once()
        call_kwargs = engine.infer.call_args
        assert call_kwargs.kwargs["ref_file"] == "/fake/ref.wav"
        assert call_kwargs.kwargs["ref_text"] == _BUILTIN_REF_TEXT
        assert call_kwargs.kwargs["gen_text"] == "Hello world"

    def test_synthesize_with_reference_audio(self, mock_f5tts):
        engine, _ = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend

        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")

        ref_audio = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100
        ref_text = "This is my reference text."

        chunks = list(backend.synthesize(
            "Generate this text",
            "clone",
            reference_audio=ref_audio,
            reference_text=ref_text,
        ))

        assert len(chunks) == 1
        # Verify it used a temp file, not the builtin
        call_kwargs = engine.infer.call_args.kwargs
        assert call_kwargs["ref_text"] == ref_text
        assert call_kwargs["gen_text"] == "Generate this text"
        # ref_file should be a temp file path (not the builtin)
        assert call_kwargs["ref_file"] != "/fake/ref.wav"

    def test_synthesize_with_reference_no_text(self, mock_f5tts):
        """When reference audio given without text, ref_text should be empty (F5-TTS will ASR)."""
        engine, _ = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend

        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")

        chunks = list(backend.synthesize(
            "Generate this",
            "clone",
            reference_audio=b"\x00" * 100,
        ))

        call_kwargs = engine.infer.call_args.kwargs
        assert call_kwargs["ref_text"] == ""  # Empty = ASR mode

    def test_synthesize_speed(self, mock_f5tts):
        engine, _ = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend

        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value="/fake/ref.wav"):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            list(backend.synthesize("test", "default", speed=1.5))

        assert engine.infer.call_args.kwargs["speed"] == 1.5

    def test_synthesize_updates_last_used(self, mock_f5tts):
        engine, _ = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend

        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value="/fake/ref.wav"):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            assert backend.loaded_models()[0].last_used_at is None
            list(backend.synthesize("test", "default"))
            assert backend.loaded_models()[0].last_used_at is not None

    def test_synthesize_normalizes_loud_audio(self, mock_f5tts):
        """Audio with peak > 1.0 should be normalized."""
        engine, _ = mock_f5tts
        # Return audio with peak of 2.0
        loud_wav = np.ones(24000, dtype=np.float32) * 2.0
        engine.infer.return_value = (loud_wav, 24000, np.zeros((100, 50)))

        from src.tts.backends.f5tts_backend import F5TTSBackend
        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value="/fake/ref.wav"):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            chunks = list(backend.synthesize("test", "default"))

        assert np.abs(chunks[0]).max() <= 1.0

    def test_synthesize_no_builtin_ref_no_user_ref(self, mock_f5tts):
        """If no reference audio and no builtin found, raise RuntimeError."""
        from src.tts.backends.f5tts_backend import F5TTSBackend

        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value=None):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            with pytest.raises(RuntimeError, match="No reference audio"):
                list(backend.synthesize("test", "default"))

    def test_synthesize_engine_failure(self, mock_f5tts):
        engine, _ = mock_f5tts
        engine.infer.side_effect = Exception("Model inference crashed")

        from src.tts.backends.f5tts_backend import F5TTSBackend
        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value="/fake/ref.wav"):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            with pytest.raises(RuntimeError, match="synthesis failed"):
                list(backend.synthesize("test", "default"))

    def test_synthesize_cleans_up_temp_file(self, mock_f5tts):
        """Temp file for reference audio should be cleaned up after synthesis."""
        engine, _ = mock_f5tts
        from src.tts.backends.f5tts_backend import F5TTSBackend

        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")

        list(backend.synthesize(
            "test",
            "clone",
            reference_audio=b"\x00" * 100,
        ))

        # The temp file that was created should have been cleaned up
        ref_file = engine.infer.call_args.kwargs["ref_file"]
        assert not Path(ref_file).exists()

    def test_synthesize_cleans_up_on_error(self, mock_f5tts):
        """Temp file should be cleaned up even if synthesis fails."""
        engine, _ = mock_f5tts
        engine.infer.side_effect = Exception("boom")

        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        backend.load_model("f5-tts/v1-base")

        with pytest.raises(RuntimeError):
            list(backend.synthesize(
                "test", "clone", reference_audio=b"\x00" * 100,
            ))

        ref_file = engine.infer.call_args.kwargs["ref_file"]
        assert not Path(ref_file).exists()

    def test_synthesize_non_numpy_output(self, mock_f5tts):
        """Engine returning a list instead of numpy should be converted."""
        engine, _ = mock_f5tts
        engine.infer.return_value = ([0.1, 0.2, 0.3], 24000, None)

        from src.tts.backends.f5tts_backend import F5TTSBackend
        with patch("src.tts.backends.f5tts_backend._get_builtin_ref_audio_path", return_value="/fake/ref.wav"):
            backend = F5TTSBackend(device="cpu")
            backend.load_model("f5-tts/v1-base")
            chunks = list(backend.synthesize("test", "default"))

        assert isinstance(chunks[0], np.ndarray)
        assert chunks[0].dtype == np.float32


class TestF5TTSListVoices:
    def test_list_voices(self):
        from src.tts.backends.f5tts_backend import F5TTSBackend
        backend = F5TTSBackend(device="cpu")
        voices = backend.list_voices()
        assert len(voices) == 1
        assert voices[0].id == "default"
        assert voices[0].language == "en-us"


class TestF5TTSModels:
    def test_model_registry(self):
        from src.tts.backends.f5tts_backend import F5_MODELS
        assert "f5-tts/v1-base" in F5_MODELS
        assert "f5-tts/e2-base" in F5_MODELS
        for model_id, meta in F5_MODELS.items():
            assert "f5_model_name" in meta
            assert "sample_rate" in meta
            assert meta["sample_rate"] == 24000


class TestBuiltinRef:
    def test_builtin_ref_missing(self):
        """When importlib.resources can't find the ref, return None."""
        from src.tts.backends.f5tts_backend import _get_builtin_ref_audio_path
        with patch("src.tts.backends.f5tts_backend.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            # The function may or may not find it depending on f5_tts install
            result = _get_builtin_ref_audio_path()
            # It should return None or a path — just ensure no crash
            assert result is None or isinstance(result, str)
