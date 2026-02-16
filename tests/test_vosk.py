"""Tests for Vosk STT backend."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_vosk_import_error_graceful():
    """Backend raises helpful ImportError when vosk not installed."""
    from src.backends.vosk_backend import VoskBackend

    backend = VoskBackend()
    with patch.dict("sys.modules", {"vosk": None}):
        with pytest.raises(ImportError, match="vosk"):
            backend.load_model("vosk-model-small-en-us-0.15")


def test_vosk_init():
    """Backend initializes without loading anything."""
    from src.backends.vosk_backend import VoskBackend

    backend = VoskBackend()
    assert backend.name == "vosk"
    assert backend.loaded_models() == []
    assert not backend.is_model_loaded("vosk-model-small-en-us-0.15")


@patch("src.backends.vosk_backend.VoskBackend._decode_to_pcm")
def test_vosk_transcribe(mock_decode):
    """Transcription returns expected format."""
    mock_decode.return_value = (b"\x00" * 32000, 16000)

    mock_recognizer = MagicMock()
    mock_recognizer.AcceptWaveform.return_value = True
    mock_recognizer.FinalResult.return_value = json.dumps({"text": "hello world"})

    from src.backends.vosk_backend import VoskBackend

    backend = VoskBackend()
    backend._models["vosk-model-small-en-us-0.15"] = MagicMock()
    backend._loaded_at["vosk-model-small-en-us-0.15"] = 0
    backend._last_used["vosk-model-small-en-us-0.15"] = 0

    with patch("src.backends.vosk_backend.KaldiRecognizer", return_value=mock_recognizer, create=True):
        with patch.dict("sys.modules", {"vosk": MagicMock(KaldiRecognizer=MagicMock(return_value=mock_recognizer))}):
            result = backend.transcribe(b"fake_audio", "vosk-model-small-en-us-0.15")
            assert result == {"text": "hello world"}


@patch("src.backends.vosk_backend.VoskBackend._decode_to_pcm")
def test_vosk_transcribe_verbose(mock_decode):
    """Verbose format includes segments."""
    mock_decode.return_value = (b"\x00" * 32000, 16000)

    final_result = {
        "text": "hello world",
        "result": [
            {"word": "hello", "start": 0.0, "end": 0.5, "conf": 0.95},
            {"word": "world", "start": 0.5, "end": 1.0, "conf": 0.90},
        ],
    }
    mock_recognizer = MagicMock()
    mock_recognizer.FinalResult.return_value = json.dumps(final_result)

    from src.backends.vosk_backend import VoskBackend

    backend = VoskBackend()
    backend._models["vosk-model-small-en-us-0.15"] = MagicMock()
    backend._loaded_at["vosk-model-small-en-us-0.15"] = 0
    backend._last_used["vosk-model-small-en-us-0.15"] = 0

    with patch.dict("sys.modules", {"vosk": MagicMock(KaldiRecognizer=MagicMock(return_value=mock_recognizer))}):
        result = backend.transcribe(
            b"fake", "vosk-model-small-en-us-0.15", response_format="verbose_json"
        )
        assert "segments" in result
        assert result["text"] == "hello world"


def test_vosk_translate_not_supported():
    """Translation raises NotImplementedError."""
    from src.backends.vosk_backend import VoskBackend

    backend = VoskBackend()
    with pytest.raises(NotImplementedError):
        backend.translate(b"fake", "vosk-model-small-en-us-0.15")


def test_vosk_load_unload():
    """Load and unload lifecycle."""
    from src.backends.vosk_backend import VoskBackend

    mock_vosk = MagicMock()
    backend = VoskBackend()

    with patch.dict("sys.modules", {"vosk": mock_vosk}):
        with patch.object(backend, "_resolve_model_path", return_value="/tmp/fake"):
            backend.load_model("vosk-model-small-en-us-0.15")
            assert backend.is_model_loaded("vosk-model-small-en-us-0.15")
            assert len(backend.loaded_models()) == 1

            backend.unload_model("vosk-model-small-en-us-0.15")
            assert not backend.is_model_loaded("vosk-model-small-en-us-0.15")


def test_backend_routing():
    """Router maps model IDs to correct backends."""
    from src.router import BackendRouter

    with patch("src.router._try_load_moonshine") as mock_moon, \
         patch("src.router._try_load_vosk") as mock_vosk:
        mock_moon_backend = MagicMock()
        mock_moon_backend.name = "moonshine"
        mock_moon.return_value = mock_moon_backend

        mock_vosk_backend = MagicMock()
        mock_vosk_backend.name = "vosk"
        mock_vosk.return_value = mock_vosk_backend

        router = BackendRouter()

        assert router.get_backend("moonshine/tiny").name == "moonshine"
        assert router.get_backend("moonshine/base").name == "moonshine"
        assert router.get_backend("vosk-model-small-en-us-0.15").name == "vosk"
        assert router.get_backend("deepdml/faster-whisper-large-v3-turbo-ct2").name == "faster-whisper"
