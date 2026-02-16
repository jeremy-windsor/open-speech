"""Tests for Moonshine STT backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_moonshine_import_error_graceful():
    """Backend raises helpful ImportError when moonshine-onnx not installed."""
    from src.backends.moonshine import MoonshineBackend

    backend = MoonshineBackend()
    with patch.dict("sys.modules", {"moonshine_onnx": None}):
        with pytest.raises(ImportError, match="moonshine-onnx"):
            backend.load_model("moonshine/tiny")


def test_moonshine_init():
    """Backend initializes without loading anything."""
    from src.backends.moonshine import MoonshineBackend

    backend = MoonshineBackend()
    assert backend.name == "moonshine"
    assert backend.loaded_models() == []
    assert not backend.is_model_loaded("moonshine/tiny")


@patch("src.backends.moonshine.MoonshineBackend._decode_audio")
def test_moonshine_transcribe(mock_decode):
    """Transcription returns expected format."""
    import numpy as np

    mock_decode.return_value = np.zeros(16000, dtype=np.float32)

    mock_model = MagicMock()
    mock_model.generate.return_value = ["hello", "world"]

    from src.backends.moonshine import MoonshineBackend

    backend = MoonshineBackend()
    backend._models["moonshine/tiny"] = mock_model
    backend._loaded_at["moonshine/tiny"] = 0
    backend._last_used["moonshine/tiny"] = 0

    result = backend.transcribe(b"fake_audio", "moonshine/tiny")
    assert result == {"text": "hello world"}
    mock_model.generate.assert_called_once()


@patch("src.backends.moonshine.MoonshineBackend._decode_audio")
def test_moonshine_transcribe_verbose(mock_decode):
    """Verbose format includes segments."""
    import numpy as np

    mock_decode.return_value = np.zeros(16000, dtype=np.float32)

    mock_model = MagicMock()
    mock_model.generate.return_value = ["hello world"]

    from src.backends.moonshine import MoonshineBackend

    backend = MoonshineBackend()
    backend._models["moonshine/tiny"] = mock_model
    backend._loaded_at["moonshine/tiny"] = 0
    backend._last_used["moonshine/tiny"] = 0

    result = backend.transcribe(b"fake", "moonshine/tiny", response_format="verbose_json")
    assert "segments" in result
    assert result["language"] == "en"
    assert result["text"] == "hello world"


def test_moonshine_rejects_non_english():
    """Moonshine raises error for non-English language requests."""
    from src.backends.moonshine import MoonshineBackend

    backend = MoonshineBackend()
    backend._models["moonshine/tiny"] = MagicMock()
    backend._loaded_at["moonshine/tiny"] = 0
    backend._last_used["moonshine/tiny"] = 0

    with pytest.raises(ValueError, match="English"):
        backend.transcribe(b"fake", "moonshine/tiny", language="fr")


def test_moonshine_translate_not_supported():
    """Translation raises NotImplementedError."""
    from src.backends.moonshine import MoonshineBackend

    backend = MoonshineBackend()
    with pytest.raises(NotImplementedError):
        backend.translate(b"fake", "moonshine/tiny")


def test_moonshine_load_unload():
    """Load and unload lifecycle."""
    from src.backends.moonshine import MoonshineBackend

    mock_model_cls = MagicMock()
    with patch("src.backends.moonshine.MoonshineOnnxModel", mock_model_cls, create=True):
        with patch.dict("sys.modules", {"moonshine_onnx": MagicMock(MoonshineOnnxModel=mock_model_cls)}):
            backend = MoonshineBackend()
            backend.load_model("moonshine/tiny")
            assert backend.is_model_loaded("moonshine/tiny")
            assert len(backend.loaded_models()) == 1

            backend.unload_model("moonshine/tiny")
            assert not backend.is_model_loaded("moonshine/tiny")
            assert len(backend.loaded_models()) == 0
