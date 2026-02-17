"""Tests for Qwen3-TTS backend (mocked transformers)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_transformers():
    """Mock transformers and torch so we don't need real models."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()

    mock_transformers = MagicMock()

    # Model mock
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.__getitem__ = MagicMock(return_value=MagicMock(
        cpu=MagicMock(return_value=MagicMock(
            float=MagicMock(return_value=MagicMock(
                numpy=MagicMock(return_value=np.random.randn(24000).astype(np.float32))
            ))
        ))
    ))
    mock_model.generate.return_value = mock_output
    mock_model.device = "cpu"
    mock_model.cpu.return_value = mock_model

    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    mock_transformers.AutoProcessor.from_pretrained.return_value = MagicMock(
        return_value={"input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))}
    )

    with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers, "accelerate": MagicMock()}):
        yield


class TestQwen3Backend:
    def test_init(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend(device="cpu")
        assert backend.name == "qwen3"
        assert backend.sample_rate == 24000

    def test_load_unload(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend(device="cpu")
        backend.load_model("qwen3-tts-0.6b")
        assert backend.is_model_loaded("qwen3-tts-0.6b")
        assert len(backend.loaded_models()) == 1
        backend.unload_model("qwen3-tts-0.6b")
        assert not backend.is_model_loaded("qwen3-tts-0.6b")

    def test_load_unknown_model(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend(device="cpu")
        with pytest.raises(ValueError, match="Unknown Qwen3"):
            backend.load_model("qwen3-tts-99b")

    def test_synthesize_not_loaded(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend(device="cpu")
        with pytest.raises(RuntimeError, match="No Qwen3-TTS model loaded"):
            list(backend.synthesize("hello", "Chelsie"))

    def test_list_voices(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend(device="cpu")
        voices = backend.list_voices()
        assert len(voices) >= 2
        assert any(v.id == "Chelsie" for v in voices)

    def test_loaded_models_info(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend(device="cpu")
        backend.load_model("qwen3-tts-0.6b")
        info = backend.loaded_models()
        assert info[0].backend == "qwen3"
        assert info[0].model == "qwen3-tts-0.6b"

    def test_load_idempotent(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend(device="cpu")
        backend.load_model("qwen3-tts-0.6b")
        backend.load_model("qwen3-tts-0.6b")  # no error
        assert len(backend.loaded_models()) == 1
