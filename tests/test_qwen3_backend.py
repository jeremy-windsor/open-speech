"""Tests for Qwen3 backend with mocked qwen-tts package."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_qwen_tts():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"

    mock_model = MagicMock()
    mock_model.generate_custom_voice.return_value = np.ones(24000, dtype=np.float32)
    mock_model.generate_voice_design.return_value = np.ones(24000, dtype=np.float32) * 0.5
    mock_model.generate_voice_clone.return_value = np.ones(24000, dtype=np.float32) * 0.25
    mock_model.create_voice_clone_prompt.return_value = {"prompt": "cached"}

    mock_qwen = MagicMock()
    mock_qwen.Qwen3TTSModel.from_pretrained.return_value = mock_model

    with patch.dict(sys.modules, {"torch": mock_torch, "qwen_tts": mock_qwen}):
        yield mock_qwen, mock_model


class TestQwen3Backend:
    def test_load_unload(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        backend.load_model("qwen3-tts/1.7B-CustomVoice")
        assert backend.is_model_loaded("qwen3-tts/1.7B-CustomVoice")
        backend.unload_model("qwen3-tts/1.7B-CustomVoice")
        assert not backend.is_model_loaded("qwen3-tts/1.7B-CustomVoice")

    def test_model_auto_selection_custom_voice(self, mock_qwen_tts):
        _, model = mock_qwen_tts
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        list(backend.synthesize("hello", "Ryan"))
        model.generate_custom_voice.assert_called_once()

    def test_model_auto_selection_voice_design(self, mock_qwen_tts):
        _, model = mock_qwen_tts
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        list(backend.synthesize("hello", "", voice_design="deep male british accent"))
        model.generate_voice_design.assert_called_once()

    def test_model_auto_selection_base_for_clone(self, mock_qwen_tts):
        _, model = mock_qwen_tts
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        list(backend.synthesize("hello", "Ryan", reference_audio=b"RIFF....", clone_transcript="hello there"))
        model.generate_voice_clone.assert_called_once()
        model.create_voice_clone_prompt.assert_called_once()

    def test_instruction_passthrough(self, mock_qwen_tts):
        _, model = mock_qwen_tts
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        list(backend.synthesize("hello", "Ryan", voice_design="Speak angrily"))
        kwargs = model.generate_custom_voice.call_args.kwargs
        assert kwargs["instruct"] == "Speak angrily"

    def test_speaker_validation(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        with pytest.raises(ValueError, match="Unsupported Qwen3 speaker"):
            list(backend.synthesize("hello", "NotARealVoice"))

    def test_language_auto_detection_from_speaker(self, mock_qwen_tts):
        _, model = mock_qwen_tts
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        list(backend.synthesize("hello", "Vivian"))
        kwargs = model.generate_custom_voice.call_args.kwargs
        assert kwargs["language"] == "zh"

    def test_list_voices_has_9_premium(self):
        from src.tts.backends.qwen3_backend import Qwen3Backend

        backend = Qwen3Backend(device="cpu")
        voices = backend.list_voices()
        assert len(voices) == 9
        assert {v.id for v in voices} >= {"Ryan", "Vivian", "Sohee"}
