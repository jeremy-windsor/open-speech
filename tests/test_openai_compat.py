"""Tests for OpenAI client compatibility: model aliases and settings defaults."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src import main as main_module
from src.main import app
from src.openai_compat import apply_speech_defaults, resolve_stt_model
from src.tts.models import TTSSpeechRequest

SETTINGS = SimpleNamespace(
    stt_model="Systran/faster-whisper-small",
    tts_model="piper/en_US-lessac-medium",
    tts_voice="piper/en_US-lessac-medium",
    tts_speed=1.25,
    tts_default_format="wav",
)


@pytest.mark.parametrize("model", [None, "", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"])
def test_openai_and_missing_stt_models_use_configured_default(model):
    assert resolve_stt_model(model, SETTINGS) == "Systran/faster-whisper-small"


def test_explicit_stt_model_is_kept():
    assert resolve_stt_model("Systran/faster-whisper-tiny", SETTINGS) == "Systran/faster-whisper-tiny"


def test_omitted_speech_fields_use_settings():
    request = apply_speech_defaults(TTSSpeechRequest(input="Hi"), SETTINGS)
    assert request.model == "piper/en_US-lessac-medium"
    assert request.voice == "piper/en_US-lessac-medium"
    assert request.speed == 1.25
    assert request.response_format == "wav"


@pytest.mark.parametrize("model", ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"])
def test_openai_tts_models_map_to_configured_model_and_keep_voice(model):
    request = apply_speech_defaults(TTSSpeechRequest(model=model, input="Hi", voice="nova"), SETTINGS)
    assert request.model == "piper/en_US-lessac-medium"
    assert request.voice == "nova"


def test_explicit_model_keeps_its_own_voice_default():
    request = apply_speech_defaults(TTSSpeechRequest(model="kokoro", input="Hi"), SETTINGS)
    assert request.model == "kokoro"
    assert request.voice == "alloy"


def test_explicit_fields_are_not_overridden():
    request = TTSSpeechRequest(model="kokoro", input="Hi", voice="af_heart", speed=1.0, response_format="mp3")
    assert apply_speech_defaults(request, SETTINGS) is request


def test_unknown_speech_fields_are_ignored():
    request = TTSSpeechRequest.model_validate({"input": "Hi", "some_future_option": True})
    assert not hasattr(request, "some_future_option")


def test_retired_speech_fields_are_still_rejected():
    with pytest.raises(ValidationError, match="voice_blend"):
        TTSSpeechRequest.model_validate({"input": "Hi", "voice_blend": "af_heart(1)+af_bella(1)"})


def test_speech_endpoint_accepts_openai_model_and_unknown_fields():
    mock_router = MagicMock()
    mock_router.synthesize.return_value = iter([np.zeros(2400, dtype=np.float32)])
    mock_router.loaded_models.return_value = []
    with patch.object(main_module, "tts_router", mock_router):
        resp = TestClient(app).post("/v1/audio/speech", json={
            "model": "tts-1",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "wav",
            "some_future_option": "x",
        })
    assert resp.status_code == 200, resp.text
    assert resp.content[:4] == b"RIFF"
    assert mock_router.synthesize.call_args.kwargs["model"] == main_module.settings.tts_model
