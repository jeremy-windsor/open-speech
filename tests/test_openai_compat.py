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
from src.openai_compat import apply_speech_defaults, is_openai_realtime_model, resolve_stt_model, resolve_tts_model
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


@pytest.mark.parametrize("model", [
    "gpt-realtime", "gpt-realtime-2025-08-28", "gpt-realtime-mini", "gpt-realtime-mini-2025-10-06",
    "gpt-4o-realtime-preview", "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-mini-realtime-preview", "gpt-4o-mini-realtime-preview-2024-12-17",
])
def test_openai_realtime_models_are_recognised_and_speak_with_configured_tts_model(model):
    assert is_openai_realtime_model(model)
    assert resolve_tts_model(model, SETTINGS) == "piper/en_US-lessac-medium"


@pytest.mark.parametrize("model", [None, "", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"])
def test_openai_and_missing_tts_models_use_configured_default(model):
    assert resolve_tts_model(model, SETTINGS) == "piper/en_US-lessac-medium"


@pytest.mark.parametrize("model", ["kokoro", "piper/en_US-lessac-medium", "whisper-1", "gpt-realtimes", None])
def test_other_models_are_not_openai_realtime_models(model):
    assert not is_openai_realtime_model(model)


def test_explicit_tts_model_is_kept():
    assert resolve_tts_model("kokoro", SETTINGS) == "kokoro"


def test_omitted_speech_fields_use_settings():
    request = apply_speech_defaults(TTSSpeechRequest(input="Hi"), SETTINGS)
    assert request.model == "piper/en_US-lessac-medium"
    assert request.voice == "piper/en_US-lessac-medium"
    assert request.speed == 1.25
    assert request.response_format == "wav"


def test_shipped_default_response_format_is_mp3(monkeypatch):
    """README: TTS_DEFAULT_FORMAT defaults to mp3, OpenAI's default response_format."""
    from src.config import Settings

    monkeypatch.delenv("TTS_DEFAULT_FORMAT", raising=False)
    assert Settings().tts_default_format == "mp3"
    request = apply_speech_defaults(TTSSpeechRequest(input="Hi"), Settings())
    assert request.response_format == "mp3"


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


class _SpeedRouter:
    """Kokoro has speed control; pocket-tts and chatterbox have a fixed speed, as their backends report."""

    def __init__(self):
        self.requests = []

    def get_backend(self, model):
        return SimpleNamespace(name=model.split("/")[0], capabilities=self.get_capabilities(model))

    def get_capabilities(self, model):
        return {"speed_control": model == "kokoro"}

    def max_input_chars_for(self, _model):
        return None

    def sample_rate_for(self, _model):
        return 24000

    def validate_voice(self, _model, _voice):
        return None

    def validate_language(self, _model, _language):
        return None

    def loaded_models(self):
        return []

    def synthesize(self, **kwargs):
        self.requests.append(kwargs)
        yield np.full(2400, 0.1, dtype=np.float32)


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("tts_speed", [1.0, 1.25], ids=["TTS_SPEED=1.0", "TTS_SPEED=1.25"])
def test_tts_speed_applies_only_to_a_model_with_speed_control(tts_speed, stream):
    # TTS_SPEED is the "speed used when /v1/audio/speech omits speed" (README), so a
    # request that sends no speed to a fixed-speed model is spoken at 1.0 instead of refused with "Speed control is not
    # supported"; a model with speed control still gets TTS_SPEED, and an explicit speed is still checked.
    router = _SpeedRouter()
    client = TestClient(app, raise_server_exceptions=False)
    path = "/v1/audio/speech?stream=true" if stream else "/v1/audio/speech"
    with (
        patch.object(main_module, "tts_router", router),
        patch.object(main_module.settings, "tts_speed", tts_speed),
        patch.object(main_module.settings, "tts_cache_enabled", False),
    ):
        spoken = {}
        for model, voice in (("pocket-tts", "alba"), ("chatterbox/turbo", "reference"), ("kokoro", "af_heart")):
            resp = client.post(path, json={"model": model, "input": "Hello there", "voice": voice,
                                           "response_format": "pcm"})
            assert resp.status_code == 200, f"TTS_SPEED={tts_speed}, {model}, no speed sent: {resp.text[:200]}"
            spoken[model] = router.requests[-1]["speed"]
        explicit = client.post(path, json={"model": "pocket-tts", "input": "Hello there", "voice": "alba",
                                           "response_format": "pcm", "speed": 1.25})

    assert spoken == {"pocket-tts": 1.0, "chatterbox/turbo": 1.0, "kokoro": tts_speed}
    assert explicit.status_code == 400, explicit.text
    assert "Speed control is not supported" in explicit.json()["error"]["message"]
