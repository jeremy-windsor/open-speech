"""Tests for TTS backend capabilities and validation responses."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module


class _FakeBackend:
    def __init__(self, name: str, capabilities: dict):
        self.name = name
        self.capabilities = capabilities

    def synthesize(self, text, voice, speed=1.0, lang_code=None, **kwargs):
        yield np.zeros(24000, dtype=np.float32)


def _mock_router():
    router = MagicMock()

    kokoro = _FakeBackend("kokoro", {
        "voice_blend": True,
        "voice_design": False,
        "voice_clone": False,
        "streaming": False,
        "instructions": False,
        "speakers": [{"name": "af_bella", "description": "Bella", "language": "en-us"}],
        "languages": ["en-us"],
        "speed_control": True,
        "ssml": "partial",
        "batch": False,
    })
    piper = _FakeBackend("piper", {
        "voice_blend": False,
        "voice_design": False,
        "voice_clone": False,
        "streaming": False,
        "instructions": False,
        "speakers": [{"name": "en_US-lessac-medium", "description": "US medium", "language": "en"}],
        "languages": ["en"],
        "speed_control": True,
        "ssml": False,
        "batch": False,
    })
    qwen3 = _FakeBackend("qwen3", {
        "voice_blend": False,
        "voice_design": True,
        "voice_clone": True,
        "streaming": True,
        "instructions": True,
        "speakers": [{"name": "Ryan", "description": "English male", "language": "en"}],
        "languages": ["Auto", "en", "zh"],
        "speed_control": True,
        "ssml": False,
        "batch": True,
    })

    def get_backend(model_id: str):
        if model_id.startswith("piper/"):
            return piper
        if model_id.startswith("qwen3"):
            return qwen3
        return kokoro

    router.get_backend.side_effect = get_backend
    router.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])
    router.loaded_models.return_value = []
    return router


def test_tts_capabilities_endpoint():
    mock_router = _mock_router()
    with patch.object(main_module, "tts_router", mock_router):
        client = TestClient(app)
        r = client.get("/api/tts/capabilities?model=qwen3-tts-0.6b")
        assert r.status_code == 200
        data = r.json()
        assert data["backend"] == "qwen3"
        assert data["capabilities"]["voice_blend"] is False
        assert data["capabilities"]["voice_design"] is True
        assert data["capabilities"]["voice_clone"] is True
        assert data["capabilities"]["streaming"] is True
        assert data["capabilities"]["instructions"] is True


def test_api_models_contains_tts_capabilities():
    mock_router = _mock_router()
    fake_model_manager = MagicMock()
    fake_model_manager.list_all.return_value = [
        SimpleNamespace(to_dict=lambda: {"id": "kokoro", "type": "tts", "provider": "kokoro", "state": "available"}),
        SimpleNamespace(to_dict=lambda: {"id": "piper/en_US-lessac-medium", "type": "tts", "provider": "piper", "state": "available"}),
    ]

    with patch.object(main_module, "tts_router", mock_router), patch.object(main_module, "model_manager", fake_model_manager):
        client = TestClient(app)
        r = client.get("/api/models")
        assert r.status_code == 200
        models = r.json()["models"]
        for m in models:
            assert "capabilities" in m
            for key in ("voice_blend", "voice_design", "voice_clone", "streaming", "instructions"):
                assert key in m["capabilities"]


def test_voice_design_rejected_on_kokoro():
    mock_router = _mock_router()
    with patch.object(main_module, "tts_router", mock_router):
        client = TestClient(app)
        r = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_bella",
            "voice_design": "speak angrily",
            "response_format": "wav",
        })
        assert r.status_code == 400
        assert r.json() == {
            "error": {"message": "voice_design is not supported by the kokoro backend. Use qwen3 for instruction-controlled speech."}
        }


def test_clone_rejected_on_piper():
    mock_router = _mock_router()
    with patch.object(main_module, "tts_router", mock_router):
        client = TestClient(app)
        files = {"reference_audio": ("ref.wav", b"fake-bytes", "audio/wav")}
        data = {"input": "hello", "model": "piper/en_US-lessac-medium", "voice": "default"}
        r = client.post("/v1/audio/speech/clone", data=data, files=files)
        assert r.status_code == 400
        assert r.json() == {
            "error": {"message": "Voice cloning is not supported by the piper backend. Use qwen3 or fish-speech."}
        }


def test_backend_capabilities_static_contract():
    from src.tts.backends.kokoro import KokoroBackend
    from src.tts.backends.piper_backend import PiperBackend
    from src.tts.backends.qwen3_backend import Qwen3Backend
    from src.tts.backends.fish_speech_backend import FishSpeechBackend
    from src.tts.backends.f5tts_backend import F5TTSBackend

    expected_matrix = {
        "kokoro": {
            "voice_blend": True,
            "voice_clone": False,
            "voice_design": False,
            "streaming": False,
            "instructions": False,
        },
        "piper": {
            "voice_blend": False,
            "voice_clone": False,
            "voice_design": False,
            "streaming": False,
            "instructions": False,
        },
        "qwen3": {
            "voice_blend": False,
            "voice_clone": True,
            "voice_design": True,
            "streaming": True,
            "instructions": True,
        },
        "fish-speech": {
            "voice_blend": False,
            "voice_clone": True,
            "voice_design": False,
            "streaming": False,
            "instructions": False,
        },
        "f5-tts": {
            "voice_blend": False,
            "voice_clone": True,
            "voice_design": False,
            "streaming": False,
            "instructions": False,
        },
    }

    backends = {
        "kokoro": KokoroBackend.capabilities,
        "piper": PiperBackend.capabilities,
        "qwen3": Qwen3Backend.capabilities,
        "fish-speech": FishSpeechBackend.capabilities,
        "f5-tts": F5TTSBackend.capabilities,
    }

    for backend_name, expected_caps in expected_matrix.items():
        caps = backends[backend_name]
        for key, expected in expected_caps.items():
            assert key in caps
            assert caps[key] is expected

    assert KokoroBackend.capabilities["ssml"] == "partial"
    assert len(KokoroBackend.capabilities["speakers"]) >= 10
    assert len(PiperBackend.capabilities["speakers"]) == 6
    assert len(Qwen3Backend.capabilities["speakers"]) == 9
