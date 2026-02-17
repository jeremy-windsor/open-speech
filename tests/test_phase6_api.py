from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module


def _wav_bytes():
    from src.audio.preprocessing import float32_mono_to_wav_bytes
    import numpy as np
    return float32_mono_to_wav_bytes(np.zeros(1600, dtype=np.float32), 16000)


def test_tts_cache_hit_header(monkeypatch):
    mock_router = MagicMock()
    mock_router.synthesize.return_value = iter([np.zeros(100, dtype=np.float32)])
    cache = MagicMock()
    cache.get.return_value = b"abc"

    with patch.object(main_module, "tts_router", mock_router), patch.object(main_module, "tts_cache", cache), patch.object(main_module.settings, "tts_cache_enabled", True):
        c = TestClient(app)
        r = c.post("/v1/audio/speech", json={"model": "kokoro", "input": "hi", "voice": "alloy", "response_format": "wav"})
        assert r.status_code == 200
        assert r.headers.get("x-cache") == "HIT"


def test_tts_cache_bypass_query(monkeypatch):
    mock_router = MagicMock()
    mock_router.synthesize.return_value = iter([np.zeros(10, dtype=np.float32)])
    cache = MagicMock()
    cache.get.return_value = b"abc"
    with patch.object(main_module, "tts_router", mock_router), patch.object(main_module, "tts_cache", cache), patch.object(main_module.settings, "tts_cache_enabled", True):
        c = TestClient(app)
        r = c.post("/v1/audio/speech?cache=false", json={"model": "kokoro", "input": "hi", "voice": "alloy", "response_format": "pcm"})
        assert r.status_code == 200
        assert r.headers.get("x-cache") is None


def test_ssml_input_type_accepted():
    mock_router = MagicMock()
    mock_router.synthesize.return_value = iter([np.zeros(10, dtype=np.float32)])
    with patch.object(main_module, "tts_router", mock_router):
        c = TestClient(app)
        r = c.post("/v1/audio/speech", json={
            "model": "kokoro", "input": "<speak>Hello <break time=\"500ms\"/> world</speak>", "input_type": "ssml", "voice": "alloy", "response_format": "pcm"
        })
        assert r.status_code == 200


def test_transcribe_diarize_disabled_returns_400():
    with patch.object(main_module.settings, "stt_diarize_enabled", False):
        c = TestClient(app)
        r = c.post(
            "/v1/audio/transcriptions?diarize=true",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"model": "x"},
        )
        assert r.status_code == 400


def test_transcribe_diarize_import_error_returns_400():
    fake_router = MagicMock()
    fake_router.transcribe.return_value = {"text": "hello"}
    class BadD:
        def __init__(self):
            raise RuntimeError("install open-speech[diarize]")

    with patch.object(main_module, "backend_router", fake_router), patch.object(main_module, "PyannoteDiarizer", BadD), patch.object(main_module.settings, "stt_diarize_enabled", True):
        c = TestClient(app)
        r = c.post(
            "/v1/audio/transcriptions?diarize=true",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"model": "x"},
        )
        assert r.status_code == 400


def test_transcribe_diarize_success_shape():
    fake_router = MagicMock()
    fake_router.transcribe.return_value = {"text": "hello world"}

    class GoodD:
        def diarize(self, wav_bytes):
            from src.diarization.pyannote_diarizer import DiarizationSegment
            return [DiarizationSegment("SPEAKER_00", 0.0, 1.0)]

    with patch.object(main_module, "backend_router", fake_router), patch.object(main_module, "PyannoteDiarizer", GoodD), patch.object(main_module.settings, "stt_diarize_enabled", True):
        c = TestClient(app)
        r = c.post(
            "/v1/audio/transcriptions?diarize=true",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"model": "x"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "text" in data and "segments" in data


def test_cache_key_uses_speed_format(monkeypatch):
    cache = MagicMock()
    cache.get.return_value = None
    mock_router = MagicMock()
    mock_router.synthesize.return_value = iter([np.zeros(10, dtype=np.float32)])
    with patch.object(main_module, "tts_cache", cache), patch.object(main_module, "tts_router", mock_router), patch.object(main_module.settings, "tts_cache_enabled", True):
        c = TestClient(app)
        r = c.post("/v1/audio/speech", json={"model": "kokoro", "input": "hi", "voice": "alloy", "speed": 1.25, "response_format": "pcm"})
        assert r.status_code == 200
        kwargs = cache.get.call_args.kwargs
        assert kwargs["speed"] == 1.25
        assert kwargs["fmt"] == "pcm"


def test_pronunciation_applied_before_tts():
    mock_router = MagicMock()
    mock_router.synthesize.return_value = iter([np.zeros(10, dtype=np.float32)])
    dict_mock = MagicMock()
    dict_mock.apply.return_value = "A W S"
    with patch.object(main_module, "tts_router", mock_router), patch.object(main_module, "pronunciation_dict", dict_mock):
        c = TestClient(app)
        r = c.post("/v1/audio/speech", json={"model": "kokoro", "input": "AWS", "voice": "alloy", "response_format": "pcm"})
        assert r.status_code == 200
        assert mock_router.synthesize.call_args.kwargs["text"] == "A W S"


def test_transcribe_still_works_without_diarize():
    fake_router = MagicMock()
    fake_router.transcribe.return_value = {"text": "ok"}
    with patch.object(main_module, "backend_router", fake_router):
        c = TestClient(app)
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav_bytes(), "audio/wav")}, data={"model": "x"})
        assert r.status_code == 200


def test_tts_response_format_validation_unchanged():
    c = TestClient(app)
    r = c.post("/v1/audio/speech", json={"model": "kokoro", "input": "x", "voice": "alloy", "response_format": "bad"})
    assert r.status_code == 400
