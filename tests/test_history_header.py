"""Tests for X-History header gating on history logging."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src import storage as storage_module
from src.diarization.pyannote_diarizer import DiarizationSegment, DiarizationSetupError
from src.main import app
from src.tts.external import ExternalProviderError


def _reset_db(tmp_path):
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    main_module.settings.os_history_enabled = True
    main_module.settings.os_history_max_entries = 1000
    storage_module._conn = None
    storage_module.init_db()


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _mock_tts():
    m = MagicMock()
    m.synthesize.return_value = iter([np.zeros(1600, dtype=np.float32)])
    return m


def _mock_stt():
    m = MagicMock()
    m.transcribe.return_value = {"text": "hello there"}
    return m


TTS_PAYLOAD = {
    "model": "kokoro",
    "input": "hello",
    "voice": "af_heart",
    "response_format": "pcm",
}


# --- TTS tests ---


def test_tts_no_header_no_history(tmp_path):
    """API call without X-History header -> NO history entry."""
    _reset_db(tmp_path)
    client = TestClient(app)
    with patch.object(main_module, "tts_router", _mock_tts()):
        resp = client.post("/v1/audio/speech", json=TTS_PAYLOAD)
        assert resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 0


def test_tts_header_true_logs_history(tmp_path):
    """API call with X-History: true -> history entry created."""
    _reset_db(tmp_path)
    client = TestClient(app)
    with patch.object(main_module, "tts_router", _mock_tts()):
        resp = client.post("/v1/audio/speech", json=TTS_PAYLOAD, headers={"X-History": "true"})
        assert resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["type"] == "tts"


def test_tts_cache_hit_with_header_true_logs_history(tmp_path):
    """A successful cached response must honor the X-History contract."""
    _reset_db(tmp_path)
    client = TestClient(app)
    cache = MagicMock()
    cache.get.return_value = b"cached-pcm"

    with (
        patch.object(main_module, "tts_router", _mock_tts()),
        patch.object(main_module, "tts_cache", cache),
        patch.object(main_module.settings, "tts_cache_enabled", True),
    ):
        resp = client.post(
            "/v1/audio/speech",
            json=TTS_PAYLOAD,
            headers={"X-History": "true"},
        )

    assert resp.status_code == 200
    assert resp.headers["x-cache"] == "HIT"
    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["output_bytes"] == len(b"cached-pcm")


def test_tts_header_false_no_history(tmp_path):
    """API call with X-History: false -> NO history entry."""
    _reset_db(tmp_path)
    client = TestClient(app)
    with patch.object(main_module, "tts_router", _mock_tts()):
        resp = client.post("/v1/audio/speech", json=TTS_PAYLOAD, headers={"X-History": "false"})
        assert resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 0


# --- STT tests ---


def test_stt_no_header_no_history(tmp_path):
    """STT call without X-History header -> NO history entry."""
    _reset_db(tmp_path)
    client = TestClient(app)
    with patch.object(main_module, "backend_router", _mock_stt()):
        files = {"file": ("sample.wav", _wav_bytes(), "audio/wav")}
        resp = client.post("/v1/audio/transcriptions", files=files, data={"model": "mock", "response_format": "json"})
        assert resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 0


def test_stt_header_true_logs_history(tmp_path):
    """STT call with X-History: true -> history entry created."""
    _reset_db(tmp_path)
    client = TestClient(app)
    with patch.object(main_module, "backend_router", _mock_stt()):
        files = {"file": ("sample.wav", _wav_bytes(), "audio/wav")}
        resp = client.post("/v1/audio/transcriptions", files=files, data={"model": "mock", "response_format": "json"}, headers={"X-History": "true"})
        assert resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["type"] == "stt"


def test_stt_header_false_no_history(tmp_path):
    """STT call with X-History: false -> NO history entry."""
    _reset_db(tmp_path)
    client = TestClient(app)
    with patch.object(main_module, "backend_router", _mock_stt()):
        files = {"file": ("sample.wav", _wav_bytes(), "audio/wav")}
        resp = client.post("/v1/audio/transcriptions", files=files, data={"model": "mock", "response_format": "json"}, headers={"X-History": "false"})
        assert resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 0


# --- Master kill switch test ---


def test_history_disabled_ignores_header(tmp_path):
    """os_history_enabled=false -> nothing logs even with X-History: true."""
    _reset_db(tmp_path)
    main_module.settings.os_history_enabled = False
    client = TestClient(app)
    with patch.object(main_module, "tts_router", _mock_tts()), patch.object(main_module, "backend_router", _mock_stt()):
        client.post("/v1/audio/speech", json=TTS_PAYLOAD, headers={"X-History": "true"})
        files = {"file": ("sample.wav", _wav_bytes(), "audio/wav")}
        client.post("/v1/audio/transcriptions", files=files, data={"model": "mock", "response_format": "json"}, headers={"X-History": "true"})

    # Re-enable to query
    main_module.settings.os_history_enabled = True
    listing = client.get("/api/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 0


# --- A request that answers an error is not logged ---
# A streamed /v1/audio/speech logged its entry before the first chunk, so a stream that failed
# before any audio (400, 503, 500) left a History row, and a transcription was logged before its diarization ran.
# README: "Send `X-History: true` to log the request"; the non-streamed path logs only what succeeded.


@pytest.fixture
def history_db(tmp_path):
    with (
        patch.object(main_module.settings, "os_studio_db_path", str(tmp_path / "studio.db")),
        patch.object(main_module.settings, "os_history_enabled", True),
        patch.object(main_module.settings, "tts_cache_enabled", False),
    ):
        storage_module._conn = None
        storage_module.init_db()
        yield
        storage_module._conn = None


def _stub_tts_router(synthesize):
    backend = MagicMock()
    backend.name = "stub"
    backend.capabilities = {"speed_control": True}
    router = MagicMock()
    router.get_backend.return_value = backend
    router.get_capabilities.return_value = backend.capabilities
    router.max_input_chars_for.return_value = None
    router.sample_rate_for.return_value = 24000
    router.synthesize.side_effect = synthesize
    return router


def _refuses(**_kwargs):
    raise ExternalProviderError("This voice doesn't speak Japanese", status_code=400, code="language_unsupported")
    yield  # pragma: no cover


def _offline(**_kwargs):
    raise ExternalProviderError("Worker offline", status_code=503, code="provider_unavailable")
    yield  # pragma: no cover


def _fails(**_kwargs):
    raise RuntimeError("backend failed")
    yield  # pragma: no cover


def _speaks(**_kwargs):
    yield np.full(2400, 0.1, dtype=np.float32)


_STUB_BODY = {"model": "stub", "input": "Hello there", "voice": "v", "response_format": "pcm"}


def _history_items(client):
    return [(i["type"], i["full_text"], i.get("streamed")) for i in client.get("/api/history").json()["items"]]


@pytest.mark.parametrize(
    ("synthesize", "status"), [(_refuses, 400), (_offline, 503), (_fails, 500)],
    ids=["provider-400", "provider-503", "backend-500"],
)
@pytest.mark.parametrize("path", ["/v1/audio/speech?stream=true", "/v1/audio/speech"], ids=["streamed", "whole"])
def test_a_speech_request_that_fails_before_any_audio_is_not_logged(history_db, path, synthesize, status):
    client = TestClient(app, raise_server_exceptions=False)
    with patch.object(main_module, "tts_router", _stub_tts_router(synthesize)):
        resp = client.post(path, json=_STUB_BODY, headers={"X-History": "true"})
    assert resp.status_code == status, resp.text
    assert _history_items(client) == []


def test_a_streamed_speech_request_that_sends_audio_is_logged_once(history_db):
    client = TestClient(app)
    with patch.object(main_module, "tts_router", _stub_tts_router(_speaks)):
        resp = client.post("/v1/audio/speech?stream=true", json=_STUB_BODY, headers={"X-History": "true"})
    assert resp.status_code == 200
    assert resp.content
    assert _history_items(client) == [("tts", "Hello there", True)]


class _RefusedDiarizer:
    def __init__(self):
        raise DiarizationSetupError("accept pyannote/speaker-diarization-3.1 on Hugging Face")


class _CrashingDiarizer:
    def diarize(self, _audio):
        raise RuntimeError("CUDA out of memory")


class _OneSpeakerDiarizer:
    def diarize(self, _audio):
        return [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=0.1)]


@pytest.mark.parametrize(
    ("diarizer", "status"),
    [(_RefusedDiarizer, 400), (_CrashingDiarizer, 500), (_OneSpeakerDiarizer, 200)],
    ids=["setup-refused-400", "diarize-failed-500", "diarized-200"],
)
def test_a_diarized_transcription_is_logged_only_when_diarization_succeeds(history_db, diarizer, status):
    stt = _mock_stt()
    client = TestClient(app)
    with (
        patch.object(main_module, "backend_router", stt),
        patch.object(main_module, "PyannoteDiarizer", diarizer),
        patch.object(main_module.settings, "stt_diarize_enabled", True),
    ):
        resp = client.post(
            "/v1/audio/transcriptions?diarize=true",
            files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
            data={"model": "mock"},
            headers={"X-History": "true"},
        )
    assert resp.status_code == status, resp.text
    assert _history_items(client) == ([("stt", "hello there", False)] if status == 200 else [])
