"""Tests for X-History header gating on history logging."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src import storage as storage_module


def _reset_db(tmp_path):
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    main_module.settings.os_history_enabled = True
    main_module.settings.os_history_max_entries = 1000
    main_module.settings.os_history_max_mb = 2000
    main_module.settings.os_history_retain_audio = True
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
