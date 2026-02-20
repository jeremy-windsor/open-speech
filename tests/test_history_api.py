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


def test_history_logs_tts_and_stt_via_api(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = iter([np.zeros(1600, dtype=np.float32)])

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = {"text": "hello there"}

    with patch.object(main_module, "tts_router", mock_tts), patch.object(main_module, "backend_router", mock_stt):
        tts_resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_heart",
            "response_format": "pcm",
        })
        assert tts_resp.status_code == 200

        files = {"file": ("sample.wav", _wav_bytes(), "audio/wav")}
        stt_resp = client.post("/v1/audio/transcriptions", files=files, data={"model": "mock", "response_format": "json"})
        assert stt_resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    types = [i["type"] for i in listing.json()["items"]]
    assert "tts" in types and "stt" in types


def test_history_filter_delete_clear_prune_and_streamed_metadata_only(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)

    hm = main_module.history_manager
    hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "one", None, 10, streamed=False)
    hm.log_stt("mock", "in.wav", "two")

    tts_only = client.get("/api/history?type=tts")
    assert tts_only.status_code == 200
    assert all(i["type"] == "tts" for i in tts_only.json()["items"])

    entry_id = tts_only.json()["items"][0]["id"]
    deleted = client.delete(f"/api/history/{entry_id}")
    assert deleted.status_code == 204

    cleared = client.delete("/api/history")
    assert cleared.status_code == 200
    assert "deleted" in cleared.json()

    main_module.settings.os_history_max_entries = 2
    hm.log_stt("m", "a.wav", "a")
    hm.log_stt("m", "b.wav", "b")
    hm.log_stt("m", "c.wav", "c")
    hm.prune()
    after = hm.list_entries()
    assert after["total"] == 2

    sid = hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "stream", "/tmp/should-not-be-stored.mp3", 111, streamed=True)
    stream_entry = next(i for i in hm.list_entries()["items"] if i["id"] == sid)
    assert stream_entry["streamed"] is True
    assert stream_entry["output_path"] is None
    assert stream_entry["output_bytes"] is None
