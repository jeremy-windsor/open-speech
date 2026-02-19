from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.main import app
from src.voice_library import VoiceLibraryManager, VoiceNotFoundError

# Minimal valid RIFF/WAVE header (12 bytes) — passes _is_wav_bytes() check
FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVE"


def test_save_and_get(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    saved = lib.save("My Voice", FAKE_WAV, "audio/wav")

    got_bytes, meta = lib.get("my voice")
    assert got_bytes == FAKE_WAV
    assert meta["name"] == "my_voice"
    assert saved["size_bytes"] == len(FAKE_WAV)


def test_save_creates_dir(tmp_path: Path):
    path = tmp_path / "nested" / "voices"
    assert not path.exists()
    VoiceLibraryManager(path)
    assert path.exists()


def test_list_empty(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    assert lib.list_voices() == []


def test_list_multiple(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    lib.save("Charlie", FAKE_WAV)
    lib.save("alpha", FAKE_WAV)
    lib.save("Bravo", FAKE_WAV)

    names = [v["name"] for v in lib.list_voices()]
    assert names == ["alpha", "bravo", "charlie"]


def test_delete(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    lib.save("Delete Me", FAKE_WAV)
    assert lib.exists("Delete Me") is True
    lib.delete("Delete Me")
    assert lib.exists("Delete Me") is False


def test_delete_missing(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(VoiceNotFoundError):
        lib.delete("missing")


def test_get_missing(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(VoiceNotFoundError):
        lib.get("missing")


def test_overwrite(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    wav_v1 = FAKE_WAV + b"\x00" * 10
    wav_v2 = FAKE_WAV + b"\x01" * 20
    lib.save("same", wav_v1, "audio/wav")
    meta2 = lib.save("same", wav_v2, "audio/wav")
    got, meta = lib.get("same")
    assert got == wav_v2
    assert meta["size_bytes"] == len(wav_v2)
    assert meta2["size_bytes"] == len(wav_v2)


def test_name_sanitization(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("My Voice!", FAKE_WAV)
    assert meta["name"] == "my_voice"


def test_name_too_long(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("a" * 100, FAKE_WAV)
    assert len(meta["name"]) == 64


def test_empty_name_raises(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(ValueError):
        lib.save("!!!", FAKE_WAV)


def test_non_wav_rejected(tmp_path: Path):
    """Non-WAV bytes (e.g. MP3) must be rejected — backends expect WAV."""
    lib = VoiceLibraryManager(tmp_path / "voices")
    mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x00\xff\xfb"  # MP3 frame
    with pytest.raises(ValueError, match="WAV format"):
        lib.save("mp3_voice", mp3_bytes, "audio/mp3")


def test_empty_audio_raises(tmp_path: Path):
    """Empty audio bytes must be rejected on save."""
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(ValueError, match="empty"):
        lib.save("empty", b"")


def test_metadata_fields(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("Meta", FAKE_WAV)
    assert set(meta) == {"name", "size_bytes", "content_type", "created_at"}
    assert meta["name"] == "meta"
    assert meta["size_bytes"] == len(FAKE_WAV)
    datetime.fromisoformat(meta["created_at"])


def test_max_count_enforced(tmp_path: Path):
    """Library rejects new voices when max_count is reached."""
    lib = VoiceLibraryManager(tmp_path / "voices", max_count=2)
    lib.save("voice1", FAKE_WAV)
    lib.save("voice2", FAKE_WAV)
    with pytest.raises(ValueError, match="full"):
        lib.save("voice3", FAKE_WAV)


def test_max_count_allows_overwrite(tmp_path: Path):
    """Overwriting an existing voice doesn't count against max_count."""
    lib = VoiceLibraryManager(tmp_path / "voices", max_count=2)
    lib.save("voice1", FAKE_WAV)
    lib.save("voice2", FAKE_WAV)
    # Overwrite existing — should succeed even at capacity
    lib.save("voice1", FAKE_WAV)


def test_max_count_zero_is_unlimited(tmp_path: Path):
    """max_count=0 means no limit."""
    lib = VoiceLibraryManager(tmp_path / "voices", max_count=0)
    for i in range(20):
        lib.save(f"voice_{i}", FAKE_WAV)
    assert len(lib.list_voices()) == 20


def test_corrupted_state_missing_audio(tmp_path: Path):
    """list_voices skips entries whose audio file is missing."""
    lib = VoiceLibraryManager(tmp_path / "voices")
    lib.save("ghost", FAKE_WAV)
    # Delete the audio file, leaving only meta.json
    for f in (tmp_path / "voices").glob("ghost.audio.*"):
        f.unlink()
    assert lib.list_voices() == []
    # get() raises VoiceNotFoundError
    with pytest.raises(VoiceNotFoundError):
        lib.get("ghost")


@pytest.fixture
def client_and_lib(tmp_path: Path, monkeypatch):
    lib = VoiceLibraryManager(tmp_path / "voices")
    monkeypatch.setattr(main_module, "voice_library", lib)
    return TestClient(app), lib


def test_upload_voice_201(client_and_lib):
    client, _ = client_and_lib
    resp = client.post(
        "/api/voices/library",
        data={"name": "Test Voice"},
        files={"audio": ("ref.wav", FAKE_WAV, "audio/wav")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test_voice"
    assert data["size_bytes"] == len(FAKE_WAV)


def test_upload_voice_invalid_name(client_and_lib):
    client, _ = client_and_lib
    resp = client.post(
        "/api/voices/library",
        data={"name": "!!!"},
        files={"audio": ("ref.wav", FAKE_WAV, "audio/wav")},
    )
    assert resp.status_code == 422


def test_upload_voice_non_wav_rejected(client_and_lib):
    """Uploading MP3 bytes must be rejected with 422."""
    client, _ = client_and_lib
    mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x00\xff\xfb"
    resp = client.post(
        "/api/voices/library",
        data={"name": "mp3voice"},
        files={"audio": ("ref.mp3", mp3_bytes, "audio/mpeg")},
    )
    assert resp.status_code == 422
    body = resp.json()
    msg = body.get("detail") or body.get("error", {}).get("message", "")
    assert "WAV" in msg


def test_upload_voice_empty_rejected(client_and_lib):
    """Uploading empty audio must be rejected."""
    client, _ = client_and_lib
    resp = client.post(
        "/api/voices/library",
        data={"name": "emptyvoice"},
        files={"audio": ("ref.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 422


def test_list_voices_empty(client_and_lib):
    client, _ = client_and_lib
    resp = client.get("/api/voices/library")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_voices_populated(client_and_lib):
    client, _ = client_and_lib
    client.post("/api/voices/library", data={"name": "b"}, files={"audio": ("a.wav", FAKE_WAV, "audio/wav")})
    client.post("/api/voices/library", data={"name": "a"}, files={"audio": ("a.wav", FAKE_WAV, "audio/wav")})
    resp = client.get("/api/voices/library")
    assert [v["name"] for v in resp.json()] == ["a", "b"]


def test_get_voice_meta(client_and_lib):
    client, _ = client_and_lib
    client.post("/api/voices/library", data={"name": "Meta Voice"}, files={"audio": ("a.wav", FAKE_WAV, "audio/wav")})
    resp = client.get("/api/voices/library/meta voice")
    assert resp.status_code == 200
    assert resp.json()["name"] == "meta_voice"


def test_get_voice_not_found(client_and_lib):
    client, _ = client_and_lib
    resp = client.get("/api/voices/library/missing")
    assert resp.status_code == 404


def test_delete_voice_204(client_and_lib):
    client, _ = client_and_lib
    client.post("/api/voices/library", data={"name": "Gone"}, files={"audio": ("a.wav", FAKE_WAV, "audio/wav")})
    resp = client.delete("/api/voices/library/gone")
    assert resp.status_code == 204


def test_delete_voice_not_found(client_and_lib):
    client, _ = client_and_lib
    resp = client.delete("/api/voices/library/nope")
    assert resp.status_code == 404


class DummyBackend:
    def __init__(self):
        self.capabilities = {"voice_clone": True, "voice_design": True}
        self.last_kwargs = None

    def synthesize(self, text, voice, speed=1.0, lang_code=None, reference_audio=None, clone_transcript=None):
        self.last_kwargs = {
            "text": text,
            "voice": voice,
            "speed": speed,
            "lang_code": lang_code,
            "reference_audio": reference_audio,
            "clone_transcript": clone_transcript,
        }
        yield np.zeros(24000, dtype=np.float32)


def test_clone_with_library_ref(client_and_lib, monkeypatch):
    client, lib = client_and_lib
    lib.save("Ref1", FAKE_WAV, "audio/wav")
    backend = DummyBackend()
    router = MagicMock()
    router.get_backend.return_value = backend
    monkeypatch.setattr(main_module, "tts_router", router)

    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "Ref1", "response_format": "wav"},
    )
    assert resp.status_code == 200
    assert backend.last_kwargs["reference_audio"] == FAKE_WAV


def test_clone_library_ref_not_found(client_and_lib):
    client, _ = client_and_lib
    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "missing", "response_format": "wav"},
    )
    assert resp.status_code == 404


def test_clone_file_takes_precedence_over_ref(client_and_lib, monkeypatch):
    client, lib = client_and_lib
    lib.save("Ref1", FAKE_WAV, "audio/wav")
    backend = DummyBackend()
    router = MagicMock()
    router.get_backend.return_value = backend
    monkeypatch.setattr(main_module, "tts_router", router)

    file_wav = FAKE_WAV + b"\xff" * 10  # different bytes
    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "Ref1", "response_format": "wav"},
        files={"reference_audio": ("ref.wav", file_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    assert backend.last_kwargs["reference_audio"] == file_wav
