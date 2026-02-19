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


def test_save_and_get(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    data = b"RIFF....WAVE"
    saved = lib.save("My Voice", data, "audio/wav")

    got_bytes, meta = lib.get("my voice")
    assert got_bytes == data
    assert meta["name"] == "my_voice"
    assert saved["size_bytes"] == len(data)


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
    lib.save("Charlie", b"a")
    lib.save("alpha", b"b")
    lib.save("Bravo", b"c")

    names = [v["name"] for v in lib.list_voices()]
    assert names == ["alpha", "bravo", "charlie"]


def test_delete(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    lib.save("Delete Me", b"x")
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
    lib.save("same", b"111", "audio/wav")
    meta2 = lib.save("same", b"22222", "audio/wav")
    got, meta = lib.get("same")
    assert got == b"22222"
    assert meta["size_bytes"] == 5
    assert meta2["size_bytes"] == 5


def test_name_sanitization(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("My Voice!", b"x")
    assert meta["name"] == "my_voice"


def test_name_too_long(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("a" * 100, b"x")
    assert len(meta["name"]) == 64


def test_empty_name_raises(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(ValueError):
        lib.save("!!!", b"x")


def test_content_type_preserved(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("mp3 one", b"123", "audio/mp3")
    assert meta["content_type"] == "audio/mp3"
    assert (tmp_path / "voices" / "mp3_one.audio.mp3").exists()


def test_metadata_fields(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("Meta", b"abc")
    assert set(meta) == {"name", "size_bytes", "content_type", "created_at"}
    assert meta["name"] == "meta"
    assert meta["size_bytes"] == 3
    datetime.fromisoformat(meta["created_at"])


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
        files={"audio": ("ref.wav", b"RIFF123", "audio/wav")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test_voice"
    assert data["size_bytes"] == 7


def test_upload_voice_invalid_name(client_and_lib):
    client, _ = client_and_lib
    resp = client.post(
        "/api/voices/library",
        data={"name": "!!!"},
        files={"audio": ("ref.wav", b"RIFF123", "audio/wav")},
    )
    assert resp.status_code == 422


def test_list_voices_empty(client_and_lib):
    client, _ = client_and_lib
    resp = client.get("/api/voices/library")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_voices_populated(client_and_lib):
    client, _ = client_and_lib
    client.post("/api/voices/library", data={"name": "b"}, files={"audio": ("a.wav", b"1", "audio/wav")})
    client.post("/api/voices/library", data={"name": "a"}, files={"audio": ("a.wav", b"1", "audio/wav")})
    resp = client.get("/api/voices/library")
    assert [v["name"] for v in resp.json()] == ["a", "b"]


def test_get_voice_meta(client_and_lib):
    client, _ = client_and_lib
    client.post("/api/voices/library", data={"name": "Meta Voice"}, files={"audio": ("a.wav", b"1", "audio/wav")})
    resp = client.get("/api/voices/library/meta voice")
    assert resp.status_code == 200
    assert resp.json()["name"] == "meta_voice"


def test_get_voice_not_found(client_and_lib):
    client, _ = client_and_lib
    resp = client.get("/api/voices/library/missing")
    assert resp.status_code == 404


def test_delete_voice_204(client_and_lib):
    client, _ = client_and_lib
    client.post("/api/voices/library", data={"name": "Gone"}, files={"audio": ("a.wav", b"1", "audio/wav")})
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
    lib.save("Ref1", b"LIBREF", "audio/wav")
    backend = DummyBackend()
    router = MagicMock()
    router.get_backend.return_value = backend
    monkeypatch.setattr(main_module, "tts_router", router)

    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "Ref1", "response_format": "wav"},
    )
    assert resp.status_code == 200
    assert backend.last_kwargs["reference_audio"] == b"LIBREF"


def test_clone_library_ref_not_found(client_and_lib):
    client, _ = client_and_lib
    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "missing", "response_format": "wav"},
    )
    assert resp.status_code == 404


def test_clone_file_takes_precedence_over_ref(client_and_lib, monkeypatch):
    client, lib = client_and_lib
    lib.save("Ref1", b"LIBREF", "audio/wav")
    backend = DummyBackend()
    router = MagicMock()
    router.get_backend.return_value = backend
    monkeypatch.setattr(main_module, "tts_router", router)

    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "Ref1", "response_format": "wav"},
        files={"reference_audio": ("ref.wav", b"FILE_REF", "audio/wav")},
    )
    assert resp.status_code == 200
    assert backend.last_kwargs["reference_audio"] == b"FILE_REF"
