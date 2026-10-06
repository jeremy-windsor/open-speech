from __future__ import annotations

import io
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.main import app
from src.voice_library import VoiceLibraryManager, VoiceNotFoundError


def _wav_bytes(
    frame_count: int = 8,
    sample: bytes = b"\x00\x00",
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16000,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sample * frame_count)
    return buffer.getvalue()


FAKE_WAV = _wav_bytes()


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
    wav_v1 = _wav_bytes(frame_count=5)
    wav_v2 = _wav_bytes(frame_count=10, sample=b"\x01\x00")
    lib.save("same", wav_v1, "audio/wav")
    meta2 = lib.save("same", wav_v2, "audio/wav")
    got, meta = lib.get("same")
    assert got == wav_v2
    assert meta["size_bytes"] == len(wav_v2)
    assert meta2["size_bytes"] == len(wav_v2)


def test_failed_overwrite_preserves_existing_audio_and_metadata(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    wav_v1 = _wav_bytes(frame_count=5)
    wav_v2 = _wav_bytes(frame_count=10, sample=b"\x01\x00")
    original_meta = lib.save("same", wav_v1, "audio/wav")

    with (
        patch.object(Path, "write_text", side_effect=OSError("metadata write failed")),
        pytest.raises(OSError, match="metadata write failed"),
    ):
        lib.save("same", wav_v2, "audio/wav")

    stored_audio, stored_meta = lib.get("same")
    assert stored_audio == wav_v1
    assert stored_meta == original_meta


def test_failed_metadata_commit_rolls_back_audio_overwrite(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    wav_v1 = _wav_bytes(frame_count=5)
    wav_v2 = _wav_bytes(frame_count=10, sample=b"\x01\x00")
    original_meta = lib.save("same", wav_v1, "audio/wav")
    meta_path = lib._meta_path("same")
    real_replace = Path.replace

    def fail_metadata_commit(path, target):
        if Path(target) == meta_path and path.name.endswith(".meta.tmp"):
            raise OSError("metadata commit failed")
        return real_replace(path, target)

    with (
        patch.object(Path, "replace", autospec=True, side_effect=fail_metadata_commit),
        pytest.raises(OSError, match="metadata commit failed"),
    ):
        lib.save("same", wav_v2, "audio/wav")

    stored_audio, stored_meta = lib.get("same")
    assert stored_audio == wav_v1
    assert stored_meta == original_meta


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


def test_truncated_wav_rejected(tmp_path: Path):
    """A RIFF/WAVE prefix without complete WAV chunks is not usable audio."""
    lib = VoiceLibraryManager(tmp_path / "voices")
    truncated_wav = b"RIFF\x00\x00\x00\x00WAVE"
    with pytest.raises(ValueError, match="valid WAV format"):
        lib.save("truncated", truncated_wav)


def test_zero_frame_wav_rejected(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(ValueError, match="audio frame"):
        lib.save("silent", _wav_bytes(frame_count=0))


def test_empty_audio_raises(tmp_path: Path):
    """Empty audio bytes must be rejected on save."""
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(ValueError, match="empty"):
        lib.save("empty", b"")


def test_metadata_fields(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("Meta", FAKE_WAV)
    assert set(meta) == {
        "name", "size_bytes", "content_type", "sha256", "created_at",
        "duration_s", "sample_rate", "channels",
    }
    assert len(meta["sha256"]) == 64
    assert meta["name"] == "meta"
    assert meta["size_bytes"] == len(FAKE_WAV)
    assert meta["duration_s"] == 0.0
    assert meta["sample_rate"] == 16000
    assert meta["channels"] == 1
    datetime.fromisoformat(meta["created_at"])


def test_metadata_reports_wav_duration_rate_and_channels(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    stereo_wav = _wav_bytes(
        frame_count=8000,
        sample=b"\x80\x80",
        channels=2,
        sample_width=1,
        sample_rate=8000,
    )

    meta = lib.save("Stereo", stereo_wav)

    assert meta["duration_s"] == 1.0
    assert meta["sample_rate"] == 8000
    assert meta["channels"] == 2


def test_max_seconds_rejects_long_reference(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices", max_seconds=1)
    long_wav = _wav_bytes(frame_count=16001)

    with pytest.raises(ValueError, match=r"too long \(1\.00s\)\. Max: 1s"):
        lib.save("too_long", long_wav)


def test_max_seconds_zero_is_unlimited(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices", max_seconds=0)
    lib.save("long", _wav_bytes(frame_count=16001))
    assert lib.exists("long")


def test_set_transcript_updates_metadata_without_touching_audio(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    before = lib.save("Narrator", FAKE_WAV, transcript="Old words")

    updated = lib.set_transcript("Narrator", "  Correct words  ")
    audio, loaded = lib.get("Narrator")

    assert audio == FAKE_WAV
    assert updated["sha256"] == before["sha256"] == loaded["sha256"]
    assert loaded["transcript"] == "Correct words"


def test_set_transcript_empty_clears_it(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    lib.save("Narrator", FAKE_WAV, transcript="Words")

    updated = lib.set_transcript("Narrator", "  ")

    assert "transcript" not in updated


def test_set_transcript_missing_voice_raises(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(VoiceNotFoundError):
        lib.set_transcript("missing", "Words")


def test_save_and_set_transcript_share_length_limit(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    with pytest.raises(ValueError, match="transcript is too long"):
        lib.save("long", FAKE_WAV, transcript="x" * 10_001)

    lib.save("valid", FAKE_WAV, transcript="Known words")
    with pytest.raises(ValueError, match="transcript is too long"):
        lib.set_transcript("valid", "x" * 10_001)


def test_optional_transcript_is_stored_with_provider_neutral_asset(tmp_path: Path):
    lib = VoiceLibraryManager(tmp_path / "voices")
    meta = lib.save("Narrator", FAKE_WAV, transcript="  Known words.  ")

    assert meta["transcript"] == "Known words."
    _audio, loaded = lib.get("Narrator")
    assert loaded["transcript"] == "Known words."


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


def test_upload_stores_reference_transcript(client_and_lib):
    client, _ = client_and_lib
    resp = client.post(
        "/api/voices/library",
        data={"name": "narrator", "transcript": "Known reference words."},
        files={"audio": ("a.wav", FAKE_WAV, "audio/wav")},
    )

    assert resp.status_code == 201
    assert resp.json()["transcript"] == "Known reference words."


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


def test_get_library_voice_audio_returns_exact_wav(client_and_lib):
    client, lib = client_and_lib
    lib.save("Preview", FAKE_WAV)

    resp = client.get("/api/voices/library/preview/audio")

    assert resp.status_code == 200
    assert resp.content == FAKE_WAV
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_upload_normalizes_spoofed_content_type_to_wav(client_and_lib):
    client, _ = client_and_lib
    upload = client.post(
        "/api/voices/library",
        data={"name": "Spoofed"},
        files={"audio": ("reference.html", FAKE_WAV, "text/html")},
    )

    assert upload.status_code == 201
    assert upload.json()["content_type"] == "audio/wav"

    preview = client.get("/api/voices/library/spoofed/audio")
    assert preview.status_code == 200
    assert preview.content == FAKE_WAV
    assert preview.headers["content-type"] == "audio/wav"
    assert preview.headers["x-content-type-options"] == "nosniff"


def test_get_library_voice_audio_not_found(client_and_lib):
    client, _ = client_and_lib
    assert client.get("/api/voices/library/missing/audio").status_code == 404


def test_patch_library_voice_transcript(client_and_lib):
    client, lib = client_and_lib
    lib.save("Narrator", FAKE_WAV, transcript="Old words")

    resp = client.patch(
        "/api/voices/library/narrator",
        json={"transcript": "Correct words"},
    )

    assert resp.status_code == 200
    assert resp.json()["transcript"] == "Correct words"
    assert client.get("/api/voices/library").json()[0]["transcript"] == "Correct words"


def test_patch_library_voice_transcript_not_found(client_and_lib):
    client, _ = client_and_lib
    resp = client.patch("/api/voices/library/missing", json={"transcript": "Words"})
    assert resp.status_code == 404


def test_patch_without_transcript_is_noop(client_and_lib):
    client, lib = client_and_lib
    lib.save("Narrator", FAKE_WAV, transcript="Keep these words")

    resp = client.patch("/api/voices/library/narrator", json={})

    assert resp.status_code == 200
    assert resp.json()["transcript"] == "Keep these words"


def test_patch_explicit_null_clears_transcript(client_and_lib):
    client, lib = client_and_lib
    lib.save("Narrator", FAKE_WAV, transcript="Remove these words")

    resp = client.patch("/api/voices/library/narrator", json={"transcript": None})

    assert resp.status_code == 200
    assert "transcript" not in resp.json()


def test_voice_library_config_reports_duration_cap(client_and_lib, monkeypatch):
    client, _ = client_and_lib
    monkeypatch.setattr(main_module.settings, "os_voice_library_max_seconds", 42)

    resp = client.get("/api/voices/library-config")

    assert resp.status_code == 200
    assert resp.json() == {"max_seconds": 42}


def test_config_is_available_as_a_voice_name(client_and_lib):
    client, _ = client_and_lib
    upload = client.post(
        "/api/voices/library",
        data={"name": "config"},
        files={"audio": ("reference.wav", FAKE_WAV, "audio/wav")},
    )

    assert upload.status_code == 201
    metadata = client.get("/api/voices/library/config")
    assert metadata.status_code == 200
    assert metadata.json()["name"] == "config"


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
    router.synthesize.side_effect = lambda model, **kwargs: backend.synthesize(**kwargs)
    monkeypatch.setattr(main_module, "tts_router", router)

    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "Ref1", "response_format": "wav"},
    )
    assert resp.status_code == 200
    assert backend.last_kwargs["reference_audio"] == FAKE_WAV


def test_openai_speech_uses_provider_neutral_library_asset(client_and_lib, monkeypatch):
    client, lib = client_and_lib
    lib.save("Narrator", FAKE_WAV, "audio/wav", transcript="Known words.")
    backend = DummyBackend()
    router = MagicMock()
    router.get_backend.return_value = backend
    router.synthesize.side_effect = lambda model, **kwargs: backend.synthesize(**kwargs)
    monkeypatch.setattr(main_module, "tts_router", router)

    resp = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello",
            "model": "qwen3/0.6b-base",
            "voice": "narrator",
            "voice_library_ref": "Narrator",
            "response_format": "wav",
        },
    )

    assert resp.status_code == 200
    assert backend.last_kwargs["reference_audio"] == FAKE_WAV
    assert backend.last_kwargs["clone_transcript"] == "Known words."


def test_patch_transcript_then_clone_uses_corrected_words(client_and_lib, monkeypatch):
    client, lib = client_and_lib
    lib.save("Narrator", FAKE_WAV, transcript="Wrong words.")
    backend = DummyBackend()
    router = MagicMock()
    router.get_backend.return_value = backend
    router.synthesize.side_effect = lambda model, **kwargs: backend.synthesize(**kwargs)
    monkeypatch.setattr(main_module, "tts_router", router)

    corrected = client.patch(
        "/api/voices/library/narrator",
        json={"transcript": "Correct reference words."},
    )
    generated = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello",
            "model": "qwen3/0.6b-base",
            "voice": "narrator",
            "voice_library_ref": "narrator",
            "response_format": "wav",
        },
    )

    assert corrected.status_code == 200
    assert generated.status_code == 200
    assert backend.last_kwargs["clone_transcript"] == "Correct reference words."


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
    router.synthesize.side_effect = lambda model, **kwargs: backend.synthesize(**kwargs)
    monkeypatch.setattr(main_module, "tts_router", router)

    file_wav = _wav_bytes(frame_count=10, sample=b"\xff\x00")
    resp = client.post(
        "/v1/audio/speech/clone",
        data={"input": "Hello", "model": "qwen3-tts-0.6b", "voice_library_ref": "Ref1", "response_format": "wav"},
        files={"reference_audio": ("ref.wav", file_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    assert backend.last_kwargs["reference_audio"] == file_wav
