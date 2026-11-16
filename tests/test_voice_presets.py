"""Tests for voice presets API."""

from __future__ import annotations

import builtins
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src import storage as storage_module
from src.main import DEFAULT_VOICE_PRESETS, _load_voice_presets, app
from src.services import tts as tts_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module.settings, "os_studio_db_path", str(tmp_path / "studio.db"))
    if storage_module._conn is not None:
        storage_module._conn.close()
    storage_module._conn = None
    storage_module.init_db()
    main_module.profile_manager.import_presets(_load_voice_presets())
    mock_router = MagicMock()
    mock_router.loaded_models.return_value = []
    with patch.object(main_module, "tts_router", mock_router):
        yield TestClient(app)


class TestVoicePresetsAPI:
    def test_get_default_presets(self, client):
        resp = client.get("/api/voice-presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "presets" in data
        assert len(data["presets"]) == 3
        names = [p["name"] for p in data["presets"]]
        assert "Will" in names
        assert "Female" in names
        assert "British Butler" in names

    def test_preset_has_required_fields(self, client):
        resp = client.get("/api/voice-presets")
        for p in resp.json()["presets"]:
            assert "name" in p
            assert "voice" in p
            assert "speed" in p

    def test_load_presets_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("presets:\n  - name: Test\n    voice: af_heart\n    speed: 1.2\n    description: Test preset\n")
            f.flush()
            with patch.dict(os.environ, {"TTS_VOICES_CONFIG": f.name}):
                presets = _load_voice_presets()
                assert len(presets) == 1
                assert presets[0]["name"] == "Test"
        os.unlink(f.name)

    def test_load_presets_missing_file(self):
        with patch.dict(os.environ, {"TTS_VOICES_CONFIG": "/nonexistent/path.yml"}):
            presets = _load_voice_presets()
            assert presets == DEFAULT_VOICE_PRESETS

    def test_load_presets_skips_entries_without_a_name_or_voice(self, tmp_path, monkeypatch, caplog):
        # Startup imports these presets, so one typo ("voices:") must not stop the server.
        # load_voice_presets warns and falls back when TTS_VOICES_CONFIG cannot be used.
        config = tmp_path / "presets.yml"
        config.write_text(
            "presets:\n  narrator:\n    voice: af_heart\n  typo:\n    voices: af_bella\n  bare: af_sky\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TTS_VOICES_CONFIG", str(config))

        with caplog.at_level("WARNING", logger="open-speech"):
            presets = tts_service.load_voice_presets()

        assert presets == [{"voice": "af_heart", "name": "narrator"}]
        assert "'typo'" in caplog.text and "'af_sky'" in caplog.text

    def test_load_presets_without_a_usable_entry_falls_back_to_defaults(self, tmp_path, monkeypatch):
        config = tmp_path / "presets.yml"
        config.write_text("presets:\n  - name: Typo\n    voices: af_bella\n", encoding="utf-8")
        monkeypatch.setenv("TTS_VOICES_CONFIG", str(config))

        assert tts_service.load_voice_presets() == DEFAULT_VOICE_PRESETS

    def test_load_presets_reads_yaml_as_utf8_regardless_of_locale(self, tmp_path, monkeypatch):
        config = tmp_path / "presets.yml"
        config.write_text(
            "presets:\n  narrator:\n    voice: af_heart\n    description: Calm — measured\n",
            encoding="utf-8",
        )
        real_open = builtins.open

        # A Windows source run defaults text files to cp1252.
        def locale_open(file, mode="r", *args, **kwargs):
            if "b" not in mode and "encoding" not in kwargs and len(args) < 2:
                kwargs["encoding"] = "cp1252"
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setenv("TTS_VOICES_CONFIG", str(config))
        monkeypatch.setattr(tts_service, "open", locale_open, raising=False)

        assert tts_service.load_voice_presets() == [
            {"voice": "af_heart", "description": "Calm — measured", "name": "narrator"}
        ]


class TestVersionEndpoint:
    def test_health_version(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["version"] == main_module.get_runtime_version()


class TestExtendedTTSAPI:
    def test_speech_with_voice_design(self):
        """voice_design field is accepted in request body."""
        import numpy as np
        mock_router = MagicMock()
        mock_router.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])
        mock_router.get_backend.return_value = mock_router
        mock_router.capabilities = {"voice_design": True, "voice_clone": True}
        # Make synthesize signature accept voice_design
        mock_router.synthesize.__wrapped__ = None

        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post("/v1/audio/speech", json={
                "model": "qwen3-tts-0.6b",
                "input": "Hello world",
                "voice": "Chelsie",
                "response_format": "wav",
                "voice_design": "warm female voice",
            })
            assert resp.status_code == 200

    def test_speech_with_reference_audio(self):
        """reference_audio field is accepted in request body."""
        import numpy as np
        mock_router = MagicMock()
        mock_router.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])
        mock_router.get_backend.return_value = mock_router
        mock_router.capabilities = {"voice_design": True, "voice_clone": True}

        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post("/v1/audio/speech", json={
                "model": "qwen3-tts-0.6b",
                "input": "Hello world",
                "voice": "default",
                "response_format": "wav",
                "reference_audio": "AAAA",  # base64
            })
            assert resp.status_code == 200

    def test_clone_endpoint(self):
        """Multipart clone endpoint works."""
        import io

        import numpy as np
        mock_router = MagicMock()
        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])
        mock_backend.capabilities = {"voice_design": True, "voice_clone": True}
        mock_router.get_backend.return_value = mock_backend

        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post(
                "/v1/audio/speech/clone",
                data={"input": "Hello", "model": "qwen3-tts-0.6b", "response_format": "wav"},
                files={"reference_audio": ("ref.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            )
            assert resp.status_code == 200

    def test_clone_endpoint_empty_text(self):
        mock_router = MagicMock()
        with patch.object(main_module, "tts_router", mock_router):
            client = TestClient(app)
            resp = client.post(
                "/v1/audio/speech/clone",
                data={"input": "  ", "model": "kokoro", "response_format": "wav"},
            )
            assert resp.status_code == 400
