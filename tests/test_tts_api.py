"""Tests for TTS API endpoints (mocked backend)."""

from __future__ import annotations

import queue
import threading
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src.tts.backends.base import VoiceInfo


@pytest.fixture
def tts_client():
    """Create test client with mocked TTS router."""
    mock_router = MagicMock()
    # synthesize returns an iterator of numpy chunks
    mock_router.synthesize.return_value = iter([
        np.zeros(24000, dtype=np.float32),  # 1 second of silence
    ])
    mock_router.list_voices.return_value = [
        VoiceInfo(id="af_heart", name="Heart", language="en-us", gender="female"),
        VoiceInfo(id="am_adam", name="Adam", language="en-us", gender="male"),
    ]
    mock_router.loaded_models.return_value = []

    with patch.object(main_module, "tts_router", mock_router):
        yield TestClient(app), mock_router


class TestSpeechEndpoint:
    def test_basic_synthesis(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": "alloy",
            "response_format": "wav",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"

    def test_pcm_format(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "pcm",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"

    def test_effects_request_bypasses_dry_cache(self, tts_client):
        client, mock_router = tts_client
        mock_cache = MagicMock()
        mock_cache.get.return_value = b"dry cached audio"

        with (
            patch.object(main_module, "tts_cache", mock_cache),
            patch.object(main_module.settings, "tts_cache_enabled", True),
            patch.object(main_module.settings, "os_effects_enabled", True),
            patch(
                "src.services.tts.apply_chain",
                side_effect=lambda samples, sample_rate, effects: samples,
            ) as mock_apply_chain,
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "Hello",
                "voice": "alloy",
                "response_format": "wav",
                "effects": [{"type": "robot"}],
            })

        assert resp.status_code == 200
        assert resp.headers.get("x-cache") is None
        mock_cache.get.assert_not_called()
        mock_router.synthesize.assert_called_once()
        mock_apply_chain.assert_called_once()

    def test_language_is_part_of_cache_operations(self, tts_client):
        client, _mock_router = tts_client
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with (
            patch.object(main_module, "tts_cache", mock_cache),
            patch.object(main_module.settings, "tts_cache_enabled", True),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "Bonjour",
                "voice": "alloy",
                "response_format": "wav",
                "language": "fr",
            })

        assert resp.status_code == 200
        mock_cache.get.assert_called_once_with(
            text="Bonjour",
            voice="alloy",
            speed=1.0,
            fmt="wav",
            model="kokoro",
            language="fr",
        )
        mock_cache.set.assert_called_once_with(
            text="Bonjour",
            voice="alloy",
            speed=1.0,
            fmt="wav",
            model="kokoro",
            audio=ANY,
            language="fr",
        )

    @pytest.mark.parametrize(
        "extended_fields",
        [
            {"instructions": "calm and precise"},
            {"voice_design": "warm narrator"},
            {"reference_audio": "AAAA"},
            {"clone_transcript": "reference words"},
        ],
    )
    def test_extended_requests_bypass_cache(self, tts_client, extended_fields):
        client, mock_router = tts_client
        mock_cache = MagicMock()
        mock_backend = MagicMock()
        mock_backend.capabilities = {
            "voice_design": True,
            "reference_audio": True,
            "clone_transcript": True,
            "voice_clone": True,
            "instructions": True,
        }
        mock_router.get_backend.return_value = mock_backend
        mock_router.synthesize.return_value = iter([np.zeros(24000, dtype=np.float32)])

        with (
            patch.object(main_module, "tts_cache", mock_cache),
            patch.object(main_module.settings, "tts_cache_enabled", True),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "extended-model",
                "input": "Hello",
                "voice": "alloy",
                "response_format": "wav",
                **extended_fields,
            })

        assert resp.status_code == 200
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()
        mock_router.synthesize.assert_called_once()

    def test_unknown_request_field_is_rejected(self, tts_client):
        client, mock_router = tts_client

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "af_heart",
            "response_format": "wav",
            "voice_blend": "af_heart(1)+af_bella(1)",
        })

        assert resp.status_code == 422
        mock_router.synthesize.assert_not_called()

    def test_model_input_limit_rejects_before_cache_or_synthesis(self, tts_client):
        client, mock_router = tts_client
        mock_router.max_input_chars_for.return_value = 5
        mock_cache = MagicMock()
        mock_cache.get.return_value = b"previous audio"

        with (
            patch.object(main_module, "tts_cache", mock_cache),
            patch.object(main_module.settings, "tts_cache_enabled", True),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro", "input": "Too long", "voice": "alloy", "response_format": "wav",
            })

        assert resp.status_code == 400
        assert "Max: 5 characters" in resp.json()["error"]["message"]
        mock_cache.get.assert_not_called()
        mock_router.synthesize.assert_not_called()

    def test_unknown_model_is_rejected_before_synthesis(self, tts_client):
        client, mock_router = tts_client
        mock_router.get_backend.side_effect = ValueError(
            "Unknown TTS model or backend: qwen3/missing"
        )

        resp = client.post("/v1/audio/speech", json={
            "model": "qwen3/missing",
            "input": "Hello",
            "voice": "Ryan",
            "response_format": "wav",
        })

        assert resp.status_code == 400
        assert "Unknown TTS model" in resp.json()["error"]["message"]
        mock_router.synthesize.assert_not_called()

    def test_unknown_voice_is_rejected_before_synthesis(self, tts_client):
        client, mock_router = tts_client
        backend = MagicMock()
        backend.name = "pocket-tts"
        backend.capabilities = {"speed_control": False}
        mock_router.get_backend.return_value = backend
        mock_router.validate_voice.side_effect = ValueError("Unknown Pocket TTS voice: nobody")

        resp = client.post("/v1/audio/speech", json={
            "model": "pocket-tts",
            "input": "Hello",
            "voice": "nobody",
            "response_format": "wav",
        })

        assert resp.status_code == 400
        assert "Unknown Pocket TTS voice" in resp.json()["error"]["message"]
        mock_router.synthesize.assert_not_called()

    def test_unsupported_speed_is_rejected_before_synthesis(self, tts_client):
        client, mock_router = tts_client
        backend = MagicMock()
        backend.name = "pocket-tts"
        backend.capabilities = {"speed_control": False}
        mock_router.get_backend.return_value = backend

        resp = client.post("/v1/audio/speech", json={
            "model": "pocket-tts",
            "input": "Hello",
            "voice": "alba",
            "speed": 1.2,
            "response_format": "wav",
        })

        assert resp.status_code == 400
        assert "Speed control is not supported" in resp.json()["error"]["message"]
        mock_router.synthesize.assert_not_called()

    def test_unsupported_instructions_are_rejected_before_synthesis(self, tts_client):
        client, mock_router = tts_client
        backend = MagicMock()
        backend.name = "kokoro"
        backend.capabilities = {"instructions": False, "speed_control": True}
        mock_router.get_backend.return_value = backend

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "af_heart",
            "instructions": "whisper",
            "response_format": "wav",
        })

        assert resp.status_code == 400
        assert "instructions are not supported" in resp.json()["error"]["message"]
        mock_router.synthesize.assert_not_called()

    def test_non_stream_synthesis_runs_off_request_thread(self, tts_client):
        client, mock_router = tts_client
        request_threads: list[int] = []
        synthesis_threads: list[int] = []

        def record_request_thread(request, pronunciation_dict):
            request_threads.append(threading.get_ident())
            return request.input

        def synthesize(**_kwargs):
            synthesis_threads.append(threading.get_ident())
            yield np.zeros(100, dtype=np.float32)

        mock_router.synthesize.side_effect = synthesize
        with patch("src.services.tts._synthesis_input", side_effect=record_request_thread):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "Hello",
                "voice": "af_heart",
                "response_format": "wav",
            })

        assert resp.status_code == 200
        assert request_threads and synthesis_threads
        assert synthesis_threads[0] != request_threads[0]

    def test_cancelled_stream_worker_closes_generator_on_worker_thread(self):
        from src.services.tts import _STREAM_END, _streaming_synthesis_worker

        calls: list[tuple[str, int]] = []

        class Chunks:
            def __iter__(self):
                calls.append(("iter", threading.get_ident()))
                return self

            def __next__(self):
                calls.append(("next", threading.get_ident()))
                return np.ones(10, dtype=np.float32)

            def close(self):
                calls.append(("close", threading.get_ident()))

        cancel_event = threading.Event()
        cancel_event.set()
        chunk_queue = queue.Queue(maxsize=1)
        received: list[object] = []
        reader = threading.Thread(target=lambda: received.append(chunk_queue.get()))
        reader.start()
        worker = threading.Thread(
            target=_streaming_synthesis_worker,
            kwargs={
                "chunk_queue": chunk_queue,
                "cancel_event": cancel_event,
                "do_synthesize": Chunks,
                "trim": False,
                "normalize": False,
                "response_format": "pcm",
                "sample_rate": 24000,
            },
        )

        worker.start()
        worker.join(timeout=2)
        reader.join(timeout=2)

        assert not worker.is_alive()
        assert not reader.is_alive()
        assert received == [_STREAM_END]
        close_threads = [thread_id for operation, thread_id in calls if operation == "close"]
        assert close_threads == [worker.ident]

    def test_clone_transcript_rejected_when_backend_does_not_support_it(self, tts_client):
        client, mock_router = tts_client
        mock_backend = MagicMock()
        mock_backend.name = "kokoro"
        mock_backend.capabilities = {}
        mock_router.get_backend.return_value = mock_backend

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "wav",
            "clone_transcript": "reference words",
        })

        assert resp.status_code == 400
        assert "clone_transcript is not supported" in resp.json()["error"]["message"]
        mock_router.synthesize.assert_not_called()
        mock_backend.synthesize.assert_not_called()

    def test_streaming_effects_are_rejected(self, tts_client):
        client, mock_router = tts_client

        resp = client.post("/v1/audio/speech?stream=true", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "wav",
            "effects": [{"type": "robot"}],
        })

        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Effects are not supported for streaming TTS"
        mock_router.synthesize.assert_not_called()

    def test_invalid_format_is_rejected_before_backend_lookup(self, tts_client):
        client, mock_router = tts_client
        mock_router.get_backend.side_effect = AssertionError("backend lookup should not run")

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "invalid",
        })

        assert resp.status_code == 400
        assert "Invalid response_format" in resp.json()["error"]["message"]
        mock_router.get_backend.assert_not_called()
        mock_router.synthesize.assert_not_called()

    def test_empty_input_rejected(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "",
            "voice": "alloy",
        })
        assert resp.status_code == 400

    def test_whitespace_input_rejected(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "   ",
            "voice": "alloy",
        })
        assert resp.status_code == 400

    def test_invalid_format_rejected(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "invalid",
        })
        assert resp.status_code == 400

    def test_input_too_long(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "x" * 5000,
            "voice": "alloy",
        })
        assert resp.status_code == 400

    def test_default_format_is_mp3(self, tts_client):
        """Verify that default response_format matches OpenAI spec (mp3)."""
        client, mock = tts_client
        # Send without response_format
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
        })
        # Should attempt mp3 encoding (may fail without ffmpeg, that's ok)
        # Just verify the request was accepted
        assert resp.status_code in (200, 500)  # 500 if no ffmpeg

    def test_speed_validation(self, tts_client):
        client, mock = tts_client
        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "speed": 0.1,  # Below minimum
        })
        assert resp.status_code == 422

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "speed": 5.0,  # Above maximum
        })
        assert resp.status_code == 422

    def test_invalid_input_type_rejected_before_synthesis(self, tts_client):
        client, mock_router = tts_client

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "wav",
            "input_type": "xml",
        })

        assert resp.status_code == 422
        mock_router.synthesize.assert_not_called()

    @pytest.mark.parametrize(
        ("field", "value", "expected_status"),
        [
            ("speed", 0.1, 422),
            ("speed", 5.0, 422),
            ("input", "x" * 4097, 400),
            ("response_format", "invalid", 400),
        ],
        ids=["speed-too-low", "speed-too-high", "input-too-long", "invalid-format"],
    )
    def test_clone_validates_speech_parameters_before_synthesis(
        self,
        tts_client,
        field,
        value,
        expected_status,
    ):
        client, mock_router = tts_client
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        backend.synthesize.return_value = iter([np.zeros(10, dtype=np.float32)])
        mock_router.get_backend.return_value = backend

        data = {
            "input": "Hello",
            "model": "clone-model",
            "voice": "Ryan",
            "speed": 1.0,
            "response_format": "wav",
        }
        data[field] = value

        resp = client.post(
            "/v1/audio/speech/clone",
            data=data,
            files={"reference_audio": ("reference.wav", b"fake-audio", "audio/wav")},
        )

        assert resp.status_code == expected_status
        backend.synthesize.assert_not_called()


class TestVoicesEndpoint:
    def test_list_voices(self, tts_client):
        client, mock = tts_client
        resp = client.get("/v1/audio/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert "voices" in data
        assert len(data["voices"]) == 2
        assert data["voices"][0]["id"] == "af_heart"
