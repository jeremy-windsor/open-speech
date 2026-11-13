"""Tests for TTS API endpoints (mocked backend)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import queue
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src.cache.tts_cache import TTSCache
from src.pronunciation.dictionary import PronunciationDictionary
from src.tts.backends.base import VoiceInfo
from src.tts.external import ExternalProviderError
from src.tts.pipeline import encode_pcm
from src.tts.router import TTSRouter
from src.voice_library import VoiceLibraryManager


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

    def test_pcm_is_24khz_whatever_the_model_native_rate(self, tts_client):
        # OpenAI's pcm format carries raw 24 kHz samples, so a 22.05 kHz model is resampled.
        client, mock_router = tts_client
        mock_router.sample_rate_for.return_value = 22050
        mock_router.synthesize.return_value = iter([np.full(22050, 0.1, dtype=np.float32)])

        with patch.object(main_module.settings, "tts_cache_enabled", False):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "Hello",
                "voice": "alloy",
                "response_format": "pcm",
            })

        assert resp.status_code == 200
        assert len(resp.content) == 2 * 24000

    def test_clone_pcm_is_24khz_whatever_the_model_native_rate(self, tts_client):
        # README: speech PCM is always 24 kHz, Piper included. The clone endpoint takes a request
        # with no reference for a model that needs none, and its pcm has no header to say otherwise.
        client, mock_router = tts_client
        backend = MagicMock()
        backend.name = "piper"
        backend.capabilities = {"speed_control": True}  # no voice_clone: no reference needed
        mock_router.get_backend.return_value = backend
        mock_router.sample_rate_for.return_value = 22050
        mock_router.synthesize.return_value = iter([np.full(22050, 0.1, dtype=np.float32)])

        resp = client.post("/v1/audio/speech/clone", data={
            "model": "piper/en_US-lessac-medium",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "pcm",
        })

        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/pcm"
        assert len(resp.content) == 2 * 24000

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
        post_processing = {
            "trim": main_module.settings.tts_trim_silence,
            "normalize": main_module.settings.tts_normalize_output,
        }
        mock_cache.get.assert_called_once_with(
            text="Bonjour",
            voice="alloy",
            speed=1.0,
            fmt="wav",
            model="kokoro",
            language="fr",
            **post_processing,
        )
        mock_cache.set.assert_called_once_with(
            text="Bonjour",
            voice="alloy",
            speed=1.0,
            fmt="wav",
            model="kokoro",
            audio=ANY,
            language="fr",
            **post_processing,
        )

    @pytest.mark.parametrize("setting", ["tts_trim_silence", "tts_normalize_output"])
    def test_cached_audio_is_not_reused_after_post_processing_setting_changes(
        self, tts_client, tmp_path, setting
    ):
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([
            np.concatenate([
                np.zeros(1000, dtype=np.float32),
                np.full(1000, 0.5, dtype=np.float32),
                np.zeros(1000, dtype=np.float32),
            ])
        ])
        payload = {"model": "kokoro", "input": "Hello", "voice": "alloy", "response_format": "wav"}

        with (
            patch.object(main_module, "tts_cache", TTSCache(str(tmp_path / "cache"), enabled=True)),
            patch.object(main_module.settings, "tts_cache_enabled", True),
            patch.object(main_module.settings, "tts_trim_silence", True),
            patch.object(main_module.settings, "tts_normalize_output", True),
        ):
            first = client.post("/v1/audio/speech", json=payload)
            with patch.object(main_module.settings, setting, False):
                second = client.post("/v1/audio/speech", json=payload)

        assert first.status_code == 200 and second.status_code == 200
        assert second.headers.get("x-cache") is None
        assert mock_router.synthesize.call_count == 2
        assert second.content != first.content

    def test_enabling_cache_does_not_change_outcome_for_lone_surrogate_text(self, tts_client, tmp_path):
        # JSON allows "\ud83d"; JavaScript clients send it when they cut text mid-emoji.
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.full(240, 0.5, np.float32)])
        body = '{"model":"kokoro","input":"Truncated \\ud83d","voice":"alloy","response_format":"wav"}'
        headers = {"Content-Type": "application/json"}

        uncached = client.post("/v1/audio/speech?cache=false", content=body, headers=headers)
        with (
            patch.object(main_module, "tts_cache", TTSCache(str(tmp_path / "cache"), enabled=True)),
            patch.object(main_module.settings, "tts_cache_enabled", True),
        ):
            stored = client.post("/v1/audio/speech", content=body, headers=headers)
            hit = client.post("/v1/audio/speech", content=body, headers=headers)

        assert uncached.status_code == stored.status_code == hit.status_code == 200
        assert stored.content == uncached.content
        assert hit.headers["x-cache"] == "HIT"
        assert hit.content == stored.content

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

    @staticmethod
    def _record_request_thread(request_threads: list[int]):
        # get_content_type is called directly in the async handler, on the event loop's thread.
        from src.services import tts as tts_service

        real = tts_service.get_content_type

        def record(fmt):
            request_threads.append(threading.get_ident())
            return real(fmt)

        return patch.object(tts_service, "get_content_type", side_effect=record)

    def test_non_stream_synthesis_runs_off_request_thread(self, tts_client):
        client, mock_router = tts_client
        request_threads: list[int] = []
        synthesis_threads: list[int] = []

        def synthesize(**_kwargs):
            synthesis_threads.append(threading.get_ident())
            yield np.zeros(100, dtype=np.float32)

        mock_router.synthesize.side_effect = synthesize
        with self._record_request_thread(request_threads):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "Hello",
                "voice": "af_heart",
                "response_format": "wav",
            })

        assert resp.status_code == 200
        assert request_threads and synthesis_threads
        assert synthesis_threads[0] != request_threads[0]

    def test_ssml_parsing_runs_off_request_thread(self, tts_client):
        # SSML parsing takes time in proportion to the input; on the event loop it would
        # stall every other HTTP and WebSocket session meanwhile.
        from src.pronunciation.dictionary import parse_ssml

        client, mock_router = tts_client
        request_threads: list[int] = []
        parse_threads: list[int] = []

        def record_parse(text):
            parse_threads.append(threading.get_ident())
            return parse_ssml(text)

        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])
        with (
            self._record_request_thread(request_threads),
            patch("src.services.tts.parse_ssml", side_effect=record_parse),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "<speak>Hello</speak>",
                "input_type": "ssml",
                "voice": "af_heart",
                "response_format": "wav",
            })

        assert resp.status_code == 200
        assert request_threads and parse_threads
        assert parse_threads[0] != request_threads[0]
        assert mock_router.synthesize.call_args.kwargs["text"] == "Hello"

    def test_cache_lookup_runs_off_request_thread(self, tts_client, tmp_path):
        # A lookup waits on the cache lock while another request or transport writes and
        # evicts, so on the event loop a hit would stall every other session meanwhile.
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])
        cache = TTSCache(str(tmp_path / "cache"), enabled=True)
        real_get = cache.get
        request_threads: list[int] = []
        lookup_threads: list[int] = []

        def record_get(**kwargs):
            lookup_threads.append(threading.get_ident())
            return real_get(**kwargs)

        payload = {"model": "kokoro", "input": "Hello", "voice": "af_heart", "response_format": "wav"}
        with (
            patch.object(main_module, "tts_cache", cache),
            patch.object(main_module.settings, "tts_cache_enabled", True),
            patch.object(cache, "get", side_effect=record_get),
        ):
            client.post("/v1/audio/speech", json=payload)
            with self._record_request_thread(request_threads):
                hit = client.post("/v1/audio/speech", json=payload)

        assert hit.status_code == 200
        assert hit.headers["x-cache"] == "HIT"
        assert request_threads and len(lookup_threads) == 2
        assert lookup_threads[-1] != request_threads[0]

    def test_library_reference_is_read_off_request_thread_and_not_cached(
        self, tts_client, tmp_path, monkeypatch
    ):
        # A library reference is a file read of up to OS_MAX_UPLOAD_MB; it is sent to the
        # backend as bytes, and a request that clones is never served from the shared cache.
        client, mock_router = tts_client
        reference = io.BytesIO()
        with wave.open(reference, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 1600)
        library = VoiceLibraryManager(tmp_path / "voices")
        library.save("narrator", reference.getvalue(), "audio/wav", transcript="Known words.")
        real_get = library.get
        request_threads: list[int] = []
        read_threads: list[int] = []

        def record_get(name):
            read_threads.append(threading.get_ident())
            return real_get(name)

        monkeypatch.setattr(library, "get", record_get)
        monkeypatch.setattr(main_module, "voice_library", library)
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        mock_router.get_backend.return_value = backend
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])
        mock_cache = MagicMock()

        with (
            self._record_request_thread(request_threads),
            patch.object(main_module, "tts_cache", mock_cache),
            patch.object(main_module.settings, "tts_cache_enabled", True),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "clone-model",
                "input": "Hello",
                "voice": "reference",
                "voice_library_ref": "narrator",
                "response_format": "wav",
            })

        assert resp.status_code == 200, resp.text
        assert request_threads and read_threads
        assert read_threads[0] != request_threads[0]
        sent = mock_router.synthesize.call_args.kwargs
        assert sent["reference_audio"] == reference.getvalue()
        assert sent["clone_transcript"] == "Known words."
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()

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

    @pytest.mark.parametrize(
        ("trim", "first_sample"),
        [(False, 0), (True, 2400 - 1200)],  # trimming keeps 50 ms of the leading silence
        ids=["untrimmed", "trimmed"],
    )
    def test_stream_worker_sends_audio_while_backend_is_still_synthesizing(self, trim, first_sample):
        from src.services.tts import _streaming_synthesis_worker

        first_chunk = np.concatenate([
            np.zeros(2400, dtype=np.float32),
            np.full(240, 0.5, dtype=np.float32),
        ])
        release = threading.Event()

        def do_synthesize():
            yield first_chunk
            # A real backend is still generating the next sentence here.
            release.wait(timeout=5)
            yield np.full(240, 0.25, dtype=np.float32)

        chunk_queue = queue.Queue(maxsize=8)
        cancel_event = threading.Event()
        worker = threading.Thread(
            target=_streaming_synthesis_worker,
            kwargs={
                "chunk_queue": chunk_queue,
                "cancel_event": cancel_event,
                "do_synthesize": do_synthesize,
                "trim": trim,
                "response_format": "pcm",
                "sample_rate": 24000,
            },
            daemon=True,
        )
        worker.start()
        try:
            first = chunk_queue.get(timeout=2)
        finally:
            release.set()
            worker.join(timeout=5)
            cancel_event.set()

        assert first == encode_pcm(first_chunk[first_sample:])

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

    @pytest.mark.parametrize(
        "effect",
        [
            {"type": "reverb", "mix": "heavy"},
            {"type": "reverb", "room": ["small"]},
            {"type": "pitch", "semitones": 1000},
            {"type": "normalize", "target_lufs": None},
        ],
        ids=["reverb-mix-text", "reverb-room-list", "pitch-out-of-range", "normalize-null-target"],
    )
    def test_effect_with_invalid_parameter_is_rejected_before_synthesis(self, tts_client, effect):
        client, mock_router = tts_client

        with patch.object(main_module.settings, "os_effects_enabled", True):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "Hello",
                "voice": "alloy",
                "response_format": "wav",
                "effects": [effect],
            })

        assert resp.status_code == 400
        mock_router.synthesize.assert_not_called()

    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (ExternalProviderError("Model returned invalid audio", code="invalid_audio"), 502),
            (RuntimeError("synthesis broke"), 500),
        ],
    )
    def test_stream_that_fails_before_audio_returns_error_status(self, tts_client, error, status):
        client, mock_router = tts_client

        def synthesize(**_kwargs):
            raise error
            yield  # pragma: no cover

        mock_router.synthesize.side_effect = synthesize
        resp = client.post("/v1/audio/speech?stream=true", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "pcm",
        })

        assert resp.status_code == status
        assert resp.json()["error"]["message"] == str(error)

    def test_stream_sends_audio_after_first_chunk(self, tts_client):
        client, mock_router = tts_client
        mock_router.synthesize.return_value = iter([
            np.full(100, 0.1, dtype=np.float32),
            np.full(100, 0.2, dtype=np.float32),
        ])

        resp = client.post("/v1/audio/speech?stream=true", json={
            "model": "kokoro",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "pcm",
        })

        assert resp.status_code == 200
        assert len(resp.content) == 400

    @pytest.mark.parametrize(
        ("fmt", "chunks"),
        [
            ("wav", [np.zeros(4800, dtype=np.float32)]),
            ("pcm", [np.zeros(4800, dtype=np.float32)]),
            ("wav", []),
        ],
        ids=["silent-wav", "silent-pcm", "no-audio-wav"],
    )
    def test_stream_without_voiced_audio_matches_the_whole_response(self, tts_client, fmt, chunks):
        # Batch trim_silence keeps all-silent audio, and a WAV body is a RIFF file even with
        # no samples. Streaming the same request must not turn that into an empty 200.
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([chunk.copy() for chunk in chunks])
        body = {"model": "kokoro", "input": "...", "voice": "alloy", "response_format": fmt}

        with (
            patch.object(main_module.settings, "tts_trim_silence", True),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            whole = client.post("/v1/audio/speech", json=body)
            streamed = client.post("/v1/audio/speech?stream=true", json=body)

        assert whole.status_code == 200
        assert streamed.status_code == 200
        assert streamed.headers["content-type"] == whole.headers["content-type"]
        assert streamed.content == whole.content
        if fmt == "wav":
            with wave.open(io.BytesIO(streamed.content)) as audio:
                assert audio.getnframes() == sum(len(chunk) for chunk in chunks)

    def test_normalize_effect_on_speech_without_samples_is_not_a_server_error(self, tts_client):
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([])
        body = {"model": "kokoro", "input": "...", "voice": "alloy", "response_format": "wav"}

        with (
            patch.object(main_module.settings, "os_effects_enabled", True),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            plain = client.post("/v1/audio/speech", json=body)
            with_effect = client.post(
                "/v1/audio/speech", json={**body, "effects": [{"type": "normalize"}]}
            )

        assert plain.status_code == 200
        assert with_effect.status_code == 200, with_effect.text
        assert with_effect.content == plain.content

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

    def test_ssml_without_speakable_text_is_rejected(self, tts_client):
        client, mock_router = tts_client

        resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "<speak> </speak>",
            "voice": "alloy",
            "response_format": "wav",
            "input_type": "ssml",
        })

        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Input text is empty"
        mock_router.synthesize.assert_not_called()

    @pytest.mark.parametrize("reason", [
        "Unknown Kokoro voice: af_nobody",
        "Voice reference 'narrator' could not be used",
    ])
    def test_a_refused_voice_answers_400_with_the_reason_in_the_error_envelope(self, tts_client, reason):
        # The web UI shows error.message, and its code in Details, when Generate or a clone test fails
        # (ported from the old UI's flow tests at the swap).
        client, mock_router = tts_client
        backend = MagicMock()
        backend.name = "kokoro"
        backend.capabilities = {"speed_control": True}
        mock_router.get_backend.return_value = backend
        mock_router.validate_voice.side_effect = ValueError(reason)

        resp = client.post("/v1/audio/speech", json={"model": "kokoro", "input": "Hello", "voice": "af_nobody"})

        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error["message"] == reason
        assert isinstance(error.get("code"), str) and error["code"]
        mock_router.synthesize.assert_not_called()

    def test_a_raised_max_input_length_accepts_longer_text(self, tts_client):
        # README: TTS_MAX_INPUT_LENGTH is the longest input. The web UI holds Generate to the limit
        # /api/tts/capabilities reports, so the route must accept everything up to that limit.
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(2400, dtype=np.float32)])
        body = {"model": "kokoro", "voice": "af_heart", "response_format": "wav"}

        with (
            patch.object(main_module.settings, "tts_max_input_length", 6000),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            accepted = client.post("/v1/audio/speech", json={**body, "input": "x" * 6000})
            refused = client.post("/v1/audio/speech", json={**body, "input": "x" * 6001})

        assert accepted.status_code == 200, accepted.text
        assert refused.status_code == 400
        assert "Max: 6000 characters" in refused.json()["error"]["message"]

    def test_the_input_limit_counts_characters_not_utf16_code_units(self, tts_client):
        # The web UI counts code points (what the server counts), not JavaScript's UTF-16 length.
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(2400, dtype=np.float32)])
        text = "Smile \U0001F600 " * 300   # 2,400 characters, 2,700 UTF-16 code units
        assert len(text) <= 2500 < len(text.encode("utf-16-le")) // 2

        with (
            patch.object(main_module.settings, "tts_max_input_length", 2500),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro", "input": text, "voice": "af_heart", "response_format": "pcm",
            })

        assert resp.status_code == 200, resp.text

    def test_ssml_expanded_past_max_input_length_is_rejected(self, tts_client):
        client, mock_router = tts_client
        ssml = '<speak>Hi<break time="10s"/><break time="10s"/></speak>'  # 55 characters

        with patch.object(main_module.settings, "tts_max_input_length", 60):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": ssml,
                "voice": "alloy",
                "response_format": "wav",
                "input_type": "ssml",
            })

        assert resp.status_code == 400
        assert "Max: 60 characters" in resp.json()["error"]["message"]
        mock_router.synthesize.assert_not_called()

    def test_text_within_max_input_length_is_accepted_when_the_dictionary_lengthens_it(
        self, tts_client, tmp_path
    ):
        # TTS_MAX_INPUT_LENGTH limits the client's input ("Input too long"), and OpenAI apps
        # chunk text at the documented 4096. The operator's dictionary is not part of it.
        client, mock_router = tts_client
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])
        dictionary = tmp_path / "dictionary.json"
        dictionary.write_text(json.dumps({"Dr.": "Doctor"}), encoding="utf-8")
        text = "Dr. Smith and Dr. Jones"

        with (
            patch.object(main_module, "pronunciation_dict", PronunciationDictionary(str(dictionary))),
            patch.object(main_module.settings, "tts_max_input_length", len(text)),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": text,
                "voice": "alloy",
                "response_format": "wav",
            })

        assert resp.status_code == 200, resp.text
        assert mock_router.synthesize.call_args.kwargs["text"] == "Doctor Smith and Doctor Jones"

    @pytest.mark.parametrize("path", ["/v1/audio/speech", "/v1/audio/speech?stream=true"])
    def test_language_the_model_cannot_speak_is_rejected_before_synthesis(self, tts_client, path):
        # TTSSpeechRequest.language: "Language code hint (e.g., en, zh, ja, ko)". Kokoro has no
        # "ko" pipeline, and KPipeline would fail its assert inside synthesis (a 500).
        client, mock_router = tts_client
        mock_router.validate_language.side_effect = ValueError("Unsupported language 'ko' for Kokoro.")
        mock_router.synthesize.side_effect = AssertionError(("ko", {"a", "b"}))

        with patch.object(main_module.settings, "tts_cache_enabled", False):
            resp = client.post(path, json={
                "model": "kokoro",
                "input": "Hello",
                "voice": "af_heart",
                "response_format": "pcm",
                "language": "ko",
            })

        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Unsupported language 'ko' for Kokoro."
        mock_router.validate_language.assert_called_with("kokoro", "ko")
        mock_router.synthesize.assert_not_called()

    @pytest.mark.parametrize("supported", [False, True], ids=["model-without-instructions", "model-with-instructions"])
    def test_openai_model_name_keeps_working_when_the_model_ignores_instructions(self, tts_client, supported):
        # gpt-4o-mini-tts takes instructions and README maps OpenAI model names to TTS_MODEL,
        # so an OpenAI app is not refused for them. A native model id keeps the 400.
        client, mock_router = tts_client
        backend = MagicMock()
        backend.name = "kokoro"
        backend.capabilities = {"instructions": supported, "speed_control": True}
        mock_router.get_backend.return_value = backend
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])
        body = {"input": "Hello", "voice": "af_heart", "instructions": "Speak calmly.", "response_format": "wav"}

        with (
            patch.object(main_module.settings, "tts_model", "kokoro"),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            alias = client.post("/v1/audio/speech", json={**body, "model": "gpt-4o-mini-tts"})
            alias_call = mock_router.synthesize.call_args
            native = client.post("/v1/audio/speech", json={**body, "model": "kokoro"})

        assert alias.status_code == 200, alias.text
        assert alias_call.kwargs["model"] == "kokoro"
        assert alias_call.kwargs.get("instructions") == ("Speak calmly." if supported else None)
        assert native.status_code == (200 if supported else 400)

    def test_inline_reference_audio_is_held_to_the_upload_limit(self, tts_client):
        # README: OS_MAX_UPLOAD_MB is the "Max upload size", also for voice references, and
        # /v1/audio/speech/clone returns 413 past it. A reference just over 1 MB still fits
        # the request-body allowance for base64 JSON, so the route holds it to the limit.
        client, mock_router = tts_client
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        mock_router.get_backend.return_value = backend
        reference = base64.b64encode(b"\0" * (1024 * 1024 + 1)).decode("ascii")

        with (
            patch.object(main_module.settings, "os_max_upload_mb", 1),
            patch.object(main_module.settings, "tts_cache_enabled", False),
        ):
            resp = client.post("/v1/audio/speech", json={
                "model": "clone-model",
                "input": "Hello",
                "voice": "reference",
                "response_format": "wav",
                "reference_audio": reference,
            })

        assert resp.status_code == 413, resp.text
        mock_router.synthesize.assert_not_called()

    @pytest.mark.parametrize(
        ("reference_audio", "status", "sent"),
        [
            ("not base64!", 400, None),
            # Decodes to nothing; /v1/audio/speech/clone answers an empty reference with 400.
            (" \n ", 400, None),
            ("UklGRi1y\nZWZlcmVuY2U=", 200, b"RIFF-reference"),
        ],
        ids=["not-base64", "whitespace-only", "line-wrapped-base64"],
    )
    def test_inline_reference_audio_must_be_base64(self, tts_client, reference_audio, status, sent):
        client, mock_router = tts_client
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        mock_router.get_backend.return_value = backend
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])

        with patch.object(main_module.settings, "tts_cache_enabled", False):
            resp = client.post("/v1/audio/speech", json={
                "model": "clone-model",
                "input": "Hello",
                "voice": "reference",
                "response_format": "wav",
                "reference_audio": reference_audio,
            })

        assert resp.status_code == status, resp.text
        if sent is None:
            mock_router.synthesize.assert_not_called()
        else:
            assert mock_router.synthesize.call_args.kwargs["reference_audio"] == sent

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
        encoded = []

        def fake_ffmpeg(audio, fmt, sample_rate=24000):  # no ffmpeg needed on the test machine
            encoded.append(fmt)
            return b"ID3" + bytes(32)

        # Send without response_format
        with patch("src.tts.pipeline.encode_with_ffmpeg", fake_ffmpeg):
            resp = client.post("/v1/audio/speech", json={
                "model": "kokoro",
                "input": "Hello",
                "voice": "alloy",
            })
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/mpeg"
        assert encoded == ["mp3"]
        assert resp.content[:3] == b"ID3"

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

    def test_clone_without_samples_is_still_a_wav_file(self, tts_client):
        # /v1/audio/speech answers a no-sample clip with a header-only WAV; a clone of the
        # same must not be a 200 audio/wav with an empty body.
        client, mock_router = tts_client
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        mock_router.get_backend.return_value = backend
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([])

        resp = client.post(
            "/v1/audio/speech/clone",
            data={"input": "...", "model": "clone-model", "voice": "Ryan", "response_format": "wav"},
            files={"reference_audio": ("reference.wav", b"fake-audio", "audio/wav")},
        )

        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(resp.content)) as audio:
            assert audio.getnframes() == 0

    @pytest.mark.parametrize("endpoint", ["speech", "clone"])
    def test_voice_library_ref_without_usable_name_is_not_found(
        self, tts_client, endpoint, tmp_path, monkeypatch
    ):
        client, mock_router = tts_client
        monkeypatch.setattr(main_module, "voice_library", VoiceLibraryManager(tmp_path / "voices"))
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        mock_router.get_backend.return_value = backend

        if endpoint == "speech":
            resp = client.post("/v1/audio/speech", json={
                "model": "clone-model",
                "input": "Hello",
                "voice": "reference",
                "voice_library_ref": "../",
                "response_format": "wav",
            })
        else:
            resp = client.post("/v1/audio/speech/clone", data={
                "model": "clone-model",
                "input": "Hello",
                "voice_library_ref": "../",
                "response_format": "wav",
            })

        assert resp.status_code == 404
        mock_router.synthesize.assert_not_called()

    def test_clone_speaks_the_pronunciation_dictionary(self, tts_client, tmp_path):
        # TTS_PRONUNCIATION_DICT is server-wide: /v1/audio/speech, Live Reader, Reader and
        # Wyoming all speak its replacements.
        client, mock_router = tts_client
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        mock_router.get_backend.return_value = backend
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])
        dictionary = tmp_path / "dictionary.json"
        dictionary.write_text(json.dumps({"Nguyen": "Win"}), encoding="utf-8")

        with patch.object(main_module, "pronunciation_dict", PronunciationDictionary(str(dictionary))):
            resp = client.post(
                "/v1/audio/speech/clone",
                data={"input": "Hello Nguyen", "model": "clone-model", "voice": "Ryan", "response_format": "wav"},
                files={"reference_audio": ("reference.wav", b"fake-audio", "audio/wav")},
            )

        assert resp.status_code == 200, resp.text
        assert mock_router.synthesize.call_args.kwargs["text"] == "Hello Win"

    def test_clone_without_model_or_format_uses_the_tts_settings(self, tts_client):
        # README: Kokoro, Piper and Pocket-TTS do not clone, so the old "kokoro" default never
        # worked. As on /v1/audio/speech, a left-out model or format comes from TTS_*.
        client, mock_router = tts_client
        backend = MagicMock()
        backend.capabilities = {"voice_clone": True}
        mock_router.get_backend.return_value = backend
        mock_router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(100, dtype=np.float32)])

        with (
            patch.object(main_module.settings, "tts_model", "clone-model"),
            patch.object(main_module.settings, "tts_default_format", "wav"),
        ):
            resp = client.post(
                "/v1/audio/speech/clone",
                data={"input": "Hello"},
                files={"reference_audio": ("reference.wav", b"fake-audio", "audio/wav")},
            )

        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"
        assert mock_router.synthesize.call_args.kwargs["model"] == "clone-model"


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/v1/audio/speech", {"json": {"model": "kokoro", "input": "Hi", "voice": "af_heart", "response_format": "wav"}}),
        ("post", "/v1/audio/speech?stream=true", {"json": {"model": "kokoro", "input": "Hi", "voice": "af_heart", "response_format": "pcm"}}),
        ("post", "/v1/audio/speech/clone", {"data": {"model": "kokoro", "input": "Hi", "response_format": "wav"}}),
        ("get", "/v1/audio/voices", {}),
        ("post", "/v1/audio/models/load", {"json": {"model": "kokoro"}}),
        ("post", "/v1/audio/models/unload", {"json": {"model": "kokoro"}}),
    ],
    ids=["speech", "speech-stream", "clone", "voices", "load", "unload"],
)
def test_tts_endpoints_report_missing_provider_as_503(method, path, body):
    with patch("src.tts.router._discover_backends", return_value={}):
        empty_router = TTSRouter(device="cpu", external_providers="")

    with patch.object(main_module, "tts_router", empty_router):
        resp = getattr(TestClient(app), method)(path, **body)

    assert resp.status_code == 503, resp.text
    assert resp.json()["error"] == {
        "message": "No TTS backends available",
        "code": "provider_missing",
    }


class _SentenceBackend:
    name = "sentences"
    sample_rate = 24000
    capabilities = {"speed_control": True}

    def synthesize(self, text, voice, speed=1.0, lang_code=None, **_options):
        for _ in range(40):
            yield np.full(2400, 0.2, dtype=np.float32)

    def validate_voice(self, voice):
        return None

    def list_voices(self):
        return []

    def loaded_models(self):
        return []


async def test_stream_client_that_stops_reading_does_not_hold_the_provider(tmp_path):
    # TTSRouter serializes inference per provider and holds that lock while its generator
    # runs. A streaming client that reads slowly or stops reading must not keep the lock,
    # or every other request for the provider waits on that client.
    from src.pronunciation.dictionary import PronunciationDictionary
    from src.services.tts import synthesize_speech_response
    from src.tts.models import TTSSpeechRequest

    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("sentences", _SentenceBackend())
    settings = SimpleNamespace(
        tts_enabled=True, tts_max_input_length=4096, os_history_enabled=False,
        tts_trim_silence=False, tts_normalize_output=False, tts_cache_enabled=False,
        os_effects_enabled=False,
    )

    def other_request() -> int:
        return sum(len(chunk) for chunk in router.synthesize(text="Other", model="sentences", voice="v"))

    async def scenario() -> tuple[int, int]:
        response = await synthesize_speech_response(
            request=TTSSpeechRequest(model="sentences", input="Hello", voice="v", response_format="pcm"),
            raw_request=SimpleNamespace(headers={}),
            stream=True,
            cache=False,
            settings=settings,
            tts_router=router,
            tts_cache=TTSCache(str(tmp_path / "cache"), enabled=False),
            pronunciation_dict=PronunciationDictionary(),
            history_manager=MagicMock(),
            voice_library=MagicMock(),
        )
        body = response.body_iterator
        try:
            streamed = len(await body.__anext__())  # the client takes its first audio, then pauses
            other = await asyncio.wait_for(asyncio.to_thread(other_request), timeout=5)
            async for chunk in body:  # the client resumes and still gets the whole clip
                streamed += len(chunk)
        finally:
            await body.aclose()
        return other, streamed

    other_samples, streamed_bytes = await asyncio.wait_for(scenario(), timeout=30)

    assert other_samples == 40 * 2400
    assert streamed_bytes == 40 * 2400 * 2


@pytest.mark.parametrize("path", ["whole", "stream", "cache-hit"])
async def test_slow_history_write_does_not_block_the_event_loop(tmp_path, path):
    # log_tts writes to the studio.db connection every thread shares, waiting for its lock;
    # on the event loop that wait would stall every other client.
    from src.pronunciation.dictionary import PronunciationDictionary
    from src.services.tts import synthesize_speech_response
    from src.tts.models import TTSSpeechRequest

    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("sentences", _SentenceBackend())
    settings = SimpleNamespace(
        tts_enabled=True, tts_max_input_length=4096, os_history_enabled=True,
        tts_trim_silence=False, tts_normalize_output=False, tts_cache_enabled=path == "cache-hit",
        os_effects_enabled=False,
    )
    cache = TTSCache(str(tmp_path / "cache"), enabled=path == "cache-hit")
    cache.set(
        text="Hello", voice="v", speed=1.0, fmt="pcm", model="sentences", audio=b"\x01\x00" * 2400,
        trim=False, normalize=False,
    )
    writing, released = threading.Event(), threading.Event()
    waits: list[bool] = []

    def slow_log_tts(**_entry):
        writing.set()
        waits.append(released.wait(timeout=2))  # set by the loop, while the loop still runs

    history_manager = MagicMock()
    history_manager.log_tts.side_effect = slow_log_tts

    async def release_while_writing() -> None:
        while not writing.is_set():
            await asyncio.sleep(0.01)
        released.set()

    async def speak() -> tuple[int, bytes]:
        response = await synthesize_speech_response(
            request=TTSSpeechRequest(model="sentences", input="Hello", voice="v", response_format="pcm"),
            raw_request=SimpleNamespace(headers={"x-history": "true"}),
            stream=path == "stream",
            cache=True,
            settings=settings,
            tts_router=router,
            tts_cache=cache,
            pronunciation_dict=PronunciationDictionary(),
            history_manager=history_manager,
            voice_library=MagicMock(),
        )
        return response.status_code, b"".join([chunk async for chunk in response.body_iterator])

    (status, body), _ = await asyncio.wait_for(asyncio.gather(speak(), release_while_writing()), timeout=30)

    assert status == 200 and body
    assert waits == [True]
    history_manager.log_tts.assert_called_once()


class _GatedBackend:
    """A provider whose synthesis, once started, waits until the test opens its gate."""

    sample_rate = 24000
    capabilities = {"speed_control": True}

    def __init__(self, name: str, gate: threading.Event | None = None) -> None:
        self.name = name
        self._gate = gate

    def synthesize(self, text, voice, speed=1.0, lang_code=None, **_options):
        if self._gate is not None:
            self._gate.wait(timeout=30)
        yield np.zeros(2400, dtype=np.float32)

    def validate_voice(self, voice):
        return None

    def list_voices(self):
        return []

    def loaded_models(self):
        return []


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
async def test_requests_queued_on_a_busy_provider_leave_the_shared_executor_free(tmp_path, stream):
    # Contract (src/tts/router.py): one provider must not serialize every other provider. Requests
    # queued behind a busy provider must not each hold a thread of asyncio's default executor,
    # which every other provider's request and every to_thread call in the process shares.
    from src.services.tts import synthesize_speech_response
    from src.tts.models import TTSSpeechRequest

    workers = 4
    queued = workers + 2
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=workers))  # this test's loop only
    gate = threading.Event()
    counts = {"arrived": 0, "applied": 0}
    counts_lock = threading.Lock()

    class CountingRouter(TTSRouter):
        def synthesize(self, text, model, voice, speed=1.0, lang_code=None, **options):
            # Runs on the thread that then waits for the provider's synthesis lock.
            if model == "slow":
                with counts_lock:
                    counts["arrived"] += 1
            yield from super().synthesize(text, model, voice, speed, lang_code, **options)

    class CountingDictionary:
        def apply(self, text):
            with counts_lock:
                counts["applied"] += 1
            return text

    with patch("src.tts.router._discover_backends", return_value={}):
        router = CountingRouter(device="cpu", external_providers="")
    router.register_backend("slow", _GatedBackend("slow", gate))
    router.register_backend("fast", _GatedBackend("fast"))
    settings = SimpleNamespace(
        tts_enabled=True, tts_max_input_length=4096, os_history_enabled=False,
        tts_trim_silence=False, tts_normalize_output=False, tts_cache_enabled=False,
        os_effects_enabled=False,
    )
    cache = TTSCache(str(tmp_path / "cache"), enabled=False)

    async def speak(model: str, streamed: bool) -> tuple[int, int]:
        response = await synthesize_speech_response(
            request=TTSSpeechRequest(model=model, input="Hello", voice="v", response_format="pcm"),
            raw_request=SimpleNamespace(headers={}), stream=streamed, cache=False, settings=settings,
            tts_router=router, tts_cache=cache, pronunciation_dict=CountingDictionary(),
            history_manager=MagicMock(), voice_library=MagicMock(),
        )
        size = 0
        async for chunk in response.body_iterator:
            size += len(chunk)
        return response.status_code, size

    def settled() -> bool:
        # One slow request synthesizes behind the gate and the others wait for the provider:
        # either `workers` of them reached the router, or all were set up and dispatched.
        with counts_lock:
            return counts["arrived"] >= workers or (counts["applied"] >= queued and counts["arrived"] >= 1)

    slow = [asyncio.create_task(speak("slow", stream)) for _ in range(queued)]
    try:
        for _ in range(1000):
            if settled():
                break
            await asyncio.sleep(0.01)
        assert settled(), f"the slow requests never reached the provider: {counts}"
        await asyncio.sleep(0.3)  # the last ones dispatch their work

        try:
            fast = await asyncio.wait_for(speak("fast", False), timeout=5)
        except TimeoutError:
            fast = "no answer within 5 s"
        try:
            bare = await asyncio.wait_for(asyncio.to_thread(lambda: "ran"), timeout=2)
        except TimeoutError:
            bare = "did not run within 2 s"
    finally:
        gate.set()
        results = await asyncio.wait_for(asyncio.gather(*slow, return_exceptions=True), timeout=30)

    assert fast == (200, 4800), f"a request for the idle provider: {fast} ({counts})"
    assert bare == "ran", f"a bare asyncio.to_thread {bare}"
    assert results == [(200, 4800)] * queued


class TestVoicesEndpoint:
    def test_list_voices(self, tts_client):
        client, mock = tts_client
        resp = client.get("/v1/audio/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert "voices" in data
        assert len(data["voices"]) == 2
        assert data["voices"][0]["id"] == "af_heart"
