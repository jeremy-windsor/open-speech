"""Model-free integration tests for the Live Reader WebSocket."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket

from src import main as main_module
from src.live_tts.server import (
    LiveTTSSession,
    _admission,
    _synthesize_worker,
    live_tts_endpoint,
    reset_live_tts_state_for_tests,
)
from src.main import app
from tests._watchdog import hard_watchdog


class _ThreadBoundChunks:
    def __init__(self, calls: list[tuple[str, int]], samples: int = 1200) -> None:
        self.calls = calls
        self.samples = samples
        self.sent = False

    def __iter__(self):
        self.calls.append(("iter", threading.get_ident()))
        return self

    def __next__(self):
        self.calls.append(("next", threading.get_ident()))
        if self.sent:
            raise StopIteration
        self.sent = True
        return np.full(self.samples, 0.25, dtype=np.float32)

    def close(self) -> None:
        self.calls.append(("close", threading.get_ident()))


class _FakeRouter:
    def __init__(self, *, sample_rate: int = 24000, samples: int = 1200) -> None:
        self.calls: list[tuple[str, int]] = []
        self.requests: list[dict] = []
        self.sample_rate = sample_rate
        self.samples = samples

    def sample_rate_for(self, _model: str) -> int:
        return self.sample_rate

    def synthesize(self, **kwargs):
        self.calls.append(("create", threading.get_ident()))
        self.requests.append(kwargs)
        return _ThreadBoundChunks(self.calls, self.samples)


class _BlockingChunks(_ThreadBoundChunks):
    def __init__(
        self,
        calls: list[tuple[str, int]],
        waiting: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(calls)
        self.waiting = waiting
        self.release = release
        self.index = 0

    def __next__(self):
        self.calls.append(("next", threading.get_ident()))
        if self.index == 0:
            self.index += 1
            return np.full(1200, 0.25, dtype=np.float32)
        if self.index == 1:
            self.waiting.set()
            if not self.release.wait(timeout=3):
                raise RuntimeError("test did not release blocking backend")
            self.index += 1
            return np.full(1200, 0.25, dtype=np.float32)
        raise StopIteration


class _BlockingRouter(_FakeRouter):
    def __init__(self) -> None:
        super().__init__()
        self.waiting = threading.Event()
        self.release = threading.Event()

    def synthesize(self, **kwargs):
        self.calls.append(("create", threading.get_ident()))
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return _BlockingChunks(self.calls, self.waiting, self.release)
        return _ThreadBoundChunks(self.calls)


class _ErrorRouter(_FakeRouter):
    def synthesize(self, **kwargs):
        self.requests.append(kwargs)

        def fail():
            raise RuntimeError("private cache path /models/user-name")
            yield

        return fail()


class _Pronunciation:
    @staticmethod
    def apply(text: str) -> str:
        return text


@pytest.fixture(autouse=True)
def _reset_admission():
    reset_live_tts_state_for_tests()
    yield
    reset_live_tts_state_for_tests()


@pytest.fixture
def live_client(monkeypatch):
    fake_router = _FakeRouter()
    monkeypatch.setattr(main_module, "tts_router", fake_router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    monkeypatch.setattr(main_module.settings, "tts_live_max_connections", 1)
    return TestClient(app), fake_router


def _configure(websocket, *, sample_rate: int = 24000, speed: float = 1.0):
    created = websocket.receive_json()
    assert created["type"] == "session.created"
    assert created["session"]["audio"] == {
        "format": "pcm16",
        "encoding": "signed little-endian",
        "sample_rate": sample_rate,
        "channels": 1,
    }
    websocket.send_json(
        {
            "type": "session.update",
            "session": {
                "model": "kokoro",
                "voice": "af_heart",
                "speed": speed,
                "latency_mode": "natural",
            },
        }
    )
    assert websocket.receive_json()["type"] == "session.updated"


def test_live_tts_streams_pcm_and_owns_generator_on_one_worker_thread(live_client):
    client, fake_router = live_client
    main_thread = threading.get_ident()

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket)
        websocket.send_json({"type": "input_text.append", "text": "Hello world."})
        assert websocket.receive_json()["type"] == "input_text.accepted"
        websocket.send_json({"type": "input_text.commit"})
        events = []
        while not any(event["type"] == "input_text.committed" for event in events):
            events.append(websocket.receive_json())
        while not any(event["type"] == "input_text.done" for event in events):
            events.append(websocket.receive_json())

    event_types = [event["type"] for event in events]
    assert "response.created" in event_types
    assert "response.output_audio.delta" in event_types
    assert "response.done" in event_types
    assert "input_text.done" in event_types
    delta = next(event for event in events if event["type"] == "response.output_audio.delta")
    assert delta["source_start"] == 0
    assert delta["source_end"] == len("Hello world.")
    assert delta["text"] == "Hello world."
    expected_pcm = (np.full(1200, 0.25, dtype=np.float32) * 32767.0).astype("<i2").tobytes()
    assert base64.b64decode(delta["delta"]) == expected_pcm
    assert len(base64.b64decode(delta["delta"])) == 2400
    assert delta["sample_rate"] == 24000
    assert fake_router.requests[0]["text"] == "Hello world."

    worker_threads = {thread_id for _operation, thread_id in fake_router.calls}
    assert len(worker_threads) == 1
    assert main_thread not in worker_threads
    assert [operation for operation, _thread_id in fake_router.calls] == [
        "create",
        "iter",
        "next",
        "next",
        "close",
    ]


def test_live_tts_resolves_voice_library_reference(monkeypatch):
    class CloneRouter(_FakeRouter):
        backend = SimpleNamespace(
            name="clone",
            capabilities={"voice_clone": True, "clone_transcript_required": True},
        )

        def get_backend(self, _model):
            return self.backend

        def get_capabilities(self, _model):
            return dict(self.backend.capabilities)

        def validate_voice(self, _model, _voice):
            return None

    class VoiceLibrary:
        @staticmethod
        def get(name):
            assert name == "alex_reference"
            return b"reference-wav", {"transcript": "Known words."}

    router = CloneRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "voice_library", VoiceLibrary())
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "clone/model",
                    "voice": "reference",
                    "voice_library_ref": "alex_reference",
                },
            }
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["voice_library_ref"] == "alex_reference"
        websocket.send_json({"type": "input_text.append", "text": "Hello clone."})
        websocket.send_json({"type": "input_text.commit"})
        while websocket.receive_json()["type"] != "input_text.done":
            pass

    assert router.requests[0]["reference_audio"] == b"reference-wav"
    assert router.requests[0]["clone_transcript"] == "Known words."


def test_session_update_rejects_unsupported_model_speed(monkeypatch):
    class CapabilityRouter(_FakeRouter):
        backend = SimpleNamespace(name="qwen3", capabilities={"speed_control": False})

        def get_backend(self, _model):
            return self.backend

        def get_capabilities(self, _model):
            return dict(self.backend.capabilities)

        def validate_voice(self, _model, _voice):
            return None

    router = CapabilityRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "qwen3/0.6b-custom-voice",
                    "voice": "Ryan",
                    "speed": 1.2,
                },
            }
        )
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert "Speed control is not supported" in event["error"]["message"]


@pytest.mark.parametrize("tts_speed", [1.0, 1.25], ids=["TTS_SPEED=1.0", "TTS_SPEED=1.25"])
def test_session_update_without_speed_uses_one_on_a_fixed_speed_model(monkeypatch, request, tts_speed):
    # TTS_SPEED is the session's default speed, not a speed the client set, so a
    # session.update that picks a fixed-speed model (pocket-tts, chatterbox) and sends no speed is spoken at 1.0.
    # An explicit speed is still refused (test_session_update_rejects_unsupported_model_speed), and a model with speed
    # control keeps TTS_SPEED.
    class CapabilityRouter(_FakeRouter):
        def get_backend(self, model):
            return SimpleNamespace(name=model.split("/")[0], capabilities=self.get_capabilities(model))

        def get_capabilities(self, model):
            return {"speed_control": model == "kokoro"}

        def validate_voice(self, _model, _voice):
            return None

    monkeypatch.setattr(main_module, "tts_router", CapabilityRouter())
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    monkeypatch.setattr(main_module.settings, "tts_speed", tts_speed)
    client = TestClient(app)

    events = []
    with hard_watchdog(request, 30), client.websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        for model, voice in (("pocket-tts", "alba"), ("chatterbox/turbo", "reference"), ("kokoro", "af_heart")):
            websocket.send_json({"type": "session.update", "session": {"model": model, "voice": voice}})
            events.append((model, websocket.receive_json()))

    for model, event in events:
        assert event["type"] == "session.updated", f"TTS_SPEED={tts_speed}, {model} with no speed: {event}"
    speeds = {model: event["session"]["speed"] for model, event in events}
    assert speeds == {"pocket-tts": 1.0, "chatterbox/turbo": 1.0, "kokoro": tts_speed}


@pytest.mark.parametrize(
    "tts_model, tts_voice, expected_speed",
    [("pocket-tts", "alba", 1.0), ("kokoro", "af_heart", 1.25)],
    ids=["fixed-speed-model", "model-with-speed-control"],
)
def test_a_session_with_no_update_speaks_a_fixed_speed_default_model_at_one(
    monkeypatch, request, tts_model, tts_voice, expected_speed
):
    # The omitted-speed rule holds for the session's start state too. A client that sends
    # text without a session.update has omitted speed, so TTS_SPEED=1.25 on a TTS_MODEL with no speed control
    # (Pocket-TTS refuses any other speed) is spoken at 1.0, and session.created says so; a model with speed control
    # keeps TTS_SPEED.
    class CapabilityRouter(_FakeRouter):
        def get_backend(self, model):
            return SimpleNamespace(name=model, capabilities=self.get_capabilities(model))

        def get_capabilities(self, model):
            return {"speed_control": model == "kokoro"}

        def validate_voice(self, _model, _voice):
            return None

        def synthesize(self, **kwargs):
            if kwargs["model"] == "pocket-tts" and kwargs.get("speed", 1.0) != 1.0:
                self.requests.append(kwargs)
                raise ValueError("Speed control is not supported by the Pocket TTS backend.")
            return super().synthesize(**kwargs)

    router = CapabilityRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    monkeypatch.setattr(main_module.settings, "tts_model", tts_model)
    monkeypatch.setattr(main_module.settings, "tts_voice", tts_voice)
    monkeypatch.setattr(main_module.settings, "tts_speed", 1.25)
    client = TestClient(app)

    events = []
    with hard_watchdog(request, 30), client.websocket_connect("/v1/audio/speech/stream") as websocket:
        created = websocket.receive_json()
        assert created["type"] == "session.created"
        websocket.send_json({"type": "input_text.append", "text": "Hello there. "})
        websocket.send_json({"type": "input_text.commit"})
        while not events or events[-1]["type"] != "input_text.done":
            events.append(websocket.receive_json())

    assert [event for event in events if event["type"] == "error"] == []
    assert [event["response"]["status"] for event in events if event["type"] == "response.done"] == ["completed"]
    assert [call["speed"] for call in router.requests] == [expected_speed]
    assert created["session"]["speed"] == expected_speed


def test_session_update_rejects_model_without_live_reader(monkeypatch):
    class CapabilityRouter(_FakeRouter):
        backend = SimpleNamespace(
            name="qwen3",
            capabilities={"speed_control": False, "live_reader": False},
        )

        def get_backend(self, _model):
            return self.backend

        def get_capabilities(self, _model):
            return dict(self.backend.capabilities)

        def validate_voice(self, _model, _voice):
            return None

    monkeypatch.setattr(main_module, "tts_router", CapabilityRouter())
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "qwen3/0.6b-base",
                    "voice": "logical-voice",
                    "speed": 1.0,
                },
            }
        )
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert "Live Reader is not supported" in event["error"]["message"]


@pytest.mark.asyncio
async def test_session_update_with_a_language_the_model_cannot_speak_keeps_the_previous_session(monkeypatch):
    # Kokoro has no Korean pipeline. The update was accepted and every segment then failed.
    from src.tts.backends import kokoro as kokoro_module
    from src.tts.router import TTSRouter

    class SilentKokoro(kokoro_module.KokoroBackend):
        def synthesize(self, text, voice, speed=1.0, lang_code=None, **_kwargs):
            yield np.zeros(1200, dtype=np.float32)

    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
    router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("kokoro", SilentKokoro(device="cpu"))

    def update(**session):
        return {"type": "websocket.receive", "text": json.dumps({"type": "session.update", "session": session})}

    sent = await _serve_asgi([
        update(model="kokoro", voice="af_heart", language="en-us"),
        update(model="kokoro", voice="af_heart", language="ko"),
        update(speed=1.0),
    ], router=router)

    events = _events(sent)
    assert [event["type"] for event in events] == ["session.created", "session.updated", "error", "session.updated"]
    assert events[2]["error"]["code"] == "invalid_event"
    assert "Unsupported language 'ko' for Kokoro" in events[2]["error"]["message"]
    assert events[3]["session"]["language"] == "en-us"


@pytest.mark.asyncio
async def test_session_update_with_a_language_pocket_tts_cannot_speak_keeps_the_previous_session(monkeypatch):
    # docs/API.md (Live Reader): a session.update with "a language it
    # cannot speak" gets an invalid_event error and the previous settings stay in effect. Pocket
    # TTS declares English only, yet Korean was accepted and every segment read in English.
    from src.tts.backends.pocket_tts_backend import PocketTTSBackend
    from src.tts.router import TTSRouter

    class SilentPocket(PocketTTSBackend):
        def synthesize(self, text, voice, speed=1.0, lang_code=None, **_kwargs):
            yield np.zeros(1200, dtype=np.float32)

    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
    router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("pocket-tts", SilentPocket(device="cpu"))

    def update(**session):
        return {"type": "websocket.receive", "text": json.dumps({"type": "session.update", "session": session})}

    sent = await _serve_asgi([
        update(model="pocket-tts", voice="alba", language="en"),
        update(model="pocket-tts", voice="alba", language="ko"),
        update(speed=1.0),
    ], router=router)

    events = _events(sent)
    assert [event["type"] for event in events] == ["session.created", "session.updated", "error", "session.updated"]
    assert events[2]["error"]["code"] == "invalid_event"
    assert "Unsupported language 'ko' for pocket-tts" in events[2]["error"]["message"]
    assert events[3]["session"]["language"] == "en"


@pytest.mark.asyncio
async def test_session_update_with_a_language_qwen3_cannot_speak_keeps_the_previous_session(monkeypatch):
    # docs/API.md (Live Reader): a session.update with "a language it
    # cannot speak" gets an invalid_event error and the previous settings stay in effect. Qwen3's
    # manifest names its languages ("Auto", "English", ...) instead of giving codes, so Dutch was
    # accepted and every segment then failed at the worker with 400 language_unsupported.
    import io
    import urllib.error

    from src.tts.external import ExternalTTSBackend
    from src.tts.router import TTSRouter

    model = "qwen3/0.6b-custom-voice"
    languages = [  # providers/qwen3/app.py LANGUAGES
        "Auto", "Chinese", "English", "Japanese", "Korean", "German",
        "French", "Russian", "Portuguese", "Spanish", "Italian",
    ]
    manifest = {
        "schema_version": 1,
        "provider": "qwen3",
        "models": [{
            "id": model, "provider": "qwen3", "sample_rate": 24000, "revision": "test-1",
            "capabilities": {
                "voice_clone": False, "instructions": False, "speed_control": False,
                "live_reader": True, "languages": languages,
            },
            "voices": [{"id": "Ryan", "name": "Ryan", "language": "en", "gender": "male"}],
        }],
    }
    health = {"status": "ok", "provider": "qwen3", "loaded_model": None}
    other_requests: list[str] = []

    class Response(io.BytesIO):
        headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def open_fn(request, **_kwargs):
        if request.full_url.endswith("/v1/manifest"):
            return Response(json.dumps(manifest).encode())
        if request.full_url.endswith("/health"):
            return Response(json.dumps(health).encode())
        other_requests.append(request.full_url)
        raise urllib.error.URLError("this worker loads and speaks nothing")

    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
    router = TTSRouter(device="cpu", external_providers="")
    backend = ExternalTTSBackend(provider="qwen3", base_url="http://qwen3:8200")
    monkeypatch.setattr(backend._opener, "open", open_fn)
    router.register_backend("qwen3", backend)

    def update(**session):
        return {"type": "websocket.receive", "text": json.dumps({"type": "session.update", "session": session})}

    sent = await _serve_asgi([
        update(model=model, voice="Ryan", language="en"),
        update(model=model, voice="Ryan", language="nl"),
        update(model=model, voice="Ryan", language="English"),
        update(model=model, voice="Ryan", language="nl-NL"),
        update(speed=1.0),
    ], router=router)

    events = _events(sent)
    assert [event["type"] for event in events] == [
        "session.created", "session.updated", "error", "session.updated", "error", "session.updated",
    ], events
    for refused, language in ((events[2], "nl"), (events[4], "nl-NL")):
        assert refused["error"]["code"] == "invalid_event"
        assert f"Unsupported language '{language}' for {model}" in refused["error"]["message"]
    assert events[1]["session"]["language"] == "en"
    assert events[3]["session"]["language"] == "English"
    assert events[5]["session"]["language"] == "English"
    assert other_requests == []


def test_live_tts_trims_non_streaming_backend_edges(monkeypatch):
    class SilentEdgeRouter(_FakeRouter):
        def synthesize(self, **kwargs):
            self.requests.append(kwargs)
            return iter(
                [
                    np.concatenate(
                        [
                            np.zeros(24000, dtype=np.float32),
                            np.full(1200, 0.25, dtype=np.float32),
                            np.zeros(24000, dtype=np.float32),
                        ]
                    )
                ]
            )

        @staticmethod
        def get_capabilities(_model: str):
            return {"streaming": False}

    router = SilentEdgeRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket)
        websocket.send_json({"type": "input_text.append", "text": "Trim this."})
        websocket.send_json({"type": "input_text.commit"})
        audio_bytes = bytearray()
        while True:
            event = websocket.receive_json()
            if event["type"] == "response.output_audio.delta":
                audio_bytes.extend(base64.b64decode(event["delta"]))
            if event["type"] == "response.done":
                break

    expected_samples = 1200 + 1200 + 3600
    assert len(audio_bytes) == expected_samples * 2


def test_live_tts_preserves_native_backend_chunks(monkeypatch):
    class NativeStreamingRouter(_FakeRouter):
        def synthesize(self, **kwargs):
            self.requests.append(kwargs)
            return iter(
                [
                    np.zeros(1200, dtype=np.float32),
                    np.full(1200, 0.25, dtype=np.float32),
                ]
            )

        @staticmethod
        def get_capabilities(_model: str):
            return {"streaming": True}

    router = NativeStreamingRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket)
        websocket.send_json({"type": "input_text.append", "text": "Stream this."})
        websocket.send_json({"type": "input_text.commit"})
        deltas = []
        while True:
            event = websocket.receive_json()
            if event["type"] == "response.output_audio.delta":
                deltas.append(event)
            if event["type"] == "response.done":
                break

    assert len(deltas) == 2
    assert base64.b64decode(deltas[0]["delta"]) == bytes(2400)


def test_live_tts_rejects_a_second_session(live_client):
    client, _fake_router = live_client

    with client.websocket_connect("/v1/audio/speech/stream") as first:
        assert first.receive_json()["type"] == "session.created"
        with client.websocket_connect("/v1/audio/speech/stream") as second:
            event = second.receive_json()
            assert event["type"] == "error"
            assert event["error"]["code"] == "session_busy"
            assert event["error"]["reason"] == "active"


def test_live_tts_get_explains_that_websocket_is_required(live_client):
    client, _fake_router = live_client

    response = client.get("/v1/audio/speech/stream")

    assert response.status_code == 426
    assert response.headers["upgrade"] == "websocket"
    assert response.json()["error"]["code"] == "websocket_upgrade_required"


def test_cancel_closes_generator_on_worker_and_next_synthesis_proceeds(monkeypatch):
    router = _BlockingRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    try:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            _configure(websocket)
            websocket.send_json({"type": "input_text.append", "text": "First sentence."})
            websocket.send_json({"type": "input_text.commit"})
            first_delta = None
            while first_delta is None:
                event = websocket.receive_json()
                if event["type"] == "response.output_audio.delta":
                    first_delta = event
            assert router.waiting.wait(timeout=1)

            websocket.send_json({"type": "response.cancel"})
            cancelled = None
            while cancelled is None:
                event = websocket.receive_json()
                if event["type"] == "response.cancelled":
                    cancelled = event
            assert cancelled["accepted_chars"] == 0
            assert cancelled["response_id"] == first_delta["response_id"]
            router.release.set()

            websocket.send_json({"type": "input_text.append", "text": "Second sentence."})
            websocket.send_json({"type": "input_text.commit"})
            completed = None
            accepted = None
            while completed is None:
                event = websocket.receive_json()
                if event["type"] == "input_text.accepted":
                    accepted = event
                if event["type"] == "input_text.done" and event["generation"] == 1:
                    completed = event
            assert accepted["accepted_chars"] == len("Second sentence.")
    finally:
        router.release.set()

    assert len(router.requests) == 2
    create_threads = [thread_id for operation, thread_id in router.calls if operation == "create"]
    close_threads = [thread_id for operation, thread_id in router.calls if operation == "close"]
    assert close_threads == create_threads


@pytest.mark.asyncio
async def test_failed_websocket_accept_does_not_consume_admission_slot():
    class FailingWebSocket:
        async def accept(self):
            raise RuntimeError("client left during handshake")

    settings = SimpleNamespace(tts_live_max_connections=1)
    with pytest.raises(RuntimeError, match="client left"):
        await live_tts_endpoint(
            FailingWebSocket(),
            tts_router=None,
            pronunciation_dict=None,
            settings=settings,
        )

    token, reason = _admission.acquire(1)
    assert token is not None
    assert reason is None
    _admission.release(token)


@pytest.mark.asyncio
async def test_receive_runtime_error_after_concurrent_disconnect_is_clean():
    class ConcurrentDisconnectWebSocket:
        def __init__(self) -> None:
            self.session: LiveTTSSession | None = None
            self.receive_calls = 0

        async def send_json(self, _event):
            return None

        async def receive(self):
            self.receive_calls += 1
            assert self.session is not None
            self.session._disconnected = True
            raise RuntimeError('WebSocket is not connected. Need to call "accept" first.')

    settings = SimpleNamespace(
        tts_model="kokoro",
        tts_voice="af_heart",
        tts_speed=1.0,
        tts_live_max_pending_segments=3,
        tts_live_max_segment_chars=200,
        tts_live_max_segment_words=20,
        tts_live_max_buffer_chars=4000,
        tts_live_max_unacked_seconds=10.0,
        tts_live_max_pause_s=60.0,
        tts_live_idle_timeout_s=60.0,
        tts_live_shutdown_timeout_s=0.1,
    )
    websocket = ConcurrentDisconnectWebSocket()
    session = LiveTTSSession(
        websocket,
        tts_router=_FakeRouter(),
        pronunciation_dict=_Pronunciation(),
        settings=settings,
    )
    websocket.session = session

    try:
        await session.run()
    finally:
        await session.shutdown()

    assert websocket.receive_calls == 1
    assert session._disconnected is True


@pytest.mark.asyncio
async def test_shutdown_has_a_bounded_wait_for_an_unresponsive_worker():
    settings = SimpleNamespace(
        tts_model="kokoro",
        tts_voice="af_heart",
        tts_speed=1.0,
        tts_live_max_pending_segments=3,
        tts_live_max_segment_chars=200,
        tts_live_max_segment_words=20,
        tts_live_shutdown_timeout_s=0.01,
    )
    session = LiveTTSSession(
        SimpleNamespace(),
        tts_router=_FakeRouter(),
        pronunciation_dict=_Pronunciation(),
        settings=settings,
    )
    session._synthesis_task = asyncio.create_task(asyncio.Event().wait())
    session._worker_future = asyncio.get_running_loop().create_future()

    worker_finished = await session.shutdown()

    assert worker_finished is False
    assert session._synthesis_task.cancelled()
    session._worker_future.cancel()


@pytest.mark.asyncio
async def test_low_speed_output_can_exceed_the_old_400_frame_guard():
    loop = asyncio.get_running_loop()
    output_queue = asyncio.Queue()
    router = _FakeRouter(samples=1200 * 500)

    await asyncio.to_thread(
        _synthesize_worker,
        loop=loop,
        output_queue=output_queue,
        cancel_event=threading.Event(),
        tts_router=router,
        text="A deliberately slow segment.",
        model="kokoro",
        voice="af_heart",
        speed=0.25,
        language=None,
        sample_rate=24000,
        frame_ms=50,
    )
    await asyncio.sleep(0)
    events = []
    while not output_queue.empty():
        events.append(output_queue.get_nowait())

    assert sum(event[0] == "audio" for event in events) == 500
    assert events[-1][0] == "done"
    assert not any(event[0] == "error" for event in events)


def test_backend_sample_rate_is_propagated_to_session_and_delta(live_client):
    client, router = live_client
    router.sample_rate = 22050

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket, sample_rate=22050)
        websocket.send_json({"type": "input_text.append", "text": "Rate check."})
        websocket.send_json({"type": "input_text.commit"})
        while True:
            event = websocket.receive_json()
            if event["type"] == "response.output_audio.delta":
                assert event["sample_rate"] == 22050
                break


def test_live_tts_example_imports_with_the_pinned_websockets_api():
    spec = importlib.util.spec_from_file_location("live_tts_client", "examples/live_tts_client.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class _ExampleClientSocket:
    """The server side of /v1/audio/speech/stream as examples/live_tts_client.py sees it.

    session.created reports the rate of the server's TTS_MODEL; session.updated and each delta
    report the native rate of the model the client chose, as LiveTTSSession does.
    """

    def __init__(self, *, created_rate: int, model_rate: int, samples: int) -> None:
        self.model_rate = model_rate
        self.samples = samples
        self.sent: list[dict] = []
        self.events: asyncio.Queue = asyncio.Queue()
        self.events.put_nowait({"type": "session.created", "session": self._session(created_rate)})

    @staticmethod
    def _session(sample_rate: int) -> dict:
        return {"audio": {"format": "pcm16", "sample_rate": sample_rate, "channels": 1}}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        if message["type"] == "session.update":
            self.events.put_nowait({"type": "session.updated", "session": self._session(self.model_rate)})
        elif message["type"] == "input_text.commit":
            asyncio.get_running_loop().call_soon(self._answer)

    def _answer(self) -> None:
        pcm = np.full(self.samples, 1000, dtype="<i2").tobytes()
        for event in (
            {"type": "response.created", "response": {"id": "resp_1", "text": "Rate check."}},
            {"type": "response.output_audio.delta", "response_id": "resp_1", "generation": 0, "sequence": 0,
             "sample_rate": self.model_rate, "delta": base64.b64encode(pcm).decode("ascii")},
            {"type": "response.done", "response": {"id": "resp_1", "status": "completed"}},
            {"type": "input_text.done"},
        ):
            self.events.put_nowait(event)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return json.dumps(await self.events.get())


@pytest.mark.parametrize(
    ("created_rate", "model", "model_rate"),
    [(24000, "piper/en_US-lessac-medium", 22050), (22050, "kokoro", 24000)],
)
async def test_live_tts_example_writes_the_wav_at_the_chosen_models_rate(
    monkeypatch, tmp_path, created_rate, model, model_rate
):
    """docs/API.md: deltas carry "the backend's actual sample_rate"; Live Reader does not resample."""
    import io
    import sys
    import wave

    spec = importlib.util.spec_from_file_location("live_tts_client", "examples/live_tts_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    socket = _ExampleClientSocket(created_rate=created_rate, model_rate=model_rate, samples=model_rate)
    monkeypatch.setattr(module, "connect", lambda *_args, **_kwargs: socket)
    monkeypatch.setattr(sys, "stdin", io.StringIO("Rate check.\n"))
    output = tmp_path / "live-reader.wav"
    args = SimpleNamespace(url="http://localhost:8100", api_key="", model=model, voice="af_heart", speed=1.0,
                           latency="natural", output=str(output), insecure=False)

    await asyncio.wait_for(module.run(args), timeout=10)

    assert [m["session"]["model"] for m in socket.sent if m["type"] == "session.update"] == [model]
    with wave.open(str(output), "rb") as wav:
        assert (wav.getnframes(), wav.getframerate()) == (model_rate, model_rate)  # one second of audio


def test_pause_prevents_a_new_segment_from_starting(live_client):
    client, router = live_client

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket)
        websocket.send_json({"type": "playback.pause"})
        assert websocket.receive_json()["type"] == "playback.paused"
        websocket.send_json({"type": "input_text.append", "text": "Wait for resume."})
        websocket.send_json({"type": "input_text.commit"})
        time.sleep(0.05)
        assert router.requests == []

        websocket.send_json({"type": "playback.resume"})
        while True:
            event = websocket.receive_json()
            if event["type"] == "input_text.done":
                break

    assert router.requests[0]["text"] == "Wait for resume."


@pytest.mark.asyncio
async def test_cancelled_generation_is_rechecked_after_flow_wait():
    sent_events = []

    async def send_json(event):
        sent_events.append(event)

    settings = SimpleNamespace(
        tts_model="kokoro",
        tts_voice="af_heart",
        tts_speed=1.0,
        tts_live_max_pending_segments=3,
        tts_live_max_segment_chars=200,
        tts_live_max_segment_words=20,
    )
    router = _FakeRouter()
    session = LiveTTSSession(
        SimpleNamespace(send_json=send_json),
        tts_router=router,
        pronunciation_dict=_Pronunciation(),
        settings=settings,
    )
    flow_waiting = asyncio.Event()
    release_flow = asyncio.Event()

    async def wait_for_flow():
        flow_waiting.set()
        await release_flow.wait()
        return True

    session._wait_for_flow = wait_for_flow
    synthesis_task = asyncio.create_task(session._synthesis_loop())
    session._segment_queue.put_nowait(
        (0, SimpleNamespace(text="Stale.", source_start=0, source_end=6))
    )
    await flow_waiting.wait()

    await session._handle_cancel({})
    release_flow.set()
    session._segment_queue.put_nowait(None)
    await synthesis_task

    assert router.requests == []
    assert not any(event["type"] == "response.created" for event in sent_events)


def test_backend_error_details_are_not_returned_to_client(monkeypatch):
    router = _ErrorRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket)
        websocket.send_json({"type": "input_text.append", "text": "Fail safely."})
        websocket.send_json({"type": "input_text.commit"})
        while True:
            event = websocket.receive_json()
            if event["type"] == "error":
                assert event["error"]["code"] == "tts_error"
                assert event["error"]["message"] == "Live TTS synthesis failed"
                assert "/models/" not in event["error"]["message"]
                break


JAPANESE_REFUSAL = (
    "chatterbox/regular does not speak Japanese, and this text has kana. Supported: en. "
    "For Japanese, use qwen3/0.6b-base."
)


class _RefusingRouter(_FakeRouter):
    """Raises on iteration, as ExternalTTSBackend does for a GPU worker's typed 400."""

    def synthesize(self, **kwargs):
        self.requests.append(kwargs)

        def refuse():
            from src.tts.external import ExternalProviderError

            raise ExternalProviderError(JAPANESE_REFUSAL, code="language_unsupported", status_code=400)
            yield

        return refuse()


def test_a_typed_provider_refusal_reaches_the_client_with_its_reason(request, monkeypatch):
    # An ExternalProviderError's message is the client-facing one /v1/audio/speech returns
    # verbatim (here a worker's Japanese refusal); the Live Reader used to replace it with "Live
    # TTS synthesis failed". Any other exception's text stays private (the test above).
    router = _RefusingRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)
    events: list[dict] = []

    with hard_watchdog(request, 30):
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            _configure(websocket)
            websocket.send_json({"type": "input_text.append", "text": "りんごを3個買いました。"})
            websocket.send_json({"type": "input_text.commit"})
            while not events or events[-1]["type"] != "response.done":
                events.append(websocket.receive_json())

    assert router.requests, "the segment never reached synthesis"
    errors = [event["error"] for event in events if event["type"] == "error"]
    assert [(error["code"], error["message"]) for error in errors] == [("tts_error", JAPANESE_REFUSAL)]
    assert events[-1]["response"]["status"] == "failed"


def test_unacknowledged_audio_closes_with_playback_stalled(monkeypatch):
    router = _FakeRouter(samples=2400)
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    monkeypatch.setattr(main_module.settings, "tts_live_max_unacked_seconds", 0.05)
    monkeypatch.setattr(main_module.settings, "tts_live_playback_stall_timeout_s", 0.05)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket)
        websocket.send_json({"type": "input_text.append", "text": "Do not acknowledge."})
        websocket.send_json({"type": "input_text.commit"})
        while True:
            event = websocket.receive_json()
            if event["type"] == "error":
                assert event["error"]["code"] == "playback_stalled"
                break


class _InputLimitRouter(_FakeRouter):
    """A provider model that advertises max_input_chars and refuses longer text, as Chatterbox does."""

    limit = 350
    backend = SimpleNamespace(name="chatterbox", capabilities={"live_reader": True, "speed_control": False})

    def max_input_chars_for(self, _model: str) -> int:
        return self.limit

    def get_backend(self, _model: str):
        return self.backend

    def synthesize(self, **kwargs):
        if len(kwargs["text"]) > self.limit:
            raise ValueError(f"Input too long for {kwargs['model']}. Max: {self.limit} characters")
        return super().synthesize(**kwargs)


@pytest.mark.parametrize(
    "more",
    ["", "Another short sentence follows here. " * 10],
    ids=["sentence-alone", "more-text-in-the-same-delta"],
)
def test_segments_respect_the_models_input_limit(monkeypatch, more):
    # 378 characters: inside the natural-mode caps (500 characters, 80 words) but over the model's 350.
    # The web UI sends up to 2,048 characters per append, so more text usually arrives with it.
    sentence = "Somewhere " * 37 + "tonight. " + more
    router = _InputLimitRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    monkeypatch.setattr(main_module.settings, "tts_live_sentence_max_chars", 500)
    monkeypatch.setattr(main_module.settings, "tts_live_sentence_max_words", 80)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json({"type": "session.update", "session": {"model": "chatterbox/turbo", "voice": "reference"}})
        assert websocket.receive_json()["type"] == "session.updated"
        websocket.send_json({"type": "input_text.append", "text": sentence})
        websocket.send_json({"type": "input_text.commit"})
        events = [websocket.receive_json()]
        while events[-1]["type"] != "input_text.done":
            events.append(websocket.receive_json())

    assert [event for event in events if event["type"] == "error"] == []
    assert len(router.requests) > 1
    assert max(len(request["text"]) for request in router.requests) <= router.limit
    assert " ".join(request["text"] for request in router.requests).split() == sentence.split()


def test_pronunciation_expansion_is_split_to_the_models_input_limit(monkeypatch):
    # The dictionary runs after the segment cap, so an expansion can pass the model's limit;
    # the pieces are spoken in order as one response.
    class ExpandingPronunciation:
        @staticmethod
        def apply(text: str) -> str:
            return text.replace("NASA", "the National Aeronautics and Space Administration")

    router = _InputLimitRouter()
    router.limit = 60
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", ExpandingPronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    client = TestClient(app)

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json({"type": "session.update", "session": {"model": "chatterbox/turbo", "voice": "reference"}})
        assert websocket.receive_json()["type"] == "session.updated"
        websocket.send_json({"type": "input_text.append", "text": "NASA said the launch went well today. "})
        websocket.send_json({"type": "input_text.commit"})
        events = [websocket.receive_json()]
        while events[-1]["type"] != "input_text.done":
            events.append(websocket.receive_json())

    assert [event for event in events if event["type"] == "error"] == []
    assert [len(request["text"]) <= 60 for request in router.requests] == [True, True]
    assert " ".join(request["text"] for request in router.requests) == (
        "the National Aeronautics and Space Administration said the launch went well today."
    )
    created = [event["response"]["text"] for event in events if event["type"] == "response.created"]
    assert created == ["NASA said the launch went well today."]
    sequences = [event["sequence"] for event in events if event["type"] == "response.output_audio.delta"]
    assert sequences == list(range(len(sequences))) and len(sequences) == 2


def test_a_segment_the_pronunciation_dictionary_removes_is_skipped_not_sent(monkeypatch, request):
    # The segment was checked for emptiness before the dictionary, so a
    # segment it removes ("an empty value removes the word", README TTS_PRONUNCIATION_DICT) went to the model as
    # "", which every GPU worker and Pocket-TTS refuse: an error event and a failed response for that sentence.
    class RemovingPronunciation:
        @staticmethod
        def apply(text: str) -> str:
            return text.replace("[Illustration]", "")

    class RouterThatRefusesEmptyText(_FakeRouter):
        def synthesize(self, **kwargs):
            if not kwargs["text"].strip():
                self.requests.append(kwargs)
                raise ValueError("Text must not be empty for Pocket TTS synthesis.")
            return super().synthesize(**kwargs)

    router = RouterThatRefusesEmptyText()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", RemovingPronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    text = "It was a fine morning.\n\n[Illustration]\n\nShe walked to the river and sat down.\n\n"

    with hard_watchdog(request, 60):
        with TestClient(app).websocket_connect("/v1/audio/speech/stream") as websocket:
            _configure(websocket)
            websocket.send_json({"type": "input_text.append", "text": text})
            websocket.send_json({"type": "input_text.commit"})
            events = [websocket.receive_json()]
            while events[-1]["type"] != "input_text.done":
                events.append(websocket.receive_json())

    assert [request["text"] for request in router.requests] == [
        "It was a fine morning.",
        "She walked to the river and sat down.",
    ]
    assert [event for event in events if event["type"] == "error"] == []
    assert [event["response"]["status"] for event in events if event["type"] == "response.done"] == [
        "completed",
        "completed",
    ]
    created = [event["response"]["text"] for event in events if event["type"] == "response.created"]
    assert created == ["It was a fine morning.", "She walked to the river and sat down."]
    skipped = [event for event in events if event["type"] == "input_text.skipped"]
    assert len(skipped) == 1
    assert text[skipped[0]["source_start"] : skipped[0]["source_end"]].strip() == "[Illustration]"


@pytest.mark.parametrize(
    "text",
    ["First sentence.", "".join(f"Sentence number {n} is here. " for n in range(1, 9))],
    ids=["one-sentence", "more-sentences-than-the-segment-queue-holds"],
)
def test_session_update_is_accepted_while_a_cancelled_response_drains(monkeypatch, text):
    # No event marks the end of a cancelled response (its response.done is suppressed),
    # so settings must be changeable as soon as response.cancelled arrives, even while
    # the segmenter task still holds the rest of the cancelled text.
    router = _BlockingRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    monkeypatch.setattr(main_module.settings, "tts_live_max_pending_segments", 3)
    client = TestClient(app)

    try:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            _configure(websocket)
            websocket.send_json({"type": "input_text.append", "text": text})
            websocket.send_json({"type": "input_text.commit"})
            while websocket.receive_json()["type"] != "response.output_audio.delta":
                pass
            assert router.waiting.wait(timeout=1)
            websocket.send_json({"type": "response.cancel"})
            while websocket.receive_json()["type"] != "response.cancelled":
                pass

            websocket.send_json({"type": "session.update", "session": {"model": "kokoro", "voice": "af_bella"}})
            reply = websocket.receive_json()
            while reply["type"] not in {"session.updated", "error"}:
                reply = websocket.receive_json()
            assert reply["type"] == "session.updated", reply

            router.release.set()
            websocket.send_json({"type": "input_text.append", "text": "Second sentence."})
            websocket.send_json({"type": "input_text.commit"})
            while websocket.receive_json()["type"] != "input_text.done":
                pass
    finally:
        router.release.set()

    # The rest of the cancelled text is never synthesized.
    assert [request["voice"] for request in router.requests] == ["af_heart", "af_bella"]


async def _serve_asgi(messages: list[dict], router=None) -> list[dict]:
    """Run one Live Reader connection over raw ASGI messages and return what the server sent."""
    incoming: asyncio.Queue = asyncio.Queue()
    for message in [{"type": "websocket.connect"}, *messages, {"type": "websocket.disconnect", "code": 1000}]:
        incoming.put_nowait(message)
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    websocket = WebSocket(
        {"type": "websocket", "path": "/v1/audio/speech/stream", "headers": [], "query_string": b""},
        incoming.get,
        send,
    )
    await asyncio.wait_for(
        live_tts_endpoint(
            websocket,
            tts_router=router or _FakeRouter(),
            pronunciation_dict=_Pronunciation(),
            settings=main_module.settings,
        ),
        timeout=5,
    )
    return sent


def _events(sent: list[dict]) -> list[dict]:
    return [json.loads(message["text"]) for message in sent if message["type"] == "websocket.send"]


@pytest.mark.asyncio
async def test_binary_frame_gets_an_error_and_the_session_continues():
    sent = await _serve_asgi([
        {"type": "websocket.receive", "bytes": b"\x00\x01"},
        {"type": "websocket.receive", "text": json.dumps({"type": "session.keepalive"})},
    ])

    events = _events(sent)
    assert [event["type"] for event in events] == ["session.created", "error", "session.keepalive.ack"]
    assert events[1]["error"]["code"] == "invalid_event"


@pytest.mark.asyncio
async def test_unexpected_session_failure_sends_a_close_frame(monkeypatch):
    # Without a close frame the client only sees the TCP connection drop (1006).
    async def explode(self, _data):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(LiveTTSSession, "_handle_keepalive", explode)

    sent = await _serve_asgi([{"type": "websocket.receive", "text": json.dumps({"type": "session.keepalive"})}])

    assert {"type": "websocket.close", "code": 1011, "reason": "Live TTS session failed"} in sent


# Ported from the old web UI's tests: the server rules Speak's Live mode
# (src/static/js/views/speak/live-session.js) is built on.


class _VoiceCheckingRouter(_FakeRouter):
    """The live server's router: a voice deleted since it was picked is refused, as validate_voice does."""

    backend = SimpleNamespace(name="kokoro", capabilities={"speed_control": True})

    def get_backend(self, _model):
        return self.backend

    def get_capabilities(self, _model):
        return dict(self.backend.capabilities)

    def validate_voice(self, _model, voice):
        if voice == "af_gone":
            raise ValueError("Voice 'af_gone' is not available for kokoro")


def _next_reply(websocket, types: set[str]) -> dict:
    """The next event of one of these types, skipping the audio and progress events of a running read."""
    event = websocket.receive_json()
    while event["type"] not in types:
        event = websocket.receive_json()
    return event


@pytest.mark.parametrize("voice", ["af_gone", ""], ids=["deleted-voice", "no-voice-yet"])
def test_a_refused_session_update_leaves_the_session_open_and_unconfigured(monkeypatch, voice):
    # A voice deleted since it was picked, or a clone model's empty voice before a reference is chosen, is
    # refused with invalid_event. The socket stays open but can never read, so the client must close it
    # and report the reason; only a later, valid update configures it.
    router = _VoiceCheckingRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    session = {"model": "kokoro", "voice": voice, "speed": 1.0, "latency_mode": "natural"}

    with TestClient(app).websocket_connect("/v1/audio/speech/stream") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json({"type": "session.update", "session": session})
        refused = websocket.receive_json()
        websocket.send_json({"type": "session.keepalive"})
        still_open = websocket.receive_json()
        websocket.send_json({"type": "session.update", "session": {**session, "voice": "af_heart"}})
        configured = websocket.receive_json()

    assert refused["type"] == "error" and refused["error"]["code"] == "invalid_event", refused
    assert refused["error"]["message"]
    assert still_open["type"] == "session.keepalive.ack"
    assert configured["type"] == "session.updated" and configured["session"]["voice"] == "af_heart"
    assert router.requests == []


def test_an_invalid_event_in_a_configured_session_fails_only_that_event(monkeypatch):
    # The same invalid_event code from a configured session is not fatal: the session reads on.
    router = _VoiceCheckingRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)

    with TestClient(app).websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure(websocket)
        websocket.send_json({"type": "input_text.append", "text": 42})
        rejected = websocket.receive_json()
        websocket.send_json({"type": "input_text.append", "text": "Still reading."})
        websocket.send_json({"type": "input_text.commit"})
        done = _next_reply(websocket, {"input_text.done", "error"})

    assert rejected["type"] == "error" and rejected["error"]["code"] == "invalid_event"
    assert done["type"] == "input_text.done", done
    assert [request["text"] for request in router.requests] == ["Still reading."]


def test_session_update_is_refused_while_text_is_read_and_accepted_right_after_cancel(monkeypatch):
    """[LR-23] session.update is refused with session_busy once text is accepted and read, and accepted right
    after response.cancelled, without waiting for the cancelled response to drain."""
    router = _BlockingRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)

    try:
        with TestClient(app).websocket_connect("/v1/audio/speech/stream") as websocket:
            _configure(websocket)
            websocket.send_json({"type": "input_text.append", "text": "First sentence. Second sentence."})
            websocket.send_json({"type": "input_text.commit"})
            _next_reply(websocket, {"response.output_audio.delta"})
            assert router.waiting.wait(timeout=1)

            websocket.send_json({"type": "session.update", "session": {"model": "kokoro", "voice": "af_bella"}})
            busy = _next_reply(websocket, {"session.updated", "error"})

            websocket.send_json({"type": "response.cancel"})
            _next_reply(websocket, {"response.cancelled"})
            websocket.send_json({"type": "session.update", "session": {"model": "kokoro", "voice": "af_bella"}})
            updated = _next_reply(websocket, {"session.updated", "error"})

            router.release.set()
            websocket.send_json({"type": "input_text.append", "text": "Third sentence."})
            websocket.send_json({"type": "input_text.commit"})
            _next_reply(websocket, {"input_text.done"})
    finally:
        router.release.set()

    assert busy["type"] == "error" and busy["error"]["code"] == "session_busy", busy
    assert updated["type"] == "session.updated" and updated["session"]["voice"] == "af_bella", updated
    assert [request["voice"] for request in router.requests] == ["af_heart", "af_bella"]


def test_a_one_time_ticket_opens_the_live_stream_once(monkeypatch):
    # A browser cannot set headers on a WebSocket: with OS_API_KEY set, Live asks POST /api/auth/ws-ticket
    # for a fresh ticket for every connection and passes it as ?ticket=, never the key.
    from starlette.websockets import WebSocketDisconnect

    import src.middleware as mw

    monkeypatch.setattr(main_module.settings, "os_api_key", "s3cret")
    monkeypatch.setattr(main_module, "tts_router", _VoiceCheckingRouter())
    monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
    monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
    mw.reset_security_state_for_tests()
    try:
        client = TestClient(app)
        issued = client.post("/api/auth/ws-ticket", headers={"Authorization": "Bearer s3cret"})
        assert issued.status_code == 200, issued.text
        ticket = issued.json()["ticket"]
        with client.websocket_connect(f"/v1/audio/speech/stream?ticket={ticket}") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/v1/audio/speech/stream?ticket={ticket}") as websocket:
                websocket.receive_json()
    finally:
        mw.reset_security_state_for_tests()


@pytest.mark.asyncio
async def test_connect_reads_an_isolated_workers_manifest_off_the_event_loop(monkeypatch):
    # For a TTS_MODEL an isolated worker serves, the rate and input limit are a manifest fetch
    # (3 s timeout each); on the event loop a hung worker froze every client for each connect.
    import io

    from src.tts.external import ExternalTTSBackend
    from src.tts.router import TTSRouter

    model = "qwen3/0.6b-custom-voice"
    manifest = {
        "schema_version": 1,
        "provider": "qwen3",
        "models": [{
            "id": model, "provider": "qwen3", "sample_rate": 48000, "revision": "test-1",
            "max_input_chars": 300,
            "capabilities": {"voice_clone": False, "instructions": False, "speed_control": False},
            "voices": [{"id": "Ryan", "name": "Ryan", "language": "en", "gender": "male"}],
        }],
    }

    class Response(io.BytesIO):
        headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    fetch_threads: list[int] = []

    def open_fn(request, **_kwargs):
        if request.full_url.endswith("/v1/manifest"):
            fetch_threads.append(threading.get_ident())
            return Response(json.dumps(manifest).encode())
        return Response(b"{}")

    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_a, **_k: {})
    router = TTSRouter(device="cpu", external_providers="")
    backend = ExternalTTSBackend(provider="qwen3", base_url="http://qwen3:8200")
    monkeypatch.setattr(backend._opener, "open", open_fn)
    router.register_backend("qwen3", backend)
    monkeypatch.setattr(main_module.settings, "tts_model", model)
    monkeypatch.setattr(main_module.settings, "tts_voice", "Ryan")
    loop_thread = threading.get_ident()

    events = _events(await _serve_asgi([], router=router))

    created = next(event for event in events if event["type"] == "session.created")
    assert created["session"]["audio"]["sample_rate"] == 48000  # the worker's rate, as before
    assert fetch_threads and loop_thread not in fetch_threads


@pytest.mark.asyncio
async def test_input_text_commits_are_bounded_while_flow_control_holds_synthesis(monkeypatch):
    # TTS_LIVE_MAX_BUFFER_CHARS bounds what waits for the segmenter; each queued commit is a unit
    # of it, so a paused client that keeps committing gets buffer_overflow, not unbounded memory.
    settings = main_module.settings
    monkeypatch.setattr(settings, "tts_live_max_buffer_chars", 64)
    commits = 500
    sent: list[dict] = []
    measured: dict[str, int] = {}
    done = asyncio.Event()

    class ScriptedSocket:
        def __init__(self):
            self.script = [{"type": "playback.pause"}] + [{"type": "input_text.commit"}] * commits
            self.session = None

        async def receive(self):
            if self.script:
                return {"type": "websocket.receive", "text": json.dumps(self.script.pop(0))}
            if not measured:
                await asyncio.sleep(0.05)  # let the segmenter take what it can
                measured["input_queue"] = self.session._input_queue.qsize()
                done.set()
            return {"type": "websocket.disconnect", "code": 1000}

        async def send_json(self, event):
            sent.append(event)

        async def close(self, code=1000, reason=""):
            pass

    websocket = ScriptedSocket()
    session = LiveTTSSession(
        websocket, tts_router=_FakeRouter(), pronunciation_dict=_Pronunciation(), settings=settings,
    )
    websocket.session = session
    try:
        await asyncio.wait_for(session.run(), timeout=30)
    finally:
        await asyncio.wait_for(session.shutdown(), timeout=30)

    assert done.is_set()
    assert measured["input_queue"] <= settings.tts_live_max_buffer_chars
    overflows = [e for e in sent if e.get("type") == "error" and e["error"]["code"] == "buffer_overflow"]
    committed = [e for e in sent if e.get("type") == "input_text.committed"]
    assert overflows and len(committed) + len(overflows) == commits


# Speak > Live sends the text from the caret with a commit (LR-08), then what the
# listener types on, with no commit (LR-10). A commit doesn't end the line, so the typed text is the same line.
# A client that resends from mid-line after response.cancel says where with input_text.context.


@pytest.fixture
def typing_client(monkeypatch, request):
    """A Live Reader that releases only at boundaries: no idle release depends on how fast the test runs."""
    # A receive that hangs ends the run, naming the test (tests/_watchdog.py)
    with hard_watchdog(request, 60):
        router = _FakeRouter(samples=240)
        monkeypatch.setattr(main_module, "tts_router", router)
        monkeypatch.setattr(main_module, "pronunciation_dict", _Pronunciation())
        monkeypatch.setattr(main_module.settings, "tts_live_enabled", True)
        monkeypatch.setattr(main_module.settings, "tts_live_segment_idle_ms", 60_000)
        monkeypatch.setattr(main_module.settings, "tts_live_sentence_idle_ms", 60_000)
        yield TestClient(app), router


def _read_until(websocket, done, events: list[dict]) -> dict:
    """Events until done(event), acking each audio chunk as a player does (LR-11)."""
    for _ in range(5000):
        event = websocket.receive_json()
        events.append(event)
        if event["type"] == "response.output_audio.delta":
            websocket.send_json({"type": "playback.ack", "response_id": event["response_id"], "sequence": event["sequence"]})
        if done(event):
            return event
    raise AssertionError(f"no end event in 5000 events: {[e['type'] for e in events[-20:]]}")


def _reaches(end: int):
    return lambda event: (
        event["type"] in ("response.done", "input_text.skipped")
        and event.get("response", event).get("source_end", 0) >= end
    )


def _configure_mode(websocket, mode: str) -> None:
    assert websocket.receive_json()["type"] == "session.created"
    websocket.send_json({"type": "session.update", "session": {"model": "kokoro", "voice": "af_heart", "latency_mode": mode}})
    assert websocket.receive_json()["type"] == "session.updated"


@pytest.mark.parametrize("mode", ["natural", "responsive", "instant_word"])
def test_text_typed_on_after_the_start_commit_is_spoken_in_full(typing_client, mode):
    # "12." typed on after Start's commit was dropped as a list marker (and, in
    # instant_word mode, answered with input_text.skipped).
    client, router = typing_client
    before, typed = "Sales rose 5% to", " 12. Next quarter is better. "
    events: list[dict] = []

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure_mode(websocket, mode)
        websocket.send_json({"type": "input_text.append", "text": before})
        _read_until(websocket, lambda event: event["type"] == "input_text.accepted", events)
        websocket.send_json({"type": "input_text.commit"})
        _read_until(websocket, lambda event: event["type"] == "input_text.done", events)
        websocket.send_json({"type": "input_text.append", "text": typed})
        _read_until(websocket, _reaches(len((before + typed).rstrip())), events)

    assert " ".join(request["text"] for request in router.requests).split() == (before + typed).split()
    assert [event for event in events if event["type"] in ("input_text.skipped", "error")] == []


@pytest.mark.parametrize("mode", ["natural", "responsive", "instant_word"])
def test_input_text_context_after_a_cancel_resumes_mid_line(typing_client, mode):
    # After response.cancel the server starts a fresh segmenter, at a line start.
    # A client resending from mid-line sends the text before the resume point first; none of it is spoken or
    # counted, and the resent text is cleaned as the continuation it is.
    client, router = typing_client
    context, typed = "Sales rose 5% to", " 12. Next quarter is better. "
    events: list[dict] = []

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure_mode(websocket, mode)
        # A draft that no mode releases (one unfinished word), then the edit's cancel.
        websocket.send_json({"type": "input_text.append", "text": "Sales"})
        _read_until(websocket, lambda event: event["type"] == "input_text.accepted", events)
        websocket.send_json({"type": "response.cancel"})
        cancelled = _read_until(websocket, lambda event: event["type"] == "response.cancelled", events)
        websocket.send_json({"type": "input_text.context", "text": context})
        websocket.send_json({"type": "input_text.append", "text": typed})
        accepted = _read_until(websocket, lambda event: event["type"] in ("input_text.accepted", "error"), events)
        _read_until(websocket, _reaches(len(typed.rstrip())), events)

    assert accepted["type"] == "input_text.accepted" and accepted["accepted_chars"] == len(typed), accepted
    assert " ".join(request["text"] for request in router.requests).split() == typed.split()
    done = [event["response"] for event in events if event["type"] == "response.done"]
    assert done and all(response["generation"] == cancelled["generation"] + 1 for response in done)
    assert all(0 <= response["source_start"] <= response["source_end"] <= len(typed) for response in done)
    assert [event for event in events if event["type"] in ("input_text.skipped", "error")] == []


def test_input_text_context_is_refused_once_text_is_accepted_and_taken_again_after_cancel(typing_client):
    # The context describes the text before a generation's first append.
    client, _router = typing_client
    events: list[dict] = []

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        _configure_mode(websocket, "natural")
        websocket.send_json({"type": "input_text.context", "text": 5})
        not_text = _read_until(websocket, lambda event: event["type"] == "error", events)
        websocket.send_json({"type": "input_text.append", "text": "Hello"})
        _read_until(websocket, lambda event: event["type"] == "input_text.accepted", events)
        websocket.send_json({"type": "input_text.context", "text": "Earlier text"})
        too_late = _read_until(websocket, lambda event: event["type"] == "error", events)
        websocket.send_json({"type": "response.cancel"})
        _read_until(websocket, lambda event: event["type"] == "response.cancelled", events)
        websocket.send_json({"type": "input_text.context", "text": "Earlier text"})
        websocket.send_json({"type": "session.keepalive"})
        next_reply = _read_until(websocket, lambda event: event["type"] in ("session.keepalive.ack", "error"), events)

    assert not_text["error"]["code"] == "invalid_event"
    assert too_late["error"]["code"] == "invalid_event" and "input_text.context" in too_late["error"]["message"]
    assert next_reply["type"] == "session.keepalive.ack", next_reply
