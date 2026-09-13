"""Model-free integration tests for the Live Reader WebSocket."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.live_tts.server import (
    LiveTTSSession,
    _admission,
    _synthesize_worker,
    live_tts_endpoint,
    reset_live_tts_state_for_tests,
)
from src.main import app


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
