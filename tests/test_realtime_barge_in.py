"""Talking over a realtime reply: streaming audio, barge-in and backchannels."""

from __future__ import annotations

import asyncio
import base64
import threading
from unittest.mock import patch

import numpy as np
import pytest

from src.realtime.audio_buffer import InputAudioBuffer
from src.realtime.backchannel import is_backchannel
from src.realtime.session import SessionConfig


class DummyWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, event):
        self.sent.append(event)

    def types(self):
        return [event["type"] for event in self.sent]

    def deltas(self):
        return [event for event in self.sent if event["type"] == "response.audio.delta"]

    def done(self):
        return [event for event in self.sent if event["type"] == "response.done"]


class TwoChunkRouter:
    """Yields one chunk, then waits for `release` before yielding the second."""

    def __init__(self):
        self.first_sent = threading.Event()
        self.release = threading.Event()

    def sample_rate_for(self, _model):
        return 24000

    def synthesize(self, **_kwargs):
        yield np.zeros(320, dtype=np.float32)
        self.first_sent.set()
        self.release.wait(timeout=2)
        yield np.zeros(320, dtype=np.float32)


async def _wait_for(predicate, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def _start_reply(session, websocket):
    await session.handle_event({
        "type": "response.create",
        "response": {"instructions": "A reply that takes a while."},
    })
    await _wait_for(lambda: len(websocket.deltas()) == 1)


# ---------------------------------------------------------------------------
# Backchannel detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["mm-hmm", "Uh-huh.", "yeah okay", "Oh, I see!", "right right", "Got it."])
def test_backchannels_are_recognised(text):
    assert is_backchannel(text)


@pytest.mark.parametrize("text", [
    "", "wait", "okay so what about Tuesday",
    "yeah but stop", "yeah yeah yeah yeah yeah yeah",
])
def test_real_speech_is_not_a_backchannel(text):
    assert not is_backchannel(text)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_audio_is_sent_before_synthesis_finishes():
    from src.realtime.server import RealtimeSession

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, router)
    try:
        await _start_reply(session, websocket)
        assert "response.done" not in websocket.types()
    finally:
        router.release.set()
    await session._response_task

    assert len(websocket.deltas()) == 2
    assert websocket.done()[-1]["response"]["status"] == "completed"


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backchannel_over_reply_holds_audio_then_resumes():
    from src.realtime.server import RealtimeSession

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, router)
    try:
        await _start_reply(session, websocket)
        await session._on_user_speech_started()
        router.release.set()
        await asyncio.sleep(0.1)
        assert len(websocket.deltas()) == 1  # held while we listen

        await session._resolve_barge_in("mm-hmm")
    finally:
        router.release.set()
    await session._response_task

    assert len(websocket.deltas()) == 2
    assert websocket.done()[-1]["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_real_words_over_reply_cancel_it_as_turn_detected():
    from src.realtime.server import RealtimeSession

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, router)
    try:
        await _start_reply(session, websocket)
        await session._on_user_speech_started()
        await session._resolve_barge_in("wait, what about Tuesday?")
    finally:
        router.release.set()

    assert session._response_task is None
    assert len(websocket.deltas()) == 1
    response = websocket.done()[-1]["response"]
    assert response["status"] == "cancelled"
    assert response["status_details"] == {"type": "cancelled", "reason": "turn_detected"}
    assert session._playback_gate.is_set()


@pytest.mark.asyncio
async def test_long_speech_over_reply_cancels_without_a_transcript():
    from src.realtime import server

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = server.RealtimeSession(websocket, router)
    try:
        with patch.object(server.settings, "os_realtime_barge_in_max_ms", 50):
            await _start_reply(session, websocket)
            await session._on_user_speech_started()
            await _wait_for(lambda: session._response_task is None)
    finally:
        router.release.set()

    response = websocket.done()[-1]["response"]
    assert response["status"] == "cancelled"
    assert response["status_details"]["reason"] == "turn_detected"


@pytest.mark.asyncio
async def test_interrupt_response_false_keeps_talking():
    from src.realtime.server import RealtimeSession

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, router)
    session.config.update_from({"session": {"turn_detection": {"interrupt_response": False}}})
    try:
        await _start_reply(session, websocket)
        await session._on_user_speech_started()
        assert session._playback_gate.is_set()
    finally:
        router.release.set()
    await session._response_task

    assert websocket.done()[-1]["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_client_cancel_reports_client_cancelled():
    from src.realtime.server import RealtimeSession

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, router)
    try:
        await _start_reply(session, websocket)
        await session.handle_event({"type": "response.cancel"})
    finally:
        router.release.set()

    assert websocket.done()[-1]["response"]["status_details"]["reason"] == "client_cancelled"


@pytest.mark.asyncio
async def test_microphone_speech_during_reply_interrupts_through_vad_and_stt():
    """End to end through the audio path: VAD hears speech, STT says it's real."""
    from src.realtime.server import RealtimeSession

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, router)
    probabilities = iter([0.9, 0.9, 0.9, 0.1, 0.1])
    session.audio_buffer = InputAudioBuffer(
        vad=lambda _audio: next(probabilities),
        threshold=0.5,
        silence_duration_ms=64,
    )
    pcm = b"\x01\x00" * (5 * 512)
    try:
        await _start_reply(session, websocket)
        with (
            patch(
                "src.realtime.server.decode_audio_to_pcm16",
                side_effect=lambda data, _fmt, target_rate: data,
            ),
            patch(
                "src.realtime.server.stt_router.transcribe",
                return_value={"text": "hang on, stop there"},
            ),
        ):
            await session._handle_input_audio_buffer_append({
                "audio": base64.b64encode(pcm).decode("ascii"),
            })
    finally:
        router.release.set()

    types = websocket.types()
    assert types.index("input_audio_buffer.speech_started") < types.index("response.done")
    assert websocket.done()[-1]["response"]["status_details"]["reason"] == "turn_detected"


# ---------------------------------------------------------------------------
# Protocol bits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_item_truncate_is_acknowledged():
    from src.realtime.server import RealtimeSession

    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, TwoChunkRouter())
    await session.handle_event({
        "type": "conversation.item.truncate",
        "item_id": "item_abc",
        "content_index": 0,
        "audio_end_ms": 1500,
    })

    assert websocket.sent[-1]["type"] == "conversation.item.truncated"
    assert websocket.sent[-1]["audio_end_ms"] == 1500


@pytest.mark.asyncio
async def test_conversation_item_truncate_rejects_bad_input():
    from src.realtime.server import RealtimeSession

    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, TwoChunkRouter())
    await session.handle_event({"type": "conversation.item.truncate", "item_id": "x", "audio_end_ms": -1})

    assert websocket.sent[-1]["error"]["code"] == "invalid_request"


def test_interrupt_response_round_trips_in_session_config():
    config = SessionConfig()
    assert config.to_dict()["turn_detection"]["interrupt_response"] is True
    config.update_from({"session": {"turn_detection": {"interrupt_response": False}}})
    assert config.to_dict()["turn_detection"]["interrupt_response"] is False
