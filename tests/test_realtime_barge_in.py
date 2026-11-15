"""Talking over a realtime reply: streaming audio, barge-in and backchannels."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import itertools
import selectors
import threading
import time
import zlib
from unittest.mock import AsyncMock, MagicMock, patch

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


class InstantRouter:
    """Fast TTS: the whole reply arrives as one chunk straight away."""

    def __init__(self, seconds):
        self.samples = int(24000 * seconds)

    def sample_rate_for(self, _model):
        return 24000

    def synthesize(self, **_kwargs):
        yield np.zeros(self.samples, dtype=np.float32)


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
    await _wait_for(lambda: len(websocket.deltas()) >= 1)


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


@pytest.mark.asyncio
async def test_reply_audio_is_paced_to_playback_when_barge_in_is_on():
    from src.realtime import server

    websocket = DummyWebSocket()
    session = server.RealtimeSession(websocket, InstantRouter(seconds=0.5))
    loop = asyncio.get_running_loop()
    with patch.object(server.settings, "os_realtime_audio_lead_ms", 100):
        started = loop.time()
        await _start_reply(session, websocket)
        await session._response_task
    assert loop.time() - started >= 0.3  # 0.5 s of audio, 0.1 s lead, last piece 0.06 s
    assert websocket.done()[-1]["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_reply_audio_is_not_paced_when_lead_is_zero():
    from src.realtime import server

    websocket = DummyWebSocket()
    session = server.RealtimeSession(websocket, InstantRouter(seconds=2))
    loop = asyncio.get_running_loop()
    with patch.object(server.settings, "os_realtime_audio_lead_ms", 0):
        started = loop.time()
        await _start_reply(session, websocket)
        await _wait_for(lambda: websocket.done())
    assert loop.time() - started < 0.5


@pytest.mark.asyncio
async def test_idle_timeout_waits_for_a_paced_reply_then_a_full_idle_window():
    """A listen-only client is not idle while its reply is still playing."""
    from starlette.websockets import WebSocketState

    from src.realtime import server

    class ListeningClient(DummyWebSocket):
        scope = {"subprotocols": ["realtime"]}
        headers = {}
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            super().__init__()
            self.closed = None
            self.requested = False

        async def accept(self, subprotocol=None):
            return None

        async def receive(self):
            if not self.requested:
                self.requested = True
                return {
                    "type": "websocket.receive",
                    "text": '{"type": "response.create", "response": {"instructions": "Read this."}}',
                }
            await asyncio.Event().wait()  # the user just listens

        async def close(self, code=1000, reason=""):
            self.closed = (code, reason)
            self.application_state = WebSocketState.DISCONNECTED

    websocket = ListeningClient()
    loop = asyncio.get_running_loop()
    with (
        patch.object(server.settings, "os_realtime_idle_timeout_s", 0.3),
        patch.object(server.settings, "os_realtime_audio_lead_ms", 100),
        patch("src.realtime.server.get_vad_model", new=AsyncMock(return_value=MagicMock())),
        patch("src.realtime.server.SileroVAD", return_value=lambda _audio: 0.0),
    ):
        endpoint = asyncio.create_task(server.realtime_endpoint(websocket, tts_router=InstantRouter(seconds=1.2)))
        try:
            await _wait_for(lambda: websocket.done(), timeout=5)
            reply_done = loop.time()
            await _wait_for(lambda: websocket.closed is not None, timeout=5)
            closed = loop.time()
        finally:
            endpoint.cancel()
            await asyncio.gather(endpoint, return_exceptions=True)

    assert websocket.done()[0]["response"]["status"] == "completed"
    assert websocket.closed == (4008, "Session idle timeout")
    assert closed - reply_done >= 0.25  # the idle window restarted when the reply ended


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_tts_reply_can_still_be_interrupted():
    """Seen in real use: the whole reply was synthesized and sent before the user could speak."""
    from src.realtime import server

    websocket = DummyWebSocket()
    session = server.RealtimeSession(websocket, InstantRouter(seconds=2))
    with patch.object(server.settings, "os_realtime_audio_lead_ms", 100):
        await _start_reply(session, websocket)
        await asyncio.sleep(0.2)
        await session._on_user_speech_started()
        await session._resolve_barge_in("wait, stop")

    response = websocket.done()[-1]["response"]
    assert response["status_details"]["reason"] == "turn_detected"
    sent = sum(len(base64.b64decode(d["delta"])) for d in websocket.deltas())
    assert sent < 24000 * 2 * 2 / 2  # well under the 2 s of audio was sent


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
async def test_speech_over_reply_with_transcription_off_cancels_at_once():
    """Without a transcript "mm-hmm" can't be told from an interruption, so the reply
    stops straight away. Only GA sessions honoured transcription null."""
    from src.realtime.server import RealtimeSession

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, router)  # a legacy session
    session.config.update_from({"session": {"input_audio_transcription": None}})
    try:
        await _start_reply(session, websocket)
        await session._on_user_speech_started()
    finally:
        router.release.set()

    assert session._response_task is None
    response = websocket.done()[-1]["response"]
    assert response["status_details"] == {"type": "cancelled", "reason": "turn_detected"}


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


@pytest.mark.asyncio
async def test_short_acknowledgement_waits_for_a_slow_transcript():
    """Seen in real use: the deadline cancelled "Yeah, okay." while it was still being transcribed."""
    import time as _time

    from src.realtime import server

    router = TwoChunkRouter()
    websocket = DummyWebSocket()
    session = server.RealtimeSession(websocket, router)
    probabilities = iter([0.9, 0.9, 0.9, 0.1, 0.1])
    session.audio_buffer = InputAudioBuffer(
        vad=lambda _audio: next(probabilities),
        threshold=0.5,
        silence_duration_ms=64,
    )
    session.config.turn_detection.silence_duration_ms = 64
    pcm = b"\x01\x00" * (5 * 512)

    def slow_transcribe(**_kwargs):
        _time.sleep(0.3)  # longer than the 50 ms barge-in limit below
        return {"text": "Yeah, okay."}

    try:
        await _start_reply(session, websocket)
        with (
            patch.object(server.settings, "os_realtime_barge_in_max_ms", 50),
            patch(
                "src.realtime.server.decode_audio_to_pcm16",
                side_effect=lambda data, _fmt, target_rate: data,
            ),
            patch("src.realtime.server.stt_router.transcribe", side_effect=slow_transcribe),
        ):
            await session._handle_input_audio_buffer_append({
                "audio": base64.b64encode(pcm).decode("ascii"),
            })
    finally:
        router.release.set()
    await session._response_task

    assert websocket.done()[-1]["response"]["status"] == "completed"


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


# ---------------------------------------------------------------------------
# The AI assistant: waiting phrases, and talking over the wait (docs/AI-ASSISTANT.md, "Waiting phrases").
# Every delay is a few milliseconds, set through the session's assistant overrides.
# ---------------------------------------------------------------------------


def _reply(text, thinking=None):
    from src.assistant import AssistantReply

    return AssistantReply(text=text, thinking=thinking, finish_reason="stop")


class FakeAssistant:
    """Stands in for src.assistant.AssistantClient. Each reply() takes the next step of its script: an
    AssistantReply is the answer, an AssistantError is raised, an asyncio.Event is waited on before
    answering with ``after``."""

    def __init__(self, *script, after=None):
        self.script = list(script)
        self.after = after
        self.calls = []
        self.cancelled = 0

    async def reply(self, messages, settings):
        self.calls.append((messages, settings))
        step = self.script.pop(0) if self.script else self.after
        if isinstance(step, asyncio.Event):
            try:
                await step.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            step = self.after
        if isinstance(step, BaseException):
            raise step
        return step


def _level(text):
    """The sample PhraseRouter speaks `text` with, as float32 and as the int16 the client receives."""
    from src.tts.pipeline import float32_to_int16

    value = np.float32((zlib.crc32(text.encode("utf-8")) % 15000 + 1000) / 32767)
    return value, int(float32_to_int16(np.array([value], dtype=np.float32))[0])


class PhraseRouter:
    """Speaks each text as `seconds` of its own constant level, so every delta shows which text it is.
    A text in `hold` waits for its threading.Event before any audio."""

    def __init__(self, seconds=0.01):
        self.samples = int(24000 * seconds)
        self.texts = []
        self.hold = {}

    def sample_rate_for(self, _model):
        return 24000

    def synthesize(self, *, text, **_kwargs):
        self.texts.append(text)
        gate = self.hold.get(text)
        if gate is not None:
            gate.wait(timeout=2)
        yield np.full(self.samples, _level(text)[0], dtype=np.float32)


class TimedWebSocket(DummyWebSocket):
    def __init__(self):
        super().__init__()
        self.times = []

    async def send_json(self, event):
        self.times.append(time.perf_counter())
        self.sent.append(event)

    def of_type(self, event_type):
        return [event for event in self.sent if event["type"] == event_type]

    def deltas(self):
        return self.of_type("response.output_audio.delta")

    def timeline(self, texts):
        """Filler events, reply text and audio in order; consecutive audio of one text is one entry."""
        levels = {_level(text)[1]: text for text in texts}
        entries = []
        for event in self.sent:
            if event["type"] == "open_speech.assistant.filler":
                entry = ("filler", event["text"])
            elif event["type"] == "response.output_audio_transcript.delta":
                entry = ("text", event["delta"])
            elif event["type"] == "response.output_audio.delta":
                samples = set(np.frombuffer(base64.b64decode(event["delta"]), dtype="<i2").tolist())
                entry = ("audio", "+".join(sorted(levels.get(sample, str(sample)) for sample in samples)))
            else:
                continue
            if not (entries and entries[-1] == entry and entry[0] == "audio"):
                entries.append(entry)
        return entries


@pytest.fixture
def assistant_env(monkeypatch, tmp_path):
    """The documented .env defaults for the presets, and a preset file of the test's own."""
    from src.config import settings

    for name, value in {
        "os_assistant_model": "test-model",
        "os_assistant_system_prompt": "",
        "os_assistant_history_turns": 20,
        "os_assistant_show_thinking": True,
        "os_assistant_speak_errors": True,
        "os_assistant_timeout_s": 120.0,
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "os_studio_db_path", str(tmp_path / "data" / "studio.db"))
    return settings


async def _assistant_session(assistant, router, *, probabilities=(), **overrides):
    """A GA session talking to `assistant`, with server VAD scripted by `probabilities`."""
    from src.realtime.server import RealtimeSession

    websocket = TimedWebSocket()
    session = RealtimeSession(websocket, router, ga=True, assistant=assistant)
    scripted = iter(probabilities)
    session.audio_buffer = InputAudioBuffer(vad=lambda _audio: next(scripted, 0.0), threshold=0.5, silence_duration_ms=64)
    await session.handle_event({"type": "session.update", "session": {
        "turn_detection": {"silence_duration_ms": 64}, "assistant": overrides,
    }})
    assert websocket.of_type("error") == [], websocket.of_type("error")
    return session, websocket


async def _ask(session):
    """A response.create with no text goes to the AI."""
    await session.handle_event({"type": "response.create", "response": {}})
    return session._response_task


async def _speak(session, transcript):
    """One turn of speech (three loud windows, then 64 ms of silence) heard as `transcript`."""
    with (
        patch("src.realtime.server.decode_audio_to_pcm16", side_effect=lambda data, _fmt, target_rate: data),
        patch("src.realtime.server.stt_router.transcribe", return_value={"text": transcript}),
    ):
        await session._handle_input_audio_buffer_append({
            "audio": base64.b64encode(b"\x01\x00" * (5 * 512)).decode("ascii"),
        })


async def _done(task, timeout=5):
    await asyncio.wait_for(asyncio.shield(task), timeout=timeout)


PHRASES = ["Hmm.", "OK, just a second.", "Let me think about that."]


@pytest.mark.asyncio
async def test_no_phrase_when_the_reply_comes_before_filler_after_ms(assistant_env):
    """A reply sooner than filler_after_ms gets no phrase."""
    router = PhraseRouter()
    session, websocket = await _assistant_session(
        FakeAssistant(_reply("Right away.")), router, fillers=PHRASES, filler_after_ms=300,
    )
    try:
        await _done(await _ask(session))
    finally:
        await session.close()

    assert websocket.of_type("open_speech.assistant.filler") == []
    assert websocket.timeline([*PHRASES, "Right away."]) == [("text", "Right away."), ("audio", "Right away.")]


@pytest.mark.asyncio
async def test_phrases_come_after_filler_after_ms_then_every_filler_every_ms_up_to_filler_max(assistant_env):
    """Each phrase's audio follows its open_speech.assistant.filler event; the
    phrases are not in the transcript, the history or response.done."""
    release = asyncio.Event()
    router = PhraseRouter()
    session, websocket = await _assistant_session(
        FakeAssistant(release, after=_reply("Paris.")), router,
        fillers=PHRASES, filler_after_ms=100, filler_every_ms=100, filler_max=3,
    )
    session.conversation.add_user("What's the capital of France?")
    try:
        task = await _ask(session)
        await _wait_for(lambda: len(websocket.of_type("open_speech.assistant.filler")) == 3, timeout=3)
        await asyncio.sleep(0.25)  # two more gaps: no fourth phrase
        assert len(websocket.of_type("open_speech.assistant.filler")) == 3
        release.set()
        await _done(task)
    finally:
        release.set()
        await session.close()

    fillers = websocket.of_type("open_speech.assistant.filler")
    said = [event["text"] for event in fillers]
    assert set(said) <= set(PHRASES)
    assert all(a != b for a, b in itertools.pairwise(said))
    expected = [entry for text in said for entry in (("filler", text), ("audio", text))]
    assert websocket.timeline([*PHRASES, "Paris."]) == [*expected, ("text", "Paris."), ("audio", "Paris.")]
    started = websocket.times[websocket.sent.index(websocket.of_type("response.created")[0])]
    at = [websocket.times[websocket.sent.index(event)] for event in fillers]
    # The event loop's timers may fire up to one clock tick early (15.6 ms on Windows).
    assert at[0] - started >= 0.07
    assert all(later - earlier >= 0.07 for earlier, later in itertools.pairwise(at))
    response_id = websocket.of_type("response.created")[0]["response"]["id"]
    assert {event["response_id"] for event in fillers} == {response_id}
    done = websocket.done()[-1]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"][0]["transcript"] == "Paris."
    assert [event["transcript"] for event in websocket.of_type("response.output_audio_transcript.done")] == ["Paris."]
    assert session.conversation.for_request(20) == [
        {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris."},
    ]


@pytest.mark.asyncio
async def test_a_phrase_is_never_said_twice_in_a_row_in_a_session(assistant_env):
    """Phrases are chosen at random, never the same one twice in a row."""
    phrases = ["A.", "B.", "C."]
    router = PhraseRouter(seconds=0.001)
    first, second = asyncio.Event(), asyncio.Event()
    session, websocket = await _assistant_session(
        FakeAssistant(first, second, after=_reply("Done.")), router,
        fillers=phrases, filler_after_ms=1, filler_every_ms=1, filler_max=10,
    )
    try:
        for release, count in ((first, 10), (second, 20)):
            task = await _ask(session)
            await _wait_for(lambda count=count: len(websocket.of_type("open_speech.assistant.filler")) == count, timeout=3)
            release.set()
            await _done(task)
    finally:
        first.set()
        second.set()
        await session.close()

    said = [event["text"] for event in websocket.of_type("open_speech.assistant.filler")]
    assert len(said) == 20
    assert all(a != b for a, b in itertools.pairwise(said)), said  # across both responses


@pytest.mark.parametrize("answer", ["reply", "error"])
@pytest.mark.asyncio
async def test_a_phrase_in_progress_is_finished_then_the_answer_follows(assistant_env, answer):
    """A phrase that has started is finished, and the reply (or the error's
    sentence) follows straight after it."""
    from src.assistant import ERROR_SENTENCES, AssistantError
    from src.realtime import server

    release = asyncio.Event()
    outcome = _reply("The answer.") if answer == "reply" else AssistantError("No whole reply", "assistant_timeout")
    final = "The answer." if answer == "reply" else ERROR_SENTENCES["assistant_timeout"]
    router = PhraseRouter(seconds=0.3)
    with patch.object(server.settings, "os_realtime_audio_lead_ms", 50):
        session, websocket = await _assistant_session(
            FakeAssistant(release, after=outcome), router, fillers=["Hmm."], filler_after_ms=0, filler_max=1,
        )
        try:
            task = await _ask(session)
            await _wait_for(lambda: len(websocket.deltas()) >= 1, timeout=3)
            release.set()  # while the phrase is still being sent, paced to playback
            await _done(task)
        finally:
            release.set()
            await session.close()

    timeline = websocket.timeline(["Hmm.", final])
    assert [entry for entry in timeline if entry[0] != "text"] == [("filler", "Hmm."), ("audio", "Hmm."), ("audio", final)]
    phrase_level = _level("Hmm.")[1]
    phrase_bytes = sum(
        len(base64.b64decode(d["delta"])) for d in websocket.deltas()
        if set(np.frombuffer(base64.b64decode(d["delta"]), dtype="<i2").tolist()) == {phrase_level}
    )
    assert phrase_bytes == router.samples * 2


@pytest.mark.asyncio
async def test_no_phrase_starts_once_the_reply_has_arrived(assistant_env):
    """A phrase whose audio is still being made when the reply comes is dropped,
    and the reply doesn't wait for it."""
    router = PhraseRouter()
    gate = threading.Event()
    router.hold["Hmm."] = gate
    release = asyncio.Event()
    session, websocket = await _assistant_session(
        FakeAssistant(release, after=_reply("Straight away.")), router, fillers=["Hmm."], filler_after_ms=0,
    )
    try:
        task = await _ask(session)
        await _wait_for(lambda: "Hmm." in router.texts, timeout=3)
        release.set()
        await _done(task, timeout=1)  # the held phrase would keep it for up to 2 s
    finally:
        release.set()
        gate.set()
        await session.close()

    assert websocket.of_type("open_speech.assistant.filler") == []
    assert websocket.timeline(["Hmm.", "Straight away."]) == [("text", "Straight away."), ("audio", "Straight away.")]


@pytest.mark.parametrize("overrides", [
    {"fillers": [], "filler_after_ms": 0},
    {"fillers": ["Hmm."], "filler_after_ms": 0, "filler_max": 0},
], ids=["empty-list", "filler-max-0"])
@pytest.mark.asyncio
async def test_no_phrases_with_an_empty_list(assistant_env, overrides):
    """An empty list (or filler_max 0) turns the phrases off."""
    release = asyncio.Event()
    router = PhraseRouter()
    session, websocket = await _assistant_session(FakeAssistant(release, after=_reply("Here.")), router, **overrides)
    try:
        task = await _ask(session)
        await asyncio.sleep(0.1)
        release.set()
        await _done(task)
    finally:
        release.set()
        await session.close()

    assert websocket.of_type("open_speech.assistant.filler") == []
    assert websocket.done()[-1]["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_real_words_over_the_wait_cancel_the_request_and_the_phrases_and_start_the_next_turn(assistant_env):
    """Talking over the wait works as talking over a reply does.
    The earlier question stays in the history with no answer."""
    router = PhraseRouter()
    never = asyncio.Event()
    fake = FakeAssistant(never, _reply("It's three o'clock."))
    session, websocket = await _assistant_session(
        fake, router, probabilities=[0.9, 0.9, 0.9, 0.1, 0.1] * 2,
        fillers=["Hmm."], filler_after_ms=0, filler_every_ms=5, filler_max=10,
    )
    try:
        await _speak(session, "What's the capital of Peru?")
        first = session._response_task
        await _wait_for(lambda: websocket.of_type("open_speech.assistant.filler"), timeout=3)
        await _speak(session, "Actually, what time is it?")
        assert first.done()
        await _done(session._response_task)
    finally:
        never.set()
        await session.close()

    assert fake.cancelled == 1
    done = websocket.done()
    assert [event["response"]["status"] for event in done] == ["cancelled", "completed"]
    assert done[0]["response"]["status_details"] == {"type": "cancelled", "reason": "turn_detected"}
    first_id = done[0]["response"]["id"]
    cancelled_at = websocket.sent.index(done[0])
    assert all(
        websocket.sent.index(event) < cancelled_at
        for event in websocket.of_type("open_speech.assistant.filler") if event["response_id"] == first_id
    )
    assert [event["role"] for event in fake.calls[1][0]] == ["system", "user", "user"]
    assert [event["content"] for event in fake.calls[1][0][1:]] == ["What's the capital of Peru?", "Actually, what time is it?"]
    assert [event["delta"] for event in websocket.of_type("response.output_audio_transcript.delta")] == ["It's three o'clock."]


@pytest.mark.asyncio
async def test_a_backchannel_over_the_wait_lets_it_continue(assistant_env):
    """"mm-hmm" over the wait doesn't cancel it and isn't a question."""
    router = PhraseRouter()
    release = asyncio.Event()
    fake = FakeAssistant(release, after=_reply("Lima."))
    session, websocket = await _assistant_session(
        fake, router, probabilities=[0.9, 0.9, 0.9, 0.1, 0.1] * 2, fillers=["Hmm."], filler_after_ms=0, filler_max=1,
    )
    try:
        await _speak(session, "What's the capital of Peru?")
        task = session._response_task
        await _wait_for(lambda: websocket.of_type("open_speech.assistant.filler"), timeout=3)
        await _speak(session, "Mm-hmm.")
        assert session._response_task is task and not task.done()
        release.set()
        await _done(task)
    finally:
        release.set()
        await session.close()

    assert fake.cancelled == 0 and len(fake.calls) == 1
    assert [event["response"]["status"] for event in websocket.done()] == ["completed"]
    assert session.conversation.for_request(20) == [
        {"role": "user", "content": "What's the capital of Peru?"}, {"role": "assistant", "content": "Lima."},
    ]


@pytest.mark.asyncio
async def test_response_cancel_during_the_wait_gives_a_cancelled_response_done(assistant_env):
    """response.cancel stops the AI request and the phrases."""
    router = PhraseRouter()
    never = asyncio.Event()
    fake = FakeAssistant(never)
    session, websocket = await _assistant_session(fake, router, fillers=["Hmm."], filler_after_ms=0, filler_every_ms=5)
    try:
        await _ask(session)
        await _wait_for(lambda: websocket.of_type("open_speech.assistant.filler"), timeout=3)
        await session.handle_event({"type": "response.cancel"})
        fillers = len(websocket.of_type("open_speech.assistant.filler"))
        await asyncio.sleep(0.05)
    finally:
        never.set()
        await session.close()

    assert fake.cancelled == 1
    assert session._response_task is None
    assert len(websocket.of_type("open_speech.assistant.filler")) == fillers
    assert websocket.done()[-1]["response"]["status_details"] == {"type": "cancelled", "reason": "client_cancelled"}
    assert websocket.sent[-1]["type"] == "response.done"


class _ListeningAssistantClient(DummyWebSocket):
    """Asks the AI once (response.create with no text), then only listens."""

    def __init__(self):
        from starlette.websockets import WebSocketState

        super().__init__()
        self.scope = {"subprotocols": []}
        self.headers = {}
        self.application_state = WebSocketState.CONNECTED
        self.closed = None
        self.requested = False

    async def accept(self, subprotocol=None):
        return None

    async def receive(self):
        if not self.requested:
            self.requested = True
            return {"type": "websocket.receive", "text": '{"type": "response.create", "response": {}}'}
        await asyncio.Event().wait()

    async def close(self, code=1000, reason=""):
        from starlette.websockets import WebSocketState

        self.closed = (code, reason, asyncio.get_running_loop().time())
        self.application_state = WebSocketState.DISCONNECTED


@pytest.mark.asyncio
async def test_the_idle_timeout_does_not_fire_during_a_long_ai_wait(assistant_env):
    """The wait counts as busy, so a slow AI doesn't close the session."""
    from src.realtime import server

    loop = asyncio.get_running_loop()
    answered = asyncio.Event()
    fake = FakeAssistant(answered, after=_reply("Finally."))
    websocket = _ListeningAssistantClient()
    with (
        patch.object(server.settings, "os_realtime_idle_timeout_s", 0.2),
        patch("src.realtime.server.on_close", side_effect=lambda callback: callback),
        patch("src.realtime.server.get_vad_model", new=AsyncMock(return_value=MagicMock())),
        patch("src.realtime.server.SileroVAD", return_value=lambda _audio: 0.0),
    ):
        started = loop.time()
        loop.call_later(0.6, answered.set)  # three idle windows of AI wait
        endpoint = asyncio.create_task(server.realtime_endpoint(websocket, tts_router=PhraseRouter(), assistant=fake))
        try:
            await _wait_for(lambda: websocket.done(), timeout=5)
            reply_done = loop.time()
            await _wait_for(lambda: websocket.closed is not None, timeout=5)
        finally:
            answered.set()
            endpoint.cancel()
            await asyncio.gather(endpoint, return_exceptions=True)

    assert websocket.done()[0]["response"]["status"] == "completed"
    assert reply_done - started >= 0.55
    assert websocket.closed[:2] == (4008, "Session idle timeout")
    assert websocket.closed[2] - reply_done >= 0.15


@pytest.mark.asyncio
async def test_shutdown_stops_a_reply_still_waiting_on_the_ai(assistant_env):
    """The realtime route registers its shutdown with src.assistant.on_close, so the
    AI requests still waiting stop before the shared client closes."""
    from src.realtime import server

    registered = []
    never = asyncio.Event()
    fake = FakeAssistant(never)
    websocket = _ListeningAssistantClient()
    with (
        patch("src.realtime.server.on_close", side_effect=lambda callback: registered.append(callback) or callback),
        patch("src.realtime.server.get_vad_model", new=AsyncMock(return_value=MagicMock())),
        patch("src.realtime.server.SileroVAD", return_value=lambda _audio: 0.0),
    ):
        endpoint = asyncio.create_task(server.realtime_endpoint(websocket, tts_router=PhraseRouter(), assistant=fake))
        try:
            await _wait_for(lambda: fake.calls, timeout=3)
            assert len(registered) == 1
            await asyncio.wait_for(registered[0](), timeout=3)
            assert fake.cancelled == 1
            assert websocket.done()[-1]["response"]["status"] == "cancelled"
        finally:
            never.set()
            endpoint.cancel()
            await asyncio.gather(endpoint, return_exceptions=True)


# ---------------------------------------------------------------------------
# The waiting phrases' timing as the listener hears it. docs/AI-ASSISTANT.md › Waiting phrases,
# `filler_every_ms` is "The gap before the next phrase while it still waits"; "A phrase that has started is
# finished, and the reply follows straight after it. No phrase starts once the reply has arrived." With barge-in
# on, audio is sent up to OS_REALTIME_AUDIO_LEAD_MS ahead of playback, and without it as fast as it is made, so
# the time a phrase is sent is not when it is heard.
#
# Phrases of a second or more would make these tests sleep for seconds, so they run on a virtual-time event loop
# (the clock jumps to the next timer instead of sleeping; a real-time guard stops a hang after 60 s), with the
# synthesis run inline and a seeded phrase choice. The listener is modelled as the server models the client
# (`_Playback.ends`): each delta plays from the later of when it was sent and when the previous delta ends.
# ---------------------------------------------------------------------------

_TIMED_REPLY = "Paris."
_VIRTUAL_REAL_TIME_LIMIT_S = 60.0


class _VirtualClock:
    def __init__(self):
        self.now = 0.0
        self.real_start = time.monotonic()


class _JumpingSelector(selectors.SelectSelector):
    """Never sleeps: when nothing is ready, the virtual clock jumps to the loop's next timer."""

    def __init__(self, clock):
        super().__init__()
        self._clock = clock

    def select(self, timeout=None):
        if time.monotonic() - self._clock.real_start > _VIRTUAL_REAL_TIME_LIMIT_S:
            raise RuntimeError(f"virtual-time loop: still running after {_VIRTUAL_REAL_TIME_LIMIT_S} s of real time")
        ready = super().select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            ready = super().select(5)  # nothing scheduled: only another thread could still wake the loop
            if not ready:
                raise RuntimeError("virtual-time loop: nothing is scheduled and nothing is ready (a deadlock)")
            return ready
        self._clock.now += timeout
        return []


class _VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        self._clock = _VirtualClock()
        super().__init__(_JumpingSelector(self._clock))
        self._clock_resolution = 1e-9  # timers fire at their time, not up to one OS clock tick early

    def time(self):
        return self._clock.now


def _run_virtual(coro):
    loop = _VirtualTimeLoop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=300))  # 300 s of virtual time
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


class _InlineExecutor(concurrent.futures.Executor):
    """Runs the phrase and reply synthesis at once, on the loop's thread: no real time passes."""

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # handed to the awaiting coroutine
            future.set_exception(exc)
        return future


class _LongPhraseRouter:
    """Speaks each text as `seconds` of its own constant level (`_level`)."""

    def __init__(self, seconds):
        self.samples = int(24000 * seconds)

    def sample_rate_for(self, _model):
        return 24000

    def synthesize(self, *, text, **_kwargs):
        yield np.full(self.samples, _level(text)[0], dtype=np.float32)


class _VirtualTimeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, event):
        self.sent.append((asyncio.get_running_loop().time(), event))


class _AnswersAfter:
    """Stands in for src.assistant.AssistantClient: answers after `after_s` seconds of virtual time."""

    def __init__(self, after_s):
        self.after_s = after_s
        self.arrived_at = None

    async def reply(self, _messages, _settings):
        await asyncio.sleep(self.after_s)
        self.arrived_at = asyncio.get_running_loop().time()
        return _reply(_TIMED_REPLY)


@pytest.fixture
def timed_env(assistant_env, monkeypatch):
    import random

    from src.realtime import server

    monkeypatch.setattr(assistant_env, "os_realtime_audio_lead_ms", 1000)  # the default
    monkeypatch.setattr(assistant_env, "os_assistant_show_thinking", False)
    monkeypatch.setattr(server, "_executor", _InlineExecutor())
    monkeypatch.setattr(server, "random", random.Random(20261107))
    return assistant_env


def _heard(sent, texts):
    """Each text's [text, start, end] as a client playing the deltas back to back hears it."""
    levels = {_level(text)[1]: text for text in texts}
    spans = []
    end = 0.0
    for at, event in sent:
        if event["type"] != "response.output_audio.delta":
            continue
        pcm = np.frombuffer(base64.b64decode(event["delta"]), dtype="<i2")
        assert len(set(pcm.tolist())) == 1, "one text per delta"
        text = levels[int(pcm[0])]
        start = max(at, end)
        end = start + len(pcm) / 24000
        if spans and spans[-1][0] == text:
            spans[-1][2] = end
        else:
            spans.append([text, start, end])
    return spans


async def _timed_turn(*, interrupt, phrase_s, after_ms, every_ms, filler_max, reply_after_s):
    """One GA response.create answered by the fake AI: the spans heard and when the reply arrived, in seconds
    from response.created."""
    from src.realtime.server import RealtimeSession

    socket = _VirtualTimeSocket()
    assistant = _AnswersAfter(reply_after_s)
    session = RealtimeSession(socket, _LongPhraseRouter(phrase_s), ga=True, assistant=assistant)
    session.audio_buffer = InputAudioBuffer(vad=lambda _audio: 0.0, threshold=0.5, silence_duration_ms=64)
    try:
        await session.handle_event({"type": "session.update", "session": {
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": interrupt}}},
            "assistant": {"fillers": PHRASES, "filler_after_ms": after_ms, "filler_every_ms": every_ms,
                          "filler_max": filler_max},
        }})
        assert session._audio_lead_s() == (1.0 if interrupt else None), "pacing as the test means it"
        await asyncio.sleep(0.5)  # the phrase cache warms (virtual time)
        await session.handle_event({"type": "response.create", "response": {}})
        await asyncio.wait_for(asyncio.shield(session._response_task), timeout=120)
    finally:
        await session.close()
    assert [event for _at, event in socket.sent if event["type"] == "error"] == []
    started = next(at for at, event in socket.sent if event["type"] == "response.created")
    spans = [[text, start - started, end - started] for text, start, end in _heard(socket.sent, [*PHRASES, _TIMED_REPLY])]
    return spans, assistant.arrived_at - started


def _show(spans):
    return [(text, round(start, 3), round(end, 3)) for text, start, end in spans]


_PACING = pytest.mark.parametrize("interrupt", [True, False], ids=["paced", "not-paced"])


@_PACING
def test_the_gap_between_phrases_is_filler_every_ms_as_heard(timed_env, interrupt):
    """With the preset's defaults (after 1200 ms, every 4000 ms, at most 3) and 1.3 s
    phrases, the listener heard 2.95 s (paced) or 2.7 s (not paced) of silence between phrases, not 4 s."""
    spans, _arrived = _run_virtual(_timed_turn(
        interrupt=interrupt, phrase_s=1.3, after_ms=1200, every_ms=4000, filler_max=3, reply_after_s=14.0,
    ))
    phrases = [span for span in spans if span[0] != _TIMED_REPLY]
    assert len(phrases) == 3, _show(spans)
    assert abs(phrases[0][1] - 1.2) < 0.001, _show(spans)  # the first phrase is on time
    gaps = [later[1] - earlier[2] for earlier, later in itertools.pairwise(phrases)]
    assert all(abs(gap - 4.0) < 0.001 for gap in gaps), (
        f"gaps of {[round(gap, 3) for gap in gaps]} s between the phrases, not 4.0 s; playback: {_show(spans)}"
    )


@_PACING
@pytest.mark.parametrize("every_ms", [200, 0], ids=["every-200ms", "every-0ms"])
def test_no_phrase_is_heard_after_the_reply_arrived(timed_env, interrupt, every_ms):
    """With a gap shorter than a phrase (200 ms, or 0, the documented minimum), the
    phrases sent ahead were heard after the reply arrived, and the reply 1.3 to 4.3 s late. The listener should
    hear the rest of the phrase playing when it arrives, then the reply."""
    spans, arrived = _run_virtual(_timed_turn(
        interrupt=interrupt, phrase_s=1.0, after_ms=0, every_ms=every_ms, filler_max=5, reply_after_s=0.7,
    ))
    assert abs(arrived - 0.7) < 0.001
    first = spans[0]
    assert first[0] != _TIMED_REPLY and first[1] < arrived < first[2], _show(spans)
    late = [(text, round(start - arrived, 3)) for text, start, _end in spans if text != _TIMED_REPLY and start > arrived]
    reply = next(span for span in spans if span[0] == _TIMED_REPLY)
    assert late == [] and abs(reply[1] - first[2]) < 0.001, (
        f"{len(late)} phrase(s) heard after the reply arrived {late}, and the reply starts "
        f"{reply[1] - arrived:.3f} s after it arrived; playback: {_show(spans)}"
    )


@_PACING
def test_the_reply_follows_the_one_phrase_in_progress_as_heard(timed_env, interrupt):
    """A control: with the default 4 s gap, a reply arriving 0.5 s
    into a 1.3 s phrase starts as that phrase ends. It shows the playback model agrees with the server's pacing."""
    spans, arrived = _run_virtual(_timed_turn(
        interrupt=interrupt, phrase_s=1.3, after_ms=0, every_ms=4000, filler_max=3, reply_after_s=0.5,
    ))
    assert [span[0] for span in spans][1:] == [_TIMED_REPLY], _show(spans)
    assert abs(spans[0][1]) < 0.001 and abs(spans[0][2] - 1.3) < 0.001, _show(spans)
    assert abs(spans[1][1] - spans[0][2]) < 0.001, _show(spans)
    assert abs(arrived - 0.5) < 0.001


# ---------------------------------------------------------------------------
# Each session.update that changed the speech settings (here the speed) left the
# waiting phrase being made for the old ones queued on the shared realtime pool, and the 32-phrase cache dropped old
# entries without cancelling them, so past 32 they still ran after the client had gone, ahead of every session's
# turn transcriptions. RealtimeSession.close()'s docstring: it cancels "the waiting phrases being made". A
# synthesis already running can't be stopped, so the bound is on those not yet started.
# ---------------------------------------------------------------------------


class _GatedPhraseRouter:
    """Every synthesis waits on ``gate`` (at most 10 s), and each one that starts is noted in ``started``."""

    def __init__(self):
        self.gate = threading.Event()
        self.lock = threading.Lock()
        self.started = []

    def sample_rate_for(self, _model):
        return 24000

    def note(self, what):
        with self.lock:
            self.started.append(what)

    def synthesize(self, *, text, **_kwargs):
        self.note(text)
        self.gate.wait(timeout=10)
        yield np.zeros(240, dtype=np.float32)


@pytest.fixture
def own_pool(monkeypatch):
    """The realtime pool as a pool of the test's own, so syntheses held on a gate never hold up another test.
    ``own_pool(workers)`` makes it; the server's has 4 workers."""
    from src.realtime import server

    pools = []

    def make(workers=4):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="test-realtime")
        pools.append(pool)
        monkeypatch.setattr(server, "_executor", pool)
        return pool

    yield make
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


async def _change_the_speed(session, times):
    """``times`` session.updates that change only the speed, each followed by one loop turn, in which the new
    warm-up starts its first phrase. The first loop turn lets the session's first warm-up start its own."""
    await asyncio.sleep(0)
    for index in range(times):
        await session.handle_event({"type": "session.update", "session": {"speed": round(0.5 + index * 0.005, 3)}})
        await asyncio.sleep(0)


@pytest.mark.parametrize("updates", [10, 40, 100])
@pytest.mark.asyncio
async def test_no_phrase_for_speech_settings_the_session_left_is_made_after_it_closes(assistant_env, own_pool, updates):
    """After many speed changes and close(), as when the
    client leaves, no phrase synthesis starts. Only the one already running finishes: the
    session holds one worker (before, each of the pool's four held one)."""
    pool = own_pool(4)
    router = _GatedPhraseRouter()
    session, _websocket = await _assistant_session(FakeAssistant(), router, fillers=["Hmm."])
    try:
        await _change_the_speed(session, updates)
        await _wait_for(lambda: len(router.started) == 1, timeout=3)  # the session's one phrase at a time
        await asyncio.sleep(0.1)  # (no other one starts meanwhile)
        await session.close()
        await asyncio.sleep(0.05)  # a cancel reaches the pool's future in a loop callback
    finally:
        router.gate.set()
    await asyncio.wait_for(asyncio.to_thread(pool.shutdown, wait=True), timeout=30)  # whatever is left, runs

    assert len(router.started) == 1, f"{len(router.started) - 1} phrases were made after the session closed"


@pytest.mark.parametrize("updates", [1, 40])
@pytest.mark.asyncio
async def test_a_turns_transcription_waits_behind_at_most_one_phrase_not_yet_started(assistant_env, own_pool, updates):
    """While the session is open, a job sent to the pool after the speed changes
    (what _commit_and_transcribe does with a turn) waits behind at most one phrase that hadn't started: the one for
    the session's speech now. Before, it waited behind one for each speed the session had left."""
    from src.realtime import server

    own_pool(4)
    router = _GatedPhraseRouter()
    session, _websocket = await _assistant_session(FakeAssistant(), router, fillers=["Hmm."])
    try:
        await _change_the_speed(session, updates)
        running = 1  # one phrase at a time per session; before, min(updates + 1, 4)
        await _wait_for(lambda: len(router.started) == running, timeout=3)
        transcription = asyncio.get_running_loop().run_in_executor(server._executor, router.note, "transcription")
        await asyncio.sleep(0.05)
    finally:
        router.gate.set()
    await asyncio.wait_for(transcription, timeout=30)
    await session.close()

    ahead = router.started.index("transcription") - running
    assert ahead <= 1, f"the transcription waited behind {ahead} phrases that hadn't started"


@pytest.mark.parametrize("off", [{"fillers": []}, {"filler_max": 0}], ids=["empty-list", "filler-max-0"])
@pytest.mark.asyncio
async def test_turning_the_phrases_off_cancels_the_ones_still_to_be_made(assistant_env, own_pool, off):
    """An update that turns the phrases off stops the warm-up, and its phrase
    still waiting for the pool is never made, nor the phrases after it."""
    pool = own_pool(1)
    router = _GatedPhraseRouter()
    busy = pool.submit(router.gate.wait, 10)  # the one worker is busy, so the phrase waits in the queue
    session, _websocket = await _assistant_session(FakeAssistant(), router, fillers=["A.", "B."])
    try:
        await asyncio.sleep(0.05)
        await session.handle_event({"type": "session.update", "session": {"assistant": off}})
        await asyncio.sleep(0.05)
    finally:
        router.gate.set()
    await asyncio.wait_for(asyncio.wrap_future(busy), timeout=30)
    await asyncio.wait_for(asyncio.to_thread(pool.shutdown, wait=True), timeout=30)
    await session.close()

    assert router.started == []


@pytest.mark.asyncio
async def test_a_phrase_a_reply_is_waiting_for_is_kept_when_the_speed_changes(assistant_env):
    """A guard: the phrase a reply is waiting
    on is for that reply's settings, so a speed change during the wait doesn't cancel it; it is said, then the
    answer follows."""
    release = asyncio.Event()
    router = PhraseRouter()
    gate = threading.Event()
    router.hold["Hmm."] = gate
    session, websocket = await _assistant_session(
        FakeAssistant(release, after=_reply("Paris.")), router, fillers=["Hmm."], filler_after_ms=0, filler_max=1,
    )
    try:
        task = await _ask(session)
        await asyncio.sleep(0.05)  # the reply waits on the phrase being made
        await session.handle_event({"type": "session.update", "session": {"speed": 1.2}})
        await asyncio.sleep(0.05)
        gate.set()
        await _wait_for(lambda: len(websocket.of_type("open_speech.assistant.filler")) == 1, timeout=3)
        release.set()
        await _done(task)
    finally:
        gate.set()
        release.set()
        await session.close()

    assert websocket.timeline(["Hmm.", "Paris."]) == [
        ("filler", "Hmm."), ("audio", "Hmm."), ("text", "Paris."), ("audio", "Paris."),
    ]
    assert websocket.done()[-1]["response"]["status"] == "completed"

# ---------------------------------------------------------------------------
# A phrase whose synthesis was already running when the session left its speech
# settings couldn't be stopped, and the next warm-up started its own phrase at once on another worker. So one
# session's speed changes held up to all four realtime workers, ahead of every session's turn transcriptions.
# _warm_filler_cache's rule: "One at a time: the workers also transcribe each turn."
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("updates", [0, 1, 3, 10])
@pytest.mark.asyncio
async def test_one_session_makes_one_waiting_phrase_at_a_time_however_often_the_speed_changes(
    assistant_env, own_pool, updates,
):
    """After ``updates`` speed changes, with every synthesis held, the session holds
    one realtime worker, and a job sent to the pool then (a turn's transcription) starts at once. ``[0]`` is the
    control. Once the held phrase ends, the warm-up makes the phrase for the session's speech now."""
    from src.realtime import server

    own_pool(4)
    router = _GatedPhraseRouter()
    session, _websocket = await _assistant_session(FakeAssistant(), router, fillers=["Hmm."])
    try:
        await _change_the_speed(session, updates)
        await asyncio.sleep(0.2)
        running = len(router.started)
        transcription = asyncio.get_running_loop().run_in_executor(server._executor, router.note, "transcription")
        await asyncio.wait_for(asyncio.shield(transcription), timeout=1)
        router.gate.set()
        # The phrase for the speed the session has now is made after the one that was running.
        await _wait_for(lambda: len(router.started) == (3 if updates else 2), timeout=3)
        await asyncio.wait_for(asyncio.wait(set(session._filler_cache.values())), timeout=3)
        made = [speech.speed for speech, _phrase in session._filler_cache]
    finally:
        router.gate.set()
        await session.close()

    assert running == 1, f"one session held {running} realtime workers with waiting phrases"
    assert made == [session.config.speed]


@pytest.mark.asyncio
async def test_a_phrase_left_while_it_runs_ends_unused_and_close_does_not_wait_for_it(assistant_env, own_pool):
    """The session forgets a phrase it left once it has ended, and close() returns at
    once while one is still running (it can't be stopped; it ends on its own)."""
    own_pool(4)
    router = _GatedPhraseRouter()
    session, websocket = await _assistant_session(FakeAssistant(), router, fillers=["Hmm."])
    try:
        await _change_the_speed(session, 2)
        await asyncio.sleep(0.1)
        assert len(router.started) == 1
        await asyncio.wait_for(session.close(), timeout=1)
    finally:
        router.gate.set()
    await _wait_for(lambda: not any(not job.done() for job in session._phrases_left_running), timeout=3)

    assert router.started == ["Hmm."]
    assert websocket.of_type("open_speech.assistant.filler") == []
