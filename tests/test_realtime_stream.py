"""Talk with a streamed AI reply: the voice starts on the first sentence, and each AI voice turn logs its timings.

The rules: OpenAI's Chat Completions streaming format; docs/AI-ASSISTANT.md "Talking to it", "What to expect for
speed" and the troubleshooting note on the timing line. The non-streamed turn's events, phrases and barge-in are in test_realtime.py
and test_realtime_barge_in.py.

No network: the AI client is the real one over an httpx.MockTransport that plays a stream of server-sent events
(thinking pieces, then answer pieces, with pauses, or waiting on an event the test sets). Every wait is bounded.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

AI_URL = "http://ai.stream.test:8080/v1"
API_KEY = "sk-stream-secret-789"
DONE = b"data: [DONE]\n\n"
QUESTION = "What is the weather like?"
THINKING = "The user asks about the weather."
ANSWER = "It's sunny today. Bring sunglasses."


def _event(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _delta(finish_reason=None, **delta) -> bytes:
    return _event({"object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})


def _usage(prompt_tokens=31, completion_tokens=12) -> bytes:
    return _event({"object": "chat.completion.chunk", "choices": [],
                   "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})


class SseStream(httpx.AsyncByteStream):
    """Plays its steps: bytes are sent, a number is a pause, an asyncio.Event is waited on, an exception raised."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.sent = 0
        self.closed = False
        self.finished = False

    async def __aiter__(self):
        for step in self.steps:
            if isinstance(step, (int, float)):
                await asyncio.sleep(step)
            elif isinstance(step, asyncio.Event):
                await step.wait()
            elif isinstance(step, BaseException):
                raise step
            else:
                self.sent += 1
                yield step
        self.finished = True

    async def aclose(self):
        self.closed = True


class StreamServer:
    """A MockTransport handler answering each request with a new SseStream of ``steps``, or with ``answer``."""

    def __init__(self, steps=(), answer=None):
        self.steps = list(steps)
        self.answer = answer
        self.requests: list[httpx.Request] = []
        self.streams: list[SseStream] = []

    def __call__(self, request):
        self.requests.append(request)
        if self.answer is not None:
            return self.answer(request)
        stream = SseStream(self.steps)
        self.streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def _client(server, **kwargs):
    from src.assistant.client import AssistantClient

    return AssistantClient(AI_URL, transport=httpx.MockTransport(server), **kwargs)


class _RecordingWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, event):
        self.sent.append(event)

    async def close(self, code=1000, reason=""):
        pass

    def of_type(self, *types):
        return [event for event in self.sent if event["type"] in types]


class _Router:
    """A TTS router that records each text it speaks; each gives 10 ms of silence."""

    def __init__(self):
        self.texts: list[str] = []

    def sample_rate_for(self, _model):
        return 24000

    def synthesize(self, **kwargs):
        self.texts.append(kwargs["text"])
        yield np.zeros(240, dtype=np.float32)


@pytest.fixture
def assistant_env(monkeypatch, tmp_path):
    """The documented .env defaults for the presets, and a preset file of the test's own."""
    from src.config import settings

    for name, value in {
        "os_assistant_model": "test-model", "os_assistant_system_prompt": "", "os_assistant_thinking": "default",
        "os_assistant_reasoning_effort": "", "os_assistant_max_tokens": 0, "os_assistant_temperature": None,
        "os_assistant_top_p": None, "os_assistant_timeout_s": 120.0, "os_assistant_history_turns": 20,
        "os_assistant_show_thinking": True, "os_assistant_speak_errors": True,
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "os_studio_db_path", str(tmp_path / "data" / "studio.db"))
    return settings


def _scripted_vad(probabilities):
    probs = iter(probabilities)
    vad_model = MagicMock()
    vad_model.session = object()
    return (
        patch("src.realtime.server.get_vad_model", new=AsyncMock(return_value=vad_model)),
        patch("src.realtime.server.SileroVAD", return_value=lambda _audio: next(probs, 0.0)),
    )


async def _session(assistant, *, router, ga=True, assistant_update=None, turns=4):
    """A session with server VAD (64 ms of end-of-turn silence) and no waiting phrases unless asked."""
    from src.realtime.server import RealtimeSession

    get_vad, silero = _scripted_vad([0.9, 0.9, 0.9, 0.1, 0.1] * turns)
    with get_vad, silero:
        session = RealtimeSession(_RecordingWebSocket(), router, ga=ga, assistant=assistant)
        await session.initialize()
        await session.handle_event({"type": "session.update", "event_id": "evt_setup", "session": {
            "turn_detection": {"silence_duration_ms": 64},
            "assistant": {"fillers": [], **(assistant_update or {})},
        }})
    return session


async def _ask(session, text=QUESTION):
    """A typed question, then the response.create that asks the AI."""
    await session.handle_event({"type": "conversation.item.create", "event_id": "evt_q", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
    await session.handle_event({"type": "response.create", "event_id": "evt_ask"})


async def _say(session, transcript, stt_seconds=0.0):
    """One spoken turn (speech, then 64 ms of silence) heard as ``transcript``; waits for its reply."""

    def transcribe(**_kwargs):
        time.sleep(stt_seconds)
        return {"text": transcript}

    pcm = b"\x01\x00" * (5 * 512)
    with patch("src.realtime.server.decode_audio_to_pcm16", side_effect=lambda data, _fmt, target_rate: data), \
            patch("src.realtime.server.stt_router.transcribe", side_effect=transcribe):
        await session.handle_event({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
    await _finish(session)


async def _finish(session):
    task = session._response_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=5)


async def _until(predicate, timeout=3.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _audio(session):
    return session.ws.of_type("response.output_audio.delta", "response.audio.delta")


def _transcript_deltas(session):
    return session.ws.of_type("response.output_audio_transcript.delta", "response.audio_transcript.delta")


def _user(text=QUESTION):
    return {"role": "user", "content": text}


def _ai(text):
    return {"role": "assistant", "content": text}


def _answer_stream(*, release=None, pause=0.02):
    """Thinking pieces, then the answer in small pieces; with ``release``, the last ones wait for it."""
    return [
        _delta(role="assistant", content=""),
        _delta(reasoning_content="The user asks"), pause, _delta(reasoning_content=" about the weather."), pause,
        _delta(content="It's sunny"), pause, _delta(content=" today."), pause, _delta(content=" Bring"),
        *([release] if release is not None else []),
        _delta(content=" sunglasses."), _delta(finish_reason="stop"), _usage(31, 12), DONE,
    ]


# ── The voice starts on the first sentence ───────────────────────────────────


@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
async def test_the_voice_starts_on_the_first_sentence_before_the_ai_sends_its_last_token(assistant_env, ga):
    """The reply is asked for with stream: true, cut into sentences as
    they complete, and each is spoken at once, its text sent just before its audio. The thinking goes out once as
    text and is never spoken; response.done and the history hold the whole answer."""
    release = asyncio.Event()
    server = StreamServer(_answer_stream(release=release))
    client, router = _client(server), _Router()
    session = await _session(client, router=router, ga=ga)
    try:
        await _ask(session)
        heard_early = await _until(lambda: _audio(session))
        spoken_early = list(router.texts)
        stream_finished_early = bool(server.streams) and server.streams[0].finished
        release.set()
        await _finish(session)
    finally:
        release.set()
        await session.close()
        await client.aclose()

    assert heard_early, "no reply audio before the AI's last token"
    assert spoken_early == ["It's sunny today."] and not stream_finished_early
    assert server.bodies[0]["stream"] is True
    assert router.texts == ["It's sunny today.", "Bring sunglasses."]
    assert THINKING not in " ".join(router.texts) and "weather" not in " ".join(router.texts)
    deltas = _transcript_deltas(session)
    assert [event["delta"] for event in deltas] == ["It's sunny today.", " Bring sunglasses."]
    sent = session.ws.sent
    for number, delta in enumerate(deltas):
        at = sent.index(delta)
        nxt = sent.index(deltas[number + 1]) if number + 1 < len(deltas) else len(sent)
        assert any(e["type"] in ("response.output_audio.delta", "response.audio.delta") for e in sent[at:nxt]), (
            "each sentence's text goes out just before its audio")
    assert sent.index(_audio(session)[0]) > sent.index(deltas[0])
    thinking = session.ws.of_type("open_speech.assistant.thinking")
    assert [event["text"] for event in thinking] == [THINKING]
    assert sent.index(thinking[0]) < sent.index(deltas[0])
    done = session.ws.of_type("response.done")[-1]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"][0]["transcript"] == ANSWER
    transcript_done = session.ws.of_type("response.output_audio_transcript.done", "response.audio_transcript.done")
    assert [event["transcript"] for event in transcript_done] == [ANSWER]
    assert session.conversation.for_request(20) == [_user(), _ai(ANSWER)]
    assert session.ws.of_type("error") == []


async def test_the_waiting_phrases_stop_once_the_first_sentence_is_ready(assistant_env):
    """"the reply has arrived" means the first sentence of the answer
    is ready: a phrase plays while the AI thinks, and none starts after that, though the rest is still coming."""
    phrase_heard, rest = asyncio.Event(), asyncio.Event()
    server = StreamServer([
        _delta(reasoning_content="Thinking hard."), phrase_heard,
        _delta(content="The first sentence is here."), _delta(content=" And"), rest,
        _delta(content=" the rest."), _delta(finish_reason="stop"), DONE,
    ])
    client, router = _client(server), _Router()
    session = await _session(client, router=router, assistant_update={
        "fillers": ["Hmm."], "filler_after_ms": 0, "filler_every_ms": 150, "filler_max": 10})
    try:
        await _ask(session)
        assert await _until(lambda: session.ws.of_type("open_speech.assistant.filler")), "a phrase while it thinks"
        phrase_heard.set()
        first_sentence = await _until(lambda: _transcript_deltas(session))
        await asyncio.sleep(0.5)  # three more phrase gaps: none may start
        rest.set()
        await _finish(session)
    finally:
        phrase_heard.set()
        rest.set()
        await session.close()
        await client.aclose()

    assert first_sentence, "the first sentence was spoken before the rest arrived"
    sent = session.ws.sent
    first_text = sent.index(_transcript_deltas(session)[0])
    phrases = [sent.index(event) for event in session.ws.of_type("open_speech.assistant.filler")]
    assert phrases and all(at < first_text for at in phrases), "no phrase once the first sentence is ready"
    assert router.texts[-2:] == ["The first sentence is here.", "And the rest."]
    assert session.ws.of_type("response.done")[-1]["response"]["status"] == "completed"


# ── Interrupting closes the stream ───────────────────────────────────────────


@pytest.mark.parametrize("how", ["response.cancel", "talked over", "connection closed"])
async def test_interrupting_closes_the_stream_and_keeps_what_was_spoken_in_the_history(assistant_env, how):
    """Talking over it, Stop reply (response.cancel) and a closed
    connection close the stream, so the AI server can stop generating; what was spoken stays in the history."""
    never = asyncio.Event()
    server = StreamServer([
        _delta(content="The first sentence is here."), _delta(content=" And then"), never,
        _delta(content=" more."), _delta(finish_reason="stop"), DONE,
    ])
    client, router = _client(server), _Router()
    session = await _session(client, router=router)
    try:
        await _ask(session)
        heard = await _until(lambda: _audio(session))
        if how == "response.cancel":
            await session.handle_event({"type": "response.cancel", "event_id": "evt_stop"})
        elif how == "talked over":
            await session._interrupt()
        else:
            await session.close()
        closed = await _until(lambda: bool(server.streams) and server.streams[0].closed, 2.0)
        await _finish(session)
    finally:
        never.set()
        await session.close()
        await client.aclose()

    assert heard, "the first sentence was spoken while the AI was still sending"
    assert closed, "the stream is closed"
    assert not server.streams[0].finished
    assert router.texts == ["The first sentence is here."]
    done = session.ws.of_type("response.done")[-1]["response"]
    assert done["status"] == "cancelled"
    assert done["status_details"]["reason"] == ("turn_detected" if how == "talked over" else "client_cancelled")
    assert session.conversation.for_request(20) == [_user(), _ai("The first sentence is here.")]


# ── Errors and the cut-off ───────────────────────────────────────────────────


@pytest.mark.parametrize(("tail", "code"), [
    ([httpx.ReadError("connection reset")], "assistant_unreachable"),
    ([_event({"error": {"message": "model crashed", "code": 500}})], "assistant_http_error"),
    ([_delta(finish_reason="length"), _usage(), DONE], "assistant_reply_cut_off"),
], ids=["dropped", "error-event", "length"])
async def test_an_error_or_a_cut_off_after_the_first_sentence_keeps_it_and_reports_an_incomplete_reply(
        assistant_env, tail, code):
    """What was said stays, nothing
    more is said (not the sentence it was in the middle of, not an error sentence), and response.done is
    "incomplete" with the code in status_details.error.code, never a finished answer."""
    from src.assistant import ERROR_SENTENCES

    server = StreamServer([
        _delta(reasoning_content="Thinking."),
        _delta(content="The first sentence is here."), _delta(content=" The second one is cut"), *tail,
    ])
    client, router = _client(server), _Router()
    session = await _session(client, router=router)
    try:
        await _ask(session)
        await _until(lambda: session.ws.of_type("response.done"), 5.0)
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert router.texts == ["The first sentence is here."]
    assert not set(ERROR_SENTENCES.values()) & set(router.texts)
    assert session.ws.of_type("error") == []
    done = session.ws.of_type("response.done")[-1]["response"]
    assert done["status"] == "incomplete"
    assert done["status_details"] == {"type": "incomplete", "error": {"type": "server_error", "code": code}}
    assert done["output"][0]["status"] == "incomplete"
    assert done["output"][0]["content"][0]["transcript"] == "The first sentence is here."
    assert "second" not in json.dumps(_transcript_deltas(session))
    assert session.conversation.for_request(20) == [_user(), _ai("The first sentence is here.")]


@pytest.mark.parametrize(("answer", "steps", "code"), [
    (None, [_delta(reasoning_content="Thinking."), _event({"error": {"message": "overloaded"}})], "assistant_http_error"),
    (None, [_delta(reasoning_content="Still thinking when"), _delta(finish_reason="length"), DONE],
     "assistant_reply_cut_off"),
    (None, [_delta(reasoning_content="Thinking."), httpx.ReadError("reset")], "assistant_unreachable"),
    (None, [_delta(content="The answer had not fin"), httpx.ReadError("reset")], "assistant_unreachable"),
    (None, [_delta(content="The answer had not fin"), _delta(finish_reason="length"), DONE], "assistant_reply_cut_off"),
    (lambda _request: httpx.Response(500, text="model crashed"), (), "assistant_http_error"),
], ids=["error-event-while-thinking", "cut-off-while-thinking", "dropped-while-thinking",
        "dropped-mid-first-sentence", "cut-off-mid-first-sentence", "http-500"])
async def test_an_error_before_anything_was_spoken_is_handled_as_before(assistant_env, answer, steps, code):
    """Before any of the answer was spoken, an error is the error
    event, the spoken sentence and a failed response.done, as for a whole reply; the history keeps the question."""
    from src.assistant import ERROR_SENTENCES

    server = StreamServer(steps, answer=answer)
    client, router = _client(server), _Router()
    session = await _session(client, router=router)
    try:
        await _ask(session)
        await _until(lambda: session.ws.of_type("response.done"), 5.0)
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert server.bodies[0]["stream"] is True
    assert [event["error"]["code"] for event in session.ws.of_type("error")] == [code]
    assert router.texts == [ERROR_SENTENCES[code]]
    assert _transcript_deltas(session) == []
    done = session.ws.of_type("response.done")[-1]["response"]
    assert done["status"] == "failed"
    assert done["status_details"] == {"type": "failed", "error": {"type": "server_error", "code": code}}
    assert session.conversation.for_request(20) == [_user()]


# ── Servers that don't stream, and the Stream replies switch ─────────────────


def _whole(content=ANSWER, thinking=None, finish_reason="stop"):
    message = {"role": "assistant", "content": content, **({"reasoning_content": thinking} if thinking else {})}
    return lambda _request: httpx.Response(
        200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})


async def test_a_plain_json_answer_to_a_streamed_request_is_spoken_as_one_piece(assistant_env):
    """A server that can't stream still works."""
    server = StreamServer(answer=_whole(thinking=THINKING))
    client, router = _client(server), _Router()
    session = await _session(client, router=router)
    try:
        await _ask(session)
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert server.bodies[0]["stream"] is True
    assert router.texts == ["It's sunny today.", "Bring sunglasses."]
    assert "".join(event["delta"] for event in _transcript_deltas(session)) == ANSWER
    assert [event["text"] for event in session.ws.of_type("open_speech.assistant.thinking")] == [THINKING]
    done = session.ws.of_type("response.done")[-1]["response"]
    assert (done["status"], done["output"][0]["content"][0]["transcript"]) == ("completed", ANSWER)
    assert session.conversation.for_request(20) == [_user(), _ai(ANSWER)]


@pytest.mark.parametrize("answer", ["It is five o'clock.", "Sure. It is five o'clock."], ids=["one", "two"])
async def test_a_whole_plain_json_answer_cut_off_by_the_token_limit_is_spoken_whole(assistant_env, answer):
    """A plain JSON answer to a streamed request arrived whole, so a finish_reason of
    "length" doesn't make it a stream cut off mid-answer. It is spoken in full and response.done is completed, as
    Stream replies off does with the same answer (docs/AI-ASSISTANT.md "Talking to it": "it works either way").
    Before, its last sentence was never flushed: a one-sentence answer failed with assistant_reply_cut_off and the error sentence."""
    from src.assistant import ERROR_SENTENCES

    server = StreamServer(answer=_whole(answer, finish_reason="length"))
    client, router = _client(server), _Router()
    session = await _session(client, router=router)
    try:
        await _ask(session)
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert server.bodies[0]["stream"] is True
    assert session.ws.of_type("error") == []
    assert not set(ERROR_SENTENCES.values()) & set(router.texts)
    assert " ".join(router.texts) == answer
    assert "".join(event["delta"] for event in _transcript_deltas(session)) == answer
    done = session.ws.of_type("response.done")[-1]["response"]
    assert (done["status"], done["output"][0]["content"][0]["transcript"]) == ("completed", answer)
    assert "status_details" not in done
    assert session.conversation.for_request(20) == [_user(), _ai(answer)]


_MINUTES = (
    "The committee met on Tuesday to discuss the budget. Several members raised concerns about the cost of the "
    "new building, which had risen by almost a third since the spring. After a long debate, they agreed to ask "
    "the architects for a cheaper design, and to meet again in two weeks.\n\n"
)


@pytest.mark.parametrize("how", ["whole", "streamed"])
async def test_a_long_answer_in_one_piece_is_cut_into_sentences_off_the_event_loop(assistant_env, how):
    """A whole JSON answer (or one large streamed piece) comes to on_piece in one
    piece, which "is called on the event loop ... and must not block" (AssistantClient.stream_reply). Cutting 400,000
    characters into sentences there stalled every session for 3.5 s. Now the loop keeps ticking, every sentence is
    handed over in order (a piece that arrives meanwhile after them), and the turn's end flushes the rest."""
    from dataclasses import replace

    from src.assistant.presets import EffectiveSettings
    from src.live_tts.segmenter import LiveTextSegmenter
    from src.realtime.server import _StreamedTurn, _TurnTiming

    long_answer = (_MINUTES * (400_000 // len(_MINUTES) + 1))[:400_000].rstrip()
    if how == "whole":
        answer = long_answer
        server = StreamServer(answer=_whole(long_answer))
    else:
        answer = long_answer + " The last words."
        server = StreamServer([_delta(content=long_answer), 0.05, _delta(content=" The last words."),
                               _delta(finish_reason="stop"), DONE])
    client = _client(server)
    ai = replace(EffectiveSettings(model="test-model"), timeout_s=120)
    turn = _StreamedTurn(_TurnTiming())
    gaps: list[float] = []
    stop = asyncio.Event()

    async def ticker():
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.05)
        reply = await asyncio.wait_for(client.stream_reply([_user()], ai, turn.on_piece), 30)
        await asyncio.wait_for(turn.end(reply), 30)
        await asyncio.sleep(0.05)
    finally:
        stop.set()
        await tick
        await client.aclose()

    handed = []
    while (item := turn.sentences.get_nowait()) is not None:
        handed.append(item)
    reference = LiveTextSegmenter(mode="natural", max_chars=assistant_env.tts_live_sentence_max_chars,
                                  max_words=assistant_env.tts_live_sentence_max_words)
    expected = [segment.text for segment in reference.append(answer) + reference.flush() if segment.text]
    assert reply.streamed is (how == "streamed") and turn.failure is None
    assert len(expected) > 1000
    assert [text for text, _covers in handed if text] == expected
    assert "".join(covers for _text, covers in handed) == answer
    assert turn.kept() == answer
    assert max(gaps) < 0.3, f"the event loop stalled {max(gaps):.2f} s while a {len(answer):,}-char answer was cut"


async def test_a_server_that_refuses_include_usage_still_answers_every_turn(assistant_env):
    """A 400 to stream_options gets one retry without it, and
    the next turn asks without it at once."""

    def answer(request):
        if "stream_options" in json.loads(request.content):
            return httpx.Response(400)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SseStream(
            [_delta(content="Fine, thanks."), _delta(finish_reason="stop"), DONE]))

    server = StreamServer(answer=answer)
    client, router = _client(server), _Router()
    session = await _session(client, router=router)
    try:
        for _ in range(2):
            await _ask(session)
            await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert ["stream_options" in body for body in server.bodies] == [True, False, False]
    assert [e["response"]["status"] for e in session.ws.of_type("response.done")] == ["completed", "completed"]
    assert router.texts == ["Fine, thanks.", "Fine, thanks."]


async def test_os_assistant_stream_false_asks_for_the_whole_reply_as_before(assistant_env, monkeypatch):
    """Stream replies off (OS_ASSISTANT_STREAM=false) sends
    stream: false, and the reply's whole text goes out before its audio, as before."""
    from src.assistant import connection
    from src.assistant.client import AssistantClient

    monkeypatch.setattr(assistant_env, "os_assistant_url", AI_URL)
    monkeypatch.setattr(assistant_env, "os_assistant_connection_locked", True)
    monkeypatch.setattr(assistant_env, "os_assistant_stream", False)
    server = StreamServer(answer=_whole())
    client = AssistantClient.from_connection(connection.effective(assistant_env), transport=httpx.MockTransport(server))
    router = _Router()
    session = await _session(client, router=router)
    try:
        await _ask(session)
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert client.stream is False
    assert server.bodies[0]["stream"] is False and "stream_options" not in server.bodies[0]
    assert [event["delta"] for event in _transcript_deltas(session)] == [ANSWER]
    assert router.texts == ["It's sunny today.", "Bring sunglasses."]
    assert session.ws.of_type("response.done")[-1]["response"]["status"] == "completed"


# ── One timing line per voice turn ───────────────────────────────────────────

_TIMING_KEYS = ["silence_ms", "stt_ms", "ai_first_token_ms", "ai_first_answer_ms", "ai_done_ms", "first_audio_ms",
                "reply_chars", "tokens_in", "tokens_out", "streamed"]


def _timing_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name == "src.realtime.server" and record.getMessage().startswith("Realtime turn")]


def _fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", line))


async def test_each_voice_turn_logs_one_line_of_its_timings_in_numbers_only(assistant_env, caplog):
    """One INFO line per AI voice turn with the end-of-turn silence,
    the transcription, the AI's first token, first answer token and end, the first audio, the reply's length, the
    token counts and whether it streamed. Numbers only: no word said, answered or thought, and never the key."""
    caplog.set_level(logging.INFO)
    server = StreamServer([
        _delta(reasoning_content="The user asks"), 0.05, _delta(reasoning_content=" about the weather."), 0.05,
        _delta(content="It's sunny today."), _delta(content=" Bring"), 0.3,
        _delta(content=" sunglasses."), _delta(finish_reason="stop"), _usage(31, 12), DONE,
    ])
    client, router = _client(server, api_key=API_KEY), _Router()
    session = await _session(client, router=router)
    try:
        await _say(session, QUESTION, stt_seconds=0.05)
        await _ask(session, "And tomorrow?")
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    lines = _timing_lines(caplog)
    assert len(lines) == 2, lines
    spoken, typed = (_fields(line) for line in lines)
    for fields in (spoken, typed):
        assert list(fields) == _TIMING_KEYS
        assert all(re.fullmatch(r"\d+|-", value) for value in fields.values()), fields
        first_token, first_answer, done, audio = (int(fields[key]) for key in (
            "ai_first_token_ms", "ai_first_answer_ms", "ai_done_ms", "first_audio_ms"))
        assert first_token + 30 <= first_answer <= done, fields
        assert first_answer <= audio < done, "the first audio went out before the AI was done"
        assert (fields["reply_chars"], fields["tokens_in"], fields["tokens_out"], fields["streamed"]) == (
            str(len(ANSWER)), "31", "12", "1")
    assert spoken["silence_ms"] == "64" and int(spoken["stt_ms"]) >= 40
    assert (typed["silence_ms"], typed["stt_ms"]) == ("-", "-")
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for secret in (API_KEY, "weather", "sunny", "sunglasses", "tomorrow"):
        assert secret not in "\n".join(lines)
    assert API_KEY not in logged


async def test_the_timing_line_says_when_the_reply_did_not_stream_and_what_was_not_reached(assistant_env, caplog):
    """streamed=0 for a whole JSON answer (where the first token is
    the end), and "-" for what a turn never reached, such as the answer of a turn that failed."""
    caplog.set_level(logging.INFO)
    answers = iter([_whole(), lambda _request: httpx.Response(503)])
    server = StreamServer(answer=lambda request: next(answers)(request))
    client, router = _client(server), _Router()
    session = await _session(client, router=router)
    try:
        await _ask(session)
        await _finish(session)
        await _ask(session, "Again?")
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    whole, failed = (_fields(line) for line in _timing_lines(caplog))
    assert whole["streamed"] == "0" and whole["reply_chars"] == str(len(ANSWER))
    assert whole["ai_first_token_ms"] == whole["ai_first_answer_ms"] == whole["ai_done_ms"] != "-"
    assert (whole["tokens_in"], whole["tokens_out"]) == ("-", "-")
    assert (failed["ai_first_token_ms"], failed["ai_first_answer_ms"], failed["reply_chars"]) == ("-", "-", "0")
    assert failed["ai_done_ms"] != "-" and failed["first_audio_ms"] != "-", "the error sentence was spoken"


# ── Sound tags reach the voice whole ─────────


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "whole"])
async def test_a_sound_tag_in_the_answer_reaches_the_voice_whole(assistant_env, monkeypatch, tmp_path, stream):
    """"[clear throat]" and "[laugh]" (Chatterbox Turbo reads each as one token)
    reach the voice model as written, through the sentence caps and the pronunciation dictionary, whether the
    answer streams or comes whole. Before, a word cap ending on "[clear" cut the tag in two."""
    from src.pronunciation.dictionary import PronunciationDictionary

    monkeypatch.setattr(assistant_env, "tts_live_sentence_max_words", 6)
    path = tmp_path / "dictionary.json"
    path.write_text('{"Done": "Done now"}', encoding="utf-8")
    answer = "One two three four five [clear throat] and on. [laugh] Done."
    if stream:
        steps = [_delta(role="assistant", content="")]
        for piece in ["One two three four five [", "clear", " throat] and on. [", "laugh] Done."]:
            steps += [_delta(content=piece), 0.01]
        server = StreamServer([*steps, _delta(finish_reason="stop"), _usage(5, 5), DONE])
    else:
        server = StreamServer(answer=_whole(answer))
    client, router = _client(server, stream=stream), _Router()
    session = await _session(client, router=router)
    session.pronounce_fn = PronunciationDictionary(str(path)).apply
    try:
        await _ask(session)
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    spoken = " ".join(router.texts)
    assert spoken.count("[clear throat]") == 1 and spoken.count("[laugh]") == 1, router.texts
    assert not any(("clear" in text) != ("[clear throat]" in text) for text in router.texts), router.texts
    assert spoken == "One two three four five [clear throat] and on. [laugh] Done now."
    assert session.ws.of_type("response.done")[-1]["response"]["status"] == "completed"
