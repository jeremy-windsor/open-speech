"""The AI client's streamed reply, and the Stream replies connection setting (OS_ASSISTANT_STREAM).

A voice turn asks the AI with ``stream: true`` and reads the server-sent events as they arrive, so the voice can
start on the first sentence. The rules: OpenAI's Chat Completions streaming format, docs/AI-ASSISTANT.md
"Talking to it" and "Thinking controls", and the connection API in docs/AI-ASSISTANT.md "The connection".

No network: every request goes to an httpx.MockTransport, at the made-up host below. Nothing here can hang: every
slow stream sleeps a fraction of a second, a stream that waits on an event is bounded by ``asyncio.wait_for``, and
pyproject's faulthandler_timeout bounds the run.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import Settings

AI_URL = "http://ai.stream.test:8080/v1"
API_KEY = "sk-stream-secret-789"
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
DONE = b"data: [DONE]\n\n"


def _event(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _delta(finish_reason=None, **delta) -> bytes:
    """One chunk of an OpenAI-style stream: a piece of the thinking or the answer in ``delta``."""
    return _event({"id": "c1", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})


def _usage(prompt_tokens=31, completion_tokens=12) -> bytes:
    """The last chunk, with stream_options.include_usage: no choices, the token counts."""
    return _event({"id": "c1", "object": "chat.completion.chunk", "choices": [],
                   "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens}})


class SseStream(httpx.AsyncByteStream):
    """A response body that plays its steps in order: bytes are sent, a number is a pause in seconds, an
    asyncio.Event is waited on, and an exception is raised (as a dropped connection does). It records what it sent,
    when, and whether the client closed it."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.sent: list[bytes] = []
        self.sent_at: list[float] = []
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
                self.sent.append(step)
                self.sent_at.append(asyncio.get_running_loop().time())
                yield step
        self.finished = True

    async def aclose(self):
        self.closed = True


class StreamServer:
    """A MockTransport handler: records each request and answers it with a new SseStream of ``steps`` (or with
    what ``answer(request)`` returns)."""

    def __init__(self, steps=(), answer=None):
        self.steps = list(steps)
        self.answer = answer
        self.requests: list[httpx.Request] = []
        self.streams: list[SseStream] = []

    def __call__(self, request: httpx.Request):
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


def _settings(**changes):
    from src.assistant.presets import EffectiveSettings

    return replace(EffectiveSettings(model="qwen3-30b"), **changes)


class Pieces:
    """The on_piece callback: each (kind, text) and the loop time it came."""

    def __init__(self):
        self.items: list[tuple[str, str]] = []
        self.at: list[float] = []

    def __call__(self, kind, text):
        self.items.append((kind, text))
        self.at.append(asyncio.get_running_loop().time())

    def text(self, kind) -> str:
        return "".join(text for k, text in self.items if k == kind)


async def _stream(server, settings=None, pieces=None, **client_kwargs):
    client = _client(server, **client_kwargs)
    try:
        return await client.stream_reply(MESSAGES, settings or _settings(), pieces if pieces is not None else Pieces())
    finally:
        await client.aclose()


THINK_THEN_ANSWER = [
    _delta(role="assistant", content=""),
    _delta(reasoning_content="The user greets me."), 0.05,
    _delta(reasoning_content=" I greet back."), 0.05,
    _delta(content="Hello"), 0.05,
    _delta(content=" there."), 0.05,
    _delta(content=" How are you?"), 0.05,
    _delta(finish_reason="stop"),
    _usage(31, 12),
    DONE,
]


class TestStreamedReply:
    async def test_the_request_is_the_replys_body_with_stream_true_and_include_usage(self):
        """The same body reply() sends (one builder), with stream and include_usage."""
        server = StreamServer(THINK_THEN_ANSWER)
        settings = _settings(reasoning_effort="low", max_tokens=256, thinking="off")
        client = _client(server, extra_body={"top_k": 20})
        try:
            await client.stream_reply(MESSAGES, settings, Pieces())
            expected = {**client.build_body(MESSAGES, settings), "stream": True,
                        "stream_options": {"include_usage": True}}
        finally:
            await client.aclose()
        request = server.requests[0]
        assert (request.method, str(request.url)) == ("POST", f"{AI_URL}/chat/completions")
        assert server.bodies == [expected]
        assert expected["top_k"] == 20 and expected["reasoning_effort"] == "low"

    async def test_the_thinking_then_the_answer_arrive_piece_by_piece_as_they_are_read(self):
        """reasoning_content is the thinking, content the answer, each handed on as it is
        read: the first answer piece comes before the server has sent its last piece."""
        server = StreamServer(THINK_THEN_ANSWER)
        pieces = Pieces()
        reply = await _stream(server, pieces=pieces)

        assert pieces.items == [
            ("thinking", "The user greets me."), ("thinking", " I greet back."),
            ("answer", "Hello"), ("answer", " there."), ("answer", " How are you?"),
        ]
        first_answer_at = pieces.at[pieces.items.index(("answer", "Hello"))]
        assert first_answer_at < server.streams[0].sent_at[-1], "the answer is handed on as it arrives"
        assert (reply.text, reply.thinking, reply.finish_reason) == (
            "Hello there. How are you?", "The user greets me. I greet back.", "stop",
        )
        assert reply.streamed is True
        assert dict(reply.usage) == {"prompt_tokens": 31, "completion_tokens": 12}

    @pytest.mark.parametrize("steps", [
        [_delta(content="<thi"), _delta(content="nk>Let me"), _delta(content=" see.</th"),
         _delta(content="ink>\n\n"), _delta(content="The "), _delta(content="answer."), _delta(finish_reason="stop"), DONE],
        [_delta(content="\n<think>Let me see.</think>The answer."), _delta(finish_reason="stop"), DONE],
        [_delta(reasoning="Let me see."), _delta(content="  The answer."), _delta(finish_reason="stop"), DONE],
    ], ids=["think-tags-split-across-pieces", "think-tags-in-one-piece", "reasoning-field"])
    async def test_a_leading_think_block_and_the_reasoning_field_are_thinking_never_answer(self, steps):
        """A leading <think>…</think> in content (even cut across pieces) is thinking;
        the answer's pieces start at its first character, without the white space before it."""
        pieces = Pieces()
        reply = await _stream(StreamServer(steps), pieces=pieces)
        assert pieces.text("thinking") == "Let me see."
        assert pieces.text("answer") == "The answer."
        assert "<think>" not in pieces.text("answer") and "</think>" not in pieces.text("answer")
        assert (reply.text, reply.thinking) == ("The answer.", "Let me see.")

    async def test_a_plain_json_answer_to_a_streamed_request_is_read_as_one_piece(self):
        """A server that can't stream answers with one JSON reply."""
        answer = {"choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Hello there.", "reasoning_content": "Hmm."}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
        server = StreamServer(answer=lambda _request: httpx.Response(200, json=answer))
        pieces = Pieces()
        reply = await _stream(server, pieces=pieces)
        assert pieces.items == [("thinking", "Hmm."), ("answer", "Hello there.")]
        assert (reply.text, reply.thinking, reply.finish_reason, reply.streamed) == ("Hello there.", "Hmm.", "stop", False)
        assert dict(reply.usage) == {"prompt_tokens": 9, "completion_tokens": 4}
        assert server.bodies[0]["stream"] is True

    async def test_a_server_that_refuses_include_usage_gets_one_retry_without_it_and_isnt_asked_again(self):
        """A 400 to stream_options gets one retry without it,
        and the client remembers, so the next reply asks without it at once."""

        def answer(request):
            if "stream_options" in json.loads(request.content):
                return httpx.Response(400, text="unknown field stream_options")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=SseStream([_delta(content="Hi."), _delta(finish_reason="stop"), DONE]))

        server = StreamServer(answer=answer)
        client = _client(server)
        try:
            first = await client.stream_reply(MESSAGES, _settings(), Pieces())
            second = await client.stream_reply(MESSAGES, _settings(), Pieces())
        finally:
            await client.aclose()
        assert (first.text, second.text) == ("Hi.", "Hi.")
        assert ["stream_options" in body for body in server.bodies] == [True, False, False]
        assert all(body["stream"] is True for body in server.bodies)
        assert dict(first.usage) == {}

    async def test_a_400_that_isnt_about_include_usage_is_an_http_error_and_isnt_remembered(self):
        from src.assistant.errors import AssistantError

        server = StreamServer(answer=lambda _request: httpx.Response(400, text=f"bad request {API_KEY}"))
        client = _client(server, api_key=API_KEY)
        try:
            for _ in range(2):
                with pytest.raises(AssistantError) as caught:
                    await client.stream_reply(MESSAGES, _settings(), Pieces())
                assert caught.value.code == "assistant_http_error"
                assert caught.value.message == "The AI server answered HTTP 400"
        finally:
            await client.aclose()
        assert ["stream_options" in body for body in server.bodies] == [True, False, True, False]

    async def test_finish_reason_length_in_the_middle_of_the_answer_returns_the_cut_answer(self):
        """max_tokens cut it mid-sentence; the caller sees finish_reason "length"."""
        steps = [_delta(reasoning_content="Thinking."), _delta(content="First sentence. Second sen"),
                 _delta(finish_reason="length"), DONE]
        reply = await _stream(StreamServer(steps))
        assert (reply.text, reply.finish_reason) == ("First sentence. Second sen", "length")

    @pytest.mark.parametrize(("steps", "code"), [
        ([_delta(reasoning_content="Still thinking when the limit"), _delta(finish_reason="length"), DONE],
         "assistant_reply_cut_off"),
        ([_delta(content="<think>Still thinking"), _delta(finish_reason="length"), DONE], "assistant_reply_cut_off"),
        ([_delta(reasoning_content="All thinking."), _delta(finish_reason="stop"), DONE], "assistant_empty_reply"),
        ([_delta(content="   "), _delta(finish_reason="stop"), DONE], "assistant_empty_reply"),
        ([b"data: {not json\n\n"], "assistant_bad_response"),
        ([_event(["not", "an", "object"])], "assistant_bad_response"),
        ([_delta(content=7), DONE], "assistant_bad_response"),
        ([_delta(content="Half an ans")], "assistant_bad_response"),  # ended with no finish_reason and no [DONE]
        ([b"data: " + b"\xff\xfe" + b"\n\n"], "assistant_bad_response"),
    ], ids=["cut-off-while-thinking", "cut-off-in-think-tags", "only-thinking", "blank", "not-json",
            "not-an-object", "content-not-text", "ended-early", "not-utf8"])
    async def test_each_error_code_from_its_cause_in_a_stream(self, steps, code):
        from src.assistant.errors import AssistantError

        with pytest.raises(AssistantError) as caught:
            await _stream(StreamServer(steps))
        assert caught.value.code == code

    @pytest.mark.parametrize(("failure", "code"), [
        (httpx.ReadError("connection reset"), "assistant_unreachable"),
        (httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
         "assistant_unreachable"),
        (httpx.ReadTimeout("timed out"), "assistant_timeout"),
    ], ids=["reset", "closed-early", "read-timeout"])
    async def test_an_error_in_the_middle_of_the_stream_raises_after_the_pieces_already_handed_on(self, failure, code):
        """What arrived before the failure was handed on; then the error, as reply()'s."""
        from src.assistant.errors import AssistantError

        steps = [_delta(reasoning_content="Hmm."), _delta(content="The first sentence."), _delta(content=" The sec"),
                 failure]
        pieces = Pieces()
        with pytest.raises(AssistantError) as caught:
            await _stream(StreamServer(steps), pieces=pieces)
        assert caught.value.code == code
        assert pieces.text("answer") == "The first sentence. The sec"

    async def test_an_error_event_in_the_stream_is_an_http_error_that_never_echoes_the_server(self):
        from src.assistant.errors import AssistantError

        steps = [_delta(content="Partial"), _event({"error": {"message": f"model crashed {API_KEY}", "code": 500}})]
        with pytest.raises(AssistantError) as caught:
            await _stream(StreamServer(steps), api_key=API_KEY)
        assert caught.value.code == "assistant_http_error"
        assert "crashed" not in caught.value.message and API_KEY not in str(caught.value)

    async def test_the_api_key_never_appears_in_a_streams_transport_error(self):
        from src.assistant.errors import AssistantError

        steps = [_delta(content="Partial"), httpx.ReadError(f"reset while sending Bearer {API_KEY}")]
        with pytest.raises(AssistantError) as caught:
            await _stream(StreamServer(steps), api_key=API_KEY)
        assert caught.value.code == "assistant_unreachable"
        assert API_KEY not in caught.value.message and "[redacted]" in caught.value.message

    async def test_the_whole_stream_must_arrive_within_timeout_s(self):
        """The same deadline as reply(): the whole reply within timeout_s."""
        from src.assistant.errors import AssistantError

        server = StreamServer([_delta(content="Slow"), 1.0, _delta(content=" answer."), _delta(finish_reason="stop"), DONE])
        started = asyncio.get_running_loop().time()
        pieces = Pieces()
        with pytest.raises(AssistantError) as caught:
            await _stream(server, _settings(timeout_s=0.2), pieces=pieces)
        assert caught.value.code == "assistant_timeout"
        assert asyncio.get_running_loop().time() - started < 0.9
        assert pieces.items == [("answer", "Slow")]
        assert server.streams[0].closed

    async def test_a_stream_larger_than_the_cap_is_a_bad_response(self, monkeypatch):
        from src.assistant import client as client_module
        from src.assistant.errors import AssistantError

        monkeypatch.setattr(client_module, "MAX_REPLY_BYTES", 2048)
        steps = [_delta(content="word ") for _ in range(100)] + [_delta(finish_reason="stop"), DONE]
        with pytest.raises(AssistantError) as caught:
            await _stream(StreamServer(steps))
        assert caught.value.code == "assistant_bad_response"

    async def test_a_cancel_closes_the_stream_so_the_server_can_stop(self):
        """Cancelling the caller closes the response at once."""
        never = asyncio.Event()
        server = StreamServer([_delta(content="The first sentence. "), never, _delta(content="Never sent."), DONE])
        pieces = Pieces()
        client = _client(server)
        try:
            task = asyncio.create_task(client.stream_reply(MESSAGES, _settings(), pieces))
            for _ in range(200):
                if pieces.items:
                    break
                await asyncio.sleep(0.01)
            assert pieces.items == [("answer", "The first sentence. ")]
            assert client.busy
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert server.streams[0].closed, "the stream is closed"
            assert not server.streams[0].finished and len(server.streams[0].sent) == 1
            assert not client.busy
        finally:
            await client.aclose()

    async def test_a_retired_client_streams_through_the_current_shared_one(self, monkeypatch):
        from src import assistant

        old_server, new_server = StreamServer(THINK_THEN_ANSWER), StreamServer(THINK_THEN_ANSWER)
        old, new = _client(old_server), _client(new_server, stream=False)
        monkeypatch.setattr(assistant, "_client", new)
        try:
            assert old.stream is True
            old.retire()
            assert old.stream is False, "a retired client follows the current one's setting"
            reply = await old.stream_reply(MESSAGES, _settings(), Pieces())
        finally:
            await old.aclose()
            await new.aclose()
        assert reply.text == "Hello there. How are you?"
        assert (len(old_server.requests), len(new_server.requests)) == (0, 1)


# ── The Stream replies setting: OS_ASSISTANT_STREAM, the saved connection, the API ──


@pytest.fixture
def clean_assistant_env(monkeypatch):
    import os

    for name in list(os.environ):
        if name.upper().startswith("OS_ASSISTANT_"):
            monkeypatch.delenv(name)


@pytest.fixture
def connection_env(monkeypatch, tmp_path):
    """The assistant on from .env, unlocked, with a data folder of its own (where the saved connection goes)."""
    from src.config import settings

    monkeypatch.setattr(settings, "os_assistant_url", AI_URL)
    monkeypatch.setattr(settings, "os_assistant_api_key", "")
    monkeypatch.setattr(settings, "os_assistant_model", "qwen3-30b")
    monkeypatch.setattr(settings, "os_assistant_extra_body", "")
    monkeypatch.setattr(settings, "os_assistant_connection_locked", False)
    monkeypatch.setattr(settings, "os_assistant_stream", True)
    monkeypatch.setattr(settings, "os_studio_db_path", str(tmp_path / "data" / "studio.db"))
    return settings


class TestStreamSetting:
    def test_os_assistant_stream_is_on_by_default_and_false_turns_it_off(self, clean_assistant_env, monkeypatch):
        from src.assistant.client import AssistantClient

        assert Settings(_env_file=None).os_assistant_stream is True
        monkeypatch.setenv("OS_ASSISTANT_URL", AI_URL)
        assert AssistantClient.from_settings(Settings(_env_file=None), transport=httpx.MockTransport(StreamServer())).stream is True
        monkeypatch.setenv("OS_ASSISTANT_STREAM", "false")
        assert Settings(_env_file=None).os_assistant_stream is False
        client = AssistantClient.from_settings(Settings(_env_file=None), transport=httpx.MockTransport(StreamServer()))
        assert client.stream is False

    def test_the_connection_saves_shows_and_forgets_stream(self, connection_env):
        from src.assistant import connection

        current = connection.effective()
        assert (current.stream, current.sources["stream"], current.view()["stream"]) == (True, "default", True)
        saved = connection.save({"stream": False})
        assert (saved.stream, saved.sources["stream"], saved.view()["stream"]) == (False, "saved", False)
        assert connection.ENV_NAMES["stream"] == "OS_ASSISTANT_STREAM"
        assert connection.effective().stream is False, "read back from the file"
        assert connection.save({"stream": None}).stream is True, "null follows .env again"

    def test_the_env_value_is_its_source_until_one_is_saved(self, connection_env, monkeypatch):
        from src.assistant import connection

        monkeypatch.setattr(connection_env, "os_assistant_stream", False)
        current = connection.effective()
        assert (current.stream, current.sources["stream"]) == (False, "env")

    @pytest.mark.parametrize("value", ["false", 0, 1, "yes", [], {}])
    def test_stream_must_be_true_or_false(self, connection_env, value):
        from src.assistant import connection

        with pytest.raises(connection.ConnectionFieldError) as caught:
            connection.save({"stream": value})
        assert caught.value.field == "stream" and "true or false" in caught.value.message
        assert connection.effective().stream is True

    def test_a_locked_connection_ignores_a_saved_stream(self, connection_env, monkeypatch):
        from src.assistant import connection

        connection.save({"stream": False})
        monkeypatch.setattr(connection_env, "os_assistant_connection_locked", True)
        assert (connection.effective().stream, connection.effective().sources["stream"]) == (True, "default")
        with pytest.raises(connection.ConnectionLockedError):
            connection.save({"stream": True})

    def test_get_and_put_api_assistant_connection_show_and_save_stream(self, connection_env, monkeypatch):
        """GET /api/assistant/connection shows stream and its source; PUT saves it and
        the rebuilt shared client follows it. No request reaches an AI server (the built client's transport fails
        any request)."""
        from src import assistant
        from src.assistant.client import AssistantClient
        from src.routes.assistant import create_router

        def no_request(request):
            raise AssertionError(f"no AI request expected, got {request.method} {request.url}")

        monkeypatch.setattr(assistant, "build_client", lambda conn: AssistantClient.from_connection(
            conn, transport=httpx.MockTransport(no_request)))
        app = FastAPI()
        app.include_router(create_router(get_settings=lambda: connection_env))
        try:
            with TestClient(app) as http:
                shown = http.get("/api/assistant/connection").json()
                assert (shown["stream"], shown["sources"]["stream"]) == (True, "default")
                saved = http.put("/api/assistant/connection", json={"stream": False})
                assert saved.status_code == 200, saved.text
                assert (saved.json()["stream"], saved.json()["sources"]["stream"]) == (False, "saved")
                assert assistant.get_client().stream is False
                refused = http.put("/api/assistant/connection", json={"stream": "no"})
                assert refused.status_code == 422
                assert refused.json()["error"]["param"] == "stream"
                status = http.get("/api/assistant").json()
                assert status["sources"]["OS_ASSISTANT_STREAM"] == "saved"
        finally:
            assistant.set_client(None)
