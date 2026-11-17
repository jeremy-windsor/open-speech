"""Text-only replies (output_modalities ["text"]): the AI's answer streams out as text, with no voice."""

from __future__ import annotations

import asyncio

import pytest

from src.assistant import AssistantError
from src.assistant.client import StreamedReply
from tests.test_realtime import (
    _assistant_session,
    _FakeAssistant,
    _reply,
    _say,
    _TextRecordingRouter,
    assistant_env,  # noqa: F401 (fixture)
)


class _StreamingAssistant:
    """Streams its scripted pieces through on_piece, then returns the whole reply."""

    stream = True

    def __init__(self, pieces, *, finish_reason="stop", gate=None):
        self.pieces = pieces
        self.finish_reason = finish_reason
        self.gate = gate
        self.calls = []

    async def stream_reply(self, messages, settings, on_piece):
        self.calls.append(messages)
        for kind, text in self.pieces:
            on_piece(kind, text)
            await asyncio.sleep(0)
        if self.gate is not None:
            await self.gate.wait()
        answer = "".join(text for kind, text in self.pieces if kind == "answer").strip()
        thinking = "".join(text for kind, text in self.pieces if kind == "thinking") or None
        return StreamedReply(text=answer, thinking=thinking, finish_reason=self.finish_reason)


@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.asyncio
async def test_a_text_session_streams_the_answer_as_text_and_speaks_nothing(assistant_env, ga):  # noqa: F811
    fake = _StreamingAssistant([("thinking", "They greet me."), ("answer", "Hello "), ("answer", "there.")])
    router = _TextRecordingRouter()
    modalities = {"output_modalities": ["text"]} if ga else {"modalities": ["text"]}
    session = await _assistant_session(fake, router=router, ga=ga, update=modalities)
    try:
        await _say(session, "Hi")
        await _say(session, "And again")
    finally:
        await session.close()

    prefix = "response.output_text" if ga else "response.text"
    deltas = [e["delta"] for e in session.ws.of_type(f"{prefix}.delta")]
    assert deltas[:2] == ["Hello ", "there."]
    assert session.ws.of_type(f"{prefix}.done")[0]["text"] == "Hello there."
    assert router.requests == [], "no voice was made"
    assert not session.ws.of_type("response.output_audio.delta") and not session.ws.of_type("response.audio.delta")
    sent = session.ws.sent
    thinking = next(i for i, e in enumerate(sent) if e["type"] == "open_speech.assistant.thinking")
    first_delta = next(i for i, e in enumerate(sent) if e["type"] == f"{prefix}.delta")
    assert thinking < first_delta, "the thinking comes before the answer"
    done = session.ws.of_type("response.done")[0]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"] == [{"type": "output_text" if ga else "text", "text": "Hello there."}]
    shown = session.ws.of_type("session.updated")[-1]["session"]
    assert (shown["output_modalities"] if ga else shown["modalities"]) == ["text"]
    second = fake.calls[1]
    assert {"role": "assistant", "content": "Hello there."} in second, "the text answer joins the conversation"


@pytest.mark.asyncio
async def test_a_whole_reply_without_streaming_is_one_delta_and_errors_fail_the_response(assistant_env):  # noqa: F811
    fake = _FakeAssistant(_reply("It's sunny."), AssistantError("The AI server answered HTTP 500", "assistant_http_error"))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, update={"output_modalities": ["text"]})
    try:
        await _say(session, "Weather?")
        await _say(session, "Tomorrow?")
    finally:
        await session.close()

    assert [e["delta"] for e in session.ws.of_type("response.output_text.delta")] == ["It's sunny."]
    statuses = [e["response"]["status"] for e in session.ws.of_type("response.done")]
    assert statuses == ["completed", "failed"]
    assert any(e["error"]["code"] == "assistant_http_error" for e in session.ws.of_type("error"))
    assert router.requests == [], "no error sentence is spoken either"


@pytest.mark.asyncio
async def test_a_text_answer_cut_by_the_token_limit_is_incomplete(assistant_env):  # noqa: F811
    fake = _StreamingAssistant([("answer", "Part of it")], finish_reason="length")
    session = await _assistant_session(fake, update={"output_modalities": ["text"]})
    try:
        await _say(session, "Tell me everything")
    finally:
        await session.close()

    done = session.ws.of_type("response.done")[0]["response"]
    assert done["status"] == "incomplete"
    assert done["status_details"]["error"]["code"] == "assistant_reply_cut_off"


@pytest.mark.asyncio
async def test_a_text_answer_can_be_cancelled(assistant_env):  # noqa: F811
    gate = asyncio.Event()
    fake = _StreamingAssistant([("answer", "Thinking about")], gate=gate)
    session = await _assistant_session(fake, update={"output_modalities": ["text"]})
    try:
        from unittest.mock import patch

        from tests.test_realtime import _append, _identity_decode

        with _identity_decode(), patch("src.realtime.server.stt_router.transcribe", return_value={"text": "Go"}):
            await session.handle_event(_append(b"\x01\x00" * (5 * 512)))
        for _ in range(50):
            if session.ws.of_type("response.output_text.delta"):
                break
            await asyncio.sleep(0.01)
        await session.handle_event({"type": "response.cancel"})
    finally:
        gate.set()
        await session.close()

    done = session.ws.of_type("response.done")[0]["response"]
    assert done["status"] == "cancelled"


@pytest.mark.asyncio
async def test_voice_replies_stay_the_default_and_a_session_can_switch_back(assistant_env):  # noqa: F811
    fake = _FakeAssistant(_reply("Spoken."), _reply("Written."), _reply("Spoken again."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router)
    try:
        assert session.ws.of_type("session.updated")[-1]["session"]["output_modalities"] == ["audio"]
        await _say(session, "One")
        await session.handle_event({"type": "session.update", "session": {"output_modalities": ["text"]}})
        await _say(session, "Two")
        await session.handle_event({"type": "session.update", "session": {"output_modalities": ["audio"]}})
        await _say(session, "Three")
        await session.handle_event({"type": "session.update", "event_id": "bad", "session": {"output_modalities": ["video"]}})
    finally:
        await session.close()

    assert [e["delta"] for e in session.ws.of_type("response.output_text.delta")] == ["Written."]
    assert len(router.requests) == 2, "the first and third replies are spoken"
    assert any(e.get("error", {}).get("event_id") == "bad" for e in session.ws.of_type("error"))
