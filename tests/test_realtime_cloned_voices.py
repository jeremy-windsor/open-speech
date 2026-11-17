"""Cloned voices in realtime: a named voice whose version clones a recording speaks with that recording."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.test_realtime import (
    _assistant_session,
    _FakeAssistant,
    _reply,
    _say,
    _TextRecordingRouter,
    assistant_env,  # noqa: F401 (fixture)
)


class _Library:
    def __init__(self):
        self.reads = []

    def get(self, name):
        self.reads.append(name)
        if name != "will_clone":
            raise KeyError(name)
        return b"RIFF-will", {"transcript": "Hi, it's me."}


@pytest.mark.asyncio
async def test_a_cloned_named_voice_speaks_with_its_recording_read_once(assistant_env):  # noqa: F811
    fake = _FakeAssistant(_reply("First answer."), _reply("Second answer."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router)
    session.voice_library = library = _Library()
    resolve = patch("src.realtime.server.resolve_named_voice", return_value=("will_clone", "will_clone"))
    try:
        with resolve:
            await session.handle_event({"type": "session.update", "session": {
                "model": "breeze/tts-2", "audio": {"output": {"voice": "voice:will"}},
            }})
            await _say(session, "Hello")
            await _say(session, "Again")
    finally:
        await session.close()

    spoken = [r for r in router.requests if r.get("model") == "breeze/tts-2"]
    assert spoken, router.requests
    assert all(r["reference_audio"] == b"RIFF-will" and r["clone_transcript"] == "Hi, it's me." for r in spoken)
    assert all(r["voice"] == "will_clone" for r in spoken)
    assert library.reads == ["will_clone"], "the recording is read once per session"
    assert not [e for e in session.ws.of_type("error") if "realtime yet" in e["error"]["message"]]


@pytest.mark.asyncio
async def test_a_cloned_voice_whose_recording_is_gone_fails_the_response_with_a_reason(assistant_env):  # noqa: F811
    fake = _FakeAssistant(_reply("Unused."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router)
    session.voice_library = _Library()
    resolve = patch("src.realtime.server.resolve_named_voice", return_value=("gone", "gone"))
    try:
        with resolve:
            await session.handle_event({"type": "session.update", "session": {
                "model": "breeze/tts-2", "audio": {"output": {"voice": "voice:old"}},
            }})
            await _say(session, "Hello")
    finally:
        await session.close()

    messages = [e["error"]["message"] for e in session.ws.of_type("error")]
    assert any("'gone'" in m and "unavailable" in m for m in messages), messages
    assert not [r for r in router.requests if r.get("model") == "breeze/tts-2"]
