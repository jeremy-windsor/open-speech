from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.client import OpenSpeechClient


def test_headers_with_api_key():
    c = OpenSpeechClient(api_key="k")
    assert c._headers()["Authorization"] == "Bearer k"


def test_headers_without_api_key():
    c = OpenSpeechClient()
    assert c._headers() == {}


def test_transcribe_sync_calls_endpoint():
    resp = MagicMock()
    resp.json.return_value = {"text": "ok"}
    resp.raise_for_status.return_value = None
    client = MagicMock()
    client.post.return_value = resp
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    with patch("httpx.Client", return_value=client):
        c = OpenSpeechClient("http://x")
        out = c.transcribe(b"wav")
        assert out["text"] == "ok"


def test_speak_sync_returns_bytes():
    resp = MagicMock()
    resp.content = b"abc"
    resp.raise_for_status.return_value = None
    client = MagicMock()
    client.post.return_value = resp
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    with patch("httpx.Client", return_value=client):
        c = OpenSpeechClient("http://x")
        out = c.speak("hi")
        assert out == b"abc"


@pytest.mark.asyncio
async def test_async_transcribe_calls_endpoint():
    resp = MagicMock()
    resp.json.return_value = {"text": "ok"}
    resp.raise_for_status.return_value = None
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    with patch("httpx.AsyncClient", return_value=client):
        c = OpenSpeechClient("http://x")
        out = await c.async_transcribe(b"wav")
        assert out["text"] == "ok"


@pytest.mark.asyncio
async def test_async_speak_returns_bytes():
    resp = MagicMock()
    resp.content = b"abc"
    resp.raise_for_status.return_value = None
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    with patch("httpx.AsyncClient", return_value=client):
        c = OpenSpeechClient("http://x")
        out = await c.async_speak("hi")
        assert out == b"abc"


def test_stream_transcribe_sync_ws_messages():
    events = [
        json.dumps({"type": "session.begin"}),
        json.dumps({"type": "transcript", "text": "hello"}),
        json.dumps({"type": "session.end"}),
    ]

    ws = MagicMock()
    ws.recv.side_effect = events
    ws.__enter__.return_value = ws
    ws.__exit__.return_value = None

    with patch("websockets.sync.client.connect", return_value=ws):
        c = OpenSpeechClient("http://x")
        out = list(c.stream_transcribe(iter([b"a", b"b"]), vad=True))

    assert [e["type"] for e in out] == ["session.begin", "transcript", "session.end"]
    assert ws.send.call_count >= 3  # two chunks + stop


@pytest.mark.asyncio
async def test_async_stream_transcribe_ws_messages():
    class AsyncWS:
        def __init__(self):
            self.sent = []
            self._events = [
                json.dumps({"type": "session.begin"}),
                json.dumps({"type": "transcript", "text": "hi"}),
                json.dumps({"type": "session.end"}),
            ]

        async def send(self, data):
            self.sent.append(data)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._events:
                raise StopAsyncIteration
            return self._events.pop(0)

    class AsyncCtx:
        def __init__(self):
            self.ws = AsyncWS()

        async def __aenter__(self):
            return self.ws

        async def __aexit__(self, exc_type, exc, tb):
            return None

    with patch("websockets.connect", return_value=AsyncCtx()):
        c = OpenSpeechClient("http://x")
        got = []
        async for ev in c.async_stream_transcribe(iter([b"a"])):
            got.append(ev["type"])

    assert got == ["session.begin", "transcript", "session.end"]


def test_realtime_session_methods_send_protocol_events():
    ws = MagicMock()
    ws.recv.side_effect = [json.dumps({"type": "session.created"}), Exception("done")]

    with patch("websockets.sync.client.connect", return_value=ws):
        c = OpenSpeechClient("http://x")
        sess = c.realtime_session()
        sess.send_audio(b"\x00\x01")
        sess.commit()
        sess.create_response("Hello", voice="alloy")
        sess.close()

    sent = [json.loads(call.args[0])["type"] for call in ws.send.call_args_list]
    assert "input_audio_buffer.append" in sent
    assert "input_audio_buffer.commit" in sent
    assert "response.create" in sent


@pytest.mark.asyncio
async def test_async_realtime_session_methods_send_protocol_events():
    class AsyncWS:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(json.loads(data))

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(0)
            raise StopAsyncIteration

        async def close(self):
            return None

    ws = AsyncWS()

    async def fake_connect(*args, **kwargs):
        return ws

    with patch("websockets.connect", side_effect=fake_connect):
        c = OpenSpeechClient("http://x")
        sess = await c.async_realtime_session()
        await sess.send_audio(b"\x00\x01")
        await sess.commit()
        await sess.create_response("Hello", voice="alloy")
        await sess.close()

    sent_types = [m["type"] for m in ws.sent]
    assert "input_audio_buffer.append" in sent_types
    assert "input_audio_buffer.commit" in sent_types
    assert "response.create" in sent_types
