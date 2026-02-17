from __future__ import annotations

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


def test_stream_transcribe_placeholder():
    c = OpenSpeechClient()
    out = list(c.stream_transcribe(iter([b"a", b"b"])))
    assert out[0]["event"] == "audio_chunk"
    assert out[1]["bytes"] == 1


@pytest.mark.asyncio
async def test_realtime_session_placeholder():
    c = OpenSpeechClient()
    events = []
    async for ev in c.realtime_session():
        events.append(ev["event"])
    assert events == ["session.started", "session.ended"]
