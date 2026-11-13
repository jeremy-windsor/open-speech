from __future__ import annotations

import asyncio
import base64
import functools
import json
import re
import shutil
import ssl
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from src.client import OpenSpeechClient
from tests.ui_support import NODE_MEMORY_LIMIT_MB, run_node_command


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
            self._stopped = asyncio.Event()
            self._events = [
                json.dumps({"type": "session.begin"}),
                json.dumps({"type": "transcript", "text": "hi"}),
                json.dumps({"type": "session.end"}),
            ]

        async def send(self, data):
            self.sent.append(data)
            if isinstance(data, str):
                self._stopped.set()

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._events:
                raise StopAsyncIteration
            if len(self._events) == 1:
                # Like the server, answer the client's stop with session.end.
                await asyncio.wait_for(self._stopped.wait(), timeout=5)
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


def test_realtime_session_vad_callback_runs_once_per_event():
    release_event = threading.Event()
    callback_event = threading.Event()
    received = []
    ws = MagicMock()

    def receive_event():
        if not release_event.is_set():
            assert release_event.wait(timeout=1.0)
            return json.dumps({"type": "input_audio_buffer.speech_started"})
        raise RuntimeError("done")

    ws.recv.side_effect = receive_event

    with patch("websockets.sync.client.connect", return_value=ws):
        session = OpenSpeechClient("http://x").realtime_session()

        def handle_vad(event):
            received.append(event)
            callback_event.set()

        session.on_vad(handle_vad)
        release_event.set()
        assert callback_event.wait(timeout=1.0)
        session.close()

    assert received == [{"type": "input_audio_buffer.speech_started"}]


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


@pytest.mark.asyncio
async def test_async_stream_transcribe_reconnect_resumes_async_generator_source(monkeypatch):
    import websockets
    from websockets.exceptions import ConnectionClosedError

    more_audio = asyncio.Event()
    first_sent = asyncio.Event()
    stop_sent = asyncio.Event()
    connections = []

    async def microphone():
        yield b"a"
        await more_audio.wait()  # idle when the connection drops
        yield b"b"

    class FakeWebSocket:
        def __init__(self):
            self.index = len(connections)
            self.sent = []
            self.ended = False
            self.begun = False
            connections.append(self)

        async def send(self, data):
            self.sent.append(data)
            if data == b"a":
                first_sent.set()
            if isinstance(data, str):
                stop_sent.set()

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.begun:  # the server's first event, before it reads any audio
                self.begun = True
                return json.dumps({"type": "session.begin"})
            if self.index == 0:
                await first_sent.wait()
                raise ConnectionClosedError(None, None)
            if self.ended:
                raise StopAsyncIteration
            more_audio.set()  # the source produces more audio after the reconnect
            await stop_sent.wait()
            self.ended = True
            return json.dumps({"type": "session.end"})

    class FakeConnect:
        async def __aenter__(self):
            return FakeWebSocket()

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: FakeConnect())
    client = OpenSpeechClient("http://x")

    async def consume():
        return [event async for event in client.async_stream_transcribe(microphone())]

    await asyncio.wait_for(consume(), timeout=10)
    delivered = [data for ws in connections for data in ws.sent if isinstance(data, bytes)]
    assert delivered == [b"a", b"b"]


def _fake_sync_stream(monkeypatch, backoff, session_end: threading.Event):
    """Patch the sync client: connection 0 drops after b"a"; later ones end on session_end."""
    import websockets.sync.client as sync_client
    from websockets.exceptions import ConnectionClosedError

    from src import client as client_module

    first_sent = threading.Event()
    connections = []

    class FakeWebSocket:
        def __init__(self):
            self.index = len(connections)
            self.sent = []
            self.begun = False
            connections.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def send(self, data):
            self.sent.append(data)
            if data == b"a":
                first_sent.set()
            if isinstance(data, str):
                session_end.set()

        def recv(self):
            if not self.begun:  # the server's first event, before it reads any audio
                self.begun = True
                return json.dumps({"type": "session.begin"})
            if self.index == 0:
                first_sent.wait(timeout=10)
                raise ConnectionClosedError(None, None)  # network drop after the first chunk
            session_end.wait(timeout=10)
            return json.dumps({"type": "session.end"})

    monkeypatch.setattr(sync_client, "connect", lambda *args, **kwargs: FakeWebSocket())
    monkeypatch.setattr(client_module, "time", SimpleNamespace(sleep=backoff))
    return connections


def test_stream_transcribe_reconnect_delivers_chunk_read_during_disconnect(monkeypatch):
    release_b = threading.Event()

    def microphone():
        yield b"a"
        release_b.wait(timeout=10)  # a live source blocks until its next chunk is captured
        yield b"b"
        yield b"c"

    def backoff(_seconds):
        # The next chunk arrives while the client waits to reconnect.
        release_b.set()
        for thread in threading.enumerate():
            if "_sender" in thread.name:
                thread.join(timeout=10)

    connections = _fake_sync_stream(monkeypatch, backoff, threading.Event())
    list(OpenSpeechClient("http://x").stream_transcribe(microphone()))

    delivered = [data for ws in connections for data in ws.sent if isinstance(data, bytes)]
    assert delivered == [b"a", b"b", b"c"]


class _IdleMicrophone:
    """Yields b"a", stays idle until released, then yields b"b" and b"c"."""

    def __init__(self, release: threading.Event, on_concurrent_read: threading.Event):
        self._chunks = [b"a", b"b", b"c"]
        self._release = release
        self._on_concurrent_read = on_concurrent_read
        self._reading = threading.Lock()
        self.concurrent_reads = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._reading.acquire(blocking=False):
            self.concurrent_reads += 1
            self._on_concurrent_read.set()
            raise RuntimeError("source read from two threads at once")
        try:
            if not self._chunks:
                raise StopIteration
            if len(self._chunks) == 2:
                self._release.wait(timeout=10)
            return self._chunks.pop(0)
        finally:
            self._reading.release()


def test_stream_transcribe_reconnect_never_reads_source_from_two_threads(monkeypatch):
    # The source stays idle past the sender's join timeout, so the old sender is still
    # inside it when the client is ready to reconnect.
    release_b = threading.Event()
    session_end = threading.Event()
    microphone = _IdleMicrophone(release_b, on_concurrent_read=session_end)

    def backoff(_seconds):
        threading.Timer(0.3, release_b.set).start()

    connections = _fake_sync_stream(monkeypatch, backoff, session_end)
    try:
        list(OpenSpeechClient("http://x").stream_transcribe(microphone))
    finally:
        release_b.set()

    assert microphone.concurrent_reads == 0
    delivered = [data for ws in connections for data in ws.sent if isinstance(data, bytes)]
    assert delivered == [b"a", b"b", b"c"]


# ---------------------------------------------------------------------------
# Streaming transcription: failures, reconnects and TLS
# ---------------------------------------------------------------------------


class _FakeSyncStream:
    """A websockets.sync connection to /v1/audio/stream.

    Like the server, it sends session.begin first, before it reads any audio. It answers
    the client's stop with session.end, drops when the client sends `drop_after`, with
    `fail_stop` drops while the stop is being sent, and with `drop_on_stop` accepts the
    stop but drops before answering it. recv() waits at most 5 s for something to happen,
    then reports a drop.
    """

    def __init__(self, *, drop_after: bytes | None = None, fail_stop: bool = False, drop_on_stop: bool = False):
        self.sent: list = []
        self.drop_after = drop_after
        self.fail_stop = fail_stop
        self.drop_on_stop = drop_on_stop
        self.closed_by_client = False
        self.state = "open"
        self.begun = False
        self._ended = False
        self._changed = threading.Condition()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def _set(self, state: str) -> None:
        with self._changed:
            self.state = state
            self._changed.notify_all()

    def send(self, data):
        if self.state != "open":
            raise ConnectionClosedError(None, None)
        if isinstance(data, str) and self.fail_stop:
            self._set("dropped")
            raise ConnectionClosedError(None, None)
        self.sent.append(data)
        if isinstance(data, str):
            self._set("dropped" if self.drop_on_stop else "stopped")
        elif data == self.drop_after:
            self._set("dropped")

    def close(self):
        self.closed_by_client = True
        if self.state == "open":
            self._set("closed")

    def recv(self):
        if not self.begun:
            self.begun = True
            return json.dumps({"type": "session.begin"})
        with self._changed:
            self._changed.wait_for(lambda: self.state != "open", timeout=5)
        if self.state == "stopped" and not self._ended:
            self._ended = True
            return json.dumps({"type": "session.end"})
        if self.state == "closed":
            raise ConnectionClosedOK(None, None)
        raise ConnectionClosedError(None, None)


class _FakeAsyncStream:
    """The asyncio counterpart of _FakeSyncStream (session.begin first); `replies` maps a chunk to an event it
    triggers.

    `drop_seen` is set once the client's receive loop has been told about a drop.
    """

    def __init__(
        self, *, drop_after: bytes | None = None, fail_stop: bool = False, drop_on_stop: bool = False, replies=None,
    ):
        self.sent: list = []
        self.drop_after = drop_after
        self.fail_stop = fail_stop
        self.drop_on_stop = drop_on_stop
        self.replies = dict(replies or {})
        self.closed_by_client = False
        self.state = "open"
        self.drop_seen = asyncio.Event()
        self._events: list[dict] = [{"type": "session.begin"}]
        self._ended = False
        self._wake = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def _set(self, state: str) -> None:
        self.state = state
        self._wake.set()

    async def send(self, data):
        if self.state != "open":
            raise ConnectionClosedError(None, None)
        if isinstance(data, str) and self.fail_stop:
            self._set("dropped")
            raise ConnectionClosedError(None, None)
        self.sent.append(data)
        if isinstance(data, str):
            self._set("dropped" if self.drop_on_stop else "stopped")
        elif data == self.drop_after:
            self._set("dropped")
        elif data in self.replies:
            self._events.append(self.replies[data])
            self._wake.set()

    async def close(self):
        self.closed_by_client = True
        if self.state == "open":
            self._set("closed")

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            if self._ended:
                raise StopAsyncIteration
            if self._events:
                return json.dumps(self._events.pop(0))
            if self.state == "stopped":
                self._ended = True
                return json.dumps({"type": "session.end"})
            if self.state == "closed":
                raise StopAsyncIteration  # websockets ends the iteration on a normal close
            if self.state == "dropped":
                self.drop_seen.set()
                raise ConnectionClosedError(None, None)
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=5)
            except asyncio.TimeoutError:
                raise ConnectionClosedError(None, None) from None


def _patch_sync_connect(monkeypatch, *sockets: _FakeSyncStream, refuse=()):
    """Serve `sockets` (then fresh ones) to stream_transcribe; the connect calls numbered in `refuse` are refused."""
    import websockets.sync.client as sync_client

    from src import client as client_module

    record = SimpleNamespace(calls=0, connections=[])
    prepared = list(sockets)

    def connect(url, **kwargs):
        record.calls += 1
        if record.calls in refuse:
            raise ConnectionRefusedError("server still restarting")
        ws = prepared.pop(0) if prepared else _FakeSyncStream()
        record.connections.append(ws)
        return ws

    monkeypatch.setattr(sync_client, "connect", connect)
    monkeypatch.setattr(client_module, "time", SimpleNamespace(sleep=lambda _seconds: None))
    return record


def _patch_async_connect(monkeypatch, *sockets: _FakeAsyncStream, refuse=()):
    """Serve `sockets` (then fresh ones) to async_stream_transcribe; see _patch_sync_connect."""
    import websockets

    record = SimpleNamespace(calls=0, connections=[])
    prepared = list(sockets)

    class Connect:
        async def __aenter__(self):
            record.calls += 1
            for _ in range(5):
                await asyncio.sleep(0)  # a real handshake awaits the network
            if record.calls in refuse:
                raise ConnectionRefusedError("server still restarting")
            ws = prepared.pop(0) if prepared else _FakeAsyncStream()
            record.connections.append(ws)
            return ws

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: Connect())
    return record


def _delivered(connections) -> list[bytes]:
    return [data for ws in connections for data in ws.sent if isinstance(data, bytes)]


async def _collect(stream) -> list[dict]:
    return [event async for event in stream]


async def _consume_async(audio, **kwargs) -> list[dict]:
    stream = OpenSpeechClient("http://x").async_stream_transcribe(audio, **kwargs)
    return await asyncio.wait_for(_collect(stream), timeout=10)


def test_stream_transcribe_raises_the_audio_source_error(monkeypatch):
    # An iterator wrapper propagates its source's exception. The server has no idle timeout,
    # so the client closes the connection itself instead of waiting for a drop.
    ws = _FakeSyncStream()
    record = _patch_sync_connect(monkeypatch, ws)

    def microphone():
        yield b"a"
        raise RuntimeError("microphone unplugged")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="microphone unplugged"):
        list(OpenSpeechClient("http://x").stream_transcribe(microphone()))
    assert ws.closed_by_client
    assert time.monotonic() - started < 4
    assert record.calls == 1


@pytest.mark.asyncio
async def test_async_stream_transcribe_raises_the_audio_source_error(monkeypatch):
    ws = _FakeAsyncStream()
    record = _patch_async_connect(monkeypatch, ws)

    async def microphone():
        yield b"a"
        raise RuntimeError("microphone unplugged")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="microphone unplugged"):
        await _consume_async(microphone())
    assert ws.closed_by_client
    assert time.monotonic() - started < 4
    assert record.calls == 1


_UNCONFIRMED = "disconnected before the server confirmed the end of the stream"


def test_stream_transcribe_raises_when_the_final_stop_is_not_delivered(monkeypatch):
    # Server sessions are per connection: without the stop the server never flushed the last
    # utterance, and a reconnect cannot recover it.
    record = _patch_sync_connect(monkeypatch, _FakeSyncStream(fail_stop=True))
    with pytest.raises(RuntimeError, match=_UNCONFIRMED) as caught:
        list(OpenSpeechClient("http://x").stream_transcribe(iter([b"a"])))
    assert isinstance(caught.value.__cause__, ConnectionClosed)
    assert record.calls == 1


@pytest.mark.asyncio
async def test_async_stream_transcribe_raises_when_the_final_stop_is_not_delivered(monkeypatch):
    record = _patch_async_connect(monkeypatch, _FakeAsyncStream(fail_stop=True))
    with pytest.raises(RuntimeError, match=_UNCONFIRMED) as caught:
        await _consume_async(iter([b"a"]))
    assert isinstance(caught.value.__cause__, ConnectionClosed)
    assert record.calls == 1


def test_stream_transcribe_raises_when_the_connection_drops_before_session_end(monkeypatch):
    # The stop usually goes out into the socket buffer and the drop shows up only in recv().
    # The server answers every stop with session.end once it has flushed the last utterance;
    # without that answer the transcript may be incomplete.
    ws = _FakeSyncStream(drop_on_stop=True)
    record = _patch_sync_connect(monkeypatch, ws)
    with pytest.raises(RuntimeError, match=_UNCONFIRMED) as caught:
        list(OpenSpeechClient("http://x").stream_transcribe(iter([b"a"])))
    assert isinstance(caught.value.__cause__, ConnectionClosedError)
    assert ws.sent == [b"a", json.dumps({"type": "stop"})]
    assert record.calls == 1


@pytest.mark.asyncio
async def test_async_stream_transcribe_raises_when_the_connection_drops_before_session_end(monkeypatch):
    ws = _FakeAsyncStream(drop_on_stop=True)
    record = _patch_async_connect(monkeypatch, ws)
    with pytest.raises(RuntimeError, match=_UNCONFIRMED) as caught:
        await _consume_async(iter([b"a"]))
    assert isinstance(caught.value.__cause__, ConnectionClosedError)
    assert ws.sent == [b"a", json.dumps({"type": "stop"})]
    assert record.calls == 1


def test_stream_transcribe_raises_when_the_source_ends_after_a_drop(monkeypatch):
    # Every chunk went out on the connection that dropped and none is left for a new one,
    # which cannot recover what the old one had not transcribed: report the drop.
    from src import client as client_module

    source_may_end = threading.Event()
    record = _patch_sync_connect(monkeypatch, _FakeSyncStream(drop_after=b"a"))
    monkeypatch.setattr(client_module, "time", SimpleNamespace(sleep=lambda _seconds: source_may_end.set()))

    def microphone():
        yield b"a"
        source_may_end.wait(timeout=10)  # the source ends while the client waits to reconnect

    with pytest.raises(RuntimeError, match=_UNCONFIRMED) as caught:
        list(OpenSpeechClient("http://x").stream_transcribe(microphone()))
    assert isinstance(caught.value.__cause__, ConnectionClosedError)
    assert record.calls == 1


@pytest.mark.asyncio
async def test_async_stream_transcribe_raises_when_the_source_ends_after_a_drop(monkeypatch):
    ws = _FakeAsyncStream(drop_after=b"a")
    record = _patch_async_connect(monkeypatch, ws)

    async def microphone():
        yield b"a"
        await ws.drop_seen.wait()  # the source ends once the client has seen the drop

    with pytest.raises(RuntimeError, match=_UNCONFIRMED) as caught:
        await _consume_async(microphone())
    assert isinstance(caught.value.__cause__, ConnectionClosedError)
    assert record.calls == 1


def test_stream_transcribe_counts_a_refused_reconnect_as_an_attempt(monkeypatch):
    # A drop usually means a server restart, so the first reconnect is often refused;
    # reconnect_attempts=2 still allows another (client-js counts failed opens the same way).
    record = _patch_sync_connect(monkeypatch, _FakeSyncStream(drop_after=b"a"), refuse={2})
    list(OpenSpeechClient("http://x").stream_transcribe(iter([b"a", b"b"]), reconnect_attempts=2))
    assert record.calls == 3
    assert _delivered(record.connections) == [b"a", b"b"]


@pytest.mark.asyncio
async def test_async_stream_transcribe_counts_a_refused_reconnect_as_an_attempt(monkeypatch):
    record = _patch_async_connect(monkeypatch, _FakeAsyncStream(drop_after=b"a"), refuse={2})
    await _consume_async(iter([b"a", b"b"]), reconnect_attempts=2)
    assert record.calls == 3
    assert _delivered(record.connections) == [b"a", b"b"]


def test_stream_transcribe_reports_the_limit_when_every_reconnect_is_refused(monkeypatch):
    record = _patch_sync_connect(monkeypatch, _FakeSyncStream(drop_after=b"a"), refuse={2, 3})
    with pytest.raises(RuntimeError, match="reconnection limit reached") as caught:
        list(OpenSpeechClient("http://x").stream_transcribe(iter([b"a", b"b"]), reconnect_attempts=2))
    assert isinstance(caught.value.__cause__, ConnectionRefusedError)
    assert record.calls == 3


@pytest.mark.asyncio
async def test_async_stream_transcribe_reports_the_limit_when_every_reconnect_is_refused(monkeypatch):
    record = _patch_async_connect(monkeypatch, _FakeAsyncStream(drop_after=b"a"), refuse={2, 3})
    with pytest.raises(RuntimeError, match="reconnection limit reached") as caught:
        await _consume_async(iter([b"a", b"b"]), reconnect_attempts=2)
    assert isinstance(caught.value.__cause__, ConnectionRefusedError)
    assert record.calls == 3


def test_stream_transcribe_raises_a_failed_first_connection_without_reading_the_source(monkeypatch):
    _patch_sync_connect(monkeypatch, refuse={1})
    read = []

    def microphone():
        read.append(True)
        yield b"a"

    with pytest.raises(ConnectionRefusedError):
        list(OpenSpeechClient("http://x").stream_transcribe(microphone()))
    assert read == []


@pytest.mark.asyncio
async def test_async_stream_transcribe_failed_first_connection_leaves_the_source_usable(monkeypatch):
    # A call that fails before it opens a connection must not consume its input: the caller
    # can retry with the same microphone and lose no audio.
    more_audio = asyncio.Event()
    second = _FakeAsyncStream()
    record = _patch_async_connect(monkeypatch, second, refuse={1})

    async def microphone():
        yield b"a"
        await more_audio.wait()  # a live source idles between buffers
        yield b"b"

    mic = microphone()
    with pytest.raises(ConnectionRefusedError):
        await _consume_async(mic)

    async def release_more_audio():
        while not second.sent:
            await asyncio.sleep(0.01)
        more_audio.set()

    releaser = asyncio.create_task(release_more_audio())
    await _consume_async(mic)
    await releaser
    assert record.calls == 2
    assert _delivered(record.connections) == [b"a", b"b"]


# The server accepts a refused /v1/audio/stream socket
# and closes it at once with 4001 (API key), 1008 (Origin) or 4004 (turned off), where it used to fail the handshake.
# The client reports that close code and reason as StreamRefusedError, and does not reconnect: a retry cannot fix it.
_REFUSALS = [(4001, "Invalid or missing API key"), (1008, "Origin not allowed"), (4004, "Live TTS is disabled")]


def _refused_close(code: int, reason: str) -> ConnectionClosedError:
    from websockets.frames import Close

    return ConnectionClosedError(Close(code, reason), Close(code, reason), True)


class _RefusedSyncStream(_FakeSyncStream):
    """A connection the server accepted and closed at once with `code` and `reason`."""

    def __init__(self, code: int, reason: str):
        super().__init__()
        self.refusal = (code, reason)

    def send(self, data):
        raise _refused_close(*self.refusal)

    def recv(self):
        raise _refused_close(*self.refusal)


class _RefusedAsyncStream(_FakeAsyncStream):
    def __init__(self, code: int, reason: str):
        super().__init__()
        self.refusal = (code, reason)

    async def send(self, data):
        raise _refused_close(*self.refusal)

    async def __anext__(self):
        raise _refused_close(*self.refusal)


@pytest.mark.parametrize(("code", "reason"), _REFUSALS)
def test_stream_transcribe_reports_a_refusal_close_with_its_code_and_reason(monkeypatch, code, reason):
    record = _patch_sync_connect(monkeypatch, *[_RefusedSyncStream(code, reason) for _ in range(3)])
    with pytest.raises(RuntimeError) as caught:
        list(OpenSpeechClient("http://x").stream_transcribe(iter([b"a"]), reconnect_attempts=2))
    assert record.calls == 1, "a refusal is not retried"
    assert str(code) in str(caught.value) and reason in str(caught.value), str(caught.value)
    from src.client import StreamRefusedError

    assert isinstance(caught.value, StreamRefusedError)
    assert (caught.value.code, caught.value.reason) == (code, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "reason"), _REFUSALS)
async def test_async_stream_transcribe_reports_a_refusal_close_with_its_code_and_reason(monkeypatch, code, reason):
    record = _patch_async_connect(monkeypatch, *[_RefusedAsyncStream(code, reason) for _ in range(3)])
    with pytest.raises(RuntimeError) as caught:
        await _consume_async(iter([b"a"]), reconnect_attempts=2)
    assert record.calls == 1, "a refusal is not retried"
    assert str(code) in str(caught.value) and reason in str(caught.value), str(caught.value)
    from src.client import StreamRefusedError

    assert isinstance(caught.value, StreamRefusedError)
    assert (caught.value.code, caught.value.reason) == (code, reason)


# Since a refusal opens the socket, the sender started on that open and read the
# caller's audio before StreamRefusedError was raised. What it read was lost to the caller's retry, and the sync
# sender could still be inside a live source after the call had raised. The source is now read only after the
# server's first event, and a refused stream reads nothing.
def test_stream_transcribe_refused_reads_nothing_so_a_retry_sends_all_the_audio(monkeypatch):
    from src.client import StreamRefusedError

    record = _patch_sync_connect(monkeypatch, _RefusedSyncStream(*_REFUSALS[0]), _FakeSyncStream())
    read = []

    def microphone():
        for chunk in (b"a", b"b"):
            read.append(chunk)
            yield chunk

    source = microphone()
    client = OpenSpeechClient("http://x")
    with pytest.raises(StreamRefusedError):
        list(client.stream_transcribe(source))
    assert read == [], "the refused call read the source"
    assert list(client.stream_transcribe(source))[-1] == {"type": "session.end"}
    assert _delivered(record.connections) == [b"a", b"b"]


def test_stream_transcribe_refused_leaves_no_thread_inside_a_live_source(monkeypatch):
    from src.client import StreamRefusedError

    record = _patch_sync_connect(monkeypatch, _RefusedSyncStream(*_REFUSALS[0]), _FakeSyncStream())
    next_buffer = threading.Event()
    inside = []

    def microphone():
        inside.append(True)
        try:
            next_buffer.wait(timeout=10)  # a live source waits for its next buffer
        finally:
            inside.pop()
        yield b"a"

    mic = microphone()
    client = OpenSpeechClient("http://x")
    try:
        with pytest.raises(StreamRefusedError):
            list(client.stream_transcribe(mic))
        assert inside == [], "stream_transcribe raised while its sender was inside the source"
        threading.Timer(0.2, next_buffer.set).start()
        list(client.stream_transcribe(mic))  # the retry: no "generator already executing"
    finally:
        next_buffer.set()
    assert _delivered(record.connections) == [b"a"]


@pytest.mark.asyncio
async def test_async_stream_transcribe_refused_leaves_the_source_usable(monkeypatch):
    from src.client import StreamRefusedError

    second = _FakeAsyncStream()
    record = _patch_async_connect(monkeypatch, _RefusedAsyncStream(*_REFUSALS[0]), second)
    more_audio = asyncio.Event()

    async def microphone():
        yield b"a"
        await more_audio.wait()  # a live source idles between buffers
        yield b"b"

    mic = microphone()
    with pytest.raises(StreamRefusedError):
        await _consume_async(mic)

    async def release_more_audio():
        while not second.sent:
            await asyncio.sleep(0.01)
        more_audio.set()

    releaser = asyncio.create_task(release_more_audio())
    await _consume_async(mic)
    await releaser
    assert _delivered(record.connections) == [b"a", b"b"]


# The client started reading the caller's audio on any first frame. When the model or the VAD
# fails to load, the server's first frame is an `error` event, not session.begin, and it then drops the connection
# without a close frame; the client read and sent the source, and reconnected to load the failing model again. Only
# session.begin starts the stream: anything else first is yielded, then the call raises, with nothing read from the
# source and no reconnect.
_LOAD_FAILED = {"type": "error", "message": "Failed to load model: Repository Not Found"}


class _LoadFailedSyncStream(_FakeSyncStream):
    """The server sends the load error, then returns: no session.begin, no close frame (a drop)."""

    def recv(self):
        if not self.begun:
            self.begun = True
            return json.dumps(_LOAD_FAILED)
        raise ConnectionClosedError(None, None)


class _LoadFailedAsyncStream(_FakeAsyncStream):
    def __init__(self):
        super().__init__()
        self._events = [_LOAD_FAILED]
        self.state = "dropped"  # once the error is read, the connection is gone


def _recorded_source(read: list):
    def microphone():
        for chunk in (b"a", b"b", b"c", b"d"):
            read.append(chunk)
            yield chunk

    return microphone()


def test_stream_transcribe_reads_nothing_and_does_not_reconnect_a_stream_that_did_not_start(monkeypatch):
    record = _patch_sync_connect(monkeypatch, *[_LoadFailedSyncStream() for _ in range(3)])
    read: list = []
    events: list = []
    with pytest.raises(RuntimeError) as caught:
        for event in OpenSpeechClient("http://x").stream_transcribe(_recorded_source(read), reconnect_attempts=2):
            events.append(event)
    assert read == [], f"the source was read for a stream that never began: {read}"
    assert record.calls == 1, f"{record.calls} connections, each loading the failing model again"
    assert events == [_LOAD_FAILED]
    assert "did not start" in str(caught.value) and _LOAD_FAILED["message"] in str(caught.value), str(caught.value)


@pytest.mark.asyncio
async def test_async_stream_transcribe_reads_nothing_and_does_not_reconnect_a_stream_that_did_not_start(monkeypatch):
    record = _patch_async_connect(monkeypatch, *[_LoadFailedAsyncStream() for _ in range(3)])
    read: list = []
    events: list = []

    async def run():
        async for event in OpenSpeechClient("http://x").async_stream_transcribe(
            _recorded_source(read), reconnect_attempts=2,
        ):
            events.append(event)

    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(run(), timeout=10)
    assert read == [], f"the source was read for a stream that never began: {read}"
    assert record.calls == 1, f"{record.calls} connections, each loading the failing model again"
    assert events == [_LOAD_FAILED]
    assert "did not start" in str(caught.value) and _LOAD_FAILED["message"] in str(caught.value), str(caught.value)


def test_stream_transcribe_stops_when_a_reconnect_does_not_start(monkeypatch):
    # The same check applies to a reconnect's first frame: the stream ends there, and the rest of the source is unread.
    record = _patch_sync_connect(
        monkeypatch, _FakeSyncStream(drop_after=b"a"), *[_LoadFailedSyncStream() for _ in range(3)],
    )
    read: list = []
    with pytest.raises(RuntimeError) as caught:
        list(OpenSpeechClient("http://x").stream_transcribe(_recorded_source(read), reconnect_attempts=2))
    assert record.calls == 2, f"{record.calls} connections"
    assert read == [b"a", b"b"], read  # "b" was read before the first connection's drop was seen
    assert "did not start" in str(caught.value), str(caught.value)


@pytest.mark.asyncio
async def test_async_stream_transcribe_stops_when_a_reconnect_does_not_start(monkeypatch):
    record = _patch_async_connect(
        monkeypatch, _FakeAsyncStream(drop_after=b"a"), *[_LoadFailedAsyncStream() for _ in range(3)],
    )
    with pytest.raises(RuntimeError) as caught:
        await _consume_async(_recorded_source([]), reconnect_attempts=2)
    assert record.calls == 2, f"{record.calls} connections"
    assert "did not start" in str(caught.value), str(caught.value)


def test_stream_transcribe_still_reconnects_after_a_drop_that_is_not_a_refusal(monkeypatch):
    # A server restart (1012) or a lost connection is still retried, as before.
    record = _patch_sync_connect(monkeypatch, _RefusedSyncStream(1012, "Service restart"))
    events = list(OpenSpeechClient("http://x").stream_transcribe(iter([b"a"]), reconnect_attempts=2))
    assert events == [{"type": "session.begin"}, {"type": "session.end"}]
    assert record.calls == 2


@pytest.mark.asyncio
async def test_async_stream_transcribe_reads_a_sync_source_off_the_event_loop(monkeypatch):
    # A live sync source blocks in next() until it has the next buffer; the event loop must
    # keep delivering the server's events meanwhile.
    _patch_async_connect(monkeypatch, _FakeAsyncStream(replies={b"a": {"type": "transcript", "text": "a"}}))
    transcript_seen = threading.Event()

    class Microphone:
        def __init__(self):
            self.calls = 0
            self.waited_for_transcript = None

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                return b"a"
            if self.calls == 2:
                self.waited_for_transcript = transcript_seen.wait(timeout=2)
                return b"b"
            raise StopIteration

    mic = Microphone()

    async def consume():
        async for event in OpenSpeechClient("http://x").async_stream_transcribe(mic):
            if event.get("type") == "transcript":
                transcript_seen.set()

    try:
        await asyncio.wait_for(consume(), timeout=10)
    finally:
        transcript_seen.set()
    assert mic.waited_for_transcript is True


async def _captured_ws_connect(flavor: str, client: OpenSpeechClient) -> dict:
    """Open one WebSocket of `flavor` against fakes and return the connect() keyword arguments."""
    captured: dict = {}

    def sync_connect(url, **kwargs):
        captured.update(kwargs, url=url)
        if flavor == "stream":
            return _FakeSyncStream()
        ws = MagicMock()
        ws.recv.side_effect = RuntimeError("closed")
        return ws

    def async_stream_connect(url, **kwargs):
        captured.update(kwargs, url=url)
        return _FakeAsyncStream()

    async def async_realtime_connect(url, **kwargs):
        captured.update(kwargs, url=url)
        ws = _FakeAsyncStream()
        ws.state = "closed"
        return ws

    if flavor == "stream":
        with patch("websockets.sync.client.connect", side_effect=sync_connect):
            list(client.stream_transcribe(iter([b"a"])))
    elif flavor == "realtime":
        with patch("websockets.sync.client.connect", side_effect=sync_connect):
            client.realtime_session().close()
    elif flavor == "async_stream":
        with patch("websockets.connect", side_effect=async_stream_connect):
            await asyncio.wait_for(_collect(client.async_stream_transcribe(iter([b"a"]))), timeout=10)
    else:
        with patch("websockets.connect", side_effect=async_realtime_connect):
            session = await client.async_realtime_session()
            await session.close()
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("flavor", ["stream", "async_stream", "realtime", "async_realtime"])
async def test_websockets_honour_ssl_verify(flavor):
    # examples/python_client.py: the server's default certificate is self-signed and
    # ssl_verify=False accepts it, as every HTTP method does.
    insecure = await _captured_ws_connect(flavor, OpenSpeechClient("https://localhost:8100", ssl_verify=False))
    assert insecure["url"].startswith("wss://")
    context = insecure.get("ssl")
    assert isinstance(context, ssl.SSLContext), insecure
    assert context.verify_mode == ssl.CERT_NONE and not context.check_hostname

    custom = ssl.create_default_context()
    with_context = await _captured_ws_connect(flavor, OpenSpeechClient("https://localhost:8100", ssl_verify=custom))
    assert with_context["ssl"] is custom
    # Verification on keeps websockets' verifying default; ws:// rejects any ssl argument.
    assert "ssl" not in await _captured_ws_connect(flavor, OpenSpeechClient("https://localhost:8100"))
    assert "ssl" not in await _captured_ws_connect(flavor, OpenSpeechClient("http://localhost:8100", ssl_verify=False))


# ---------------------------------------------------------------------------
# client-js (run under node against WebSocket, AudioContext and fetch fakes)
# ---------------------------------------------------------------------------


_CLIENT_JS_FAKES = r"""
class FakeWebSocket {
  static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
  static instances = [];
  constructor(url, protocols) {
    this.url = url; this.protocols = protocols; this.readyState = 0; this.sent = []; this.closeCalls = 0;
    this.onopen = null; this.onmessage = null; this.onerror = null; this.onclose = null;
    FakeWebSocket.instances.push(this);
  }
  open() { this.readyState = 1; this.onopen?.({ type: "open" }); }
  fail() { this.onerror?.({ type: "error" }); }
  message(data) { this.onmessage?.({ data }); }
  send(data) { if (this.readyState !== 1) throw new Error("send called while not open"); this.sent.push(data); }
  // The client's close(): as in browsers (WHATWG WebSockets, close() step 1), a code other than
  // 1000 or 3000-4999 throws and leaves the socket open.
  close(code = 1000, reason = "") {
    if (code !== 1000 && !(code >= 3000 && code <= 4999)) throw new DOMException("invalid code", "InvalidAccessError");
    this.closeCalls++;
    this.closedByServer(code, reason);
  }
  // The server or the network ending the connection, with any code.
  closedByServer(code, reason = "") {
    if (this.readyState === 3) return;
    this.readyState = 3;
    this.onclose?.({ code, reason });
  }
}
class FakeNode { connect() { return this; } disconnect() { this.disconnected = true; } }
class FakeAudioContext {
  static instances = [];
  constructor() {
    this.destination = new FakeNode(); this.processor = new FakeNode(); this.closed = false;
    FakeAudioContext.instances.push(this);
  }
  createMediaStreamSource() { return new FakeNode(); }
  createAnalyser() { return new FakeNode(); }
  createScriptProcessor() { return this.processor; }
  close() { this.closed = true; return Promise.resolve(); }
}
globalThis.WebSocket = FakeWebSocket;
globalThis.AudioContext = FakeAudioContext;
async function waitFor(predicate) {
  const deadline = Date.now() + 3000;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for test condition");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
// "resolved", "rejected: <name>: <message>", or "pending" if `promise` has not settled within ms.
function settled(promise, ms = 500) {
  return Promise.race([
    promise.then(() => "resolved", (error) => `rejected: ${error?.name}: ${error?.message}`),
    new Promise((resolve) => setTimeout(() => resolve("pending"), ms)),
  ]);
}
const { OpenSpeechClient } = await import(process.argv[2]);
"""


@functools.lru_cache(maxsize=1)
def _node_running_typescript() -> str | None:
    """node, if it runs client-js/src/index.ts as is (type stripping is on by default from 22.18 / 23.6)."""
    node = shutil.which("node")
    if node is None:
        return None
    check = [node, "-e", "console.log(process.features.typescript ? 'yes' : 'no')"]
    output = run_node_command(check, what="node's TypeScript check", timeout=60)
    return node if output.strip().splitlines()[-1:] == ["yes"] else None


def _run_client_js(tmp_path: Path, body: str, *, memory_limit_mb: int = NODE_MEMORY_LIMIT_MB) -> dict:
    # Node runs under the suite's memory cap and process-tree kill
    # (ui_support.run_node_command), as every other node run does, so a test that loops while
    # allocating fails instead of starving the host.
    node = _node_running_typescript()
    if node is None:
        pytest.skip("needs node 22.18+ (runs TypeScript without a build step)")
    script = tmp_path / "client_test.mjs"
    script.write_text(_CLIENT_JS_FAKES + body, encoding="utf-8")
    index = (Path(__file__).resolve().parents[1] / "client-js" / "src" / "index.ts").as_uri()
    output = run_node_command([node, str(script), index], what=f"node {script.name}", timeout=60,
                              memory_limit_mb=memory_limit_mb)
    return json.loads(output.strip().splitlines()[-1])


def test_js_stream_transcribe_releases_the_microphone_and_socket_when_the_consumer_stops(tmp_path):
    # Breaking out of `for await (const event of client.streamTranscribe(media))` calls
    # iterator.return(); the README offers no other way to stop dictation.
    result = _run_client_js(tmp_path, r"""
const events = new OpenSpeechClient({ baseUrl: "http://example.test" }).streamTranscribe({}, 0);
const first = events.next();
await waitFor(() => FakeWebSocket.instances.length === 1);
const ws = FakeWebSocket.instances[0];
ws.open();
await waitFor(() => FakeAudioContext.instances.length === 1);
const context = FakeAudioContext.instances[0];
ws.message(JSON.stringify({ type: "transcript", text: "hello" }));
const { value } = await first;
const done = await events.return(undefined);
// Audio the browser captures after the consumer stopped must not reach the server.
context.processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array(4) } });
console.log(JSON.stringify({
  value, done: done.done, socketClosed: ws.readyState === FakeWebSocket.CLOSED, sent: ws.sent.length,
  contextClosed: context.closed, processorDisconnected: context.processor.disconnected === true,
}));
""")
    assert result == {
        "value": {"type": "transcript", "text": "hello"},
        "done": True,
        "socketClosed": True,
        "sent": 0,
        "contextClosed": True,
        "processorDisconnected": True,
    }


def test_js_realtime_send_audio_encodes_a_multi_second_chunk_without_buffer(tmp_path):
    # Browsers have no Buffer; the README sets no chunk size limit and the server accepts up
    # to OS_REALTIME_MAX_BUFFER_MB, so a push-to-talk recording goes out in one call.
    result = _run_client_js(tmp_path, r"""
globalThis.Buffer = undefined;  // as in a browser
const session = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
const ws = FakeWebSocket.instances[0];
ws.open();
await session.ready;
const bytes = new Uint8Array(24000 * 2 * 5);  // 5 s of 24 kHz PCM16
for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 7) % 256;
await session.sendAudio(bytes.buffer);
const message = JSON.parse(ws.sent[0]);
console.log(JSON.stringify({ messages: ws.sent.length, type: message.type, audio: message.audio }));
""")
    assert result["messages"] == 1
    assert result["type"] == "input_audio_buffer.append"
    assert base64.b64decode(result["audio"]) == bytes((i * 7) % 256 for i in range(24000 * 2 * 5))


def test_js_websockets_authenticate_with_a_one_time_ticket(tmp_path):
    # Browsers cannot set WebSocket headers, so with an apiKey every connection,
    # reconnects included, first gets a single-use ticket from POST /api/auth/ws-ticket.
    result = _run_client_js(tmp_path, r"""
const requests = [];
let issued = 0;
globalThis.fetch = async (url, init = {}) => {
  requests.push({ url: String(url), method: init.method, auth: new Headers(init.headers).get("authorization") });
  return Response.json({ ticket: `t${++issued}/+`, expires_in: 60 });
};
const client = new OpenSpeechClient({ baseUrl: "https://speech.test:8100", apiKey: "secret" });

const realtime = client.realtimeSession();
await waitFor(() => FakeWebSocket.instances.length === 1);
FakeWebSocket.instances[0].open();
await realtime.ready;

const live = client.liveSpeechSession({ voice: "af_heart" });
await waitFor(() => FakeWebSocket.instances.length === 2);
const liveSocket = FakeWebSocket.instances[1];
liveSocket.open();
liveSocket.message(JSON.stringify({ type: "session.created" }));
liveSocket.message(JSON.stringify({ type: "session.updated" }));
await live.ready;

const events = client.streamTranscribe({}, 1);
const first = events.next();
await waitFor(() => FakeWebSocket.instances.length === 3);
const dropped = FakeWebSocket.instances[2];
dropped.open();
await waitFor(() => FakeAudioContext.instances.length === 1);
dropped.fail();
dropped.closedByServer(1006, "connection lost");
await waitFor(() => FakeWebSocket.instances.length === 4);
const reconnected = FakeWebSocket.instances[3];
reconnected.open();
await waitFor(() => FakeAudioContext.instances.length === 2);
reconnected.message(JSON.stringify({ type: "transcript", text: "hi" }));
await first;
await events.return(undefined);
realtime.close();
live.close();

// Closed before its ticket arrived: no socket is opened afterwards.
const socketsBefore = FakeWebSocket.instances.length;
const early = client.realtimeSession();
early.close();
let earlyError = null;
try { await early.ready; } catch (error) { earlyError = error.message; }
await new Promise((resolve) => setTimeout(resolve, 20));
const openedAfterClose = FakeWebSocket.instances.length - socketsBefore;

new OpenSpeechClient({ baseUrl: "http://speech.test" }).realtimeSession();
console.log(JSON.stringify({
  urls: FakeWebSocket.instances.map((ws) => ws.url), requests, earlyError, openedAfterClose,
}));
""")
    ticket_request = {"url": "https://speech.test:8100/api/auth/ws-ticket", "method": "POST", "auth": "Bearer secret"}
    assert result["requests"] == [ticket_request] * 5
    assert result["urls"] == [
        "wss://speech.test:8100/v1/realtime?ticket=t1%2F%2B",
        "wss://speech.test:8100/v1/audio/speech/stream?ticket=t2%2F%2B",
        "wss://speech.test:8100/v1/audio/stream?sample_rate=16000&vad=true&ticket=t3%2F%2B",
        "wss://speech.test:8100/v1/audio/stream?sample_rate=16000&vad=true&ticket=t4%2F%2B",
        "ws://speech.test/v1/realtime",  # no apiKey: no ticket request
    ]
    assert result["earlyError"] == "Realtime session is closed"
    assert result["openedAfterClose"] == 0


def test_js_websocket_helpers_report_a_refused_ticket(tmp_path):
    result = _run_client_js(tmp_path, r"""
globalThis.fetch = async () => new Response("Invalid or missing API key", { status: 401 });
const client = new OpenSpeechClient({ baseUrl: "http://speech.test", apiKey: "wrong" });
const errors = {};
try { await client.realtimeSession().ready; } catch (error) { errors.realtime = error.message; }
try { await client.liveSpeechSession().ready; } catch (error) { errors.live = error.message; }
try { for await (const _event of client.streamTranscribe({}, 0)) { /* none */ } } catch (error) { errors.stream = error.message; }
console.log(JSON.stringify({ errors, sockets: FakeWebSocket.instances.length }));
""")
    assert result["errors"] == {
        "realtime": "WebSocket ticket request failed (401)",
        "live": "WebSocket ticket request failed (401)",
        "stream": "WebSocket ticket request failed (401)",
    }
    assert result["sockets"] == 0


def test_js_stream_transcribe_stops_at_once_when_aborted_while_waiting_for_an_event(tmp_path):
    # A Stop button usually fires during silence, while the consumer waits in `for await` and
    # no event arrives to let a `break` run; the signal must release capture and the socket then.
    result = _run_client_js(tmp_path, r"""
const stop = new AbortController();
const events = new OpenSpeechClient({ baseUrl: "http://example.test" }).streamTranscribe({}, { signal: stop.signal });
const waiting = events.next();
await waitFor(() => FakeWebSocket.instances.length === 1);
const ws = FakeWebSocket.instances[0];
ws.open();
await waitFor(() => FakeAudioContext.instances.length === 1);
const context = FakeAudioContext.instances[0];
stop.abort();
const releasedAtOnce = {
  socketClosed: ws.readyState === FakeWebSocket.CLOSED, contextClosed: context.closed,
  processorDisconnected: context.processor.disconnected === true,
};
// The user keeps talking after Stop.
context.processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array(4) } });
console.log(JSON.stringify({ releasedAtOnce, sent: ws.sent.length, next: await settled(waiting) }));
""")
    assert result == {
        "releasedAtOnce": {"socketClosed": True, "contextClosed": True, "processorDisconnected": True},
        "sent": 0,
        "next": "rejected: AbortError: This operation was aborted",
    }


def test_js_stream_transcribe_aborted_before_it_streams_opens_nothing(tmp_path):
    result = _run_client_js(tmp_path, r"""
globalThis.fetch = (url, init = {}) => new Promise((resolve, reject) => {
  init.signal?.addEventListener("abort", () => reject(init.signal.reason));  // a ticket that never comes
});
const outcomes = {};

// While it waits for its ticket.
let stop = new AbortController();
let events = new OpenSpeechClient({ baseUrl: "http://speech.test", apiKey: "k" }).streamTranscribe({}, { signal: stop.signal });
let waiting = events.next();
await new Promise((resolve) => setTimeout(resolve, 20));
stop.abort();
outcomes.ticket = await settled(waiting);
const socketsAfterTicket = FakeWebSocket.instances.length;

// While its socket connects.
stop = new AbortController();
events = new OpenSpeechClient({ baseUrl: "http://speech.test" }).streamTranscribe({}, { signal: stop.signal });
waiting = events.next();
await waitFor(() => FakeWebSocket.instances.length === 1);
stop.abort();
outcomes.connecting = await settled(waiting);
const connecting = FakeWebSocket.instances[0];

// While it waits to reconnect after a failed open.
stop = new AbortController();
events = new OpenSpeechClient({ baseUrl: "http://speech.test" }).streamTranscribe({}, { reconnectAttempts: 2, signal: stop.signal });
waiting = events.next();
await waitFor(() => FakeWebSocket.instances.length === 2);
FakeWebSocket.instances[1].fail();
await new Promise((resolve) => setTimeout(resolve, 20));
stop.abort();
outcomes.backoff = await settled(waiting, 100);
await new Promise((resolve) => setTimeout(resolve, 400));

console.log(JSON.stringify({
  outcomes, socketsAfterTicket, connectingClosed: connecting.readyState === FakeWebSocket.CLOSED,
  sockets: FakeWebSocket.instances.length, contexts: FakeAudioContext.instances.length,
}));
""")
    aborted = "rejected: AbortError: This operation was aborted"
    assert result == {
        "outcomes": {"ticket": aborted, "connecting": aborted, "backoff": aborted},
        "socketsAfterTicket": 0,
        "connectingClosed": True,
        "sockets": 2,
        "contexts": 0,
    }


def test_js_stream_transcribe_does_not_retry_a_refused_ticket(tmp_path):
    # A wrong key is not transient, and every refused request counts against the server's
    # failed-auth limit (OS_AUTH_FAILURE_LIMIT); only network and server errors are retried.
    result = _run_client_js(tmp_path, r"""
let requests = 0;
globalThis.fetch = async () => { requests++; return new Response("Invalid or missing API key", { status: 401 }); };
const client = new OpenSpeechClient({ baseUrl: "http://speech.test", apiKey: "wrong" });
let refused = null;
try { for await (const _event of client.streamTranscribe({}, 2)) { /* none */ } } catch (error) { refused = error.message; }
const refusedRequests = requests;

requests = 0;
globalThis.fetch = async () => (++requests === 1
  ? new Response("busy", { status: 503 })
  : Response.json({ ticket: "t", expires_in: 60 }));
const stop = new AbortController();
const events = client.streamTranscribe({}, { reconnectAttempts: 2, signal: stop.signal });
const waiting = events.next();
await waitFor(() => FakeWebSocket.instances.length === 1);
stop.abort();
await settled(waiting);
console.log(JSON.stringify({ refused, refusedRequests, retriedRequests: requests, url: FakeWebSocket.instances[0].url }));
""")
    assert result == {
        "refused": "WebSocket ticket request failed (401)",
        "refusedRequests": 1,
        "retriedRequests": 2,
        "url": "ws://speech.test/v1/audio/stream?sample_rate=16000&vad=true&ticket=t",
    }


def test_js_sessions_reject_ready_when_the_socket_cannot_be_created_after_the_ticket(tmp_path):
    # Without a key the WebSocket constructor's error reaches the caller directly; with one the
    # socket is created later, so the error must reject `ready` instead of escaping unhandled.
    result = _run_client_js(tmp_path, r"""
const unhandled = [];
process.on("unhandledRejection", (error) => unhandled.push(String(error)));
globalThis.fetch = async () => Response.json({ ticket: "t", expires_in: 60 });
globalThis.WebSocket = class extends FakeWebSocket {
  constructor(url, protocols) { super(url, protocols); throw new DOMException("insecure", "SecurityError"); }
};
const client = new OpenSpeechClient({ baseUrl: "http://speech.test", apiKey: "k" });
const realtime = await settled(client.realtimeSession().ready);
const live = await settled(client.liveSpeechSession().ready);
console.log(JSON.stringify({ realtime, live, unhandled }));
""")
    assert result == {
        "realtime": "rejected: SecurityError: insecure",
        "live": "rejected: SecurityError: insecure",
        "unhandled": [],
    }


def test_js_live_speech_refused_session_update_closes_the_socket_and_reports_the_error(tmp_path):
    # The server answers a session.update it refuses (an unknown voice, say) with an error and
    # keeps the socket open, holding its Live slot (TTS_LIVE_MAX_CONNECTIONS=1) until
    # TTS_LIVE_IDLE_TIMEOUT_S. `ready` rejects, onEvent still sees the error, and the SDK closes
    # the socket with a code a browser accepts.
    result = _run_client_js(tmp_path, r"""
const live = new OpenSpeechClient({ baseUrl: "http://example.test" }).liveSpeechSession({ voice: "no_such_voice" });
const seen = [];
live.onEvent((event) => seen.push(event.type));
const ws = FakeWebSocket.instances[0];
ws.open();
ws.message(JSON.stringify({ type: "session.created", session: {} }));
const update = JSON.parse(ws.sent[0]).type;
let thrown = null;
try {
  ws.message(JSON.stringify({ type: "error", error: { code: "invalid_event", message: "Named voice is unavailable" } }));
} catch (error) { thrown = `${error.name}: ${error.message}`; }
console.log(JSON.stringify({
  update, thrown, ready: await settled(live.ready), seen, socketClosed: ws.readyState === FakeWebSocket.CLOSED,
}));
""")
    assert result == {
        "update": "session.update",
        "thrown": None,
        "ready": "rejected: Error: Named voice is unavailable",
        "seen": ["session.created", "error"],
        "socketClosed": True,
    }


def test_js_sdk_own_tests_pass_under_node():
    # client-js/tests/index.test.ts, the SDK's own tests, used
    # bun:test, and Bun is on neither CI nor the dev host, so they ran nowhere. Two SDK fixes are
    # guarded only there: transcribe() returns the text formats as strings (client-js/README.md), and
    # RealtimeSession queues sends until its socket opens. Run the file with the node CI sets up,
    # through the capped runner the web UI tests use.
    if _node_running_typescript() is None:
        pytest.skip("needs node 22.18+ (runs TypeScript without a build step)")
    from tests.ui_support import run_node_test

    output = run_node_test("client-js/tests/index.test.ts")
    counts = dict(re.findall(r"^\S* ?(tests|pass) (\d+)\r?$", output, re.M))
    assert int(counts.get("tests", 0)) > 0 and counts.get("pass") == counts["tests"], output[-4000:]


# 400 MB of typed arrays in 10 MB steps: memory outside V8's heap limit, as audio buffers are.
_ALLOCATE_400_MB = r"""
const held = [];
let allocated = 0;
const allocate = () => {
  for (let i = 0; i < 40; i += 1) { held.push(new Uint8Array(10 * 1024 * 1024).fill(1)); allocated += 10; }
};
"""


@pytest.mark.skipif(sys.platform != "win32", reason="the memory cap uses a Windows job object")
@pytest.mark.parametrize("runner", ["_run_client_js", "run_node_script"])
def test_js_node_runs_are_refused_memory_past_their_cap(tmp_path, runner):
    # _run_client_js (the SDK behaviour tests above) and
    # ui_support.run_node_script ran node with subprocess.run, outside the job-object cap that every
    # other node run in the suite has, so a regression that loops while allocating audio buffers could
    # take the 16 GB host down. They go through the capped runner now: an allocation past the cap fails
    # inside node, and an uncaught one fails the test naming the cap.
    from tests.ui_support import run_node_script

    def run(body: str) -> dict:
        if runner == "_run_client_js":
            return _run_client_js(tmp_path, body, memory_limit_mb=256)
        return json.loads(run_node_script(body, memory_limit_mb=256).strip().splitlines()[-1])

    caught = run(_ALLOCATE_400_MB + r"""
let refused = false;
try { allocate(); } catch (error) { refused = `${error.name}: ${error.message}`; }
console.log(JSON.stringify({ allocated_mb: allocated, refused }));
""")
    assert caught["refused"] is not False, f"node allocated {caught['allocated_mb']} MB under a 256 MB cap"
    assert caught["refused"].startswith("RangeError") and caught["allocated_mb"] < 256, caught

    with pytest.raises(pytest.fail.Exception) as failed:
        run(_ALLOCATE_400_MB + "allocate();\nconsole.log(JSON.stringify({ allocated_mb: allocated }));\n")
    assert "memory cap" in str(failed.value)
