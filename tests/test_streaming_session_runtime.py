from __future__ import annotations

import json

import pytest

import src.streaming as streaming
from src.streaming import StreamingSession, streaming_endpoint


class DummyWS:
    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.sent = []
        self.accepted = False
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def close(self, code: int, reason: str):
        self.closed = (code, reason)

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))


class _BackendOK:
    def is_model_loaded(self, _model):
        return True

    def load_model(self, _model):
        return None

    def transcribe(self, **_kwargs):
        return {"text": "hello world"}


class _BackendErr(_BackendOK):
    def transcribe(self, **_kwargs):
        raise RuntimeError("boom")


class _FakeVADModel:
    session = object()


class _VADState:
    def __init__(self, probs):
        self._probs = iter(probs)

    def __call__(self, _samples):
        return next(self._probs, 0.0)


@pytest.mark.asyncio
async def test_streaming_endpoint_rejects_invalid_sample_rate():
    ws = DummyWS()
    await streaming_endpoint(ws, sample_rate=1000)
    assert ws.closed[0] == 1008


@pytest.mark.asyncio
async def test_streaming_endpoint_rejects_when_max_connections_reached(monkeypatch):
    ws = DummyWS()
    monkeypatch.setattr(streaming.settings, "os_stream_max_connections", 0)
    await streaming_endpoint(ws)
    assert ws.closed[0] == 1013


@pytest.mark.asyncio
async def test_streaming_endpoint_accepts_and_tracks_session(monkeypatch):
    ws = DummyWS(messages=[{"type": "websocket.receive", "text": '{"type":"stop"}'}])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    monkeypatch.setattr(streaming, "get_vad_model", lambda: _FakeVADModel())
    monkeypatch.setattr(streaming, "SileroVAD", lambda *_a, **_k: _VADState([0.0]))

    await streaming_endpoint(ws, vad=False)

    assert ws.accepted is True
    assert ws.sent[0]["type"] == "session.begin"
    assert ws.sent[-1]["type"] == "session.end"
    assert len(streaming._active_sessions) == 0


@pytest.mark.asyncio
async def test_session_stop_message_ends_with_client_stop(monkeypatch):
    ws = DummyWS(messages=[{"type": "websocket.receive", "text": '{"type":"stop"}'}])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())

    s = StreamingSession(ws, model="m", language=None, sample_rate=16000, interim_results=True, endpointing_ms=300, vad_enabled=False)
    await s.run()

    assert ws.sent[-1]["type"] == "session.end"
    assert ws.sent[-1]["reason"] == "client_stop"


@pytest.mark.asyncio
async def test_session_disconnect_ends_with_disconnect(monkeypatch):
    ws = DummyWS(messages=[{"type": "websocket.disconnect"}])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())

    s = StreamingSession(ws, model="m", language=None, sample_rate=16000, interim_results=True, endpointing_ms=300, vad_enabled=False)
    await s.run()

    assert ws.sent[-1]["reason"] == "disconnect"


@pytest.mark.asyncio
async def test_vad_emits_speech_start_and_end(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    chunk = (b"\x01\x00" * 3200)
    ws = DummyWS(messages=[
        {"type": "websocket.receive", "bytes": chunk},
        {"type": "websocket.receive", "bytes": chunk},
        {"type": "websocket.disconnect"},
    ])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    async def _get_vad_model():
        return _FakeVADModel()

    monkeypatch.setattr(streaming, "get_vad_model", _get_vad_model)
    monkeypatch.setattr(streaming, "SileroVAD", lambda *_a, **_k: _VADState([0.9, 0.0]))

    s = StreamingSession(ws, model="m", language=None, sample_rate=16000, interim_results=True, endpointing_ms=100, vad_enabled=True)
    await s.run()

    states = [e.get("state") for e in ws.sent if e.get("type") == "vad"]
    assert "speech_start" in states
    assert "speech_end" in states


@pytest.mark.asyncio
async def test_disconnect_mid_stream_flushes_final_transcript(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    chunk = (b"\x01\x00" * 3200)
    ws = DummyWS(messages=[
        {"type": "websocket.receive", "bytes": chunk},
        {"type": "websocket.disconnect"},
    ])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())

    s = StreamingSession(ws, model="m", language=None, sample_rate=16000, interim_results=True, endpointing_ms=300, vad_enabled=False)
    await s.run()

    transcripts = [e for e in ws.sent if e.get("type") == "transcript"]
    assert transcripts
    assert ws.sent[-1]["type"] == "session.end"
    assert ws.sent[-1]["reason"] == "disconnect"


@pytest.mark.asyncio
async def test_transcription_error_propagates_as_error_event(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    chunk = (b"\x01\x00" * 3200)
    ws = DummyWS(messages=[
        {"type": "websocket.receive", "bytes": chunk},
        {"type": "websocket.disconnect"},
    ])
    monkeypatch.setattr(streaming, "backend_router", _BackendErr())

    s = StreamingSession(ws, model="m", language=None, sample_rate=16000, interim_results=True, endpointing_ms=300, vad_enabled=False)
    await s.run()

    errors = [e for e in ws.sent if e.get("type") == "error"]
    assert errors and "Transcription failed" in errors[0]["message"]


@pytest.mark.asyncio
async def test_model_load_failure_emits_error(monkeypatch):
    class _BackendLoadFail(_BackendOK):
        def is_model_loaded(self, _model):
            return False

        def load_model(self, _model):
            raise RuntimeError("load failed")

    ws = DummyWS(messages=[])
    monkeypatch.setattr(streaming, "backend_router", _BackendLoadFail())

    s = StreamingSession(ws, model="m", language=None, sample_rate=16000, interim_results=True, endpointing_ms=300, vad_enabled=False)
    await s.run()

    assert ws.sent[-1]["type"] == "error"
    assert "Failed to load model" in ws.sent[-1]["message"]


@pytest.mark.asyncio
async def test_malformed_json_text_frame_is_ignored(monkeypatch):
    ws = DummyWS(messages=[
        {"type": "websocket.receive", "text": "{not-json"},
        {"type": "websocket.disconnect"},
    ])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())

    s = StreamingSession(ws, model="m", language=None, sample_rate=16000, interim_results=True, endpointing_ms=300, vad_enabled=False)
    await s.run()

    assert ws.sent[0]["type"] == "session.begin"
    assert ws.sent[-1]["type"] == "session.end"
