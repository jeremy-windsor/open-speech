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
async def test_disconnect_flushes_valid_audio_shorter_than_configured_chunk(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 200)
    audio = b"\x01\x00" * 1600  # 100ms / 3200 bytes: valid, but below chunk_bytes
    ws = DummyWS(messages=[
        {"type": "websocket.receive", "bytes": audio},
        {"type": "websocket.disconnect"},
    ])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())

    session = StreamingSession(
        ws,
        model="m",
        language=None,
        sample_rate=16000,
        interim_results=True,
        endpointing_ms=300,
        vad_enabled=False,
    )
    await session.run()

    final_transcripts = [
        event
        for event in ws.sent
        if event.get("type") == "transcript" and event.get("speech_final") is True
    ]
    assert final_transcripts[-1]["text"] == "hello world"


@pytest.mark.asyncio
async def test_silence_cannot_grow_utterance_past_max_bytes(monkeypatch):
    monkeypatch.setattr(streaming, "MAX_UTTERANCE_BYTES", 3201)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    ws = DummyWS()
    session = StreamingSession(
        ws,
        model="m",
        language=None,
        sample_rate=16000,
        interim_results=True,
        endpointing_ms=999_999,
        vad_enabled=True,
    )
    session.vad_state = _VADState([0.0])
    session.speech_active = True
    session.utterance_audio = bytearray(3200)
    session._chunk_count = 3

    await session._process_chunk(b"\x00\x00" * 512)

    final_transcripts = [
        event
        for event in ws.sent
        if event.get("type") == "transcript" and event.get("speech_final") is True
    ]
    assert [event["text"] for event in final_transcripts] == ["hello world"]
    assert session.speech_active is False
    assert session.silence_samples == 0
    assert session.utterance_audio == b""


@pytest.mark.asyncio
async def test_vad_processes_mixed_windows_individually(monkeypatch):
    ws = DummyWS()
    session = StreamingSession(
        ws,
        model="m",
        language=None,
        sample_rate=16000,
        interim_results=True,
        endpointing_ms=64,
        vad_enabled=True,
    )
    session.vad_state = _VADState([0.9, 0.0, 0.0])
    session._chunk_count = 3

    async def skip_transcription(*_args, **_kwargs):
        return None

    session._transcribe_utterance = skip_transcription

    await session._process_chunk(b"\x01\x00" * (3 * 512))

    states = [event["state"] for event in ws.sent if event["type"] == "vad"]
    assert states == ["speech_start", "speech_end"]
    assert session.speech_active is False


@pytest.mark.asyncio
async def test_vad_final_transcript_ends_at_window_boundary(monkeypatch):
    ws = DummyWS()
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    session = StreamingSession(
        ws,
        model="m",
        language=None,
        sample_rate=16000,
        interim_results=True,
        endpointing_ms=64,
        vad_enabled=True,
    )
    session.vad_state = _VADState([0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.9])
    session._chunk_count = 3
    session.total_samples = 7 * 512  # receive loop has already accepted the whole frame

    await session._process_chunk(b"\x01\x00" * (7 * 512))

    final_transcripts = [
        event
        for event in ws.sent
        if event.get("type") == "transcript" and event.get("speech_final") is True
    ]
    assert len(final_transcripts) == 1
    assert final_transcripts[0]["start"] == 0.0
    assert final_transcripts[0]["end"] == pytest.approx(6 * 512 / 16000)
    assert session.speech_active is True
    assert session.utterance_start == pytest.approx(6 * 512 / 16000)


@pytest.mark.asyncio
async def test_vad_carries_unaligned_chunks_and_flushes_active_tail():
    ws = DummyWS()
    session = StreamingSession(
        ws,
        model="m",
        language=None,
        sample_rate=16000,
        interim_results=True,
        endpointing_ms=300,
        vad_enabled=True,
    )
    window_lengths = []

    def always_speech(samples):
        window_lengths.append(len(samples))
        return 0.9

    session.vad_state = always_speech
    session._chunk_count = 3
    finalized_audio = []

    async def skip_transcription(*_args, **_kwargs):
        return None

    async def record_finalize(*_args, **_kwargs):
        finalized_audio.append(bytes(session.utterance_audio))

    session._transcribe_utterance = skip_transcription
    session._finalize_utterance = record_finalize

    chunk = b"\x01\x00" * 1600
    for _ in range(3):
        await session._process_chunk(chunk)
    await session._flush()

    assert window_lengths == [512] * 9
    assert finalized_audio == [chunk * 3]
    assert session._vad_buffer == b""


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
