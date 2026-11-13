from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
import wave

import numpy as np
import pytest

import src.streaming as streaming
from src.audio.resample import resample
from src.streaming import StreamingSession, streaming_endpoint


class DummyWS:
    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.sent = []
        self.accepted = False
        self.closed = None
        self.scope = {"type": "websocket", "subprotocols": []}  # a Starlette WebSocket's; refuse_websocket reads it

    async def accept(self, subprotocol=None):
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
    # Accepted first, then closed, so a browser gets the code and reason.
    ws = DummyWS()
    await streaming_endpoint(ws, sample_rate=1000)
    assert ws.accepted
    assert ws.closed == (1008, "Invalid sample_rate: must be 8000-192000")
    assert ws.sent == []


# /v1/audio/stream refused an invalid
# sample_rate by closing with 1008 before accept(), so a browser saw only 1006 and no reason. The socket must open,
# then close at once with 1008 and the reason, with no event sent and no session started.
@pytest.mark.parametrize("sample_rate", [1000, 7999, 192001])
def test_an_invalid_sample_rate_opens_the_stream_then_closes_it_with_1008_and_the_reason(monkeypatch, sample_rate):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from src import main as main_module

    started = []

    class _NoSession:
        def __init__(self, *args, **kwargs):
            started.append(kwargs)
            raise AssertionError("a refused stream must not start a session")

    monkeypatch.setattr(streaming, "StreamingSession", _NoSession)
    monkeypatch.setattr(main_module.settings, "os_api_key", "")

    url = f"/v1/audio/stream?sample_rate={sample_rate}"
    # websocket_connect returns only once the socket is accepted; the first receive then gets the close.
    with (
        TestClient(main_module.app).websocket_connect(url) as websocket,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        websocket.receive_json()

    assert (closed.value.code, closed.value.reason) == (1008, "Invalid sample_rate: must be 8000-192000")
    assert started == []
    assert len(streaming._active_sessions) == 0


@pytest.mark.parametrize("encoding", ["opus", "mulaw", "PCM_S16LE", ""])
@pytest.mark.asyncio
async def test_streaming_endpoint_refuses_an_unsupported_encoding_before_reading_audio(monkeypatch, encoding):
    """Only PCM16 (encoding=pcm_s16le) is documented. Other encodings were ignored and
    their bytes transcribed as PCM16 noise."""
    ws = DummyWS(messages=[{"type": "websocket.receive", "bytes": b"\x00\x01" * 800}])
    monkeypatch.setattr(streaming, "backend_router", _BackendErr())

    await streaming_endpoint(ws, encoding=encoding, vad=False)

    assert ws.closed == (1003, "Unsupported encoding")
    assert [event["type"] for event in ws.sent] == ["error"]
    assert "pcm_s16le" in ws.sent[0]["message"]
    assert len(ws._messages) == 1  # the audio frame was never read
    assert len(streaming._active_sessions) == 0


@pytest.mark.parametrize("encoding", ["pcm_s16le", "linear16"])
@pytest.mark.asyncio
async def test_streaming_endpoint_accepts_pcm16_encodings(monkeypatch, encoding):
    ws = DummyWS(messages=[{"type": "websocket.receive", "text": '{"type":"stop"}'}])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())

    await streaming_endpoint(ws, encoding=encoding, vad=False)

    assert ws.closed == (1000, "Stream ended")  # a normal end
    assert [event["type"] for event in ws.sent] == ["session.begin", "session.end"]


@pytest.mark.asyncio
async def test_streaming_endpoint_rejects_when_max_connections_reached(monkeypatch):
    ws = DummyWS()
    monkeypatch.setattr(streaming.settings, "os_stream_max_connections", 0)
    await streaming_endpoint(ws)
    assert ws.closed[0] == 1013


class _UnansweredCloseWS(DummyWS):
    """A socket whose close() waits until the test lets it end, as the close handshake with a client that doesn't
    answer it does (uvicorn's websockets protocol waits up to 10 s for the close frame, then 10 s for the TCP end)."""

    def __init__(self):
        super().__init__()
        self.closing = asyncio.Event()
        self.answered = asyncio.Event()

    async def close(self, code: int, reason: str):
        self.closing.set()
        await self.answered.wait()
        self.closed = (code, reason)


@pytest.mark.asyncio
async def test_a_refused_streams_close_handshake_does_not_hold_the_session_lock(monkeypatch):
    """Over OS_STREAM_MAX_CONNECTIONS a new stream is closed with 1013, but the
    close runs after the session-count lock is released. Before, it ran inside it, so a client that didn't answer the
    close kept every session from starting or giving its slot back for up to about 20 s."""
    monkeypatch.setattr(streaming.settings, "os_stream_max_connections", 0)
    ws = _UnansweredCloseWS()
    refusal = asyncio.create_task(streaming_endpoint(ws))
    try:
        await asyncio.wait_for(ws.closing.wait(), 5)
        held_while_closing = streaming._active_sessions_lock.locked()
    finally:
        ws.answered.set()
        await asyncio.wait_for(refusal, 5)

    assert not held_while_closing, "the session-count lock was held through the refused socket's close handshake"
    assert ws.closed == (1013, "Too many concurrent streams")
    assert len(streaming._active_sessions) == 0


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


class _RecordingBackend(_BackendOK):
    def __init__(self):
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return super().transcribe(**kwargs)


def _session(ws, **overrides):
    kwargs = dict(
        model="m",
        language=None,
        sample_rate=16000,
        interim_results=True,
        endpointing_ms=300,
        vad_enabled=False,
    )
    kwargs.update(overrides)
    return StreamingSession(ws, **kwargs)


def _audio(data: bytes) -> dict:
    return {"type": "websocket.receive", "bytes": data}


def _text(data: str) -> dict:
    return {"type": "websocket.receive", "text": data}


@pytest.mark.parametrize("frame", ["null", "[]", '"stop"', "3"])
@pytest.mark.asyncio
async def test_non_object_json_text_frame_is_ignored(monkeypatch, frame):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    ws = DummyWS(messages=[_text(frame), _audio(b"\x01\x00" * 3200), _text('{"type":"stop"}')])

    await _session(ws).run()

    finals = [e for e in ws.sent if e.get("type") == "transcript" and e.get("speech_final")]
    assert [e["text"] for e in finals] == ["hello world"]
    assert ws.sent[-1]["reason"] == "client_stop"


@pytest.mark.asyncio
async def test_odd_length_frames_carry_the_split_sample_into_the_next_frame(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    session = _session(DummyWS())
    processed = []

    async def record(chunk):
        processed.append(chunk)

    session._process_chunk = record
    pcm = np.arange(1, 1601, dtype=np.int16).tobytes()  # exactly one 100 ms chunk

    for piece in (pcm[:3], pcm[3:1001], pcm[1001:]):  # 3, 998 and 2199 bytes
        await session._handle_audio(piece)

    assert processed == [pcm]
    assert session.total_samples == 1600


@pytest.mark.asyncio
async def test_vad_off_transcript_times_follow_processed_audio(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    ws = DummyWS(messages=[_audio(b"\x01\x00" * 4800), _text('{"type":"stop"}')])  # three 100 ms chunks

    await _session(ws).run()

    transcripts = [e for e in ws.sent if e.get("type") == "transcript"]
    assert transcripts and all(e["start"] == 0.0 for e in transcripts)
    first_interim = next(e for e in transcripts if not e["is_final"])
    assert first_interim["end"] == pytest.approx(0.1)  # only the first chunk had been transcribed
    assert transcripts[-1]["speech_final"] and transcripts[-1]["end"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_config_message_changes_the_client_sample_rate(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    backend = _RecordingBackend()
    monkeypatch.setattr(streaming, "backend_router", backend)
    ws = DummyWS(messages=[
        _text(json.dumps({"type": "config", "sample_rate": 8000})),
        _audio(b"\x01\x00" * 8000),  # 1 s at 8 kHz
        _text('{"type":"stop"}'),
    ])

    await _session(ws, sample_rate=16000).run()

    final = [e for e in ws.sent if e.get("type") == "transcript" and e.get("speech_final")][-1]
    assert final["end"] == pytest.approx(1.0, abs=0.02)
    with wave.open(io.BytesIO(backend.calls[-1]["audio"]), "rb") as wav_file:
        assert wav_file.getnframes() == pytest.approx(16000, abs=320)


@pytest.mark.parametrize("sample_rate", [1000, 16000.0, True, "16000"])
@pytest.mark.asyncio
async def test_config_message_rejects_an_invalid_sample_rate(monkeypatch, sample_rate):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    ws = DummyWS(messages=[_text(json.dumps({"type": "config", "sample_rate": sample_rate}))])
    session = _session(ws)

    await session.run()

    assert [e["message"] for e in ws.sent if e["type"] == "error"] == [
        "Invalid sample_rate: must be 8000-192000"
    ]
    assert session.client_sample_rate == 16000


@pytest.mark.parametrize("language", ["klingon", "xx-YY"])
@pytest.mark.asyncio
async def test_unknown_language_is_reported_and_the_stream_transcribes_with_auto_detect(monkeypatch, language):
    """faster-whisper checks the language only after decoding, so every transcription of a
    stream opened with an unknown code failed."""
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    backend = _RecordingBackend()
    monkeypatch.setattr(streaming, "backend_router", backend)
    ws = DummyWS(messages=[_audio(b"\x01\x00" * 3200), _text('{"type":"stop"}')])

    await streaming_endpoint(ws, language=language, vad=False)

    # An error before session.begin means the stream failed to start (model or VAD load),
    # so this one follows it, says the stream goes on, and session.begin shows the language used.
    assert [e["type"] for e in ws.sent[:2]] == ["session.begin", "error"]
    assert ws.sent[0]["language"] is None
    assert [e["message"] for e in ws.sent if e["type"] == "error"] == [
        f"Invalid language {language!r}. Use an ISO-639-1 code such as 'en'. "
        "The stream continues with auto-detect."
    ]
    assert backend.calls and all(call["language"] is None for call in backend.calls)
    finals = [e for e in ws.sent if e.get("type") == "transcript" and e.get("speech_final")]
    assert [e["text"] for e in finals] == ["hello world"]


@pytest.mark.parametrize(("language", "expected"), [
    ("en-US", "en"), ("DE", "de"), ("iw", "he"), ("auto", None), ("", None), (None, None),
])
@pytest.mark.asyncio
async def test_stream_language_reaches_the_backend_as_a_whisper_code(monkeypatch, language, expected):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    backend = _RecordingBackend()
    monkeypatch.setattr(streaming, "backend_router", backend)
    ws = DummyWS(messages=[_audio(b"\x01\x00" * 3200), _text('{"type":"stop"}')])

    await streaming_endpoint(ws, language=language, vad=False)

    assert [e for e in ws.sent if e["type"] == "error"] == []
    assert ws.sent[0]["type"] == "session.begin" and ws.sent[0]["language"] == expected
    assert backend.calls and all(call["language"] == expected for call in backend.calls)


@pytest.mark.asyncio
async def test_vad_model_load_failure_emits_error(monkeypatch):
    async def failing_vad():
        raise RuntimeError("Silero download blocked")

    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    monkeypatch.setattr(streaming, "get_vad_model", failing_vad)
    ws = DummyWS(messages=[_text('{"type":"stop"}')])

    await _session(ws, vad_enabled=True).run()

    assert ws.sent[-1]["type"] == "error"
    assert "Failed to load VAD model" in ws.sent[-1]["message"]


@pytest.mark.asyncio
async def test_final_transcription_error_emits_error_event(monkeypatch):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)

    class FinalFails(_RecordingBackend):
        def transcribe(self, **kwargs):
            if self.calls:
                raise RuntimeError("CUDA out of memory")
            return super().transcribe(**kwargs)

    monkeypatch.setattr(streaming, "backend_router", FinalFails())
    ws = DummyWS(messages=[_audio(b"\x01\x00" * 1600), _text('{"type":"stop"}')])

    await _session(ws).run()

    errors = [e for e in ws.sent if e.get("type") == "error"]
    assert errors and "Transcription failed" in errors[0]["message"]


@pytest.mark.parametrize("chunk_ms", [0, -100])
@pytest.mark.asyncio
async def test_non_positive_chunk_window_still_consumes_each_frame(monkeypatch, chunk_ms):
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", chunk_ms)
    session = _session(DummyWS())
    processed = []

    async def record(chunk):
        processed.append(chunk)
        assert len(processed) < 100, "frame is never consumed"

    session._process_chunk = record

    await session._handle_audio(b"\x01\x00" * 1600)

    assert [len(chunk) for chunk in processed] == [streaming.VAD_WINDOW_BYTES] * 3


@pytest.mark.asyncio
async def test_a_rate_coprime_with_16k_is_resampled_without_stalling_the_event_loop(monkeypatch):
    """Frames are resampled on the event loop, and 191999 Hz is an accepted rate. With the
    exact 16000/191999 ratio the stream designed a filter of millions of taps first (about
    a second here), and every other request waited for it."""
    import scipy.signal  # noqa: F401  - keep the one-time import out of the measurement

    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    resampled = []

    async def record(chunk_16k, transcribe=True):
        resampled.append(chunk_16k)

    started = time.perf_counter()
    session = _session(DummyWS(), sample_rate=191_999)
    session._process_16k = record
    for _ in range(3):  # 100 ms frames
        await session._handle_audio(np.zeros(19_199, dtype=np.int16).tobytes())
    elapsed = time.perf_counter() - started

    assert len(resampled) == 3
    # 300 ms at 16 kHz, less the resampler's lookahead (about 10 samples) it still holds back
    assert 4800 - 16 <= len(b"".join(resampled)) // 2 <= 4800
    assert elapsed < 0.25, f"three 100 ms frames at 191999 Hz took {elapsed:.2f} s"


@pytest.mark.asyncio
async def test_resampled_chunks_join_without_a_click_at_each_chunk_end(monkeypatch):
    """Each 100 ms chunk used to be resampled on its own, so 8 kHz telephony audio
    reached VAD and STT with a spike of up to a quarter of the peak at every chunk end."""
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    backend = _RecordingBackend()
    monkeypatch.setattr(streaming, "backend_router", backend)
    t = np.arange(8000) / 8000  # 1 s at 8 kHz
    tones = sum(np.sin(2 * np.pi * hz * t) for hz in (300, 1000, 2000))
    pcm = (6000 * tones).astype(np.int16)
    ws = DummyWS(messages=[_audio(pcm.tobytes()), _text('{"type":"stop"}')])

    await _session(ws, sample_rate=8000).run()

    with wave.open(io.BytesIO(backend.calls[-1]["audio"]), "rb") as wav_file:
        streamed = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    whole = resample(pcm, 8000, 16000)
    assert len(streamed) == len(whole) == 16000
    assert np.max(np.abs(streamed - whole)) <= 1  # rounding only


@pytest.mark.asyncio
async def test_config_repeating_the_current_sample_rate_keeps_the_stream_seamless(monkeypatch):
    """A config message with the rate already in use restarted the resampler from
    silence, which put a click into the audio at that point."""
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    backend = _RecordingBackend()
    monkeypatch.setattr(streaming, "backend_router", backend)
    t = np.arange(8000) / 8000  # 1 s at 8 kHz
    tones = sum(np.sin(2 * np.pi * hz * t) for hz in (300, 1000, 2000))
    pcm = (6000 * tones).astype(np.int16).tobytes()
    ws = DummyWS(messages=[
        _audio(pcm[:7000]),  # 437.5 ms, so part of a chunk is still buffered
        _text(json.dumps({"type": "config", "sample_rate": 8000})),
        _audio(pcm[7000:]),
        _text('{"type":"stop"}'),
    ])

    await _session(ws, sample_rate=8000).run()

    with wave.open(io.BytesIO(backend.calls[-1]["audio"]), "rb") as wav_file:
        streamed = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    whole = resample(np.frombuffer(pcm, dtype=np.int16), 8000, 16000)
    assert len(streamed) == len(whole) == 16000
    assert np.max(np.abs(streamed - whole)) <= 1  # rounding only


@pytest.mark.asyncio
async def test_interim_and_final_decodes_run_once_without_temperature_fallback(monkeypatch):
    # Each live window decodes once at temperature 0. With faster-whisper's
    # hotter retries a window cut mid-word, which often fails the thresholds, cost up to six.
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    backend = _RecordingBackend()
    monkeypatch.setattr(streaming, "backend_router", backend)
    ws = DummyWS(messages=[_audio(b"\x01\x00" * 16000), _text('{"type":"stop"}')])

    await _session(ws).run()

    assert len(backend.calls) >= 2, backend.calls  # interim decodes, then the final one
    assert {(call["temperature"], call["temperature_fallback"]) for call in backend.calls} == {(0.0, False)}


# An utterance started at the first window over STT_VAD_THRESHOLD, so a
# word's onset, which Silero scores below the threshold while it is already audible, never reached
# Whisper. Speech keeps a 300 ms lead-in, as /v1/realtime's default prefix_padding_ms does, but none
# from before the previous utterance's end.
PREROLL_SAMPLES = 300 * 16000 // 1000


def _wav_samples(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert (wav_file.getframerate(), wav_file.getsampwidth(), wav_file.getnchannels()) == (16000, 2, 1)
        return np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)


@pytest.mark.asyncio
async def test_vad_utterance_keeps_300_ms_before_its_first_speech_window(monkeypatch):
    backend = _RecordingBackend()
    monkeypatch.setattr(streaming, "backend_router", backend)
    monkeypatch.setattr(streaming.settings, "stt_vad_threshold", 0.5)
    # 12 idle windows (384 ms) ending in an onset scored under the threshold, as Silero scores real
    # speech; an utterance of 4 speech windows and the 64 ms endpointing silence; one idle window;
    # a second utterance.
    probs = [0.01] * 10 + [0.41, 0.39] + [0.97] * 4 + [0.0] * 2 + [0.0] + [0.97] * 4 + [0.0] * 2
    stream = np.arange(len(probs) * 512, dtype=np.int16)  # sample i holds i
    session = _session(DummyWS(), vad_enabled=True, endpointing_ms=64)
    session.vad_state = _VADState(probs)
    session._chunk_count = 3

    await asyncio.wait_for(session._process_chunk(stream.tobytes()), timeout=60)

    trigger, first_end = 12 * 512, 18 * 512
    assert len(backend.calls) == 2, backend.calls  # the two final decodes
    first, second = (_wav_samples(call["audio"]) for call in backend.calls)
    assert int(first[0]) == trigger - PREROLL_SAMPLES, (
        f"the utterance Whisper got starts at sample {int(first[0])} and the first window over the "
        f"threshold at {trigger}: the onset before it was cut, not kept for 300 ms ({PREROLL_SAMPLES} samples)"
    )
    np.testing.assert_array_equal(first, stream[trigger - PREROLL_SAMPLES:first_end])
    # Only one idle window separates the utterances: the second one's lead-in stops at the first one's end.
    np.testing.assert_array_equal(second, stream[first_end:])
    finals = [e for e in session.ws.sent if e.get("type") == "transcript" and e.get("speech_final")]
    assert [(e["start"], e["end"]) for e in finals] == [
        pytest.approx(((trigger - PREROLL_SAMPLES) / 16000, first_end / 16000)),
        pytest.approx((first_end / 16000, len(stream) / 16000)),
    ]


# At INFO a session logged only "started" and "ended
# (transcriptions=0, errors=0)", so a client that sends silence looked the same as one whose speech the VAD
# rejects, or whose speech is transcribed to nothing. The "ended" line now also says how much audio arrived,
# its peak level, the highest VAD score and how often speech started: numbers only, never audio or text.
def _ended_line(caplog) -> str:
    lines = [
        record.getMessage() for record in caplog.records
        if record.name == "src.streaming" and record.levelno == logging.INFO
        and re.match(r"Streaming session \S+ ended \(", record.getMessage())
    ]
    assert len(lines) == 1, lines
    return lines[0]


def _clip(samples: int, values: dict[int, int] | None = None) -> bytes:
    """``samples`` of digital silence, with the given sample values put in."""
    pcm = np.zeros(samples, dtype=np.int16)
    for index, value in (values or {}).items():
        pcm[index] = value
    return pcm.tobytes()


async def _fake_vad_model():
    return _FakeVADModel()


@pytest.mark.parametrize(("pcm", "probs", "levels"), [
    (_clip(16000), [], "audio=1.0 s, peak=silent, vad_max=0.00, speech=0"),
    (_clip(16000, {100: 1000, 200: -1000}), [0.12] * 31, "audio=1.0 s, peak=-30.3 dBFS, vad_max=0.12, speech=0"),
], ids=["a-client-sending-zeros", "sound-the-vad-scores-below-the-threshold"])
@pytest.mark.asyncio
async def test_ended_line_tells_a_silent_client_from_sound_the_vad_rejects(monkeypatch, caplog, pcm, probs, levels):
    caplog.set_level(logging.INFO, logger="src.streaming")
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming.settings, "stt_vad_threshold", 0.5)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    monkeypatch.setattr(streaming, "get_vad_model", _fake_vad_model)
    monkeypatch.setattr(streaming, "SileroVAD", lambda *_a, **_k: _VADState(probs))
    ws = DummyWS(messages=[_audio(pcm), _text('{"type":"stop"}')])

    await asyncio.wait_for(streaming_endpoint(ws, vad=True), timeout=60)

    line = _ended_line(caplog)
    assert line.endswith(f" ended (transcriptions=0, errors=0, {levels})"), line


@pytest.mark.asyncio
async def test_ended_line_counts_speech_starts_and_leaves_the_transcript_out(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.streaming")
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming.settings, "stt_vad_threshold", 0.5)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    monkeypatch.setattr(streaming, "get_vad_model", _fake_vad_model)
    # Two utterances, each ended by the 64 ms endpointing silence, in 2 s of audio
    probs = [0.1] * 5 + [0.93] * 10 + [0.0] * 10 + [0.88] * 10 + [0.0] * 27
    monkeypatch.setattr(streaming, "SileroVAD", lambda *_a, **_k: _VADState(probs))
    ws = DummyWS(messages=[_audio(_clip(32000, {6000: -16384, 7000: 9000})), _text('{"type":"stop"}')])

    await asyncio.wait_for(streaming_endpoint(ws, vad=True, endpointing=64), timeout=60)

    finals = [e["text"] for e in ws.sent if e.get("type") == "transcript" and e.get("speech_final")]
    assert finals == ["hello world", "hello world"]
    line = _ended_line(caplog)
    assert re.search(
        r" ended \(transcriptions=[1-9]\d*, errors=0, audio=2\.0 s, peak=-6\.0 dBFS, vad_max=0\.93, speech=2\)$", line,
    ), line
    assert "hello" not in line.lower()


@pytest.mark.asyncio
async def test_ended_line_counts_audio_at_the_clients_rate_and_a_full_scale_negative_peak(monkeypatch, caplog):
    """Without the VAD there is no score (vad_max=none). The seconds follow each frame's client rate, and
    -32768, which has no int16 absolute value, is a 0 dBFS peak."""
    caplog.set_level(logging.INFO, logger="src.streaming")
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    ws = DummyWS(messages=[
        _audio(_clip(4000, {10: -32768})),  # 0.5 s at 8 kHz
        _text(json.dumps({"type": "config", "sample_rate": 16000})),
        _audio(_clip(8000, {10: 120})),  # 0.5 s at 16 kHz
        _text('{"type":"stop"}'),
    ])

    await asyncio.wait_for(streaming_endpoint(ws, sample_rate=8000, vad=False), timeout=60)

    line = _ended_line(caplog)
    assert re.search(
        r" ended \(transcriptions=[1-9]\d*, errors=0, audio=1\.0 s, peak=0\.0 dBFS, vad_max=none, speech=0\)$", line,
    ), line


@pytest.mark.asyncio
async def test_the_ended_line_numbers_keep_no_audio(monkeypatch):
    """The numbers cost O(1) per frame. The session keeps a few numbers for them, never audio."""
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    frames = [_audio(_clip(16000, {5: 700})) for _ in range(5)]
    session = _session(DummyWS(messages=[*frames, _text('{"type":"stop"}')]))

    await asyncio.wait_for(session.run(), timeout=60)

    assert session.levels.summary() == "audio=5.0 s, peak=-33.4 dBFS, vad_max=none, speech=0"
    kept = vars(session.levels)
    assert all(value is None or type(value) in (int, float) for value in kept.values()), kept


# The session logged what the listener said at INFO, the level
# src/main.py gives every logger ("Transcription #N: '...'", "Confirmed: '...'", "Final utterance: '...'"), so the
# words of every Transcribe session reached the server log and `docker logs`. HTTP transcriptions, Wyoming and
# /v1/realtime keep them out. INFO now gives sizes only, and the words are on a DEBUG line.
SAID = "tell nobody the door code is four four one seven"  # 48 chars, 10 words
SAID_WORDS = ("nobody", "door code", "four four one seven")


class _HearsSaid(_RecordingBackend):
    def transcribe(self, **kwargs):
        super().transcribe(**kwargs)
        return {"text": SAID}


def _streaming_lines(caplog, level: int) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == "src.streaming" and record.levelno == level
    ]


@pytest.mark.parametrize("vad", [False, True], ids=["vad-off", "vad-on"])
@pytest.mark.asyncio
async def test_stream_keeps_what_was_said_out_of_the_info_log(monkeypatch, caplog, vad):
    caplog.set_level(logging.INFO)  # the level src/main.py gives every logger
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming.settings, "stt_vad_threshold", 0.5)
    monkeypatch.setattr(streaming, "backend_router", _HearsSaid())
    monkeypatch.setattr(streaming, "get_vad_model", _fake_vad_model)
    # Speech for 15 windows (480 ms), then the 64 ms endpointing silence ends the utterance
    monkeypatch.setattr(streaming, "SileroVAD", lambda *_a, **_k: _VADState([0.95] * 15 + [0.0] * 20))
    frame = _clip(3200, {index: 4000 for index in range(0, 3200, 50)})  # 200 ms at 16 kHz
    ws = DummyWS(messages=[_audio(frame), _audio(frame), _audio(frame), _text('{"type":"stop"}')])

    await asyncio.wait_for(streaming_endpoint(ws, vad=vad, endpointing=64), timeout=60)

    # The client gets the words, as it should
    assert any(e.get("type") == "transcript" and e.get("text") == SAID for e in ws.sent), ws.sent
    # ...and no log line at INFO or above, from any logger, has them
    leaked = [
        f"{record.levelname} {record.name}: {record.getMessage()}" for record in caplog.records
        if record.levelno >= logging.INFO and any(word in record.getMessage() for word in SAID_WORDS)
    ]
    assert leaked == []
    # The INFO lines keep the sizes
    info = _streaming_lines(caplog, logging.INFO)
    for pattern in (
        r"\[[0-9a-f]{8}\] Transcription #1: 48 chars \(\d+\.\ds audio\)",
        r"\[[0-9a-f]{8}\] Confirmed: 10 words",
        r"\[[0-9a-f]{8}\] Final utterance: 48 chars \(\d+\.\ds audio\)",
    ):
        assert any(re.fullmatch(pattern, line) for line in info), (pattern, info)


@pytest.mark.asyncio
async def test_stream_logs_the_words_at_debug_and_an_empty_decode_as_0_chars(monkeypatch, caplog):
    """The words move to DEBUG rather than out of the log, and the INFO line still shows a
    decode that came back empty."""
    caplog.set_level(logging.DEBUG, logger="src.streaming")
    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)

    class FirstDecodeEmpty(_RecordingBackend):
        def transcribe(self, **kwargs):
            super().transcribe(**kwargs)
            return {"text": "" if len(self.calls) == 1 else SAID}

    monkeypatch.setattr(streaming, "backend_router", FirstDecodeEmpty())
    ws = DummyWS(messages=[_audio(b"\x01\x00" * 1600) for _ in range(3)] + [_text('{"type":"stop"}')])

    await asyncio.wait_for(_session(ws).run(), timeout=60)

    info = _streaming_lines(caplog, logging.INFO)
    assert any(re.fullmatch(r"\[[0-9a-f]{8}\] Transcription #1: 0 chars \(0\.1s audio\)", line) for line in info), info
    debug = _streaming_lines(caplog, logging.DEBUG)
    for prefix in ("Transcription #2 text", "Confirmed text", "Final utterance text"):
        assert any(re.fullmatch(rf"\[[0-9a-f]{{8}}\] {prefix}: '{SAID}'", line) for line in debug), (prefix, debug)


# A stream the server ends itself (a stop answered by session.end, an error before
# session.begin, a failure in the receive loop) returned from the ASGI app without ws.close(). uvicorn's websockets
# protocol then cuts the TCP connection, so the client got no Close frame and saw 1006, a dropped connection
# (RFC 6455 7.1.5 and 7.4.1). It must close with 1000 after a normal end (7.4.1: "the purpose for which the
# connection was established has been fulfilled") and 1011 with a reason after a failure. Starlette's TestClient
# doesn't go through uvicorn's run_asgi, so these run a real uvicorn on 127.0.0.1 and read the Close frame.


def _free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture()
def stream_server(monkeypatch):
    """A real uvicorn serving only the /v1/audio/stream router on a free 127.0.0.1 port, stopped afterwards."""
    import threading

    import uvicorn
    from fastapi import FastAPI

    from src.routes.streaming import create_router

    monkeypatch.setattr(streaming.settings, "os_api_key", "")
    app = FastAPI()
    app.include_router(create_router())
    port = _free_port()
    # ws="auto", as src.main.main() leaves it: uvicorn picks its websockets protocol.
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.05)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=15)
    assert not thread.is_alive(), "uvicorn did not stop"


def _wire_session(port: int, query: str, *, send_stop: bool) -> tuple[list[str], object]:
    """The event types a stream sends, and its Close frame's (code, reason), or a note that none came."""
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    events: list[str] = []
    close: object = "the server never closed the socket"
    # proxy=None: websockets follows the host's proxy settings by default.
    url = f"ws://127.0.0.1:{port}/v1/audio/stream?{query}"
    with connect(url, open_timeout=5, close_timeout=5, proxy=None) as ws:
        try:
            while True:
                event = json.loads(ws.recv(timeout=5))
                events.append(event["type"])
                if event["type"] == "session.begin" and send_stop:
                    ws.send(json.dumps({"type": "stop"}))
        except ConnectionClosed as closed:
            close = (closed.rcvd.code, closed.rcvd.reason) if closed.rcvd else "no close frame (1006)"
    return events, close


def _model_load_fails(monkeypatch):
    monkeypatch.setattr(streaming.backend_router, "is_model_loaded", lambda _model: False)

    def fail(_model):
        raise RuntimeError("model files missing")

    monkeypatch.setattr(streaming.backend_router, "load_model", fail)


def _silero_load_fails(monkeypatch):
    async def fail():
        raise RuntimeError("Silero files missing")

    monkeypatch.setattr(streaming, "get_vad_model", fail)


def _receive_loop_fails(monkeypatch):
    async def fail(self, _text):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(streaming.StreamingSession, "_handle_text", fail)


@pytest.mark.parametrize(
    ("setup", "query", "send_stop", "events", "close"),
    [
        (None, "vad=false", True, ["session.begin", "session.end"], (1000, "Stream ended")),
        (_model_load_fails, "vad=false", False, ["error"], (1011, "Failed to load model")),
        (_silero_load_fails, "vad=true", False, ["error"], (1011, "Failed to load VAD model")),
        (_receive_loop_fails, "vad=false", True, ["session.begin", "session.end"], (1011, "Streaming session failed")),
    ],
    ids=["clean-stop", "model-load-fails", "silero-load-fails", "receive-loop-fails"],
)
def test_a_stream_the_server_ends_reaches_the_client_with_a_close_code(
    stream_server, monkeypatch, setup, query, send_stop, events, close
):
    # The test's client must dial the uvicorn it started, not the host's proxy. A proxy
    # on loopback with nothing listening (nothing leaves the machine) fails a client that follows it.
    import os

    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setattr(streaming.backend_router, "is_model_loaded", lambda _model: True)
    if setup is not None:
        setup(monkeypatch)

    got_events, got_close = _wire_session(stream_server, query, send_stop=send_stop)

    assert got_events == events
    assert got_close == close


@pytest.mark.asyncio
async def test_a_stream_the_client_left_is_not_closed_again(monkeypatch):
    """After the client's disconnect there is nothing to close (Starlette raises on a
    send after it), so only a stream the server ends gets a close."""
    ws = DummyWS(messages=[{"type": "websocket.disconnect"}])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())

    await asyncio.wait_for(streaming_endpoint(ws, vad=False), 5)

    assert [event["type"] for event in ws.sent] == ["session.begin", "session.end"]
    assert ws.closed is None
    assert len(streaming._active_sessions) == 0


@pytest.mark.asyncio
async def test_a_failure_out_of_the_session_closes_with_1011(monkeypatch):
    """An exception out of run() (here the final flush) ends the stream with 1011, not a
    dropped connection."""
    async def fail(self):
        raise RuntimeError("flush failed")

    ws = DummyWS(messages=[_text('{"type":"stop"}')])
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    monkeypatch.setattr(streaming.StreamingSession, "_flush", fail)

    await asyncio.wait_for(streaming_endpoint(ws, vad=False), 5)

    # README.md says a server failure "also sends session.end, then closes with 1011".
    assert [event["type"] for event in ws.sent] == ["session.begin", "session.end"]
    assert ws.closed == (1011, "Streaming session failed")
    assert len(streaming._active_sessions) == 0


ONNX_RUNTIME_ERROR = "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION"


def _tone(samples: int) -> dict:
    return _audio((np.sin(np.arange(samples) / 8.0) * 8000).astype(np.int16).tobytes())


async def _stream_with_a_failing_vad(monkeypatch, frames, sample_rate, fails) -> tuple[list[str], object]:
    """One /v1/audio/stream session whose Silero raises as onnxruntime does whenever ``fails(state)`` is true; the
    event types sent and the close."""
    state = {"flushing": False}
    real_flush = StreamingSession._flush

    async def flush(self):
        state["flushing"] = True
        return await real_flush(self)

    def vad(_samples):
        if fails(state):
            raise RuntimeError(ONNX_RUNTIME_ERROR)
        return 0.0

    monkeypatch.setattr(streaming.settings, "os_stream_chunk_ms", 100)
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    monkeypatch.setattr(streaming, "get_vad_model", _fake_vad_model)
    monkeypatch.setattr(streaming, "SileroVAD", lambda *_a, **_k: vad)
    monkeypatch.setattr(StreamingSession, "_flush", flush)
    ws = DummyWS(messages=frames)

    await asyncio.wait_for(streaming_endpoint(ws, vad=True, sample_rate=sample_rate), 30)

    assert len(streaming._active_sessions) == 0
    return [event["type"] for event in ws.sent], ws.closed


@pytest.mark.parametrize(("frames", "sample_rate"), [
    ([_tone(4800)], 16000),  # one 300 ms frame: the first chunk's VAD run fails with two chunks still buffered
    ([_tone(2048)] * 3, 48000),  # the web app's AudioWorklet frames (core/mic-worklet.js): 1344 samples left over
], ids=["300-ms-frame", "web-app-48k-frames"])
@pytest.mark.asyncio
async def test_a_vad_failure_with_audio_still_buffered_sends_session_end_then_1011(monkeypatch, frames, sample_rate):
    """The final flush scores the audio still buffered with the same broken VAD. Its
    failure used to leave run() before session.end, so the client got a bare 1011; README.md promises session.end,
    then 1011 Streaming session failed."""
    events, close = await _stream_with_a_failing_vad(monkeypatch, frames, sample_rate, lambda _state: True)

    assert events == ["session.begin", "session.end"]
    assert close == (1011, "Streaming session failed")


@pytest.mark.asyncio
async def test_a_stop_whose_flush_fails_is_still_answered_with_session_end_then_1011(monkeypatch):
    """README.md and docs/AI-AGENTS.md say a stop is answered with session.end. When
    only the stop's flush fails (700 samples left after one 100 ms chunk), session.end still goes out, and the close
    says the stream failed."""
    events, close = await _stream_with_a_failing_vad(
        monkeypatch, [_tone(1600 + 700), _text('{"type":"stop"}')], 16000, lambda state: state["flushing"]
    )

    assert events == ["session.begin", "session.end"]
    assert close == (1011, "Streaming session failed")


@pytest.mark.asyncio
async def test_a_flush_that_fails_after_the_client_left_closes_nothing(monkeypatch):
    """After the client's own disconnect there is still nothing to close,
    even when the flush fails."""
    events, close = await _stream_with_a_failing_vad(
        monkeypatch, [_tone(1600 + 700)], 16000, lambda state: state["flushing"]
    )

    assert events == ["session.begin", "session.end"]
    assert close is None


class _HangingWS(DummyWS):
    """A client that sends nothing until ``gone`` is set, then leaves."""

    def __init__(self):
        super().__init__()
        self.gone = asyncio.Event()

    async def receive(self):
        await self.gone.wait()
        return {"type": "websocket.disconnect"}


@pytest.mark.asyncio
async def test_streams_waiting_to_load_their_model_leave_the_shared_executor_free(monkeypatch):
    """As for transcriptions: faster-whisper's load_model waits for the
    inference lock that every decode holds. Streams for a model that isn't loaded must not each hold a thread of
    asyncio's default executor while they wait, or every asyncio.to_thread caller in the server stalls."""
    import concurrent.futures
    import threading

    workers = 4
    streams = workers + 2
    shared = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shared-default")
    asyncio.get_running_loop().set_default_executor(shared)
    inference_lock, counts_lock = threading.Lock(), threading.Lock()
    loads: list[str] = []

    class _BackendLoading(_BackendOK):
        def is_model_loaded(self, _model):
            return False

        def load_model(self, _model):
            with counts_lock:
                loads.append(threading.current_thread().name)
            with inference_lock:  # FasterWhisperBackend.load_model takes it before it checks its models
                return None

    monkeypatch.setattr(streaming, "backend_router", _BackendLoading())
    monkeypatch.setattr(streaming.settings, "os_stream_max_connections", 10)
    sockets = [_HangingWS() for _ in range(streams)]
    inference_lock.acquire()  # a long decode, a batch file or a download holds it
    tasks = [asyncio.create_task(streaming_endpoint(ws, model="other/model", vad=False)) for ws in sockets]
    try:
        for _ in range(500):
            with counts_lock:
                if loads:
                    break
            await asyncio.sleep(0.01)
        assert loads, "no stream reached load_model"
        await asyncio.sleep(0.3)  # the other streams dispatch their loads
        try:
            bare = await asyncio.wait_for(asyncio.to_thread(lambda: "ran"), timeout=2)
        except TimeoutError:
            bare = "did not run within 2 s"
        with counts_lock:
            waiting = sorted(loads)
    finally:
        inference_lock.release()
        for ws in sockets:
            ws.gone.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=15)
        shared.shutdown(wait=False)

    assert bare == "ran", (
        f"streams waiting to load their model held the shared default executor (waiting threads {waiting}), so an "
        f"unrelated asyncio.to_thread job {bare}"
    )
    assert all(ws.sent[0]["type"] == "session.begin" for ws in sockets)


@pytest.mark.asyncio
async def test_the_close_after_a_stream_ends_does_not_hold_the_session_lock(monkeypatch):
    """The close after session.end runs once the session has
    given its slot back and outside the session-count lock, so a client that doesn't answer the close holds no other
    stream's start or end."""
    ws = _UnansweredCloseWS()
    ws._messages = [_text('{"type":"stop"}')]
    monkeypatch.setattr(streaming, "backend_router", _BackendOK())
    stream = asyncio.create_task(streaming_endpoint(ws, vad=False))
    try:
        await asyncio.wait_for(ws.closing.wait(), 5)
        held_while_closing = streaming._active_sessions_lock.locked()
        slots_while_closing = len(streaming._active_sessions)
    finally:
        ws.answered.set()
        await asyncio.wait_for(stream, 5)

    assert not held_while_closing, "the session-count lock was held through the close handshake"
    assert slots_while_closing == 0
    assert ws.closed == (1000, "Stream ended")
