"""Tests for the OpenAI Realtime API compatibility layer."""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import io
import json
import logging
import sys
import threading
import warnings
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.realtime import g711
from src.realtime.audio_buffer import (
    InputAudioBuffer,
    decode_audio_to_pcm16,
    encode_pcm16_to_format,
)
from src.realtime.events import (
    _event_id,
    _item_id,
    _response_id,
    conversation_item_created,
    conversation_item_input_audio_transcription_completed,
    error,
    input_audio_buffer_committed,
    input_audio_buffer_speech_started,
    input_audio_buffer_speech_stopped,
    response_audio_delta,
    response_audio_done,
    response_created,
    response_done,
    session_created,
    session_updated,
)
from src.realtime.session import VALID_AUDIO_FORMATS, SessionConfig
from tests._watchdog import hard_watchdog

# ---------------------------------------------------------------------------
# Event serialization tests
# ---------------------------------------------------------------------------


class TestEvents:
    def test_event_id_format(self):
        eid = _event_id()
        assert eid.startswith("evt_")
        assert len(eid) > 10

    def test_item_id_format(self):
        iid = _item_id()
        assert iid.startswith("item_")

    def test_response_id_format(self):
        rid = _response_id()
        assert rid.startswith("resp_")

    def test_session_created_event(self):
        session = {"id": "sess_123", "model": "test"}
        evt = session_created(session)
        assert evt["type"] == "session.created"
        assert evt["session"]["id"] == "sess_123"
        assert "event_id" in evt

    def test_session_updated_event(self):
        session = {"id": "sess_123"}
        evt = session_updated(session)
        assert evt["type"] == "session.updated"

    def test_error_event(self):
        evt = error("something broke", code="test_error")
        assert evt["type"] == "error"
        assert evt["error"]["message"] == "something broke"
        assert evt["error"]["code"] == "test_error"

    def test_error_event_with_event_id(self):
        evt = error("bad", event_id="evt_orig")
        assert evt["error"]["event_id"] == "evt_orig"

    def test_speech_started_event(self):
        evt = input_audio_buffer_speech_started(1500, "item_abc")
        assert evt["type"] == "input_audio_buffer.speech_started"
        assert evt["audio_start_ms"] == 1500
        assert evt["item_id"] == "item_abc"

    def test_speech_stopped_event(self):
        evt = input_audio_buffer_speech_stopped(3000, "item_abc")
        assert evt["type"] == "input_audio_buffer.speech_stopped"
        assert evt["audio_end_ms"] == 3000

    def test_committed_event(self):
        evt = input_audio_buffer_committed("item_1", "item_0")
        assert evt["type"] == "input_audio_buffer.committed"
        assert evt["item_id"] == "item_1"
        assert evt["previous_item_id"] == "item_0"

    def test_conversation_item_created_event(self):
        item = {"id": "item_1", "type": "message"}
        evt = conversation_item_created(item)
        assert evt["type"] == "conversation.item.created"
        assert evt["item"]["id"] == "item_1"

    def test_transcription_completed_event(self):
        evt = conversation_item_input_audio_transcription_completed("item_1", 0, "hello")
        assert evt["type"] == "conversation.item.input_audio_transcription.completed"
        assert evt["transcript"] == "hello"

    def test_response_created_event(self):
        resp = {"id": "resp_1", "status": "in_progress"}
        evt = response_created(resp)
        assert evt["type"] == "response.created"

    def test_response_audio_delta_event(self):
        evt = response_audio_delta("resp_1", "item_1", 0, 0, "base64data")
        assert evt["type"] == "response.audio.delta"
        assert evt["delta"] == "base64data"

    def test_response_audio_done_event(self):
        evt = response_audio_done("resp_1", "item_1", 0, 0)
        assert evt["type"] == "response.audio.done"

    def test_response_done_event(self):
        resp = {"id": "resp_1", "status": "completed"}
        evt = response_done(resp)
        assert evt["type"] == "response.done"


# ---------------------------------------------------------------------------
# Session config tests
# ---------------------------------------------------------------------------


class TestSessionConfig:
    def test_default_config(self):
        cfg = SessionConfig()
        assert cfg.input_audio_format == "pcm16"
        assert cfg.output_audio_format == "pcm16"
        assert cfg.voice == "alloy"
        assert cfg.vad_enabled is True

    def test_to_dict(self):
        cfg = SessionConfig(model="test-model")
        d = cfg.to_dict()
        assert d["model"] == "test-model"
        assert d["object"] == "realtime.session"
        assert "turn_detection" in d
        assert d["turn_detection"]["type"] == "server_vad"

    def test_update_voice(self):
        cfg = SessionConfig()
        cfg.update_from({"session": {"voice": "nova"}})
        assert cfg.voice == "nova"

    def test_update_audio_format(self):
        cfg = SessionConfig()
        cfg.update_from({"session": {"input_audio_format": "g711_ulaw"}})
        assert cfg.input_audio_format == "g711_ulaw"

    def test_update_invalid_format_ignored(self):
        cfg = SessionConfig()
        cfg.update_from({"session": {"input_audio_format": "mp3"}})
        assert cfg.input_audio_format == "pcm16"

    def test_update_turn_detection(self):
        cfg = SessionConfig()
        cfg.update_from({"session": {"turn_detection": {"threshold": 0.8, "silence_duration_ms": 1000}}})
        assert cfg.turn_detection.threshold == 0.8
        assert cfg.turn_detection.silence_duration_ms == 1000

    def test_disable_turn_detection(self):
        cfg = SessionConfig()
        cfg.update_from({"session": {"turn_detection": None}})
        assert cfg.vad_enabled is False

    def test_session_id_unique(self):
        c1 = SessionConfig()
        c2 = SessionConfig()
        assert c1.id != c2.id

    def test_valid_audio_formats(self):
        assert "pcm16" in VALID_AUDIO_FORMATS
        assert "g711_ulaw" in VALID_AUDIO_FORMATS
        assert "g711_alaw" in VALID_AUDIO_FORMATS


# ---------------------------------------------------------------------------
# Audio buffer tests
# ---------------------------------------------------------------------------


def _make_pcm16_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM16 audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


def _make_pcm16_tone(duration_ms: int, freq: int = 440, sample_rate: int = 16000) -> bytes:
    """Generate a sine wave PCM16 audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.arange(num_samples) / sample_rate
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return samples.tobytes()


def _rms(audio: bytes | np.ndarray) -> float:
    if isinstance(audio, bytes):
        audio = np.frombuffer(audio, dtype=np.int16)
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


class TestAudioBuffer:
    def test_append_and_commit(self):
        buf = InputAudioBuffer()
        pcm = _make_pcm16_silence(100)
        buf.append(pcm)
        data = buf.commit()
        assert data == pcm

    def test_commit_clears_buffer(self):
        buf = InputAudioBuffer()
        buf.append(_make_pcm16_silence(100))
        buf.commit()
        data = buf.commit()
        assert data == b""

    def test_clear(self):
        buf = InputAudioBuffer()
        buf.append(_make_pcm16_silence(100))
        buf.clear()
        assert buf.get_audio() == b""

    def test_no_vad_no_events(self):
        buf = InputAudioBuffer()
        events = buf.append(_make_pcm16_tone(100))
        assert events == []

    def test_vad_speech_detection(self):
        """With a mock VAD that always returns high probability."""
        mock_vad = MagicMock()
        mock_vad.return_value = 0.9  # Always speech

        buf = InputAudioBuffer(vad=mock_vad, threshold=0.5)
        events = buf.append(_make_pcm16_tone(100))
        assert len(events) == 1
        assert events[0]["type"] == "speech_started"

    def test_vad_aggregates_partial_websocket_frames(self):
        """Two partial frames should form one complete Silero window."""
        from src.vad.silero import SileroVAD

        class AlwaysSpeechSession:
            def __init__(self):
                self.call_count = 0

            def run(self, _output_names, inputs):
                self.call_count += 1
                return [np.array([[0.9]], dtype=np.float32), inputs["state"]]

        onnx_session = AlwaysSpeechSession()
        buf = InputAudioBuffer(vad=SileroVAD(onnx_session), threshold=0.5)
        half_window = np.zeros(256, dtype=np.int16).tobytes()

        assert buf.append(half_window) == []
        events = buf.append(half_window)

        assert onnx_session.call_count == 1
        assert [event["type"] for event in events] == ["speech_started"]

    def test_vad_handles_speech_then_silence_in_one_append(self):
        """Each complete window must drive its own VAD transition."""
        probabilities = iter([0.9, 0.1, 0.1, 0.9])
        window_lengths = []

        def sequenced_vad(audio):
            window_lengths.append(len(audio))
            return next(probabilities)

        buf = InputAudioBuffer(
            vad=sequenced_vad,
            threshold=0.5,
            silence_duration_ms=64,
        )
        audio = np.zeros(3 * 512, dtype=np.int16).tobytes()

        events = buf.append(audio)

        assert [
            {key: value for key, value in event.items() if not key.startswith("_")}
            for event in events
        ] == [
            {"type": "speech_started", "audio_start_ms": 0},
            {"type": "speech_stopped", "audio_end_ms": 96},
        ]
        assert events[1]["_buffer_end_bytes"] == 3 * 512 * 2
        assert buf.in_speech is False

        resumed_events = buf.append(np.zeros(512, dtype=np.int16).tobytes())

        assert window_lengths == [512, 512, 512, 512]
        assert resumed_events == [
            {"type": "speech_started", "audio_start_ms": 96},
        ]

    def test_vad_silence_after_speech(self):
        """VAD detects speech then silence."""
        mock_vad = MagicMock()
        buf = InputAudioBuffer(vad=mock_vad, threshold=0.5, silence_duration_ms=100)

        # Speech
        mock_vad.return_value = 0.9
        events1 = buf.append(_make_pcm16_tone(100))
        assert any(e["type"] == "speech_started" for e in events1)

        # Silence (multiple chunks to exceed silence_duration_ms)
        mock_vad.return_value = 0.1
        all_events = []
        for _ in range(10):
            evts = buf.append(_make_pcm16_silence(50))
            all_events.extend(evts)

        assert any(e["type"] == "speech_stopped" for e in all_events)

    def test_vad_turn_keeps_only_prefix_padding_before_speech(self):
        probabilities = iter([0.1] * 40 + [0.9] * 3 + [0.1] * 2)
        buf = InputAudioBuffer(
            vad=lambda _audio: next(probabilities),
            silence_duration_ms=64,
            prefix_padding_ms=96,  # 3 windows
        )

        events = buf.append(b"\x01\x00" * (45 * 512))

        # OpenAI: audio_start_ms marks the start of the audio sent, padding included.
        assert events[0] == {"type": "speech_started", "audio_start_ms": 1280 - 96}
        start, end = events[1]["_buffer_start_bytes"], events[1]["_buffer_end_bytes"]
        assert len(buf.commit_through(end, start=start)) == (3 + 5) * 512 * 2
        assert events[1]["audio_end_ms"] - events[0]["audio_start_ms"] == (3 + 5) * 32
        assert buf.buffered_bytes == 0

    def test_vad_keeps_idle_audio_up_to_the_commit_backlog(self):
        buf = InputAudioBuffer(vad=lambda _audio: 0.1, prefix_padding_ms=96, max_backlog_ms=320)

        for _ in range(10):
            buf.append(b"\x00\x00" * 700)  # unaligned with the 512-sample windows

        pending_samples = 10 * 700 % 512
        assert buf.buffered_bytes == (320 * 16 + pending_samples) * 2

    def test_vad_turns_in_one_frame_commit_only_their_own_audio(self):
        """Two turns in one frame: each commits its speech and padding, none of the other turn."""
        probabilities = iter([0.9] * 3 + [0.1] * 2 + [0.1] * 20 + [0.9] * 3 + [0.1] * 2)
        speech = np.arange(1, 512 * 5 + 1, dtype=np.int16).tobytes()
        gap = np.zeros(512 * 20, dtype=np.int16).tobytes()
        buf = InputAudioBuffer(
            vad=lambda _audio: next(probabilities),
            silence_duration_ms=64,
            prefix_padding_ms=64,  # 2 windows
        )

        events = buf.append(speech + gap + speech)

        started = [e["audio_start_ms"] for e in events if e["type"] == "speech_started"]
        turns = [
            (e["_buffer_start_bytes"], e["_buffer_end_bytes"]) for e in events if e["type"] == "speech_stopped"
        ]
        assert started == [0, (5 + 20 - 2) * 32]
        # As the realtime handler does: later offsets move back by what was committed.
        assert buf.commit_through(turns[0][1], start=turns[0][0]) == speech
        assert buf.commit_through(turns[1][1] - turns[0][1], start=turns[1][0] - turns[0][1]) == (
            gap[-2 * 512 * 2:] + speech
        )
        assert buf.buffered_bytes == 0

    def test_backlog_trimming_leaves_an_uncommitted_turn_in_place(self):
        probabilities = iter([0.9] * 3 + [0.1] * 2)
        speech = np.arange(1, 512 * 5 + 1, dtype=np.int16).tobytes()
        buf = InputAudioBuffer(
            vad=lambda _audio: next(probabilities, 0.1),
            silence_duration_ms=64,
            prefix_padding_ms=0,
            max_backlog_ms=64,  # 2 windows
        )

        stopped = [e for e in buf.append(speech) if e["type"] == "speech_stopped"][0]
        buf.append(np.zeros(512 * 20, dtype=np.int16).tobytes())

        assert buf.buffered_bytes == len(speech) + 2 * 512 * 2
        assert buf.commit_through(stopped["_buffer_end_bytes"], start=stopped["_buffer_start_bytes"]) == speech

    def test_configure_keeps_audio_clock_and_buffered_audio(self):
        vad = MagicMock(return_value=0.1)
        buf = InputAudioBuffer(vad=vad, prefix_padding_ms=1000)
        buf.append(b"\x00\x00" * (32 * 512))

        buf.configure(vad=vad, threshold=0.5, silence_duration_ms=200, prefix_padding_ms=500)
        vad.return_value = 0.9
        events = buf.append(b"\x01\x00" * 512)

        # Speech 1.024 s into the session, with the new 500 ms of padding
        assert events == [{"type": "speech_started", "audio_start_ms": 1024 - 500}]
        assert buf.buffered_bytes == (32 * 512 + 512) * 2

    def test_buffer_limit_rejects_append(self):
        buf = InputAudioBuffer(max_buffer_bytes=1000)
        with pytest.raises(BufferError):
            buf.append(b"\x00" * 2000)

    def test_buffer_limit_rejects_growth(self):
        buf = InputAudioBuffer(max_buffer_bytes=1000)
        buf.append(b"\x00" * 800)
        with pytest.raises(BufferError):
            buf.append(b"\x00" * 400)


@pytest.mark.asyncio
async def test_realtime_auto_commit_preserves_next_utterance_in_same_frame():
    from src.realtime.server import RealtimeSession

    probabilities = iter([0.9, 0.1, 0.1, 0.9])

    class DummyWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.audio_buffer = InputAudioBuffer(
        vad=lambda _audio: next(probabilities),
        threshold=0.5,
        silence_duration_ms=64,
    )
    pcm = b"\x01\x00" * (4 * 512)

    with (
        patch(
            "src.realtime.server.decode_audio_to_pcm16",
            side_effect=lambda data, _fmt, target_rate: data,
        ),
        patch("src.realtime.server.stt_router.transcribe", return_value={"text": "first"}) as transcribe,
    ):
        await session._handle_input_audio_buffer_append({
            "audio": base64.b64encode(pcm).decode("ascii"),
        })

    assert session.audio_buffer.get_audio() == pcm[3 * 512 * 2:]
    with wave.open(io.BytesIO(transcribe.call_args.kwargs["audio"]), "rb") as wav_file:
        assert wav_file.getnframes() == 3 * 512
    event_types = [event["type"] for event in websocket.sent]
    assert event_types.count("input_audio_buffer.speech_started") == 2
    assert event_types.count("input_audio_buffer.speech_stopped") == 1
    assert event_types.count("input_audio_buffer.committed") == 1


@pytest.mark.parametrize("invalid_audio", ["!!!!", {"not": "base64"}])
@pytest.mark.asyncio
async def test_realtime_rejects_malformed_base64_audio(invalid_audio):
    from src.realtime.server import RealtimeSession

    class DummyWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.audio_buffer = InputAudioBuffer()

    await session._handle_input_audio_buffer_append({"audio": invalid_audio})

    assert session.audio_buffer.get_audio() == b""
    assert websocket.sent[-1]["type"] == "error"
    assert websocket.sent[-1]["error"]["code"] == "invalid_audio"


@pytest.mark.asyncio
async def test_voice_only_session_update_preserves_buffered_microphone_audio():
    from src.realtime.server import RealtimeSession

    class DummyWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

    session = RealtimeSession(DummyWebSocket(), MagicMock())
    buffered_audio = _make_pcm16_tone(100)
    session.audio_buffer = InputAudioBuffer()
    session.audio_buffer.append(buffered_audio)

    vad_model = MagicMock()
    vad_model.session = MagicMock()
    with (
        patch(
            "src.realtime.server.get_vad_model",
            new=AsyncMock(return_value=vad_model),
        ),
        patch("src.realtime.server.SileroVAD", return_value=MagicMock()),
    ):
        await session._handle_session_update({
            "type": "session.update",
            "session": {"voice": "shimmer"},
        })

    assert session.config.voice == "shimmer"
    assert session.audio_buffer.get_audio() == buffered_audio


@pytest.mark.asyncio
async def test_realtime_tts_uses_tts_default_instead_of_stt_session_model():
    from src.realtime import server
    from src.tts.router import TTSRouter

    class DummyWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

    class KokoroStub:
        name = "kokoro"
        sample_rate = 24000

        def synthesize(self, text, voice, speed=1.0, lang_code=None):
            yield np.zeros(100, dtype=np.float32)

    with (
        patch("src.tts.router._discover_backends", return_value={}),
        patch.object(server.settings, "stt_model", "Systran/faster-whisper-base"),
        patch.object(server.settings, "tts_model", "kokoro"),
    ):
        router = TTSRouter(device="cpu")
        router.register_backend("kokoro", KokoroStub())
        websocket = DummyWebSocket()
        session = server.RealtimeSession(websocket, router)
        await session._handle_response_create({
            "response": {"instructions": "Hello from realtime"},
        })

    event_types = [event["type"] for event in websocket.sent]
    assert "error" not in event_types
    assert "response.audio.delta" in event_types
    assert "response.done" in event_types


@pytest.mark.asyncio
async def test_realtime_tts_encodes_from_selected_models_sample_rate():
    from src.realtime import server

    class DummyWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

    class Router:
        def sample_rate_for(self, model):
            assert model == "piper/test-voice"
            return 16000

        def synthesize(self, **_kwargs):
            yield np.zeros(320, dtype=np.float32)

    websocket = DummyWebSocket()
    session = server.RealtimeSession(websocket, Router())
    await session._handle_response_create({
        "response": {
            "instructions": "Use the selected model rate.",
            "model": "piper/test-voice",
        },
    })

    audio = b"".join(
        base64.b64decode(event["delta"])
        for event in websocket.sent
        if event["type"] == "response.audio.delta"
    )
    # 20 ms at the model's 16 kHz is 20 ms of 24 kHz pcm16, not 13 ms
    assert len(audio) // 2 == 480


@pytest.mark.asyncio
async def test_realtime_response_cancel_interrupts_synthesis_and_cleans_task():
    from src.realtime.server import RealtimeSession

    started = threading.Event()
    release = threading.Event()
    second_chunk_requested = threading.Event()
    release_second_chunk = threading.Event()
    finished = threading.Event()

    class DummyWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

    class BlockingRouter:
        def sample_rate_for(self, _model):
            return 24000

        def synthesize(self, **_kwargs):
            started.set()
            try:
                release.wait(timeout=2)
                yield np.zeros(320, dtype=np.float32)
                second_chunk_requested.set()
                release_second_chunk.wait(timeout=2)
                yield np.zeros(320, dtype=np.float32)
            finally:
                finished.set()

    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, BlockingRouter())

    await session.handle_event({
        "type": "response.create",
        "response": {"instructions": "Cancel this response."},
    })
    try:
        assert await asyncio.to_thread(started.wait, 1)
        await session.handle_event({"type": "response.cancel"})
    finally:
        release.set()
        release_second_chunk.set()

    assert await asyncio.to_thread(finished.wait, 1)
    assert not second_chunk_requested.is_set()
    assert session._response_task is None
    assert session._current_response_id is None
    event_types = [event["type"] for event in websocket.sent]
    assert "response.audio.delta" not in event_types
    done = [event for event in websocket.sent if event["type"] == "response.done"]
    assert done[-1]["response"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_realtime_rejects_overlapping_response_create():
    from src.realtime.server import RealtimeSession

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    synthesis_calls = 0

    class DummyWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            self.sent.append(event)

    class BlockingRouter:
        def sample_rate_for(self, _model):
            return 24000

        def synthesize(self, **_kwargs):
            nonlocal synthesis_calls
            synthesis_calls += 1
            started.set()
            try:
                release.wait(timeout=2)
                yield np.zeros(320, dtype=np.float32)
            finally:
                finished.set()

    websocket = DummyWebSocket()
    session = RealtimeSession(websocket, BlockingRouter())
    await session.handle_event({
        "type": "response.create",
        "response": {"instructions": "First response."},
    })
    first_task = session._response_task
    try:
        assert await asyncio.to_thread(started.wait, 1)
        await session.handle_event({
            "type": "response.create",
            "event_id": "evt_second",
            "response": {"instructions": "Second response."},
        })

        assert session._response_task is first_task
        assert synthesis_calls == 1
        errors = [event for event in websocket.sent if event["type"] == "error"]
        assert errors[-1]["error"]["code"] == "response_in_progress"
        assert errors[-1]["error"]["event_id"] == "evt_second"
        await session.handle_event({"type": "response.cancel"})
    finally:
        release.set()

    assert await asyncio.to_thread(finished.wait, 1)


@pytest.mark.asyncio
async def test_realtime_cancel_after_completed_done_does_not_emit_second_terminal_event():
    from src.realtime.server import RealtimeSession

    completed_delivery_started = asyncio.Event()
    allow_completed_delivery = asyncio.Event()

    class PausingWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, event):
            snapshot = copy.deepcopy(event)
            self.sent.append(snapshot)
            if (
                snapshot["type"] == "response.done"
                and snapshot["response"]["status"] == "completed"
            ):
                completed_delivery_started.set()
                await allow_completed_delivery.wait()

    class Router:
        def sample_rate_for(self, _model):
            return 24000

        def synthesize(self, **_kwargs):
            yield np.zeros(320, dtype=np.float32)

    websocket = PausingWebSocket()
    session = RealtimeSession(websocket, Router())
    await session.handle_event({
        "type": "response.create",
        "response": {"instructions": "Complete this response."},
    })
    task = session._response_task
    assert task is not None

    try:
        await asyncio.wait_for(completed_delivery_started.wait(), timeout=1)
        # The cancel's error reply queues behind the paused response.done.
        cancel = asyncio.create_task(session.handle_event({"type": "response.cancel"}))
        await asyncio.sleep(0)
    finally:
        allow_completed_delivery.set()
    await task
    await asyncio.wait_for(cancel, timeout=1)

    done = [event for event in websocket.sent if event["type"] == "response.done"]
    assert [event["response"]["status"] for event in done] == ["completed"]
    errors = [event for event in websocket.sent if event["type"] == "error"]
    assert [event["error"]["code"] for event in errors] == ["response_cancel_not_active"]
    assert session._response_task is None
    assert session._current_response_id is None


class _RecordingWebSocket:
    def __init__(self):
        self.sent = []
        self.closed = None

    async def send_json(self, event):
        self.sent.append(event)

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)

    def of_type(self, event_type):
        return [event for event in self.sent if event["type"] == event_type]


def _patched_vad(detector):
    """Patch Silero with a detector that scores each 512-sample window."""
    vad_model = MagicMock()
    vad_model.session = object()
    return (
        patch("src.realtime.server.get_vad_model", new=AsyncMock(return_value=vad_model)),
        patch("src.realtime.server.SileroVAD", return_value=detector),
    )


def _scripted_vad(probabilities):
    """Patch Silero so each 512-sample window gets the next probability."""
    probs = iter(probabilities)
    return _patched_vad(lambda _audio: next(probs, 0.0))


def _identity_decode():
    return patch(
        "src.realtime.server.decode_audio_to_pcm16",
        side_effect=lambda data, _fmt, target_rate: data,
    )


def _append(pcm: bytes) -> dict:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}


# 12 MiB of 24 kHz PCM16 is a 16 MiB base64 text frame, uvicorn's default ws_max_size
LARGE_APPEND_PCM_BYTES = 12 * 1024 * 1024
# 65 s of 16 kHz audio, about 0.4 s of Silero time on a desktop CPU
MAX_VAD_WINDOWS_PER_LOOP_TURN = 2048


@pytest.fixture
def watchdog(request):
    """End the run if the test hangs (there is no pytest-timeout), naming it and printing every thread's stack."""
    with hard_watchdog(request, 300):  # tests/test_streaming_units.py tests it
        yield


class _LoopTurns:
    """A task that counts its own turns on the event loop."""

    def __init__(self):
        self.count = 0
        self.stop = False

    async def run(self):
        while not self.stop:
            self.count += 1
            await asyncio.sleep(0)


class _WhereVADRuns:
    """Silence for every window; records whether it ran on the loop thread, and at which turn."""

    def __init__(self, turns: _LoopTurns):
        self.turns = turns
        self.loop_thread = threading.get_ident()
        self.calls: list[tuple[bool, int]] = []

    def __call__(self, _window) -> float:
        self.calls.append((threading.get_ident() == self.loop_thread, self.turns.count))
        return 0.0

    def longest_run_on_the_loop(self) -> int:
        """The most windows scored on the loop thread while no other task got a turn."""
        longest = run = 0
        last_turn = None
        for on_loop, turn in self.calls:
            if not on_loop:
                run, last_turn = 0, None
                continue
            run = run + 1 if turn == last_turn else 1
            last_turn = turn
            longest = max(longest, run)
        return longest


@pytest.mark.asyncio
async def test_large_silent_append_scores_its_vad_windows_off_the_event_loop(watchdog):
    """One 16 MiB append is 8k Silero windows (~1.6 s) and idle audio is trimmed, so a
    client could repeat it: all of it ran on the event loop and stalled every other request."""
    from src.realtime.server import RealtimeSession

    turns = _LoopTurns()
    vad = _WhereVADRuns(turns)
    get_vad, silero = _patched_vad(vad)
    with get_vad, silero:
        session = RealtimeSession(_RecordingWebSocket(), MagicMock(), ga=True)
        await session.initialize()  # server VAD by default
    assert session.audio_buffer.vad is vad
    event = _append(bytes(LARGE_APPEND_PCM_BYTES))
    counter = asyncio.create_task(turns.run())
    await asyncio.sleep(0)

    await asyncio.wait_for(session.handle_event(event), timeout=240)
    turns.stop = True
    await counter

    assert session.ws.of_type("error") == []
    # 24 kHz to 16 kHz; the resampler holds back its last few samples for the next append
    assert len(vad.calls) >= LARGE_APPEND_PCM_BYTES // 2 * 2 // 3 // 512 - 1
    assert vad.longest_run_on_the_loop() <= MAX_VAD_WINDOWS_PER_LOOP_TURN, (
        f"{vad.longest_run_on_the_loop()} of {len(vad.calls)} VAD windows ran on the event loop "
        "with no other task getting a turn"
    )


@pytest.mark.asyncio
async def test_vad_turn_events_and_its_transcript_share_one_item_id():
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    get_vad, silero = _scripted_vad([0.9, 0.9, 0.9, 0.1, 0.1])
    with get_vad, silero, _identity_decode(), patch(
        "src.realtime.server.stt_router.transcribe", return_value={"text": "hi there"},
    ):
        session = RealtimeSession(websocket, MagicMock())
        await session.initialize()
        await session.handle_event({
            "type": "session.update",
            "session": {"turn_detection": {"silence_duration_ms": 64}},
        })
        await session.handle_event(_append(b"\x01\x00" * (5 * 512)))

    ids = {
        websocket.of_type("input_audio_buffer.speech_started")[0]["item_id"],
        websocket.of_type("input_audio_buffer.speech_stopped")[0]["item_id"],
        websocket.of_type("input_audio_buffer.committed")[0]["item_id"],
        websocket.of_type("conversation.item.created")[0]["item"]["id"],
        websocket.of_type("conversation.item.input_audio_transcription.completed")[0]["item_id"],
    }
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_server_vad_session_streaming_quiet_audio_is_not_closed_as_idle():
    from src.realtime import server

    websocket = _RecordingWebSocket()
    get_vad, silero = _scripted_vad([0.1] * 10)
    with get_vad, silero, _identity_decode():
        session = server.RealtimeSession(websocket, MagicMock())
        await session.initialize()
        # The client has streamed silence for longer than the idle timeout.
        session._last_commit_at -= server.settings.os_realtime_idle_timeout_s + 1
        await session.handle_event(_append(b"\x00\x00" * 1024))

    assert websocket.closed is None
    assert websocket.of_type("error") == []


@pytest.mark.asyncio
async def test_manual_commit_session_still_times_out_waiting_for_commit():
    from src.realtime import server

    websocket = _RecordingWebSocket()
    session = server.RealtimeSession(websocket, MagicMock())
    session.config.update_from({"session": {"turn_detection": None}})
    session.audio_buffer = InputAudioBuffer()
    session._last_commit_at -= server.settings.os_realtime_idle_timeout_s + 1

    await session.handle_event(_append(b"\x00\x00" * 1024))

    assert websocket.closed == (4008, "Session idle timeout")


@pytest.mark.asyncio
async def test_switching_server_vad_to_manual_commit_starts_the_commit_wait_at_the_switch():
    """OS_REALTIME_IDLE_TIMEOUT_S closes idle sessions. A hands-free session streaming
    quiet audio owed no commit, so after turn_detection null its first append must not
    be closed as "idle timeout waiting for commit"."""
    from src.realtime import server

    websocket = _RecordingWebSocket()
    get_vad, silero = _scripted_vad([0.1] * 10)
    with get_vad, silero, _identity_decode():
        session = server.RealtimeSession(websocket, MagicMock())
        await session.initialize()
        # Hands-free for longer than the idle timeout, the client streaming its microphone
        session._last_commit_at -= server.settings.os_realtime_idle_timeout_s + 1
        await session.handle_event(_append(b"\x00\x00" * 1024))
        await session.handle_event({"type": "session.update", "session": {"turn_detection": None}})
        await session.handle_event(_append(b"\x01\x00" * 1024))
        assert websocket.closed is None, websocket.of_type("error")

        # From the switch on, the manual-commit wait applies as usual.
        session._last_commit_at -= server.settings.os_realtime_idle_timeout_s + 1
        await session.handle_event(_append(b"\x01\x00" * 1024))

    assert websocket.closed == (4008, "Session idle timeout")


@pytest.mark.asyncio
async def test_clear_restarts_the_manual_commit_wait():
    from src.realtime import server

    websocket = _RecordingWebSocket()
    session = server.RealtimeSession(websocket, MagicMock())
    session.config.update_from({"session": {"turn_detection": None}})
    session.audio_buffer = InputAudioBuffer()
    session._last_commit_at -= server.settings.os_realtime_idle_timeout_s + 1

    await session.handle_event({"type": "input_audio_buffer.clear"})
    await session.handle_event(_append(b"\x00\x00" * 1024))

    assert websocket.closed is None
    assert websocket.of_type("error") == []


@pytest.mark.asyncio
async def test_vad_turn_events_span_exactly_the_audio_sent_to_stt():
    """OpenAI: speech_started.audio_start_ms "will correspond to the beginning of audio
    sent to the model, and thus includes the prefix_padding_ms", and speech_stopped's
    audio_end_ms marks its end. It reported the detection time, 300 ms too late."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    windows = [0.1] * 32 + [0.9] * 4 + [0.1] * 16  # 1.024 s quiet, speech, 512 ms quiet
    get_vad, silero = _scripted_vad(windows)
    with get_vad, silero, _identity_decode(), patch(
        "src.realtime.server.stt_router.transcribe", return_value={"text": "hi"},
    ) as transcribe:
        session = RealtimeSession(websocket, MagicMock())
        await session.initialize()
        await session.handle_event(_append(b"\x01\x00" * (len(windows) * 512)))

    started = websocket.of_type("input_audio_buffer.speech_started")[0]
    stopped = websocket.of_type("input_audio_buffer.speech_stopped")[0]
    with wave.open(io.BytesIO(transcribe.call_args.kwargs["audio"]), "rb") as wav_file:
        sent_ms = wav_file.getnframes() * 1000 // 16000

    assert started["audio_start_ms"] == 1024 - 300
    assert stopped["audio_end_ms"] - started["audio_start_ms"] == sent_ms == 300 + 20 * 32


@pytest.mark.asyncio
async def test_manual_commit_under_server_vad_transcribes_audio_the_vad_missed():
    """Speech too quiet for Silero still reaches STT when the client commits it. Only
    VAD turns are cut to prefix_padding_ms before their speech."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    get_vad, silero = _scripted_vad([0.1] * 200)
    with get_vad, silero, _identity_decode(), patch(
        "src.realtime.server.stt_router.transcribe", return_value={"text": "hi"},
    ) as transcribe:
        session = RealtimeSession(websocket, MagicMock())
        await session.initialize()
        for _ in range(20):  # 2 s in 100 ms appends
            await session.handle_event(_append(b"\x01\x00" * 1600))
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert websocket.of_type("input_audio_buffer.speech_started") == []
    with wave.open(io.BytesIO(transcribe.call_args.kwargs["audio"]), "rb") as wav_file:
        assert wav_file.getnframes() == 2 * 16000


def test_manual_commit_under_server_vad_keeps_at_most_the_commit_backlog():
    from src.realtime import audio_buffer

    buf = InputAudioBuffer(vad=lambda _audio: 0.1)
    for _ in range(40):  # 40 s of quiet audio, 1 s per append
        buf.append(b"\x00\x00" * 16000)

    assert audio_buffer.MAX_COMMIT_BACKLOG_MS == 30_000
    assert len(buf.commit()) == 30 * 16000 * 2


@pytest.mark.parametrize("buffered", [b"", b"\x01\x00" * 799], ids=["empty", "under_50ms"])
@pytest.mark.asyncio
async def test_commit_of_empty_or_too_short_buffer_returns_commit_empty_error(buffered):
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.audio_buffer = InputAudioBuffer()
    session.audio_buffer.append(buffered)

    await session.handle_event({"type": "input_audio_buffer.commit", "event_id": "evt_commit"})

    assert [event["type"] for event in websocket.sent] == ["error"]
    assert websocket.sent[0]["error"]["code"] == "input_audio_buffer_commit_empty"
    assert websocket.sent[0]["error"]["event_id"] == "evt_commit"


@pytest.mark.asyncio
async def test_response_cancel_without_active_response_returns_error():
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())

    await session.handle_event({"type": "response.cancel", "event_id": "evt_cancel"})

    assert [event["type"] for event in websocket.sent] == ["error"]
    assert websocket.sent[0]["error"]["code"] == "response_cancel_not_active"
    assert websocket.sent[0]["error"]["event_id"] == "evt_cancel"


@pytest.mark.parametrize("model", ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"])
@pytest.mark.asyncio
async def test_ga_transcription_maps_openai_model_names_to_stt_model(model):
    from src.realtime import server

    session = server.RealtimeSession(_RecordingWebSocket(), MagicMock(), ga=True)
    session.audio_buffer = InputAudioBuffer()
    session.config.update_from({"session": {
        "type": "realtime",
        "audio": {"input": {"transcription": {"model": model}}},
    }})
    session.audio_buffer.append(b"\x01\x00" * 1600)

    with (
        patch.object(server.settings, "stt_model", "Systran/faster-whisper-base"),
        patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hi"}) as transcribe,
    ):
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert transcribe.call_args.kwargs["model"] == "Systran/faster-whisper-base"


@pytest.mark.asyncio
async def test_turn_transcription_decodes_once_without_temperature_fallback():
    # A live turn decodes once at temperature 0; faster-whisper's hotter retries
    # (up to six decodes) are for file transcription.
    from src.realtime import server

    session = server.RealtimeSession(_RecordingWebSocket(), MagicMock(), ga=True)
    session.audio_buffer = InputAudioBuffer()
    session.audio_buffer.append(b"\x01\x00" * 1600)

    with patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hi"}) as transcribe:
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert transcribe.call_args.kwargs["temperature"] == 0.0
    assert transcribe.call_args.kwargs["temperature_fallback"] is False


@pytest.mark.parametrize(("session_model", "transcription_model", "expected"), [
    # OpenAI beta clients connect with ?model=<realtime model> and pick STT with
    # input_audio_transcription.model.
    ("gpt-4o-realtime-preview", "whisper-1", "Systran/faster-whisper-base"),
    ("gpt-4o-realtime-preview-2024-12-17", "Systran/faster-whisper-tiny", "Systran/faster-whisper-tiny"),
    # Project clients written for the legacy events pick STT with ?model=.
    ("Systran/faster-whisper-tiny", "whisper-1", "Systran/faster-whisper-tiny"),
    ("whisper-1", "whisper-1", "Systran/faster-whisper-base"),
    ("gpt-4o-transcribe", "whisper-1", "Systran/faster-whisper-base"),
])
@pytest.mark.asyncio
async def test_beta_session_transcribes_with_the_stt_model_its_client_chose(
    session_model, transcription_model, expected,
):
    from src.realtime import server

    session = server.RealtimeSession(_RecordingWebSocket(), MagicMock(), model=session_model)
    session.audio_buffer = InputAudioBuffer()
    await session.handle_event({"type": "session.update", "session": {
        "input_audio_transcription": {"model": transcription_model}, "turn_detection": None,
    }})
    session.audio_buffer.append(b"\x01\x00" * 1600)

    with (
        patch.object(server.settings, "stt_model", "Systran/faster-whisper-base"),
        patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hi"}) as transcribe,
    ):
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert transcribe.call_args.kwargs["model"] == expected


@pytest.mark.asyncio
async def test_openai_beta_header_client_gets_a_beta_session_that_transcribes_with_stt_model():
    """OpenAI beta clients send 'OpenAI-Beta: realtime=v1' without the 'realtime' subprotocol,
    connect with ?model=<realtime model> and pick STT with input_audio_transcription.model."""
    from starlette.websockets import WebSocket

    from src.realtime import server

    incoming = [
        {"type": "websocket.connect"},
        {"type": "websocket.receive", "text": json.dumps({"type": "session.update", "session": {
            "input_audio_transcription": {"model": "whisper-1"}, "turn_detection": None,
        }})},
        {"type": "websocket.receive", "text": json.dumps(_append(b"\x01\x00" * 4800))},  # 200 ms at 24 kHz
        {"type": "websocket.receive", "text": json.dumps({"type": "input_audio_buffer.commit"})},
    ]
    sent = []
    transcribed = asyncio.Event()

    async def receive():
        if incoming:
            return incoming.pop(0)
        await transcribed.wait()
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)
        if "input_audio_transcription.completed" in message.get("text", ""):
            transcribed.set()

    scope = {
        "type": "websocket", "path": "/v1/realtime", "headers": [(b"openai-beta", b"realtime=v1")],
        "subprotocols": [], "query_string": b"model=gpt-4o-realtime-preview",
    }
    get_vad, silero = _scripted_vad([])
    with (
        get_vad, silero,
        patch.object(server.settings, "stt_model", "Systran/faster-whisper-base"),
        patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hi"}) as transcribe,
    ):
        await asyncio.wait_for(server.realtime_endpoint(
            WebSocket(scope, receive, send), tts_router=MagicMock(), model="gpt-4o-realtime-preview",
        ), timeout=5)

    received = [json.loads(m["text"]) for m in sent if m["type"] == "websocket.send"]
    assert received[0]["type"] == "session.created"
    assert received[0]["session"]["object"] == "realtime.session"  # beta event shapes, not GA
    assert received[0]["session"]["model"] == "gpt-4o-realtime-preview"  # the client's name is echoed
    assert received[-1]["type"] == "conversation.item.input_audio_transcription.completed", received[-1]
    assert transcribe.call_args.kwargs["model"] == "Systran/faster-whisper-base"


class _ModelRecordingRouter:
    TTS_MODELS = {"kokoro", "piper/en_US-lessac-medium"}

    def __init__(self):
        self.models = []

    def get_backend(self, model):
        if model not in self.TTS_MODELS:
            raise ValueError(f"Unknown TTS model or backend: {model}")
        return object()

    def sample_rate_for(self, model):
        self.models.append(model)
        return 24000

    def synthesize(self, **kwargs):
        self.models.append(kwargs["model"])
        yield np.zeros(240, dtype=np.float32)


@pytest.mark.parametrize("model", [
    "gpt-realtime", "gpt-realtime-mini-2025-10-06",
    "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview-2024-12-17",
])
@pytest.mark.asyncio
async def test_ga_session_with_an_openai_realtime_model_speaks_with_tts_model(model):
    """The OpenAI SDK's realtime.connect(model=...) puts its realtime model on ?model=.
    OpenAI model names map to TTS_MODEL; the realtime name reached the TTS router."""
    from src.realtime import server

    websocket = _RecordingWebSocket()
    router = _ModelRecordingRouter()
    with patch.object(server.settings, "tts_model", "kokoro"):
        session = server.RealtimeSession(websocket, router, model=model, ga=True)
        await session._handle_response_create({"response": {"instructions": "Hello there"}})

    assert websocket.of_type("error") == []
    assert set(router.models) == {"kokoro"}
    assert websocket.of_type("response.done")[0]["response"]["status"] == "completed"


@pytest.mark.parametrize(("query_model", "update", "expected_model"), [
    ("", {"audio": {"output": {"voice": "alloy"}}}, "kokoro"),
    ("", {"model": "piper/en_US-lessac-medium", "audio": {"output": {"voice": "alloy"}}}, "piper/en_US-lessac-medium"),
    ("Systran/faster-whisper-small", {"audio": {"output": {"voice": "alloy"}}}, "kokoro"),
    # GA browser clients must offer the 'realtime' subprotocol, so they start as legacy
    ("piper/en_US-lessac-medium", {"audio": {"output": {"voice": "alloy"}}}, "piper/en_US-lessac-medium"),
], ids=["stt-default", "update-names-tts-model", "stt-query-model", "tts-query-model"])
@pytest.mark.asyncio
async def test_nested_ga_update_of_a_legacy_session_speaks_with_its_tts_model(query_model, update, expected_model):
    """README: 'realtime' subprotocol clients keep legacy events until they send a nested
    GA session update. The legacy session model is the STT model, which then became the
    TTS model and failed every reply. A TTS model the client chose in ?model= is kept."""
    from src.realtime import server

    websocket = _RecordingWebSocket()
    router = _ModelRecordingRouter()
    with (
        patch.object(server.settings, "tts_model", "kokoro"),
        patch.object(server.settings, "stt_model", "Systran/faster-whisper-base"),
    ):
        session = server.RealtimeSession(websocket, router, model=query_model)
        session.audio_buffer = InputAudioBuffer()
        await session.handle_event({"type": "session.update", "session": {"type": "realtime", **update}})
        await session._handle_response_create({"response": {"instructions": "Hello there"}})

    assert websocket.of_type("session.updated")[0]["session"]["model"] == expected_model
    assert websocket.of_type("error") == []
    assert set(router.models) == {expected_model}


@pytest.mark.asyncio
async def test_legacy_session_with_an_openai_realtime_model_keeps_it_on_upgrade():
    from src.realtime import server

    websocket = _RecordingWebSocket()
    router = _ModelRecordingRouter()
    with patch.object(server.settings, "tts_model", "kokoro"):
        session = server.RealtimeSession(websocket, router, model="gpt-realtime")
        session.audio_buffer = InputAudioBuffer()
        await session.handle_event({"type": "session.update", "session": {"audio": {"output": {"voice": "alloy"}}}})
        await session._handle_response_create({"response": {"instructions": "Hello there"}})

    assert websocket.of_type("session.updated")[0]["session"]["model"] == "gpt-realtime"
    assert set(router.models) == {"kokoro"}


@pytest.mark.parametrize("make_session", [
    lambda server: server.RealtimeSession(_RecordingWebSocket(), MagicMock()),
    lambda server: server.RealtimeSession(_RecordingWebSocket(), MagicMock(), model="gpt-4o-realtime-preview"),
], ids=["legacy", "beta"])
@pytest.mark.asyncio
async def test_transcription_null_commits_without_transcribing(make_session):
    """OpenAI beta session.input_audio_transcription "can be set to null to turn off once
    on", and session.updated echoes null. Only GA sessions honoured it."""
    from src.realtime import server

    session = make_session(server)
    session.audio_buffer = InputAudioBuffer()
    with patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hi"}) as transcribe:
        await session.handle_event({"type": "session.update", "session": {
            "input_audio_transcription": None, "turn_detection": None,
        }})
        session.audio_buffer.append(b"\x01\x00" * 1600)
        await session.handle_event({"type": "input_audio_buffer.commit"})

    websocket = session.ws
    assert websocket.of_type("session.updated")[0]["session"]["input_audio_transcription"] is None
    assert len(websocket.of_type("input_audio_buffer.committed")) == 1
    assert websocket.of_type("conversation.item.input_audio_transcription.completed") == []
    assert not transcribe.called


@pytest.mark.asyncio
async def test_input_audio_transcription_language_and_prompt_reach_stt():
    from src.realtime.server import RealtimeSession

    session = RealtimeSession(_RecordingWebSocket(), MagicMock())
    session.audio_buffer = InputAudioBuffer()
    await session.handle_event({"type": "session.update", "session": {
        "input_audio_transcription": {"model": "whisper-1", "language": "de", "prompt": "Open Speech"},
        "turn_detection": None,
    }})
    session.audio_buffer.append(b"\x01\x00" * 1600)

    with patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hallo"}) as transcribe:
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert transcribe.call_args.kwargs["language"] == "de"
    assert transcribe.call_args.kwargs["prompt"] == "Open Speech"


def _transcription_update(transcription: dict, *, ga: bool) -> dict:
    if ga:
        return {"type": "realtime", "audio": {"input": {"transcription": transcription}}}
    return {"input_audio_transcription": transcription}


@pytest.mark.parametrize("ga", [False, True], ids=["legacy", "ga"])
@pytest.mark.parametrize("language", ["klingon", "xx", 5])
@pytest.mark.asyncio
async def test_unknown_transcription_language_is_refused_and_the_previous_one_kept(ga, language):
    """faster-whisper checks the language only after decoding a turn, so a session with an
    unknown code failed every transcription. The update is refused when it arrives."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.audio_buffer = InputAudioBuffer()
    await session.handle_event({"type": "session.update", "session": {
        **_transcription_update({"model": "whisper-1", "language": "de-AT"}, ga=ga), "turn_detection": None,
    }})
    await session.handle_event({
        "type": "session.update", "event_id": "evt_language",
        "session": _transcription_update({"model": "whisper-1", "language": language}, ga=ga),
    })
    session.audio_buffer.append(b"\x01\x00" * 1600)
    with patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hallo"}) as transcribe:
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert len(websocket.of_type("session.updated")) == 1
    errors = websocket.of_type("error")
    assert len(errors) == 1
    assert errors[0]["error"]["type"] == "invalid_request_error"
    assert errors[0]["error"]["event_id"] == "evt_language"
    assert "language" in errors[0]["error"]["message"]
    assert session.config.input_audio_transcription["language"] == "de"
    assert transcribe.call_args.kwargs["language"] == "de"


_PARAKEET = "nvidia/parakeet-tdt-0.6b-v2"


@pytest.mark.parametrize(("ga", "stt_model"), [
    (False, "whisper-1"), (True, "whisper-1"), (True, _PARAKEET),
], ids=["legacy", "ga-stt-model", "ga-named-parakeet"])
@pytest.mark.asyncio
async def test_a_language_parakeet_cant_transcribe_is_reported_once_and_the_session_continues_in_english(
    ga, stt_model,
):
    """Parakeet transcribes English only. A session whose language it can't transcribe failed every turn with
    transcription_failed; as /v1/audio/stream does, the update now says so once, naming Whisper, and the turns
    are transcribed in English."""
    from src.realtime.server import RealtimeSession

    parakeet = _PARAKEET
    websocket = _RecordingWebSocket()
    with patch("src.realtime.server.settings.stt_model", parakeet):
        session = RealtimeSession(websocket, MagicMock())
        session.audio_buffer = InputAudioBuffer()
        transcription = {"model": stt_model, "language": "de"}
        await session.handle_event({
            "type": "session.update", "event_id": "evt_de",
            "session": {**_transcription_update(transcription, ga=ga), "turn_detection": None},
        })
        session.audio_buffer.append(b"\x01\x00" * 1600)
        with patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hello"}) as transcribe:
            await session.handle_event({"type": "input_audio_buffer.commit"})

    assert len(websocket.of_type("session.updated")) == 1
    errors = websocket.of_type("error")
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "language_unsupported"
    assert errors[0]["error"]["event_id"] == "evt_de"
    assert "Whisper" in errors[0]["error"]["message"]
    assert "continues in English" in errors[0]["error"]["message"]
    assert session.config.input_audio_transcription["language"] is None
    assert transcribe.call_args.kwargs["language"] is None
    assert transcribe.call_args.kwargs["model"] == parakeet
    assert websocket.of_type("conversation.item.input_audio_transcription.failed") == []
    assert websocket.of_type("conversation.item.input_audio_transcription.completed")[0]["transcript"] == "hello"


@pytest.mark.asyncio
async def test_a_whisper_session_keeps_its_language_when_parakeet_is_the_default():
    """Only the model that transcribes the turns counts: a session naming a Whisper model keeps German."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    with patch("src.realtime.server.settings.stt_model", _PARAKEET):
        session = RealtimeSession(websocket, MagicMock())
        session.audio_buffer = InputAudioBuffer()
        await session.handle_event({"type": "session.update", "session": {
            **_transcription_update({"model": "Systran/faster-whisper-small", "language": "de"}, ga=True),
            "turn_detection": None,
        }})
    assert websocket.of_type("error") == []
    assert session.config.input_audio_transcription["language"] == "de"


@pytest.mark.parametrize(("language", "expected"), [
    ("en-US", "en"), ("EN", "en"), ("nb", "no"), ("auto", None), ("", None), (None, None),
])
@pytest.mark.asyncio
async def test_transcription_language_is_reported_and_used_as_its_whisper_code(language, expected):
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.audio_buffer = InputAudioBuffer()
    transcription = {"model": "whisper-1", "language": language}
    await session.handle_event({"type": "session.update", "session": {
        "input_audio_transcription": transcription, "turn_detection": None,
    }})
    session.audio_buffer.append(b"\x01\x00" * 1600)
    with patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hi"}) as transcribe:
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert websocket.of_type("error") == []
    reported = websocket.of_type("session.updated")[0]["session"]["input_audio_transcription"]
    assert reported == {"model": "whisper-1", "language": expected}
    assert transcribe.call_args.kwargs["language"] == expected
    assert transcription["language"] == language  # the client's event is left as it was


@pytest.mark.asyncio
async def test_session_update_reports_turn_detection_off_when_vad_cannot_load():
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.config.update_from({"session": {"turn_detection": None}})
    session.audio_buffer = InputAudioBuffer()
    with patch("src.realtime.server.get_vad_model", new=AsyncMock(side_effect=RuntimeError("offline"))):
        await session.handle_event({
            "type": "session.update", "event_id": "evt_vad",
            "session": {"turn_detection": {"type": "server_vad"}},
        })

    assert [event["type"] for event in websocket.sent] == ["session.updated", "error"]
    assert websocket.sent[0]["session"]["turn_detection"] is None
    assert websocket.sent[1]["error"]["code"] == "vad_unavailable"
    assert websocket.sent[1]["error"]["event_id"] == "evt_vad"
    assert not session.config.vad_enabled


@pytest.mark.parametrize("response", [
    {"instructions": "x" * 51},
    {"input": [{"type": "message", "content": [{"type": "input_text", "text": "x" * 51}]}]},
], ids=["instructions", "input_text"])
@pytest.mark.asyncio
async def test_response_create_refuses_text_over_tts_max_input_length(response):
    """TTS_MAX_INPUT_LENGTH bounds every speech request; realtime replies ignored it."""
    from src.realtime import server

    websocket = _RecordingWebSocket()
    router = _ModelRecordingRouter()
    session = server.RealtimeSession(websocket, router)
    with patch.object(server.settings, "tts_max_input_length", 50):
        await session._handle_response_create({"event_id": "evt_long", "response": response})
        assert router.models == []
        await session._handle_response_create({"response": {"instructions": "x" * 50}})

    errors = websocket.of_type("error")
    assert [event["error"]["code"] for event in errors] == ["string_above_max_length"]
    assert errors[0]["error"]["event_id"] == "evt_long"
    assert [event["response"]["status"] for event in websocket.of_type("response.done")] == ["completed"]


@pytest.mark.asyncio
async def test_binary_frame_gets_an_error_event_and_the_session_goes_on():
    """Every other malformed frame gets an error event and the session goes on. A binary
    frame raised KeyError in receive_text, ending the session without a close frame."""
    from starlette.websockets import WebSocket

    from src.realtime import server

    incoming = [
        {"type": "websocket.connect"},
        {"type": "websocket.receive", "bytes": b"\x00\x01"},
        {"type": "websocket.receive", "text": json.dumps({"type": "input_audio_buffer.clear"})},
    ]
    sent = []
    cleared = asyncio.Event()

    async def receive():
        if incoming:
            return incoming.pop(0)
        await cleared.wait()
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)
        if "input_audio_buffer.cleared" in message.get("text", ""):
            cleared.set()

    scope = {
        "type": "websocket", "path": "/v1/realtime", "headers": [],
        "subprotocols": ["realtime"], "query_string": b"",
    }
    get_vad, silero = _scripted_vad([])
    with get_vad, silero:
        await asyncio.wait_for(
            server.realtime_endpoint(WebSocket(scope, receive, send), tts_router=MagicMock()), timeout=5,
        )

    received = [json.loads(m["text"]) for m in sent if m["type"] == "websocket.send"]
    assert [event["type"] for event in received] == ["session.created", "error", "input_audio_buffer.cleared"]
    assert received[1]["error"]["code"] == "invalid_event"


@pytest.mark.asyncio
async def test_vad_load_failure_reports_turn_detection_off():
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    with patch("src.realtime.server.get_vad_model", new=AsyncMock(side_effect=RuntimeError("offline"))):
        await session.initialize()

    assert [event["type"] for event in websocket.sent] == ["session.created", "error"]
    assert websocket.sent[0]["session"]["turn_detection"] is None
    assert websocket.sent[1]["error"]["code"] == "vad_unavailable"


@pytest.mark.asyncio
async def test_turn_detection_update_keeps_buffered_audio_and_session_audio_clock():
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    get_vad, silero = _scripted_vad([0.1] * 32 + [0.9] * 4)
    with get_vad, silero, _identity_decode():
        session = RealtimeSession(websocket, MagicMock())
        await session.initialize()
        await session.handle_event(_append(b"\x00\x00" * (32 * 512)))  # 1.024 s of silence
        buffered = session.audio_buffer.get_audio()
        await session.handle_event({"type": "session.update", "session": {
            "turn_detection": {"interrupt_response": False},
        }})
        assert session.audio_buffer.get_audio() == buffered
        await session.handle_event(_append(b"\x01\x00" * (4 * 512)))

    started = websocket.of_type("input_audio_buffer.speech_started")
    # Speech 1.024 s in; audio_start_ms includes the default 300 ms of prefix padding.
    assert [event["audio_start_ms"] for event in started] == [1024 - 300]


@pytest.mark.asyncio
async def test_failed_input_transcription_is_reported_for_its_item():
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.audio_buffer = InputAudioBuffer()
    session.audio_buffer.append(b"\x01\x00" * 1600)

    with patch("src.realtime.server.stt_router.transcribe", side_effect=RuntimeError("model unavailable")):
        await session.handle_event({"type": "input_audio_buffer.commit"})

    committed = websocket.of_type("input_audio_buffer.committed")[0]
    failed = websocket.of_type("conversation.item.input_audio_transcription.failed")
    assert [event["item_id"] for event in failed] == [committed["item_id"]]
    assert failed[0]["error"]["message"] == "model unavailable"


@pytest.mark.asyncio
async def test_a_failed_input_transcription_names_no_folders():
    # The log keeps the full message; the client's two events get the file's name only.
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.audio_buffer = InputAudioBuffer()
    session.audio_buffer.append(b"\x01\x00" * 1600)
    error = RuntimeError("Load model from /root/.cache/huggingface/hub/models--x/snapshots/abc/encoder.onnx failed")

    with patch("src.realtime.server.stt_router.transcribe", side_effect=error):
        await session.handle_event({"type": "input_audio_buffer.commit"})

    failed = websocket.of_type("conversation.item.input_audio_transcription.failed")
    errors = [e for e in websocket.of_type("error") if e["error"]["code"] == "transcription_error"]
    assert failed[0]["error"]["message"] == "Load model from encoder.onnx failed"
    assert [e["error"]["message"] for e in errors] == ["Load model from encoder.onnx failed"]


@pytest.mark.asyncio
async def test_speech_start_time_does_not_drift_with_small_pcm16_appends():
    """OpenAI: audio_start_ms counts from the start of all audio written to the buffer.
    Resampling each append on its own turned 64 samples at 24 kHz into 42 at 16 kHz
    instead of 42 2/3, so speech written 4.096 s in was reported 64 ms early."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    get_vad, silero = _patched_vad(lambda audio: 0.9 if np.abs(audio).max() > 0.05 else 0.0)
    quiet = _append(b"\x00\x00" * 64)
    loud = _append(np.full(64, 16000, dtype=np.int16).tobytes())
    with get_vad, silero:
        session = RealtimeSession(websocket, MagicMock())
        await session.initialize()
        # No padding, so audio_start_ms is where the speech itself starts
        await session.handle_event({"type": "session.update", "session": {
            "turn_detection": {"prefix_padding_ms": 0},
        }})
        for _ in range(1536):  # 4.096 s at 24 kHz
            await session.handle_event(quiet)
        for _ in range(24):  # 64 ms
            await session.handle_event(loud)

    started = websocket.of_type("input_audio_buffer.speech_started")
    assert len(started) == 1
    # VAD decides per 32 ms window, so that is the resolution to expect.
    assert abs(started[0]["audio_start_ms"] - 4096) <= 32


@pytest.mark.asyncio
async def test_input_format_switch_keeps_the_audio_the_resampler_held_back():
    from src.realtime.server import RealtimeSession

    session = RealtimeSession(_RecordingWebSocket(), MagicMock())
    session.config.update_from({"session": {"turn_detection": None}})
    session.audio_buffer = InputAudioBuffer()
    for _ in range(100):  # 1 s at 24 kHz in 10 ms appends
        await session.handle_event(_append(b"\x01\x00" * 240))
    await session.handle_event({"type": "session.update", "session": {"input_audio_format": "g711_ulaw"}})
    await session.handle_event(_append(g711.pcm16_to_ulaw(b"\x01\x00" * 4000)))  # 0.5 s at 8 kHz

    held_back = session._input_audio.flush()
    assert (session.audio_buffer.buffered_bytes + len(held_back)) // 2 == 16000 + 8000


@pytest.mark.parametrize("update", [
    {"input_audio_format": "g711_ulaw"},
    {"type": "realtime", "audio": {"input": {"format": {"type": "audio/pcmu"}}}},
], ids=["legacy", "ga"])
@pytest.mark.asyncio
async def test_input_format_update_keeps_buffered_audio(update):
    """Buffered audio is already 16 kHz PCM, so a new input format must not discard it."""
    from src.realtime.server import RealtimeSession

    session = RealtimeSession(_RecordingWebSocket(), MagicMock())
    session.audio_buffer = InputAudioBuffer()
    buffered = _make_pcm16_tone(100)
    session.audio_buffer.append(buffered)

    await session.handle_event({"type": "session.update", "session": update})

    assert session.config.input_audio_format == "g711_ulaw"
    assert session.audio_buffer.get_audio() == buffered


@pytest.mark.parametrize("audio_format, audio", [
    ("pcm16", np.full(1200, 3000, dtype="<i2").tobytes()),  # 50 ms at 24 kHz
    ("g711_ulaw", g711.pcm16_to_ulaw(np.full(400, 3000, dtype="<i2").tobytes())),  # 50 ms at 8 kHz
])
@pytest.mark.asyncio
async def test_manual_commit_of_exactly_50_ms_transcribes_all_of_it(audio_format, audio):
    """The input resampler holds back its newest ~1 ms until the next append. A commit
    left it there, so 50 ms of audio was rejected as "shorter than 50 ms"."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, MagicMock())
    session.config.update_from({"session": {"turn_detection": None, "input_audio_format": audio_format}})
    session.audio_buffer = InputAudioBuffer()
    await session.handle_event(_append(audio))

    with patch("src.realtime.server.stt_router.transcribe", return_value={"text": "hi"}) as transcribe:
        await session.handle_event({"type": "input_audio_buffer.commit"})

    assert websocket.of_type("error") == []
    with wave.open(io.BytesIO(transcribe.call_args.kwargs["audio"]), "rb") as wav_file:
        assert wav_file.getnframes() == 800  # all 50 ms at 16 kHz


@pytest.mark.asyncio
async def test_clear_drops_the_audio_the_resampler_held_back_but_keeps_the_session_clock():
    """input_audio_buffer.clear left the input resampler's newest ~1 ms in place, so
    cleared audio came out at the start of the next buffer."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    get_vad, silero = _patched_vad(lambda audio: 0.9 if np.abs(audio).max() > 0.25 else 0.0)
    with get_vad, silero:
        session = RealtimeSession(websocket, MagicMock())
        await session.initialize()
        # 100 ms at 24 kHz, too quiet for the detector
        await session.handle_event(_append(np.full(2400, 6000, dtype="<i2").tobytes()))
        await session.handle_event({"type": "input_audio_buffer.clear"})
        await session.handle_event(_append(b"\x00\x00" * 2304))  # three 32 ms VAD windows
        assert not np.frombuffer(session.audio_buffer.get_audio(), dtype="<i2").any()
        await session.handle_event(_append(np.full(2304, 16000, dtype="<i2").tobytes()))

    started = websocket.of_type("input_audio_buffer.speech_started")
    # OpenAI: the clock counts all audio written to the buffer, cleared audio included.
    # Speech came 96 ms after the clear, and the turn's 300 ms of padding reaches back to it.
    assert [event["audio_start_ms"] for event in started] == [100]


@pytest.mark.asyncio
async def test_g711_reply_chunks_are_encoded_as_one_stream():
    """Resampling each TTS chunk on its own turned 100 samples at 24 kHz into 33 at
    8 kHz instead of 33 1/3, and left a seam at every chunk boundary."""
    from src.realtime.server import RealtimeSession

    t = np.arange(24000) / 24000
    reply = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    class Router:
        def sample_rate_for(self, _model):
            return 24000

        def synthesize(self, **_kwargs):
            for start in range(0, len(reply), 100):
                yield reply[start:start + 100]

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, Router())
    session.config.update_from({"session": {"output_audio_format": "g711_ulaw", "turn_detection": None}})
    await session._handle_response_create({"response": {"instructions": "Hello"}})

    streamed = b"".join(
        base64.b64decode(event["delta"]) for event in websocket.of_type("response.audio.delta")
    )
    whole = encode_pcm16_to_format((reply * 32767).astype(np.int16).tobytes(), 24000, "g711_ulaw")
    assert len(streamed) == len(whole) == 8000
    assert streamed == whole


@pytest.mark.parametrize(("bad_sample", "expected"), [
    (np.nan, 0), (np.inf, 32767), (-np.inf, -32767),
], ids=["nan", "inf", "-inf"])
@pytest.mark.asyncio
async def test_non_finite_reply_samples_are_sanitized_like_every_other_tts_output(bad_sample, expected):
    """Casting NaN to int16 is undefined (numpy warns "invalid value encountered in cast");
    the other TTS paths map NaN to silence and infinities to full scale."""
    from src.realtime.server import RealtimeSession

    class Router:
        def sample_rate_for(self, _model):
            return 24000

        def synthesize(self, **_kwargs):
            yield np.array([0.5, bad_sample, -0.5], dtype=np.float32)

    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, Router())
    with warnings.catch_warnings():
        # The filters are process-wide, so this also covers the synthesis thread.
        warnings.simplefilter("error", RuntimeWarning)
        await session._handle_response_create({"response": {"instructions": "Hello"}})

    assert websocket.of_type("error") == []
    streamed = b"".join(
        base64.b64decode(event["delta"]) for event in websocket.of_type("response.audio.delta")
    )
    assert np.frombuffer(streamed, dtype=np.int16).tolist() == [16383, expected, -16383]


class _TextRecordingRouter:
    def __init__(self):
        self.requests = []

    def sample_rate_for(self, _model):
        return 24000

    def synthesize(self, **kwargs):
        self.requests.append(kwargs)
        yield np.zeros(240, dtype=np.float32)


@pytest.mark.parametrize("ga", [False, True], ids=["legacy", "ga"])
@pytest.mark.asyncio
async def test_reply_speaks_the_pronunciation_dictionary_and_reports_the_text_as_written(ga):
    """TTS_PRONUNCIATION_DICT applies server-wide; realtime replies spoke the raw text."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    router = _TextRecordingRouter()
    session = RealtimeSession(
        websocket, router, ga=ga, pronounce_fn=lambda text: text.replace("Nguyen", "Win"),
    )
    await session._handle_response_create({"response": {"instructions": "Hello Nguyen"}})

    assert [request["text"] for request in router.requests] == ["Hello Win"]
    content = websocket.of_type("response.done")[0]["response"]["output"][0]["content"][0]
    assert content["transcript"] == "Hello Nguyen"
    if ga:
        assert websocket.of_type("response.output_audio_transcript.delta")[0]["delta"] == "Hello Nguyen"
        assert websocket.of_type("response.output_audio_transcript.done")[0]["transcript"] == "Hello Nguyen"


class _LimitedRouter(_TextRecordingRouter):
    def max_input_chars_for(self, _model):
        return 20


@pytest.mark.parametrize(("pronounce_fn", "message"), [
    (lambda text: text.replace("NASA", "National Aeronautics and Space Administration"),
     "Input too long for {model}. Max: 20 characters"),
    (lambda text: text.replace("NASA", " "), "Input text is empty"),
], ids=["too-long", "empty"])
@pytest.mark.asyncio
async def test_reply_the_dictionary_makes_unspeakable_is_refused_as_for_speech(pronounce_fn, message):
    """/v1/audio/speech holds what the dictionary makes of the text to the model's own limit
    and refuses an empty result; realtime sent it to the model unchecked."""
    from src.config import settings
    from src.openai_compat import resolve_tts_model
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    router = _LimitedRouter()
    session = RealtimeSession(websocket, router, pronounce_fn=pronounce_fn)
    await session._handle_response_create({"response": {"instructions": "NASA"}})

    assert router.requests == []
    model = resolve_tts_model(None, settings)
    assert [event["error"]["message"] for event in websocket.of_type("error")] == [message.format(model=model)]
    assert websocket.of_type("response.done")[0]["response"]["status"] == "failed"


@pytest.mark.asyncio
async def test_reply_within_the_models_own_limit_is_spoken():
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    router = _LimitedRouter()
    session = RealtimeSession(websocket, router, pronounce_fn=lambda text: text)
    await session._handle_response_create({"response": {"instructions": "x" * 20}})

    assert [request["text"] for request in router.requests] == ["x" * 20]
    assert websocket.of_type("error") == []
    assert websocket.of_type("response.done")[0]["response"]["status"] == "completed"


def test_realtime_endpoint_gets_the_servers_pronunciation_dictionary(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from src import main as main_module
    from src.pronunciation.dictionary import PronunciationDictionary

    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps({"Nguyen": "Win"}), encoding="utf-8")
    monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(str(path)))
    received = {}

    async def endpoint(websocket, **kwargs):
        received.update(kwargs)
        await websocket.accept()
        await websocket.close()

    monkeypatch.setattr("src.realtime.server.realtime_endpoint", endpoint)
    with TestClient(main_module.app).websocket_connect("/v1/realtime") as websocket:
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert received["pronounce_fn"]("Hello Nguyen") == "Hello Win"


@pytest.mark.parametrize("voice", ["ash", "ballad", "coral", "sage", "verse", "marin", "cedar"])
@pytest.mark.asyncio
async def test_openai_ga_voices_speak_with_kokoro(monkeypatch, voice):
    """GA SDK clients pick these voices; Kokoro answered "Unknown Kokoro voice"."""
    from src.realtime.server import RealtimeSession
    from src.tts.backends import kokoro as kokoro_module
    from src.tts.router import TTSRouter

    class SilentKokoro(kokoro_module.KokoroBackend):
        def synthesize(self, text, voice, speed=1.0, lang_code=None, **_kwargs):
            yield np.zeros(240, dtype=np.float32)

    # Unknown voices fall back to scanning the local Hugging Face cache.
    monkeypatch.setattr(kokoro_module, "_discover_voices_from_package", lambda: None)
    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("kokoro", SilentKokoro(device="cpu"))
    websocket = _RecordingWebSocket()
    session = RealtimeSession(websocket, router, model="kokoro", ga=True)
    await session.handle_event({"type": "session.update", "session": {
        "type": "realtime", "audio": {"output": {"voice": voice}}, "turn_detection": None,
    }})
    await session._handle_response_create({"response": {"instructions": "Hello"}})

    assert websocket.of_type("error") == []
    assert websocket.of_type("response.done")[0]["response"]["status"] == "completed"


class TestAudioFormatConversion:
    def test_pcm16_passthrough_same_rate(self):
        """pcm16 at 24kHz → 24kHz should be ~passthrough (resampled to 16k target)."""
        data = _make_pcm16_tone(100, sample_rate=24000)
        result = decode_audio_to_pcm16(data, "pcm16", target_rate=16000)
        # Result should be shorter (16kHz vs 24kHz)
        expected_samples = int(24000 * 0.1 * 16000 / 24000)
        actual_samples = len(result) // 2
        assert abs(actual_samples - expected_samples) < 10

    def test_g711_ulaw_decode(self):
        """g711_ulaw should produce PCM16 output."""
        # Create some ulaw bytes
        pcm = _make_pcm16_tone(100, sample_rate=8000)
        ulaw = g711.pcm16_to_ulaw(pcm)
        result = decode_audio_to_pcm16(ulaw, "g711_ulaw", target_rate=16000)
        assert len(result) > 0
        # Should be resampled from 8kHz to 16kHz (~double)
        assert len(result) > len(ulaw)

    def test_g711_alaw_decode(self):
        pcm = _make_pcm16_tone(100, sample_rate=8000)
        alaw = g711.pcm16_to_alaw(pcm)
        result = decode_audio_to_pcm16(alaw, "g711_alaw", target_rate=16000)
        assert len(result) > 0

    def test_g711_codecs_match_audioop_for_every_sample_and_code(self):
        """audioop is gone in Python 3.13; its replacement must give the same bytes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            audioop = pytest.importorskip("audioop")
        pcm = np.arange(-32768, 32768, dtype="<i2").tobytes()
        codes = bytes(range(256))

        assert g711.pcm16_to_ulaw(pcm) == audioop.lin2ulaw(pcm, 2)
        assert g711.pcm16_to_alaw(pcm) == audioop.lin2alaw(pcm, 2)
        assert g711.ulaw_to_pcm16(codes) == audioop.ulaw2lin(codes, 2)
        assert g711.alaw_to_pcm16(codes) == audioop.alaw2lin(codes, 2)

    def test_realtime_audio_imports_without_audioop(self, monkeypatch):
        """Python 3.13 has no audioop module, and /v1/realtime must still load."""
        monkeypatch.setitem(sys.modules, "audioop", None)  # "import audioop" now fails
        for name in ("src.realtime.audio_buffer", "src.realtime.g711"):
            # Run a fresh copy; the module the other tests use stays in place.
            spec = importlib.util.find_spec(name)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    @pytest.mark.parametrize(
        ("conversion", "tone_hz"),
        [("input_pcm16_24k_to_16k", 10000), ("output_g711_24k_to_8k", 9000)],
    )
    def test_tones_above_the_new_nyquist_do_not_alias_into_the_speech_band(self, conversion, tone_hz):
        """Decimating without a low-pass filter folded a 10 kHz microphone tone into
        STT as 6 kHz, and a 9 kHz reply tone into G.711 as 1 kHz, at over 60% RMS."""
        def convert(pcm: bytes) -> np.ndarray:
            if conversion.startswith("input"):
                converted = decode_audio_to_pcm16(pcm, "pcm16", target_rate=16000)
            else:
                converted = g711.ulaw_to_pcm16(encode_pcm16_to_format(pcm, 24000, "g711_ulaw"))
            return np.frombuffer(converted, dtype=np.int16)[100:-100]

        tone = _make_pcm16_tone(1000, freq=tone_hz, sample_rate=24000)
        speech_band = _make_pcm16_tone(1000, freq=1000, sample_rate=24000)

        assert _rms(convert(tone)) < 0.01 * _rms(tone)  # -40 dB
        assert _rms(convert(speech_band)) == pytest.approx(_rms(speech_band), rel=0.02)

    def test_encode_pcm16_to_ulaw(self):
        pcm = _make_pcm16_tone(100, sample_rate=24000)
        result = encode_pcm16_to_format(pcm, 24000, "g711_ulaw")
        assert len(result) > 0
        # ulaw is 1 byte per sample at 8kHz
        expected_samples = int(24000 * 0.1 * 8000 / 24000)
        assert abs(len(result) - expected_samples) < 10

    def test_encode_pcm16_to_pcm16(self):
        pcm = _make_pcm16_tone(100, sample_rate=16000)
        result = encode_pcm16_to_format(pcm, 16000, "pcm16")
        # Should resample to 24kHz
        expected_samples = int(16000 * 0.1 * 24000 / 16000)
        actual_samples = len(result) // 2
        assert abs(actual_samples - expected_samples) < 10

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            decode_audio_to_pcm16(b"\x00" * 100, "mp3")

    def test_encode_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            encode_pcm16_to_format(b"\x00" * 100, 16000, "mp3")


# ---------------------------------------------------------------------------
# WebSocket integration tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------


class TestRealtimeWebSocket:
    """Integration tests for the /v1/realtime endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked backends."""
        import os
        os.environ["STT_SSL_ENABLED"] = "false"
        os.environ["OS_REALTIME_ENABLED"] = "true"

        from fastapi.testclient import TestClient

        from src.main import app
        return TestClient(app)

    def test_websocket_connect_and_session_created(self, client):
        """Connecting should receive a session.created event."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_session = MagicMock()
            mock_model = MagicMock()
            mock_model.session = mock_session
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                mock_vad_instance = MagicMock()
                MockVAD.return_value = mock_vad_instance

                with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                    data = ws.receive_json()
                    assert data["type"] == "session.created"
                    assert "session" in data
                    assert data["session"]["object"] == "realtime.session"

    def test_session_update(self, client):
        """session.update should return session.updated with new config."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                    ws.receive_json()  # session.created

                    ws.send_json({
                        "event_id": "evt_test",
                        "type": "session.update",
                        "session": {
                            "voice": "shimmer",
                            "input_audio_format": "g711_ulaw",
                        }
                    })

                    data = ws.receive_json()
                    assert data["type"] == "session.updated"
                    assert data["session"]["voice"] == "shimmer"
                    assert data["session"]["input_audio_format"] == "g711_ulaw"

    def test_unknown_event_returns_error(self, client):
        """Unknown event types should return an error."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                    ws.receive_json()  # session.created

                    ws.send_json({"type": "bogus.event", "event_id": "evt_1"})
                    data = ws.receive_json()
                    assert data["type"] == "error"
                    assert "unknown" in data["error"]["message"].lower()

    def test_invalid_json_returns_error(self, client):
        """Non-JSON text should return an error."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                    ws.receive_json()  # session.created
                    ws.send_text("not json {{{")
                    data = ws.receive_json()
                    assert data["type"] == "error"

    def test_input_audio_buffer_commit_transcription(self, client):
        """Committing audio should trigger transcription."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                with patch("src.realtime.server.stt_router") as mock_stt:
                    mock_stt.transcribe.return_value = {"text": "hello world"}
                    MockVAD.return_value.return_value = 0.1  # Low prob = no speech

                    with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                        ws.receive_json()  # session.created

                        # Send enough audio for transcription (>50ms at 16kHz)
                        # pcm16 format is 24kHz, so we need to send 24kHz audio
                        pcm = _make_pcm16_tone(200, sample_rate=24000)
                        audio_b64 = base64.b64encode(pcm).decode()

                        ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        })

                        ws.send_json({"type": "input_audio_buffer.commit"})

                        # Should get: committed, conversation.item.created, transcription.completed
                        events_received = []
                        for _ in range(3):
                            evt = ws.receive_json()
                            events_received.append(evt["type"])

                        assert "input_audio_buffer.committed" in events_received
                        assert "conversation.item.created" in events_received
                        assert "conversation.item.input_audio_transcription.completed" in events_received

    def test_input_audio_buffer_clear(self, client):
        """Clear should return cleared event."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                    ws.receive_json()  # session.created
                    ws.send_json({"type": "input_audio_buffer.clear"})
                    data = ws.receive_json()
                    assert data["type"] == "input_audio_buffer.cleared"

    def test_response_create_with_text(self, client):
        """response.create with instructions should stream TTS audio."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                # Mock the tts_router used by the session
                from src.main import tts_router as real_tts_router
                with patch.object(real_tts_router, "synthesize") as mock_synth:
                    mock_synth.return_value = iter([np.zeros(2400, dtype=np.float32)])

                    with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                        ws.receive_json()  # session.created

                        ws.send_json({
                            "type": "response.create",
                            "response": {
                                "instructions": "Hello, how are you?",
                            }
                        })

                        events_received = []
                        for _ in range(10):
                            try:
                                evt = ws.receive_json()
                                events_received.append(evt["type"])
                                if evt["type"] == "response.done":
                                    break
                            except Exception:
                                break

                        types = [e for e in events_received]
                        assert "response.created" in types
                        assert "response.done" in types

    def test_response_create_text_only_rejected(self, client):
        """response.create with modalities: ['text'] should be rejected."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                    ws.receive_json()  # session.created

                    ws.send_json({
                        "type": "response.create",
                        "response": {
                            "modalities": ["text"],
                            "instructions": "Hello",
                        }
                    })

                    data = ws.receive_json()
                    assert data["type"] == "error"
                    assert "text-only" in data["error"]["message"].lower() or "text" in data["error"]["message"].lower()

    def test_response_create_no_text_returns_error(self, client):
        """response.create without text should return error."""
        with patch("src.realtime.server.get_vad_model", new_callable=AsyncMock) as mock_vad:
            mock_model = MagicMock()
            mock_model.session = MagicMock()
            mock_vad.return_value = mock_model

            with patch("src.realtime.server.SileroVAD") as MockVAD:
                MockVAD.return_value = MagicMock()

                with client.websocket_connect("/v1/realtime", subprotocols=["realtime"]) as ws:
                    ws.receive_json()  # session.created

                    ws.send_json({
                        "type": "response.create",
                        "response": {}
                    })

                    data = ws.receive_json()
                    assert data["type"] == "error"


class TestRealtimeDisabled:
    """Test that the endpoint is disabled when OS_REALTIME_ENABLED=false."""

    def test_disabled_closes_connection(self, monkeypatch):
        monkeypatch.setenv("OS_REALTIME_ENABLED", "false")  # put back even when the test fails

        # Need to reload settings
        with patch("src.main.settings") as mock_settings:
            mock_settings.os_realtime_enabled = False
            mock_settings.os_api_key = ""
            mock_settings.stt_api_key = ""

            from fastapi.testclient import TestClient
            from starlette.websockets import WebSocketDisconnect

            from src.main import app

            client = TestClient(app)
            # The refusal for a disabled API, not any failure to connect.
            # The socket is accepted first, then closed, so a browser gets the reason.
            with client.websocket_connect("/v1/realtime") as websocket:
                with pytest.raises(WebSocketDisconnect) as closed:
                    websocket.receive_json()
            assert (getattr(closed.value, "code", None), closed.value.reason) == (4004, "Realtime API is disabled")


# A refused /v1/realtime socket is accepted and then closed with its
# code and reason, so a browser can say why (closed before accept(), it saw only 1006). No session is started.
@pytest.mark.parametrize(
    ("refusal", "code", "reason"),
    [
        ("disabled", 4004, "Realtime API is disabled"),
        ("origin", 1008, "Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it."),
        ("key", 4001, "Invalid or missing API key"),
    ],
)
def test_a_refused_realtime_socket_is_accepted_then_closed_with_its_code(monkeypatch, refusal, code, reason):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from src import main as main_module
    from src.routes import realtime as realtime_routes

    started = []

    async def endpoint(websocket, **kwargs):
        started.append(kwargs)
        await websocket.accept()
        await websocket.close()

    monkeypatch.setattr("src.realtime.server.realtime_endpoint", endpoint)
    monkeypatch.setattr(main_module.settings, "os_realtime_enabled", refusal != "disabled")
    if refusal == "origin":
        # The middleware refuses a foreign Origin before the route; this reaches the route's own check.
        monkeypatch.setattr(realtime_routes, "verify_ws_origin", lambda websocket: False)
    if refusal == "key":
        monkeypatch.setattr(main_module.settings, "os_api_key", "fake-key-for-realtime-refusal")

    with TestClient(main_module.app).websocket_connect("/v1/realtime") as websocket:  # accepted
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()

    assert (closed.value.code, closed.value.reason) == (code, reason)
    assert started == []


# ---------------------------------------------------------------------------
# The AI assistant over /v1/realtime (docs/AI-ASSISTANT.md, "Talking to it" and "Waiting phrases"). The waiting
# phrases' timing, talking over the wait and response.cancel are in test_realtime_barge_in.py.
# ---------------------------------------------------------------------------


def _reply(text, thinking=None, finish_reason="stop"):
    from src.assistant import AssistantReply

    return AssistantReply(text=text, thinking=thinking, finish_reason=finish_reason)


class _FakeAssistant:
    """Stands in for src.assistant.AssistantClient; nothing reaches a network.

    Each call to reply() takes the next step of its script: an AssistantReply is the answer, an
    AssistantError is raised, and an asyncio.Event is waited on before answering with ``after``.
    """

    def __init__(self, *script, after=None):
        self.script = list(script)
        self.after = after
        self.calls = []
        self.cancelled = 0

    async def reply(self, messages, settings):
        self.calls.append((copy.deepcopy(messages), settings))
        step = self.script.pop(0) if self.script else self.after
        if isinstance(step, asyncio.Event):
            try:
                await step.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            step = self.after
        if isinstance(step, BaseException):
            raise step
        return step


@pytest.fixture
def assistant_env(monkeypatch, tmp_path):
    """The documented .env defaults for the presets, and a preset file of the test's own."""
    from src.config import settings

    for name, value in {
        "os_assistant_model": "test-model",
        "os_assistant_system_prompt": "",
        "os_assistant_thinking": "default",
        "os_assistant_reasoning_effort": "",
        "os_assistant_max_tokens": 0,
        "os_assistant_temperature": None,
        "os_assistant_top_p": None,
        "os_assistant_timeout_s": 120.0,
        "os_assistant_history_turns": 20,
        "os_assistant_show_thinking": True,
        "os_assistant_speak_errors": True,
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "os_studio_db_path", str(tmp_path / "data" / "studio.db"))
    return settings


async def _assistant_session(assistant, *, router=None, ga=True, update=None, turns=4):
    """A session with server VAD whose turns each take one _say(); no waiting phrases unless asked."""
    from src.realtime.server import RealtimeSession

    update = dict(update or {})
    session_update = {
        "turn_detection": {"silence_duration_ms": 64},
        **update,
        "assistant": {"fillers": [], **update.get("assistant", {})},
    }
    get_vad, silero = _scripted_vad([0.9, 0.9, 0.9, 0.1, 0.1] * turns)
    with get_vad, silero:
        session = RealtimeSession(_RecordingWebSocket(), router or _TextRecordingRouter(), ga=ga, assistant=assistant)
        await session.initialize()
        await session.handle_event({"type": "session.update", "event_id": "evt_setup", "session": session_update})
    return session


async def _finish(session):
    task = session._response_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=5)


async def _say(session, transcript):
    """One spoken turn (speech, then 64 ms of silence) heard as ``transcript``; waits for its reply."""
    with _identity_decode(), patch("src.realtime.server.stt_router.transcribe", return_value={"text": transcript}):
        await session.handle_event(_append(b"\x01\x00" * (5 * 512)))
    await _finish(session)


def _first(sent, event_type, start=0):
    return next(index for index, event in enumerate(sent) if index >= start and event["type"] == event_type)


@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.asyncio
async def test_a_turn_goes_to_the_ai_with_the_prompt_and_history_and_the_reply_is_spoken(assistant_env, ga):
    """The transcript, rendered with the preset's user template, goes to the AI
    after the system prompt and the history; the reply's text is sent before its audio; the history holds both."""
    from src.assistant.presets import VOICE_SYSTEM_PROMPT

    fake = _FakeAssistant(_reply("It's sunny."), _reply("About twenty degrees."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, ga=ga, update={"assistant": {"user_template": "Asked: {text}"}})
    try:
        await _say(session, "What's the weather?")
        await _say(session, "  And the temperature? ")
    finally:
        await session.close()

    system = {"role": "system", "content": VOICE_SYSTEM_PROMPT}
    first = {"role": "user", "content": "Asked: What's the weather?"}
    answer = {"role": "assistant", "content": "It's sunny."}
    second = {"role": "user", "content": "Asked: And the temperature?"}
    assert [messages for messages, _settings in fake.calls] == [[system, first], [system, first, answer, second]]
    assert [request["text"] for request in router.requests] == ["It's sunny.", "About twenty degrees."]
    assert session.conversation.for_request(20) == [first, answer, second, {"role": "assistant", "content": "About twenty degrees."}]

    sent = session.ws.sent
    prefix = "response.output_audio" if ga else "response.audio"
    order = [
        _first(sent, "conversation.item.input_audio_transcription.completed"),
        _first(sent, "response.created"),
        _first(sent, f"{prefix}_transcript.delta"),
        _first(sent, f"{prefix}.delta"),
        _first(sent, f"{prefix}.done"),
        _first(sent, f"{prefix}_transcript.done"),
        _first(sent, "response.done"),
    ]
    assert order == sorted(order), [sent[index]["type"] for index in order]
    assert sent[order[2]]["delta"] == "It's sunny."
    assert sent[order[5]]["transcript"] == "It's sunny."
    done = session.ws.of_type("response.done")
    assert [event["response"]["status"] for event in done] == ["completed", "completed"]
    assert done[0]["response"]["output"][0]["content"][0]["transcript"] == "It's sunny."
    assert session.ws.of_type("error") == []


@pytest.mark.parametrize("case", ["create_response_false", "transcription_session", "no_assistant"])
@pytest.mark.asyncio
async def test_no_ai_call_without_create_response_in_a_transcription_session_or_without_an_assistant(assistant_env, case):
    """A turn is only transcribed in these cases, as before."""
    fake = None if case == "no_assistant" else _FakeAssistant(_reply("Unused."))
    update = {
        "create_response_false": {"turn_detection": {"silence_duration_ms": 64, "create_response": False}},
        "transcription_session": {"type": "transcription"},
        "no_assistant": {},
    }[case]
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, update=update)
    try:
        await _say(session, "Is anyone there?")
    finally:
        await session.close()

    assert fake is None or fake.calls == []
    assert [e["transcript"] for e in session.ws.of_type("conversation.item.input_audio_transcription.completed")] == [
        "Is anyone there?"
    ]
    assert session.ws.of_type("response.created") == []
    assert router.requests == []
    assert session.ws.of_type("error") == []


@pytest.mark.parametrize(("assistant", "asked", "reported"), [
    (True, None, True), (True, False, False), (False, None, False), (False, True, False),
], ids=["assistant-default", "assistant-asked-false", "no-assistant-default", "no-assistant-asked-true"])
@pytest.mark.asyncio
async def test_create_response_defaults_to_whether_the_assistant_is_on_and_is_reported_honestly(assistant, asked, reported):
    """As in OpenAI's API, create_response true means a reply after each turn,
    so a session that can't reply reports false even when the client asks for true."""
    from src.realtime.server import RealtimeSession

    websocket = _RecordingWebSocket()
    get_vad, silero = _scripted_vad([])
    with get_vad, silero:
        session = RealtimeSession(websocket, MagicMock(), ga=True, assistant=_FakeAssistant() if assistant else None)
        await session.initialize()
        if asked is not None:
            await session.handle_event({"type": "session.update", "session": {
                "type": "realtime", "audio": {"input": {"turn_detection": {"create_response": asked}}},
            }})

    shown = (websocket.of_type("session.updated") or websocket.of_type("session.created"))[-1]["session"]
    assert shown["audio"]["input"]["turn_detection"]["create_response"] is reported
    assert session.config.turn_detection.create_response is reported


@pytest.mark.asyncio
async def test_response_create_with_text_is_still_spoken_word_for_word_with_the_assistant_on(assistant_env):
    """Text in response.create is spoken as given,
    with no AI call and no waiting phrases, and the conversation is left alone."""
    fake = _FakeAssistant(_reply("Unused."))
    router = _TextRecordingRouter()
    session = await _assistant_session(
        fake, router=router, update={"assistant": {"fillers": ["Hmm."], "filler_after_ms": 0, "filler_every_ms": 0}},
    )
    try:
        await session.handle_event({"type": "response.create", "response": {"instructions": "Read this."}})
        await _finish(session)
    finally:
        await session.close()

    assert fake.calls == []
    assert "Read this." in [request["text"] for request in router.requests]
    assert session.ws.of_type("open_speech.assistant.filler") == []
    done = session.ws.of_type("response.done")[0]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"][0]["transcript"] == "Read this."
    assert session.conversation.for_request(20) == []


@pytest.mark.asyncio
async def test_a_preset_and_overrides_in_session_update_reach_the_ai(assistant_env):
    """session.update's assistant block picks the preset and overrides its fields."""
    from src.assistant import presets

    presets.create_preset({"name": "Short answers", "system_prompt": "Answer in five words.", "max_tokens": 64})
    fake = _FakeAssistant(_reply("Hello to you too."))
    session = await _assistant_session(fake, update={"assistant": {"preset": "short-answers", "temperature": 0.2}})
    try:
        await _say(session, "Hello")
    finally:
        await session.close()

    messages, used = fake.calls[0]
    assert messages[0] == {"role": "system", "content": "Answer in five words."}
    assert (used.preset_id, used.max_tokens, used.temperature) == ("short-answers", 64, 0.2)
    shown = session.ws.of_type("session.updated")[-1]["session"]
    assert shown["assistant"] == {"preset": "short-answers", "fillers": [], "temperature": 0.2}
    assert "os_assistant_url" not in json.dumps(shown) and "api_key" not in json.dumps(shown)


@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.asyncio
async def test_session_instructions_are_the_system_prompt(assistant_env, ga):
    """OpenAI's session instructions are the session's system prompt."""
    fake = _FakeAssistant(_reply("Arr."))
    session = await _assistant_session(fake, ga=ga, update={"instructions": "You are a pirate."})
    try:
        await _say(session, "Hello")
    finally:
        await session.close()

    assert fake.calls[0][0][0] == {"role": "system", "content": "You are a pirate."}
    assert session.ws.of_type("session.updated")[-1]["session"]["instructions"] == "You are a pirate."


@pytest.mark.parametrize("bad", [
    {"assistant": {"filler_max": 11}},
    {"assistant": {"preset": "no-such-preset"}},
    {"assistant": {"colour": "red"}},
    {"assistant": {"user_template": "No placeholder"}},
    {"assistant": "voice"},
    {"instructions": 5},
    {"instructions": "x" * 8001},
], ids=["out-of-range", "unknown-preset", "unknown-field", "template-without-text", "not-an-object",
        "instructions-not-text", "instructions-too-long"])
@pytest.mark.asyncio
async def test_a_bad_assistant_value_refuses_the_whole_session_update(assistant_env, bad):
    """Values are validated through presets.effective; a bad one changes nothing."""
    fake = _FakeAssistant(_reply("Unused."))
    session = await _assistant_session(fake)
    try:
        await session.handle_event({"type": "session.update", "event_id": "evt_bad", "session": {"voice": "nova", **bad}})
    finally:
        await session.close()

    assert len(session.ws.of_type("session.updated")) == 1  # only the setup's
    errors = session.ws.of_type("error")
    assert [(e["error"]["code"], e["error"]["event_id"]) for e in errors] == [("invalid_request", "evt_bad")]
    assert session.config.voice != "nova"


@pytest.mark.parametrize("code", [
    "assistant_unreachable", "assistant_timeout", "assistant_http_error",
    "assistant_bad_response", "assistant_empty_reply", "assistant_reply_cut_off",
])
@pytest.mark.asyncio
async def test_an_ai_error_is_reported_then_spoken_then_fails_the_response(assistant_env, code):
    """Error with the code, the preset's sentence as audio, a failed response.done
    with the code, and the history as it was."""
    from src.assistant import ERROR_SENTENCES, AssistantError

    fake = _FakeAssistant(AssistantError("The AI server answered HTTP 500", code))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router)
    try:
        await _say(session, "Hello?")
    finally:
        await session.close()

    sent = session.ws.sent
    errors = session.ws.of_type("error")
    assert [event["error"]["code"] for event in errors] == [code]
    assert [request["text"] for request in router.requests] == [ERROR_SENTENCES[code]]
    error_at = _first(sent, "error")
    assert error_at < _first(sent, "response.output_audio.delta") < _first(sent, "response.done")
    done = session.ws.of_type("response.done")[-1]["response"]
    assert done["status"] == "failed"
    assert done["status_details"] == {"type": "failed", "error": {"type": "server_error", "code": code}}
    assert session.conversation.for_request(20) == [{"role": "user", "content": "Hello?"}]


@pytest.mark.parametrize(("assistant", "spoken"), [
    ({"speak_errors": False}, []),
    ({"error_sentences": {"assistant_timeout": "Ada is slow today."}}, ["Ada is slow today."]),
], ids=["speak-errors-off", "preset-sentence"])
@pytest.mark.asyncio
async def test_an_ai_error_speaks_the_presets_sentence_or_nothing(assistant_env, assistant, spoken):
    """A preset's error sentence replaces the default; speak_errors off says nothing."""
    from src.assistant import AssistantError

    fake = _FakeAssistant(AssistantError("No whole reply within 120 s", "assistant_timeout"))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, update={"assistant": assistant})
    try:
        await _say(session, "Hello?")
    finally:
        await session.close()

    assert [request["text"] for request in router.requests] == spoken
    assert [event["error"]["code"] for event in session.ws.of_type("error")] == ["assistant_timeout"]
    assert session.ws.of_type("response.done")[-1]["response"]["status"] == "failed"


@pytest.mark.parametrize("show", [True, False], ids=["shown", "hidden"])
@pytest.mark.asyncio
async def test_thinking_is_sent_as_text_never_spoken_and_never_kept(assistant_env, show):
    """open_speech.assistant.thinking carries the thinking when show_thinking is on."""
    fake = _FakeAssistant(_reply("Hello!", thinking="The user greets me, so I greet back."), _reply("Fine."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, update={"assistant": {"show_thinking": show}})
    try:
        await _say(session, "Hi")
        await _say(session, "How are you?")
    finally:
        await session.close()

    thinking = session.ws.of_type("open_speech.assistant.thinking")
    assert [event["text"] for event in thinking] == (["The user greets me, so I greet back."] if show else [])
    if show:
        sent = session.ws.sent
        assert sent.index(thinking[0]) < _first(sent, "response.output_audio_transcript.delta")
        assert thinking[0]["response_id"] == session.ws.of_type("response.created")[0]["response"]["id"]
    assert [request["text"] for request in router.requests] == ["Hello!", "Fine."]
    assert "greet back" not in json.dumps(fake.calls[1][0])
    assert "greet back" not in json.dumps(session.ws.of_type("response.done"))


class _CappedRouter(_TextRecordingRouter):
    def max_input_chars_for(self, _model):
        return 60


@pytest.mark.asyncio
async def test_a_long_reply_is_spoken_in_full_in_segments(assistant_env):
    """An AI reply longer than TTS_MAX_INPUT_LENGTH and the model's own limit is
    never refused; it is spoken segment by segment, each within the model's limit."""
    from src.realtime import server

    reply = " ".join(f"Sentence number {index} is part of a long answer." for index in range(12))
    fake = _FakeAssistant(_reply(reply))
    router = _CappedRouter()
    session = await _assistant_session(fake, router=router)
    try:
        with patch.object(server.settings, "tts_max_input_length", 50):
            await _say(session, "Tell me everything.")
    finally:
        await session.close()

    texts = [request["text"] for request in router.requests]
    assert len(texts) > 1
    assert all(len(text) <= 60 for text in texts), texts
    assert " ".join(texts).split() == reply.split()
    assert session.ws.of_type("error") == []
    done = session.ws.of_type("response.done")[0]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"][0]["transcript"] == reply


@pytest.mark.asyncio
async def test_a_markdown_reply_is_spoken_cleaned(assistant_env):
    """Emphasis removed, links read by their label,
    a code block replaced with "Code block skipped."."""
    reply = "Here is **how** it works:\n\n```python\nprint('secret code')\n```\n\nSee [the guide](https://example.test/guide)."
    fake = _FakeAssistant(_reply(reply))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router)
    try:
        await _say(session, "How?")
    finally:
        await session.close()

    spoken = " ".join(request["text"] for request in router.requests)
    assert "Code block skipped." in spoken
    assert "how it works" in spoken and "the guide" in spoken
    for raw in ("**", "print(", "https://", "```", "["):
        assert raw not in spoken, spoken


@pytest.mark.asyncio
async def test_a_named_voice_is_resolved_for_the_reply(assistant_env):
    """A named voice is resolved with resolve_named_voice."""
    from src.realtime import server

    fake = _FakeAssistant(_reply("Hello."))
    router = _TextRecordingRouter()
    with patch("src.realtime.server.resolve_named_voice", return_value=("af_heart+am_adam", None)) as resolve:
        session = await _assistant_session(fake, router=router, update={"voice": "voice:narrator"})
        try:
            await _say(session, "Hi")
            await session.handle_event({"type": "response.create", "response": {"instructions": "Word for word."}})
            await _finish(session)
        finally:
            await session.close()

    assert [request["voice"] for request in router.requests] == ["af_heart+am_adam", "af_heart+am_adam"]
    assert resolve.call_args.args == (server.settings.tts_model, "voice:narrator")
    assert session.ws.of_type("session.updated")[-1]["session"]["audio"]["output"]["voice"] == "voice:narrator"


@pytest.mark.asyncio
async def test_an_unknown_named_voice_refuses_the_session_update(assistant_env):
    """As in the Live Reader, a named voice that doesn't resolve is refused."""
    from src.voice_identities import NamedVoiceNotFoundError

    fake = _FakeAssistant(_reply("Unused."))
    session = await _assistant_session(fake)
    try:
        with patch("src.realtime.server.resolve_named_voice", side_effect=NamedVoiceNotFoundError("Named voice not found")):
            await session.handle_event({"type": "session.update", "event_id": "evt_voice", "session": {"voice": "voice:gone"}})
    finally:
        await session.close()

    assert [e["error"]["event_id"] for e in session.ws.of_type("error")] == ["evt_voice"]
    assert session.config.voice != "voice:gone"


def test_the_realtime_route_passes_the_shared_ai_client(monkeypatch):
    """Each connection gets src.assistant.get_client() as its assistant."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from src import assistant
    from src import main as main_module

    fake = _FakeAssistant()
    received = {}

    async def endpoint(websocket, **kwargs):
        received.update(kwargs)
        await websocket.accept()
        await websocket.close()

    monkeypatch.setattr("src.realtime.server.realtime_endpoint", endpoint)
    previous = assistant.get_client()
    assistant.set_client(fake)
    try:
        with (
            TestClient(main_module.app).websocket_connect("/v1/realtime") as websocket,
            pytest.raises(WebSocketDisconnect),
        ):
            websocket.receive_json()
    finally:
        assistant.set_client(previous)

    assert received["assistant"] is fake


# ---------------------------------------------------------------------------
# Typed questions: conversation.item.create with a user input_text message. The
# contract is OpenAI's Realtime protocol (GA and legacy shapes). The item joins the conversation as a
# spoken turn does; as in OpenAI's API, the client's response.create asks for the answer.
# ---------------------------------------------------------------------------


def _typed(text, *, event_id="evt_typed", **item):
    return {
        "type": "conversation.item.create", "event_id": event_id,
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}], **item},
    }


async def _until(predicate, timeout=3.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.asyncio
async def test_a_typed_question_joins_the_conversation_and_response_create_answers_it_aloud(assistant_env, ga):
    """The item is acknowledged as OpenAI does (conversation.item.added and .done
    in GA, conversation.item.created in legacy), joins the conversation through the preset's user template like
    a transcript, starts no reply of its own (even with create_response on), and response.create answers it
    with the spoken-turn flow. A later spoken turn shares the same conversation."""
    from src.assistant.presets import VOICE_SYSTEM_PROMPT

    fake = _FakeAssistant(_reply("It's sunny."), _reply("Rain, later."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, ga=ga, update={"assistant": {"user_template": "Asked: {text}"}})
    assert session.config.turn_detection.create_response is True
    try:
        await session.handle_event(_typed("  What's the weather? "))
        acknowledged = list(session.ws.sent)
        assert session.ws.of_type("response.created") == [] and fake.calls == []
        await session.handle_event({"type": "response.create", "event_id": "evt_answer"})
        await _finish(session)
        await _say(session, "And tomorrow?")
    finally:
        await session.close()

    added = "conversation.item.added" if ga else "conversation.item.created"
    item = next(event for event in acknowledged if event["type"] == added)["item"]
    assert item["id"].startswith("item_")
    assert (item["object"], item["type"], item["role"], item["status"]) == ("realtime.item", "message", "user", "completed")
    assert item["content"] == [{"type": "input_text", "text": "  What's the weather? "}]
    done_items = [event for event in acknowledged if event["type"] == "conversation.item.done"]
    assert [event["item"] for event in done_items] == ([item] if ga else [])
    assert [event["type"] for event in acknowledged[-(2 if ga else 1):]] == ([added, "conversation.item.done"] if ga else [added])

    system = {"role": "system", "content": VOICE_SYSTEM_PROMPT}
    typed = {"role": "user", "content": "Asked: What's the weather?"}
    answer = {"role": "assistant", "content": "It's sunny."}
    spoken = {"role": "user", "content": "Asked: And tomorrow?"}
    assert [messages for messages, _settings in fake.calls] == [[system, typed], [system, typed, answer, spoken]]
    assert [request["text"] for request in router.requests] == ["It's sunny.", "Rain, later."]
    assert session.conversation.for_request(20) == [typed, answer, spoken, {"role": "assistant", "content": "Rain, later."}]

    sent = session.ws.sent
    prefix = "response.output_audio" if ga else "response.audio"
    order = [
        _first(sent, added),
        _first(sent, "response.created"),
        _first(sent, f"{prefix}_transcript.delta"),
        _first(sent, f"{prefix}.delta"),
        _first(sent, f"{prefix}.done"),
        _first(sent, f"{prefix}_transcript.done"),
        _first(sent, "response.done"),
    ]
    assert order == sorted(order), [sent[index]["type"] for index in order]
    assert sent[order[2]]["delta"] == "It's sunny."
    done = session.ws.of_type("response.done")
    assert [event["response"]["status"] for event in done] == ["completed", "completed"]
    assert done[0]["response"]["output"][0]["content"][0]["transcript"] == "It's sunny."
    assert session.ws.of_type("error") == []


@pytest.mark.asyncio
async def test_a_typed_item_keeps_the_clients_id_and_joins_its_text_parts(assistant_env):
    """As in OpenAI's API, a client may name the item; several input_text parts
    are one message. Text of exactly the limit is accepted."""
    from src.config import ASSISTANT_PROMPT_MAX_CHARS

    fake = _FakeAssistant(_reply("Noted."))
    session = await _assistant_session(fake)
    try:
        await session.handle_event({
            "type": "conversation.item.create",
            "previous_item_id": None,
            "item": {
                "id": "msg_001", "object": "realtime.item", "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "First line."}, {"type": "input_text", "text": "Second line."}],
            },
        })
        await session.handle_event(_typed("x" * ASSISTANT_PROMPT_MAX_CHARS))
    finally:
        await session.close()

    items = [event["item"] for event in session.ws.of_type("conversation.item.added")]
    assert items[0]["id"] == "msg_001"
    assert [part["text"] for part in items[0]["content"]] == ["First line.", "Second line."]
    assert session.conversation.for_request(20) == [
        {"role": "user", "content": "First line.\nSecond line."},
        {"role": "user", "content": "x" * ASSISTANT_PROMPT_MAX_CHARS},
    ]
    assert session.ws.of_type("error") == []


@pytest.mark.parametrize(("event", "code"), [
    ({"type": "conversation.item.create"}, "invalid_request"),
    ({"type": "conversation.item.create", "item": "Hello"}, "invalid_request"),
    ({"type": "conversation.item.create", "item": {
        "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}}, "invalid_request"),
    ({"type": "conversation.item.create", "item": {
        "type": "function_call_output", "call_id": "call_1", "output": "{}"}}, "invalid_request"),
    (_typed("Hello", role="assistant"), "invalid_request"),
    (_typed("Hello", role="system"), "invalid_request"),
    ({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "AAAA"}]}}, "invalid_request"),
    ({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "item_reference", "id": "item_1"}]}}, "invalid_request"),
    ({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"},
                                                        {"type": "input_image", "image_url": "data:,"}]}}, "invalid_request"),
    ({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": []}}, "invalid_request"),
    ({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": "Hello"}}, "invalid_request"),
    ({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": 5}]}}, "invalid_request"),
    (_typed(" \n "), "invalid_request"),
    (_typed("x" * 8001), "string_above_max_length"),
    (_typed("Hello", id=7), "invalid_request"),
    (_typed("Hello", id="x" * 33), "invalid_request"),
    ({**_typed("Hello"), "previous_item_id": "root"}, "invalid_request"),
], ids=[
    "no-item", "item-not-an-object", "function-call", "function-call-output", "assistant-role", "system-role",
    "input-audio", "item-reference", "input-image", "no-content", "content-not-a-list", "text-not-a-string",
    "blank-text", "text-too-long", "id-not-text", "id-too-long", "previous-item-id",
])
@pytest.mark.asyncio
async def test_a_typed_item_that_isnt_a_users_text_message_is_refused(assistant_env, event, code):
    """Other item types, roles and content, and text over the limit, get an
    invalid_request_error with the event's id. Nothing joins the conversation and nothing is asked."""
    fake = _FakeAssistant(_reply("Unused."))
    session = await _assistant_session(fake)
    try:
        await session.handle_event({**event, "event_id": "evt_bad"})
    finally:
        await session.close()

    errors = session.ws.of_type("error")
    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in errors] == [
        ("invalid_request_error", code, "evt_bad")
    ]
    assert session.ws.of_type("conversation.item.added") == []
    assert session.conversation.for_request(20) == []
    assert fake.calls == []


@pytest.mark.parametrize("case", ["no_assistant", "transcription_session", "tts_disabled"])
@pytest.mark.asyncio
async def test_a_typed_item_is_refused_when_nothing_would_answer_it(assistant_env, monkeypatch, case):
    """Without the AI assistant, in a transcription session or with TTS off, a
    typed question is refused with an invalid_request_error; response.create behaves as before."""
    if case == "tts_disabled":
        monkeypatch.setattr(assistant_env, "tts_enabled", False)
    fake = None if case == "no_assistant" else _FakeAssistant(_reply("Unused."))
    update = {"type": "transcription"} if case == "transcription_session" else {}
    session = await _assistant_session(fake, update=update)
    try:
        await session.handle_event(_typed("Hello?"))
        if case != "transcription_session":
            await session.handle_event({"type": "response.create", "event_id": "evt_answer"})
            await _finish(session)
    finally:
        await session.close()

    errors = [(e["error"]["type"], e["error"]["code"]) for e in session.ws.of_type("error")]
    refusal = {"no_assistant": "assistant_disabled", "transcription_session": "invalid_request", "tts_disabled": "tts_disabled"}[case]
    # Then response.create with no text, as before this change.
    after = {"no_assistant": ["missing_input"], "transcription_session": [], "tts_disabled": ["tts_disabled"]}[case]
    assert errors == [("invalid_request_error", refusal)] + [("invalid_request_error", code) for code in after]
    assert session.ws.of_type("error")[0]["error"]["event_id"] == "evt_typed"
    assert session.ws.of_type("conversation.item.added") == []
    assert session.conversation.for_request(20) == []
    assert fake is None or fake.calls == []


@pytest.mark.asyncio
async def test_a_typed_question_gets_the_waiting_phrases_and_a_spoken_error_like_a_spoken_turn(assistant_env):
    """The answer to a typed question goes through the spoken turn's path, so the
    waiting phrases play while the AI works, and its error is reported, spoken and fails the response."""
    from src.assistant import ERROR_SENTENCES, AssistantError

    release = asyncio.Event()
    fake = _FakeAssistant(release, after=AssistantError("No whole reply within 120 s", "assistant_timeout"))
    router = _TextRecordingRouter()
    session = await _assistant_session(
        fake, router=router, update={"assistant": {"fillers": ["Hmm."], "filler_after_ms": 1, "filler_max": 1}},
    )
    try:
        await session.handle_event(_typed("Hello?"))
        await session.handle_event({"type": "response.create"})
        await _until(lambda: session.ws.of_type("open_speech.assistant.filler"))
        release.set()
        await _finish(session)
    finally:
        release.set()
        await session.close()

    sent = session.ws.sent
    assert [event["text"] for event in session.ws.of_type("open_speech.assistant.filler")] == ["Hmm."]
    assert _first(sent, "open_speech.assistant.filler") < _first(sent, "error") < _first(sent, "response.done")
    assert [event["error"]["code"] for event in session.ws.of_type("error")] == ["assistant_timeout"]
    texts = [request["text"] for request in router.requests]  # the phrase may also be made ahead
    assert set(texts) == {"Hmm.", ERROR_SENTENCES["assistant_timeout"]}
    assert texts[-1] == ERROR_SENTENCES["assistant_timeout"]
    assert session.ws.of_type("response.done")[-1]["response"]["status"] == "failed"
    assert session.conversation.for_request(20) == [{"role": "user", "content": "Hello?"}]


@pytest.mark.asyncio
async def test_talking_over_a_typed_questions_wait_cancels_it_and_asks_the_spoken_words(assistant_env):
    """barge-in works on a typed question's answer as on a spoken one. The typed
    question stays in the history with no answer, and the spoken words become the next question."""
    never = asyncio.Event()
    fake = _FakeAssistant(never, _reply("Three o'clock."))
    session = await _assistant_session(fake)
    try:
        await session.handle_event(_typed("What's the capital of Peru?"))
        await session.handle_event({"type": "response.create"})
        await _until(lambda: fake.calls)
        await _say(session, "Actually, what time is it?")
    finally:
        never.set()
        await session.close()

    assert fake.cancelled == 1
    done = session.ws.of_type("response.done")
    assert [event["response"]["status"] for event in done] == ["cancelled", "completed"]
    assert done[0]["response"]["status_details"] == {"type": "cancelled", "reason": "turn_detected"}
    assert [message["content"] for message in fake.calls[1][0][1:]] == [
        "What's the capital of Peru?", "Actually, what time is it?",
    ]


@pytest.mark.asyncio
async def test_a_typed_question_during_a_reply_is_refused_and_one_after_response_cancel_is_answered(assistant_env):
    """While a reply runs, a typed item is refused (response_in_progress), so the
    history never puts it before the answer it interrupts. The Talk view's sequence (response.cancel, then
    conversation.item.create, then response.create, sent back to back) cancels the reply and answers the new
    question."""
    never = asyncio.Event()
    fake = _FakeAssistant(never, _reply("Lima."))
    session = await _assistant_session(fake)
    try:
        await session.handle_event(_typed("Tell me a long story."))
        await session.handle_event({"type": "response.create"})
        await _until(lambda: fake.calls)
        await session.handle_event(_typed("Stop. What's the capital of Peru?", event_id="evt_early"))
        refused = session.ws.of_type("error")
        await session.handle_event({"type": "response.cancel"})
        await session.handle_event(_typed("What's the capital of Peru?"))
        await session.handle_event({"type": "response.create"})
        await _finish(session)
    finally:
        never.set()
        await session.close()

    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in refused] == [
        ("invalid_request_error", "response_in_progress", "evt_early")
    ]
    assert session.ws.of_type("error") == refused
    assert fake.cancelled == 1
    done = session.ws.of_type("response.done")
    assert [event["response"]["status"] for event in done] == ["cancelled", "completed"]
    assert done[0]["response"]["status_details"] == {"type": "cancelled", "reason": "client_cancelled"}
    assert len(session.ws.of_type("conversation.item.added")) == 2
    assert session.conversation.for_request(20) == [
        {"role": "user", "content": "Tell me a long story."},
        {"role": "user", "content": "What's the capital of Peru?"},
        {"role": "assistant", "content": "Lima."},
    ]


# A /v1/realtime session logged nothing at INFO when it ended, so
# a client sending silence looked like one whose speech the VAD rejected. Its "ended" line says, as
# /v1/audio/stream's does, how much audio arrived, its peak level, the VAD's top score and how often speech
# started: numbers only, never audio or text.
def _realtime_ended_lines(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == "src.realtime.server" and record.levelno == logging.INFO
        and record.getMessage().startswith("Realtime session ")
    ]


def _pcm16_with(samples: int, values: dict[int, int]) -> bytes:
    """``samples`` of 16 kHz digital silence, with the given sample values put in."""
    pcm = np.zeros(samples, dtype=np.int16)
    for index, value in values.items():
        pcm[index] = value
    return pcm.tobytes()


@pytest.mark.parametrize(("update", "probs", "pcm", "levels", "speech_started"), [
    (None, [0.0] * 3 + [0.91] * 40, _pcm16_with(16000, {700: -16384, 900: 5000}),
     "audio=1.0 s, peak=-6.0 dBFS, vad_max=0.91, speech=1", 1),
    # The detector would score 0.99, but manual turns never run it.
    ({"turn_detection": None}, [0.99] * 40, _pcm16_with(8000, {10: 1000}),
     "audio=0.5 s, peak=-30.3 dBFS, vad_max=none, speech=0", 0),
], ids=["server-vad", "manual-turns"])
@pytest.mark.asyncio
async def test_realtime_session_end_logs_the_audio_it_received(caplog, update, probs, pcm, levels, speech_started):
    from starlette.websockets import WebSocket

    from src.realtime import server

    caplog.set_level(logging.INFO, logger="src.realtime.server")
    incoming = [{"type": "websocket.connect"}]
    if update is not None:
        incoming.append({"type": "websocket.receive", "text": json.dumps({"type": "session.update", "session": update})})
    half = len(pcm) // 2
    incoming += [
        {"type": "websocket.receive", "text": json.dumps(_append(pcm[:half]))},
        {"type": "websocket.receive", "text": json.dumps(_append(pcm[half:]))},
    ]
    sent = []

    async def receive():
        if incoming:
            return incoming.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket", "path": "/v1/realtime", "headers": [],
        "subprotocols": ["realtime"], "query_string": b"",
    }
    get_vad, silero = _scripted_vad(probs)
    with get_vad, silero, _identity_decode():
        await asyncio.wait_for(
            server.realtime_endpoint(WebSocket(scope, receive, send), tts_router=MagicMock()), timeout=5,
        )

    received = [json.loads(m["text"]) for m in sent if m["type"] == "websocket.send"]
    assert [event for event in received if event["type"] == "error"] == []
    assert sum(event["type"] == "input_audio_buffer.speech_started" for event in received) == speech_started
    session_id = received[0]["session"]["id"]
    assert _realtime_ended_lines(caplog) == [f"Realtime session {session_id} ended ({levels})"]


# ---------------------------------------------------------------------------
# With the AI assistant on, a client's own input_audio_buffer.commit started an
# AI reply by itself under server VAD, so the SDKs' documented flow (commit, then response.create with text)
# heard the AI and got response_in_progress. OpenAI's reference, input_audio_buffer.commit: "it will not
# create a response from the model"; turn_detection.create_response is about a VAD stop event. So
# server turn detection's own end of turn still replies, and a client's response.create after its
# commit answers it.
# ---------------------------------------------------------------------------


async def _commit_heard_as(session, transcript, *, event_id="evt_commit"):
    """500 ms of audio the VAD hears nothing in, then the client's own commit, heard as ``transcript``."""
    with _identity_decode(), patch("src.realtime.server.stt_router.transcribe", return_value={"text": transcript}):
        for _ in range(5):
            await session.handle_event(_append(b"\x01\x00" * 1600))
        await session.handle_event({"type": "input_audio_buffer.commit", "event_id": event_id})


@pytest.mark.parametrize("ga", [False, True], ids=["legacy-as-the-sdks", "ga"])
@pytest.mark.asyncio
async def test_a_clients_commit_starts_no_ai_reply_so_its_response_create_text_is_spoken(assistant_env, ga):
    """Under the default server VAD with create_response on, a client's commit is
    transcribed and its words join the conversation, but nothing is asked; the response.create with text that
    follows (the SDKs' flow) is spoken word for word, with no error."""
    fake = _FakeAssistant(_reply("The AI's own answer."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, ga=ga, turns=0)  # the VAD hears no turn
    assert session.config.turn_detection is not None and session.config.turn_detection.create_response is True
    try:
        await _commit_heard_as(session, "Repeat after me.")
        assert session._response_task is None and session.ws.of_type("response.created") == []
        await session.handle_event({"type": "response.create", "event_id": "evt_say",
                                    "response": {"instructions": "Say this back"}})
        await _finish(session)
    finally:
        await session.close()

    assert fake.calls == []
    assert [request["text"] for request in router.requests] == ["Say this back"]
    assert session.ws.of_type("error") == []
    assert len(session.ws.of_type("input_audio_buffer.committed")) == 1
    completed = session.ws.of_type("conversation.item.input_audio_transcription.completed")
    assert [event["transcript"] for event in completed] == ["Repeat after me."]
    assert session.conversation.for_request(20) == [{"role": "user", "content": "Repeat after me."}]
    assert [event["response"]["status"] for event in session.ws.of_type("response.done")] == ["completed"]


@pytest.mark.parametrize("ga", [False, True], ids=["legacy", "ga"])
@pytest.mark.asyncio
async def test_a_clients_commit_is_answered_by_the_ai_on_its_response_create_with_no_text(assistant_env, ga):
    """OpenAI's manual flow. The commit asks nothing; the client's response.create
    with no text asks the AI once with the committed words, and its reply is spoken."""
    from src.assistant.presets import VOICE_SYSTEM_PROMPT

    fake = _FakeAssistant(_reply("It's noon."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, ga=ga, turns=0)
    try:
        await _commit_heard_as(session, "What time is it?")
        assert session._response_task is None and session.ws.of_type("response.created") == []
        await session.handle_event({"type": "response.create", "event_id": "evt_answer"})
        await _finish(session)
    finally:
        await session.close()

    assert [messages for messages, _settings in fake.calls] == [[
        {"role": "system", "content": VOICE_SYSTEM_PROMPT}, {"role": "user", "content": "What time is it?"},
    ]]
    assert [request["text"] for request in router.requests] == ["It's noon."]
    assert [event["response"]["status"] for event in session.ws.of_type("response.done")] == ["completed"]
    assert session.ws.of_type("error") == []


@pytest.mark.asyncio
async def test_server_vads_end_of_turn_still_replies_and_a_later_client_commit_does_not(assistant_env):
    """A control: a turn server VAD ends is answered by itself (create_response),
    in the same session where a client's own commit then waits for its response.create."""
    fake = _FakeAssistant(_reply("Hello."), _reply("Unused."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, turns=1)
    try:
        await _say(session, "Hi there.")
        await _commit_heard_as(session, "Anything else?")
        await _finish(session)
    finally:
        await session.close()

    assert [messages[-1]["content"] for messages, _settings in fake.calls] == ["Hi there."]
    assert [request["text"] for request in router.requests] == ["Hello."]
    assert len(session.ws.of_type("response.created")) == 1
    assert session.conversation.for_request(20) == [
        {"role": "user", "content": "Hi there."}, {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Anything else?"},
    ]
    assert session.ws.of_type("error") == []


# ---------------------------------------------------------------------------
# A typed question with a lone UTF-16 surrogate (the JSON escape "\ud83d", half an
# emoji, as JavaScript's slice can leave) joined the conversation, and from then on every AI request of the
# session failed to encode (UTF-8 can't encode a surrogate code point, RFC 3629 section 3): tts_error, nothing
# spoken. It is refused with invalid_request_error, so it never joins the conversation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_typed_question_with_a_lone_surrogate_is_refused_and_the_next_one_is_answered(assistant_env):
    """With the real AI client over an httpx.MockTransport (no
    network), the half emoji is refused with the event's id and never joins the conversation, so the next
    question reaches the AI and its answer is spoken."""
    import httpx

    from src.assistant.client import AssistantClient

    requests = []

    def ai_server(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": "It is noon."}, "finish_reason": "stop"}]})

    client = AssistantClient("http://ai-server.local:8000/v1", transport=httpx.MockTransport(ai_server))
    router = _TextRecordingRouter()
    session = await _assistant_session(client, router=router)
    try:
        await session.handle_event(_typed("Show me a smiley \ud83d", event_id="evt_half_emoji"))
        refused = session.ws.of_type("error")
        await session.handle_event({"type": "response.create", "event_id": "evt_1"})
        await _finish(session)
        await session.handle_event(_typed("What time is it?", event_id="evt_time"))
        await session.handle_event({"type": "response.create", "event_id": "evt_2"})
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in refused] == [
        ("invalid_request_error", "invalid_request", "evt_half_emoji"),
    ]
    assert "surrogate" in refused[0]["error"]["message"]
    assert [event["item"]["content"][0]["text"] for event in session.ws.of_type("conversation.item.added")] == [
        "What time is it?",
    ]
    # The first response.create had nothing new to answer, as after any refused item; the second is answered.
    statuses = [event["response"]["status"] for event in session.ws.of_type("response.done")]
    assert statuses[-1] == "completed"
    assert all(e["error"]["code"] != "tts_error" for e in session.ws.of_type("error"))
    assert len(requests) == len(statuses)
    assert json.loads(requests[-1].content)["messages"][-1] == {"role": "user", "content": "What time is it?"}
    assert router.requests[-1]["text"] == "It is noon."
    assert session.conversation.for_request(20)[-2:] == [
        {"role": "user", "content": "What time is it?"}, {"role": "assistant", "content": "It is noon."},
    ]


@pytest.mark.parametrize("item", [
    {"content": [{"type": "input_text", "text": "Half \ud83d emoji"}]},
    {"content": [{"type": "input_text", "text": "Low half \ude00"}]},
    {"content": [{"type": "input_text", "text": "Reversed \ude00\ud83d pair"}]},
    {"content": [{"type": "input_text", "text": "Fine."}, {"type": "input_text", "text": "\ud83d"}]},
    {"id": "msg_\ud83d", "content": [{"type": "input_text", "text": "Fine."}]},
], ids=["high-half", "low-half", "reversed-pair", "second-part", "item-id"])
@pytest.mark.asyncio
async def test_a_typed_item_with_a_lone_surrogate_is_refused_and_nothing_joins(assistant_env, item):
    """A lone surrogate anywhere in the item's text, or in the item id it would echo
    back, gets invalid_request_error with the event's id; nothing is acknowledged, added or asked."""
    fake = _FakeAssistant(_reply("Unused."))
    session = await _assistant_session(fake)
    try:
        await session.handle_event({
            "type": "conversation.item.create", "event_id": "evt_bad",
            "item": {"type": "message", "role": "user", **item},
        })
    finally:
        await session.close()

    errors = session.ws.of_type("error")
    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in errors] == [
        ("invalid_request_error", "invalid_request", "evt_bad"),
    ]
    assert session.ws.of_type("conversation.item.added") == []
    assert session.conversation.for_request(20) == []
    assert fake.calls == []


@pytest.mark.asyncio
async def test_a_typed_question_with_a_whole_emoji_is_taken(assistant_env):
    """A control: a whole emoji (the JSON pair "\\ud83d\\ude00", decoded to one code
    point) is ordinary text and joins the conversation."""
    fake = _FakeAssistant(_reply("Hi!"))
    session = await _assistant_session(fake)
    try:
        await session.handle_event(json.loads('{"type": "conversation.item.create", "event_id": "evt_ok", "item": '
                                              '{"type": "message", "role": "user", "content": '
                                              '[{"type": "input_text", "text": "Smile \\ud83d\\ude00"}]}}'))
    finally:
        await session.close()

    assert session.ws.of_type("error") == []
    assert session.conversation.for_request(20) == [{"role": "user", "content": "Smile \U0001F600"}]


# ---------------------------------------------------------------------------
# A session.update whose instructions (OpenAI's name for the
# system prompt) had a lone UTF-16 surrogate was accepted, and every later AI request of the session failed
# to encode: tts_error, nothing spoken. README: "A value that isn't valid refuses the whole update (error
# with code invalid_request), and the session stays as it was." The HTTP fields are in test_assistant.py.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_instructions_with_a_lone_surrogate_are_refused_and_the_next_question_is_answered(assistant_env):
    """With the real AI client over an
    httpx.MockTransport (no network), the update is refused with its event's id and the session keeps its
    system prompt, so the next typed question reaches the AI and is answered aloud."""
    import httpx

    from src.assistant.client import AssistantClient
    from src.assistant.presets import VOICE_SYSTEM_PROMPT

    requests = []

    def ai_server(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": "It is noon."}, "finish_reason": "stop"}]})

    client = AssistantClient("http://ai-server.local:8000/v1", transport=httpx.MockTransport(ai_server))
    router = _TextRecordingRouter()
    session = await _assistant_session(client, router=router)
    updated_before = len(session.ws.of_type("session.updated"))
    try:
        await session.handle_event({"type": "session.update", "event_id": "evt_instr",
                                    "session": {"type": "realtime", "instructions": "Be brief \ud83d"}})
        refused = session.ws.of_type("error")
        await session.handle_event(_typed("What time is it?", event_id="evt_time"))
        await session.handle_event({"type": "response.create", "event_id": "evt_1"})
        await _finish(session)
    finally:
        await session.close()
        await client.aclose()

    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in refused] == [
        ("invalid_request_error", "invalid_request", "evt_instr"),
    ]
    assert "surrogate" in refused[0]["error"]["message"]
    assert len(session.ws.of_type("session.updated")) == updated_before
    assert session.config.instructions is None
    assert session.ws.of_type("error") == refused
    assert [event["response"]["status"] for event in session.ws.of_type("response.done")] == ["completed"]
    assert len(requests) == 1
    assert json.loads(requests[0].content)["messages"] == [
        {"role": "system", "content": VOICE_SYSTEM_PROMPT}, {"role": "user", "content": "What time is it?"},
    ]
    assert router.requests[-1]["text"] == "It is noon."


@pytest.mark.parametrize("assistant", [True, False], ids=["assistant", "no-assistant"])
@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.parametrize("instructions", ["Be brief \ud83d", "\ude00 first", "Keep \ude00\ud83d it"],
                         ids=["high-half", "low-half", "reversed-pair"])
@pytest.mark.asyncio
async def test_a_session_update_with_a_lone_surrogate_in_instructions_is_refused(assistant_env, assistant, ga, instructions):
    """Refused with invalid_request_error and the event's id, the session as
    it was (also its other fields in the same update), with or without the assistant: session.updated would echo
    the instructions, and they can't be encoded for the client either."""
    fake = _FakeAssistant(_reply("Unused.")) if assistant else None
    session = await _assistant_session(fake, ga=ga, update={"instructions": "Be kind."})
    before = session.config.to_dict()
    try:
        await session.handle_event({"type": "session.update", "event_id": "evt_bad",
                                    "session": {"instructions": instructions, "voice": "nova"}})
    finally:
        await session.close()

    errors = session.ws.of_type("error")
    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in errors] == [
        ("invalid_request_error", "invalid_request", "evt_bad"),
    ]
    assert session.config.to_dict() == before
    assert session.config.instructions == "Be kind."
    assert fake is None or fake.calls == []


@pytest.mark.asyncio
async def test_instructions_with_a_whole_emoji_are_taken_and_used(assistant_env):
    """A control: a whole emoji (the JSON pair "\\ud83d\\ude00") is
    ordinary text; the update is taken and the AI gets it as the system prompt."""
    fake = _FakeAssistant(_reply("Short."))
    session = await _assistant_session(fake)
    try:
        await session.handle_event(json.loads('{"type": "session.update", "event_id": "evt_ok", "session": '
                                              '{"instructions": "Be brief \\ud83d\\ude00"}}'))
        await session.handle_event(_typed("Hi?"))
        await session.handle_event({"type": "response.create"})
        await _finish(session)
    finally:
        await session.close()

    assert session.ws.of_type("error") == []
    assert session.ws.of_type("session.updated")[-1]["session"]["instructions"] == "Be brief \U0001F600"
    assert fake.calls[0][0][0] == {"role": "system", "content": "Be brief \U0001F600"}


# ---------------------------------------------------------------------------
# Several errors a client event caused didn't carry its event_id, though their
# siblings in the same handlers did. OpenAI's Realtime reference, server event error: error.event_id is "the
# event_id of the client event that caused the error, if applicable". It is echoed on every
# error a client event causes.
# ---------------------------------------------------------------------------


class _StubResampler:
    """Stands in for the session's InputAudioStream: what it holds back comes out at the commit."""

    def __init__(self, held_back: bytes):
        self.held_back = held_back

    def flush(self) -> bytes:
        held_back, self.held_back = self.held_back, b""
        return held_back


def _undecodable(*_args, **_kwargs):
    raise ValueError("The audio can't be decoded")


@pytest.mark.parametrize(("events", "code"), [
    ([{"type": "response.create", "response": {"output_modalities": ["text"]}}], "unsupported_modality"),
    ([{"type": "response.create", "response": {}}], "missing_input"),
    ([{"type": "input_audio_buffer.append", "audio": "not base64!"}], "invalid_audio"),
    ([{"type": "input_audio_buffer.append", "audio": 5}], "invalid_audio"),
    ([{"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01\x00" * 160).decode("ascii")}],
     "invalid_audio:undecodable"),
    ([_append(b"\x01\x00" * (1024 * 1024 // 2 + 1))], "buffer_overflow"),
    ([{"type": "session.update", "session": {"turn_detection": None}}, _append(b"\x01\x00" * 160)], "idle_timeout"),
], ids=["text-only", "no-text", "bad-base64", "audio-not-text", "undecodable", "over-the-buffer", "commit-idle-timeout"])
@pytest.mark.asyncio
async def test_each_refused_client_event_gets_its_event_id_back(monkeypatch, events, code):
    """A GA session with no assistant. The last event of each case is
    refused, and the error names it."""
    from src.realtime import server

    monkeypatch.setattr(server.settings, "os_realtime_max_buffer_mb", 1)
    code, _, variant = code.partition(":")
    get_vad, silero = _scripted_vad([])
    decode = patch("src.realtime.server.decode_audio_to_pcm16",
                   side_effect=_undecodable if variant else lambda data, _fmt, target_rate: data)
    with get_vad, silero, decode:
        session = server.RealtimeSession(_RecordingWebSocket(), _TextRecordingRouter(), ga=True)
        await session.initialize()
        try:
            for event in events[:-1]:
                await session.handle_event(event)
            if code == "idle_timeout":
                session._last_commit_at -= server.settings.os_realtime_idle_timeout_s + 1
            await session.handle_event({**events[-1], "event_id": "evt_refused"})
            await _finish(session)
        finally:
            await session.close()

    assert [(e["error"]["code"], e["error"].get("event_id")) for e in session.ws.of_type("error")] == [
        (code, "evt_refused"),
    ]


@pytest.mark.asyncio
async def test_an_overflow_while_a_commit_flushes_the_resampler_names_the_commit():
    """The audio the resampler still held back is added at the commit; when it overflows
    the buffer, the buffer_overflow is the commit's, as the commit_empty that follows already was."""
    from src.realtime.server import RealtimeSession

    session = RealtimeSession(_RecordingWebSocket(), _TextRecordingRouter(), ga=True)
    session.audio_buffer = InputAudioBuffer(max_buffer_bytes=64)
    session._input_audio = _StubResampler(b"\x01\x00" * 40)
    await session.handle_event({"type": "input_audio_buffer.commit", "event_id": "evt_commit"})

    assert [(e["error"]["code"], e["error"].get("event_id")) for e in session.ws.of_type("error")] == [
        ("buffer_overflow", "evt_commit"), ("input_audio_buffer_commit_empty", "evt_commit"),
    ]


@pytest.mark.asyncio
async def test_a_replys_failure_names_the_response_create_that_asked_for_it():
    """A reply the client's response.create started that then fails to synthesize (here
    the dictionary makes its text too long for the model) gets tts_error with that event's id, as the
    internal_error of the same request already did."""
    from src.realtime.server import RealtimeSession

    session = RealtimeSession(
        _RecordingWebSocket(), _LimitedRouter(),
        pronounce_fn=lambda text: text.replace("NASA", "National Aeronautics and Space Administration"),
    )
    await session.handle_event({"type": "response.create", "event_id": "evt_say", "response": {"instructions": "NASA"}})
    await _finish(session)

    assert [(e["error"]["code"], e["error"].get("event_id")) for e in session.ws.of_type("error")] == [
        ("tts_error", "evt_say"),
    ]
    assert session.ws.of_type("response.done")[0]["response"]["status"] == "failed"


@pytest.mark.asyncio
async def test_an_ai_error_names_the_clients_response_create_and_a_vad_turns_has_none(assistant_env):
    """The AI's error (server_error, an assistant_* code) for a reply the client asked for
    with response.create names that event; a reply that server VAD started was asked for by no client event."""
    from src.assistant import AssistantError

    fake = _FakeAssistant(
        AssistantError("No whole reply within 120 s", "assistant_timeout"),
        AssistantError("The AI server can't be reached", "assistant_unreachable"),
    )
    session = await _assistant_session(fake)
    try:
        await session.handle_event(_typed("Hello?"))
        await session.handle_event({"type": "response.create", "event_id": "evt_answer"})
        await _finish(session)
        await _say(session, "Are you there?")
    finally:
        await session.close()

    errors = [(e["error"]["type"], e["error"]["code"], e["error"].get("event_id")) for e in session.ws.of_type("error")]
    assert errors == [("server_error", "assistant_timeout", "evt_answer"), ("server_error", "assistant_unreachable", None)]
    assert [event["response"]["status"] for event in session.ws.of_type("response.done")] == ["failed", "failed"]


@pytest.mark.asyncio
async def test_an_event_without_a_type_gets_its_event_id_back():
    """An event object with no type is refused with invalid_event, naming the event_id
    it carries."""
    from starlette.websockets import WebSocket

    from src.realtime import server

    incoming = [
        {"type": "websocket.connect"},
        {"type": "websocket.receive", "text": json.dumps({"event_id": "evt_untyped", "audio": ""})},
    ]
    sent = []

    async def receive():
        if incoming:
            return incoming.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket", "path": "/v1/realtime", "headers": [],
        "subprotocols": ["realtime"], "query_string": b"",
    }
    get_vad, silero = _scripted_vad([])
    with get_vad, silero:
        await asyncio.wait_for(
            server.realtime_endpoint(WebSocket(scope, receive, send), tts_router=MagicMock()), timeout=5,
        )

    received = [json.loads(m["text"]) for m in sent if m["type"] == "websocket.send"]
    errors = [event["error"] for event in received if event["type"] == "error"]
    assert [(error["code"], error.get("event_id")) for error in errors] == [("invalid_event", "evt_untyped")]


# ---------------------------------------------------------------------------
# The rest of the client's text that /v1/realtime sends back refuses a lone UTF-16
# surrogate too, as the instructions and typed items do. A session.update's voice, model
# or transcription settings (their prompt) with one was taken, and session.updated, which echoes them, couldn't
# be encoded for the client; a word-for-word response.create's text was spoken, and its transcript and
# response.done couldn't be encoded either. Either way the client heard of nothing. Each is now
# refused with invalid_request_error and the client's event_id, the session as it was.
# ---------------------------------------------------------------------------


class _Utf8WebSocket(_RecordingWebSocket):
    """Sends as the real endpoint does: starlette's send_json (0.50) dumps with ensure_ascii=False, and the text
    frame is UTF-8, which can't encode a surrogate code point (RFC 3629, section 3). An event that can't be sent
    is kept in ``unsendable``, and the error is raised, as the real socket raises it (``_send`` swallows it)."""

    def __init__(self):
        super().__init__()
        self.unsendable = []

    async def send_json(self, event):
        try:
            json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            self.unsendable.append(event)
            raise
        await super().send_json(event)


def _surrogate_update(field, ga, text):
    """A session.update payload that sets ``field`` to ``text``, in the GA or the legacy shape."""
    if field == "voice":
        return {"audio": {"output": {"voice": text}}} if ga else {"voice": text}
    if field == "model":
        return {"model": text}
    transcription = {"model": "whisper-1", "prompt": text} if field == "transcription-prompt" else {"model": text}
    return {"audio": {"input": {"transcription": transcription}}} if ga else {"input_audio_transcription": transcription}


@pytest.mark.parametrize("half", ["\ud83d", "\ude00"], ids=["high-half", "low-half"])
@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.parametrize("field", ["voice", "model", "transcription-prompt", "transcription-model"])
@pytest.mark.asyncio
async def test_a_session_update_with_a_lone_surrogate_in_its_voice_model_or_transcription_is_refused(
    assistant_env, field, ga, half,
):
    """Refused with invalid_request_error and the event's id, and the whole update
    with it (its instructions too), so the session is as it was: the next word-for-word reply is spoken in its
    voice, and every event reaches the client. The transcription's model is checked as well as its prompt:
    session.updated echoes the whole transcription object, so any text in it fails the same way."""
    fake = _FakeAssistant(_reply("Unused."))
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, ga=ga, update={"voice": "nova", "instructions": "Be kind."})
    before = session.config.to_dict()
    session.ws = _Utf8WebSocket()  # from here on, every event is encoded as the real endpoint sends it
    try:
        await session.handle_event({"type": "session.update", "event_id": "evt_bad", "session": {
            **_surrogate_update(field, ga, f"name{half}"), "instructions": "Changed."}})
        refused = session.ws.of_type("error")
        await session.handle_event({"type": "response.create", "event_id": "evt_say",
                                    "response": {"instructions": "Hello."}})
        await _finish(session)
    finally:
        await session.close()

    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in refused] == [
        ("invalid_request_error", "invalid_request", "evt_bad"),
    ]
    assert "surrogate" in refused[0]["error"]["message"]
    assert session.ws.of_type("session.updated") == []
    assert session.config.to_dict() == before
    assert session.ws.unsendable == []
    assert session.ws.of_type("error") == refused
    assert [event["response"]["status"] for event in session.ws.of_type("response.done")] == ["completed"]
    assert [(request["text"], request["voice"]) for request in router.requests] == [("Hello.", "nova")]
    assert fake.calls == []


@pytest.mark.parametrize("assistant", [True, False], ids=["assistant", "no-assistant"])
@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.parametrize("where", ["instructions", "input"])
@pytest.mark.asyncio
async def test_a_word_for_word_response_with_a_lone_surrogate_is_refused_before_it_starts(
    assistant_env, where, ga, assistant,
):
    """A response.create whose text (its instructions, or its input's text) has a lone
    surrogate gets invalid_request_error with the event's id before any response starts: nothing is spoken or
    asked, and the next response.create is answered as before."""
    fake = _FakeAssistant(_reply("Unused.")) if assistant else None
    router = _TextRecordingRouter()
    session = await _assistant_session(fake, router=router, ga=ga)
    session.ws = _Utf8WebSocket()  # from here on, every event is encoded as the real endpoint sends it
    text = "Smile \ud83d please"
    if where == "instructions":
        response = {"instructions": text}
    else:
        response = {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]}
    try:
        await session.handle_event({"type": "response.create", "event_id": "evt_half", "response": response})
        await _finish(session)
        refused = session.ws.of_type("error")
        refused_events = list(session.ws.sent)
        await session.handle_event({"type": "response.create", "event_id": "evt_say",
                                    "response": {"instructions": "Hello."}})
        await _finish(session)
    finally:
        await session.close()

    assert [(e["error"]["type"], e["error"]["code"], e["error"]["event_id"]) for e in refused] == [
        ("invalid_request_error", "invalid_request", "evt_half"),
    ]
    assert "surrogate" in refused[0]["error"]["message"]
    assert [event["type"] for event in refused_events] == ["error"]  # no response.created, no response.done
    assert session.ws.unsendable == []
    assert session.ws.of_type("error") == refused
    assert [event["response"]["status"] for event in session.ws.of_type("response.done")] == ["completed"]
    assert [request["text"] for request in router.requests] == ["Hello."]
    assert fake is None or fake.calls == []


@pytest.mark.parametrize("ga", [True, False], ids=["ga", "legacy"])
@pytest.mark.asyncio
async def test_a_whole_emoji_in_the_voice_the_transcription_prompt_and_a_spoken_text_is_taken(assistant_env, ga):
    """A control: a whole emoji (the JSON pair "\\ud83d\\ude00", decoded to one code
    point) is ordinary text. The update is taken and echoed, and the text is spoken, its transcript sent."""
    router = _TextRecordingRouter()
    session = await _assistant_session(None, router=router, ga=ga)
    session.ws = _Utf8WebSocket()
    smile = json.loads('"\\ud83d\\ude00"')
    voice, transcription = f"nova{smile}", {"model": "whisper-1", "prompt": f"Zo{smile}"}
    if ga:
        update = {"audio": {"output": {"voice": voice}, "input": {"transcription": transcription}}}
    else:
        update = {"voice": voice, "input_audio_transcription": transcription}
    try:
        await session.handle_event({"type": "session.update", "event_id": "evt_ok", "session": update})
        await session.handle_event({"type": "response.create", "event_id": "evt_say",
                                    "response": {"instructions": f"Smile {smile}"}})
        await _finish(session)
    finally:
        await session.close()

    assert session.ws.unsendable == []
    assert session.ws.of_type("error") == []
    assert session.config.voice == "nova\U0001F600"
    assert session.config.input_audio_transcription["prompt"] == "Zo\U0001F600"
    assert len(session.ws.of_type("session.updated")) == 1
    assert [(request["text"], request["voice"]) for request in router.requests] == [("Smile \U0001F600", "nova\U0001F600")]
    done = session.ws.of_type("response.done")
    assert [event["response"]["status"] for event in done] == ["completed"]
    assert done[0]["response"]["output"][0]["content"][0]["transcript"] == "Smile \U0001F600"


class _UnansweredCloseWebSocket:
    """An accepted socket whose close() waits until the test lets it end, as the close handshake with a client that
    doesn't answer it does (uvicorn's websockets protocol waits up to 10 s for the close frame, then for the TCP)."""

    def __init__(self):
        self.scope = {"subprotocols": []}
        self.headers = {}
        self.closing = asyncio.Event()
        self.answered = asyncio.Event()
        self.closed_with = None

    async def accept(self, subprotocol=None):
        return None

    async def close(self, code=1000, reason=""):
        self.closing.set()
        await self.answered.wait()
        self.closed_with = (code, reason)


@pytest.mark.asyncio
async def test_a_refused_realtime_sessions_close_handshake_does_not_hold_the_session_lock(monkeypatch):
    """Over OS_STREAM_MAX_CONNECTIONS a new session is closed with 1013, but the
    close runs after the session-count lock is released. Before, it ran inside it, so a client that didn't answer the
    close kept every session from starting or giving its slot back for up to about 20 s."""
    from src.realtime import server

    monkeypatch.setattr(server.settings, "os_stream_max_connections", 0)
    ws = _UnansweredCloseWebSocket()
    refusal = asyncio.create_task(server.realtime_endpoint(ws, tts_router=MagicMock()))
    try:
        await asyncio.wait_for(ws.closing.wait(), 5)
        held_while_closing = server._active_realtime_lock.locked()
    finally:
        ws.answered.set()
        await asyncio.wait_for(refusal, 5)

    assert not held_while_closing, "the session-count lock was held through the refused socket's close handshake"
    assert ws.closed_with == (1013, "Too many concurrent realtime sessions")
    assert len(server._active_realtime_sessions) == 0
