"""Tests for the OpenAI Realtime API compatibility layer."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.realtime.events import (
    session_created,
    session_updated,
    error,
    input_audio_buffer_speech_started,
    input_audio_buffer_speech_stopped,
    input_audio_buffer_committed,
    conversation_item_created,
    conversation_item_input_audio_transcription_completed,
    response_created,
    response_audio_delta,
    response_audio_done,
    response_done,
    _event_id,
    _item_id,
    _response_id,
)
from src.realtime.session import SessionConfig, TurnDetectionConfig, VALID_AUDIO_FORMATS
from src.realtime.audio_buffer import (
    InputAudioBuffer,
    decode_audio_to_pcm16,
    encode_pcm16_to_format,
)


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
        import audioop
        pcm = _make_pcm16_tone(100, sample_rate=8000)
        ulaw = audioop.lin2ulaw(pcm, 2)
        result = decode_audio_to_pcm16(ulaw, "g711_ulaw", target_rate=16000)
        assert len(result) > 0
        # Should be resampled from 8kHz to 16kHz (~double)
        assert len(result) > len(ulaw)

    def test_g711_alaw_decode(self):
        import audioop
        pcm = _make_pcm16_tone(100, sample_rate=8000)
        alaw = audioop.lin2alaw(pcm, 2)
        result = decode_audio_to_pcm16(alaw, "g711_alaw", target_rate=16000)
        assert len(result) > 0

    def test_encode_pcm16_to_ulaw(self):
        import audioop
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

                with patch.object(
                    __import__("src.realtime.server", fromlist=["RealtimeSession"]).RealtimeSession,
                    "_handle_response_create",
                ) as mock_handler:
                    # Simpler: just test that the TTS path works with a mock
                    pass

        # Test with full mock of tts_router
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

    def test_disabled_closes_connection(self):
        import os
        os.environ["OS_REALTIME_ENABLED"] = "false"

        # Need to reload settings
        with patch("src.main.settings") as mock_settings:
            mock_settings.os_realtime_enabled = False
            mock_settings.os_api_key = ""
            mock_settings.stt_api_key = ""

            from fastapi.testclient import TestClient
            from src.main import app

            client = TestClient(app)
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/realtime") as ws:
                    pass

        os.environ.pop("OS_REALTIME_ENABLED", None)
