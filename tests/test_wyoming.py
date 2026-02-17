"""Tests for Wyoming Protocol integration."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_stt_router():
    router = MagicMock()
    router.loaded_models.return_value = []
    router.transcribe.return_value = {"text": "hello world"}
    return router


@pytest.fixture
def mock_tts_router():
    router = MagicMock()
    router.loaded_models.return_value = []
    router.list_voices.return_value = []
    router.synthesize.return_value = iter([np.zeros(2400, dtype=np.float32)])
    return router


def _make_mock_settings(**overrides):
    s = MagicMock()
    s.stt_model = overrides.get("stt_model", "test-stt-model")
    s.tts_enabled = overrides.get("tts_enabled", True)
    s.tts_voice = overrides.get("tts_voice", "af_heart")
    s.tts_model = overrides.get("tts_model", "kokoro")
    return s


# ---------------------------------------------------------------------------
# build_info / Describe
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_build_info_has_asr(self, mock_stt_router, mock_tts_router):
        from src.wyoming import server
        orig = server.settings
        server.settings = _make_mock_settings()
        try:
            info = server.build_info(mock_stt_router, mock_tts_router)
            assert len(info.asr) == 1
            assert info.asr[0].name == "open-speech"
            assert len(info.asr[0].models) >= 1
        finally:
            server.settings = orig

    def test_build_info_has_tts(self, mock_stt_router, mock_tts_router):
        from src.wyoming import server
        orig = server.settings
        server.settings = _make_mock_settings()
        try:
            info = server.build_info(mock_stt_router, mock_tts_router)
            assert len(info.tts) == 1
            assert info.tts[0].name == "open-speech"
        finally:
            server.settings = orig

    def test_build_info_no_tts_when_disabled(self, mock_stt_router, mock_tts_router):
        from src.wyoming import server
        orig = server.settings
        server.settings = _make_mock_settings(tts_enabled=False)
        try:
            info = server.build_info(mock_stt_router, mock_tts_router)
            assert len(info.tts) == 0
        finally:
            server.settings = orig

    def test_build_info_default_model_included(self, mock_stt_router, mock_tts_router):
        from src.wyoming import server
        orig = server.settings
        server.settings = _make_mock_settings(stt_model="my-stt-model", tts_enabled=False)
        try:
            info = server.build_info(mock_stt_router, mock_tts_router)
            model_names = [m.name for m in info.asr[0].models]
            assert "my-stt-model" in model_names
        finally:
            server.settings = orig


# ---------------------------------------------------------------------------
# STT handler
# ---------------------------------------------------------------------------


class TestSTTHandler:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self, mock_stt_router):
        from src.wyoming.stt_handler import handle_transcribe

        pcm = b"\x00\x00" * 1600
        text = await handle_transcribe(
            audio_chunks=[pcm], rate=16000, width=2, channels=1,
            stt_router=mock_stt_router,
        )
        assert text == "hello world"
        mock_stt_router.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self, mock_stt_router):
        from src.wyoming.stt_handler import handle_transcribe

        text = await handle_transcribe(
            audio_chunks=[], rate=16000, width=2, channels=1,
            stt_router=mock_stt_router,
        )
        assert text == ""
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_transcribe_wav_header(self, mock_stt_router):
        from src.wyoming.stt_handler import handle_transcribe

        pcm = b"\x00\x00" * 1600
        await handle_transcribe(
            audio_chunks=[pcm], rate=16000, width=2, channels=1,
            stt_router=mock_stt_router,
        )
        wav_bytes = mock_stt_router.transcribe.call_args[1]["audio"]
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"


# ---------------------------------------------------------------------------
# TTS handler
# ---------------------------------------------------------------------------


class TestTTSHandler:
    @pytest.mark.asyncio
    async def test_synthesize_sends_audio_events(self, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings()
        try:
            events: list[Event] = []

            async def capture(event):
                events.append(event)

            await tts_handler.handle_synthesize(
                text="hello", voice=None, tts_router=mock_tts_router,
                write_event=capture,
            )
            types = [e.type for e in events]
            assert "audio-start" in types
            assert "audio-chunk" in types
            assert "audio-stop" in types
        finally:
            tts_handler.settings = orig

    @pytest.mark.asyncio
    async def test_synthesize_disabled(self, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings(tts_enabled=False)
        try:
            events: list[Event] = []
            await tts_handler.handle_synthesize(
                text="hello", voice=None, tts_router=mock_tts_router,
                write_event=lambda e: events.append(e),
            )
            assert len(events) == 0
        finally:
            tts_handler.settings = orig

    @pytest.mark.asyncio
    async def test_synthesize_uses_voice(self, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings()
        try:
            await tts_handler.handle_synthesize(
                text="hello", voice="custom_voice", tts_router=mock_tts_router,
                write_event=AsyncMock(),
            )
            call_kwargs = mock_tts_router.synthesize.call_args[1]
            assert call_kwargs["voice"] == "custom_voice"
        finally:
            tts_handler.settings = orig

    @pytest.mark.asyncio
    async def test_synthesize_audio_is_16khz(self, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings()
        try:
            events: list[Event] = []

            async def capture(event):
                events.append(event)

            await tts_handler.handle_synthesize(
                text="hello", voice=None, tts_router=mock_tts_router,
                write_event=capture,
            )
            for e in events:
                if AudioStart.is_type(e.type):
                    start = AudioStart.from_event(e)
                    assert start.rate == 16000
                    assert start.width == 2
                    assert start.channels == 1
        finally:
            tts_handler.settings = orig


# ---------------------------------------------------------------------------
# Event handler integration
# ---------------------------------------------------------------------------


class TestEventHandler:
    def _make_handler(self, mock_stt_router, mock_tts_router, **settings_kw):
        from src.wyoming import server
        orig = server.settings
        server.settings = _make_mock_settings(**settings_kw)
        info = server.build_info(mock_stt_router, mock_tts_router)
        server.settings = orig

        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = MagicMock(spec=asyncio.StreamWriter)
        handler = server.OpenSpeechEventHandler(
            reader, writer, mock_stt_router, mock_tts_router, info
        )
        handler.write_event = AsyncMock()
        return handler

    @pytest.mark.asyncio
    async def test_describe_event(self, mock_stt_router, mock_tts_router):
        handler = self._make_handler(mock_stt_router, mock_tts_router)
        result = await handler.handle_event(Describe().event())
        assert result is True
        handler.write_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_flow(self, mock_stt_router, mock_tts_router):
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        await handler.handle_event(Transcribe().event())
        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 1600)
        await handler.handle_event(chunk.event())
        await handler.handle_event(AudioStop().event())

        handler.write_event.assert_called()
        event = handler.write_event.call_args[0][0]
        assert Transcript.is_type(event.type)

    @pytest.mark.asyncio
    async def test_synthesize_flow(self, mock_stt_router, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings()
        try:
            handler = self._make_handler(mock_stt_router, mock_tts_router)
            synth = Synthesize(text="hello world", voice=SynthesizeVoice(name="af_heart"))
            await handler.handle_event(synth.event())

            calls = handler.write_event.call_args_list
            types = [c[0][0].type for c in calls]
            assert "audio-start" in types
            assert "audio-chunk" in types
            assert "audio-stop" in types
        finally:
            tts_handler.settings = orig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_wyoming_disabled_by_default(self):
        from src.config import Settings
        s = Settings()
        assert s.os_wyoming_enabled is False

    def test_wyoming_default_port(self):
        from src.config import Settings
        s = Settings()
        assert s.os_wyoming_port == 10400

    def test_wyoming_enabled_via_env(self, monkeypatch):
        from src.config import Settings
        monkeypatch.setenv("OS_WYOMING_ENABLED", "true")
        s = Settings()
        assert s.os_wyoming_enabled is True

    def test_wyoming_port_via_env(self, monkeypatch):
        from src.config import Settings
        monkeypatch.setenv("OS_WYOMING_PORT", "10500")
        s = Settings()
        assert s.os_wyoming_port == 10500


# ---------------------------------------------------------------------------
# Resample utility
# ---------------------------------------------------------------------------


class TestResample:
    def test_resample_24k_to_16k(self):
        from src.wyoming.tts_handler import _resample_to_16k
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 24000)).astype(np.float32)
        resampled = _resample_to_16k(audio, 24000)
        assert len(resampled) == 16000

    def test_resample_noop_at_16k(self):
        from src.wyoming.tts_handler import _resample_to_16k
        audio = np.zeros(16000, dtype=np.float32)
        resampled = _resample_to_16k(audio, 16000)
        assert len(resampled) == 16000
        assert resampled is audio


# ---------------------------------------------------------------------------
# PCM to WAV
# ---------------------------------------------------------------------------


class TestPcmToWav:
    def test_wav_header(self):
        from src.wyoming.stt_handler import _pcm_to_wav
        pcm = b"\x00\x00" * 100
        wav = _pcm_to_wav(pcm, rate=16000, width=2, channels=1)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        data_size = struct.unpack("<I", wav[40:44])[0]
        assert data_size == 200
