"""Tests for Wyoming Protocol integration."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import os
import struct
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe
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
    router.sample_rate_for.return_value = 24000
    return router


@pytest.fixture(autouse=True)
def _no_silero_load(request, monkeypatch):
    """Wyoming loads Silero for STT_VAD_ENABLED; no test reads the real model file or downloads it.

    A test that set silero._vad_model gets it; otherwise loading fails and the audio stays as it
    is. Tests using cached_silero_not_loaded keep the real loader, with a fake model file.
    """
    if "cached_silero_not_loaded" in request.fixturenames:
        return
    import src.vad.silero as silero
    from src.wyoming import stt_handler

    async def loaded_model_or_none():
        if silero._vad_model is None:
            raise RuntimeError("no Silero VAD model in this test")
        return silero._vad_model

    monkeypatch.setattr(stt_handler, "get_vad_model", loaded_model_or_none)


def _make_mock_settings(**overrides):
    s = MagicMock()
    s.stt_model = overrides.get("stt_model", "test-stt-model")
    s.tts_enabled = overrides.get("tts_enabled", True)
    s.tts_voice = overrides.get("tts_voice", "af_heart")
    s.tts_model = overrides.get("tts_model", "kokoro")
    s.tts_cache_enabled = overrides.get("tts_cache_enabled", False)
    s.tts_trim_silence = overrides.get("tts_trim_silence", False)
    s.tts_normalize_output = overrides.get("tts_normalize_output", False)
    s.tts_max_input_length = overrides.get("tts_max_input_length", 4096)
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

    def test_build_info_advertises_each_stt_model_languages(self, mock_stt_router, mock_tts_router, monkeypatch):
        # Home Assistant offers a Wyoming STT engine only to pipelines in the languages its
        # Info lists; README lists the faster-whisper models as 99+ languages.
        from src.wyoming import server

        monkeypatch.setattr(server, "settings", _make_mock_settings(
            stt_model="deepdml/faster-whisper-large-v3-turbo-ct2", tts_enabled=False,
        ))
        mock_stt_router.loaded_models.return_value = [
            MagicMock(model="Systran/faster-whisper-base.en", backend="faster-whisper"),
        ]

        info = server.build_info(mock_stt_router, mock_tts_router)

        languages = {m.name: m.languages for m in info.asr[0].models}
        assert {"en", "de", "fr", "es", "zh"} <= set(languages["deepdml/faster-whisper-large-v3-turbo-ct2"])
        assert languages["Systran/faster-whisper-base.en"] == ["en"]

    @pytest.mark.parametrize("loaded", [False, True], ids=["default", "loaded"])
    def test_build_info_advertises_english_alone_for_parakeet(self, mock_stt_router, mock_tts_router, monkeypatch,
                                                               loaded):
        # Parakeet transcribes English only: Home Assistant must not offer it to a pipeline in another language.
        from src.wyoming import server

        parakeet = "nvidia/parakeet-tdt-0.6b-v2"
        monkeypatch.setattr(server, "settings", _make_mock_settings(stt_model=parakeet, tts_enabled=False))
        mock_stt_router.loaded_models.return_value = (
            [MagicMock(model=parakeet, backend="parakeet")] if loaded else []
        )

        info = server.build_info(mock_stt_router, mock_tts_router)

        assert {m.name: m.languages for m in info.asr[0].models} == {parakeet: ["en"]}


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
    async def test_transcribe_decodes_once_without_temperature_fallback(self, mock_stt_router):
        # A voice command waits for its decode, so it runs once at temperature 0
        # instead of up to six times with faster-whisper's hotter retries.
        from src.wyoming.stt_handler import handle_transcribe

        await handle_transcribe(
            audio_chunks=[b"\x00\x00" * 1600], rate=16000, width=2, channels=1,
            stt_router=mock_stt_router,
        )

        assert mock_stt_router.transcribe.call_args.kwargs["temperature_fallback"] is False

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

    @pytest.mark.asyncio
    async def test_audio_preparation_runs_off_event_loop(self, mock_stt_router, monkeypatch):
        # Wyoming shares the app's event loop. Like HTTP uploads, the Silero VAD pass and
        # STT_NORMALIZE / STT_NOISE_REDUCE preprocessing must not stall other clients.
        from src.wyoming import stt_handler

        loop_thread = threading.get_ident()
        on_loop: dict[str, bool] = {}

        def vad(pcm, rate, width, channels):
            on_loop["vad"] = threading.get_ident() == loop_thread
            return pcm

        def preprocess(wav_bytes, *, noise_reduce, normalize):
            on_loop["preprocess"] = threading.get_ident() == loop_thread
            return wav_bytes

        monkeypatch.setattr(stt_handler, "_extract_speech_segments", vad)
        monkeypatch.setattr(stt_handler, "preprocess_stt_audio", preprocess)

        text = await asyncio.wait_for(stt_handler.handle_transcribe(
            audio_chunks=[b"\x10\x00" * 1600], rate=16000, width=2, channels=1,
            stt_router=mock_stt_router,
        ), timeout=10)

        assert text == "hello world"
        assert on_loop == {"vad": False, "preprocess": False}

    @pytest.fixture
    def cached_silero_not_loaded(self, monkeypatch, tmp_path):
        """Silero is cached on disk (the vad-cache volume), but no session has loaded it yet."""
        import hashlib
        import urllib.request

        import onnxruntime

        import src.vad.silero as silero
        from src.wyoming import stt_handler

        class EnergySession:
            """Fake Silero ONNX session: speech wherever the 512-sample window has energy."""

            def run(self, _output_names, inputs):
                window = inputs["input"][0, silero.VAD_CONTEXT_SIZE:]
                prob = 0.9 if float(np.sqrt(np.mean(window ** 2))) > 0.05 else 0.0
                return [np.array([[prob]], dtype=np.float32), inputs["state"]]

        def no_network(*_args, **_kwargs):
            raise AssertionError("tests must not download")

        content = b"onnx" * 1000
        monkeypatch.setattr(silero, "SILERO_CACHE_DIR", tmp_path)
        monkeypatch.setattr(silero, "SILERO_ONNX_BYTES", len(content))
        monkeypatch.setattr(silero, "SILERO_ONNX_SHA256", hashlib.sha256(content).hexdigest())
        (tmp_path / silero.SILERO_ONNX_FILENAME).write_bytes(content)
        monkeypatch.setattr(silero, "_vad_model", None)
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: EnergySession())
        monkeypatch.setattr(urllib.request, "urlopen", no_network)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_enabled", True)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_threshold", 0.5)
        monkeypatch.setattr(stt_handler.settings, "stt_normalize", False)
        monkeypatch.setattr(stt_handler.settings, "stt_noise_reduce", False)
        return silero

    @staticmethod
    async def _seconds_sent_to_stt(pcm: bytes) -> float:
        import io
        import wave

        from src.wyoming import stt_handler

        router = MagicMock()
        router.transcribe.return_value = {"text": "hello"}
        await asyncio.wait_for(stt_handler.handle_transcribe(
            audio_chunks=[pcm], rate=16000, width=2, channels=1, stt_router=router,
        ), timeout=30)
        with wave.open(io.BytesIO(router.transcribe.call_args.kwargs["audio"]), "rb") as wav:
            return wav.getnframes() / wav.getframerate()

    @staticmethod
    def _silence_then_speech() -> bytes:
        t = np.arange(16000) / 16000
        speech = (0.5 * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
        return np.concatenate([np.zeros(2 * 16000, dtype=np.int16), speech]).tobytes()

    @pytest.mark.asyncio
    async def test_vad_trims_wyoming_audio_when_no_other_session_loaded_silero(self, cached_silero_not_loaded):
        # A Wyoming-only deployment (Home Assistant) with STT_VAD_ENABLED=true: no streaming or
        # realtime session ever loads Silero, yet the 2 s of leading silence is still dropped.
        seconds = await self._seconds_sent_to_stt(self._silence_then_speech())

        assert seconds < 2.0, f"{seconds:.2f} s of audio (2 s silence + 1 s speech) reached the STT model"

    @pytest.mark.asyncio
    async def test_vad_trimming_does_not_depend_on_an_earlier_streaming_session(self, cached_silero_not_loaded):
        pcm = self._silence_then_speech()

        before = await self._seconds_sent_to_stt(pcm)
        # /v1/audio/stream or /v1/realtime opens once; both load Silero.
        await asyncio.wait_for(cached_silero_not_loaded.get_vad_model(), timeout=10)
        after = await self._seconds_sent_to_stt(pcm)

        assert before == pytest.approx(after, abs=0.05)

    @pytest.mark.asyncio
    async def test_unavailable_vad_leaves_the_audio_as_it_is(self, mock_stt_router, monkeypatch):
        # The autouse fixture's loader fails (no model): the command is still transcribed, whole.
        from src.wyoming import stt_handler

        monkeypatch.setattr(stt_handler.settings, "stt_vad_enabled", True)
        pcm = self._silence_then_speech()

        text = await asyncio.wait_for(stt_handler.handle_transcribe(
            audio_chunks=[pcm], rate=16000, width=2, channels=1, stt_router=mock_stt_router,
        ), timeout=10)

        assert text == "hello world"
        assert len(mock_stt_router.transcribe.call_args.kwargs["audio"]) == 44 + len(pcm)

    @pytest.mark.asyncio
    async def test_transcribe_failure_is_raised_not_masked(self, mock_stt_router):
        # An empty transcript reads as "nothing recognized"; the caller reports the failure.
        from src.wyoming.stt_handler import handle_transcribe

        mock_stt_router.transcribe.side_effect = RuntimeError("model failed to load")

        with pytest.raises(RuntimeError, match="model failed to load"):
            await handle_transcribe(
                audio_chunks=[b"\x00\x00" * 1600], rate=16000, width=2, channels=1,
                stt_router=mock_stt_router,
            )

    @pytest.mark.parametrize("rate, width, channels", [
        (1, 2, 1), (7999, 2, 1), (192001, 2, 1), ("16000", 2, 1), (16000, 0, 1), (16000, 5, 1), (16000, 2, 0),
    ])
    def test_audio_format_error_rejects_unusable_formats(self, rate, width, channels):
        from src.wyoming.stt_handler import audio_format_error

        assert audio_format_error(rate, width, channels)

    @pytest.mark.parametrize("rate, width, channels", [(8000, 2, 1), (16000, 2, 1), (192000, 4, 2), (44100, 1, 1)])
    def test_audio_format_error_accepts_pcm_speech_formats(self, rate, width, channels):
        from src.wyoming.stt_handler import audio_format_error

        assert audio_format_error(rate, width, channels) is None

    @pytest.mark.parametrize("rate", [16000, 22050, 44100, 11025])
    def test_vad_trimming_keeps_all_the_detected_speech(self, monkeypatch, rate):
        # Segment times were mapped back at rate // 1000 samples per ms (22 at 22.05 kHz),
        # so cuts drifted early and the end of the utterance never reached Whisper.
        import src.vad.silero as silero
        from src.wyoming import stt_handler

        class EnergySession:
            """Fake Silero ONNX session: speech wherever the 512-sample window has energy."""

            def run(self, _output_names, inputs):
                window = inputs["input"][0, silero.VAD_CONTEXT_SIZE:]
                prob = 0.9 if float(np.sqrt(np.mean(window ** 2))) > 0.05 else 0.0
                return [np.array([[prob]], dtype=np.float32), inputs["state"]]

        monkeypatch.setattr(stt_handler.settings, "stt_vad_enabled", True)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_threshold", 0.5)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_min_speech_ms", 250)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_silence_ms", 800)
        monkeypatch.setattr(silero, "_vad_model", silero.SileroVAD(EnergySession()))
        # One second of silence, then speech (no zero samples) to the end of the utterance
        t = np.arange(4 * rate) / rate
        speech = (6000 + 3000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
        audio = np.concatenate([np.zeros(rate, dtype=np.int16), speech])

        out = np.frombuffer(
            stt_handler._extract_speech_segments(audio.tobytes(), rate=rate, width=2, channels=1),
            dtype=np.int16,
        )

        assert len(out) < len(audio)  # the leading silence was trimmed
        kept = int(np.count_nonzero(out))
        assert kept == len(speech), f"lost the last {len(speech) - kept} samples of the utterance"
        np.testing.assert_array_equal(out[-len(speech):], speech)

    @pytest.mark.parametrize("rate", [16000, 22050, 44100, 11025])
    def test_vad_segment_times_map_to_the_same_times_at_the_client_rate(self, monkeypatch, rate):
        import src.vad.silero as silero
        from src.wyoming import stt_handler

        class OneSegment:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_speech_segments(self, *_args, **_kwargs):
                return [silero.Segment(start_ms=60_000, end_ms=60_500)]

        monkeypatch.setattr(stt_handler.settings, "stt_vad_enabled", True)
        monkeypatch.setattr(silero, "_vad_model", MagicMock())
        monkeypatch.setattr(silero, "SileroVAD", OneSegment)
        audio =(np.arange(61 * rate) % 32768).astype(np.int16)  # sample i holds i mod 32768

        out = np.frombuffer(
            stt_handler._extract_speech_segments(audio.tobytes(), rate=rate, width=2, channels=1),
            dtype=np.int16,
        )

        # 60.0 s to 60.5 s; at 22.05 and 44.1 kHz the slice was 136 ms early. It starts at 59.7 s:
        # the segment keeps its 300 ms lead-in.
        start, end = 59_700 * rate // 1000, -(-60_500 * rate // 1000)
        assert (int(out[0]), len(out)) == (start % 32768, end - start)
        np.testing.assert_array_equal(out, audio[start:end])

    def test_vad_trimming_keeps_300_ms_before_the_first_speech_window(self, monkeypatch):
        # Each segment was cut at the first window over
        # STT_VAD_THRESHOLD, so a word's onset, which Silero scores below the threshold while it is
        # already audible, never reached Whisper. A segment keeps 300 ms before that window, the same
        # as /v1/realtime's default prefix_padding_ms.
        import src.vad.silero as silero
        from src.wyoming import stt_handler

        # 12 idle windows (384 ms) ending in an onset scored under the threshold, as Silero scores
        # real speech, then 12 windows of speech to the end
        probs = [0.01] * 10 + [0.41, 0.39] + [0.97] * 12

        class ScriptedSession:
            """Fake Silero ONNX session: the scripted probability of each window in turn."""

            def __init__(self):
                self._probs = iter(probs)

            def run(self, _output_names, inputs):
                return [np.array([[next(self._probs, 0.0)]], dtype=np.float32), inputs["state"]]

        monkeypatch.setattr(stt_handler.settings, "stt_vad_enabled", True)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_threshold", 0.5)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_min_speech_ms", 250)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_silence_ms", 800)
        monkeypatch.setattr(silero, "_vad_model", silero.SileroVAD(ScriptedSession()))
        audio = np.arange(len(probs) * 512, dtype=np.int16)  # sample i holds i

        out = np.frombuffer(
            stt_handler._extract_speech_segments(audio.tobytes(), rate=16000, width=2, channels=1),
            dtype=np.int16,
        )

        trigger = 12 * 512  # 384 ms
        assert int(out[0]) == trigger - 4800, (
            f"the trimmed command starts at {int(out[0]) / 16:.0f} ms and the first window over the "
            f"threshold at {trigger / 16:.0f} ms: the onset before it was cut, not kept for 300 ms"
        )
        np.testing.assert_array_equal(out, audio[trigger - 4800:])

    @pytest.mark.parametrize("rate", [16000, 22050, 44100, 11025])
    def test_vad_lead_in_stops_at_the_audio_start_and_at_the_previous_segment(self, monkeypatch, rate):
        # The 300 ms lead-in is clamped to the start of the audio and to
        # the previous segment, which it joins when it reaches it, so no audio is sent twice.
        import src.vad.silero as silero
        from src.wyoming import stt_handler

        class Segments:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_speech_segments(self, *_args, **_kwargs):
                return [
                    silero.Segment(start_ms=100, end_ms=600),
                    silero.Segment(start_ms=800, end_ms=1500),
                    silero.Segment(start_ms=3000, end_ms=3500),
                ]

        monkeypatch.setattr(stt_handler.settings, "stt_vad_enabled", True)
        monkeypatch.setattr(silero, "_vad_model", MagicMock())
        monkeypatch.setattr(silero, "SileroVAD", Segments)
        audio = (np.arange(4 * rate) % 32768).astype(np.int16)  # sample i holds i mod 32768

        out = np.frombuffer(
            stt_handler._extract_speech_segments(audio.tobytes(), rate=rate, width=2, channels=1),
            dtype=np.int16,
        )

        def start_at(ms):
            return ms * rate // 1000

        def end_at(ms):
            return -(-ms * rate // 1000)

        # 100 ms starts at 0; 800 ms reaches back past the first segment's end (600 ms), so the two
        # are one stretch to 1500 ms; 3000 ms keeps its whole lead-in, from 2700 ms.
        expected = np.concatenate([audio[:end_at(1500)], audio[start_at(2700):end_at(3500)]])
        assert len(out) == len(expected), (
            f"{len(out) * 1000 / rate:.0f} ms of audio kept; expected 0-1500 ms and 2700-3500 ms "
            f"({len(expected) * 1000 / rate:.0f} ms)"
        )
        np.testing.assert_array_equal(out, expected)

    # Wyoming accepts 8-, 16-, 24- and 32-bit PCM with any channel count
    # (README, Wyoming), and STT_NORMALIZE, STT_NOISE_REDUCE and STT_VAD_ENABLED are not limited to
    # 16-bit mono. Each format must reach the STT model processed like the same sound as 16-bit mono.
    PCM_FORMATS = [("8bit-mono", 1, 1), ("24bit-mono", 3, 1), ("32bit-mono", 4, 1), ("16bit-stereo", 2, 2), ("24bit-stereo", 3, 2)]

    @staticmethod
    def _encode_pcm(x: np.ndarray, width: int, channels: int) -> bytes:
        """Float mono audio in [-1, 1] as little-endian PCM, with WAV's unsigned 8-bit samples."""
        if width == 1:
            pcm = (np.round(x * 127.0) + 128).astype(np.uint8)
        elif width == 2:
            pcm = np.round(x * 32767.0).astype("<i2")
        elif width == 3:
            pcm = np.round(x * 8388607.0).astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].copy()
        else:
            pcm = np.round(x * 2147483647.0).astype("<i4")
        frames = pcm.reshape(len(x), -1)
        return np.repeat(frames, channels, axis=0).reshape(len(x), -1).tobytes()  # the same sound on each channel

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """A PCM WAV of any accepted width and channel count as float mono, and its rate."""
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            width, channels, rate = wav.getsampwidth(), wav.getnchannels(), wav.getframerate()
            raw = wav.readframes(wav.getnframes())
        if width == 1:
            x = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128.0) / 128.0
        elif width == 2:
            x = np.frombuffer(raw, "<i2") / 32768.0
        elif width == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            x = np.where(v >= 1 << 23, v - (1 << 24), v) / 8388608.0
        else:
            x = np.frombuffer(raw, "<i4") / 2147483648.0
        return x.reshape(-1, channels).mean(axis=1), rate

    @classmethod
    def _received_by_stt(cls, x: np.ndarray, width: int, channels: int) -> tuple[np.ndarray, int]:
        """Send x through the Wyoming STT path in this format; return the audio the STT model got."""
        from src.wyoming import stt_handler

        router = MagicMock()
        router.transcribe.return_value = {"text": "ok"}
        text = asyncio.run(asyncio.wait_for(stt_handler.handle_transcribe(
            audio_chunks=[cls._encode_pcm(x, width, channels)], rate=16000, width=width, channels=channels,
            stt_router=router,
        ), timeout=30))
        assert text == "ok"
        return cls._decode_wav(router.transcribe.call_args.kwargs["audio"])

    @staticmethod
    def _tone(seconds: float, amplitude: float) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * 220 * np.arange(int(seconds * 16000)) / 16000)

    @pytest.fixture
    def no_stt_preprocessing(self, monkeypatch):
        from src.wyoming import stt_handler

        for name in ("stt_vad_enabled", "stt_normalize", "stt_noise_reduce"):
            monkeypatch.setattr(stt_handler.settings, name, False)
        return stt_handler.settings

    @pytest.mark.parametrize("label, width, channels", PCM_FORMATS, ids=[f[0] for f in PCM_FORMATS])
    def test_stt_normalize_applies_to_every_accepted_format(self, no_stt_preprocessing, monkeypatch, label, width, channels):
        # 8-, 24- and 32-bit audio reached the model at its input level.
        monkeypatch.setattr(no_stt_preprocessing, "stt_normalize", True)
        quiet = self._tone(1.0, 0.05)  # about -29 dBFS RMS, which STT_NORMALIZE boosts

        def rms_db(x):
            return float(20 * np.log10(np.sqrt(np.mean(np.square(x)))))

        reference, _ = self._received_by_stt(quiet, 2, 1)
        received, rate = self._received_by_stt(quiet, width, channels)

        assert rate == 16000
        assert rms_db(reference) - rms_db(quiet) > 6, "the 16-bit mono reference is normalized"
        assert abs(rms_db(received) - rms_db(reference)) < 1.0, (
            f"{label}: the STT model got {rms_db(received):.1f} dBFS (input {rms_db(quiet):.1f}); "
            f"the same audio as 16-bit mono got {rms_db(reference):.1f}"
        )

    @pytest.mark.parametrize("label, width, channels", PCM_FORMATS, ids=[f[0] for f in PCM_FORMATS])
    def test_stt_noise_reduce_applies_to_every_accepted_format(self, no_stt_preprocessing, monkeypatch, label, width, channels):
        # Denoise never ran on 8-, 24- or 32-bit audio.
        from src.audio import preprocessing

        calls = []

        def recording_reduce_noise(audio, sample_rate):
            calls.append((len(audio), sample_rate))
            return audio

        monkeypatch.setattr(no_stt_preprocessing, "stt_noise_reduce", True)
        monkeypatch.setattr(preprocessing, "reduce_noise", recording_reduce_noise)

        self._received_by_stt(self._tone(1.0, 0.3), width, channels)

        assert calls == [(16000, 16000)], f"{label}: denoise ran on {calls}, not once on the 1 s utterance"

    @pytest.mark.parametrize("label, width, channels", PCM_FORMATS, ids=[f[0] for f in PCM_FORMATS])
    def test_vad_trims_every_accepted_format(self, no_stt_preprocessing, monkeypatch, label, width, channels):
        # VAD trimming skipped every format but 16-bit mono.
        import src.vad.silero as silero

        class EnergySession:
            """Fake Silero ONNX session: speech wherever the 512-sample window has energy."""

            def run(self, _output_names, inputs):
                window = inputs["input"][0, silero.VAD_CONTEXT_SIZE:]
                prob = 0.9 if float(np.sqrt(np.mean(window ** 2))) > 0.05 else 0.0
                return [np.array([[prob]], dtype=np.float32), inputs["state"]]

        monkeypatch.setattr(silero, "_vad_model", silero.SileroVAD(EnergySession()))
        monkeypatch.setattr(no_stt_preprocessing, "stt_vad_enabled", True)
        monkeypatch.setattr(no_stt_preprocessing, "stt_vad_threshold", 0.5)
        monkeypatch.setattr(no_stt_preprocessing, "stt_vad_min_speech_ms", 250)
        monkeypatch.setattr(no_stt_preprocessing, "stt_vad_silence_ms", 800)
        audio = np.concatenate([np.zeros(16000), self._tone(2.0, 0.3)])  # 1 s of silence, then 2 s of speech

        reference, _ = self._received_by_stt(audio, 2, 1)
        received, rate = self._received_by_stt(audio, width, channels)

        assert len(reference) / 16000 < 2.5, "the 16-bit mono reference is trimmed"
        assert abs(len(received) / rate - len(reference) / 16000) < 0.05, (
            f"{label}: {len(received) / rate:.2f} s reached the STT model; "
            f"the same audio as 16-bit mono was trimmed to {len(reference) / 16000:.2f} s"
        )

    @pytest.mark.parametrize("label, width, channels", PCM_FORMATS, ids=[f[0] for f in PCM_FORMATS])
    def test_every_accepted_format_reaches_stt_as_the_same_sound(self, no_stt_preprocessing, label, width, channels):
        # A control: whatever the format, the model gets the same waveform,
        # within 8-bit rounding (1/128) or a few 16-bit steps: no sign, byte-order or offset slips.
        tone = self._tone(0.5, 0.4)

        received, rate = self._received_by_stt(tone, width, channels)

        assert rate == 16000 and len(received) == len(tone)
        np.testing.assert_allclose(received, tone, atol=1 / 128 if width == 1 else 4 / 32768)

    def test_an_8bit_wav_sent_by_wav_to_chunks_reaches_stt_as_the_same_sound(self, no_stt_preprocessing):
        # The Wyoming protocol doesn't say whether 8-bit PCM is
        # signed. Open Speech reads it as unsigned, as in WAV, which is what the wyoming package's own WAV path
        # sends: wav_to_chunks, used by its HTTP-to-Wyoming STT bridge, copies the WAV's frames into the chunks.
        import io
        import wave

        from wyoming.audio import wav_to_chunks

        from src.wyoming import stt_handler

        tone = self._tone(0.5, 0.4)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(16000)
            wav.writeframes((np.round(tone * 127.0) + 128).astype(np.uint8).tobytes())  # 0x80 is silence
        with wave.open(io.BytesIO(buf.getvalue()), "rb") as wav:
            start, *chunks = wav_to_chunks(wav, samples_per_chunk=1024, start_event=True)
        assert isinstance(start, AudioStart) and (start.rate, start.width, start.channels) == (16000, 1, 1)
        assert len(chunks) > 1 and all(isinstance(chunk, AudioChunk) for chunk in chunks)
        router = MagicMock()
        router.transcribe.return_value = {"text": "ok"}

        text = asyncio.run(asyncio.wait_for(stt_handler.handle_transcribe(
            audio_chunks=[chunk.audio for chunk in chunks], rate=start.rate, width=start.width,
            channels=start.channels, stt_router=router,
        ), timeout=30))

        assert text == "ok"
        received, rate = self._decode_wav(router.transcribe.call_args.kwargs["audio"])
        assert rate == 16000 and len(received) == len(tone)
        np.testing.assert_allclose(received, tone, atol=1 / 128)


# ---------------------------------------------------------------------------
# TTS handler
# ---------------------------------------------------------------------------


class TestTTSHandler:
    @pytest.mark.asyncio
    async def test_synthesis_waits_for_its_provider_on_the_providers_own_thread(self, monkeypatch):
        # Voice commands queued behind a busy provider must not each hold a thread of asyncio's
        # default executor, which every other request and to_thread call in the process shares.
        from src.tts.router import TTSRouter
        from src.wyoming import tts_handler

        threads: list[str] = []

        class Backend:
            name = "fake"
            sample_rate = 16000

            def synthesize(self, text, voice, speed=1.0, lang_code=None, **_options):
                threads.append(threading.current_thread().name)
                yield np.zeros(1600, dtype=np.float32)

        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu", external_providers="")
        router.register_backend("fake", Backend())
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings(tts_model="fake"))
        events: list[Event] = []

        async def capture(event):
            events.append(event)

        assert await asyncio.wait_for(tts_handler.handle_synthesize(
            text="hello", voice="v", tts_router=router, write_event=capture,
        ), timeout=10)

        assert [e.type for e in events][-1] == "audio-stop"
        assert len(threads) == 1 and threads[0].startswith("tts-fake")

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
    async def test_synthesize_disabled_sends_error_event(self, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings(tts_enabled=False)
        try:
            events: list[Event] = []

            async def capture(event):
                events.append(event)

            await tts_handler.handle_synthesize(
                text="hello", voice=None, tts_router=mock_tts_router,
                write_event=capture,
            )
            assert [e.type for e in events] == ["error"]
            mock_tts_router.synthesize.assert_not_called()
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

    @pytest.mark.asyncio
    async def test_synthesize_uses_backend_native_sample_rate(self, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings()
        mock_tts_router.sample_rate_for.return_value = 22050
        mock_tts_router.synthesize.return_value = iter([
            np.ones(2205, dtype=np.float32) * 0.25,
        ])
        try:
            events: list[Event] = []

            async def capture(event):
                events.append(event)

            await tts_handler.handle_synthesize(
                text="hello", voice=None, tts_router=mock_tts_router,
                write_event=capture,
            )

            audio_events = [
                AudioChunk.from_event(event)
                for event in events
                if AudioChunk.is_type(event.type)
            ]
            assert len(audio_events) == 1
            assert len(audio_events[0].audio) == 1600 * 2
        finally:
            tts_handler.settings = orig

    @pytest.mark.asyncio
    async def test_synthesize_runs_backend_off_event_loop(self, mock_tts_router):
        from src.wyoming import tts_handler
        orig = tts_handler.settings
        tts_handler.settings = _make_mock_settings()
        event_loop_thread = threading.get_ident()
        synthesis_threads: list[int] = []

        def synthesize(**_kwargs):
            synthesis_threads.append(threading.get_ident())
            yield np.zeros(100, dtype=np.float32)

        mock_tts_router.synthesize.side_effect = synthesize
        try:
            await tts_handler.handle_synthesize(
                text="hello", voice=None, tts_router=mock_tts_router,
                write_event=AsyncMock(),
            )
            assert synthesis_threads
            assert synthesis_threads[0] != event_loop_thread
        finally:
            tts_handler.settings = orig

    @pytest.mark.asyncio
    async def test_synthesize_rejects_text_over_max_input_length(self, mock_tts_router, monkeypatch):
        # README: TTS_MAX_INPUT_LENGTH "Max text length"; /v1/audio/speech answers 400.
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings(tts_max_input_length=20))
        events: list[Event] = []

        async def capture(event):
            events.append(event)

        assert await tts_handler.handle_synthesize(
            text="word " * 200, voice=None, tts_router=mock_tts_router, write_event=capture,
        ) is False

        assert [e.type for e in events] == ["error"]
        error = Error.from_event(events[0])
        assert error.code == "input_too_long"
        assert "Max: 20 characters" in error.text
        mock_tts_router.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_rejects_text_over_the_model_input_limit(self, mock_tts_router, monkeypatch):
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings())
        mock_tts_router.max_input_chars_for.return_value = 10
        events: list[Event] = []

        async def capture(event):
            events.append(event)

        await tts_handler.handle_synthesize(
            text="eleven char", voice=None, tts_router=mock_tts_router, write_event=capture,
        )

        assert [(e.type, e.data.get("code")) for e in events] == [("error", "input_too_long")]
        mock_tts_router.synthesize.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing", ["_pronunciation_dict", "_tts_cache"])
    async def test_failure_before_synthesis_sends_error_event(self, mock_tts_router, monkeypatch, failing):
        # The dictionary and the cache lookup run before synthesis; a failure there must
        # still end the request with an error event, or the client waits on a silent socket.
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings(tts_cache_enabled=True))
        broken = MagicMock()
        broken.apply.side_effect = RuntimeError("bad replacement template")
        broken.get.side_effect = PermissionError(13, "Permission denied")
        monkeypatch.setattr(tts_handler, failing, broken)
        events: list[Event] = []

        async def capture(event):
            events.append(event)

        await asyncio.wait_for(tts_handler.handle_synthesize(
            text="hello", voice=None, tts_router=mock_tts_router, write_event=capture,
        ), timeout=10)

        assert [e.type for e in events] == ["error"]

    @pytest.mark.asyncio
    async def test_a_text_the_pronunciation_dictionary_removes_is_an_empty_utterance(
        self, mock_tts_router, monkeypatch
    ):
        # Wyoming had the same gap: "an empty value removes the word" (README
        # TTS_PRONUNCIATION_DICT), and what the dictionary left ("") went to the model, which Pocket-TTS and
        # the GPU workers refuse, so Home Assistant got an error event and the connection was closed.
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings())
        removing = MagicMock()
        removing.apply.side_effect = lambda text: text.replace("[chime]", "")
        monkeypatch.setattr(tts_handler, "_pronunciation_dict", removing)

        def synthesize(**kwargs):
            if not kwargs["text"].strip():
                raise ValueError("Text must not be empty for Pocket TTS synthesis.")
            yield np.zeros(100, dtype=np.float32)

        mock_tts_router.synthesize.side_effect = synthesize
        events: list[Event] = []

        async def capture(event):
            events.append(event)

        assert await asyncio.wait_for(tts_handler.handle_synthesize(
            text=" [chime] ", voice=None, tts_router=mock_tts_router, write_event=capture,
        ), timeout=10) is True

        assert [e.type for e in events] == ["audio-start", "audio-stop"]
        mock_tts_router.synthesize.assert_not_called()


# ---------------------------------------------------------------------------
# TTS cache shared with /v1/audio/speech
# ---------------------------------------------------------------------------


def _constant_router(sample_rate: int, seconds: float = 1.0):
    """A backend at sample_rate whose output keeps its length through trim and normalize."""
    router = MagicMock()
    router.loaded_models.return_value = []
    router.list_voices.return_value = []
    router.sample_rate_for.return_value = sample_rate
    router.synthesize.side_effect = lambda **_kw: iter(
        [np.full(int(sample_rate * seconds), 0.25, dtype=np.float32)]
    )
    return router


class TestTTSCache:
    @pytest.fixture
    def cache(self, tmp_path, monkeypatch):
        from src.cache.tts_cache import TTSCache
        from src.wyoming import tts_handler

        cache = TTSCache(str(tmp_path / "tts-cache"), 50, enabled=True)
        monkeypatch.setattr(tts_handler, "_tts_cache", cache)
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings(
            tts_cache_enabled=True, tts_model="piper/test",
            tts_trim_silence=True, tts_normalize_output=True,
        ))
        return cache

    @pytest.mark.asyncio
    async def test_native_rate_audio_is_cached_as_24khz_pcm(self, cache):
        from src.wyoming import tts_handler

        await tts_handler.handle_synthesize(
            text="hello", voice="v1", tts_router=_constant_router(22050),
            write_event=AsyncMock(),
        )

        cached = cache.get(text="hello", voice="v1", speed=1.0, fmt="pcm", model="piper/test")
        assert cached is not None
        assert len(cached) // 2 == 24000  # one second of 24 kHz PCM16

    @pytest.mark.asyncio
    async def test_cache_hit_plays_cached_pcm_as_24khz(self, cache):
        from src.wyoming import tts_handler

        one_second_24k = np.full(24000, 8000, dtype=np.int16).tobytes()
        cache.set(text="hello", voice="v1", speed=1.0, fmt="pcm", model="piper/test", audio=one_second_24k)
        router = _constant_router(22050)
        events: list[Event] = []

        async def capture(event):
            events.append(event)

        await tts_handler.handle_synthesize(
            text="hello", voice="v1", tts_router=router, write_event=capture,
        )

        router.synthesize.assert_not_called()
        samples = sum(
            len(AudioChunk.from_event(e).audio) // 2 for e in events if AudioChunk.is_type(e.type)
        )
        assert samples == 16000  # one second at Wyoming's 16 kHz

    @pytest.mark.asyncio
    @pytest.mark.parametrize("setting", ["tts_trim_silence", "tts_normalize_output"])
    async def test_cache_entry_is_keyed_by_post_processing_settings(self, cache, monkeypatch, setting):
        # A restart with another TTS_TRIM_SILENCE or TTS_NORMALIZE_OUTPUT must not replay
        # audio shaped the old way.
        from src.wyoming import tts_handler

        router = _constant_router(22050)
        await tts_handler.handle_synthesize(
            text="hello", voice="v1", tts_router=router, write_event=AsyncMock(),
        )
        monkeypatch.setattr(tts_handler.settings, setting, False)
        await tts_handler.handle_synthesize(
            text="hello", voice="v1", tts_router=router, write_event=AsyncMock(),
        )

        assert router.synthesize.call_count == 2

    def test_speech_pcm_cache_hit_after_wyoming_synthesis_is_24khz(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from src import main as main_module
        from src.cache.tts_cache import TTSCache
        from src.wyoming import tts_handler

        # Production: both transports read one settings object and one TTS_CACHE_DIR.
        cache_dir = str(tmp_path / "tts-cache")
        monkeypatch.setattr(tts_handler, "_tts_cache", TTSCache(cache_dir, 50, enabled=True))
        monkeypatch.setattr(main_module, "tts_cache", TTSCache(cache_dir, 50, enabled=True))
        monkeypatch.setattr(main_module.settings, "tts_cache_enabled", True)
        monkeypatch.setattr(main_module.settings, "tts_model", "piper/test")
        router = _constant_router(22050)
        monkeypatch.setattr(main_module, "tts_router", router)

        asyncio.run(asyncio.wait_for(tts_handler.handle_synthesize(
            text="hello", voice="v1", tts_router=router, write_event=AsyncMock(),
        ), timeout=30))
        response = TestClient(main_module.app).post("/v1/audio/speech", json={
            "model": "piper/test", "input": "hello", "voice": "v1", "response_format": "pcm",
        })

        assert response.status_code == 200
        assert response.headers.get("x-cache") == "HIT"
        assert len(response.content) // 2 == 24000  # one second at 24 kHz

    def test_wyoming_and_speech_share_one_cache_instance(self):
        # One TTSCache per directory: its lock orders every write and eviction, and two
        # instances on one directory evict under each other's renames.
        from src import main as main_module
        from src.cache.tts_cache import get_shared_tts_cache
        from src.wyoming import tts_handler

        assert tts_handler._tts_cache is main_module.tts_cache
        assert main_module.tts_cache is get_shared_tts_cache(main_module.settings.tts_cache_dir)


# ---------------------------------------------------------------------------
# Event handler integration
# ---------------------------------------------------------------------------


def _frame(header: dict, body: bytes = b"") -> bytes:
    """One event as raw bytes: the JSON header line, then its data section and payload."""
    return json.dumps(header).encode() + b"\n" + body


def _stream_of(raw: bytes) -> asyncio.StreamReader:
    """A reader holding raw, then the end of the stream."""
    reader = asyncio.StreamReader()
    reader.feed_data(raw)
    reader.feed_eof()
    return reader


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
    async def test_audio_stop_without_audio_answers_with_empty_transcript(
        self, mock_stt_router, mock_tts_router,
    ):
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        await handler.handle_event(Transcribe().event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(AudioStop().event())

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [e.type for e in events] == ["transcript"]
        assert Transcript.from_event(events[0]).text == ""
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("channels", "audio"), [(1, b"\x00"), (2, b"\x00\x00")], ids=["mono-one-byte", "stereo-half-frame"],
    )
    async def test_audio_shorter_than_one_frame_answers_with_one_transcript(
        self, mock_stt_router, mock_tts_router, monkeypatch, channels, audio,
    ):
        # _pcm_to_mono16 drops the partial frame, so no sample is left,
        # and STT_NORMALIZE (on by default) raised on the empty clip. The client got an error event and
        # the connection closed; AudioStop must answer with a transcript, as when no audio arrived.
        from src.wyoming import stt_handler

        monkeypatch.setattr(stt_handler.settings, "stt_normalize", True)
        monkeypatch.setattr(stt_handler.settings, "stt_noise_reduce", False)
        monkeypatch.setattr(stt_handler.settings, "stt_vad_enabled", False)
        mock_stt_router.transcribe.return_value = {"text": ""}
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        for event in (
            Transcribe().event(),
            AudioStart(rate=16000, width=2, channels=channels).event(),
            AudioChunk(rate=16000, width=2, channels=channels, audio=audio).event(),
        ):
            assert await asyncio.wait_for(handler.handle_event(event), timeout=10) is True
        keep_open = await asyncio.wait_for(handler.handle_event(AudioStop().event()), timeout=30)

        events = [c[0][0] for c in handler.write_event.call_args_list]
        errors = [Error.from_event(e).text for e in events if Error.is_type(e.type)]
        assert [e.type for e in events] == ["transcript"], errors
        assert Transcript.from_event(events[0]).text == ""
        assert keep_open is True

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

    @pytest.mark.asyncio
    async def test_synthesize_failure_sends_error_event(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings())
        mock_tts_router.synthesize.side_effect = RuntimeError("voice not available")
        handler = self._make_handler(mock_stt_router, mock_tts_router)

        # False ends the connection: Home Assistant ignores error events and waits for
        # audio-stop until the connection closes.
        assert await handler.handle_event(Synthesize(text="hello").event()) is False

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [e.type for e in events] == ["error"]
        assert "voice not available" in Error.from_event(events[0]).text

    @pytest.mark.asyncio
    async def test_transcribe_failure_sends_error_event_and_ends_connection(
        self, mock_stt_router, mock_tts_router,
    ):
        # An empty transcript would read as "nothing recognized"; HTTP STT answers 5xx.
        mock_stt_router.transcribe.side_effect = RuntimeError("model 'missing/model' failed to load")
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        await handler.handle_event(Transcribe(name="missing/model").event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1600).event())
        assert await handler.handle_event(AudioStop().event()) is False

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [e.type for e in events] == ["error"]
        error = Error.from_event(events[0])
        assert "failed to load" in error.text
        assert error.code == "RuntimeError"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("requested", "decoded"), [
        ("de-DE", "de"), ("EN", "en"), ("", None), ("auto", None), (None, None),
    ])
    async def test_transcribe_language_reaches_whisper_as_its_code(
        self, mock_stt_router, mock_tts_router, requested, decoded,
    ):
        # Home Assistant sends its pipeline language, region tag and all; faster-whisper knows
        # only bare codes, and "" or "auto" mean auto-detect, as on /v1/audio/transcriptions.
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        assert await handler.handle_event(Transcribe(language=requested).event()) is True
        await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1600).event())
        assert await handler.handle_event(AudioStop().event()) is True

        assert mock_stt_router.transcribe.call_args.kwargs["language"] == decoded

    @pytest.mark.asyncio
    @pytest.mark.parametrize("named", [True, False], ids=["named", "default"])
    async def test_a_language_parakeet_does_not_transcribe_is_refused_before_any_audio(
        self, mock_stt_router, mock_tts_router, monkeypatch, named,
    ):
        from src.wyoming import server

        parakeet = "nvidia/parakeet-tdt-0.6b-v2"
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        monkeypatch.setattr(server, "settings", _make_mock_settings(
            stt_model=parakeet if not named else "Systran/faster-whisper-base", tts_enabled=False,
        ))

        event = Transcribe(name=parakeet, language="de-DE") if named else Transcribe(language="de-DE")
        assert await handler.handle_event(event.event()) is False

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [(e.type, e.data.get("code")) for e in events] == [("error", "language_unsupported")]
        assert "English only" in Error.from_event(events[0]).text
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["en-US", "en", "", None])
    async def test_parakeet_takes_english_and_no_language(self, mock_stt_router, mock_tts_router, monkeypatch, language):
        from src.wyoming import server

        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        monkeypatch.setattr(server, "settings", _make_mock_settings(stt_model="nvidia/parakeet-tdt-0.6b-v2",
                                                                    tts_enabled=False))

        assert await handler.handle_event(Transcribe(language=language).event()) is True

    @pytest.mark.asyncio
    async def test_unknown_transcribe_language_is_refused_before_any_audio(
        self, mock_stt_router, mock_tts_router,
    ):
        # HTTP answers 400. faster-whisper would raise only after decoding the whole utterance.
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        assert await handler.handle_event(Transcribe(language="klingon").event()) is False

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [(e.type, e.data.get("code")) for e in events] == [("error", "invalid_language")]
        assert "'klingon'" in Error.from_event(events[0]).text
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", [5, ["en"], {"code": "en"}], ids=["number", "list", "object"])
    async def test_non_string_transcribe_language_is_refused_as_invalid(
        self, mock_stt_router, mock_tts_router, caplog, language,
    ):
        # Any client on the network can send this: it gets an answer, not a logged traceback.
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        with caplog.at_level(logging.WARNING, logger="src.wyoming.server"):
            assert await handler.handle_event(Event(type="transcribe", data={"language": language})) is False

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [(e.type, e.data.get("code")) for e in events] == [("error", "invalid_language")]
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.asyncio
    async def test_audio_over_the_upload_limit_is_refused(self, mock_stt_router, mock_tts_router, monkeypatch):
        # The port has no authentication; like every HTTP and WebSocket audio input, what
        # one transcription buffers is bounded (OS_MAX_UPLOAD_MB).
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_max_upload_mb", 1)
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * (256 * 1024)).event()  # 512 KiB

        await handler.handle_event(Transcribe().event())
        assert await handler.handle_event(chunk) is True
        assert await handler.handle_event(chunk) is True  # exactly 1 MiB
        assert await handler.handle_event(chunk) is False

        assert sum(len(c) for c in handler._audio_chunks) == 0
        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [(e.type, e.data.get("code")) for e in events] == [("error", "audio_too_large")]
        await handler.handle_event(AudioStop().event())
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("event", [
        AudioStart(rate=1, width=2, channels=1).event(),
        AudioChunk(rate=1, width=2, channels=1, audio=b"\x10\x00" * 200).event(),
    ], ids=["audio-start", "audio-chunk"])
    async def test_out_of_range_sample_rate_is_refused_before_decoding(
        self, mock_stt_router, mock_tts_router, event,
    ):
        # The decoder resamples to 16 kHz: 200 samples at 1 Hz would decode to 3.2M.
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        await handler.handle_event(Transcribe().event())
        assert await handler.handle_event(event) is False
        await handler.handle_event(AudioStop().event())

        error = handler.write_event.call_args_list[0][0][0]
        assert Error.from_event(error).code == "unsupported_audio"
        assert "1 Hz" in Error.from_event(error).text
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_event_is_answered_with_an_error_event(self, mock_stt_router, mock_tts_router):
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)

        assert await handler.handle_event(Event(type="audio-chunk", data={}, payload=b"\0\0")) is False

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [e.type for e in events] == ["error"]

    @pytest.mark.asyncio
    async def test_data_section_larger_than_metadata_needs_is_refused(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        # Audio travels as payload; the data section is JSON metadata. Parsed JSON takes
        # many times its size in memory, and parsing runs on the loop HTTP clients share,
        # so the audio limit (OS_MAX_UPLOAD_MB) is far too generous for it.
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_max_upload_mb", 100)
        monkeypatch.setattr(server.settings, "tts_max_input_length", 4096)
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        handler.reader = _stream_of(_frame({"type": "describe", "data_length": 10 * 1024 * 1024}))

        await asyncio.wait_for(handler.run(), timeout=5)

        events = [c[0][0] for c in handler.write_event.call_args_list]
        assert [(e.type, e.data.get("code")) for e in events] == [("error", "event_too_large")]
        handler.writer.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_section_fits_the_longest_synthesize_text(self, monkeypatch):
        # Text TTS_MAX_INPUT_LENGTH admits must reach the length check however the client
        # escapes it: "\ud83d\ude00" is 12 bytes for one character.
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "tts_max_input_length", 100_000)
        text = "\U0001F600" * 100_000
        data = json.dumps(Synthesize(text=text).event().data).encode()
        assert len(data) > 1024 * 1024

        event = await server.read_event(_stream_of(_frame({"type": "synthesize", "data_length": len(data)}, data)))

        assert Synthesize.from_event(event).text == text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", [
        # Within the reader's 64 KiB line limit.
        b'{"type": "describe", "data": ' + b"[" * 30_000 + b"]" * 30_000 + b"}\n",
        _frame({"type": "describe", "data_length": 60_000}, b"[" * 30_000 + b"]" * 30_000),
    ], ids=["header", "data-section"])
    async def test_deeply_nested_json_ends_the_connection_like_any_malformed_event(
        self, mock_stt_router, mock_tts_router, raw,
    ):
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        handler.reader = _stream_of(raw)

        await asyncio.wait_for(handler.run(), timeout=5)  # json.loads raises RecursionError

        handler.writer.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_reset_ends_the_handler_without_an_exception(self, mock_stt_router, mock_tts_router):
        # handler.run is a task nobody awaits: an exception from it is only logged as
        # "Task exception was never retrieved".
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        handler.reader = asyncio.StreamReader()
        handler.reader.set_exception(ConnectionResetError("connection reset by peer"))

        await asyncio.wait_for(handler.run(), timeout=5)

        handler.writer.close.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [
        OSError(errno.EHOSTUNREACH, "No route to host"),
        TimeoutError(errno.ETIMEDOUT, "Connection timed out"),  # from the socket, not the idle wait
    ], ids=["unreachable", "timed-out"])
    async def test_any_socket_error_ends_the_handler_like_a_reset(
        self, mock_stt_router, mock_tts_router, caplog, error,
    ):
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        handler.reader = asyncio.StreamReader()
        handler.reader.set_exception(error)

        with caplog.at_level(logging.INFO, logger="src.wyoming.server"):
            await asyncio.wait_for(handler.run(), timeout=5)

        handler.writer.close.assert_called_once()
        assert "OS_WYOMING_IDLE_TIMEOUT_S" not in caplog.text  # a broken socket is not a slow client

    @pytest.mark.asyncio
    async def test_client_that_stops_reading_is_disconnected(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        # A client that sends a request and never reads the reply would hold the handler in
        # drain(), and its connection slot, for good once the socket buffers fill.
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_wyoming_idle_timeout_s", 0.1)
        handler = self._make_handler(mock_stt_router, mock_tts_router, tts_enabled=False)
        del handler.write_event  # the handler's own, writing to the stalled writer below

        async def never_drains():
            await asyncio.Event().wait()

        handler.writer.drain = never_drains

        assert await asyncio.wait_for(handler.handle_event(Describe().event()), timeout=3) is False
        handler.writer.transport.abort.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_output", [[], [np.zeros(0, dtype=np.float32)]])
    async def test_synthesize_without_audio_sends_empty_audio_stream(
        self, mock_stt_router, mock_tts_router, monkeypatch, backend_output,
    ):
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings())
        mock_tts_router.synthesize.return_value = iter(backend_output)
        handler = self._make_handler(mock_stt_router, mock_tts_router)

        await handler.handle_event(Synthesize(text="hello").event())

        types = [c[0][0].type for c in handler.write_event.call_args_list]
        assert types == ["audio-start", "audio-stop"]


# ---------------------------------------------------------------------------
# Over TCP, as Home Assistant connects
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _tcp_server(stt_router, tts_router):
    """Serve OpenSpeechEventHandler with wyoming's own TCP server on a free local port."""
    from wyoming.server import AsyncTcpServer

    from src.wyoming import server

    tcp_server = AsyncTcpServer("127.0.0.1", 0)
    info = MagicMock()
    await tcp_server.start(
        lambda reader, writer: server.OpenSpeechEventHandler(reader, writer, stt_router, tts_router, info)
    )
    try:
        yield tcp_server._server.sockets[0].getsockname()[1]
    finally:
        await tcp_server.stop()


async def _exchange(port: int, events: list[Event], *, raw: bytes = b"", end_input: bool = False) -> list[Event]:
    """Send events (then any raw bytes), and read replies until the server closes the connection.

    A server that leaves the connection open fails the read's timeout, as a waiting
    Home Assistant would hang.
    """
    from wyoming.event import async_read_event, async_write_event

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        for event in events:
            await async_write_event(event, writer)
        if raw:
            writer.write(raw)
            await writer.drain()
        if end_input:
            writer.write_eof()
        received = []
        while (event := await asyncio.wait_for(async_read_event(reader), timeout=5)) is not None:
            received.append(event)
        return received
    finally:
        writer.close()


@contextlib.asynccontextmanager
async def _capped_tcp_server(stt_router, tts_router):
    """Serve through make_handler_factory, whose connections share one cap; yields connect()."""
    from wyoming.info import Info
    from wyoming.server import AsyncTcpServer

    from src.wyoming import server

    tcp_server = AsyncTcpServer("127.0.0.1", 0)
    await tcp_server.start(server.make_handler_factory(stt_router, tts_router, Info()))
    port = tcp_server._server.sockets[0].getsockname()[1]
    writers = []

    async def connect():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writers.append(writer)
        return reader, writer

    try:
        yield connect
    finally:
        for writer in writers:
            writer.close()
        await tcp_server.stop()


async def _described(reader, writer) -> bool:
    """Whether the server answers a Describe on this connection with its info."""
    from wyoming.event import async_read_event, async_write_event
    from wyoming.info import Info

    with contextlib.suppress(ConnectionError):
        await async_write_event(Describe().event(), writer)
        event = await asyncio.wait_for(async_read_event(reader), timeout=5)
        return event is not None and Info.is_type(event.type)
    return False


class TestTCPServer:
    @pytest.mark.asyncio
    async def test_failed_transcription_is_answered_and_the_connection_closed(
        self, mock_stt_router, mock_tts_router,
    ):
        mock_stt_router.transcribe.side_effect = RuntimeError("model failed to load")

        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            received = await _exchange(port, [
                Transcribe().event(),
                AudioStart(rate=16000, width=2, channels=1).event(),
                AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1600).event(),
                AudioStop().event(),
            ])

        assert [e.type for e in received] == ["error"]

    @pytest.mark.asyncio
    async def test_failed_synthesis_is_answered_and_the_connection_closed(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings())
        mock_tts_router.synthesize.side_effect = RuntimeError("voice not available")

        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            received = await _exchange(port, [Synthesize(text="hello").event()])

        assert [e.type for e in received] == ["error"]

    @pytest.mark.asyncio
    async def test_oversized_event_is_refused_before_its_payload_arrives(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        # wyoming's own reader buffers whatever payload length a header declares.
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_max_upload_mb", 1)
        header = json.dumps({
            "type": "audio-chunk",
            "data": {"rate": 16000, "width": 2, "channels": 1},
            "payload_length": 1024 * 1024 + 1,
        }).encode() + b"\n"

        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            received = await _exchange(port, [Transcribe().event()], raw=header)

        assert [(e.type, e.data.get("code")) for e in received] == [("error", "event_too_large")]
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_client_still_streaming_over_the_limit_receives_the_error(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        # Closing a socket with unread input resets the connection, and the reset makes
        # the client drop the error event it has not read yet.
        from wyoming.event import async_read_event, async_write_event

        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_max_upload_mb", 1)
        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1024).event()  # 2 KiB
        answered = asyncio.Event()

        async def stream_audio(writer):
            with contextlib.suppress(ConnectionError):
                await async_write_event(Transcribe().event(), writer)
                for _ in range(4096):  # 8 MiB, as a microphone keeps sending
                    if answered.is_set():
                        return
                    await async_write_event(chunk, writer)
                    await asyncio.sleep(0)

        async def read_replies(reader):
            received = []
            try:
                while (event := await async_read_event(reader)) is not None:
                    received.append(event)
            finally:
                answered.set()
            return received

        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            streaming = asyncio.create_task(stream_audio(writer))
            try:
                received = await asyncio.wait_for(read_replies(reader), timeout=10)
            finally:
                answered.set()
                await asyncio.wait_for(streaming, timeout=10)
                writer.close()

        assert [(e.type, e.data.get("code")) for e in received] == [("error", "audio_too_large")]
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_connection_closes_when_the_client_is_done(self, mock_stt_router, mock_tts_router):
        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            received = await _exchange(port, [
                Transcribe().event(),
                AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1600).event(),
                AudioStop().event(),
            ], end_input=True)

        assert [e.type for e in received] == ["transcript"]
        assert Transcript.from_event(received[0]).text == "hello world"

    @pytest.mark.asyncio
    async def test_unknown_language_is_answered_and_the_connection_closed(self, mock_stt_router, mock_tts_router):
        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            received = await _exchange(port, [
                Transcribe(language="xx").event(),
                AudioStart(rate=16000, width=2, channels=1).event(),
                AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1600).event(),
                AudioStop().event(),
            ])

        assert [(e.type, e.data.get("code")) for e in received] == [("error", "invalid_language")]
        mock_stt_router.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_connection_is_closed(self, mock_stt_router, mock_tts_router, monkeypatch):
        # The port has no authentication: a client that connects and goes quiet must not
        # hold a connection for good.
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_wyoming_idle_timeout_s", 0.2)

        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                assert await asyncio.wait_for(reader.read(), timeout=5) == b""
            finally:
                writer.close()

    @pytest.mark.asyncio
    async def test_idle_timeout_does_not_count_transcription_time(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        # Only waiting for the client's next event is idle; the client is waiting on us.
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_wyoming_idle_timeout_s", 0.2)

        def slow_transcription(**_kwargs):
            time.sleep(0.6)
            return {"text": "hello world"}

        mock_stt_router.transcribe.side_effect = slow_transcription

        async with _tcp_server(mock_stt_router, mock_tts_router) as port:
            received = await _exchange(port, [
                Transcribe().event(),
                AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1600).event(),
                AudioStop().event(),
            ])  # the server ends the connection once the client is idle again

        assert [e.type for e in received] == ["transcript"]

    @pytest.mark.asyncio
    async def test_connections_over_the_cap_are_closed_at_once_while_all_are_busy(
        self, mock_stt_router, mock_tts_router, monkeypatch, caplog,
    ):
        from wyoming.event import async_read_event, async_write_event

        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_wyoming_max_connections", 2)
        release = threading.Event()

        def busy_transcription(**_kwargs):
            release.wait(timeout=10)
            return {"text": "hello world"}

        mock_stt_router.transcribe.side_effect = busy_transcription

        async def transcribing() -> None:
            while mock_stt_router.transcribe.call_count < 2:
                await asyncio.sleep(0.01)

        async with _capped_tcp_server(mock_stt_router, mock_tts_router) as connect:
            try:
                busy = [await connect(), await connect()]
                for _reader, writer in busy:
                    for event in (
                        Transcribe().event(),
                        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x10\x00" * 1600).event(),
                        AudioStop().event(),
                    ):
                        await async_write_event(event, writer)
                await asyncio.wait_for(transcribing(), timeout=5)

                with caplog.at_level(logging.WARNING, logger="src.wyoming.server"):
                    for _ in range(3):  # one burst
                        reader, _writer = await connect()
                        assert await asyncio.wait_for(reader.read(), timeout=5) == b""
                warnings = [
                    r for r in caplog.records if r.name == "src.wyoming.server" and r.levelno == logging.WARNING
                ]
                assert len(warnings) == 1
                assert "OS_WYOMING_MAX_CONNECTIONS" in warnings[0].getMessage()

                release.set()
                for reader, _writer in busy:  # the requests in progress are unaffected
                    event = await asyncio.wait_for(async_read_event(reader), timeout=5)
                    assert Transcript.from_event(event).text == "hello world"
                reader, writer = await connect()
                assert await _described(reader, writer)
            finally:
                release.set()

    @pytest.mark.asyncio
    async def test_idle_connections_make_room_for_new_ones_at_the_cap(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        # The port has no authentication: a client that holds connections open, or keeps
        # them alive with an event now and then, must not lock Home Assistant out.
        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_wyoming_max_connections", 2)

        async with _capped_tcp_server(mock_stt_router, mock_tts_router) as connect:
            first, second = await connect(), await connect()
            assert await _described(*first) and await _described(*second)  # both then wait

            third = await connect()
            assert await _described(*third)
            assert await asyncio.wait_for(first[0].read(), timeout=5) == b""  # idle the longest
            assert await _described(*second) and await _described(*third)

    @pytest.mark.asyncio
    async def test_shutdown_closes_open_connections_instead_of_waiting_for_them(
        self, mock_stt_router, mock_tts_router, monkeypatch,
    ):
        # main.py cancels the server task and awaits it, and Docker kills the container 10 s
        # after asking it to stop. A connected client must not hold that up.
        from wyoming.event import async_read_event, async_write_event
        from wyoming.info import Info
        from wyoming.server import AsyncTcpServer

        from src.wyoming import server
        monkeypatch.setattr(server.settings, "os_wyoming_idle_timeout_s", 60)
        tcp_server = AsyncTcpServer("127.0.0.1", 0)
        task = asyncio.create_task(
            server._run_server(tcp_server, server.make_handler_factory(mock_stt_router, mock_tts_router, Info()))
        )
        writer = None

        async def listening() -> None:
            while tcp_server._server is None or not tcp_server._server.sockets:
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(listening(), timeout=5)
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", tcp_server._server.sockets[0].getsockname()[1]
            )
            await async_write_event(Describe().event(), writer)
            assert Info.is_type((await asyncio.wait_for(async_read_event(reader), timeout=5)).type)

            task.cancel()
            await asyncio.wait_for(task, timeout=3)
            assert await asyncio.wait_for(reader.read(), timeout=5) == b""
        finally:
            if writer is not None:
                writer.close()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ---------------------------------------------------------------------------
# A Whisper model's first use
# ---------------------------------------------------------------------------

# A Wyoming Transcribe naming a Whisper model that isn't downloaded fetched its files
# inside a thread of the STT pool (two threads, src/services/stt.py), so two unauthenticated voice commands naming an
# uncached repo held both threads for the whole download, and every transcription on the loaded STT_MODEL (HTTP and
# Home Assistant alike) waited, though the outsiders had hung up. A Whisper model's first
# download now holds up no other STT request, however it starts. Nothing is downloaded: the fake faster_whisper's
# download_model blocks until released, as a hub download does.
_LOADED_WHISPER = "deepdml/faster-whisper-large-v3-turbo-ct2"
_UNCACHED_WHISPER = "Systran/faster-whisper-medium"
_ONE_SECOND_PCM = b"\x00\x00" * 16000


def _whisper_segment(text: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        seek=0, start=0.0, end=1.0, text=text, tokens=[], temperature=0.0,
        avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
    )


@pytest.fixture
def first_use_router(tmp_path, monkeypatch):
    """A real BackendRouter with _LOADED_WHISPER loaded, over a fake faster_whisper whose download blocks.

    Yields the router, downloads (the fetches started), release (lets them end) and failure (a download raises its
    first item).
    """
    import sys
    from types import ModuleType, SimpleNamespace

    from src.config import settings
    from src.router import BackendRouter

    downloads: list[str] = []
    release = threading.Event()
    failure: list[Exception] = []

    def download_model(size_or_id, output_dir=None, local_files_only=False, cache_dir=None, **kwargs):
        downloads.append(size_or_id)
        release.wait(20)  # a multi-GB hub download
        if failure:
            raise failure[0]
        path = tmp_path / ("models--" + size_or_id.replace("/", "--"))
        path.mkdir(exist_ok=True)
        return str(path)

    class WhisperModel:
        def __init__(self, model_size_or_path, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            return iter([_whisper_segment(" first use")]), SimpleNamespace(language="en", duration=1.0)

    class LoadedWhisper:
        def transcribe(self, path, **kwargs):
            return iter([_whisper_segment(" hello")]), SimpleNamespace(language="en", duration=1.0)

    utils = ModuleType("faster_whisper.utils")
    utils.download_model = download_model
    module = ModuleType("faster_whisper")
    module.utils, module.WhisperModel = utils, WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))  # an empty cache
    monkeypatch.setattr(settings, "stt_vad_enabled", False)
    router = BackendRouter()
    backend = router.get_backend(_LOADED_WHISPER)
    backend._models[_LOADED_WHISPER] = LoadedWhisper()
    backend._loaded_at[_LOADED_WHISPER] = 0.0
    assert not router.is_model_loaded(_UNCACHED_WHISPER)
    try:
        yield SimpleNamespace(router=router, downloads=downloads, release=release, failure=failure)
    finally:
        release.set()


def _voice_command(model: str) -> list[Event]:
    return [
        Transcribe(name=model).event(),
        AudioStart(rate=16000, width=2, channels=1).event(),
        AudioChunk(rate=16000, width=2, channels=1, audio=_ONE_SECOND_PCM).event(),
        AudioStop().event(),
    ]


class TestFirstUseFetch:
    @pytest.mark.asyncio
    async def test_outsiders_naming_an_uncached_model_hold_up_no_transcription_on_the_loaded_one(
        self, first_use_router, mock_tts_router,
    ):
        # Two outsiders send a voice command naming an uncached Whisper repo and hang
        # up. While its download runs, a Home Assistant command and an HTTP decode (every HTTP route decodes in the
        # same STT pool) on the loaded model are answered at once.
        from wyoming.event import async_read_event, async_write_event

        from src.services.stt import run_stt_inference
        from src.wyoming.stt_handler import _pcm_to_wav

        router, downloads = first_use_router.router, first_use_router.downloads
        async with _capped_tcp_server(router, mock_tts_router) as connect:
            for _ in range(2):
                _reader, writer = await connect()
                for event in _voice_command(_UNCACHED_WHISPER):
                    await async_write_event(event, writer)
                writer.close()
            for _ in range(500):
                if downloads:
                    break
                await asyncio.sleep(0.01)
            assert downloads == [_UNCACHED_WHISPER], f"the outsiders' commands never reached the fetch: {downloads}"
            await asyncio.sleep(0.5)  # both outsider commands reach the fetch

            reader, writer = await connect()
            for event in _voice_command(_LOADED_WHISPER):
                await async_write_event(event, writer)
            home_assistant = asyncio.ensure_future(async_read_event(reader))
            http = asyncio.ensure_future(run_stt_inference(
                router.transcribe, audio=_pcm_to_wav(_ONE_SECOND_PCM, 16000, 2, 1), model=_LOADED_WHISPER,
            ))
            done, _ = await asyncio.wait([home_assistant, http], timeout=5)
            answered = {"Home Assistant": home_assistant in done, "HTTP": http in done}
            first_use_router.release.set()
            ha_event, http_result = await asyncio.wait_for(asyncio.gather(home_assistant, http), timeout=15)

        assert answered == {"Home Assistant": True, "HTTP": True}, (
            f"with two Wyoming commands naming the uncached {_UNCACHED_WHISPER} downloading, requests on the loaded "
            f"{_LOADED_WHISPER} were still waiting after 5 s ({answered}): the Wyoming fetch holds the STT pool's "
            "threads"
        )
        assert Transcript.from_event(ha_event).text == "hello"
        assert http_result["text"] == "hello"
        assert downloads == [_UNCACHED_WHISPER]  # the outsiders shared one download

    @pytest.mark.asyncio
    async def test_a_failed_first_download_is_answered_with_the_error_event(self, first_use_router, mock_tts_router):
        # The fetch now runs before the decode, outside the STT pool; its failure
        # still reaches the client as the error event a failed transcription gets, and the connection ends.
        first_use_router.failure.append(OSError("the hub could not be reached"))
        first_use_router.release.set()

        async with _tcp_server(first_use_router.router, mock_tts_router) as port:
            received = await _exchange(port, _voice_command(_UNCACHED_WHISPER))

        assert [e.type for e in received] == ["error"]
        error = Error.from_event(received[0])
        assert error.code == "OSError"
        assert error.text == "STT failed: the hub could not be reached"
        assert first_use_router.downloads == [_UNCACHED_WHISPER]

    @pytest.mark.asyncio
    async def test_a_failed_first_download_names_no_folders(self, first_use_router, mock_tts_router):
        first_use_router.failure.append(OSError("No file /root/.cache/huggingface/hub/models--x/snapshots/a/model.bin"))
        first_use_router.release.set()

        async with _tcp_server(first_use_router.router, mock_tts_router) as port:
            received = await _exchange(port, _voice_command(_UNCACHED_WHISPER))

        assert Error.from_event(received[0]).text == "STT failed: No file model.bin"

    @pytest.mark.asyncio
    async def test_a_model_id_the_fetch_refuses_is_answered_with_the_error_event(
        self, first_use_router, mock_tts_router, monkeypatch,
    ):
        # Whatever the fetch step raises for a model id it refuses reaches the
        # client as the error event, not a dropped connection, and nothing is fetched or decoded.
        from src.wyoming import stt_handler

        async def refuse(backend_router, model):
            raise ValueError(f"model id {model!r} is not allowed")

        monkeypatch.setattr(stt_handler, "await_stt_fetch", refuse)
        first_use_router.release.set()

        async with _tcp_server(first_use_router.router, mock_tts_router) as port:
            received = await _exchange(port, _voice_command("//attacker/share/model"))

        assert [(e.type, e.data.get("code")) for e in received] == [("error", "ValueError")]
        assert "is not allowed" in Error.from_event(received[0]).text
        assert first_use_router.downloads == []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_wyoming_disabled_by_default(self):
        from src.config import Settings
        with patch.dict(os.environ, {}, clear=True):
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

    def test_wyoming_connection_limits(self, monkeypatch):
        # Home Assistant opens one short-lived connection per describe, transcribe or synthesize.
        from src.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            defaults = Settings()
        assert (defaults.os_wyoming_idle_timeout_s, defaults.os_wyoming_max_connections) == (120, 32)
        monkeypatch.setenv("OS_WYOMING_IDLE_TIMEOUT_S", "0")
        monkeypatch.setenv("OS_WYOMING_MAX_CONNECTIONS", "0")
        unlimited = Settings()
        assert (unlimited.os_wyoming_idle_timeout_s, unlimited.os_wyoming_max_connections) == (0, 0)


# ---------------------------------------------------------------------------
# Resample utility
# ---------------------------------------------------------------------------


def _pcm_samples(pieces: list[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(pieces), dtype=np.int16)


class TestResample:
    def test_resample_24k_to_16k(self):
        from src.wyoming.tts_handler import _to_wyoming_pcm
        audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)).astype(np.float32)
        assert len(_pcm_samples(_to_wyoming_pcm([audio], 24000))) == 16000

    def test_resample_noop_at_16k(self):
        from src.tts.pipeline import float32_to_int16
        from src.wyoming.tts_handler import _to_wyoming_pcm
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        np.testing.assert_array_equal(_pcm_samples(_to_wyoming_pcm([audio], 16000)), float32_to_int16(audio))

    def test_chunk_boundaries_leave_no_seams(self):
        # One filter runs across the chunks, so chunked output matches resampling the
        # whole utterance at once.
        from src.wyoming.tts_handler import _to_wyoming_pcm
        audio = (0.3 * np.random.default_rng(0).standard_normal(24000)).astype(np.float32)
        whole = _pcm_samples(_to_wyoming_pcm([audio], 24000)).astype(np.int32)
        chunked = _pcm_samples(_to_wyoming_pcm(np.array_split(audio, 7), 24000)).astype(np.int32)
        assert len(chunked) == len(whole) == 16000
        assert np.max(np.abs(chunked - whole)) <= 1

    @pytest.mark.asyncio
    async def test_content_above_8khz_is_filtered_out_not_aliased(self, mock_tts_router, monkeypatch):
        # Nyquist: 16 kHz audio cannot carry a 10 kHz tone (sibilant energy). Linear
        # interpolation folded it back as a 6 kHz tone at about 60% of its level.
        from src.wyoming import tts_handler
        monkeypatch.setattr(tts_handler, "settings", _make_mock_settings())
        tone = (0.5 * np.sin(2 * np.pi * 10_000 * np.arange(24000) / 24000)).astype(np.float32)
        mock_tts_router.synthesize.return_value = iter([tone])
        events: list[Event] = []

        async def capture(event):
            events.append(event)

        await tts_handler.handle_synthesize(
            text="hiss", voice=None, tts_router=mock_tts_router, write_event=capture,
        )

        pcm = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
        output = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767
        assert output.size == 16000
        tone_rms = float(np.sqrt(np.mean(tone ** 2)))
        assert float(np.sqrt(np.mean(output ** 2))) < 0.02 * tone_rms


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
