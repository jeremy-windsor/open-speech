"""Real-time streaming transcription via WebSocket.

Protocol (Deepgram-compatible):
  Client → Server: binary PCM16 LE mono 16kHz frames, or JSON text {"type":"stop"}
  Server → Client: JSON text frames with transcript events
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect, Query

from src.config import settings
from src.router import router as backend_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silero VAD (ONNX)
# ---------------------------------------------------------------------------

_vad_model = None
_vad_lock = asyncio.Lock()

SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SILERO_CACHE_DIR = Path.home() / ".cache" / "silero-vad"


async def _get_vad_model():
    """Lazy-load Silero VAD ONNX model."""
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    async with _vad_lock:
        if _vad_model is not None:
            return _vad_model

        import onnxruntime as ort

        model_path = SILERO_CACHE_DIR / "silero_vad.onnx"
        if not model_path.exists():
            logger.info("Downloading Silero VAD model...")
            SILERO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Download in executor to not block event loop
            import urllib.request
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: urllib.request.urlretrieve(SILERO_ONNX_URL, str(model_path))
            )
            logger.info("Silero VAD model downloaded to %s", model_path)

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _vad_model = SileroVAD(sess)
        logger.info("Silero VAD model loaded")
        return _vad_model


class SileroVAD:
    """Wrapper around the Silero VAD ONNX model."""

    def __init__(self, session):
        self.session = session
        self.sample_rate = 16000
        # Internal state tensors
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> float:
        """Run VAD on audio chunk. Returns speech probability 0-1.

        Audio should be float32, mono, 16kHz, shape (N,) where N is
        a multiple of 512 samples (32ms at 16kHz). For best results
        use 512 or 1536 sample windows.
        """
        if len(audio) == 0:
            return 0.0

        # Process in 512-sample windows and return max probability
        window_size = 512
        max_prob = 0.0

        for start in range(0, len(audio) - window_size + 1, window_size):
            chunk = audio[start:start + window_size]
            input_data = chunk.reshape(1, -1).astype(np.float32)
            sr = np.array([self.sample_rate], dtype=np.int64)

            ort_inputs = {
                "input": input_data,
                "h": self._h,
                "c": self._c,
                "sr": sr,
            }
            out, self._h, self._c = self.session.run(None, ort_inputs)
            prob = float(out[0][0])
            if prob > max_prob:
                max_prob = prob

        return max_prob


# ---------------------------------------------------------------------------
# LocalAgreement2 — stable partial emission
# ---------------------------------------------------------------------------

class LocalAgreement2:
    """Compare consecutive transcriptions and emit only stable (agreed) words.

    When two consecutive transcription results share the same prefix words,
    those words are considered "confirmed" and emitted as final.
    """

    def __init__(self):
        self.previous_words: list[str] = []
        self.confirmed_words: list[str] = []

    def process(self, current_text: str) -> tuple[list[str], list[str]]:
        """Process a new transcription result.

        Returns (new_confirmed, pending):
          - new_confirmed: newly confirmed words (stable, won't change)
          - pending: words that may still change
        """
        current_words = current_text.split() if current_text.strip() else []

        # Find longest common prefix between previous and current
        common_len = 0
        min_len = min(len(self.previous_words), len(current_words))
        for i in range(min_len):
            if self.previous_words[i].lower() == current_words[i].lower():
                common_len = i + 1
            else:
                break

        # Words in common prefix that weren't already confirmed are new confirmed
        already_confirmed = len(self.confirmed_words)
        new_confirmed = []
        if common_len > already_confirmed:
            new_confirmed = current_words[already_confirmed:common_len]
            self.confirmed_words = current_words[:common_len]

        # Pending = words after confirmed portion
        pending = current_words[len(self.confirmed_words):]

        self.previous_words = current_words
        return new_confirmed, pending

    def flush(self) -> list[str]:
        """Flush remaining unconfirmed words as final."""
        remaining = self.previous_words[len(self.confirmed_words):]
        self.confirmed_words.extend(remaining)
        return remaining

    def reset(self):
        self.previous_words = []
        self.confirmed_words = []


# ---------------------------------------------------------------------------
# Streaming session
# ---------------------------------------------------------------------------

_active_sessions: dict[str, "StreamingSession"] = {}


class StreamingSession:
    """Manages one WebSocket streaming transcription session."""

    def __init__(
        self,
        ws: WebSocket,
        model: str,
        language: str | None,
        sample_rate: int,
        interim_results: bool,
        endpointing_ms: int,
    ):
        self.ws = ws
        self.session_id = str(uuid.uuid4())
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        self.endpointing_ms = endpointing_ms

        self.audio_buffer = bytearray()
        self.chunk_samples = int(sample_rate * settings.stt_stream_chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * 2  # 16-bit = 2 bytes per sample

        self.agreement = LocalAgreement2()
        self.vad: SileroVAD | None = None

        self.utterance_start = 0.0
        self.total_samples = 0
        self.silence_samples = 0
        self.endpointing_samples = int(sample_rate * endpointing_ms / 1000)
        self.speech_active = False
        self.utterance_audio = bytearray()

        self._running = False

    async def run(self):
        """Main session loop."""
        self._running = True
        self.vad = await _get_vad_model()
        # Each session gets its own VAD state
        self.vad_state = SileroVAD(self.vad.session)

        await self._send_event({
            "type": "session.begin",
            "session_id": self.session_id,
            "model": self.model,
            "sample_rate": self.sample_rate,
        })

        try:
            while self._running:
                try:
                    msg = await self.ws.receive()
                except WebSocketDisconnect:
                    break

                if msg["type"] == "websocket.disconnect":
                    break

                if msg["type"] == "websocket.receive":
                    if "bytes" in msg and msg["bytes"]:
                        await self._handle_audio(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await self._handle_text(msg["text"])
        except Exception as e:
            logger.exception("Streaming session error: %s", e)
        finally:
            # Flush remaining audio
            await self._flush()
            await self._send_event({
                "type": "session.end",
                "reason": "client_stop" if not self._running else "disconnect",
            })

    async def _handle_text(self, text: str):
        data = json.loads(text)
        if data.get("type") == "stop":
            self._running = False

    async def _handle_audio(self, data: bytes):
        self.audio_buffer.extend(data)
        self.total_samples += len(data) // 2

        # Process complete chunks
        while len(self.audio_buffer) >= self.chunk_bytes:
            chunk = bytes(self.audio_buffer[:self.chunk_bytes])
            del self.audio_buffer[:self.chunk_bytes]
            await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: bytes):
        """Process one audio chunk through VAD + transcription."""
        # Convert to float32 for VAD
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # Run VAD
        speech_prob = self.vad_state(samples)
        is_speech = speech_prob >= settings.stt_stream_vad_threshold

        if is_speech:
            self.silence_samples = 0
            if not self.speech_active:
                self.speech_active = True
                self.utterance_start = (self.total_samples - len(chunk) // 2) / self.sample_rate
                self.utterance_audio = bytearray()
                self.agreement.reset()

            self.utterance_audio.extend(chunk)
            await self._transcribe_utterance()
        else:
            if self.speech_active:
                self.silence_samples += len(chunk) // 2
                self.utterance_audio.extend(chunk)  # Include trailing audio

                if self.silence_samples >= self.endpointing_samples:
                    # Utterance ended — finalize
                    await self._finalize_utterance()
                else:
                    # Still within endpointing window, transcribe what we have
                    await self._transcribe_utterance()

    async def _transcribe_utterance(self):
        """Transcribe current utterance audio and emit interim/confirmed results."""
        if len(self.utterance_audio) < 3200:  # < 0.1s, skip
            return

        # Build WAV in memory
        wav_data = self._pcm_to_wav(bytes(self.utterance_audio))

        # Run transcription in executor (it's CPU/GPU bound)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: backend_router.transcribe(
                    audio=wav_data,
                    model=self.model,
                    language=self.language,
                    response_format="json",
                    temperature=0.0,
                ),
            )
        except Exception as e:
            logger.warning("Transcription error: %s", e)
            return

        text = result.get("text", "").strip()
        if not text:
            return

        new_confirmed, pending = self.agreement.process(text)
        now = self.total_samples / self.sample_rate

        # Emit confirmed words
        if new_confirmed:
            confirmed_text = " ".join(self.agreement.confirmed_words)
            await self._send_event({
                "type": "transcript",
                "is_final": True,
                "speech_final": False,
                "text": confirmed_text,
                "start": self.utterance_start,
                "end": now,
                "confidence": 0.95,
            })

        # Emit interim result with pending words
        if self.interim_results and pending:
            full_text = " ".join(self.agreement.confirmed_words + pending)
            await self._send_event({
                "type": "transcript",
                "is_final": False,
                "speech_final": False,
                "text": full_text,
                "start": self.utterance_start,
                "end": now,
                "confidence": 0.90,
            })

    async def _finalize_utterance(self):
        """Finalize the current utterance."""
        if len(self.utterance_audio) < 3200:
            self.speech_active = False
            self.silence_samples = 0
            return

        # One final transcription
        wav_data = self._pcm_to_wav(bytes(self.utterance_audio))
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: backend_router.transcribe(
                    audio=wav_data,
                    model=self.model,
                    language=self.language,
                    response_format="json",
                    temperature=0.0,
                ),
            )
        except Exception as e:
            logger.warning("Final transcription error: %s", e)
            self.speech_active = False
            self.silence_samples = 0
            return

        text = result.get("text", "").strip()
        now = self.total_samples / self.sample_rate

        if text:
            await self._send_event({
                "type": "transcript",
                "is_final": True,
                "speech_final": True,
                "text": text,
                "start": self.utterance_start,
                "end": now,
                "confidence": 0.95,
            })

        # Reset for next utterance
        self.speech_active = False
        self.silence_samples = 0
        self.utterance_audio = bytearray()
        self.agreement.reset()

    async def _flush(self):
        """Flush any remaining audio buffer."""
        if self.audio_buffer:
            remaining = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            if self.speech_active and len(self.utterance_audio) > 0:
                self.utterance_audio.extend(remaining)
                await self._finalize_utterance()

    def _pcm_to_wav(self, pcm: bytes) -> bytes:
        """Wrap raw PCM16 data in a WAV header."""
        num_channels = 1
        sample_width = 2
        data_size = len(pcm)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,  # chunk size
            1,   # PCM format
            num_channels,
            self.sample_rate,
            self.sample_rate * num_channels * sample_width,
            num_channels * sample_width,
            sample_width * 8,
            b"data",
            data_size,
        )
        return header + pcm

    async def _send_event(self, event: dict):
        try:
            await self.ws.send_text(json.dumps(event))
        except Exception:
            pass  # Connection may be closed


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

async def streaming_endpoint(
    ws: WebSocket,
    model: str | None = None,
    language: str | None = None,
    sample_rate: int = 16000,
    encoding: str = "pcm_s16le",
    interim_results: bool = True,
    endpointing: int = 300,
):
    """WebSocket endpoint for real-time streaming transcription."""
    # Check connection limit
    if len(_active_sessions) >= settings.stt_stream_max_connections:
        await ws.close(code=1013, reason="Too many concurrent streams")
        return

    await ws.accept()

    session = StreamingSession(
        ws=ws,
        model=model or settings.stt_default_model,
        language=language,
        sample_rate=sample_rate,
        interim_results=interim_results,
        endpointing_ms=endpointing,
    )

    _active_sessions[session.session_id] = session
    try:
        logger.info("Streaming session %s started (model=%s)", session.session_id, session.model)
        await session.run()
    finally:
        _active_sessions.pop(session.session_id, None)
        logger.info("Streaming session %s ended", session.session_id)
