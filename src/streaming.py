"""Real-time streaming transcription via WebSocket.

Protocol (Deepgram-compatible):
  Client → Server: binary PCM16 LE mono frames, or JSON text messages:
    {"type":"stop"}           — end session gracefully
    {"type":"config", ...}    — update session config (e.g. sample_rate)
  Server → Client: JSON text frames with transcript events:
    {"type":"session.begin"}  — session started
    {"type":"transcript"}     — transcription result (is_final, speech_final)
    {"type":"error"}          — transcription error
    {"type":"session.end"}    — session ended

The client's actual audio sample rate is sent via the `sample_rate` query parameter.
The server resamples to 16kHz internally for VAD and model inference.
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

# Internal processing sample rate — VAD and whisper models expect 16kHz
INTERNAL_SAMPLE_RATE = 16000

# Max utterance duration in seconds — force-finalize if exceeded to prevent
# unbounded memory growth and quadratic transcription time
MAX_UTTERANCE_SECONDS = 30
MAX_UTTERANCE_BYTES = MAX_UTTERANCE_SECONDS * INTERNAL_SAMPLE_RATE * 2  # PCM16

# Valid sample rate range
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000

# Dedicated thread pool for streaming transcription to avoid starving REST API
import concurrent.futures
_streaming_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="stream-transcribe"
)

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
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: urllib.request.urlretrieve(SILERO_ONNX_URL, str(model_path))
            )
            logger.info("Silero VAD model downloaded to %s", model_path)

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _vad_model = SileroVAD(sess)
        logger.info("Silero VAD model loaded")
        return _vad_model


def resample_pcm16(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM16 LE mono audio from one sample rate to another.

    Uses linear interpolation — adequate for speech at nearby rates.
    Returns PCM16 LE bytes at the target rate.
    """
    if from_rate == to_rate:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm_bytes

    # Calculate output length
    ratio = to_rate / from_rate
    out_len = int(len(samples) * ratio)
    if out_len == 0:
        return b""

    # Linear interpolation
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, out_len)
    resampled = np.interp(x_new, x_old, samples)

    return resampled.astype(np.int16).tobytes()


class SileroVAD:
    """Wrapper around the Silero VAD ONNX model."""

    def __init__(self, session):
        self.session = session
        self.sample_rate = INTERNAL_SAMPLE_RATE  # VAD expects 16kHz
        # Internal state tensor: shape [2, 1, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> float:
        """Run VAD on audio chunk. Returns speech probability 0-1.

        Audio MUST be float32, mono, 16kHz, shape (N,) where N is
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
            sr = np.array(self.sample_rate, dtype=np.int64)

            ort_inputs = {
                "input": input_data,
                "state": self._state,
                "sr": sr,
            }
            out, self._state = self.session.run(None, ort_inputs)
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
        self.client_sample_rate = sample_rate  # What the client actually sends
        self.needs_resample = sample_rate != INTERNAL_SAMPLE_RATE
        self.interim_results = interim_results
        self.endpointing_ms = endpointing_ms

        self.audio_buffer = bytearray()
        # Chunk size in CLIENT sample rate (what we receive)
        self.chunk_samples = int(sample_rate * settings.stt_stream_chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * 2  # 16-bit = 2 bytes per sample

        self.agreement = LocalAgreement2()
        self.vad: SileroVAD | None = None

        self.utterance_start = 0.0
        self.total_samples = 0  # In client sample rate
        self.silence_samples = 0  # In internal (16kHz) sample rate
        self.endpointing_samples = int(INTERNAL_SAMPLE_RATE * endpointing_ms / 1000)
        self.speech_active = False
        self.utterance_audio = bytearray()  # Always stored at INTERNAL_SAMPLE_RATE

        self._running = False
        self._transcription_count = 0
        self._error_count = 0

    async def run(self):
        """Main session loop."""
        self._running = True

        # Pre-validate model is loaded
        loop = asyncio.get_running_loop()
        try:
            if not backend_router.is_model_loaded(self.model):
                logger.info("[%s] Model %s not loaded — loading now...", self.session_id[:8], self.model)
                await loop.run_in_executor(None, lambda: backend_router.load_model(self.model))
                logger.info("[%s] Model %s loaded successfully", self.session_id[:8], self.model)
        except Exception as e:
            logger.error("[%s] Failed to load model %s: %s", self.session_id[:8], self.model, e)
            await self._send_event({
                "type": "error",
                "message": f"Failed to load model: {e}",
            })
            return

        self.vad = await _get_vad_model()
        # Each session gets its own VAD state
        self.vad_state = SileroVAD(self.vad.session)

        if self.needs_resample:
            logger.info(
                "[%s] Client sample rate %d Hz — will resample to %d Hz",
                self.session_id[:8], self.client_sample_rate, INTERNAL_SAMPLE_RATE,
            )

        await self._send_event({
            "type": "session.begin",
            "session_id": self.session_id,
            "model": self.model,
            "sample_rate": self.client_sample_rate,
            "internal_sample_rate": INTERNAL_SAMPLE_RATE,
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
            logger.exception("[%s] Streaming session error: %s", self.session_id[:8], e)
        finally:
            # Flush remaining audio
            await self._flush()
            await self._send_event({
                "type": "session.end",
                "reason": "client_stop" if not self._running else "disconnect",
                "transcriptions": self._transcription_count,
                "errors": self._error_count,
            })

    async def _handle_text(self, text: str):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[%s] Received malformed JSON text frame", self.session_id[:8])
            return
        if data.get("type") == "stop":
            self._running = False

    async def _handle_audio(self, data: bytes):
        # Drop trailing odd byte — PCM16 requires even-length frames
        if len(data) % 2 != 0:
            data = data[:-1]
        if not data:
            return
        self.audio_buffer.extend(data)
        self.total_samples += len(data) // 2
        logger.debug("[%s] Audio recv %d bytes, buffer %d/%d",
                     self.session_id[:8], len(data), len(self.audio_buffer), self.chunk_bytes)

        # Process complete chunks
        while len(self.audio_buffer) >= self.chunk_bytes:
            chunk = bytes(self.audio_buffer[:self.chunk_bytes])
            del self.audio_buffer[:self.chunk_bytes]
            await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: bytes):
        """Process one audio chunk through VAD + transcription.

        Input chunk is at client_sample_rate. We resample to INTERNAL_SAMPLE_RATE
        (16kHz) for VAD and transcription.
        """
        # Resample to 16kHz if needed
        if self.needs_resample:
            chunk_16k = resample_pcm16(chunk, self.client_sample_rate, INTERNAL_SAMPLE_RATE)
        else:
            chunk_16k = chunk

        # Convert to float32 for VAD (at 16kHz)
        samples = np.frombuffer(chunk_16k, dtype=np.int16).astype(np.float32) / 32768.0

        # Run VAD
        speech_prob = self.vad_state(samples)
        is_speech = speech_prob >= settings.stt_stream_vad_threshold
        logger.debug("[%s] VAD prob=%.3f speech=%s active=%s utterance=%d bytes",
                     self.session_id[:8], speech_prob, is_speech, self.speech_active,
                     len(self.utterance_audio))

        if is_speech:
            self.silence_samples = 0
            if not self.speech_active:
                self.speech_active = True
                self.utterance_start = (self.total_samples - len(chunk) // 2) / self.client_sample_rate
                self.utterance_audio = bytearray()
                self.agreement.reset()
                logger.info("[%s] Speech started at %.2fs", self.session_id[:8], self.utterance_start)

            self.utterance_audio.extend(chunk_16k)  # Store at 16kHz

            # Force-finalize if utterance exceeds max length (prevents memory bomb)
            if len(self.utterance_audio) >= MAX_UTTERANCE_BYTES:
                logger.info("[%s] Utterance exceeded %ds max, force-finalizing",
                           self.session_id[:8], MAX_UTTERANCE_SECONDS)
                await self._finalize_utterance()
            else:
                await self._transcribe_utterance()
        else:
            if self.speech_active:
                # Count silence in 16kHz samples
                self.silence_samples += len(chunk_16k) // 2
                self.utterance_audio.extend(chunk_16k)  # Include trailing audio at 16kHz

                if self.silence_samples >= self.endpointing_samples:
                    # Utterance ended — finalize
                    logger.info("[%s] Speech ended (%.0fms silence), finalizing",
                               self.session_id[:8], self.silence_samples / INTERNAL_SAMPLE_RATE * 1000)
                    await self._finalize_utterance()
                else:
                    # Still within endpointing window, transcribe what we have
                    await self._transcribe_utterance()

    async def _transcribe_utterance(self):
        """Transcribe current utterance audio and emit interim/confirmed results.

        utterance_audio is always at INTERNAL_SAMPLE_RATE (16kHz).
        """
        if len(self.utterance_audio) < 3200:  # < 0.1s at 16kHz, skip
            return

        # Build WAV in memory (audio is at 16kHz)
        wav_data = self._pcm_to_wav(bytes(self.utterance_audio), INTERNAL_SAMPLE_RATE)

        # Run transcription in executor (it's CPU/GPU bound)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                _streaming_executor,
                lambda: backend_router.transcribe(
                    audio=wav_data,
                    model=self.model,
                    language=self.language,
                    response_format="json",
                    temperature=0.0,
                ),
            )
            self._transcription_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error("[%s] Transcription error (#%d): %s",
                        self.session_id[:8], self._error_count, e, exc_info=True)
            await self._send_event({
                "type": "error",
                "message": f"Transcription failed: {e}",
            })
            return

        text = result.get("text", "").strip()
        logger.info("[%s] Transcription #%d: '%s' (%.1fs audio)",
                    self.session_id[:8], self._transcription_count, text,
                    len(self.utterance_audio) / 2 / INTERNAL_SAMPLE_RATE)
        if not text:
            return

        new_confirmed, pending = self.agreement.process(text)
        now = self.total_samples / self.client_sample_rate

        # Emit confirmed words
        if new_confirmed:
            confirmed_text = " ".join(self.agreement.confirmed_words)
            logger.info("[%s] Confirmed: '%s'", self.session_id[:8], confirmed_text)
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
        """Finalize the current utterance. utterance_audio is at 16kHz."""
        if len(self.utterance_audio) < 3200:
            self.speech_active = False
            self.silence_samples = 0
            return

        # One final transcription (audio is at 16kHz)
        wav_data = self._pcm_to_wav(bytes(self.utterance_audio), INTERNAL_SAMPLE_RATE)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                _streaming_executor,
                lambda: backend_router.transcribe(
                    audio=wav_data,
                    model=self.model,
                    language=self.language,
                    response_format="json",
                    temperature=0.0,
                ),
            )
            self._transcription_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error("[%s] Final transcription error (#%d): %s",
                        self.session_id[:8], self._error_count, e, exc_info=True)
            self.speech_active = False
            self.silence_samples = 0
            return

        text = result.get("text", "").strip()
        now = self.total_samples / self.client_sample_rate

        if text:
            logger.info("[%s] Final utterance: '%s' (%.1fs audio)",
                       self.session_id[:8], text,
                       len(self.utterance_audio) / 2 / INTERNAL_SAMPLE_RATE)
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
                # Resample remaining to 16kHz before appending
                if self.needs_resample:
                    remaining = resample_pcm16(remaining, self.client_sample_rate, INTERNAL_SAMPLE_RATE)
                self.utterance_audio.extend(remaining)
                await self._finalize_utterance()

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
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
            sample_rate,
            sample_rate * num_channels * sample_width,
            num_channels * sample_width,
            sample_width * 8,
            b"data",
            data_size,
        )
        return header + pcm

    async def _send_event(self, event: dict):
        try:
            text = json.dumps(event)
            await self.ws.send_text(text)
            logger.debug("[%s] Sent: %s", self.session_id[:8], event.get("type", "?"))
        except Exception as e:
            logger.warning("[%s] Failed to send event %s: %s",
                          self.session_id[:8], event.get("type", "?"), e)


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

    # Validate sample rate
    if sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        await ws.close(code=1008, reason=f"Invalid sample_rate: must be {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}")
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
        logger.info(
            "Streaming session %s started (model=%s, client_rate=%d, resample=%s)",
            session.session_id, session.model, sample_rate, session.needs_resample,
        )
        await session.run()
    finally:
        _active_sessions.pop(session.session_id, None)
        logger.info(
            "Streaming session %s ended (transcriptions=%d, errors=%d)",
            session.session_id, session._transcription_count, session._error_count,
        )
