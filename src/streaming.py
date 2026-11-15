"""Real-time streaming transcription via WebSocket.

Protocol (Deepgram-compatible):
  Client → Server: binary PCM16 LE mono frames, or JSON text messages:
    {"type":"stop"}           — end session gracefully
    {"type":"config", ...}    — update session config (e.g. sample_rate)
  Server → Client: JSON text frames with transcript events:
    {"type":"session.begin"}  — session started
    {"type":"transcript"}     — transcription result (is_final, speech_final)
    {"type":"vad"}            — VAD state change (speech_start / speech_end)
    {"type":"error"}          — error; one before session.begin means the stream did not start
    {"type":"session.end"}    — session ended
  A stream the server ends is then closed with 1000, or 1011 after a failure (an error before session.begin).

The client's actual audio sample rate is sent via the `sample_rate` query parameter.
The server resamples to 16kHz internally for VAD and model inference. The
`encoding` query parameter must be `pcm_s16le` (or Deepgram's `linear16`).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import struct
import threading
import uuid
from collections import deque

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from src.audio.levels import InputLevels
from src.audio.resample import StreamingResampler, resample
from src.backends.parakeet import STREAM_MAX_UTTERANCE_S as PARAKEET_MAX_UTTERANCE_SECONDS
from src.backends.parakeet import check_language, is_parakeet_model
from src.config import settings
from src.middleware import WS_CLOSE_ORIGIN, refuse_websocket
from src.openai_compat import resolve_stt_model
from src.router import router as backend_router
from src.services.stt import whisper_language_code
from src.utils.redact import redact_paths
from src.vad.silero import SileroVAD, get_vad_model, VAD_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Internal processing sample rate — VAD and whisper models expect 16kHz
INTERNAL_SAMPLE_RATE = VAD_SAMPLE_RATE
VAD_WINDOW_SAMPLES = 512
VAD_WINDOW_BYTES = VAD_WINDOW_SAMPLES * 2
# Speech keeps this much audio from before its first window over STT_VAD_THRESHOLD, here and in
# Wyoming's trimming: Silero scores a word's onset below the threshold while it is already audible.
# The same as /v1/realtime's default prefix_padding_ms.
VAD_PREROLL_MS = 300

# Max utterance duration in seconds — force-finalize if exceeded to prevent
# unbounded memory growth and quadratic transcription time. Parakeet's is shorter
# (PARAKEET_MAX_UTTERANCE_SECONDS, 24 s): its positional encoding and GPU arena are sized
# for pieces of up to 25 s, and an utterance is cut a VAD window or a chunk past the limit.
MAX_UTTERANCE_SECONDS = 30
MAX_UTTERANCE_BYTES = MAX_UTTERANCE_SECONDS * INTERNAL_SAMPLE_RATE * 2  # PCM16


def max_utterance_seconds(model: str) -> int:
    """The length at which a stream's utterance is finalized even while the speaker goes on."""
    return PARAKEET_MAX_UTTERANCE_SECONDS if is_parakeet_model(model) else MAX_UTTERANCE_SECONDS

# Valid sample rate range
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000

# Binary frames carry PCM16 LE mono; "linear16" is Deepgram's name for it
SUPPORTED_ENCODINGS = ("pcm_s16le", "linear16")

# Dedicated thread pool for streaming transcription to avoid starving REST API
_streaming_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="stream-transcribe"
)
# Silero runs off the event loop, in its own pool: one large frame is thousands of
# windows, and the transcription pool can be busy for seconds with other decodes.
_vad_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="stream-vad"
)
# A stream's model load waits for faster-whisper's inference lock, held by every decode and load. It waits here,
# never in asyncio's shared default executor, as transcriptions wait in their own pool.
_stream_load_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="stream-load"
)


def _score_vad_windows(vad_state, samples: np.ndarray, lock: threading.Lock) -> list[float]:
    """Silero's speech probability for each 512-sample window of ``samples`` (blocking).

    The lock keeps a session's windows in order, since the VAD carries state between them.
    """
    with lock:
        return [
            vad_state(samples[offset:offset + VAD_WINDOW_SAMPLES])
            for offset in range(0, len(samples), VAD_WINDOW_SAMPLES)
        ]


def resample_pcm16(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample a whole clip of PCM16 LE mono audio to another sample rate.

    Uses an anti-aliased polyphase FIR (src.audio.resample) to avoid
    downsampling aliasing that can degrade VAD quality. A stream of chunks
    needs a StreamingResampler instead, so the chunks join without seams.
    """
    if from_rate == to_rate:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm_bytes

    if len(samples) == 1:
        out_len = int(len(samples) * (to_rate / from_rate))
        if out_len <= 0:
            return b""
        return np.full(out_len, samples[0], dtype=np.int16).tobytes()

    resampled = resample(samples, from_rate, to_rate, padtype="line")
    resampled = np.clip(resampled, -32768, 32767)
    return resampled.astype(np.int16).tobytes()


def _to_pcm16(samples: np.ndarray) -> bytes:
    return np.clip(np.rint(samples), -32768, 32767).astype(np.int16).tobytes()


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

        already_confirmed = len(self.confirmed_words)
        new_confirmed = []
        if common_len > already_confirmed:
            new_confirmed = current_words[already_confirmed:common_len]
            self.confirmed_words = current_words[:common_len]

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
_active_sessions_lock = asyncio.Lock()

# The close code and reason for a stream the server ends itself: without a Close frame the client sees only a dropped
# connection (1006, RFC 6455 7.1.5). 1000 is a normal end; 1011 a server failure, after an error event that names it.
CLOSE_ENDED = (1000, "Stream ended")
CLOSE_MODEL_FAILED = (1011, "Failed to load model")
CLOSE_VAD_FAILED = (1011, "Failed to load VAD model")
CLOSE_FAILED = (1011, "Streaming session failed")


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
        vad_enabled: bool = True,
        language_error: str | None = None,
    ):
        self.ws = ws
        self.session_id = str(uuid.uuid4())
        self.model = model
        self.language = language
        self.interim_results = interim_results
        self.endpointing_ms = endpointing_ms
        self.vad_enabled = vad_enabled
        # Why the language the client asked for was not used. It is sent after session.begin,
        # since an error before it means the stream failed to start.
        self.language_error = language_error
        self.max_utterance_seconds = max_utterance_seconds(model)
        self.max_utterance_bytes = (
            self.max_utterance_seconds * INTERNAL_SAMPLE_RATE * 2  # PCM16
            if is_parakeet_model(model) else MAX_UTTERANCE_BYTES
        )

        self.audio_buffer = bytearray()
        self._set_sample_rate(sample_rate)
        # The second half of a sample split across two binary frames
        self._odd_byte = b""
        self._chunk_count = 0

        self.agreement = LocalAgreement2()
        self.vad: SileroVAD | None = None
        self.vad_state: SileroVAD | None = None
        self._vad_lock = threading.Lock()

        self.utterance_start = 0.0
        self.total_samples = 0
        self.silence_samples = 0
        self.endpointing_samples = int(INTERNAL_SAMPLE_RATE * endpointing_ms / 1000)
        self.speech_active = False
        self.utterance_audio = bytearray()
        self._vad_buffer = bytearray()
        # The latest windows while no one speaks; an utterance starts with their last VAD_PREROLL_MS.
        # Only windows after the previous utterance come here, so it never reaches back into that one.
        self._pre_roll_bytes = VAD_PREROLL_MS * INTERNAL_SAMPLE_RATE // 1000 * 2
        self._pre_roll: deque[bytes] = deque(maxlen=-(-self._pre_roll_bytes // VAD_WINDOW_BYTES))
        # 16 kHz samples processed so far: the clock for transcript start/end
        self._samples_processed = 0

        self._running = False
        self._transcription_count = 0
        self._error_count = 0
        # The audio received, its peak, the VAD's top score and the speech starts, for the "ended" log line
        self.levels = InputLevels()

    def _set_sample_rate(self, sample_rate: int) -> None:
        self.client_sample_rate = sample_rate
        self.needs_resample = sample_rate != INTERNAL_SAMPLE_RATE
        # One resampler for the whole stream, so chunks join without seams
        self._resampler = StreamingResampler(sample_rate, INTERNAL_SAMPLE_RATE) if self.needs_resample else None
        # Never below one Silero window; a 0 ms window would never consume a frame.
        chunk_ms = max(settings.stt_stream_chunk_ms, VAD_WINDOW_SAMPLES * 1000 / INTERNAL_SAMPLE_RATE)
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * 2

    async def run(self) -> tuple[int, str] | None:
        """Main session loop. Returns the close code and reason for the socket, or None when the client left."""
        self._running = True

        # Pre-validate model is loaded
        loop = asyncio.get_running_loop()
        try:
            if not backend_router.is_model_loaded(self.model):
                logger.info("[%s] Model %s not loaded — loading now...", self.session_id[:8], self.model)
                await loop.run_in_executor(_stream_load_executor, lambda: backend_router.load_model(self.model))
                logger.info("[%s] Model %s loaded successfully", self.session_id[:8], self.model)
        except Exception as e:
            logger.error("[%s] Failed to load model %s: %s", self.session_id[:8], self.model, e)
            await self._send_event({
                "type": "error",
                "message": f"Failed to load model: {redact_paths(str(e))}",
            })
            return CLOSE_MODEL_FAILED

        if self.vad_enabled:
            try:
                self.vad = await get_vad_model()
            except Exception as e:
                logger.error("[%s] Failed to load VAD model: %s", self.session_id[:8], e)
                await self._send_event({
                    "type": "error",
                    "message": f"Failed to load VAD model: {e}",
                })
                return CLOSE_VAD_FAILED
            self.vad_state = SileroVAD(self.vad.session, threshold=settings.stt_vad_threshold)
        else:
            self.vad = None
            self.vad_state = None

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
            "vad_enabled": self.vad_enabled,
            "language": self.language,
        })
        if self.language_error:
            await self._send_event({"type": "error", "message": self.language_error})

        ending: tuple[int, str] | None = CLOSE_ENDED
        try:
            while self._running:
                try:
                    msg = await self.ws.receive()
                except WebSocketDisconnect:
                    ending = None
                    break

                if msg["type"] == "websocket.disconnect":
                    ending = None
                    break

                if msg["type"] == "websocket.receive":
                    if "bytes" in msg and msg["bytes"]:
                        await self._handle_audio(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await self._handle_text(msg["text"])
        except Exception as e:
            logger.exception("[%s] Streaming session error: %s", self.session_id[:8], e)
            ending = CLOSE_FAILED
        finally:
            # The flush runs the audio still buffered through the same VAD and decoder. Its failure must not skip
            # session.end (docs/API.md): the stream then closes with 1011.
            try:
                await self._flush()
            except Exception as e:
                logger.exception("[%s] Streaming session error in the final flush: %s", self.session_id[:8], e)
                if ending is not None:
                    ending = CLOSE_FAILED
            await self._send_event({
                "type": "session.end",
                "reason": "client_stop" if not self._running else "disconnect",
                "transcriptions": self._transcription_count,
                "errors": self._error_count,
            })
        return ending

    async def _handle_text(self, text: str):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[%s] Received malformed JSON text frame", self.session_id[:8])
            return
        if not isinstance(data, dict):
            logger.warning("[%s] Received non-object JSON text frame", self.session_id[:8])
            return
        if data.get("type") == "stop":
            self._running = False
        elif data.get("type") == "config":
            await self._handle_config(data)

    async def _handle_config(self, data: dict):
        if "sample_rate" not in data:
            return
        sample_rate = data["sample_rate"]
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE
        ):
            await self._send_event({
                "type": "error",
                "message": f"Invalid sample_rate: must be {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}",
            })
            return
        if sample_rate == self.client_sample_rate:
            return  # restarting the resampler would only put a seam in the stream
        # Audio already received was recorded at the old rate
        await self._process_buffered_audio()
        self._set_sample_rate(sample_rate)
        logger.info("[%s] Client sample rate changed to %d Hz", self.session_id[:8], sample_rate)

    async def _handle_audio(self, data: bytes):
        data = self._odd_byte + data
        self._odd_byte = b""
        if len(data) % 2 != 0:
            data, self._odd_byte = data[:-1], data[-1:]
        if not data:
            return
        self.audio_buffer.extend(data)
        self.total_samples += len(data) // 2
        self.levels.add_pcm16(data, self.client_sample_rate)
        logger.debug(
            "[%s] Audio recv %d bytes, buffer %d/%d",
            self.session_id[:8],
            len(data),
            len(self.audio_buffer),
            self.chunk_bytes,
        )

        while len(self.audio_buffer) >= self.chunk_bytes:
            chunk = bytes(self.audio_buffer[:self.chunk_bytes])
            del self.audio_buffer[:self.chunk_bytes]
            await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: bytes):
        """Process one audio chunk through VAD + transcription."""
        await self._process_16k(self._to_16k(chunk))

    def _to_16k(self, chunk: bytes) -> bytes:
        """Resample client audio to 16kHz if needed, as part of one continuous stream."""
        if self._resampler is None:
            return chunk
        return _to_pcm16(self._resampler.process(np.frombuffer(chunk, dtype=np.int16)))

    async def _process_16k(self, chunk_16k: bytes, transcribe: bool = True):
        """Run 16 kHz audio through VAD, and transcription unless told not to."""
        if not chunk_16k:
            return

        # If VAD is disabled, treat all audio as speech
        if not self.vad_enabled or self.vad_state is None:
            if not self.speech_active:
                self.speech_active = True
                self.utterance_start = self._samples_processed / INTERNAL_SAMPLE_RATE
                self.utterance_audio = bytearray()
                self.agreement.reset()
            self._samples_processed += len(chunk_16k) // 2
            self.utterance_audio.extend(chunk_16k)
            if len(self.utterance_audio) >= self.max_utterance_bytes:
                await self._finalize_utterance()
            elif transcribe:
                await self._transcribe_utterance()
            return

        # Silero decisions are meaningful per 512-sample window. Keep trailing
        # samples for the next client chunk so packet boundaries cannot discard
        # audio or collapse speech followed by silence into one max decision.
        self._vad_buffer.extend(chunk_16k)
        process_bytes = len(self._vad_buffer) // VAD_WINDOW_BYTES * VAD_WINDOW_BYTES
        if process_bytes == 0:
            return
        vad_pcm = bytes(self._vad_buffer[:process_bytes])
        del self._vad_buffer[:process_bytes]

        samples = np.frombuffer(vad_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._chunk_count < 3:
            logger.debug(
                "[%s] Audio samples min=%.4f max=%.4f rms=%.4f",
                self.session_id[:8],
                samples.min(),
                samples.max(),
                float(np.sqrt(np.mean(samples ** 2))),
            )
            self._chunk_count += 1

        # On the event loop, Silero over a large frame stalled every request and stream.
        # The VAD's state does not depend on the decisions below, so scoring first is the same.
        probs = await asyncio.get_running_loop().run_in_executor(
            _vad_executor, _score_vad_windows, self.vad_state, samples, self._vad_lock,
        )
        if probs:
            self.levels.add_vad(max(probs))
        for index, speech_prob in enumerate(probs):
            offset = index * VAD_WINDOW_SAMPLES
            window_pcm = vad_pcm[offset * 2:(offset + VAD_WINDOW_SAMPLES) * 2]
            window_start_sample = self._samples_processed
            self._samples_processed += VAD_WINDOW_SAMPLES
            window_end = self._samples_processed / INTERNAL_SAMPLE_RATE

            is_speech = speech_prob >= settings.stt_vad_threshold
            logger.debug(
                "[%s] VAD prob=%.3f speech=%s active=%s utterance=%d bytes",
                self.session_id[:8],
                speech_prob,
                is_speech,
                self.speech_active,
                len(self.utterance_audio),
            )

            if is_speech:
                self.silence_samples = 0
                if not self.speech_active:
                    self.speech_active = True
                    pre_roll = b"".join(self._pre_roll)[-self._pre_roll_bytes:]
                    self._pre_roll.clear()
                    self.utterance_start = (window_start_sample - len(pre_roll) // 2) / INTERNAL_SAMPLE_RATE
                    self.utterance_audio = bytearray(pre_roll)
                    self.agreement.reset()
                    logger.info("[%s] Speech started at %.2fs",
                                self.session_id[:8], self.utterance_start)
                    self.levels.speech_started()
                    await self._send_event({"type": "vad", "state": "speech_start"})

                self.utterance_audio.extend(window_pcm)

                if len(self.utterance_audio) >= self.max_utterance_bytes:
                    logger.info("[%s] Utterance exceeded %ds max, force-finalizing",
                                self.session_id[:8], self.max_utterance_seconds)
                    await self._finalize_utterance(end_time=window_end)
            elif self.speech_active:
                self.silence_samples += VAD_WINDOW_SAMPLES
                self.utterance_audio.extend(window_pcm)

                if len(self.utterance_audio) >= self.max_utterance_bytes:
                    logger.info("[%s] Utterance exceeded %ds max during silence, force-finalizing",
                                self.session_id[:8], self.max_utterance_seconds)
                    await self._finalize_utterance(end_time=window_end)
                elif self.silence_samples >= self.endpointing_samples:
                    logger.info("[%s] Speech ended (%.0fms silence), finalizing",
                                self.session_id[:8],
                                self.silence_samples / INTERNAL_SAMPLE_RATE * 1000)
                    await self._finalize_utterance(end_time=window_end)
            else:
                self._pre_roll.append(window_pcm)

        if self.speech_active and transcribe:
            await self._transcribe_utterance(
                end_time=self._samples_processed / INTERNAL_SAMPLE_RATE,
            )

    async def _transcribe_utterance(self, end_time: float | None = None):
        """Transcribe current utterance audio and emit interim/confirmed results."""
        if len(self.utterance_audio) < 3200:
            return

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
                    temperature_fallback=False,  # one decode per window keeps the stream live
                ),
            )
            self._transcription_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error("[%s] Transcription error (#%d): %s",
                        self.session_id[:8], self._error_count, e, exc_info=True)
            await self._send_event({
                "type": "error",
                "message": f"Transcription failed: {redact_paths(str(e))}",
            })
            return

        text = result.get("text", "").strip()
        # What the listener said stays out of the INFO log; "0 chars" still shows an empty decode.
        logger.info("[%s] Transcription #%d: %d chars (%.1fs audio)",
                    self.session_id[:8], self._transcription_count, len(text),
                    len(self.utterance_audio) / 2 / INTERNAL_SAMPLE_RATE)
        logger.debug("[%s] Transcription #%d text: '%s'",
                     self.session_id[:8], self._transcription_count, text)
        if not text:
            return

        new_confirmed, pending = self.agreement.process(text)
        now = (
            end_time
            if end_time is not None
            else self._samples_processed / INTERNAL_SAMPLE_RATE
        )

        if new_confirmed:
            confirmed_text = " ".join(self.agreement.confirmed_words)
            logger.info("[%s] Confirmed: %d words", self.session_id[:8], len(self.agreement.confirmed_words))
            logger.debug("[%s] Confirmed text: '%s'", self.session_id[:8], confirmed_text)
            await self._send_event({
                "type": "transcript",
                "is_final": True,
                "speech_final": False,
                "text": confirmed_text,
                "start": self.utterance_start,
                "end": now,
                "confidence": 0.95,
            })

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

    async def _finalize_utterance(self, end_time: float | None = None):
        """Finalize the current utterance."""
        if len(self.utterance_audio) < 3200:
            was_active = self.speech_active
            self._reset_utterance()
            if was_active and self.vad_enabled:
                await self._send_event({"type": "vad", "state": "speech_end"})
            return

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
                    temperature_fallback=False,  # one decode per window keeps the stream live
                ),
            )
            self._transcription_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error("[%s] Final transcription error (#%d): %s",
                        self.session_id[:8], self._error_count, e, exc_info=True)
            self._reset_utterance()
            await self._send_event({
                "type": "error",
                "message": f"Transcription failed: {redact_paths(str(e))}",
            })
            if self.vad_enabled:
                await self._send_event({"type": "vad", "state": "speech_end"})
            return

        text = result.get("text", "").strip()
        now = (
            end_time
            if end_time is not None
            else self._samples_processed / INTERNAL_SAMPLE_RATE
        )

        if text:
            logger.info("[%s] Final utterance: %d chars (%.1fs audio)",
                       self.session_id[:8], len(text),
                       len(self.utterance_audio) / 2 / INTERNAL_SAMPLE_RATE)
            logger.debug("[%s] Final utterance text: '%s'", self.session_id[:8], text)
            await self._send_event({
                "type": "transcript",
                "is_final": True,
                "speech_final": True,
                "text": text,
                "start": self.utterance_start,
                "end": now,
                "confidence": 0.95,
            })

        # Send VAD speech_end event
        if self.vad_enabled:
            await self._send_event({"type": "vad", "state": "speech_end"})

        self._reset_utterance()

    def _reset_utterance(self) -> None:
        """Reset all state owned by the current utterance."""
        self.speech_active = False
        self.silence_samples = 0
        self.utterance_audio = bytearray()
        self.agreement.reset()

    async def _process_buffered_audio(self):
        """Process audio shorter than a chunk that is still waiting in the buffer.

        This ends the stream at the current sample rate, so the resampler's
        last few samples go out with it.
        """
        remaining = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        audio_16k = self._to_16k(remaining)
        if self._resampler is not None:
            audio_16k += _to_pcm16(self._resampler.flush())
        # The resampler's tail alone is too short to be worth a transcription.
        await self._process_16k(audio_16k, transcribe=bool(remaining))

    async def _flush(self):
        """Flush any remaining audio buffer."""
        await self._process_buffered_audio()
        if self.vad_enabled and self._vad_buffer:
            tail_samples = len(self._vad_buffer) // 2
            if self.speech_active:
                self.utterance_audio.extend(self._vad_buffer)
            self._vad_buffer.clear()
            self._samples_processed += tail_samples
        if self.speech_active and self.utterance_audio:
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
            16,
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
    endpointing: int | None = None,
    vad: bool | None = None,
):
    """WebSocket endpoint for real-time streaming transcription."""
    if sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        # Accepted and then closed, so a browser gets the code and reason, not 1006. 1008 is RFC 6455's policy
        # violation, the code the Origin refusal also uses.
        await refuse_websocket(
            ws, WS_CLOSE_ORIGIN, f"Invalid sample_rate: must be {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}"
        )
        return
    if encoding not in SUPPORTED_ENCODINGS:
        # Audio in another encoding would be transcribed as noise, so refuse it
        # before reading any; 1003 is the close code for data we cannot accept.
        await ws.accept()
        await ws.send_text(json.dumps({
            "type": "error",
            "message": f"Unsupported encoding {encoding!r}: send PCM16 LE mono audio with encoding=pcm_s16le",
        }))
        await ws.close(code=1003, reason="Unsupported encoding")
        return

    # VAD: use query param if provided, otherwise use config default
    vad_enabled = vad if vad is not None else settings.stt_vad_enabled

    # faster-whisper checks the language only after decoding, so an unknown one would fail
    # every transcription: report it and auto-detect instead.
    language_error = None
    try:
        language = whisper_language_code(language)
    except ValueError as exc:
        language, language_error = None, f"{exc}. The stream continues with auto-detect."
    stt_model = resolve_stt_model(model, settings)
    try:
        check_language(stt_model, language)  # Parakeet transcribes English only
    except ValueError as exc:
        language, language_error = None, f"{exc} The stream continues in English."

    await ws.accept()

    session = StreamingSession(
        ws=ws,
        model=stt_model,
        language=language,
        sample_rate=sample_rate,
        interim_results=interim_results,
        endpointing_ms=endpointing if endpointing is not None else settings.stt_stream_endpointing_ms,
        vad_enabled=vad_enabled,
        language_error=language_error,
    )

    # The refusal's close handshake runs outside the lock: a client that doesn't answer it would hold every other
    # session's start and end for up to about 20 s.
    async with _active_sessions_lock:
        full = len(_active_sessions) >= settings.stt_stream_max_connections
        if not full:
            _active_sessions[session.session_id] = session
    if full:
        await ws.close(code=1013, reason="Too many concurrent streams")
        return
    ending: tuple[int, str] | None = None
    try:
        logger.info(
            "Streaming session %s started (model=%s, client_rate=%d, resample=%s, vad=%s)",
            session.session_id, session.model, sample_rate, session.needs_resample, vad_enabled,
        )
        ending = await session.run()
    except Exception:
        logger.exception("Streaming session %s failed", session.session_id)
        ending = CLOSE_FAILED
    finally:
        async with _active_sessions_lock:
            _active_sessions.pop(session.session_id, None)
        # The audio in numbers, never its content or the text: silence, sound the VAD rejects
        # and speech transcribed to nothing each read differently here.
        logger.info(
            "Streaming session %s ended (transcriptions=%d, errors=%d, %s)",
            session.session_id, session._transcription_count, session._error_count,
            session.levels.summary(),
        )
    # A stream the server ended gets a Close frame, or the client sees a dropped connection. Like the refusal's, the
    # handshake runs outside the lock. After the client's own disconnect there is nothing to close, and Starlette raises
    # on a send after it.
    if ending is not None:
        try:
            await ws.close(code=ending[0], reason=ending[1])
        except Exception as exc:
            logger.debug("Streaming session %s: close failed: %s", session.session_id, exc)
