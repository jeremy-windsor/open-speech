"""Input audio buffer with optional Silero VAD integration.

Handles buffering, format conversion, and speech boundary detection
for the OpenAI Realtime API compatibility layer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.audio.levels import InputLevels
from src.audio.resample import StreamingResampler
from src.realtime.g711 import alaw_to_pcm16, pcm16_to_alaw, pcm16_to_ulaw, ulaw_to_pcm16
from src.realtime.session import FORMAT_SAMPLE_RATES
from src.vad.silero import VAD_SAMPLE_RATE, SileroVAD

logger = logging.getLogger(__name__)

VAD_WINDOW_SAMPLES = 512

# Under server VAD a turn keeps prefix_padding_ms of audio before speech, but
# a manual commit takes the audio since the last commit, up to this much of it.
MAX_COMMIT_BACKLOG_MS = 30_000

# Wire format <-> PCM16 LE at the format's own rate (FORMAT_SAMPLE_RATES)
_DECODERS = {"pcm16": bytes, "g711_ulaw": ulaw_to_pcm16, "g711_alaw": alaw_to_pcm16}
_ENCODERS = {"pcm16": bytes, "g711_ulaw": pcm16_to_ulaw, "g711_alaw": pcm16_to_alaw}


class _PCM16Resampler:
    """Anti-aliased resampling of a PCM16 LE mono stream, chunk by chunk."""

    def __init__(self, from_rate: int, to_rate: int):
        self._resampler = None if from_rate == to_rate else StreamingResampler(from_rate, to_rate)

    def process(self, pcm16: bytes) -> bytes:
        if self._resampler is None:
            return pcm16
        return self._to_pcm16(self._resampler.process(np.frombuffer(pcm16, dtype="<i2")))

    def flush(self) -> bytes:
        if self._resampler is None:
            return b""
        return self._to_pcm16(self._resampler.flush())

    @staticmethod
    def _to_pcm16(samples: np.ndarray) -> bytes:
        return np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()


class InputAudioStream:
    """A session's input audio, decoded to PCM16 at target_rate as one stream.

    Appends are resampled as parts of one continuous stream, so they join
    without seams and the session clock does not drift with the client's
    frame size. The newest ~1 ms waits for the next append, or for flush().
    """

    def __init__(self, fmt: str, target_rate: int = VAD_SAMPLE_RATE):
        if fmt not in _DECODERS:
            raise ValueError(f"Unsupported audio format: {fmt}")
        self.format = fmt
        self.target_rate = target_rate
        self._resampler = _PCM16Resampler(FORMAT_SAMPLE_RATES[fmt], target_rate)

    def decode(self, data: bytes) -> bytes:
        return self._resampler.process(_DECODERS[self.format](data))

    def flush(self) -> bytes:
        """End the stream and return the audio still held back."""
        return self._resampler.flush()


class OutputAudioStream:
    """A response's PCM16 audio, encoded to an output format as one stream."""

    def __init__(self, from_rate: int, fmt: str):
        if fmt not in _ENCODERS:
            raise ValueError(f"Unsupported audio format: {fmt}")
        self.format = fmt
        self._resampler = _PCM16Resampler(from_rate, FORMAT_SAMPLE_RATES[fmt])

    def encode(self, pcm16: bytes) -> bytes:
        return _ENCODERS[self.format](self._resampler.process(pcm16))

    def flush(self) -> bytes:
        """End the stream and return the encoded audio still held back."""
        return _ENCODERS[self.format](self._resampler.flush())


def decode_audio_to_pcm16(data: bytes, fmt: str | InputAudioStream, target_rate: int = 16000) -> bytes:
    """Decode audio from the given format to PCM16 mono at target_rate.

    Args:
        data: Raw audio bytes (already base64-decoded).
        fmt: One of 'pcm16', 'g711_ulaw', 'g711_alaw' to convert a whole
            clip, or a session's InputAudioStream to convert its next append.
        target_rate: Target sample rate in Hz.

    Returns:
        PCM16 LE mono bytes at target_rate.
    """
    if isinstance(fmt, InputAudioStream):
        if fmt.target_rate != target_rate:
            raise ValueError(f"Input audio stream converts to {fmt.target_rate} Hz, not {target_rate} Hz")
        return fmt.decode(data)
    stream = InputAudioStream(fmt, target_rate)
    return stream.decode(data) + stream.flush()


def encode_pcm16_to_format(pcm16_data: bytes, from_rate: int, fmt: str) -> bytes:
    """Encode PCM16 mono audio to the specified output format.

    Args:
        pcm16_data: PCM16 LE mono bytes at from_rate.
        from_rate: Source sample rate.
        fmt: Target format ('pcm16', 'g711_ulaw', 'g711_alaw').

    Returns:
        Audio bytes in target format at the appropriate sample rate.
    """
    stream = OutputAudioStream(from_rate, fmt)
    return stream.encode(pcm16_data) + stream.flush()


class InputAudioBuffer:
    """Manages the input audio buffer with optional VAD.

    Audio is stored as PCM16 16kHz mono internally (for STT backends).
    """

    def __init__(self, vad: SileroVAD | None = None, threshold: float = 0.5,
                 silence_duration_ms: int = 500, max_buffer_bytes: int = 50 * 1024 * 1024,
                 prefix_padding_ms: int = 300, max_backlog_ms: int = MAX_COMMIT_BACKLOG_MS):
        self._buffer = bytearray()
        self._vad_buffer = bytearray()
        self._vad = vad
        self._threshold = threshold
        self._silence_duration_ms = silence_duration_ms
        self._prefix_padding_ms = prefix_padding_ms
        self._max_backlog_ms = max_backlog_ms
        self._in_speech = False
        self._silence_samples = 0
        self._speech_start_ms = 0
        self._total_samples = 0  # total samples appended (at 16kHz)
        self._max_buffer_bytes = max_buffer_bytes
        # Audio before this offset belongs to a turn that has stopped but has
        # not been committed yet, so silence trimming must leave it in place.
        self._uncommitted_turn_end = 0
        # Where the current VAD turn's audio, prefix padding included, begins
        self._turn_start = 0
        # Every append in numbers, for the session's "ended" log line; clear() and commits keep them
        self.levels = InputLevels()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def vad(self) -> SileroVAD | None:
        return self._vad

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def configure(self, vad: SileroVAD | None, threshold: float,
                  silence_duration_ms: int, prefix_padding_ms: int) -> None:
        """Apply new turn detection settings, keeping buffered audio and the session clock."""
        if vad is not self._vad:
            # A different detector, or none, starts from silence.
            self._vad_buffer.clear()
            self._in_speech = False
            self._silence_samples = 0
        self._vad = vad
        self._threshold = threshold
        self._silence_duration_ms = silence_duration_ms
        self._prefix_padding_ms = prefix_padding_ms

    def clear(self, discarded_bytes: int = 0) -> None:
        """Clear the audio buffer.

        ``discarded_bytes`` of written audio that the caller dropped before it
        reached the buffer still count toward the session clock.
        """
        self._total_samples += discarded_bytes // 2
        self._buffer.clear()
        self._vad_buffer.clear()
        self._silence_samples = 0
        self._uncommitted_turn_end = 0
        self._turn_start = 0

    def append(self, pcm16_16khz: bytes) -> list[dict[str, Any]]:
        """Append PCM16 16kHz audio and run VAD if enabled.

        Returns a list of events to send (speech_started, speech_stopped).
        """
        events: list[dict[str, Any]] = []
        self.levels.add_pcm16(pcm16_16khz, VAD_SAMPLE_RATE)  # received, even if refused below
        frame_size = len(pcm16_16khz)
        if frame_size > self._max_buffer_bytes:
            self.clear()
            raise BufferError(f"Audio frame exceeds max buffer size ({self._max_buffer_bytes} bytes)")
        if len(self._buffer) + frame_size > self._max_buffer_bytes:
            raise BufferError(f"Input audio buffer exceeded max size ({self._max_buffer_bytes} bytes)")
        pending_vad_bytes = len(self._vad_buffer)
        first_window_buffer_offset = len(self._buffer) - pending_vad_bytes
        self._buffer.extend(pcm16_16khz)

        num_samples = len(pcm16_16khz) // 2
        pending_vad_samples = pending_vad_bytes // 2
        first_window_sample = self._total_samples - pending_vad_samples
        self._total_samples += num_samples

        if self._vad is None:
            return events

        # Silero consumes 512-sample windows. Preserve partial WebSocket frames
        # so VAD decisions do not depend on how the client packetizes audio.
        self._vad_buffer.extend(pcm16_16khz)
        vad_window_bytes = VAD_WINDOW_SAMPLES * 2
        process_bytes = len(self._vad_buffer) // vad_window_bytes * vad_window_bytes
        if process_bytes == 0:
            return events
        vad_pcm = bytes(self._vad_buffer[:process_bytes])
        del self._vad_buffer[:process_bytes]
        audio = np.frombuffer(vad_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # A turn starts prefix_padding_ms before its speech (OpenAI's
        # audio_start_ms includes the padding), but not before an earlier turn ends.
        padding_bytes = self._prefix_padding_ms * VAD_SAMPLE_RATE // 1000 * 2

        for offset in range(0, len(audio), VAD_WINDOW_SAMPLES):
            window = audio[offset:offset + VAD_WINDOW_SAMPLES]
            window_buffer_offset = first_window_buffer_offset + offset * 2
            window_start_sample = first_window_sample + offset
            window_end_ms = (
                (window_start_sample + VAD_WINDOW_SAMPLES) * 1000
                // VAD_SAMPLE_RATE
            )
            speech_prob = self._vad(window)
            self.levels.add_vad(speech_prob)
            is_speech = speech_prob >= self._threshold

            if is_speech:
                self._silence_samples = 0
                if not self._in_speech:
                    self._in_speech = True
                    self._turn_start = max(self._uncommitted_turn_end, window_buffer_offset - padding_bytes)
                    turn_start_sample = window_start_sample - (window_buffer_offset - self._turn_start) // 2
                    self._speech_start_ms = turn_start_sample * 1000 // VAD_SAMPLE_RATE
                    self.levels.speech_started()
                    events.append({
                        "type": "speech_started",
                        "audio_start_ms": self._speech_start_ms,
                    })
            elif self._in_speech:
                self._silence_samples += VAD_WINDOW_SAMPLES
                silence_ms = self._silence_samples * 1000 // VAD_SAMPLE_RATE
                if silence_ms >= self._silence_duration_ms:
                    self._in_speech = False
                    self._silence_samples = 0
                    self._uncommitted_turn_end = window_buffer_offset + VAD_WINDOW_SAMPLES * 2
                    events.append({
                        "type": "speech_stopped",
                        "audio_end_ms": window_end_ms,
                        # Private offsets used by the realtime handler to commit
                        # only this turn. A later utterance may already exist
                        # in the same WebSocket frame.
                        "_buffer_start_bytes": self._turn_start,
                        "_buffer_end_bytes": self._uncommitted_turn_end,
                    })

        if not self._in_speech:
            # Idle audio waits for a manual commit, up to the backlog limit.
            backlog_bytes = min(
                self._max_backlog_ms * VAD_SAMPLE_RATE // 1000 * 2, self._max_buffer_bytes // 2,
            )
            unprocessed_offset = len(self._buffer) - len(self._vad_buffer)
            self._drop_audio_before(unprocessed_offset - max(backlog_bytes, padding_bytes))
        return events

    def _drop_audio_before(self, offset: int) -> None:
        """Drop idle audio before ``offset``, keeping any uncommitted turn."""
        start = self._uncommitted_turn_end
        if offset > start:
            del self._buffer[start:offset]

    def commit(self) -> bytes:
        """Commit and return the current buffer contents, then clear."""
        data = bytes(self._buffer)
        self.clear()
        return data

    def commit_through(self, byte_count: int, start: int = 0) -> bytes:
        """Commit buffer[start:byte_count], drop the audio before it, and keep later audio."""
        if not 0 <= start <= byte_count <= len(self._buffer):
            raise ValueError("Commit boundary is outside the audio buffer")
        data = bytes(self._buffer[start:byte_count])
        del self._buffer[:byte_count]
        self._uncommitted_turn_end = max(0, self._uncommitted_turn_end - byte_count)
        self._turn_start = max(0, self._turn_start - byte_count)
        return data

    def get_audio(self) -> bytes:
        """Get current buffer contents without clearing."""
        return bytes(self._buffer)
