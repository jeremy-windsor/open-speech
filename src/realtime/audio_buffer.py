"""Input audio buffer with optional Silero VAD integration.

Handles buffering, format conversion, and speech boundary detection
for the OpenAI Realtime API compatibility layer.
"""

from __future__ import annotations

import audioop
import logging
import struct
from typing import Any

import numpy as np

from src.vad.silero import SileroVAD, VAD_SAMPLE_RATE

logger = logging.getLogger(__name__)


def _resample_linear(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Simple linear interpolation resample for PCM16 mono."""
    if from_rate == to_rate:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm_bytes
    ratio = to_rate / from_rate
    out_len = int(len(samples) * ratio)
    if out_len == 0:
        return b""
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, out_len)
    resampled = np.interp(x_new, x_old, samples)
    return resampled.astype(np.int16).tobytes()


def decode_audio_to_pcm16(data: bytes, fmt: str, target_rate: int = 16000) -> bytes:
    """Decode audio from the given format to PCM16 mono at target_rate.

    Args:
        data: Raw audio bytes (already base64-decoded).
        fmt: One of 'pcm16', 'g711_ulaw', 'g711_alaw'.
        target_rate: Target sample rate in Hz.

    Returns:
        PCM16 LE mono bytes at target_rate.
    """
    if fmt == "pcm16":
        # OpenAI Realtime pcm16 is 24kHz mono 16-bit LE
        return _resample_linear(data, 24000, target_rate)
    elif fmt == "g711_ulaw":
        pcm = audioop.ulaw2lin(data, 2)
        return _resample_linear(pcm, 8000, target_rate)
    elif fmt == "g711_alaw":
        pcm = audioop.alaw2lin(data, 2)
        return _resample_linear(pcm, 8000, target_rate)
    else:
        raise ValueError(f"Unsupported audio format: {fmt}")


def encode_pcm16_to_format(pcm16_data: bytes, from_rate: int, fmt: str) -> bytes:
    """Encode PCM16 mono audio to the specified output format.

    Args:
        pcm16_data: PCM16 LE mono bytes at from_rate.
        from_rate: Source sample rate.
        fmt: Target format ('pcm16', 'g711_ulaw', 'g711_alaw').

    Returns:
        Audio bytes in target format at the appropriate sample rate.
    """
    if fmt == "pcm16":
        return _resample_linear(pcm16_data, from_rate, 24000)
    elif fmt == "g711_ulaw":
        pcm_8k = _resample_linear(pcm16_data, from_rate, 8000)
        return audioop.lin2ulaw(pcm_8k, 2)
    elif fmt == "g711_alaw":
        pcm_8k = _resample_linear(pcm16_data, from_rate, 8000)
        return audioop.lin2alaw(pcm_8k, 2)
    else:
        raise ValueError(f"Unsupported audio format: {fmt}")


class InputAudioBuffer:
    """Manages the input audio buffer with optional VAD.

    Audio is stored as PCM16 16kHz mono internally (for STT backends).
    """

    def __init__(self, vad: SileroVAD | None = None, threshold: float = 0.5,
                 silence_duration_ms: int = 500):
        self._buffer = bytearray()
        self._vad = vad
        self._threshold = threshold
        self._silence_duration_ms = silence_duration_ms
        self._in_speech = False
        self._silence_samples = 0
        self._speech_start_ms = 0
        self._total_samples = 0  # total samples appended (at 16kHz)

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def clear(self) -> None:
        """Clear the audio buffer."""
        self._buffer.clear()
        self._silence_samples = 0

    def append(self, pcm16_16khz: bytes) -> list[dict[str, Any]]:
        """Append PCM16 16kHz audio and run VAD if enabled.

        Returns a list of events to send (speech_started, speech_stopped).
        """
        events: list[dict[str, Any]] = []
        self._buffer.extend(pcm16_16khz)

        num_samples = len(pcm16_16khz) // 2
        current_ms = (self._total_samples * 1000) // VAD_SAMPLE_RATE
        self._total_samples += num_samples

        if self._vad is None:
            return events

        # Run VAD on the new chunk
        audio = np.frombuffer(pcm16_16khz, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) == 0:
            return events

        prob = self._vad(audio)
        is_speech = prob >= self._threshold

        if is_speech:
            self._silence_samples = 0
            if not self._in_speech:
                self._in_speech = True
                self._speech_start_ms = current_ms
                events.append({"type": "speech_started", "audio_start_ms": current_ms})
        else:
            if self._in_speech:
                self._silence_samples += num_samples
                silence_ms = (self._silence_samples * 1000) // VAD_SAMPLE_RATE
                if silence_ms >= self._silence_duration_ms:
                    end_ms = current_ms
                    self._in_speech = False
                    self._silence_samples = 0
                    events.append({"type": "speech_stopped", "audio_end_ms": end_ms})

        return events

    def commit(self) -> bytes:
        """Commit and return the current buffer contents, then clear."""
        data = bytes(self._buffer)
        self.clear()
        return data

    def get_audio(self) -> bytes:
        """Get current buffer contents without clearing."""
        return bytes(self._buffer)
