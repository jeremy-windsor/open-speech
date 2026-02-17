"""Silero VAD wrapper — lightweight voice activity detection via ONNX.

Uses the Silero VAD ONNX model (<2MB, MIT licensed) for speech/silence
classification. Works on CPU; no PyTorch dependency required.

Usage:
    vad = await get_vad_model()
    # Per-stream: create a new instance sharing the ONNX session
    stream_vad = SileroVAD(vad.session)
    prob = stream_vad(audio_float32_16khz)

    # Or use higher-level helpers:
    stream_vad.is_speech(pcm16_bytes)  # -> bool
    stream_vad.get_speech_segments(pcm16_bytes)  # -> list[Segment]
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SILERO_CACHE_DIR = Path.home() / ".cache" / "silero-vad"

# VAD expects 16kHz mono audio
VAD_SAMPLE_RATE = 16000

_vad_model: SileroVAD | None = None
_vad_lock = asyncio.Lock()


@dataclass
class Segment:
    """A detected speech segment."""
    start_ms: int
    end_ms: int


class SileroVAD:
    """Wrapper around the Silero VAD ONNX model.

    Each streaming session should create its own SileroVAD instance
    (sharing the same ONNX session) to maintain independent state.
    """

    def __init__(self, session, threshold: float = 0.5):
        self.session = session
        self.sample_rate = VAD_SAMPLE_RATE
        self.threshold = threshold
        # Internal state tensor: shape [2, 1, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def reset(self):
        """Reset internal VAD state for a new audio stream."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> float:
        """Run VAD on audio chunk. Returns speech probability 0-1.

        Audio MUST be float32, mono, 16kHz, shape (N,) where N is
        a multiple of 512 samples (32ms at 16kHz). For best results
        use 512 or 1536 sample windows.
        """
        if len(audio) == 0:
            return 0.0

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

    def is_speech(self, pcm16_bytes: bytes, threshold: float | None = None) -> bool:
        """Check if a PCM16 audio chunk contains speech.

        Args:
            pcm16_bytes: Raw PCM16 LE mono 16kHz audio bytes.
            threshold: Speech probability threshold (default: self.threshold).

        Returns:
            True if speech probability exceeds threshold.
        """
        if not pcm16_bytes:
            return False
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        prob = self(audio)
        return prob >= (threshold if threshold is not None else self.threshold)

    def get_speech_segments(
        self,
        pcm16_bytes: bytes,
        threshold: float | None = None,
        min_speech_ms: int = 250,
        silence_ms: int = 800,
    ) -> list[Segment]:
        """Detect speech segments in an audio buffer.

        Args:
            pcm16_bytes: Raw PCM16 LE mono 16kHz audio bytes.
            threshold: Speech probability threshold.
            min_speech_ms: Minimum speech duration to include.
            silence_ms: Silence duration to end a segment.

        Returns:
            List of Segment(start_ms, end_ms).
        """
        if not pcm16_bytes:
            return []

        thresh = threshold if threshold is not None else self.threshold
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        window_size = 512  # 32ms at 16kHz
        window_ms = window_size * 1000 // self.sample_rate
        silence_windows = max(1, silence_ms // window_ms)
        min_speech_windows = max(1, min_speech_ms // window_ms)

        segments: list[Segment] = []
        in_speech = False
        speech_start = 0
        silence_count = 0
        speech_windows = 0

        for start in range(0, len(audio) - window_size + 1, window_size):
            chunk = audio[start:start + window_size]
            input_data = chunk.reshape(1, -1).astype(np.float32)
            sr = np.array(self.sample_rate, dtype=np.int64)
            ort_inputs = {"input": input_data, "state": self._state, "sr": sr}
            out, self._state = self.session.run(None, ort_inputs)
            prob = float(out[0][0])

            current_ms = start * 1000 // self.sample_rate

            if prob >= thresh:
                silence_count = 0
                if not in_speech:
                    in_speech = True
                    speech_start = current_ms
                    speech_windows = 0
                speech_windows += 1
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= silence_windows:
                        end_ms = current_ms
                        if speech_windows >= min_speech_windows:
                            segments.append(Segment(start_ms=speech_start, end_ms=end_ms))
                        in_speech = False
                        silence_count = 0
                        speech_windows = 0

        # Close any open segment
        if in_speech and speech_windows >= min_speech_windows:
            end_ms = len(audio) * 1000 // self.sample_rate
            segments.append(Segment(start_ms=speech_start, end_ms=end_ms))

        return segments


async def get_vad_model() -> SileroVAD:
    """Lazy-load Silero VAD ONNX model (singleton).

    Returns a SileroVAD instance. For per-stream use, create a new
    SileroVAD(model.session) to get independent state.
    """
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
            import urllib.request
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: urllib.request.urlretrieve(SILERO_ONNX_URL, str(model_path))
            )
            logger.info("Silero VAD model downloaded to %s", model_path)

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _vad_model = SileroVAD(sess)
        logger.info("Silero VAD model loaded")
        return _vad_model
