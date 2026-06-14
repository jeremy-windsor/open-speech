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
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SILERO_TAG = "v5.1.2"
SILERO_ONNX_FILENAME = f"silero_vad_{SILERO_TAG}.onnx"
SILERO_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    f"{SILERO_TAG}/src/silero_vad/data/silero_vad.onnx"
)
# Verified against the tagged raw GitHub asset for v5.1.2.
SILERO_ONNX_BYTES = 2_327_524
SILERO_ONNX_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
SILERO_CACHE_DIR = Path.home() / ".cache" / "silero-vad"

# VAD expects 16kHz mono audio
VAD_SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
VAD_CONTEXT_SIZE = 64

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
        # Silero VAD v5 expects 64 samples of prior audio before each 512-sample frame.
        self._context = np.zeros(VAD_CONTEXT_SIZE, dtype=np.float32)

    def reset(self):
        """Reset internal VAD state for a new audio stream."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(VAD_CONTEXT_SIZE, dtype=np.float32)

    def _run_window(self, window: np.ndarray) -> float:
        """Run one 512-sample window with the required v5 audio context."""
        window = window.astype(np.float32, copy=False)
        input_data = np.concatenate((self._context, window)).reshape(1, -1)
        sr = np.array(self.sample_rate, dtype=np.int64)

        ort_inputs = {
            "input": input_data,
            "state": self._state,
            "sr": sr,
        }
        out, self._state = self.session.run(None, ort_inputs)
        self._context = window[-VAD_CONTEXT_SIZE:].copy()
        return float(out[0][0])

    def __call__(self, audio: np.ndarray) -> float:
        """Run VAD on audio chunk. Returns speech probability 0-1.

        Audio MUST be float32, mono, 16kHz, shape (N,). Full 512-sample
        windows are evaluated; trailing partial windows are ignored.
        """
        if len(audio) == 0:
            return 0.0

        max_prob = 0.0

        for start in range(0, len(audio) - VAD_WINDOW_SIZE + 1, VAD_WINDOW_SIZE):
            chunk = audio[start:start + VAD_WINDOW_SIZE]
            prob = self._run_window(chunk)
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

        window_ms = VAD_WINDOW_SIZE * 1000 // self.sample_rate
        silence_windows = max(1, silence_ms // window_ms)
        min_speech_windows = max(1, min_speech_ms // window_ms)

        segments: list[Segment] = []
        in_speech = False
        speech_start = 0
        silence_count = 0
        speech_windows = 0

        for start in range(0, len(audio) - VAD_WINDOW_SIZE + 1, VAD_WINDOW_SIZE):
            chunk = audio[start:start + VAD_WINDOW_SIZE]
            prob = self._run_window(chunk)
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_file_matches(path: Path) -> bool:
    if not path.exists() or path.stat().st_size != SILERO_ONNX_BYTES:
        return False
    return _sha256(path) == SILERO_ONNX_SHA256


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

        model_path = SILERO_CACHE_DIR / SILERO_ONNX_FILENAME
        if not _model_file_matches(model_path):
            if model_path.exists():
                logger.warning(
                    "Cached Silero VAD model at %s does not match %s metadata; re-downloading",
                    model_path,
                    SILERO_TAG,
                )
            logger.info("Downloading Silero VAD model %s...", SILERO_TAG)
            SILERO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            import urllib.request
            tmp_path = model_path.with_suffix(".onnx.tmp")
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: urllib.request.urlretrieve(SILERO_ONNX_URL, str(tmp_path))
            )
            if not _model_file_matches(tmp_path):
                raise RuntimeError(
                    f"Downloaded Silero VAD model {SILERO_TAG} failed size/SHA256 validation"
                )
            tmp_path.replace(model_path)
            logger.info("Silero VAD model downloaded to %s", model_path)

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _vad_model = SileroVAD(sess)
        logger.info("Silero VAD model loaded")
        return _vad_model
