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
import os
import tempfile
import threading
import time
import urllib.request
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
# The whole download must finish in this time; one stalled read fails it too.
SILERO_DOWNLOAD_TIMEOUT_S = 60.0

# VAD expects 16kHz mono audio
VAD_SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
VAD_CONTEXT_SIZE = 64

_vad_model: SileroVAD | None = None
_vad_lock = asyncio.Lock()
# One download at a time; sessions wait on it, not on _vad_lock.
_download_lock = asyncio.Lock()
# Finished download attempts and how the last one failed, so sessions that
# waited on an attempt share its failure instead of each waiting out another.
_download_attempts = 0
_download_error: Exception | None = None


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
        # Round up so a segment always lasts at least the requested duration
        silence_windows = max(1, -(-silence_ms // window_ms))
        min_speech_windows = max(1, -(-min_speech_ms // window_ms))

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


def _fetch_model(deadline: float) -> bytes:
    """Read the model into memory, at most one chunk past its size (blocking).

    Every socket step waits at most SILERO_DOWNLOAD_TIMEOUT_S and the reads
    stop at the deadline, so an attempt _download_model gave up on ends on
    its own.
    """
    data = bytearray()
    with urllib.request.urlopen(SILERO_ONNX_URL, timeout=SILERO_DOWNLOAD_TIMEOUT_S) as response:
        # read1 makes one socket read, so the deadline is checked as data trickles in
        while chunk := response.read1(64 * 1024):
            data += chunk
            if len(data) > SILERO_ONNX_BYTES:
                break  # the size check in _download_model rejects it
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Silero VAD model download took longer than {SILERO_DOWNLOAD_TIMEOUT_S:g} s"
                )
    return bytes(data)


def _download_model(model_path: Path) -> None:
    """Download and verify the model, then move it into place (blocking).

    Raises TimeoutError when the download takes longer than
    SILERO_DOWNLOAD_TIMEOUT_S. Socket timeouts alone would give the lookup,
    each address's connect, the TLS handshake and each read their own full
    timeout, so the network part runs in a daemon thread that is given up
    on at the deadline; it writes nothing to disk.
    """
    timeout_s = SILERO_DOWNLOAD_TIMEOUT_S
    deadline = time.monotonic() + timeout_s
    outcome: dict = {}

    def fetch() -> None:
        try:
            outcome["data"] = _fetch_model(deadline)
        except BaseException as exc:  # noqa: BLE001 - raised again below
            outcome["error"] = exc

    worker = threading.Thread(target=fetch, name="silero-vad-download", daemon=True)
    worker.start()
    worker.join(max(0.0, deadline - time.monotonic()))
    if worker.is_alive():
        raise TimeoutError(f"Silero VAD model download took longer than {timeout_s:g} s")
    if "error" in outcome:
        raise outcome["error"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=model_path.parent, prefix=model_path.name, suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(outcome["data"])
        if not _model_file_matches(tmp_path):
            raise RuntimeError(
                f"Downloaded Silero VAD model {SILERO_TAG} failed size/SHA256 validation"
            )
        tmp_path.replace(model_path)
    finally:
        tmp_path.unlink(missing_ok=True)


async def _ensure_model_file(model_path: Path) -> None:
    """Download the model file unless a valid copy is cached.

    A failed download is not cached: the next session to ask tries again.
    """
    global _download_attempts, _download_error
    if _model_file_matches(model_path):
        return
    attempts_seen = _download_attempts
    async with _download_lock:
        if _model_file_matches(model_path):
            return  # another session just downloaded it
        if _download_attempts != attempts_seen and _download_error is not None:
            raise RuntimeError(f"Silero VAD model download failed: {_download_error}") from _download_error
        if model_path.exists():
            logger.warning(
                "Cached Silero VAD model at %s does not match %s metadata; re-downloading",
                model_path,
                SILERO_TAG,
            )
        logger.info("Downloading Silero VAD model %s...", SILERO_TAG)
        _download_error = None
        try:
            await asyncio.get_running_loop().run_in_executor(None, _download_model, model_path)
        except Exception as e:
            _download_error = e
            raise
        finally:
            _download_attempts += 1
        logger.info("Silero VAD model downloaded to %s", model_path)


_blocking_load_lock = threading.Lock()


def load_vad_model_blocking() -> SileroVAD:
    """The shared Silero model for a worker thread, which has no event loop to await get_vad_model() on.

    Downloads and verifies the file first when it isn't cached, as get_vad_model does; the file is written to a
    temporary name and renamed, so this and an async load never see a half-written file.
    """
    global _vad_model
    if _vad_model is not None:
        return _vad_model
    model_path = SILERO_CACHE_DIR / SILERO_ONNX_FILENAME
    with _blocking_load_lock:
        if _vad_model is not None:
            return _vad_model
        if not _model_file_matches(model_path):
            logger.info("Downloading Silero VAD model %s...", SILERO_TAG)
            _download_model(model_path)

        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if _vad_model is None:  # an async load may have finished meanwhile; keep the one sessions already share
            _vad_model = SileroVAD(sess)
            logger.info("Silero VAD model loaded")
        return _vad_model


async def get_vad_model() -> SileroVAD:
    """Lazy-load Silero VAD ONNX model (singleton).

    Returns a SileroVAD instance. For per-stream use, create a new
    SileroVAD(model.session) to get independent state.
    """
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    # The download is bounded and runs outside _vad_lock, so a stalled network
    # cannot leave every streaming and realtime session waiting on the lock.
    model_path = SILERO_CACHE_DIR / SILERO_ONNX_FILENAME
    await _ensure_model_file(model_path)

    async with _vad_lock:
        if _vad_model is not None:
            return _vad_model

        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _vad_model = SileroVAD(sess)
        logger.info("Silero VAD model loaded")
        return _vad_model
