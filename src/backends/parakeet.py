"""NVIDIA Parakeet TDT speech-to-text through onnx-asr (ONNX Runtime).

English only, with punctuation and capitals; several times faster than Whisper large-v3-turbo on a GPU, and as fast
as Whisper base on a CPU with turbo-level accuracy.

- Files: the published ONNX export (``istupakov/parakeet-tdt-0.6b-v2-onnx``) at a pinned revision. On a GPU the fp32
  files are fetched and converted to fp16 once (src/backends/parakeet_fp16.py); on a CPU the published int8 files are
  used as they are.
- Memory: the CUDA session's arena is capped (STT_PARAKEET_GPU_MEM_LIMIT_MB) and never sees more than
  MAX_PIECE_S of audio per item: past the card's memory Windows pages the other models out instead of failing.
- Long audio: the export's positional encoding has a fixed table and full attention grows fast, so audio longer
  than SINGLE_PIECE_MAX_S is cut with Silero VAD into pieces of at most MAX_PIECE_S (a piece with no pause is
  cut at its quietest window), decoded in batches of up to BATCH_SIZE pieces and BATCH_MAX_S of audio.
- Requests: a language other than English is refused (LanguageUnsupportedError, ``language_unsupported``), as is
  translation; ``prompt`` and ``temperature`` have no meaning for the greedy TDT decoder and are ignored.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import io
import logging
import math
import shutil
import threading
import time
import wave
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.backends import parakeet_fp16
from src.backends.faster_whisper import _disk_size_mb, refuse_network_model_path
from src.config import hf_hub_cache_dir, settings
from src.formatters import _fmt_time_srt, _fmt_time_vtt
from src.models import LoadedModelInfo

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Audio up to this long is decoded as one piece; longer audio is cut with VAD.
SINGLE_PIECE_MAX_S = 25.0
MAX_PIECE_S = 20.0
BATCH_SIZE = 4
# A batch holds at most this much padded audio, one piece of SINGLE_PIECE_MAX_S: under the default GPU arena cap
# (STT_PARAKEET_GPU_MEM_LIMIT_MB) two 20 s pieces at once don't fit, while one 25 s piece or four 6 s ones do.
BATCH_MAX_S = 25.0
# Speech is kept with this much audio either side of the VAD's windows over the threshold.
PIECE_PAD_S = 0.2
# A pause this long ends a speech region; regions are then joined into pieces of at most MAX_PIECE_S.
MIN_PAUSE_S = 0.3
VAD_WINDOW = 512
# onnx-asr gives each token's start; the last word of a piece ends at most this long after its last token starts.
LAST_TOKEN_S = 0.4
# A segment ends after a word ending in one of these, unless the word is a title or an initial ("Mr.", "J.").
SENTENCE_END = (".", "?", "!")
NOT_SENTENCE_END = frozenset({"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "prof.", "vs.", "e.g.", "i.e."})
# A streaming utterance is cut at this length for Parakeet (src/streaming.py).
STREAM_MAX_UTTERANCE_S = 25

# Whisper models to name when Parakeet is asked for another language.
MULTILINGUAL_ALTERNATIVE = "deepdml/faster-whisper-large-v3-turbo-ct2"


@dataclass(frozen=True)
class ParakeetModel:
    onnx_repo: str
    revision: str  # a commit of onnx_repo, so the files (and the fp16 cache made from them) never change unseen
    onnx_asr_name: str


PARAKEET_MODELS: dict[str, ParakeetModel] = {
    "nvidia/parakeet-tdt-0.6b-v2": ParakeetModel(
        onnx_repo="istupakov/parakeet-tdt-0.6b-v2-onnx",
        revision="0bbb45a3365852604aef28b538a8f066f4ccaa85",
        onnx_asr_name="nemo-parakeet-tdt-0.6b-v2",
    ),
}

INT8_FILES = (
    "config.json", "vocab.txt", "nemo128.onnx", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx",
)

_fetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="parakeet-fetch")


def is_parakeet_model(model_id: object) -> bool:
    return isinstance(model_id, str) and model_id in PARAKEET_MODELS


class LanguageUnsupportedError(ValueError):
    """A request asked Parakeet for a language other than English. HTTP answers it 400 with this code."""

    code = "language_unsupported"


class TranslationUnsupportedError(ValueError):
    code = "translation_unsupported"


def is_english_or_auto(language: str | None) -> bool:
    """Whether a request's language lets an English-only model answer: none, ``auto``, or English (``en``, ``en-US``)."""
    if language is None:
        return True
    code = language.strip().lower().replace("_", "-")
    return code in ("", "auto", "en") or code.startswith("en-")


def language_unsupported_message(model_id: str, language: str) -> str:
    return (
        f"{model_id} transcribes English only, so it can't transcribe language {language!r}. "
        f"Use a Whisper model for other languages, such as {MULTILINGUAL_ALTERNATIVE}."
    )


def check_language(model_id: str, language: str | None) -> None:
    """Raise LanguageUnsupportedError when ``model_id`` is a Parakeet model and ``language`` isn't English."""
    if is_parakeet_model(model_id) and not is_english_or_auto(language):
        raise LanguageUnsupportedError(language_unsupported_message(model_id, language or ""))


# --- Audio --------------------------------------------------------------------------------------------------------


def decode_audio(audio: bytes) -> np.ndarray:
    """16 kHz mono float32 samples from an upload; the routes give 16 kHz mono PCM16 WAV, read here directly."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (
                SAMPLE_RATE, 1, 2, "NONE",
            ):
                frames = wav.readframes(wav.getnframes())
                return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    except (EOFError, wave.Error):
        pass
    from faster_whisper.audio import decode_audio as decode_any  # any format and rate, through PyAV

    return np.asarray(decode_any(io.BytesIO(audio), sampling_rate=SAMPLE_RATE), dtype=np.float32)


def plan_pieces(
    probs: Sequence[float],
    n_samples: int,
    *,
    threshold: float,
    window: int = VAD_WINDOW,
    max_samples: int = int(MAX_PIECE_S * SAMPLE_RATE),
    pad_samples: int = int(PIECE_PAD_S * SAMPLE_RATE),
    min_pause_windows: int = max(1, round(MIN_PAUSE_S * SAMPLE_RATE / VAD_WINDOW)),
) -> list[tuple[int, int]]:
    """Cut audio into pieces of speech, each at most ``max_samples`` long, from Silero's score for each window.

    Speech regions (windows at or over ``threshold``, ended by ``min_pause_windows`` below it) are padded by
    ``pad_samples``; a region longer than ``max_samples`` is cut at its quietest window in the second half of each
    ``max_samples`` span, or at ``max_samples`` exactly when it has none quieter than the rest; then neighbouring
    regions are joined while the piece they make stays within ``max_samples``. Returns (start, end) sample offsets;
    an empty list when no window is speech.
    """
    regions: list[tuple[int, int]] = []
    start: int | None = None
    quiet = 0
    for index, prob in enumerate(probs):
        if prob >= threshold:
            if start is None:
                start = index
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= min_pause_windows:
                regions.append((start, index - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        regions.append((start, len(probs) - quiet))

    spans: list[tuple[int, int]] = []
    for first, end in regions:
        lo = max(0, first * window - pad_samples)
        hi = min(n_samples, end * window + pad_samples)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))

    cut: list[tuple[int, int]] = []
    for lo, hi in spans:
        while hi - lo > max_samples:
            # The quietest window in the second half of the next max_samples, so no piece is tiny.
            first_window = -(-(lo + max_samples // 2) // window)
            last_window = (lo + max_samples) // window - 1
            candidates = [i for i in range(first_window, min(last_window, len(probs) - 1) + 1)]
            at = lo + max_samples
            if candidates:
                quietest = min(candidates, key=lambda i: (probs[i], -i))
                if probs[quietest] < max(probs[i] for i in candidates):
                    at = min(lo + max_samples, (quietest + 1) * window)
            cut.append((lo, at))
            lo = at
        if hi > lo:
            cut.append((lo, hi))

    pieces: list[tuple[int, int]] = []
    for lo, hi in cut:
        if pieces and hi - pieces[-1][0] <= max_samples:
            pieces[-1] = (pieces[-1][0], hi)
        else:
            pieces.append((lo, hi))
    return pieces


def even_pieces(n_samples: int, max_samples: int = int(MAX_PIECE_S * SAMPLE_RATE)) -> list[tuple[int, int]]:
    """Equal pieces of at most ``max_samples``, for when the VAD can't run."""
    count = max(1, math.ceil(n_samples / max_samples))
    bounds = [round(i * n_samples / count) for i in range(count + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(count) if bounds[i + 1] > bounds[i]]


def score_speech(samples: np.ndarray) -> list[float]:
    """Silero's speech probability for each 512-sample window of 16 kHz audio (the core's shared model)."""
    from src.vad.silero import SileroVAD, load_vad_model_blocking

    shared = load_vad_model_blocking()
    vad = SileroVAD(shared.session)  # its own state; the ONNX session is shared and thread-safe
    return [
        vad(samples[offset:offset + VAD_WINDOW])
        for offset in range(0, len(samples) - VAD_WINDOW + 1, VAD_WINDOW)
    ]


# --- Results ------------------------------------------------------------------------------------------------------


def _words(result: Any, offset: float, piece_end: float) -> list[dict[str, Any]]:
    """Words with start and end times from one piece's tokens: a token with a leading space starts a word, and a
    word ends where the next one starts."""
    tokens = list(getattr(result, "tokens", None) or [])
    times = list(getattr(result, "timestamps", None) or [])
    logprobs = list(getattr(result, "logprobs", None) or [])
    if not tokens or len(times) != len(tokens):
        return []
    words: list[dict[str, Any]] = []
    for index, (token, start) in enumerate(zip(tokens, times)):
        if not words or token.startswith(" "):
            words.append({"word": "", "start": offset + float(start), "logprobs": []})
        words[-1]["word"] += token
        if index < len(logprobs):
            words[-1]["logprobs"].append(float(logprobs[index]))
    last_token_start = offset + float(times[-1])
    for index, word in enumerate(words):
        word["word"] = word["word"].strip()
        if index + 1 < len(words):
            word["end"] = words[index + 1]["start"]
        else:
            word["end"] = max(word["start"], min(piece_end, last_token_start + LAST_TOKEN_S))
    return [word for word in words if word["word"]]


def _ends_sentence(word: str) -> bool:
    if not word.endswith(SENTENCE_END):
        return False
    bare = word.lstrip("\"'(")
    return not (bare.lower() in NOT_SENTENCE_END or (len(bare) == 2 and bare[0].isupper() and bare[1] == "."))


def _compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def _segment(index: int, words: list[dict[str, Any]], *, start: float | None = None,
             end: float | None = None, text: str | None = None, logprobs: list[float] | None = None) -> dict:
    text = text if text is not None else " ".join(word["word"] for word in words)
    logprobs = logprobs if logprobs is not None else [p for word in words for p in word["logprobs"]]
    return {
        "id": index,
        "seek": 0,
        "start": round(start if start is not None else words[0]["start"], 3),
        "end": round(end if end is not None else words[-1]["end"], 3),
        "text": " " + text.strip(),
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": round(sum(logprobs) / len(logprobs), 4) if logprobs else 0.0,
        "compression_ratio": round(_compression_ratio(text.strip()), 4),
        # Every piece is speech the VAD found, and the model gives no such score.
        "no_speech_prob": 0.0,
    }


def build_result(pieces: list[tuple[float, float, Any]], duration: float) -> dict[str, Any]:
    """verbose_json from each piece's (start s, end s, onnx-asr result): one segment per sentence, and words."""
    segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []
    for start, end, result in pieces:
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            continue
        words = _words(result, start, end)
        if not words:
            logprobs = [float(p) for p in (getattr(result, "logprobs", None) or [])]
            segments.append(_segment(len(segments), [], start=start, end=end, text=text, logprobs=logprobs))
            continue
        all_words.extend(words)
        sentence: list[dict[str, Any]] = []
        for word in words:
            sentence.append(word)
            if _ends_sentence(word["word"]):
                segments.append(_segment(len(segments), sentence))
                sentence = []
        if sentence:
            segments.append(_segment(len(segments), sentence))
    return {
        "task": "transcribe",
        "language": "en",
        "duration": round(duration, 3),
        "text": " ".join(segment["text"].strip() for segment in segments),
        "segments": segments,
        "words": [
            {"word": word["word"], "start": round(word["start"], 3), "end": round(word["end"], 3)}
            for word in all_words
        ],
    }


# --- Backend ------------------------------------------------------------------------------------------------------


class ParakeetBackend:
    """STT backend for NVIDIA Parakeet TDT through onnx-asr, with FasterWhisperBackend's locking."""

    name = "parakeet"

    def __init__(self, score_speech_fn: Callable[[np.ndarray], list[float]] | None = None) -> None:
        self._models: dict[str, Any] = {}  # model_id -> onnx-asr adapter with timestamps
        self._runtime_of: dict[str, tuple[str, str]] = {}  # model_id -> (device, compute type)
        self._loaded_at: dict[str, float] = {}
        self._last_used: dict[str, float] = {}
        self._active: set[str] = set()
        # As FasterWhisperBackend: _lock guards the state above only; _inference_lock serializes load, unload and
        # inference. Take it before _lock, never after.
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._fetches: dict[str, concurrent.futures.Future] = {}
        self._fetched_paths: dict[str, str] = {}
        self._score_speech = score_speech_fn or score_speech

    # --- Runtime ---

    @staticmethod
    def _onnx_asr():
        import onnx_asr

        return onnx_asr

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        return "CUDAExecutionProvider" in ort.get_available_providers()

    def runtime(self) -> tuple[str, str]:
        """("cuda", "float16") when STT_DEVICE is cuda or auto and ONNX Runtime has CUDA; else ("cpu", "int8")."""
        device = (settings.stt_device or "cpu").strip().lower()
        if device in ("cuda", "auto") or device.startswith("cuda:"):
            if self._cuda_available():
                return "cuda", "float16"
            if device != "auto":
                logger.warning("STT_DEVICE=%s but ONNX Runtime has no CUDA provider: Parakeet runs int8 on the CPU",
                               device)
        return "cpu", "int8"

    def _providers(self, device: str) -> list:
        if device != "cuda":
            return ["CPUExecutionProvider"]
        options: dict[str, Any] = {
            # EXHAUSTIVE searches again for every new input length (~0.7 s each).
            "cudnn_conv_algo_search": "HEURISTIC",
            # The default doubles the arena on each growth, past the cap's intent.
            "arena_extend_strategy": "kSameAsRequested",
        }
        configured = (settings.stt_device or "").strip().lower()
        if configured.startswith("cuda:") and configured[5:].isdigit():
            options["device_id"] = int(configured[5:])
        if settings.stt_parakeet_gpu_mem_limit_mb > 0:
            options["gpu_mem_limit"] = settings.stt_parakeet_gpu_mem_limit_mb * 1024 * 1024
        return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]

    # --- Files ---

    @staticmethod
    def _download_root() -> str | None:
        return settings.stt_model_dir or None

    def _cache_dir(self) -> Path:
        return Path(settings.stt_model_dir) if settings.stt_model_dir else hf_hub_cache_dir()

    def _repo_dir(self, model_id: str) -> Path:
        return self._cache_dir() / ("models--" + PARAKEET_MODELS[model_id].onnx_repo.replace("/", "--"))

    @staticmethod
    def _spec(model_id: str) -> ParakeetModel:
        refuse_network_model_path(model_id)
        spec = PARAKEET_MODELS.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown Parakeet model {model_id!r}")
        return spec

    def _ready_files(self, model_id: str, device: str) -> Path | None:
        """The folder a load reads, when every file it needs is in the cache already (no network needed)."""
        spec = PARAKEET_MODELS[model_id]
        repo_dir = self._repo_dir(model_id)
        if device == "cuda":
            return parakeet_fp16.cached_fp16_dir(repo_dir, spec.revision)
        snapshot = repo_dir / "snapshots" / spec.revision
        return snapshot if all((snapshot / name).is_file() for name in INT8_FILES) else None

    def _fetch(self, model_id: str) -> str:
        return self.start_fetch(model_id).result()

    def start_fetch(self, model_id: str) -> concurrent.futures.Future:
        """Start fetching (and on a GPU converting) a model's files, or join the fetch running; the future gives the
        folder its load reads. One fetch per model at a time, in this module's pool, outside every lock."""
        self._spec(model_id)
        done: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            future = self._fetches.get(model_id)
            if future is not None:
                return future
            path = self._fetched_paths.get(model_id)
            if path is not None and Path(path).is_dir():
                done.set_result(path)
                return done
            future = _fetch_executor.submit(self._download, model_id)
            self._fetches[model_id] = future
        return future

    def _download(self, model_id: str) -> str:
        try:
            device, _ = self.runtime()
            path = self._ready_files(model_id, device)
            if path is None:
                path = self._download_files(model_id, device)
            with self._lock:
                self._fetched_paths[model_id] = str(path)
            return str(path)
        finally:
            with self._lock:
                self._fetches.pop(model_id, None)

    def _download_files(self, model_id: str, device: str) -> Path:
        from huggingface_hub import snapshot_download

        spec = PARAKEET_MODELS[model_id]
        files = parakeet_fp16.SOURCE_FILES if device == "cuda" else INT8_FILES
        logger.info("Downloading %s (%s files of %s)", model_id, "fp32" if device == "cuda" else "int8",
                    spec.onnx_repo)
        snapshot = Path(snapshot_download(
            spec.onnx_repo, revision=spec.revision, allow_patterns=list(files), cache_dir=self._download_root(),
        ))
        if device != "cuda":
            return snapshot
        return parakeet_fp16.ensure_fp16(snapshot, self._repo_dir(model_id), spec.revision)

    def download_model(self, model_id: str) -> bool:
        if not self.is_model_loaded(model_id):
            self._fetch(model_id)
        return True

    # --- Load / unload ---

    def load_model(self, model_id: str, _model_path: str | None = None) -> None:
        spec = self._spec(model_id)
        if self.is_model_loaded(model_id):
            return
        model_path = _model_path or self._fetch(model_id)
        with self._inference_lock:
            if model_id in self._models:
                return
            device, compute_type = self.runtime()
            onnx_asr = self._onnx_asr()
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.log_severity_level = 3  # ORT's warnings about nodes it places on the CPU, on every load
            logger.info("Loading model %s (device=%s, compute=%s)", model_id, device, compute_type)
            model = onnx_asr.load_model(
                spec.onnx_asr_name,
                model_path,
                quantization="fp16" if device == "cuda" else "int8",
                sess_options=options,
                providers=self._providers(device),
            ).with_timestamps()
            with self._lock:
                self._models[model_id] = model
                self._runtime_of[model_id] = (device, compute_type)
                self._loaded_at[model_id] = time.time()
                self._last_used[model_id] = time.time()
                self._fetched_paths.pop(model_id, None)
            logger.info("Model %s loaded successfully", model_id)

    def unload_model(self, model_id: str) -> None:
        with self._inference_lock:
            with self._lock:
                if model_id not in self._models:
                    return
                del self._models[model_id]
                self._runtime_of.pop(model_id, None)
                self._loaded_at.pop(model_id, None)
                self._last_used.pop(model_id, None)
            # The sessions, and with them their CUDA arenas, go when the last reference does.
            gc.collect()
            logger.info("Model %s unloaded", model_id)

    def unload_if_idle(self, model_id: str) -> bool:
        if not self._inference_lock.acquire(blocking=False):
            return False
        try:
            self.unload_model(model_id)
        finally:
            self._inference_lock.release()
        return True

    def loaded_models(self) -> list[LoadedModelInfo]:
        now = time.time()
        ttl = settings.stt_model_ttl
        default_model = settings.stt_default_model
        pinned = default_model if settings.os_pin_default_models else None
        with self._lock:
            last_used = {**self._last_used, **dict.fromkeys(self._active, now)}
            return [
                LoadedModelInfo(
                    model=mid,
                    backend=self.name,
                    device=self._runtime_of.get(mid, ("cpu", "int8"))[0],
                    compute_type=self._runtime_of.get(mid, ("cpu", "int8"))[1],
                    loaded_at=self._loaded_at[mid],
                    last_used_at=last_used.get(mid),
                    is_default=(mid == default_model),
                    ttl_remaining=(
                        None if ttl == 0 or mid == pinned
                        else max(0.0, ttl - (now - last_used.get(mid, now)))
                    ),
                )
                for mid in self._models
            ]

    def is_model_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._models

    # --- Cache ---

    def list_cached_models(self) -> list[dict[str, Any]]:
        with self._lock:
            loaded = set(self._models)
        results = []
        for model_id in PARAKEET_MODELS:
            repo_dir = self._repo_dir(model_id)
            if not repo_dir.is_dir() and model_id not in loaded:
                continue
            results.append({
                "model": model_id,
                "backend": self.name,
                "loaded": model_id in loaded,
                "is_default": model_id == settings.stt_default_model,
                "size_mb": round(_disk_size_mb(repo_dir), 1) if repo_dir.is_dir() else 0,
            })
        return results

    def is_model_cached(self, model_id: str) -> bool:
        return model_id in PARAKEET_MODELS and self._repo_dir(model_id).is_dir()

    def delete_cached_model(self, model_id: str) -> bool:
        if model_id not in PARAKEET_MODELS:
            return False
        with self._lock:
            self._fetched_paths.pop(model_id, None)
        repo_dir = self._repo_dir(model_id)
        if not repo_dir.is_dir():
            return False
        shutil.rmtree(repo_dir)  # the source files and the fp16 folders made from them
        logger.info("Deleted cached model %s from %s", model_id, repo_dir)
        return True

    # --- Inference ---

    def _ensure_model(self, model_id: str, model_path: str | None = None) -> Any:
        if model_id not in self._models:
            self.load_model(model_id, model_path)
        with self._lock:
            self._last_used[model_id] = time.time()
            return self._models[model_id]

    @contextlib.contextmanager
    def _in_use(self, model_id: str) -> Iterator[None]:
        with self._lock:
            self._active.add(model_id)
        try:
            yield
        finally:
            with self._lock:
                self._active.discard(model_id)
                if model_id in self._models:
                    self._last_used[model_id] = time.time()

    def _pieces(self, samples: np.ndarray) -> list[tuple[int, int]]:
        if len(samples) <= SINGLE_PIECE_MAX_S * SAMPLE_RATE:
            return [(0, len(samples))] if len(samples) else []
        try:
            probs = self._score_speech(samples)
        except Exception as exc:
            logger.warning("Silero VAD is unavailable, so long audio is cut every %g s: %s", MAX_PIECE_S, exc)
            return even_pieces(len(samples))
        return plan_pieces(probs, len(samples), threshold=settings.stt_vad_threshold)

    @staticmethod
    def _batches(lengths: list[int]) -> list[list[int]]:
        """Piece indexes in batches, shortest first so a batch pads little: up to BATCH_SIZE pieces, and no more
        than BATCH_MAX_S of padded audio (the batch's longest piece times its count)."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batches: list[list[int]] = []
        for i in order:
            batch = batches[-1] if batches else None
            if (batch is None or len(batch) >= BATCH_SIZE
                    or (len(batch) + 1) * lengths[i] > BATCH_MAX_S * SAMPLE_RATE):
                batches.append([i])
            else:
                batch.append(i)
        return batches

    @classmethod
    def _recognize(cls, model: Any, waveforms: list[np.ndarray]) -> list[Any]:
        """Decode pieces in batches (_batches); one at a time after a batch fails (the arena cap reached)."""
        results: list[Any] = [None] * len(waveforms)
        for batch in cls._batches([len(w) for w in waveforms]):
            try:
                decoded = model.recognize([waveforms[i] for i in batch], sample_rate=SAMPLE_RATE)
            except Exception as exc:
                if len(batch) == 1:
                    raise
                logger.warning("A batch of %d pieces failed (%s); decoding them one at a time", len(batch),
                               str(exc).splitlines()[0][-200:])
                decoded = [model.recognize([waveforms[i]], sample_rate=SAMPLE_RATE)[0] for i in batch]
            for i, result in zip(batch, decoded):
                results[i] = result
        return results

    def transcribe(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
        temperature_fallback: bool = True,
    ) -> dict[str, Any]:
        """Transcribe English audio. ``temperature``, ``temperature_fallback`` and ``prompt`` are accepted and
        ignored: the TDT decoder is greedy and takes no prompt."""
        self._spec(model)
        check_language(model, language)
        samples = decode_audio(audio)
        duration = len(samples) / SAMPLE_RATE
        model_path = None if self.is_model_loaded(model) else self._fetch(model)
        with self._inference_lock, self._in_use(model):
            asr = self._ensure_model(model, model_path)
            pieces = self._pieces(samples)
            results = self._recognize(asr, [samples[lo:hi] for lo, hi in pieces]) if pieces else []
        result = build_result(
            [(lo / SAMPLE_RATE, hi / SAMPLE_RATE, r) for (lo, hi), r in zip(pieces, results)], duration,
        )
        if response_format == "verbose_json":
            return result
        if response_format == "text":
            return {"text": result["text"], "raw_text": True}
        if response_format == "srt":
            return {"text": self._to_srt(result["segments"]), "raw_text": True}
        if response_format == "vtt":
            return {"text": self._to_vtt(result["segments"]), "raw_text": True}
        return {"text": result["text"]}

    def translate(
        self,
        audio: bytes,
        model: str,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        raise TranslationUnsupportedError(f"{model} can't translate: it transcribes English only.")

    @staticmethod
    def _to_srt(segments: list[dict]) -> str:
        return "\n".join(
            f"{i}\n{_fmt_time_srt(s['start'])} --> {_fmt_time_srt(s['end'])}\n{s['text'].strip()}\n"
            for i, s in enumerate(segments, 1)
        )

    @staticmethod
    def _to_vtt(segments: list[dict]) -> str:
        lines = ["WEBVTT\n"]
        lines.extend(f"{_fmt_time_vtt(s['start'])} --> {_fmt_time_vtt(s['end'])}\n{s['text'].strip()}\n" for s in segments)
        return "\n".join(lines)
