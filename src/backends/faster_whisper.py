"""faster-whisper backend using ctranslate2."""

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import io
import logging
import shutil
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Iterator

from src.config import hf_hub_cache_dir, settings
from src.formatters import _fmt_time_srt, _fmt_time_vtt
from src.models import LoadedModelInfo

logger = logging.getLogger(__name__)

DISTILLED_ENGLISH_LONG_FORM_MODELS = frozenset({
    "Systran/faster-distil-whisper-small.en",
    "Systran/faster-distil-whisper-medium.en",
})
LONG_FORM_THRESHOLD_S = 30
LONG_FORM_CHUNK_S = 15
# OpenAI's temperature=0 means "start greedy, re-decode hotter when a segment fails
# the compression-ratio or log-probability thresholds"; a scalar disables the retries.
# Live paths turn the retries off (temperature_fallback=False): they decode every window
# once, as a window cut mid-word often fails the thresholds and would cost up to six decodes.
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Model files are fetched here, outside the STT inference pool and every STT lock, one fetch per model at a time, so
# a request or a load waits for its model's download without holding a decode thread or a queue. More first
# downloads than this at once wait here for a free thread.
_fetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="stt-fetch")


class NetworkModelPathError(ValueError):
    """A request named a network (UNC) or device path as its STT model.

    On Windows a stat of ``\\\\host\\share`` (or ``//host/share``) connects to that host over SMB and offers the server
    account's credentials, so such an id is refused before anything reads the file system. HTTP answers it ``400``
    with this code.
    """

    code = "invalid_model"


def is_network_or_device_path(model_id: str) -> bool:
    """Whether Windows reads the id as a UNC or device path: two leading slashes or backslashes in any mix
    (``\\\\host\\share``, ``//host/share``, ``\\\\?\\UNC\\...``, ``\\\\.\\pipe\\...``), or the NT prefix ``\\??\\``."""
    text = model_id.lstrip()
    separators = "\\/"
    if len(text) >= 2 and text[0] in separators and text[1] in separators:
        return True
    return len(text) >= 4 and text[0] in separators and text[1:3] == "??" and text[3] in separators


def refuse_network_model_path(model_id: str) -> None:
    """Raise NetworkModelPathError for a network or device path chosen by a request, before any stat of it.

    A model the operator configured (``STT_MODEL``, ``STT_TRANSLATE_MODEL``, ``STT_PRELOAD_MODELS``) is the
    operator's choice and is left as it is; a local folder is a model as before.
    """
    if not isinstance(model_id, str) or not is_network_or_device_path(model_id):
        return
    configured = {settings.stt_model, settings.stt_translate_model}
    configured.update(part.strip() for part in (settings.stt_preload_models or "").split(","))
    if model_id in configured:
        return
    raise NetworkModelPathError(
        f"Model {model_id!r} is a network or device path, which a request can't use. "
        "Use a model name or a local folder."
    )


def _disk_size_mb(path: Path) -> float:
    """MB a model directory takes on disk, counting each file once.

    The Hugging Face cache stores a file once in blobs/ and links it from every snapshot revision, so a file
    reached by several paths (a symlink or a hard link) is counted once, by its device and inode.
    """
    seen: set = set()
    total = 0
    for f in path.rglob("*"):
        try:
            if not f.is_file():
                continue
            st = f.stat()
        except OSError:
            continue  # removed while scanning, or a broken link
        key = (st.st_dev, st.st_ino) if st.st_ino else f
        if key not in seen:
            seen.add(key)
            total += st.st_size
    return total / (1024 * 1024)


def _wav_duration_seconds(audio: bytes) -> float | None:
    """Read the duration of a WAV upload without decoding it twice."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except (EOFError, ValueError, wave.Error, ZeroDivisionError):
        return None


class FasterWhisperBackend:
    """STT backend using faster-whisper (ctranslate2)."""

    name = "faster-whisper"

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}  # model_id -> WhisperModel
        self._loaded_at: dict[str, float] = {}
        self._last_used: dict[str, float] = {}
        # A long request would otherwise look idle to the TTL and LRU checks.
        self._active: set[str] = set()
        # _lock guards only the state above, so status reads (/health) never wait on a model.
        # _inference_lock serializes load, unload and inference, so an idle unload cannot
        # free a model in the middle of a request. Take it before _lock, never after.
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()
        # Under _lock: the fetch running for each model, and the files a finished fetch gave, until the model is
        # built from them or its cache is deleted.
        self._fetches: dict[str, concurrent.futures.Future] = {}
        self._fetched_paths: dict[str, str] = {}

    def load_model(self, model_id: str, _model_path: str | None = None) -> None:
        """Load a faster-whisper model into memory, downloading its files first if needed."""
        if self.is_model_loaded(model_id):
            logger.info("Model %s already loaded", model_id)
            return
        # The download runs before the inference lock: a first use (a 1.5 GB STT_TRANSLATE_MODEL) must not hold
        # every decode, load and unload of the other models for its length.
        model_path = _model_path or self._fetch(model_id)
        with self._inference_lock:
            if model_id in self._models:
                logger.info("Model %s already loaded", model_id)
                return

            from faster_whisper import WhisperModel

            logger.info("Loading model %s (device=%s, compute=%s)",
                         model_id, settings.stt_device, settings.stt_compute_type)

            model = WhisperModel(
                model_path,
                device=settings.stt_device,
                compute_type=settings.stt_compute_type,
                download_root=self._download_root(),
            )

            with self._lock:
                self._models[model_id] = model
                self._loaded_at[model_id] = time.time()
                self._last_used[model_id] = time.time()
                self._fetched_paths.pop(model_id, None)  # a later load checks the hub again
            logger.info("Model %s loaded successfully", model_id)

    @staticmethod
    def _download_root() -> str | None:
        # snapshot_download falls back to the Hugging Face cache only for None: an empty
        # STT_MODEL_DIR would put models--<id> folders in the working directory instead.
        return settings.stt_model_dir or None

    def _fetch(self, model_id: str) -> str:
        """Return the path of a model's files, fetching them into the cache first; takes no lock."""
        return self.start_fetch(model_id).result()

    def start_fetch(self, model_id: str) -> concurrent.futures.Future:
        """Start fetching a model's files into the cache, or join the fetch running; the future gives their path.

        One fetch per model at a time, in this module's fetch pool: a request for a model that is downloading waits
        for that download instead of starting its own, and an async caller can await the future without holding a
        thread. The fetch is not skipped for a model whose cache folder exists, as a download that stopped half way
        leaves one; on a complete cache it costs huggingface_hub one metadata request. Files fetched and not built
        into a model since are given at once.
        """
        refuse_network_model_path(model_id)  # before the stat below
        done: concurrent.futures.Future = concurrent.futures.Future()
        if Path(model_id).is_dir():
            done.set_result(model_id)  # WhisperModel reads a local directory as is
            return done
        with self._lock:
            future = self._fetches.get(model_id)
            if future is not None:
                return future
            path = self._fetched_paths.get(model_id)
            if path is not None and Path(path).is_dir():
                done.set_result(path)
                return done
            # Submitted and recorded under _lock, which _download takes before it forgets the fetch.
            future = _fetch_executor.submit(self._download, model_id)
            self._fetches[model_id] = future
        return future

    def _download(self, model_id: str) -> str:
        """The fetch start_fetch runs; it records its files before its future gives them, then forgets itself."""
        try:
            from faster_whisper.utils import download_model

            logger.info("Downloading model %s", model_id)
            path = download_model(model_id, cache_dir=self._download_root())
            with self._lock:
                self._fetched_paths[model_id] = path
            return path
        finally:
            with self._lock:
                self._fetches.pop(model_id, None)

    def download_model(self, model_id: str) -> bool:
        """Fetch a model's files into the cache without loading it into memory."""
        if not self.is_model_loaded(model_id):  # a loaded model is cached
            self._fetch(model_id)
        return True

    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        with self._inference_lock:
            with self._lock:
                if model_id not in self._models:
                    return
                del self._models[model_id]
                del self._loaded_at[model_id]
                self._last_used.pop(model_id, None)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Model %s unloaded", model_id)

    def unload_if_idle(self, model_id: str) -> bool:
        """Unload a model unless a request or a load is running; False when one is.

        TTL and LRU eviction use it, so they neither wait for a decode nor unload the model
        serving it. One inference lock serves every model, so any running decode counts.
        """
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
                    device=settings.stt_device,
                    compute_type=settings.stt_compute_type,
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

    # --- Cache management ---

    def _get_cache_dir(self) -> Path:
        """Return the HuggingFace hub cache directory used by faster-whisper."""
        if settings.stt_model_dir:
            return Path(settings.stt_model_dir)
        # Without a download root, faster-whisper downloads into the default HF hub cache.
        return hf_hub_cache_dir()

    def list_cached_models(self) -> list[dict[str, Any]]:
        """List models found in the HuggingFace cache directory."""
        cache_dir = self._get_cache_dir()
        results = []
        with self._lock:
            loaded_ids = set(self._models.keys())

        if not cache_dir.exists():
            # Still return loaded models (they may use a custom path)
            for mid in loaded_ids:
                results.append({
                    "model": mid,
                    "loaded": True,
                    "is_default": mid == settings.stt_default_model,
                    "size_mb": 0,
                })
            return results

        seen = set()

        if settings.stt_model_dir:
            # Custom model dir: models stored as models--Org--Name or directly
            for p in cache_dir.iterdir():
                if not p.is_dir() or p.name.startswith("."):
                    continue
                if p.name.startswith("models--"):
                    # Convert models--Org--Name -> Org/Name
                    parts = p.name.split("--", 2)
                    if len(parts) == 3:
                        model_id = f"{parts[1]}/{parts[2]}"
                    else:
                        model_id = p.name
                else:
                    model_id = p.name
                seen.add(model_id)
                size_mb = _disk_size_mb(p)
                results.append({
                    "model": model_id,
                    "loaded": model_id in loaded_ids,
                    "is_default": model_id == settings.stt_default_model,
                    "size_mb": round(size_mb, 1),
                })
        else:
            # Default HF cache: directories named models--Org--Name
            for p in cache_dir.iterdir():
                if not p.is_dir() or not p.name.startswith("models--"):
                    continue
                parts = p.name.split("--", 2)
                if len(parts) != 3:
                    continue
                model_id = f"{parts[1]}/{parts[2]}"
                seen.add(model_id)
                size_mb = _disk_size_mb(p)
                results.append({
                    "model": model_id,
                    "loaded": model_id in loaded_ids,
                    "is_default": model_id == settings.stt_default_model,
                    "size_mb": round(size_mb, 1),
                })

        # Include loaded models not found in cache scan
        for mid in loaded_ids:
            if mid not in seen:
                results.append({
                    "model": mid,
                    "loaded": True,
                    "is_default": mid == settings.stt_default_model,
                    "size_mb": 0,
                })

        return results

    def _find_cache_path(self, model_id: str) -> Path | None:
        """Find the cache directory for a model."""
        cache_dir = self._get_cache_dir()
        if not cache_dir.exists():
            return None

        def cached(p: Path) -> bool:
            # Only a direct child: a "..", or on Windows a backslash or drive, in the ID would
            # otherwise name a directory outside the cache for delete_cached_model to remove.
            return p.parent == cache_dir and p.name != ".." and p.exists()

        if settings.stt_model_dir:
            # Check models--Org--Name format
            safe_name = "models--" + model_id.replace("/", "--")
            p = cache_dir / safe_name
            if cached(p):
                return p
            # Check direct name
            p = cache_dir / model_id.split("/")[-1]
            if cached(p):
                return p
        else:
            safe_name = "models--" + model_id.replace("/", "--")
            p = cache_dir / safe_name
            if cached(p):
                return p
        return None

    def delete_cached_model(self, model_id: str) -> bool:
        """Delete a model from the cache. Returns True if found and deleted."""
        with self._lock:
            self._fetched_paths.pop(model_id, None)
        p = self._find_cache_path(model_id)
        if p and p.exists():
            shutil.rmtree(p)
            logger.info("Deleted cached model %s from %s", model_id, p)
            return True
        return False

    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model exists in the cache."""
        return self._find_cache_path(model_id) is not None

    def _ensure_model(self, model_id: str, model_path: str | None = None) -> Any:
        """Ensure model is loaded, auto-load if not."""
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

    def _run_inference(
        self,
        audio: bytes,
        model_id: str,
        task: str = "transcribe",
        language: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
        temperature_fallback: bool = True,
    ) -> dict[str, Any]:
        """Run transcription/translation inference."""
        # A first use downloads before the lock, as load_model does.
        model_path = None if self.is_model_loaded(model_id) else self._fetch(model_id)
        with self._inference_lock, self._in_use(model_id):
            whisper_model = self._ensure_model(model_id, model_path)

            # Close the input handle before the decoder reopens it on Windows.
            # Keep the directory alive until lazy segment iteration completes.
            with tempfile.TemporaryDirectory(prefix="open-speech-whisper-") as temp_dir:
                audio_path = Path(temp_dir) / "input.wav"
                audio_path.write_bytes(audio)

                fallback = temperature_fallback and not temperature
                kwargs: dict[str, Any] = {
                    "task": task,
                    "beam_size": 5,
                    "temperature": TEMPERATURE_FALLBACK if fallback else temperature,
                }
                if language and task == "transcribe":
                    kwargs["language"] = language
                if prompt:
                    kwargs["initial_prompt"] = prompt

                if (model_id in DISTILLED_ENGLISH_LONG_FORM_MODELS
                        and (duration := _wav_duration_seconds(audio)) is not None
                        and duration > LONG_FORM_THRESHOLD_S):
                    from faster_whisper import BatchedInferencePipeline

                    logger.info("Transcribing %.1fs with 15s VAD chunks: %s", duration, model_id)
                    pipeline = BatchedInferencePipeline(whisper_model)
                    segments_gen, info = pipeline.transcribe(
                        str(audio_path), chunk_length=LONG_FORM_CHUNK_S, batch_size=1,
                        **kwargs,
                    )
                else:
                    segments_gen, info = whisper_model.transcribe(str(audio_path), **kwargs)
                segments = list(segments_gen)

        # Build response based on format
        full_text = "".join(s.text for s in segments).strip()

        if response_format in ("verbose_json",):
            return {
                "task": task,
                # info.language is the language heard. OpenAI's translation object names the output's, always
                # English: "en" in this server's codes.
                "language": "en" if task == "translate" else info.language,
                "duration": info.duration,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "seek": int(s.seek),
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "tokens": list(s.tokens) if s.tokens else [],
                        "temperature": s.temperature,
                        "avg_logprob": s.avg_logprob,
                        "compression_ratio": s.compression_ratio,
                        "no_speech_prob": s.no_speech_prob,
                    }
                    for i, s in enumerate(segments)
                ],
            }
        elif response_format == "text":
            return {"text": full_text, "raw_text": True}
        elif response_format == "srt":
            return {"text": self._to_srt(segments), "raw_text": True}
        elif response_format == "vtt":
            return {"text": self._to_vtt(segments), "raw_text": True}
        else:
            # json (default)
            return {"text": full_text}

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
        return self._run_inference(
            audio, model, task="transcribe",
            language=language, response_format=response_format,
            temperature=temperature, prompt=prompt,
            temperature_fallback=temperature_fallback,
        )

    def translate(
        self,
        audio: bytes,
        model: str,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        return self._run_inference(
            audio, model, task="translate",
            response_format=response_format,
            temperature=temperature, prompt=prompt,
        )

    @staticmethod
    def _to_srt(segments: list) -> str:
        lines = []
        for i, s in enumerate(segments, 1):
            start = _fmt_time_srt(s.start)
            end = _fmt_time_srt(s.end)
            lines.append(f"{i}\n{start} --> {end}\n{s.text.strip()}\n")
        return "\n".join(lines)

    @staticmethod
    def _to_vtt(segments: list) -> str:
        lines = ["WEBVTT\n"]
        for s in segments:
            start = _fmt_time_vtt(s.start)
            end = _fmt_time_vtt(s.end)
            lines.append(f"{start} --> {end}\n{s.text.strip()}\n")
        return "\n".join(lines)
