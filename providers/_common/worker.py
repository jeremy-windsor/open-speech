"""Scaffolding shared by the isolated GPU TTS workers (worker contract v1).

Each worker keeps its model runtime and streaming strategy in ``providers/<name>/app.py``.
This module holds what they share: request models, typed errors and their HTTP envelope,
reference-audio checks, the reference-conditioning cache, CUDA housekeeping, /health,
model load/unload, float32le streaming, sentence rules, and language-tag lookup.

Model libraries (torch, soundfile) are imported where they are used, so tests can stub them.
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import gc
import hashlib
import io
import logging
import os
import re
import string
import tempfile
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, TypeVar

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

SCHEMA_VERSION = 1
AUDIO_FORMAT = "float32le"
OUT_OF_MEMORY_CODE = "out_of_memory"
# 507 Insufficient Storage: the GPU, not the request, lacks room; the core maps it to a
# typed insufficient_gpu_memory error.
OUT_OF_MEMORY_STATUS = 507
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_REFERENCE_CACHE_SIZE = 8

CJK_SENTENCE_ENDINGS = frozenset("。！？")
LATIN_SENTENCE_ENDINGS = frozenset(".!?")
SENTENCE_CLOSERS = frozenset("」』”’）)》〉】]〉〕〗〙〛〞〟\"'")
# Copied from the harness's sentence rules (src/live_tts/segmenter.py), which a worker
# cannot import: a period after one of these words, a single letter (initials, "e.g."),
# or a bare list number does not end a sentence.
NONTERMINAL_ABBREVIATIONS = frozenset(
    {"dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs"}
)
_ASCII_LETTERS = frozenset(string.ascii_letters)
_LIST_NUMBER = re.compile(r"\s*\d+")

# uvicorn configures this logger, so the message reaches `docker logs`.
logger = logging.getLogger("uvicorn.error")

T = TypeVar("T")


def env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in TRUE_VALUES


class WorkerFailure(RuntimeError):
    def __init__(self, message: str, *, code: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def error_detail(exc: WorkerFailure) -> dict[str, str]:
    return {"code": exc.code, "message": str(exc)}


class OutOfMemoryHTTPException(HTTPException):
    """Rendered as HTTP 507 with ``{"error": {"code": "out_of_memory", "message": ...}}``."""


def http_error(exc: WorkerFailure) -> HTTPException:
    if exc.code == OUT_OF_MEMORY_CODE:
        return OutOfMemoryHTTPException(exc.status_code, detail=error_detail(exc))
    return HTTPException(exc.status_code, detail=error_detail(exc))


def ensure_idle(lock: asyncio.Lock) -> None:
    if lock.locked():
        raise HTTPException(429, detail={"code": "worker_busy", "message": "Worker is busy"})


def unknown_model_error(model_id: str) -> HTTPException:
    return HTTPException(
        400, detail={"code": "unknown_model", "message": f"Unknown model: {model_id}"}
    )


def create_app(title: str) -> FastAPI:
    app = FastAPI(title=title, version="1")

    @app.exception_handler(OutOfMemoryHTTPException)
    async def out_of_memory(_request: Request, exc: OutOfMemoryHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    return app


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str


class SynthesisRequest(BaseModel):
    """Worker contract v1 synthesis body; each worker narrows text, voice, and speed."""

    model_config = ConfigDict(extra="forbid")
    model: str
    text: str = Field(min_length=1, max_length=4096)
    voice: str = "reference"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str | None = None
    instructions: str | None = None
    voice_design: str | None = None
    reference_audio: str | None = None
    clone_transcript: str | None = None


def decode_reference_audio(encoded: str | None, *, max_bytes: int, missing_message: str) -> bytes:
    if not encoded:
        raise WorkerFailure(missing_message, code="reference_audio_required", status_code=400)
    try:
        audio = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise WorkerFailure(
            "reference_audio must be valid base64",
            code="invalid_reference_audio",
            status_code=400,
        ) from exc
    if not audio or len(audio) > max_bytes:
        raise WorkerFailure(
            "reference_audio is empty or exceeds the worker limit",
            code="invalid_reference_audio",
            status_code=413,
        )
    return audio


def reference_audio_info(audio: bytes) -> Any:
    """Return soundfile's info for reference audio sampled at 16 kHz or higher."""
    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(audio))
    except Exception as exc:
        raise WorkerFailure(
            "reference_audio is not a supported audio file",
            code="invalid_reference_audio",
            status_code=400,
        ) from exc
    if info.frames <= 0 or info.samplerate < 16000:
        raise WorkerFailure(
            "reference_audio must contain speech sampled at 16 kHz or higher",
            code="invalid_reference_audio",
            status_code=400,
        )
    return info


@contextmanager
def reference_file(audio: bytes) -> Iterator[str]:
    """Hold reference audio in a temporary WAV path for libraries that only read files."""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio)
            temp_path = temp_file.name
        yield temp_path
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def float32_audio(value: Any, *, message: str = "Model returned invalid audio") -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all():
        raise WorkerFailure(message, code="invalid_audio")
    return np.clip(audio, -1.0, 1.0).astype("<f4", copy=False)


class ReferenceCache:
    """Small LRU of per-reference model conditioning, cleared when the model unloads.

    Keys hash the model ID, the reference audio, and the text the conditioning was built
    from, so a changed transcript or recording never reuses stale conditioning.
    """

    def __init__(self, size: int = DEFAULT_REFERENCE_CACHE_SIZE) -> None:
        self.size = max(1, size)
        self._entries: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def key(model_id: str | None, audio: bytes, text: str = "") -> str:
        digest = hashlib.sha256()
        digest.update((model_id or "").encode())
        digest.update(audio)
        digest.update(hashlib.sha256(text.encode()).digest())
        return digest.hexdigest()

    def get(self, key: str) -> Any | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> list[Any]:
        """Store ``value`` and return the entries evicted to stay within the size."""
        self._entries[key] = value
        self._entries.move_to_end(key)
        evicted = []
        while len(self._entries) > self.size:
            evicted.append(self._entries.popitem(last=False)[1])
        return evicted

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# sm_86 is 8.6 and sm_120 is 12.0: the last digit is the minor version.
_CUDA_ARCH = re.compile(r"(sm|compute)_(\d+)(\d)(a|f)?")


def _arch_runs_on(arch: str, capability: tuple[int, int]) -> bool:
    """Whether code built for one ``torch.cuda.get_arch_list()`` entry runs on this GPU.

    A cubin for X.y (``sm_XY``) runs on any GPU of capability X.z with z >= y (CUDA C
    Programming Guide, Binary Compatibility), so official wheels list one per family: no
    sm_89 or sm_61, yet they run on 8.9 and 6.1. torch's own ``_check_cubins`` applies the
    same rule. PTX (``compute_XY``) is JIT-compiled onto any equal or newer GPU. A
    family-specific entry (``sm_100f``) stays within its major version, and an
    arch-specific one (``sm_90a``, ``compute_90a``) runs only on its exact capability.
    """
    match = _CUDA_ARCH.fullmatch(arch.strip())
    if match is None:
        return False
    kind, major, minor, suffix = match.groups()
    built = (int(major), int(minor))
    device = (int(capability[0]), int(capability[1]))
    if suffix == "a":
        return built == device
    if kind == "sm" or suffix == "f":
        return built[0] == device[0] and built[1] <= device[1]
    return built <= device


def arch_failure(capability: tuple[int, int], arch_list: Sequence[str]) -> WorkerFailure | None:
    """The failure every load gets on this GPU when the torch build has no code for it."""
    if any(_arch_runs_on(arch, capability) for arch in arch_list):
        return None
    return WorkerFailure(
        f"Torch wheel has no code that runs on sm_{capability[0]}{capability[1]} "
        f"(built for: {' '.join(arch_list) or 'none'})",
        code="cuda_arch_unsupported",
        status_code=503,
    )


def require_cuda() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        raise WorkerFailure("CUDA is unavailable", code="cuda_unavailable", status_code=503)
    capability = torch.cuda.get_device_capability(0)
    failure = arch_failure(capability, list(torch.cuda.get_arch_list()))
    if failure is not None:
        raise failure
    return capability


def release_memory() -> None:
    """Return an unloaded model's memory to the GPU driver and the OS."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        # cuBLAS keeps a workspace per handle as a live block, and empty_cache() frees only
        # segments with no live block. Qwen3's weights sit in one model-sized warmup segment,
        # so its workspace alone kept about 3.5 GB reserved after an unload. torch's own leak
        # checker clears these workspaces the same way before emptying the cache.
        clear_workspaces = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
        if callable(clear_workspaces):
            clear_workspaces()
        torch.cuda.empty_cache()
    if os.name == "posix":
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim(0)


def is_out_of_memory(exc: BaseException) -> bool:
    import torch

    error_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(error_type, type) and isinstance(exc, error_type):
        return True
    # Some CUDA paths still raise a plain RuntimeError with the allocator's message.
    return isinstance(exc, RuntimeError) and "cuda out of memory" in str(exc).lower()


def guard_out_of_memory(call: Callable[[], T], message: str) -> T:
    """Run a model call; CUDA out-of-memory becomes a typed 507 and the worker stays healthy."""
    try:
        return call()
    except Exception as exc:
        if not is_out_of_memory(exc):
            raise
        logger.warning("%s: %s", message, exc)
    # The failing frames, and the tensors they held, are gone once the except block ends,
    # so emptying the cache now hands that memory back to other GPU processes.
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    raise WorkerFailure(message, code=OUT_OF_MEMORY_CODE, status_code=OUT_OF_MEMORY_STATUS)


class WorkerRuntime:
    """The worker's one loaded model, its timings, and its reference-conditioning cache."""

    display_name = "the model"
    # Each model's GPU memory once loaded, in MiB, as the manifest gives it (gpu_memory_mb); /health repeats the loaded
    # model's (gpu_memory.need_mb) next to what it holds (gpu_memory.model_mb).
    gpu_memory_need_mb: Mapping[str, int] = MappingProxyType({})

    def __init__(self, *, reference_cache_size: int = DEFAULT_REFERENCE_CACHE_SIZE) -> None:
        self.model: Any | None = None
        self.model_id: str | None = None
        self.loaded_at: float | None = None
        self.last_used_at: float | None = None
        self.load_seconds: float | None = None
        self.references = ReferenceCache(reference_cache_size)

    def is_known_model(self, model_id: str) -> bool:
        raise NotImplementedError

    def build_model(self, model_id: str, capability: tuple[int, int]) -> Any:
        """Load and return the model's weights; called after CUDA has been checked."""
        raise NotImplementedError

    def check_usable(self) -> None:
        """Raise a WorkerFailure when this process can no longer run CUDA work."""

    def after_load(self) -> None:
        """Record diagnostics once the new model is in place."""

    def load_failure(self, exc: Exception) -> WorkerFailure | None:
        """Return a failure to raise without unloading, or None for the generic path."""
        return None

    def load(self, model_id: str) -> None:
        if not self.is_known_model(model_id):
            raise WorkerFailure(f"Unknown model: {model_id}", code="unknown_model", status_code=400)
        self.check_usable()
        if self.model_id == model_id and self.model is not None:
            return
        self.unload()
        capability = require_cuda()
        started = time.perf_counter()
        try:
            model = guard_out_of_memory(
                lambda: self.build_model(model_id, capability),
                f"CUDA out of memory while loading {self.display_name}",
            )
            self.model = model
            self.model_id = model_id
            self.load_seconds = time.perf_counter() - started
            self.loaded_at = time.time()
            self.last_used_at = None
            self.after_load()
        except WorkerFailure:
            self.unload()
            raise
        except Exception as exc:
            failure = self.load_failure(exc)
            if failure is not None:
                raise failure from exc
            self.unload()
            raise WorkerFailure(
                f"Failed to load {model_id}: {exc}",
                code="model_load_failed",
            ) from exc

    def unload(self) -> None:
        self.model = None
        self.model_id = None
        self.loaded_at = None
        self.last_used_at = None
        self.load_seconds = None
        self.references.clear()
        release_memory()

    def require_model(self, model_id: str) -> Any:
        if self.model is None or self.model_id != model_id:
            raise WorkerFailure(
                f"Model {model_id} is not loaded",
                code="model_not_loaded",
                status_code=409,
            )
        return self.model


def gpu_health(
    *, query: bool = True
) -> tuple[dict[str, Any], tuple[int, int] | None, WorkerFailure | None]:
    """Return the /health gpu block, the device capability, and any failure.

    The failure is a query error, or ``cuda_arch_unsupported`` when the torch build has no code
    for the GPU: every load would get that one from ``require_cuda()``, so health fails too.
    """
    import torch

    available = torch.cuda.is_available()
    gpu: dict[str, Any] = {"available": available}
    if not available or not query:
        return gpu, None, None
    try:
        capability = torch.cuda.get_device_capability(0)
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "capability": list(capability),
                "arch_list": torch.cuda.get_arch_list(),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
            }
        )
    except Exception as exc:
        logger.warning("CUDA health query failed: %s", exc)
        gpu["error"] = "CUDA runtime details are unavailable"
        failure = WorkerFailure("CUDA runtime query failed", code="cuda_query_failed", status_code=503)
        return gpu, None, failure
    return gpu, capability, arch_failure(capability, list(gpu["arch_list"]))


def health_body(
    provider: str,
    runtime: WorkerRuntime,
    *,
    status: str,
    busy: bool,
    gpu: dict[str, Any],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    loaded = runtime.model_id
    return {
        "status": status,
        "provider": provider,
        "loaded_model": loaded,
        "device": "cuda:0" if gpu["available"] else "unavailable",
        **details,
        "loaded_at": runtime.loaded_at,
        "last_used_at": runtime.last_used_at,
        "load_seconds": runtime.load_seconds,
        "busy": busy,
        "gpu": gpu,
        # The GPU memory the loaded model holds (torch's reserved memory, in MiB) and the model's expected need: the
        # core's GPU memory guard counts them when it makes room for another load. Both null with no model loaded.
        "gpu_memory": {
            "model_mb": gpu.get("memory_reserved_mb") if loaded else None,
            "need_mb": runtime.gpu_memory_need_mb.get(loaded) if loaded else None,
        },
    }


def health_payload(
    provider: str,
    runtime: WorkerRuntime,
    *,
    busy: bool,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gpu, _capability, failure = gpu_health()
    if failure is not None:
        status = "failed"
    else:
        status = "ok" if gpu["available"] else "unavailable"
    return health_body(
        provider,
        runtime,
        status=status,
        busy=busy,
        gpu=gpu,
        details={**(details or {}), "failure": error_detail(failure) if failure else None},
    )


async def load_model(
    runtime: WorkerRuntime,
    lock: asyncio.Lock,
    model_id: str,
    *,
    on_failure: Callable[[WorkerFailure], None] | None = None,
) -> dict[str, Any]:
    ensure_idle(lock)
    async with lock:
        try:
            await asyncio.to_thread(runtime.load, model_id)
        except WorkerFailure as exc:
            if on_failure is not None:
                on_failure(exc)
            raise http_error(exc) from exc
    return {"status": "loaded", "model": model_id, "load_seconds": runtime.load_seconds}


async def unload_model(runtime: WorkerRuntime, lock: asyncio.Lock, model_id: str) -> dict[str, str]:
    ensure_idle(lock)
    async with lock:
        if runtime.model_id and runtime.model_id != model_id:
            raise HTTPException(
                404,
                detail={"code": "model_not_loaded", "message": f"Model {model_id} is not loaded"},
            )
        await asyncio.to_thread(runtime.unload)
    return {"status": "unloaded", "model": model_id}


def audio_response(
    body: AsyncIterator[bytes],
    *,
    sample_rate: int,
    cancellation: str,
    background: BackgroundTask,
    headers: Mapping[str, str] | None = None,
) -> StreamingResponse:
    return StreamingResponse(
        body,
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": AUDIO_FORMAT,
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Cancellation-Granularity": cancellation,
            **(headers or {}),
        },
        background=background,
    )


def next_chunk(iterator: Iterator[np.ndarray]) -> tuple[bool, np.ndarray | None]:
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


async def stream_chunks(
    chunks: Generator[np.ndarray, None, None],
    *,
    lock: asyncio.Lock,
    request: Request,
    sample_rate: int,
    cancellation: str,
    headers: Mapping[str, str] | None = None,
) -> StreamingResponse:
    """Stream audio chunks that the model produces one step at a time on worker threads.

    The caller holds ``lock``; it is released exactly once, when the stream ends or fails.
    The first chunk is produced before the response starts, so its failure is a typed
    HTTP error. Later steps finish even after a disconnect, so the worker stays busy
    until the model thread is idle again.
    """
    lock_released = False

    def release_lock_once() -> None:
        nonlocal lock_released
        if lock_released:
            return
        lock_released = True
        lock.release()

    try:
        finished, first = await asyncio.to_thread(next_chunk, chunks)
    except WorkerFailure as exc:
        release_lock_once()
        raise http_error(exc) from exc
    except BaseException:
        release_lock_once()
        raise
    if finished or first is None:
        release_lock_once()
        raise HTTPException(
            500, detail={"code": "invalid_audio", "message": "Model returned no audio"}
        )

    async def body() -> AsyncIterator[bytes]:
        try:
            yield first.tobytes()
            while not await request.is_disconnected():
                # A disconnect cancels this generator but not the model thread, so finish
                # the current chunk while the lock still marks the worker busy.
                with anyio.CancelScope(shield=True):
                    done, chunk = await asyncio.to_thread(next_chunk, chunks)
                if done:
                    break
                assert chunk is not None
                yield chunk.tobytes()
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.to_thread(chunks.close)
                finally:
                    release_lock_once()

    async def close_stream() -> None:
        release_lock_once()

    return audio_response(
        body(),
        sample_rate=sample_rate,
        cancellation=cancellation,
        headers=headers,
        background=BackgroundTask(close_stream),
    )


def period_is_nonterminal(text: str, index: int, start: int = 0) -> bool:
    """Whether the period at ``index`` follows an abbreviation, an initial, or a list number.

    ``start`` is where the current sentence begins: like the harness, a number counts as a
    list number when it starts its line or its sentence ("Add the flour. 2. Mix well.").
    """
    line = text[max(text.rfind("\n", 0, index) + 1, start) : index]
    if _LIST_NUMBER.fullmatch(line):
        return True
    word_start = len(line)
    while word_start and line[word_start - 1] in _ASCII_LETTERS:
        word_start -= 1
    word = line[word_start:].lower()
    return len(word) == 1 or word in NONTERMINAL_ABBREVIATIONS


def sentence_parts(text: str) -> list[str]:
    """Split text after each sentence, keeping closing quotes and brackets with it.

    A CJK full stop always ends a sentence. A Latin '.', '!' or '?' ends one only before
    whitespace, and a period after "Mr", "Dr", an initial, or a list number does not.
    A fragment without letters or digits joins the part before it.
    """
    result: list[str] = []

    def append_part(part: str) -> None:
        normalized = part.strip()
        if not normalized:
            return
        if result and not any(character.isalnum() for character in normalized):
            result[-1] += normalized
        else:
            result.append(normalized)

    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character not in CJK_SENTENCE_ENDINGS and character not in LATIN_SENTENCE_ENDINGS:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in SENTENCE_CLOSERS:
            end += 1
        if character in CJK_SENTENCE_ENDINGS:
            split_here = True
        else:
            split_here = (
                end < len(text)
                and text[end].isspace()
                and not (character == "." and period_is_nonterminal(text, index, start))
            )
        if not split_here:
            index = end
            continue
        append_part(text[start:end])
        while end < len(text) and text[end].isspace():
            end += 1
        start = end
        index = end

    append_part(text[start:])
    return result


def language_name(hint: str | None, names: Mapping[str, str]) -> str | None:
    """Map a language hint to a model language name, or None when it is unsupported.

    The whole tag is tried first, then its primary subtag (RFC 4647 lookup), so regional
    tags such as zh-CN or pt-BR reach the model as their language.
    """
    key = (hint or "").strip().lower().replace("_", "-")
    return names.get(key) or names.get(key.split("-", 1)[0])
