"""Disposable Qwen3-TTS worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import os
import re
import tempfile
import time
from collections import OrderedDict
from typing import Any, AsyncIterator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask


PROVIDER = "qwen3"
SCHEMA_VERSION = 1
SAMPLE_RATE = 24000
MAX_REFERENCE_BYTES = int(os.environ.get("QWEN3_MAX_REFERENCE_MB", "100")) * 1024 * 1024
MAX_SEGMENT_CHARS = int(os.environ.get("QWEN3_MAX_SEGMENT_CHARS", "400"))
PROMPT_CACHE_SIZE = int(os.environ.get("QWEN3_PROMPT_CACHE_SIZE", "8"))
STREAM_QUEUE_TIMEOUT_S = float(os.environ.get("QWEN3_STREAM_QUEUE_TIMEOUT_S", "30"))

MODEL_IDS = {
    "qwen3/0.6b-custom-voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "qwen3/0.6b-base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
MODEL_REVISIONS = {
    "qwen3/0.6b-custom-voice": "85e237c12c027371202489a0ec509ded67b5e4b5",
    "qwen3/0.6b-base": "5d83992436eae1d760afd27aff78a71d676296fc",
}

VOICES = [
    {"id": "Vivian", "name": "Vivian", "language": "zh", "gender": "female"},
    {"id": "Serena", "name": "Serena", "language": "zh", "gender": "female"},
    {"id": "Uncle_Fu", "name": "Uncle Fu", "language": "zh", "gender": "male"},
    {"id": "Dylan", "name": "Dylan", "language": "zh", "gender": "male"},
    {"id": "Eric", "name": "Eric", "language": "zh", "gender": "male"},
    {"id": "Ryan", "name": "Ryan", "language": "en", "gender": "male"},
    {"id": "Aiden", "name": "Aiden", "language": "en", "gender": "male"},
    {"id": "Ono_Anna", "name": "Ono Anna", "language": "ja", "gender": "female"},
    {"id": "Sohee", "name": "Sohee", "language": "ko", "gender": "female"},
]

LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

COMMON_CAPABILITIES = {
    "voice_blend": False,
    "voice_design": False,
    "streaming": True,
    "instructions": False,
    "languages": LANGUAGES,
    "speed_control": False,
    "speed_mode": "none",
    "ssml": False,
    "batch": False,
    "cancellation": "between_segments",
    "live_reader": True,
}

MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "provider": PROVIDER,
    "models": [
        {
            "id": "qwen3/0.6b-custom-voice",
            "provider": PROVIDER,
            "sample_rate": SAMPLE_RATE,
            "revision": MODEL_REVISIONS["qwen3/0.6b-custom-voice"],
            "max_input_chars": 4096,
            "capabilities": {
                **COMMON_CAPABILITIES,
                "voice_clone": False,
                "reference_audio": False,
                "clone_transcript": False,
                "speakers": VOICES,
            },
            "voices": VOICES,
        },
        {
            "id": "qwen3/0.6b-base",
            "provider": PROVIDER,
            "sample_rate": SAMPLE_RATE,
            "revision": MODEL_REVISIONS["qwen3/0.6b-base"],
            "max_input_chars": 4096,
            "capabilities": {
                **COMMON_CAPABILITIES,
                "live_reader": False,
                "voice_clone": True,
                "reference_audio": True,
                "clone_transcript": True,
                "clone_transcript_required": True,
                "speakers": [],
            },
            "voices": [],
        },
    ],
}


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    text: str = Field(min_length=1, max_length=4096)
    voice: str = "Ryan"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str | None = None
    instructions: str | None = None
    voice_design: str | None = None
    reference_audio: str | None = None
    clone_transcript: str | None = None


class WorkerFailure(RuntimeError):
    def __init__(self, message: str, *, code: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _detail(exc: WorkerFailure) -> dict[str, str]:
    return {"code": exc.code, "message": str(exc)}


def _normalize_language(language: str | None) -> str:
    if not language:
        return "Auto"
    key = language.strip().lower().replace("_", "-")
    mapping = {
        "auto": "Auto",
        "zh": "Chinese",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }
    return mapping.get(key, language)


def _split_text(text: str) -> list[str]:
    """Split at sentence boundaries, then hard-bound unusually long sentences."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    result: list[str] = []
    for sentence in sentences or [text.strip()]:
        remaining = sentence
        while len(remaining) > MAX_SEGMENT_CHARS:
            split_at = remaining.rfind(" ", 0, MAX_SEGMENT_CHARS + 1)
            if split_at < MAX_SEGMENT_CHARS // 2:
                split_at = MAX_SEGMENT_CHARS
            result.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            result.append(remaining)
    return result


class QwenRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_id: str | None = None
        self.loaded_at: float | None = None
        self.last_used_at: float | None = None
        self.load_seconds: float | None = None
        self.prompt_cache: OrderedDict[str, Any] = OrderedDict()

    def _require_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA is unavailable", code="cuda_unavailable", status_code=503)
        capability = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        if required_arch not in torch.cuda.get_arch_list():
            raise WorkerFailure(
                f"Torch wheel does not contain {required_arch}",
                code="cuda_arch_unsupported",
                status_code=503,
            )

    def load(self, model_id: str) -> None:
        if model_id not in MODEL_IDS:
            raise WorkerFailure(f"Unknown model: {model_id}", code="unknown_model", status_code=400)
        if self.model_id == model_id and self.model is not None:
            return
        self.unload()
        self._require_cuda()
        try:
            from qwen_tts import Qwen3TTSModel

            started = time.perf_counter()
            self.model = Qwen3TTSModel.from_pretrained(
                MODEL_IDS[model_id],
                revision=MODEL_REVISIONS[model_id],
                device_map="cuda:0",
                dtype=torch.float16,
                attn_implementation="sdpa",
            )
            self.load_seconds = time.perf_counter() - started
            self.model_id = model_id
            self.loaded_at = time.time()
            self.last_used_at = None
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise WorkerFailure(
                "CUDA out of memory while loading Qwen3-TTS",
                code="cuda_out_of_memory",
                status_code=503,
            ) from exc
        except WorkerFailure:
            raise
        except Exception as exc:
            self.unload()
            raise WorkerFailure(
                f"Failed to load {model_id}: {exc}",
                code="model_load_failed",
                status_code=500,
            ) from exc

    def unload(self) -> None:
        self.model = None
        self.model_id = None
        self.loaded_at = None
        self.last_used_at = None
        self.load_seconds = None
        self.prompt_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_reference(self, encoded: str | None) -> bytes:
        if not encoded:
            raise WorkerFailure(
                "reference_audio is required for the Base model",
                code="reference_audio_required",
                status_code=400,
            )
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise WorkerFailure(
                "reference_audio must be valid base64",
                code="invalid_reference_audio",
                status_code=400,
            ) from exc
        if not data or len(data) > MAX_REFERENCE_BYTES:
            raise WorkerFailure(
                "reference_audio is empty or exceeds the worker limit",
                code="invalid_reference_audio",
                status_code=413,
            )
        return data

    def _clone_prompt(self, reference_audio: bytes, transcript: str) -> Any:
        assert self.model is not None
        key_hash = hashlib.sha256()
        key_hash.update((self.model_id or "").encode())
        key_hash.update(reference_audio)
        key_hash.update(hashlib.sha256(transcript.encode()).digest())
        key = key_hash.hexdigest()
        cached = self.prompt_cache.get(key)
        if cached is not None:
            self.prompt_cache.move_to_end(key)
            return cached

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(reference_audio)
                temp_path = temp_file.name
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=temp_path,
                ref_text=transcript,
                x_vector_only_mode=False,
            )
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        self.prompt_cache[key] = prompt
        self.prompt_cache.move_to_end(key)
        while len(self.prompt_cache) > max(1, PROMPT_CACHE_SIZE):
            self.prompt_cache.popitem(last=False)
        return prompt

    def generate(self, request: SynthesisRequest, text: str) -> np.ndarray:
        if self.model is None or self.model_id != request.model:
            raise WorkerFailure(
                f"Model {request.model} is not loaded",
                code="model_not_loaded",
                status_code=409,
            )
        if request.speed != 1.0:
            raise WorkerFailure(
                f"Numeric speed is not supported by {request.model}",
                code="speed_unsupported",
                status_code=400,
            )
        if request.instructions or request.voice_design:
            raise WorkerFailure(
                f"Instructions and voice design are not supported by {request.model}",
                code="instructions_unsupported",
                status_code=400,
            )

        try:
            if request.model.endswith("custom-voice"):
                voice_ids = {voice["id"] for voice in VOICES}
                if request.voice not in voice_ids:
                    raise WorkerFailure(
                        f"Unsupported voice '{request.voice}'",
                        code="voice_unsupported",
                        status_code=400,
                    )
                wavs, sample_rate = self.model.generate_custom_voice(
                    text=text,
                    language=_normalize_language(request.language),
                    speaker=request.voice,
                )
            else:
                transcript = (request.clone_transcript or "").strip()
                if not transcript:
                    raise WorkerFailure(
                        "clone_transcript is required for the Base model",
                        code="clone_transcript_required",
                        status_code=400,
                    )
                reference_audio = self._decode_reference(request.reference_audio)
                prompt = self._clone_prompt(reference_audio, transcript)
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    language=_normalize_language(request.language),
                    voice_clone_prompt=prompt,
                )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise WorkerFailure(
                "CUDA out of memory during Qwen3-TTS generation",
                code="cuda_out_of_memory",
                status_code=503,
            ) from exc
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(
                f"Qwen3-TTS generation failed: {exc}",
                code="generation_failed",
            ) from exc

        if int(sample_rate) != SAMPLE_RATE:
            raise WorkerFailure(
                f"Model returned {sample_rate} Hz; manifest declares {SAMPLE_RATE} Hz",
                code="sample_rate_mismatch",
            )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if not audio.size or not np.isfinite(audio).all():
            raise WorkerFailure(
                "Model returned empty or non-finite audio",
                code="invalid_audio",
            )
        self.last_used_at = time.time()
        return np.clip(audio, -1.0, 1.0).astype("<f4", copy=False)


runtime = QwenRuntime()
operation_lock = asyncio.Lock()
app = FastAPI(title="Open Speech Qwen3 Worker", version="1")


@app.get("/health")
async def health() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "arch_list": torch.cuda.get_arch_list(),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
            }
        )
    return {
        "status": "ok" if cuda_available else "unavailable",
        "provider": PROVIDER,
        "loaded_model": runtime.model_id,
        "device": "cuda:0" if cuda_available else "unavailable",
        "dtype": "float16",
        "attention": "sdpa",
        "loaded_at": runtime.loaded_at,
        "last_used_at": runtime.last_used_at,
        "load_seconds": runtime.load_seconds,
        "busy": operation_lock.locked(),
        "gpu": gpu,
    }


@app.get("/v1/manifest")
async def manifest() -> dict[str, Any]:
    return MANIFEST


@app.post("/v1/models/load")
async def load_model(payload: LoadRequest) -> dict[str, Any]:
    if operation_lock.locked():
        raise HTTPException(429, detail={"code": "worker_busy", "message": "Worker is busy"})
    async with operation_lock:
        try:
            await asyncio.to_thread(runtime.load, payload.model)
        except WorkerFailure as exc:
            raise HTTPException(exc.status_code, detail=_detail(exc)) from exc
    return {"status": "loaded", "model": payload.model, "load_seconds": runtime.load_seconds}


@app.post("/v1/models/unload")
async def unload_model(payload: LoadRequest) -> dict[str, str]:
    if operation_lock.locked():
        raise HTTPException(429, detail={"code": "worker_busy", "message": "Worker is busy"})
    async with operation_lock:
        if runtime.model_id and runtime.model_id != payload.model:
            raise HTTPException(
                404,
                detail={"code": "model_not_loaded", "message": f"Model {payload.model} is not loaded"},
            )
        await asyncio.to_thread(runtime.unload)
    return {"status": "unloaded", "model": payload.model}


@app.post("/v1/audio/speech")
async def synthesize(payload: SynthesisRequest, request: Request) -> StreamingResponse:
    if operation_lock.locked():
        raise HTTPException(429, detail={"code": "worker_busy", "message": "Worker is busy"})
    await operation_lock.acquire()
    segments = _split_text(payload.text)

    audio_queue: asyncio.Queue[bytes | WorkerFailure | Exception | None] = asyncio.Queue(
        maxsize=2
    )
    consumer_closed = asyncio.Event()

    async def put_unless_closed(item: bytes | WorkerFailure | Exception | None) -> bool:
        put_task = asyncio.create_task(audio_queue.put(item))
        closed_task = asyncio.create_task(consumer_closed.wait())
        done, pending = await asyncio.wait(
            {put_task, closed_task},
            timeout=STREAM_QUEUE_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return put_task in done and not put_task.cancelled()

    def force_terminal(error: Exception | None = None) -> None:
        """Ensure a stalled consumer can still observe termination if it resumes."""
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if error is not None:
            audio_queue.put_nowait(error)
        audio_queue.put_nowait(None)

    async def produce() -> None:
        forced_error: Exception | None = None
        try:
            for index, segment in enumerate(segments):
                if consumer_closed.is_set():
                    break
                if index and await request.is_disconnected():
                    break
                audio = await asyncio.to_thread(runtime.generate, payload, segment)
                if consumer_closed.is_set():
                    break
                if not await put_unless_closed(audio.tobytes()):
                    forced_error = WorkerFailure(
                        "Audio consumer stopped reading the worker stream",
                        code="stream_consumer_timeout",
                        status_code=503,
                    )
                    consumer_closed.set()
                    break
        except WorkerFailure as exc:
            if not await put_unless_closed(exc):
                forced_error = exc
                consumer_closed.set()
        except Exception as exc:
            if not await put_unless_closed(exc):
                forced_error = exc
                consumer_closed.set()
        finally:
            try:
                if forced_error is not None:
                    force_terminal(forced_error)
                elif not await put_unless_closed(None):
                    force_terminal()
            finally:
                operation_lock.release()

    producer_task = asyncio.create_task(produce(), name="qwen3-synthesis")
    try:
        first_item = await audio_queue.get()
    except BaseException:
        consumer_closed.set()
        raise
    if isinstance(first_item, WorkerFailure):
        consumer_closed.set()
        await producer_task
        raise HTTPException(first_item.status_code, detail=_detail(first_item)) from first_item
    if isinstance(first_item, Exception):
        consumer_closed.set()
        await producer_task
        raise first_item
    if first_item is None:
        consumer_closed.set()
        await producer_task
        raise HTTPException(
            500,
            detail={"code": "invalid_audio", "message": "Model returned no audio"},
        )

    async def generate() -> AsyncIterator[bytes]:
        try:
            yield first_item
            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            consumer_closed.set()

    async def close_consumer() -> None:
        consumer_closed.set()

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "float32le",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Cancellation-Granularity": "between_segments",
        },
        background=BackgroundTask(close_consumer),
    )
