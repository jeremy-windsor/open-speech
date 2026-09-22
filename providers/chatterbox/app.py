"""Isolated Chatterbox worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import os
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

PROVIDER = "chatterbox"
SCHEMA_VERSION = 1
SAMPLE_RATE = 24000
MAX_INPUT_CHARS = int(os.environ.get("CHATTERBOX_MAX_INPUT_CHARS", "350"))
MAX_REFERENCE_BYTES = int(os.environ.get("CHATTERBOX_MAX_REFERENCE_MB", "100")) * 1024 * 1024

MODEL_IDS = {
    "chatterbox/regular": "ResembleAI/chatterbox",
    "chatterbox/turbo": "ResembleAI/chatterbox-turbo",
}
MODEL_REVISIONS = {
    "chatterbox/regular": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    "chatterbox/turbo": "749d1c1a46eb10492095d68fbcf55691ccf137cd",
}


def _capabilities(*, tags: bool) -> dict[str, Any]:
    return {
        "voice_blend": False,
        "voice_design": False,
        "voice_clone": True,
        "reference_audio": True,
        "clone_transcript": False,
        "clone_transcript_required": False,
        "streaming": False,
        "instructions": False,
        "languages": ["en"],
        "speed_control": False,
        "speed_mode": "none",
        "ssml": False,
        "batch": False,
        "cancellation": "after_utterance",
        "live_reader": True,
        "native_tags": tags,
        "speakers": [],
    }


MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "provider": PROVIDER,
    "models": [
        {
            "id": model_id,
            "provider": PROVIDER,
            "sample_rate": SAMPLE_RATE,
            "revision": MODEL_REVISIONS[model_id],
            "max_input_chars": MAX_INPUT_CHARS,
            "capabilities": _capabilities(tags=model_id.endswith("turbo")),
            "voices": [],
        }
        for model_id in MODEL_IDS
    ],
}


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    voice: str = "reference"
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


def _decode_reference(encoded: str | None, *, turbo: bool) -> bytes:
    if not encoded:
        raise WorkerFailure(
            "reference_audio is required for Chatterbox voice cloning",
            code="reference_audio_required",
            status_code=400,
        )
    try:
        audio = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise WorkerFailure(
            "reference_audio must be valid base64",
            code="invalid_reference_audio",
            status_code=400,
        ) from exc
    if not audio or len(audio) > MAX_REFERENCE_BYTES:
        raise WorkerFailure(
            "reference_audio is empty or exceeds the worker limit",
            code="invalid_reference_audio",
            status_code=413,
        )
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
    if turbo and info.duration <= 5.0:
        raise WorkerFailure(
            "Chatterbox Turbo reference audio must be longer than five seconds",
            code="reference_audio_too_short",
            status_code=400,
        )
    return audio


def _numpy_audio(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all():
        raise WorkerFailure("Model returned invalid audio", code="invalid_audio")
    return np.clip(audio, -1.0, 1.0).astype("<f4", copy=False)


class ChatterboxRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_id: str | None = None
        self.loaded_at: float | None = None
        self.last_used_at: float | None = None
        self.load_seconds: float | None = None

    @staticmethod
    def _require_cuda() -> None:
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA is unavailable", code="cuda_unavailable", status_code=503)
        capability = torch.cuda.get_device_capability(0)
        architecture = f"sm_{capability[0]}{capability[1]}"
        if architecture not in torch.cuda.get_arch_list():
            raise WorkerFailure(
                f"Torch wheel does not contain {architecture}",
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
            from huggingface_hub import snapshot_download

            started = time.perf_counter()
            model_dir = snapshot_download(
                repo_id=MODEL_IDS[model_id],
                revision=MODEL_REVISIONS[model_id],
            )
            if model_id.endswith("turbo"):
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                model = ChatterboxTurboTTS.from_local(model_dir, "cuda")
            else:
                from chatterbox.tts import ChatterboxTTS

                model = ChatterboxTTS.from_local(model_dir, "cuda")
            if int(getattr(model, "sr", 0)) != SAMPLE_RATE:
                raise WorkerFailure(
                    f"Model sample rate {getattr(model, 'sr', None)} does not match {SAMPLE_RATE}",
                    code="sample_rate_mismatch",
                )
            self.model = model
            self.model_id = model_id
            self.load_seconds = time.perf_counter() - started
            self.loaded_at = time.time()
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise WorkerFailure(
                "CUDA out of memory while loading Chatterbox",
                code="cuda_out_of_memory",
                status_code=503,
            ) from exc
        except WorkerFailure:
            self.unload()
            raise
        except Exception as exc:
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, payload: SynthesisRequest) -> np.ndarray:
        if self.model is None or self.model_id != payload.model:
            raise WorkerFailure(
                f"Model {payload.model} is not loaded",
                code="model_not_loaded",
                status_code=409,
            )
        if payload.speed != 1.0:
            raise WorkerFailure(
                "Chatterbox does not support numeric speed control",
                code="speed_unsupported",
                status_code=400,
            )
        if payload.instructions:
            raise WorkerFailure(
                "Chatterbox does not support separate delivery instructions",
                code="instructions_unsupported",
                status_code=400,
            )
        reference = _decode_reference(
            payload.reference_audio,
            turbo=payload.model.endswith("turbo"),
        )
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(reference)
                temp_path = temp_file.name
            with torch.inference_mode():
                result = self.model.generate(payload.text, audio_prompt_path=temp_path)
            audio = _numpy_audio(result)
            self.last_used_at = time.time()
            return audio
        except torch.cuda.OutOfMemoryError as exc:
            raise WorkerFailure(
                "CUDA out of memory during Chatterbox synthesis",
                code="cuda_out_of_memory",
                status_code=503,
            ) from exc
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(
                f"Chatterbox synthesis failed: {exc}",
                code="synthesis_failed",
            ) from exc
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


runtime = ChatterboxRuntime()
operation_lock = asyncio.Lock()
app = FastAPI(title="Open Speech Chatterbox Worker", version="1")


@app.get("/health")
async def health() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
            }
        )
    return {
        "status": "ok" if cuda_available else "unavailable",
        "provider": PROVIDER,
        "loaded_model": runtime.model_id,
        "device": "cuda:0" if cuda_available else "unavailable",
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
    if payload.model not in MODEL_IDS:
        raise HTTPException(
            400, detail={"code": "unknown_model", "message": f"Unknown model: {payload.model}"}
        )
    if operation_lock.locked():
        raise HTTPException(429, detail={"code": "worker_busy", "message": "Worker is busy"})
    await operation_lock.acquire()
    lock_released = False

    def release_lock_once() -> None:
        nonlocal lock_released
        if lock_released:
            return
        lock_released = True
        operation_lock.release()

    try:
        audio = await asyncio.to_thread(runtime.generate, payload)
    except WorkerFailure as exc:
        release_lock_once()
        raise HTTPException(exc.status_code, detail=_detail(exc)) from exc
    except BaseException:
        release_lock_once()
        raise

    async def generate():
        try:
            if not await request.is_disconnected():
                yield audio.tobytes()
        finally:
            release_lock_once()

    async def close_stream() -> None:
        release_lock_once()

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "float32le",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Cancellation-Granularity": "after_utterance",
        },
        background=BackgroundTask(close_stream),
    )
