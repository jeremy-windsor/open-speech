"""Isolated CosyVoice 2/3 worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import os
import tempfile
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

PROVIDER = "cosyvoice"
SCHEMA_VERSION = 1
SAMPLE_RATE = 24000
MAX_INPUT_CHARS = int(os.environ.get("COSYVOICE_MAX_INPUT_CHARS", "1500"))
MAX_REFERENCE_BYTES = int(os.environ.get("COSYVOICE_MAX_REFERENCE_MB", "100")) * 1024 * 1024
FP16 = os.environ.get("COSYVOICE_FP16", "true").strip().lower() in {"1", "true", "yes", "on"}
COSYVOICE3_PROMPT = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"

MODEL_IDS = {
    "cosyvoice/2-0.5b": "FunAudioLLM/CosyVoice2-0.5B",
    "cosyvoice/3-0.5b": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}
MODEL_REVISIONS = {
    "cosyvoice/2-0.5b": "eec1ae6c79877dbd9379285cf8789c9e0879293d",
    "cosyvoice/3-0.5b": "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
}
LANGUAGES = ["zh", "en", "ja", "ko", "de", "fr", "ru", "es", "it"]


def _capabilities() -> dict[str, Any]:
    return {
        "voice_blend": False,
        "voice_design": False,
        "voice_clone": True,
        "reference_audio": True,
        "clone_transcript": True,
        "clone_transcript_required": True,
        "streaming": True,
        "instructions": True,
        "languages": LANGUAGES,
        "speed_control": True,
        "speed_mode": "native_non_streaming",
        "ssml": False,
        "batch": False,
        "cancellation": "between_chunks",
        "live_reader": True,
        "native_tags": True,
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
            "capabilities": _capabilities(),
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
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
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


def _decode_reference(encoded: str | None) -> bytes:
    if not encoded:
        raise WorkerFailure(
            "reference_audio is required for CosyVoice voice cloning",
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
    if info.duration > 30.0:
        raise WorkerFailure(
            "CosyVoice reference audio cannot exceed 30 seconds",
            code="reference_audio_too_long",
            status_code=400,
        )
    return audio


def _prompt_text(model_id: str, transcript: str) -> str:
    if model_id.endswith("3-0.5b"):
        return f"{COSYVOICE3_PROMPT}{END_OF_PROMPT}{transcript}"
    return transcript


def _instruction_text(model_id: str, instructions: str) -> str:
    clean = instructions.replace(END_OF_PROMPT, "").strip()
    if model_id.endswith("3-0.5b"):
        clean = f"{COSYVOICE3_PROMPT} {clean}".strip()
    return f"{clean}{END_OF_PROMPT}"


def _numpy_audio(output: Any) -> np.ndarray:
    if not isinstance(output, dict) or "tts_speech" not in output:
        raise WorkerFailure("Model returned an invalid result", code="invalid_audio")
    value = output["tts_speech"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all():
        raise WorkerFailure("Model returned invalid audio", code="invalid_audio")
    return np.clip(audio, -1.0, 1.0).astype("<f4", copy=False)


class CosyVoiceRuntime:
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
            from cosyvoice.cli.cosyvoice import AutoModel
            from huggingface_hub import snapshot_download

            started = time.perf_counter()
            model_dir = snapshot_download(
                repo_id=MODEL_IDS[model_id],
                revision=MODEL_REVISIONS[model_id],
            )
            model = AutoModel(model_dir=model_dir, fp16=FP16)
            if int(getattr(model, "sample_rate", 0)) != SAMPLE_RATE:
                raise WorkerFailure(
                    f"Model sample rate {getattr(model, 'sample_rate', None)} does not match {SAMPLE_RATE}",
                    code="sample_rate_mismatch",
                )
            self.model = model
            self.model_id = model_id
            self.load_seconds = time.perf_counter() - started
            self.loaded_at = time.time()
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise WorkerFailure(
                "CUDA out of memory while loading CosyVoice",
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

    def generate(self, payload: SynthesisRequest) -> Iterator[np.ndarray]:
        if self.model is None or self.model_id != payload.model:
            raise WorkerFailure(
                f"Model {payload.model} is not loaded",
                code="model_not_loaded",
                status_code=409,
            )
        transcript = (payload.clone_transcript or "").strip()
        if not transcript:
            raise WorkerFailure(
                "clone_transcript is required for CosyVoice voice cloning",
                code="clone_transcript_required",
                status_code=400,
            )
        reference = _decode_reference(payload.reference_audio)
        native_stream = payload.speed == 1.0
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(reference)
                temp_path = temp_file.name
            if payload.instructions:
                iterator = self.model.inference_instruct2(
                    payload.text,
                    _instruction_text(payload.model, payload.instructions),
                    temp_path,
                    stream=native_stream,
                    speed=payload.speed,
                )
            else:
                iterator = self.model.inference_zero_shot(
                    payload.text,
                    _prompt_text(payload.model, transcript),
                    temp_path,
                    stream=native_stream,
                    speed=payload.speed,
                )
            produced = False
            with torch.inference_mode():
                for output in iterator:
                    produced = True
                    self.last_used_at = time.time()
                    yield _numpy_audio(output)
            if not produced:
                raise WorkerFailure("Model returned no audio", code="invalid_audio")
        except torch.cuda.OutOfMemoryError as exc:
            raise WorkerFailure(
                "CUDA out of memory during CosyVoice synthesis",
                code="cuda_out_of_memory",
                status_code=503,
            ) from exc
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(
                f"CosyVoice synthesis failed: {exc}",
                code="synthesis_failed",
            ) from exc
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def _next_chunk(iterator: Iterator[np.ndarray]) -> tuple[bool, np.ndarray | None]:
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


runtime = CosyVoiceRuntime()
operation_lock = asyncio.Lock()
app = FastAPI(title="Open Speech CosyVoice Worker", version="1")


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
        "dtype": "float16" if FP16 else "float32",
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
    iterator = runtime.generate(payload)
    try:
        finished, first = await asyncio.to_thread(_next_chunk, iterator)
    except WorkerFailure as exc:
        operation_lock.release()
        raise HTTPException(exc.status_code, detail=_detail(exc)) from exc
    except BaseException:
        operation_lock.release()
        raise
    if finished or first is None:
        operation_lock.release()
        raise HTTPException(
            500, detail={"code": "invalid_audio", "message": "Model returned no audio"}
        )

    async def generate():
        try:
            yield first.tobytes()
            while not await request.is_disconnected():
                done, chunk = await asyncio.to_thread(_next_chunk, iterator)
                if done:
                    break
                assert chunk is not None
                yield chunk.tobytes()
        finally:
            await asyncio.to_thread(iterator.close)
            if operation_lock.locked():
                operation_lock.release()

    async def close_stream() -> None:
        if operation_lock.locked():
            operation_lock.release()

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "float32le",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Cancellation-Granularity": "between_chunks",
            "X-Native-Streaming": "true" if payload.speed == 1.0 else "false",
        },
        background=BackgroundTask(close_stream),
    )
