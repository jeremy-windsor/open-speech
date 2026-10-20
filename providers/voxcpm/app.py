"""Isolated VoxCPM2 worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import base64
import ctypes
import gc
import hashlib
import io
import json
import logging
import os
import re
import signal
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

logger = logging.getLogger("uvicorn.error")

PROVIDER = "voxcpm"
SCHEMA_VERSION = 1
SAMPLE_RATE = 48000
MAX_INPUT_CHARS = int(os.environ.get("VOXCPM_MAX_INPUT_CHARS", "1500"))
MAX_SEGMENT_CHARS = int(os.environ.get("VOXCPM_MAX_SEGMENT_CHARS", "300"))
MAX_REFERENCE_BYTES = int(os.environ.get("VOXCPM_MAX_REFERENCE_MB", "100")) * 1024 * 1024
MAX_REFERENCE_SECONDS = float(os.environ.get("VOXCPM_MAX_REFERENCE_SECONDS", "30"))
DTYPE_SETTING = os.environ.get("VOXCPM_DTYPE", "auto").strip().lower()
INFERENCE_TIMESTEPS = int(os.environ.get("VOXCPM_INFERENCE_TIMESTEPS", "10"))
CFG_VALUE = float(os.environ.get("VOXCPM_CFG_VALUE", "2.0"))
PROMPT_CACHE_SIZE = int(os.environ.get("VOXCPM_PROMPT_CACHE_SIZE", "4"))
RESTART_AFTER_UNLOAD_S = float(os.environ.get("VOXCPM_RESTART_AFTER_UNLOAD_S", "0"))

MODEL_IDS = {"voxcpm/2": "openbmb/VoxCPM2"}
MODEL_REVISIONS = {"voxcpm/2": "32279effe8c19989596f05d353d1447f51d9e915"}
# The repository's tokenization_voxcpm2.py is not needed: the stock fast tokenizer reads it.
MODEL_FILES = ["*.json", "*.safetensors", "*.pth"]
LANGUAGES = [
    "ar", "my", "zh", "da", "nl", "en", "fi", "fr", "de", "el", "he", "hi", "id", "it", "ja",
    "km", "ko", "lo", "ms", "no", "pl", "pt", "ru", "es", "sw", "sv", "tl", "th", "tr", "vi",
]
SENTENCE_END = re.compile(r"(?<=[.!?;。！？；])\s+|(?<=[。！？；])")


def _capabilities() -> dict[str, Any]:
    return {
        "voice_blend": False,
        "voice_design": True,
        "voice_clone": True,
        "reference_audio": True,
        "clone_transcript": True,
        "clone_transcript_required": False,
        "streaming": True,
        "instructions": True,
        "languages": LANGUAGES,
        "speed_control": False,
        "speed_mode": "none",
        "ssml": False,
        "batch": False,
        "cancellation": "between_chunks",
        "live_reader": False,
        "native_tags": False,
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


def _resolve_dtype_name(setting: str, capability: tuple[int, int]) -> str:
    if setting == "auto":
        # VoxCPM2 ships bf16 weights. Cards before sm_80 have no bf16 hardware, and fp32 would
        # need 9 GB for the weights alone, so they run fp16.
        return "bfloat16" if capability >= (8, 0) else "float16"
    if setting in {"bfloat16", "float16", "float32"}:
        return setting
    raise WorkerFailure(
        "VOXCPM_DTYPE must be one of: auto, bfloat16, float16, float32",
        code="dtype_invalid",
        status_code=503,
    )


def _decode_reference(encoded: str) -> bytes:
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
    if info.duration > MAX_REFERENCE_SECONDS:
        raise WorkerFailure(
            f"VoxCPM2 reference audio cannot exceed {MAX_REFERENCE_SECONDS:g} seconds",
            code="reference_audio_too_long",
            status_code=400,
        )
    return audio


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _segments(text: str, limit: int) -> list[str]:
    """Group whole sentences into pieces of at most `limit` characters."""
    pieces: list[str] = []
    for sentence in (part.strip() for part in SENTENCE_END.split(text)):
        if not sentence:
            continue
        if pieces and len(pieces[-1]) + 1 + len(sentence) <= limit:
            pieces[-1] = f"{pieces[-1]} {sentence}"
        else:
            pieces.append(sentence)
    return pieces


def _style(text: str) -> str:
    # VoxCPM2 reads a leading "(...)" as the voice or style description.
    return _clean(re.sub(r"[()]", " ", text))


def _numpy_audio(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all():
        raise WorkerFailure("Model returned invalid audio", code="invalid_audio")
    return np.clip(audio, -1.0, 1.0).astype("<f4", copy=False)


def _load_voxcpm2(model_dir: str, dtype_name: str, device: str = "cuda") -> Any:
    """Build VoxCPM2 straight onto the GPU.

    Upstream's from_local builds the 2B model in fp32 on the CPU and then loads a second copy of
    the weights, about 14 GB of RAM at peak. The kitchen PC's Docker VM has 7.7 GB, so this builds
    the model on the GPU in its final dtype and copies the weights in one tensor at a time.
    """
    from safetensors import safe_open
    from transformers import LlamaTokenizerFast
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as config_file:
        config = VoxCPMConfig.model_validate_json(config_file.read())
    config.dtype = dtype_name
    tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
    audio_vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()

    dtype = getattr(torch, dtype_name)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = VoxCPM2Model(config, tokenizer, audio_vae, None, device=device)
    finally:
        torch.set_default_dtype(default_dtype)

    targets = {
        name: tensor for name, tensor in model.state_dict().items() if not name.startswith("audio_vae.")
    }
    with torch.no_grad(), safe_open(
        os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu"
    ) as weights:
        for name in set(weights.keys()) & set(targets):
            targets.pop(name).copy_(weights.get_tensor(name))
    if targets:
        logger.warning("VoxCPM2 weights missing %d tensors, e.g. %s", len(targets), list(targets)[:3])
    # Same order and final layout as upstream: everything cast to the model dtype, then the audio
    # VAE back to fp32 before its own weights load, so they keep full precision.
    model = model.to(dtype)
    model.audio_vae = model.audio_vae.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(
        os.path.join(model_dir, "audiovae.pth"), map_location="cpu", weights_only=True
    )
    model.audio_vae.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    return model.eval()


class VoxCPMRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_id: str | None = None
        self.loaded_at: float | None = None
        self.last_used_at: float | None = None
        self.load_seconds: float | None = None
        self.dtype_name: str | None = None
        self.cuda_used = False
        # (reference hash, transcript) -> prompt cache built from the reference clip.
        self.prompts: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    @staticmethod
    def _require_cuda() -> tuple[int, int]:
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
        return tuple(capability)

    def load(self, model_id: str) -> None:
        if model_id not in MODEL_IDS:
            raise WorkerFailure(f"Unknown model: {model_id}", code="unknown_model", status_code=400)
        if self.model_id == model_id and self.model is not None:
            return
        self.unload()
        dtype_name = _resolve_dtype_name(DTYPE_SETTING, self._require_cuda())
        self.cuda_used = True
        try:
            from huggingface_hub import snapshot_download

            started = time.perf_counter()
            model_dir = snapshot_download(
                repo_id=MODEL_IDS[model_id],
                revision=MODEL_REVISIONS[model_id],
                allow_patterns=MODEL_FILES,
            )
            with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as config_file:
                architecture = json.load(config_file).get("architecture", "").lower()
            if architecture != "voxcpm2":
                raise WorkerFailure(
                    f"Expected a VoxCPM2 checkpoint, found {architecture or 'unknown'}",
                    code="model_load_failed",
                )
            model = _load_voxcpm2(model_dir, dtype_name)
            if int(model.sample_rate) != SAMPLE_RATE:
                raise WorkerFailure(
                    f"Model sample rate {model.sample_rate} does not match {SAMPLE_RATE}",
                    code="sample_rate_mismatch",
                )
            self.model = model
            self.model_id = model_id
            self.dtype_name = dtype_name
            self.load_seconds = time.perf_counter() - started
            self.loaded_at = time.time()
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise WorkerFailure(
                "CUDA out of memory while loading VoxCPM2",
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
        self.prompts.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.name == "posix":
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim(0)

    def _prompt(self, encoded: str, transcript: str) -> dict[str, Any]:
        reference = _decode_reference(encoded)
        key = (hashlib.sha256(reference).hexdigest(), transcript)
        cached = self.prompts.get(key)
        if cached is not None:
            self.prompts.move_to_end(key)
            return cached
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(reference)
                temp_path = temp_file.name
            with torch.inference_mode():
                if transcript:
                    # Upstream's "ultimate" clone: the clip as both reference and continuation.
                    cache = self.model.build_prompt_cache(
                        prompt_text=transcript,
                        prompt_wav_path=temp_path,
                        reference_wav_path=temp_path,
                    )
                else:
                    cache = self.model.build_prompt_cache(reference_wav_path=temp_path)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        if PROMPT_CACHE_SIZE > 0:
            self.prompts[key] = cache
            while len(self.prompts) > PROMPT_CACHE_SIZE:
                self.prompts.popitem(last=False)
        return cache

    def generate(self, payload: SynthesisRequest) -> Iterator[np.ndarray]:
        if self.model is None or self.model_id != payload.model:
            raise WorkerFailure(
                f"Model {payload.model} is not loaded",
                code="model_not_loaded",
                status_code=409,
            )
        if payload.speed != 1.0:
            raise WorkerFailure(
                "VoxCPM2 does not support speed control",
                code="speed_unsupported",
                status_code=400,
            )
        text = _clean(payload.text)
        design = _style(payload.voice_design or "")
        instructions = _style(payload.instructions or "")
        try:
            if payload.reference_audio:
                if design:
                    raise WorkerFailure(
                        "Use either a reference clip or a voice design, not both",
                        code="voice_conflict",
                        status_code=400,
                    )
                # Style directions only work in reference mode, so they set the transcript aside.
                transcript = "" if instructions else _clean(payload.clone_transcript or "")
                cache = self._prompt(payload.reference_audio, transcript)
                prefix = f"({instructions})" if instructions else ""
                # The clip keeps the voice steady across pieces, so long text goes in pieces.
                pieces = [prefix + piece for piece in _segments(text, MAX_SEGMENT_CHARS)]
            elif design:
                if instructions:
                    design = f"{design}, {instructions}"
                # Nothing anchors a designed voice between pieces, so the text stays whole.
                cache = None
                pieces = [f"({design}){text}"]
            else:
                raise WorkerFailure(
                    "VoxCPM2 needs a reference clip or a voice design",
                    code="reference_audio_required",
                    status_code=400,
                )
            produced = False
            for piece in pieces:
                stream = self.model.generate_with_prompt_cache_streaming(
                    target_text=piece,
                    prompt_cache=cache,
                    inference_timesteps=INFERENCE_TIMESTEPS,
                    cfg_value=CFG_VALUE,
                )
                try:
                    for audio, _, _ in stream:
                        produced = True
                        self.last_used_at = time.time()
                        yield _numpy_audio(audio)
                finally:
                    stream.close()
            if not produced:
                raise WorkerFailure("Model returned no audio", code="invalid_audio")
        except torch.cuda.OutOfMemoryError as exc:
            raise WorkerFailure(
                "CUDA out of memory during VoxCPM2 synthesis",
                code="cuda_out_of_memory",
                status_code=503,
            ) from exc
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(f"VoxCPM2 synthesis failed: {exc}", code="synthesis_failed") from exc


def _next_chunk(iterator: Iterator[np.ndarray]) -> tuple[bool, np.ndarray | None]:
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


runtime = VoxCPMRuntime()
operation_lock = asyncio.Lock()
restart_task: asyncio.Task[None] | None = None
app = FastAPI(title="Open Speech VoxCPM Worker", version="1")


def _cancel_restart() -> None:
    global restart_task
    if restart_task is not None:
        restart_task.cancel()
        restart_task = None


def _schedule_restart() -> None:
    global restart_task
    if RESTART_AFTER_UNLOAD_S <= 0 or not runtime.cuda_used:
        return
    _cancel_restart()
    restart_task = asyncio.get_running_loop().create_task(_restart_when_idle())


async def _restart_when_idle() -> None:
    # A load that arrives during the wait cancels this task.
    await asyncio.sleep(RESTART_AFTER_UNLOAD_S)
    async with operation_lock:
        if runtime.model is not None:
            return
        logger.info("Exiting after unload so the container restarts with its memory released")
        os.kill(os.getpid(), signal.SIGTERM)


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
        "dtype": runtime.dtype_name or DTYPE_SETTING,
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
    _cancel_restart()
    if operation_lock.locked():
        raise HTTPException(429, detail={"code": "worker_busy", "message": "Worker is busy"})
    async with operation_lock:
        try:
            await asyncio.to_thread(runtime.load, payload.model)
        except WorkerFailure as exc:
            _schedule_restart()
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
    _schedule_restart()
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

    iterator = runtime.generate(payload)
    try:
        finished, first = await asyncio.to_thread(_next_chunk, iterator)
    except WorkerFailure as exc:
        release_lock_once()
        raise HTTPException(exc.status_code, detail=_detail(exc)) from exc
    except BaseException:
        release_lock_once()
        raise
    if finished or first is None:
        release_lock_once()
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
            try:
                await asyncio.to_thread(iterator.close)
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
            "X-Cancellation-Granularity": "between_chunks",
            "X-Native-Streaming": "true",
        },
        background=BackgroundTask(close_stream),
    )
