"""Isolated Ming-3.0-tts worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import base64
import ctypes
import gc
import hashlib
import io
import logging
import os
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

PROVIDER = "ming"
SCHEMA_VERSION = 1
SAMPLE_RATE = 44100
MAX_INPUT_CHARS = int(os.environ.get("MING_MAX_INPUT_CHARS", "1500"))
MAX_REFERENCE_BYTES = int(os.environ.get("MING_MAX_REFERENCE_MB", "100")) * 1024 * 1024
MAX_REFERENCE_SECONDS = float(os.environ.get("MING_MAX_REFERENCE_SECONDS", "20"))
DTYPE_SETTING = os.environ.get("MING_DTYPE", "auto").strip().lower()
TEXT_NORMALIZATION = os.environ.get("MING_TEXT_NORMALIZATION", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROMPT_CACHE_SIZE = int(os.environ.get("MING_PROMPT_CACHE_SIZE", "4"))
RESTART_AFTER_UNLOAD_S = float(os.environ.get("MING_RESTART_AFTER_UNLOAD_S", "0"))

MODEL_IDS = {"ming/3-0.5b": "inclusionAI/Ming-3.0-tts-0.5B"}
MODEL_REVISIONS = {"ming/3-0.5b": "e5bd8da31ecf1b9276cdbdda821fa891068f1368"}
MODEL_FILES = ["*.json", "*.txt", "*.safetensors", "*.onnx"]
LANGUAGES = ["zh", "en", "ja", "ko", "de", "fr", "es"]
# Chinese and English need no language line. Ming requires the other five to be named.
NAMED_LANGUAGES = {
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "de": "de",
    "german": "de",
    "fr": "fr",
    "french": "fr",
    "es": "es",
    "spanish": "es",
}

# Ming decodes into a 512-position cache shared by the prompt (template, transcript, reference
# latents and text) and the speech it writes, one position per 80 ms. Each text fragment has to
# leave room for its speech, so fragments get shorter as the reference clip gets longer.
CACHE_POSITIONS = 512
POSITIONS_PER_UNIT = 8  # one word or Chinese character: its tokens and its speech, with margin
MIN_FRAGMENT_UNITS = 8
MAX_FRAGMENT_UNITS = 50  # the upstream default
SPEAKER_EMBEDDING_DIM = 192
CFG = 2.0
SIGMA = 0.25
TEMPERATURE = 0.0


def _capabilities() -> dict[str, Any]:
    return {
        "voice_blend": False,
        "voice_design": True,
        "voice_clone": True,
        "reference_audio": True,
        "clone_transcript": True,
        "clone_transcript_required": True,
        "streaming": True,
        "instructions": False,
        "languages": LANGUAGES,
        "speed_control": False,
        "speed_mode": "none",
        "ssml": False,
        "batch": False,
        "cancellation": "between_chunks",
        "live_reader": False,
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


def _resolve_dtype_name(setting: str) -> str:
    if setting == "auto":
        # Ming ships bf16 weights and overflows in fp16: on an RTX 2070 SUPER every fp16 clip came
        # back non-finite. Cards before sm_80 run bf16 slower but correctly.
        return "bfloat16"
    if setting in {"bfloat16", "float16", "float32"}:
        return setting
    raise WorkerFailure(
        "MING_DTYPE must be one of: auto, bfloat16, float16, float32",
        code="dtype_invalid",
        status_code=503,
    )


def _language_id(language: str | None) -> str | None:
    if not language:
        return None
    key = language.strip().lower().replace("_", "-")
    return NAMED_LANGUAGES.get(key) or NAMED_LANGUAGES.get(key.split("-")[0])


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
            f"Ming reference audio cannot exceed {MAX_REFERENCE_SECONDS:g} seconds",
            code="reference_audio_too_long",
            status_code=400,
        )
    return audio


def _halves(text: str) -> list[str] | None:
    words = text.split()
    if len(words) >= 2:
        middle = len(words) // 2
        return [" ".join(words[:middle]), " ".join(words[middle:])]
    if len(text) >= 2:
        middle = len(text) // 2
        return [text[:middle], text[middle:]]
    return None


def _numpy_audio(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all():
        raise WorkerFailure("Model returned invalid audio", code="invalid_audio")
    return np.clip(audio, -1.0, 1.0).astype("<f4", copy=False)


def _normalizer_works(model: Any) -> bool:
    try:
        model.normalizer.normalize("Open Speech 2.0 costs $5.")
    except Exception:
        logger.exception("Ming text normalization is off")
        return False
    return True


class MingRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_id: str | None = None
        self.loaded_at: float | None = None
        self.last_used_at: float | None = None
        self.load_seconds: float | None = None
        self.dtype_name: str | None = None
        self.text_normalization = False
        self.cuda_used = False
        self.tasks: Any | None = None
        self.contract: Any | None = None
        self.trim: Any | None = None
        # Reference clip hash -> (prompt latents, speaker embeddings). Encoding samples noise,
        # so reusing the first encoding also keeps a saved voice sounding the same.
        self.prompts: OrderedDict[str, tuple[Any, list[Any]]] = OrderedDict()

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
        dtype_name = _resolve_dtype_name(DTYPE_SETTING)
        self.cuda_used = True
        try:
            import ming_dtype
            from huggingface_hub import snapshot_download
            from loguru import logger as ming_logger

            # Set before the upstream modules are imported; the Dockerfile patched them to read it.
            ming_dtype.MING_DTYPE = getattr(torch, dtype_name)
            import prompt_contract
            import tts_tasks
            from ming_tts import MingTTS, trim_trailing_silence

            # Upstream logs every prompt it decodes, user text included.
            ming_logger.disable("modeling_bailingmm")

            started = time.perf_counter()
            model_dir = snapshot_download(
                repo_id=MODEL_IDS[model_id],
                revision=MODEL_REVISIONS[model_id],
                allow_patterns=MODEL_FILES,
            )
            model = MingTTS(
                model_dir,
                device="cuda:0",
                text_normalization=TEXT_NORMALIZATION,
                cache_prompts=False,
            )
            if int(model.sample_rate) != SAMPLE_RATE:
                raise WorkerFailure(
                    f"Model sample rate {model.sample_rate} does not match {SAMPLE_RATE}",
                    code="sample_rate_mismatch",
                )
            model.text_normalization = TEXT_NORMALIZATION and _normalizer_works(model)
            self.model = model
            self.model_id = model_id
            self.dtype_name = dtype_name
            self.text_normalization = model.text_normalization
            self.tasks = tts_tasks
            self.contract = prompt_contract
            self.trim = trim_trailing_silence
            self.load_seconds = time.perf_counter() - started
            self.loaded_at = time.time()
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise WorkerFailure(
                "CUDA out of memory while loading Ming",
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

    def _prompt(self, encoded: str) -> tuple[Any, list[Any]]:
        reference = _decode_reference(encoded)
        key = hashlib.sha256(reference).hexdigest()
        cached = self.prompts.get(key)
        if cached is not None:
            self.prompts.move_to_end(key)
            return cached
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(reference)
                temp_path = temp_file.name
            # Upstream's own prompt cache is keyed by file path, so the worker keeps its own.
            waveform, speaker = self.model._load_prompt([temp_path])
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        with torch.inference_mode():
            latent = self.model.model.encode_prompt_latent(waveform)
        entry = (latent, speaker)
        if PROMPT_CACHE_SIZE > 0:
            self.prompts[key] = entry
            while len(self.prompts) > PROMPT_CACHE_SIZE:
                self.prompts.popitem(last=False)
        return entry

    def _input_length(self, voice: dict[str, Any], text: str) -> int:
        model = self.model.model
        latent = voice["prompt_latent"]
        input_ids, _, _ = self.contract.build_input_ids(
            model.tokenizer,
            model.model_type,
            voice["prompt"],
            text,
            num_speakers=len(voice["speaker"]),
            instruction=voice["instruction"],
            prompt_text=voice["prompt_text"],
            prompt_latent_steps=0 if latent is None else latent.shape[1] // model.patch_size,
        )
        return len(input_ids)

    def _speak(self, voice: dict[str, Any], fragment: str, depth: int = 0) -> Iterator[np.ndarray]:
        model = self.model.model
        with torch.inference_mode():
            speech = model.generate(
                prompt=voice["prompt"],
                text=fragment,
                spk_emb=[embedding.clone() for embedding in voice["speaker"]],
                instruction=voice["instruction"],
                prompt_latent=voice["prompt_latent"],
                prompt_text=voice["prompt_text"],
                max_decode_steps=1000,
                cfg=CFG,
                sigma=SIGMA,
                temperature=TEMPERATURE,
                use_zero_spk_emb=voice["zero_speaker"],
            )
        budget = (CACHE_POSITIONS - self._input_length(voice, fragment)) // model.patch_size
        steps = round(speech.shape[-1] / self.model._pad_align())
        halves = _halves(fragment) if steps >= budget and depth < 2 else None
        if halves:
            # Ming stops at the end of its cache without warning, which would cut the words off.
            logger.warning("Ming ran out of room on a fragment; generating it in two halves")
            for half in halves:
                yield from self._speak(voice, half, depth + 1)
            return
        yield _numpy_audio(self.trim(speech, SAMPLE_RATE))

    def generate(self, payload: SynthesisRequest) -> Iterator[np.ndarray]:
        if self.model is None or self.model_id != payload.model:
            raise WorkerFailure(
                f"Model {payload.model} is not loaded",
                code="model_not_loaded",
                status_code=409,
            )
        if payload.speed != 1.0:
            raise WorkerFailure(
                "Ming does not support speed control",
                code="speed_unsupported",
                status_code=400,
            )
        if payload.instructions:
            raise WorkerFailure(
                "Ming does not support instructions with a voice",
                code="instructions_unsupported",
                status_code=400,
            )
        design = (payload.voice_design or "").strip()
        language = _language_id(payload.language)
        voice: dict[str, Any] = {"prompt": self.tasks.build_tts_prompt(language)}
        try:
            if payload.reference_audio:
                if design:
                    raise WorkerFailure(
                        "Use either a reference clip or a voice design, not both",
                        code="voice_conflict",
                        status_code=400,
                    )
                transcript = (payload.clone_transcript or "").strip()
                if not transcript:
                    raise WorkerFailure(
                        "clone_transcript is required for Ming voice cloning",
                        code="clone_transcript_required",
                        status_code=400,
                    )
                latent, speaker = self._prompt(payload.reference_audio)
                voice.update(
                    instruction=None,
                    prompt_latent=latent,
                    prompt_text=transcript,
                    speaker=speaker,
                    zero_speaker=False,
                )
            elif design:
                voice.update(
                    instruction=self.tasks.build_task_instruction("VOICE_DESIGN", style=design),
                    prompt_latent=None,
                    prompt_text=None,
                    speaker=[torch.zeros(1, SPEAKER_EMBEDDING_DIM)],
                    zero_speaker=True,
                )
            else:
                raise WorkerFailure(
                    "Ming needs a reference clip with its transcript, or a voice design",
                    code="reference_audio_required",
                    status_code=400,
                )
            free = CACHE_POSITIONS - self._input_length(voice, "")
            units = max(MIN_FRAGMENT_UNITS, min(MAX_FRAGMENT_UNITS, free // POSITIONS_PER_UNIT))
            # Upstream keeps voice-design text in one piece, which overruns the cache on a
            # paragraph, so it is split like clone text.
            try:
                fragments = self.model.prepare_text(payload.text, "TTS", language, units)
            except ValueError as exc:
                raise WorkerFailure(str(exc), code="invalid_text", status_code=400) from exc
            produced = False
            for fragment in fragments:
                for audio in self._speak(voice, fragment):
                    produced = True
                    self.last_used_at = time.time()
                    yield audio
            if not produced:
                raise WorkerFailure("Model returned no audio", code="invalid_audio")
        except torch.cuda.OutOfMemoryError as exc:
            raise WorkerFailure(
                "CUDA out of memory during Ming synthesis",
                code="cuda_out_of_memory",
                status_code=503,
            ) from exc
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(f"Ming synthesis failed: {exc}", code="synthesis_failed") from exc


def _next_chunk(iterator: Iterator[np.ndarray]) -> tuple[bool, np.ndarray | None]:
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


runtime = MingRuntime()
operation_lock = asyncio.Lock()
restart_task: asyncio.Task[None] | None = None
app = FastAPI(title="Open Speech Ming Worker", version="1")


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
        "text_normalization": runtime.text_normalization,
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
            # Audio arrives one sentence-sized fragment at a time, not frame by frame.
            "X-Native-Streaming": "false",
        },
        background=BackgroundTask(close_stream),
    )
