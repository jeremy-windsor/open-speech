"""Disposable Qwen3-TTS worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import inspect
import logging
import math
import os
import tempfile
import time
import unicodedata
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

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
MAX_SEGMENT_UNITS = float(os.environ.get("QWEN3_MAX_SEGMENT_UNITS", "400"))
MAX_NEW_TOKENS_CEILING = int(os.environ.get("QWEN3_MAX_NEW_TOKENS_CEILING", "1200"))
PROMPT_CACHE_SIZE = int(os.environ.get("QWEN3_PROMPT_CACHE_SIZE", "8"))
STREAM_QUEUE_TIMEOUT_S = float(os.environ.get("QWEN3_STREAM_QUEUE_TIMEOUT_S", "30"))
DTYPE_SETTING = os.environ.get("QWEN3_DTYPE", "auto").strip().lower()
ENABLE_BASE = os.environ.get("QWEN3_ENABLE_BASE", "false").strip().lower() in {"1", "true", "yes", "on"}
WIDE_CHARACTER_UNITS = 3.2
GENERATION_TOKENS_PER_UNIT = 2.25
GENERATION_TOKEN_OVERHEAD = 48
GENERATION_TOKEN_FLOOR = 192
CODEC_TOKENS_PER_SECOND = 12.0
GENERATION_LIMIT_MARGIN_S = 1.0
CJK_SENTENCE_ENDINGS = frozenset("。！？")
LATIN_SENTENCE_ENDINGS = frozenset(".!?")
SENTENCE_CLOSERS = frozenset("」』”’）)》〉】]〉〕〗〙〛〞〟\"'")
CUDA_CONTEXT_ERROR_MARKERS = (
    "device-side assert",
    "cuda error",
    "cublas_status",
    "probability tensor contains",
)
WORKER_RESTART_DELAY_S = 1.0

logger = logging.getLogger(__name__)

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
if not ENABLE_BASE:
    MANIFEST["models"] = [
        model for model in MANIFEST["models"] if model["id"] != "qwen3/0.6b-base"
    ]


def _model_is_advertised(model_id: str) -> bool:
    return any(model["id"] == model_id for model in MANIFEST["models"])


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


def _resolve_dtype_name(setting: str, capability: tuple[int, int]) -> str:
    """Resolve a stable inference dtype without relying on emulated BF16 checks."""
    normalized = setting.strip().lower()
    if normalized == "auto":
        return "bfloat16" if capability >= (8, 0) else "float32"
    if normalized == "bfloat16":
        if capability < (8, 0):
            raise WorkerFailure(
                "bfloat16 requires CUDA compute capability 8.0 or newer",
                code="dtype_unsupported",
                status_code=503,
            )
        return normalized
    if normalized in {"float16", "float32"}:
        return normalized
    raise WorkerFailure(
        "QWEN3_DTYPE must be one of: auto, float32, float16, bfloat16",
        code="dtype_invalid",
        status_code=503,
    )


def _is_cuda_context_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return type(exc).__name__ == "AcceleratorError" or any(
        marker in message for marker in CUDA_CONTEXT_ERROR_MARKERS
    )


def _parameter_dtype_summary(model: Any) -> dict[str, list[str]]:
    """Report parameter dtypes per top-level model component."""
    root = getattr(model, "model", model)
    summary: dict[str, list[str]] = {}
    seen: set[int] = set()

    def add_component(name: str, component: Any) -> None:
        if component is None or id(component) in seen:
            return
        parameters = getattr(component, "parameters", None)
        if not callable(parameters):
            return
        seen.add(id(component))
        dtypes = sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in parameters()
            }
        )
        if dtypes:
            summary[str(name)] = dtypes

    named_children = getattr(root, "named_children", None)
    if callable(named_children):
        for name, component in named_children():
            add_component(f"model.{name}", component)

    for name, component in vars(model).items():
        if not name.startswith("_"):
            add_component(name, component)
    for name in (
        "speech_tokenizer",
        "audio_tokenizer",
        "tokenizer",
        "speaker_encoder",
        "codec",
    ):
        add_component(name, getattr(model, name, None))
    if not summary:
        add_component("model", root)
    return summary


def _schedule_process_restart() -> None:
    """Let the typed response leave the worker before Docker recreates it."""
    asyncio.get_running_loop().call_later(WORKER_RESTART_DELAY_S, os._exit, 1)


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


def _character_units(character: str) -> float:
    if unicodedata.east_asian_width(character) in {"W", "F"}:
        return WIDE_CHARACTER_UNITS
    return 1.0


def _text_units(text: str) -> float:
    return math.fsum(_character_units(character) for character in text)


def _max_new_tokens(text: str) -> int:
    estimated = math.ceil(GENERATION_TOKENS_PER_UNIT * _text_units(text))
    estimated += GENERATION_TOKEN_OVERHEAD
    return min(MAX_NEW_TOKENS_CEILING, max(GENERATION_TOKEN_FLOOR, estimated))


def _effective_segment_unit_limit() -> float:
    ceiling_units = (
        MAX_NEW_TOKENS_CEILING - GENERATION_TOKEN_OVERHEAD
    ) / GENERATION_TOKENS_PER_UNIT
    return min(MAX_SEGMENT_UNITS, ceiling_units)


def _validate_generation_settings() -> None:
    if MAX_SEGMENT_UNITS <= 0:
        raise RuntimeError("QWEN3_MAX_SEGMENT_UNITS must be greater than zero")
    if MAX_NEW_TOKENS_CEILING < GENERATION_TOKEN_FLOOR:
        raise RuntimeError(
            f"QWEN3_MAX_NEW_TOKENS_CEILING must be at least {GENERATION_TOKEN_FLOOR}"
        )


def _prefix_within_units(text: str, unit_limit: float) -> int:
    units = 0.0
    for index, character in enumerate(text):
        next_units = units + _character_units(character)
        if index and next_units > unit_limit + 1e-9:
            return index
        units = next_units
    return len(text)


def _sentence_parts(text: str) -> list[str]:
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
        if character not in CJK_SENTENCE_ENDINGS | LATIN_SENTENCE_ENDINGS:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in SENTENCE_CLOSERS:
            end += 1
        split_here = character in CJK_SENTENCE_ENDINGS
        split_here = split_here or (end < len(text) and text[end].isspace())
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


def _split_text(text: str) -> list[str]:
    """Split at sentence boundaries, then bound segments by estimated speech work."""
    sentences = _sentence_parts(text)
    unit_limit = _effective_segment_unit_limit()
    result: list[str] = []
    for sentence in sentences or [text.strip()]:
        remaining = sentence
        while _text_units(remaining) > unit_limit:
            hard_split = _prefix_within_units(remaining, unit_limit)
            candidate = remaining[:hard_split]
            split_at = candidate.rfind(" ")
            if split_at <= 0 or _text_units(candidate[:split_at]) < unit_limit / 2:
                split_at = hard_split
            result.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            result.append(remaining)
    return result


def _validate_generation_api(model: Any, model_id: str) -> None:
    method_name = (
        "generate_custom_voice"
        if model_id.endswith("custom-voice")
        else "generate_voice_clone"
    )
    method = getattr(model, method_name, None)
    if not callable(method):
        raise WorkerFailure(
            f"Pinned qwen-tts is missing {method_name}",
            code="provider_api_mismatch",
            status_code=503,
        )
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError) as exc:
        raise WorkerFailure(
            f"Cannot inspect pinned qwen-tts API {method_name}",
            code="provider_api_mismatch",
            status_code=503,
        ) from exc
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    missing = []
    if "non_streaming_mode" not in parameters:
        missing.append("non_streaming_mode")
    if "max_new_tokens" not in parameters and not accepts_kwargs:
        missing.append("max_new_tokens")
    if missing:
        raise WorkerFailure(
            f"Pinned qwen-tts {method_name} does not accept: {', '.join(missing)}",
            code="provider_api_mismatch",
            status_code=503,
        )


class QwenRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_id: str | None = None
        self.dtype_name: str | None = None
        self.parameter_dtypes: dict[str, list[str]] = {}
        self.loaded_at: float | None = None
        self.last_used_at: float | None = None
        self.load_seconds: float | None = None
        self.failed_code: str | None = None
        self.failed_message: str | None = None
        self.prompt_cache: OrderedDict[str, Any] = OrderedDict()

    def _require_cuda(self) -> tuple[int, int]:
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
        return capability

    def _mark_cuda_context_failed(self, exc: BaseException) -> WorkerFailure:
        self.failed_code = "cuda_context_failed"
        self.failed_message = str(exc)
        return WorkerFailure(
            "CUDA context failed; the Qwen worker is restarting",
            code=self.failed_code,
            status_code=503,
        )

    def _refresh_parameter_dtypes(self) -> None:
        try:
            self.parameter_dtypes = _parameter_dtype_summary(self.model)
        except Exception as exc:
            logger.warning("Could not inspect Qwen parameter dtypes: %s", exc)

    def load(self, model_id: str) -> None:
        if not _model_is_advertised(model_id):
            raise WorkerFailure(f"Unknown model: {model_id}", code="unknown_model", status_code=400)
        if self.failed_code:
            raise WorkerFailure(
                "CUDA context failed; the Qwen worker must restart",
                code=self.failed_code,
                status_code=503,
            )
        if self.model_id == model_id and self.model is not None:
            return
        self.unload()
        capability = self._require_cuda()
        dtype_name = _resolve_dtype_name(DTYPE_SETTING, capability)
        if dtype_name == "float16":
            logger.warning(
                "QWEN3_DTYPE=float16 is an explicit experimental override; "
                "use auto for the hardware-safe policy"
            )
        try:
            from qwen_tts import Qwen3TTSModel

            started = time.perf_counter()
            self.model = Qwen3TTSModel.from_pretrained(
                MODEL_IDS[model_id],
                revision=MODEL_REVISIONS[model_id],
                device_map="cuda:0",
                dtype=getattr(torch, dtype_name),
                attn_implementation="sdpa",
            )
            _validate_generation_api(self.model, model_id)
            self.load_seconds = time.perf_counter() - started
            self.model_id = model_id
            self.dtype_name = dtype_name
            self._refresh_parameter_dtypes()
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
            self.unload()
            raise
        except Exception as exc:
            if _is_cuda_context_failure(exc):
                raise self._mark_cuda_context_failed(exc) from exc
            self.unload()
            raise WorkerFailure(
                f"Failed to load {model_id}: {exc}",
                code="model_load_failed",
                status_code=500,
            ) from exc

    def unload(self) -> None:
        self.model = None
        self.model_id = None
        self.dtype_name = None
        self.parameter_dtypes = {}
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
        if self.failed_code:
            raise WorkerFailure(
                "CUDA context failed; the Qwen worker is restarting",
                code=self.failed_code,
                status_code=503,
            )
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

        max_new_tokens = _max_new_tokens(text)
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
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
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
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
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
            if _is_cuda_context_failure(exc):
                raise self._mark_cuda_context_failed(exc) from exc
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
        audio_seconds = audio.size / SAMPLE_RATE
        generation_limit_seconds = max_new_tokens / CODEC_TOKENS_PER_SECOND
        if audio_seconds >= max(0.0, generation_limit_seconds - GENERATION_LIMIT_MARGIN_S):
            logger.warning(
                "Qwen generation hit safety limit: audio_seconds=%.2f "
                "max_new_tokens=%d segment_units=%.1f",
                audio_seconds,
                max_new_tokens,
                _text_units(text),
            )
            raise WorkerFailure(
                "Qwen3-TTS reached its generation safety limit; no partial audio was returned",
                code="generation_limit_reached",
            )
        self._refresh_parameter_dtypes()
        self.last_used_at = time.time()
        return np.clip(audio, -1.0, 1.0).astype("<f4", copy=False)


_validate_generation_settings()
runtime = QwenRuntime()
operation_lock = asyncio.Lock()
app = FastAPI(title="Open Speech Qwen3 Worker", version="1")


@app.get("/health")
async def health() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] = {"available": cuda_available}
    health_error: WorkerFailure | None = None
    resolved_dtype = runtime.dtype_name
    if cuda_available and runtime.failed_code is None:
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
            if resolved_dtype is None:
                resolved_dtype = _resolve_dtype_name(DTYPE_SETTING, capability)
        except WorkerFailure as exc:
            health_error = exc
        except Exception as exc:
            gpu["error"] = "CUDA runtime details are unavailable"
            health_error = WorkerFailure(
                "CUDA runtime query failed",
                code="cuda_query_failed",
                status_code=503,
            )
            logger.warning("CUDA health query failed: %s", exc)
    elif runtime.failed_code:
        gpu["error"] = "CUDA context failed"

    if runtime.failed_code:
        status = "failed"
        failure = {"code": runtime.failed_code, "message": runtime.failed_message}
    elif health_error is not None:
        status = "failed"
        failure = _detail(health_error)
    else:
        status = "ok" if cuda_available else "unavailable"
        failure = None
    return {
        "status": status,
        "provider": PROVIDER,
        "loaded_model": runtime.model_id,
        "device": "cuda:0" if cuda_available else "unavailable",
        "dtype_config": DTYPE_SETTING,
        "dtype": resolved_dtype,
        "parameter_dtypes": runtime.parameter_dtypes,
        "attention": "sdpa",
        "generation": {
            "non_streaming_mode": True,
            "max_segment_units": MAX_SEGMENT_UNITS,
            "effective_segment_units": _effective_segment_unit_limit(),
            "max_new_tokens_ceiling": MAX_NEW_TOKENS_CEILING,
        },
        "failure": failure,
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
            if exc.code == "cuda_context_failed":
                _schedule_process_restart()
            raise HTTPException(exc.status_code, detail=_detail(exc)) from exc
    return {"status": "loaded", "model": payload.model, "load_seconds": runtime.load_seconds}


@app.post("/v1/models/unload")
async def unload_model(payload: LoadRequest) -> dict[str, str]:
    if runtime.failed_code:
        raise HTTPException(
            503,
            detail={
                "code": runtime.failed_code,
                "message": "CUDA context failed; the Qwen worker is restarting",
            },
        )
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
    if not _model_is_advertised(payload.model):
        raise HTTPException(
            400, detail={"code": "unknown_model", "message": f"Unknown model: {payload.model}"}
        )
    if runtime.failed_code:
        raise HTTPException(
            503,
            detail={
                "code": runtime.failed_code,
                "message": "CUDA context failed; the Qwen worker is restarting",
            },
        )
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
            if exc.code == "cuda_context_failed":
                _schedule_process_restart()
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
