"""Disposable Qwen3-TTS worker for the Open Speech harness.

It holds one checkpoint at a time: loading another, 0.6B or 1.7B, unloads the one before.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import secrets
import time
import unicodedata
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import Field
from starlette.background import BackgroundTask

from providers._common import worker as common
from providers._common.worker import LoadRequest, WorkerFailure

PROVIDER = "qwen3"
SCHEMA_VERSION = common.SCHEMA_VERSION
SAMPLE_RATE = 24000
MAX_REFERENCE_BYTES = int(os.environ.get("QWEN3_MAX_REFERENCE_MB", "100")) * 1024 * 1024
MAX_SEGMENT_UNITS = float(os.environ.get("QWEN3_MAX_SEGMENT_UNITS", "400"))
MAX_NEW_TOKENS_CEILING = int(os.environ.get("QWEN3_MAX_NEW_TOKENS_CEILING", "1200"))
PROMPT_CACHE_SIZE = int(os.environ.get("QWEN3_PROMPT_CACHE_SIZE", "8"))
STREAM_QUEUE_TIMEOUT_S = float(os.environ.get("QWEN3_STREAM_QUEUE_TIMEOUT_S", "30"))
DTYPE_SETTING = os.environ.get("QWEN3_DTYPE", "auto").strip().lower()
ENABLE_BASE = common.env_flag("QWEN3_ENABLE_BASE", "false")
WIDE_CHARACTER_UNITS = 3.2
GENERATION_TOKENS_PER_UNIT = 2.25
GENERATION_TOKEN_OVERHEAD = 48
GENERATION_TOKEN_FLOOR = 192
# The 12Hz speech tokenizer decodes 1920 samples per codec token at 24 kHz, 12.5 tokens a second
# (speech_tokenizer/config.json: decode_upsample_rate 1920, encoder_config._frame_rate 12.5).
CODEC_SAMPLES_PER_TOKEN = 1920
GENERATION_LIMIT_MARGIN_S = 1.0
CUDA_CONTEXT_ERROR_MARKERS = (
    "device-side assert",
    "cuda error",
    "cublas_status",
    "probability tensor contains",
)
WORKER_RESTART_DELAY_S = 1.0
OUT_OF_MEMORY_MESSAGE = "CUDA out of memory during Qwen3-TTS generation"

logger = logging.getLogger(__name__)

MODEL_IDS = {
    "qwen3/0.6b-custom-voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "qwen3/0.6b-base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "qwen3/1.7b-custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen3/1.7b-voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "qwen3/1.7b-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
MODEL_REVISIONS = {
    "qwen3/0.6b-custom-voice": "85e237c12c027371202489a0ec509ded67b5e4b5",
    "qwen3/0.6b-base": "5d83992436eae1d760afd27aff78a71d676296fc",
    "qwen3/1.7b-custom-voice": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
    "qwen3/1.7b-voice-design": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
    "qwen3/1.7b-base": "fd4b254389122332181a7c3db7f27e918eec64e3",
}
# Each checkpoint's kind and the qwen-tts method that speaks with it.
MODEL_KINDS = {
    "qwen3/0.6b-custom-voice": "custom_voice",
    "qwen3/0.6b-base": "base",
    "qwen3/1.7b-custom-voice": "custom_voice",
    "qwen3/1.7b-voice-design": "voice_design",
    "qwen3/1.7b-base": "base",
}
GENERATE_METHODS = {
    "custom_voice": "generate_custom_voice",
    "voice_design": "generate_voice_design",
    "base": "generate_voice_clone",
}
# qwen-tts drops a style instruction on the 0.6B CustomVoice; the 1.7B one follows it.
TAKES_INSTRUCT = frozenset({"qwen3/1.7b-custom-voice", "qwen3/1.7b-voice-design"})
# In float32 the 1.7B checkpoints need about 7.7 GB, more than an 8 GB card has; float16 runs them before sm_80.
FLOAT16_BEFORE_SM80 = frozenset({"qwen3/1.7b-custom-voice", "qwen3/1.7b-voice-design", "qwen3/1.7b-base"})

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
# The model's own names (any case) and ISO 639-1 codes; regional tags fall back to these.
LANGUAGE_NAMES = {
    **{name.lower(): name for name in LANGUAGES},
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

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

# GPU memory once loaded, in MiB (worker contract: the manifest's gpu_memory_mb). Measured on an RTX 2070 SUPER, which
# runs the 0.6B in float32 (no bfloat16 before sm_80): about 3490 MiB reserved, and bfloat16 on a newer card takes
# less. The 1.7B in float16 there held 5.1 GiB.
GPU_MEMORY_MB = {
    "qwen3/0.6b-custom-voice": 3500,
    "qwen3/0.6b-base": 3500,
    "qwen3/1.7b-custom-voice": 5200,
    "qwen3/1.7b-voice-design": 5200,
    "qwen3/1.7b-base": 5200,
}

KIND_CAPABILITIES = {
    "custom_voice": {
        "voice_clone": False,
        "reference_audio": False,
        "clone_transcript": False,
        "speakers": VOICES,
    },
    "base": {
        "live_reader": False,
        "voice_clone": True,
        "reference_audio": True,
        "clone_transcript": True,
        "clone_transcript_required": True,
        "speakers": [],
    },
    # A voice made from a description (voice_design, or instructions): a new one for each request without a seed,
    # so not for the Live Reader, which speaks each sentence as a request of its own.
    "voice_design": {
        "voice_design": True,
        "live_reader": False,
        "voice_clone": False,
        "reference_audio": False,
        "clone_transcript": False,
        "speakers": [],
    },
}


def _model_manifest(model_id: str) -> dict[str, Any]:
    kind = MODEL_KINDS[model_id]
    capabilities = {**COMMON_CAPABILITIES, **KIND_CAPABILITIES[kind]}
    if model_id in TAKES_INSTRUCT:
        capabilities["instructions"] = True
    return {
        "id": model_id,
        "provider": PROVIDER,
        "sample_rate": SAMPLE_RATE,
        "revision": MODEL_REVISIONS[model_id],
        "max_input_chars": 4096,
        "gpu_memory_mb": GPU_MEMORY_MB[model_id],
        "capabilities": capabilities,
        "voices": VOICES if kind == "custom_voice" else [],
    }


MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "provider": PROVIDER,
    "models": [
        _model_manifest(model_id)
        for model_id in MODEL_IDS
        if ENABLE_BASE or MODEL_KINDS[model_id] != "base"
    ],
}


def _model_is_advertised(model_id: str) -> bool:
    return any(model["id"] == model_id for model in MANIFEST["models"])


class SynthesisRequest(common.SynthesisRequest):
    text: str = Field(min_length=1, max_length=4096)
    voice: str = "Ryan"
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


_detail = common.error_detail


def _random_seed() -> int:
    return secrets.randbelow(2**31)


def _resolve_dtype_name(
    setting: str, capability: tuple[int, int], model_id: str | None = None
) -> str:
    """Resolve a stable inference dtype without relying on emulated BF16 checks."""
    normalized = setting.strip().lower()
    if normalized == "auto":
        if capability >= (8, 0):
            return "bfloat16"
        return "float16" if model_id in FLOAT16_BEFORE_SM80 else "float32"
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
    if not language or not language.strip():
        return "Auto"
    name = common.language_name(language, LANGUAGE_NAMES)
    if name is None:
        raise WorkerFailure(
            f"Unsupported language '{language}'. Supported: {', '.join(LANGUAGES)}",
            code="language_unsupported",
            status_code=400,
        )
    return name


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


_sentence_parts = common.sentence_parts


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
    method_name = GENERATE_METHODS[MODEL_KINDS[model_id]]
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
    if model_id in TAKES_INSTRUCT and "instruct" not in parameters:
        missing.append("instruct")
    if missing:
        raise WorkerFailure(
            f"Pinned qwen-tts {method_name} does not accept: {', '.join(missing)}",
            code="provider_api_mismatch",
            status_code=503,
        )


VOICE_IDS = frozenset(voice["id"] for voice in VOICES)


def _voice_description(request: SynthesisRequest) -> str:
    parts = ((request.voice_design or "").strip(), (request.instructions or "").strip())
    return " ".join(part for part in parts if part)


def check_request(request: SynthesisRequest) -> None:
    """Refuse what the model can't do, before any generation."""
    kind = MODEL_KINDS[request.model]
    if request.speed != 1.0:
        raise WorkerFailure(
            f"Numeric speed is not supported by {request.model}",
            code="speed_unsupported",
            status_code=400,
        )
    if kind == "custom_voice" and request.voice not in VOICE_IDS:
        raise WorkerFailure(
            f"Unsupported voice '{request.voice}'",
            code="voice_unsupported",
            status_code=400,
        )
    if kind == "voice_design":
        if not _voice_description(request):
            raise WorkerFailure(
                f"{request.model} speaks in a voice made from a description: "
                "send one as instructions (or voice_design)",
                code="voice_required",
                status_code=400,
            )
        return
    if (request.voice_design or "").strip():
        raise WorkerFailure(
            f"Voice design is not supported by {request.model}",
            code="voice_design_unsupported",
            status_code=400,
        )
    if (request.instructions or "").strip() and request.model not in TAKES_INSTRUCT:
        raise WorkerFailure(
            f"Instructions are not supported by {request.model}",
            code="instructions_unsupported",
            status_code=400,
        )


def _snapshot(model_id: str, *, local_files_only: bool = False) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_IDS[model_id],
        revision=MODEL_REVISIONS[model_id],
        local_files_only=local_files_only,
    )


def _model_path(model_id: str) -> str:
    """The checkpoint's local folder, fetched first if it isn't cached. Given a repository ID, transformers asks the
    Hub about its processor even when the files are cached."""
    try:
        return _snapshot(model_id, local_files_only=True)
    except Exception:
        return _snapshot(model_id)


class QwenRuntime(common.WorkerRuntime):
    display_name = "Qwen3-TTS"
    gpu_memory_need_mb = GPU_MEMORY_MB

    def __init__(self) -> None:
        super().__init__(reference_cache_size=PROMPT_CACHE_SIZE)
        self.dtype_name: str | None = None
        self.parameter_dtypes: dict[str, list[str]] = {}
        self.failed_code: str | None = None
        self.failed_message: str | None = None

    def _context_failure(self) -> WorkerFailure:
        assert self.failed_code is not None
        return WorkerFailure(
            "CUDA context failed; the Qwen worker is restarting",
            code=self.failed_code,
            status_code=503,
        )

    def _mark_cuda_context_failed(self, exc: BaseException) -> WorkerFailure:
        self.failed_code = "cuda_context_failed"
        self.failed_message = str(exc)
        return self._context_failure()

    def _refresh_parameter_dtypes(self) -> None:
        try:
            self.parameter_dtypes = _parameter_dtype_summary(self.model)
        except Exception as exc:
            logger.warning("Could not inspect Qwen parameter dtypes: %s", exc)

    def is_known_model(self, model_id: str) -> bool:
        return _model_is_advertised(model_id)

    def check_usable(self) -> None:
        if self.failed_code:
            raise WorkerFailure(
                "CUDA context failed; the Qwen worker must restart",
                code=self.failed_code,
                status_code=503,
            )

    def download(self, model_id: str) -> None:
        """Fetch the checkpoint into the cache without loading it, so no GPU memory is held meanwhile."""
        if not self.is_known_model(model_id):
            raise WorkerFailure(f"Unknown model: {model_id}", code="unknown_model", status_code=400)
        try:
            _snapshot(model_id)
        except Exception as exc:
            raise WorkerFailure(
                f"Failed to fetch {MODEL_IDS[model_id]}: {exc}", code="model_download_failed"
            ) from exc

    def build_model(self, model_id: str, capability: tuple[int, int]) -> Any:
        dtype_name = _resolve_dtype_name(DTYPE_SETTING, capability, model_id)
        if DTYPE_SETTING == "float16" and model_id not in FLOAT16_BEFORE_SM80:
            logger.warning(
                "QWEN3_DTYPE=float16 is an explicit experimental override; "
                "use auto for the hardware-safe policy"
            )
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            _model_path(model_id),
            device_map="cuda:0",
            dtype=getattr(torch, dtype_name),
            attn_implementation="sdpa",
        )
        _validate_generation_api(model, model_id)
        self.dtype_name = dtype_name
        return model

    def after_load(self) -> None:
        self._refresh_parameter_dtypes()

    def load_failure(self, exc: Exception) -> WorkerFailure | None:
        # Unloading would call into the poisoned context; the process restarts instead.
        if _is_cuda_context_failure(exc):
            return self._mark_cuda_context_failed(exc)
        return None

    def unload(self) -> None:
        self.dtype_name = None
        self.parameter_dtypes = {}
        super().unload()

    def _decode_reference(self, encoded: str | None) -> bytes:
        return common.decode_reference_audio(
            encoded,
            max_bytes=MAX_REFERENCE_BYTES,
            missing_message="reference_audio is required for the Base model",
        )

    def _clone_prompt(self, reference_audio: bytes, transcript: str) -> Any:
        assert self.model is not None
        key = self.references.key(self.model_id, reference_audio, transcript)
        cached = self.references.get(key)
        if cached is not None:
            return cached
        with common.reference_file(reference_audio) as reference_path:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=reference_path,
                ref_text=transcript,
                x_vector_only_mode=False,
            )
        self.references.put(key, prompt)
        return prompt

    def _run_model(
        self,
        model: Any,
        request: SynthesisRequest,
        text: str,
        language: str,
        max_new_tokens: int,
    ) -> tuple[Any, int]:
        kind = MODEL_KINDS[request.model]
        instruct = (request.instructions or "").strip()
        if kind == "custom_voice":
            return model.generate_custom_voice(
                text=text,
                language=language,
                speaker=request.voice,
                non_streaming_mode=True,
                max_new_tokens=max_new_tokens,
                **({"instruct": instruct} if instruct else {}),
            )
        if kind == "voice_design":
            return model.generate_voice_design(
                text=text,
                language=language,
                instruct=_voice_description(request),
                non_streaming_mode=True,
                max_new_tokens=max_new_tokens,
            )
        transcript = (request.clone_transcript or "").strip()
        if not transcript:
            raise WorkerFailure(
                "clone_transcript is required for the Base model",
                code="clone_transcript_required",
                status_code=400,
            )
        reference_audio = self._decode_reference(request.reference_audio)
        prompt = self._clone_prompt(reference_audio, transcript)
        return model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            non_streaming_mode=True,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, request: SynthesisRequest, text: str) -> np.ndarray:
        if self.failed_code:
            raise self._context_failure()
        model = self.require_model(request.model)
        check_request(request)
        language = _normalize_language(request.language)

        max_new_tokens = _max_new_tokens(text)
        if request.seed is not None:
            # Every segment of a request starts from the same seed, so a designed voice stays the same throughout.
            torch.manual_seed(request.seed)
        try:
            wavs, sample_rate = common.guard_out_of_memory(
                lambda: self._run_model(model, request, text, language, max_new_tokens),
                OUT_OF_MEMORY_MESSAGE,
            )
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
        audio = common.float32_audio(wavs[0], message="Model returned empty or non-finite audio")
        audio_seconds = audio.size / SAMPLE_RATE
        # A run that spent its whole token budget decodes to max_new_tokens codec tokens.
        generation_limit_samples = max_new_tokens * CODEC_SAMPLES_PER_TOKEN
        margin_samples = round(GENERATION_LIMIT_MARGIN_S * SAMPLE_RATE)
        if audio.size >= max(0, generation_limit_samples - margin_samples):
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
        return audio


_validate_generation_settings()
runtime = QwenRuntime()
operation_lock = asyncio.Lock()
app = common.create_app("Open Speech Qwen3 Worker")


def _raise_if_context_failed() -> None:
    if runtime.failed_code:
        raise HTTPException(503, detail=_detail(runtime._context_failure()))


def _after_load_failure(exc: WorkerFailure) -> None:
    if exc.code == "cuda_context_failed":
        _schedule_process_restart()


@app.get("/health")
async def health() -> dict[str, Any]:
    gpu, capability, health_error = common.gpu_health(query=runtime.failed_code is None)
    resolved_dtype = runtime.dtype_name
    if runtime.failed_code:
        gpu["error"] = "CUDA context failed"
    elif capability is not None and resolved_dtype is None:
        try:
            resolved_dtype = _resolve_dtype_name(DTYPE_SETTING, capability)
        except WorkerFailure as exc:
            # A load checks the GPU architecture first (cuda_arch_unsupported), so that one stays.
            health_error = health_error or exc

    if runtime.failed_code:
        status = "failed"
        failure = {"code": runtime.failed_code, "message": runtime.failed_message}
    elif health_error is not None:
        status = "failed"
        failure = _detail(health_error)
    else:
        status = "ok" if gpu["available"] else "unavailable"
        failure = None
    return common.health_body(
        PROVIDER,
        runtime,
        status=status,
        busy=operation_lock.locked(),
        gpu=gpu,
        details={
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
        },
    )


@app.get("/v1/manifest")
async def manifest() -> dict[str, Any]:
    return MANIFEST


@app.post("/v1/models/load")
async def load_model(payload: LoadRequest) -> dict[str, Any]:
    return await common.load_model(
        runtime, operation_lock, payload.model, on_failure=_after_load_failure
    )


@app.post("/v1/models/download")
async def download_model(payload: LoadRequest) -> dict[str, Any]:
    return await common.download_model(runtime, payload.model)


@app.post("/v1/models/unload")
async def unload_model(payload: LoadRequest) -> dict[str, str]:
    _raise_if_context_failed()
    return await common.unload_model(runtime, operation_lock, payload.model)


@app.post("/v1/audio/speech")
async def synthesize(payload: SynthesisRequest, request: Request) -> StreamingResponse:
    if not _model_is_advertised(payload.model):
        raise common.unknown_model_error(payload.model)
    _raise_if_context_failed()
    if MODEL_KINDS[payload.model] == "voice_design" and payload.seed is None:
        payload = payload.model_copy(update={"seed": _random_seed()})
    common.ensure_idle(operation_lock)
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
        raise common.http_error(first_item) from first_item
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

    return common.audio_response(
        generate(),
        sample_rate=SAMPLE_RATE,
        cancellation="between_segments",
        background=BackgroundTask(close_consumer),
    )
