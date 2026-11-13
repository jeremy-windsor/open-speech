"""Isolated Chatterbox worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException, Request  # noqa: F401 - callers catch worker.HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from providers._common import worker as common
from providers._common.worker import LoadRequest, WorkerFailure

PROVIDER = "chatterbox"
SCHEMA_VERSION = common.SCHEMA_VERSION
SAMPLE_RATE = 24000
MAX_INPUT_CHARS = int(os.environ.get("CHATTERBOX_MAX_INPUT_CHARS", "350"))
MAX_REFERENCE_BYTES = int(os.environ.get("CHATTERBOX_MAX_REFERENCE_MB", "100")) * 1024 * 1024
REFERENCE_CACHE_SIZE = int(
    os.environ.get("CHATTERBOX_REFERENCE_CACHE_SIZE", str(common.DEFAULT_REFERENCE_CACHE_SIZE))
)
# Unloading drops the weights, but the process keeps the CUDA context, the CUDA libraries and
# allocator arenas (about 1.5 GiB of host RAM). On a small Docker Desktop VM that is enough to
# get another worker OOM-killed, so after an unload the worker exits once it has stayed idle this
# many seconds and Compose's restart policy starts a fresh, empty process. 0 disables it.
RESTART_AFTER_UNLOAD_S = float(os.environ.get("CHATTERBOX_RESTART_AFTER_UNLOAD_S", "0"))

logger = common.logger

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


class SynthesisRequest(common.SynthesisRequest):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


def _decode_reference(encoded: str | None, *, turbo: bool) -> bytes:
    audio = common.decode_reference_audio(
        encoded,
        max_bytes=MAX_REFERENCE_BYTES,
        missing_message="reference_audio is required for Chatterbox voice cloning",
    )
    info = common.reference_audio_info(audio)
    if turbo and info.duration <= 5.0:
        raise WorkerFailure(
            "Chatterbox Turbo reference audio must be longer than five seconds",
            code="reference_audio_too_short",
            status_code=400,
        )
    return audio


class ChatterboxRuntime(common.WorkerRuntime):
    display_name = "Chatterbox"

    def __init__(self) -> None:
        super().__init__(reference_cache_size=REFERENCE_CACHE_SIZE)
        self.cuda_used = False

    def is_known_model(self, model_id: str) -> bool:
        return model_id in MODEL_IDS

    def build_model(self, model_id: str, capability: tuple[int, int]) -> Any:
        self.cuda_used = True
        from huggingface_hub import snapshot_download

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
        return model

    def _generate_with_reference(self, model: Any, text: str, reference: bytes) -> Any:
        """Reuse the reference's conditionals (speaker embedding, prompt tokens) when cached.

        Chatterbox prepares ``model.conds`` from ``audio_prompt_path`` on every call, which
        repeats the reference analysis for each Live Reader sentence. Only conditionals the
        call rebuilt are cached; models that leave ``conds`` alone always take the path.
        """
        key = self.references.key(self.model_id, reference)
        if hasattr(model, "conds"):
            cached = self.references.get(key)
            if cached is not None:
                model.conds = cached
                return model.generate(text)
        previous = getattr(model, "conds", None)
        with common.reference_file(reference) as reference_path:
            result = model.generate(text, audio_prompt_path=reference_path)
        conds = getattr(model, "conds", None)
        if conds is not None and conds is not previous:
            self.references.put(key, conds)
        return result

    def generate(self, payload: SynthesisRequest) -> np.ndarray:
        model = self.require_model(payload.model)
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
        try:
            with torch.inference_mode():
                result = common.guard_out_of_memory(
                    lambda: self._generate_with_reference(model, payload.text, reference),
                    "CUDA out of memory during Chatterbox synthesis",
                )
            audio = common.float32_audio(result)
            self.last_used_at = time.time()
            return audio
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(
                f"Chatterbox synthesis failed: {exc}",
                code="synthesis_failed",
            ) from exc


runtime = ChatterboxRuntime()
operation_lock = asyncio.Lock()
restart_task: asyncio.Task[None] | None = None
app = common.create_app("Open Speech Chatterbox Worker")


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
    # A load that arrives during the wait (switching regular <-> turbo) cancels this task.
    await asyncio.sleep(RESTART_AFTER_UNLOAD_S)
    async with operation_lock:
        if runtime.model is not None:
            return
        logger.info("Exiting after unload so the container restarts with its memory released")
        os.kill(os.getpid(), signal.SIGTERM)


def _utterance(payload: SynthesisRequest) -> Iterator[np.ndarray]:
    yield runtime.generate(payload)


@app.get("/health")
async def health() -> dict[str, Any]:
    return common.health_payload(PROVIDER, runtime, busy=operation_lock.locked())


@app.get("/v1/manifest")
async def manifest() -> dict[str, Any]:
    return MANIFEST


@app.post("/v1/models/load")
async def load_model(payload: LoadRequest) -> dict[str, Any]:
    _cancel_restart()
    return await common.load_model(
        runtime, operation_lock, payload.model, on_failure=lambda _exc: _schedule_restart()
    )


@app.post("/v1/models/unload")
async def unload_model(payload: LoadRequest) -> dict[str, str]:
    result = await common.unload_model(runtime, operation_lock, payload.model)
    _schedule_restart()
    return result


@app.post("/v1/audio/speech")
async def synthesize(payload: SynthesisRequest, request: Request) -> StreamingResponse:
    if payload.model not in MODEL_IDS:
        raise common.unknown_model_error(payload.model)
    common.ensure_idle(operation_lock)
    await operation_lock.acquire()
    return await common.stream_chunks(
        _utterance(payload),
        lock=operation_lock,
        request=request,
        sample_rate=SAMPLE_RATE,
        cancellation="after_utterance",
    )

