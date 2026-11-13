"""Isolated Fish Audio S2 Pro worker for the Open Speech harness.

S2 Pro runs through a community fork of fish-speech that quantizes the 4B Slow AR to bitsandbytes NF4
(scarxity/fish-speech-int4-patch, weights scarxity/fish-speech-s2-pro-nf4), so it fits next to the core's
models on an 8 GB card. The GPU holds a decode-only
codec; each reference recording is encoded once, on the CPU, and handed to the engine as prompt tokens.

The fork's ModelManager keeps the model on a daemon thread and has no teardown, so an unload cannot free
the GPU: the worker exits FISH_RESTART_AFTER_UNLOAD_S seconds after an unload and Compose restarts it
empty, as the Chatterbox worker does.
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import signal
import time
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import Field

from providers._common import worker as common
from providers._common.worker import LoadRequest, WorkerFailure

PROVIDER = "fish"
SCHEMA_VERSION = common.SCHEMA_VERSION
MODEL_ID = "fish/s2-pro"
MODEL_REPO = "scarxity/fish-speech-s2-pro-nf4"
MODEL_REVISION = "279f07f238504911b6724157cc9f07fb7e0a23d2"
# The fork the image installs (providers/fish/Dockerfile, FISH_COMMIT); reported in /health.
FORK_COMMIT = "3fd7da6a65dfeb9f8ad1975813301ed16c4da47a"
# Everything the fork reads from the snapshot: the NF4 Slow/Fast AR, the codec, and the tokenizer (about 5.0 GB).
MODEL_FILES = (
    "model.pth",
    "codec.pth",
    "tokenizer.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "config.json",
    "tokenizer_config.json",
    "LICENSE.md",
)
CODEC_CONFIG = "modded_dac_vq"
# The codec decodes 44.1 kHz audio at about 21.5 frames a second: 2048 samples per codec token.
SAMPLE_RATE = 44100
CODEC_SAMPLES_PER_TOKEN = 2048
GENERATION_LIMIT_MARGIN_S = 1.0
MAX_NEW_TOKENS_CEILING = 1024
TOKENS_PER_CHARACTER = 2.2
TOKEN_OVERHEAD = 64
WIDE_CHARACTER_UNITS = 3.2
CHUNK_LENGTH = 200

MAX_INPUT_CHARS = int(os.environ.get("FISH_MAX_INPUT_CHARS", "1500"))
SEGMENT_CHARS = int(os.environ.get("FISH_SEGMENT_CHARS", "300"))
MAX_REFERENCE_BYTES = int(os.environ.get("FISH_MAX_REFERENCE_MB", "100")) * 1024 * 1024
MAX_REFERENCE_SECONDS = float(os.environ.get("FISH_MAX_REFERENCE_SECONDS", "30"))
COMPILE = common.env_flag("FISH_COMPILE", "true")
MAX_SEQ_LEN = int(os.environ.get("FISH_MAX_SEQ_LEN", "2048"))
REFERENCE_CACHE_SIZE = int(
    os.environ.get("FISH_REFERENCE_CACHE_SIZE", str(common.DEFAULT_REFERENCE_CACHE_SIZE))
)
TEMPERATURE = float(os.environ.get("FISH_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("FISH_TOP_P", "0.7"))
# One seed for every segment, so the segments of one text keep one voice.
SEED = int(os.environ.get("FISH_SEED", "42"))
RESTART_AFTER_UNLOAD_S = float(os.environ.get("FISH_RESTART_AFTER_UNLOAD_S", "1"))
# The fork's ReferenceLoader.load_by_id reads references/<id>/ under the working directory.
REFERENCES_DIR = Path("references")
OUT_OF_MEMORY_MESSAGE = "CUDA out of memory during Fish Audio S2 synthesis"
COMPILE_ERROR_MARKERS = ("torch._dynamo", "torch._inductor", "triton", "inductor")

logger = common.logger

# Tiers 1 and 2 of the model card; only English is tested.
LANGUAGES = ["en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de"]
LANGUAGE_CODES = {code: code for code in LANGUAGES}

MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "provider": PROVIDER,
    "models": [
        {
            "id": MODEL_ID,
            "provider": PROVIDER,
            "sample_rate": SAMPLE_RATE,
            "revision": MODEL_REVISION,
            "max_input_chars": MAX_INPUT_CHARS,
            "capabilities": {
                "voice_blend": False,
                "voice_design": False,
                "voice_clone": True,
                "reference_audio": True,
                "clone_transcript": True,
                "clone_transcript_required": True,
                "streaming": True,
                "instructions": False,
                "languages": list(LANGUAGES),
                "speed_control": False,
                "speed_mode": "none",
                "ssml": False,
                "batch": False,
                "cancellation": "between_segments",
                "live_reader": True,
                "native_tags": True,
                "speakers": [],
            },
            "voices": [],
        }
    ],
}


def _validate_settings() -> None:
    if MAX_INPUT_CHARS <= 0:
        raise RuntimeError("FISH_MAX_INPUT_CHARS must be greater than zero")
    if SEGMENT_CHARS <= 0:
        raise RuntimeError("FISH_SEGMENT_CHARS must be greater than zero")


class SynthesisRequest(common.SynthesisRequest):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


def _text_units(text: str) -> float:
    """Text length weighted for speech time: a wide (CJK) character is spoken about as long as 3.2 Latin ones."""
    return math.fsum(
        WIDE_CHARACTER_UNITS if unicodedata.east_asian_width(character) in {"W", "F"} else 1.0
        for character in text
    )


def _max_new_tokens(segment: str) -> int:
    """Codec tokens one segment may take: about 1.5 a Latin character in speech, with headroom."""
    return min(MAX_NEW_TOKENS_CEILING, math.ceil(_text_units(segment) * TOKENS_PER_CHARACTER) + TOKEN_OVERHEAD)


def _prefix_within_units(text: str, limit: float) -> int:
    units = 0.0
    for index, character in enumerate(text):
        units += _text_units(character)
        if index and units > limit + 1e-9:
            return index
    return len(text)


def _hard_split(sentence: str) -> list[str]:
    """Cut a sentence longer than SEGMENT_CHARS units at its last space within the limit (or at the limit)."""
    parts: list[str] = []
    remaining = sentence
    while _text_units(remaining) > SEGMENT_CHARS:
        hard = _prefix_within_units(remaining, SEGMENT_CHARS)
        split_at = remaining[:hard].rfind(" ")
        if split_at <= 0 or _text_units(remaining[:split_at]) < SEGMENT_CHARS / 2:
            split_at = hard
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _segments(text: str) -> list[str]:
    """Whole sentences packed into segments of at most SEGMENT_CHARS units; a longer sentence is cut."""
    segments: list[str] = []
    for sentence in common.sentence_parts(text) or [text.strip()]:
        for part in _hard_split(sentence):
            if segments and _text_units(f"{segments[-1]} {part}") <= SEGMENT_CHARS:
                segments[-1] = f"{segments[-1]} {part}"
            else:
                segments.append(part)
    return [segment for segment in segments if segment]


def _check_request(payload: SynthesisRequest) -> tuple[bytes, str]:
    """Refuse what S2 Pro can't do before any work; return the reference audio and its transcript."""
    if payload.speed != 1.0:
        raise WorkerFailure(
            "Fish Audio S2 does not support numeric speed control",
            code="speed_unsupported",
            status_code=400,
        )
    if payload.instructions or payload.voice_design:
        raise WorkerFailure(
            "Fish Audio S2 does not take separate instructions or a voice design",
            code="instructions_unsupported",
            status_code=400,
        )
    if payload.language and payload.language.strip():
        if common.language_name(payload.language, LANGUAGE_CODES) is None:
            raise WorkerFailure(
                f"Unsupported language '{payload.language}' for {MODEL_ID}. "
                f"Supported: {', '.join(LANGUAGES)}",
                code="language_unsupported",
                status_code=400,
            )
    audio = common.decode_reference_audio(
        payload.reference_audio,
        max_bytes=MAX_REFERENCE_BYTES,
        missing_message="reference_audio is required for Fish Audio S2 voice cloning",
    )
    transcript = (payload.clone_transcript or "").strip()
    if not transcript:
        raise WorkerFailure(
            "clone_transcript is required for Fish Audio S2 voice cloning",
            code="clone_transcript_required",
            status_code=400,
        )
    info = common.reference_audio_info(audio)
    if info.duration > MAX_REFERENCE_SECONDS:
        raise WorkerFailure(
            f"Fish Audio S2 reference audio cannot exceed {MAX_REFERENCE_SECONDS:g} seconds",
            code="reference_audio_too_long",
            status_code=400,
        )
    return audio, transcript


# ── The fork's API, at FISH_COMMIT. The image build asserts each name (providers/fish/Dockerfile). ──


def _snapshot() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES)
    )


def _load_engine(snapshot: str) -> tuple[Any, Any]:
    """Build the fork's ModelManager (it warms up on "Hello world.") and return it and its engine."""
    from tools.server.model_manager import ModelManager

    manager = ModelManager(
        mode="tts",
        device="cuda",
        # Turing has no bfloat16, the fork's default precision; NF4 computes in float16.
        half=True,
        compile=COMPILE,
        bnb4=True,
        llama_checkpoint_path=snapshot,
        decoder_checkpoint_path=str(Path(snapshot) / "codec.pth"),
        decoder_config_name=CODEC_CONFIG,
        max_seq_len=MAX_SEQ_LEN,
        lazy_load=False,
        codec_decode_only=True,
    )
    return manager, manager.tts_inference_engine


def _load_reference_codec(snapshot: str) -> Any:
    """The full codec, encoder included, held on the CPU for references (about 1.9 GB of host RAM)."""
    from fish_speech.models.dac.inference import load_model

    return load_model(
        config_name=CODEC_CONFIG,
        checkpoint_path=str(Path(snapshot) / "codec.pth"),
        device="cpu",
    )


def _encode_reference(codec: Any, wav_path: Path, tokens_path: Path) -> Any:
    """Encode a reference recording into prompt tokens, as the fork's ReferenceLoader does, and save them."""
    import soundfile as sf
    import torchaudio

    samples, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(samples.mean(axis=1))
    target = int(getattr(getattr(codec, "spec_transform", None), "sample_rate", 0) or codec.sample_rate)
    if rate != target:
        audio = torchaudio.functional.resample(audio, rate, target)
    audios = audio[None, None, :]
    lengths = torch.tensor([audios.shape[2]], dtype=torch.long)
    with torch.inference_mode():
        tokens = codec.encode(audios, lengths)[0][0]
    torch.save(tokens, tokens_path)
    return tokens


def _serve_request(**fields: Any) -> Any:
    from fish_speech.utils.schema import ServeTTSRequest

    return ServeTTSRequest(**fields)


def _engine_sample_rate(engine: Any) -> int | None:
    decoder = getattr(engine, "decoder_model", None)
    rate = getattr(getattr(decoder, "spec_transform", None), "sample_rate", None)
    if rate is None:
        rate = getattr(decoder, "sample_rate", None)
    return int(rate) if rate is not None else None


def _is_compile_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        where = f"{type(current).__module__}.{type(current).__name__} {current}".lower()
        if any(marker in where for marker in COMPILE_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


class FishRuntime(common.WorkerRuntime):
    display_name = "Fish Audio S2"

    def __init__(self) -> None:
        super().__init__(reference_cache_size=REFERENCE_CACHE_SIZE)
        self.cuda_used = False
        self.manager: Any | None = None
        self.snapshot: str | None = None
        self.codec: Any | None = None
        self.reference_keys: set[str] = set()

    def is_known_model(self, model_id: str) -> bool:
        return model_id == MODEL_ID

    def build_model(self, model_id: str, capability: tuple[int, int]) -> Any:
        self.cuda_used = True
        snapshot = _snapshot()
        manager, engine = _load_engine(snapshot)
        self.manager = manager
        self.snapshot = snapshot
        rate = _engine_sample_rate(engine)
        if rate is not None and rate != SAMPLE_RATE:
            raise WorkerFailure(
                f"Model sample rate {rate} does not match {SAMPLE_RATE}",
                code="sample_rate_mismatch",
            )
        return engine

    def load_failure(self, exc: Exception) -> WorkerFailure | None:
        if not COMPILE or not _is_compile_failure(exc):
            return None
        self.unload()
        return WorkerFailure(
            f"torch.compile failed on this GPU ({type(exc).__name__}: {exc}); "
            "set FISH_COMPILE=false to run without it",
            code="compile_failed",
            status_code=503,
        )

    def unload(self) -> None:
        for key in self.reference_keys:
            shutil.rmtree(REFERENCES_DIR / key, ignore_errors=True)
        self.reference_keys.clear()
        self.manager = None
        self.snapshot = None
        self.codec = None
        super().unload()

    def _forget_reference(self, engine: Any, key: str) -> None:
        cache = getattr(engine, "ref_by_id", None)
        if isinstance(cache, dict):
            cache.pop(key, None)
        shutil.rmtree(REFERENCES_DIR / key, ignore_errors=True)
        self.reference_keys.discard(key)

    def _reference_id(self, engine: Any, audio: bytes, transcript: str) -> str:
        """The engine's reference id for this recording and transcript, encoding it on a cache miss.

        The tokens go into the engine's own reference cache (``ref_by_id``, read when a request asks
        for ``use_memory_cache="on"``) and, beside the recording and its ``.lab`` transcript, into
        references/<id>/, where the fork's ``load_by_id`` looks.
        """
        key = self.references.key(self.model_id, audio, transcript)
        if self.references.get(key) is not None:
            return key
        folder = REFERENCES_DIR / key
        folder.mkdir(parents=True, exist_ok=True)
        self.reference_keys.add(key)
        (folder / "ref.wav").write_bytes(audio)
        (folder / "ref.lab").write_text(transcript, encoding="utf-8")
        if self.codec is None:
            assert self.snapshot is not None
            self.codec = _load_reference_codec(self.snapshot)
        tokens = _encode_reference(self.codec, folder / "ref.wav", folder / "ref.tokens.pt")
        cache = getattr(engine, "ref_by_id", None)
        if isinstance(cache, dict):
            cache[key] = ([tokens], [transcript])
        for evicted in self.references.put(key, key):
            self._forget_reference(engine, evicted)
        return key

    def _speak(self, engine: Any, segment: str, reference_id: str) -> np.ndarray:
        max_new_tokens = _max_new_tokens(segment)
        request = _serve_request(
            text=segment,
            reference_id=reference_id,
            use_memory_cache="on",
            max_new_tokens=max_new_tokens,
            chunk_length=CHUNK_LENGTH,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            seed=SEED,
            format="wav",
            streaming=False,
        )
        final: Any = None
        for result in engine.inference(request):
            code = getattr(result, "code", None)
            if code == "error":
                error = getattr(result, "error", None)
                raise error if isinstance(error, BaseException) else RuntimeError(str(error))
            if code == "final":
                final = getattr(result, "audio", None)
        if final is None:
            raise WorkerFailure("Model returned no audio", code="invalid_audio")
        rate, samples = final
        if int(rate) != SAMPLE_RATE:
            raise WorkerFailure(
                f"Model returned {rate} Hz; manifest declares {SAMPLE_RATE} Hz",
                code="sample_rate_mismatch",
            )
        audio = common.float32_audio(samples, message="Model returned empty or non-finite audio")
        # A run that spent its whole token budget decodes to max_new_tokens codec tokens.
        limit = max_new_tokens * CODEC_SAMPLES_PER_TOKEN - round(GENERATION_LIMIT_MARGIN_S * SAMPLE_RATE)
        if audio.size >= max(0, limit):
            logger.warning(
                "Fish generation hit its token limit: audio_seconds=%.2f max_new_tokens=%d chars=%d",
                audio.size / SAMPLE_RATE,
                max_new_tokens,
                len(segment),
            )
            raise WorkerFailure(
                "Fish Audio S2 reached its generation limit; no partial audio was returned",
                code="generation_limit_reached",
            )
        return audio

    def generate(self, payload: SynthesisRequest) -> Iterator[np.ndarray]:
        """Yield each segment's audio; every refusal comes before any generation."""
        audio, transcript = _check_request(payload)
        engine = self.require_model(payload.model)
        try:
            with torch.inference_mode():
                reference_id = common.guard_out_of_memory(
                    lambda: self._reference_id(engine, audio, transcript), OUT_OF_MEMORY_MESSAGE
                )
            for segment in _segments(payload.text):
                # inference_mode is thread-local, and each chunk may be pulled on another thread.
                with torch.inference_mode():
                    chunk = common.guard_out_of_memory(
                        lambda: self._speak(engine, segment, reference_id), OUT_OF_MEMORY_MESSAGE
                    )
                self.last_used_at = time.time()
                yield chunk
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(
                f"Fish Audio S2 generation failed: {exc}",
                code="generation_failed",
            ) from exc


_validate_settings()
runtime = FishRuntime()
operation_lock = asyncio.Lock()
restart_task: asyncio.Task[None] | None = None
app = common.create_app("Open Speech Fish Audio S2 Worker")


def _restart_pending() -> bool:
    return restart_task is not None and not restart_task.done()


def _schedule_restart() -> None:
    """Exit soon after an unload (or a failed load) so Compose restarts the worker with its memory released."""
    global restart_task
    if RESTART_AFTER_UNLOAD_S <= 0 or not runtime.cuda_used or _restart_pending():
        return
    restart_task = asyncio.get_running_loop().create_task(_restart_when_idle())


async def _restart_when_idle() -> None:
    await asyncio.sleep(RESTART_AFTER_UNLOAD_S)
    async with operation_lock:
        if runtime.model is not None:
            return
        logger.info("Exiting after unload so the container restarts with the model's memory released")
        os.kill(os.getpid(), signal.SIGTERM)


@app.get("/health")
async def health() -> dict[str, Any]:
    return common.health_payload(
        PROVIDER,
        runtime,
        busy=operation_lock.locked(),
        details={
            "quantization": "nf4",
            "precision": "float16",
            "compile": COMPILE,
            "max_seq_len": MAX_SEQ_LEN,
            "codec": "decode-only",
            "reference_encoder": "cpu",
            "fork_commit": FORK_COMMIT,
            "restart_pending": _restart_pending(),
        },
    )


@app.get("/v1/manifest")
async def manifest() -> dict[str, Any]:
    return MANIFEST


@app.post("/v1/models/load")
async def load_model(payload: LoadRequest) -> dict[str, Any]:
    if _restart_pending():
        # The unloaded model still holds the GPU on the fork's thread; a second one would not fit beside it.
        raise HTTPException(
            503,
            detail={
                "code": "worker_restarting",
                "message": "The Fish worker is restarting to release the unloaded model; load again once it is back",
            },
        )
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
    if payload.model != MODEL_ID:
        raise common.unknown_model_error(payload.model)
    common.ensure_idle(operation_lock)
    await operation_lock.acquire()
    return await common.stream_chunks(
        runtime.generate(payload),
        lock=operation_lock,
        request=request,
        sample_rate=SAMPLE_RATE,
        cancellation="between_segments",
    )
