"""Text-to-speech service helpers."""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.audio.postprocessing import process_tts_chunks
from src.effects.chain import apply_chain
from src.pronunciation.dictionary import parse_ssml
from src.tts.models import VoiceListResponse, VoiceObject
from src.tts.pipeline import encode_audio, encode_audio_streaming, get_content_type
from src.voice_library import VoiceNotFoundError

logger = logging.getLogger("open-speech")

DEFAULT_VOICE_PRESETS = [
    {
        "name": "Will",
        "voice": "am_puck(1)+am_liam(1)+am_onyx(0.5)",
        "speed": 1.2,
        "description": "Dry wit genius blend — Puck + Liam + Onyx",
    },
    {
        "name": "Female",
        "voice": "af_jessica(1)+af_heart(1)",
        "speed": 1.2,
        "description": "Warm female blend — Jessica + Heart",
    },
    {
        "name": "British Butler",
        "voice": "bm_george",
        "speed": 0.9,
        "description": "Refined British male",
    },
]


def load_voice_presets() -> list[dict]:
    """Load voice presets from config file or defaults."""
    config_path = os.environ.get("TTS_VOICES_CONFIG")
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as file:
                data = yaml.safe_load(file)
            if isinstance(data, dict) and "presets" in data:
                return data["presets"]
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to load voice presets from %s: %s", config_path, exc)
    return DEFAULT_VOICE_PRESETS


def synthesize_array(*, text: str, model: str, voice: str, speed: float, sample_rate: int = 24000, language: str | None = None, tts_router, settings) -> np.ndarray:
    """Synthesize a TTS request into a single float32 array."""
    del sample_rate
    chunks = process_tts_chunks(
        tts_router.synthesize(text=text, model=model, voice=voice, speed=speed, lang_code=language),
        trim=settings.tts_trim_silence,
        normalize=settings.tts_normalize_output,
    )
    all_chunks = list(chunks)
    if not all_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(all_chunks).astype(np.float32, copy=False)


def tts_backend_name(*, tts_router, model_id: str) -> str:
    backend = tts_router.get_backend(model_id)
    return getattr(backend, "name", model_id)


def tts_capabilities(*, tts_router, model_id: str) -> dict:
    backend = tts_router.get_backend(model_id)
    capabilities = getattr(backend, "capabilities", {})
    return dict(capabilities)


def validate_tts_feature_support(*, tts_router, model_id: str, voice_design: str | None = None, reference_audio: bytes | str | None = None) -> str | None:
    backend_name = tts_backend_name(tts_router=tts_router, model_id=model_id)
    capabilities = tts_capabilities(tts_router=tts_router, model_id=model_id)
    if voice_design and not capabilities.get("voice_design", False):
        if backend_name == "kokoro":
            return "voice_design is not supported by the kokoro backend."
        return f"voice_design is not supported by the {backend_name} backend."

    if reference_audio is not None and not capabilities.get("voice_clone", False):
        if backend_name == "piper":
            return "Voice cloning is not supported by the piper backend."
        return f"Voice cloning is not supported by the {backend_name} backend."
    return None


def list_tts_models(*, settings, tts_router):
    """List TTS models and their load state."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    loaded = tts_router.loaded_models()
    loaded_ids = {model.model for model in loaded}
    models = [
        {
            "model": model.model,
            "backend": model.backend,
            "device": model.device,
            "status": "loaded",
            "loaded_at": model.loaded_at,
            "last_used_at": model.last_used_at,
        }
        for model in loaded
    ]
    if settings.tts_model not in loaded_ids:
        models.append({
            "model": settings.tts_model,
            "backend": "kokoro",
            "status": "not_loaded",
        })
    return {"models": models}


def list_voices(*, settings, tts_router, model: str | None = None):
    """List available TTS voices, optionally filtered by provider."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    if model:
        provider = model.split("/")[0] if "/" in model else model
        voices = tts_router.list_voices(provider)
    else:
        voices = tts_router.list_voices()

    return VoiceListResponse(
        voices=[VoiceObject(id=voice.id, name=voice.name, language=voice.language, gender=voice.gender) for voice in voices]
    )


def get_tts_capabilities_response(*, settings, tts_router, model: str | None = None):
    """Return TTS backend capabilities."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    model_id = model or settings.tts_model
    return {
        "backend": tts_backend_name(tts_router=tts_router, model_id=model_id),
        "capabilities": tts_capabilities(tts_router=tts_router, model_id=model_id),
    }


def load_tts_model(*, settings, tts_router, model_id: str):
    """Load a TTS model into memory."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    for loaded in tts_router.loaded_models():
        if loaded.model != model_id:
            try:
                tts_router.unload_model(loaded.model)
                logger.info("Auto-unloaded TTS model %s to load %s", loaded.model, model_id)
            except Exception as exc:
                logger.warning("Failed to auto-unload TTS model %s: %s", loaded.model, exc)

    try:
        tts_router.load_model(model_id)
    except Exception as exc:
        logger.exception("Failed to load TTS model %s", model_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "loaded", "model": model_id}


def unload_tts_model(*, settings, tts_router, model_id: str):
    """Unload a TTS model from memory."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    if not tts_router.is_model_loaded(model_id):
        raise HTTPException(status_code=404, detail=f"TTS model {model_id} is not loaded")
    tts_router.unload_model(model_id)
    return {"status": "unloaded", "model": model_id}


def _synthesis_input(request, pronunciation_dict) -> str:
    synth_input = request.input
    if request.input_type == "ssml":
        synth_input = parse_ssml(synth_input)
    return pronunciation_dict.apply(synth_input)


def _build_synth_call(*, request, synth_input: str, tts_router):
    has_extended = bool(request.voice_design or request.reference_audio)

    def _do_synthesize():
        if has_extended:
            backend = tts_router.get_backend(request.model)
            capabilities = tts_capabilities(tts_router=tts_router, model_id=request.model)
            kwargs: dict[str, Any] = dict(
                text=synth_input,
                voice=request.voice,
                speed=request.speed,
                lang_code=request.language,
            )
            if request.voice_design and (capabilities.get("voice_design") or capabilities.get("voice_clone")):
                kwargs["voice_design"] = request.voice_design
            if request.reference_audio and (capabilities.get("reference_audio") or capabilities.get("voice_clone")):
                try:
                    ref_bytes = base64.b64decode(request.reference_audio)
                except Exception:
                    ref_bytes = request.reference_audio.encode()
                kwargs["reference_audio"] = ref_bytes
            if request.clone_transcript and (capabilities.get("clone_transcript") or capabilities.get("voice_clone")):
                kwargs["clone_transcript"] = request.clone_transcript
            return backend.synthesize(**kwargs)
        return tts_router.synthesize(
            text=synth_input,
            model=request.model,
            voice=request.voice,
            speed=request.speed,
            lang_code=request.language,
        )

    return _do_synthesize


def _sample_rate_for_model(*, tts_router, model_id: str) -> int:
    sample_rate_for = getattr(tts_router, "sample_rate_for", None)
    if callable(sample_rate_for):
        try:
            return sample_rate_for(model_id) or 24000
        except Exception:
            return 24000
    return 24000


async def _read_upload_limited(upload: UploadFile, max_bytes: int, *, too_large_detail: str) -> bytes:
    """Read an upload without letting oversized bodies grow unbounded in memory."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=too_large_detail)
        chunks.append(chunk)
    return b"".join(chunks)


async def synthesize_speech_response(*, request, raw_request, stream: bool, cache: bool, settings, tts_router, tts_cache, pronunciation_dict, history_manager):
    """Handle an OpenAI-compatible TTS request."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    if len(request.input) > settings.tts_max_input_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long. Max: {settings.tts_max_input_length} characters",
        )

    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    feature_error = validate_tts_feature_support(
        tts_router=tts_router,
        model_id=request.model,
        voice_design=request.voice_design,
        reference_audio=request.reference_audio,
    )
    if feature_error:
        raise HTTPException(status_code=400, detail=feature_error)

    valid_formats = {"mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"}
    if request.response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {', '.join(sorted(valid_formats))}",
        )

    content_type = get_content_type(request.response_format)
    synth_input = _synthesis_input(request, pronunciation_dict)
    do_synthesize = _build_synth_call(request=request, synth_input=synth_input, tts_router=tts_router)

    if stream:
        if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
            try:
                history_manager.log_tts(
                    model=request.model,
                    voice=request.voice,
                    speed=request.speed,
                    format=request.response_format,
                    text=synth_input,
                    output_path=None,
                    output_bytes=None,
                    streamed=True,
                )
            except Exception:
                logger.exception("Failed to log streamed TTS history entry")

        async def _generate():
            loop = asyncio.get_running_loop()
            import queue
            import threading

            chunk_queue: queue.Queue = queue.Queue()
            sample_rate = _sample_rate_for_model(tts_router=tts_router, model_id=request.model)

            def _producer():
                try:
                    for chunk in encode_audio_streaming(
                        process_tts_chunks(
                            do_synthesize(),
                            trim=settings.tts_trim_silence,
                            normalize=settings.tts_normalize_output,
                        ),
                        fmt=request.response_format,
                        sample_rate=sample_rate,
                    ):
                        chunk_queue.put(chunk)
                except Exception as exc:
                    chunk_queue.put(exc)
                finally:
                    chunk_queue.put(None)

            thread = threading.Thread(target=_producer, daemon=True)
            thread.start()

            while True:
                item = await loop.run_in_executor(None, chunk_queue.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        return StreamingResponse(
            _generate(),
            media_type=content_type,
            headers={"Transfer-Encoding": "chunked"},
        )

    loop = asyncio.get_running_loop()

    if cache and settings.tts_cache_enabled and not stream:
        cached = tts_cache.get(
            text=synth_input,
            voice=request.voice,
            speed=request.speed,
            fmt=request.response_format,
            model=request.model,
        )
        if cached is not None:
            return StreamingResponse(
                iter([cached]),
                media_type=content_type,
                headers={"Content-Length": str(len(cached)), "X-Cache": "HIT"},
            )

    try:
        processed_chunks = process_tts_chunks(
            do_synthesize(),
            trim=settings.tts_trim_silence,
            normalize=settings.tts_normalize_output,
        )
        chunks_list = list(processed_chunks)
        samples = np.concatenate(chunks_list).astype(np.float32, copy=False) if chunks_list else np.zeros(0, dtype=np.float32)

        sample_rate = _sample_rate_for_model(tts_router=tts_router, model_id=request.model)

        if settings.os_effects_enabled and request.effects:
            samples = apply_chain(samples, sample_rate, request.effects)

        audio_bytes = await loop.run_in_executor(
            None,
            lambda: encode_audio(iter([samples]), fmt=request.response_format, sample_rate=sample_rate),
        )
        if cache and settings.tts_cache_enabled and not stream and not request.effects:
            await loop.run_in_executor(
                None,
                lambda: tts_cache.set(
                    text=synth_input,
                    voice=request.voice,
                    speed=request.speed,
                    fmt=request.response_format,
                    model=request.model,
                    audio=audio_bytes,
                ),
            )
    except Exception as exc:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
        try:
            history_manager.log_tts(
                model=request.model,
                voice=request.voice,
                speed=request.speed,
                format=request.response_format,
                text=synth_input,
                output_path=None,
                output_bytes=len(audio_bytes),
                streamed=False,
            )
        except Exception:
            logger.exception("Failed to log TTS history entry")

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes))},
    )


async def upload_voice_reference(*, name: str, audio: UploadFile, settings, voice_library) -> JSONResponse:
    """Upload a voice-library reference sample."""
    max_bytes = settings.os_max_upload_mb * 1024 * 1024
    audio_bytes = await _read_upload_limited(
        audio,
        max_bytes,
        too_large_detail=f"Voice file too large. Max: {settings.os_max_upload_mb}MB",
    )
    content_type = audio.content_type or "audio/wav"
    try:
        metadata = voice_library.save(name, audio_bytes, content_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return JSONResponse(metadata, status_code=201)


def list_library_voices(*, voice_library) -> JSONResponse:
    return JSONResponse(voice_library.list_voices())


def get_library_voice_metadata(*, name: str, voice_library) -> JSONResponse:
    try:
        _, metadata = voice_library.get(name)
    except VoiceNotFoundError:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return JSONResponse(metadata)


def delete_library_voice(*, name: str, voice_library) -> Response:
    try:
        voice_library.delete(name)
    except VoiceNotFoundError:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return Response(status_code=204)


async def clone_speech_response(*, input_text: str, model: str, reference_audio: UploadFile | None, voice_library_ref: str | None, voice: str, speed: float, response_format: str, transcript: str | None, language: str | None, settings, tts_router, voice_library):
    """Handle multipart voice cloning requests."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    if not input_text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    ref_bytes = None
    if voice_library_ref and reference_audio is None:
        try:
            ref_bytes, _metadata = voice_library.get(voice_library_ref)
        except VoiceNotFoundError:
            raise HTTPException(status_code=404, detail=f"Voice library entry '{voice_library_ref}' not found")

    if reference_audio:
        feature_error = validate_tts_feature_support(tts_router=tts_router, model_id=model, reference_audio=b"provided")
        if feature_error:
            raise HTTPException(status_code=400, detail=feature_error)
        max_bytes = settings.os_max_upload_mb * 1024 * 1024
        ref_bytes = await _read_upload_limited(
            reference_audio,
            max_bytes,
            too_large_detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB",
        )
        if len(ref_bytes) == 0:
            raise HTTPException(status_code=400, detail="Reference audio is empty")

    if ref_bytes is not None:
        feature_error = validate_tts_feature_support(tts_router=tts_router, model_id=model, reference_audio=b"provided")
        if feature_error:
            raise HTTPException(status_code=400, detail=feature_error)
        max_bytes = settings.os_max_upload_mb * 1024 * 1024
        if len(ref_bytes) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB")
        if len(ref_bytes) == 0:
            raise HTTPException(status_code=400, detail="Reference audio is empty")

    content_type = get_content_type(response_format)
    loop = asyncio.get_running_loop()

    try:
        def _synth():
            backend = tts_router.get_backend(model)
            synth_kwargs = dict(text=input_text, voice=voice, speed=speed, lang_code=language)
            signature = inspect.signature(backend.synthesize)
            if "reference_audio" in signature.parameters:
                synth_kwargs["reference_audio"] = ref_bytes
            if transcript and "clone_transcript" in signature.parameters:
                synth_kwargs["clone_transcript"] = transcript
            return encode_audio(
                process_tts_chunks(
                    backend.synthesize(**synth_kwargs),
                    trim=settings.tts_trim_silence,
                    normalize=settings.tts_normalize_output,
                ),
                fmt=response_format,
                sample_rate=_sample_rate_for_model(tts_router=tts_router, model_id=model),
            )

        audio_bytes = await loop.run_in_executor(None, _synth)
    except Exception as exc:
        logger.exception("Voice cloning synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes))},
    )
