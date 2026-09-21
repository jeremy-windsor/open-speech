"""Text-to-speech service helpers."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import queue
import threading
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
from src.tts.external import ExternalProviderError
from src.voice_library import VoiceNotFoundError

logger = logging.getLogger("open-speech")

VALID_TTS_RESPONSE_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"}

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


def synthesize_array(*, text: str, model: str, voice: str, speed: float, sample_rate: int = 24000, language: str | None = None, voice_library_ref: str | None = None, tts_router, settings, voice_library=None) -> np.ndarray:
    """Synthesize a TTS request into a single float32 array."""
    del sample_rate
    backend_options: dict[str, Any] = {}
    if voice_library_ref:
        if voice_library is None:
            raise RuntimeError("Voice library is unavailable")
        reference_audio, metadata = voice_library.get(voice_library_ref)
        backend_options["reference_audio"] = reference_audio
        if metadata.get("transcript"):
            backend_options["clone_transcript"] = metadata["transcript"]
    feature_error = validate_tts_feature_support(
        tts_router=tts_router,
        model_id=model,
        reference_audio=backend_options.get("reference_audio"),
        clone_transcript=backend_options.get("clone_transcript"),
        speed=speed,
        voice=voice,
    )
    if feature_error:
        raise ValueError(feature_error)
    input_error = validate_tts_input_limit(tts_router=tts_router, model_id=model, text=text)
    if input_error:
        raise ValueError(input_error)
    chunks = process_tts_chunks(
        tts_router.synthesize(
            text=text,
            model=model,
            voice=voice,
            speed=speed,
            lang_code=language,
            **backend_options,
        ),
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
    get_capabilities = getattr(type(tts_router), "get_capabilities", None)
    if callable(get_capabilities):
        return dict(get_capabilities(tts_router, model_id))
    backend = tts_router.get_backend(model_id)
    return dict(getattr(backend, "capabilities", {}))


def validate_tts_input_limit(*, tts_router, model_id: str, text: str) -> str | None:
    get_limit = getattr(tts_router, "max_input_chars_for", None)
    if not callable(get_limit):
        return None
    limit = get_limit(model_id)
    if type(limit) is int and len(text) > limit:
        return f"Input too long for {model_id}. Max: {limit} characters"
    return None


def validate_tts_feature_support(
    *,
    tts_router,
    model_id: str,
    voice_design: str | None = None,
    reference_audio: bytes | str | None = None,
    clone_transcript: str | None = None,
    instructions: str | None = None,
    speed: float = 1.0,
    voice: str | None = None,
    live_reader: bool = False,
) -> str | None:
    try:
        backend = tts_router.get_backend(model_id)
    except ValueError as exc:
        return str(exc)

    backend_name = getattr(backend, "name", model_id)
    capabilities = tts_capabilities(tts_router=tts_router, model_id=model_id)
    if live_reader and capabilities.get("live_reader") is False:
        return f"Live Reader is not supported by the {backend_name} model."
    if voice_design and not capabilities.get("voice_design", False):
        if backend_name == "kokoro":
            return "voice_design is not supported by the kokoro backend."
        return f"voice_design is not supported by the {backend_name} backend."

    if reference_audio is not None and not capabilities.get("voice_clone", False):
        if backend_name == "piper":
            return "Voice cloning is not supported by the piper backend."
        return f"Voice cloning is not supported by the {backend_name} backend."
    if clone_transcript and not (
        capabilities.get("clone_transcript", False)
        or capabilities.get("voice_clone", False)
    ):
        return f"clone_transcript is not supported by the {backend_name} backend."
    if (
        reference_audio is not None
        and capabilities.get("clone_transcript_required", False)
        and not clone_transcript
    ):
        return f"clone_transcript is required by the {backend_name} model."
    if instructions and not capabilities.get("instructions", False):
        return f"instructions are not supported by the {backend_name} backend."
    if speed != 1.0 and not capabilities.get("speed_control", False):
        return f"Speed control is not supported by the {backend_name} backend."

    if voice is not None:
        validate_voice = getattr(tts_router, "validate_voice", None)
        if callable(validate_voice):
            try:
                validate_voice(model_id, voice)
            except ValueError as exc:
                return str(exc)
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

    try:
        voices = tts_router.list_voices(model or settings.tts_model)
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return VoiceListResponse(
        voices=[VoiceObject(id=voice.id, name=voice.name, language=voice.language, gender=voice.gender) for voice in voices]
    )


def get_tts_capabilities_response(*, settings, tts_router, model: str | None = None):
    """Return TTS backend capabilities."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    model_id = model or settings.tts_model
    try:
        return {
            "backend": tts_backend_name(tts_router=tts_router, model_id=model_id),
            "model": model_id,
            "sample_rate": _sample_rate_for_model(
                tts_router=tts_router, model_id=model_id
            ),
            "capabilities": tts_capabilities(tts_router=tts_router, model_id=model_id),
        }
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def load_tts_model(*, settings, tts_router, model_id: str):
    """Load a TTS model into memory."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    try:
        # Resolve the target before evicting a working model. An invalid model
        # request must not disrupt the currently loaded provider.
        tts_router.get_backend(model_id)
        get_capabilities = getattr(tts_router, "get_capabilities", None)
        if callable(get_capabilities):
            get_capabilities(model_id)
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    evicted_models: list[str] = []
    for loaded in tts_router.loaded_models():
        if loaded.model != model_id:
            try:
                tts_router.unload_model(loaded.model)
                evicted_models.append(loaded.model)
                logger.info("Auto-unloaded TTS model %s to load %s", loaded.model, model_id)
            except Exception as exc:
                logger.warning("Failed to auto-unload TTS model %s: %s", loaded.model, exc)
                _restore_tts_models(tts_router, evicted_models)
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not unload TTS model {loaded.model}; {model_id} was not loaded",
                ) from exc

    try:
        tts_router.load_model(model_id)
    except ExternalProviderError as exc:
        _restore_tts_models(tts_router, evicted_models)
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except ValueError as exc:
        _restore_tts_models(tts_router, evicted_models)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _restore_tts_models(tts_router, evicted_models)
        logger.exception("Failed to load TTS model %s", model_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "loaded", "model": model_id}


def _restore_tts_models(tts_router, model_ids: list[str]) -> None:
    """Best-effort restoration after a replacement model fails to load."""
    for previous_model in model_ids:
        try:
            tts_router.load_model(previous_model)
            logger.info("Restored TTS model %s after load failure", previous_model)
        except Exception:
            logger.exception("Failed to restore TTS model %s", previous_model)


def unload_tts_model(*, settings, tts_router, model_id: str):
    """Unload a TTS model from memory."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    try:
        if not tts_router.is_model_loaded(model_id):
            raise HTTPException(status_code=404, detail=f"TTS model {model_id} is not loaded")
        tts_router.unload_model(model_id)
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "unloaded", "model": model_id}


def _synthesis_input(request, pronunciation_dict) -> str:
    synth_input = request.input
    if request.input_type == "ssml":
        synth_input = parse_ssml(synth_input)
    return pronunciation_dict.apply(synth_input)


def _build_synth_call(*, request, synth_input: str, tts_router):
    has_extended = bool(
        request.instructions
        or request.voice_design
        or request.reference_audio
        or request.clone_transcript
    )

    def _do_synthesize():
        backend_options: dict[str, Any] = {}
        if has_extended:
            capabilities = tts_capabilities(tts_router=tts_router, model_id=request.model)
            if request.instructions and capabilities.get("instructions"):
                backend_options["instructions"] = request.instructions
            if request.voice_design and (
                capabilities.get("voice_design") or capabilities.get("voice_clone")
            ):
                backend_options["voice_design"] = request.voice_design
            if request.reference_audio and (
                capabilities.get("reference_audio") or capabilities.get("voice_clone")
            ):
                try:
                    ref_bytes = base64.b64decode(request.reference_audio)
                except Exception:
                    ref_bytes = request.reference_audio.encode()
                backend_options["reference_audio"] = ref_bytes
            if request.clone_transcript and (
                capabilities.get("clone_transcript") or capabilities.get("voice_clone")
            ):
                backend_options["clone_transcript"] = request.clone_transcript
        return tts_router.synthesize(
            text=synth_input,
            model=request.model,
            voice=request.voice,
            speed=request.speed,
            lang_code=request.language,
            **backend_options,
        )

    return _do_synthesize


def _sample_rate_for_model(*, tts_router, model_id: str) -> int:
    sample_rate_for = getattr(tts_router, "sample_rate_for", None)
    if callable(sample_rate_for):
        try:
            return sample_rate_for(model_id) or 24000
        except ExternalProviderError:
            raise
        except Exception:
            return 24000
    return 24000


_STREAM_END = object()


def _put_stream_item(
    chunk_queue: queue.Queue,
    item: object,
    cancel_event: threading.Event,
) -> bool:
    """Put a worker result without allowing a disconnected client to block it forever."""
    while not cancel_event.is_set():
        try:
            chunk_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _close_iterator(iterator: object | None) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.exception("Failed to close TTS stream iterator")


def _finish_stream(chunk_queue: queue.Queue, cancel_event: threading.Event) -> None:
    """Deliver the terminal marker, dropping buffered audio only after cancellation."""
    if not cancel_event.is_set():
        _put_stream_item(chunk_queue, _STREAM_END, cancel_event)
        return

    while True:
        try:
            chunk_queue.put_nowait(_STREAM_END)
            return
        except queue.Full:
            try:
                chunk_queue.get_nowait()
            except queue.Empty:
                continue


def _streaming_synthesis_worker(
    *,
    chunk_queue: queue.Queue,
    cancel_event: threading.Event,
    do_synthesize,
    trim: bool,
    normalize: bool,
    response_format: str,
    sample_rate: int,
) -> None:
    """Own synthesis and encoder generators on one worker thread."""
    raw_chunks = None
    processed_chunks = None
    encoded_chunks = None

    def _cancelable_chunks():
        assert raw_chunks is not None
        for chunk in raw_chunks:
            if cancel_event.is_set():
                break
            yield chunk

    try:
        raw_chunks = do_synthesize()
        processed_chunks = process_tts_chunks(
            _cancelable_chunks(),
            trim=trim,
            normalize=normalize,
        )
        encoded_chunks = encode_audio_streaming(
            processed_chunks,
            fmt=response_format,
            sample_rate=sample_rate,
        )
        for chunk in encoded_chunks:
            if cancel_event.is_set():
                break
            if not _put_stream_item(chunk_queue, chunk, cancel_event):
                break
    except Exception as exc:
        _put_stream_item(chunk_queue, exc, cancel_event)
    finally:
        _close_iterator(encoded_chunks)
        _close_iterator(processed_chunks)
        _close_iterator(raw_chunks)
        _finish_stream(chunk_queue, cancel_event)


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


async def synthesize_speech_response(*, request, raw_request, stream: bool, cache: bool, settings, tts_router, tts_cache, pronunciation_dict, history_manager, voice_library):
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

    if stream and request.effects:
        raise HTTPException(
            status_code=400,
            detail="Effects are not supported for streaming TTS",
        )

    if request.response_format not in VALID_TTS_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {', '.join(sorted(VALID_TTS_RESPONSE_FORMATS))}",
        )

    if request.voice_library_ref and request.reference_audio:
        raise HTTPException(
            status_code=400,
            detail="Use either voice_library_ref or reference_audio, not both",
        )
    if request.voice_library_ref:
        try:
            reference_bytes, metadata = voice_library.get(request.voice_library_ref)
        except VoiceNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Voice library entry '{request.voice_library_ref}' not found",
            )
        request = request.model_copy(
            update={
                "reference_audio": base64.b64encode(reference_bytes).decode("ascii"),
                "clone_transcript": request.clone_transcript or metadata.get("transcript"),
            }
        )

    try:
        feature_error = await asyncio.to_thread(
            validate_tts_feature_support,
            tts_router=tts_router,
            model_id=request.model,
            voice_design=request.voice_design,
            reference_audio=request.reference_audio,
            clone_transcript=request.clone_transcript,
            instructions=request.instructions,
            speed=request.speed,
            voice=request.voice,
        )
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if feature_error:
        raise HTTPException(status_code=400, detail=feature_error)

    content_type = get_content_type(request.response_format)
    synth_input = _synthesis_input(request, pronunciation_dict)
    try:
        input_error = await asyncio.to_thread(
            validate_tts_input_limit,
            tts_router=tts_router, model_id=request.model, text=synth_input,
        )
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if input_error:
        raise HTTPException(status_code=400, detail=input_error)
    do_synthesize = _build_synth_call(request=request, synth_input=synth_input, tts_router=tts_router)
    has_extended_request = bool(
        request.instructions
        or request.voice_design
        or request.reference_audio
        or request.clone_transcript
    )
    cache_eligible = (
        cache
        and settings.tts_cache_enabled
        and not stream
        and not request.effects
        and not has_extended_request
    )

    if stream:
        try:
            sample_rate = await asyncio.to_thread(
                _sample_rate_for_model,
                tts_router=tts_router,
                model_id=request.model,
            )
        except ExternalProviderError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc))

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
            chunk_queue: queue.Queue = queue.Queue(maxsize=8)
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=_streaming_synthesis_worker,
                kwargs={
                    "chunk_queue": chunk_queue,
                    "cancel_event": cancel_event,
                    "do_synthesize": do_synthesize,
                    "trim": settings.tts_trim_silence,
                    "normalize": settings.tts_normalize_output,
                    "response_format": request.response_format,
                    "sample_rate": sample_rate,
                },
                daemon=True,
                name=f"tts-http-stream-{request.model}",
            )
            thread.start()
            try:
                while True:
                    item = await asyncio.to_thread(chunk_queue.get)
                    if item is _STREAM_END:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                cancel_event.set()
                await asyncio.to_thread(thread.join, 1.0)

        return StreamingResponse(
            _generate(),
            media_type=content_type,
            headers={"Transfer-Encoding": "chunked"},
        )

    # Cache only ordinary synthesis. Extended requests can contain sensitive
    # reference audio and require more identity inputs than the shared cache.
    if cache_eligible:
        cached = tts_cache.get(
            text=synth_input,
            voice=request.voice,
            speed=request.speed,
            fmt=request.response_format,
            model=request.model,
            language=request.language,
        )
        if cached is not None:
            if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
                try:
                    history_manager.log_tts(
                        model=request.model,
                        voice=request.voice,
                        speed=request.speed,
                        format=request.response_format,
                        text=synth_input,
                        output_path=None,
                        output_bytes=len(cached),
                        streamed=False,
                    )
                except Exception:
                    logger.exception("Failed to log cached TTS history entry")
            return StreamingResponse(
                iter([cached]),
                media_type=content_type,
                headers={"Content-Length": str(len(cached)), "X-Cache": "HIT"},
            )

    try:
        def _synthesize_and_encode() -> bytes:
            processed_chunks = process_tts_chunks(
                do_synthesize(),
                trim=settings.tts_trim_silence,
                normalize=settings.tts_normalize_output,
            )
            chunks_list = list(processed_chunks)
            samples = (
                np.concatenate(chunks_list).astype(np.float32, copy=False)
                if chunks_list
                else np.zeros(0, dtype=np.float32)
            )
            sample_rate = _sample_rate_for_model(tts_router=tts_router, model_id=request.model)
            if settings.os_effects_enabled and request.effects:
                samples = apply_chain(samples, sample_rate, request.effects)
            return encode_audio(
                iter([samples]),
                fmt=request.response_format,
                sample_rate=sample_rate,
            )

        audio_bytes = await asyncio.to_thread(_synthesize_and_encode)
        if cache_eligible:
            await asyncio.to_thread(
                lambda: tts_cache.set(
                    text=synth_input,
                    voice=request.voice,
                    speed=request.speed,
                    fmt=request.response_format,
                    model=request.model,
                    audio=audio_bytes,
                    language=request.language,
                ),
            )
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
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


async def upload_voice_reference(*, name: str, audio: UploadFile, transcript: str | None = None, settings, voice_library) -> JSONResponse:
    """Upload a voice-library reference sample."""
    max_bytes = settings.os_max_upload_mb * 1024 * 1024
    audio_bytes = await _read_upload_limited(
        audio,
        max_bytes,
        too_large_detail=f"Voice file too large. Max: {settings.os_max_upload_mb}MB",
    )
    content_type = audio.content_type or "audio/wav"
    try:
        metadata = voice_library.save(name, audio_bytes, content_type, transcript=transcript)
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


def get_library_voice_audio(*, name: str, voice_library) -> Response:
    try:
        audio_bytes, metadata = voice_library.get(name)
    except VoiceNotFoundError:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{metadata["name"]}.wav"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def update_library_voice_transcript(
    *, name: str, transcript: str | None, voice_library
) -> JSONResponse:
    try:
        metadata = voice_library.set_transcript(name, transcript)
    except VoiceNotFoundError:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
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

    if len(input_text) > settings.tts_max_input_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long. Max: {settings.tts_max_input_length} characters",
        )

    if not 0.25 <= speed <= 4.0:
        raise HTTPException(status_code=422, detail="speed must be between 0.25 and 4.0")

    if response_format not in VALID_TTS_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {', '.join(sorted(VALID_TTS_RESPONSE_FORMATS))}",
        )

    ref_bytes = None
    if voice_library_ref and reference_audio is None:
        try:
            ref_bytes, metadata = voice_library.get(voice_library_ref)
            transcript = transcript or metadata.get("transcript")
        except VoiceNotFoundError:
            raise HTTPException(status_code=404, detail=f"Voice library entry '{voice_library_ref}' not found")

    try:
        initial_feature_error = await asyncio.to_thread(
            validate_tts_feature_support,
            tts_router=tts_router,
            model_id=model,
            reference_audio=(
                b"provided" if reference_audio is not None or ref_bytes is not None else None
            ),
            clone_transcript=transcript,
            speed=speed,
            voice=voice,
        )
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if initial_feature_error:
        raise HTTPException(status_code=400, detail=initial_feature_error)

    try:
        input_error = await asyncio.to_thread(
            validate_tts_input_limit,
            tts_router=tts_router, model_id=model, text=input_text,
        )
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if input_error:
        raise HTTPException(status_code=400, detail=input_error)

    if reference_audio:
        max_bytes = settings.os_max_upload_mb * 1024 * 1024
        ref_bytes = await _read_upload_limited(
            reference_audio,
            max_bytes,
            too_large_detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB",
        )
        if len(ref_bytes) == 0:
            raise HTTPException(status_code=400, detail="Reference audio is empty")

    if ref_bytes is not None:
        max_bytes = settings.os_max_upload_mb * 1024 * 1024
        if len(ref_bytes) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB")
        if len(ref_bytes) == 0:
            raise HTTPException(status_code=400, detail="Reference audio is empty")

    content_type = get_content_type(response_format)
    loop = asyncio.get_running_loop()

    try:
        def _synth():
            synth_kwargs = dict(text=input_text, voice=voice, speed=speed, lang_code=language)
            if ref_bytes is not None:
                synth_kwargs["reference_audio"] = ref_bytes
            if transcript:
                synth_kwargs["clone_transcript"] = transcript
            return encode_audio(
                process_tts_chunks(
                    tts_router.synthesize(model=model, **synth_kwargs),
                    trim=settings.tts_trim_silence,
                    normalize=settings.tts_normalize_output,
                ),
                fmt=response_format,
                sample_rate=_sample_rate_for_model(tts_router=tts_router, model_id=model),
            )

        audio_bytes = await loop.run_in_executor(None, _synth)
    except ExternalProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception as exc:
        logger.exception("Voice cloning synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes))},
    )
