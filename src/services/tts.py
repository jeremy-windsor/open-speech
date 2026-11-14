"""Text-to-speech service helpers."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextvars
import functools
import json
import logging
import math
import os
import queue
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from scipy.signal import resample_poly

from src.audio.postprocessing import StreamingEdgeTrimmer, process_tts_chunks
from src.cache.tts_cache import PCM_SAMPLE_RATE
from src.effects.chain import apply_chain, effect_parameter_error
from src.model_registry import get_known_model, model_display_name
from src.pronunciation.dictionary import parse_ssml
from src.tts.external import ExternalProviderError
from src.tts.models import VoiceListResponse, VoiceObject
from src.tts.pipeline import encode_audio, encode_audio_streaming, get_content_type
from src.tts.router import NoTTSBackendsError, speed_in_range
from src.voice_identities import VoiceIdentityManager, resolve_named_voice
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


def _usable_voice_presets(entries: list, config_path: str) -> list[dict]:
    """Keep the entries that name a preset and its voice, warning about each one skipped."""
    presets = []
    for entry in entries:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str) and entry["name"].strip()
            and isinstance(entry.get("voice"), str) and entry["voice"].strip()
        ):
            presets.append(entry)
        else:
            logger.warning("Skipping voice preset %r in %s: it needs a name and a voice", entry, config_path)
    return presets


def load_voice_presets() -> list[dict]:
    """Load voice presets from config file or defaults."""
    config_path = os.environ.get("TTS_VOICES_CONFIG")
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, encoding="utf-8") as file:
                data = yaml.safe_load(file)
            if isinstance(data, dict) and "presets" in data:
                data = data["presets"]
                if isinstance(data, dict):
                    data = [
                        {**values, "name": str(name)} if isinstance(values, dict) else values
                        for name, values in data.items()
                    ]
            if isinstance(data, list):
                # Startup imports these, so one malformed entry is skipped instead of stopping it.
                presets = _usable_voice_presets(data, config_path)
                if presets or not data:
                    return presets
                logger.warning("No usable voice presets in %s; using the defaults", config_path)
        except Exception as exc:
            logger.warning("Failed to load voice presets from %s: %s", config_path, exc)
    return DEFAULT_VOICE_PRESETS


class TTSDisabledError(RuntimeError):
    """TTS_ENABLED=false: no route may load or run a TTS model."""

    def __init__(self) -> None:
        super().__init__("TTS is disabled")


def synthesize_array(*, text: str, model: str, voice: str, speed: float, sample_rate: int = PCM_SAMPLE_RATE, language: str | None = None, voice_library_ref: str | None = None, instructions: str | None = None, tts_router, settings, voice_library=None) -> np.ndarray:
    """Synthesize a TTS request into a single float32 array (Studio and the Reader)."""
    if not getattr(settings, "tts_enabled", True):
        raise TTSDisabledError()
    backend_options: dict[str, Any] = {}
    voice, voice_library_ref = resolve_named_voice(model, voice, voice_library_ref)
    if instructions:
        backend_options["instructions"] = instructions
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
        instructions=instructions,
        language=language,
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
    audio = np.concatenate(all_chunks).astype(np.float32, copy=False)
    native_rate = _sample_rate_for_model(tts_router=tts_router, model_id=model)
    if native_rate != sample_rate:
        divisor = math.gcd(native_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, native_rate // divisor)
    return audio.astype(np.float32, copy=False)


def _provider_missing(exc: NoTTSBackendsError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"message": str(exc), "code": "provider_missing"},
    )


def provider_http_error(exc: ExternalProviderError) -> HTTPException:
    """A worker's failure, or the GPU memory guard's refusal, with its status and code."""
    detail = {"message": exc.message, "code": exc.code}
    if exc.code == "insufficient_gpu_memory" and getattr(exc, "details", None):
        detail["gpu"] = exc.details
    return HTTPException(status_code=exc.status_code, detail=detail)


def tts_backend_name(*, tts_router, model_id: str) -> str:
    backend = tts_router.get_backend(model_id)
    return getattr(backend, "name", model_id)


def tts_capabilities(*, tts_router, model_id: str) -> dict:
    get_capabilities = getattr(type(tts_router), "get_capabilities", None)
    if callable(get_capabilities):
        return dict(get_capabilities(tts_router, model_id))
    backend = tts_router.get_backend(model_id)
    return dict(getattr(backend, "capabilities", {}))


# Request fields a model may declare in capabilities["options"]; a model that doesn't ignores them.
MODEL_OPTION_FIELDS = ("steps",)


def request_option_values(request) -> dict:
    """The model options a request sets."""
    return {name: value for name in MODEL_OPTION_FIELDS if (value := getattr(request, name, None)) is not None}


def _for_model(request, *, tts_router):
    """The request at a speed its model takes and without options it doesn't declare; the cache key's options (those
    not at their default)."""
    capabilities = tts_capabilities(tts_router=tts_router, model_id=request.model)
    update: dict[str, Any] = {}
    speed = speed_in_range(capabilities, request.speed)
    if speed != request.speed:
        update["speed"] = speed
    values = request_option_values(request)
    declared = capabilities.get("options") or {}
    update.update({name: None for name in values if name not in declared})
    keyed = {
        name: value for name, value in values.items() if name in declared and value != declared[name].get("default")
    }
    return (request.model_copy(update=update) if update else request), ({"options": keyed} if keyed else {})


def declared_options(capabilities: dict, values: dict) -> dict:
    """The options among ``values`` that the model declares."""
    declared = capabilities.get("options") or {}
    return {name: value for name, value in values.items() if name in declared}


def _option_error(capabilities: dict, values: dict, backend_name: str) -> str | None:
    declared = capabilities.get("options") or {}
    for name, value in declared_options(capabilities, values).items():
        low, high = declared[name].get("min"), declared[name].get("max")
        if (low is not None and value < low) or (high is not None and value > high):
            return f"{name} must be from {low} to {high} for the {backend_name} model."
    return None


def validate_tts_text(*, tts_router, model_id: str, text: str) -> str | None:
    """Why the model would read the text as nothing ("😀" to Supertonic), else None."""
    validate_text = getattr(tts_router, "validate_text", None)
    if not callable(validate_text):
        return None
    try:
        validate_text(model_id, text)
    except ValueError as exc:
        return str(exc)
    return None


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
    language: str | None = None,
    options: dict | None = None,
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
    if options:
        option_error = _option_error(capabilities, options, backend_name)
        if option_error:
            return option_error

    if voice is not None:
        validate_voice = getattr(tts_router, "validate_voice", None)
        if callable(validate_voice):
            try:
                validate_voice(model_id, voice)
            except ValueError as exc:
                return str(exc)
    if language:
        # A hint the model has no pipeline for would otherwise fail inside synthesis.
        validate_language = getattr(tts_router, "validate_language", None)
        if callable(validate_language):
            try:
                validate_language(model_id, language)
            except ValueError as exc:
                return str(exc)
    return None


# ModelManager states of a TTS model no request can use now: its provider is not installed, or its worker is
# configured but offline or unable to run. /v1/audio/models leaves these out (the default TTS_MODEL excepted).
_UNUSABLE_TTS_STATES = frozenset({"provider_missing", "provider_unavailable"})


def _tts_model_row(
    *, model_id: str, backend: str, state: str, is_default: bool, device=None, loaded_at=None,
    last_used_at=None, description: str | None = None,
) -> dict:
    # id and name are what OpenAI-style clients read (Open WebUI lists each row's id); model, backend, device and
    # status (loaded / not_loaded) are this endpoint's own fields, and state is the /api/models state.
    row = {
        "id": model_id,
        "name": model_display_name(model_id, backend),
        "object": "model",
        "model": model_id,
        "backend": backend,
        "device": device,
        "status": "loaded" if state == "loaded" else "not_loaded",
        "state": state,
        "loaded_at": loaded_at,
        "last_used_at": last_used_at,
        "is_default": is_default,
    }
    if description:
        row["description"] = description
    return row


def list_tts_models(*, settings, tts_router, model_manager=None, loaded_only: bool = False):
    """List the TTS models this server can use, each with its load state.

    Every TTS model /api/models lists whose provider can run it (state provider_installed, downloaded or loaded),
    and TTS_MODEL always, whatever its state.
    ``loaded_only`` lists only the loaded models.
    """
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    rows: dict[str, dict] = {}
    for model in tts_router.loaded_models():
        known = get_known_model(model.model) or {}
        rows[model.model] = _tts_model_row(
            model_id=model.model, backend=model.backend, state="loaded",
            is_default=model.model == settings.tts_model, device=model.device,
            loaded_at=model.loaded_at, last_used_at=model.last_used_at, description=known.get("description"),
        )
    if loaded_only:
        return {"models": list(rows.values())}

    inventory = model_manager.list_all() if model_manager is not None else []
    for info in inventory:
        if info.type != "tts" or info.id in rows:
            continue
        state = getattr(info.state, "value", info.state)
        if state in _UNUSABLE_TTS_STATES and info.id != settings.tts_model:
            continue
        rows[info.id] = _tts_model_row(
            model_id=info.id, backend=info.provider, state=state, is_default=info.id == settings.tts_model,
            device=info.device, loaded_at=info.loaded_at, last_used_at=info.last_used_at,
            description=info.description,
        )
    if settings.tts_model not in rows:
        # A TTS_MODEL no provider takes is still listed, as "unknown", so a client sees what is configured.
        state = "provider_installed"
        try:
            backend = tts_backend_name(tts_router=tts_router, model_id=settings.tts_model)
        except (ValueError, NoTTSBackendsError):
            backend = settings.tts_model.split("/")[0]
            state = "unknown"
        known = get_known_model(settings.tts_model) or {}
        rows[settings.tts_model] = _tts_model_row(
            model_id=settings.tts_model, backend=backend, state=state, is_default=True,
            description=known.get("description"),
        )
    return {"models": list(rows.values())}


def list_voices(*, settings, tts_router, model: str | None = None):
    """List available TTS voices, optionally filtered by provider."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    model = model or settings.tts_model
    try:
        voices = tts_router.list_voices(model)
    except ExternalProviderError as exc:
        raise provider_http_error(exc)
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    listed = [
        VoiceObject(id=voice.id, name=voice.name, language=voice.language, gender=voice.gender, model=model)
        for voice in voices
    ]
    if not listed and _clones(tts_router, model):
        # A cloning model has no voices of its own: its named voices are what a client can choose.
        listed = [
            VoiceObject(id=identity["voice"], name=identity["name"], model=model, kind="named_voice")
            for identity in VoiceIdentityManager().list_all()
            if any(realization["model"] == model for realization in identity["realizations"])
        ]
    return VoiceListResponse(voices=listed)


def _clones(tts_router, model: str) -> bool:
    try:
        return bool(tts_capabilities(tts_router=tts_router, model_id=model).get("voice_clone"))
    except (ValueError, ExternalProviderError, NoTTSBackendsError):
        return False


def named_voice_catalog(*, model: str | None, tts_router, voice_library) -> list[dict]:
    """Retain all identities, marking each explicit implementation's availability."""
    identities = VoiceIdentityManager().list_all()
    for identity in identities:
        for realization in identity["realizations"]:
            realization["available"] = False
            if model and realization["model"] != model:
                realization["reason"] = "Select this realization's model"
                continue
            try:
                reference = None
                transcript = None
                if realization["reference_audio_id"]:
                    reference, metadata = voice_library.get(realization["reference_audio_id"])
                    transcript = metadata.get("transcript")
                error = validate_voice_realization(
                    tts_router=tts_router, model_id=realization["model"],
                    voice=realization["voice"], reference_audio=reference,
                    clone_transcript=transcript,
                )
                realization["available"] = error is None
                realization["reason"] = error
            except (KeyError, ValueError, OSError, NoTTSBackendsError, ExternalProviderError):
                realization["reason"] = "Provider, model, or reference is unavailable"
        matching = [r for r in identity["realizations"] if not model or r["model"] == model]
        identity["available"] = any(r["available"] for r in matching)
    return identities


def validate_voice_realization(*, tts_router, model_id, voice, reference_audio, clone_transcript):
    capabilities = tts_capabilities(tts_router=tts_router, model_id=model_id)
    if (reference_audio is None and capabilities.get("voice_clone")
            and not tts_router.list_voices(model_id)):
        return "A saved reference recording is required for this model"
    return validate_tts_feature_support(
        tts_router=tts_router, model_id=model_id, voice=voice,
        reference_audio=reference_audio, clone_transcript=clone_transcript,
    )


def get_tts_capabilities_response(*, settings, tts_router, model: str | None = None):
    """Return TTS backend capabilities."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    model_id = model or settings.tts_model
    try:
        response = {
            "backend": tts_backend_name(tts_router=tts_router, model_id=model_id),
            "model": model_id,
            "sample_rate": _sample_rate_for_model(
                tts_router=tts_router, model_id=model_id
            ),
            "capabilities": tts_capabilities(tts_router=tts_router, model_id=model_id),
            # TTS_MAX_INPUT_LENGTH, which every request is held to.
            "max_input_length": settings.tts_max_input_length,
        }
        get_limit = getattr(tts_router, "max_input_chars_for", None)
        model_limit = get_limit(model_id) if callable(get_limit) else None
        if type(model_limit) is int:
            # The model's own, usually lower, limit (CHATTERBOX_MAX_INPUT_CHARS and the like).
            response["max_input_chars"] = model_limit
        return response
    except ExternalProviderError as exc:
        raise provider_http_error(exc)
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def load_tts_model(*, settings, tts_router, model_id: str, operation_lock=None, make_room=None):
    """Load a TTS model into memory, in place of the other loaded TTS models.

    operation_lock is ModelManager.operation_lock("tts"), which every TTS model load, unload and
    eviction takes: two loads at once could each find nothing to evict and both stay loaded.
    make_room is ModelManager.make_room_for.
    """
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
        raise provider_http_error(exc)
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with operation_lock or nullcontext():
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
                    detail = f"Could not unload TTS model {loaded.model}; {model_id} was not loaded"
                    raise HTTPException(status_code=500, detail=detail) from exc

        try:
            with make_room(model_id, user=True) if make_room else nullcontext() as room:
                tts_router.load_model(model_id)
        except ExternalProviderError as exc:
            _restore_tts_models(tts_router, evicted_models)
            raise provider_http_error(exc)
        except ValueError as exc:
            _restore_tts_models(tts_router, evicted_models)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            _restore_tts_models(tts_router, evicted_models)
            logger.exception("Failed to load TTS model %s", model_id)
            raise HTTPException(status_code=500, detail=str(exc))
    response = {"status": "loaded", "model": model_id}
    if room is not None and room.notice:
        response["notice"] = room.notice
    return response


def _restore_tts_models(tts_router, model_ids: list[str]) -> None:
    """Best-effort restoration after a replacement model fails to load."""
    for previous_model in model_ids:
        try:
            tts_router.load_model(previous_model)
            logger.info("Restored TTS model %s after load failure", previous_model)
        except Exception:
            logger.exception("Failed to restore TTS model %s", previous_model)


def unload_tts_model(*, settings, tts_router, model_id: str, operation_lock=None):
    """Unload a TTS model from memory, under the TTS operation lock as load_tts_model."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    try:
        with operation_lock or nullcontext():
            if not tts_router.is_model_loaded(model_id):
                raise HTTPException(status_code=404, detail=f"TTS model {model_id} is not loaded")
            tts_router.unload_model(model_id)
    except ExternalProviderError as exc:
        raise provider_http_error(exc)
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "unloaded", "model": model_id}


def _synthesis_input(request, pronunciation_dict, *, max_input_length: int) -> str:
    """Return the text to synthesize: SSML reduced to text and pauses, then the dictionary.

    SSML pauses lengthen the text, so TTS_MAX_INPUT_LENGTH also covers what the client's
    markup expands to. The operator's dictionary is not part of the client's input, so its
    expansions are held only to the model's own limit, which is checked afterwards.
    """
    spoken = request.input
    if request.input_type == "ssml":
        spoken = parse_ssml(spoken)
        if len(spoken) > max_input_length:
            raise HTTPException(
                status_code=400,
                detail=f"Input too long. Max: {max_input_length} characters",
            )
    return pronunciation_dict.apply(spoken)


def _decode_reference_audio(reference_audio: str, *, max_upload_mb: int) -> bytes:
    """Decode base64 reference_audio, held to the upload limit of /v1/audio/speech/clone."""
    try:
        reference = base64.b64decode("".join(reference_audio.split()), validate=True)
    except ValueError:  # binascii.Error, or a character outside ASCII
        raise HTTPException(status_code=400, detail="reference_audio must be base64-encoded audio")
    if len(reference) > max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Reference audio too large. Max: {max_upload_mb}MB")
    if not reference:  # whitespace only, as /v1/audio/speech/clone rejects an empty upload
        raise HTTPException(status_code=400, detail="Reference audio is empty")
    return reference


def _without_unsupported_instructions(request, *, tts_router):
    """Drop instructions an OpenAI model name brought to a model that has none.

    OpenAI's gpt-4o-mini-tts takes instructions and its name maps to TTS_MODEL, so an OpenAI
    app keeps working when that model cannot follow them. A native model id keeps the 400.
    """
    try:
        supported = tts_capabilities(tts_router=tts_router, model_id=request.model).get("instructions", False)
    except ValueError:
        return request  # the feature check reports an unknown model
    if supported:
        return request
    logger.debug("Ignoring instructions sent with an OpenAI model name: %s does not take them", request.model)
    return request.model_copy(update={"instructions": None})


def _without_default_speed(request, *, tts_router):
    """Speak at 1.0 on a fixed-speed model when the speed is TTS_SPEED, not one the client sent.

    TTS_SPEED fills an omitted ``speed`` only where the model has speed control (pocket-tts and chatterbox have none),
    as TTS_VOICE fills ``voice`` only for TTS_MODEL. A sent speed keeps the 400.
    """
    try:
        supported = tts_capabilities(tts_router=tts_router, model_id=request.model).get("speed_control", False)
    except ValueError:
        return request  # the feature check reports an unknown model
    return request if supported else request.model_copy(update={"speed": 1.0})


def _history_meta(request) -> dict:
    """What Re-generate needs to repeat a request, beyond the logged model, voice and text."""
    return {
        "reference_voice": request.voice_library_ref,
        "instructions": request.instructions,
        "effects": request.effects,
        "input_type": request.input_type,
        "language": request.language,
        **request_option_values(request),
    }


def _build_synth_call(*, request, synth_input: str, tts_router, reference_audio: bytes | None = None):
    has_extended = bool(
        request.instructions
        or request.voice_design
        or reference_audio is not None
        or request.clone_transcript
    )

    option_values = request_option_values(request)

    def _do_synthesize():
        backend_options: dict[str, Any] = {}
        if option_values:
            capabilities = tts_capabilities(tts_router=tts_router, model_id=request.model)
            backend_options.update(declared_options(capabilities, option_values))
        if has_extended:
            capabilities = tts_capabilities(tts_router=tts_router, model_id=request.model)
            if request.instructions and capabilities.get("instructions"):
                backend_options["instructions"] = request.instructions
            if request.voice_design and (
                capabilities.get("voice_design") or capabilities.get("voice_clone")
            ):
                backend_options["voice_design"] = request.voice_design
            if reference_audio is not None and (
                capabilities.get("reference_audio") or capabilities.get("voice_clone")
            ):
                backend_options["reference_audio"] = reference_audio
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
    """Return the model's native rate, or 24 kHz when the router reports no usable one."""
    sample_rate_for = getattr(tts_router, "sample_rate_for", None)
    if callable(sample_rate_for):
        try:
            sample_rate = sample_rate_for(model_id)
            return sample_rate if isinstance(sample_rate, int) and sample_rate > 0 else PCM_SAMPLE_RATE
        except ExternalProviderError:
            raise
        except Exception:
            return PCM_SAMPLE_RATE
    return PCM_SAMPLE_RATE


# One thread per TTS provider runs the work that waits for the provider's synthesis lock, so
# requests queued behind a busy provider wait in that provider's queue. In asyncio's default
# executor they would hold every thread it has, and with them every other provider's requests
# and every to_thread call in the process (TTSRouter: one provider must not serialize others).
_provider_executors: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_provider_executors_lock = threading.Lock()


def _registered_provider(tts_router, model_id: str) -> str | None:
    """The registered provider a model ID routes to, as TTSRouter.get_backend picks it."""
    list_backends = getattr(tts_router, "list_backends", None)
    try:
        registered = list_backends() if callable(list_backends) else None
    except Exception:
        return None
    if not isinstance(registered, (list, tuple, set, frozenset)):
        return None
    if model_id in registered:
        return model_id
    prefix = model_id.split("/", 1)[0]
    return prefix if "/" in model_id and prefix in registered else None


def _provider_executor(tts_router, model_id: str) -> concurrent.futures.ThreadPoolExecutor | None:
    """The provider's own single thread, or None (the default executor) for an unknown model."""
    provider = _registered_provider(tts_router, model_id)
    if provider is None:
        return None  # get_backend fails at once, without waiting on any lock
    with _provider_executors_lock:
        try:
            executors = _provider_executors.setdefault(tts_router, {})
        except TypeError:  # a router that cannot be weakly referenced
            return None
        executor = executors.get(provider)
        if executor is None:
            executor = executors[provider] = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"tts-{provider}"
            )
        return executor


async def run_on_provider(tts_router, model_id: str, fn, /, *args, **kwargs):
    """Run blocking work that takes the model's provider lock, on that provider's thread."""
    call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(
        _provider_executor(tts_router, model_id), call
    )


def _to_pcm_rate(samples: np.ndarray, sample_rate: int, fmt: str) -> tuple[np.ndarray, int]:
    """Resample for pcm, which has no header and is always 24 kHz; other formats keep the rate."""
    if fmt != "pcm" or sample_rate == PCM_SAMPLE_RATE:
        return samples, sample_rate
    divisor = math.gcd(sample_rate, PCM_SAMPLE_RATE)
    resampled = resample_poly(samples, PCM_SAMPLE_RATE // divisor, sample_rate // divisor)
    return resampled.astype(np.float32), PCM_SAMPLE_RATE


def _concatenated(chunks) -> np.ndarray:
    chunks_list = list(chunks)
    if not chunks_list:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks_list).astype(np.float32, copy=False)


_STREAM_END = object()


class _LoopQueue:
    """An unbounded queue a worker thread fills and the event loop awaits, with no thread waiting.

    put() and put_nowait() take queue.Queue's arguments, for _streaming_synthesis_worker.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._items: asyncio.Queue = asyncio.Queue()

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        try:
            self._loop.call_soon_threadsafe(self._items.put_nowait, item)
        except RuntimeError:
            pass  # the loop has closed, so nobody reads this stream any more

    def put_nowait(self, item: object) -> None:
        self.put(item)

    def get_nowait(self) -> object:
        raise queue.Empty  # never full, so _finish_stream never needs to make room

    async def get(self) -> object:
        return await self._items.get()


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
    response_format: str,
    sample_rate: int,
    normalize: bool = False,
) -> None:
    """Own synthesis and encoder generators on one worker thread.

    Each backend chunk is encoded as soon as it arrives. Peak normalization needs the
    whole clip, so, as in Live Reader, a stream is never normalized (``normalize`` is
    ignored) and only its edges are trimmed. The router holds the provider's synthesis
    lock while this thread iterates the backend, so an HTTP stream passes an unbounded
    ``chunk_queue`` and the thread never waits on the client.
    """
    raw_chunks = None
    processed_chunks = None
    encoded_chunks = None

    def _trimmed_chunks():
        assert raw_chunks is not None
        trimmer = StreamingEdgeTrimmer(sample_rate) if trim else None
        # Batch trim_silence keeps all-silent audio, so hold the raw silence until the
        # trimmer emits voiced audio and send it if none ever comes.
        silence: list[np.ndarray] | None = []
        for chunk in raw_chunks:
            if cancel_event.is_set():
                return
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if trimmer is None:
                yield samples
                continue
            trimmed = trimmer.push(samples)
            if silence is not None:
                if not trimmed.size:
                    silence.append(samples)
                    continue
                silence = None
            yield trimmed
        if trimmer is None:
            return
        if silence is None:
            yield trimmer.finish()
        elif silence:
            yield np.concatenate(silence)

    try:
        raw_chunks = do_synthesize()
        processed_chunks = _trimmed_chunks()
        encoded_chunks = encode_audio_streaming(
            processed_chunks,
            fmt=response_format,
            sample_rate=sample_rate,
            output_sample_rate=PCM_SAMPLE_RATE if response_format == "pcm" else None,
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


async def synthesize_speech_response(*, request, raw_request, stream: bool, cache: bool, settings, tts_router, tts_cache, pronunciation_dict, history_manager, voice_library, openai_alias: bool = False, speed_from_settings: bool = False):
    """Handle an OpenAI-compatible TTS request.

    ``openai_alias`` marks a request whose model was an OpenAI model name (tts-1,
    gpt-4o-mini-tts), already mapped to TTS_MODEL. ``speed_from_settings`` marks a
    request whose ``speed`` is TTS_SPEED because the client sent none.
    """
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    sse = request.stream_format == "sse"
    stream = stream or request.stream_format is not None

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
    effect_error = effect_parameter_error(request.effects)
    if effect_error:
        raise HTTPException(status_code=400, detail=effect_error)

    if request.response_format not in VALID_TTS_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {', '.join(sorted(VALID_TTS_RESPONSE_FORMATS))}",
        )

    if request.voice.startswith("voice:"):
        if request.reference_audio or request.voice_design or request.clone_transcript:
            raise HTTPException(status_code=400, detail="Named voice inputs cannot be overridden")
        try:
            voice, reference = await asyncio.to_thread(
                resolve_named_voice, request.model, request.voice, request.voice_library_ref
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        request = request.model_copy(update={"voice": voice, "voice_library_ref": reference})

    if request.voice_library_ref and request.reference_audio:
        raise HTTPException(
            status_code=400,
            detail="Use either voice_library_ref or reference_audio, not both",
        )
    reference_bytes = None
    if request.reference_audio:
        reference_bytes = await asyncio.to_thread(
            _decode_reference_audio, request.reference_audio, max_upload_mb=settings.os_max_upload_mb
        )
    if request.voice_library_ref:
        try:
            # Off the event loop: a reference is a file read of up to OS_MAX_UPLOAD_MB.
            reference_bytes, metadata = await asyncio.to_thread(
                voice_library.get, request.voice_library_ref
            )
        except VoiceNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Voice library entry '{request.voice_library_ref}' not found",
            )
        request = request.model_copy(
            update={"clone_transcript": request.clone_transcript or metadata.get("transcript")}
        )

    try:
        if openai_alias and request.instructions:
            request = await asyncio.to_thread(
                _without_unsupported_instructions, request, tts_router=tts_router
            )
        if speed_from_settings and request.speed != 1.0:
            request = await asyncio.to_thread(_without_default_speed, request, tts_router=tts_router)
        feature_error = await asyncio.to_thread(
            validate_tts_feature_support,
            tts_router=tts_router,
            model_id=request.model,
            voice_design=request.voice_design,
            reference_audio=reference_bytes,
            clone_transcript=request.clone_transcript,
            instructions=request.instructions,
            speed=request.speed,
            voice=request.voice,
            language=request.language,
            options=request_option_values(request),
        )
    except ExternalProviderError as exc:
        raise provider_http_error(exc)
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    if feature_error:
        raise HTTPException(status_code=400, detail=feature_error)
    requested_speed = request.speed
    request, cache_options = await asyncio.to_thread(_for_model, request, tts_router=tts_router)
    speed_headers = {"X-Speed-Applied": f"{request.speed:g}"} if request.speed != requested_speed else {}

    content_type = get_content_type(request.response_format)
    # Off the event loop: SSML parsing and the dictionary take time in proportion to the input.
    synth_input = await asyncio.to_thread(
        _synthesis_input, request, pronunciation_dict,
        max_input_length=settings.tts_max_input_length,
    )
    if not synth_input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")
    try:
        input_error = await asyncio.to_thread(
            validate_tts_text, tts_router=tts_router, model_id=request.model, text=synth_input,
        ) or await asyncio.to_thread(
            validate_tts_input_limit,
            tts_router=tts_router, model_id=request.model, text=synth_input,
        )
    except ExternalProviderError as exc:
        raise provider_http_error(exc) from exc
    if input_error:
        raise HTTPException(status_code=400, detail=input_error)
    do_synthesize = _build_synth_call(
        request=request, synth_input=synth_input, tts_router=tts_router, reference_audio=reference_bytes
    )
    has_extended_request = bool(
        request.instructions
        or request.voice_design
        or reference_bytes is not None
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
            raise provider_http_error(exc)

        # Unbounded, so synthesis finishes and releases the provider however slowly the client
        # reads. The input limit bounds the clip, and the non-stream path holds it all too. The
        # loop awaits it, so a stream waiting for its provider holds no shared executor thread.
        loop = asyncio.get_running_loop()
        chunk_queue = _LoopQueue(loop)
        cancel_event = threading.Event()
        worker_done = asyncio.Event()

        def _run_worker(**kwargs) -> None:
            try:
                _streaming_synthesis_worker(**kwargs)
            finally:
                try:
                    loop.call_soon_threadsafe(worker_done.set)
                except RuntimeError:
                    pass  # the loop has closed

        async def _wait_for_worker() -> None:
            try:
                await asyncio.wait_for(worker_done.wait(), timeout=1.0)
            except TimeoutError:
                pass  # it ends on its own once it sees the cancel

        thread = threading.Thread(
            target=_run_worker,
            kwargs={
                "chunk_queue": chunk_queue,
                "cancel_event": cancel_event,
                "do_synthesize": do_synthesize,
                "trim": settings.tts_trim_silence,
                "response_format": request.response_format,
                "sample_rate": sample_rate,
            },
            daemon=True,
            name=f"tts-http-stream-{request.model}",
        )
        thread.start()
        # Wait for the first chunk before sending headers, so a request that fails before any
        # audio gets an error status instead of a 200 with an empty body.
        try:
            first_item = await chunk_queue.get()
        except BaseException:
            cancel_event.set()
            raise
        if isinstance(first_item, Exception):
            cancel_event.set()
            await _wait_for_worker()
            if isinstance(first_item, ExternalProviderError):
                raise provider_http_error(first_item)
            logger.error("TTS synthesis failed", exc_info=first_item)
            raise HTTPException(status_code=500, detail=str(first_item))

        # Logged once the first chunk is audio, so a stream that fails before any audio leaves no entry; one that
        # fails later keeps it, since audio was sent.
        if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
            try:
                # Off the event loop: a write to the studio.db connection every thread shares.
                await asyncio.to_thread(
                    history_manager.log_tts,
                    model=request.model,
                    voice=request.voice,
                    speed=request.speed,
                    format=request.response_format,
                    text=request.input,  # the client's text: Open in Speak sends it again
                    output_path=None,
                    output_bytes=None,
                    streamed=True,
                    meta=_history_meta(request),
                )
            except Exception:
                logger.exception("Failed to log streamed TTS history entry")
            except BaseException:
                cancel_event.set()  # cancelled while logging: the synthesis thread is running
                raise

        async def _generate():
            item = first_item
            try:
                while True:
                    if item is _STREAM_END:
                        if sse:
                            yield 'data: {"type":"speech.audio.done"}\n\n'
                        break
                    if isinstance(item, Exception):
                        if sse:
                            error = {"message": str(item)}
                            if isinstance(item, ExternalProviderError):
                                error["code"] = item.code
                            error_event = {"type": "error", "error": error}
                            yield f"data: {json.dumps(error_event)}\n\n"
                            break
                        raise item
                    if sse:
                        audio_event = {
                            "type": "speech.audio.delta",
                            "audio": base64.b64encode(item).decode("ascii"),
                        }
                        yield f"data: {json.dumps(audio_event)}\n\n"
                    else:
                        yield item
                    item = await chunk_queue.get()
            finally:
                cancel_event.set()
                await _wait_for_worker()

        body = _generate()
        # The worker is already running; stop it even if the client leaves before the body starts.
        weakref.finalize(body, cancel_event.set)
        return StreamingResponse(
            body,
            media_type="text/event-stream" if sse else content_type,
            headers={"Transfer-Encoding": "chunked", **speed_headers},
        )

    # Cache only ordinary synthesis. Extended requests can contain sensitive
    # reference audio and require more identity inputs than the shared cache.
    if cache_eligible:
        # Off the event loop: a read that can wait on the cache lock while another
        # request or transport writes and evicts.
        cached = await asyncio.to_thread(
            tts_cache.get,
            text=synth_input,
            voice=request.voice,
            speed=request.speed,
            fmt=request.response_format,
            model=request.model,
            language=request.language,
            trim=settings.tts_trim_silence,
            normalize=settings.tts_normalize_output,
            **cache_options,
        )
        if cached is not None:
            if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
                try:
                    await asyncio.to_thread(
                        history_manager.log_tts,
                        model=request.model,
                        voice=request.voice,
                        speed=request.speed,
                        format=request.response_format,
                        text=request.input,
                        output_path=None,
                        output_bytes=len(cached),
                        streamed=False,
                        meta=_history_meta(request),
                    )
                except Exception:
                    logger.exception("Failed to log cached TTS history entry")
            return StreamingResponse(
                iter([cached]),
                media_type=content_type,
                headers={"Content-Length": str(len(cached)), "X-Cache": "HIT", **speed_headers},
            )

    try:
        def _synthesize() -> np.ndarray:
            return _concatenated(process_tts_chunks(
                do_synthesize(),
                trim=settings.tts_trim_silence,
                normalize=settings.tts_normalize_output,
            ))

        def _encode(samples: np.ndarray) -> bytes:
            sample_rate = _sample_rate_for_model(tts_router=tts_router, model_id=request.model)
            if settings.os_effects_enabled and request.effects:
                samples = apply_chain(samples, sample_rate, request.effects)
            samples, sample_rate = _to_pcm_rate(samples, sample_rate, request.response_format)
            return encode_audio(
                iter([samples]),
                fmt=request.response_format,
                sample_rate=sample_rate,
            )

        samples = await run_on_provider(tts_router, request.model, _synthesize)
        audio_bytes = await asyncio.to_thread(_encode, samples)
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
                    trim=settings.tts_trim_silence,
                    normalize=settings.tts_normalize_output,
                    **cache_options,
                ),
            )
    except ExternalProviderError as exc:
        raise provider_http_error(exc)
    except Exception as exc:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
        try:
            await asyncio.to_thread(
                history_manager.log_tts,
                model=request.model,
                voice=request.voice,
                speed=request.speed,
                format=request.response_format,
                text=request.input,
                output_path=None,
                output_bytes=len(audio_bytes),
                streamed=False,
                meta=_history_meta(request),
            )
        except Exception:
            logger.exception("Failed to log TTS history entry")

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes)), **speed_headers},
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
        # Off the event loop: saving reads every frame of up to OS_MAX_UPLOAD_MB and writes a copy.
        metadata = await asyncio.to_thread(voice_library.save, name, audio_bytes, content_type, transcript=transcript)
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


async def clone_speech_response(*, input_text: str, model: str, reference_audio: UploadFile | None, voice_library_ref: str | None, voice: str, speed: float, response_format: str, transcript: str | None, language: str | None, settings, tts_router, voice_library, pronunciation_dict=None):
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

    # TTS_PRONUNCIATION_DICT applies server-wide, as on /v1/audio/speech.
    if pronunciation_dict is not None:
        input_text = await asyncio.to_thread(pronunciation_dict.apply, input_text)
        if not input_text.strip():
            raise HTTPException(status_code=400, detail="Input text is empty")

    ref_bytes = None
    if voice_library_ref and reference_audio is None:
        try:
            ref_bytes, metadata = await asyncio.to_thread(voice_library.get, voice_library_ref)
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
            language=language,
        )
    except ExternalProviderError as exc:
        raise provider_http_error(exc)
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    if initial_feature_error:
        raise HTTPException(status_code=400, detail=initial_feature_error)

    try:
        input_error = await asyncio.to_thread(
            validate_tts_input_limit,
            tts_router=tts_router, model_id=model, text=input_text,
        )
    except ExternalProviderError as exc:
        raise provider_http_error(exc) from exc
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

    try:
        def _synth() -> np.ndarray:
            synth_kwargs = dict(text=input_text, voice=voice, speed=speed, lang_code=language)
            if ref_bytes is not None:
                synth_kwargs["reference_audio"] = ref_bytes
            if transcript:
                synth_kwargs["clone_transcript"] = transcript
            return _concatenated(process_tts_chunks(
                tts_router.synthesize(model=model, **synth_kwargs),
                trim=settings.tts_trim_silence,
                normalize=settings.tts_normalize_output,
            ))

        def _encode(samples: np.ndarray) -> bytes:
            # pcm is 24 kHz here too, as on /v1/audio/speech (Piper speaks at 22.05 or 16 kHz).
            samples, sample_rate = _to_pcm_rate(
                samples, _sample_rate_for_model(tts_router=tts_router, model_id=model), response_format
            )
            return encode_audio(iter([samples]), fmt=response_format, sample_rate=sample_rate)

        samples = await run_on_provider(tts_router, model, _synth)
        audio_bytes = await asyncio.to_thread(_encode, samples)
    except ExternalProviderError as exc:
        raise provider_http_error(exc)
    except Exception as exc:
        logger.exception("Voice cloning synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes))},
    )
