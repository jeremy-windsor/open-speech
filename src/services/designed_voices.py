"""Designed voices: named voices made from a description, each owning a sample its versions clone.

The sample is a voice-library entry named designed_<identity id> whose metadata carries a ``designed`` block
(identity_id, description, model, seed, options, created_at). GET /api/voices/library leaves it out; a version's
reference_audio_id still finds it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from pydantic import TypeAdapter, ValidationError

from src.services.tts import (
    _provider_missing,
    _read_upload_limited,
    declared_options,
    provider_http_error,
    tts_capabilities,
    validate_tts_feature_support,
    validate_voice_realization,
)
from src.services.voice_design import MAX_DESCRIPTION_CHARS, MAX_SAMPLE_SECONDS
from src.storage import transaction
from src.tts.external import ExternalProviderError
from src.tts.models import ModelOptionValues
from src.tts.router import NoTTSBackendsError
from src.voice_identities import VoiceIdentityManager
from src.voice_library import MAX_TRANSCRIPT_CHARS, VoiceNotFoundError, _wav_info, designed_sample_name

logger = logging.getLogger(__name__)

MAX_NAME_CHARS = 200
MAX_OPTIONS = 8
# Model states whose weights are on this server: the cloning models a new designed voice speaks with by default.
_INSTALLED_STATES = frozenset({"downloaded", "loaded"})
_options_adapter = TypeAdapter(ModelOptionValues)


def design_block(identity_id: str, voice_library) -> dict | None:
    """A named voice's ``design``: what it was designed from and its sample, or None for a voice not designed."""
    try:
        metadata = voice_library.metadata(designed_sample_name(identity_id))
    except (VoiceNotFoundError, OSError, ValueError):
        return None
    designed = metadata.get("designed")
    if not isinstance(designed, dict) or designed.get("identity_id") != identity_id:
        return None
    return {
        "description": designed.get("description"),
        "model": designed.get("model"),
        "seed": designed.get("seed"),
        "options": designed.get("options") or {},
        "sample": metadata.get("name"),
        "transcript": metadata.get("transcript"),
        "duration_s": metadata.get("duration_s"),
    }


def with_design(identity: dict, voice_library) -> dict:
    identity["design"] = design_block(identity["id"], voice_library)
    return identity


def delete_named_voice(identity_id: str, voice_library) -> None:
    """Delete a named voice, and its designed sample when it has one."""
    designed = design_block(identity_id, voice_library) is not None
    VoiceIdentityManager().delete(identity_id)
    if designed:
        try:
            voice_library.delete(designed_sample_name(identity_id), designed=True)
        except (VoiceNotFoundError, OSError):
            logger.warning("Could not remove the designed sample of named voice %s", identity_id)


def _text(value: str | None, field: str, limit: int) -> str:
    text = (value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field} is required")
    if len(text) > limit:
        raise HTTPException(status_code=422, detail=f"{field} is too long. Max: {limit} characters")
    return text


def _options(raw: str | None) -> dict:
    if raw is None or not raw.strip():
        return {}
    try:
        options = _options_adapter.validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="options must be a JSON object of option names and numbers") from exc
    if len(options) > MAX_OPTIONS:
        raise HTTPException(status_code=422, detail=f"options can name at most {MAX_OPTIONS} options")
    return options


def _speak_with(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    models: list[str] = []
    for value in values:
        for model in value.split(","):
            model = model.strip()
            if model and model not in models:
                models.append(model)
    return models


async def _sample(audio: UploadFile, settings) -> bytes:
    data = await _read_upload_limited(
        audio,
        settings.os_max_upload_mb * 1024 * 1024,
        too_large_detail=f"Voice file too large. Max: {settings.os_max_upload_mb}MB",
    )
    info = await asyncio.to_thread(_wav_info, data) if data else None
    if info is None:
        raise HTTPException(status_code=422, detail="The sample must be a WAV file with at least one audio frame")
    if info.duration_seconds > MAX_SAMPLE_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=f"A voice sample can be at most {MAX_SAMPLE_SECONDS:g} s (this one is "
            f"{info.duration_seconds:.1f} s)",
        )
    return data


def _default_speak_with(*, model: str, tts_router, model_manager) -> list[str]:
    """The design model when it clones, else the installed cloning models of its provider."""
    if tts_capabilities(tts_router=tts_router, model_id=model).get("voice_clone"):
        return [model]
    if model_manager is None:
        return []
    provider = model.split("/", 1)[0]
    models = []
    for info in model_manager.list_all():
        state = getattr(info.state, "value", info.state)
        if info.type != "tts" or info.id == model or info.provider != provider or state not in _INSTALLED_STATES:
            continue
        try:
            clones = tts_capabilities(tts_router=tts_router, model_id=info.id).get("voice_clone")
        except (ValueError, ExternalProviderError, NoTTSBackendsError):
            continue
        if clones:
            models.append(info.id)
    return sorted(models)


def _check_design(*, model: str, description: str, options: dict, tts_router) -> dict:
    """The options the design model declares; 400 for a model that can't design a voice or an option out of range."""
    error = validate_tts_feature_support(
        tts_router=tts_router, model_id=model, voice_design=description, options=options,
    )
    if error:
        raise HTTPException(status_code=400, detail=error)
    return declared_options(tts_capabilities(tts_router=tts_router, model_id=model), options)


def _check_versions(models: list[str], *, sample: str, audio: bytes, transcript: str, tts_router) -> None:
    for model in models:
        try:
            error = validate_voice_realization(
                tts_router=tts_router, model_id=model, voice=sample, reference_audio=audio,
                clone_transcript=transcript,
            )
        except ValueError as exc:
            error = str(exc)
        if error:
            raise HTTPException(status_code=400, detail=f"{model}: {error}")


def _designed(*, identity_id: str, model: str, description: str, seed: int | None, options: dict) -> dict:
    return {
        "identity_id": identity_id,
        "description": description,
        "model": model,
        "seed": seed,
        "options": options,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _create(*, name, model, description, transcript, audio, seed, options, speak_with, tts_router, voice_library,
            model_manager) -> dict:
    options = _check_design(model=model, description=description, options=options, tts_router=tts_router)
    if speak_with is None:
        speak_with = _default_speak_with(model=model, tts_router=tts_router, model_manager=model_manager)
    identity_id = str(uuid4())
    sample = designed_sample_name(identity_id)
    _check_versions(speak_with, sample=sample, audio=audio, transcript=transcript, tts_router=tts_router)
    manager = VoiceIdentityManager()
    storing = False
    try:
        # The named voice, its versions and its sample together: any failure leaves none of them.
        with transaction():
            try:
                manager.create(name, identity_id)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            for version in speak_with:
                manager.add_realization(identity_id, model=version, voice=sample, reference_audio_id=sample)
            storing = True
            try:
                voice_library.save(
                    sample, audio, "audio/wav", transcript=transcript, max_seconds=MAX_SAMPLE_SECONDS,
                    designed=_designed(identity_id=identity_id, model=model, description=description, seed=seed,
                                       options=options),
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BaseException:
        if storing:
            with contextlib.suppress(VoiceNotFoundError, OSError):
                voice_library.delete(sample, designed=True)
        raise
    return with_design(manager.get(identity_id), voice_library)


def _replace(*, identity_id, model, description, transcript, audio, seed, options, tts_router, voice_library) -> dict:
    manager = VoiceIdentityManager()
    manager.get(identity_id)
    if design_block(identity_id, voice_library) is None:
        raise HTTPException(status_code=404, detail="This named voice is not a designed voice")
    options = _check_design(model=model, description=description, options=options, tts_router=tts_router)
    with transaction():
        manager.get(identity_id)
        try:
            voice_library.save(
                designed_sample_name(identity_id), audio, "audio/wav", transcript=transcript,
                max_seconds=MAX_SAMPLE_SECONDS,
                designed=_designed(identity_id=identity_id, model=model, description=description, seed=seed,
                                   options=options),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return with_design(manager.get(identity_id), voice_library)


async def _run(fn, **kwargs) -> dict:
    try:
        return await asyncio.to_thread(fn, **kwargs)
    except ExternalProviderError as exc:
        raise provider_http_error(exc) from exc
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc


async def create_designed_voice(*, name: str, model: str, description: str, transcript: str, audio: UploadFile,
                                seed: int | None, options: str | None, speak_with: list[str] | None, settings,
                                tts_router, voice_library, model_manager=None) -> dict:
    """POST /api/voices/designed: the named voice, its sample and a version for each ``speak_with`` model."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    fields = dict(
        name=_text(name, "name", MAX_NAME_CHARS),
        model=_text(model, "model", MAX_NAME_CHARS),
        description=_text(description, "description", MAX_DESCRIPTION_CHARS),
        transcript=_text(transcript, "transcript", MAX_TRANSCRIPT_CHARS),
        options=_options(options),
        seed=seed,
        speak_with=_speak_with(speak_with),
    )
    fields["audio"] = await _sample(audio, settings)
    return await _run(_create, **fields, tts_router=tts_router, voice_library=voice_library,
                      model_manager=model_manager)


async def replace_designed_voice(*, identity_id: str, model: str, description: str, transcript: str,
                                 audio: UploadFile, seed: int | None, options: str | None, settings, tts_router,
                                 voice_library) -> dict:
    """PUT /api/voices/designed/{identity_id}: a new sample and design for the same voice; versions keep it."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    fields = dict(
        model=_text(model, "model", MAX_NAME_CHARS),
        description=_text(description, "description", MAX_DESCRIPTION_CHARS),
        transcript=_text(transcript, "transcript", MAX_TRANSCRIPT_CHARS),
        options=_options(options),
        seed=seed,
    )
    fields["audio"] = await _sample(audio, settings)
    return await _run(_replace, identity_id=identity_id, **fields, tts_router=tts_router, voice_library=voice_library)
