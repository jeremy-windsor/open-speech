"""Studio, voice-library, and persistence routes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from src.assistant.presets import has_lone_surrogate
from src.composer import MAX_RENDER_SECONDS
from src.conversation import ConversationChangedError, ConversationNotFoundError
from src.effects.chain import effect_parameter_error
from src.profiles import ProfileNotFoundError
from src.services import designed_voices
from src.services import tts as tts_service
from src.tts.external import ExternalProviderError
from src.tts.models import ModelOptionValues
from src.tts.pipeline import get_content_type
from src.tts.router import NoTTSBackendsError
from src.voice_identities import NamedVoiceNotFoundError, SampleOwnedError, VoiceIdentityManager, check_reference
from src.voice_library import MAX_TRANSCRIPT_CHARS, VoiceNotFoundError

# Renders wait for synthesis locks (a conversation speaks through each turn's provider) or for
# the composer's one-render-at-a-time lock. Their own pools hold a queue of them, so it never
# fills asyncio's default executor, which every other request and to_thread call shares.
_conversation_renders = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="conversation-render"
)
_composer_renders = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="composer-render")


async def _run_render(executor: concurrent.futures.Executor, fn: Callable, /, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(executor, functools.partial(fn, **kwargs))


# What SQLite binds as an INTEGER: a page past it was an OverflowError 500.
SqliteInt = Annotated[int, Query(ge=-(2**63), le=2**63 - 1)]


def _refuse_lone_surrogate(value, info: ValidationInfo):
    """A text field holding half of a UTF-16 pair (the JSON escape "\\ud83d") can't be stored in studio.db,
    whose UTF-8 encode would raise a 500, or a 409 for a preset: refuse it with 422 naming the field."""
    if isinstance(value, str) and has_lone_surrogate(value):
        raise ValueError(f"{info.field_name} has a lone surrogate (half of a UTF-16 pair, such as half an emoji), "
                         "which can't be stored; send whole characters")
    return value


class ProfilePayload(BaseModel):
    name: str
    backend: str
    model: str | None = None
    voice: str
    speed: float = Field(default=1.0, ge=0.25, le=4.0, allow_inf_nan=False)   # as /v1/audio/speech
    format: str = "mp3"
    blend: str | None = None
    reference_audio_id: str | None = None
    voice_identity_id: str | None = None
    instructions: str | None = None
    effects: list[dict | str] = Field(default_factory=list, max_length=16)   # one 422 error, not one per item
    # A model's own options, as /v1/audio/speech takes them ({"steps": 8}, {"cfg_scale": 2.5}).
    options: ModelOptionValues = Field(default_factory=dict, max_length=8)

    _text_is_storable = field_validator("name", "backend", "model", "voice", "format", "blend",
                                        "reference_audio_id", "voice_identity_id", "instructions")(
        _refuse_lone_surrogate)


class VoiceTranscriptPayload(BaseModel):
    transcript: str | None = Field(default=None, max_length=MAX_TRANSCRIPT_CHARS)


class VoiceIdentityPayload(BaseModel):
    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Voice name is required")
        return value


class VoiceRealizationPayload(BaseModel):
    model: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    reference_audio_id: str | None = None


class ProfileListResponse(BaseModel):
    profiles: list[dict]
    default_profile_id: str | None = None


class HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["tts", "stt"]
    created_at: str
    model: str | None = None
    voice: str | None = None
    speed: float | None = None
    format: str | None = None
    text_preview: str | None = None
    full_text: str | None = None
    input_filename: str | None = None
    output_path: str | None = None
    output_bytes: int | None = None
    streamed: bool = False
    meta: dict = Field(default_factory=dict)


class HistoryListResponse(BaseModel):
    items: list[HistoryEntry]
    total: int
    limit: int
    offset: int


class ConversationTurnPayload(BaseModel):
    speaker: str
    text: str
    profile_id: str | None = None
    effects: list[dict] | None = Field(default=None, max_length=16)

    _text_is_storable = field_validator("speaker", "text", "profile_id")(_refuse_lone_surrogate)


class ConversationCreatePayload(BaseModel):
    name: str
    # Bounded: Pydantic makes one error per invalid item.
    turns: list[ConversationTurnPayload] = Field(default_factory=list, max_length=500)

    _text_is_storable = field_validator("name")(_refuse_lone_surrogate)


# Render buffers are sized by the sample rate, so it is bounded like every other audio input.
MIN_RENDER_SAMPLE_RATE = 8000
MAX_RENDER_SAMPLE_RATE = 48000
MAX_COMPOSER_TRACKS = 64


class ConversationRenderPayload(BaseModel):
    format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm", "m4a"] = "wav"
    sample_rate: int = Field(default=24000, ge=MIN_RENDER_SAMPLE_RATE, le=MAX_RENDER_SAMPLE_RATE)
    save_turn_audio: bool = True


class ComposerTrack(BaseModel):
    source_path: str
    offset_s: float = Field(default=0.0, allow_inf_nan=False, le=MAX_RENDER_SECONDS)
    volume: float = Field(default=1.0, allow_inf_nan=False)
    muted: bool = False
    solo: bool = False
    effects: list[dict] | None = Field(default=None, max_length=16)


class ComposerRenderRequest(BaseModel):
    name: str | None = None
    format: Literal["wav", "mp3"] = "wav"
    sample_rate: int = Field(default=24000, ge=MIN_RENDER_SAMPLE_RATE, le=MAX_RENDER_SAMPLE_RATE)
    tracks: list[ComposerTrack] = Field(max_length=MAX_COMPOSER_TRACKS)

    _text_is_storable = field_validator("name")(_refuse_lone_surrogate)


def _not_found_detail(exc: Exception) -> str:
    # str(KeyError("x")) is "'x'", quotes included.
    return exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)


def _reject_invalid_effects(effects: list[dict | str] | None, label: str | None = None) -> None:
    """Refuse effect parameters apply_chain cannot use before they are stored or rendered."""
    # Presets may name an effect by its type alone.
    error = effect_parameter_error([{"type": fx} if isinstance(fx, str) else fx for fx in effects or []])
    if error:
        raise HTTPException(status_code=400, detail=f"{label}: {error}" if label else error)


def create_router(*, get_settings: Callable, get_voice_library: Callable, get_profile_manager: Callable, get_history_manager: Callable, get_conversation_manager: Callable, get_composer_manager: Callable, get_tts_router: Callable = lambda: None, get_model_manager: Callable = lambda: None) -> APIRouter:
    router = APIRouter()

    def reject_invalid_turn(turn: ConversationTurnPayload, label: str) -> None:
        max_length = get_settings().tts_max_input_length
        if len(turn.text) > max_length:
            raise HTTPException(status_code=400, detail=f"{label} is too long. Max: {max_length} characters")
        _reject_invalid_effects(turn.effects, label)

    @router.post("/api/voices/library", status_code=201)
    async def upload_voice(
        name: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
        transcript: Annotated[str | None, Form()] = None,
    ):
        return await tts_service.upload_voice_reference(
            name=name,
            audio=audio,
            transcript=transcript,
            settings=get_settings(),
            voice_library=get_voice_library(),
        )

    # The library handlers below are plain functions, which FastAPI runs in its threadpool: they
    # read reference files, and wait for the library's lock while a TTS request or a Live
    # session reads one, which must not hold up the event loop.
    @router.get("/api/voices/library")
    def list_library_voices():
        return tts_service.list_library_voices(voice_library=get_voice_library())

    @router.get("/api/voices/library-config")
    async def get_library_config():
        return {"max_seconds": get_settings().voice_library_max_seconds}

    @router.get("/api/voices/library/{name}")
    def get_library_voice_meta(name: str):
        return tts_service.get_library_voice_metadata(name=name, voice_library=get_voice_library())

    @router.get("/api/voices/library/{name}/audio")
    def get_library_voice_audio(name: str):
        return tts_service.get_library_voice_audio(name=name, voice_library=get_voice_library())

    @router.patch("/api/voices/library/{name}")
    def update_library_voice_transcript(name: str, payload: VoiceTranscriptPayload):
        if "transcript" not in payload.model_fields_set:
            return tts_service.get_library_voice_metadata(
                name=name,
                voice_library=get_voice_library(),
            )
        return tts_service.update_library_voice_transcript(
            name=name,
            transcript=payload.transcript,
            voice_library=get_voice_library(),
        )

    @router.delete("/api/voices/library/{name}", status_code=204)
    def delete_library_voice(name: str):
        return tts_service.delete_library_voice(name=name, voice_library=get_voice_library())

    # Handlers that read or write studio.db are plain functions, which FastAPI runs in its
    # threadpool: every statement on the shared connection waits for any other thread's
    # transaction, and that wait must not hold up the event loop.
    @router.get("/api/voice-presets")
    def get_voice_presets():
        manager = get_profile_manager()
        return {"presets": manager.list_all()}

    def with_design(identity: dict) -> dict:
        return designed_voices.with_design(identity, get_voice_library())

    @router.post("/api/voices/identities", status_code=201)
    def create_voice_identity(payload: VoiceIdentityPayload):
        try:
            return with_design(VoiceIdentityManager().create(payload.name))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/voices/identities")
    async def list_voice_identities(model: str | None = None, check_availability: bool = True):
        def listed():
            if not check_availability:
                identities = VoiceIdentityManager().list_all()
            else:
                identities = tts_service.named_voice_catalog(
                    model=model, tts_router=get_tts_router(), voice_library=get_voice_library(),
                )
            return [with_design(identity) for identity in identities]

        return {"voices": await asyncio.to_thread(listed)}

    @router.patch("/api/voices/identities/{identity_id}")
    def rename_voice_identity(identity_id: str, payload: VoiceIdentityPayload):
        try:
            return with_design(VoiceIdentityManager().rename(identity_id, payload.name))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.delete("/api/voices/identities/{identity_id}")
    def delete_voice_identity(identity_id: str):
        # moved_to: the recording a designed voice's sample became because a preset or another voice still uses it.
        try:
            return {"moved_to": designed_voices.delete_named_voice(identity_id, get_voice_library())}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc

    @router.post("/api/voices/designed", status_code=201)
    async def create_designed_voice(
        name: Annotated[str, Form()],
        model: Annotated[str, Form()],
        description: Annotated[str, Form()],
        transcript: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
        seed: Annotated[int | None, Form(ge=0, le=2**31 - 1)] = None,
        options: Annotated[str | None, Form()] = None,
        speak_with: Annotated[list[str] | None, Form()] = None,
    ):
        try:
            return await designed_voices.create_designed_voice(
                name=name, model=model, description=description, transcript=transcript, audio=audio, seed=seed,
                options=options, speak_with=speak_with, settings=get_settings(), tts_router=get_tts_router(),
                voice_library=get_voice_library(), model_manager=get_model_manager(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.put("/api/voices/designed/{identity_id}")
    async def replace_designed_voice(
        identity_id: str,
        model: Annotated[str, Form()],
        description: Annotated[str, Form()],
        transcript: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
        seed: Annotated[int | None, Form(ge=0, le=2**31 - 1)] = None,
        options: Annotated[str | None, Form()] = None,
    ):
        try:
            return await designed_voices.replace_designed_voice(
                identity_id=identity_id, model=model, description=description, transcript=transcript, audio=audio,
                seed=seed, options=options, settings=get_settings(), tts_router=get_tts_router(),
                voice_library=get_voice_library(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc

    @router.post("/api/voices/identities/{identity_id}/realizations", status_code=201)
    async def create_voice_realization(identity_id: str, payload: VoiceRealizationPayload):
        return await save_voice_realization(identity_id, payload, replace=False)

    @router.put("/api/voices/identities/{identity_id}/realizations/{model:path}")
    async def update_voice_realization(identity_id: str, model: str, payload: VoiceRealizationPayload):
        if payload.model != model:
            raise HTTPException(status_code=400, detail="Model in the path and payload must match")
        return await save_voice_realization(identity_id, payload, replace=True)

    async def save_voice_realization(identity_id: str, payload: VoiceRealizationPayload, *, replace: bool):
        try:
            identity = await asyncio.to_thread(VoiceIdentityManager().get, identity_id)
            if replace and not any(item["model"] == payload.model for item in identity["realizations"]):
                raise HTTPException(status_code=404, detail="Voice model version not found")
            check_reference(identity["id"], payload.reference_audio_id)
            reference, transcript = None, None
            if payload.reference_audio_id:
                reference, metadata = await asyncio.to_thread(get_voice_library().get, payload.reference_audio_id)
                transcript = metadata.get("transcript")
            error = await asyncio.to_thread(
                tts_service.validate_voice_realization,
                tts_router=get_tts_router(), model_id=payload.model,
                voice=payload.voice, reference_audio=reference, clone_transcript=transcript,
            )
            if error:
                raise HTTPException(status_code=400, detail=error)
            manager = VoiceIdentityManager()
            save = manager.update_realization if replace else manager.add_realization
            return await asyncio.to_thread(save, identity_id, **payload.model_dump())
        except SampleOwnedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc
        except ExternalProviderError as exc:
            raise tts_service.provider_http_error(exc) from exc
        except NoTTSBackendsError as exc:
            raise tts_service._provider_missing(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.delete("/api/voices/identities/{identity_id}/realizations/{model:path}", status_code=204)
    def delete_voice_realization(identity_id: str, model: str):
        try:
            VoiceIdentityManager().delete_realization(identity_id, model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc
        return Response(status_code=204)

    @router.post("/api/profiles", status_code=201)
    def create_profile(payload: ProfilePayload):
        _reject_invalid_effects(payload.effects)
        try:
            return get_profile_manager().create(**payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc

    @router.get("/api/profiles", response_model=ProfileListResponse)
    def list_profiles():
        # The default comes from the same read as the list, so it is always one of them.
        profiles = get_profile_manager().list_all()
        default_id = next((profile["id"] for profile in profiles if profile.get("is_default")), None)
        return {"profiles": profiles, "default_profile_id": default_id}

    @router.get("/api/profiles/{profile_id}")
    def get_profile(profile_id: str):
        profile = get_profile_manager().get(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile

    @router.get("/api/profiles/{profile_id}/resolve")
    def resolve_profile(profile_id: str):
        try:
            return get_profile_manager().resolve(profile_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.put("/api/profiles/{profile_id}")
    def update_profile(profile_id: str, payload: ProfilePayload):
        _reject_invalid_effects(payload.effects)
        try:
            return get_profile_manager().update(profile_id, **payload.model_dump(exclude_unset=True))
        except ProfileNotFoundError:
            raise HTTPException(status_code=404, detail="Profile not found")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.delete("/api/profiles/{profile_id}", status_code=204)
    def delete_profile(profile_id: str):
        if not get_profile_manager().delete(profile_id):
            raise HTTPException(status_code=404, detail="Profile not found")
        return Response(status_code=204)

    @router.post("/api/profiles/{profile_id}/default", response_model=ProfileListResponse)
    def set_profile_default(profile_id: str):
        try:
            get_profile_manager().set_default(profile_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Profile not found")
        profiles = get_profile_manager().list_all()
        return {"profiles": profiles, "default_profile_id": profile_id}

    @router.get("/api/history", response_model=HistoryListResponse)
    async def list_history(type: str | None = None, limit: SqliteInt = 50, offset: SqliteInt = 0, search: str | None = None):
        return await asyncio.to_thread(
            get_history_manager().list_entries, type_filter=type, limit=limit, offset=offset, search=search
        )

    @router.delete("/api/history/{entry_id}", status_code=204)
    async def delete_history_entry(entry_id: str):
        # Off the event loop: deletes wait on studio.db's shared transaction lock.
        if not await asyncio.to_thread(get_history_manager().delete_entry, entry_id):
            raise HTTPException(status_code=404, detail="History entry not found")
        return Response(status_code=204)

    @router.delete("/api/history")
    async def clear_history():
        return {"deleted": await asyncio.to_thread(get_history_manager().clear_all)}

    @router.post("/api/conversations", status_code=201)
    def create_conversation(payload: ConversationCreatePayload):
        for n, turn in enumerate(payload.turns, start=1):
            reject_invalid_turn(turn, f"Turn {n}")
        return get_conversation_manager().create(payload.name, [turn.model_dump() for turn in payload.turns])

    @router.get("/api/conversations")
    def list_conversations(limit: SqliteInt = 50, offset: SqliteInt = 0):
        return get_conversation_manager().list_all(limit=limit, offset=offset)

    @router.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        item = get_conversation_manager().get(conversation_id)
        if not item:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return item

    @router.post("/api/conversations/{conversation_id}/turns", status_code=201)
    def add_conversation_turn(conversation_id: str, payload: ConversationTurnPayload):
        reject_invalid_turn(payload, "Turn")
        try:
            return get_conversation_manager().add_turn(
                conversation_id=conversation_id,
                speaker=payload.speaker,
                text=payload.text,
                profile_id=payload.profile_id,
                effects=payload.effects,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Conversation not found")

    @router.delete("/api/conversations/{conversation_id}/turns/{turn_id}", status_code=204)
    def delete_conversation_turn(conversation_id: str, turn_id: str):
        if not get_conversation_manager().delete_turn(conversation_id, turn_id):
            raise HTTPException(status_code=404, detail="Turn not found")
        return Response(status_code=204)

    @router.post("/api/conversations/{conversation_id}/render")
    async def render_conversation(conversation_id: str, payload: ConversationRenderPayload):
        try:
            return await _run_render(
                _conversation_renders,
                get_conversation_manager().render,
                conversation_id=conversation_id,
                format=payload.format,
                sample_rate=payload.sample_rate,
                save_turn_audio=payload.save_turn_audio,
            )
        except ConversationNotFoundError:
            raise HTTPException(status_code=404, detail="Conversation not found")
        except ConversationChangedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except tts_service.TTSDisabledError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NoTTSBackendsError as exc:
            raise tts_service._provider_missing(exc) from exc
        except ExternalProviderError as exc:
            raise tts_service.provider_http_error(exc) from exc
        except VoiceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Voice library entry '{exc.args[0]}' not found") from exc
        except NamedVoiceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=_not_found_detail(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/api/conversations/{conversation_id}/audio")
    def get_conversation_audio(conversation_id: str):
        item = get_conversation_manager().get(conversation_id)
        if not item:
            raise HTTPException(status_code=404, detail="Conversation not found")
        output_path = item.get("render_output_path")
        if not output_path:
            raise HTTPException(status_code=404, detail="Conversation has no rendered output")
        path = Path(output_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Rendered audio file not found")
        suffix = path.suffix.lower().lstrip(".")
        return FileResponse(path, media_type=get_content_type(suffix or "wav"))

    @router.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str):
        if not get_conversation_manager().delete(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return Response(status_code=204)

    @router.post("/api/composer/render")
    async def render_composer(payload: ComposerRenderRequest):
        try:
            return await _run_render(
                _composer_renders,
                get_composer_manager().render,
                tracks=[track.model_dump() for track in payload.tracks],
                format=payload.format,
                sample_rate=payload.sample_rate,
                name=payload.name,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/api/composer/renders")
    def list_composer_renders(limit: SqliteInt = 100, offset: SqliteInt = 0):
        return get_composer_manager().list_renders(limit=limit, offset=offset)

    @router.get("/api/composer/render/{composition_id}/audio")
    def get_composer_audio(composition_id: str):
        item = get_composer_manager().get_render(composition_id)
        if not item:
            raise HTTPException(status_code=404, detail="Composition not found")
        output_path = item.get("render_output_path")
        if not output_path:
            raise HTTPException(status_code=404, detail="Composition has no rendered output")
        path = Path(output_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise HTTPException(status_code=404, detail="Rendered audio file not found")
        suffix = path.suffix.lower().lstrip(".")
        return FileResponse(path, media_type=get_content_type(suffix or "wav"))

    @router.delete("/api/composer/render/{composition_id}", status_code=204)
    def delete_composer_render(composition_id: str):
        if not get_composer_manager().delete_render(composition_id):
            raise HTTPException(status_code=404, detail="Composition not found")
        return Response(status_code=204)

    return router
