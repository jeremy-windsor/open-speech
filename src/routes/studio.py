"""Studio, voice-library, and persistence routes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.services import tts as tts_service
from src.tts.pipeline import get_content_type


class ProfilePayload(BaseModel):
    name: str
    backend: str
    model: str | None = None
    voice: str
    speed: float = 1.0
    format: str = "mp3"
    blend: str | None = None
    reference_audio_id: str | None = None
    effects: list[dict | str] = Field(default_factory=list)


class ProfileListResponse(BaseModel):
    profiles: list[dict]
    default_profile_id: str | None = None


class HistoryListResponse(BaseModel):
    items: list[dict]
    total: int
    limit: int
    offset: int


class ConversationTurnPayload(BaseModel):
    speaker: str
    text: str
    profile_id: str | None = None
    effects: list[dict] | None = None


class ConversationCreatePayload(BaseModel):
    name: str
    turns: list[ConversationTurnPayload] = Field(default_factory=list)


class ConversationRenderPayload(BaseModel):
    format: str = "wav"
    sample_rate: int = 24000
    save_turn_audio: bool = True


class ComposerTrack(BaseModel):
    source_path: str
    offset_s: float = 0.0
    volume: float = 1.0
    muted: bool = False
    solo: bool = False
    effects: list[dict] | None = None


class ComposerRenderRequest(BaseModel):
    name: str | None = None
    format: str = "wav"
    sample_rate: int = 24000
    tracks: list[ComposerTrack]



def create_router(*, get_settings: Callable, get_voice_library: Callable, get_profile_manager: Callable, get_history_manager: Callable, get_conversation_manager: Callable, get_composer_manager: Callable) -> APIRouter:
    router = APIRouter()

    @router.post("/api/voices/library", status_code=201)
    async def upload_voice(
        name: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
    ):
        return await tts_service.upload_voice_reference(
            name=name,
            audio=audio,
            settings=get_settings(),
            voice_library=get_voice_library(),
        )

    @router.get("/api/voices/library")
    async def list_library_voices():
        return tts_service.list_library_voices(voice_library=get_voice_library())

    @router.get("/api/voices/library/{name}")
    async def get_library_voice_meta(name: str):
        return tts_service.get_library_voice_metadata(name=name, voice_library=get_voice_library())

    @router.delete("/api/voices/library/{name}", status_code=204)
    async def delete_library_voice(name: str):
        return tts_service.delete_library_voice(name=name, voice_library=get_voice_library())

    @router.get("/api/voice-presets")
    async def get_voice_presets():
        return {"presets": tts_service.load_voice_presets()}

    @router.post("/api/profiles", status_code=201)
    async def create_profile(payload: ProfilePayload):
        try:
            return get_profile_manager().create(**payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.get("/api/profiles", response_model=ProfileListResponse)
    async def list_profiles():
        profiles = get_profile_manager().list_all()
        default_profile = get_profile_manager().get_default()
        return {"profiles": profiles, "default_profile_id": default_profile["id"] if default_profile else None}

    @router.get("/api/profiles/{profile_id}")
    async def get_profile(profile_id: str):
        profile = get_profile_manager().get(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile

    @router.put("/api/profiles/{profile_id}")
    async def update_profile(profile_id: str, payload: ProfilePayload):
        try:
            return get_profile_manager().update(profile_id, **payload.model_dump())
        except KeyError:
            raise HTTPException(status_code=404, detail="Profile not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.delete("/api/profiles/{profile_id}", status_code=204)
    async def delete_profile(profile_id: str):
        if not get_profile_manager().delete(profile_id):
            raise HTTPException(status_code=404, detail="Profile not found")
        return Response(status_code=204)

    @router.post("/api/profiles/{profile_id}/default", response_model=ProfileListResponse)
    async def set_profile_default(profile_id: str):
        try:
            get_profile_manager().set_default(profile_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Profile not found")
        profiles = get_profile_manager().list_all()
        return {"profiles": profiles, "default_profile_id": profile_id}

    @router.get("/api/history", response_model=HistoryListResponse)
    async def list_history(type: str | None = None, limit: int = 50, offset: int = 0):
        return get_history_manager().list_entries(type_filter=type, limit=limit, offset=offset)

    @router.delete("/api/history/{entry_id}", status_code=204)
    async def delete_history_entry(entry_id: str):
        if not get_history_manager().delete_entry(entry_id):
            raise HTTPException(status_code=404, detail="History entry not found")
        return Response(status_code=204)

    @router.delete("/api/history")
    async def clear_history():
        return {"deleted": get_history_manager().clear_all()}

    @router.post("/api/conversations", status_code=201)
    async def create_conversation(payload: ConversationCreatePayload):
        return get_conversation_manager().create(payload.name, [turn.model_dump() for turn in payload.turns])

    @router.get("/api/conversations")
    async def list_conversations(limit: int = 50, offset: int = 0):
        return get_conversation_manager().list_all(limit=limit, offset=offset)

    @router.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        item = get_conversation_manager().get(conversation_id)
        if not item:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return item

    @router.post("/api/conversations/{conversation_id}/turns", status_code=201)
    async def add_conversation_turn(conversation_id: str, payload: ConversationTurnPayload):
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
    async def delete_conversation_turn(conversation_id: str, turn_id: str):
        if not get_conversation_manager().delete_turn(conversation_id, turn_id):
            raise HTTPException(status_code=404, detail="Turn not found")
        return Response(status_code=204)

    @router.post("/api/conversations/{conversation_id}/render")
    async def render_conversation(conversation_id: str, payload: ConversationRenderPayload):
        try:
            return get_conversation_manager().render(
                conversation_id=conversation_id,
                format=payload.format,
                sample_rate=payload.sample_rate,
                save_turn_audio=payload.save_turn_audio,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Conversation not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/api/conversations/{conversation_id}/audio")
    async def get_conversation_audio(conversation_id: str):
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
        return Response(content=path.read_bytes(), media_type=get_content_type(suffix or "wav"))

    @router.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str):
        if not get_conversation_manager().delete(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return Response(status_code=204)

    @router.post("/api/composer/render")
    async def render_composer(payload: ComposerRenderRequest):
        try:
            return get_composer_manager().render(
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
    async def list_composer_renders(limit: int = 100, offset: int = 0):
        return get_composer_manager().list_renders(limit=limit, offset=offset)

    @router.get("/api/composer/render/{composition_id}/audio")
    async def get_composer_audio(composition_id: str):
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
        return Response(content=path.read_bytes(), media_type=get_content_type(suffix or "wav"))

    @router.delete("/api/composer/render/{composition_id}", status_code=204)
    async def delete_composer_render(composition_id: str):
        if not get_composer_manager().delete_render(composition_id):
            raise HTTPException(status_code=404, detail="Composition not found")
        return Response(status_code=204)

    return router
