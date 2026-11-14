"""Pydantic models for TTS API requests and responses."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpeechVoiceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)


# Fields that once had a meaning here. Silently ignoring them would change the
# audio a client expects, so they are still rejected.
RETIRED_SPEECH_FIELDS = frozenset({"voice_blend"})


class TTSSpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request with extended fields.

    Unknown fields are ignored, as OpenAI clients may send newer parameters.
    """

    model_config = ConfigDict(extra="ignore")

    model: str = "kokoro"
    input: str
    voice: str | SpeechVoiceReference = "alloy"
    response_format: str = "mp3"
    stream_format: Literal["audio", "sse"] | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(
        default=None,
        description="Natural-language delivery instructions",
    )
    voice_design: str | None = Field(
        default=None,
        description="Text description of desired voice (Qwen3 only)",
    )
    reference_audio: str | None = Field(
        default=None,
        description="Base64-encoded reference audio for voice cloning",
    )
    voice_library_ref: str | None = Field(
        default=None,
        description="Provider-neutral voice-library asset name for cloning",
    )
    language: str | None = Field(default=None, description="Language code hint (e.g., en, zh, ja, ko)")
    clone_transcript: str | None = Field(
        default=None,
        description="Reference transcript for voice cloning prompt creation",
    )
    input_type: Literal["text", "ssml"] = Field(default="text", description="text or ssml")
    steps: int | None = Field(
        default=None, ge=1, le=100,
        description="Quality steps for a model that has them (Supertonic); other models ignore it",
    )
    # e.g. [{"type":"reverb","room":"small"}]; bounded, so a long invalid list is one 422 error
    effects: list[dict] | None = Field(default=None, max_length=16)

    @model_validator(mode="before")
    @classmethod
    def reject_retired_fields(cls, data):
        if isinstance(data, dict):
            retired = sorted(RETIRED_SPEECH_FIELDS.intersection(data))
            if retired:
                raise ValueError(f"Unsupported field(s): {', '.join(retired)}")
        return data

    @field_validator("voice", mode="after")
    @classmethod
    def resolve_voice_reference(cls, value):
        if not isinstance(value, SpeechVoiceReference):
            return value
        try:
            identity_id = UUID(value.id)
        except ValueError:
            return value.id
        # Identities are stored in canonical form; UUID input is case-insensitive (RFC 4122).
        return f"voice:{identity_id}"


class VoiceObject(BaseModel):
    """Voice metadata."""
    id: str
    name: str
    language: str = "en-us"
    gender: str = "unknown"
    model: str | None = None


class VoiceListResponse(BaseModel):
    """Response for GET /v1/audio/voices."""
    voices: list[VoiceObject] = []


class ModelLoadRequest(BaseModel):
    """Request to load a TTS model; an omitted model means TTS_MODEL."""
    model: str | None = None


class ModelUnloadRequest(BaseModel):
    """Request to unload a TTS model; an omitted model means TTS_MODEL."""
    model: str | None = None
