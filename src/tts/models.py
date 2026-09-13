"""Pydantic models for TTS API requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TTSSpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request with extended fields."""

    model_config = ConfigDict(extra="forbid")

    model: str = "kokoro"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
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
    language: str | None = Field(default=None, description="Language code hint (e.g., en, zh, ja, ko)")
    clone_transcript: str | None = Field(
        default=None,
        description="Reference transcript for voice cloning prompt creation",
    )
    input_type: Literal["text", "ssml"] = Field(default="text", description="text or ssml")
    effects: list[dict] | None = None  # e.g. [{"type":"reverb","room":"small"}]


class VoiceObject(BaseModel):
    """Voice metadata."""
    id: str
    name: str
    language: str = "en-us"
    gender: str = "unknown"


class VoiceListResponse(BaseModel):
    """Response for GET /v1/audio/voices."""
    voices: list[VoiceObject] = []


class ModelLoadRequest(BaseModel):
    """Request to load a TTS model."""
    model: str = "kokoro"


class ModelUnloadRequest(BaseModel):
    """Request to unload a TTS model."""
    model: str = "kokoro"
