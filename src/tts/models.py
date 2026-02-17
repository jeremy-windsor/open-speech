"""Pydantic models for TTS API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TTSSpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request with extended fields."""
    model: str = "kokoro"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    voice_design: str | None = Field(default=None, description="Text description of desired voice (Qwen3 only)")
    reference_audio: str | None = Field(default=None, description="Base64 or URL of reference audio for voice cloning")
    input_type: str = Field(default="text", description="text or ssml")


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
