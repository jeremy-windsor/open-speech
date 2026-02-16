"""Pydantic models for TTS API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TTSSpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""
    model: str = "kokoro"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


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
