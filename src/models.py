"""Pydantic models for API requests and responses."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field


# --- OpenAI-compatible response models ---


class TranscriptionResponse(BaseModel):
    """Simple JSON transcription response."""
    text: str


class TranscriptionVerboseResponse(BaseModel):
    """Verbose JSON transcription response."""
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[Segment] = []


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class ModelObject(BaseModel):
    """OpenAI model object."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "open-speech"


class ModelListResponse(BaseModel):
    """OpenAI model list response."""
    object: str = "list"
    data: list[ModelObject] = []


# --- Management API models ---


class LoadedModelInfo(BaseModel):
    model: str
    backend: str
    device: str
    compute_type: str
    loaded_at: float
    last_used_at: float | None = None
    is_default: bool = False
    ttl_remaining: float | None = None


class LoadedModelsResponse(BaseModel):
    models: list[LoadedModelInfo] = []


class PullResponse(BaseModel):
    status: str
    model: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    models_loaded: int = 0


# Rebuild forward refs
TranscriptionVerboseResponse.model_rebuild()
