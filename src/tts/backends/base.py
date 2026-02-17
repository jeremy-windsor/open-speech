"""Base protocol for TTS backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass
class VoiceInfo:
    """Metadata about an available voice."""
    id: str
    name: str
    language: str = "en-us"
    gender: str = "unknown"


@dataclass
class TTSLoadedModelInfo:
    """Info about a loaded TTS model."""
    model: str
    backend: str
    device: str
    loaded_at: float
    last_used_at: float | None = None




DEFAULT_TTS_CAPABILITIES: dict[str, Any] = {
    "voice_design": False,
    "voice_clone": False,
    "streaming": False,
    "speakers": [],
    "languages": ["en"],
    "speed_control": True,
    "ssml": False,
    "batch": False,
}

@runtime_checkable
class TTSBackend(Protocol):
    """Protocol that all TTS backends must implement."""

    name: str
    sample_rate: int
    capabilities: dict[str, Any]

    def load_model(self, model_id: str) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def is_model_loaded(self, model_id: str) -> bool: ...
    def loaded_models(self) -> list[TTSLoadedModelInfo]: ...

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks as float32 numpy arrays at native sample rate.
        
        Yields chunks (typically per-sentence) for streaming support.
        """
        ...

    def list_voices(self) -> list[VoiceInfo]: ...
