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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VoiceInfo":
        voice_id = payload.get("id")
        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ValueError("Voice manifest entries require a non-empty id")
        return cls(
            id=voice_id,
            name=str(payload.get("name") or voice_id),
            language=str(payload.get("language") or "en-us"),
            gender=str(payload.get("gender") or "unknown"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
        }


@dataclass(frozen=True)
class TTSModelManifest:
    """Model-specific controls exposed by one TTS provider."""

    id: str
    provider: str
    sample_rate: int
    capabilities: dict[str, Any]
    voices: tuple[VoiceInfo, ...] = ()
    max_input_chars: int | None = None
    revision: str = "1"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TTSModelManifest":
        model_id = payload.get("id")
        provider = payload.get("provider")
        sample_rate = payload.get("sample_rate")
        capabilities = payload.get("capabilities")
        voices = payload.get("voices", [])
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Model manifest requires a non-empty id")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Model manifest requires a non-empty provider")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("Model manifest requires a positive integer sample_rate")
        if not isinstance(capabilities, dict):
            raise ValueError("Model manifest requires a capabilities object")
        if not isinstance(voices, list):
            raise ValueError("Model manifest voices must be a list")
        max_input_chars = payload.get("max_input_chars")
        if max_input_chars is not None and (
            not isinstance(max_input_chars, int) or max_input_chars <= 0
        ):
            raise ValueError("Model manifest max_input_chars must be a positive integer")
        return cls(
            id=model_id,
            provider=provider,
            sample_rate=sample_rate,
            capabilities=dict(capabilities),
            voices=tuple(VoiceInfo.from_dict(item) for item in voices),
            max_input_chars=max_input_chars,
            revision=str(payload.get("revision") or "1"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "provider": self.provider,
            "sample_rate": self.sample_rate,
            "capabilities": dict(self.capabilities),
            "voices": [voice.to_dict() for voice in self.voices],
            "revision": self.revision,
        }
        if self.max_input_chars is not None:
            payload["max_input_chars"] = self.max_input_chars
        return payload


@dataclass
class TTSLoadedModelInfo:
    """Info about a loaded TTS model."""
    model: str
    backend: str
    device: str
    loaded_at: float
    last_used_at: float | None = None




DEFAULT_TTS_CAPABILITIES: dict[str, Any] = {
    "voice_blend": False,
    "voice_design": False,
    "voice_clone": False,
    "streaming": False,
    "instructions": False,
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

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this backend's required packages are installed."""
        return True

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
        **backend_options: Any,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks as float32 numpy arrays at native sample rate.

        Provider-specific options are passed only after capability validation.
        Yields chunks (typically per-sentence) for streaming support.
        """
        ...

    def list_voices(self) -> list[VoiceInfo]: ...
