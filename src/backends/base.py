"""Base protocol for STT backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.models import LoadedModelInfo


@runtime_checkable
class STTBackend(Protocol):
    """Protocol that all STT backends must implement."""

    name: str

    def load_model(self, model_id: str) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def loaded_models(self) -> list[LoadedModelInfo]: ...
    def is_model_loaded(self, model_id: str) -> bool: ...

    def transcribe(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]: ...

    def translate(
        self,
        audio: bytes,
        model: str,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]: ...
