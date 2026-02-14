"""Backend router — maps model IDs to backend implementations."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.backends.base import STTBackend
from src.backends.faster_whisper import FasterWhisperBackend
from src.models import LoadedModelInfo

logger = logging.getLogger(__name__)


class BackendRouter:
    """Routes requests to the appropriate STT backend based on model ID."""

    def __init__(self) -> None:
        self._backends: dict[str, STTBackend] = {}
        self._lock = asyncio.Lock()
        # Register faster-whisper as the default (and only Phase 1) backend
        fw = FasterWhisperBackend()
        self._backends["faster-whisper"] = fw
        self._default_backend = fw

    def get_backend(self, model_id: str) -> STTBackend:
        """Get the backend for a given model ID.
        
        Phase 1: everything routes to faster-whisper.
        Future: route based on model_id prefix or registry.
        """
        # For now, all models go to faster-whisper
        return self._default_backend

    def load_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        backend.load_model(model_id)

    def unload_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        backend.unload_model(model_id)

    def loaded_models(self) -> list[LoadedModelInfo]:
        result = []
        for backend in self._backends.values():
            result.extend(backend.loaded_models())
        return result

    def is_model_loaded(self, model_id: str) -> bool:
        backend = self.get_backend(model_id)
        return backend.is_model_loaded(model_id)

    def transcribe(self, audio: bytes, model: str, **kwargs: Any) -> dict[str, Any]:
        backend = self.get_backend(model)
        return backend.transcribe(audio, model, **kwargs)

    def translate(self, audio: bytes, model: str, **kwargs: Any) -> dict[str, Any]:
        backend = self.get_backend(model)
        return backend.translate(audio, model, **kwargs)


# Global singleton
router = BackendRouter()
