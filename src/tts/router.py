"""TTS Router — routes model IDs to TTS backend instances."""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo
from src.tts.backends.kokoro import KokoroBackend

logger = logging.getLogger(__name__)


class TTSRouter:
    """Routes TTS requests to the appropriate backend based on model ID."""

    def __init__(self, device: str = "auto") -> None:
        self._backends: dict[str, TTSBackend] = {}
        self._device = device
        # Register kokoro as the default backend
        kokoro = KokoroBackend(device=device)
        self._backends["kokoro"] = kokoro
        self._default_backend = kokoro

    def get_backend(self, model_id: str) -> TTSBackend:
        """Get the backend for a given model ID."""
        if model_id in self._backends:
            return self._backends[model_id]
        return self._default_backend

    def load_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        backend.load_model(model_id)

    def unload_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        backend.unload_model(model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        backend = self.get_backend(model_id)
        return backend.is_model_loaded(model_id)

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        result = []
        for backend in self._backends.values():
            result.extend(backend.loaded_models())
        return result

    def synthesize(
        self,
        text: str,
        model: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Synthesize text to audio chunks."""
        backend = self.get_backend(model)
        return backend.synthesize(text, voice, speed, lang_code)

    def list_voices(self, model: str | None = None) -> list[VoiceInfo]:
        """List available voices."""
        if model and model in self._backends:
            return self._backends[model].list_voices()
        # Aggregate from all backends
        voices = []
        for backend in self._backends.values():
            voices.extend(backend.list_voices())
        return voices
