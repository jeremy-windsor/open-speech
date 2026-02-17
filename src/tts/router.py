"""TTS Router — routes model IDs to TTS backend instances."""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
import pkgutil
import threading
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo

logger = logging.getLogger(__name__)


def _discover_backends() -> dict[str, type]:
    """Auto-discover TTSBackend implementations in src.tts.backends package."""
    discovered: dict[str, type] = {}
    try:
        import src.tts.backends as backends_pkg
        for importer, modname, ispkg in pkgutil.iter_modules(backends_pkg.__path__):
            if modname.startswith("_") or modname == "base":
                continue
            try:
                module = importlib.import_module(f"src.tts.backends.{modname}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        obj is not TTSBackend
                        and hasattr(obj, "name")
                        and hasattr(obj, "sample_rate")
                        and hasattr(obj, "synthesize")
                        and hasattr(obj, "load_model")
                    ):
                        backend_name = getattr(obj, "name", modname)
                        discovered[backend_name] = obj
                        logger.debug("Discovered TTS backend: %s from %s", backend_name, modname)
            except Exception as e:
                logger.warning("Failed to import TTS backend module %s: %s", modname, e)
    except Exception as e:
        logger.warning("Backend auto-discovery failed: %s", e)
    return discovered


class TTSRouter:
    """Routes TTS requests to the appropriate backend based on model ID."""

    def __init__(self, device: str = "auto") -> None:
        self._backends: dict[str, TTSBackend] = {}
        self._device = device
        self._default_backend: TTSBackend | None = None
        self._lock = threading.RLock()

        # Auto-discover and register backends
        for name, cls in _discover_backends().items():
            try:
                backend = cls(device=device)
                self._backends[name] = backend
                logger.info("Auto-registered TTS backend: %s", name)
            except Exception as e:
                logger.warning("Failed to instantiate backend %s: %s", name, e)

        # Set default
        if "kokoro" in self._backends:
            self._default_backend = self._backends["kokoro"]
        elif self._backends:
            self._default_backend = next(iter(self._backends.values()))

    def register_backend(self, name: str, backend: TTSBackend) -> None:
        """Register a TTS backend instance.
        
        Community contributors can use this to add custom backends:
            router.register_backend("my_tts", MyTTSBackend(device="cpu"))
        """
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
            self._backends[name] = backend
            logger.info("Registered TTS backend: %s", name)
            if self._default_backend is None:
                self._default_backend = backend

    def get_backend(self, model_id: str) -> TTSBackend:
        """Get the backend for a given model ID."""
        # Direct match
        if model_id in self._backends:
            return self._backends[model_id]
        # Prefix-based routing (e.g. piper/en_US-lessac-medium → piper backend)
        prefix = model_id.split("/")[0] if "/" in model_id else None
        if prefix and prefix in self._backends:
            return self._backends[prefix]

        # Heuristic routing for known model-id families
        if model_id.startswith("qwen3-tts") and "qwen3" in self._backends:
            return self._backends["qwen3"]
        if model_id.startswith("fish-speech") and "fish-speech" in self._backends:
            return self._backends["fish-speech"]

        if self._default_backend is not None:
            return self._default_backend
        raise RuntimeError("No TTS backends available")

    def list_backends(self) -> list[str]:
        """List registered backend names."""
        return list(self._backends.keys())

    def get_capabilities(self, model_id: str) -> dict[str, Any]:
        """Get capabilities for the backend selected by model ID."""
        backend = self.get_backend(model_id)
        return copy.deepcopy(getattr(backend, "capabilities", {}))

    def load_model(self, model_id: str) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
            backend = self.get_backend(model_id)
            backend.load_model(model_id)

    def unload_model(self, model_id: str) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
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
