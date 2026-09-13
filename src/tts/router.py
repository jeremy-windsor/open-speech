"""TTS Router — routes model IDs to TTS backend instances."""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
import os
import pkgutil
import threading
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo
from src.tts.external import ExternalProviderError, ExternalTTSBackend, parse_external_provider_urls

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

    def __init__(self, device: str = "auto", external_providers: str | None = None) -> None:
        self._backends: dict[str, TTSBackend] = {}
        self._device = device
        self._default_backend: TTSBackend | None = None
        self._lock = threading.RLock()
        self._synthesis_locks: dict[str, threading.RLock] = {}

        # Auto-discover and register backends
        for name, cls in _discover_backends().items():
            try:
                is_available = getattr(cls, "is_available", None)
                if callable(is_available) and not cls.is_available():
                    logger.info(
                        "Skipping TTS backend %s — package not installed (rebuild with BAKED_PROVIDERS=%s)",
                        name,
                        name,
                    )
                    continue
                backend = cls(device=device)
                self._backends[name] = backend
                logger.info("Auto-registered TTS backend: %s", name)
            except Exception as e:
                logger.warning("Failed to instantiate backend %s: %s", name, e)

        for name, url in parse_external_provider_urls(
            os.environ.get("TTS_EXTERNAL_PROVIDERS", "")
            if external_providers is None
            else external_providers
        ).items():
            if name in self._backends:
                raise ValueError(f"External TTS provider '{name}' conflicts with a local backend")
            self._backends[name] = ExternalTTSBackend(
                provider=name,
                base_url=url,
                device="external",
            )
            logger.info("Registered isolated TTS provider: %s", name)

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
            synthesis_locks = getattr(self, "_synthesis_locks", None)
            if synthesis_locks is None:
                self._synthesis_locks = {}
                synthesis_locks = self._synthesis_locks
            synthesis_locks.setdefault(name, threading.RLock())
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
            backend = self._backends[prefix]
            supports_model = getattr(backend, "supports_model", None)
            if not callable(supports_model) or supports_model(model_id):
                return backend

        if not self._backends:
            raise RuntimeError("No TTS backends available")
        raise ValueError(f"Unknown TTS model or backend: {model_id}")

    def _synthesis_lock_for(
        self,
        backend: TTSBackend,
        fallback_name: str,
    ) -> threading.RLock:
        synthesis_locks = getattr(self, "_synthesis_locks", None)
        if synthesis_locks is None:
            self._synthesis_locks = {}
            synthesis_locks = self._synthesis_locks
        backend_name = getattr(backend, "name", fallback_name)
        return synthesis_locks.setdefault(backend_name, threading.RLock())

    def list_backends(self) -> list[str]:
        """List registered backend names."""
        return list(self._backends.keys())

    def get_capabilities(self, model_id: str) -> dict[str, Any]:
        """Get capabilities for the backend selected by model ID."""
        backend = self.get_backend(model_id)
        get_capabilities = getattr(backend, "get_capabilities", None)
        if callable(get_capabilities):
            return copy.deepcopy(get_capabilities(model_id))
        return copy.deepcopy(getattr(backend, "capabilities", {}))

    def provider_is_available(self, provider: str) -> bool:
        """Return runtime health for a registered provider."""
        backend = self._backends.get(provider)
        if backend is None:
            return False
        is_provider_available = getattr(backend, "is_provider_available", None)
        if callable(is_provider_available):
            return bool(is_provider_available())
        return True

    def sample_rate_for(self, model_id: str) -> int:
        """Return the native sample rate for the backend selected by model ID."""
        backend = self.get_backend(model_id)
        get_sample_rate = getattr(backend, "get_sample_rate", None)
        if callable(get_sample_rate):
            return int(get_sample_rate(model_id) or 24000)
        return int(getattr(backend, "sample_rate", 24000) or 24000)

    def load_model(self, model_id: str) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
            backend = self.get_backend(model_id)
            synthesis_lock = self._synthesis_lock_for(backend, model_id)
        with synthesis_lock:
            backend.load_model(model_id)

    def unload_model(self, model_id: str) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
            backend = self.get_backend(model_id)
            synthesis_lock = self._synthesis_lock_for(backend, model_id)
        with synthesis_lock:
            backend.unload_model(model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
            backend = self.get_backend(model_id)
            synthesis_lock = self._synthesis_lock_for(backend, model_id)
        if getattr(backend, "requires_model_id", False):
            return backend.is_model_loaded(model_id)
        with synthesis_lock:
            return backend.is_model_loaded(model_id)

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
            backends = [
                (backend, self._synthesis_lock_for(backend, name))
                for name, backend in self._backends.items()
            ]
        result = []
        for backend, synthesis_lock in backends:
            try:
                if getattr(backend, "requires_model_id", False):
                    result.extend(backend.loaded_models())
                else:
                    with synthesis_lock:
                        result.extend(backend.loaded_models())
            except ExternalProviderError:
                logger.debug("External TTS provider %s is unavailable", backend.name)
        return result

    def synthesize(
        self,
        text: str,
        model: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
        **backend_options: Any,
    ) -> Iterator[np.ndarray]:
        """Synthesize text to audio chunks."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._lock = threading.RLock()
            lock = self._lock
        with lock:
            backend = self.get_backend(model)
            # For single-speaker backends (e.g. Piper) the model_id doubles as
            # the voice selector — pass it so the backend picks the right model.
            effective_voice = model if getattr(backend, "single_speaker", False) else voice
            validate_voice = getattr(backend, "validate_voice", None)
            if callable(validate_voice):
                if getattr(backend, "requires_model_id", False):
                    validate_voice(effective_voice, model)
                else:
                    validate_voice(effective_voice)
            synthesis_lock = self._synthesis_lock_for(backend, model)

        # Backends may not be safe for concurrent inference, but one provider
        # must not serialize every other provider. The generator is owned and
        # closed by the caller's thread while this provider-scoped lock is held.
        with synthesis_lock:
            if getattr(backend, "requires_model_id", False):
                backend_options["model_id"] = model
            yield from backend.synthesize(
                text=text,
                voice=effective_voice,
                speed=speed,
                lang_code=lang_code,
                **backend_options,
            )

    def validate_voice(self, model: str, voice: str) -> None:
        """Validate a voice without loading a model when the backend supports it."""
        backend = self.get_backend(model)
        effective_voice = model if getattr(backend, "single_speaker", False) else voice
        validate_voice = getattr(backend, "validate_voice", None)
        if callable(validate_voice):
            if getattr(backend, "requires_model_id", False):
                validate_voice(effective_voice, model)
            else:
                validate_voice(effective_voice)

    def list_voices(self, model: str | None = None) -> list[VoiceInfo]:
        """List available voices."""
        if model:
            with self._lock:
                backend = self.get_backend(model)
                synthesis_lock = self._synthesis_lock_for(backend, model)
            if getattr(backend, "requires_model_id", False):
                return backend.list_voices(model)
            with synthesis_lock:
                return backend.list_voices()
        # Aggregate from all backends
        with self._lock:
            backends = [
                (backend, self._synthesis_lock_for(backend, name))
                for name, backend in self._backends.items()
            ]
        voices: list[VoiceInfo] = []
        for backend, synthesis_lock in backends:
            if getattr(backend, "requires_model_id", False):
                continue
            with synthesis_lock:
                voices.extend(backend.list_voices())
        return voices
