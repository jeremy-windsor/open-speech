"""TTS Router — routes model IDs to TTS backend instances."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib
import inspect
import logging
import os
import pkgutil
import threading
import time
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo
from src.tts.external import ExternalProviderError, ExternalTTSBackend, parse_external_provider_urls

logger = logging.getLogger(__name__)


class NoTTSBackendsError(RuntimeError):
    """Raised when TTS is enabled but no backend is installed or configured."""


class _RequestUsage:
    """Per-model synthesis and load activity, which status reads see without waiting for it.

    In-process backends stamp their use when a synthesis starts, so a long request, or a
    stream a slow client is still reading, would otherwise look idle to the TTL and LRU
    checks from its first second. And a backend reports a model only once its load is done,
    so a first load, explicit or started by a request, would be missed by the eviction that
    keeps one TTS model loaded.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, int] = {}
        self._loading: dict[str, int] = {}
        self._finished_at: dict[str, float] = {}

    @staticmethod
    def _enter(counts: dict[str, int], model_id: str) -> None:
        counts[model_id] = counts.get(model_id, 0) + 1

    @staticmethod
    def _leave(counts: dict[str, int], model_id: str) -> None:
        counts[model_id] -= 1
        if not counts[model_id]:
            del counts[model_id]

    @contextlib.contextmanager
    def track(self, model_id: str) -> Iterator[None]:
        with self._lock:
            self._enter(self._active, model_id)
        try:
            yield
        finally:
            with self._lock:
                self._leave(self._active, model_id)
                self._finished_at[model_id] = time.time()

    @contextlib.contextmanager
    def loading(self, model_id: str) -> Iterator[None]:
        with self._lock:
            self._enter(self._loading, model_id)
        try:
            yield
        finally:
            with self._lock:
                self._leave(self._loading, model_id)

    def pending(self) -> set[str]:
        """Models that a load or a request is bringing in or keeping in memory."""
        with self._lock:
            return {*self._active, *self._loading}

    def apply(self, info: TTSLoadedModelInfo) -> TTSLoadedModelInfo:
        with self._lock:
            active = info.model in self._active
            finished_at = self._finished_at.get(info.model)
        if active:
            return dataclasses.replace(info, last_used_at=time.time())
        # A request that ended before the current load belongs to an earlier residency.
        if finished_at is not None and finished_at >= (info.loaded_at or 0.0):
            return dataclasses.replace(
                info, last_used_at=max(info.last_used_at or 0.0, finished_at)
            )
        return info


def speed_in_range(capabilities: Any, speed: float) -> float:
    """The speed held to the model's capabilities["speed_range"] (Supertonic's 0.7 to 1.9); as it is without one."""
    speed_range = capabilities.get("speed_range") if isinstance(capabilities, dict) else None
    if not isinstance(speed_range, dict):
        return speed
    low, high = speed_range.get("min"), speed_range.get("max")
    if low is not None:
        speed = max(low, speed)
    return speed if high is None else min(high, speed)


def _discover_backends() -> dict[str, type]:
    """Auto-discover TTSBackend implementations in src.tts.backends package."""
    discovered: dict[str, type] = {}
    try:
        import src.tts.backends as backends_pkg
        for _importer, modname, _ispkg in pkgutil.iter_modules(backends_pkg.__path__):
            if modname.startswith("_") or modname == "base":
                continue
            try:
                module = importlib.import_module(f"src.tts.backends.{modname}")
                for _name, obj in inspect.getmembers(module, inspect.isclass):
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

    # Set by ModelManager: the GPU memory guard's admission of a load.
    gpu_admission: Any = None

    def __init__(self, device: str = "auto", external_providers: str | None = None) -> None:
        self._backends: dict[str, TTSBackend] = {}
        self._device = device
        self._default_backend: TTSBackend | None = None
        self._lock = threading.RLock()
        self._synthesis_locks: dict[str, threading.RLock] = {}
        self._usage = _RequestUsage()

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

    def _router_lock(self) -> threading.RLock:
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = self._lock = threading.RLock()
        return lock

    def get_backend(self, model_id: str) -> TTSBackend:
        """Get the backend for a given model ID.

        The router lock covers only the registry lookup. An isolated worker answers
        supports_model() from its manifest over HTTP, and a slow or unreachable worker must not
        hold up the other providers' synthesis and model reads while it does.
        """
        with self._router_lock():
            # Direct match
            backend = self._backends.get(model_id)
            if backend is not None:
                return backend
            # Prefix-based routing (e.g. piper/en_US-lessac-medium → piper backend)
            prefix = model_id.split("/")[0] if "/" in model_id else None
            candidate = self._backends.get(prefix) if prefix else None
            has_backends = bool(self._backends)
        if candidate is not None:
            supports_model = getattr(candidate, "supports_model", None)
            if not callable(supports_model) or supports_model(model_id):
                return candidate

        if not has_backends:
            raise NoTTSBackendsError("No TTS backends available")
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

    def _request_usage(self) -> _RequestUsage:
        usage = getattr(self, "_usage", None)
        if usage is None:
            usage = self._usage = _RequestUsage()
        return usage

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

    def max_input_chars_for(self, model_id: str) -> int | None:
        """Return a model-specific limit when its provider advertises one."""
        backend = self.get_backend(model_id)
        get_manifest = getattr(backend, "get_model_manifest", None)
        if callable(get_manifest):
            return get_manifest(model_id).max_input_chars
        return None

    def _admission(self, backend: TTSBackend, model_id: str) -> Any:
        """The guard's admission of loading this model; None when there is none. Entered before the provider's
        synthesis lock, so making room can unload another of its models."""
        admit = getattr(self, "gpu_admission", None)
        is_loaded = getattr(backend, "is_model_loaded", None)
        if admit is None or (callable(is_loaded) and is_loaded(model_id)):
            return None
        return admit(model_id, "tts")

    def fetch_weights(self, model_id: str) -> None:
        """Have an isolated worker download a model's weights before its load is admitted: a first download takes
        minutes and holds no GPU memory, so the guard makes no room for it. A worker without the call downloads
        as it loads."""
        backend = self.get_backend(model_id)
        fetch = getattr(backend, "fetch_weights", None)
        if callable(fetch) and not backend.is_model_loaded(model_id):
            fetch(model_id)

    def load_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        self.fetch_weights(model_id)
        with self._router_lock():
            synthesis_lock = self._synthesis_lock_for(backend, model_id)
        with (
            self._request_usage().loading(model_id),
            self._admission(backend, model_id) or contextlib.nullcontext(),
            synthesis_lock,
        ):
            backend.load_model(model_id)

    def unload_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        with self._router_lock():
            synthesis_lock = self._synthesis_lock_for(backend, model_id)
        with synthesis_lock:
            backend.unload_model(model_id)

    # Status and voice reads do not take the synthesis lock: a stream holds it for as long as
    # its client reads, and model queries (even for STT models) must not wait for that.
    # Backends keep their loaded state consistent for these reads on their own. An in-process
    # model still being loaded counts as loaded, as it did when these reads waited for the
    # load, so the eviction that keeps one TTS model loaded finds it and waits to unload it.
    # Isolated workers report their own state in /health.
    def is_model_loaded(self, model_id: str) -> bool:
        backend = self.get_backend(model_id)
        pending = model_id in self._request_usage().pending()
        if backend.is_model_loaded(model_id):
            return True
        return pending and not isinstance(backend, ExternalTTSBackend)

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        with self._router_lock():
            backends = list(self._backends.values())
        usage = self._request_usage()
        # Taken first, so a load that finishes between the two reads is still listed.
        pending = usage.pending()
        result = []
        for backend in backends:
            try:
                result.extend(usage.apply(info) for info in backend.loaded_models())
            except ExternalProviderError:
                logger.debug("External TTS provider %s is unavailable", backend.name)
        listed = {info.model for info in result}
        for model_id in sorted(pending - listed):
            backend = self._registered_backend(model_id)
            if backend is None or isinstance(backend, ExternalTTSBackend):
                continue
            now = time.time()
            result.append(TTSLoadedModelInfo(
                model=model_id,
                backend=getattr(backend, "name", model_id.split("/", 1)[0]),
                device=getattr(self, "_device", "auto"),
                loaded_at=now,
                last_used_at=now,
            ))
        return result

    def _registered_backend(self, model_id: str) -> TTSBackend | None:
        """The backend a model ID that already routed resolves to, without asking it again."""
        with self._router_lock():
            backend = self._backends.get(model_id)
            if backend is None:
                backend = self._backends.get(model_id.split("/", 1)[0])
            return backend

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
        # Routing and voice checks run outside the router lock: a worker answers both from its
        # manifest over HTTP.
        backend = self.get_backend(model)
        # Some backends select their sole voice through the model ID.
        effective_voice = model if getattr(backend, "single_speaker", False) else voice
        validate_voice = getattr(backend, "validate_voice", None)
        if callable(validate_voice):
            if getattr(backend, "requires_model_id", False):
                validate_voice(effective_voice, model)
            else:
                validate_voice(effective_voice)
        # Every caller (Speak, Reader, Live, Talk, Studio, realtime) gets a speed the model can take.
        speed = speed_in_range(getattr(backend, "capabilities", None), speed)
        with self._router_lock():
            synthesis_lock = self._synthesis_lock_for(backend, model)
        # An in-process model that loads onto the GPU loads here, through the guard, before its first synthesis.
        if not isinstance(backend, ExternalTTSBackend):
            admission = self._admission(backend, model)
            if admission is not None:
                with self._request_usage().loading(model), admission, synthesis_lock:
                    backend.load_model(model)

        # Backends may not be safe for concurrent inference, but one provider
        # must not serialize every other provider. The generator is owned and
        # closed by the caller's thread while this provider-scoped lock is held.
        # A request waiting for the lock already counts as use of its model.
        with self._request_usage().track(model), synthesis_lock:
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

    def validate_language(self, model: str, language: str | None) -> None:
        """Reject a language hint the model cannot speak, before any synthesis starts.

        A backend without its own check takes any hint, as the hint is optional. A backend that
        serves several models (requires_model_id) gets the model ID, as for validate_voice.
        """
        backend = self.get_backend(model)
        validate_language = getattr(backend, "validate_language", None)
        if language and callable(validate_language):
            if getattr(backend, "requires_model_id", False):
                validate_language(language, model)
            else:
                validate_language(language)

    def validate_text(self, model: str, text: str) -> None:
        """Reject text the model would read as nothing (Supertonic and "😀"), for a backend that checks it."""
        validate_text = getattr(self.get_backend(model), "validate_text", None)
        if callable(validate_text):
            validate_text(text)

    def list_voices(self, model: str | None = None) -> list[VoiceInfo]:
        """List available voices."""
        # Like the status reads above, a catalog read does not wait for a running synthesis.
        if model:
            backend = self.get_backend(model)
            if getattr(backend, "requires_model_id", False):
                return backend.list_voices(model)
            return backend.list_voices()
        # Aggregate from all backends
        with self._router_lock():
            backends = list(self._backends.values())
        voices: list[VoiceInfo] = []
        for backend in backends:
            if isinstance(backend, ExternalTTSBackend):
                continue
            voices.extend(backend.list_voices())
        return voices
