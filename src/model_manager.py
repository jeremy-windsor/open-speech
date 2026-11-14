"""Unified Model Manager — wraps STT and TTS routers with a single interface."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import hf_hub_cache_dir, settings
from src.gpu_memory import (
    DEFAULT_GPU_MB,
    LOCKED,
    RELEASE_WAIT_S,
    GpuMemoryGuard,
    GpuResident,
    InsufficientGpuMemoryError,
)
from src.model_registry import PROVIDER_DISPLAY_NAMES, get_known_model, get_known_models, known_model_size_mb
from src.tts.external import ExternalProviderError, ExternalTTSBackend
from src.utils.redact import redact_paths

logger = logging.getLogger(__name__)

# Hugging Face repositories an in-process provider downloads, when not models--<model id>.
# pocket-tts fetches kyutai/pocket-tts once its terms are accepted, else the variant
# without voice cloning, which also holds its predefined voices.
_PROVIDER_REPOS: dict[str, tuple[str, ...]] = {
    "kokoro": ("hexgrad/Kokoro-82M", "hexgrad/Kokoro-82M-v1.1-zh"),
    "pocket-tts": ("kyutai/pocket-tts", "kyutai/pocket-tts-without-voice-cloning"),
    "supertonic": ("Supertone/supertonic-3",),
}


def _check_provider(model_type: str, provider: str, stt_router: Any, tts_router: Any) -> bool:
    """Return whether the provider backend is currently registered/available."""
    if model_type == "tts":
        return provider in getattr(tts_router, "_backends", {})

    stt_backends = getattr(stt_router, "_backends", None)
    if not stt_backends:
        return True
    return provider in stt_backends


def pinned_default_models() -> set[str]:
    """The defaults that stay loaded (OS_PIN_DEFAULT_MODELS): TTL and LRU eviction leave them, and a load with
    keep_pinned loads next to them."""
    if not settings.os_pin_default_models:
        return set()
    return {settings.stt_model, settings.tts_model}


class ModelState(str, Enum):
    AVAILABLE = "available"
    PROVIDER_MISSING = "provider_missing"
    PROVIDER_INSTALLED = "provider_installed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADED = "loaded"


@dataclass
class ModelLifecycleError(Exception):
    message: str
    code: str
    model_id: str
    provider: str | None = None
    action: str | None = None
    details: dict[str, Any] | None = None
    # None means a client error: 404 for unknown_model, 400 otherwise.
    status_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "message": self.message,
            "code": self.code,
            "model": self.model_id,
            "provider": self.provider,
            "action": self.action,
        }
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class ModelInfo:
    id: str
    type: str  # "stt" or "tts"
    provider: str
    device: str | None = None
    state: ModelState = ModelState.AVAILABLE
    size_mb: int | None = None
    loaded_at: float | None = None
    last_used_at: float | None = None
    is_default: bool = False
    description: str | None = None
    source: str | None = None
    model_format: str | None = None
    provider_available: bool = True
    # Why a worker that answers says it can't run ({code, message} from its /health), or None.
    provider_failure: dict[str, str] | None = None
    notice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.type,
            "provider": self.provider,
            "device": self.device,
            "state": self.state.value,
            "size_mb": self.size_mb,
            "loaded_at": self.loaded_at,
            "last_used_at": self.last_used_at,
            "is_default": self.is_default,
            # A pinned default stays loaded next to another TTS model's load, and an STT load with keep_pinned.
            "pinned": self.is_default and self.id in pinned_default_models(),
            "provider_available": self.provider_available,
            "provider_failure": dict(self.provider_failure) if self.provider_failure else None,
        }
        if self.description:
            d["description"] = self.description
        if self.source:
            d["source"] = self.source
        if self.model_format:
            d["model_format"] = self.model_format
        if self.notice:
            d["notice"] = self.notice
        return d


# How long after a displaced STT model comes back the same model can't unload it again, unless a person loads it.
DISPLACE_COOLDOWN_S = 60.0


@dataclass
class _Room:
    model_id: str
    user: bool
    # Set when the load unloaded the pinned STT model.
    notice: str | None = None


class ModelManager:
    """Unified model lifecycle for STT and TTS."""

    def __init__(self, stt_router: Any, tts_router: Any) -> None:
        self._stt = stt_router
        self._tts = tts_router
        # Every load, unload, download and eviction of a model takes its type's lock, whichever
        # route or timer starts it (services/tts.py and the legacy /api/ps routes take them too).
        # Each reads what is loaded before it changes it, so two at once could both find nothing
        # to evict and leave two models where one is promised. Loads evict only their own type,
        # so STT and TTS each have a lock: an STT unload waits for a running decode, and a TTS
        # switch must not wait behind it.
        self._operation_locks = {"stt": threading.RLock(), "tts": threading.RLock()}
        # (provider, STT_DEVICE) -> the device an STT backend runs on; asked once, not on every inventory poll.
        self._stt_runtime_devices: dict[tuple[str, str], str | None] = {}
        self.gpu_guard = GpuMemoryGuard()
        self._room = threading.local()
        # A pinned STT model unloaded to make room -> the model it made room for.
        self._displaced: dict[str, str] = {}
        # A model the pinned STT made room for -> when the STT came back.
        self._stt_restored_at: dict[str, float] = {}
        for router in (stt_router, tts_router):
            with contextlib.suppress(AttributeError, TypeError):
                router.gpu_admission = self.gpu_admission

    def operation_lock(self, model_type: str) -> threading.RLock:
        """The lock that serializes every load, unload, download and eviction of this type."""
        return self._operation_locks[model_type]

    def _tts_backend_rejects(self, model_id: str) -> bool:
        """Whether the registered backend for this ID's prefix does not serve it."""
        tts_backends = getattr(self._tts, "_backends", {})
        prefix = model_id.split("/", 1)[0]
        if model_id in tts_backends or "/" not in model_id or prefix not in tts_backends:
            return False
        get_backend = getattr(self._tts, "get_backend", None)
        try:
            if callable(get_backend):
                get_backend(model_id)
            return False
        except ValueError:
            return True

    def _resolve_type(self, model_id: str) -> str:
        known = get_known_model(model_id)
        if known:
            return known["type"]
        # A registered prefix is not enough, even for TTS_MODEL: "kokoro/typo" must not act
        # on Kokoro's files, and "piper/<typo>" has no Piper voice files to look up.
        rejected = self._tts_backend_rejects(model_id)
        if model_id == settings.tts_model and not rejected:
            return "tts"
        if model_id == settings.stt_model:
            return "stt"
        for m in self._tts.loaded_models():
            if m.model == model_id:
                return "tts"
        for m in self._stt.loaded_models():
            if m.model == model_id:
                return "stt"
        tts_backends = getattr(self._tts, "_backends", {})
        prefix = model_id.split("/", 1)[0]
        routed = model_id in tts_backends or ("/" in model_id and prefix in tts_backends)
        if routed and not rejected:
            return "tts"
        raise ModelLifecycleError(
            message=f"Unknown model '{model_id}'",
            code="unknown_model",
            model_id=model_id,
        )

    def _provider_from_model(self, model_id: str) -> str:
        known = get_known_model(model_id)
        if known:
            return known["provider"]
        if model_id.startswith("piper/") or model_id.startswith("piper-"):
            return "piper"
        if model_id.startswith("pocket-tts"):
            return "pocket-tts"
        if model_id == "kokoro":
            return "kokoro"
        prefix = model_id.split("/", 1)[0] if "/" in model_id else None
        if prefix and prefix in getattr(self._tts, "_backends", {}):
            return prefix
        return "faster-whisper"

    def _provider_runtime_available(self, provider: str) -> bool:
        checker = getattr(self._tts, "provider_is_available", None)
        if callable(checker):
            return bool(checker(provider))
        return provider in getattr(self._tts, "_backends", {})

    def _provider_failure(self, provider: str) -> dict[str, str] | None:
        """Why a registered TTS provider's worker says it can't run, as {code, message}; None when it gives no
        reason (it is ok, only unavailable, or doesn't answer)."""
        backend = getattr(self._tts, "_backends", {}).get(provider)
        get_failure = getattr(backend, "provider_failure", None)
        failure = get_failure() if callable(get_failure) else None
        if isinstance(failure, dict) and isinstance(failure.get("code"), str) and isinstance(failure.get("message"), str):
            return {"code": failure["code"], "message": failure["message"]}
        return None

    def _stt_runtime_device(self, provider: str) -> str | None:
        """The device an STT backend loads its models on ("cuda" or "cpu"), or None when it doesn't say."""
        key = (provider, str(settings.stt_device))
        if key not in self._stt_runtime_devices:
            backend = getattr(self._stt, "_backends", {}).get(provider)
            runtime = getattr(backend, "runtime", None)
            device = None
            if callable(runtime):
                try:
                    device = runtime()[0]
                except Exception:
                    logger.debug("Could not ask the %s backend for its device", provider, exc_info=True)
            self._stt_runtime_devices[key] = device
        return self._stt_runtime_devices[key]

    def _registry_size_mb(self, known: dict, device: str | None = None) -> int | None:
        """A registry model's size on disk on the device it runs on: a loaded model's own, else its backend's."""
        if known.get("size_mb_by_device") and not device and known.get("type") == "stt":
            device = self._stt_runtime_device(known["provider"])
        return known_model_size_mb(known, device)

    def resolve_provider(self, model_id: str) -> str:
        return self._provider_from_model(model_id)

    # --- GPU memory guard ---

    @staticmethod
    def _on_gpu(device: str | None) -> bool:
        device = str(device or "").strip().lower()
        return device.startswith("cuda") or device == "auto"

    def _tts_backend_of(self, model_id: str) -> Any:
        get_backend = getattr(self._tts, "get_backend", None)
        try:
            return get_backend(model_id) if callable(get_backend) else None
        except Exception:
            return None

    @staticmethod
    def _reported_mb(value: Any, total_mb: int | None) -> int | None:
        """A worker's size figure, or None when it isn't one (a value the card couldn't hold)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return None
        if total_mb and value >= total_mb:
            return None
        return round(value)

    def gpu_need_mb(self, model_id: str, model_type: str, total_mb: int | None = None) -> int | None:
        """The GPU memory, in MiB, a model will hold once loaded; None when it loads on the CPU."""
        known = get_known_model(model_id) or {}
        if model_type == "tts":
            backend = self._tts_backend_of(model_id)
            if isinstance(backend, ExternalTTSBackend):
                reported = self._reported_mb(backend.gpu_memory_need_mb(model_id), total_mb)
                return reported or known.get("gpu_mb") or DEFAULT_GPU_MB
            if not known.get("gpu_mb"):
                return None
            get_device = getattr(backend, "_get_device", None)
            device = get_device() if callable(get_device) else settings.tts_effective_device
            return known["gpu_mb"] if self._on_gpu(device) else None
        provider = known.get("provider") or self._provider_from_model(model_id)
        device = self._stt_runtime_device(provider) or settings.stt_device
        if not self._on_gpu(device):
            return None
        return known.get("gpu_mb") or DEFAULT_GPU_MB

    def _resident_mb(self, model_id: str, reported: Any = None, total_mb: int | None = None) -> int:
        known = get_known_model(model_id) or {}
        return self._reported_mb(reported, total_mb) or int(known.get("gpu_mb") or DEFAULT_GPU_MB)

    @staticmethod
    def _growth_mb(model_id: str, gpu_mb: int) -> int:
        """How much more a loaded model may take while it runs: Parakeet up to its arena cap."""
        known = get_known_model(model_id) or {}
        cap = settings.stt_parakeet_gpu_mem_limit_mb
        if known.get("provider") == "parakeet" and cap > 0:
            return max(cap - gpu_mb, 0)
        return int(known.get("gpu_growth_mb") or 0)

    def _gpu_residents(self, target_id: str, target_type: str, total_mb: int | None = None, *,
                       displace_stt: bool = False) -> list[GpuResident]:
        pinned = self._pinned_models()
        target_backend = self._tts_backend_of(target_id) if target_type == "tts" else None
        target_worker = target_backend.name if isinstance(target_backend, ExternalTTSBackend) else None
        workers = {
            backend.name for backend in list(getattr(self._tts, "_backends", {}).values())
            if isinstance(backend, ExternalTTSBackend)
        }
        residents: list[GpuResident] = []
        for model_type, router in (("stt", self._stt), ("tts", self._tts)):
            for m in router.loaded_models():
                if not self._on_gpu(m.device):
                    continue
                if model_type == "tts" and m.backend not in workers and self.gpu_need_mb(m.model, "tts") is None:
                    continue  # an in-process model that runs on the CPU, listed while it loads
                gpu_mb = self._resident_mb(m.model, getattr(m, "gpu_memory_mb", None), total_mb)
                is_pinned = m.model in pinned
                residents.append(GpuResident(
                    id=m.model,
                    type=model_type,
                    gpu_mb=gpu_mb,
                    last_used_at=m.last_used_at if m.last_used_at is not None else (m.loaded_at or 0.0),
                    pinned=is_pinned,
                    replaced_by_load=model_type == "tts" and target_worker is not None and m.backend == target_worker,
                    growth_mb=self._growth_mb(m.model, gpu_mb) if is_pinned else 0,
                    displaceable=displace_stt and is_pinned and model_type == "stt",
                ))
        return residents

    def _unload_for_gpu(self, resident: GpuResident, for_model: str) -> bool | str:
        operation_lock = self.operation_lock(resident.type)
        if not operation_lock.acquire(blocking=False):
            return LOCKED
        try:
            unloaded = self._unload_if_idle(ModelInfo(id=resident.id, type=resident.type, provider=""))
        finally:
            operation_lock.release()
        if unloaded and resident.pinned:
            self._displaced[resident.id] = for_model
            room = getattr(self._room, "current", None)
            if room is not None:
                room.notice = self.displaced_notice(resident.id)
            logger.info("Unloaded the pinned %s to make room for %s; it reloads on its next use", resident.id,
                        for_model)
        return unloaded

    @contextlib.contextmanager
    def make_room_for(self, model_id: str, *, user: bool = False) -> Iterator[_Room]:
        """Mark a load of ``model_id`` in this thread as one a person asked for: when it doesn't fit, it may unload the
        pinned STT model. ``user`` (Models › Load) also skips DISPLACE_COOLDOWN_S."""
        room = _Room(model_id, user)
        previous = getattr(self._room, "current", None)
        self._room.current = room
        try:
            yield room
        finally:
            self._room.current = previous

    def _may_displace_stt(self, model_id: str, model_type: str) -> bool:
        room = getattr(self._room, "current", None)
        if model_type != "tts" or room is None or room.model_id != model_id or model_id in self._pinned_models():
            return False
        restored_at = self._stt_restored_at.get(model_id)
        return room.user or restored_at is None or time.monotonic() - restored_at >= DISPLACE_COOLDOWN_S

    def displaced_notice(self, model_id: str) -> str:
        known = get_known_model(model_id) or {}
        name = PROVIDER_DISPLAY_NAMES.get(known.get("provider", ""), model_id)
        return f"{name} was unloaded to make room; it reloads on the next transcription."

    @contextlib.contextmanager
    def _restoring(self, model_id: str, admission: contextlib.AbstractContextManager) -> Iterator[None]:
        with admission:
            yield
        made_room_for = self._displaced.pop(model_id, None)
        if made_room_for is not None:
            self._stt_restored_at[made_room_for] = time.monotonic()

    def _worker_releasing(self) -> bool:
        """Whether a worker unloaded a model in the last RELEASE_WAIT_S and may still hold its memory."""
        now = time.monotonic()
        return any(
            isinstance(backend, ExternalTTSBackend) and backend.last_unload_at is not None
            and now - backend.last_unload_at < RELEASE_WAIT_S
            for backend in list(getattr(self._tts, "_backends", {}).values())
        )

    def gpu_admission(self, model_id: str, model_type: str) -> contextlib.AbstractContextManager | None:
        """The context a load onto the GPU runs in (it makes room or raises InsufficientGpuMemoryError); None when
        the model runs on the CPU, the guard is off, or there is no GPU to read."""
        need_mb = self.gpu_need_mb(model_id, model_type)
        if need_mb is None:
            return None
        memory = self.gpu_guard.read()
        if memory is None:
            return None
        if need_mb >= memory.total_mb:
            need_mb = self.gpu_need_mb(model_id, model_type, memory.total_mb)
        displace_stt = self._may_displace_stt(model_id, model_type)
        admission = self.gpu_guard.admission(
            model_id,
            need_mb,
            residents=lambda: self._gpu_residents(model_id, model_type, memory.total_mb, displace_stt=displace_stt),
            unload_idle=lambda resident: self._unload_for_gpu(resident, model_id),
            pinned=model_id in self._pinned_models(),
            releasing=self._worker_releasing,
        )
        if model_type == "stt" and model_id in self._displaced:
            return self._restoring(model_id, admission)
        return admission

    def _require_provider(self, model_id: str, action: str) -> str:
        self._resolve_type(model_id)
        provider = self._provider_from_model(model_id)
        return provider

    def _check_loadable(self, model_id: str, model_type: str, provider: str, action: str) -> None:
        """Raise unless the model's provider is installed, online and serves this model."""
        if not _check_provider(model_type, provider, self._stt, self._tts):
            known = get_known_model(model_id)
            if model_type == "tts" and known and known.get("optional_provider"):
                # The Dockerfile bakes in only the in-process providers and rejects these names.
                advice = (
                    f"Run '{provider}' as an isolated worker (docker-compose.voice-models.yml) "
                    "and list it in TTS_EXTERNAL_PROVIDERS."
                )
            else:
                advice = f"Rebuild with BAKED_PROVIDERS including '{provider}'."
            raise ModelLifecycleError(
                message=f"Provider '{provider}' is not installed for model '{model_id}'. {advice}",
                code="provider_missing",
                model_id=model_id,
                provider=provider,
                action=action,
            )
        if model_type == "tts" and not self._provider_runtime_available(provider):
            failure = self._provider_failure(provider)
            reason = f"its worker can't run: {failure['message'].rstrip('.')}" if failure else "unavailable"
            raise ModelLifecycleError(
                message=f"Provider '{provider}' is configured but {reason}.",
                code="provider_unavailable",
                model_id=model_id,
                provider=provider,
                action=action,
                status_code=503,
            )

        if model_type == "tts":
            try:
                get_capabilities = getattr(self._tts, "get_capabilities", None)
                if callable(get_capabilities):
                    get_capabilities(model_id)
            except ExternalProviderError as e:
                raise ModelLifecycleError(
                    message=str(e),
                    code=e.code,
                    model_id=model_id,
                    provider=provider,
                    action=action,
                    status_code=e.status_code,
                ) from e
            except ValueError as e:
                raise ModelLifecycleError(
                    message=str(e),
                    code="unknown_model",
                    model_id=model_id,
                    provider=provider,
                    action=action,
                ) from e

    @staticmethod
    def _require_enabled(model_id: str, model_type: str, action: str) -> None:
        """TTS_ENABLED=false: no TTS model is loaded or downloaded, whichever route asks."""
        if model_type == "tts" and not settings.tts_enabled:
            raise ModelLifecycleError(
                message="TTS is disabled",
                code="tts_disabled",
                model_id=model_id,
                action=action,
                status_code=404,
            )

    def load(
        self, model_id: str, device: str | None = None, _evict_others: bool = True, keep_pinned: bool = False,
        user: bool = False,
    ) -> ModelInfo:
        """Load a model, unloading the other loaded models of its type first. A pinned default (OS_PIN_DEFAULT_MODELS)
        stays loaded next to it: always for a TTS load, with ``keep_pinned`` for an STT load, as when a request loads a
        model on demand. ``user``: a person pressed Load (see make_room_for)."""
        model_type = self._resolve_type(model_id)
        self._require_enabled(model_id, model_type, "load")
        if model_type == "stt":
            self._fetch_stt_before_load(model_id)
        else:
            self._fetch_tts_before_load(model_id)
        explicit = model_type == "tts" and _evict_others
        with (
            self.operation_lock(model_type),
            self.make_room_for(model_id, user=user) if explicit else nullcontext() as room,
        ):
            info = self._load(model_id, model_type, _evict_others=_evict_others, keep_pinned=keep_pinned)
        if room is not None:
            info.notice = room.notice
        return info

    def _fetch_tts_before_load(self, model_id: str) -> None:
        """Have an isolated worker download the model's weights before the load takes the TTS lock and the guard's
        room, so the models in use keep serving through a first download."""
        fetch = getattr(self._tts, "fetch_weights", None)
        if not callable(fetch):
            return
        provider = self._require_provider(model_id, "load")
        self._check_loadable(model_id, "tts", provider, "load")
        try:
            fetch(model_id)
        except ExternalProviderError as e:
            raise ModelLifecycleError(
                message=str(e), code=e.code, model_id=model_id, provider=provider, action="load",
                status_code=e.status_code,
            ) from e

    def _fetch_stt_before_load(self, model_id: str) -> None:
        """Download an STT model's files before its load takes the STT lock.

        So no other STT load or unload waits for the download, and the model in use, which the load unloads
        first, keeps serving meanwhile.
        """
        future = self.start_stt_fetch(model_id)
        if future is None:
            return
        try:
            future.result()
        except Exception as e:
            raise self.stt_fetch_failed(model_id, e) from e

    def start_stt_fetch(self, model_id: str) -> concurrent.futures.Future | None:
        """Start, or join, the fetch of the files an STT model's load needs; None when it needs none.

        The backend fetches outside every STT lock, one fetch per model, so the Models route awaits it before it
        takes its STT queue. A model whose cache folder exists is fetched too: a download that stopped half way
        leaves one, and the load would finish it after unloading the model in use.
        """
        start_fetch = getattr(self._stt, "start_fetch", None)
        if not callable(start_fetch) or self._stt.is_model_loaded(model_id):
            return None
        provider = self._require_provider(model_id, "load")
        self._check_loadable(model_id, "stt", provider, "load")
        return start_fetch(model_id)

    def stt_fetch_failed(self, model_id: str, error: Exception) -> ModelLifecycleError:
        """The load failure of an STT model whose files could not be fetched."""
        return ModelLifecycleError(
            message=f"Failed to load model '{model_id}': {redact_paths(str(error))}",
            code="load_failed",
            model_id=model_id,
            provider=self._provider_from_model(model_id),
            action="load",
            details={"exception": type(error).__name__},
            status_code=500,
        )

    def _load(self, model_id: str, model_type: str, *, _evict_others: bool, keep_pinned: bool = False) -> ModelInfo:
        provider = self._require_provider(model_id, "load")
        self._check_loadable(model_id, model_type, provider, "load")

        # Only evict when explicitly loading (not when called from download/prefetch)
        evicted_models: list[str] = []
        if _evict_others:
            kept = self._pinned_models() if keep_pinned or model_type == "tts" else set()
            for m in self.list_loaded():
                if m.type == model_type and m.id != model_id and m.id not in kept:
                    try:
                        self.unload(m.id)
                        evicted_models.append(m.id)
                        logger.info("Auto-unloaded %s model %s to load %s", model_type.upper(), m.id, model_id)
                    except Exception as e:
                        self._restore_models(evicted_models)
                        raise ModelLifecycleError(
                            message=f"Could not unload '{m.id}' before loading '{model_id}'",
                            code="unload_failed",
                            model_id=model_id,
                            provider=provider,
                            action="load",
                            details={"previous_model": m.id, "exception": type(e).__name__},
                            status_code=500,
                        ) from e

        try:
            if model_type == "tts":
                self._tts.load_model(model_id)
                for m in self._tts.loaded_models():
                    if m.model == model_id:
                        return ModelInfo(
                            id=model_id, type="tts", provider=m.backend,
                            device=m.device, state=ModelState.LOADED,
                            loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                            is_default=(model_id == settings.tts_model),
                            provider_available=True,
                        )
                return ModelInfo(id=model_id, type="tts", provider=provider,
                                 state=ModelState.LOADED, is_default=(model_id == settings.tts_model), provider_available=True)

            self._stt.load_model(model_id)
            for m in self._stt.loaded_models():
                if m.model == model_id:
                    return ModelInfo(
                        id=model_id, type="stt", provider=m.backend,
                        device=m.device, state=ModelState.LOADED,
                        loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                        is_default=(model_id == settings.stt_model),
                        provider_available=True,
                    )
            return ModelInfo(id=model_id, type="stt", provider=provider,
                             state=ModelState.LOADED, is_default=(model_id == settings.stt_model), provider_available=True)
        except ModelLifecycleError:
            self._restore_models(evicted_models)
            raise
        except InsufficientGpuMemoryError as e:
            self._restore_models(evicted_models)
            raise ModelLifecycleError(
                message=str(e),
                code=e.code,
                model_id=model_id,
                provider=provider,
                action="load",
                details={"gpu": e.details},
                status_code=e.status_code,
            ) from e
        except ExternalProviderError as e:
            self._restore_models(evicted_models)
            raise ModelLifecycleError(
                message=str(e),
                code=e.code,
                model_id=model_id,
                provider=provider,
                action="load",
                status_code=e.status_code,
            ) from e
        except Exception as e:
            self._restore_models(evicted_models)
            raise ModelLifecycleError(
                message=f"Failed to load model '{model_id}': {redact_paths(str(e))}",
                code="load_failed",
                model_id=model_id,
                provider=provider,
                action="load",
                details={"exception": type(e).__name__},
                status_code=500,
            ) from e

    def _restore_models(self, model_ids: list[str]) -> None:
        for model_id in model_ids:
            try:
                if self._resolve_type(model_id) == "tts":
                    self._tts.load_model(model_id)
                else:
                    self._stt.load_model(model_id)
                logger.info("Restored model %s after replacement load failure", model_id)
            except Exception:
                logger.exception("Failed to restore model %s after replacement load failure", model_id)

    def download(self, model_id: str) -> ModelInfo:
        provider = self._require_provider(model_id, "download")
        model_type = self._resolve_type(model_id)
        self._require_enabled(model_id, model_type, "download")
        # A download changes no residency, so a fetch that needs no load runs outside the lock.
        if not self._fetch_without_loading(model_id, model_type, provider):
            with self.operation_lock(model_type):
                self._download_by_loading(model_id, model_type)
        info = self.status(model_id)
        info.provider = provider
        return info

    def _fetch_without_loading(self, model_id: str, model_type: str, provider: str) -> bool:
        """Fetch a model's weights into its cache without loading it; False when only a load can.

        faster-whisper, and the TTS backends with a download_model hook (Kokoro, Piper),
        download from the Hugging Face hub into the cache the artifact scan reads, and so does a
        worker with POST /v1/models/download (Breeze) into its own volume. Pocket-TTS picks its
        repository while it loads, and the other workers have no download-only call, so those
        still load once and unload again.
        """
        if model_type == "stt":
            self._check_loadable(model_id, model_type, provider, "download")
            download_model = getattr(self._stt, "download_model", None)
            return callable(download_model) and bool(download_model(model_id))

        backend = getattr(self._tts, "_backends", {}).get(provider)
        fetch_weights = getattr(backend, "fetch_weights", None)
        if callable(fetch_weights):
            self._check_loadable(model_id, model_type, provider, "download")
            try:
                return self._tts.is_model_loaded(model_id) or fetch_weights(model_id)
            except ExternalProviderError as e:
                raise ModelLifecycleError(
                    message=str(e), code=e.code, model_id=model_id, provider=provider, action="download",
                    status_code=e.status_code,
                ) from e
        download_model = getattr(backend, "download_model", None)
        if not callable(download_model):
            return False
        self._check_loadable(model_id, model_type, provider, "download")
        if self._tts.is_model_loaded(model_id):
            return True  # its weights are in the cache already
        download_model(model_id)
        return True

    def _download_by_loading(self, model_id: str, model_type: str) -> None:
        previously_loaded = [
            model.id for model in self.list_loaded() if model.type == model_type
        ]
        # Load to trigger the weights fetch, then unload if not already loaded.
        was_loaded = False
        try:
            if model_type == "tts":
                was_loaded = self._tts.is_model_loaded(model_id)
            else:
                was_loaded = self._stt.is_model_loaded(model_id)
        except Exception:
            was_loaded = False

        try:
            self.load(model_id, _evict_others=False)
            if not was_loaded:
                self.unload(model_id)
        finally:
            # Some providers can hold only one model and replace the active
            # model even when the manager does not explicitly evict it.
            loaded_after = {
                model.id for model in self.list_loaded() if model.type == model_type
            }
            self._restore_models([
                previous for previous in previously_loaded if previous not in loaded_after
            ])

    def unload(self, model_id: str) -> None:
        model_type = self._resolve_type(model_id)
        with self.operation_lock(model_type):
            if model_type == "tts":
                self._tts.unload_model(model_id)
            else:
                self._stt.unload_model(model_id)

    def _unload_if_idle(self, model: ModelInfo) -> bool:
        """Unload a model for TTL or LRU eviction unless it is busy; returns whether it did.

        Idle eviction never waits. A running request or load holds its backend's lock
        (faster-whisper's inference lock, a TTS provider's synthesis lock), so waiting for that
        lock would unload a model that is serving, while holding the operation lock that every
        load of the type needs. A busy model is left for a later pass.
        """
        operation_lock = self.operation_lock(model.type)
        if not operation_lock.acquire(blocking=False):
            return False  # an explicit load or unload of this type is running
        try:
            if model.type == "stt":
                unload_if_idle = getattr(self._stt, "unload_if_idle", None)
                if callable(unload_if_idle):
                    return bool(unload_if_idle(model.id))
                self._stt.unload_model(model.id)
                return True
            synthesis_lock = self._tts_synthesis_lock(model.id)
            if not synthesis_lock.acquire(blocking=False):
                return False
            try:
                self._tts.unload_model(model.id)  # takes the same, reentrant, lock
            finally:
                synthesis_lock.release()
            return True
        finally:
            operation_lock.release()

    def _tts_synthesis_lock(self, model_id: str) -> Any:
        """The lock a synthesis or a load of this TTS model holds (TTSRouter's per provider)."""
        lock_for = getattr(self._tts, "_synthesis_lock_for", None)
        get_backend = getattr(self._tts, "get_backend", None)
        if callable(lock_for) and callable(get_backend):
            try:
                return lock_for(get_backend(model_id), model_id)
            except ValueError:
                pass
        return threading.RLock()  # a router without per-provider locks

    def preload(self, model_id: str, model_type: str) -> None:
        """Load a startup model (STT_/TTS_PRELOAD_MODELS) next to the models already loaded.

        An STT model's files are fetched first, outside the STT lock, as a load's are: a first download (Parakeet on
        a GPU: about 7 minutes) holds up no other STT load or unload, and a request for the same model joins it.
        """
        self._require_enabled(model_id, model_type, "preload")
        router = self._tts if model_type == "tts" else self._stt
        future = self.start_preload_fetch(model_id, model_type)
        if future is not None:
            future.result()
        with self.operation_lock(model_type):
            router.load_model(model_id)

    def start_preload_fetch(self, model_id: str, model_type: str) -> concurrent.futures.Future | None:
        """Start, or join, the fetch of a startup STT model's files; None for a TTS model, a loaded model, or a
        backend that fetches only as it loads. The startup preload awaits it without holding a thread."""
        start_fetch = getattr(self._stt, "start_fetch", None)
        if model_type != "stt" or not callable(start_fetch) or self._stt.is_model_loaded(model_id):
            return None
        future = start_fetch(model_id)
        return future if isinstance(future, concurrent.futures.Future) else None

    def warm_default_tts(self) -> bool | None:
        """Load TTS_MODEL, unless it is loaded already or its provider is missing.

        Returns whether it loaded the model, or None while the provider is offline (an isolated
        worker still starting), so the caller can try again later. Run in the background at
        startup, so the first speech request does not wait for the default model to load.
        """
        model_id = settings.tts_model
        if not settings.tts_enabled:
            return False
        try:
            info = self.status(model_id)
        except ModelLifecycleError:
            logger.warning("Not preloading TTS_MODEL %s: no TTS provider serves it", model_id)
            return False
        if info.state == ModelState.PROVIDER_UNAVAILABLE:
            return None
        if info.state not in (ModelState.PROVIDER_INSTALLED, ModelState.DOWNLOADED):
            return False
        # A first start downloads the weights here, before the load takes the TTS lock, so no
        # TTS load or unload waits for the download.
        self._fetch_without_loading(model_id, "tts", info.provider)
        with self.operation_lock("tts"):
            if any(m.model == model_id for m in self._tts.loaded_models()):
                return False  # a request loaded it meanwhile
            logger.info("Preloading default TTS model: %s", model_id)
            self._tts.load_model(model_id)
            return True

    def _hf_cache_roots(self) -> list[Path]:
        roots: list[Path] = []
        if settings.stt_model_dir:
            roots.append(Path(settings.stt_model_dir).expanduser())
        # Where this process downloads Kokoro, Piper, Pocket-TTS and (without STT_MODEL_DIR)
        # faster-whisper models, and the default cache other Hugging Face apps may share.
        for root in (hf_hub_cache_dir(), Path.home() / ".cache" / "huggingface" / "hub"):
            if root not in roots:
                roots.append(root)
        return roots

    def _safe_remove_dir(self, path: Path, allowed_roots: list[Path]) -> bool:
        rp = path.resolve()
        for root in allowed_roots:
            rr = root.resolve()
            # Never the cache root itself: it holds every model and is shared with other HF apps.
            if rr in rp.parents:
                if rp.exists() and rp.is_dir():
                    shutil.rmtree(rp)
                    return True
        return False

    def _candidate_artifact_paths(self, model_id: str, provider: str) -> list[Path]:
        candidates: list[Path] = []
        for root in self._hf_cache_roots():
            if provider == "piper":
                from src.tts.backends.piper_backend import PIPER_MODELS, _hf_path_for_model
                meta = PIPER_MODELS.get(model_id)
                if meta is None:
                    continue  # no such voice, so no files of its own
                filename = _hf_path_for_model(meta["name"])[0]
                snapshots = root / "models--rhasspy--piper-voices" / "snapshots"
                for snapshot in snapshots.glob("*"):
                    candidates.append(snapshot / filename)
                continue
            safe_hf = root / f"models--{model_id.replace('/', '--')}"
            # On Windows a backslash in the ID is a separator too, and could reach another
            # model's directory; the repo directory must be a direct child of the root.
            if safe_hf.parent == root:
                candidates.append(safe_hf)
            for repo in _PROVIDER_REPOS.get(provider, ()):
                candidates.append(root / f"models--{repo.replace('/', '--')}")
        return candidates

    def _delete_piper_artifacts(self, model_id: str) -> list[str]:
        """Remove only this voice's snapshots and unreferenced weight blobs."""
        from src.tts.backends.piper_backend import PIPER_MODELS, _hf_path_for_model

        filenames = _hf_path_for_model(PIPER_MODELS[model_id]["name"])
        removed: list[str] = []
        for root in self._hf_cache_roots():
            repo = root / "models--rhasspy--piper-voices"
            snapshots, blobs = repo / "snapshots", repo / "blobs"
            if repo.is_symlink() or snapshots.is_symlink() or blobs.is_symlink():
                continue
            blob_targets: set[Path] = set()
            paths = [snapshot / filename for snapshot in snapshots.glob("*") for filename in filenames]
            for path in paths:
                if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
                    continue
                if path.is_symlink():
                    # Resolve only the snapshot link. A repository blob can
                    # itself link into the newer cache-wide shared blob store.
                    target = Path(os.path.abspath(path.parent / path.readlink()))
                    if target.parent == blobs.absolute():
                        blob_targets.add(target)
                if path.is_symlink() or path.is_file():
                    path.unlink()
                    removed.append(str(path))
            # A blob can be shared by another model or snapshot revision.
            for path in snapshots.rglob("*"):
                if path.is_symlink():
                    blob_targets.discard(Path(os.path.abspath(path.parent / path.readlink())))
            shared_targets: set[Path] = set()
            shared_store = root / "blobs"
            for target in blob_targets:
                if target.is_symlink():
                    shared = target.resolve()
                    if not shared_store.is_symlink() and shared_store.resolve() in shared.parents:
                        shared_targets.add(shared)
                    target.unlink()
                    removed.append(str(target))
                elif target.is_file():
                    target.unlink()
                    removed.append(str(target))
            if shared_targets:
                # Keep shared weights referenced anywhere else in this cache.
                for path in root.rglob("*"):
                    if path.is_symlink():
                        shared_targets.discard(path.resolve())
                for target in shared_targets:
                    if target.is_file() and not target.is_symlink():
                        target.unlink()
                        removed.append(str(target))
        return removed

    def delete_artifacts(self, model_id: str) -> dict[str, Any]:
        model_type = self._resolve_type(model_id)
        with self.operation_lock(model_type):  # no load may read the files while they go
            return self._delete_artifacts(model_id)

    def _delete_artifacts(self, model_id: str) -> dict[str, Any]:
        self._resolve_type(model_id)
        provider = self._provider_from_model(model_id)
        removed_paths: list[str] = []

        if provider == "piper":
            from src.tts.backends.piper_backend import PIPER_MODELS

            if model_id not in PIPER_MODELS:
                # TTS_MODEL with a voice name Piper does not have, and no Piper backend to ask.
                return {
                    "status": "not_found", "model": model_id, "provider": provider, "deleted_paths": [],
                }
            backend = getattr(self._tts, "_backends", {}).get(provider)
            lock_for = getattr(self._tts, "_synthesis_lock_for", None)
            deletion_lock = lock_for(backend, model_id) if backend and callable(lock_for) else nullcontext()
            with deletion_lock:
                if backend is not None:
                    backend.unload_model(model_id)
                elif self.status(model_id).state == ModelState.LOADED:
                    self.unload(model_id)
                removed_paths = self._delete_piper_artifacts(model_id)
            return {
                "status": "deleted" if removed_paths else "not_found",
                "model": model_id,
                "provider": provider,
                "deleted_paths": removed_paths,
            }

        # unload first if loaded
        try:
            if self.status(model_id).state == ModelState.LOADED:
                self.unload(model_id)
        except Exception:
            pass

        # STT backends may support precise deletion
        deleted = False
        if self._resolve_type(model_id) == "stt" and hasattr(self._stt, "delete_cached_model"):
            try:
                deleted = bool(self._stt.delete_cached_model(model_id))
            except Exception as e:
                if getattr(e, "code", None) == "model_fetching":
                    # A first-use fetch is writing the files: deleting them now would only have it write them
                    # back (and the folder sweep below must not pull them out from under it either).
                    raise ModelLifecycleError(
                        message=str(e),
                        code="model_fetching",
                        model_id=model_id,
                        provider=provider,
                        action="delete",
                        status_code=409,
                    ) from e
                deleted = False

        allowed_roots = self._hf_cache_roots()
        for path in self._candidate_artifact_paths(model_id, provider):
            try:
                if self._safe_remove_dir(path, allowed_roots):
                    removed_paths.append(str(path))
                    deleted = True
            except Exception:
                logger.warning("Failed deleting path %s", path, exc_info=True)

        return {
            "status": "deleted" if deleted else "not_found",
            "model": model_id,
            "provider": provider,
            "deleted_paths": removed_paths,
        }

    def list_loaded(self) -> list[ModelInfo]:
        result: list[ModelInfo] = []
        for m in self._stt.loaded_models():
            result.append(ModelInfo(
                id=m.model, type="stt", provider=m.backend,
                device=m.device, state=ModelState.LOADED,
                loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                is_default=(m.model == settings.stt_model),
                provider_available=True,
            ))
        for m in self._tts.loaded_models():
            result.append(ModelInfo(
                id=m.model, type="tts", provider=m.backend,
                device=m.device, state=ModelState.LOADED,
                loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                is_default=(m.model == settings.tts_model),
                provider_available=True,
            ))
        return result

    def _base_state_for_model(self, model_id: str, provider: str, is_downloaded: bool) -> ModelState:
        if is_downloaded:
            return ModelState.DOWNLOADED
        return ModelState.PROVIDER_INSTALLED

    def list_all(self) -> list[ModelInfo]:
        models: dict[str, ModelInfo] = {}

        for m in self.list_loaded():
            models[m.id] = m

        # Only expose cached STT models that are explicitly known as STT in the
        # curated registry, or the configured STT_MODEL (as status() does). This
        # prevents unrelated HF repos (e.g., TTS assets) from showing up as
        # loadable STT models.
        known_types = {m["id"]: m["type"] for m in get_known_models()}
        for cached in self._stt.list_cached_models():
            mid = cached.get("model", cached.get("id", ""))
            # What the files take on disk; 0 when there are none to measure (the registry's figure stands in).
            measured = cached.get("size_mb") or None
            if mid in models:
                if models[mid].type == "stt" and models[mid].size_mb is None:
                    models[mid].size_mb = measured  # a loaded model's files
                continue
            if not mid:
                continue
            if known_types.get(mid, "stt" if mid == settings.stt_model else None) != "stt":
                continue
            provider = cached.get("backend", self._provider_from_model(mid))
            models[mid] = ModelInfo(
                id=mid, type="stt", provider=provider,
                state=self._base_state_for_model(mid, provider, is_downloaded=True),
                size_mb=measured,
                is_default=(mid == settings.stt_model),
                provider_available=True,
            )

        for km in get_known_models():
            mid = km["id"]
            provider = km["provider"]
            is_tts = km["type"] == "tts"
            provider_registered = _check_provider(km["type"], provider, self._stt, self._tts)
            if km.get("optional_provider") and not provider_registered:
                continue
            advertised: bool | None = None
            manifest_checked = False
            if is_tts and km.get("optional_provider"):
                backend = getattr(self._tts, "_backends", {})[provider]
                check_advertised = getattr(backend, "advertises_model", None)
                if callable(check_advertised):
                    manifest_checked = True
                    advertised = check_advertised(mid)
                if advertised is False:
                    continue
            provider_available = (
                self._provider_runtime_available(provider)
                if is_tts and provider_registered
                else provider_registered
            )
            if manifest_checked and advertised is None:
                provider_available = False
            provider_failure = (
                self._provider_failure(provider) if is_tts and provider_registered and not provider_available else None
            )
            if mid not in models:
                is_dl = False
                if is_tts:
                    is_dl = any(p.exists() for p in self._candidate_artifact_paths(mid, provider))
                state = (
                    ModelState.PROVIDER_MISSING
                    if is_tts and not provider_registered
                    else ModelState.PROVIDER_UNAVAILABLE
                    if is_tts and not provider_available
                    else self._base_state_for_model(mid, provider, is_downloaded=is_dl)
                )
                models[mid] = ModelInfo(
                    id=mid,
                    type=km["type"],
                    provider=provider,
                    state=state,
                    size_mb=self._registry_size_mb(km),
                    is_default=(mid == settings.stt_model or mid == settings.tts_model),
                    description=km.get("description"),
                    source=km.get("source"),
                    model_format=km.get("model_format"),
                    provider_available=provider_available,
                    provider_failure=provider_failure,
                )
            else:
                if not models[mid].size_mb:
                    models[mid].size_mb = self._registry_size_mb(km, models[mid].device)
                if not getattr(models[mid], "description", None) and km.get("description"):
                    models[mid].description = km.get("description")
                if not getattr(models[mid], "source", None) and km.get("source"):
                    models[mid].source = km.get("source")
                if not getattr(models[mid], "model_format", None) and km.get("model_format"):
                    models[mid].model_format = km.get("model_format")
                if is_tts and not provider_registered:
                    models[mid].provider_available = False
                    if models[mid].state != ModelState.LOADED:
                        models[mid].state = ModelState.PROVIDER_MISSING
                elif is_tts and not provider_available:
                    models[mid].provider_available = False
                    models[mid].provider_failure = provider_failure
                    if models[mid].state != ModelState.LOADED:
                        models[mid].state = ModelState.PROVIDER_UNAVAILABLE

        if settings.stt_model not in models:
            stt_provider = self._provider_from_model(settings.stt_model)
            models[settings.stt_model] = ModelInfo(
                id=settings.stt_model, type="stt", provider=stt_provider,
                state=self._base_state_for_model(settings.stt_model, stt_provider, is_downloaded=False),
                is_default=True,
                provider_available=True,
            )
        # A TTS_MODEL its provider's backend rejects is unknown to status() and every operation.
        if settings.tts_model not in models and not self._tts_backend_rejects(settings.tts_model):
            tts_provider = self._provider_from_model(settings.tts_model)
            provider_registered = _check_provider("tts", tts_provider, self._stt, self._tts)
            known_default = get_known_model(settings.tts_model)
            optional = bool(known_default and known_default.get("optional_provider"))
            advertised: bool | None = None
            manifest_checked = False
            if optional and provider_registered:
                backend = getattr(self._tts, "_backends", {})[tts_provider]
                check_advertised = getattr(backend, "advertises_model", None)
                if callable(check_advertised):
                    manifest_checked = True
                    advertised = check_advertised(settings.tts_model)
            # The same runtime check as status() and the registry rows: a worker that doesn't answer, or answers
            # that it can't run, makes the default unavailable, with the worker's reason.
            provider_available = (
                provider_registered
                and (not manifest_checked or advertised is True)
                and self._provider_runtime_available(tts_provider)
            )
            models[settings.tts_model] = ModelInfo(
                id=settings.tts_model, type="tts", provider=tts_provider,
                state=(
                    ModelState.PROVIDER_MISSING
                    if not provider_registered
                    else ModelState.PROVIDER_UNAVAILABLE
                    if not provider_available
                    else self._base_state_for_model(settings.tts_model, tts_provider, is_downloaded=False)
                ),
                is_default=True,
                provider_available=provider_available,
                provider_failure=(
                    self._provider_failure(tts_provider) if provider_registered and not provider_available else None
                ),
            )

        return [self._with_notice(info) for info in models.values()]

    def _with_notice(self, info: ModelInfo) -> ModelInfo:
        if info.id in self._displaced and info.state != ModelState.LOADED:
            info.notice = self.displaced_notice(info.id)
        return info

    def status(self, model_id: str) -> ModelInfo:
        return self._with_notice(self._status(model_id))

    def _status(self, model_id: str) -> ModelInfo:
        for m in self.list_loaded():
            if m.id == model_id:
                return m

        for cached in self._stt.list_cached_models():
            mid = cached.get("model", cached.get("id", ""))
            if mid == model_id:
                known = get_known_model(model_id)
                if (known and known.get("type") != "stt") or (not known and model_id != settings.stt_model):
                    continue
                provider = cached.get("backend", self._provider_from_model(model_id))
                return ModelInfo(
                    id=model_id, type="stt", provider=provider,
                    state=self._base_state_for_model(model_id, provider, is_downloaded=True),
                    size_mb=cached.get("size_mb"),
                    is_default=(model_id == settings.stt_model),
                    provider_available=True,
                )

        model_type = self._resolve_type(model_id)
        provider = self.resolve_provider(model_id)
        is_dl = False
        provider_available = True
        if model_type == "tts":
            is_dl = any(p.exists() for p in self._candidate_artifact_paths(model_id, provider))
            provider_registered = _check_provider("tts", provider, self._stt, self._tts)
            provider_available = (
                self._provider_runtime_available(provider) if provider_registered else False
            )
            known = get_known_model(model_id)
            if provider_registered and known and known.get("optional_provider"):
                backend = getattr(self._tts, "_backends", {})[provider]
                check_advertised = getattr(backend, "advertises_model", None)
                if callable(check_advertised) and check_advertised(model_id) is not True:
                    provider_available = False
        else:
            provider_registered = True

        state = (
            ModelState.PROVIDER_MISSING
            if model_type == "tts" and not provider_registered
            else ModelState.PROVIDER_UNAVAILABLE
            if model_type == "tts" and not provider_available
            else self._base_state_for_model(model_id, provider, is_downloaded=is_dl)
        )

        return ModelInfo(
            id=model_id, type=model_type, provider=provider,
            state=state,
            is_default=(model_id == settings.stt_model or model_id == settings.tts_model),
            provider_available=provider_available,
            provider_failure=(
                self._provider_failure(provider)
                if model_type == "tts" and provider_registered and not provider_available
                else None
            ),
        )

    @staticmethod
    def _pinned_models() -> set[str]:
        """The defaults that TTL and LRU eviction leave loaded (OS_PIN_DEFAULT_MODELS)."""
        return pinned_default_models()

    def evict_lru(self) -> bool:
        """Unload the least recently used idle model; False when there is none to unload.

        The pinned defaults are not counted against OS_MAX_LOADED_MODELS: they stay loaded on
        top of it, so the cap bounds the other models and the defaults never crowd out the model
        a client is using. A busy model is skipped (see _unload_if_idle).
        """
        pinned = self._pinned_models()
        candidates = [m for m in self.list_loaded() if m.id not in pinned]
        max_loaded = settings.os_max_loaded_models
        if 0 < max_loaded and len(candidates) <= max_loaded:
            return False
        candidates.sort(
            key=lambda m: m.last_used_at if m.last_used_at is not None else (m.loaded_at or 0)
        )
        for model in candidates:
            if self._unload_if_idle(model):
                logger.info("LRU eviction: unloaded %s", model.id)
                return True
        return False

    def check_ttl(self) -> None:
        ttl = settings.os_model_ttl
        if ttl <= 0:
            return
        pinned = self._pinned_models()
        now = time.time()
        for m in self.list_loaded():
            if m.id in pinned:
                continue
            last_used = m.last_used_at or m.loaded_at or now
            if (now - last_used) > ttl:
                if self._unload_if_idle(m):
                    logger.info("TTL eviction: unloaded %s (idle %.0fs)", m.id, now - last_used)
                else:
                    logger.debug("TTL eviction: %s is busy, retrying on the next pass", m.id)
