"""Unified Model Manager — wraps STT and TTS routers with a single interface."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import hf_hub_cache_dir, settings
from src.model_registry import get_known_model, get_known_models
from src.tts.external import ExternalProviderError

logger = logging.getLogger(__name__)

# Hugging Face repositories an in-process provider downloads, when not models--<model id>.
# pocket-tts fetches kyutai/pocket-tts once its terms are accepted, else the variant
# without voice cloning, which also holds its predefined voices.
_PROVIDER_REPOS: dict[str, tuple[str, ...]] = {
    "kokoro": ("hexgrad/Kokoro-82M", "hexgrad/Kokoro-82M-v1.1-zh"),
    "pocket-tts": ("kyutai/pocket-tts", "kyutai/pocket-tts-without-voice-cloning"),
}


def _check_provider(model_type: str, provider: str, stt_router: Any, tts_router: Any) -> bool:
    """Return whether the provider backend is currently registered/available."""
    if model_type == "tts":
        return provider in getattr(tts_router, "_backends", {})

    stt_backends = getattr(stt_router, "_backends", None)
    if not stt_backends:
        return True
    return provider in stt_backends


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
            "provider_available": self.provider_available,
            "provider_failure": dict(self.provider_failure) if self.provider_failure else None,
        }
        if self.description:
            d["description"] = self.description
        if self.source:
            d["source"] = self.source
        if self.model_format:
            d["model_format"] = self.model_format
        return d


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

    def resolve_provider(self, model_id: str) -> str:
        return self._provider_from_model(model_id)

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

    def load(self, model_id: str, device: str | None = None, _evict_others: bool = True) -> ModelInfo:
        model_type = self._resolve_type(model_id)
        self._require_enabled(model_id, model_type, "load")
        if model_type == "stt":
            self._fetch_stt_before_load(model_id)
        with self.operation_lock(model_type):
            return self._load(model_id, model_type, _evict_others=_evict_others)

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
            message=f"Failed to load model '{model_id}': {error}",
            code="load_failed",
            model_id=model_id,
            provider=self._provider_from_model(model_id),
            action="load",
            details={"exception": type(error).__name__},
            status_code=500,
        )

    def _load(self, model_id: str, model_type: str, *, _evict_others: bool) -> ModelInfo:
        provider = self._require_provider(model_id, "load")
        self._check_loadable(model_id, model_type, provider, "load")

        # Only evict when explicitly loading (not when called from download/prefetch)
        evicted_models: list[str] = []
        if _evict_others:
            for m in self.list_loaded():
                if m.type == model_type and m.id != model_id:
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
                message=f"Failed to load model '{model_id}': {e}",
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
        download from the Hugging Face hub into the cache the artifact scan reads. Pocket-TTS
        picks its repository while it loads, and the isolated workers have no download-only
        call, so those still load once and unload again.
        """
        if model_type == "stt":
            self._check_loadable(model_id, model_type, provider, "download")
            download_model = getattr(self._stt, "download_model", None)
            return callable(download_model) and bool(download_model(model_id))

        backend = getattr(self._tts, "_backends", {}).get(provider)
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
        """Load a startup model (STT_/TTS_PRELOAD_MODELS) next to the models already loaded."""
        self._require_enabled(model_id, model_type, "preload")
        router = self._tts if model_type == "tts" else self._stt
        with self.operation_lock(model_type):
            router.load_model(model_id)

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
            except Exception:
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
            if not mid or mid in models:
                continue
            if known_types.get(mid, "stt" if mid == settings.stt_model else None) != "stt":
                continue
            provider = cached.get("backend", self._provider_from_model(mid))
            models[mid] = ModelInfo(
                id=mid, type="stt", provider=provider,
                state=self._base_state_for_model(mid, provider, is_downloaded=True),
                size_mb=cached.get("size_mb"),
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
                    size_mb=km.get("size_mb"),
                    is_default=(mid == settings.stt_model or mid == settings.tts_model),
                    description=km.get("description"),
                    source=km.get("source"),
                    model_format=km.get("model_format"),
                    provider_available=provider_available,
                    provider_failure=provider_failure,
                )
            else:
                if models[mid].size_mb is None and km.get("size_mb"):
                    models[mid].size_mb = km["size_mb"]
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

        return list(models.values())

    def status(self, model_id: str) -> ModelInfo:
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
        if not settings.os_pin_default_models:
            return set()
        return {settings.stt_model, settings.tts_model}

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
