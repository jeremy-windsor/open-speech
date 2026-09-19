"""Unified Model Manager — wraps STT and TTS routers with a single interface."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import settings
from src.model_registry import get_known_model, get_known_models
from src.tts.external import ExternalProviderError

logger = logging.getLogger(__name__)


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

    def _resolve_type(self, model_id: str) -> str:
        known = get_known_model(model_id)
        if known:
            return known["type"]
        tts_prefixes = ("kokoro", "piper/", "piper-", "pocket-tts")
        if model_id in getattr(self._tts, "_backends", {}) or any(model_id.startswith(p) for p in tts_prefixes):
            return "tts"
        for m in self._tts.loaded_models():
            if m.model == model_id:
                return "tts"
        return "stt"

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

    def resolve_provider(self, model_id: str) -> str:
        return self._provider_from_model(model_id)

    def _require_provider(self, model_id: str, action: str) -> str:
        provider = self._provider_from_model(model_id)
        return provider

    def load(self, model_id: str, device: str | None = None, _evict_others: bool = True) -> ModelInfo:
        model_type = self._resolve_type(model_id)
        provider = self._require_provider(model_id, "load")

        if not _check_provider(model_type, provider, self._stt, self._tts):
            raise ModelLifecycleError(
                message=(
                    f"Provider '{provider}' is not installed for model '{model_id}'. "
                    f"Rebuild with BAKED_PROVIDERS including '{provider}'."
                ),
                code="provider_missing",
                model_id=model_id,
                provider=provider,
                action="load",
            )
        if model_type == "tts" and not self._provider_runtime_available(provider):
            raise ModelLifecycleError(
                message=f"Provider '{provider}' is configured but unavailable.",
                code="provider_unavailable",
                model_id=model_id,
                provider=provider,
                action="load",
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
                    action="load",
                ) from e
            except ValueError as e:
                raise ModelLifecycleError(
                    message=str(e),
                    code="unknown_model",
                    model_id=model_id,
                    provider=provider,
                    action="load",
                ) from e

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
                        logger.warning("Failed to auto-unload %s model %s: %s", model_type.upper(), m.id, e)

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
        # Manual download path: load to trigger weights fetch, then unload if not already loaded.
        was_loaded = False
        try:
            if self._resolve_type(model_id) == "tts":
                was_loaded = self._tts.is_model_loaded(model_id)
            else:
                was_loaded = self._stt.is_model_loaded(model_id)
        except Exception:
            was_loaded = False

        self.load(model_id, _evict_others=False)
        if not was_loaded:
            self.unload(model_id)
        info = self.status(model_id)
        info.provider = provider
        return info

    def unload(self, model_id: str) -> None:
        model_type = self._resolve_type(model_id)
        if model_type == "tts":
            self._tts.unload_model(model_id)
        else:
            self._stt.unload_model(model_id)

    def _hf_cache_roots(self) -> list[Path]:
        roots: list[Path] = []
        if settings.stt_model_dir:
            roots.append(Path(settings.stt_model_dir).expanduser())
        env_roots = [
            os.environ.get("HF_HUB_CACHE"),
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ]
        for root in env_roots:
            if root:
                p = Path(root).expanduser()
                if p not in roots:
                    roots.append(p)
        return roots

    def _safe_remove_dir(self, path: Path, allowed_roots: list[Path]) -> bool:
        rp = path.resolve()
        for root in allowed_roots:
            rr = root.resolve()
            if rp == rr or rr in rp.parents:
                if rp.exists() and rp.is_dir():
                    shutil.rmtree(rp)
                    return True
        return False

    def _candidate_artifact_paths(self, model_id: str, provider: str) -> list[Path]:
        candidates: list[Path] = []
        for root in self._hf_cache_roots():
            safe_hf = root / f"models--{model_id.replace('/', '--')}"
            candidates.append(safe_hf)
            if provider == "kokoro":
                candidates.append(root / "models--hexgrad--Kokoro-82M")
                candidates.append(root / "models--hexgrad--Kokoro-82M-v1.1-zh")
        return candidates

    def delete_artifacts(self, model_id: str) -> dict[str, Any]:
        provider = self._provider_from_model(model_id)
        removed_paths: list[str] = []

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
        # curated registry. This prevents unrelated HF repos (e.g., TTS assets)
        # from showing up as loadable STT models.
        known_types = {m["id"]: m["type"] for m in get_known_models()}
        for cached in self._stt.list_cached_models():
            mid = cached.get("model", cached.get("id", ""))
            if not mid or mid in models:
                continue
            if known_types.get(mid) != "stt":
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
        if settings.tts_model not in models:
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
            provider_available = provider_registered and (not manifest_checked or advertised is True)
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
                if known and known.get("type") != "stt":
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
        )

    def evict_lru(self) -> None:
        loaded = self.list_loaded()
        if not loaded:
            return
        loaded.sort(
            key=lambda m: m.last_used_at if m.last_used_at is not None else (m.loaded_at or 0)
        )
        oldest = loaded[0]
        logger.info("LRU eviction: unloading %s", oldest.id)
        self.unload(oldest.id)

    def check_ttl(self) -> None:
        ttl = settings.os_model_ttl
        if ttl <= 0:
            return
        now = time.time()
        for m in self.list_loaded():
            last_used = m.last_used_at or m.loaded_at or now
            if (now - last_used) > ttl:
                logger.info("TTL eviction: unloading %s (idle %.0fs)", m.id, now - last_used)
                self.unload(m.id)
