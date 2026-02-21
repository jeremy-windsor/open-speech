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

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    AVAILABLE = "available"
    PROVIDER_MISSING = "provider_missing"
    PROVIDER_INSTALLED = "provider_installed"
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
        return d


class ModelManager:
    """Unified model lifecycle for STT and TTS."""

    def __init__(self, stt_router: Any, tts_router: Any) -> None:
        self._stt = stt_router
        self._tts = tts_router

    def _resolve_type(self, model_id: str) -> str:
        tts_prefixes = ("kokoro", "piper/", "piper-", "pocket-tts", "qwen3-tts")
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
        if model_id.startswith("qwen3-tts"):
            return "qwen3"
        if model_id == "kokoro":
            return "kokoro"
        return "faster-whisper"

    def resolve_provider(self, model_id: str) -> str:
        return self._provider_from_model(model_id)

    def _require_provider(self, model_id: str, action: str) -> str:
        provider = self._provider_from_model(model_id)
        return provider

    def load(self, model_id: str, device: str | None = None) -> ModelInfo:
        model_type = self._resolve_type(model_id)
        provider = self._require_provider(model_id, "load")
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
            raise
        except Exception as e:
            raise ModelLifecycleError(
                message=f"Failed to load model '{model_id}': {e}",
                code="load_failed",
                model_id=model_id,
                provider=provider,
                action="load",
                details={"exception": type(e).__name__},
            ) from e

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

        self.load(model_id)
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
            if provider == "qwen3":
                for suffix in ["1.7B-Base", "1.7B-CustomVoice", "1.7B-VoiceDesign", "0.6B-Base", "0.6B-CustomVoice", "Tokenizer-12Hz"]:
                    candidates.append(root / f"models--Qwen--Qwen3-TTS-{suffix}")
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
            if mid not in models:
                is_dl = False
                if km["type"] == "tts" and True:
                    is_dl = any(p.exists() for p in self._candidate_artifact_paths(mid, provider))
                models[mid] = ModelInfo(
                    id=mid,
                    type=km["type"],
                    provider=provider,
                    state=self._base_state_for_model(mid, provider, is_downloaded=is_dl),
                    size_mb=km.get("size_mb"),
                    is_default=(mid == settings.stt_model or mid == settings.tts_model),
                    description=km.get("description"),
                    provider_available=True,
                )
            else:
                if models[mid].size_mb is None and km.get("size_mb"):
                    models[mid].size_mb = km["size_mb"]
                if not getattr(models[mid], "description", None) and km.get("description"):
                    models[mid].description = km.get("description")

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
            models[settings.tts_model] = ModelInfo(
                id=settings.tts_model, type="tts", provider=tts_provider,
                state=self._base_state_for_model(settings.tts_model, tts_provider, is_downloaded=False),
                is_default=True,
                provider_available=True,
            )

        return list(models.values())

    def status(self, model_id: str) -> ModelInfo:
        for m in self.list_loaded():
            if m.id == model_id:
                return m

        for cached in self._stt.list_cached_models():
            mid = cached.get("model", cached.get("id", ""))
            if mid == model_id:
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
        if model_type == "tts" and True:
            is_dl = any(p.exists() for p in self._candidate_artifact_paths(model_id, provider))
        return ModelInfo(
            id=model_id, type=model_type, provider=provider,
            state=self._base_state_for_model(model_id, provider, is_downloaded=is_dl),
            is_default=(model_id == settings.stt_model or model_id == settings.tts_model),
            provider_available=True,
        )

    def evict_lru(self) -> None:
        loaded = self.list_loaded()
        non_default = [m for m in loaded if not m.is_default]
        if not non_default:
            return
        non_default.sort(key=lambda m: m.last_used_at or 0)
        oldest = non_default[0]
        logger.info("LRU eviction: unloading %s", oldest.id)
        self.unload(oldest.id)

    def check_ttl(self) -> None:
        ttl = settings.os_model_ttl
        if ttl <= 0:
            return
        now = time.time()
        for m in self.list_loaded():
            if m.is_default:
                continue
            last_used = m.last_used_at or m.loaded_at or now
            if (now - last_used) > ttl:
                logger.info("TTL eviction: unloading %s (idle %.0fs)", m.id, now - last_used)
                self.unload(m.id)
