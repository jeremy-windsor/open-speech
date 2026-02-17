"""Unified Model Manager — wraps STT and TTS routers with a single interface."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config import settings
from src.model_registry import get_known_models

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    AVAILABLE = "available"
    DOWNLOADED = "downloaded"
    LOADED = "loaded"


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


def _is_provider_available(provider: str) -> bool:
    """Check if a provider's required package is importable."""
    _provider_imports = {
        "moonshine": "moonshine_onnx",
        "piper": "piper",
        "fish-speech": "fish_speech",
    }
    pkg = _provider_imports.get(provider)
    if pkg is None:
        return True  # no special import needed
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


# Cache provider availability checks
_provider_avail_cache: dict[str, bool] = {}


def _check_provider(provider: str) -> bool:
    if provider not in _provider_avail_cache:
        _provider_avail_cache[provider] = _is_provider_available(provider)
    return _provider_avail_cache[provider]


class ModelManager:
    """Unified model lifecycle for STT and TTS."""

    def __init__(self, stt_router: Any, tts_router: Any) -> None:
        self._stt = stt_router
        self._tts = tts_router

    def _resolve_type(self, model_id: str) -> str:
        """Determine if a model is STT or TTS."""
        # TTS models
        tts_prefixes = ("kokoro", "piper/", "piper-")
        if model_id in self._tts._backends or any(model_id.startswith(p) for p in tts_prefixes):
            return "tts"
        # Check if loaded as TTS
        for m in self._tts.loaded_models():
            if m.model == model_id:
                return "tts"
        return "stt"

    def _get_stt_provider(self, model_id: str) -> str:
        """Get STT provider name for model."""
        backend = self._stt.get_backend(model_id)
        return getattr(backend, "name", "unknown")

    def _get_tts_provider(self, model_id: str) -> str:
        """Get TTS provider name for model."""
        backend = self._tts.get_backend(model_id)
        return getattr(backend, "name", "unknown")

    def resolve_provider(self, model_id: str) -> str:
        model_type = self._resolve_type(model_id)
        if model_type == "tts":
            return self._get_tts_provider(model_id)
        return self._get_stt_provider(model_id)

    def load(self, model_id: str, device: str | None = None) -> ModelInfo:
        """Load a model into memory."""
        model_type = self._resolve_type(model_id)
        if model_type == "tts":
            self._tts.load_model(model_id)
            # Find loaded info
            for m in self._tts.loaded_models():
                if m.model == model_id:
                    return ModelInfo(
                        id=model_id, type="tts", provider=m.backend,
                        device=m.device, state=ModelState.LOADED,
                        loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                        is_default=(model_id == settings.tts_model),
                    )
            return ModelInfo(id=model_id, type="tts", provider=self._get_tts_provider(model_id),
                           state=ModelState.LOADED, is_default=(model_id == settings.tts_model))
        else:
            self._stt.load_model(model_id)
            for m in self._stt.loaded_models():
                if m.model == model_id:
                    return ModelInfo(
                        id=model_id, type="stt", provider=m.backend,
                        device=m.device, state=ModelState.LOADED,
                        loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                        is_default=(model_id == settings.stt_model),
                    )
            return ModelInfo(id=model_id, type="stt", provider=self._get_stt_provider(model_id),
                           state=ModelState.LOADED, is_default=(model_id == settings.stt_model))

    def unload(self, model_id: str) -> None:
        """Unload a model from memory."""
        model_type = self._resolve_type(model_id)
        if model_type == "tts":
            self._tts.unload_model(model_id)
        else:
            self._stt.unload_model(model_id)

    def list_loaded(self) -> list[ModelInfo]:
        """List all currently loaded models."""
        result: list[ModelInfo] = []
        for m in self._stt.loaded_models():
            result.append(ModelInfo(
                id=m.model, type="stt", provider=m.backend,
                device=m.device, state=ModelState.LOADED,
                loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                is_default=(m.model == settings.stt_model),
            ))
        for m in self._tts.loaded_models():
            result.append(ModelInfo(
                id=m.model, type="tts", provider=m.backend,
                device=m.device, state=ModelState.LOADED,
                loaded_at=m.loaded_at, last_used_at=m.last_used_at,
                is_default=(m.model == settings.tts_model),
            ))
        return result

    def list_all(self) -> list[ModelInfo]:
        """List all models: loaded + cached/downloaded + defaults."""
        models: dict[str, ModelInfo] = {}

        # Loaded models
        for m in self.list_loaded():
            models[m.id] = m

        # Known TTS HuggingFace repos that should NOT appear as STT
        _tts_hf_repos = {"hexgrad/Kokoro-82M", "hexgrad/Kokoro-82M-v1.1-zh"}

        # Cached STT models (on disk but maybe not loaded)
        for cached in self._stt.list_cached_models():
            mid = cached.get("model", cached.get("id", ""))
            if mid and mid not in models and mid not in _tts_hf_repos:
                models[mid] = ModelInfo(
                    id=mid, type="stt",
                    provider=cached.get("backend", "faster-whisper"),
                    state=ModelState.DOWNLOADED,
                    size_mb=cached.get("size_mb"),
                    is_default=(mid == settings.stt_model),
                )

        # Merge curated registry (available models not yet seen)
        for km in get_known_models():
            mid = km["id"]
            if mid not in models:
                models[mid] = ModelInfo(
                    id=mid,
                    type=km["type"],
                    provider=km["provider"],
                    state=ModelState.AVAILABLE,
                    size_mb=km.get("size_mb"),
                    is_default=(mid == settings.stt_model or mid == settings.tts_model),
                    description=km.get("description"),
                    provider_available=_check_provider(km["provider"]),
                )
            else:
                # Enrich existing entries with registry metadata
                if models[mid].size_mb is None and km.get("size_mb"):
                    models[mid].size_mb = km["size_mb"]
                if not getattr(models[mid], "description", None) and km.get("description"):
                    models[mid].description = km.get("description")

        # Ensure defaults are always listed
        if settings.stt_model not in models:
            models[settings.stt_model] = ModelInfo(
                id=settings.stt_model, type="stt",
                provider=self._get_stt_provider(settings.stt_model),
                state=ModelState.AVAILABLE,
                is_default=True,
            )
        if settings.tts_model not in models:
            models[settings.tts_model] = ModelInfo(
                id=settings.tts_model, type="tts",
                provider=self._get_tts_provider(settings.tts_model),
                state=ModelState.AVAILABLE,
                is_default=True,
            )

        return list(models.values())

    def status(self, model_id: str) -> ModelInfo:
        """Get status of a specific model."""
        # Check loaded
        for m in self.list_loaded():
            if m.id == model_id:
                return m
        # Check cached
        for cached in self._stt.list_cached_models():
            mid = cached.get("model", cached.get("id", ""))
            if mid == model_id:
                return ModelInfo(
                    id=model_id, type="stt",
                    provider=cached.get("backend", "faster-whisper"),
                    state=ModelState.DOWNLOADED,
                    size_mb=cached.get("size_mb"),
                    is_default=(model_id == settings.stt_model),
                )
        # Available
        model_type = self._resolve_type(model_id)
        provider = self.resolve_provider(model_id)
        return ModelInfo(
            id=model_id, type=model_type, provider=provider,
            state=ModelState.AVAILABLE,
            is_default=(model_id == settings.stt_model or model_id == settings.tts_model),
        )

    def evict_lru(self) -> None:
        """Evict least recently used non-default model."""
        loaded = self.list_loaded()
        non_default = [m for m in loaded if not m.is_default]
        if not non_default:
            return
        # Sort by last_used_at ascending
        non_default.sort(key=lambda m: m.last_used_at or 0)
        oldest = non_default[0]
        logger.info("LRU eviction: unloading %s", oldest.id)
        self.unload(oldest.id)

    def check_ttl(self) -> None:
        """Unload models that have been idle beyond TTL."""
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
