"""Model lifecycle manager — TTL-based auto-eviction and max-models LRU."""

from __future__ import annotations

import asyncio
import logging
import time

from src.config import settings

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Background task that evicts idle models based on TTL and max-loaded limits."""

    def __init__(self, router) -> None:
        self._router = router
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Model lifecycle started (ttl=%ds, max_loaded=%d)",
            settings.stt_model_ttl,
            settings.stt_max_loaded_models,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                await self._evict()
            except Exception:
                logger.exception("Lifecycle eviction error")

    async def _evict(self) -> None:
        backend = self._router._default_backend
        default_model = settings.stt_default_model
        ttl = settings.stt_model_ttl
        max_loaded = settings.stt_max_loaded_models
        now = time.time()

        # TTL eviction
        if ttl > 0:
            to_evict = [
                mid for mid in list(backend._models)
                if mid != default_model
                and (now - backend._last_used.get(mid, now)) > ttl
            ]
            for mid in to_evict:
                logger.info("TTL eviction: unloading %s (idle %.0fs)", mid, now - backend._last_used.get(mid, 0))
                async with self._router._lock:
                    backend.unload_model(mid)

        # Max loaded eviction (LRU)
        if max_loaded > 0:
            loaded = [mid for mid in backend._models if mid != default_model]
            excess = len(backend._models) - max_loaded
            if excess > 0:
                # Sort non-default by last_used ascending (oldest first)
                loaded.sort(key=lambda m: backend._last_used.get(m, 0))
                for mid in loaded[:excess]:
                    logger.info("LRU eviction: unloading %s (max_loaded=%d)", mid, max_loaded)
                    async with self._router._lock:
                        backend.unload_model(mid)
