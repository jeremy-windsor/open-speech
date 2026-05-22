"""Model lifecycle manager — TTL-based auto-eviction and max-models LRU."""

from __future__ import annotations

import asyncio
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Background task that evicts idle models based on TTL and max-loaded limits."""

    def __init__(self, model_manager) -> None:
        self._model_manager = model_manager
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Model lifecycle started (ttl=%ds, max_loaded=%d)",
            settings.os_model_ttl,
            settings.os_max_loaded_models,
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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._evict_sync)

    def _evict_sync(self) -> None:
        self._model_manager.check_ttl()

        max_loaded = settings.os_max_loaded_models
        if max_loaded <= 0:
            return

        while len(self._model_manager.list_loaded()) > max_loaded:
            before = len(self._model_manager.list_loaded())
            self._model_manager.evict_lru()
            if len(self._model_manager.list_loaded()) >= before:
                logger.warning(
                    "LRU eviction did not reduce loaded model count; stopping lifecycle pass"
                )
                break
