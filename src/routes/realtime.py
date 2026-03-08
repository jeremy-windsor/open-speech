"""Realtime audio routes."""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, WebSocket

from src.middleware import verify_ws_api_key, verify_ws_origin


def create_router(*, get_settings: Callable, get_tts_router: Callable) -> APIRouter:
    router = APIRouter()

    @router.websocket("/v1/realtime")
    async def ws_realtime(
        websocket: WebSocket,
        model: str | None = None,
    ):
        settings = get_settings()
        if not settings.os_realtime_enabled:
            await websocket.close(code=4004, reason="Realtime API is disabled")
            return
        if not verify_ws_origin(websocket):
            await websocket.close(code=1008, reason="Origin not allowed")
            return
        if not verify_ws_api_key(websocket):
            await websocket.close(code=4001, reason="Invalid or missing API key")
            return
        from src.realtime.server import realtime_endpoint

        await realtime_endpoint(websocket, tts_router=get_tts_router(), model=model or "")

    return router
