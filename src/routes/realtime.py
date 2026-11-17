"""Realtime audio routes."""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, WebSocket

from src.middleware import (
    WS_CLOSE_FEATURE_OFF,
    WS_CLOSE_KEY,
    WS_CLOSE_ORIGIN,
    WS_KEY_REFUSED,
    WS_ORIGIN_REFUSED,
    refuse_websocket,
    verify_ws_api_key,
    verify_ws_origin,
)


def create_router(
    *,
    get_settings: Callable,
    get_tts_router: Callable,
    get_pronunciation_dict: Callable | None = None,
    get_voice_library: Callable | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/v1/realtime")
    async def ws_realtime(
        websocket: WebSocket,
        model: str | None = None,
    ):
        # A refused socket is accepted and then closed, so a browser gets the code and reason, not 1006. The
        # middleware has already refused a foreign Origin the same way.
        settings = get_settings()
        if not settings.os_realtime_enabled:
            await refuse_websocket(websocket, WS_CLOSE_FEATURE_OFF, "Realtime API is disabled")
            return
        if not verify_ws_origin(websocket):
            await refuse_websocket(websocket, WS_CLOSE_ORIGIN, WS_ORIGIN_REFUSED)
            return
        if not verify_ws_api_key(websocket):
            await refuse_websocket(websocket, WS_CLOSE_KEY, WS_KEY_REFUSED)
            return
        from src import assistant
        from src.realtime.server import realtime_endpoint

        pronunciation_dict = get_pronunciation_dict() if get_pronunciation_dict is not None else None
        await realtime_endpoint(
            websocket,
            tts_router=get_tts_router(),
            model=model or "",
            pronounce_fn=pronunciation_dict.apply if pronunciation_dict is not None else None,
            # The shared AI client (OS_ASSISTANT_URL), or None: each turn can then be answered by the AI.
            assistant=assistant.get_client(),
            voice_library=get_voice_library() if get_voice_library is not None else None,
        )

    return router
