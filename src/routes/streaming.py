"""Streaming STT routes."""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, WebSocket
from fastapi.responses import JSONResponse

from src.middleware import verify_ws_api_key, verify_ws_origin
from src.streaming import streaming_endpoint


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/audio/stream")
    async def ws_stream_info():
        return JSONResponse(
            status_code=426,
            content={
                "error": {
                    "message": (
                        "/v1/audio/stream is a WebSocket endpoint. "
                        "Connect with ws:// or wss:// using a WebSocket client."
                    ),
                    "code": "websocket_upgrade_required",
                }
            },
            headers={"Upgrade": "websocket"},
        )

    @router.websocket("/v1/audio/stream")
    async def ws_stream(
        websocket: WebSocket,
        model: str | None = None,
        language: str | None = None,
        sample_rate: int = 16000,
        encoding: str = "pcm_s16le",
        interim_results: bool = True,
        endpointing: int = 300,
        vad: bool | None = None,
    ):
        if not verify_ws_origin(websocket):
            await websocket.close(code=1008, reason="Origin not allowed")
            return
        if not verify_ws_api_key(websocket):
            await websocket.close(code=4001, reason="Invalid or missing API key")
            return
        await streaming_endpoint(
            websocket,
            model=model,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            interim_results=interim_results,
            endpointing=endpointing,
            vad=vad,
        )

    return router
