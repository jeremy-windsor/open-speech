"""Streaming STT routes."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from src.middleware import (
    WS_CLOSE_KEY,
    WS_CLOSE_ORIGIN,
    WS_KEY_REFUSED,
    WS_ORIGIN_REFUSED,
    refuse_websocket,
    verify_ws_api_key,
    verify_ws_origin,
)
from src.streaming import (
    MAX_SAMPLE_RATE,
    MAX_UTTERANCE_SECONDS,
    MIN_SAMPLE_RATE,
    streaming_endpoint,
)

_WHOLE_NUMBER = TypeAdapter(int)
_TRUE_OR_FALSE = TypeAdapter(bool)
# An utterance is finalized at the cap anyway, so a longer silence could never end one. Past the range, a huge value
# would raise OverflowError in StreamingSession after accept().
_MAX_ENDPOINTING_MS = MAX_UTTERANCE_SECONDS * 1000


def _query_values(**raw: str | None) -> tuple[dict, str | None]:
    """The typed query parameters read as FastAPI read them (the same pydantic types, so "16000.0" and "on" still
    work), and the reason to refuse the first one that can't be read, or None. The route reads them itself, after
    the Origin and key checks: a typed parameter makes FastAPI close before accept(), with a list of errors as the
    reason, and a browser sees only 1006."""
    rules = {
        "sample_rate": (_WHOLE_NUMBER, 16000, f"Invalid sample_rate: must be {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}"),
        "interim_results": (_TRUE_OR_FALSE, True, "Invalid interim_results: must be true or false"),
        "endpointing": (
            _WHOLE_NUMBER, None, f"Invalid endpointing: must be a whole number of milliseconds, 0-{_MAX_ENDPOINTING_MS}",
        ),
        "vad": (_TRUE_OR_FALSE, None, "Invalid vad: must be true or false"),
    }
    ranges = {"endpointing": (0, _MAX_ENDPOINTING_MS)}
    values = {}
    for name, (adapter, default, reason) in rules.items():
        try:
            value = default if raw[name] is None else adapter.validate_python(raw[name])
        except ValidationError:
            return values, reason
        if value is not None and name in ranges and not ranges[name][0] <= value <= ranges[name][1]:
            return values, reason
        values[name] = value
    return values, None


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
        sample_rate: str | None = None,
        encoding: str = "pcm_s16le",
        interim_results: str | None = None,
        endpointing: str | None = None,
        vad: str | None = None,
    ):
        # Accepted and then closed, so a browser gets the code and reason, not 1006.
        if not verify_ws_origin(websocket):
            await refuse_websocket(websocket, WS_CLOSE_ORIGIN, WS_ORIGIN_REFUSED)
            return
        if not verify_ws_api_key(websocket):
            await refuse_websocket(websocket, WS_CLOSE_KEY, WS_KEY_REFUSED)
            return
        values, refused = _query_values(
            sample_rate=sample_rate, interim_results=interim_results, endpointing=endpointing, vad=vad,
        )
        if refused is not None:
            await refuse_websocket(websocket, WS_CLOSE_ORIGIN, refused)
            return
        await streaming_endpoint(
            websocket,
            model=model,
            language=language,
            sample_rate=values["sample_rate"],
            encoding=encoding,
            interim_results=values["interim_results"],
            endpointing=values["endpointing"],
            vad=values["vad"],
        )

    return router
