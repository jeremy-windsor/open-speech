"""Exercise the security boundary through real Uvicorn HTTP parsers.

Use production routes with disposable history and stub only the audio work.
Never start the full application's model or persistence lifecycle here.
"""

from __future__ import annotations

import asyncio
import socket
import sqlite3
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from websockets.exceptions import InvalidStatusCode
from websockets.legacy.client import connect

import src.history as history_module
from src import middleware
from src.config import Settings
from src.history import HistoryManager
from src.realtime import server as realtime_server
from src.routes import realtime, streaming, studio
from src.storage import SCHEMA_SQL

API_KEY = "test-only-api-key"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}
UPGRADE_HEADERS = [
    pytest.param([], id="ordinary-http"),
    pytest.param([("Upgrade", "websocket")], id="upgrade-only"),
    pytest.param([("Upgrade", "WeBsOcKeT")], id="mixed-case"),
    pytest.param(
        [("Upgrade", "websocket"), ("Connection", "close")],
        id="non-upgrade-connection",
    ),
    pytest.param(
        [("Upgrade", "websocket"), ("Sec-WebSocket-Key", "invalid")],
        id="invalid-websocket-key",
    ),
    pytest.param(
        [("Upgrade", "websocket"), ("Upgrade", "h2c")],
        id="duplicate-upgrade",
    ),
]


@pytest.fixture
def security_app(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None,
        os_api_key=API_KEY,
        os_rate_limit=0,
        os_rate_limit_burst=0,
        os_trust_proxy=False,
        os_ws_allowed_origins="",
        os_realtime_enabled=True,
        os_history_retain_audio=True,
        os_history_max_entries=1000,
        os_history_max_mb=2000,
    )
    monkeypatch.setattr(middleware, "settings", settings)
    monkeypatch.setattr(middleware, "_rate_limiter", None)
    monkeypatch.setattr(history_module, "settings", settings)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA_SQL)
    monkeypatch.setattr(history_module, "get_db", lambda: db)
    history = HistoryManager()
    audio = tmp_path / "history.wav"
    audio.write_bytes(b"disposable audio")
    entry_id = history.log_tts(
        "test", "test", 1.0, "wav", "private history", str(audio), audio.stat().st_size
    )

    async def handle_audio(websocket, **kwargs):
        await websocket.accept()
        await websocket.send_text("authenticated")
        await websocket.close()

    monkeypatch.setattr(streaming, "streaming_endpoint", handle_audio)
    monkeypatch.setattr(realtime_server, "realtime_endpoint", handle_audio)
    app = FastAPI()
    app.add_middleware(middleware.SecurityMiddleware)
    app.include_router(studio.create_router(
        get_settings=lambda: settings,
        get_history_manager=lambda: history,
        get_voice_library=lambda: None,
        get_profile_manager=lambda: None,
        get_conversation_manager=lambda: None,
        get_composer_manager=lambda: None,
    ))
    app.include_router(streaming.create_router())
    app.include_router(realtime.create_router(
        get_settings=lambda: settings, get_tts_router=lambda: None
    ))

    @app.get("/health")
    @app.get("/web")
    async def handle_public_request():
        return {"status": "ok"}

    try:
        yield SimpleNamespace(
            app=app, settings=settings, history=history, entry_id=entry_id, audio=audio
        )
    finally:
        db.close()


@pytest.fixture(params=["h11", "httptools"])
def live_server(request, security_app):
    @asynccontextmanager
    async def start():
        # Reserve an ephemeral port before starting, avoiding port-selection races.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(
                security_app.app,
                http=request.param,
                ws="websockets",
                lifespan="off",
                log_config=None,
                access_log=False,
            ))
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                async with asyncio.timeout(5):
                    while not server.started:
                        if task.done():
                            await task
                            raise RuntimeError("Uvicorn exited before startup")
                        await asyncio.sleep(0.01)
                yield f"http://127.0.0.1:{port}"
            finally:
                server.should_exit = True
                await asyncio.wait_for(task, timeout=5)

    return start


@pytest.mark.parametrize("headers", UPGRADE_HEADERS)
@pytest.mark.parametrize("authorization", [None, "Bearer wrong-key"])
async def test_private_reads_and_deletes_require_auth(
    live_server, security_app, headers, authorization
):
    headers = list(headers)
    if authorization:
        headers.append(("Authorization", authorization))
    async with live_server() as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        for method, path in [
            ("GET", "/api/history"),
            ("DELETE", f"/api/history/{security_app.entry_id}"),
            ("DELETE", "/api/history"),
        ]:
            response = await client.request(method, path, headers=headers)
            assert response.status_code == 401
            assert "private history" not in response.text
            assert security_app.history.list_entries()["total"] == 1
            assert security_app.audio.exists()


@pytest.mark.parametrize("headers", UPGRADE_HEADERS)
async def test_authenticated_history_reads_and_deletes_work(live_server, security_app, headers):
    headers = [*headers, *AUTH_HEADERS.items()]
    async with live_server() as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        response = await client.get("/api/history", headers=headers)
        assert response.status_code == 200
        assert response.json()["items"][0]["full_text"] == "private history"
        response = await client.delete(
            f"/api/history/{security_app.entry_id}", headers=headers
        )
        assert response.status_code == 204
        assert not security_app.audio.exists()
        security_app.history.log_stt("test", "test.wav", "second entry")
        response = await client.delete("/api/history", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"deleted": 1}
        assert security_app.history.list_entries()["total"] == 0


@pytest.mark.parametrize("auth_enabled", [True, False])
async def test_upgrade_header_cannot_skip_rate_limit(live_server, security_app, auth_enabled):
    security_app.settings.os_api_key = API_KEY if auth_enabled else ""
    security_app.settings.os_rate_limit = 1
    security_app.settings.os_rate_limit_burst = 2
    headers = {"Upgrade": "websocket"}
    if auth_enabled:
        headers.update(AUTH_HEADERS)
    async with live_server() as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        for _ in range(2):
            response = await client.get("/api/history", headers=headers)
            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
        response = await client.get("/api/history", headers=headers)
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        # Ordinary HTTP shares the same exhausted bucket.
        response = await client.get("/api/history", headers=AUTH_HEADERS if auth_enabled else {})
        assert response.status_code == 429
        for path in ("/health", "/web", "/docs", "/openapi.json"):
            response = await client.get(path, headers={"Upgrade": "websocket"})
            assert response.status_code == 200


async def test_http_stream_route_keeps_auth_and_upgrade_response(live_server):
    async with live_server() as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        response = await client.get("/v1/audio/stream", headers={"Upgrade": "websocket"})
        assert response.status_code == 401
        response = await client.get(
            "/v1/audio/stream", headers={**AUTH_HEADERS, "Upgrade": "websocket"}
        )
        assert response.status_code == 426


async def test_invalid_websocket_handshake_cannot_reach_history(live_server, security_app):
    headers = {
        "Connection": "Upgrade",
        "Upgrade": "websocket",
        "Sec-WebSocket-Key": "invalid",
        "Sec-WebSocket-Version": "13",
    }
    async with live_server() as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        response = await client.get("/api/history", headers=headers)
        assert response.status_code == 400
        assert "private history" not in response.text
        assert security_app.history.list_entries()["total"] == 1


@pytest.mark.parametrize("path", ["/v1/audio/stream", "/v1/realtime"])
async def test_actual_websockets_keep_route_auth(live_server, path):
    async with live_server() as url:
        ws_url = url.replace("http://", "ws://", 1) + path
        for headers in ({}, {"Authorization": "Bearer wrong-key"}):
            with pytest.raises(InvalidStatusCode) as error:
                async with connect(ws_url, extra_headers=headers):
                    pytest.fail("Unauthenticated WebSocket was accepted")
            assert error.value.status_code == 403
        async with connect(ws_url, extra_headers=AUTH_HEADERS) as websocket:
            assert await websocket.recv() == "authenticated"
        async with connect(ws_url + f"?api_key={API_KEY}") as websocket:
            assert await websocket.recv() == "authenticated"
