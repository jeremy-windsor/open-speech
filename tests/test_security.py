"""Tests for security middleware — Host and Origin checks, API key auth, rate limiting,
input validation, CORS and response headers."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.requests import Request
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from src.config import Settings
from src.services import stt as stt_service

MB = 1024 * 1024


class _GuardedChunkedUpload:
    """Upload fake that rejects unbounded reads and yields fixed-size chunks."""

    def __init__(self, total_bytes: int) -> None:
        self._remaining = total_bytes
        self.read_sizes: list[int] = []
        self.filename = "oversized.wav"
        self.content_type = "audio/wav"

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 1:
            raise AssertionError("upload reads must specify a positive size")
        chunk_size = min(size, self._remaining)
        self._remaining -= chunk_size
        return b"x" * chunk_size


@contextmanager
def _make_client(
    api_key: str = "",
    rate_limit: int = 0,
    rate_limit_burst: int = 0,
    max_upload_mb: int = 100,
    cors_origins: str = "",
    ws_allowed_origins: str = "",
    **overrides,
):
    """Create a test client with custom security settings and mocked backend."""
    test_settings = Settings(
        os_api_key=api_key,
        os_rate_limit=rate_limit,
        os_rate_limit_burst=rate_limit_burst,
        os_max_upload_mb=max_upload_mb,
        os_cors_origins=cors_origins,
        os_ws_allowed_origins=ws_allowed_origins,
        stt_model="test-model",
        stt_device="cpu",
        **overrides,
    )

    # Reset global rate limiter so each test starts fresh
    import src.middleware as mw
    mw._rate_limiter = None
    # Imported before the patch, so a first import binds the real settings.
    import src.main  # noqa: F401

    with patch("src.config.settings", test_settings), \
         patch("src.middleware.settings", test_settings):
        # Re-import app to pick up patched settings
        # Instead, we patch the module-level references
        from src.main import app
        from src import router as router_module

        mock_backend = MagicMock()
        mock_backend.name = "faster-whisper"
        mock_backend.loaded_models.return_value = []
        mock_backend.is_model_loaded.return_value = False
        mock_backend.transcribe.return_value = {"text": "hello world"}
        mock_backend.translate.return_value = {"text": "hello world"}

        with patch.object(router_module.router, "_default_backend", mock_backend), \
             patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}):
            with TestClient(app) as client:
                yield client


@contextmanager
def _security_client(**overrides):
    """A bare app behind SecurityMiddleware: no models, no storage, only the settings given."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    import src.middleware as mw

    test_settings = Settings(_env_file=None, **overrides)
    app = FastAPI()
    app.add_middleware(mw.SecurityMiddleware)
    app.include_router(mw.create_auth_router())

    @app.get("/v1/ping")
    async def ping():
        return {"ok": True}

    @app.post("/v1/echo")
    async def echo(request: Request):
        return {"received": len(await request.body())}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/web")
    async def web():
        return HTMLResponse("<!doctype html><title>UI</title>")

    @app.websocket("/v1/ws")
    async def ws(websocket: WebSocket):
        if not mw.verify_ws_origin(websocket):
            await websocket.close(code=1008)
            return
        if not mw.verify_ws_api_key(websocket):
            await websocket.close(code=4001)
            return
        await websocket.accept()
        await websocket.send_text("ok")
        await websocket.close()

    with patch("src.middleware.settings", test_settings):
        yield TestClient(app)


def _forwarded_request(*forwarded_lines: str) -> Request:
    """A real request whose X-Forwarded-For field lines arrive exactly as given."""
    headers = [(b"x-forwarded-for", line.encode("latin-1")) for line in forwarded_lines]
    return Request({
        "type": "http", "method": "GET", "path": "/v1/models", "query_string": b"",
        "headers": headers, "client": ("10.0.0.2", 40000), "server": ("testserver", 80), "scheme": "http",
    })


async def _call_asgi(app, scope: dict, body_chunks: list[bytes] = ()) -> tuple[int, int]:
    """Run one raw ASGI request; return (status, request body bytes the app pulled)."""
    pieces = list(body_chunks)
    pulled = 0
    status = 0
    done = asyncio.Event()

    async def receive():
        nonlocal pulled
        if pieces:
            piece = pieces.pop(0)
            pulled += len(piece)
            return {"type": "http.request", "body": piece, "more_body": bool(pieces)}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        if message["type"] == "http.response.body" and not message.get("more_body"):
            done.set()

    base = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/", "raw_path": b"/", "query_string": b"", "root_path": "",
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80), "headers": [(b"host", b"testserver")],
    }
    await asyncio.wait_for(app({**base, **scope}, receive, send), timeout=30)
    return status, pulled


def test_make_client_first_imports_src_main_outside_its_settings_patch(monkeypatch):
    # When _make_client was the first to import src.main, the import ran inside its
    # patch of src.config.settings, so src.main bound the throwaway Settings for the rest of the run and the refusal
    # tests (TestRefusedWebSocketsSayWhy) switched features off on an object the app no longer read.
    # To make "first import" happen here whatever ran before, without running src/main.py twice (a second app would
    # hold its own databases open), src.main is taken out of sys.modules and a finder hands the import the same module
    # back, recording the src.config.settings that `from src.config import settings` would bind at that moment.
    import importlib.abc
    import importlib.util
    import sys

    import src
    import src.config
    import src.main as main_module

    bound = []

    class _FirstImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            return importlib.util.spec_from_loader(name, self) if name == "src.main" else None

        def create_module(self, spec):
            return main_module

        def exec_module(self, module):
            bound.append(sys.modules["src.config"].settings)

    monkeypatch.setattr(main_module, "__spec__", main_module.__spec__)  # the import sets it to the finder's spec
    monkeypatch.delitem(sys.modules, "src.main")
    monkeypatch.delattr(src, "main")
    monkeypatch.setattr(sys, "meta_path", [_FirstImport(), *sys.meta_path])
    with _make_client():
        pass

    assert bound, "_make_client did not import src.main"
    assert bound[0] is src.config.settings, "src.main was first imported inside _make_client's settings patch"


# ---- API Key Auth Tests ----

class TestApiKeyAuth:
    """Test API key authentication."""

    def test_no_key_configured_allows_all(self):
        """When STT_API_KEY is empty, all requests pass."""
        with _make_client(api_key="") as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200

    def test_key_required_rejects_missing(self):
        """When STT_API_KEY is set, requests without key get 401."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 401
            assert "API key" in resp.json()["error"]["message"]
            # RFC 9110 15.5.2 / RFC 6750 3: a 401 names the scheme the server accepts.
            assert resp.headers["www-authenticate"] == "Bearer"

    def test_key_bearer_header_accepted(self):
        """Valid Bearer token in Authorization header passes."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/v1/models", headers={"Authorization": "Bearer secret123"})
            assert resp.status_code == 200

    def test_key_wrong_bearer_rejected(self):
        """Wrong Bearer token gets 401."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
            assert resp.status_code == 401
            assert resp.headers["www-authenticate"] == 'Bearer error="invalid_token"'

    def test_key_query_param_accepted(self):
        """API key via query parameter works."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/v1/models?api_key=secret123")
            assert resp.status_code == 200

    def test_bearer_scheme_is_case_insensitive(self):
        """RFC 7235 section 2.1: the auth-scheme token is case-insensitive."""
        with _make_client(api_key="secret123") as client:
            for scheme in ("bearer", "BEARER"):
                resp = client.get("/v1/models", headers={"Authorization": f"{scheme} secret123"})
                assert resp.status_code == 200

    def test_non_ascii_query_key_rejected_with_401(self):
        """A non-ASCII candidate key is a wrong key, not a server error."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/v1/models", params={"api_key": "sécret"})
            assert resp.status_code == 401

    def test_non_ascii_configured_key(self):
        """A non-ASCII OS_API_KEY still rejects wrong keys with 401 and accepts its own."""
        with _make_client(api_key="sécret") as client:
            assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert client.get("/v1/models", params={"api_key": "sécret"}).status_code == 200

    def test_health_exempt(self):
        """Health endpoint never requires auth."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_web_exempt(self):
        """Web UI endpoint never requires auth."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/web")
            # 200 or 404 (no static file), but NOT 401
            assert resp.status_code != 401

    def test_transcribe_requires_key(self):
        """Audio endpoints require auth when key is set."""
        with _make_client(api_key="secret123") as client:
            audio = b"RIFF" + b"\x00" * 100
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", audio, "audio/wav")},
                data={"model": "test-model"},
            )
            assert resp.status_code == 401

    def test_transcribe_with_key_works(self):
        """Audio endpoints pass with valid key."""
        with _make_client(api_key="secret123") as client:
            audio = b"RIFF" + b"\x00" * 100
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", audio, "audio/wav")},
                data={"model": "test-model"},
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status_code == 200


# ---- Rate Limiting Tests ----

class TestRateLimiting:
    """Test per-IP rate limiting."""

    def test_no_rate_limit(self):
        """When rate_limit=0, no limiting applied."""
        with _make_client(rate_limit=0) as client:
            for _ in range(20):
                resp = client.get("/v1/models")
                assert resp.status_code == 200

    def test_rate_limit_allows_within_burst(self):
        """Requests within burst limit succeed."""
        with _make_client(rate_limit=60, rate_limit_burst=5) as client:
            for _ in range(5):
                resp = client.get("/v1/models")
                assert resp.status_code == 200

    def test_rate_limit_blocks_over_burst(self):
        """Requests beyond burst get 429."""
        # One token a minute: a slow runner cannot refill the bucket between requests.
        with _make_client(rate_limit=1, rate_limit_burst=3) as client:
            for _ in range(3):
                resp = client.get("/v1/models")
                assert resp.status_code == 200
            resp = client.get("/v1/models")
            assert resp.status_code == 429
            assert "Rate limit" in resp.json()["error"]["message"]

    def test_rate_limit_headers_present(self):
        """Rate limit response headers are set."""
        with _make_client(rate_limit=60, rate_limit_burst=5) as client:
            resp = client.get("/v1/models")
            assert "X-RateLimit-Limit" in resp.headers

    def test_health_exempt_from_rate_limit(self):
        """Health endpoint is not rate limited."""
        with _make_client(rate_limit=60, rate_limit_burst=1) as client:
            client.get("/v1/models")  # consume the 1 token
            client.get("/v1/models")  # this would be 429
            # But health should always work
            resp = client.get("/health")
            assert resp.status_code == 200


# ---- Input Validation Tests ----

class TestInputValidation:
    """Test upload size and format validation."""

    def test_empty_file_rejected(self):
        """Empty audio file returns 400."""
        with _make_client() as client:
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"", "audio/wav")},
                data={"model": "test-model"},
            )
            assert resp.status_code == 400
            assert "Empty" in resp.json()["error"]["message"]

    def test_oversized_file_rejected(self):
        """File exceeding max upload size returns 413."""
        with _make_client(max_upload_mb=1) as client:
            # Patch the settings at the point where main.py reads it
            import src.main as main_mod
            with patch.object(main_mod, "settings", Settings(
                os_max_upload_mb=1, stt_model="test-model", stt_device="cpu",
            )):
                # 1.5 MB of zeros
                big_audio = b"RIFF" + b"\x00" * (1024 * 1024 + 500000)
                resp = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", big_audio, "audio/wav")},
                    data={"model": "test-model"},
                )
                assert resp.status_code == 413
                assert "too large" in resp.json()["error"]["message"]

    def test_normal_file_accepted(self):
        """Normal-sized file passes validation."""
        with _make_client(max_upload_mb=100) as client:
            audio = b"RIFF" + b"\x00" * 100
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", audio, "audio/wav")},
                data={"model": "test-model"},
            )
            assert resp.status_code == 200

    def test_translate_empty_file_rejected(self):
        """Empty file on translate endpoint also returns 400."""
        with _make_client() as client:
            resp = client.post(
                "/v1/audio/translations",
                files={"file": ("test.wav", b"", "audio/wav")},
                data={"model": "test-model"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_stt_upload_limit_uses_bounded_reads(self):
        """The upload limit must be enforced without reading the whole file at once."""
        max_bytes = 1024 * 1024
        upload = _GuardedChunkedUpload(max_bytes + 1)
        settings = SimpleNamespace(
            os_max_upload_mb=1,
            stt_noise_reduce=False,
            stt_normalize=False,
        )

        with (
            patch.object(
                stt_service,
                "convert_to_wav",
                side_effect=AssertionError("oversized uploads must not reach conversion"),
            ) as convert,
            pytest.raises(HTTPException) as exc_info,
        ):
            await stt_service.read_and_prepare_upload(file=upload, settings=settings)

        assert exc_info.value.status_code == 413
        assert upload.read_sizes
        assert all(0 < size <= max_bytes + 1 for size in upload.read_sizes)
        convert.assert_not_called()

    @pytest.mark.parametrize(
        ("path", "backend_method"),
        [
            ("/v1/audio/transcriptions", "transcribe"),
            ("/v1/audio/translations", "translate"),
        ],
        ids=["transcription", "translation"],
    )
    def test_invalid_stt_response_format_rejected_before_backend_work(
        self,
        path,
        backend_method,
    ):
        """Unsupported formats must fail before starting model inference."""
        import src.main as main_mod

        client = TestClient(main_mod.app)
        with (
            patch(
                "src.services.stt.read_and_prepare_upload",
                new=AsyncMock(return_value=b"prepared-wav"),
            ),
            patch.object(
                main_mod.backend_router,
                backend_method,
                return_value={"text": "unexpected backend result"},
            ) as backend_call,
        ):
            resp = client.post(
                path,
                files={"file": ("test.wav", b"RIFF", "audio/wav")},
                data={"model": "test-model", "response_format": "bogus"},
            )

        backend_call.assert_not_called()
        assert resp.status_code == 400
        assert "response_format" in resp.json()["error"]["message"]


# ---- CORS Tests ----

class TestCors:
    """Test CORS configuration."""

    def test_cors_wildcard_allows_any_origin(self):
        """Default * allows any origin."""
        with _make_client(cors_origins="*") as client:
            resp = client.options(
                "/v1/models",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            # FastAPI CORS with allow_origins=["*"] reflects the request origin
            assert resp.headers.get("access-control-allow-origin") in ("*", "https://example.com")

    def test_cors_headers_present(self):
        """CORS headers are present on preflight responses."""
        with _make_client(cors_origins="*") as client:
            resp = client.options(
                "/v1/models",
                headers={
                    "Origin": "https://myapp.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            # CORS middleware is configured at app init; verify headers exist
            assert "access-control-allow-origin" in resp.headers
            assert "access-control-allow-methods" in resp.headers


# ---- Middleware Unit Tests ----

class TestMiddlewareUnits:
    """Unit tests for middleware helper functions."""

    def test_rate_limiter_cleanup(self):
        """Stale entries are cleaned up."""
        from src.middleware import RateLimiter
        rl = RateLimiter(requests_per_minute=60)
        # Simulate an old entry
        rl._buckets["1.2.3.4"] = (5.0, 0.0)  # time 0 = very old
        rl.cleanup(max_age=1.0)
        assert "1.2.3.4" not in rl._buckets

    def test_rate_limiter_refill(self):
        """Tokens refill over time (on a fake clock: a loaded machine refilled the bucket mid-test)."""
        import src.middleware as mw
        rl = mw.RateLimiter(requests_per_minute=6000, burst=5)  # 100/sec

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "1.2.3.4"
        mock_request.url.path = "/v1/models"

        now = [1000.0]
        with patch.object(mw, "time", SimpleNamespace(monotonic=lambda: now[0])):
            # Drain all tokens
            for _ in range(5):
                allowed, _ = rl.check(mock_request)
                assert allowed

            allowed, _ = rl.check(mock_request)
            assert not allowed  # depleted

            now[0] += 0.005  # half a token
            assert rl.check(mock_request)[0] is False

            now[0] += 0.02  # two tokens' worth since the last check
            assert rl.check(mock_request)[0] is True
            assert rl.check(mock_request)[0] is True
            assert rl.check(mock_request)[0] is False

            now[0] += 60  # never more than the burst
            assert [rl.check(mock_request)[0] for _ in range(6)] == [True] * 5 + [False]

    def test_rate_limiter_keys_on_proxy_appended_forwarded_for(self):
        """With OS_TRUST_PROXY, a client-supplied X-Forwarded-For prefix gets no fresh bucket."""
        from src.middleware import RateLimiter
        rl = RateLimiter(requests_per_minute=60, burst=1)

        with patch("src.middleware.settings", Settings(os_trust_proxy=True)):
            assert rl.check(_forwarded_request("198.51.100.1, 203.0.113.7"))[0] is True
            assert rl.check(_forwarded_request("198.51.100.2, 203.0.113.7"))[0] is False
            assert rl.check(_forwarded_request("203.0.113.8"))[0] is True

    def test_rate_limiter_keys_on_a_proxy_added_forwarded_for_line(self):
        """RFC 9110 5.3: repeated field lines are one list, so the proxy's own line counts.

        HAProxy 'option forwardfor' adds a line after the client's instead of appending to it.
        """
        from src.middleware import RateLimiter
        rl = RateLimiter(requests_per_minute=60, burst=1)

        with patch("src.middleware.settings", Settings(os_trust_proxy=True)):
            assert rl.check(_forwarded_request("198.51.100.1", "203.0.113.7"))[0] is True
            # Same client behind the proxy, a new spoofed first line: the bucket is spent.
            assert rl.check(_forwarded_request("198.51.100.2", "203.0.113.7"))[0] is False

    def test_trusted_proxy_hops_picks_the_entry_the_outer_proxy_saw(self):
        """Behind a CDN and nginx (OS_TRUSTED_PROXY_HOPS=2) the rightmost entry is the CDN."""
        from src.middleware import client_ip

        two_hops = Settings(os_trust_proxy=True, os_trusted_proxy_hops=2)
        with patch("src.middleware.settings", two_hops):
            assert client_ip(_forwarded_request("6.6.6.6, 203.0.113.7", "192.0.2.50")) == "203.0.113.7"
            assert client_ip(_forwarded_request("203.0.113.7")) == "203.0.113.7"
            assert client_ip(_forwarded_request()) == "10.0.0.2"
        with patch("src.middleware.settings", Settings(os_trust_proxy=False)):
            assert client_ip(_forwarded_request("203.0.113.7")) == "10.0.0.2"

    def test_auth_exempt_paths(self):
        """Verify exempt path detection."""
        from src.middleware import _is_auth_exempt
        assert _is_auth_exempt("/health") is True
        assert _is_auth_exempt("/web") is True
        assert _is_auth_exempt("/web/assets/style.css") is True
        assert _is_auth_exempt("/v1/models") is False
        assert _is_auth_exempt("/v1/audio/transcriptions") is False


class TestSecurityWarnings:
    def test_auth_disabled_logs_startup_warning(self, caplog):
        with caplog.at_level("WARNING"):
            with _make_client(api_key="") as client:
                resp = client.get("/health")
        assert resp.status_code == 200
        assert "No API key set" in caplog.text

    def test_query_param_auth_logs_deprecation(self, caplog):
        with _make_client(api_key="secret123") as client:
            with caplog.at_level("WARNING"):
                resp = client.get("/v1/models?api_key=secret123")
            assert resp.status_code == 200
            assert "API key in query string is deprecated" in caplog.text


class TestWebSocketApiKey:
    def test_non_ascii_query_key_is_rejected(self):
        from src.middleware import verify_ws_api_key

        ws = SimpleNamespace(query_params={"api_key": "sécret"}, headers={})
        with patch("src.middleware.settings", Settings(os_api_key="secret123")):
            assert verify_ws_api_key(ws) is False

    def test_lowercase_bearer_header_is_accepted(self):
        from src.middleware import verify_ws_api_key

        ws = SimpleNamespace(query_params={}, headers={"authorization": "bearer secret123"})
        with patch("src.middleware.settings", Settings(os_api_key="secret123")):
            assert verify_ws_api_key(ws) is True


class TestWebSocketOriginValidation:
    def test_verify_ws_origin_refuses_a_foreign_origin_by_default(self):
        """Secure default: an empty OS_WS_ALLOWED_ORIGINS allows only this server's own pages."""
        from src.middleware import verify_ws_origin

        ws = MagicMock()
        ws.headers = {"origin": "https://evil.example", "host": "lan-box:8100"}
        with patch("src.middleware.settings", Settings(os_ws_allowed_origins="")):
            assert verify_ws_origin(ws) is False

    def test_verify_ws_origin_allows_same_origin_and_origin_less_clients(self):
        from src.middleware import verify_ws_origin

        with patch("src.middleware.settings", Settings(os_ws_allowed_origins="")):
            assert verify_ws_origin(SimpleNamespace(headers={"origin": "http://lan-box:8100", "host": "lan-box:8100"}))
            assert verify_ws_origin(SimpleNamespace(headers={"origin": "https://LAN-BOX", "host": "lan-box:443"}))
            assert verify_ws_origin(SimpleNamespace(headers={"origin": "http://[::1]:8100", "host": "[::1]:8100"}))
            # The SDK, Home Assistant and CLI tools send no Origin.
            assert verify_ws_origin(SimpleNamespace(headers={"host": "lan-box:8100"}))
            # Another port is another origin.
            assert not verify_ws_origin(SimpleNamespace(headers={"origin": "http://lan-box:3000", "host": "lan-box:8100"}))
            assert not verify_ws_origin(SimpleNamespace(headers={"origin": "null", "host": "lan-box:8100"}))

    def test_verify_ws_origin_star_allows_any_origin(self):
        from src.middleware import verify_ws_origin

        ws = SimpleNamespace(headers={"origin": "https://evil.example", "host": "lan-box:8100"})
        with patch("src.middleware.settings", Settings(os_ws_allowed_origins="*")):
            assert verify_ws_origin(ws) is True

    def test_verify_ws_origin_blocks_not_allowed(self):
        from src.middleware import verify_ws_origin

        ws = MagicMock()
        ws.headers = {"origin": "https://evil.example"}
        with patch("src.middleware.settings", Settings(os_ws_allowed_origins="https://good.example")):
            assert verify_ws_origin(ws) is False

    def test_verify_ws_origin_allows_listed_origin(self):
        from src.middleware import verify_ws_origin

        ws = MagicMock()
        ws.headers = {"origin": "https://good.example"}
        with patch("src.middleware.settings", Settings(os_ws_allowed_origins="https://good.example,https://other.example")):
            assert verify_ws_origin(ws) is True


GOOD_KEY = {"Authorization": "Bearer secret123"}


class TestRoutedPathDecidesAuth:
    """Auth and rate limiting follow the path the router matches, never a URL rebuilt from Host."""

    def test_host_with_a_path_cannot_skip_auth_on_reads_or_deletes(self):
        with _make_client(api_key="secret123") as client:
            read = client.get("/api/history", headers={"host": "lan-box/web/?"})
            wipe = client.delete("/api/history", headers={"host": "lan-box/v1/reader/media/#"})
        assert (read.status_code, wipe.status_code) == (400, 400)
        assert read.json() == {"error": {"message": "Invalid Host header"}}

    def test_auth_uses_the_routed_path_when_any_host_is_allowed(self):
        with _make_client(api_key="secret123", os_allowed_hosts="*") as client:
            assert client.get("/api/history", headers={"host": "lan-box/web/?"}).status_code == 401
            assert client.delete("/api/history", headers={"host": "lan-box/v1/reader/media/#"}).status_code == 401
            assert client.get("/api/history", headers={"host": "lan-box/web/?", **GOOD_KEY}).status_code == 200

    def test_rate_limit_uses_the_routed_path_when_any_host_is_allowed(self):
        # One token a minute: a slow runner cannot refill the bucket between requests.
        with _make_client(rate_limit=1, rate_limit_burst=2, os_allowed_hosts="*") as client:
            codes = [client.get("/v1/models", headers={"host": "lan-box/health?"}).status_code for _ in range(4)]
        assert codes == [200, 200, 429, 429]

    def test_routed_path_strips_root_path_like_the_router(self):
        from src.middleware import _routed_path

        assert _routed_path({"path": "/speech/api/history", "root_path": "/speech"}) == "/api/history"
        assert _routed_path({"path": "/speechx/web", "root_path": "/speech"}) == "/speechx/web"
        assert _routed_path({"path": "/api/history", "root_path": ""}) == "/api/history"


class TestHostValidation:
    """DNS rebinding: only IP literals, LAN names and OS_ALLOWED_HOSTS reach the app."""

    @pytest.mark.parametrize("host", [
        "127.0.0.1:8100", "192.168.1.20", "[::1]:8100", "[fe80::1]", "lan-box", "lan-box:8100", "LAN-BOX",
        "localhost:8100", "app.localhost", "speech.local", "box.lan:8100", "nas.home.arpa", "gpu.internal",
        "testserver", "lan-box:",  # RFC 3986 3.2.3: an empty port means the default one
    ])
    def test_ip_literals_and_lan_names_are_accepted_by_default(self, host):
        with _security_client() as client:
            assert client.get("/v1/ping", headers={"host": host}).status_code == 200

    @pytest.mark.parametrize("host", [
        "lan-box/web/?", "lan-box?x", "lan-box#x", "user@lan-box", "lan box", "lan-box:80:80", "[::1",
        "lan-box:123456", "[not-an-ip]", "lan-box:65536", "lan-box:99999",
    ])
    def test_malformed_host_is_refused(self, host):
        with _security_client(os_allowed_hosts="*.example.com") as client:
            resp = client.get("/v1/ping", headers={"host": host})
        assert resp.status_code == 400
        assert resp.json() == {"error": {"message": "Invalid Host header"}}

    async def test_two_host_lines_are_refused(self):
        with _security_client() as client:
            status, _ = await _call_asgi(
                client.app, {"path": "/v1/ping", "headers": [(b"host", b"lan-box"), (b"host", b"evil.example")]}
            )
        assert status == 400

    def test_other_domains_are_refused_with_a_one_time_warning(self, caplog):
        with _security_client() as client, caplog.at_level(logging.WARNING):
            first = client.get("/v1/ping", headers={"host": "speech.example.com"})
            second = client.get("/v1/ping", headers={"host": "rebind.attacker.example:8100"})
        assert (first.status_code, second.status_code) == (400, 400)
        assert "OS_ALLOWED_HOSTS" in first.json()["error"]["message"]
        assert len([r for r in caplog.records if "OS_ALLOWED_HOSTS" in r.getMessage()]) == 1

    def test_scanner_garbage_does_not_use_up_the_warning_for_a_refused_name(self, caplog):
        with _security_client() as client, caplog.at_level(logging.WARNING):
            client.get("/v1/ping", headers={"host": "lan-box/../../etc"})
            client.get("/v1/ping", headers={"host": "lan-box?x"})
            client.get("/v1/ping", headers={"host": "speech.example.com"})
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        # Listing a malformed Host cannot help, so only the refused name points at OS_ALLOWED_HOSTS.
        name_warnings = [w for w in warnings if "OS_ALLOWED_HOSTS" in w]
        assert len(name_warnings) == 1 and "speech.example.com" in name_warnings[0]
        assert len([w for w in warnings if "malformed" in w.lower()]) == 1

    def test_allowed_hosts_lists_names_and_wildcards(self):
        with _security_client(os_allowed_hosts="speech.example.com, *.example.org") as client:
            assert client.get("/v1/ping", headers={"host": "speech.example.com:8443"}).status_code == 200
            assert client.get("/v1/ping", headers={"host": "a.b.example.org"}).status_code == 200
            assert client.get("/v1/ping", headers={"host": "example.org"}).status_code == 400
            assert client.get("/v1/ping", headers={"host": "other.example.com"}).status_code == 400
        with _security_client(os_allowed_hosts="*") as client:
            assert client.get("/v1/ping", headers={"host": "anything.example.net"}).status_code == 200
            assert client.get("/v1/ping", headers={"host": "lan-box/web/?"}).status_code == 200

    def test_this_machines_own_names_are_accepted(self):
        with _security_client() as client, \
             patch("src.middleware._machine_names", return_value=frozenset({"workstation.corp.example"})):
            assert client.get("/v1/ping", headers={"host": "workstation.corp.example:8100"}).status_code == 200

    def test_this_machines_names_are_looked_up_off_the_event_loop_at_startup(self):
        """socket.getfqdn() can wait seconds on reverse DNS; a request must not stall every stream on it."""
        import src.middleware as mw

        lookups = []

        def getfqdn():
            try:
                asyncio.get_running_loop()
                lookups.append("event loop")
            except RuntimeError:
                lookups.append("worker thread")
            return "workstation.corp.example"

        mw._machine_names.cache_clear()
        try:
            with patch("src.middleware.socket.getfqdn", getfqdn), _security_client() as client, client:
                own = client.get("/v1/ping", headers={"host": "workstation.corp.example"})
                other = client.get("/v1/ping", headers={"host": "unlisted.example.com"})
        finally:
            mw._machine_names.cache_clear()
        assert (own.status_code, other.status_code) == (200, 400)
        assert lookups == ["worker thread"]


class TestSecureCorsDefault:
    def test_no_cors_headers_by_default(self):
        assert Settings(_env_file=None).os_cors_origins == ""
        with _make_client() as client:
            preflight = client.options(
                "/v1/models",
                headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"},
            )
            simple = client.get("/v1/models", headers={"Origin": "https://example.com"})
        assert "access-control-allow-origin" not in preflight.headers
        assert simple.status_code == 200
        assert "access-control-allow-origin" not in simple.headers

    def test_listed_origins_get_credentialed_cors(self):
        with _security_client(os_cors_origins="https://app.example") as client:
            listed = client.get("/v1/ping", headers={"Origin": "https://app.example"})
            other = client.get("/v1/ping", headers={"Origin": "https://evil.example"})
        assert listed.headers["access-control-allow-origin"] == "https://app.example"
        assert listed.headers["access-control-allow-credentials"] == "true"
        assert "access-control-allow-origin" not in other.headers


class TestCrossSiteWriteGuard:
    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_a_foreign_page_cannot_write(self, method):
        with _security_client() as client:
            resp = client.request(method, "/v1/echo", content=b"x", headers={"Origin": "https://evil.example"})
        assert resp.status_code == 403
        assert "OS_CORS_ORIGINS" in resp.json()["error"]["message"]

    def test_a_foreign_page_cannot_wipe_history(self):
        with _make_client() as client:
            assert client.delete("/api/history", headers={"Origin": "https://evil.example"}).status_code == 403
            assert client.delete("/api/history", headers={"Origin": "null"}).status_code == 403

    def test_same_origin_listed_and_origin_less_writes_pass(self):
        with _security_client(os_cors_origins="https://app.example") as client:
            assert client.post("/v1/echo", content=b"x", headers={"Origin": "http://testserver"}).status_code == 200
            assert client.post("/v1/echo", content=b"x", headers={"Origin": "https://app.example"}).status_code == 200
            assert client.post("/v1/echo", content=b"x").status_code == 200  # SDKs and curl send no Origin
            # Reads are not refused: without CORS headers the page cannot see the response.
            assert client.get("/v1/ping", headers={"Origin": "https://evil.example"}).status_code == 200

    def test_cross_site_fetch_metadata_without_origin_is_refused(self):
        with _security_client() as client:
            assert client.post("/v1/echo", content=b"x", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
            assert client.post("/v1/echo", content=b"x", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200

    def test_star_cors_reopens_cross_site_writes(self):
        with _security_client(os_cors_origins="*") as client:
            assert client.post("/v1/echo", content=b"x", headers={"Origin": "https://evil.example"}).status_code == 200


HARDENING_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "same-origin",
    "x-frame-options": "DENY",
    "cross-origin-opener-policy": "same-origin",
    "permissions-policy": "microphone=(self), camera=(), geolocation=()",
}


class TestSecurityHeaders:

    def test_every_response_gets_the_hardening_headers(self):
        with _security_client(os_api_key="secret123") as client:
            responses = [
                client.get("/health"),
                client.get("/v1/ping"),  # 401
                client.get("/v1/ping", headers={"host": "evil.example.com"}),  # 400
                client.get("/v1/ping", headers=GOOD_KEY),
            ]
        for resp in responses:
            assert {name: resp.headers.get(name) for name in HARDENING_HEADERS} == HARDENING_HEADERS

    @pytest.mark.parametrize("api_docs", [False, True])
    def test_api_docs_pages_are_served_only_with_os_api_docs(self, monkeypatch, api_docs):
        # Swagger UI and ReDoc load unpinned scripts from a CDN on the origin where the web UI keeps its API
        # key, so the real app serves them only on request (OS_API_DOCS). The schema itself is always there.
        import src.main as main_module

        monkeypatch.setattr(main_module.settings, "os_api_docs", api_docs)
        client = TestClient(main_module.create_app())  # no lifespan: no models, no storage
        docs, redoc, schema = client.get("/docs"), client.get("/redoc"), client.get("/openapi.json")
        expected = 200 if api_docs else 404
        assert (docs.status_code, redoc.status_code) == (expected, expected)
        assert schema.status_code == 200 and schema.json()["info"]["title"] == "Open Speech"
        if api_docs:
            assert "swagger-ui" in docs.text and "redoc" in redoc.text
        assert Settings(_env_file=None).os_api_docs is False, "off unless asked for"

    def test_web_ui_html_gets_the_csp_and_api_docs_do_not(self):
        with _security_client() as client:
            web, docs, api = client.get("/web"), client.get("/docs"), client.get("/v1/ping")
        csp = web.headers["content-security-policy"]
        for directive in (
            "default-src 'self'", "script-src 'self'", "connect-src 'self' ws: wss:", "object-src 'none'",
            "base-uri 'none'", "frame-ancestors 'none'", "media-src 'self' blob: data:",
        ):
            assert directive in csp
        assert docs.status_code == 200 and "content-security-policy" not in docs.headers
        assert "content-security-policy" not in api.headers

    def test_the_bundled_web_ui_is_served_with_the_csp(self):
        with _make_client() as client:
            resp = client.get("/web")
        assert resp.status_code == 200
        assert "script-src 'self'" in resp.headers["content-security-policy"]

    def test_security_headers_can_be_turned_off(self):
        with _security_client(os_security_headers=False) as client:
            resp = client.get("/web")
        assert "x-frame-options" not in resp.headers
        assert "content-security-policy" not in resp.headers


# An exception no route maps went to Starlette's ServerErrorMiddleware, which sits
# outside SecurityMiddleware and answered a plain-text "Internal Server Error" without the headers.
INTERNAL_ERROR = {"error": {"message": "Internal server error", "code": "internal_error"}}


class TestUnexpectedErrors:

    @pytest.fixture
    def secure_defaults(self, monkeypatch):
        # The documented defaults, whatever a local .env says: security headers on, no API key.
        from src.config import settings

        monkeypatch.setattr(settings, "os_security_headers", True)
        monkeypatch.setattr(settings, "os_api_key", "")
        return settings

    @staticmethod
    def _crashing_app():
        import src.main as main_module

        app = main_module.create_app()  # its exception handlers and middleware; no lifespan, no models

        @app.get("/api/test-crash")
        def crash():
            raise RuntimeError(r"CUDA out of memory in C:\Users\someone\models\secret")

        return app

    def test_a_route_that_raises_answers_the_json_envelope_with_the_hardening_headers(self, secure_defaults):
        client = TestClient(self._crashing_app(), raise_server_exceptions=False)
        resp = client.get("/api/test-crash")
        assert resp.status_code == 500
        assert resp.headers["content-type"] == "application/json", resp.text[:200]
        assert resp.json() == INTERNAL_ERROR, "the exception text can hold paths: never echoed"
        assert {name: resp.headers.get(name) for name in HARDENING_HEADERS} == HARDENING_HEADERS

    def test_the_handler_logs_the_route_and_leaves_the_traceback_to_the_server(self, secure_defaults, caplog):
        # Starlette re-raises after the handler answers, and uvicorn logs the traceback
        # ("Exception in ASGI application"), so the handler names the request without repeating it.
        client = TestClient(self._crashing_app(), raise_server_exceptions=False)
        with caplog.at_level(logging.ERROR, logger="open-speech"):
            client.get("/api/test-crash?token=abc")
        lines = [record.getMessage() for record in caplog.records if record.name == "open-speech"]
        assert any("GET /api/test-crash" in line and "RuntimeError" in line for line in lines), lines
        assert not any("token=abc" in line or "secret" in line for line in lines), lines
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            TestClient(self._crashing_app()).get("/api/test-crash")

    def test_with_security_headers_off_the_envelope_stays_and_the_headers_go(self, secure_defaults, monkeypatch):
        monkeypatch.setattr(secure_defaults, "os_security_headers", False)
        client = TestClient(self._crashing_app(), raise_server_exceptions=False)
        resp = client.get("/api/test-crash")
        assert resp.status_code == 500 and resp.json() == INTERNAL_ERROR
        assert not any(name in resp.headers for name in HARDENING_HEADERS)

    def test_a_real_route_whose_render_crashes_answers_the_envelope(self, secure_defaults, monkeypatch):
        # A composer MP3 render without ffmpeg, or an in-process backend out of GPU memory, raises an error the
        # route doesn't map.
        import src.main as main_module

        def render(**_kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        monkeypatch.setattr(main_module.composer_manager, "render", render)
        client = TestClient(main_module.create_app(), raise_server_exceptions=False)
        resp = client.post("/api/composer/render", json={"format": "mp3", "tracks": [{"source_path": "a.wav"}]})
        assert resp.status_code == 500
        assert resp.json() == INTERNAL_ERROR
        assert {name: resp.headers.get(name) for name in HARDENING_HEADERS} == HARDENING_HEADERS

    # The catch-all 500 leaves from ServerErrorMiddleware, outside SecurityMiddleware's
    # CORSMiddleware, so a listed origin's page (or any page under `*`) got a network error instead of this JSON.
    # README "Browser origins": listed origins get CORS headers with credentials; `*` gets them without.
    _NO_CORS = {"access-control-allow-origin": None, "access-control-allow-credentials": None, "vary": None}
    _LISTED_CORS = {
        "access-control-allow-origin": "https://app.example", "access-control-allow-credentials": "true", "vary": "Origin",
    }

    @pytest.mark.parametrize(
        ("cors_origins", "headers", "expected"),
        [
            pytest.param("https://app.example", {"Origin": "https://app.example"}, _LISTED_CORS, id="listed-origin"),
            pytest.param("https://other.example, https://APP.example/", {"Origin": "https://app.example"}, _LISTED_CORS,
                         id="second-of-two-listed"),
            pytest.param("*", {"Origin": "https://evil.example"},
                         {"access-control-allow-origin": "*", "access-control-allow-credentials": None, "vary": None},
                         id="star"),
            pytest.param("*", {"Origin": "https://evil.example", "Cookie": "a=b"},
                         {"access-control-allow-origin": "https://evil.example", "access-control-allow-credentials": None,
                          "vary": "Origin"}, id="star-with-a-cookie"),
            pytest.param("https://app.example", {"Origin": "https://evil.example"}, _NO_CORS, id="unlisted-origin"),
            pytest.param("", {"Origin": "https://evil.example"}, _NO_CORS, id="cors-off"),
            pytest.param("https://app.example", {}, _NO_CORS, id="no-origin"),
        ],
    )
    def test_the_500_carries_the_cors_headers_the_cors_layer_gives(
        self, secure_defaults, monkeypatch, cors_origins, headers, expected,
    ):
        monkeypatch.setattr(secure_defaults, "os_cors_origins", cors_origins)
        client = TestClient(self._crashing_app(), raise_server_exceptions=False)
        resp = client.get("/api/test-crash", headers=headers)
        assert resp.status_code == 500 and resp.json() == INTERNAL_ERROR
        assert {name: resp.headers.get(name) for name in HARDENING_HEADERS} == HARDENING_HEADERS
        assert {name: resp.headers.get(name) for name in expected} == expected, dict(resp.headers)
        if expected["access-control-allow-origin"]:
            # The same headers as CORSMiddleware's own answer to a handled error (the router's 404).
            handled = client.get("/api/test-missing-route", headers=headers)
            assert handled.status_code == 404
            assert {name: handled.headers.get(name) for name in expected} == expected, dict(handled.headers)

    def test_a_listed_origin_can_read_a_real_routes_crash(self, secure_defaults, monkeypatch):
        """A composer render out of GPU memory, sent by a listed origin's page."""
        import src.main as main_module

        def render(**_kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        monkeypatch.setattr(secure_defaults, "os_cors_origins", "https://app.example")
        monkeypatch.setattr(main_module.composer_manager, "render", render)
        client = TestClient(main_module.create_app(), raise_server_exceptions=False)
        resp = client.post(
            "/api/composer/render", json={"format": "mp3", "tracks": [{"source_path": "a.wav"}]},
            headers={"Origin": "https://app.example"},
        )
        assert resp.status_code == 500, "the cross-site write guard lets a listed origin through"
        assert resp.json() == INTERNAL_ERROR
        assert {name: resp.headers.get(name) for name in self._LISTED_CORS} == self._LISTED_CORS, dict(resp.headers)


class TestWebSocketTickets:
    def test_ticket_needs_the_api_key_and_opens_one_connection(self):
        with _security_client(os_api_key="secret123") as client:
            assert client.post("/api/auth/ws-ticket").status_code == 401
            resp = client.post("/api/auth/ws-ticket", headers=GOOD_KEY)
            assert resp.status_code == 200
            body = resp.json()
            assert body["expires_in"] == 60
            assert len(body["ticket"]) >= 32
            with client.websocket_connect(f"/v1/ws?ticket={body['ticket']}") as ws:
                assert ws.receive_text() == "ok"
            with pytest.raises(WebSocketDisconnect) as reused:
                with client.websocket_connect(f"/v1/ws?ticket={body['ticket']}"):
                    pytest.fail("a used ticket opened a second connection")
            assert reused.value.code == 4001

    def test_header_and_deprecated_query_key_still_work(self):
        with _security_client(os_api_key="secret123") as client:
            with client.websocket_connect("/v1/ws", headers=GOOD_KEY) as ws:
                assert ws.receive_text() == "ok"
            with client.websocket_connect("/v1/ws?api_key=secret123") as ws:
                assert ws.receive_text() == "ok"

    def test_tickets_expire_after_60_seconds_and_stay_bounded(self):
        import src.middleware as mw

        now = [1000.0]
        with patch.object(mw, "time", SimpleNamespace(monotonic=lambda: now[0])):
            tickets = mw.WebSocketTickets(max_tickets=3)
            expired = tickets.issue()
            now[0] += 61
            assert tickets.redeem(expired) is False
            issued = [tickets.issue() for _ in range(5)]
            assert tickets.redeem(issued[0]) is False  # the oldest made room
            assert tickets.redeem(issued[-1]) is True
            assert tickets.redeem("made-up") is False


class TestFailedAuthThrottling:
    def test_ten_wrong_keys_lock_the_client_out(self):
        with _security_client(os_api_key="secret123") as client:
            wrong = [client.get("/v1/ping", headers={"Authorization": "Bearer guess"}).status_code for _ in range(10)]
            locked = client.get("/v1/ping", headers=GOOD_KEY)
        assert wrong == [401] * 10
        assert locked.status_code == 429
        assert 0 < int(locked.headers["Retry-After"]) <= 60

    def test_lockout_ends_after_a_minute_and_missing_keys_do_not_count(self):
        import src.middleware as mw

        now = [1000.0]
        with _security_client(os_api_key="secret123", os_auth_failure_limit=3) as client, \
             patch.object(mw, "time", SimpleNamespace(monotonic=lambda: now[0])):
            # A client that is not set up yet guesses nothing.
            assert [client.get("/v1/ping").status_code for _ in range(5)] == [401] * 5
            assert client.get("/v1/ping", headers=GOOD_KEY).status_code == 200
            for _ in range(3):
                client.get("/v1/ping", params={"api_key": "guess"})
            assert client.get("/v1/ping", headers=GOOD_KEY).status_code == 429
            now[0] += 61
            assert client.get("/v1/ping", headers=GOOD_KEY).status_code == 200

    def test_zero_turns_the_lockout_off(self):
        with _security_client(os_api_key="secret123", os_auth_failure_limit=0) as client:
            codes = {client.get("/v1/ping", headers={"Authorization": "Bearer guess"}).status_code for _ in range(15)}
            assert codes == {401}
            assert client.get("/v1/ping", headers=GOOD_KEY).status_code == 200

    def test_wrong_websocket_keys_count_and_the_lockout_covers_handshakes(self):
        with _security_client(os_api_key="secret123", os_auth_failure_limit=2) as client:
            for _ in range(2):
                with pytest.raises(WebSocketDisconnect):
                    with client.websocket_connect("/v1/ws?api_key=guess"):
                        pytest.fail("a wrong key opened a connection")
            with pytest.raises(WebSocketDenialResponse) as refused:
                with client.websocket_connect("/v1/ws", headers=GOOD_KEY):
                    pytest.fail("a locked-out client opened a connection")
            assert refused.value.status_code == 429
            assert client.get("/v1/ping", headers=GOOD_KEY).status_code == 429


class TestWebSocketHandshakeChecks:
    def test_handshakes_share_the_per_ip_rate_limit(self):
        # One token a minute: a slow runner cannot refill the bucket between handshakes.
        with _security_client(os_rate_limit=1, os_rate_limit_burst=1) as client:
            with client.websocket_connect("/v1/ws") as ws:
                assert ws.receive_text() == "ok"
            with pytest.raises(WebSocketDenialResponse) as refused:
                with client.websocket_connect("/v1/ws"):
                    pytest.fail("a rate-limited client opened a connection")
            assert refused.value.status_code == 429
            assert "retry-after" in refused.value.headers
            assert client.get("/v1/ping").status_code == 429

    def test_foreign_origin_and_unknown_host_are_refused_before_the_route_runs(self):
        with _security_client() as client:
            # A foreign Origin is accepted and closed at once with
            # 1008 and the reason, so the page can say why; the route never runs (it would send "ok").
            with client.websocket_connect("/v1/ws", headers={"origin": "https://evil.example"}) as ws:
                foreign = ws.receive()
            with pytest.raises(WebSocketDenialResponse) as rebinding:
                with client.websocket_connect("/v1/ws", headers={"host": "rebind.attacker.example"}):
                    pytest.fail("a rebinding page opened a connection")
            with client.websocket_connect("/v1/ws", headers={"origin": "http://testserver"}) as ws:
                assert ws.receive_text() == "ok"
        assert foreign == {"type": "websocket.close", "code": 1008, "reason": _ORIGIN_REFUSED}
        assert rebinding.value.status_code == 400
        with _security_client(os_ws_allowed_origins="https://app.example") as client:
            with client.websocket_connect("/v1/ws", headers={"origin": "https://app.example"}) as ws:
                assert ws.receive_text() == "ok"


_LIVE_READER = "/v1/audio/speech/stream"
_STT_STREAM = "/v1/audio/stream"
_ORIGIN_REFUSED = "Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it."
_KEY_REFUSED = "Invalid or missing API key"
_ENDPOINTING_REFUSED = "Invalid endpointing: must be a whole number of milliseconds, 0-30000"


class TestRefusedWebSocketsSayWhy:
    """A WebSocket closed before accept() reaches a
    browser as close code 1006 with no reason, so the web app could only say "didn't open". A refused Live Reader
    or STT stream socket is accepted and then closed with the code and reason: 4004 the feature is off, 1008 the
    Origin is not allowed, 4001 a bad or missing API key or ticket. No session starts for a refused socket."""

    @pytest.fixture
    def app_client(self, monkeypatch):
        import src.live_tts.server as live_server
        import src.main as main_module
        import src.routes.streaming as streaming_routes
        from src.config import settings

        for name, value in {
            "os_api_key": "", "os_ws_allowed_origins": "", "os_allowed_hosts": "", "os_rate_limit": 0,
            "tts_enabled": True, "tts_live_enabled": True,
        }.items():
            monkeypatch.setattr(settings, name, value)
        started: list[str] = []

        async def live_session(websocket, **_kwargs):
            started.append(_LIVE_READER)
            await websocket.accept()
            await websocket.send_text("session")
            await websocket.close()

        async def stt_session(websocket, **_kwargs):
            started.append(_STT_STREAM)
            await websocket.accept()
            await websocket.send_text("session")
            await websocket.close()

        monkeypatch.setattr(live_server, "live_tts_endpoint", live_session)
        monkeypatch.setattr(streaming_routes, "streaming_endpoint", stt_session)
        client = TestClient(main_module.create_app())  # its middleware and routes; no lifespan, no models
        return SimpleNamespace(client=client, settings=settings, started=started)

    @staticmethod
    def _accepted_then_closed(client, path, **kwargs) -> dict:
        """The first message after the handshake. Entering the block means the socket was accepted."""
        with client.websocket_connect(path, **kwargs) as ws:
            return ws.receive()

    @pytest.mark.parametrize("path", [_LIVE_READER, _STT_STREAM])
    def test_a_foreign_origin_is_accepted_and_closed_with_1008(self, app_client, path):
        closed = self._accepted_then_closed(app_client.client, path, headers={"origin": "https://evil.example"})
        assert closed == {"type": "websocket.close", "code": 1008, "reason": _ORIGIN_REFUSED}
        assert app_client.started == []

    @pytest.mark.parametrize("path", [_LIVE_READER, _STT_STREAM])
    @pytest.mark.parametrize("credential", ["none", "wrong-key", "used-ticket"])
    def test_a_missing_or_wrong_key_is_accepted_and_closed_with_4001(self, app_client, monkeypatch, path, credential):
        monkeypatch.setattr(app_client.settings, "os_api_key", "fake-key-for-websocket-tests")
        query = {"none": "", "wrong-key": "?api_key=guess", "used-ticket": "?ticket=never-issued"}[credential]
        closed = self._accepted_then_closed(app_client.client, path + query)
        assert closed == {"type": "websocket.close", "code": 4001, "reason": _KEY_REFUSED}
        assert app_client.started == []

    @pytest.mark.parametrize(("refusal", "code", "reason"), [
        ("middleware-origin", 1008, _ORIGIN_REFUSED), ("key", 4001, _KEY_REFUSED),
        ("realtime-off", 4004, "Realtime API is disabled"),
    ])
    @pytest.mark.parametrize("offered", [["realtime"], None], ids=["offers-realtime", "offers-none"])
    def test_a_refusal_accepts_with_the_offered_subprotocol(self, app_client, monkeypatch, refusal, code, reason,
                                                            offered):
        """A refusal accepted with no subprotocol although the client offered one, and a
        browser fails such a handshake (WHATWG WebSockets, "establish a WebSocket connection"): the JS SDK's
        RealtimeSession, which offers "realtime", saw 1006 and never the code and reason."""
        monkeypatch.setattr(app_client.settings, "os_realtime_enabled", refusal != "realtime-off")
        if refusal == "key":
            monkeypatch.setattr(app_client.settings, "os_api_key", "fake-key-for-subprotocol-test")
        headers = {"origin": "https://evil.example"} if refusal == "middleware-origin" else {}
        with app_client.client.websocket_connect("/v1/realtime", subprotocols=offered, headers=headers) as ws:
            assert ws.accepted_subprotocol == (offered[0] if offered else None)
            assert ws.receive() == {"type": "websocket.close", "code": code, "reason": reason}

    @pytest.mark.parametrize(("query", "reason"), [
        ("sample_rate=16k", "Invalid sample_rate: must be 8000-192000"),
        ("sample_rate=", "Invalid sample_rate: must be 8000-192000"),
        ("sample_rate=16000.5", "Invalid sample_rate: must be 8000-192000"),
        ("endpointing=abc", _ENDPOINTING_REFUSED),
        ("vad=maybe", "Invalid vad: must be true or false"),
        ("interim_results=2", "Invalid interim_results: must be true or false"),
    ])
    def test_a_stream_query_it_cant_parse_is_accepted_and_closed_with_1008(self, app_client, monkeypatch, query,
                                                                           reason):
        """Typed query parameters made FastAPI close the socket before accept(), with a
        list of errors as the reason, so a browser saw 1006; and the Origin and key checks never ran."""
        closed = self._accepted_then_closed(app_client.client, f"{_STT_STREAM}?{query}")
        assert closed == {"type": "websocket.close", "code": 1008, "reason": reason}
        assert app_client.started == []
        monkeypatch.setattr(app_client.settings, "os_api_key", "fake-key-checked-first")  # the key is checked first
        closed = self._accepted_then_closed(app_client.client, f"{_STT_STREAM}?{query}")
        assert closed == {"type": "websocket.close", "code": 4001, "reason": _KEY_REFUSED}

    def test_a_stream_query_it_can_parse_still_reaches_the_session(self, app_client, monkeypatch):
        import src.routes.streaming as streaming_routes

        received = {}

        async def stt_session(websocket, **kwargs):
            received.update(kwargs)
            await websocket.accept()
            await websocket.close()

        monkeypatch.setattr(streaming_routes, "streaming_endpoint", stt_session)
        query = "sample_rate=16000.0&interim_results=0&endpointing=500&vad=on&model=m&language=en&encoding=pcm_s16le"
        with app_client.client.websocket_connect(f"{_STT_STREAM}?{query}") as ws:
            ws.receive()
        assert received == {"model": "m", "language": "en", "sample_rate": 16000, "encoding": "pcm_s16le",
                            "interim_results": False, "endpointing": 500, "vad": True}
        received.clear()
        with app_client.client.websocket_connect(_STT_STREAM) as ws:
            ws.receive()
        assert received == {"model": None, "language": None, "sample_rate": 16000, "encoding": "pcm_s16le",
                            "interim_results": True, "endpointing": None, "vad": None}

    @pytest.mark.parametrize("endpointing", ["1" + "0" * 400, "-1", "30001"], ids=["401-digits", "negative", "30001"])
    def test_a_stream_endpointing_out_of_range_is_accepted_and_closed_with_1008(self, app_client, endpointing):
        """The route took any whole number, and a huge one raised OverflowError in
        StreamingSession after accept(): no 1008 and no reason, only the server's internal-error close."""
        closed = self._accepted_then_closed(app_client.client, f"{_STT_STREAM}?endpointing={endpointing}")
        assert closed == {"type": "websocket.close", "code": 1008, "reason": _ENDPOINTING_REFUSED}
        assert app_client.started == []

    def test_a_stream_endpointing_at_the_ends_of_its_range_reaches_the_session(self, app_client, monkeypatch):
        import src.routes.streaming as streaming_routes

        received = []

        async def stt_session(websocket, **kwargs):
            received.append(kwargs["endpointing"])
            await websocket.accept()
            await websocket.close()

        monkeypatch.setattr(streaming_routes, "streaming_endpoint", stt_session)
        for endpointing in (0, 30000):
            with app_client.client.websocket_connect(f"{_STT_STREAM}?endpointing={endpointing}") as ws:
                ws.receive()
        assert received == [0, 30000]

    @pytest.mark.parametrize("switch", ["tts_enabled", "tts_live_enabled"])
    def test_the_live_reader_turned_off_is_accepted_and_closed_with_4004(self, app_client, monkeypatch, switch):
        monkeypatch.setattr(app_client.settings, switch, False)
        closed = self._accepted_then_closed(app_client.client, _LIVE_READER)
        assert closed == {"type": "websocket.close", "code": 4004, "reason": "Live TTS is disabled"}
        assert app_client.started == []

    @pytest.mark.parametrize("path", [_LIVE_READER, _STT_STREAM])
    def test_an_allowed_socket_still_opens_its_session(self, app_client, monkeypatch, path):
        monkeypatch.setattr(app_client.settings, "os_api_key", "fake-key-for-websocket-tests")
        headers = {"origin": "http://testserver", "authorization": "Bearer fake-key-for-websocket-tests"}
        with app_client.client.websocket_connect(path, headers=headers) as ws:
            assert ws.receive_text() == "session"
        assert app_client.started == [path]

    def test_an_unknown_host_is_still_refused_in_place_of_the_handshake(self, app_client):
        # Unlike the refusals above, a DNS-rebinding Host stays an HTTP 400 answer to the handshake.
        with pytest.raises(WebSocketDenialResponse) as rebinding:
            with app_client.client.websocket_connect(_LIVE_READER, headers={"host": "rebind.attacker.example"}):
                pytest.fail("a rebinding page opened a connection")
        assert rebinding.value.status_code == 400
        assert app_client.started == []


class TestRequestBodyLimit:
    """OS_MAX_UPLOAD_MB and READER_MAX_UPLOAD_MB bound what the server reads, not only what it keeps."""

    async def test_a_declared_oversized_body_is_refused_unread(self):
        with _security_client(os_max_upload_mb=1) as client:
            status, pulled = await _call_asgi(client.app, {
                "method": "POST", "path": "/v1/echo",
                "headers": [(b"host", b"testserver"), (b"content-length", str(3 * MB).encode())],
            }, [b"x" * (64 * 1024)] * 48)
        assert (status, pulled) == (413, 0)

    async def test_a_chunked_body_is_cut_off_once_it_passes_the_limit(self):
        chunk = 64 * 1024
        with _security_client(os_max_upload_mb=1) as client:
            status, pulled = await _call_asgi(client.app, {
                "method": "POST", "path": "/v1/echo", "headers": [(b"host", b"testserver")],
            }, [b"x" * chunk] * 64)  # 4 MB without a Content-Length
        assert status == 413
        assert pulled <= 2 * MB + chunk

    @pytest.mark.parametrize(("path", "content_type", "expected"), [
        ("/v1/reader/documents", "multipart/form-data; boundary=x", 3 * MB + MB),
        ("/v1/audio/transcriptions/batch", "multipart/form-data; boundary=x", 20 * MB + MB),
        ("/v1/audio/transcriptions", "multipart/form-data; boundary=x", 10 * MB + MB),
        ("/v1/audio/speech", "application/json", 10 * MB * 4 // 3 + 64 * 1024),
    ])
    def test_each_route_family_gets_its_configured_limit(self, path, content_type, expected):
        from src.middleware import _body_limit

        limits = Settings(_env_file=None, os_max_upload_mb=10, os_batch_max_total_mb=20, reader_max_upload_mb=3)
        with patch("src.middleware.settings", limits):
            assert _body_limit(path, content_type)[0] == expected

    def test_json_bodies_get_room_for_base64(self):
        with _security_client(os_max_upload_mb=6) as client:
            body = b"x" * int(7.5 * MB)
            assert client.post("/v1/echo", content=body, headers={"content-type": "application/json"}).status_code == 200
            assert client.post("/v1/echo", content=body, headers={"content-type": "audio/wav"}).status_code == 413
            too_big = b"x" * (8 * MB + 128 * 1024)
            assert client.post("/v1/echo", content=too_big, headers={"content-type": "application/json"}).status_code == 413

    @pytest.mark.parametrize(("path", "limit", "files", "data"), [
        ("/v1/audio/transcriptions", {"max_upload_mb": 1}, {"file": ("a.wav", b"\0" * (4 * MB), "audio/wav")},
         {"model": "test-model"}),
        ("/v1/reader/documents", {"reader_max_upload_mb": 1}, {"file": ("book.txt", b"a" * (4 * MB), "text/plain")},
         None),
    ], ids=["transcription", "reader-import"])
    def test_oversized_multipart_uploads_are_not_spooled(self, monkeypatch, path, limit, files, data):
        from starlette.datastructures import UploadFile

        written: list[int] = []
        original_write = UploadFile.write

        async def counting_write(self, chunk):
            written.append(len(chunk))
            return await original_write(self, chunk)

        monkeypatch.setattr(UploadFile, "write", counting_write)
        with _make_client(**limit) as client:
            resp = client.post(path, files=files, data=data)
        assert resp.status_code == 413
        assert sum(written) <= 2 * MB, f"spooled {sum(written):,} bytes of a 4 MB upload with a 1 MB limit"

    async def test_a_chunked_reader_import_is_cut_off_once_it_passes_the_limit(self):
        boundary = b"limitboundary"
        head = (
            b"--" + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="big.txt"\r\n'
            b"Content-Type: text/plain\r\n\r\n"
        )
        chunk = 64 * 1024
        pieces = [head] + [b"a" * chunk] * 96 + [b"\r\n--" + boundary + b"--\r\n"]  # 6 MB
        with _make_client(reader_max_upload_mb=1) as client:
            status, pulled = await _call_asgi(client.app, {
                "method": "POST", "path": "/v1/reader/documents",
                "headers": [(b"host", b"testserver"), (b"content-type", b"multipart/form-data; boundary=" + boundary)],
            }, pieces)
        assert status == 413
        assert pulled <= 2 * MB + chunk


# Pydantic makes one error per invalid list item, and validation_exception_handler
# formatted every one with str(exc) on the event loop (FastAPI builds that string with += in a loop). A 200 KB body
# of zeros froze the server for about 8 s and answered a 16 MB 422.
class _UnboundedListItem(BaseModel):
    name: str


class _UnboundedListBody(BaseModel):
    items: list[_UnboundedListItem]


# The web UI reads each error's field and message from this repr (src/static/js/core/api.js, parseValidation).
_UI_VALIDATION_PATTERN = re.compile(r"""'loc':\s*\(([^)]*)\)[^{}]*?'msg':\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")""")
_EFFECT = {"type": "normalize"}
_TURN = {"speaker": "a", "text": "b"}
_PROFILE = {"name": "p", "backend": "kokoro", "voice": "af_heart"}


def _zeros(n: int) -> str:
    return "[" + ",".join(["0"] * n) + "]"


class TestValidationErrorsStayBounded:

    @pytest.fixture
    def secure_defaults(self, monkeypatch):
        # The documented defaults, whatever a local .env says: no API key.
        from src.config import settings

        monkeypatch.setattr(settings, "os_api_key", "")
        return settings

    @staticmethod
    def _app():
        import src.main as main_module

        app = main_module.create_app()  # its exception handlers and middleware; no lifespan, no models

        @app.post("/api/test-unbounded-list")
        def unbounded(payload: _UnboundedListBody):
            return {"items": len(payload.items)}

        return app

    async def test_a_long_list_of_invalid_items_does_not_stall_the_event_loop(self, secure_defaults):
        app = self._app()
        body = ('{"name":"x","turns":' + _zeros(100_000) + "}").encode()  # 200 KB
        gaps: list[float] = []
        done = asyncio.Event()

        async def tick() -> None:
            # The gap is recorded before `done` is checked, so a block that lasts until the response is measured.
            last = time.perf_counter()
            while True:
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now
                if done.is_set():
                    return

        ticker = asyncio.create_task(tick())
        await asyncio.sleep(0.05)  # ASGITransport runs the app inline in this task: the ticker starts first
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                resp = await asyncio.wait_for(
                    client.post("/api/conversations", content=body, headers={"content-type": "application/json"}),
                    timeout=120,
                )
        finally:
            done.set()
            await asyncio.wait_for(ticker, timeout=5)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"
        assert max(gaps) < 0.5, f"event loop blocked for {max(gaps):.2f} s by a {len(body)} B JSON body"
        assert len(resp.content) < len(body) // 10, f"a {len(resp.content)} B 422 for a {len(body)} B body"

    def test_the_422_names_the_first_errors_and_counts_the_rest(self, secure_defaults):
        # A list no model bounds (a test route): the work and the message stay the same size for any count.
        body = ('{"items":' + _zeros(100_000) + "}").encode()
        resp = TestClient(self._app()).post(
            "/api/test-unbounded-list", content=body, headers={"content-type": "application/json"}
        )
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        message = error["message"]
        assert len(message) < 4096, f"a {len(message)} character message for a {len(body)} B body"
        lines = message.splitlines()
        assert (lines[0], lines[-1]) == ("100000 validation errors:", "  ... and 99990 more")
        # The first ten keep the repr the web UI reads; the message names no source file.
        fields = [(match.group(1), match.group(2)) for match in _UI_VALIDATION_PATTERN.finditer(message)]
        expected = "Input should be a valid dictionary or object to extract fields from"  # FastAPI: from_attributes
        assert fields == [(f"'body', 'items', {index}", expected) for index in range(10)]
        assert "test_security.py" not in message

    def test_a_long_input_is_clipped_in_the_message(self, secure_defaults):
        # A too_long error carries the whole list as its input, and a missing field the whole body.
        client = TestClient(self._app())
        long_list = ('{"name":"x","turns":' + _zeros(20_000) + "}").encode()
        long_text = ('{"backend":"kokoro","voice":"af_heart","instructions":"' + "x" * 100_000 + '"}').encode()
        for path, body, kind in (("/api/conversations", long_list, "too_long"), ("/api/profiles", long_text, "missing")):
            resp = client.post(path, content=body, headers={"content-type": "application/json"})
            assert resp.status_code == 422, resp.text[:300]
            message = resp.json()["error"]["message"]
            assert message.startswith("1 validation error:") and f"'type': '{kind}'" in message, message[:300]
            assert len(message) < 4096, f"{path}: a {len(message)} character message for a {len(body)} B body"

    @pytest.mark.parametrize(("method", "path", "build", "limit", "loc", "breaker"), [
        ("POST", "/api/conversations", lambda n: {"name": "x", "turns": [_TURN] * n}, 500,
         "'body', 'turns'", ("name", 5)),
        ("POST", "/api/conversations", lambda n: {"name": "x", "turns": [{**_TURN, "effects": [_EFFECT] * n}]}, 16,
         "'body', 'turns', 0, 'effects'", ("name", 5)),
        ("POST", "/api/conversations/c1/turns", lambda n: {**_TURN, "effects": [_EFFECT] * n}, 16,
         "'body', 'effects'", ("speaker", 5)),
        ("POST", "/api/profiles", lambda n: {**_PROFILE, "effects": [_EFFECT] * n}, 16,
         "'body', 'effects'", ("name", 5)),
        ("PUT", "/api/profiles/p1", lambda n: {**_PROFILE, "effects": [_EFFECT] * n}, 16,
         "'body', 'effects'", ("name", 5)),
        ("POST", "/api/composer/render", lambda n: {"tracks": [{"source_path": "a.wav", "effects": [_EFFECT] * n}]}, 16,
         "'body', 'tracks', 0, 'effects'", ("sample_rate", 1)),
        ("POST", "/v1/audio/speech", lambda n: {"input": "hi", "effects": [_EFFECT] * n}, 16,
         "'body', 'effects'", ("input", 5)),
    ], ids=["conversation-turns", "conversation-turn-effects", "turn-effects", "profile-effects",
            "profile-update-effects", "composer-track-effects", "speech-effects"])
    def test_json_lists_are_bounded(self, secure_defaults, method, path, build, limit, loc, breaker):
        # One more item than the limit is refused with one too_long error, whatever the items are. Another field
        # is made invalid in both requests, so the route itself never runs.
        field, bad = breaker
        client = TestClient(self._app())
        over = client.request(method, path, json={**build(limit + 1), field: bad})
        assert over.status_code == 422, over.text[:300]
        message = over.json()["error"]["message"]
        assert message.startswith("2 validation errors:"), message[:300]
        assert f"'type': 'too_long', 'loc': ({loc})" in message, message[:300]
        at_limit = client.request(method, path, json={**build(limit), field: bad})
        assert at_limit.status_code == 422, at_limit.text[:300]
        message = at_limit.json()["error"]["message"]
        assert message.startswith("1 validation error:"), message[:300]
        assert f"'loc': ('body', '{field}')" in message, message[:300]
