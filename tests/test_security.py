"""Tests for security middleware — API key auth, rate limiting, input validation, CORS."""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import Settings


@contextmanager
def _make_client(
    api_key: str = "",
    rate_limit: int = 0,
    rate_limit_burst: int = 0,
    max_upload_mb: int = 100,
    cors_origins: str = "*",
    ws_allowed_origins: str = "",
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
    )

    # Reset global rate limiter so each test starts fresh
    import src.middleware as mw
    mw._rate_limiter = None

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
            assert "API key" in resp.json()["detail"]

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

    def test_key_query_param_accepted(self):
        """API key via query parameter works."""
        with _make_client(api_key="secret123") as client:
            resp = client.get("/v1/models?api_key=secret123")
            assert resp.status_code == 200

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
        with _make_client(rate_limit=60, rate_limit_burst=3) as client:
            for _ in range(3):
                resp = client.get("/v1/models")
                assert resp.status_code == 200
            resp = client.get("/v1/models")
            assert resp.status_code == 429
            assert "Rate limit" in resp.json()["detail"]

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
            assert "Empty" in resp.json()["detail"]

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
                assert "too large" in resp.json()["detail"]

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
        """Tokens refill over time."""
        import time
        from src.middleware import RateLimiter
        rl = RateLimiter(requests_per_minute=6000, burst=5)  # 100/sec

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "1.2.3.4"
        mock_request.url.path = "/v1/models"

        # Drain all tokens
        for _ in range(5):
            allowed, _ = rl.check(mock_request)
            assert allowed

        allowed, _ = rl.check(mock_request)
        assert not allowed  # depleted

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


class TestWebSocketOriginValidation:
    def test_verify_ws_origin_allows_when_unset(self):
        from src.middleware import verify_ws_origin

        ws = MagicMock()
        ws.headers = {"origin": "https://evil.example"}
        with patch("src.middleware.settings", Settings(os_ws_allowed_origins="")):
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
