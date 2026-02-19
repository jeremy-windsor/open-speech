"""Security middleware — API key auth, rate limiting, input validation, CORS."""

from __future__ import annotations

import hmac
import logging
import random
import time
from typing import Callable

from fastapi import Request, Response, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

# Paths that never require auth (health, docs, web UI static assets)
AUTH_EXEMPT_PATHS = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/web",
})


def _is_auth_exempt(path: str) -> bool:
    """Check if a path is exempt from API key auth."""
    if path in AUTH_EXEMPT_PATHS:
        return True
    # Static assets under /web/
    if path.startswith("/web/") or path.startswith("/static/"):
        return True
    return False


def verify_api_key(request: Request) -> None:
    """Verify the API key from Authorization header or query param.

    Only enforced when STT_API_KEY is set. Raises HTTPException on failure.
    """
    if not settings.stt_api_key:
        return  # Auth disabled

    if _is_auth_exempt(request.url.path):
        return

    # Check Authorization: Bearer <key>
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if hmac.compare_digest(token, settings.stt_api_key):
            return

    # Check query param ?api_key=<key> (for WebSocket clients)
    query_key = request.query_params.get("api_key")
    if query_key and hmac.compare_digest(query_key, settings.stt_api_key):
        logger.warning("API key in query string is deprecated — use Authorization: Bearer header")
        return

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key. Set Authorization: Bearer <key> header.",
    )


def verify_ws_api_key(websocket: WebSocket) -> bool:
    """Verify API key for WebSocket connections. Returns True if valid."""
    if not settings.stt_api_key:
        return True

    # Check query param
    query_key = websocket.query_params.get("api_key")
    if query_key and hmac.compare_digest(query_key, settings.stt_api_key):
        logger.warning("API key in query string is deprecated — use Authorization: Bearer header")
        return True

    # Check header (some WS clients support this)
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if hmac.compare_digest(token, settings.stt_api_key):
            return True

    return False


def _allowed_ws_origins() -> set[str]:
    raw = settings.os_ws_allowed_origins.strip()
    if not raw:
        return set()
    return {o.strip() for o in raw.split(",") if o.strip()}


def verify_ws_origin(websocket: WebSocket) -> bool:
    """Validate WebSocket Origin header when allowlist is configured."""
    allowed = _allowed_ws_origins()
    if not allowed:
        return True

    origin = websocket.headers.get("origin", "")
    return origin in allowed



# ---------------------------------------------------------------------------
# Rate Limiting (token bucket per IP)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory token bucket rate limiter per IP address."""

    def __init__(self, requests_per_minute: int, burst: int | None = None):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = burst or requests_per_minute  # max bucket size
        self._buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_time)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP. Only trusts X-Forwarded-For when STT_TRUST_PROXY=true."""
        if settings.stt_trust_proxy:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> tuple[bool, dict[str, str]]:
        """Check if request is allowed. Returns (allowed, headers)."""
        ip = self._get_client_ip(request)
        now = time.monotonic()

        if ip in self._buckets:
            tokens, last_time = self._buckets[ip]
            elapsed = now - last_time
            tokens = min(self.burst, tokens + elapsed * self.rate)
        else:
            tokens = float(self.burst)

        headers = {
            "X-RateLimit-Limit": str(self.burst),
            "X-RateLimit-Remaining": str(max(0, int(tokens) - 1)),
        }

        if tokens >= 1.0:
            self._buckets[ip] = (tokens - 1.0, now)
            allowed = True
            headers["X-RateLimit-Remaining"] = str(max(0, int(tokens) - 1))
        else:
            self._buckets[ip] = (tokens, now)
            retry_after = (1.0 - tokens) / self.rate
            headers["Retry-After"] = str(int(retry_after) + 1)
            headers["X-RateLimit-Remaining"] = "0"
            allowed = False

        # Probabilistic cleanup (~1% of requests) to prevent unbounded growth
        if random.random() < 0.01:
            self.cleanup()

        return allowed, headers

    def cleanup(self, max_age: float = 3600.0) -> None:
        """Remove stale entries older than max_age seconds."""
        now = time.monotonic()
        stale = [ip for ip, (_, t) in self._buckets.items() if now - t > max_age]
        for ip in stale:
            del self._buckets[ip]


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter | None:
    """Get or create the rate limiter (lazy init)."""
    global _rate_limiter
    if settings.stt_rate_limit <= 0:
        return None
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=settings.stt_rate_limit,
            burst=settings.stt_rate_limit_burst or settings.stt_rate_limit,
        )
    return _rate_limiter


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------


# Note: Upload size validation is done in the endpoint handlers (main.py)
# after reading the actual bytes, not in middleware. This avoids the
# Content-Length header spoofing issue and handles chunked transfer correctly.


# ---------------------------------------------------------------------------
# Combined Security Middleware
# ---------------------------------------------------------------------------

class SecurityMiddleware(BaseHTTPMiddleware):
    """Combined middleware: auth → rate limit → validation."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip for WebSocket upgrades (handled separately in the endpoint)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # 1. API Key Auth
        try:
            verify_api_key(request)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"message": e.detail}},
            )

        # 2. Rate Limiting
        rl_headers: dict[str, str] = {}
        limiter = get_rate_limiter()
        if limiter and not _is_auth_exempt(request.url.path):
            allowed, rl_headers = limiter.check(request)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": {"message": "Rate limit exceeded. Try again later."}},
                    headers=rl_headers,
                )

        # Process request (upload validation in endpoint handlers)
        response = await call_next(request)

        # Attach rate limit headers to successful responses
        for k, v in rl_headers.items():
            response.headers[k] = v

        return response
