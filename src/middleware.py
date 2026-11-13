"""Security middleware — Host and Origin checks, CORS, API key auth, rate limiting, body size, headers."""

from __future__ import annotations

import hmac
import ipaddress
import logging
import math
import random
import re
import secrets
import socket
import time
from collections import OrderedDict, deque
from functools import lru_cache
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.config import settings
from src.reader.media import MEDIA_PREFIX as READER_MEDIA_PREFIX

logger = logging.getLogger(__name__)

MB = 1024 * 1024


def _routed_path(scope: Scope) -> str:
    """The path the router matches: scope["path"] without root_path.

    Never use request.url for security decisions: Starlette rebuilds it from the Host header,
    so a Host such as "lan-box/web/?" would make any route look like /web/.
    """
    path = scope.get("path", "")
    root_path = scope.get("root_path", "")
    if root_path and path.startswith(root_path) and path[len(root_path):len(root_path) + 1] in ("", "/"):
        return path[len(root_path):]
    return path


# (status code, message, extra headers) of a refused request.
_Refusal = tuple[int, str, "dict[str, str] | None"]


def _error(status_code: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"message": message}}, headers=headers)


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
    # Reader audio links carry their own expiring signature (checked by the route),
    # because an HTML audio element cannot send an Authorization header.
    if path.startswith(READER_MEDIA_PREFIX):
        return True
    return False


def _key_matches(candidate: str) -> bool:
    # compare_digest rejects non-ASCII str with TypeError; compare the encoded bytes.
    return hmac.compare_digest(candidate.encode(), settings.stt_api_key.encode())


def _bearer_token(auth_header: str) -> str | None:
    # The auth-scheme is case-insensitive (RFC 7235 section 2.1).
    scheme, _, token = auth_header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else None


def _credential_status(conn: HTTPConnection) -> bool | None:
    """True for a valid API key, False for a wrong one, None when the request carries none."""
    auth_header = conn.headers.get("authorization", "")
    token = _bearer_token(auth_header)
    if token is not None and _key_matches(token):
        return True

    # Check query param ?api_key=<key> (for WebSocket clients)
    query_key = conn.query_params.get("api_key")
    if query_key and _key_matches(query_key):
        logger.warning("API key in query string is deprecated — use Authorization: Bearer header")
        return True

    return False if auth_header or query_key else None


def verify_api_key(request: Request) -> None:
    """Verify the API key from Authorization header or query param.

    Only enforced when STT_API_KEY is set. Raises HTTPException on failure.
    """
    if not settings.stt_api_key:
        return  # Auth disabled

    if _is_auth_exempt(_routed_path(request.scope)):
        return

    status = _credential_status(request)
    if not status:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set Authorization: Bearer <key> header.",
            headers=_bearer_challenge(status),
        )


# ---------------------------------------------------------------------------
# WebSocket tickets (browsers cannot set an Authorization header on a WebSocket)
# ---------------------------------------------------------------------------

WS_TICKET_TTL_S = 60


class WebSocketTickets:
    """Single-use, short-lived WebSocket credentials, kept in memory."""

    def __init__(self, ttl_s: float = WS_TICKET_TTL_S, max_tickets: int = 1000):
        self.ttl_s = ttl_s
        self.max_tickets = max_tickets
        # Every ticket lives equally long, so insertion order is also expiry order.
        self._expiry: OrderedDict[str, float] = OrderedDict()

    def issue(self) -> str:
        now = time.monotonic()
        while self._expiry and (len(self._expiry) >= self.max_tickets or next(iter(self._expiry.values())) <= now):
            self._expiry.popitem(last=False)
        ticket = secrets.token_urlsafe(32)
        self._expiry[ticket] = now + self.ttl_s
        return ticket

    def redeem(self, ticket: str) -> bool:
        expiry = self._expiry.pop(ticket, None)
        return expiry is not None and time.monotonic() < expiry

    def clear(self) -> None:
        self._expiry.clear()


_ws_tickets = WebSocketTickets()


def create_auth_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/auth/ws-ticket")
    async def issue_ws_ticket():
        """A one-time ?ticket= for a WebSocket connection; the middleware has already checked the API key."""
        return {"ticket": _ws_tickets.issue(), "expires_in": WS_TICKET_TTL_S}

    return router


def verify_ws_api_key(websocket: WebSocket) -> bool:
    """Verify API key for WebSocket connections. Returns True if valid."""
    if not settings.stt_api_key:
        return True

    ticket = websocket.query_params.get("ticket")
    if ticket and _ws_tickets.redeem(ticket):
        return True

    # Check query param
    query_key = websocket.query_params.get("api_key")
    if query_key and _key_matches(query_key):
        logger.warning("API key in query string is deprecated — use Authorization: Bearer header")
        return True

    # Check header (some WS clients support this)
    auth_header = websocket.headers.get("authorization", "")
    token = _bearer_token(auth_header)
    if token is not None and _key_matches(token):
        return True

    failures = get_auth_failure_limiter()
    if failures and (ticket or query_key or auth_header):
        failures.record_failure(client_ip(websocket))
    return False


# ---------------------------------------------------------------------------
# Host and Origin checks (DNS rebinding, cross-site requests)
# ---------------------------------------------------------------------------

# RFC 9110 7.2: uri-host [ ":" port ]. Anything else (a path, "?", "#", "@", spaces) is malformed.
# An empty port is allowed and means the default one (RFC 3986 3.2.3).
_HOST_RE = re.compile(
    r"(?:\[(?P<ipv6>[0-9A-Fa-f:.]+)\]|(?P<name>[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?\.?))"
    r"(?::(?P<port>[0-9]{0,5}))?"
)
# Names a public DNS rebinding attack cannot use.
_LAN_SUFFIXES = (".localhost", ".local", ".lan", ".home.arpa", ".internal")
_DEFAULT_PORTS = {"http": 80, "https": 443}
# One warning each, so scanner garbage cannot hide the refusal of a real name.
_host_warning_logged = False
_malformed_host_warning_logged = False


def _split_host(value: str) -> tuple[str, int | None] | None:
    """(hostname, port) of a Host header value, or None when it is malformed."""
    match = _HOST_RE.fullmatch(value)
    if match is None:
        return None
    port = int(match["port"]) if match["port"] else None
    if port is not None and port > 65535:
        return None
    if match["ipv6"] is not None:
        try:
            hostname = str(ipaddress.IPv6Address(match["ipv6"]))
        except ValueError:
            return None
    else:
        hostname = match["name"].lower().rstrip(".")
    return hostname, port


@lru_cache(maxsize=1)
def _machine_names() -> frozenset[str]:
    """This machine's hostname and FQDN. getfqdn() may wait on reverse DNS, so SecurityMiddleware
    looks them up in a worker thread at startup, before the event loop serves requests."""
    names = set()
    for lookup in (socket.gethostname, socket.getfqdn):
        try:
            names.add(lookup().lower().rstrip("."))
        except OSError:
            pass
    return frozenset(name for name in names if name)


@lru_cache(maxsize=8)
def _allowed_host_patterns(raw: str) -> tuple[str, ...]:
    patterns = []
    for entry in raw.split(","):
        entry = entry.strip().lower()
        if entry.startswith("*") or not entry:
            patterns.append(entry)
            continue
        parsed = _split_host(entry)
        patterns.append(parsed[0] if parsed else entry)
    return tuple(pattern for pattern in patterns if pattern)


def _host_is_allowed(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    # Single-label names (LAN machine names, "localhost", "testserver") and LAN-only suffixes.
    if "." not in hostname or hostname.endswith(_LAN_SUFFIXES):
        return True
    for pattern in _allowed_host_patterns(settings.os_allowed_hosts):
        if hostname == pattern or (pattern.startswith("*.") and hostname.endswith(pattern[1:])):
            return True
    return hostname in _machine_names()


def _host_problem(scope: Scope) -> str | None:
    """Why the request's Host header is refused, or None when it is acceptable."""
    if "*" in _allowed_host_patterns(settings.os_allowed_hosts):
        return None
    hosts = [value.decode("latin-1") for name, value in scope.get("headers", ()) if name == b"host"]
    if not hosts:
        return None  # HTTP/1.0 without Host: not a browser, so not a DNS rebinding attack
    # RFC 9112 3.2: a request with more than one Host line or an invalid one gets 400.
    parsed = _split_host(hosts[0]) if len(hosts) == 1 else None
    if parsed is not None and _host_is_allowed(parsed[0]):
        return None
    global _host_warning_logged, _malformed_host_warning_logged
    if parsed is None:
        if not _malformed_host_warning_logged:
            _malformed_host_warning_logged = True
            logger.warning(
                "Refused a request with a malformed Host header %r. Later malformed Host headers are not logged.",
                ", ".join(hosts),
            )
        return "Invalid Host header"
    if not _host_warning_logged:
        _host_warning_logged = True
        logger.warning(
            "Refused a request for Host %r. Add the name to OS_ALLOWED_HOSTS (comma-separated; "
            "'*.example.com' works, '*' turns the check off) to serve it. Later refusals are not logged.",
            hosts[0],
        )
    return "Host not allowed. Add it to OS_ALLOWED_HOSTS on the server."


@lru_cache(maxsize=8)
def _origin_list(raw: str) -> tuple[str, ...]:
    return tuple(o.strip().rstrip("/").lower() for o in raw.split(",") if o.strip())


def _is_same_origin(origin: str, host_header: str | None) -> bool:
    """True when the browser's Origin names this server as the client addressed it (its Host)."""
    try:
        parts = urlsplit(origin)
        origin_port = parts.port
    except ValueError:
        return False
    host = _split_host(host_header) if host_header else None
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname or host is None:
        return False
    origin_host = parts.hostname.lower().rstrip(".")
    if ":" in origin_host:
        try:
            origin_host = str(ipaddress.IPv6Address(origin_host))
        except ValueError:
            return False
    default_port = _DEFAULT_PORTS[parts.scheme]
    return host[0] == origin_host and (host[1] or default_port) == (origin_port or default_port)


def _origin_allowed(origin: str, host_header: str | None, allowed: tuple[str, ...]) -> bool:
    return "*" in allowed or origin.rstrip("/").lower() in allowed or _is_same_origin(origin, host_header)


_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _is_cross_site_write(conn: HTTPConnection) -> bool:
    """A browser write sent by a page that is neither this server's own nor in OS_CORS_ORIGINS."""
    if conn.scope["method"] not in _STATE_CHANGING_METHODS:
        return False
    allowed = _origin_list(settings.os_cors_origins)
    origin = conn.headers.get("origin")
    if origin:
        return not _origin_allowed(origin, conn.headers.get("host"), allowed)
    # SDKs and curl send no Origin; a browser that omits it still reports a cross-site sender.
    return "*" not in allowed and conn.headers.get("sec-fetch-site", "").lower() == "cross-site"


def verify_ws_origin(websocket: WebSocket) -> bool:
    """Accept a WebSocket Origin that is this server's own or listed in OS_WS_ALLOWED_ORIGINS.

    Non-browser clients (SDK, Home Assistant, CLI tools) send no Origin and are accepted;
    browsers always send one.
    """
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    return _origin_allowed(origin, websocket.headers.get("host"), _origin_list(settings.os_ws_allowed_origins))


# A refused WebSocket is accepted and then closed with one of these codes and reasons: closed before accept(), a
# browser sees only 1006 and no reason. 4000-4999 are the application's codes (RFC 6455 section 7.4.2).
WS_CLOSE_FEATURE_OFF = 4004
WS_CLOSE_ORIGIN = 1008
WS_CLOSE_KEY = 4001
WS_ORIGIN_REFUSED = "Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it."
WS_KEY_REFUSED = "Invalid or missing API key"


def _first_offered_subprotocol(scope: Scope) -> str | None:
    """The first subprotocol the client offered, or None. A browser fails a handshake that offered some and whose
    answer names none (WHATWG WebSockets, "establish a WebSocket connection"), and then sees only 1006, so a refusal
    accepts with one of them."""
    return next((name for name in scope.get("subprotocols") or () if name), None)


async def refuse_websocket(websocket: WebSocket, code: int, reason: str) -> None:
    """Accept, then close at once with the code and reason; nothing is read or started."""
    await websocket.accept(subprotocol=_first_offered_subprotocol(websocket.scope))
    await websocket.close(code=code, reason=reason)


async def _accept_and_close(scope: Scope, receive: Receive, send: Send, code: int, reason: str) -> None:
    """refuse_websocket for the middleware, which has the ASGI calls and no WebSocket."""
    if (await receive())["type"] == "websocket.connect":
        await send({"type": "websocket.accept", "subprotocol": _first_offered_subprotocol(scope)})
        await send({"type": "websocket.close", "code": code, "reason": reason})


# ---------------------------------------------------------------------------
# Rate Limiting (token bucket per IP)
# ---------------------------------------------------------------------------

def client_ip(conn: HTTPConnection) -> str:
    """Client address. Only trusts X-Forwarded-For when OS_TRUST_PROXY=true."""
    client = getattr(conn, "client", None)
    peer = client.host if client else "unknown"
    if not settings.stt_trust_proxy:
        return peer
    # RFC 9110 5.3: repeated field lines are one comma-joined list. Each proxy appends the address
    # it saw, so the entry OS_TRUSTED_PROXY_HOPS from the right is the one the nearest trusted
    # proxy reported; entries left of it are client-supplied.
    entries = [
        entry.strip()
        for line in conn.headers.getlist("x-forwarded-for")
        for entry in line.split(",")
        if entry.strip()
    ]
    if not entries:
        return peer
    hops = max(1, settings.os_trusted_proxy_hops)
    return entries[-hops] if len(entries) >= hops else entries[0]


class RateLimiter:
    """Simple in-memory token bucket rate limiter per IP address."""

    def __init__(self, requests_per_minute: int, burst: int | None = None):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = burst or requests_per_minute  # max bucket size
        self._buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_time)

    def _get_client_ip(self, request: HTTPConnection) -> str:
        return client_ip(request)

    def check(self, request: HTTPConnection) -> tuple[bool, dict[str, str]]:
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


class AuthFailureLimiter:
    """Refuses a client IP for ``lockout_s`` after ``limit`` wrong API keys within ``window_s``.

    Independent of OS_RATE_LIMIT, so guessing the key is slow even when rate limiting is off.
    """

    def __init__(self, limit: int, window_s: float = 60.0, lockout_s: float = 60.0, max_tracked: int = 10_000):
        self.limit = limit
        self.window_s = window_s
        self.lockout_s = lockout_s
        self.max_tracked = max_tracked
        self._failures: dict[str, deque[float]] = {}
        self._locked_until: dict[str, float] = {}

    def retry_after(self, ip: str) -> int | None:
        """Seconds until ``ip`` may try again, or None when it is not locked out."""
        until = self._locked_until.get(ip)
        if until is None:
            return None
        remaining = until - time.monotonic()
        if remaining <= 0:
            del self._locked_until[ip]
            return None
        return math.ceil(remaining)

    def record_failure(self, ip: str) -> None:
        now = time.monotonic()
        if ip not in self._failures and len(self._failures) >= self.max_tracked:
            self._forget_stale(now)
        failures = self._failures.setdefault(ip, deque(maxlen=self.limit))
        failures.append(now)
        while failures and now - failures[0] > self.window_s:
            failures.popleft()
        if len(failures) >= self.limit:
            del self._failures[ip]
            self._locked_until[ip] = now + self.lockout_s
            logger.warning(
                "Too many wrong API keys from %s; refusing it for %ds, even with the right key. Clients "
                "that share this address (NAT, Docker Desktop port publishing, a proxy without "
                "OS_TRUST_PROXY=true) are refused too; OS_AUTH_FAILURE_LIMIT=0 turns the lockout off.",
                ip,
                self.lockout_s,
            )

    def _forget_stale(self, now: float) -> None:
        for ip in [ip for ip, times in self._failures.items() if not times or now - times[-1] > self.window_s]:
            del self._failures[ip]
        for ip in [ip for ip, until in self._locked_until.items() if until <= now]:
            del self._locked_until[ip]


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None
_auth_failure_limiter: AuthFailureLimiter | None = None


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


def get_auth_failure_limiter() -> AuthFailureLimiter | None:
    """Get or create the failed-auth limiter for OS_AUTH_FAILURE_LIMIT (0 disables it)."""
    global _auth_failure_limiter
    limit = settings.os_auth_failure_limit
    if limit <= 0:
        return None
    if _auth_failure_limiter is None or _auth_failure_limiter.limit != limit:
        _auth_failure_limiter = AuthFailureLimiter(limit)
    return _auth_failure_limiter


def reset_security_state_for_tests() -> None:
    """Forget rate-limit buckets, failed-auth counts, WebSocket tickets and the one-time Host warnings."""
    global _rate_limiter, _auth_failure_limiter, _host_warning_logged, _malformed_host_warning_logged
    _rate_limiter = None
    _auth_failure_limiter = None
    _host_warning_logged = False
    _malformed_host_warning_logged = False
    _ws_tickets.clear()


# ---------------------------------------------------------------------------
# Request body size
# ---------------------------------------------------------------------------

# Room for multipart part headers and form fields next to the file itself.
_FORM_OVERHEAD_BYTES = MB
# JSON carries uploads as base64 (4/3 of the bytes) next to a few small fields.
_JSON_OVERHEAD_BYTES = 64 * 1024


class _BodyTooLarge(HTTPException):
    """Raised from receive(); FastAPI re-raises HTTPException from body parsing as it is."""


def _body_limit(path: str, content_type: str) -> tuple[int, int]:
    """(largest accepted body in bytes, the configured limit in MB) for a request.

    Routes still apply their own exact limits; this one bounds how much is read before they can.
    """
    if path == "/v1/reader/documents":
        limit_mb = settings.reader_max_upload_mb
    elif path == "/v1/audio/transcriptions/batch":
        # Several files, each up to OS_MAX_UPLOAD_MB, up to OS_BATCH_MAX_TOTAL_MB together.
        limit_mb = max(settings.os_max_upload_mb, settings.os_batch_max_total_mb)
    else:
        limit_mb = settings.os_max_upload_mb
    form_limit = limit_mb * MB + _FORM_OVERHEAD_BYTES
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type == "application/json" or media_type.endswith("+json"):
        # Never below the form allowance: the Reader takes its text limit plus 1 MB of JSON.
        return max(limit_mb * MB * 4 // 3 + _JSON_OVERHEAD_BYTES, form_limit), limit_mb
    return form_limit, limit_mb


# ---------------------------------------------------------------------------
# Response headers
# ---------------------------------------------------------------------------

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
}
# Only for the web UI: Swagger UI on /docs and /redoc loads its scripts from a CDN.
WEB_UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; media-src 'self' blob: data:; connect-src 'self' ws: wss:; "
    "font-src 'self'; manifest-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)


def hardening_headers() -> dict[str, str]:
    """SECURITY_HEADERS when OS_SECURITY_HEADERS is on, for a response sent outside SecurityMiddleware.

    Starlette sends the catch-all 500 handler's response from ServerErrorMiddleware, outside every user
    middleware. It is JSON, so it never needs the web UI's CSP.
    """
    return dict(SECURITY_HEADERS) if settings.os_security_headers else {}


def cors_headers(request_headers: Headers) -> dict[str, str]:
    """The CORS headers SecurityMiddleware's CORSMiddleware adds to a simple response, for one sent outside it.

    As Starlette's CORSMiddleware.send for the current OS_CORS_ORIGINS: a listed origin gets itself, credentials
    and ``Vary: Origin``; ``*`` gets ``*`` without credentials, or the origin itself when the request has a cookie.
    CORS off, no Origin, or an origin not listed: none.
    """
    origins = _origin_list(settings.os_cors_origins)
    origin = request_headers.get("origin")
    if not origins or origin is None:
        return {}
    if "*" in origins:
        if "cookie" in request_headers:
            return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
        return {"Access-Control-Allow-Origin": "*"}
    if origin in origins:
        return {"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true", "Vary": "Origin"}
    return {}


def _is_web_ui_path(path: str) -> bool:
    return path == "/web" or path.startswith(("/web/", "/static/"))


def _with_security_headers(path: str, send: Send) -> Send:
    async def send_with_headers(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = MutableHeaders(scope=message)
            for name, value in SECURITY_HEADERS.items():
                headers.setdefault(name, value)
            if _is_web_ui_path(path) and headers.get("content-type", "").startswith("text/html"):
                headers.setdefault("Content-Security-Policy", WEB_UI_CSP)
        await send(message)

    return send_with_headers


# ---------------------------------------------------------------------------
# Combined Security Middleware
# ---------------------------------------------------------------------------

class SecurityMiddleware:
    """Pure ASGI middleware for HTTP and WebSocket scopes.

    HTTP: Host check -> CORS -> cross-site write guard -> failed-auth lockout -> API key -> rate limit
    -> body size. WebSocket handshakes get the Host, Origin, lockout and rate-limit checks before the
    route accepts them; the route still checks the API key or ticket. Settings are read per request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._cors: tuple[tuple[str, ...], ASGIApp] | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            await self._http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._websocket(scope, receive, send)
        else:
            if scope["type"] == "lifespan":
                # The Host check needs these; looking them up per request would block the event loop.
                await run_in_threadpool(_machine_names)
            await self.app(scope, receive, send)

    async def _http(self, scope: Scope, receive: Receive, send: Send) -> None:
        if settings.os_security_headers:
            send = _with_security_headers(_routed_path(scope), send)
        problem = _host_problem(scope)
        if problem:
            await _error(400, problem)(scope, receive, send)
            return
        origins = _origin_list(settings.os_cors_origins)
        await (self._cors_app(origins) if origins else self._checked_http)(scope, receive, send)

    def _cors_app(self, origins: tuple[str, ...]) -> ASGIApp:
        # Outside the auth check: preflights carry no credentials, and errors still get CORS headers.
        if self._cors is None or self._cors[0] != origins:
            cors = CORSMiddleware(
                self._checked_http,
                allow_origins=list(origins),
                allow_credentials="*" not in origins,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self._cors = (origins, cors)
        return self._cors[1]

    async def _checked_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        conn = HTTPConnection(scope)
        refusal: _Refusal | None = None
        rl_headers: dict[str, str] = {}
        path = _routed_path(scope)
        if _is_cross_site_write(conn):
            refusal = (403, "Cross-site request refused. List the page's origin in OS_CORS_ORIGINS to allow it.", None)
        elif not _is_auth_exempt(path):
            refusal = _refuse_client(conn, check_key=True)
            limiter = get_rate_limiter() if refusal is None else None
            if limiter:
                allowed, rl_headers = limiter.check(conn)
                if not allowed:
                    refusal = (429, "Rate limit exceeded. Try again later.", rl_headers)

        # Refuse an oversized body before reading it, and stop reading one that grows past the limit
        # (chunked, or a wrong Content-Length): multipart parsing would otherwise spool it to disk.
        limit, limit_mb = _body_limit(path, conn.headers.get("content-type", ""))
        too_large = f"Request body too large. Max: {limit_mb}MB"
        declared = conn.headers.get("content-length", "")
        if refusal is None and declared.isdigit() and int(declared) > limit:
            refusal = (413, too_large, None)
        if refusal is not None:
            await _error(*refusal)(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge(status_code=413, detail=too_large)
            return message

        async def send_with_rate_headers(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                for name, value in rl_headers.items():
                    headers[name] = value
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_rate_headers)
        except _BodyTooLarge:
            if response_started:
                raise
            await _error(413, too_large)(scope, receive, send)

    async def _websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        conn = HTTPConnection(scope)
        refusal: _Refusal | None = None
        problem = _host_problem(scope)
        if problem:
            refusal = (400, problem, None)
        elif not verify_ws_origin(conn):
            # Accepted and closed with the reason, before the route runs.
            await _accept_and_close(scope, receive, send, WS_CLOSE_ORIGIN, WS_ORIGIN_REFUSED)
            return
        elif not _is_auth_exempt(_routed_path(scope)):
            # The route checks the key or ticket itself (a ticket works once).
            refusal = _refuse_client(conn, check_key=False)
            limiter = get_rate_limiter() if refusal is None else None
            if limiter:
                allowed, rl_headers = limiter.check(conn)
                if not allowed:
                    refusal = (429, "Rate limit exceeded. Try again later.", rl_headers)
        if refusal is None:
            await self.app(scope, receive, send)
        elif "websocket.http.response" in (scope.get("extensions") or {}):
            await _error(*refusal)(scope, receive, send)  # an HTTP response in place of the handshake
        else:
            await send({"type": "websocket.close", "code": 1008, "reason": refusal[1][:120]})


def _refuse_client(conn: HTTPConnection, *, check_key: bool) -> _Refusal | None:
    """The 429 for a locked-out client, or (with check_key) the 401 for a missing or wrong API key."""
    if not settings.stt_api_key:
        return None
    ip = client_ip(conn)
    failures = get_auth_failure_limiter()
    retry_after = failures.retry_after(ip) if failures else None
    if retry_after is not None:
        return 429, "Too many failed API key attempts. Try again later.", {"Retry-After": str(retry_after)}
    if not check_key:
        return None
    status = _credential_status(conn)
    if status:
        return None
    # Only a wrong key counts: a request without one (a client not set up yet) guesses nothing.
    if status is False and failures:
        failures.record_failure(ip)
    return 401, "Invalid or missing API key. Set Authorization: Bearer <key> header.", _bearer_challenge(status)


def _bearer_challenge(status: bool | None) -> dict[str, str]:
    """A 401 names the scheme it accepts (RFC 9110 15.5.2); a wrong key is an invalid token (RFC 6750 3)."""
    return {"WWW-Authenticate": 'Bearer error="invalid_token"' if status is False else "Bearer"}
