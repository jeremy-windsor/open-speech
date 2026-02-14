# Open Speech — STATE.md

## Current Status
Phase 3 security features implemented (2026-02-14)

## Last Session: 2026-02-14 Nightly (Session 5, 4:00 AM)
**Task:** Phase 3 Security — API key auth, rate limiting, input validation, CORS

### Changes Made
1. **`src/middleware.py`** — New security middleware module:
   - `SecurityMiddleware` (Starlette BaseHTTPMiddleware) — chains auth → rate limit → validation
   - `verify_api_key()` — Bearer header or `?api_key=` query param, exempt paths for health/web
   - `verify_ws_api_key()` — WebSocket auth (header or query param)
   - `RateLimiter` — token bucket per IP with burst, X-Forwarded-For support, stale cleanup
   - `validate_upload()` — Content-Length pre-check
   - Auth-exempt: `/health`, `/docs`, `/openapi.json`, `/redoc`, `/web`, `/web/*`, `/static/*`

2. **`src/config.py`** — 5 new settings:
   - `STT_API_KEY` (empty = disabled), `STT_RATE_LIMIT` (0 = disabled)
   - `STT_RATE_LIMIT_BURST`, `STT_MAX_UPLOAD_MB` (100), `STT_CORS_ORIGINS` (*)

3. **`src/main.py`** — Wired middleware + CORS + endpoint-level validation:
   - `SecurityMiddleware` added, `CORSMiddleware` added
   - WebSocket auth check before streaming
   - Empty file (400) and oversize file (413) checks in transcribe/translate

4. **`tests/test_security.py`** — 23 new tests:
   - Auth: 9 tests (no key, missing, bearer, wrong, query, exempt, transcribe)
   - Rate limiting: 5 tests (disabled, within burst, over burst, headers, exempt)
   - Input validation: 4 tests (empty, oversized, normal, translate empty)
   - CORS: 2 tests (wildcard, specific origin)
   - Middleware units: 3 tests (cleanup, refill, exempt paths)

5. **README.md** — Security section with usage examples

### Test Results
47/47 passing (6 API + 18 streaming + 23 security)

### Next Steps (Phase 3 remaining)
- Model management in web UI
- Language selection with auto-detect
- Transcription history (session persistence)
- Export: SRT, VTT, JSON, plain text download
- WebSocket connection hardening (origin checks)
- Dependency audit (`pip-audit`)
- Professional landing page
- Full test suite expansion
