# Open Speech Security & Code Quality Review

## Executive summary
Open Speech is promising and functionally rich, but **it is not yet hardened for hostile networks or public internet exposure**. The biggest concerns are: insecure-by-default deployment (auth disabled unless explicitly set), unauthenticated Wyoming TCP service, WebSocket attack surface lacking origin/abuse controls, and unsafe model supply-chain handling (including a zip extraction vulnerability in Vosk model download). The codebase is generally readable and test-heavy, but concurrency controls, input limits on some endpoints, and production Docker hardening are not at a mature security baseline yet.

---

## Findings

### 🔴 Critical — Zip Slip path traversal in Vosk model extraction
**Where:** `src/backends/vosk_backend.py:89-90`

`zipfile.ZipFile.extractall(dest)` is used directly on downloaded archives from remote URLs (`src/backends/vosk_backend.py:17-20`, `75-90`) without validating member paths. A malicious or compromised zip can write outside `dest` (e.g. `../../...`) and overwrite arbitrary files.

**Impact:** Remote file overwrite / potential code execution depending on filesystem permissions and what gets overwritten.

**Recommended fix:**
- Replace `extractall` with safe extraction that validates each member resolves under `dest`.
- Reject absolute paths and `..` traversal entries.
- Add integrity verification (checksum/signature) before extraction.

---

### 🔴 Critical — Authentication is disabled by default across HTTP + WebSocket API
**Where:** `src/config.py:92`, `src/middleware.py:50-51`, `src/middleware.py:76-77`

`os_api_key` defaults to empty string. When empty, HTTP and WS auth checks short-circuit to allow all access.

**Impact:** If deployed with defaults (or forgotten env var), anyone with network access can use transcription, TTS, model loading, and clone endpoints.

**Recommended fix:**
- Fail startup in non-dev mode when `OS_API_KEY` is unset.
- Introduce explicit `OS_AUTH_DISABLED=true` guard for local/dev only.
- Update compose examples to default-secure patterns.

---

### 🔴 Critical — Realtime audio buffer can grow unbounded (memory DoS)
**Where:** `src/realtime/audio_buffer.py:93,117`, `src/realtime/server.py:125-143,159-160`

Realtime `input_audio_buffer.append` adds data to a bytearray continuously, with no max buffer size enforcement before commit/clear.

**Impact:** A client can stream endless audio (or giant frames) and exhaust server RAM.

**Recommended fix:**
- Enforce per-session max buffered bytes/time.
- Reject oversized append frames with explicit error.
- Force periodic commit/flush and close abusive sessions.

---

### 🟡 Warning — Wyoming TCP service has no authentication or transport security
**Where:** `src/main.py:112-117`, `src/wyoming/server.py:198-213`, `src/config.py:103-104`

Wyoming server is started as plain TCP on configured host/port with no API key, mTLS, token, or IP ACL in application layer.

**Impact:** Anyone who can reach the port can use STT/TTS over Wyoming protocol.

**Recommended fix:**
- Bind Wyoming to localhost/private interface by default.
- Add optional shared secret or mTLS.
- Strongly document reverse-proxy or firewall requirement.

---

### 🟡 Warning — No WebSocket origin validation (CSWSH risk)
**Where:** `src/main.py:388-431`, `src/realtime/server.py:362`, `src/streaming.py:543`

WS endpoints accept connections based on API key only; no Origin allowlist check is performed.

**Impact:** Browser-based cross-site WebSocket abuse is possible if credentials/key material are available in browser context or query string.

**Recommended fix:**
- Validate `Origin` header against configured allowlist for WS endpoints.
- Add `OS_WS_ALLOWED_ORIGINS` separate from broad CORS policy.

---

### 🟡 Warning — Rate limiting is not applied to WebSocket connections/events
**Where:** `src/middleware.py:191-193`, `src/main.py:388-431`

Security middleware intentionally skips websocket upgrades. Only streaming has a max connection count; there is no per-IP/message/frame rate control on `/v1/realtime` and no frame-size throttling.

**Impact:** Resource exhaustion via rapid event flood or oversized WS messages.

**Recommended fix:**
- Add WS handshake/IP token bucket.
- Add per-session event rate and max frame size limits.
- Add backpressure and idle timeouts.

---

### 🟡 Warning — API key accepted in query string (credential leakage risk)
**Where:** `src/middleware.py:63-65`, `src/middleware.py:80-82`

`api_key` query parameter is accepted for HTTP and WS auth.

**Impact:** Keys leak through logs, browser history, referrers, and intermediary telemetry.

**Recommended fix:**
- Deprecate query param auth; prefer `Authorization: Bearer` only.
- If temporary compatibility needed, gate behind explicit legacy flag and warn loudly.

---

### 🟡 Warning — TLS implementation uses self-signed certs in `/tmp` with weak operational safeguards
**Where:** `src/ssl_utils.py:11-13,31-40`, `docker-compose.yml:59`

Default cert/key path is `/tmp/open-speech-certs`. Keys are generated without explicit restrictive chmod. `/tmp` semantics and shared environments increase operational risk.

**Impact:** Easier key misuse/exposure in containerized multi-tenant or misconfigured hosts; self-signed certs also invite insecure client behavior (`-k` patterns).

**Recommended fix:**
- Use dedicated app-owned directory (e.g., `/var/lib/open-speech/certs`).
- Set key permissions explicitly (`0600`) and directory permissions (`0700`).
- Treat self-signed as dev-only; recommend ACME/proxy TLS for production.

---

### 🟡 Warning — Model downloads are not pinned/verified (supply-chain risk)
**Where:** `src/vad/silero.py:196-203`, `src/tts/backends/piper_backend.py:101-103`, `src/backends/vosk_backend.py:17-20,85-90`

Downloads are pulled from remote URLs/HF without checksum/signature verification or pinned immutable revisions.

**Impact:** Compromised upstream artifact can poison runtime behavior.

**Recommended fix:**
- Pin Hugging Face downloads to commit `revision` and verify SHA256.
- For direct URLs, ship known hashes and verify before use.
- Add allowlist and artifact integrity policy.

---

### 🟡 Warning — Docker images run as root and expose broad attack surface
**Where:** `Dockerfile:11-56`, `Dockerfile.cpu:1-35`, `docker-compose.yml:25-27`

No `USER` is configured; service runs as root. Compose maps service ports directly to host; cache volumes are under `/root`.

**Impact:** Container breakout or app compromise has higher blast radius.

**Recommended fix:**
- Create and run as non-root user.
- Use least-privileged filesystem permissions.
- Default-bind to localhost unless explicitly externalized.

---

### 🟡 Warning — Voice cloning endpoint lacks upload size limit
**Where:** `src/main.py:626-646`

`reference_audio` is read fully into memory with no max size check, unlike transcription endpoints which enforce `OS_MAX_UPLOAD_MB`.

**Impact:** Memory exhaustion via oversized multipart file uploads.

**Recommended fix:**
- Apply same upload-size guard used in `/v1/audio/transcriptions` and `/v1/audio/translations`.

---

### 🟡 Warning — Thread/concurrency safety gaps in model management paths
**Where:** `src/main.py:325-348`, `src/tts/router.py:50-113`

Global mutable dict `_download_progress` and backend collections are modified/read across concurrent requests without explicit locking.

**Impact:** Race conditions, inconsistent state, rare runtime errors under load.

**Recommended fix:**
- Add `asyncio.Lock` / thread-safe structures for shared mutable maps.
- Serialize load/unload operations per model.

---

### 🟡 Warning — Realtime API compatibility deviations
**Where:** `src/realtime/server.py:219-223`

`response.create` synthesis always uses `settings.tts_model` instead of requested/session model semantics.

**Impact:** Behavior diverges from expected OpenAI-style model selection; confusing client interoperability.

**Recommended fix:**
- Resolve model from session/request payload and validate availability.
- Return structured spec-aligned errors for unsupported fields.

---

### 🟡 Warning — Dependency policy is permissive and unpinned
**Where:** `pyproject.toml:14-49`

All dependencies are lower-bounded only (`>=`), no lockfile, no hash pinning.

**Impact:** Non-reproducible builds and accidental uptake of vulnerable/breaking transitive releases.

**Recommended fix:**
- Introduce locked dependency set (`requirements.lock`/`uv.lock`/poetry lock).
- Add CI vulnerability scanning (e.g., `pip-audit`, `safety`, OSV).

---

### 🔵 Info — Good practices observed
**Where:**
- Constant-time key comparison: `src/middleware.py:60,65,81,88`
- YAML safe parser used: `src/main.py:607`
- Explicit HTTP codes and errors in many endpoints (e.g., size checks in `src/main.py:169-173,219-223`)
- Presence of dedicated security tests (`tests/test_security.py`)

These are positive foundations, but they do not offset the critical hardening gaps above.

---

## Dependency audit notes
- **No hardcoded live credentials found** in repository scan (only examples/tests).
- **License posture:** Project is MIT (`pyproject.toml:10`), but optional dependencies may include non-MIT licenses. A formal SBOM/license scan is still recommended before public distribution.
- **Known CVEs:** Could not be authoritatively established from source alone without lockfile + resolved dependency graph scan.

---

## Docker review summary
- `.dockerignore` exists, but is minimal (`.dockerignore`).
- Dockerfiles are single-stage and root-based.
- No explicit read-only filesystem, dropped capabilities, or seccomp profile guidance.
- Compose defaults expose service ports directly and persist certs in `/tmp`-mounted volume.

---

## Overall risk assessment
**Current risk: HIGH** for internet-exposed or semi-trusted deployments. In a fully private/lab network with strict firewalling, risk is manageable but still non-trivial due to supply-chain and DoS vectors. Before public release, address all 🔴 items and at least the major 🟡 networking/auth hardening items (WS origin/rate controls, Wyoming auth posture, non-root containerization, artifact integrity verification).