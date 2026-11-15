# Security

Open Speech's defaults are locked down. A deployment reached by IP address, `localhost`, or a LAN name
works without changing any security setting; set `OS_API_KEY` to require a key. A public name, a
reverse proxy, a page on another site, or an iframe needs one of the opt-outs below. The Wyoming port
is separate and has no authentication; see [Wyoming](API.md#wyoming).

- [API key](#api-key), [Host names](#host-names), [Browser origins](#browser-origins),
  [Reverse proxies](#reverse-proxies), [Security headers](#security-headers), [Rate limiting](#rate-limiting),
  [TLS certificates](#tls-certificates)
- The AI assistant's own rules: [Security and privacy](AI-ASSISTANT.md#security-and-privacy)

## API key

Set the key in `.env`. A key of letters, digits, `-` and `_` reads the same everywhere;
`python -c "import secrets; print(secrets.token_urlsafe(32))"` prints one. Single-quote a key that holds a `$`, as
in `OS_API_KEY='Xk9$mP2q'`: Compose and a source install's `uv run --env-file` both read it as written, while `$$`
works under Compose only (see [Writing values in .env](CONFIGURATION.md#writing-values-in-env)). Single quotes can't hold an apostrophe: in a source
install's `.env`, write it as `\'` (`OS_API_KEY=Xk9\'mP2q`) or export the key in the shell, because uv stops reading
`.env` at a bare one and drops the key and `OS_AUTH_REQUIRED` below it, which leaves authentication off. For example:

```bash
OS_API_KEY=my-secret-key

# Stop at startup when OS_API_KEY is empty
OS_AUTH_REQUIRED=true
```

Then restart with the same `-f` files and profiles that started the server: `docker compose up -d` for the plain
CPU launch, `docker compose -f docker-compose.cpu.yml up -d` for the `docker-compose.cpu.yml` one, or
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` for the GPU one. A bare
`docker compose up -d` recreates any other launch from `docker-compose.yml` alone: a GPU server as the CPU image (see
[GPU launch](CONFIGURATION.md#gpu-launch)), and a `docker-compose.cpu.yml` server with Wyoming on `0.0.0.0`, where it has no
authentication, the Turbo STT model and a two-model cap. Then check:

```bash
curl -sk https://localhost:8100/health  # public liveness check
curl -sk -H "Authorization: Bearer my-secret-key" https://localhost:8100/v1/models
```

- With `OS_API_KEY` set, API requests need `Authorization: Bearer KEY`; the scheme is
  case-insensitive. The paths that need no key are listed under [API Reference](API.md).
- WebSocket clients that can set headers send the same header. Browsers cannot, so they first call
  `POST /api/auth/ws-ticket` with the key and then connect with `?ticket=TICKET`. A ticket works for
  one connection within 60 seconds; see [Authentication](API.md#authentication) for what a bad ticket
  gets. The deprecated `?api_key=KEY` query parameter works and logs a warning each time. Open
  Speech's own log shows the values of `api_key`, `ticket`, `exp` and `sig` in a query string as `***`, but a
  reverse proxy's logs and browser history still keep the key.
- **Failed-auth lockout.** After `OS_AUTH_FAILURE_LIMIT` (default `10`) wrong keys from one client
  address within 60 seconds, counting `Authorization` headers, `?api_key=` values, and WebSocket
  tickets, that address gets `429` with `Retry-After` for 60 seconds on HTTP requests and WebSocket
  handshakes, even with the right key. Requests without any key do not count. Clients that share one
  address, such as clients behind NAT, through Docker Desktop's port publishing, or behind a reverse
  proxy without `OS_TRUST_PROXY=true`, are locked out together. Behind a proxy, set
  `OS_TRUST_PROXY=true`; otherwise `OS_AUTH_FAILURE_LIMIT=0` turns the lockout off. The lockout
  applies only when `OS_API_KEY` is set.

## Host names

Every HTTP request and WebSocket handshake must name an accepted host in its `Host` header, which
blocks DNS rebinding attacks from web pages. Accepted without configuration:

- IP addresses, and names without a dot (`lan-box`, `localhost`)
- `*.localhost`, `*.local`, `*.lan`, `*.home.arpa`, and `*.internal`
- the server's own hostname and FQDN (inside Docker, the container's)

Any other name gets `400 Host not allowed. Add it to OS_ALLOWED_HOSTS on the server.` until you list
it in `OS_ALLOWED_HOSTS`, a comma-separated list. `*.example.com` matches subdomains of `example.com`
but not `example.com` itself, and `*` turns the check off. Router names that end in `.fritz.box`,
`.localdomain`, or `.home` need listing too:

```bash
OS_ALLOWED_HOSTS=speech.example.com,*.fritz.box,*.localdomain,*.home
```

A malformed `Host` header (one with a path, `?`, `#`, `@`, or spaces, a port above 65535, or a request
with two `Host` lines) gets `400 Invalid Host header`. The first refused name is logged once with a
hint about `OS_ALLOWED_HOSTS`.

## Browser origins

- **CORS.** `OS_CORS_ORIGINS` is empty by default, so the server sends no CORS headers and only its own
  pages can read API responses. A comma-separated list of origins gets CORS headers, with
  credentials, and may send writes; `*` sends CORS headers to any origin, without credentials. The
  answer to an unexpected server error (`500`, code `internal_error`) carries the same headers, so the page
  can read it.
- **Cross-site write guard.** A `POST`, `PUT`, `PATCH`, or `DELETE` whose `Origin` is neither this
  server's own (the same host and port as the `Host` header) nor listed in `OS_CORS_ORIGINS` gets
  `403`. So does `Origin: null`, and a request without an `Origin` that sends
  `Sec-Fetch-Site: cross-site`. Other requests without an `Origin`, such as SDK and curl calls, are not
  affected. `OS_CORS_ORIGINS=*` lets every origin through the guard.
- **WebSockets.** `OS_WS_ALLOWED_ORIGINS` is empty by default, which accepts only this server's own
  pages. List other origins to accept them too, or set `*` for any. Clients that send no `Origin`
  (SDKs, Home Assistant, CLI tools) are always accepted. A refused page's socket is opened and closed at once
  with `1008` ([Refused WebSockets](API.md#refused-websockets)). With `OS_API_KEY` set, a page on another origin also
  needs that origin in `OS_CORS_ORIGINS`:
  its `POST /api/auth/ws-ticket` is a cross-site write, and without CORS it cannot read the ticket.
- **Wildcards.** Without `OS_API_KEY`, `OS_CORS_ORIGINS=*` lets any web page the user opens send
  writes (synthesis, history deletes, model loads) and read the answers, `OS_WS_ALLOWED_ORIGINS=*`
  lets it open WebSockets, and `OS_ALLOWED_HOSTS=*` lets a DNS rebinding page do the same. Set
  `OS_API_KEY` before using any of these `*` values.

```bash
OS_CORS_ORIGINS=https://myapp.com,https://staging.myapp.com
OS_WS_ALLOWED_ORIGINS=https://myapp.com,https://staging.myapp.com
```

## Reverse proxies

1. Point the proxy at `https://HOST:8100`. Open Speech serves HTTPS with a self-signed certificate by
   default, so turn off the proxy's certificate verification for this upstream, or give Open Speech a
   certificate the proxy trusts (see [TLS certificates](#tls-certificates)). When only the proxy can
   reach Open Speech, `OS_SSL_ENABLED=false` serves plain HTTP to it instead.
2. Pass the client's `Host` header through to Open Speech, port included, and list the public name in
   `OS_ALLOWED_HOSTS`. nginx sends the upstream's name by default; set
   `proxy_set_header Host $http_host;`. nginx's `$host` drops the port, so it works only when the
   proxy listens on the default port (443 or 80). The same-origin checks compare the host and port of
   the browser's `Origin` with `Host`.
3. If the proxy rewrites `Host` instead, same-origin checks no longer match the browser's `Origin`:
   list the public origin, such as `https://speech.example.com`, in both `OS_CORS_ORIGINS` and
   `OS_WS_ALLOWED_ORIGINS`.
4. Set `OS_TRUST_PROXY=true`, so rate limits and the failed-auth lockout see each client's own address
   from `X-Forwarded-For`. The client is the entry `OS_TRUSTED_PROXY_HOPS` (default `1`) from the right,
   counting every `X-Forwarded-For` line; raise it when several proxies each append an entry. Leave
   `OS_TRUST_PROXY=false` when clients can reach Open Speech directly, because they could forge the
   header.

## Security headers

With `OS_SECURITY_HEADERS=true` (the default), every HTTP response carries
`X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`, `X-Frame-Options: DENY`,
`Cross-Origin-Opener-Policy: same-origin`, and
`Permissions-Policy: microphone=(self), camera=(), geolocation=()`. HTML from `/web` and `/static` also
gets a strict `Content-Security-Policy` (`script-src 'self'`, `frame-ancestors 'none'`); `/docs` and
`/redoc` do not. To embed the web UI in an iframe, such as a Home Assistant webpage panel, set
`OS_SECURITY_HEADERS=false`, which turns all of these headers off.

`/docs` and `/redoc` are off unless `OS_API_DOCS=true`. FastAPI's Swagger UI and ReDoc pages load their
scripts from a CDN, unpinned, on the server's own origin, where the web UI keeps its API key in browser
storage, so a compromised CDN could read it. Turn them on only where that is acceptable, such as a
trusted LAN without `OS_API_KEY`.

## Rate limiting

```bash
# Requests per minute per client address; 0 = off
OS_RATE_LIMIT=60
OS_RATE_LIMIT_BURST=10
```

WebSocket handshakes share the per-address bucket with HTTP requests. A limited request gets `429` with
`Retry-After`, and responses carry `X-RateLimit-Limit` and `X-RateLimit-Remaining`. Request bodies are
limited separately; see [Upload and body limits](API.md#upload-and-body-limits).

## TLS certificates

```bash
# Your own certificate: set both, to files that exist
OS_SSL_CERTFILE=/certs/cert.pem
OS_SSL_KEYFILE=/certs/key.pem

# Extra SANs for the generated self-signed certificate
OS_TLS_EXTRA_SANS=192.0.2.24,openspeech.local
```

- With neither set, Open Speech generates a self-signed pair in `/var/lib/open-speech/certs`, valid
  for 365 days for `localhost`, `127.0.0.1`, `0.0.0.0`, and the `OS_TLS_EXTRA_SANS` entries. At
  startup it generates a new pair once the certificate expires within 30 days. Clients that trusted
  the old certificate must then trust the new one.
- Set `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE` together. Open Speech never replaces your own files or
  changes their permissions. When neither file exists, it generates a self-signed pair at those paths.
  When only one of them exists, startup stops with an error rather than overwrite it.
- The SANs are read only when a pair is generated. If you change `OS_TLS_EXTRA_SANS` after a cert has
  been generated, remove the generated cert and key so Open Speech can generate them again. With the
  checked-in Compose files, that usually means removing the project's `ssl-certs` Docker volume.
