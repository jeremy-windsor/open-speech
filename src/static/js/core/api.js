/**
 * The only module that calls fetch() or constructs a WebSocket. Every request stays on this origin, carries
 * the API key as an Authorization header, and turns the server's error envelope into an ApiError with
 * plain-language copy.
 * @module core/api
 */
import * as auth from './auth.js';
import { app } from './store.js';
import { refreshHealth } from './status.js';

/**
 * A failed request.
 * - `status`: HTTP status, 0 for network, abort and timeout
 * - `code`: the server's code, or 'network' | 'aborted' | 'timeout' | 'unauthorized' | 'auth_locked'
 * - `kind`: 'network' | 'http' | 'auth' | 'locked' | 'aborted' | 'timeout'
 * - `detail`: raw server text or JSON (kept as it came, for logs and tests), cut at DETAIL_MAX for display
 * - `detailText`: the same, readable: the JSON envelope becomes its message and code
 *   ("CUDA out of memory (tts_error)"), never the envelope itself; other text as it came. When not given it
 *   is derived from `detail` with readableDetail().
 * - `body`: the parsed JSON body, whole (never cut), or null when the body wasn't JSON: for a caller that needs
 *   the error's own fields, such as Ask AI's spoken error sentence and its audio
 * - `retryAfter`: seconds from Retry-After, or null
 * - `fields`: [{field, message}] parsed from a validation_error, when possible
 */
export class ApiError extends Error {
  constructor(message, { status = 0, code = '', kind = 'http', detail = '', detailText, body = null, retryAfter = null, fields = [], statusText = '' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.statusText = statusText;
    this.code = code;
    this.kind = kind;
    this.detail = detail;
    this.detailText = typeof detailText === 'string' ? detailText : readableDetail(detail);
    this.body = body === undefined ? null : body;
    this.retryAfter = retryAfter;
    this.fields = fields;
  }
}

const EXEMPT = ['/health', '/web', '/static/', '/v1/reader/media/'];
const DETAIL_MAX = 4000;
const PROBE_WAIT_MS = 5000;

// Single flight for a key the server has not answered yet (AU-05): the first non-exempt request carries it
// and the others wait for its answer (at most 5 s), so a wrong key reaches the server once, not once per
// request that boot fires together. Every wrong key counts toward OS_AUTH_FAILURE_LIMIT.
let confirmedKey = '';
let keyProbe = null;   // {key, done}

function isExempt(pathname) {
  return EXEMPT.some((p) => (p.endsWith('/') ? pathname.startsWith(p) : pathname === p || pathname.startsWith(`${p}/`)));
}

function setConnection(patch) {
  const current = app.get().connection;
  const next = { ...current, ...patch };
  if (Object.keys(patch).some((k) => current[k] !== next[k])) app.set({ connection: next });
}

/** Resolve `path` against this page and refuse any other origin. */
function sameOrigin(path) {
  const url = new URL(String(path), location.href);
  if (url.origin !== location.origin) throw new TypeError('cross-origin request blocked');
  return url;
}

function headerObject(init) {
  const headers = new Headers();
  if (!init) return headers;
  const source = init instanceof Headers ? init : new Headers(init);
  source.forEach((value, name) => headers.set(name, value));
  return headers;
}

/** Python repr of a validation error -> [{field, message}] (best effort). */
export function parseValidation(text) {
  const out = [];
  const re = /'loc':\s*\(([^)]*)\)[^{}]*?'msg':\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")/g;
  let m;
  while ((m = re.exec(String(text || ''))) !== null) {
    const parts = (m[1].match(/'([^']*)'|"([^"]*)"|\d+/g) || []).map((p) => p.replace(/^['"]|['"]$/g, ''));
    const named = parts.filter((p) => !/^\d+$/.test(p) && !['body', 'query', 'path', 'header', 'form'].includes(p));
    out.push({ field: named[named.length - 1] || parts[parts.length - 1] || '', message: m[2] ?? m[3] ?? '' });
  }
  return out;
}

function envelopeMessage(payload) {
  if (!payload || typeof payload !== 'object') return '';
  if (payload.error && typeof payload.error === 'object' && payload.error.message) return String(payload.error.message);
  if (payload.detail && typeof payload.detail === 'object' && !Array.isArray(payload.detail) && payload.detail.message) return String(payload.detail.message);
  if (typeof payload.detail === 'string' && payload.detail) return payload.detail;
  if (typeof payload.error === 'string' && payload.error) return payload.error;
  return '';
}

function envelopeCode(payload) {
  if (!payload || typeof payload !== 'object') return '';
  if (payload.error && typeof payload.error === 'object' && payload.error.code) return String(payload.error.code);
  if (payload.detail && typeof payload.detail === 'object' && !Array.isArray(payload.detail) && payload.detail.code) return String(payload.detail.code);
  return '';
}

function statusLine(res) {
  return res.statusText ? `${res.status} ${res.statusText}` : `HTTP ${res.status}`;
}

// The code the server gives any HTTPException without one of its own (src/main.py): it says nothing more.
const GENERIC_CODES = new Set(['http_error']);

const clip = (text) => (text.length > DETAIL_MAX ? `${text.slice(0, DETAIL_MAX)}…` : text);

/** The readable form of a parsed error body, or '' when it has no message to read. */
function readablePayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return '';
  if (Array.isArray(payload.detail)) {
    // FastAPI's default validation body: one "field: message" line per item.
    return payload.detail
      .map((d) => {
        const field = Array.isArray(d?.loc) ? String(d.loc[d.loc.length - 1] ?? '') : '';
        const msg = String(d?.msg || '');
        return field && msg ? `${field}: ${msg}` : msg;
      })
      .filter(Boolean)
      .join('\n');
  }
  const message = envelopeMessage(payload);
  if (!message) return '';
  const code = envelopeCode(payload);
  return code && !GENERIC_CODES.has(code) ? `${message} (${code})` : message;
}

/**
 * PURE: the text for a Details disclosure, never the JSON envelope. The server's JSON
 * envelope ({"error": {"message", "code"}}, {"detail": "…"} or {"detail": {"message", "code"}}) reads as
 * its message plus its code, "CUDA out of memory (tts_error)" (a generic "http_error" code is left out);
 * FastAPI's validation list reads as "field: message" lines; anything else (a proxy's HTML page, a
 * validation repr, a close reason) comes back as it came, trimmed.
 * @param {*} text the raw body text (ApiError.detail)
 * @returns {string}
 */
export function readableDetail(text) {
  const raw = String(text ?? '').trim();
  if (!raw || !/^[[{]/.test(raw)) return raw;
  let payload;
  try { payload = JSON.parse(raw); } catch { return raw; }
  return readablePayload(payload) || raw;
}

async function errorFor(res, { sentKey = '', ownAuth = false } = {}) {
  let text = '';
  try { text = await res.text(); } catch { text = ''; }
  const contentType = res.headers.get('content-type') || '';
  let payload = null;
  if (text && (contentType.includes('json') || /^\s*[[{]/.test(text))) {
    try { payload = JSON.parse(text); } catch { payload = null; }
  }
  const detail = clip(text);
  // Read from the whole body: a long envelope cut at DETAIL_MAX would no longer parse.
  const detailText = clip(readablePayload(payload) || text.trim());
  // The parsed body goes along whole, for a caller that reads its fields.
  const base = { status: res.status, statusText: res.statusText || '', detail, detailText, body: payload };

  if (res.status === 401) {
    if (ownAuth) {
      // The caller sent its own key (testing one before saving it): the app's key and state are untouched.
      return new ApiError("The server didn't accept that API key.", { ...base, code: 'unauthorized', kind: 'auth' });
    }
    if (sentKey !== auth.getKey()) {
      // The key changed while this request was out: its 401 says nothing about the key in use now.
      return new ApiError('The API key changed while this request was running. Try again.', { ...base, code: 'unauthorized', kind: 'auth' });
    }
    if (sentKey) auth.markRefused(sentKey);
    setConnection({ state: 'unauthorized', error: auth.refusalMessage() });
    return new ApiError(auth.refusalMessage(), { ...base, code: 'unauthorized', kind: 'auth' });
  }
  const serverMessage = envelopeMessage(payload);
  const code = envelopeCode(payload);
  if (res.status === 429 && /API key/i.test(serverMessage)) {
    const wait = Number.parseInt(res.headers.get('Retry-After') || '', 10);
    const retryAfter = Number.isFinite(wait) && wait >= 0 ? wait : null;
    const message = `After too many wrong API keys from this address, the server refuses it for ${retryAfter !== null ? `${retryAfter} s` : 'a while'}, even with the right key. Check the key in Settings › Connection, then try again.`;
    const retryInMs = retryAfter !== null ? retryAfter * 1000 : 0;
    setConnection({ state: 'locked', error: message, retryInMs, retryAt: retryInMs ? Date.now() + retryInMs : 0 });
    return new ApiError(message, { ...base, code: 'auth_locked', kind: 'locked', retryAfter });
  }
  if (code === 'validation_error') {
    // The repr is the useful part ("the repr goes in Details"); the code would add nothing to it. An
    // envelope that names its field in `param` (the AI preset API, src/routes/assistant.py) gives that field.
    const param = payload && payload.error && typeof payload.error === 'object' && typeof payload.error.param === 'string' ? payload.error.param : '';
    const fields = param ? [{ field: param, message: serverMessage }] : parseValidation(serverMessage);
    return new ApiError('Some values were not accepted.', { ...base, code, fields, detail: serverMessage || detail, detailText: clip(serverMessage) || detailText });
  }
  if (text && !payload) {
    return new ApiError(`The server returned an unexpected response (HTTP ${res.status})`, { ...base, code: 'unexpected_response' });
  }
  if (payload && Array.isArray(payload.detail)) {
    // FastAPI's default validation body: [{loc, msg, type}]
    const fields = payload.detail.map((d) => ({ field: Array.isArray(d?.loc) ? String(d.loc[d.loc.length - 1]) : '', message: String(d?.msg || '') }));
    return new ApiError('Some values were not accepted.', { ...base, code: 'validation_error', fields });
  }
  return new ApiError(serverMessage || statusLine(res), { ...base, code });
}

/** Wait for `promise` or `ms`, whichever comes first (the timer is cleared either way). */
function waitAtMost(promise, ms) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    promise.then(() => { clearTimeout(timer); resolve(); }, () => { clearTimeout(timer); resolve(); });
  });
}

function beginProbe(key) {
  let resolve;
  const probe = { key, done: new Promise((r) => { resolve = r; }) };
  keyProbe = probe;
  return (confirmed) => {
    if (confirmed) confirmedKey = key;
    if (keyProbe === probe) keyProbe = null;
    resolve();
  };
}

/** An answer that passed the server's key check: anything but a 401, or the refusals made before it. */
function confirms(status) {
  return status < 400 || ![400, 401, 403, 429].includes(status);
}

function localRefusal() {
  setConnection({ state: 'unauthorized', error: auth.refusalMessage() });
  return new ApiError(auth.refusalMessage(), { status: 401, statusText: 'Unauthorized', code: 'unauthorized', kind: 'auth' });
}

function networkFailure(error, url) {
  const wasOffline = app.get().connection.state === 'offline';
  setConnection({ state: 'offline', error: "Can't reach the server." });
  if (!wasOffline && url.pathname !== '/health') {
    Promise.resolve().then(() => refreshHealth()).catch(() => {});
  }
  return new ApiError("Can't reach the server.", { code: 'network', kind: 'network', detail: String(error?.message || error || '') });
}

/**
 * Make a same-origin request.
 * @param {string} path a path ('/api/models') or a same-origin URL; another origin throws TypeError
 * @param {object} [options]
 * @param {string} [options.method='GET']
 * @param {*} [options.json] sent as a JSON body with Content-Type: application/json
 * @param {BodyInit} [options.body] any other body (FormData, Blob, string)
 * @param {HeadersInit} [options.headers] a caller's own Authorization header is kept as is (its 401 never
 *   touches the app's key or connection state)
 * @param {AbortSignal} [options.signal] aborts the request and, for a Response the caller reads itself
 *   (accept 'response', or 'auto' with a non-JSON body), the body download too
 * @param {number} [options.timeoutMs=0] 0 = no timeout; it covers the headers and the body read (for a
 *   returned Response, the headers only); a timeout rejects with kind 'timeout'
 * @param {'auto'|'json'|'text'|'blob'|'response'} [options.accept='auto'] auto: JSON for a JSON content
 *   type, else the Response
 * @param {boolean} [options.history=false] adds X-History: true
 * @param {boolean} [options.keepalive=false]
 * @returns {Promise<*>} 204 resolves null
 * @throws {ApiError}
 */
export async function request(path, {
  method = 'GET', json, body, headers = {}, signal, timeoutMs = 0,
  accept = 'auto', history = false, keepalive = false,
} = {}) {
  const url = sameOrigin(path);
  const h = headerObject(headers);
  const ownAuth = h.has('Authorization');
  const exempt = isExempt(url.pathname);
  let sentKey = '';
  let release = null;
  if (!ownAuth && auth.hasKey()) {
    if (!exempt) {
      // An auth-exempt path (/health, static files, signed Reader media) still goes out without the key when
      // the key is refused, so a refused key never makes the health check look like an outage.
      let waited = false;
      for (;;) {
        if (auth.isRefused() || !auth.isSendable(auth.getKey())) throw localRefusal();
        const keyNow = auth.getKey();
        if (keyNow === confirmedKey || waited) break;
        if (!keyProbe || keyProbe.key !== keyNow) {
          release = beginProbe(keyNow);
          break;
        }
        await waitAtMost(keyProbe.done, PROBE_WAIT_MS);
        waited = true;
      }
    }
    for (const [name, value] of Object.entries(auth.authHeaders())) h.set(name, value);
    if (h.has('Authorization')) sentKey = auth.getKey();
  }
  let payload = body;
  if (json !== undefined) {
    payload = JSON.stringify(json);
    h.set('Content-Type', 'application/json');
  }
  if (history) h.set('X-History', 'true');

  const controller = new AbortController();
  let timedOut = false;
  let timer = null;
  const onAbort = () => controller.abort(signal.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  if (timeoutMs > 0) {
    timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
  }
  const stopTimer = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  };
  const cleanup = () => {
    stopTimer();
    if (signal) signal.removeEventListener('abort', onAbort);
  };
  const cancelled = (error) => {
    if (timedOut) return new ApiError("The server didn't answer in time.", { code: 'timeout', kind: 'timeout' });
    if (controller.signal.aborted || error?.name === 'AbortError') return new ApiError('Canceled', { code: 'aborted', kind: 'aborted' });
    return null;
  };

  let res;
  try {
    res = await fetch(url.href, {
      method, headers: h, body: payload, signal: controller.signal, keepalive, credentials: 'same-origin',
    });
  } catch (error) {
    cleanup();
    if (release) release(false);
    throw cancelled(error) || networkFailure(error, url);
  }
  // Requests waiting on this one's key go once the answer is known: at once for a success, and for a refusal
  // only after errorFor() has marked the key refused (so none of them sends it again).
  if (release && res.ok) {
    release(true);
    release = null;
  }

  // A Response handed back to the caller keeps following the caller's signal while its body is read (the
  // streaming Stop); the timeout only covered the wait for the headers.
  let handedOver = false;
  try {
    if (!res.ok) {
      const failure = await errorFor(res, { sentKey, ownAuth });
      if (release) {
        release(confirms(res.status));
        release = null;
      }
      throw failure;
    }

    const state = app.get().connection.state;
    if (state !== 'online' && (!exempt || state === 'offline' || state === 'connecting')) {
      setConnection({ state: 'online', error: '', retryInMs: 0, retryAt: 0 });
    }
    if (accept === 'response') {
      handedOver = true;
      return res;
    }
    if (res.status === 204) return null;
    if (accept === 'blob') return await res.blob();
    if (accept === 'text') return await res.text();
    const contentType = res.headers.get('content-type') || '';
    if (accept === 'json' || contentType.includes('json')) {
      const text = await res.text();
      if (!text) return null;
      try {
        return JSON.parse(text);
      } catch {
        throw new ApiError(`The server returned an unexpected response (HTTP ${res.status})`, { status: res.status, code: 'unexpected_response', detail: text.slice(0, DETAIL_MAX) });
      }
    }
    handedOver = true;
    return res;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    // The body read failed: cancelled, timed out, or the connection dropped mid-transfer.
    throw cancelled(error) || networkFailure(error, url);
  } finally {
    if (handedOver) stopTimer();
    else cleanup();
  }
}

export const get = (p, o) => request(p, o);
export const post = (p, json, o) => request(p, { ...o, method: 'POST', json });
export const put = (p, json, o) => request(p, { ...o, method: 'PUT', json });
export const patch = (p, json, o) => request(p, { ...o, method: 'PATCH', json });
export const del = (p, o) => request(p, { ...o, method: 'DELETE' });

/**
 * Open a same-origin WebSocket: wss: on https pages, ws: otherwise. With a key set it first fetches a fresh
 * one-time ticket and adds ?ticket= (never the key). Without a key no ticket is requested.
 * @param {string} path e.g. '/v1/audio/speech/stream'
 * @param {URLSearchParams} [query] copied, not modified
 * @returns {Promise<WebSocket>}
 */
export async function openSocket(path, query = new URLSearchParams()) {
  const url = sameOrigin(path);
  const params = new URLSearchParams(query);
  params.delete('api_key');
  params.delete('ticket');
  if (auth.hasKey()) {
    const ticket = await auth.wsTicket();
    if (ticket) params.set('ticket', ticket);
  }
  const search = params.toString();
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return new WebSocket(`${scheme}://${location.host}${url.pathname}${search ? `?${search}` : ''}`);
}

let liveObjectUrls = new Set();
let pagehideBound = false;

// Only a real unload revokes: a page kept in the back/forward cache (event.persisted) comes back with the views
// still holding these URLs (a render or a recording preview in the mini-player), so they must stay playable.
function onPagehide(event) {
  if (!event || !event.persisted) revokeAllMedia();
}

function trackObjectUrl(objectUrl) {
  liveObjectUrls.add(objectUrl);
  if (!pagehideBound && typeof window !== 'undefined' && window.addEventListener) {
    pagehideBound = true;
    window.addEventListener('pagehide', onPagehide);
  }
}

/** Revoke every object URL made by mediaUrl() (run on a pagehide that unloads the page). */
export function revokeAllMedia() {
  for (const objectUrl of liveObjectUrls) URL.revokeObjectURL(objectUrl);
  liveObjectUrls = new Set();
}

/**
 * A URL an <audio> element or a link can use. Without a key, and for signed /v1/reader/media/ URLs, it is
 * the URL itself (revoke is a no-op). With a key the file is fetched with the key into an object URL;
 * revoke() releases it.
 * @param {string} url same-origin
 * @returns {Promise<{url: string, revoke: () => void}>}
 */
export async function mediaUrl(url) {
  const resolved = sameOrigin(url);
  if (!auth.hasKey() || resolved.pathname.startsWith('/v1/reader/media/')) {
    return { url: String(url), revoke() {} };
  }
  const blob = await request(url, { accept: 'blob' });
  const objectUrl = URL.createObjectURL(blob);
  trackObjectUrl(objectUrl);
  let revoked = false;
  return {
    url: objectUrl,
    revoke() {
      if (revoked) return;
      revoked = true;
      liveObjectUrls.delete(objectUrl);
      URL.revokeObjectURL(objectUrl);
    },
  };
}

/** Banner titles per situation (they replace today's message prefixes). */
export const BANNER_TITLES = Object.freeze({
  speak: "Couldn't generate speech",
  transcribe: "Couldn't transcribe",
  mic: 'Microphone problem',
  live: 'Live stopped',
  delete: "Couldn't delete",
  'reader-add': "Couldn't add the document",
  'reader-open': "Couldn't open the document",
  'reader-play': "Couldn't play",
  'default-preset': "The default preset wasn't applied",
});

/**
 * Plain-language copy for an error.
 * @param {*} error usually an ApiError
 * @param {string|{title?: string, notFound?: string}} [context] a BANNER_TITLES key (or any title) for the
 *   banner title, and optionally the 404 message
 * @returns {{title: string, cause: string, message: string, detail: string, detailText: string,
 *   retryable: boolean, action: ''|'open-settings'|'retry'}} `cause` is the generic title ("API key
 *   needed"); `title` is the context's banner title when given, else the cause. `detailText` is what a
 *   Details disclosure shows: the server's message and code, never its JSON envelope, and ''
 *   when it would only repeat `message`. `detail` stays the raw body, as before (views that parse it keep
 *   working; new code uses `detailText`).
 */
export function explain(error, context = '') {
  const ctx = typeof context === 'string' ? { title: BANNER_TITLES[context] || context } : (context || {});
  const e = error || {};
  const status = e.status || 0;
  const serverMessage = String(e.message || '');
  let cause = 'Something went wrong';
  let message = serverMessage;
  let detail = e.detail ? String(e.detail) : '';
  let detailText = typeof e.detailText === 'string' ? e.detailText.trim() : readableDetail(detail);
  let retryable = false;
  let action = '';
  if (e.kind === 'aborted') {
    return { title: '', cause: '', message: '', detail: '', detailText: '', retryable: false, action: '' };
  }
  if (e.kind === 'auth' || status === 401) {
    cause = 'API key needed';
    message = serverMessage || auth.refusalMessage();
    action = 'open-settings';
  } else if (e.kind === 'locked') {
    cause = 'Too many wrong keys';
    retryable = true;
    action = 'open-settings';
  } else if (e.kind === 'network') {
    cause = "Can't reach the server";
    message = 'Check that Open Speech is running.';
    retryable = true;
    action = 'retry';
  } else if (e.kind === 'timeout') {
    cause = "The server didn't answer";
    message = 'It may be busy. Try again.';
    retryable = true;
    action = 'retry';
  } else if (e.code === 'unexpected_response') {
    cause = 'Something went wrong on the server';
    message = `The server returned an unexpected response (HTTP ${status}).`;
    retryable = status >= 500;
    action = retryable ? 'retry' : '';
  } else if (e.code === 'validation_error') {
    cause = 'Some values were not accepted';
    const first = (e.fields || [])[0];
    message = first && first.message ? `${first.field ? `${first.field}: ` : ''}${first.message}` : 'Some values were not accepted.';
  } else if (status === 413) {
    cause = 'Too large';
  } else if (status === 409) {
    cause = 'Conflict';
  } else if (status === 404) {
    cause = 'Not found';
    if (ctx.notFound) message = ctx.notFound;
  } else if (status === 503) {
    cause = 'Not available right now';
    retryable = true;
    action = 'retry';
  } else if (status >= 500) {
    cause = 'Something went wrong on the server';
    detail = detail || serverMessage;
    detailText = detailText || serverMessage;
    message = serverMessage || 'Try again in a moment.';
    retryable = true;
    action = 'retry';
  } else if (!(error instanceof ApiError)) {
    message = serverMessage || String(error || '');
  }
  // Details that only repeat the message add nothing (every view already dropped them).
  if (detailText && detailText === String(message || '').trim()) detailText = '';
  return { title: ctx.title || cause, cause, message, detail, detailText, retryable, action };
}

const CLOSE_COPY = {
  4001: 'The server refused the API key or ticket. Check Settings › Connection.',
  1006: 'The connection to the server dropped.',
};

const STREAM_NAMES = { live: 'The Live speech stream', transcribe: 'The live transcription stream' };

/**
 * Copy for a WebSocket close code.
 *
 * The server accepts a socket it refuses, then closes it at once with the code and reason
 * (src/middleware.py WS_*): 4001 a missing or wrong key or ticket, 1008 an Origin not
 * allowed (or, on Transcribe, a sample rate), 4004 the feature turned off on the server. Such a socket fires
 * `open` first. A refusal the server still answers as HTTP in place of the handshake (an unknown Host, a rate
 * limit or key lockout) reaches the page as close code 1006 with no reason, exactly what an unreachable server
 * gives. So a socket that never fired `open` and closes with 1006 (or no code) "didn't open"; it never
 * "dropped".
 * @param {number} code
 * @param {string} [reason]
 * @param {'live'|'transcribe'} [endpoint]
 * @param {{opened?: boolean}} [options] opened: whether the socket fired `open` (default true, the old
 *   behaviour)
 * @returns {{message: string, detail: string}}
 */
export function explainClose(code, reason = '', endpoint = '', options = {}) {
  const r = String(reason || '');
  const opened = !(options && options.opened === false);
  if (!opened && (code === 1006 || !code)) {
    const what = STREAM_NAMES[endpoint] || 'The connection to the server';
    return { message: `${what} didn't open. Check that Open Speech is running and that this page comes from its own address.`, detail: r };
  }
  if (code === 4001 && !auth.hasKey()) return { message: 'This server needs an API key. Enter it in Settings › Connection.', detail: r };
  if (CLOSE_COPY[code]) return { message: CLOSE_COPY[code], detail: r };
  if (code === 1008 && /origin not allowed/i.test(r)) {
    return { message: "This page's address isn't allowed to connect. Open Open Speech from its own address, or set OS_WS_ALLOWED_ORIGINS.", detail: r };
  }
  // The handshake refusals the server makes before the route runs (src/middleware.py).
  if (code === 1008 && /api key/i.test(r)) {
    return { message: 'After too many wrong API keys from this address, the server refuses connections for a while, even with the right key. Check the key in Settings › Connection, then try again.', detail: r };
  }
  if (code === 1008 && /rate limit/i.test(r)) return { message: 'Too many requests right now. Try again shortly.', detail: r };
  if (endpoint === 'live') {
    if (code === 4004) return { message: 'Live speech is turned off on this server (TTS_ENABLED or TTS_LIVE_ENABLED).', detail: r };
    if (code === 1013) return { message: 'Another Live session is running. Stop it first or try again shortly.', detail: r };
  }
  if (endpoint === 'transcribe') {
    if (code === 1008) return { message: "The microphone's sample rate wasn't accepted.", detail: r };
    if (code === 1003) return { message: 'Unsupported audio encoding.', detail: r };
    if (code === 1013) return { message: 'Too many live transcriptions at once. Try again shortly.', detail: r };
  }
  return { message: `Disconnected (code ${code}).`, detail: r };
}

/**
 * What helps after a WebSocket close, shared by Live, Transcribe and Talk: 'open-settings'
 * for a refused key or ticket (4001, or a key lockout's 1008), '' for a refusal a retry from this page can't change
 * (1008 for an Origin not allowed, 4004 for a feature turned off on the server), 'retry' for anything else (a
 * drop, a socket that didn't open, a busy server, a sample rate).
 * @param {number} code
 * @param {string} [reason]
 * @returns {'open-settings'|'retry'|''}
 */
export function closeAction(code, reason = '') {
  const r = String(reason || '');
  if (code === 4001 || (code === 1008 && /api key/i.test(r))) return 'open-settings';
  if (code === 4004 || (code === 1008 && /origin not allowed/i.test(r))) return '';
  return 'retry';
}

/** Test hook. */
export function __resetForTests() {
  liveObjectUrls = new Set();
  pagehideBound = false;
  confirmedKey = '';
  keyProbe = null;
}
