import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { json, text, bytes, networkError, delay } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

async function setup({ key = '', ...options } = {}) {
  env = installBrowserEnv({ ...options, seed: { ...(key ? { 'open-speech-api-key': key } : {}), ...(options.seed || {}) } });
  const api = await importFresh('core/api.js');
  const auth = await import('../../../src/static/js/core/auth.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  auth.init();
  return { api, auth, app };
}

describe('core/api: the API key', () => {
  it('[AU-02] every same-origin request carries Authorization: Bearer <key>, explicitly (no window.fetch patch)', async () => {
    const fakeFetch = () => globalThis.fetch;
    const { api } = await setup({ key: 'k1', routes: { 'GET /api/models': () => json(200, { models: [] }), 'GET /health': () => json(200, { status: 'ok' }) } });
    const before = fakeFetch();
    await api.get('/api/models');
    await api.get('/health');
    assert.equal(globalThis.fetch, before, 'fetch is not replaced');
    assert.deepEqual(env.fetch.calls.map((c) => c.headers.authorization), ['Bearer k1', 'Bearer k1']);
  });

  it("[AU-02] a caller's own Authorization header is kept as is", async () => {
    const { api } = await setup({ key: 'k1', routes: { 'GET /x': () => json(200, {}) } });
    await api.get('/x', { headers: { Authorization: 'Bearer other' } });
    assert.equal(env.fetch.calls[0].headers.authorization, 'Bearer other');
  });

  it('without a key nothing is added', async () => {
    const { api } = await setup({ routes: { 'GET /x': () => json(200, {}) } });
    await api.get('/x');
    assert.equal(env.fetch.calls[0].headers.authorization, undefined);
  });

  it('[AU-03][GS-26] a URL on another origin throws before any request, so the key never leaves this origin', async () => {
    const { api } = await setup({ key: 'k1' });
    await assert.rejects(api.get('https://evil.example/api'), (e) => e instanceof TypeError && /cross-origin/.test(e.message));
    await assert.rejects(api.get('//evil.example/x'), TypeError);
    await assert.rejects(api.openSocket('https://evil.example/v1/audio/stream'), TypeError);
    await assert.rejects(api.mediaUrl('http://other:9/a.wav'), TypeError);
    assert.equal(env.fetch.calls.length, 0);
    assert.equal(env.sockets.length, 0);
  });

  it('[GS-26] relative paths resolve against this page (same-origin calls only)', async () => {
    const { api } = await setup({ routes: { 'GET /api/models': () => json(200, {}) } });
    await api.get('/api/models?x=1');
    assert.equal(env.fetch.calls[0].url, 'http://127.0.0.1:8100/api/models?x=1');
  });

  it('[AU-04] a 401 without a key says the server needs one, with kind auth and the connection state unauthorized', async () => {
    const { api, app } = await setup({ routes: { 'GET /api/models': () => json(401, { error: { message: 'Invalid or missing API key. Set Authorization: Bearer <key> header.', code: 'unauthorized' } }) } });
    const error = await api.get('/api/models').catch((e) => e);
    assert.ok(error instanceof api.ApiError);
    assert.equal(error.status, 401);
    assert.equal(error.kind, 'auth');
    assert.equal(error.code, 'unauthorized');
    assert.equal(error.message, 'This server needs an API key. Enter it in Settings › Connection.');
    assert.equal(app.get().connection.state, 'unauthorized');
    assert.match(error.detail, /Invalid or missing API key/, 'the server text is kept for Details only');
  });

  it('[AU-04] a 401 for a wrong key says the server did not accept it', async () => {
    const { api } = await setup({ key: 'wrong', routes: { 'GET /api/models': () => json(401, { error: { message: 'Invalid API key' } }) } });
    const error = await api.get('/api/models').catch((e) => e);
    assert.equal(error.message, "The server didn't accept the API key. Check it in Settings › Connection.");
  });

  it('[AU-05] a refused key is sent once: later requests fail locally until a key is entered', async () => {
    const { api, auth, app } = await setup({ key: 'wrong', routes: {
      'GET /api/models': (req) => (req.headers.authorization === 'Bearer right' ? json(200, { models: [] }) : json(401, { error: { message: 'Invalid API key' } })),
    } });
    await assert.rejects(api.get('/api/models'), { status: 401 });
    await assert.rejects(api.get('/api/models'), { status: 401, kind: 'auth' });
    await assert.rejects(api.get('/v1/reader/documents'), { status: 401 });
    assert.equal(env.fetch.calls.length, 1, 'the wrong key reached the server once');
    auth.setKey('right');
    assert.deepEqual(await api.get('/api/models'), { models: [] });
    assert.equal(env.fetch.calls.length, 2);
    assert.equal(app.get().connection.state, 'online');
  });

  it('[AU-05] a refused key never blocks the auth-exempt /health: it goes out without the key', async () => {
    const { api, app } = await setup({ key: 'wrong', routes: {
      'GET /api/models': () => json(401, { error: { message: 'Invalid API key' } }),
      'GET /health': () => json(200, { status: 'ok' }),
    } });
    await api.get('/api/models').catch(() => {});
    assert.deepEqual(await api.get('/health'), { status: 'ok' });
    const health = env.fetch.callsTo('GET /health');
    assert.equal(health.length, 1);
    assert.equal(health[0].headers.authorization, undefined);
    assert.equal(app.get().connection.state, 'unauthorized', 'the banner keeps saying the key is needed');
  });

  it('[AU-07] an unsendable key fails locally with its message and is never sent', async () => {
    const { api, auth } = await setup({ routes: { 'GET /api/models': () => json(200, {}) } });
    auth.setKey('curly“key');
    const error = await api.get('/api/models').catch((e) => e);
    assert.equal(error.status, 401);
    assert.match(error.message, /can't be sent, such as a curly quote/);
    assert.equal(env.fetch.calls.length, 0);
  });

  it('[AU-06] a 429 about the API key becomes the lockout message with Retry-After; a plain 429 passes through', async () => {
    const { api, app } = await setup({ key: 'k', routes: {
      'GET /locked': () => json(429, { error: { message: 'Too many failed API key attempts' } }, { 'Retry-After': '30' }),
      'GET /busy': () => json(429, { error: { message: 'Rate limit exceeded' } }, { 'Retry-After': '2' }),
    } });
    const locked = await api.get('/locked').catch((e) => e);
    assert.equal(locked.kind, 'locked');
    assert.equal(locked.code, 'auth_locked');
    assert.equal(locked.retryAfter, 30);
    assert.equal(locked.message, 'After too many wrong API keys from this address, the server refuses it for 30 s, even with the right key. Check the key in Settings › Connection, then try again.');
    assert.equal(app.get().connection.state, 'locked');
    assert.equal(app.get().connection.retryInMs, 30000);
    const busy = await api.get('/busy').catch((e) => e);
    assert.equal(busy.kind, 'http');
    assert.equal(busy.status, 429);
    assert.equal(busy.message, 'Rate limit exceeded');
  });
});

describe('core/api: responses and errors', () => {
  it('[TE-02] the error message precedence is error.message, detail.message, detail, error, then the status line', async () => {
    const { api } = await setup({ routes: {
      'GET /a': () => json(400, { error: { message: 'from error.message' }, detail: 'no' }),
      'GET /b': () => json(400, { detail: { message: 'from detail.message' } }),
      'GET /c': () => json(400, { detail: 'from detail' }),
      'GET /d': () => json(400, { error: 'from error' }),
      'GET /e': () => json(418, {}),
    } });
    const messages = [];
    for (const p of ['/a', '/b', '/c', '/d', '/e']) messages.push(await api.get(p).catch((e) => [e.message, e.status]));
    assert.deepEqual(messages, [
      ['from error.message', 400], ['from detail.message', 400], ['from detail', 400], ['from error', 400], ['HTTP 418', 418],
    ]);
  });

  it('[TE-03] JSON content types parse; anything else returns the Response; 204 is null; accept picks the type', async () => {
    const { api } = await setup({ routes: {
      'GET /json': () => json(200, { a: 1 }),
      'GET /audio': () => bytes(200, [1, 2, 3, 4], 'audio/mpeg'),
      'DELETE /thing': () => text(204, ''),
      'GET /text': () => text(200, 'hello'),
    } });
    assert.deepEqual(await api.get('/json'), { a: 1 });
    const res = await api.get('/audio');
    assert.ok(res instanceof Response);
    assert.equal((await res.blob()).size, 4);
    assert.equal(await api.del('/thing'), null);
    assert.equal(await api.get('/text', { accept: 'text' }), 'hello');
    assert.equal((await api.get('/audio', { accept: 'blob' })).size, 4);
    assert.ok((await api.get('/json', { accept: 'response' })) instanceof Response);
  });

  it('[TE-04] a failed request shows the envelope message, or "<status> <statusText>" for an empty body, never raw JSON', async () => {
    const { api } = await setup({ routes: {
      'POST /v1/audio/speech': () => json(400, { error: { message: 'Unknown Kokoro voice: zz', code: 'invalid_voice' } }),
      'POST /v1/audio/transcriptions': () => new Response(null, { status: 500, statusText: 'Internal Server Error' }),
    } });
    const a = await api.post('/v1/audio/speech', { input: 'x' }).catch((e) => e);
    assert.equal(a.message, 'Unknown Kokoro voice: zz');
    assert.equal(a.code, 'invalid_voice');
    assert.ok(!a.message.includes('{'));
    const b = await api.post('/v1/audio/transcriptions', {}).catch((e) => e);
    assert.equal(b.message, '500 Internal Server Error');
  });

  it('[BUG-01] a non-JSON error body (a proxy HTML 502) gets plain copy; the raw text goes in detail only', async () => {
    const { api } = await setup({ routes: { 'GET /api/models': () => text(502, '<html><body><h1>502 Bad Gateway</h1>nginx</body></html>', { 'content-type': 'text/html' }) } });
    const error = await api.get('/api/models').catch((e) => e);
    assert.equal(error.message, 'The server returned an unexpected response (HTTP 502)');
    assert.match(error.detail, /nginx/);
    const copy = api.explain(error);
    assert.equal(copy.cause, 'Something went wrong on the server');
    assert.equal(copy.message, 'The server returned an unexpected response (HTTP 502).');
    assert.equal(copy.retryable, true);
    assert.ok(!copy.message.includes('<html>'));
  });

  it('[BUG-01] a validation_error (Python repr) says "Some values were not accepted." and names the first field', async () => {
    const repr = "1 validation error:\n  {'type': 'missing', 'loc': ('body', 'input'), 'msg': 'Field required', 'input': {'model': 'kokoro'}}";
    const { api } = await setup({ routes: { 'POST /v1/audio/speech': () => json(422, { error: { message: repr, code: 'validation_error' } }) } });
    const error = await api.post('/v1/audio/speech', {}).catch((e) => e);
    assert.equal(error.message, 'Some values were not accepted.');
    assert.equal(error.detail, repr);
    assert.deepEqual(error.fields, [{ field: 'input', message: 'Field required' }]);
    const copy = api.explain(error, 'speak');
    assert.equal(copy.title, "Couldn't generate speech");
    assert.equal(copy.cause, 'Some values were not accepted');
    assert.equal(copy.message, 'input: Field required');
  });

  it('a validation_error that names its field in `param` (the AI preset API) gives that field and the message', async () => {
    // src/routes/assistant.py: {"error": {"message", "code": "validation_error", "param": <field>}}.
    const message = 'user_template must contain {text}';
    const { api } = await setup({ routes: { 'PUT /api/assistant/presets/voice': () => json(422, { error: { message, code: 'validation_error', param: 'user_template' } }) } });
    const error = await api.put('/api/assistant/presets/voice', {}).catch((e) => e);
    assert.ok(error instanceof api.ApiError);
    assert.equal(error.status, 422);
    assert.equal(error.code, 'validation_error');
    assert.deepEqual(error.fields, [{ field: 'user_template', message }]);
    assert.equal(api.explain(error).message, `user_template: ${message}`);
    // A param that isn't text is ignored, and the repr parsing still applies.
    const { api: api2 } = await setup({ routes: { 'POST /x': () => json(422, { error: { message: 'plain', code: 'validation_error', param: 7 } }) } });
    assert.deepEqual((await api2.post('/x', {}).catch((e) => e)).fields, []);
  });

  it('parses messages with apostrophes from the repr', async () => {
    const { api } = await setup();
    const fields = api.parseValidation("{'type': 'x', 'loc': ('body', 'speed'), 'msg': \"Input should be 'a' or 'b'\", 'input': 1}");
    assert.deepEqual(fields, [{ field: 'speed', message: "Input should be 'a' or 'b'" }]);
  });

  it('[TE-05] the message prefixes became banner titles, and explain() gives the plain cause', async () => {
    const { api } = await setup();
    assert.deepEqual({ ...api.BANNER_TITLES }, {
      speak: "Couldn't generate speech", transcribe: "Couldn't transcribe", mic: 'Microphone problem', live: 'Live stopped',
      delete: "Couldn't delete", 'reader-add': "Couldn't add the document", 'reader-open': "Couldn't open the document",
      'reader-play': "Couldn't play", 'default-preset': "The default preset wasn't applied",
    });
    const e = new api.ApiError('Unknown voice', { status: 400 });
    assert.deepEqual(api.explain(e, 'transcribe').title, "Couldn't transcribe");
    assert.equal(api.explain(e, 'transcribe').message, 'Unknown voice');
    const notFound = api.explain(new api.ApiError('Document not found', { status: 404 }), { title: "Couldn't open the document", notFound: 'That document is no longer in the library' });
    assert.equal(notFound.message, 'That document is no longer in the library');
    assert.equal(api.explain(new api.ApiError('x', { status: 413 })).cause, 'Too large');
    assert.equal(api.explain(new api.ApiError('x', { status: 409 })).cause, 'Conflict');
    assert.equal(api.explain(new api.ApiError('x', { status: 503 })).cause, 'Not available right now');
    assert.equal(api.explain(new api.ApiError('', { kind: 'aborted', code: 'aborted' })).message, '');
    const auth = api.explain(new api.ApiError('This server needs an API key. Enter it in Settings › Connection.', { status: 401, kind: 'auth' }));
    assert.equal(auth.cause, 'API key needed');
    assert.equal(auth.action, 'open-settings');
  });

  it('a network failure is kind network, sets the connection offline and schedules a health check', async () => {
    const { api, app } = await setup({ routes: { 'GET /api/models': () => networkError(), 'GET /health': () => networkError() } });
    const error = await api.get('/api/models').catch((e) => e);
    assert.equal(error.kind, 'network');
    assert.equal(error.status, 0);
    assert.equal(error.message, "Can't reach the server.");
    assert.equal(app.get().connection.state, 'offline');
    await env.flush();
    assert.equal(env.fetch.callsTo('GET /health').length, 1, 'refreshHealth ran once');
    const copy = api.explain(error);
    assert.equal(copy.cause, "Can't reach the server");
    assert.equal(copy.message, 'Check that Open Speech is running.');
  });

  it('timeoutMs aborts with kind timeout; a caller abort is kind aborted', async () => {
    const { api } = await setup({ routes: { 'GET /slow': () => delay(10000, json(200, {})) } });
    const pending = api.get('/slow', { timeoutMs: 5000 }).catch((e) => e);
    await env.clock.tickAsync(5000);
    const timeout = await pending;
    assert.equal(timeout.kind, 'timeout');
    const controller = new AbortController();
    const aborted = api.get('/slow', { signal: controller.signal }).catch((e) => e);
    controller.abort();
    assert.equal((await aborted).kind, 'aborted');
  });

  it('json bodies set Content-Type; history adds X-History; keepalive passes through', async () => {
    const { api } = await setup({ routes: { 'PUT /v1/reader/documents/:id/progress': () => json(200, { ok: true }) } });
    await api.put('/v1/reader/documents/d1/progress', { block_index: 3 }, { history: true, keepalive: true });
    const call = env.fetch.calls[0];
    assert.equal(call.method, 'PUT');
    assert.equal(call.headers['content-type'], 'application/json');
    assert.equal(call.headers['x-history'], 'true');
    assert.equal(call.keepalive, true);
    assert.deepEqual(call.json, { block_index: 3 });
  });

  it('a 2xx returns the connection to online, but an auth-exempt /health does not clear "unauthorized"', async () => {
    const { api, app } = await setup({ routes: {
      'GET /health': () => json(200, { status: 'ok' }),
      'GET /api/models': () => json(200, {}),
      'GET /api/denied': () => json(401, {}),
    } });
    await api.get('/api/denied').catch(() => {});
    assert.equal(app.get().connection.state, 'unauthorized');
    await api.get('/health');
    assert.equal(app.get().connection.state, 'unauthorized');
    await api.get('/api/models');
    assert.equal(app.get().connection.state, 'online');
  });
});

describe('core/api: sockets and media', () => {
  it('[AU-13] with a key, each WebSocket gets its own one-time ticket; wss on https; the key never goes in the URL', async () => {
    let n = 0;
    const { api } = await setup({ key: 'secret-key', url: 'https://speech.lan:8100/web', routes: { 'POST /api/auth/ws-ticket': () => json(200, { ticket: `tk${(n += 1)}`, expires_in: 60 }) } });
    const q = new URLSearchParams({ sample_rate: '48000', interim_results: 'true', api_key: 'never' });
    const a = await api.openSocket('/v1/audio/stream', q);
    const b = await api.openSocket('/v1/audio/speech/stream');
    assert.equal(a.url, 'wss://speech.lan:8100/v1/audio/stream?sample_rate=48000&interim_results=true&ticket=tk1');
    assert.equal(b.url, 'wss://speech.lan:8100/v1/audio/speech/stream?ticket=tk2');
    assert.equal(env.fetch.callsTo('POST /api/auth/ws-ticket').length, 2);
    for (const s of env.sockets) assert.ok(!s.url.includes('secret-key') && !s.url.includes('api_key'));
    assert.equal(q.get('api_key'), 'never', "the caller's params are copied, not modified");
  });

  it('[AU-13] without a key no ticket is requested and ws: is used on http', async () => {
    const { api } = await setup();
    const ws = await api.openSocket('/v1/audio/stream', new URLSearchParams({ model: 'm' }));
    assert.equal(ws.url, 'ws://127.0.0.1:8100/v1/audio/stream?model=m');
    assert.equal(env.fetch.calls.length, 0);
  });

  it('[AU-14] with a key, media an element cannot authenticate is fetched with the key into an object URL; revoke releases it', async () => {
    const { api } = await setup({ key: 'k1', routes: { 'GET /api/voices/library/:name/audio': () => bytes(200, [1, 2, 3], 'audio/wav') } });
    const media = await api.mediaUrl('/api/voices/library/demo/audio');
    assert.match(media.url, /^blob:/);
    assert.equal(env.fetch.calls[0].headers.authorization, 'Bearer k1');
    media.revoke();
    media.revoke();
    assert.deepEqual(env.revokedUrls, [media.url]);
    const second = await api.mediaUrl('/api/voices/library/demo/audio');
    const kept = new Event('pagehide');
    kept.persisted = true;   // into the back/forward cache: the views still hold the URL
    env.window.dispatchEvent(kept);
    assert.equal(env.revokedUrls.includes(second.url), false, 'a page kept for Back keeps its media playable');
    env.window.dispatchEvent(new Event('pagehide'));
    assert.ok(env.revokedUrls.includes(second.url), 'object URLs are revoked on a pagehide that unloads the page');
  });

  it('[AU-14] without a key the plain URL is used and nothing is fetched', async () => {
    const { api } = await setup();
    const media = await api.mediaUrl('/api/composer/render/r1/audio');
    assert.equal(media.url, '/api/composer/render/r1/audio');
    assert.equal(env.fetch.calls.length, 0);
    assert.doesNotThrow(() => media.revoke());
  });

  it('[AU-15] signed Reader media URLs are used directly, even with a key (no blob copy, so iOS can stream)', async () => {
    const { api } = await setup({ key: 'k1' });
    const url = '/v1/reader/media/d1/p3.mp3?exp=1760000000&sig=abc';
    const media = await api.mediaUrl(url);
    assert.equal(media.url, url);
    assert.equal(env.fetch.calls.length, 0);
  });

  it('[REV-38] WebSocket close codes have copy: 4004, 4001, 1008 origin and 1013 for Live; the Transcribe codes; others', async () => {
    const { api } = await setup({ key: 'k1' });
    assert.equal(api.explainClose(4004, '', 'live').message, 'Live speech is turned off on this server (TTS_ENABLED or TTS_LIVE_ENABLED).');
    assert.equal(api.explainClose(4001, 'bad ticket', 'live').message, 'The server refused the API key or ticket. Check Settings › Connection.');
    assert.equal(api.explainClose(4001, '', 'transcribe').message, 'The server refused the API key or ticket. Check Settings › Connection.');
    assert.equal(api.explainClose(1008, 'Origin not allowed', 'live').message, "This page's address isn't allowed to connect. Open Open Speech from its own address, or set OS_WS_ALLOWED_ORIGINS.");
    assert.equal(api.explainClose(1013, '', 'live').message, 'Another Live session is running. Stop it first or try again shortly.');
    assert.equal(api.explainClose(1006, '', 'live').message, 'The connection to the server dropped.');
    assert.equal(api.explainClose(1008, 'bad rate', 'transcribe').message, "The microphone's sample rate wasn't accepted.");
    assert.equal(api.explainClose(1003, '', 'transcribe').message, 'Unsupported audio encoding.');
    assert.equal(api.explainClose(1013, '', 'transcribe').message, 'Too many live transcriptions at once. Try again shortly.');
    assert.deepEqual(api.explainClose(4321, 'weird', 'live'), { message: 'Disconnected (code 4321).', detail: 'weird' });
  });

  it("[REV-38] the server's handshake refusals: 4001 with no key at all asks for one; a lockout or rate limit (1008) says so on both endpoints", async () => {
    const { api } = await setup();
    assert.equal(api.explainClose(4001, '', 'live').message, 'This server needs an API key. Enter it in Settings › Connection.', 'nothing was sent, so nothing was refused');
    for (const endpoint of ['live', 'transcribe']) {
      assert.match(api.explainClose(1008, 'Too many failed API key attempts. Try again later.', endpoint).message, /^After too many wrong API keys from this address/);
      assert.equal(api.explainClose(1008, 'Rate limit exceeded. Try again later.', endpoint).message, 'Too many requests right now. Try again shortly.');
    }
    assert.equal(api.explainClose(1008, 'bad rate', 'transcribe').message, "The microphone's sample rate wasn't accepted.", 'any other 1008 on Transcribe is still the sample rate');
  });

  // A refused socket is accepted, then closed with 4001, 1008 or 4004
  // and the reason (src/middleware.py WS_*), so it opens first. One mapping says why, and one says
  // what helps: Open Settings for a key, nothing for an address or a feature the server turned off, Retry otherwise.
  it('the server\'s refusals (opened, then closed with their code and reason) say why on every endpoint; closeAction says what helps', async () => {
    const { api } = await setup({ key: 'k1' });
    const KEY = 'Invalid or missing API key';
    const ORIGIN = 'Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it.';
    for (const endpoint of ['live', 'transcribe', 'talk']) {
      assert.equal(api.explainClose(4001, KEY, endpoint).message, 'The server refused the API key or ticket. Check Settings › Connection.', endpoint);
      assert.equal(api.explainClose(1008, ORIGIN, endpoint).message, "This page's address isn't allowed to connect. Open Open Speech from its own address, or set OS_WS_ALLOWED_ORIGINS.", endpoint);
      assert.equal(api.closeAction(4001, KEY), 'open-settings', endpoint);
      assert.equal(api.closeAction(1008, ORIGIN), '', `${endpoint}: a retry from this address is refused again`);
    }
    assert.equal(api.explainClose(4004, 'Live TTS is disabled', 'live').message, 'Live speech is turned off on this server (TTS_ENABLED or TTS_LIVE_ENABLED).');
    assert.equal(api.closeAction(4004, 'Live TTS is disabled'), '', 'turned off on the server: a retry can\'t help');
    assert.equal(api.closeAction(4004, 'Realtime API is disabled'), '');
    assert.equal(api.closeAction(1008, 'Too many failed API key attempts. Try again later.'), 'open-settings');
    assert.equal(api.closeAction(1008, 'Invalid sample_rate: must be 8000-192000'), 'retry', 'a sample rate: Retry, as before');
    assert.equal(api.closeAction(1013, 'Too many concurrent streams'), 'retry');
    assert.equal(api.closeAction(1006, ''), 'retry', 'a drop');
    assert.equal(api.closeAction(1006, '', { opened: false }), 'retry', "a socket that didn't open (no server, or a refusal as an HTTP answer)");
    assert.equal(api.explainClose(1006, '', 'transcribe', { opened: false }).message,
      "The live transcription stream didn't open. Check that Open Speech is running and that this page comes from its own address.", '1006 keeps its words');
  });

  // The Live socket is closed with 4004 "Live TTS is disabled" when
  // TTS_ENABLED or TTS_LIVE_ENABLED is false (src/routes/tts.py), so Live's sentence names both settings.
  it("Live's 4004 sentence names both settings that close it, TTS_ENABLED and TTS_LIVE_ENABLED", async () => {
    const { api } = await setup({ key: 'k1' });
    for (const reason of ['Live TTS is disabled', '']) {
      const { message } = api.explainClose(4004, reason, 'live');
      assert.equal(message, 'Live speech is turned off on this server (TTS_ENABLED or TTS_LIVE_ENABLED).', reason || 'no reason');
    }
  });
});

describe('core/api: cancelling, timeouts and key races (foundation repair)', () => {
  const settleAll = async () => { for (let i = 0; i < 6; i += 1) await new Promise((r) => setImmediate(r)); };

  it('a caller abort after the headers still aborts the download: the streamed body errors (streaming Stop)', async () => {
    const stream = () => new Response(new ReadableStream({ start(c) { c.enqueue(new Uint8Array([1, 2, 3])); } }), { status: 200, headers: { 'content-type': 'audio/mpeg' } });
    const { api } = await setup({ routes: { 'POST /v1/audio/speech': stream } });
    const controller = new AbortController();
    const res = await api.post('/v1/audio/speech', { input: 'hi' }, { accept: 'response', signal: controller.signal });
    const reader = res.body.getReader();
    assert.deepEqual([...(await reader.read()).value], [1, 2, 3]);
    controller.abort();
    assert.equal(env.fetch.calls[0].signal.aborted, true, 'the signal handed to fetch() follows the caller for the life of the response');
    await assert.rejects(reader.read(), { name: 'AbortError' });
  });

  it('a caller abort during a blob or JSON body read rejects with kind aborted', async () => {
    const slow = () => new Response(new ReadableStream({ start() {} }), { status: 200, headers: { 'content-type': 'application/json' } });
    const { api } = await setup({ routes: { 'GET /api/history': slow, 'GET /api/voices/library/:n/audio': slow } });
    for (const [path, accept] of [['/api/history', 'json'], ['/api/voices/library/a/audio', 'blob']]) {
      const controller = new AbortController();
      const pending = api.get(path, { accept, signal: controller.signal });
      await settleAll();
      controller.abort();
      await assert.rejects(pending, (e) => e.kind === 'aborted');
    }
  });

  it('timeoutMs covers the body read too: a body that stalls after the headers times out', async () => {
    const stalled = () => new Response(new ReadableStream({ start() {} }), { status: 200, headers: { 'content-type': 'application/json' } });
    const { api } = await setup({ routes: { 'GET /health': stalled } });
    const pending = api.get('/health', { timeoutMs: 5000 });
    let outcome = null;
    pending.then(() => { outcome = 'resolved'; }, (e) => { outcome = e; });
    await env.clock.tickAsync(4999);
    assert.equal(outcome, null);
    await env.clock.tickAsync(1);
    await settleAll();
    assert.equal(outcome.kind, 'timeout');
  });

  it('[AU-05] a 401 for a request sent with the old key never marks the new key refused', async () => {
    let release;
    const { api, auth, app } = await setup({ key: 'old-wrong', routes: {
      'GET /api/models': (req) => (req.headers.authorization === 'Bearer right'
        ? json(200, { models: [] })
        : new Promise((r) => { release = () => r(json(401, { error: { message: 'Invalid API key' } })); })),
    } });
    const inflight = api.get('/api/models').catch((e) => e);
    await settleAll();
    auth.setKey('right');   // the listener saves the corrected key while the old request is out
    release();
    const late = await inflight;
    assert.equal(late.kind, 'auth');
    assert.equal(auth.isRefused(), false, 'the new key is not refused');
    assert.notEqual(app.get().connection.state, 'unauthorized');
    assert.deepEqual(await api.get('/api/models'), { models: [] });
    assert.equal(env.fetch.calls.at(-1).headers.authorization, 'Bearer right', 'the right key is sent');
  });

  it("[AU-05] a 401 for a caller's own Authorization header leaves the app's key and connection alone", async () => {
    const { api, auth, app } = await setup({ key: 'saved', routes: { 'GET /api/models': () => json(401, { error: { message: 'Invalid API key' } }) } });
    await assert.rejects(api.get('/api/models', { headers: { Authorization: 'Bearer candidate' } }), (e) => e.kind === 'auth' && e.message === "The server didn't accept that API key.");
    assert.equal(auth.isRefused(), false);
    assert.notEqual(app.get().connection.state, 'unauthorized');
  });

  it('[AU-05] requests fired together with a key the server has not answered yet send it once: a wrong key fails the rest locally', async () => {
    let answer;
    const { api, auth } = await setup({ key: 'wrong', routes: {
      'GET /api/models': () => new Promise((r) => { answer = () => r(json(401, { error: { message: 'Invalid API key' } })); }),
      'GET /api/profiles': () => json(200, { profiles: [] }),
      'GET /v1/audio/voices': () => json(200, { voices: [] }),
      'GET /health': () => json(200, { status: 'ok' }),
    } });
    const results = Promise.allSettled([api.get('/api/models'), api.get('/api/profiles'), api.get('/v1/audio/voices'), api.get('/health')]);
    await settleAll();
    assert.deepEqual(env.fetch.calls.map((c) => c.path), ['/api/models', '/health'], 'one keyed request is out (the exempt /health is never held)');
    answer();
    const out = await results;
    assert.deepEqual(out.map((r) => r.status), ['rejected', 'rejected', 'rejected', 'fulfilled']);
    assert.equal(env.fetch.calls.filter((c) => c.headers.authorization).length, 2, 'the wrong key reached the server once (plus the exempt /health, which never counts)');
    assert.equal(env.fetch.calls.filter((c) => c.path !== '/health').length, 1);
    assert.equal(auth.isRefused(), true);
  });

  it('[AU-05] once the server accepted a key, requests run in parallel; a slow first answer holds the others at most 5 s', async () => {
    let answer;
    const { api } = await setup({ key: 'right', routes: {
      'POST /api/models/:id/download': () => new Promise((r) => { answer = () => r(json(200, {})); }),
      'GET /api/profiles': () => json(200, { profiles: [] }),
      'GET /api/models': () => json(200, { models: [] }),
    } });
    const slow = api.post('/api/models/x/download');
    await settleAll();
    const held = api.get('/api/profiles');
    await env.clock.tickAsync(4000);
    assert.equal(env.fetch.callsTo('GET /api/profiles').length, 0, 'held while the first keyed request is out');
    await env.clock.tickAsync(1000);
    await held;
    assert.equal(env.fetch.callsTo('GET /api/profiles').length, 1, 'released after 5 s');
    answer();
    await slow;
    await Promise.all([api.get('/api/models'), api.get('/api/profiles')]);
    const last = env.fetch.calls.slice(-2).map((c) => c.path).sort();
    assert.deepEqual(last, ['/api/models', '/api/profiles']);
  });
});

describe('core/api: readable details and sockets that never opened', () => {
  it('[TE-04][BUG-01] explain() gives detailText: the envelope reads as its message and code, never the JSON envelope; detail stays the raw body', async () => {
    const { api } = await setup({ routes: {
      'POST /v1/audio/speech': () => json(500, { error: { message: 'CUDA out of memory', code: 'tts_error' } }),
      'GET /api/conversations/c1': () => json(404, { error: { message: 'Conversation not found', code: 'http_error' } }),
      'GET /api/profiles/p1': () => json(404, { detail: 'Profile not found' }),
      'GET /api/voices/library': () => json(400, { detail: { message: 'Reference audio is too short', code: 'invalid_reference' } }),
    } });
    const oom = await api.post('/v1/audio/speech', { input: 'x' }).catch((e) => e);
    assert.equal(oom.detail, '{"error":{"message":"CUDA out of memory","code":"tts_error"}}', 'the raw body is kept for the views that still parse it');
    assert.equal(oom.detailText, 'CUDA out of memory (tts_error)');
    const copy = api.explain(oom, 'speak');
    assert.deepEqual([copy.title, copy.message, copy.detailText], ["Couldn't generate speech", 'CUDA out of memory', 'CUDA out of memory (tts_error)']);
    assert.ok(!copy.detailText.includes('{'), 'never the JSON envelope');
    // The server's generic code says nothing more, so Details would only repeat the message: none.
    const missing = await api.get('/api/conversations/c1').catch((e) => e);
    assert.equal(missing.detailText, 'Conversation not found');
    assert.equal(api.explain(missing, "Couldn't open").detailText, '');
    const fastapi = await api.get('/api/profiles/p1').catch((e) => e);
    assert.equal(fastapi.detailText, 'Profile not found');
    assert.equal(api.explain(fastapi).detailText, '', 'it only repeats the message');
    const nested = await api.get('/api/voices/library').catch((e) => e);
    assert.equal(api.explain(nested).detailText, 'Reference audio is too short (invalid_reference)');
  });

  it("[BUG-01] a proxy page keeps its raw text in detailText; a validation error keeps its repr (no code); FastAPI's list reads as field lines", async () => {
    const repr = "1 validation error:\n  {'type': 'missing', 'loc': ('body', 'input'), 'msg': 'Field required', 'input': {'model': 'kokoro'}}";
    const { api } = await setup({ routes: {
      'GET /api/models': () => text(502, '<html><body>502 Bad Gateway</body></html>', { 'content-type': 'text/html' }),
      'POST /v1/audio/speech': () => json(422, { error: { message: repr, code: 'validation_error' } }),
      'POST /api/profiles': () => json(422, { detail: [{ loc: ['body', 'speed'], msg: 'Input should be less than 4', type: 'less_than' }, { loc: ['body'], msg: 'Bad body', type: 'x' }] }),
    } });
    const proxy = await api.get('/api/models').catch((e) => e);
    const proxyCopy = api.explain(proxy);
    assert.equal(proxyCopy.message, 'The server returned an unexpected response (HTTP 502).');
    assert.equal(proxyCopy.detailText, '<html><body>502 Bad Gateway</body></html>', 'the raw text goes in Details only, as text');
    const invalid = api.explain(await api.post('/v1/audio/speech', {}).catch((e) => e), 'speak');
    assert.equal(invalid.message, 'input: Field required');
    assert.equal(invalid.detailText, repr, 'the repr goes in Details, without "(validation_error)"');
    const list = api.explain(await api.post('/api/profiles', {}).catch((e) => e));
    assert.equal(list.detailText, 'speed: Input should be less than 4\nbody: Bad body');
  });

  it('[TE-04] a long envelope is read from the whole body (a cut-off detail would no longer parse), and both are capped', async () => {
    const long = 'x'.repeat(5000);
    const { api } = await setup({ routes: { 'POST /v1/audio/speech': () => json(500, { error: { message: `Worker crashed: ${long}`, code: 'tts_error' } }) } });
    const error = await api.post('/v1/audio/speech', {}).catch((e) => e);
    assert.ok(error.detail.startsWith('{"error"') && error.detail.endsWith('…'), 'the raw detail is capped');
    assert.ok(error.detailText.startsWith('Worker crashed: xxx'), 'never the envelope, even when it is long');
    assert.ok(error.detailText.length <= 4001);
  });

  // A caller that needs an error's own fields (Ask AI's spoken error sentence and its
  // audio) reads them from the parsed body, which is kept whole; `detail` stays cut for display.
  it('an error keeps its parsed JSON body whole as `body`, never cut; `detail` and `detailText` stay as they were', async () => {
    const base64 = 'A'.repeat(42668);   // a 32 KB MP3 clip in base64, as POST /api/assistant/ask answers a 502
    const envelope = { error: { message: 'connect ECONNREFUSED 127.0.0.1:9', code: 'assistant_unreachable' }, elapsed_ms: 3, sentence: "I couldn't reach the AI server.", audio: { format: 'mp3', content_type: 'audio/mpeg', base64, bytes: 32000 } };
    const { api } = await setup({ routes: {
      'POST /api/assistant/ask': () => json(502, envelope),
      'POST /api/profiles': () => json(422, { error: { message: 'speed: too fast', code: 'validation_error', param: 'speed' } }),
      'GET /api/models': () => text(502, '<html><body>502 Bad Gateway</body></html>', { 'content-type': 'text/html' }),
      'GET /api/conversations/c1': () => json(404, { error: { message: 'Conversation not found', code: 'http_error' } }),
    } });
    const error = await api.post('/api/assistant/ask', { input: { kind: 'text', text: 'Hi' } }).catch((e) => e);
    assert.equal(error.code, 'assistant_unreachable');
    assert.ok(error.detail.length <= 4001 && error.detail.endsWith('…'), 'the raw detail is still cut for display');
    assert.equal(error.detailText, 'connect ECONNREFUSED 127.0.0.1:9 (assistant_unreachable)');
    assert.deepEqual(error.body, envelope, 'the whole body, parsed');
    assert.equal(error.body.audio.base64.length, base64.length, 'never cut');
    const invalid = await api.post('/api/profiles', {}).catch((e) => e);
    assert.deepEqual(invalid.body, { error: { message: 'speed: too fast', code: 'validation_error', param: 'speed' } });
    assert.equal((await api.get('/api/conversations/c1').catch((e) => e)).body.error.code, 'http_error');
    assert.equal((await api.get('/api/models').catch((e) => e)).body, null, 'not JSON: no body');
    assert.equal(new api.ApiError('x', { detail: '{"a":1}' }).body, null, 'a hand-made error has none unless given');
    assert.deepEqual(new api.ApiError('x', { body: { a: 1 } }).body, { a: 1 });
  });

  it('readableDetail() (pure) and hand-made errors: an ApiError built with a JSON detail derives detailText; other errors and aborts have none', async () => {
    const { api } = await setup();
    assert.equal(api.readableDetail('{"error":{"message":"CUDA out of memory","code":"tts_error"}}'), 'CUDA out of memory (tts_error)');
    assert.equal(api.readableDetail('{"error":{"message":"Not found","code":"http_error"}}'), 'Not found');
    assert.equal(api.readableDetail('{"error":"plain"}'), 'plain');
    assert.equal(api.readableDetail('{"detail":"Profile not found"}'), 'Profile not found');
    assert.equal(api.readableDetail('<html>502 Bad Gateway</html>'), '<html>502 Bad Gateway</html>');
    assert.equal(api.readableDetail('{not json'), '{not json');
    assert.equal(api.readableDetail('{"other":1}'), '{"other":1}', 'JSON without a message is shown as it came');
    assert.equal(api.readableDetail('  spaced  '), 'spaced');
    assert.equal(api.readableDetail(undefined), '');
    const made = new api.ApiError('Worker crashed', { status: 500, detail: '{"error":{"message":"Worker crashed","code":"worker_error"}}' });
    assert.equal(made.detailText, 'Worker crashed (worker_error)');
    assert.equal(api.explain(made).detailText, 'Worker crashed (worker_error)');
    assert.equal(new api.ApiError('x', { detail: 'raw', detailText: 'given' }).detailText, 'given');
    assert.equal(api.explain(new Error('boom')).detailText, '');
    assert.equal(api.explain(new api.ApiError('Canceled', { kind: 'aborted', code: 'aborted', detail: '{"error":{"message":"m"}}' })).detailText, '');
    const empty500 = api.explain(new api.ApiError('500 Internal Server Error', { status: 500 }));
    assert.equal(empty500.detailText, '', 'the status line is already the message');
  });

  it("[REV-38][BUG-02] a socket that never opened and closes with 1006 (or no code) \"didn't open\", per endpoint; an open socket still \"dropped\"", async () => {
    const { api } = await setup({ key: 'k1' });
    assert.deepEqual(api.explainClose(1006, '', 'transcribe', { opened: false }), {
      message: "The live transcription stream didn't open. Check that Open Speech is running and that this page comes from its own address.", detail: '',
    });
    assert.equal(api.explainClose(1006, '', 'live', { opened: false }).message, "The Live speech stream didn't open. Check that Open Speech is running and that this page comes from its own address.");
    assert.equal(api.explainClose(undefined, '', 'live', { opened: false }).message, "The Live speech stream didn't open. Check that Open Speech is running and that this page comes from its own address.");
    assert.equal(api.explainClose(0, '', '', { opened: false }).message, "The connection to the server didn't open. Check that Open Speech is running and that this page comes from its own address.");
    assert.equal(api.explainClose(1006, '', 'live', { opened: true }).message, 'The connection to the server dropped.');
    assert.equal(api.explainClose(1006, '', 'live').message, 'The connection to the server dropped.', 'without the option: as before');
    // Codes that do reach a page that never saw `open` keep their own copy.
    assert.equal(api.explainClose(4001, 'bad ticket', 'live', { opened: false }).message, 'The server refused the API key or ticket. Check Settings › Connection.');
    assert.equal(api.explainClose(1013, '', 'transcribe', { opened: false }).message, 'Too many live transcriptions at once. Try again shortly.');
  });
});
