import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { json, networkError } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

const SHELL = '<div id="app-sr-status" role="status"></div><div id="app-sr-alert" role="alert"></div>';
const MODELS = { models: [
  { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' },
  { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'loaded', device: 'cpu' },
], default_tts_model: 'kokoro', default_stt_model: 'Systran/faster-whisper-base' };

async function setup(routes = {}, options = {}) {
  env = installBrowserEnv({ html: SHELL, routes: { 'GET /health': () => json(200, { status: 'ok', version: '0.8.0', models_loaded: 1, revision: 'abc123' }), 'GET /api/models': () => json(200, MODELS), ...routes }, ...options });
  const status = await importFresh('core/status.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  return { status, app };
}

const base = (patch = {}) => ({
  connection: { state: 'online', error: '' }, jobs: [], recording: null, player: null,
  models: { list: [], defaults: { tts: '', stt: '' }, fetchedAt: 0, error: '' }, ...patch,
});

describe('core/status.chipModel (pure)', () => {
  it('follows the status chip priority table, first match wins', async () => {
    const { status } = await setup();
    const c = status.chipModel;
    assert.equal(c(base({ recording: { owner: 'transcribe', startedAt: 5 }, connection: { state: 'offline' } })).row, 'recording');
    assert.equal(c(base({ connection: { state: 'offline' }, jobs: [{ key: 'k', label: 'Loading Kokoro…', startedAt: 1 }] })).label, 'Offline');
    assert.deepEqual([c(base({ connection: { state: 'unauthorized' } })).label, c(base({ connection: { state: 'unauthorized' } })).tone], ['API key needed', 'warn']);
    assert.deepEqual([c(base({ connection: { state: 'locked' } })).label, c(base({ connection: { state: 'locked' } })).tone], ['Locked out', 'error']);
    const job = c(base({ jobs: [{ key: 'a', label: 'Loading Kokoro…', startedAt: 100 }, { key: 'b', label: 'Downloading Piper…', startedAt: 200 }] }));
    assert.deepEqual([job.tone, job.label, job.elapsedFrom], ['busy', 'Loading Kokoro… +1', 100]);
    const loaded = c(base({ models: { list: MODELS.models } }));
    assert.deepEqual([loaded.tone, loaded.label], ['ok', 'Kokoro · GPU']);
    assert.equal(loaded.ariaLabel, 'Kokoro · GPU. Status: connected. Loaded: Kokoro on GPU, whisper base on CPU. Open status.');
    assert.match(loaded.detail, /whisper base on CPU/);
    assert.deepEqual([c(base()).tone, c(base()).label], ['idle', 'No model loaded']);
    assert.equal(c(base({ connection: { state: 'connecting' } })).label, 'Connecting…');
  });

  it('[GS-19] activity is app.recording (and app.player for the mini-player), never a body class or ambient animation', async () => {
    const { status, app } = await setup();
    const rec = status.chipModel(base({ recording: { owner: 'voices', startedAt: 1000 } }));
    assert.deepEqual([rec.tone, rec.label, rec.elapsedFrom], ['rec', 'Recording', 1000]);
    status.setRecording({ owner: 'transcribe', startedAt: 42 });
    assert.deepEqual(app.get().recording, { owner: 'transcribe', startedAt: 42 });
    status.setRecording(null);
    assert.equal(app.get().recording, null);
    assert.equal(document.body.getAttribute('class'), null, 'no body.is-live');
  });

  it('[REV-10] labels and accessible names never carry a ticking counter; time comes from elapsedFrom', async () => {
    const { status } = await setup();
    const state = base({ jobs: [{ key: 'model:kokoro', label: 'Loading Kokoro…', startedAt: Date.now() }] });
    const first = status.chipModel(state);
    env.clock.tick(6000);
    const later = status.chipModel(state);
    assert.deepEqual(later, first, 'the model does not change as time passes');
    assert.doesNotMatch(first.ariaLabel, /\d+\s*s\b/);
    assert.equal(first.ariaLabel, 'Status: Loading Kokoro. Open status.');
    const rec = status.chipModel(base({ recording: { owner: 'transcribe', startedAt: 1 } }));
    assert.equal(rec.ariaLabel, 'Status: recording. Open status.');
  });
});

describe('core/status jobs', () => {
  it('[TE-06] one job per key: a second caller shares it (its label wins) and it ends with the last caller', async () => {
    const { status, app } = await setup();
    const endA = status.beginJob('model:kokoro', 'Loading Kokoro…');
    env.raf.flush();
    assert.equal(document.getElementById('app-sr-status').textContent, 'Loading Kokoro…', 'the first begin announces');
    const endB = status.beginJob('model:kokoro', 'Loading Kokoro…', { detail: 'Unloads Piper' });
    assert.deepEqual(app.get().jobs.map((j) => [j.label, j.detail]), [['Loading Kokoro…', 'Unloads Piper']]);
    endA();
    endA();
    assert.equal(app.get().jobs.length, 1, 'ending twice is idempotent');
    env.clock.tick(2500);
    endB();
    assert.equal(app.get().jobs.length, 0);
    env.clock.tick(500);
    env.raf.flush();
    assert.equal(document.getElementById('app-sr-status').textContent, 'Loading Kokoro done');
  });

  it('a job detail ("Unloads Piper") goes in the chip detail and the first announcement, never in the 22-character label', async () => {
    const { status, app } = await setup();
    status.beginJob('model:kokoro', 'Loading Kokoro…', { detail: 'Unloads Piper' });
    env.raf.flush();
    assert.equal(document.getElementById('app-sr-status').textContent, 'Loading Kokoro… Unloads Piper.');
    status.beginJob('model:whisper', 'Downloading whisper base…');
    const chip = status.chipModel(app.get());
    assert.deepEqual([chip.label, chip.detail], ['Loading Kokoro… +1', 'Unloads Piper · Downloading whisper base…']);
    assert.ok(chip.label.length <= 22);
  });

  it('[TE-06] a job shorter than 2 s ends without a "done" announcement', async () => {
    const { status } = await setup();
    const end = status.beginJob('model:piper', 'Loading Piper…');
    env.raf.flush();
    env.clock.tick(1000);
    end();
    env.clock.tick(1000);
    env.raf.flush();
    assert.equal(document.getElementById('app-sr-status').textContent, 'Loading Piper…');
  });
});

describe('core/status polling', () => {
  it('start() checks /health and the models at once, not awaited, then every 30 s and 60 s while visible', async () => {
    const { status, app } = await setup();
    status.start();
    await env.flush();
    assert.equal(app.get().connection.state, 'online');
    assert.equal(app.get().connection.version, '0.8.0');
    assert.equal(app.get().connection.revision, 'abc123');
    assert.equal(app.get().models.list.length, 2);
    assert.equal(env.fetch.callsTo('GET /health').length, 1);
    await env.clock.tickAsync(30000);
    assert.equal(env.fetch.callsTo('GET /health').length, 2);
    await env.clock.tickAsync(30000);
    assert.equal(env.fetch.callsTo('GET /health').length, 3);
    assert.equal(env.fetch.callsTo('GET /api/models').length, 2);
    status.stop();
  });

  it('offline: backoff 2, 4, 8, 16, 30 s, and an immediate re-check on "online"', async () => {
    let up = false;
    const { status, app } = await setup({ 'GET /health': () => (up ? json(200, { status: 'ok', version: '1' }) : networkError()), 'GET /api/models': () => networkError() });
    status.start();
    await env.flush();
    assert.equal(app.get().connection.state, 'offline');
    assert.equal(app.get().connection.retryInMs, 2000);
    const count = () => env.fetch.callsTo('GET /health').length;
    await env.clock.tickAsync(2000);
    assert.equal(count(), 2);
    assert.equal(app.get().connection.retryInMs, 4000);
    await env.clock.tickAsync(4000);
    assert.equal(count(), 3);
    await env.clock.tickAsync(8000);
    assert.equal(count(), 4);
    up = true;
    env.setOnline(true);
    await env.flush();
    assert.equal(count(), 5);
    assert.equal(app.get().connection.state, 'online');
    status.stop();
  });

  it('a request that fails with a network error while online starts the backoff: /health is asked again 2 s later, as the banner says', async () => {
    // The check a failed request makes used to leave the 30 s timer armed, so the banner counted down
    // "Retrying in 2 s" to 0 s and stayed there for ~28 s with no request.
    let up = true;
    const { status, app } = await setup({
      'GET /health': () => (up ? json(200, { status: 'ok', version: '1' }) : networkError()),
      'GET /api/history': () => networkError(),
    });
    const api = await import('../../../src/static/js/core/api.js');
    status.start();
    await env.flush();
    await env.clock.tickAsync(5000);
    const count = () => env.fetch.callsTo('GET /health').length;
    const before = count();
    up = false;
    await api.get('/api/history').catch(() => {});
    await env.flush();
    assert.equal(count(), before + 1, 'the failed request checks /health at once');
    assert.equal(app.get().connection.state, 'offline');
    assert.equal(app.get().connection.retryInMs, 2000);
    assert.equal(app.get().connection.retryAt, Date.now() + 2000);
    await env.clock.tickAsync(2000);
    assert.equal(count(), before + 2, 'the retry comes when the banner said it would');
    assert.equal(app.get().connection.retryInMs, 4000);
    assert.equal(app.get().connection.retryAt, Date.now() + 4000, 'the countdown never sits at 0 s');
    up = true;
    await env.clock.tickAsync(4000);
    assert.equal(count(), before + 3);
    assert.equal(app.get().connection.state, 'online');
    await env.clock.tickAsync(29999);
    assert.equal(count(), before + 3, 'back online: every 30 s again');
    await env.clock.tickAsync(1);
    assert.equal(count(), before + 4);
    status.stop();
  });

  it("the banner's Retry now (refreshHealth while offline) moves the next retry to the delay it reports", async () => {
    const { status, app } = await setup({ 'GET /health': () => networkError(), 'GET /api/models': () => networkError() });
    status.start();
    await env.flush();
    await env.clock.tickAsync(2000);
    await env.clock.tickAsync(4000);   // three failures: the next check is due in 8 s
    await env.clock.tickAsync(1000);
    const count = () => env.fetch.callsTo('GET /health').length;
    const before = count();
    await status.refreshHealth();      // Retry now
    await env.flush();
    const { retryInMs, retryAt } = app.get().connection;
    assert.equal(retryInMs, 16000);
    assert.equal(retryAt, Date.now() + 16000);
    await env.clock.tickAsync(15999);
    assert.equal(count(), before + 1, 'the old 8 s timer no longer fires (7 s later)');
    await env.clock.tickAsync(1);
    assert.equal(count(), before + 2, 'the next check comes when the banner said');
    status.stop();
    await env.clock.tickAsync(120000);
    assert.equal(count(), before + 2, 'stop() leaves nothing armed');
  });

  it('polling pauses while the page is hidden and checks again when it shows', async () => {
    const { status } = await setup();
    status.start();
    await env.flush();
    env.setHidden(true);
    await env.clock.tickAsync(120000);
    assert.equal(env.fetch.callsTo('GET /health').length, 1);
    env.setHidden(false);
    await env.flush();
    assert.equal(env.fetch.callsTo('GET /health').length, 2);
    assert.equal(env.fetch.callsTo('GET /api/models').length, 2, 'models were stale, so they refresh too');
    status.stop();
  });

  it('a lockout answer (429) on /health, e.g. from a proxy, keeps the state "locked", not "offline"', async () => {
    const { status, app } = await setup({ 'GET /health': () => json(429, { error: { message: 'Too many failed API key attempts. Try again later.' } }, { 'Retry-After': '42' }) });
    assert.equal(await status.refreshHealth(), null);
    assert.equal(app.get().connection.state, 'locked');
    assert.equal(app.get().connection.retryInMs, 42000);
  });

  it('a /health timeout (5 s) counts as offline', async () => {
    const { status, app } = await setup({ 'GET /health': () => new Promise(() => {}) });
    const done = status.refreshHealth();
    await env.clock.tickAsync(5000);
    assert.equal(await done, null);
    assert.equal(app.get().connection.state, 'offline');
  });
});
