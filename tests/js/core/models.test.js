import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { json, delay } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

const SHELL = '<main id="main"></main><div id="app-sr-status" role="status"></div><div id="app-toasts-polite" role="status"></div><div id="app-toasts-alert" role="alert"></div>';

const row = (id, state, extra = {}) => ({ id, type: extra.type || (id.includes('whisper') ? 'stt' : 'tts'), provider: extra.provider || (id.includes('/') ? id.split('/')[0] : id), state, provider_available: true, ...extra });

async function setup({ routes = {}, list = [], defaults = { tts: 'kokoro', stt: '' }, inventoryAgeMs = 0, route = 'speak' } = {}) {
  env = installBrowserEnv({ html: SHELL, routes });
  const models = await importFresh('core/models.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  app.set({ models: { list, defaults, fetchedAt: Date.now() - inventoryAgeMs, error: '' }, route: { view: route, sub: '', id: '', path: route, shot: '' } });
  return { models, app };
}

/** A fake model server: status per id, POST load/download moving the state. */
function server(states, { loadTo = 'loaded', downloadTo = 'downloaded', inventory = null } = {}) {
  const s = { ...states };
  return {
    states: s,
    routes: {
      'GET /api/models/*/status': (req) => {
        const id = decodeURIComponent(req.path.replace('/api/models/', '').replace(/\/status$/, ''));
        return json(200, { model: id, state: s[id] ?? 'available', progress: 0 });
      },
      'POST /api/models/*/load': (req) => { const id = decodeURIComponent(req.path.split('/')[3]); s[id] = loadTo; return json(200, { status: 'loaded' }); },
      'POST /api/models/*/download': (req) => { const id = decodeURIComponent(req.path.split('/')[3]); s[id] = downloadTo; return json(200, { status: 'downloaded' }); },
      'POST /api/models/*/prefetch': (req) => { const id = decodeURIComponent(req.path.split('/')[3]); s[id] = downloadTo; return json(200, { status: 'downloaded' }); },
      'GET /api/models': () => json(200, inventory ? inventory() : { models: Object.entries(s).map(([id, st]) => row(id, st)), default_tts_model: 'kokoro' }),
    },
  };
}

const paths = () => env.fetch.calls.map((c) => `${c.method} ${c.path}`);

describe('core/models: labels and lists', () => {
  it('[TE-07] modelLabel: the provider name plus the variant; an unknown provider gives the model name', async () => {
    const { models } = await setup();
    assert.equal(models.modelLabel('chatterbox/turbo'), 'Chatterbox turbo');
    assert.equal(models.modelLabel('kokoro'), 'Kokoro');
    assert.equal(models.modelLabel('piper/en_US-amy-medium'), 'Piper en_US-amy-medium');
    assert.equal(models.modelLabel('qwen3/0.6b-base'), 'Qwen3 TTS 0.6b-base');
    // A variant the provider's name already ends with is not said twice.
    assert.equal(models.modelLabel('breeze/tts-2'), 'Breeze TTS 2');
    assert.equal(models.modelLabel('mystery/thing'), 'thing');
    assert.equal(models.modelLabel('acme-tts'), 'acme-tts');
    assert.equal(models.providerOf('piper/en_US-amy-medium'), 'piper');
    assert.equal(models.providerOf('kokoro-v1'), 'kokoro');
    assert.equal(models.providerOf(''), 'kokoro');
  });

  it('shortName and deviceLabel', async () => {
    const { models } = await setup();
    assert.equal(models.shortName({ id: 'Systran/faster-whisper-base', type: 'stt' }), 'base');
    assert.equal(models.shortName({ id: 'Systran/faster-distil-whisper-large-v3', type: 'stt' }), 'distil-large-v3');
    assert.equal(models.shortName({ id: 'deepdml/faster-whisper-large-v3-turbo-ct2', type: 'stt' }), 'large-v3-turbo');
    assert.equal(models.shortName({ id: 'piper/en_US-amy-medium', type: 'tts' }), 'Amy (US)');
    assert.equal(models.shortName({ id: 'piper/en_GB-northern_english_male-medium', type: 'tts' }), 'Northern English Male (GB)');
    assert.equal(models.shortName({ id: 'kokoro', type: 'tts' }), 'Kokoro');
    assert.equal(models.shortName({ id: 'chatterbox/turbo', type: 'tts' }), 'turbo');
    assert.equal(models.shortName({ id: 'nvidia/parakeet-tdt-0.6b-v2', type: 'stt' }), 'parakeet-tdt-0.6b-v2');
    assert.deepEqual(['cuda', 'cuda:1', 'mps', 'cpu', '', null].map(models.deviceLabel), ['GPU', 'GPU', 'GPU', 'CPU', '', '']);
  });

  it('sttLabel: Whisper models say "whisper", Parakeet says its own name', async () => {
    const { models } = await setup();
    assert.equal(models.sttLabel({ id: 'Systran/faster-whisper-base', type: 'stt' }), 'whisper base');
    assert.equal(models.sttLabel('deepdml/faster-whisper-large-v3-turbo-ct2'), 'whisper large-v3-turbo');
    assert.equal(models.sttLabel({ id: 'nvidia/parakeet-tdt-0.6b-v2', type: 'stt', provider: 'parakeet' }), 'parakeet-tdt-0.6b-v2');
    assert.equal(models.sttLabel('nvidia/parakeet-tdt-0.6b-v2'), 'parakeet-tdt-0.6b-v2');
    assert.equal(models.sttLabel(''), '');
    assert.equal(models.classifyKind({ id: 'nvidia/parakeet-tdt-0.6b-v2' }), 'stt');
  });

  it('[SP-01] TTS models exclude provider_missing and provider_available === false; Kokoro, Piper, then A–Z', async () => {
    const { models } = await setup();
    const list = [
      row('xtts/v2', 'available'), row('chatterbox/turbo', 'provider_missing'), row('piper/en_US-amy-medium', 'downloaded'),
      row('fish/s2-pro', 'available', { provider_available: false }), row('kokoro', 'loaded'), row('Systran/faster-whisper-base', 'loaded'),
      row('fish/s2-pro', 'available'), row('piper/en_US-ryan-high', 'available'),
    ];
    assert.deepEqual(models.ttsModels(list).map((m) => m.id), ['kokoro', 'piper/en_US-amy-medium', 'piper/en_US-ryan-high', 'fish/s2-pro', 'xtts/v2']);
    assert.equal(models.classifyKind({ id: 'vosk-small' }), 'stt');
    assert.equal(models.classifyKind({ id: 'kokoro' }), 'tts');
  });

  it('sttModels: faster-whisper and Parakeet, loaded > downloaded > available > missing, then id', async () => {
    const { models } = await setup();
    const list = [
      row('Systran/faster-whisper-small', 'available'), row('Systran/faster-whisper-base', 'downloaded'), row('Systran/faster-whisper-tiny', 'loaded'),
      row('Systran/faster-whisper-medium', 'provider_missing'), row('Systran/faster-whisper-large-v3', 'available'), row('kokoro', 'loaded'),
      row('nvidia/parakeet-tdt-0.6b-v2', 'available', { provider: 'parakeet', type: 'stt' }),
    ];
    assert.deepEqual(models.sttModels(list).map((m) => m.id), [
      'Systran/faster-whisper-tiny', 'Systran/faster-whisper-base', 'nvidia/parakeet-tdt-0.6b-v2', 'Systran/faster-whisper-large-v3',
      'Systran/faster-whisper-small', 'Systran/faster-whisper-medium',
    ]);
  });

  it('[SP-02] the pick: preferred > configured default > Kokoro > loaded > downloaded > first; a loaded canary never beats the default', async () => {
    const { models } = await setup();
    const list = [row('piper/en_US-amy-medium', 'downloaded'), row('qwen3/0.6b-base', 'loaded'), row('kokoro', 'available')];
    assert.equal(models.pickTtsModel(list, { configuredDefault: 'kokoro' }), 'kokoro');
    assert.equal(models.pickTtsModel(list, { preferred: 'qwen3/0.6b-base', configuredDefault: 'kokoro' }), 'qwen3/0.6b-base');
    assert.equal(models.pickTtsModel(list, { preferred: 'gone', configuredDefault: 'piper/en_US-amy-medium' }), 'piper/en_US-amy-medium');
    const noKokoro = list.filter((m) => m.id !== 'kokoro');
    assert.equal(models.pickTtsModel(noKokoro, { configuredDefault: 'missing' }), 'qwen3/0.6b-base');
    assert.equal(models.pickTtsModel([row('a/x', 'available'), row('a/y', 'downloaded')]), 'a/y');
    assert.equal(models.pickTtsModel([row('a/x', 'available')]), 'a/x');
    assert.equal(models.pickTtsModel([]), '');
  });

  it('[MD-14] otherLoadedTts names the loaded TTS model a load would replace; STT loads replace none', async () => {
    const { models } = await setup();
    const list = [row('kokoro', 'loaded'), row('piper/en_US-amy-medium', 'downloaded'), row('Systran/faster-whisper-base', 'loaded')];
    assert.equal(models.otherLoadedTts('piper/en_US-amy-medium', list).id, 'kokoro');
    assert.equal(models.otherLoadedTts('kokoro', list), null);
    assert.equal(models.otherLoadedTts('Systran/faster-whisper-base', list), null);
    assert.equal(models.otherLoadedTts('Systran/faster-whisper-small', list), null);
  });

  it('[MD-14][SP-12] with two speech models loaded, the note, the switch question and the job name both: an explicit load unloads both', async () => {
    // A Reader, Live or API request loads its model next to the others; an explicit load unloads every other
    // loaded speech model (README "Model residency"), so naming only the first one undersold the switch.
    const list = [row('kokoro', 'loaded'), row('piper/en_US-amy-medium', 'loaded'), row('qwen3/0.6b-base', 'downloaded')];
    const srv = server({ kokoro: 'loaded', 'piper/en_US-amy-medium': 'loaded', 'qwen3/0.6b-base': 'downloaded' });
    const { models, app } = await setup({ list, routes: srv.routes });
    assert.deepEqual(models.otherLoadedTtsModels('qwen3/0.6b-base', list).map((m) => m.id), ['kokoro', 'piper/en_US-amy-medium']);
    assert.equal(models.uiState(list[2], null, { list }).note, 'Unloads Kokoro and Amy (US)');
    assert.equal(models.joinNames(['A', 'B', 'C']), 'A, B and C');
    const seen = [];
    app.select((s) => s.jobs, (j) => { if (j.length) seen.push([j[0].label, j[0].detail]); });
    const ready = models.ensureReady('qwen3/0.6b-base');
    await env.flush();
    assert.equal(document.getElementById('app-confirm-msg').textContent,
      'Loading Qwen3 TTS 0.6b-base will unload Kokoro and Piper en_US-amy-medium. Continue?');
    document.getElementById('app-confirm-ok').click();
    assert.equal(await ready, true);
    assert.equal(seen[0][1], 'Unloads Kokoro and Piper');
  });

  it('[MD-14] an STT row\'s note says what its load unloads and which pinned default stays loaded', async () => {
    const { models } = await setup();
    const parakeet = row('nvidia/parakeet-tdt-0.6b-v2', 'loaded', { type: 'stt', provider: 'parakeet', is_default: true, pinned: true });
    const small = row('Systran/faster-whisper-small', 'loaded', { provider: 'faster-whisper' });
    const base = row('Systran/faster-whisper-base', 'downloaded', { provider: 'faster-whisper' });
    const kokoro = row('kokoro', 'loaded', { pinned: true });
    const note = (list, m) => models.uiState(m, null, { list }).note;
    assert.equal(note([parakeet, base, kokoro], base), 'Parakeet TDT 0.6B v2 stays loaded');
    assert.equal(note([parakeet, small, base, kokoro], base), 'Unloads Whisper small · Parakeet TDT 0.6B v2 stays loaded');
    // OS_PIN_DEFAULT_MODELS off: nothing is pinned, so the default is unloaded too.
    assert.equal(note([{ ...parakeet, pinned: false }, base, kokoro], base), 'Unloads Parakeet TDT 0.6B v2');
    // Loading the default itself unloads the other STT models.
    const idle = { ...parakeet, state: 'downloaded' };
    assert.equal(note([idle, small, kokoro], idle), 'Unloads Whisper small');
    assert.equal(note([base, kokoro], base), '');
    assert.deepEqual(models.otherLoadedSttModels('kokoro', [parakeet, kokoro]), { unloads: [], stays: [] });
    // A pinned STT model a big speech model's load unloaded says so until it is back.
    const notice = 'Parakeet was unloaded to make room; it reloads on the next transcription.';
    assert.equal(note([{ ...idle, notice }, kokoro], { ...idle, notice }), notice);
    assert.equal(models.sttName({ id: 'nvidia/parakeet-tdt-0.6b-v2', provider: 'parakeet' }), 'Parakeet TDT 0.6B v2');
    assert.equal(models.sttName('Systran/faster-whisper-large-v3'), 'Whisper large-v3');
  });

  it('[MD-14] a TTS row\'s note: a TTS load unloads the other TTS models but the pinned Kokoro stays loaded', async () => {
    const { models } = await setup();
    const kokoro = row('kokoro', 'loaded', { is_default: true, pinned: true });
    const amy = row('piper/en_US-amy-medium', 'loaded');
    const breeze = row('breeze/tts-2', 'downloaded');
    const note = (list, m) => models.uiState(m, null, { list }).note;
    assert.equal(note([kokoro, amy, breeze], breeze), 'Unloads Amy (US) · Kokoro stays loaded');
    assert.equal(note([kokoro, breeze], breeze), 'Kokoro stays loaded');
    assert.deepEqual(models.otherLoadedTtsModels('breeze/tts-2', [kokoro, amy, breeze]).map((m) => m.id), ['piper/en_US-amy-medium']);
    // OS_PIN_DEFAULT_MODELS off: Kokoro is unloaded too.
    assert.equal(note([{ ...kokoro, pinned: false }, breeze], breeze), 'Unloads Kokoro');
  });

  it('installHint: the image rebuild for a baked provider, the worker\'s own Compose profile for a GPU worker', async () => {
    const { models } = await setup();
    assert.deepEqual(models.installHint('piper'), { kind: 'rebuild', text: 'To install, rebuild your image:', command: 'docker build --build-arg BAKED_PROVIDERS=kokoro,piper,pocket-tts,supertonic .' });
    // Each worker in docker-compose.voice-models.yml starts only when its
    // profile is named (README "GPU Launch"); layering the file alone starts the core and no worker.
    // The Fish Audio S2 worker has the profile `fish`, the Breeze TTS 2 one `breeze`.
    for (const [p, name] of [['chatterbox', 'Chatterbox'], ['qwen3', 'Qwen3 TTS'], ['fish', 'Fish Audio S2'], ['breeze', 'Breeze TTS 2']]) {
      const worker = models.installHint(p);
      assert.equal(worker.kind, 'worker');
      assert.equal(worker.command, `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile ${p} up -d --build`);
      assert.equal(worker.text, `${name} runs as an isolated voice worker. Turn it on with its Compose profile:`);
    }
  });

  it('installHint: a worker provider with no service in docker-compose.voice-models.yml gets no command', async () => {
    const { models } = await setup();
    // The stale in-process `fish-speech` provider is gone; a model of it is a stranger.
    for (const [p, name] of [['fish-speech', 'fish-speech'], ['f5-tts', 'F5 TTS'], ['xtts', 'XTTS v2'], ['mystery', 'mystery']]) {
      const hint = models.installHint(p);
      assert.equal(hint.kind, 'worker');
      assert.equal(hint.command, '', `${p} has no profile to name, so no command can start it`);
      assert.equal(hint.text, `${name} runs as an isolated voice worker, and docker-compose.voice-models.yml has none for it.`);
      const state = models.uiState(row(`${p}/x`, 'provider_missing'));
      assert.deepEqual([state.key, state.label, state.action], ['worker-missing', 'Worker not configured', null], `${p}: no Copy command`);
    }
    assert.equal(models.uiState(row('qwen3/0.6b-base', 'provider_missing')).action.command,
      'docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile qwen3 up -d --build');
  });

  it('uiState: the model state vocabulary', async () => {
    const { models } = await setup();
    const u = (m, op, o) => models.uiState(m, op, o);
    const list = [row('kokoro', 'loaded', { device: 'cuda' }), row('piper/en_US-amy-medium', 'downloaded')];
    assert.deepEqual([u(list[0]).label, u(list[0]).tone, u(list[0]).action.kind, u(list[0]).overflow], ['Loaded · GPU', 'ok', 'unload', ['delete']]);
    const d = u(list[1], null, { list });
    assert.deepEqual([d.label, d.tone, d.action.kind, d.note], ['Downloaded', 'idle', 'load', 'Unloads Kokoro']);
    assert.deepEqual([u(row('piper/x', 'available')).label, u(row('piper/x', 'available')).action.kind], ['Not downloaded', 'download']);
    assert.deepEqual([u(row('qwen3/a', 'available', { model_format: 'external' })).label, u(row('qwen3/a', 'available', { model_format: 'external' })).action.kind], ['Downloads on first load', 'load']);
    assert.equal(u(row('x/y', 'queued')).label, 'Queued');
    assert.equal(u(row('piper/x', 'available'), { kind: 'download', startedAt: 5, progress: 0.42 }).label, 'Downloading · 42%');
    // The server's own markers (0.1 while downloading, 0.5 while loading) are phases, never shown as a percent.
    const phase = u(row('piper/x', 'downloading', { progress: 0.1 }), { kind: 'download', startedAt: 5, progress: 0.1 });
    assert.deepEqual([phase.label, phase.progress, phase.elapsedFrom], ['Downloading', null, 5]);
    assert.equal(u(row('piper/x', 'downloading', { progress: 0.1 })).label, 'Downloading');
    const loading = u(row('kokoro', 'available'), { kind: 'load', startedAt: 7 });
    assert.deepEqual([loading.label, loading.tone, loading.elapsedFrom, loading.action], ['Loading', 'busy', 7, null]);
    assert.deepEqual([u(row('kokoro', 'provider_missing')).label, u(row('kokoro', 'provider_missing')).action.command], ['Not installed', 'docker build --build-arg BAKED_PROVIDERS=kokoro,piper,pocket-tts,supertonic .']);
    assert.equal(u(row('chatterbox/turbo', 'provider_missing')).label, 'Worker not configured');
    const offline = u(row('chatterbox/turbo', 'provider_unavailable'));
    assert.deepEqual([offline.label, offline.tone, offline.hint.text], ['Worker offline', 'error', "The voice worker isn't responding. Check the worker service."]);
    const failed = u(row('kokoro', 'available'), { kind: 'load', error: 'Model reverted to available — load failed' });
    assert.deepEqual([failed.label, failed.tone, failed.action.kind, failed.error], ["Couldn't load", 'error', 'retry', 'Model reverted to available — load failed']);
    assert.notEqual(u(list[0]).signature, u(list[0], { kind: 'unload', startedAt: 1 }).signature);
  });
});

describe('core/models.ensureReady', () => {
  it('[SP-12] fast path: a model listed as loaded in a fresh inventory needs no request at all', async () => {
    const { models } = await setup({ list: [row('kokoro', 'loaded')] });
    assert.equal(await models.ensureReady('kokoro'), true);
    assert.deepEqual(paths(), []);
  });

  it('[SP-12] with a stale inventory, one status request; a recent markUsed keeps the fast path open', async () => {
    const srv = server({ kokoro: 'loaded' });
    const { models } = await setup({ list: [row('kokoro', 'loaded')], inventoryAgeMs: 60000, routes: srv.routes });
    assert.equal(await models.ensureReady('kokoro'), true);
    assert.deepEqual(paths(), ['GET /api/models/kokoro/status']);
    env.fetch.reset();
    assert.equal(await models.ensureReady('kokoro'), true, 'the status check counts as a recent success');
    assert.deepEqual(paths(), []);
    env.clock.tick(21000);
    models.markUsed('kokoro');
    assert.equal(await models.ensureReady('kokoro'), true);
    assert.deepEqual(paths(), []);
  });

  it('[SP-12] after a failed request for the model the fast path is off until a status check succeeds', async () => {
    const srv = server({ kokoro: 'loaded' });
    const { models } = await setup({ list: [row('kokoro', 'loaded')], routes: srv.routes });
    models.markFailed('kokoro');
    assert.equal(await models.ensureReady('kokoro'), true);
    assert.deepEqual(paths(), ['GET /api/models/kokoro/status']);
  });

  it('[SP-12] provider_missing and provider_unavailable throw with the install advice', async () => {
    const srv = server({ kokoro: 'provider_missing', 'chatterbox/turbo': 'provider_missing', 'xtts/v2': 'provider_missing', 'fish/s2-pro': 'provider_unavailable' });
    const { models } = await setup({ routes: srv.routes });
    await assert.rejects(models.ensureReady('kokoro'), /rebuild the image with kokoro in BAKED_PROVIDERS/);
    // The worker starts only with its Compose profile named.
    await assert.rejects(models.ensureReady('chatterbox/turbo'), { message: 'The chatterbox voice worker is not configured — turn it on with docker-compose.voice-models.yml and --profile chatterbox' });
    await assert.rejects(models.ensureReady('xtts/v2'), { message: 'The xtts voice worker is not configured — docker-compose.voice-models.yml has none for it' });
    await assert.rejects(models.ensureReady('fish/s2-pro'), /The selected voice provider is offline/);
  });

  it('[SP-12] a TTS load that would unload another model asks first; cancelling loads nothing and returns false', async () => {
    const srv = server({ kokoro: 'loaded', 'piper/en_US-amy-medium': 'downloaded' });
    const { models } = await setup({ list: [row('kokoro', 'loaded'), row('piper/en_US-amy-medium', 'downloaded')], routes: srv.routes });
    const ready = models.ensureReady('piper/en_US-amy-medium');
    await env.flush();
    const dialog = document.getElementById('app-confirm');
    assert.equal(dialog.open, true);
    assert.equal(document.getElementById('app-confirm-msg').textContent, 'Loading Piper en_US-amy-medium will unload Kokoro. Continue?');
    document.getElementById('app-confirm-cancel').click();
    assert.equal(await ready, false);
    assert.ok(!paths().some((p) => p.startsWith('POST')), 'nothing was loaded');
  });

  it('[SP-12] a confirmed switch shows "Loading Piper…" in the chip with "Unloads Kokoro" as its detail', async () => {
    const srv = server({ kokoro: 'loaded', 'piper/en_US-amy-medium': 'downloaded' });
    const { models, app } = await setup({ list: [row('kokoro', 'loaded'), row('piper/en_US-amy-medium', 'downloaded')], routes: srv.routes });
    const seen = [];
    app.select((s) => s.jobs, (j) => { if (j.length) seen.push([j[0].label, j[0].detail]); });
    const ready = models.ensureReady('piper/en_US-amy-medium');
    await env.flush();
    document.getElementById('app-confirm-ok').click();
    assert.equal(await ready, true);
    assert.deepEqual(seen[0], ['Loading Piper…', 'Unloads Kokoro']);
  });

  it('[SP-12] available -> load (TTS); downloaded -> load; the status job runs throughout; refresh at the end', async () => {
    const srv = server({ 'piper/en_US-amy-medium': 'available' });
    const { models, app } = await setup({ list: [row('piper/en_US-amy-medium', 'available')], routes: srv.routes });
    const jobs = [];
    app.select((s) => s.jobs, (j) => jobs.push(j.map((x) => x.label)));
    assert.equal(await models.ensureReady('piper/en_US-amy-medium'), true);
    assert.deepEqual(paths(), [
      'GET /api/models/piper%2Fen_US-amy-medium/status', 'POST /api/models/piper%2Fen_US-amy-medium/load',
      'GET /api/models/piper%2Fen_US-amy-medium/status', 'GET /api/models/piper%2Fen_US-amy-medium/status', 'GET /api/models',
    ]);
    assert.deepEqual(jobs, [['Loading Piper…'], []]);
  });

  it('[SP-12] STT: available -> download, then load; a final state other than loaded throws', async () => {
    const srv = server({ 'Systran/faster-whisper-base': 'available' });
    const { models } = await setup({ routes: srv.routes });
    assert.equal(await models.ensureReady('Systran/faster-whisper-base', { kind: 'stt' }), true);
    assert.deepEqual(paths().filter((p) => p.startsWith('POST')), ['POST /api/models/Systran%2Ffaster-whisper-base/download', 'POST /api/models/Systran%2Ffaster-whisper-base/load']);
    // It loads next to the pinned STT_MODEL (Parakeet in the images) instead of unloading it.
    const load = env.fetch.calls.find((c) => c.method === 'POST' && c.path.endsWith('/load'));
    assert.equal(load.query.get('keep_pinned'), 'true');
    const stuck = server({ kokoro: 'available' }, { loadTo: 'downloaded' });
    env.fetch.route('GET /api/models/*/status', stuck.routes['GET /api/models/*/status']);
    env.fetch.route('POST /api/models/*/load', stuck.routes['POST /api/models/*/load']);
    await assert.rejects(models.ensureReady('kokoro'), /TTS model not ready: state=downloaded/);
  });
});

describe('core/models.runOp', () => {
  it('[MD-05][BUG-12] POST, then poll the status right away and every 2 s; progress goes to app.ops; refresh at the end', async () => {
    let state = 'available';
    let progress = 0;
    const statusCalls = [];
    const { models, app } = await setup({ list: [row('piper/en_US-ryan-high', 'available')], route: 'models', routes: {
      'POST /api/models/:id/prefetch': () => delay(5000, () => { state = 'downloaded'; return json(200, {}); }),
      'GET /api/models/:id/status': () => { statusCalls.push(Date.now()); if (state === 'available' && statusCalls.length > 1) { state = 'downloading'; } progress = Math.min(0.9, progress + 0.3); return json(200, { state, progress }); },
      'GET /api/models': () => json(200, { models: [row('piper/en_US-ryan-high', state)] }),
    } });
    const t0 = Date.now();
    const op = models.runOp('piper/en_US-ryan-high', 'prefetch');
    await env.flush();
    assert.equal(statusCalls.length, 1, 'the first poll is immediate (not after 3 s)');
    assert.equal(statusCalls[0], t0);
    assert.equal(app.get().ops['piper/en_US-ryan-high'].kind, 'prefetch');
    await env.clock.tickAsync(2000);
    assert.equal(statusCalls.length, 2);
    assert.equal(statusCalls[1] - t0, 2000);
    assert.ok(app.get().ops['piper/en_US-ryan-high'].progress > 0);
    assert.equal(app.get().jobs[0].label, 'Downloading Piper…');
    await env.clock.tickAsync(4000);
    assert.equal(await op, true);
    assert.equal(app.get().ops['piper/en_US-ryan-high'], undefined);
    assert.equal(app.get().jobs.length, 0);
    assert.equal(app.get().models.list[0].state, 'downloaded');
    assert.ok(env.fetch.callsTo('GET /api/models').length >= 1);
  });

  it('[REV-08] runOp owns its polling whatever view is showing; a failure is kept in app.ops and toasted elsewhere', async () => {
    const { models, app } = await setup({ list: [row('kokoro', 'available')], route: 'speak', routes: {
      'POST /api/models/:id/load': () => json(200, {}),
      'GET /api/models/:id/status': () => json(200, { state: 'available' }),
      'GET /api/models': () => json(200, { models: [row('kokoro', 'available')] }),
    } });
    const op = models.runOp('kokoro', 'load');
    app.set({ route: { view: 'history', sub: '', id: '', path: 'history', shot: '' } });
    await env.clock.tickAsync(10000);
    assert.equal(await op, false);
    assert.equal(app.get().ops.kokoro.error, 'Model reverted to available — load failed');
    assert.match(document.getElementById('app-toasts-alert').textContent, /Couldn't load Kokoro: Model reverted to available — load failed/);
    models.clearOp('kokoro');
    assert.equal(app.get().ops.kokoro, undefined);
  });

  it('[MD-05] queued/downloading/loading keep waiting; provider states map to their messages', async () => {
    const states = ['queued', 'downloading', 'loading', 'provider_unavailable'];
    const { models, app } = await setup({ route: 'models', routes: {
      'POST /api/models/:id/download': () => json(200, {}),
      'GET /api/models/:id/status': () => json(200, { state: states.length > 1 ? states.shift() : states[0] }),
      'GET /api/models': () => json(200, { models: [] }),
    } });
    const op = models.runOp('chatterbox/turbo', 'download');
    await env.clock.tickAsync(10000);
    assert.equal(await op, false);
    assert.equal(app.get().ops['chatterbox/turbo'].error, 'Voice provider is offline');
  });

  it('a second runOp for the same model joins the first', async () => {
    const srv = server({ kokoro: 'available' });
    const { models } = await setup({ routes: srv.routes });
    const a = models.runOp('kokoro', 'load');
    const b = models.runOp('kokoro', 'load');
    assert.equal(a, b);
    await env.clock.tickAsync(5000);
    assert.equal(await a, true);
    assert.equal(env.fetch.callsTo('POST /api/models/:id/load').length, 1);
  });

  it('[MD-05] Models › Load of an STT model keeps the pinned default STT model loaded (keep_pinned), as Transcribe does; a TTS load does not ask', async () => {
    const parakeet = 'nvidia/parakeet-tdt-0.6b-v2';
    const list = [
      row(parakeet, 'loaded', { type: 'stt', provider: 'parakeet', device: 'cuda', is_default: true, pinned: true }),
      row('Systran/faster-whisper-small', 'loaded', { provider: 'faster-whisper' }),
      row('Systran/faster-whisper-base', 'downloaded', { provider: 'faster-whisper' }),
      row('kokoro', 'loaded', { pinned: true }), row('piper/en_US-amy-medium', 'downloaded'),
    ];
    const srv = server({ [parakeet]: 'loaded', 'Systran/faster-whisper-small': 'loaded', 'Systran/faster-whisper-base': 'downloaded', kokoro: 'loaded', 'piper/en_US-amy-medium': 'downloaded' });
    const { models, app } = await setup({ list, routes: srv.routes, route: 'models' });
    const details = [];
    app.select((s) => s.jobs, (j) => { if (j.length) details.push([j[0].label, j[0].detail]); });
    const op = models.runOp('Systran/faster-whisper-base', 'load');
    await env.clock.tickAsync(5000);
    assert.equal(await op, true);
    const load = env.fetch.calls.find((c) => c.method === 'POST' && c.path.endsWith('/load'));
    assert.equal(load.query.get('keep_pinned'), 'true');
    assert.equal(load.query.get('explicit'), null);
    // The status job names only what the load unloads, never the pinned default.
    assert.deepEqual(details[0], ['Loading whisper base…', 'Unloads whisper small']);
    env.fetch.reset();
    const tts = models.runOp('piper/en_US-amy-medium', 'load');
    await env.clock.tickAsync(5000);
    assert.equal(await tts, true);
    assert.equal(env.fetch.calls.find((c) => c.method === 'POST').query.get('keep_pinned'), null);
    // A person's TTS load may unload the pinned STT model to make room.
    assert.equal(env.fetch.calls.find((c) => c.method === 'POST').query.get('explicit'), 'true');
  });

  it('restoreDefault always loads the default through runOp, even when the inventory says it is loaded (SP-11)', async () => {
    const srv = server({ kokoro: 'loaded' });
    const { models } = await setup({ list: [row('kokoro', 'loaded')], routes: srv.routes });
    const done = models.restoreDefault();
    await env.clock.tickAsync(5000);
    assert.equal(await done, 'kokoro');
    assert.equal(env.fetch.callsTo('POST /api/models/:id/load').length, 1);
  });
});

describe('core/models unload and delete', () => {
  it('[MD-06] unload is only DELETE /api/models/{id}; on failure it refreshes silently and rethrows the server reason', async () => {
    const { models } = await setup({ list: [row('Systran/faster-whisper-base', 'loaded')], routes: {
      'DELETE /api/models/:id': () => json(404, { error: { message: 'Model Systran/faster-whisper-base is not loaded', code: 'not_loaded' } }),
      'GET /api/models': () => json(200, { models: [] }),
    } });
    await assert.rejects(models.unload('Systran/faster-whisper-base'), /is not loaded/);
    assert.deepEqual(paths(), ['DELETE /api/models/Systran%2Ffaster-whisper-base', 'GET /api/models']);
    env.fetch.route('DELETE /api/models/:id', () => json(200, { status: 'unloaded' }));
    env.fetch.reset();
    assert.equal(await models.unload('kokoro'), true);
    assert.deepEqual(paths(), ['DELETE /api/models/kokoro', 'GET /api/models']);
  });

  it('[MD-07] Delete files confirms first; cancel sends nothing; confirming uses the artifacts route and refreshes', async () => {
    const { models } = await setup({ routes: {
      'DELETE /api/models/*/artifacts': () => json(200, { deleted: true }),
      'GET /api/models': () => json(200, { models: [] }),
    } });
    assert.equal(models.deleteFilesMessage('kokoro'), 'Delete downloaded files for Kokoro? This unloads the model and removes its cached files.');
    let pending = models.deleteArtifacts('kokoro', { confirm: true });
    await env.flush();
    assert.equal(document.getElementById('app-confirm-msg').textContent, models.deleteFilesMessage('kokoro'));
    assert.equal(document.activeElement.id, 'app-confirm-cancel', 'a destructive confirm focuses Cancel');
    document.getElementById('app-confirm-cancel').click();
    assert.equal(await pending, false);
    assert.deepEqual(paths(), []);
    pending = models.deleteArtifacts('piper/en_US-amy-medium', { confirm: true });
    await env.flush();
    document.getElementById('app-confirm-ok').click();
    assert.equal(await pending, true);
    assert.deepEqual(paths(), ['DELETE /api/models/piper%2Fen_US-amy-medium/artifacts', 'GET /api/models']);
  });

  it('[MD-08] model ids are encodeURIComponent-ed in every model route', async () => {
    const id = 'piper/en_US-amy-medium';
    const srv = server({ [id]: 'downloaded' });
    const { models } = await setup({ routes: { ...srv.routes, 'DELETE /api/models/*': () => json(200, {}) } });
    await models.getStatus(id);
    await models.ensureReady(id, { confirmSwitch: false });
    await models.unload(id);
    await models.deleteArtifacts(id);
    const op = models.runOp(id, 'download');
    await env.clock.tickAsync(3000);
    await op;
    const modelRoutes = env.fetch.calls.filter((c) => c.path.startsWith('/api/models/'));
    assert.ok(modelRoutes.length >= 6);
    for (const c of modelRoutes) assert.ok(c.path.startsWith('/api/models/piper%2Fen_US-amy-medium'), c.path);
  });
});

describe('core/models.refresh', () => {
  it('deduplicates concurrent refreshes and records failures without throwing when silent', async () => {
    let n = 0;
    const { models, app } = await setup({ routes: { 'GET /api/models': () => { n += 1; return n > 1 ? json(500, { error: { message: 'boom' } }) : json(200, { models: [row('kokoro', 'loaded')], default_tts_model: 'kokoro', default_stt_model: 'Systran/faster-whisper-base' }); } } });
    await Promise.all([models.refresh(), models.refresh()]);
    assert.equal(n, 1);
    assert.deepEqual(app.get().models.defaults, { tts: 'kokoro', stt: 'Systran/faster-whisper-base' });
    assert.equal(await models.refresh({ silent: true }), null);
    assert.equal(app.get().models.error, 'boom');
    assert.equal(app.get().models.list.length, 1, 'the last good list is kept');
    await assert.rejects(models.refresh(), /boom/);
  });

  it('capabilities() normalises flags and the input limit', async () => {
    const { models } = await setup({ routes: { 'GET /api/tts/capabilities': () => json(200, { capabilities: { voice_blend: true, streaming: true }, max_input_length: 4096, max_input_chars: 2000 }) } });
    const caps = await models.capabilities('kokoro');
    assert.deepEqual(caps, { voice_blend: true, streaming: true, max_input_length: 2000 });
    assert.equal(env.fetch.calls[0].search, '?model=kokoro');
  });
});

describe('core/models: freshness, joining and timeouts (foundation repair)', () => {
  const turn = async (n = 10) => { for (let i = 0; i < n; i += 1) await new Promise((r) => setImmediate(r)); };

  it('[SP-12] an inventory request that started before a load finished never overwrites the merged "loaded" row', async () => {
    let releaseList;
    let state = 'downloaded';
    const { models, app } = await setup({
      list: [row('kokoro', 'downloaded')],
      routes: {
        'GET /api/models': () => { const snap = state; return new Promise((r) => { releaseList = () => r(json(200, { models: [row('kokoro', snap)] })); }); },
        'GET /api/models/:id/status': () => json(200, { state }),
        'POST /api/models/:id/load': () => { state = 'loaded'; return json(200, {}); },
      },
    });
    const background = models.refresh({ silent: true });   // the 60 s poll starts while kokoro is 'downloaded'
    await turn();
    const firstRelease = releaseList;
    const ready = models.ensureReady('kokoro', { confirmSwitch: false });
    await turn(20);
    // ensureReady's closing refresh() must not join the older request: a new one goes out.
    assert.equal(env.fetch.callsTo('GET /api/models').length, 2, 'a fresh inventory request, not a join of the stale one');
    releaseList();
    firstRelease();
    assert.equal(await ready, true);
    await background;
    assert.equal(app.get().models.list.find((m) => m.id === 'kokoro').state, 'loaded');
  });

  it('[SP-12] an older inventory answer that lands after a newer one is dropped', async () => {
    const answers = [];
    const { models, app } = await setup({
      list: [row('kokoro', 'downloaded')],
      routes: { 'GET /api/models': () => new Promise((r) => answers.push(r)), 'GET /api/models/:id/status': () => json(200, { state: 'loaded' }) },
    });
    const old = models.refresh({ silent: true });
    await turn();
    await models.ensureReady('kokoro', { confirmSwitch: false, fastPath: false });   // merges 'loaded' (a status check)
    const newer = models.refresh({ silent: true });
    await turn();
    assert.equal(answers.length, 2);
    answers[1](json(200, { models: [row('kokoro', 'loaded')] }));
    await newer;
    answers[0](json(200, { models: [row('kokoro', 'downloaded')] }));
    await old;
    assert.equal(app.get().models.list[0].state, 'loaded');
  });

  it('[REV-08] Generate during a Models-view load waits for that load instead of failing with "state=loading"', async () => {
    let finishLoad;
    let state = 'downloaded';
    const { models } = await setup({
      route: 'models',
      list: [row('piper/en_US-amy-low', 'downloaded')],
      defaults: { tts: '', stt: '' },
      routes: {
        'GET /api/models': () => json(200, { models: [row('piper/en_US-amy-low', state)] }),
        'GET /api/models/:id/status': () => json(200, { state }),
        'POST /api/models/:id/load': () => { state = 'loading'; return new Promise((r) => { finishLoad = () => { state = 'loaded'; r(json(200, {})); }; }); },
      },
    });
    const op = models.runOp('piper/en_US-amy-low', 'load');
    await turn();
    let outcome = null;
    models.ensureReady('piper/en_US-amy-low', { confirmSwitch: false }).then((v) => { outcome = v; }, (e) => { outcome = e; });
    await env.clock.tickAsync(4000);
    assert.equal(outcome, null, 'still waiting for the load');
    finishLoad();
    await env.clock.tickAsync(2000);
    assert.equal(await op, true);
    await turn();
    assert.equal(outcome, true);
    assert.equal(env.fetch.callsTo('POST /api/models/:id/load').length, 1, 'no second load');
  });

  it('[REV-08] a model loading elsewhere (state=loading in its status) is waited for, polling every 2 s', async () => {
    let state = 'loading';
    const { models } = await setup({ list: [row('kokoro', 'downloaded')], routes: { 'GET /api/models/:id/status': () => json(200, { state }), 'GET /api/models': () => json(200, { models: [row('kokoro', state)] }) } });
    let outcome = null;
    models.ensureReady('kokoro', { confirmSwitch: false }).then((v) => { outcome = v; }, (e) => { outcome = e; });
    await env.clock.tickAsync(4000);
    assert.equal(outcome, null);
    state = 'loaded';
    await env.clock.tickAsync(2000);
    assert.equal(outcome, true);
    assert.equal(env.fetch.callsTo('POST /api/models/:id/load').length, 0, 'it never loads it again');
  });

  it('concurrent ensureReady calls for one model share one run (one status check, one load)', async () => {
    const srv = server({ kokoro: 'downloaded' });
    const { models } = await setup({ list: [row('kokoro', 'downloaded')], routes: srv.routes });
    const [a, b] = await Promise.all([models.ensureReady('kokoro', { confirmSwitch: false }), models.ensureReady('kokoro', { confirmSwitch: false })]);
    assert.deepEqual([a, b], [true, true]);
    assert.equal(env.fetch.callsTo('POST /api/models/:id/load').length, 1);
  });

  it('a stalled /api/models or status request times out after 15 s, so later refreshes are not stuck behind it', async () => {
    let stall = true;
    const { models, app } = await setup({ routes: { 'GET /api/models': () => (stall ? new Promise(() => {}) : json(200, { models: [row('kokoro', 'loaded')] })), 'GET /api/models/:id/status': () => new Promise(() => {}) } });
    const first = models.refresh().catch((e) => e);
    await env.clock.tickAsync(15000);
    assert.equal((await first).kind, 'timeout');
    stall = false;
    await models.refresh();
    assert.equal(app.get().models.list[0].state, 'loaded');
    const status = models.getStatus('kokoro').catch((e) => e);
    await env.clock.tickAsync(15000);
    assert.equal((await status).kind, 'timeout');
  });
});

describe('core/models: an offline worker with a loaded model, refresh after a key change', () => {
  const turn = async (n = 10) => { for (let i = 0; i < n; i += 1) await new Promise((r) => setImmediate(r)); };

  it('[MD-02] uiState: a loaded model whose worker stopped answering reads "Worker offline", never "Worker not configured"; a baked provider keeps its advice', async () => {
    const { models } = await setup();
    // src/model_manager.py list_all keeps state 'loaded' when the worker stops answering and only sets provider_available false.
    const stale = row('chatterbox/regular', 'loaded', { device: 'cuda:0', provider_available: false });
    const s = models.uiState(stale);
    assert.deepEqual([s.key, s.label, s.tone, s.action], ['worker-offline', 'Worker offline', 'error', null]);
    assert.equal(s.hint.text, "The voice worker isn't responding. Check the worker service.");
    assert.doesNotMatch(JSON.stringify(s), /not configured|docker compose/i, 'an offline worker is never described as not installed');
    assert.equal(s.signature, models.uiState(row('chatterbox/regular', 'provider_unavailable', { provider_available: false })).signature, 'the same row as an offline worker');
    // The id alone names the provider when the row has none.
    assert.equal(models.uiState({ id: 'fish/s2-pro', type: 'tts', state: 'loaded', provider_available: false }).key, 'worker-offline');
    // Unchanged: a loaded model with its worker up, and the baked providers (no worker to go offline).
    assert.equal(models.uiState(row('chatterbox/regular', 'loaded', { device: 'cuda:0' })).key, 'loaded');
    assert.equal(models.uiState(row('kokoro', 'loaded', { provider_available: false })).key, 'not-installed');
    assert.equal(models.uiState(row('chatterbox/turbo', 'provider_missing', { provider_available: false })).label, 'Worker not configured');
    // An operation in flight or a failed one still wins over the row's state.
    assert.equal(models.uiState(stale, { kind: 'unload', startedAt: 1 }).key, 'unloading');
    assert.equal(models.uiState(stale, { kind: 'unload', error: 'Worker gone' }).key, 'error');
  });

  // A worker that answers /health with status "failed"
  // and a usable failure reaches its rows as provider_failure {code, message}, on a row
  // with provider_available false whose state is provider_unavailable, or loaded for a model the worker still
  // reports loaded. It answers, so it isn't "offline": the vocabulary says it can't run, with the worker's reason.
  const REASON = 'Torch wheel has no code that runs on sm_120 (built for: sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90)';
  const FAILURE = Object.freeze({ code: 'cuda_arch_unsupported', message: REASON });
  const unavailable = (id, extra = {}) => row(id, 'provider_unavailable', { provider_available: false, ...extra });

  it("[MD-02] uiState: a worker that answers but can't run reads \"Worker can't run\" with its reason, never \"Worker offline\"; without a reason it stays \"Worker offline\"", async () => {
    const { models } = await setup();
    const failed = models.uiState(unavailable('chatterbox/turbo', { provider_failure: FAILURE }));
    assert.deepEqual([failed.key, failed.label, failed.tone, failed.action, failed.busy], ['worker-failed', "Worker can't run", 'error', null, false]);
    assert.deepEqual(failed.hint, { kind: 'failed', text: `The voice worker can't run: ${REASON}.` });
    assert.doesNotMatch(JSON.stringify(failed), /offline|isn't responding|not configured|docker compose/i, 'it answers: never offline, never not installed');
    assert.notEqual(failed.signature, models.uiState(unavailable('chatterbox/turbo')).signature, 'a row that changes between the two is patched');
    // A model the worker still reports loaded keeps state 'loaded' on the server: the same reason.
    const loaded = models.uiState(row('chatterbox/regular', 'loaded', { device: 'cuda:0', provider_available: false, provider_failure: FAILURE }));
    assert.deepEqual([loaded.key, loaded.label, loaded.hint.text], ['worker-failed', "Worker can't run", failed.hint.text]);
    // No usable reason (the server sends null when the worker doesn't answer): "Worker offline", as before.
    for (const none of [null, undefined, {}, { code: 'cuda_arch_unsupported' }, { code: 'x', message: '' }, { code: 'x', message: '   ' }, { code: 'x', message: 5 }, 'Torch says no', ['Torch says no']]) {
      const s = models.uiState(unavailable('chatterbox/turbo', { provider_failure: none }));
      assert.deepEqual([s.key, s.label, s.hint.text], ['worker-offline', 'Worker offline', "The voice worker isn't responding. Check the worker service."], JSON.stringify(none));
    }
    // A reason that ends its own sentence keeps its stop; one the server cut keeps its ellipsis.
    const said = (message) => models.uiState(unavailable('qwen3/0.6b-base', { provider_failure: { code: 'dtype_unsupported', message } })).hint.text;
    assert.equal(said('bfloat16 needs compute capability 8.0 or newer.'), "The voice worker can't run: bfloat16 needs compute capability 8.0 or newer.");
    assert.equal(said(`${'x'.repeat(20)}…`), `The voice worker can't run: ${'x'.repeat(20)}…`);
    // A row whose worker is up ignores a stale reason; an operation in flight or a failed one still wins.
    assert.equal(models.uiState(row('chatterbox/turbo', 'downloaded', { provider_failure: FAILURE })).key, 'downloaded');
    assert.equal(models.uiState(unavailable('chatterbox/turbo', { provider_failure: FAILURE }), { kind: 'load', startedAt: 1 }).key, 'loading');
    assert.equal(models.uiState(unavailable('chatterbox/turbo', { provider_failure: FAILURE }), { kind: 'load', error: 'boom' }).key, 'error');
  });

  it("a status check carries the worker's reason into the row and takes it away again, so the row never keeps a stale one", async () => {
    let failure = FAILURE;
    const routes = {
      'GET /api/models/*/status': () => json(200, { model: 'chatterbox/turbo', state: 'provider_unavailable', progress: 0, provider_available: false, provider_failure: failure }),
    };
    const { models, app } = await setup({ routes, list: [unavailable('chatterbox/turbo', { provider_failure: null })] });
    const listed = () => app.get().models.list.find((m) => m.id === 'chatterbox/turbo');
    await assert.rejects(models.ensureReady('chatterbox/turbo'));
    assert.deepEqual(listed().provider_failure, FAILURE);
    assert.equal(models.uiState(listed()).key, 'worker-failed');
    failure = null;   // the worker stopped answering
    await assert.rejects(models.ensureReady('chatterbox/turbo'));
    assert.equal(listed().provider_failure, null);
    assert.equal(models.uiState(listed()).key, 'worker-offline');
  });

  // ensureReady() (Speak's Generate, the Reader, Live, Voices) said "The selected voice
  // provider is offline" for a worker that answers but can't run, losing the reason the status check carries.
  it("[SP-12] ensureReady: a worker that answers but can't run fails with its reason, never \"offline\"; one that doesn't answer is still offline", async () => {
    let failure = FAILURE;
    const routes = {
      'GET /api/models/*/status': () => json(200, { model: 'chatterbox/turbo', state: 'provider_unavailable', progress: 0, provider_available: false, provider_failure: failure }),
    };
    const { models } = await setup({ routes, list: [unavailable('chatterbox/turbo')] });
    const said = async () => {
      try {
        await models.ensureReady('chatterbox/turbo');
      } catch (error) {
        return error.message;
      }
      return 'no error';
    };
    assert.equal(await said(), `The voice worker can't run: ${REASON}.`);
    failure = { code: 'dtype_unsupported', message: 'bfloat16 needs compute capability 8.0 or newer.' };
    assert.equal(await said(), "The voice worker can't run: bfloat16 needs compute capability 8.0 or newer.", 'its own stop is kept');
    failure = null;   // the worker doesn't answer
    assert.equal(await said(), 'The selected voice provider is offline');
    assert.deepEqual(paths().filter((p) => p.startsWith('POST')), [], 'nothing is loaded either way');
  });

  it('[AU-09] a refresh after a key change never joins the request still out without it: a second request carries the new key', async () => {
    const answers = [];
    const { models, app } = await setup({ routes: { 'GET /api/models': (req) => new Promise((r) => answers.push({ req, r })) } });
    const auth = await import('../../../src/static/js/core/auth.js');
    auth.init();
    const old = models.refresh().catch((e) => e);   // the 60 s poll, sent without a key on a keyed server
    await turn();
    assert.equal(answers.length, 1);
    assert.equal(answers[0].req.headers.authorization, undefined);
    assert.equal(models.refresh() === models.refresh(), true, 'the same key still shares one request');
    auth.setKey('k-new');
    const fresh = models.refresh();
    await turn();
    assert.equal(answers.length, 2, 'a request with the new key goes out at once');
    assert.equal(answers[1].req.headers.authorization, 'Bearer k-new');
    // The old request's 401 arrives first: it is about the key no longer in use, and it is not recorded.
    answers[0].r(json(401, { error: { message: 'Invalid API key' } }));
    const oldError = await old;
    assert.match(oldError.message, /The API key changed while this request was running/);
    assert.equal(app.get().models.error, '', "the replaced request's failure is not the inventory's state");
    assert.equal(auth.isRefused(), false, 'the new key was not judged by the old answer');
    answers[1].r(json(200, { models: [row('kokoro', 'loaded')], default_tts_model: 'kokoro' }));
    await fresh;
    assert.deepEqual(app.get().models.list.map((m) => m.id), ['kokoro']);
    assert.equal(app.get().connection.state, 'online');
  });

  it("[AU-09] an old key's success never stands for a new, wrong key: the new key's own answer is the one callers get", async () => {
    const answers = [];
    const { models, app } = await setup({ routes: { 'GET /api/models': (req) => new Promise((r) => answers.push({ req, r })) } });
    const auth = await import('../../../src/static/js/core/auth.js');
    auth.init();
    const old = models.refresh({ silent: true });
    await turn();
    auth.setKey('k-wrong');
    const check = models.refresh().catch((e) => e);
    await turn();
    assert.equal(answers.length, 2);
    answers[0].r(json(200, { models: [row('kokoro', 'loaded')] }));   // the keyless server answered the old request
    await old;
    answers[1].r(json(401, { error: { message: 'Invalid API key' } }));
    const error = await check;
    assert.equal(error.kind, 'auth', 'the caller that checks the new key learns it was refused');
    assert.equal(auth.isRefused(), true);
    assert.equal(app.get().connection.state, 'unauthorized');
    assert.equal(app.get().models.error, error.message);
  });
});
