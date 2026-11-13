// Shared setup for the Models tests: the real shell markup (live regions, dialogs, toasts) with the
// real partial mounted in #view-models, the store seeded with an inventory like the server's, and the
// fake server routes for /api/models and the model operations.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, loadHtml, importFresh, flushMicrotasks, STATIC_JS, REPO_ROOT } from '../harness/env.js';
import { json, text, delay, networkError } from '../harness/fakes.js';

export { json, text, delay, networkError };

export const PARTIAL = readFileSync(resolve(REPO_ROOT, 'src/static/partials/models.html'), 'utf8');
export const STT_DEFAULT = 'Systran/faster-whisper-base';

const speakers = (n) => Array.from({ length: n }, (_, i) => ({ name: `voice_${i}`, description: `Voice ${i}` }));

/** An inventory shaped like GET /api/models (src/model_manager.py ModelInfo.to_dict plus capabilities). */
export function inventory() {
  return [
    { id: 'kokoro', type: 'tts', provider: 'kokoro', device: 'cuda:0', state: 'loaded', size_mb: 330, is_default: true, provider_available: true,
      description: 'Fast, 52 voices, voice blending', capabilities: { voice_blend: true, streaming: false, voice_clone: false, speakers: speakers(54) } },
    { id: 'pocket-tts', type: 'tts', provider: 'pocket-tts', device: null, state: 'provider_missing', size_mb: 220, provider_available: false },
    { id: 'chatterbox/regular', type: 'tts', provider: 'chatterbox', device: null, state: 'provider_unavailable', size_mb: 8800, provider_available: false, model_format: 'external' },
    { id: 'chatterbox/turbo', type: 'tts', provider: 'chatterbox', device: null, state: 'provider_unavailable', size_mb: 5500, provider_available: false, model_format: 'external' },
    // The server lists Piper voices in registry order; the card sorts them (MD-10).
    { id: 'piper/en_GB-alan-medium', type: 'tts', provider: 'piper', state: 'downloaded', size_mb: 35, provider_available: true, capabilities: {} },
    { id: 'piper/en_US-ryan-high', type: 'tts', provider: 'piper', state: 'available', size_mb: 75, provider_available: true, capabilities: {} },
    { id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'downloaded', size_mb: 35, provider_available: true, capabilities: {} },
    { id: 'piper/en_US-lessac-low', type: 'tts', provider: 'piper', state: 'available', size_mb: 6, provider_available: true, capabilities: {} },
    { id: 'piper/en_US-arctic-medium', type: 'tts', provider: 'piper', state: 'available', size_mb: 35, provider_available: true, capabilities: {} },
    { id: 'piper/en_GB-cori-high', type: 'tts', provider: 'piper', state: 'available', size_mb: 75, provider_available: true, capabilities: {} },
    { id: 'piper/en_US-joe-medium', type: 'tts', provider: 'piper', state: 'available', size_mb: 35, provider_available: true, capabilities: {} },
    { id: STT_DEFAULT, type: 'stt', provider: 'faster-whisper', device: 'cpu', state: 'loaded', size_mb: 150, source: 'Systran', model_format: 'CT2', description: 'Good balance for CPU', is_default: true, provider_available: true },
    { id: 'Systran/faster-whisper-tiny', type: 'stt', provider: 'faster-whisper', state: 'available', size_mb: 75, source: 'Systran', model_format: 'CT2', description: 'Fastest, lowest quality', provider_available: true },
    { id: 'Systran/faster-distil-whisper-small.en', type: 'stt', provider: 'faster-whisper', state: 'downloaded', size_mb: 250, source: 'Systran', model_format: 'CT2', description: 'Distil-small English-only, fast CPU', provider_available: true },
    { id: 'deepdml/faster-whisper-large-v3-turbo-ct2', type: 'stt', provider: 'faster-whisper', state: 'available', size_mb: 1600, source: 'deepdml', model_format: 'CT2', description: 'Large-v3-turbo, near large-v3 accuracy at 3-4x speed', provider_available: true },
    { id: 'Systran/faster-whisper-small', type: 'stt', provider: 'faster-whisper', state: 'available', size_mb: 500, source: 'Systran', model_format: 'CT2', description: 'Better accuracy', provider_available: true },
    { id: 'Systran/faster-whisper-medium', type: 'stt', provider: 'faster-whisper', state: 'available', size_mb: 1500, source: 'Systran', model_format: 'CT2', description: 'High accuracy', provider_available: true },
    { id: 'Systran/faster-whisper-large-v3', type: 'stt', provider: 'faster-whisper', state: 'available', size_mb: 3000, source: 'Systran', model_format: 'CT2', description: 'Large-v3, high accuracy', provider_available: true },
  ];
}

/** Import a module from src/static/js: the same instance the view uses. */
export function load(rel) {
  return import(pathToFileURL(resolve(STATIC_JS, rel)).href);
}

/** Let promises, timers due now and frames settle. */
export async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

/**
 * A tiny model server: the inventory lives in `server.list`; GET /api/models answers it, the status route
 * answers one row, and load/unload/download/prefetch/artifacts change it (instantly unless told otherwise).
 */
export function fakeServer(list = inventory()) {
  const server = {
    list,
    defaults: { tts: 'kokoro', stt: STT_DEFAULT },
    loadMs: 0,
    fail: {},   // id -> {op: 'load'|'unload'|'artifacts'|..., status, message}
    find(id) { return server.list.find((m) => m.id === id); },
    set(id, patch) {
      server.list = server.list.map((m) => (m.id === id ? { ...m, ...patch } : m));
    },
  };
  const failure = (id, op) => {
    const f = server.fail[id];
    if (!f || f.op !== op) return null;
    return json(f.status || 500, { error: { message: f.message || 'boom', code: f.code || 'failed' } });
  };
  const routes = {
    'GET /api/models': () => json(200, { models: server.list, default_tts_model: server.defaults.tts, default_stt_model: server.defaults.stt }),
    'GET /api/models/:id/status': (req) => {
      const row = server.find(req.params.id);
      return row ? json(200, { model: row.id, state: row.state, device: row.device || null, progress: row.progress ?? null }) : json(404, { error: { message: 'unknown model', code: 'unknown_model' } });
    },
    'POST /api/models/:id/load': (req) => {
      const id = req.params.id;
      const bad = failure(id, 'load');
      if (bad) return bad;
      const row = server.find(id);
      const finish = () => {
        // One speech model at a time: loading a TTS model unloads the other.
        if (row.type === 'tts') for (const m of server.list) if (m.type === 'tts' && m.state === 'loaded' && m.id !== id) server.set(m.id, { state: 'downloaded', device: null });
        server.set(id, { state: 'loaded', device: row.type === 'tts' ? 'cuda:0' : 'cpu' });
        return json(200, { model: id, state: 'loaded' });
      };
      if (server.loadMs) {
        server.set(id, { state: 'loading' });
        return delay(server.loadMs, finish);
      }
      return finish();
    },
    'POST /api/models/:id/prefetch': (req) => {
      const id = req.params.id;
      const bad = failure(id, 'prefetch');
      if (bad) return bad;
      server.set(id, { state: 'downloaded' });
      return json(200, { model: id, state: 'downloaded' });
    },
    'POST /api/models/:id/download': (req) => {
      server.set(req.params.id, { state: 'downloaded' });
      return json(200, { model: req.params.id, state: 'downloaded' });
    },
    'DELETE /api/models/:id/artifacts': (req) => {
      const id = req.params.id;
      const bad = failure(id, 'artifacts');
      if (bad) return bad;
      server.set(id, { state: 'available', device: null });
      return json(200, { model: id, deleted: true });
    },
    'DELETE /api/models/:id': (req) => {
      const id = req.params.id;
      const bad = failure(id, 'unload');
      if (bad) return bad;
      server.set(id, { state: 'downloaded', device: null });
      return json(200, { model: id, state: 'downloaded' });
    },
  };
  return { server, routes };
}

/**
 * The fake browser with the shell markup and the Models partial, the view initialised and (by default)
 * shown on `sub`.
 * @param {object} [options]
 * @param {string} [options.sub] 'tts' | 'stt'
 * @param {object} [options.routes] extra fetch routes (they win over the fake server's)
 * @param {object[]|null} [options.models] the inventory already in app.models (null: none fetched yet)
 * @param {object} [options.env] more installBrowserEnv options
 * @param {boolean} [options.show]
 */
export async function mountModels({ sub = 'tts', routes = {}, models: list, env: envOptions = {}, show = true, shot = '' } = {}) {
  const { server, routes: serverRoutes } = fakeServer(list === undefined || list === null ? inventory() : list);
  const env = installBrowserEnv({ document: loadHtml('src/static/index.html'), routes: { ...routes, ...serverRoutes, ...routes }, ...envOptions });
  const mod = await importFresh('views/models/index.js');
  const { app } = await load('core/store.js');
  const dom = await load('core/dom.js');
  const models = await load('core/models.js');
  const route = { view: 'models', sub, id: '', path: `models/${sub}`, shot };
  app.set({
    models: list === null
      ? { list: [], defaults: { tts: '', stt: '' }, fetchedAt: 0, error: '' }
      : { list: server.list, defaults: { tts: 'kokoro', stt: STT_DEFAULT }, fetchedAt: Date.now(), error: '' },
    connection: { ...app.get().connection, state: 'online' },
    route,
  });
  const root = env.document.getElementById('view-models');
  dom.render(root, dom.raw(PARTIAL));
  root.hidden = false;
  const ctl = mod.default.init(root);
  if (show) ctl.show(route, { focus: 'none', scrolled: false, reason: 'route' });
  await settle(env);
  const $ = (id) => env.document.getElementById(id);
  const row = (id) => root.querySelector(`[data-model-id="${id}"]`);
  const card = (provider) => root.querySelector(`.mo-card[data-provider="${provider}"]`);
  return { env, mod, ctl, app, root, $, row, card, server, models, route, doc: env.document };
}

/** Answer the open confirm dialog. */
export async function answerConfirm(t, ok) {
  const dialog = t.$('app-confirm');
  if (!dialog.open) throw new Error('answerConfirm(): no confirm is open');
  t.env.click(t.$(ok ? 'app-confirm-ok' : 'app-confirm-cancel'));
  await settle(t.env);
}

/** The text of the polite announcement region (after its frame). */
export function announced(t) {
  t.env.raf.flush();
  return t.$('app-sr-status').textContent;
}

/** The visible text of an element, whitespace collapsed. */
export function textOf(el) {
  return String(el ? el.textContent : '').replace(/\s+/g, ' ').trim();
}

/** The ids of the rows shown in a card, in order. */
export function rowIds(cardEl) {
  return cardEl.querySelectorAll('.mo-row').map((li) => li.getAttribute('data-model-id'));
}

/** Toasts on screen: [{kind, text}]. */
export function toasts(t) {
  return t.doc.querySelectorAll('.toast').map((el) => ({ kind: (el.className.match(/toast--(\w+)/) || [])[1], text: textOf(el.querySelector('.toast__msg')) }));
}
