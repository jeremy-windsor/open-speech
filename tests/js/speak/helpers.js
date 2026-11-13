// Shared setup for the Speak tests: the real shell and partial in the fake browser, the server routes
// Speak calls, and the view mounted the way the view host mounts it (render the partial, init, show).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, loadHtml, importFresh, STATIC_JS, REPO_ROOT, flushMicrotasks } from '../harness/env.js';
import { json, bytes, text, delay, networkError } from '../harness/fakes.js';

export { json, bytes, text, delay, networkError, flushMicrotasks };

export const PARTIAL = readFileSync(resolve(REPO_ROOT, 'src/static/partials/speak.html'), 'utf8');

export const KOKORO_VOICES = ['af_heart', 'af_bella', 'af_sky', 'am_adam', 'bf_emma', 'bm_george'].map((id) => ({ id, name: id }));
export const PIPER_VOICES = [{ id: 'en_US-amy-medium', name: 'en_US-amy-medium' }];

export function modelsPayload({ kokoroState = 'loaded', extra = [] } = {}) {
  return {
    models: [
      { id: 'kokoro', type: 'tts', provider: 'kokoro', state: kokoroState, device: 'cuda:0' },
      { id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'downloaded', device: 'cpu' },
      { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'downloaded', device: 'cuda:0' },
      { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'downloaded' },
      ...extra,
    ],
    default_tts_model: 'kokoro',
    default_stt_model: 'Systran/faster-whisper-base',
  };
}

export const CAPS = {
  kokoro: { capabilities: { voice_blend: true, streaming: true, speed_control: true, instructions: false, voice_clone: false, live_reader: true }, max_input_length: 4096 },
  'piper/en_US-amy-medium': { capabilities: ['streaming'], max_input_length: 4096 },
  chatterbox: { capabilities: { voice_blend: false, streaming: false, speed_control: false, instructions: true, voice_clone: true, clone_transcript_required: true, live_reader: false }, max_input_length: 2000, max_input_chars: 3000 },
};

export const NAMED = {
  voices: [
    { id: 'n1', name: 'Narrator', voice: 'voice:n1', available: true, realizations: [{ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', reference_audio_id: null, available: true }] },
    { id: 'n2', name: 'Old clone', voice: 'voice:n2', available: false, realizations: [{ model: 'kokoro', voice: 'af_sky', available: false, reason: 'A saved reference recording is required for this model' }] },
  ],
};

export const RECORDINGS = [{ name: 'demo_reference', transcript: 'Hello there.' }, { name: 'raw_take' }];

export const PRESETS = {
  profiles: [
    { id: 'p1', name: 'Audiobook 1.1×', backend: 'kokoro', model: 'kokoro', voice: 'af_bella', speed: 1.1, format: 'mp3', effects: [] },
    { id: 'p2', name: 'Podcast EQ', backend: 'kokoro', model: 'kokoro', voice: 'af_heart', speed: 1, format: 'mp3', effects: [{ type: 'podcast_eq' }] },
  ],
  default_profile_id: null,
};

export function speechBytes(n = 64) {
  return new Uint8Array(Array.from({ length: n }, (_, i) => i % 251));
}

/** The routes Speak calls, with `extra` first (the fake fetch takes the first match). */
export function routes(extra = {}, { models = modelsPayload(), presets = PRESETS } = {}) {
  // The server's model states: a load makes that model 'loaded' and the other TTS model 'downloaded'.
  const states = new Map(models.models.map((m) => [m.id, m.state]));
  const inventory = () => ({ ...models, models: models.models.map((m) => ({ ...m, state: states.get(m.id) || m.state })) });
  const base = {
    'GET /health': () => json(200, { status: 'ok', version: '0.8.0', models_loaded: 1 }),
    'GET /api/models': () => json(200, inventory()),
    'GET /api/models/:id/status': (req) => json(200, { id: req.params.id, state: states.get(req.params.id) || 'available' }),
    'POST /api/models/:id/load': (req) => {
      for (const m of models.models) if (m.type === 'tts' && states.get(m.id) === 'loaded') states.set(m.id, 'downloaded');
      states.set(req.params.id, 'loaded');
      return json(200, { status: 'loaded' });
    },
    'GET /api/tts/capabilities': (req) => json(200, CAPS[req.query.get('model')] || { capabilities: {}, max_input_length: 4096 }),
    'GET /v1/audio/voices': (req) => {
      const model = req.query.get('model');
      if (model === 'kokoro') return json(200, { voices: KOKORO_VOICES });
      if (model && model.startsWith('piper')) return json(200, { voices: PIPER_VOICES });
      return json(200, { voices: [] });
    },
    'GET /api/voices/identities': () => json(200, NAMED),
    'GET /api/voices/library': () => json(200, RECORDINGS),
    'GET /api/voices/library/:name': (req) => (RECORDINGS.some((r) => r.name === req.params.name) ? json(200, RECORDINGS.find((r) => r.name === req.params.name)) : json(404, { error: { message: 'Voice not found' } })),
    'GET /api/profiles': () => json(200, presets),
    'GET /api/profiles/:id/resolve': (req) => {
      const p = presets.profiles.find((x) => x.id === req.params.id);
      return p ? json(200, { ...p, voice: p.blend || p.voice }) : json(404, { error: { message: 'Preset not found' } });
    },
    'POST /v1/audio/speech': () => bytes(200, speechBytes(), 'audio/mpeg'),
    'GET /static/partials/:name': (req) => text(200, readFileSync(resolve(REPO_ROOT, 'src/static/partials', req.params.name), 'utf8'), { 'content-type': 'text/html; charset=utf-8' }),
  };
  const out = { ...extra };
  for (const [pattern, handler] of Object.entries(base)) if (!(pattern in out)) out[pattern] = handler;
  return out;
}

/** Import a module from src/static/js (the same instance the view uses). */
export function load(rel) {
  return import(pathToFileURL(resolve(STATIC_JS, rel)).href);
}

/** Let promises, due timers and frames settle. */
export async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

/**
 * Mount Speak in the fake browser.
 * @param {object} [options]
 * @param {object} [options.routes] extra fetch routes
 * @param {object} [options.models] the /api/models payload (also put in app.models unless loaded: false)
 * @param {boolean} [options.loaded] app.models is filled before the view shows (default true)
 * @param {object} [options.seed] localStorage
 * @param {object} [options.route] the route passed to show()
 * @param {object} [options.env] more installBrowserEnv options
 */
export async function mountSpeak({ routes: extra = {}, models = modelsPayload(), presets = PRESETS, loaded = true, seed = {}, route = {}, env: envOptions = {}, beforeShow = null } = {}) {
  const env = installBrowserEnv({ document: loadHtml('src/static/index.html'), routes: routes(extra, { models, presets }), seed, ...envOptions });
  const view = await importFresh('views/speak/index.js');
  const { app, setIntent } = await load('core/store.js');
  const dom = await load('core/dom.js');
  const fullRoute = { view: 'speak', sub: '', id: '', path: 'speak', shot: '', ...route };
  app.set({ route: fullRoute, connection: { ...app.get().connection, state: 'online' } });
  if (loaded) {
    app.set({ models: { list: models.models, defaults: { tts: models.default_tts_model || '', stt: models.default_stt_model || '' }, fetchedAt: Date.now(), error: '' } });
  }
  const section = document.getElementById('view-speak');
  dom.render(section, dom.raw(PARTIAL));
  section.hidden = false;
  const ctl = view.default.init(section);
  if (typeof beforeShow === 'function') await beforeShow({ app, setIntent, env });
  ctl.show(fullRoute, { focus: 'none' });
  await settle(env);
  await settle(env);
  const $ = (id) => document.getElementById(id);
  return { env, view, ctl, section, app, setIntent, $, parts: ctl.__parts, ctx: ctl.__ctx };
}

/** Wait for the view's first setup (models, presets, the Default preset). */
export async function ready(t) {
  await t.ctx.setupDone;
  await settle(t.env);
}

/** Type into the editor. */
export function type(t, value) {
  t.env.input(t.$('sp-input'), value);
}

/** The requests made to a path pattern ('POST /v1/audio/speech'). */
export function callsTo(t, pattern) {
  return t.env.fetch.callsTo(pattern);
}

/** Record download-link clicks: [{href, download, blob}] (the blob is the object URL's Blob). */
export function captureDownloads(env) {
  const clicks = [];
  const original = document.createElement.bind(document);
  document.createElement = (tag, ...rest) => {
    const el = original(tag, ...rest);
    if (String(tag).toLowerCase() === 'a') {
      el.addEventListener('click', () => {
        const href = el.getAttribute('href');
        clicks.push({ href, download: el.getAttribute('download'), blob: env.objectUrls.get(href) || null });
      });
    }
    return el;
  };
  return clicks;
}

/** The fake MediaSource instances created so far. */
export async function mediaSources() {
  const { FakeMediaSource } = await import('../harness/fakes.js');
  return FakeMediaSource.instances;
}

/** Bytes of a Blob as a plain array. */
export async function blobBytes(blob) {
  return Array.from(new Uint8Array(await blob.arrayBuffer()));
}

/** Answer the shell's confirm dialog (#app-confirm) once it is open. */
export async function answerConfirm(t, ok = true) {
  await settle(t.env);
  const dialog = document.getElementById('app-confirm');
  if (!dialog || !dialog.open) throw new Error('no confirm is open');
  t.env.click(document.getElementById(ok ? 'app-confirm-ok' : 'app-confirm-cancel'));
  await settle(t.env);
  await settle(t.env);
}
