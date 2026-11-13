// Shared setup for the Studio tests: the real shell (index.html) and the real studio partial, the real view
// host and router, and a small in-memory server that behaves like src/conversation.py and src/composer.py
// (turns appended at the end, re-indexed after a delete, a turn change retires the render). Not a test file.
import assert from 'node:assert/strict';
import { shellEnv, load, settle, json, text } from '../shell/helpers.js';
import { bytes, delay, networkError } from '../harness/fakes.js';

export { load, settle, json, text, bytes, delay, networkError };

/**
 * A short, bounded description of a node for a failure message: `<button#st-play data-action="play-render">
 * "Play render"`, `#text "Hello"`, `#document`, `null`.
 * @param {*} n
 * @returns {string}
 */
export function describeNode(n) {
  if (n === null || n === undefined) return String(n);
  if (typeof n !== 'object') return `${typeof n} ${String(n).slice(0, 60)}`;
  if (n.nodeType === 9) return '#document';
  if (n.nodeType === 3) return `#text ${JSON.stringify(String(n.textContent || '').slice(0, 40))}`;
  if (n.nodeType !== 1) return `node (type ${n.nodeType})`;
  const attr = (name) => (typeof n.getAttribute === 'function' ? n.getAttribute(name) : null);
  const id = attr('id');
  const action = attr('data-action');
  const cls = attr('class');
  const words = String(n.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 40);
  return `<${n.localName || '?'}${id ? `#${id}` : ''}${action ? ` data-action="${action}"` : ''}${!id && cls ? ` class="${cls}"` : ''}>${words ? ` "${words}"` : ''}`;
}

/**
 * Node identity checks. Never pass a harness node to assert.equal/deepEqual: when it fails, node:assert inspects
 * actual and expected deeply, and a harness node reaches the whole document, window and env through its links,
 * so one failure allocates without bound (a mutation run reached the 2 GB cap that way). These fail with a
 * short description of each node instead (the same helpers as tests/js/transcribe/helpers.js).
 * @param {*} actual
 * @param {*} expected
 * @param {string} [message]
 */
export function sameNode(actual, expected, message = '') {
  if (actual === expected) return;
  assert.fail(`${message ? `${message}: ` : ''}got ${describeNode(actual)}, expected ${describeNode(expected)}`);
}

/** No such node (null or undefined), with the same bounded failure message as sameNode(). */
export function noNode(actual, message = '') {
  if (actual === null || actual === undefined) return;
  assert.fail(`${message ? `${message}: ` : ''}expected no node, got ${describeNode(actual)}`);
}

export const MODELS = {
  models: [
    { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' },
    { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'downloaded', device: 'cpu' },
  ],
  default_tts_model: 'kokoro',
  default_stt_model: 'Systran/faster-whisper-base',
};

export function presets() {
  return {
    profiles: [
      { id: 'p1', name: 'Audiobook 1.1×', model: 'kokoro', voice: 'af_heart', speed: 1.1, format: 'mp3', effects: [], is_default: true },
      { id: 'p2', name: 'Podcast EQ', model: 'kokoro', voice: 'am_adam', speed: 1, format: 'mp3', effects: [{ type: 'podcast_eq' }], is_default: false },
    ],
    default_profile_id: 'p1',
  };
}

const ENVELOPE = (status, message) => json(status, { error: { message, code: 'http_error' } });

/**
 * A stateful fake of the conversation and composer routes.
 * @param {{conversations?: object[], mixes?: object[], presets?: object}} [seed]
 */
export function studioServer(seed = {}) {
  let n = 0;
  const next = (prefix) => `${prefix}${(n += 1)}`;
  const iso = () => new Date(Date.now()).toISOString();
  const state = {
    convs: new Map(),
    mixes: [],
    presets: seed.presets || presets(),
    renderDurationMs: 4200,
    mixDurationMs: 9000,
  };
  for (const c of seed.conversations || []) {
    state.convs.set(c.id, {
      id: c.id, name: c.name ?? '', created_at: c.created_at || iso(), updated_at: c.updated_at || iso(),
      render_output_path: c.render_output_path || null, meta_json: '{}',
      turns: (c.turns || []).map((t, i) => ({
        id: t.id || next('t'), conversation_id: c.id, turn_index: i, speaker: t.speaker || 'Speaker', text: t.text || '',
        profile_id: t.profile_id || null, effects: t.effects || [], audio_path: null, duration_ms: t.duration_ms ?? null, created_at: iso(),
      })),
    });
  }
  for (const m of seed.mixes || []) state.mixes.push({ meta: {}, tracks: [], ...m });

  const summary = (c) => ({ id: c.id, name: c.name, created_at: c.created_at, updated_at: c.updated_at, render_output_path: c.render_output_path, meta_json: '{}' });
  const full = (c) => ({ ...summary(c), turns: c.turns.map((t) => ({ ...t })) });
  const changed = (c) => {
    c.render_output_path = null;
    c.updated_at = iso();
  };

  const routes = {
    'GET /api/models': () => json(200, MODELS),
    'GET /api/profiles': () => json(200, state.presets),
    'GET /api/conversations': () => {
      const items = [...state.convs.values()].sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at))).map(summary);
      return json(200, { items, total: items.length });
    },
    'POST /api/conversations': (req) => {
      const id = next('c');
      const c = { id, name: req.json.name, created_at: iso(), updated_at: iso(), render_output_path: null, meta_json: '{}', turns: [] };
      state.convs.set(id, c);
      return json(201, full(c));
    },
    'GET /api/conversations/:id': (req) => {
      const c = state.convs.get(req.params.id);
      return c ? json(200, full(c)) : ENVELOPE(404, 'Conversation not found');
    },
    'POST /api/conversations/:id/turns': (req) => {
      const c = state.convs.get(req.params.id);
      if (!c) return ENVELOPE(404, 'Conversation not found');
      const body = req.json;
      const turn = {
        id: next('t'), conversation_id: c.id, turn_index: c.turns.length, speaker: body.speaker, text: body.text,
        profile_id: body.profile_id, effects: body.effects || [], audio_path: null, duration_ms: null, created_at: iso(),
      };
      c.turns.push(turn);
      changed(c);
      return json(201, { ...turn });
    },
    'DELETE /api/conversations/:id/turns/:tid': (req) => {
      const c = state.convs.get(req.params.id);
      const index = c ? c.turns.findIndex((t) => t.id === req.params.tid) : -1;
      if (index < 0) return ENVELOPE(404, 'Turn not found');
      c.turns.splice(index, 1);
      c.turns.forEach((t, i) => { t.turn_index = i; });
      changed(c);
      return json(204);
    },
    'POST /api/conversations/:id/render': (req) => {
      const c = state.convs.get(req.params.id);
      if (!c) return ENVELOPE(404, 'Conversation not found');
      if (!c.turns.length) return ENVELOPE(400, 'Conversation has no turns');
      const format = req.json.format;
      c.render_output_path = `/data/conversations/${c.id}/render.${format}`;
      c.updated_at = iso();
      for (const t of c.turns) t.duration_ms = 1200;
      return json(200, { conversation_id: c.id, output_path: c.render_output_path, download_url: `/api/conversations/${c.id}/audio`, duration_ms: state.renderDurationMs, turn_count: c.turns.length });
    },
    'GET /api/conversations/:id/audio': (req) => {
      const c = state.convs.get(req.params.id);
      if (!c || !c.render_output_path) return ENVELOPE(404, 'Conversation has no rendered output');
      return bytes(200, [82, 73, 70, 70], c.render_output_path.endsWith('.mp3') ? 'audio/mpeg' : 'audio/wav');
    },
    'DELETE /api/conversations/:id': (req) => {
      if (!state.convs.delete(req.params.id)) return ENVELOPE(404, 'Conversation not found');
      return json(204);
    },
    'GET /api/composer/renders': () => json(200, { items: state.mixes.map((m) => ({ ...m })), total: state.mixes.length, limit: 50, offset: 0 }),
    'POST /api/composer/render': (req) => {
      const body = req.json;
      const id = next('m');
      const ext = body.format === 'mp3' ? 'mp3' : 'wav';
      const mix = {
        id, name: body.name, sample_rate: body.sample_rate, created_at: iso(), updated_at: iso(),
        render_output_path: `data/composer/render_${id}.${ext}`, tracks: body.tracks, meta: { format: ext, duration_ms: state.mixDurationMs },
      };
      state.mixes.unshift(mix);
      return json(200, { composition_id: id, output_path: mix.render_output_path, download_url: `/api/composer/render/${id}/audio`, duration_ms: state.mixDurationMs });
    },
    'GET /api/composer/render/:id/audio': (req) => {
      const mix = state.mixes.find((m) => m.id === req.params.id);
      return mix ? bytes(200, [82, 73, 70, 70], 'audio/wav') : ENVELOPE(404, 'Composition not found');
    },
    'DELETE /api/composer/render/:id': (req) => {
      const index = state.mixes.findIndex((m) => m.id === req.params.id);
      if (index < 0) return ENVELOPE(404, 'Composition not found');
      state.mixes.splice(index, 1);
      return json(204);
    },
  };
  return { state, routes, envelope: ENVELOPE };
}

/** The seeded conversation of the preview server: three turns rendered to WAV. */
export function teaParty(extra = {}) {
  return {
    id: 'conv-1', name: 'The Mad Tea-Party', render_output_path: '/data/conversations/conv-1/render.wav',
    turns: [
      { id: 't-a', speaker: 'Hatter', text: 'Your hair wants cutting.', profile_id: 'p1', duration_ms: 1200 },
      { id: 't-b', speaker: 'Alice', text: 'You should learn not to make personal remarks.\nIt’s very rude.', profile_id: 'p2', duration_ms: 1200 },
      { id: 't-c', speaker: 'March Hare', text: 'Then you should say what you mean.', profile_id: null, duration_ms: 1200 },
    ],
    ...extra,
  };
}

const HISTORY_STUB = '<header class="view-head"><h1 id="hi-title" class="view-head__title" data-view-heading tabindex="-1">History</h1></header>';

function stubView(id, title, partial) {
  return async () => ({ default: { id, title, partial, init: () => ({ show() {}, hide() {} }) } });
}

/**
 * Boot the fake browser on #<path> with the studio view mounted through the real view host and router.
 * @param {{path?: string, server?: object, routes?: object, seed?: object}} [options] routes override the
 *   server's; other options go to installBrowserEnv (seed, playPolicy…)
 */
export async function studioEnv({ path = 'studio/conversations', server = studioServer(), routes = {}, ...options } = {}) {
  const env = await shellEnv({
    routes: {
      'GET /static/partials/history.html': () => text(200, HISTORY_STUB, { 'content-type': 'text/html' }),
      ...server.routes,
      ...routes,
    },
    ...options,
  });
  const viewHost = await load('shell/view-host.js');
  const router = await load('core/router.js');
  const store = await load('core/store.js');
  const auth = await load('core/auth.js');
  auth.init();
  viewHost.initViewHost({
    registry: {
      studio: () => load('views/studio/index.js'),
      history: stubView('history', 'History', '/static/partials/history.html'),
    },
  });
  env.setHash(`#${path}`);
  await router.start(viewHost.show);
  await settle(env);
  await settle(env);
  const section = document.getElementById('view-studio');
  const ctx = {
    env, router, viewHost, store, app: store.app, section, server,
    controller: viewHost.controllerFor('studio'),
    $: (id) => section.querySelector(`#${id}`),
    $$: (sel) => [...section.querySelectorAll(sel)],
    calls: (pattern) => env.fetch.callsTo(pattern),
    settle: (ms = 0) => settle(env, ms),
    async go(to, options) {
      await router.navigate(to, options);
      await settle(env);
      await settle(env);
    },
    async click(el) {
      if (!el) throw new Error('click(): no element');
      env.click(el);
      await settle(env);
      await settle(env);
    },
    /** Open an item's "More actions" menu and choose an item by its text. */
    async menu(button, label) {
      env.click(button);
      await settle(env);
      const menu = document.getElementById(button.getAttribute('aria-controls'));
      if (!menu || menu.hidden) throw new Error(`menu ${button.getAttribute('aria-controls')} did not open`);
      const item = [...menu.querySelectorAll('[role="menuitem"]')].find((el) => el.textContent.trim() === label);
      if (!item) throw new Error(`no menu item ${label}`);
      env.click(item);
      await settle(env);
      await settle(env);
      return item;
    },
    menuItems(button) {
      const menu = document.getElementById(button.getAttribute('data-menu-btn'));
      return menu ? [...menu.querySelectorAll('[role="menuitem"]')] : [];
    },
    async answerConfirm(yes = true) {
      const dialog = document.getElementById('app-confirm');
      if (!dialog.open) throw new Error('no confirm is open');
      env.click(document.getElementById(yes ? 'app-confirm-ok' : 'app-confirm-cancel'));
      await settle(env);
      await settle(env);
      await settle(env);
    },
    confirmText() {
      return {
        title: document.getElementById('app-confirm-title').textContent,
        message: document.getElementById('app-confirm-msg').textContent,
        open: document.getElementById('app-confirm').open,
      };
    },
    async answerPrompt(value) {
      const dialog = document.getElementById('app-prompt');
      if (!dialog.open) throw new Error('no prompt is open');
      if (value === null) {
        env.click(document.getElementById('app-prompt-cancel'));
      } else {
        document.getElementById('app-prompt-input').value = value;
        env.click(document.getElementById('app-prompt-ok'));
      }
      await settle(env);
      await settle(env);
      await settle(env);
    },
    toasts() {
      return [...document.querySelectorAll('.toast')].map((t) => ({ kind: t.className.replace(/.*toast--(\w+).*/, '$1'), text: t.querySelector('.toast__msg').textContent }));
    },
  };
  return ctx;
}

/** Record the <a download> links the app clicks (downloadFromApi / downloadBlob). */
export function captureDownloads(env) {
  const clicks = [];
  const original = document.createElement.bind(document);
  document.createElement = (tag, ...rest) => {
    const el = original(tag, ...rest);
    if (String(tag).toLowerCase() === 'a') {
      el.addEventListener('click', (event) => {
        event.preventDefault();
        const href = el.getAttribute('href');
        clicks.push({ href, download: el.getAttribute('download'), blob: env.objectUrls.get(href) || null });
      });
    }
    return el;
  };
  return clicks;
}
