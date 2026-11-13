// Shared setup for the History tests: the real shell and history partial, the real view host and
// router, and a fake /api/history that filters, searches and pages like src/history.py (newest first; search
// over the whole text and file name; total counts the matches). Not a test file.
import { shellEnv, load, settle, json, text } from '../shell/helpers.js';
import { delay, networkError } from '../harness/fakes.js';
import { routes as speakRoutes, modelsPayload } from '../speak/helpers.js';

export { load, settle, json, text, delay, networkError };
export { describeNode, sameNode, noNode } from '../studio/helpers.js';

const DAY = 86400000;
/** The fake clock's start (installBrowserEnv's default `now`): fixtures are built before it is installed. */
export const NOW = 1760000000000;

/** Midnight today on the fake clock, local time (so day groups do not depend on the time zone). */
export function startOfToday(now = NOW) {
  const d = new Date(now);
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}

/** An ISO time `days` days before today at `hour`:`minute` local time. */
export function at(days, hour = 10, minute = 0) {
  const base = startOfToday() - days * DAY;
  const d = new Date(base);
  return new Date(d.getFullYear(), d.getMonth(), d.getDate(), hour, minute).toISOString();
}

export function speech(i, extra = {}) {
  const textValue = extra.full_text || `Speech entry number ${i}. In a hole in the ground there lived a hobbit.`;
  return {
    id: `h${i}`, type: 'tts', created_at: at(0, 9, 0), model: 'kokoro', voice: 'af_heart', speed: 1, format: 'mp3',
    text_preview: textValue.slice(0, 180), full_text: textValue, input_filename: null, output_path: null, output_bytes: 1000,
    streamed: false, meta: { reference_voice: null, instructions: null, effects: null }, ...extra,
  };
}

export function transcript(i, extra = {}) {
  const textValue = extra.full_text || `Okay so the plan for meeting ${i} is simple.`;
  return {
    id: `s${i}`, type: 'stt', created_at: at(0, 8, 0), model: 'Systran/faster-whisper-base', voice: null, speed: null, format: null,
    text_preview: textValue.slice(0, 180), full_text: textValue, input_filename: `meeting-${i}.m4a`, output_path: null,
    output_bytes: null, streamed: false, meta: {}, ...extra,
  };
}

/** A fake of GET/DELETE /api/history over a list of entries (newest first). */
export function historyServer(entries = []) {
  const state = { entries: entries.slice() };
  const matching = (type, search) => state.entries.filter((e) => (!type || e.type === type)
    && (!search || `${e.full_text || ''} ${e.input_filename || ''}`.toLowerCase().includes(search.toLowerCase())));
  const routes = {
    'GET /api/history': (req) => {
      const type = req.query.get('type') || '';
      const search = (req.query.get('search') || '').trim();
      const limit = Number(req.query.get('limit') || 50);
      const offset = Number(req.query.get('offset') || 0);
      const all = matching(type, search);
      return json(200, { items: all.slice(offset, offset + limit), total: all.length, limit, offset });
    },
    'DELETE /api/history/:id': (req) => {
      const index = state.entries.findIndex((e) => e.id === req.params.id);
      if (index < 0) return json(404, { error: { message: 'History entry not found', code: 'http_error' } });
      state.entries.splice(index, 1);
      return json(204);
    },
    'DELETE /api/history': () => {
      const deleted = state.entries.length;
      state.entries = [];
      return json(200, { deleted });
    },
  };
  return { state, routes };
}

const SPEAK_STUB = '<header class="view-head"><h1 id="sp-title" class="view-head__title" data-view-heading tabindex="-1">Speak</h1></header>';
const STUDIO_STUB = '<header class="view-head"><h1 id="st-title" class="view-head__title" data-view-heading tabindex="-1">Studio</h1></header>';

function stubView(id, title, partial) {
  return async () => ({ default: { id, title, partial, init: () => ({ show() {}, hide() {} }) } });
}

/**
 * Boot the fake browser on #history with the history view mounted through the real view host and router.
 * @param {{server?: object, routes?: object, path?: string, speak?: boolean}} [options] speak: mount the real
 *   Speak view (its partial, its routes and a loaded Kokoro) instead of a stub; other options go to
 *   installBrowserEnv
 */
export async function historyEnv({ server = historyServer(), routes = {}, path = 'history', speak = false, ...options } = {}) {
  const env = await shellEnv({
    routes: {
      // Speak's routes first: the server's and the test's own routes replace any of them.
      ...(speak ? speakRoutes() : { 'GET /static/partials/speak.html': () => text(200, SPEAK_STUB, { 'content-type': 'text/html' }) }),
      'GET /static/partials/studio.html': () => text(200, STUDIO_STUB, { 'content-type': 'text/html' }),
      ...server.routes,
      ...routes,
    },
    ...options,
  });
  const viewHost = await load('shell/view-host.js');
  const router = await load('core/router.js');
  const store = await load('core/store.js');
  const prefs = await load('core/prefs.js');
  const auth = await load('core/auth.js');
  prefs.load();
  auth.init();
  if (speak) {
    const models = modelsPayload();
    (await load('views/speak/index.js')).__resetForTests();
    store.app.set({
      connection: { ...store.app.get().connection, state: 'online' },
      models: { list: models.models, defaults: { tts: models.default_tts_model || '', stt: models.default_stt_model || '' }, fetchedAt: Date.now(), error: '' },
    });
  }
  viewHost.initViewHost({
    registry: {
      history: () => load('views/history/index.js'),
      speak: speak ? () => load('views/speak/index.js') : stubView('speak', 'Speak', '/static/partials/speak.html'),
      studio: stubView('studio', 'Studio', '/static/partials/studio.html'),
    },
  });
  env.setHash(`#${path}`);
  await router.start(viewHost.show);
  await settle(env);
  await settle(env);
  const section = document.getElementById('view-history');
  const ctx = {
    env, router, viewHost, store, prefs, app: store.app, section, server,
    controller: viewHost.controllerFor('history'),
    $: (id) => section.querySelector(`#${id}`),
    $$: (sel) => [...section.querySelectorAll(sel)],
    calls: (pattern) => env.fetch.callsTo(pattern),
    settle: (ms = 0) => settle(env, ms),
    rows: () => [...section.querySelectorAll('.hi-entry')],
    async go(to) {
      await router.navigate(to);
      await settle(env);
      await settle(env);
    },
    async click(el) {
      if (!el) throw new Error('click(): no element');
      env.click(el);
      await settle(env);
      await settle(env);
    },
    async menu(button, label) {
      env.click(button);
      await settle(env);
      const menu = document.getElementById(button.getAttribute('aria-controls'));
      if (!menu || menu.hidden) throw new Error('menu did not open');
      const item = [...menu.querySelectorAll('[role="menuitem"]')].find((el) => el.textContent.trim() === label);
      if (!item) throw new Error(`no menu item ${label}: ${[...menu.querySelectorAll('[role="menuitem"]')].map((el) => el.textContent.trim()).join('|')}`);
      env.click(item);
      await settle(env);
      await settle(env);
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
    toasts() {
      return [...document.querySelectorAll('.toast')].map((x) => ({ kind: x.className.replace(/.*toast--(\w+).*/, '$1'), text: x.querySelector('.toast__msg').textContent }));
    },
  };
  return ctx;
}
