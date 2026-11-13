// Shared setup for the Settings tests: the real shell markup with the real partial mounted in
// #view-settings, or the whole shell booted (for the auth:changed hand-off, AU-09).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, loadHtml, importFresh, flushMicrotasks, STATIC_JS, REPO_ROOT } from '../harness/env.js';
import { json, text, delay, networkError } from '../harness/fakes.js';

export { json, text, delay, networkError };

export const PARTIAL = readFileSync(resolve(REPO_ROOT, 'src/static/partials/settings.html'), 'utf8');
export const KEY_STORAGE = 'open-speech-api-key';
export const ROUTE = Object.freeze({ view: 'settings', sub: '', id: '', path: 'settings', shot: '' });

export const INVENTORY = {
  models: [
    { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' },
    { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'loaded', device: 'cpu' },
    { id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'downloaded' },
  ],
  default_tts_model: 'kokoro',
  default_stt_model: 'Systran/faster-whisper-base',
};

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

/** Serve the real partials from disk. */
export function partialRoute() {
  return (req) => text(200, readFileSync(resolve(REPO_ROOT, 'src/static/partials', req.params.name), 'utf8'), { 'content-type': 'text/html; charset=utf-8' });
}

/** A server whose /api/models wants `key` (null: no key needed). */
export function keyedModels(key) {
  return (req) => {
    if (key === null) return json(200, INVENTORY);
    const sent = req.headers.authorization || '';
    if (sent === `Bearer ${key}`) return json(200, INVENTORY);
    return json(401, { error: { message: 'Invalid or missing API key', code: 'unauthorized' } });
  };
}

/**
 * The fake browser with the shell markup and the Settings partial, the view initialised and shown.
 * @param {object} [options]
 * @param {object} [options.routes]
 * @param {object} [options.seed] localStorage
 * @param {object} [options.env] more installBrowserEnv options
 * @param {boolean} [options.show]
 * @param {string} [options.shot]
 */
export async function mountSettings({ routes = {}, seed = {}, env: envOptions = {}, show = true, shot = '' } = {}) {
  const allRoutes = {
    'GET /api/models': keyedModels(null),
    'GET /health': () => json(200, { status: 'ok', version: '0.8.0', models_loaded: 1, revision: 'abc1234' }),
    ...routes,
  };
  const env = installBrowserEnv({ document: loadHtml('src/static/index.html'), routes: { ...routes, ...allRoutes }, seed, ...envOptions });
  const mod = await importFresh('views/settings/index.js');
  const { app } = await load('core/store.js');
  const prefs = await load('core/prefs.js');
  const auth = await load('core/auth.js');
  const theme = await load('core/theme.js');
  const dom = await load('core/dom.js');
  prefs.load();
  theme.init();
  auth.init();
  const route = { ...ROUTE, shot };
  app.set({
    connection: { ...app.get().connection, state: 'online', version: '0.8.0', revision: 'abc1234', checkedAt: Date.now() },
    models: { list: INVENTORY.models, defaults: { tts: 'kokoro', stt: 'Systran/faster-whisper-base' }, fetchedAt: Date.now(), error: '' },
    route,
  });
  const root = env.document.getElementById('view-settings');
  dom.render(root, dom.raw(PARTIAL));
  root.hidden = false;
  const ctl = mod.default.init(root);
  if (show) ctl.show(route, { focus: 'none', scrolled: false, reason: 'route' });
  await settle(env);
  const $ = (id) => env.document.getElementById(id);
  return { env, mod, ctl, app, prefs, auth, theme, root, $, doc: env.document, route };
}

/** A stand-in for a view the test does not exercise. */
function fakeView(id) {
  return () => Promise.resolve({ default: { id, title: id, partial: `/static/partials/${id}.html`, init: () => ({ show() {}, hide() {} }) } });
}

/**
 * Boot the whole shell (shell/index.js boot()) at #settings with the real Settings view; every other view is
 * a stand-in. Used for the auth:changed hand-off (AU-09).
 */
export async function bootShell({ routes = {}, seed = {}, env: envOptions = {} } = {}) {
  const allRoutes = {
    'GET /health': () => json(200, { status: 'ok', version: '0.8.0', models_loaded: 1, revision: 'abc1234' }),
    'GET /static/partials/:name': partialRoute(),
    ...routes,
  };
  const env = installBrowserEnv({
    document: loadHtml('src/static/index.html'), url: 'http://127.0.0.1:8100/web#settings',
    routes: { ...routes, ...allRoutes }, seed, ...envOptions,
  });
  const shell = await importFresh('shell/index.js');
  const settingsMod = await load('views/settings/index.js');
  if (typeof settingsMod.__resetForTests === 'function') settingsMod.__resetForTests();
  const registry = {
    settings: () => Promise.resolve(settingsMod),
    reader: fakeView('reader'), speak: fakeView('speak'), transcribe: fakeView('transcribe'), voices: fakeView('voices'),
    studio: fakeView('studio'), history: fakeView('history'), models: fakeView('models'),
  };
  await shell.boot({ registry });
  await settle(env, 5);
  const { app } = await load('core/store.js');
  const auth = await load('core/auth.js');
  const $ = (id) => env.document.getElementById(id);
  return { env, shell, app, auth, $, doc: env.document, root: $('view-settings') };
}

/** The visible text of an element, whitespace collapsed. */
export function textOf(el) {
  return String(el ? el.textContent : '').replace(/\s+/g, ' ').trim();
}

/** Toasts on screen: [{kind, text}]. */
export function toasts(t) {
  return t.doc.querySelectorAll('.toast').map((el) => ({ kind: (el.className.match(/toast--(\w+)/) || [])[1], text: textOf(el.querySelector('.toast__msg')) }));
}

/** The text of the polite announcement region (after its frame). */
export function announced(t) {
  t.env.raf.flush();
  return t.$('app-sr-status').textContent;
}

/** Calls to a route since a mark: [{method, path, headers}]. */
export function callsSince(env, mark) {
  return env.fetch.calls.slice(mark).map((c) => ({ method: c.method, path: c.path, headers: c.headers, url: c.url }));
}
