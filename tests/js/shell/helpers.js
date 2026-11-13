// Shared setup for the shell and component tests: the real shell (index.html) in the fake browser, the
// server routes the shell calls at boot, and a reset of every stateful shell and component module.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, loadHtml, resetAll as resetEverything, STATIC_JS, REPO_ROOT, flushMicrotasks } from '../harness/env.js';
import { json, text } from '../harness/fakes.js';

export const SHELL = 'src/static/index.html';


export const MODELS = {
  models: [
    { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' },
    { id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'downloaded', device: 'cpu' },
  ],
  default_tts_model: 'kokoro',
  default_stt_model: 'Systran/faster-whisper-base',
};

/** Serve the real partials from disk. */
export function partialRoute() {
  return (req) => {
    const name = (req.params && req.params.name) || req.path.split('/').pop();
    return text(200, readFileSync(resolve(REPO_ROOT, 'src/static/partials', name), 'utf8'), { 'content-type': 'text/html; charset=utf-8' });
  };
}

/** A Reader document as GET /v1/reader/documents/{id} answers it (empty: no blocks, chapters or progress). */
export function readerDocument(id, extra = {}) {
  return { id, title: 'Doc', author: '', blocks: [], chapters: [], progress: null, ...extra };
}

/**
 * The Reader's library and document routes, for a shell test that routes to `#reader/<id>`: the
 * real Reader treats a 404 for a document as a dead deep link (RE-60) and goes back to `#reader`, so without
 * them such a test would test that redirect. Opt-in (pass them as `routes`), not part of serverRoutes(): every
 * shell boot lands on the Reader, and a 2xx from its library request would make the connection `online` in the
 * tests that keep /health or /api/models unanswered, refused or offline.
 * @param {(id: string) => object} [doc] the document for an id
 */
export function readerRoutes(doc = (id) => readerDocument(id)) {
  return {
    'GET /v1/reader/documents': () => json(200, { documents: [] }),
    'GET /v1/reader/documents/:id': (req) => json(200, doc(req.params.id)),
  };
}

/** The routes the shell calls at boot, plus `extra`. The fake fetch takes the first matching pattern, so the
 *  extra routes come first and win over the generic ones. */
export function serverRoutes(extra = {}) {
  const base = {
    'GET /health': () => json(200, { status: 'ok', version: '0.8.0', models_loaded: 1, revision: 'abc1234' }),
    'GET /api/models': () => json(200, MODELS),
    'GET /static/partials/:name': partialRoute(),
  };
  const out = { ...extra };
  for (const [pattern, handler] of Object.entries(base)) if (!(pattern in out)) out[pattern] = handler;
  return out;
}

/** Import a module from src/static/js (the same instance the shell uses). */
export function load(rel) {
  return import(pathToFileURL(resolve(STATIC_JS, rel)).href);
}

/** Reset every shell and component module, then the core (store last): the harness's resetAll(). */
export async function resetAll() {
  await resetEverything();
}

/**
 * The fake browser with the real shell markup.
 * @param {object} [options] installBrowserEnv options; `routes` are added to serverRoutes()
 */
export async function shellEnv({ routes = {}, document: doc, ...options } = {}) {
  const env = installBrowserEnv({ document: doc || loadHtml(SHELL), routes: serverRoutes(routes), ...options });
  await resetAll();
  return env;
}

/** Let promises, timers due now and animation frames settle. */
export async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

/** A keydown at an element (the harness also runs the browser's default action unless prevented). */
export function press(env, key, init = {}, target) {
  return env.key(key, init, target);
}

export { json, text };
