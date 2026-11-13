// Shared setup for the Reader player tests: a fake Reader server (render, parts, progress, model status),
// documents with chapters and sentence marks, and the engine on a clean fake browser.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { installBrowserEnv, importFresh, flushMicrotasks, loadHtml, REPO_ROOT, STATIC_JS } from '../harness/env.js';
import { json, text, delay, networkError } from '../harness/fakes.js';
import { pathToFileURL } from 'node:url';

export { json, text, delay, networkError, flushMicrotasks };

export const MODELS = [
  { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' },
  { id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'downloaded', device: 'cpu' },
];

/** The shell bits the engine and player touch: the audio host, the toast and announcement regions, main. */
export const SHELL_BITS = [
  '<main id="main" tabindex="-1"><section class="view" id="view-reader" data-view="reader"><div id="rd-reading"><div class="rd-player-slot" id="rd-player-slot" data-slot="reader-player"></div></div></section></main>',
  '<div class="toasts" id="app-toasts"><div id="app-toasts-polite" role="status" aria-live="polite"></div><div id="app-toasts-alert" role="alert" aria-live="assertive"></div></div>',
  '<div id="app-sr-status" class="visually-hidden" role="status" aria-live="polite"></div>',
  '<div id="app-sr-alert" class="visually-hidden" role="alert"></div>',
  '<div id="app-audio-host" hidden><audio id="app-audio" preload="auto"></audio></div>',
].join('');

/** Import a module from src/static/js without resetting anything (the instance the code under test uses). */
export function load(rel) {
  return import(pathToFileURL(resolve(STATIC_JS, rel)).href);
}

/**
 * A document: `chapters` is a list of [title, paragraphs]. Each chapter starts with a heading block; each
 * paragraph has two sentences ("Chapter c, paragraph p, first sentence. And the second one.").
 */
export function makeDoc({ id = 'doc1', title = "Alice's Adventures in Wonderland", author = 'Lewis Carroll', chapters = [['Down the Rabbit-Hole', 4], ['The Pool of Tears', 3], ['A Caucus-Race', 3]], progress = null, headings = true } = {}) {
  const blocks = [];
  const outline = [];
  chapters.forEach(([chapterTitle, paragraphs], c) => {
    const first = blocks.length;
    if (headings) blocks.push({ index: blocks.length, chapter_index: c, kind: 'heading', text: chapterTitle, sentences: [[0, chapterTitle.length]] });
    for (let p = 0; p < paragraphs; p += 1) {
      const one = `Chapter ${c + 1}, paragraph ${p + 1}, first sentence.`;
      const two = 'And the second one, which is a little longer than the first.';
      const textValue = `${one} ${two}`;
      blocks.push({ index: blocks.length, chapter_index: c, kind: 'paragraph', text: textValue, sentences: [[0, one.length], [one.length + 1, textValue.length]] });
    }
    outline.push({ index: c, title: chapterTitle, first_block: first, block_count: blocks.length - first });
  });
  return { id, title, author, source_type: 'text', block_count: blocks.length, chapter_count: outline.length, char_count: 0, blocks, chapters: outline, progress };
}

/** The duration the fake server gives a block's part: 2 s per sentence. */
export const SENTENCE_MS = 2000;

/** One part per block, ready, with a sentence mark per sentence. */
export function partFor(doc, block, overrides = {}) {
  const b = doc.blocks[block];
  const marks = (b.sentences || [[0, 1]]).map((_, i) => ({ block, index: i, start_ms: i * SENTENCE_MS }));
  return {
    id: `p${block}`, first_block: block, last_block: block, status: 'ready',
    duration_ms: marks.length * SENTENCE_MS, url: `/v1/reader/media/p${block}.mp3?exp=1&sig=s1`, error: null, sentences: marks,
    ...overrides,
  };
}

/**
 * A fake Reader server. By default every block's part is ready at once and a window holds 4 parts. Tweak
 * with `pending` (block -> polls before ready), `errors` (block -> message), `override` (block -> fields).
 */
export class ReaderServer {
  constructor(doc, { window = 4 } = {}) {
    this.doc = doc;
    this.window = window;
    this.pending = new Map();
    this.errors = new Map();
    this.override = new Map();
    this.saved = [];
    this.renders = [];
    this.polls = [];
    this.status = 'loaded';
    this.sig = 's1';
    this.renderHandler = null;
    this.partsHandler = null;
  }

  partsFrom(from) {
    const out = [];
    for (let b = from; b < Math.min(this.doc.blocks.length, from + this.window); b += 1) {
      const waiting = this.pending.get(b) || 0;
      if (this.errors.has(b)) out.push(partFor(this.doc, b, { status: 'error', url: null, duration_ms: 0, error: this.errors.get(b) }));
      else if (waiting > 0) out.push(partFor(this.doc, b, { status: 'rendering', url: null, duration_ms: 0 }));
      else out.push(partFor(this.doc, b, { url: `/v1/reader/media/p${b}.mp3?exp=1&sig=${this.sig}`, ...(this.override.get(b) || {}) }));
    }
    return out;
  }

  tickPending() {
    for (const [b, n] of this.pending) this.pending.set(b, Math.max(0, n - 1));
  }

  routes(extra = {}) {
    const base = {
      'POST /v1/reader/documents/:id/render': (req) => {
        this.renders.push(req.json);
        if (this.renderHandler) return this.renderHandler(req);
        return json(200, { parts: this.partsFrom(req.json.from_block) });
      },
      'GET /v1/reader/documents/:id/parts': (req) => {
        const from = Number(req.query.get('from_block'));
        this.polls.push({ from, model: req.query.get('model'), voice: req.query.get('voice') });
        if (this.partsHandler) return this.partsHandler(req);
        this.tickPending();
        return json(200, { parts: this.partsFrom(from) });
      },
      'PUT /v1/reader/documents/:id/progress': (req) => {
        this.saved.push({ ...req.json, keepalive: req.keepalive });
        return json(200, { progress: { ...req.json, updated_at: '2026-10-23T10:00:00Z' } });
      },
      'GET /api/models/:id/status': () => json(200, { state: this.status }),
      'POST /api/models/:id/load': () => { this.status = 'loaded'; return json(200, { status: 'loaded' }); },
      'GET /api/models': () => json(200, { models: MODELS, default_tts_model: 'kokoro', default_stt_model: '' }),
      'GET /api/voices/identities': () => json(200, { voices: [{ id: 'n1', name: 'Narrator', realizations: [{ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)' }] }] }),
      'GET /v1/audio/voices': () => json(200, { voices: [{ id: 'af_heart' }, { id: 'af_bella' }, { id: 'am_adam' }, { id: 'bf_emma' }] }),
      'GET /static/partials/:name': (req) => text(200, readFileSync(resolve(REPO_ROOT, 'src/static/partials', req.params.name), 'utf8'), { 'content-type': 'text/html; charset=utf-8' }),
    };
    return { ...extra, ...Object.fromEntries(Object.entries(base).filter(([k]) => !(k in extra))) };
  }
}

/**
 * The fake browser plus a fresh engine. Models are known and Kokoro is loaded (the ensureReady fast path), the
 * Reader voice is af_heart unless `seed` says otherwise.
 */
export async function setup({ doc = makeDoc(), server = null, seed = null, routes = {}, playPolicy = 'resolve', prefs = null, models = MODELS, modelsKnown = true, html = '', viewport, media, shell = false } = {}) {
  const srv = server || new ReaderServer(doc);
  const storageSeed = seed || { 'open-speech-reader-voice': JSON.stringify({ model: 'kokoro', voice: 'af_heart' }) };
  const base = { routes: srv.routes(routes), seed: storageSeed, playPolicy, ...(viewport ? { viewport } : {}), ...(media ? { media } : {}) };
  // shell: the real index.html (the voice picker, dialogs and live regions), with the Reader's slot in its section.
  const env = shell
    ? installBrowserEnv({ ...base, document: loadHtml('src/static/index.html') })
    : installBrowserEnv({ ...base, html: SHELL_BITS + html });
  if (shell) {
    const section = document.getElementById('view-reader');
    section.hidden = false;
    env.mount('<div id="rd-reading"><div class="rd-player-slot" id="rd-player-slot" data-slot="reader-player"></div></div>', section);
  }
  const mod = await importFresh('views/reader/engine.js');
  const { app } = await load('core/store.js');
  const prefsMod = await load('core/prefs.js');
  prefsMod.load();
  if (prefs) prefsMod.set(prefs);
  if (modelsKnown) app.set({ models: { list: models, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
  const audio = await load('core/audio.js');
  const engine = mod.getEngine();
  const el = (id) => document.getElementById(id);
  return { env, engine, app, mod, server: srv, doc, audio, prefs: prefsMod, a: () => el('rd-audio-a'), b: () => el('rd-audio-b') };
}

/** Let promises, due timers and frames settle. */
export async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

/** The element playing now (the engine's active element). */
export function playingEl() {
  return ['rd-audio-a', 'rd-audio-b'].map((id) => document.getElementById(id)).find((el) => el && !el.paused && el.dataset.part) || null;
}

/** Load the document and start playback; resolves once the first part plays. */
export async function startPlaying(ctx, { block = null } = {}) {
  ctx.engine.load(ctx.doc);
  if (block === null) ctx.engine.play();
  else ctx.engine.playFrom(block);
  await settle(ctx.env);
  const el = playingEl();
  if (el) loadMeta(ctx, el);
  await settle(ctx.env);
  return playingEl();
}

/** Fire loadedmetadata on an element holding a part, with that part's duration. */
export function loadMeta(ctx, el) {
  if (!el || el.duration > 0) return;
  const block = Number(String(el.dataset.part || '').replace(/^p/, ''));
  const part = Number.isFinite(block) && ctx.doc.blocks[block] ? (ctx.server ? ctx.server.partsFrom(block)[0] : partFor(ctx.doc, block)) : null;
  el.__loadMetadata(part && part.duration_ms > 0 ? part.duration_ms / 1000 : 4);
}

/** Collect an engine event into an array. */
export function collect(engine, type) {
  const out = [];
  engine.on(type, (detail) => out.push(detail));
  return out;
}

/** Mount the player in the slot; resolves with the controller once its partial is in. */
export async function mountPlayer(ctx, options = {}) {
  const { mountPlayer: mount } = await load('views/reader/player.js');
  const player = mount(document.getElementById('rd-player-slot'), ctx.engine, options);
  await player.ready;
  return player;
}

/** Show the reading view of the document (the route the Reader view sets), so the player's shortcuts and sheets apply. */
export async function showReading(ctx, docId = ctx.doc.id) {
  const router = await load('core/router.js');
  await router.navigate(`reader/${encodeURIComponent(docId)}`);
  await settle(ctx.env);
  return router;
}

/** An element by id. */
export const $ = (id) => document.getElementById(id);

/**
 * Record every DOM write inside `root`: the attribute methods, the hidden/disabled/value/textContent setters
 * (a same-value write counts: in a browser it still queues a mutation record) and style.setProperty. Each entry
 * is '<what> <id or tag> <detail>'. restore() puts everything back.
 */
export function spyWrites(root) {
  const list = [];
  const restores = [];
  const inside = (node) => node === root || (!!root.contains && root.contains(node));
  const nameOf = (node) => node.id || node.localName;
  const owner = (obj, prop) => {
    for (let p = obj; p; p = Object.getPrototypeOf(p)) if (Object.prototype.hasOwnProperty.call(p, prop)) return p;
    return null;
  };
  for (const method of ['setAttribute', 'removeAttribute', 'toggleAttribute']) {
    const p = owner(root, method);
    const orig = p[method];
    p[method] = function spy(...args) {
      if (inside(this)) list.push(`${method} ${nameOf(this)} ${args[0]}`);
      return orig.apply(this, args);
    };
    restores.push(() => { p[method] = orig; });
  }
  for (const prop of ['hidden', 'disabled', 'value', 'textContent']) {
    const p = owner(root, prop);
    const desc = Object.getOwnPropertyDescriptor(p, prop);
    Object.defineProperty(p, prop, {
      ...desc,
      set(v) {
        if (inside(this)) list.push(`${prop} ${nameOf(this)}`);
        desc.set.call(this, v);
      },
    });
    restores.push(() => Object.defineProperty(p, prop, desc));
  }
  for (const node of [root, ...root.querySelectorAll('*')]) {
    const real = node.style;
    if (!real) continue;
    node.style = new Proxy(real, {
      get(target, prop) {
        if (prop === 'setProperty') {
          return (key, value) => {
            list.push(`style ${nameOf(node)} ${key}`);
            return real.setProperty(key, value);
          };
        }
        return real[prop];
      },
      set(target, prop, value) {
        real[prop] = value;
        return true;
      },
    });
    restores.push(() => { node.style = real; });
  }
  return { list, restore() { for (const undo of restores.splice(0).reverse()) undo(); } };
}
