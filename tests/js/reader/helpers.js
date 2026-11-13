// Shared setup for the Reader tests: the real shell and reader partial in the fake browser, the Reader
// controller wired to the fake engine (the real engine is replaced through the controller's deps), and a fake
// Reader API holding documents in memory.
import { shellEnv, load, settle, json, text } from '../shell/helpers.js';
import { createFakeEngine } from './fake-engine.js';

export { load, settle, json, text };

/** A sentence list for a paragraph: [start, end) code-point offsets of each sentence. */
function sentencesOf(textValue) {
  const out = [];
  const points = Array.from(textValue);
  let start = 0;
  for (let i = 0; i < points.length; i += 1) {
    if ('.!?'.includes(points[i]) && (i + 1 === points.length || points[i + 1] === ' ')) {
      out.push([start, i + 1]);
      start = i + 2;
    }
  }
  if (start < points.length) out.push([start, points.length]);
  return out;
}

/**
 * A full document as GET /v1/reader/documents/{id} returns it (blocks and chapters), plus `summary()`, the
 * library row (no blocks).
 */
export function makeDoc({
  id = 'alice', title = "Alice's Adventures in Wonderland", author = 'Lewis Carroll', sourceType = 'text',
  chapters = 3, perChapter = 5, progress = null, chapterTitles = null, paragraph = null,
  updatedAt = '2026-10-20T10:00:00Z',
} = {}) {
  const blocks = [];
  const list = [];
  for (let c = 0; c < chapters; c += 1) {
    const first = blocks.length;
    const name = chapterTitles ? chapterTitles[c] : `Chapter ${c + 1}`;
    blocks.push({ index: blocks.length, chapter_index: c, kind: 'heading', text: name, sentences: [[0, Array.from(name).length]] });
    for (let p = 1; p < perChapter; p += 1) {
      const body = paragraph ? paragraph(c, p) : `Paragraph ${p} of chapter ${c + 1}. It has a second sentence. And a third one.`;
      blocks.push({ index: blocks.length, chapter_index: c, kind: 'paragraph', text: body, sentences: sentencesOf(body) });
    }
    list.push({ index: c, title: name, first_block: first, block_count: blocks.length - first });
  }
  const charCount = blocks.reduce((n, b) => n + Array.from(b.text).length, 0);
  const doc = {
    id, title, author, source_type: sourceType, created_at: updatedAt, updated_at: updatedAt,
    chapter_count: list.length, block_count: blocks.length, char_count: charCount, progress, chapters: list, blocks,
  };
  doc.summary = () => {
    const { blocks: _b, chapters: _c, summary: _s, ...rest } = doc;
    return { ...rest };
  };
  return doc;
}

export function progressAt(blockIndex, { model = 'kokoro', voice = 'af_heart', speed = 1.25, updatedAt = '2026-10-21T09:00:00Z' } = {}) {
  return { block_index: blockIndex, offset_ms: 0, model, voice, speed, updated_at: updatedAt };
}

/**
 * The fake Reader API over `docs` (an array of makeDoc() results, mutated in place by PUT and DELETE).
 * @returns {object} createFetch routes
 */
export function readerRoutes(docs) {
  const find = (id) => docs.find((d) => d.id === id);
  return {
    'GET /v1/reader/documents': () => json(200, { documents: docs.map((d) => d.summary()) }),
    'POST /v1/reader/documents': (req) => {
      const body = req.json || {};
      const doc = makeDoc({ id: `new-${docs.length + 1}`, title: body.title || 'Pasted text', author: '' });
      docs.unshift(doc);
      return json(201, doc.summary());
    },
    'GET /v1/reader/documents/:id': (req) => {
      const doc = find(req.params.id);
      if (!doc) return json(404, { detail: 'Document not found' });
      const { summary: _s, ...full } = doc;
      return json(200, full);
    },
    'PUT /v1/reader/documents/:id/progress': (req) => {
      const doc = find(req.params.id);
      if (!doc) return json(404, { detail: 'Document not found' });
      doc.progress = { ...req.json, updated_at: '2026-10-22T12:00:00Z' };
      return json(200, { progress: doc.progress });
    },
    'DELETE /v1/reader/documents/:id': (req) => {
      const index = docs.findIndex((d) => d.id === req.params.id);
      if (index < 0) return json(404, { detail: 'Document not found' });
      docs.splice(index, 1);
      return text(204, '');
    },
  };
}

/**
 * Boot the shell on `hash` with the Reader controller using the fake engine.
 * `mountPlayer(slot, engine, options)` replaces the recording stub (which mounts nothing); every call is still
 * recorded in `mounted`.
 * @param {{docs?: object[], routes?: object, hash?: string, engine?: object, loadEngine?: Function, mountPlayer?: Function, viewport?: object}} [options]
 */
export async function readerEnv({ docs = null, routes = {}, hash = '#reader', engine = createFakeEngine(), loadEngine, mountPlayer, ...options } = {}) {
  const library = docs || [
    makeDoc({ progress: progressAt(12) }),
    makeDoc({ id: 'note', title: 'Notes for Tuesday', author: '', chapters: 1, perChapter: 3, updatedAt: '2026-10-21T08:00:00Z' }),
  ];
  const env = await shellEnv({
    url: `http://127.0.0.1:8100/web${hash}`,
    routes: { ...readerRoutes(library), ...routes },
    ...options,
  });
  const mounted = [];
  const shell = await load('shell/index.js');
  const viewHost = await load('shell/view-host.js');
  const mod = await load('views/reader/index.js');
  mod.__resetForTests();
  const deps = {
    loadEngine: loadEngine || (async () => engine),
    mountPlayer: async (slot, eng, opts) => {
      mounted.push({ slot, engine: eng, options: opts });
      return mountPlayer ? mountPlayer(slot, eng, opts) : undefined;
    },
  };
  const registry = {
    ...viewHost.VIEW_MODULES,
    reader: async () => ({ default: { ...mod.default, init: (root) => mod.createReaderController(root, deps) } }),
  };
  await shell.boot({ registry });
  await settle(env, 5);
  const ctl = viewHost.controllerFor('reader');
  const router = await load('core/router.js');
  const store = await load('core/store.js');
  return { env, engine, ctl, mounted, docs: library, router, store, viewHost, mod, shell };
}

/** document.getElementById shortcut. */
export const $ = (id) => document.getElementById(id);

/** The visible text of an element, whitespace collapsed. */
export function textOf(el) {
  return String(el ? el.textContent : '').replace(/\s+/g, ' ').trim();
}

/** Navigate and let the view settle (fetches, frames, timers due). */
export async function go(ctx, path, ms = 0) {
  await ctx.router.navigate(path);
  await settle(ctx.env, ms);
}
