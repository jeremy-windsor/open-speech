// Home: the greeting, what "Pick up where you left off" and "Recent" show (loaded, empty, failed), reloads,
// and where its controls go. The real shell markup and partial, a fake server; no AI route is ever asked.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, loadHtml, importFresh, flushMicrotasks, STATIC_JS, REPO_ROOT } from '../harness/env.js';
import { json } from '../harness/fakes.js';

const PARTIAL = readFileSync(resolve(REPO_ROOT, 'src/static/partials/home.html'), 'utf8');
const load = (rel) => import(pathToFileURL(resolve(STATIC_JS, rel)).href);
const NOW = new Date(2026, 10, 16, 9, 30).getTime();
const ago = (minutes) => new Date(NOW - minutes * 60000).toISOString();

const doc = (id, title, extra = {}) => ({ id, title, chapter_count: 12, block_count: 200, created_at: ago(5000), updated_at: ago(5000), progress: null, ...extra });
const speech = (n) => ({ id: `h${n}`, type: 'tts', created_at: ago(n * 10), voice: 'af_heart', text_preview: `Take number ${n}` });
const transcript = (n) => ({ id: `h${n}`, type: 'stt', created_at: ago(n * 10), input_filename: `meeting-${n}.m4a`, text_preview: '' });

let t;
afterEach(() => t?.env?.restore());

async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

async function mount({ documents = [], items = [], routes = {}, show = true } = {}) {
  const hits = { documents: 0, history: 0, limit: '' };
  const server = { documents, items };
  const env = installBrowserEnv({
    document: loadHtml('src/static/index.html'),
    now: NOW,
    routes: {
      ...routes,
      'GET /v1/reader/documents': () => { hits.documents += 1; return json(200, { documents: server.documents }); },
      'GET /api/history': (req) => { hits.history += 1; hits.limit = req.query.get('limit'); return json(200, { items: server.items }); },
      ...routes,
    },
  });
  const mod = await importFresh('views/home/index.js');
  const store = await load('core/store.js');
  const router = await load('core/router.js');
  const dom = await load('core/dom.js');
  const root = env.document.getElementById('view-home');
  dom.render(root, dom.raw(PARTIAL));
  root.hidden = false;
  const ctl = mod.createHomeView(root, { now: () => new Date(Date.now()) });
  const route = { view: 'home', sub: '', id: '', path: 'home', shot: '' };
  if (show) ctl.show(route, { focus: 'none' });
  await settle(env);
  const $ = (id) => env.document.getElementById(id);
  return { env, mod, ctl, store, router, root, $, hits, server, route };
}

describe('home: pure models', () => {
  it('greeting follows the local hour', async () => {
    t = await mount({ show: false });
    const at = (h) => t.mod.greeting(new Date(2026, 10, 16, h, 0));
    assert.deepEqual([4, 5, 11, 12, 17, 18, 23].map(at),
      ['Good evening', 'Good morning', 'Good morning', 'Good afternoon', 'Good afternoon', 'Good evening', 'Good evening']);
  });

  it('continueModel: the unfinished document saved last; else the newest to start; null for an empty library', async () => {
    t = await mount({ show: false });
    const { continueModel } = t.mod;
    assert.equal(continueModel([]), null);
    assert.equal(continueModel(undefined), null);
    const docs = [
      doc('a', 'Alice', { progress: { block_index: 50, updated_at: ago(600) } }),
      doc('b', 'Bleak House', { progress: { block_index: 20, updated_at: ago(30) } }),
      doc('c', 'Finished', { progress: { block_index: 200, updated_at: ago(1) } }),
      doc('d', 'Dracula'),
    ];
    const picked = continueModel(docs);
    assert.deepEqual({ id: picked.id, started: picked.started, pct: picked.pct }, { id: 'b', started: true, pct: 10 });
    assert.match(picked.meta, /^10% listened · /);
    const fresh = continueModel([doc('x', 'Older', { updated_at: ago(900) }), doc('y', '', { updated_at: ago(5) })]);
    assert.deepEqual(fresh, { id: 'y', title: 'Untitled', started: false, pct: null, meta: '12 chapters · Not started' });
    const done = continueModel([doc('c', 'Finished', { progress: { block_index: 200, updated_at: ago(1) } })]);
    assert.equal(done.started, false, 'a finished book is offered to start again');
  });

  it('recentModel: at most three rows, speech with its voice and a transcript with its file name', async () => {
    t = await mount({ show: false });
    const rows = t.mod.recentModel([speech(1), transcript(2), speech(3), speech(4)], NOW);
    assert.equal(rows.length, 3);
    assert.deepEqual(rows[0], { id: 'h1', type: 'tts', text: 'Take number 1', meta: 'Speech · af_heart · 10 min ago' });
    assert.deepEqual(rows[1], { id: 'h2', type: 'stt', text: 'meeting-2.m4a', meta: 'Transcript · meeting-2.m4a · 20 min ago' });
    assert.deepEqual(t.mod.recentModel(null), []);
  });
});

describe('home: the view', () => {
  it('shows the greeting, the document to resume with its progress, and the newest history', async () => {
    t = await mount({ documents: [doc('b', 'Bleak House', { progress: { block_index: 20, updated_at: ago(30) } })], items: [speech(1), transcript(2)] });
    assert.equal(t.$('hm-title').textContent, 'Good morning');
    assert.equal(t.hits.limit, '3');
    const card = t.root.querySelector('.hm-doc');
    assert.equal(card.getAttribute('data-id'), 'b');
    assert.match(card.textContent, /Bleak House/);
    assert.match(card.textContent, /Resume/);
    assert.equal(card.querySelector('progress').getAttribute('value'), '10');
    assert.equal(t.$('hm-continue').hasAttribute('aria-busy'), false);
    assert.deepEqual([...t.root.querySelectorAll('.hm-item__title')].map((n) => n.textContent), ['Take number 1', 'meeting-2.m4a']);
    assert.equal(t.$('hm-all').hidden, false);
    assert.deepEqual([...t.root.querySelectorAll('.hm-tile')].map((a) => a.getAttribute('href')), ['#reader', '#speak', '#transcribe', '#talk']);
  });

  it('empty states: an empty library points to the Reader, an empty history says what will appear', async () => {
    t = await mount();
    assert.match(t.$('hm-continue').textContent, /Your library is empty/);
    assert.equal(t.$('hm-continue').querySelector('a').getAttribute('href'), '#reader');
    assert.match(t.$('hm-recent').textContent, /Nothing yet/);
    assert.equal(t.$('hm-all').hidden, true);
    assert.equal(t.root.querySelector('.hm-doc'), null);
  });

  it('a list that fails says why and retries alone; the other list still shows', async () => {
    let fail = true;
    t = await mount({
      items: [speech(1)],
      routes: { 'GET /v1/reader/documents': () => (fail ? json(500, { error: { message: 'boom' } }) : json(200, { documents: [doc('d', 'Dracula')] })) },
    });
    const note = t.$('hm-continue').querySelector('[role="alert"]');
    assert.ok(note && note.textContent.trim().length > 0);
    assert.equal(t.root.querySelectorAll('.hm-item').length, 1);
    fail = false;
    const before = t.hits.history;
    t.env.click(t.$('hm-continue').querySelector('[data-action="retry-continue"]'));
    await settle(t.env);
    assert.match(t.root.querySelector('.hm-doc').textContent, /Dracula/);
    assert.match(t.root.querySelector('.hm-doc').textContent, /Start/);
    assert.equal(t.hits.history, before, 'only the failed list was asked again');
  });

  it('reloads when stale or after a change elsewhere, not on every visit', async () => {
    t = await mount({ items: [speech(1)] });
    assert.deepEqual([t.hits.documents, t.hits.history], [1, 1]);
    t.ctl.hide();
    t.ctl.show(t.route, { focus: 'none' });
    await settle(t.env);
    assert.deepEqual([t.hits.documents, t.hits.history], [1, 1], 'fresh data is kept');
    t.ctl.hide();
    t.server.items = [speech(5), speech(1)];
    t.store.emit('history:changed', {});
    await settle(t.env);
    assert.equal(t.hits.history, 1, 'hidden: it only notes the change');
    t.ctl.show(t.route, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.hits.history, 2);
    assert.equal(t.root.querySelector('.hm-item__title').textContent, 'Take number 5');
    t.store.emit('library:changed', {});
    await settle(t.env);
    assert.equal(t.hits.documents, 3, 'shown: a change reloads at once');
    await t.env.clock.tickAsync(t.mod.STALE_MS + 1000);
    t.ctl.hide();
    t.ctl.show(t.route, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.hits.documents, 4, 'older than 30 s');
  });

  it('the card opens its document in the Reader and a recent row opens History', async () => {
    t = await mount({ documents: [doc('a b', 'Alice', { progress: { block_index: 2, updated_at: ago(3) } })], items: [speech(1)] });
    await t.router.start(() => {});
    t.env.click(t.root.querySelector('.hm-doc'));
    await settle(t.env);
    assert.equal(location.hash, '#reader/a%20b');
    t.env.click(t.root.querySelector('.hm-item'));
    await settle(t.env);
    assert.equal(location.hash, '#history');
  });

  it('the status pill follows the app state and names itself like the status chip', async () => {
    t = await mount();
    assert.equal(t.$('hm-status-label').textContent, 'No model loaded', 'the lists answered, so the server is reachable');
    t.store.app.set({
      connection: { ...t.store.app.get().connection, state: 'online' },
      models: { list: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' }], defaults: { tts: 'kokoro', stt: '' }, fetchedAt: NOW, error: '' },
    });
    await settle(t.env);
    assert.equal(t.$('hm-status-label').textContent, 'Kokoro · GPU');
    assert.ok(t.$('hm-status-pill').classList.contains('pill--ok'));
    assert.match(t.$('hm-status').getAttribute('aria-label'), /Open status\.$/);
  });

  it('asks only the library and the history: never the AI assistant', async () => {
    const seen = [];
    t = await mount({
      documents: [doc('d', 'Dracula')], items: [speech(1)],
      routes: {
        'GET /api/assistant': () => { seen.push('assistant'); return json(200, { enabled: true }); },
        'POST /api/assistant/ask': () => { seen.push('ask'); return json(200, {}); },
      },
    });
    assert.deepEqual(seen, []);
    assert.deepEqual([t.hits.documents, t.hits.history], [1, 1]);
  });
});
