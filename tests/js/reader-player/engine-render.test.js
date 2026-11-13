import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, json, delay, partFor, load } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

describe('reader-player/engine render protocol', () => {
  it('[RE-07] POST /render {from_block, model, voice}, then GET /parts with the same window until the part is ready', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.pending.set(3, 2);
    engine.load(ctx.doc);
    engine.playFrom(3);
    await settle(env, 5000);
    assert.deepEqual(server.renders[0], { from_block: 3, model: 'kokoro', voice: 'af_heart' });
    assert.ok(server.polls.length >= 2);
    for (const poll of server.polls) assert.deepEqual(poll, { from: 3, model: 'kokoro', voice: 'af_heart' });
    assert.equal(playingEl().dataset.part, 'p3');
  });

  it('[RE-07] an answer for an older start is ignored and never overwrites the parts of a newer one', async () => {
    ctx = await setup({
      routes: {
        'POST /v1/reader/documents/:id/render': (req) => {
          ctx.server.renders.push(req.json);
          const from = req.json.from_block;
          return delay(from === 0 ? 3000 : 5, () => json(200, { parts: ctx.server.partsFrom(from) }));
        },
      },
    });
    const { engine, env } = ctx;
    engine.load(ctx.doc);
    engine.play();                 // from block 0: its render answers in 3 s
    await settle(env, 10);
    engine.playFrom(5);            // a newer start answers at once
    await settle(env, 100);
    assert.equal(playingEl().dataset.part, 'p5');
    await settle(env, 3500);       // the old answer (parts 0–3) arrives now
    assert.equal(playingEl().dataset.part, 'p5', 'the stale answer started nothing');
    assert.equal(engine.state.cursor, 5);
    engine.jumpTo(1);              // chapter 1 (blocks 0–4) was never rendered for this window
    await settle(env, 5);
    assert.equal(engine.state.timeline.estimated, true, 'the stale parts 0–3 were not taken in');
  });

  it('[RE-08] when the server job was replaced, our part stays pending in no job and the render is asked again after 3 s', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    // As src/reader/renderer.py answers: GET /parts plans the same window with the same ids whatever the job,
    // and a part in no job is 'pending'. Only the window a POST /render points the single job at is rendered.
    let job = null;
    const inNoJob = (from) => server.partsFrom(from).map((part) => ({ ...part, status: 'pending', url: null, duration_ms: null, sentences: [] }));
    server.renderHandler = (req) => {
      job = server.renders.length === 1 ? 'another tab' : req.json.from_block;   // another tab asks right after us
      return json(200, { parts: inNoJob(req.json.from_block) });
    };
    server.partsHandler = (req) => {
      const from = Number(req.query.get('from_block'));
      return json(200, { parts: job === from ? server.partsFrom(from) : inNoJob(from) });
    };
    engine.load(ctx.doc);
    engine.playFrom(2);
    await settle(env, 2900);
    assert.equal(server.renders.length, 1);
    assert.equal(playingEl(), null);
    await settle(env, 1200);
    assert.equal(server.renders.length, 2, 're-requested once our part had been pending for 3 s');
    await settle(env, 4000);
    assert.equal(playingEl().dataset.part, 'p2');
    assert.equal(server.renders.length, 2, 'a part that is ours is not asked for again');
  });
});

// The server has one render job and a request replaces it; the worker drops a
// part that left the job at its next sentence boundary and saves nothing (src/reader/renderer.py, _wanted). Two
// listeners waiting for a first part at once, each asking again every 3 s, took the job from each other for ever
// once a part needed more than about 3 s. Each re-request now waits twice as long as the one before.
const RERENDER_WAITS = [3000, 6000, 12000, 24000];   // the wait before the 1st, 2nd, 3rd and every later re-request
const waitBefore = (n) => RERENDER_WAITS[Math.min(n, RERENDER_WAITS.length - 1)];
const SENTENCE_RENDER_MS = 400;                      // synthesis time of one sentence
const LOOKAHEAD = 3;                                 // ReaderRenderer lookahead_parts default

/** The server's single render job, as src/reader/renderer.py runs it: request() points the one job at a window,
 * the worker renders the job's first unrendered part sentence by sentence and drops it (nothing saved) once it
 * left the job, and the statuses are 'ready' once saved, 'rendering' for the worker's part, else 'pending'. */
class OneJobRenderer {
  constructor(clock, { sentences, docs }) {
    this.clock = clock;
    this.sentences = sentences;
    this.docs = docs;               // document id -> block count
    this.job = null;                // part keys `${doc}:${block}`
    this.worker = null;             // { key, done }
    this.ready = new Set();
    this.requests = [];
    this.dropped = 0;
  }

  window(doc, from) {
    const keys = [];
    for (let b = from; b < Math.min(this.docs[doc], from + LOOKAHEAD); b += 1) keys.push(`${doc}:${b}`);
    return keys;
  }

  describe(doc, from) {
    return this.window(doc, from).map((key) => ({
      block: Number(key.split(':')[1]),
      status: this.ready.has(key) ? 'ready' : this.worker?.key === key ? 'rendering' : 'pending',
    }));
  }

  request(doc, from, who) {
    this.job = this.window(doc, from);
    this.requests.push({ who, at: this.clock.now() });
    this.start();
    return this.describe(doc, from);
  }

  start() {
    if (this.worker || !this.job) return;
    const key = this.job.find((k) => !this.ready.has(k));
    if (!key) return;
    this.worker = { key, done: 0 };
    this.clock.setTimeout(() => this.sentence(), SENTENCE_RENDER_MS);
  }

  sentence() {
    const w = this.worker;
    w.done += 1;
    if (!this.job.includes(w.key)) {           // _wanted(), before the next sentence or before saving
      this.dropped += 1;
      this.worker = null;
      this.start();
      return;
    }
    if (w.done >= this.sentences) {
      this.ready.add(w.key);
      this.worker = null;
      this.start();
      return;
    }
    this.clock.setTimeout(() => this.sentence(), SENTENCE_RENDER_MS);
  }
}

/** Another tab or device on document 'other', running the engine's startAt() loop with the given waits. */
function otherListener(renderer, clock, pollMs, waits, startMs = 500) {
  const me = { readyAt: null };
  let parts = null;
  let again = 0;
  let lastRender = 0;
  const step = () => {
    const seen = parts?.find((p) => p.block === 0);
    if (!parts || ((!seen || seen.status === 'pending') && clock.now() - lastRender > waits(again))) {
      if (parts) again += 1;
      lastRender = clock.now();
      parts = renderer.request('other', 0, 'B');
    } else {
      parts = renderer.describe('other', 0);
    }
    if (parts.find((p) => p.block === 0)?.status === 'ready') {
      me.readyAt = clock.now();                  // it plays; its lookahead only polls /parts
      return;
    }
    clock.setTimeout(step, pollMs);
  };
  clock.setTimeout(step, startMs);
  return me;
}

describe('reader-player/engine render protocol: two listeners and the single render job', () => {
  const others = [
    ['the other on this page too', waitBefore],
    ['the other on a page opened before this fix, asking again every 3 s', () => RERENDER_WAITS[0]],
  ];
  for (const [who, waits] of others) for (const seconds of [4, 8]) {
    it(`[RE-08] two listeners waiting at once for ${seconds} s parts both play (${who}): this one asks again after 3 s, then 6, 12 and 24 s`, async () => {
      ctx = await setup();
      const { engine, env, server, doc } = ctx;
      const { POLL_MS } = await load('views/reader/play-logic.js');
      const renderer = new OneJobRenderer(env.clock, { sentences: (seconds * 1000) / SENTENCE_RENDER_MS, docs: { [doc.id]: doc.blocks.length, other: 1 } });
      const answer = (rows) => rows.map(({ block, status }) => (status === 'ready'
        ? partFor(doc, block)
        : partFor(doc, block, { status, url: null, duration_ms: null, sentences: [] })));
      server.renderHandler = (req) => json(200, { parts: answer(renderer.request(doc.id, req.json.from_block, 'A')) });
      server.partsHandler = (req) => json(200, { parts: answer(renderer.describe(doc.id, Number(req.query.get('from_block')))) });
      const other = otherListener(renderer, env.clock, POLL_MS, waits);
      const t0 = env.clock.now();
      engine.load(doc);
      engine.playFrom(1);
      await settle(env, 60000);
      const asked = renderer.requests.filter((q) => q.who === 'A').map((q) => q.at - t0);
      const why = `this player asked at ${JSON.stringify(asked)} ms, the other ${renderer.requests.length - asked.length}x; `
        + `${renderer.dropped} half-rendered parts dropped; the other ready at ${other.readyAt === null ? 'never' : `${other.readyAt - t0} ms`}; `
        + `status ${JSON.stringify(engine.state.status?.text ?? engine.state.status)}`;
      assert.equal(playingEl()?.dataset.part, 'p1', `after 60 s this player does not play: ${why}`);
      assert.notEqual(other.readyAt, null, `after 60 s the other listener's part is not ready: ${why}`);
      asked.slice(1).forEach((at, n) => assert.ok(at - asked[n] > waitBefore(n),
        `re-request ${n + 1} came ${at - asked[n]} ms after the one before, not after more than ${waitBefore(n)} ms: ${why}`));
    });
  }

  it('[RE-08] a part that stays in no job is asked for again after 3 s, then 6, 12 and every 24 s; a new start begins at 3 s again', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    // Another listener holds the job the whole time: our part is always 'pending' in no job.
    const inNoJob = (from) => server.partsFrom(from).map((part) => ({ ...part, status: 'pending', url: null, duration_ms: null, sentences: [] }));
    const asked = [];
    server.renderHandler = (req) => {
      asked.push({ from: req.json.from_block, at: env.clock.now() });
      return json(200, { parts: inNoJob(req.json.from_block) });
    };
    server.partsHandler = (req) => json(200, { parts: inNoJob(Number(req.query.get('from_block'))) });
    engine.load(ctx.doc);
    engine.playFrom(2);
    await settle(env, 75000);
    const gaps = asked.slice(1).map((q, n) => q.at - asked[n].at);
    assert.ok(gaps.length >= 5, `only ${gaps.length} re-requests in 75 s: ${JSON.stringify(gaps)}`);
    gaps.forEach((gap, n) => assert.ok(gap > waitBefore(n) && gap <= waitBefore(n) + 1000,
      `re-request ${n + 1} came ${gap} ms after the one before, expected just over ${waitBefore(n)} ms (all gaps ${JSON.stringify(gaps)})`));
    assert.equal(playingEl(), null);
    const before = asked.length;
    engine.playFrom(5);                          // a new start: its count starts again at 3 s
    await settle(env, 4100);
    const fresh = asked.slice(before);
    assert.deepEqual(fresh.map((q) => q.from), [5, 5], `the new start asked ${JSON.stringify(fresh)}`);
    assert.ok(fresh[1].at - fresh[0].at <= RERENDER_WAITS[0] + 1000, `the new start asked again after ${fresh[1].at - fresh[0].at} ms`);
  });
});
