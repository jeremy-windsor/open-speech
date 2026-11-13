// The Reader controller on the real engine (views/reader/engine.js) instead of the fake: the contracts the controller
// relies on hold there too. Unload keeps the document unless {forget: true}, a reset document is never saved over,
// and the controller reads no extra state snapshot per 'state' event (the real getter rebuilds one on every read).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, makeDoc, progressAt, $, textOf, settle, go, load } from './helpers.js';

let ctx;
let engineModule;
afterEach(() => {
  if (engineModule) engineModule.__resetForTests();
  ctx?.env.restore();
});

async function withRealEngine(options = {}) {
  engineModule = await load('views/reader/engine.js');
  engineModule.__resetForTests();
  ctx = await readerEnv({ loadEngine: async () => engineModule.getEngine(), ...options });
  return engineModule.getEngine();
}

const puts = () => ctx.env.fetch.callsTo('PUT /v1/reader/documents/:id/progress');

describe('views/reader with the real engine', () => {
  it('[REV-03] Start over releases the open document with unload({forget: true}); a plain unload (Close player) keeps it', async () => {
    const engine = await withRealEngine({ hash: '#reader/alice' });
    assert.equal(engine.state.docId, 'alice', 'opening loads the document into the engine');
    assert.equal(engine.state.cursor, 12, 'at the saved position');
    await engine.unload();   // Close player: stop and save; the document stays loaded
    await settle(ctx.env);
    assert.equal(engine.state.docId, 'alice');
    await go(ctx, 'reader', 50);
    ctx.env.click(document.querySelector('#rd-list .rd-row[data-doc="alice"] [data-action="row-menu"]'));
    await settle(ctx.env);
    ctx.env.click($('rd-row-menu').querySelector('[data-menu="restart"]'));
    await settle(ctx.env);
    ctx.env.click($('app-confirm-ok'));
    await settle(ctx.env, 50);
    assert.equal(engine.state.docId, '', 'the engine dropped it');
    const bodies = puts().map((p) => p.json.block_index);
    assert.deepEqual(bodies, [0], 'only the Start over position was saved: no save from the engine overwrote it');
    assert.equal(textOf(document.querySelector('#rd-list .rd-row[data-doc="alice"] .rd-row__pct')), '0%');
    await settle(ctx.env, 3000);
    assert.deepEqual(puts().map((p) => p.json.block_index), [0], 'and none later');
  });

  it('[RE-53] a speed change reaches the aside from the snapshot the engine sends: no extra snapshot is built per event', async () => {
    const long = (c, p) => `Paragraph ${p} of chapter ${c + 1}. ${'It goes on for a while, so the time left is minutes, not seconds. '.repeat(5)}`;
    const engine = await withRealEngine({
      hash: '#reader/alice',
      docs: [makeDoc({ perChapter: 10, paragraph: long, progress: progressAt(12) })],
    });
    let reads = 0;
    const getter = Object.getOwnPropertyDescriptor(engine, 'state').get;
    Object.defineProperty(engine, 'state', { configurable: true, get() { reads += 1; return getter.call(engine); } });
    const before = textOf($('rd-side-meta'));
    engine.setSpeed(2);
    await settle(ctx.env);
    assert.equal(reads, 0, "the controller's 'state' handler used the snapshot it was given");
    assert.match(before, /^Lewis Carroll · 40% · ~\d+ min left$/);
    assert.notEqual(textOf($('rd-side-meta')), before, 'the time left follows the new speed');
    assert.match(textOf($('rd-side-meta')), /^Lewis Carroll · 40% · ~\d+ min left$/);
  });
});
