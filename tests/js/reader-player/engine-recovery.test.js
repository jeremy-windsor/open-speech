import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, json, partFor } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

describe('reader-player/engine recovery', () => {
  it('[RE-11] a part the server could not render stops playback with its error (or a plain default)', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.errors.set(2, 'The model ran out of memory');
    server.errors.set(4, null);
    engine.load(ctx.doc);
    engine.playFrom(2);
    await settle(env);
    assert.equal(engine.state.status.text, "Couldn't play: The model ran out of memory");
    assert.equal(engine.state.playing, false);
    engine.playFrom(4);
    await settle(env);
    assert.equal(engine.state.status.text, "Couldn't play: The audio for this paragraph could not be rendered");
  });

  it('[RE-19] on ended: an errored next part halts on that block (Play retries it, the saved position points at it); the last part finishes', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.errors.set(1, 'Render failed');
    const a = await startPlaying(ctx);
    a.__end();
    await settle(env);
    assert.equal(engine.state.cursor, 1);
    assert.equal(engine.state.status.text, "Couldn't play: Render failed");
    assert.equal(engine.state.playing, false);
    await engine.saveProgress();
    assert.equal(server.saved.at(-1).block_index, 1);
    server.errors.delete(1);
    engine.play();
    await settle(env);
    assert.equal(server.renders.at(-1).from_block, 1, 'Play retries the block that failed');
    assert.equal(playingEl().dataset.part, 'p1');
    engine.jumpTo(ctx.doc.blocks.length - 1);
    await settle(env);
    const last = playingEl();
    loadMeta(ctx, last);
    last.__end();
    await settle(env);
    assert.equal(engine.state.status.text, 'Finished');
  });

  it('[RE-20] an error on the preloaded element only drops the preload: the next part is rendered fresh when it is due', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    const a = await startPlaying(ctx);
    const b = ctx.b();
    assert.equal(b.dataset.part, 'p1');
    b.__fail(2);
    assert.equal(b.dataset.part, undefined);
    assert.equal(engine.state.status.text, '', 'nothing is reported for a preload');
    const renders = server.renders.length;
    a.__end();
    await settle(env);
    assert.equal(server.renders.length, renders + 1);
    assert.equal(server.renders.at(-1).from_block, 1);
    assert.equal(playingEl().dataset.part, 'p1');
  });

  it('[RE-21] an error on the playing element refetches the parts and resumes at the same position; a final error or 10 failures stop; no fresh part restarts from the saved spot', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    let a = await startPlaying(ctx);
    a.__timeupdate(2.5);
    const polls = server.polls.length;
    server.sig = 's2';
    a.__fail(2);
    await settle(env);
    assert.equal(server.polls.length, polls + 1, 'the parts are fetched again for a fresh signed URL');
    assert.match(a.getAttribute('src'), /sig=s2/);
    assert.equal(a.dataset.seek, '2.5');
    assert.equal(engine.state.playing, true);

    // Ten failures in a row (each fresh source fails before it plays) give up.
    a.__playPolicy = 'pending';
    for (let i = 0; i < 12 && engine.state.playing; i += 1) {
      a.__fail(4);
      await settle(env, 5000);
    }
    assert.equal(engine.state.status.text, "Couldn't play: The audio could not be loaded");
    assert.equal(engine.state.playing, false);

    // A final answer (404: the document is gone) stops at once.
    ctx.env.restore();
    ctx = await setup();
    a = await startPlaying(ctx);
    ctx.server.partsHandler = () => json(404, { error: { message: 'Document not found' } });
    a.__fail(2);
    await settle(ctx.env);
    assert.equal(ctx.engine.state.status.text, "Couldn't play: Document not found");

    // No fresh part: save where it stopped and start again from there.
    ctx.env.restore();
    ctx = await setup();
    a = await startPlaying(ctx, { block: 2 });
    a.__timeupdate(2.4);
    ctx.server.partsHandler = () => json(200, { parts: [partFor(ctx.doc, 2, { status: 'rendering', url: null, duration_ms: 0 })] });
    a.__fail(2);
    await settle(ctx.env);
    ctx.server.partsHandler = null;
    await settle(ctx.env, 1000);
    const el = playingEl();
    assert.equal(el.dataset.part, 'p2');
    assert.equal(el.dataset.seek, '2', 'restarted at the start of the sentence it stopped in');
  });

  it('[RE-22] the failure count and "Reconnecting…" clear once the element really plays', async () => {
    ctx = await setup();
    const { engine, env } = ctx;
    const a = await startPlaying(ctx);
    a.__playPolicy = 'pending';
    a.__fail(4);
    await settle(env, 100);
    a.__fail(4);
    await settle(env, 100);
    assert.equal(engine.state.status.text, 'Reconnecting…');
    await settle(env, 2000);
    a.__resolvePlay();
    await settle(env);
    assert.equal(engine.state.status.text, '');
    // The counter restarted: one more failure does not say "Reconnecting…" yet.
    a.__playPolicy = 'resolve';
    a.__fail(4);
    await settle(env, 10);
    assert.notEqual(engine.state.status.text, 'Reconnecting…');
  });

  it('[RE-23] a failure keeps where it stopped, halts, shows "Couldn\'t play: …" and an error toast whose Retry plays again', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    const a = await startPlaying(ctx, { block: 3 });
    a.__timeupdate(2.2);
    server.partsHandler = () => json(422, { error: { message: 'The voice is not available' } });
    a.__fail(2);
    await settle(env);
    assert.equal(engine.state.status.text, "Couldn't play: The voice is not available");
    assert.equal(engine.state.status.kind, 'error');
    assert.equal(a.paused, true);
    const toastEl = document.querySelector('#app-toasts-alert .toast--error');
    assert.ok(toastEl, 'an error toast');
    assert.match(toastEl.textContent, /Couldn't play: The voice is not available/);
    server.partsHandler = null;
    const retry = [...toastEl.querySelectorAll('button')].find((b) => b.textContent === 'Retry');
    env.click(retry);
    await settle(env);
    const el = playingEl();
    assert.equal(el.dataset.part, 'p3');
    assert.equal(el.dataset.seek, '2', 'Retry resumes at the sentence where it stopped');
    assert.equal(document.querySelector('#app-toasts-alert .toast--error'), null, 'the toast goes once audio plays');
  });
});
