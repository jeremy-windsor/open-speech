import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, json, networkError } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

describe('reader-player/engine parts', () => {
  it('[RE-01] two <audio> elements in #app-audio-host alternate: the active one plays, the idle one preloads the next part, and ended hands over without a gap', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    engine.load(ctx.doc);
    const a = ctx.a();
    const b = ctx.b();
    assert.equal(a.localName, 'audio');
    assert.equal(a.getAttribute('preload'), 'auto');
    assert.equal(b.getAttribute('preload'), 'auto');
    assert.equal(a.parentNode.id, 'app-audio-host');
    assert.equal(document.querySelectorAll('#app-audio-host audio').length, 3, 'the shared #app-audio and the Reader pair');
    await startPlaying(ctx);
    assert.equal(playingEl(), a);
    assert.equal(b.dataset.part, 'p1', 'the idle element preloads the next part');
    assert.match(b.getAttribute('src'), /p1\.mp3/);
    assert.equal(b.paused, true);
    const renders = server.renders.length;
    a.__end();
    assert.equal(b.paused, false, 'the next part starts inside the ended handler');
    await settle(env);
    assert.equal(playingEl(), b);
    assert.equal(engine.state.cursor, 1);
    assert.equal(a.dataset.part, 'p2', 'and the freed element preloads the one after');
    assert.ok(server.renders.length >= renders);
  });

  it('[RE-12] a part start reloads only when the part id or signed URL changed or the element failed; it seeks the offset, applies the speed and pauses the previous element', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    const a = await startPlaying(ctx);
    const b = ctx.b();
    engine.setSpeed(1.5);
    let emptied = 0;
    b.addEventListener('emptied', () => { emptied += 1; });
    a.__end();
    await settle(env);
    assert.equal(emptied, 0, 'the preloaded part plays as loaded');
    assert.equal(b.playbackRate, 1.5);
    assert.equal(b.defaultPlaybackRate, 1.5);
    assert.equal(a.paused, true);
    // An expired link: the same part comes back with a fresh signature and loads again at the same position.
    loadMeta(ctx, b);
    b.__timeupdate(1.25);
    server.sig = 's2';
    b.__fail(2);
    await settle(env, 100);
    assert.match(b.getAttribute('src'), /sig=s2/);
    assert.equal(b.dataset.seek, '1.25', 'the position is kept for the reloaded source');
    assert.equal(playingEl(), b);
  });

  it('[RE-16] every src change goes through one path: a parked seek and loop are cleared and data-part follows the source', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.pending.set(0, 3);
    engine.load(ctx.doc);
    engine.play();
    await settle(env, 10);
    const keepAlive = ctx.b();
    assert.match(keepAlive.getAttribute('src'), /^data:audio\/wav/, 'the keep-alive silence while waiting');
    assert.equal(keepAlive.loop, true);
    assert.equal(keepAlive.dataset.part, undefined, 'silence never counts as a part');
    keepAlive.dataset.seek = '9';
    await settle(env, 3000);
    const a = playingEl();
    assert.equal(a, ctx.a());
    loadMeta(ctx, a);
    await settle(env);
    assert.equal(keepAlive.loop, false);
    assert.equal(keepAlive.dataset.part, 'p1', 'the preload replaced the silence');
    assert.equal(keepAlive.dataset.seek, undefined, 'and dropped the parked seek');
    assert.equal(keepAlive.currentTime, 0);
  });

  it('[RE-17] when a part starts: the resume is used up, a retry line clears, the lock screen and the cursor update, progress saves, and the render job is re-pointed at the playing part', async () => {
    const progress = { block_index: 3, offset_ms: 1200, model: 'kokoro', voice: 'af_heart', speed: 1, updated_at: 'x' };
    ctx = await setup();
    ctx.doc.progress = progress;
    const { engine, env, server } = ctx;
    engine.load(ctx.doc);
    assert.equal(engine.state.timeline.elapsedMs > 0, true, 'the saved offset shows before playing');
    engine.play();
    await settle(env);
    const a = playingEl();
    assert.equal(a.dataset.part, 'p3');
    loadMeta(ctx, a);
    await settle(env);
    assert.equal(a.currentTime, 1.2);
    assert.equal(env.mediaSession.metadata.title, 'Down the Rabbit-Hole');
    assert.equal(env.mediaSession.metadata.album, "Alice's Adventures in Wonderland");
    assert.ok(env.mediaSession.positionStates.length > 0);
    assert.equal(server.renders.length, 1, 'the start already asked for this window');
    a.__end();
    await settle(env);
    assert.equal(playingEl().dataset.part, 'p4');
    assert.deepEqual(server.renders.at(-1), { from_block: 4, model: 'kokoro', voice: 'af_heart' }, 're-pointed at the part now playing');
    assert.equal(engine.state.cursor, 4);
    assert.equal(server.saved.at(-1).block_index, 4);
    a.__fail(2);   // a failure on the old element changes nothing
    assert.equal(engine.state.status.text, '');
  });

  it('[RE-18] the lookahead polls at least 1 s apart (backing off on failures) until the next part is ready, preloads it, stops while paused and restarts on play', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.pending.set(1, 4);
    const a = await startPlaying(ctx);
    const b = ctx.b();
    assert.notEqual(b.dataset.part, 'p1');
    const polls0 = server.polls.length;
    await settle(env, 900);
    assert.equal(server.polls.length, polls0, 'no poll before 1 s');
    await settle(env, 200);
    assert.equal(server.polls.length, polls0 + 1);
    server.partsHandler = () => networkError();
    await settle(env, 1000);
    await settle(env, 1000);
    await settle(env, 1000);       // three failed polls in a row
    const failed = server.polls.length;
    await settle(env, 1000);
    assert.equal(server.polls.length, failed, 'backing off after failures: the next poll waits 2 s');
    await settle(env, 1000);
    assert.equal(server.polls.length, failed + 1);
    server.partsHandler = null;
    engine.pause();
    await settle(env, 10000);
    const paused = server.polls.length;
    await settle(env, 10000);
    assert.equal(server.polls.length, paused, 'no polling while paused');
    engine.play();
    await settle(env, 1100);
    assert.ok(server.polls.length > paused, 'play restarts the lookahead');
    server.pending.set(1, 0);
    await settle(env, 2100);
    assert.equal(b.dataset.part, 'p1');
    assert.equal(a.paused, false);
    void json;
  });
});
