import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, collect, showReading, load } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

describe('reader-player/engine tracking', () => {
  it('[RE-51] a frame ticker follows the sentence while playing (Safari fires timeupdate only ~4×/s); a multi-block part without marks estimates the block from its characters', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.override.set(2, { last_block: 3, duration_ms: 8000, sentences: [{ block: 2, index: 0, start_ms: 0 }, { block: 2, index: 1, start_ms: 2000 }, { block: 3, index: 0, start_ms: 4000 }, { block: 3, index: 1, start_ms: 6000 }] });
    const cursors = collect(engine, 'cursor');
    const el = await startPlaying(ctx, { block: 2 });
    el.__currentTime = 2.2;   // no timeupdate yet: the frame ticker reads currentTime
    env.raf.flush();
    assert.deepEqual([engine.state.cursor, engine.state.sentence], [2, 1]);
    assert.equal(cursors.at(-1).reason, 'playback');
    el.__currentTime = 4.5;
    env.raf.flush();
    assert.deepEqual(cursors.at(-1), { block: 3, sentence: 0, chapter: 0, reason: 'playback' });
    engine.pause();
    await settle(env);
    assert.equal(env.raf.pending, 0, 'no ticker while paused');
    ctx.env.restore();
    // No marks: the share of characters (three blocks of equal length over 9 s).
    ctx = await setup();
    ctx.server.override.set(6, { last_block: 8, duration_ms: 9000, sentences: [] });
    const el2 = await startPlaying(ctx, { block: 6 });
    el2.__currentTime = 4.2;
    ctx.env.raf.flush();
    assert.equal(ctx.engine.state.cursor, 7);
    assert.equal(ctx.engine.state.sentence, -1);
  });

  it('[REV-34] with the page hidden (a locked phone, no animation frames), timeupdate drives the cursor, the sleep timer, the saves and the chapter on the lock screen', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    // Block 5 (chapter 2's heading) is a 90 s part with a sentence every 2 s, so the position keeps moving.
    server.override.set(5, { duration_ms: 90000, sentences: Array.from({ length: 45 }, (_, i) => ({ block: 5, index: i, start_ms: i * 2000 })) });
    const el = await startPlaying(ctx, { block: 3 });
    env.setHidden(true);
    await settle(env);
    env.raf.flush();
    assert.equal(env.raf.pending, 0, 'nothing runs on frames while hidden');
    el.__timeupdate(2.3);
    assert.equal(engine.state.sentence, 1, 'the cursor follows timeupdate');
    // The chapter changes at a part start, on the lock screen too.
    el.__end();
    await settle(env);
    const next = playingEl();
    loadMeta(ctx, next);
    next.__end();
    await settle(env);
    assert.equal(engine.state.chapter, 1);
    assert.equal(env.mediaSession.metadata.title, 'The Pool of Tears');
    // A one-minute sleep timer runs out on timeupdate alone.
    engine.setSleep({ minutes: 1 });
    const now = playingEl();
    loadMeta(ctx, now);
    const saves = server.saved.length;
    for (let i = 1; i <= 260 && engine.state.playing; i += 1) {
      await env.clock.tickAsync(250);
      now.__timeupdate(i / 4);
    }
    assert.equal(engine.state.playing, false);
    assert.equal(engine.state.status.text, 'Sleep timer ended');
    assert.ok(server.saved.length > saves, 'saves kept coming while hidden');
  });

  it('keeps the screen on while the reading view plays, when the wake-lock pref is on, and lets go on pause, hide and leaving the view', async () => {
    ctx = await setup({ prefs: { wakeLock: true } });
    const { engine, env } = ctx;
    engine.load(ctx.doc);
    await showReading(ctx);
    await startPlaying(ctx, { block: 1 });
    assert.equal(env.wakeLocks.length, 1);
    assert.equal(env.wakeLocks[0].type, 'screen');
    engine.pause();
    await settle(env);
    assert.equal(env.wakeLocks[0].released, true, 'released on pause');
    engine.play();
    await settle(env);
    assert.equal(env.wakeLocks.length, 2);
    env.setHidden(true);
    await settle(env);
    assert.equal(env.wakeLocks[1].released, true, 'released when hidden');
    env.setHidden(false);
    await settle(env);
    assert.equal(env.wakeLocks.length, 3);
    await (await load('core/router.js')).navigate('speak');
    await settle(env);
    assert.equal(env.wakeLocks[2].released, true, 'released when the reading view is left (audio keeps playing)');
    assert.equal(engine.state.playing, true);
    ctx.prefs.set({ wakeLock: false });
    await showReading(ctx);
    assert.equal(env.wakeLocks.length, 3, 'off by default and when the pref is off');
  });

  it('a quick Play, Pause, Play keeps one wake lock and leaks none; a lock the system lets go is asked for again at the next play', async () => {
    ctx = await setup({ prefs: { wakeLock: true } });
    const { engine, env } = ctx;
    engine.load(ctx.doc);
    await showReading(ctx);
    await startPlaying(ctx, { block: 1 });
    engine.pause();
    await settle(env);
    // Faster than the browser answers the first request: both requests are answered after the last Play.
    engine.play();
    engine.pause();
    engine.play();
    await settle(env);
    const held = () => env.wakeLocks.filter((lock) => !lock.released);
    assert.equal(engine.state.playing, true);
    assert.equal(held().length, 1, `one lock held while playing (${env.wakeLocks.length} asked for)`);
    engine.pause();
    await settle(env);
    assert.equal(held().length, 0, 'none held once paused: the older answer was released, not overwritten');
    // The system releases the lock (battery saver) while the view plays: the next visible, playing moment asks again.
    engine.play();
    await settle(env);
    assert.equal(held().length, 1);
    const asked = env.wakeLocks.length;
    await held()[0].release();
    await settle(env);
    assert.equal(env.wakeLocks.length, asked, 'nothing re-asks from the release itself');
    env.setHidden(false);   // the page shows (again) while still playing
    await settle(env);
    assert.equal(env.wakeLocks.length, asked + 1, 'a dead lock is not mistaken for a held one');
    assert.equal(held().length, 1);
    engine.pause();
    await settle(env);
    assert.equal(held().length, 0);
  });
});
