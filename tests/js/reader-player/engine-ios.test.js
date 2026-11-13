// iOS behaviours, on fake media elements. The on-device checks (a locked iPhone, a Home Screen app) are
// done by hand.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, load } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

describe('reader-player/engine on iOS', () => {
  it('[RE-25] while the first part renders, the idle element loops unmuted silence (iOS keeps the audio session and the lock screen); a system pause of it counts as the listener pausing', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.pending.set(0, 100);
    engine.load(ctx.doc);
    engine.play();
    await settle(env, 10);
    const keep = ctx.b();
    assert.match(keep.getAttribute('src'), /^data:audio\/wav;base64,/);
    assert.equal(keep.loop, true);
    assert.equal(keep.muted, false);
    assert.equal(keep.paused, false);
    keep.__systemPause();   // AirPods out while waiting
    await settle(env);
    assert.equal(engine.state.playing, false);
    assert.equal(engine.state.waiting, false);
    server.pending.set(0, 0);
    await settle(env, 5000);
    assert.equal(playingEl(), null, 'the start was cancelled');
    engine.play();
    await settle(env);
    assert.equal(playingEl(), ctx.a());
    assert.equal(keep.loop, false, 'the keep-alive stops once a part plays');
  });

  it('[RE-26] every gesture in the Reader unlocks both elements (an unmuted silent play inside the gesture) until each has played once; a part held for later is muted and keeps its position', async () => {
    ctx = await setup({ playPolicy: 'gesture' });
    const { engine, env } = ctx;
    const { mountPlayer } = await load('views/reader/player.js');
    const player = mountPlayer(document.getElementById('rd-player-slot'), engine, {});
    await player.ready;
    engine.load(ctx.doc);
    const a = ctx.a();
    const b = ctx.b();
    const section = document.getElementById('view-reader');
    env.withGesture(() => section.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    await settle(env);
    for (const el of [a, b]) {
      assert.equal(el.dataset.unlocked, '1');
      assert.equal(el.__plays[0].gesture, true);
      assert.match(el.__plays[0].src, /^data:audio\/wav/);
      assert.equal(el.paused, true, 'the silent clip is paused right away');
      assert.equal(el.muted, false);
    }
    const plays = a.__plays.length + b.__plays.length;
    env.withGesture(() => section.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    await settle(env);
    assert.equal(a.__plays.length + b.__plays.length, plays, 'once both are unlocked, gestures no longer touch them');
    // A later play after a network wait (no gesture) is allowed now.
    engine.play();
    await settle(env);
    assert.ok(playingEl(), 'the part plays outside the gesture');
    player.destroy();
    ctx.env.restore();
    // An element that holds a preloaded part is unlocked muted, and its position is put back.
    ctx = await setup({ playPolicy: 'gesture' });
    ctx.engine.load(ctx.doc);
    const b2 = ctx.b();
    b2.setAttribute('src', '/v1/reader/media/p9.mp3?exp=1&sig=s1');
    b2.dataset.part = 'p9';
    b2.__loadMetadata(4);
    b2.currentTime = 1.5;
    let mutedDuringPlay = null;
    b2.addEventListener('play', () => { mutedDuringPlay = b2.muted; });
    ctx.env.withGesture(() => ctx.engine.unlock());
    await settle(ctx.env);
    assert.equal(mutedDuringPlay, true);
    assert.equal(b2.muted, false);
    assert.equal(b2.paused, true);
    assert.equal(b2.currentTime, 1.5);
    assert.equal(b2.dataset.unlocked, '1');
  });

  it('[RE-27] a system pause (headphones out, a call) stops and saves; a system play on the active element plays; nothing ever resumes on its own', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    const el = await startPlaying(ctx, { block: 2 });
    el.__timeupdate(2.2);
    const saves = server.saved.length;
    el.__systemPause();
    await settle(env);
    assert.equal(engine.state.playing, false);
    assert.equal(server.saved.length, saves + 1);
    assert.equal(env.mediaSession.playbackState, 'paused');
    const plays = el.__plays.length;
    await settle(env, 60000);
    assert.equal(el.__plays.length, plays, 'never auto-resumed');
    el.paused = false;
    el.dispatchEvent(new Event('play'));   // the system resumed this element (a headphone button)
    await settle(env);
    assert.equal(engine.state.playing, true);
    assert.equal(env.mediaSession.playbackState, 'playing');
  });
});
