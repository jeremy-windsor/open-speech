import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, makeDoc, load } from './helpers.js';

const ms = await load('views/reader/media-session.js');

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

/** A canvas good enough for tiles.artworkDataUrl (the fake DOM has none). */
function fakeCanvas() {
  const create = document.createElement.bind(document);
  document.createElement = (tag) => {
    if (String(tag).toLowerCase() !== 'canvas') return create(tag);
    const canvas = { width: 0, height: 0 };
    canvas.getContext = () => ({ createLinearGradient: () => ({ addColorStop() {} }), fillRect() {}, fillText() {} });
    canvas.toDataURL = () => `data:image/png;base64,${canvas.width}`;
    return canvas;
  };
}

describe('reader-player/media session', () => {
  it('[RE-47] the lock screen drives the Reader: play, pause, previous (restart or back), next, skips by the offset or the prefs, seekto on the chapter, and stop; a clip that takes over replaces the handlers', async () => {
    ctx = await setup({ prefs: { snapToSentence: false } });
    const { engine, env, audio, server } = ctx;
    const marks = Array.from({ length: 30 }, (_, i) => ({ block: 2, index: i, start_ms: i * 2000 }));
    server.override.set(2, { duration_ms: 60000, sentences: marks });
    const el = await startPlaying(ctx, { block: 2 });
    const session = env.mediaSession;
    session.action('pause');
    assert.equal(engine.state.playing, false);
    session.action('play');
    assert.equal(engine.state.playing, true);
    el.__timeupdate(30);
    session.action('seekbackward', {});
    assert.equal(el.currentTime, 15, 'the skip prefs (15 s back)');
    session.action('seekforward', { seekOffset: 10 });
    assert.equal(el.currentTime, 25, 'the offset the system asked for');
    session.action('seekforward', {});
    assert.equal(el.currentTime, 55);
    const partStart = engine.state.timeline.elapsedMs - 55000;   // where block 2's part starts on the chapter timeline
    session.action('seekto', { seekTime: (partStart + 22000) / 1000 });
    assert.equal(engine.state.cursor, 2);
    assert.equal(Math.round(el.currentTime * 1000) / 1000, 22, 'seekto is in chapter time');
    session.action('previoustrack');
    assert.equal(engine.state.cursor, 2, 'more than 3 s in: the paragraph restarts');
    assert.equal(el.currentTime, 0);
    session.action('previoustrack');
    await settle(env);
    assert.equal(engine.state.cursor, 1, 'at its start: back one paragraph');
    loadMeta(ctx, playingEl());
    session.action('nexttrack');
    await settle(env);
    assert.equal(engine.state.cursor, 2);
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:take' });
    assert.equal(session.handlers.previoustrack, undefined, "a take's session has no track handlers");
    assert.equal(engine.state.playing, false, 'and the Reader paused under it');
    audio.closeCurrent();
    assert.equal(typeof session.handlers.nexttrack, 'function', 'the displaced Reader is offered again');
    session.action('stop');
    assert.equal(ctx.app.get().player, null);
    assert.equal(session.handlers.play, undefined, 'cleared on stop');
    assert.equal(session.metadata, null);
    assert.equal(session.playbackState, 'none');
  });

  it("[RE-47][RE-50] after a clip that displaced the Reader ends, the re-offered Reader's own release (the Reader's stop, a deleted document) clears the lock screen; while the clip plays, it leaves the clip's alone", async () => {
    ctx = await setup();
    const { engine, env, audio } = ctx;
    const session = env.mediaSession;
    await startPlaying(ctx, { block: 2 });
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:take' });
    audio.closeCurrent();
    assert.equal(typeof session.handlers.play, 'function', 'the displaced Reader is offered again');
    assert.equal(session.metadata.album, "Alice's Adventures in Wonderland");
    await engine.unload();   // what the Reader's own stop does (also the mini-player's Close through the session)
    assert.equal(session.handlers.play, undefined, 'its handlers are gone');
    assert.equal(session.metadata, null, 'and its metadata');
    assert.equal(session.playbackState, 'none');
    // A clip still playing over the displaced Reader keeps its own lock screen when the Reader lets go.
    engine.play();
    await settle(env);
    await audio.playClip({ id: 'take:2', owner: 'speak', title: 'Take 2', src: 'blob:take2' });
    assert.equal(session.metadata.title, 'Take 2');
    await engine.unload({ forget: true });
    assert.equal(session.metadata.title, 'Take 2', "the clip's metadata stays");
    assert.equal(typeof session.handlers.pause, 'function', "and so do its handlers");
    audio.closeCurrent();
  });

  it('[RE-47] with prefs.trackUnit = chapter, previous and next track move by chapter (previous restarts the chapter after 3 s)', async () => {
    ctx = await setup({ prefs: { trackUnit: 'chapter' } });
    const { engine, env } = ctx;
    await startPlaying(ctx, { block: 6 });
    env.mediaSession.action('nexttrack');
    await settle(env);
    assert.equal(engine.state.cursor, 9, 'the next chapter starts at block 9');
    loadMeta(ctx, playingEl());
    env.mediaSession.action('previoustrack');
    await settle(env);
    assert.equal(engine.state.cursor, 5, 'at the chapter start: the previous chapter');
    const el = playingEl();
    loadMeta(ctx, el);
    el.__end();
    await settle(env);
    loadMeta(ctx, playingEl());
    playingEl().__timeupdate(2);   // 4 s into the chapter
    env.mediaSession.action('previoustrack');
    await settle(env);
    assert.equal(engine.state.cursor, 5, 'past 3 s: the chapter restarts');
  });

  it('[RE-48] metadata: the chapter title, the author (or the voice when there is none), the document as album, and the tile as artwork at 96, 256 and 512; refreshed at chapter changes', async () => {
    assert.deepEqual(ms.metadata({ doc: makeDoc(), chapterIndex: 1, voiceLabel: 'Heart' }), { title: 'The Pool of Tears', artist: 'Lewis Carroll', album: "Alice's Adventures in Wonderland", artwork: [] });
    const single = makeDoc({ author: '', chapters: [['Only', 2]] });
    assert.deepEqual(ms.metadata({ doc: single, chapterIndex: 0, voiceLabel: 'Heart' }), { title: "Alice's Adventures in Wonderland", artist: 'Heart', album: "Alice's Adventures in Wonderland", artwork: [] });
    assert.deepEqual(ms.artworkList({ 96: 'data:a', 256: '', 512: 'data:c' }), [{ src: 'data:a', sizes: '96x96', type: 'image/png' }, { src: 'data:c', sizes: '512x512', type: 'image/png' }]);
    ctx = await setup();
    fakeCanvas();
    const { env } = ctx;
    const el = await startPlaying(ctx, { block: 4 });
    const md = env.mediaSession.metadata;
    assert.equal(md.title, 'Down the Rabbit-Hole');
    assert.equal(md.artist, 'Lewis Carroll');
    assert.equal(md.album, "Alice's Adventures in Wonderland");
    assert.deepEqual(md.artwork.map((a) => a.sizes), ['96x96', '256x256', '512x512']);
    assert.ok(md.artwork.every((a) => a.type === 'image/png' && a.src.startsWith('data:image/png')));
    assert.equal(ctx.app.get().player.title, 'Down the Rabbit-Hole', 'the mini-player shows the chapter');
    assert.equal(ctx.app.get().player.subtitle, "Alice's Adventures in Wonderland");
    el.__end();
    await settle(env);
    assert.equal(env.mediaSession.metadata.title, 'The Pool of Tears');
    assert.equal(ctx.app.get().player.title, 'The Pool of Tears');
  });

  it('[RE-49] the position state is chapter-relative, set after part starts, seeks, rate changes and the 4 Hz ticks, only with valid values, and an estimate that moves by 2% or less is not re-reported', async () => {
    assert.equal(ms.POSITION_SCOPE, 'chapter');
    assert.equal(ms.positionState({ durationMs: 0, elapsedMs: 0, rate: 1 }), null);
    assert.equal(ms.positionState({ durationMs: 10000, elapsedMs: 0, rate: 0 }), null);
    assert.deepEqual(ms.positionState({ durationMs: 10000, elapsedMs: 12000, rate: 1.5 }), { duration: 10, position: 10, rate: 1.5 });
    assert.deepEqual(ms.positionState({ scope: 'part', partDuration: 4, partPosition: 1, rate: 1 }), { duration: 4, position: 1, rate: 1 });
    assert.equal(ms.smoothDuration(100000, 101500), 100000, 'a 1.5% wobble is ignored');
    assert.equal(ms.smoothDuration(100000, 103000), 103000);
    assert.equal(ms.smoothDuration(100000, 101000, { partReady: true }), 101000, 'a part became ready');
    ctx = await setup();
    const { env, engine } = ctx;
    const el = await startPlaying(ctx, { block: 2 });
    const states = env.mediaSession.positionStates;
    assert.ok(states.length > 0);
    const last = states.at(-1);
    const chapter = engine.state.timeline;
    assert.equal(last.duration, chapter.durationMs / 1000, 'the whole chapter, not the part');
    assert.ok(last.duration > 4);
    const n = states.length;
    engine.setSpeed(1.5);
    assert.equal(states.at(-1).playbackRate, 1.5);
    el.__timeupdate(1);
    await env.clock.tickAsync(300);
    el.__timeupdate(1.3);
    assert.ok(states.length > n + 1);
    assert.ok(states.every((s) => s.duration > 0 && s.position <= s.duration && s.playbackRate > 0));
  });

  it('[RE-50] playbackState mirrors playing and paused, and is none after finishing and after stop', async () => {
    ctx = await setup();
    const { env, engine, audio } = ctx;
    const count = ctx.doc.blocks.length;
    await startPlaying(ctx, { block: count - 2 });
    assert.equal(env.mediaSession.playbackState, 'playing');
    engine.pause();
    assert.equal(env.mediaSession.playbackState, 'paused');
    engine.play();
    await settle(env);
    assert.equal(env.mediaSession.playbackState, 'playing');
    audio.closeCurrent();
    assert.equal(env.mediaSession.playbackState, 'none');
    engine.playFrom(count - 1);
    await settle(env);
    const el = playingEl();
    loadMeta(ctx, el);
    el.__end();
    await settle(env);
    assert.equal(env.mediaSession.playbackState, 'none');
  });

  it('[BUG-07] there is a stop handler, the position covers the chapter (not each part), the album is the document, and pitch is preserved on every element at every speed', async () => {
    ctx = await setup();
    const { env, engine } = ctx;
    await startPlaying(ctx, { block: 1 });
    assert.equal(typeof env.mediaSession.handlers.stop, 'function');
    assert.notEqual(env.mediaSession.metadata.album, 'Open Speech');
    assert.equal(env.mediaSession.metadata.album, "Alice's Adventures in Wonderland");
    assert.ok(env.mediaSession.positionStates.at(-1).duration > 4, 'longer than the 4 s part');
    engine.setSpeed(2.5);
    for (const el of [ctx.a(), ctx.b()]) {
      assert.equal(el.preservesPitch, true);
      assert.equal(el.webkitPreservesPitch, true);
      assert.equal(el.playbackRate, 2.5);
    }
  });
});
