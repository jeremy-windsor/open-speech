import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, makeDoc, json, networkError, collect, load } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

const saved = (progress) => ({ block_index: 2, offset_ms: 2000, model: 'kokoro', voice: 'af_heart', speed: 1, updated_at: '2026-10-22T09:00:00Z', ...progress });

async function resumed(progress = {}) {
  ctx = await setup({ doc: makeDoc({ progress: saved(progress) }), prefs: { snapToSentence: false } });
  ctx.engine.load(ctx.doc);
  ctx.engine.play();
  await settle(ctx.env);
  return playingEl();
}

describe('reader-player/engine saved position', () => {
  it('[RE-15] while a resume is still loading, its parked offset is where skips, previous track and saves start from', async () => {
    const el = await resumed();
    assert.equal(el.dataset.seek, '2', 'parked: currentTime is still 0');
    ctx.engine.seekBy(1);
    assert.equal(el.dataset.seek, '3');
    ctx.engine.seekBy(-2.5);
    assert.equal(el.dataset.seek, '0.5');
    const atHalf = ctx.engine.state.timeline.elapsedMs;
    ctx.engine.seekBy(2.5);
    assert.equal(el.dataset.seek, '3');
    assert.equal(Math.round(ctx.engine.state.timeline.elapsedMs - atHalf), 2500, 'the timeline counts the parked position');
    ctx.engine.setSpeed(1.5);   // a different payload, so the save goes out
    await ctx.engine.saveProgress();
    assert.equal(ctx.server.saved.at(-1).offset_ms, 2000, 'saved at the sentence the parked position is in');
    ctx.engine.prevTrack();     // 3 s into the paragraph at 1.5× is 2 s of listening: back one paragraph
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 1);
  });

  it('[RE-24] stopping (Close player, the lock screen, opening another document) keeps the position, halts, clears the status and releases the session; leaving the Reader does not stop it', async () => {
    ctx = await setup();
    const { engine, env, app, audio, server } = ctx;
    const router = await load('core/router.js');
    const el = await startPlaying(ctx, { block: 2 });
    el.__timeupdate(2.6);
    await router.navigate('speak');
    await settle(env);
    assert.equal(engine.state.playing, true, 'Back to library (any route change) keeps playing');
    assert.equal(app.get().player.id, 'reader:doc1');
    audio.closeCurrent();   // the mini-player's Close player and the lock screen's stop
    await settle(env);
    assert.equal(el.paused, true);
    assert.equal(engine.state.playing, false);
    assert.equal(engine.state.status.text, '');
    assert.equal(app.get().player, null);
    assert.equal(app.get().resumable, null);
    assert.equal(env.mediaSession.playbackState, 'none');
    assert.deepEqual({ block: server.saved.at(-1).block_index, offset: server.saved.at(-1).offset_ms }, { block: 2, offset: 2000 });
    engine.play();
    await settle(env);
    const again = playingEl();
    assert.equal(Number(again.dataset.seek ?? again.currentTime), 2, 'play resumes where it stopped');
    assert.equal(app.get().player.id, 'reader:doc1', 'and re-claims the session');
    const other = makeDoc({ id: 'doc2', title: 'Notes for Tuesday', chapters: [['Notes', 2]] });
    engine.load(other);
    await settle(env);
    assert.equal(engine.state.docId, 'doc2');
    assert.equal(engine.state.playing, false);
    assert.equal(app.get().player, null, 'opening another document released the old session');
  });

  it('[RE-32] the saved offset applies only to the saved paragraph read with the same voice, never negative, and only once', async () => {
    let el = await resumed();
    assert.equal(el.dataset.seek, '2');
    loadMeta(ctx, el);
    await settle(ctx.env);
    ctx.engine.pause();
    ctx.engine.jumpTo(2);
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal(playingEl().currentTime, 0, 'used once: the next start of that paragraph begins at 0');
    ctx.env.restore();
    el = await resumed({ offset_ms: -500 });
    assert.equal(el.dataset.seek, undefined, 'never negative');
    ctx.env.restore();
    // Another voice (the saved model is not available here): the offset belongs to other audio.
    el = await resumed({ model: 'chatterbox', voice: 'Ryan' });
    assert.equal(el.dataset.seek, undefined);
    assert.equal(ctx.server.renders[0].voice, 'af_heart');
    ctx.env.restore();
    ctx = await setup({ doc: makeDoc({ progress: saved({}) }) });
    ctx.engine.load(ctx.doc);
    ctx.engine.playFrom(3);
    await settle(ctx.env);
    assert.equal(playingEl().dataset.seek, undefined, 'another paragraph starts at its beginning');
  });

  it('[RE-33] Back to library saves and keeps playing; opening a document and leaving without playing sends nothing', async () => {
    ctx = await setup({ doc: makeDoc({ progress: saved({}) }) });
    const { engine, env, server } = ctx;
    engine.load(ctx.doc);
    await engine.saveProgress();
    await engine.unload();
    assert.equal(server.saved.length, 0, 'the saved position is left exactly as it was');
    engine.play();
    await settle(env);
    const el = playingEl();
    loadMeta(ctx, el);
    el.__timeupdate(1.2);          // the listener went back to the first sentence
    await engine.saveProgress();   // what the Reader view's Back to library does
    assert.equal(server.saved.at(-1).offset_ms, 0);
    assert.equal(engine.state.playing, true);
    assert.equal(el.paused, false);
  });

  it('[RE-34] closing the player while a resume is still loading keeps the saved offset', async () => {
    const el = await resumed();
    assert.equal(el.readyState, 0);
    await ctx.engine.unload();
    const last = ctx.server.saved.at(-1);
    assert.ok(!last || last.offset_ms === 2000, 'nothing new, or the same offset');
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal(playingEl().dataset.seek, '2');
  });

  it('[RE-35] the payload is {block_index, offset_ms, model, voice, speed}: the start of the sentence when known, else the position in the block', async () => {
    ctx = await setup({ prefs: { snapToSentence: false } });
    const { engine, env, server } = ctx;
    const el = await startPlaying(ctx, { block: 3 });
    engine.setSpeed(1.25);
    el.__timeupdate(3.4);
    await engine.saveProgress();
    assert.deepEqual(server.saved.at(-1), { block_index: 3, offset_ms: 2000, model: 'kokoro', voice: 'af_heart', speed: 1.25, keepalive: false });
    ctx.env.restore();
    // A part without sentence marks: the position in the block.
    ctx = await setup();
    ctx.server.override.set(6, { last_block: 6, sentences: [], duration_ms: 9000 });
    const el2 = await startPlaying(ctx, { block: 6 });
    el2.__timeupdate(1.25);
    await ctx.engine.saveProgress();
    const last = ctx.server.saved.at(-1);
    assert.equal(last.block_index, 6);
    assert.equal(last.offset_ms, 1250);
  });

  it('[RE-36] saves PUT the progress, skip an identical save, use keepalive when the page hides, retry after a retryable failure and report the answer', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    const progress = collect(engine, 'progress');
    const el = await startPlaying(ctx, { block: 2 });
    await engine.saveProgress();
    const n = server.saved.length;
    await engine.saveProgress();
    assert.equal(server.saved.length, n, 'identical: skipped');
    assert.equal(progress.at(-1).docId, 'doc1');
    assert.equal(progress.at(-1).progress.block_index, 2);
    el.__timeupdate(2.2);
    server.saved.length = 0;
    env.fetch.route('PUT /v1/reader/documents/:id/progress', () => networkError());
    await engine.saveProgress();
    env.fetch.route('PUT /v1/reader/documents/:id/progress', (req) => { server.saved.push({ ...req.json, keepalive: req.keepalive }); return json(200, { progress: req.json }); });
    await engine.saveProgress();
    assert.equal(server.saved.length, 1, 'the failed save is retried by the next one');
    env.fetch.route('PUT /v1/reader/documents/:id/progress', () => json(404, { error: { message: 'gone' } }));
    el.__timeupdate(0.3);
    await engine.saveProgress();
    env.fetch.route('PUT /v1/reader/documents/:id/progress', (req) => { server.saved.push({ ...req.json, keepalive: req.keepalive }); return json(200, { progress: req.json }); });
    await engine.saveProgress();
    assert.equal(server.saved.length, 1, 'a final failure is not repeated');
    el.__timeupdate(2.4);
    env.setHidden(true);
    await settle(env);
    assert.equal(server.saved.at(-1).keepalive, true);
  });

  it('[RE-37] saves come every 10 s of playing (driven by timeupdate, so a locked phone saves too), on part start, pause, a system pause, a jump while paused, hide and pagehide', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    // Block 1 is a 30 s part with a sentence every 2 s, so the position keeps changing.
    const marks = Array.from({ length: 15 }, (_, i) => ({ block: 1, index: i, start_ms: i * 2000 }));
    server.override.set(1, { duration_ms: 30000, sentences: marks });
    const el = await startPlaying(ctx, { block: 1 });
    assert.ok(server.saved.length >= 1, 'the part start saved');
    el.__timeupdate(2.5);
    env.setHidden(true);   // locked: no animation frames, and a keepalive save
    await settle(env);
    assert.equal(server.saved.at(-1).keepalive, true);
    const hidden = server.saved.length;
    for (let i = 11; i <= 120; i += 1) {   // 4 Hz timeupdates for ~27 s
      await env.clock.tickAsync(250);
      el.__timeupdate(i / 4);
    }
    const periodic = server.saved.length - hidden;
    assert.ok(periodic >= 2 && periodic <= 3, `about one save per 10 s (got ${periodic})`);
    env.setHidden(false);
    engine.pause();
    await settle(env);
    assert.equal(server.saved.at(-1).keepalive, false);
    const paused = server.saved.length;
    engine.jumpTo(6);
    await settle(env);
    assert.equal(server.saved.length, paused + 1, 'a jump while paused saves');
    assert.equal(server.saved.at(-1).block_index, 6);
    engine.play();
    await settle(env);
    const playing = playingEl();
    loadMeta(ctx, playing);
    playing.__timeupdate(2.1);
    const before = server.saved.length;
    playing.__systemPause();
    await settle(env);
    assert.equal(server.saved.length, before + 1, 'a system pause saves');
    engine.play();
    await settle(env);
    playing.__timeupdate(0.5);
    window.dispatchEvent(new Event('pagehide'));
    await settle(env);
    assert.equal(server.saved.at(-1).keepalive, true, 'pagehide saves with keepalive');
  });

  it('[RE-38] finishing: "Finished", the cursor on the last block, saved past it (100% in the library), and the lock screen cleared', async () => {
    ctx = await setup();
    const { engine, env, server, app } = ctx;
    const finished = collect(engine, 'finished');
    const count = ctx.doc.blocks.length;
    const el = await startPlaying(ctx, { block: count - 1 });
    el.__end();
    await settle(env);
    assert.equal(engine.state.status.text, 'Finished');
    assert.equal(engine.state.cursor, count - 1);
    assert.equal(engine.state.finished, true);
    assert.equal(server.saved.at(-1).block_index, count);
    assert.equal(env.mediaSession.playbackState, 'none');
    assert.equal(app.get().player, null);
    assert.deepEqual(finished, [{ docId: 'doc1' }]);
  });

  it('[RE-39] a finished document reopens finished (last block, 100%) until played; Play replays the last paragraph and then saves block count − 1', async () => {
    const doc = makeDoc();
    const count = doc.blocks.length;
    doc.progress = saved({ block_index: count, offset_ms: 0 });
    ctx = await setup({ doc });
    const { engine, env, server } = ctx;
    engine.load(doc);
    assert.equal(engine.state.finished, true);
    assert.equal(engine.state.cursor, count - 1);
    await engine.saveProgress();
    assert.equal(server.saved.length, 0, 'still saved as finished');
    engine.play();
    await settle(env);
    assert.equal(playingEl().dataset.part, `p${count - 1}`);
    assert.equal(engine.state.finished, false);
    assert.equal(server.saved.at(-1).block_index, count - 1);
  });

  it('[RE-40] tapping a paragraph or changing the voice on a finished document leaves 100% in the view and in what is saved', async () => {
    const doc = makeDoc();
    const count = doc.blocks.length;
    doc.progress = saved({ block_index: count, offset_ms: 0 });
    ctx = await setup({ doc });
    ctx.engine.load(doc);
    ctx.engine.jumpTo(4);
    await settle(ctx.env);
    assert.equal(ctx.engine.state.finished, false);
    assert.equal(ctx.server.saved.at(-1).block_index, 4);
    ctx.env.restore();
    const doc2 = makeDoc();
    doc2.progress = saved({ block_index: count, offset_ms: 0 });
    ctx = await setup({ doc: doc2 });
    ctx.engine.load(doc2);
    ctx.engine.setVoice({ model: 'kokoro', voice: 'bf_emma' });
    await settle(ctx.env);
    assert.equal(ctx.engine.state.finished, false);
    assert.equal(ctx.server.saved.at(-1).block_index, count - 1);
    assert.equal(ctx.server.saved.at(-1).voice, 'bf_emma');
  });
});
