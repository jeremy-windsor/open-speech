import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, makeDoc, json, delay, networkError, load, MODELS } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

const calls = (env) => env.fetch.calls.map((c) => `${c.method} ${c.path}`);
const PIPER = 'piper/en_US-amy-medium';

describe('reader-player/engine start', () => {
  it('[RE-03] a start claims the app audio session at once: another player pauses, Live stops, the lock screen says playing before the first part renders, and a newer start makes the older one stale', async () => {
    ctx = await setup({ routes: { 'POST /v1/reader/documents/:id/render': (req) => delay(req.json.from_block === 0 ? 3000 : 10, () => json(200, { parts: ctx.server.partsFrom(req.json.from_block) })) } });
    const { engine, env, audio, app } = ctx;
    const take = await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:take' });
    assert.ok(take);
    const shared = audio.element();
    assert.equal(shared.paused, false);
    engine.load(ctx.doc);
    engine.play();
    assert.equal(shared.paused, true, 'the Speak take paused when the Reader started');
    assert.equal(app.get().player.id, 'reader:doc1');
    assert.equal(app.get().player.kind, 'long');
    assert.equal(engine.state.playing, true);
    assert.equal(engine.state.waiting, true);
    assert.equal(env.mediaSession.playbackState, 'playing', 'lock-screen controls work while the first part renders');
    for (const action of ['play', 'pause', 'stop', 'seekbackward', 'seekforward', 'seekto', 'previoustrack', 'nexttrack']) {
      assert.equal(typeof env.mediaSession.handlers[action], 'function', action);
    }
    await settle(env, 100);
    engine.playFrom(5);   // a newer start while the first render is still in flight
    await settle(env, 4000);
    assert.equal(engine.state.cursor, 5, 'the stale first start never moved the cursor back');
    assert.equal(playingEl().dataset.part, 'p5');
    // Live is stopped (not paused) when the Reader starts: its render slot would stall the Reader.
    engine.pause();
    const stopped = [];
    audio.claim({ id: 'live', owner: 'live', kind: 'live', title: 'Live', controls: { play() {}, pause() {}, toggle() {}, stop: (why) => stopped.push(why) }, getState: () => ({ playing: true }) });
    engine.play();
    await settle(env, 50);
    assert.deepEqual(stopped, ['Another player started']);
    assert.equal(app.get().player.id, 'reader:doc1');
  });

  it('[RE-06] the voice model loads before rendering as a status job ("Loading Piper…", "Unloads Kokoro") without a confirm; a failed load stops with its reason; no status answer carries on; a loaded model costs no request', async () => {
    ctx = await setup({
      seed: { 'open-speech-reader-voice': JSON.stringify({ model: PIPER, voice: 'en_US-amy-medium' }) },
      routes: { 'POST /api/models/:id/load': () => delay(3000, () => { ctx.server.status = 'loaded'; return json(200, { status: 'loaded' }); }) },
    });
    const { engine, env, app, server } = ctx;
    server.status = 'downloaded';
    engine.load(ctx.doc);
    engine.play();
    await settle(env, 100);
    assert.equal(engine.state.status.text, 'Loading Piper…');
    assert.equal(engine.state.status.kind, 'preparing');
    const job = app.get().jobs.find((j) => j.key === `model:${PIPER}`);
    assert.equal(job.label, 'Loading Piper…');
    assert.equal(job.detail, 'Unloads Kokoro');
    const dialogs = await load('core/dialogs.js');
    assert.equal(dialogs.isModalOpen(), false, 'no "Switch speech model?" confirm for the Reader');
    assert.equal(server.renders.length, 0, 'nothing renders before the model is loaded');
    await settle(env, 3500);
    const order = calls(env).filter((c) => /status|load|render/.test(c));
    const id = encodeURIComponent(PIPER);
    assert.deepEqual(order.slice(0, 4), [`GET /api/models/${id}/status`, `POST /api/models/${id}/load`, `GET /api/models/${id}/status`, 'POST /v1/reader/documents/doc1/render']);
    assert.ok(playingEl());
    assert.equal(app.get().jobs.length, 0, 'the job ended');
  });

  it('[RE-06] a load failure stops playback with the server reason; an unreachable status check renders anyway', async () => {
    ctx = await setup({
      seed: { 'open-speech-reader-voice': JSON.stringify({ model: PIPER, voice: 'en_US-amy-medium' }) },
      routes: { 'POST /api/models/:id/load': () => json(500, { error: { message: 'CUDA out of memory' } }) },
    });
    const { engine, env, server } = ctx;
    server.status = 'downloaded';
    engine.load(ctx.doc);
    engine.play();
    await settle(env, 500);
    assert.equal(engine.state.status.text, "Couldn't play: CUDA out of memory");
    assert.equal(engine.state.status.kind, 'error');
    assert.equal(engine.state.playing, false);
    assert.equal(server.renders.length, 0);
    ctx.env.restore();
    ctx = await setup({
      seed: { 'open-speech-reader-voice': JSON.stringify({ model: PIPER, voice: 'en_US-amy-medium' }) },
      routes: { 'GET /api/models/:id/status': () => networkError() },
    });
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(ctx.env, 500);
    assert.equal(ctx.server.renders.length, 1, 'no status answer: the render reports any real problem');
    assert.ok(playingEl());
  });

  it('[RE-06] with the model known to be loaded (the fast path) the first request is the render', async () => {
    ctx = await setup();
    ctx.engine.load(ctx.doc);
    ctx.env.fetch.reset();
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal(calls(ctx.env)[0], 'POST /v1/reader/documents/doc1/render');
    assert.ok(!calls(ctx.env).some((c) => c.includes('/status')), 'no status request before audio');
  });

  it('[RE-13] an autoplay refusal stops waiting and says "Tap play to start listening"', async () => {
    ctx = await setup({ playPolicy: 'NotAllowedError' });
    const { engine, env } = ctx;
    engine.load(ctx.doc);
    engine.play();
    await settle(env, 50);
    assert.equal(engine.state.status.text, 'Tap play to start listening');
    assert.equal(engine.state.waiting, false);
    assert.equal(engine.state.playing, false);
    assert.equal(engine.state.busy, false);
  });

  it('[RE-28] waiting feedback: "Preparing audio…" after 1 s, "Still preparing…" after 20 s, a busy play button; a retry line is not overwritten and settles back; nothing ticks', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.pending.set(0, 1000);   // block 0 keeps rendering
    engine.load(ctx.doc);
    engine.play();
    await settle(env, 900);
    assert.equal(engine.state.status.text, '');
    assert.equal(engine.state.busy, false, 'no spinner flash for a fast start');
    await settle(env, 200);
    assert.equal(engine.state.status.text, 'Preparing audio…');
    assert.equal(engine.state.status.kind, 'preparing');
    assert.equal(engine.state.busy, true);
    server.partsHandler = () => networkError();
    await settle(env, 3000);
    assert.equal(engine.state.status.text, 'Reconnecting…');
    assert.equal(engine.state.status.kind, 'retry');
    await settle(env, 18000);
    assert.equal(engine.state.status.text, 'Reconnecting…', 'the 20 s line does not overwrite a retry');
    server.partsHandler = null;
    await settle(env, 5000);
    assert.equal(engine.state.status.text, 'Still preparing audio. The model may still be loading, or Live is busy.');
    assert.doesNotMatch(engine.state.status.text, /\d/, 'no elapsed counter in the status line');
    server.pending.set(0, 0);
    await settle(env, 1000);
    assert.equal(engine.state.status.text, '');
    assert.equal(engine.state.busy, false);
    assert.ok(playingEl());
    assert.ok(MODELS.length);
  });
});
