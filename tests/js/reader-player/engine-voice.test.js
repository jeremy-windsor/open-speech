import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, makeDoc, collect, load, json, MODELS } from './helpers.js';

let ctx = null;
let speakVoiceModule = null;
afterEach(() => {
  if (speakVoiceModule) speakVoiceModule.__resetForTests();   // stop the boot publisher
  ctx?.env.restore();
  ctx = null;
});

/** Start Speak's boot publisher, as main.js does once the first view mounted. */
async function startBootVoice() {
  speakVoiceModule = await load('views/speak/speak-voice.js');
  speakVoiceModule.__resetForTests();
  return speakVoiceModule.startBootVoice();
}

const progress = (voice) => ({ block_index: 0, offset_ms: 0, speed: 1, updated_at: 'x', ...voice });
const STORED = { 'open-speech-reader-voice': JSON.stringify({ model: 'kokoro', voice: 'bf_emma' }) };
const CHATTERBOX = { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded' };
// Speak's last setup was a clone model reading with a raw reference recording: Speak publishes it as kind
// 'recording', which the Reader can't use (REV-09).
const SPEAK_RECORDING = { 'open-speech-speak': JSON.stringify({ model: 'chatterbox', voice: 'demo_reference', reference: 'demo_reference', identity: '' }) };

async function voiceUsed(options) {
  ctx = await setup(options);
  if (options.boot) await startBootVoice();
  await settle(ctx.env);
  ctx.engine.load(ctx.doc);
  ctx.engine.play();
  await settle(ctx.env, options.wait || 0);
  return ctx.server.renders[0] ? ctx.server.renders[0].voice : null;
}

describe('reader-player/engine voice', () => {
  it("[RE-04] the voice is the document's saved one (when its model is available), else the last Reader voice, else Speak's", async () => {
    assert.equal(await voiceUsed({ doc: makeDoc({ progress: progress({ model: 'kokoro', voice: 'am_adam' }) }), seed: STORED }), 'am_adam');
    ctx.env.restore();
    assert.equal(await voiceUsed({ doc: makeDoc({ progress: progress({ model: 'chatterbox', voice: 'Ryan' }) }), seed: STORED }), 'bf_emma', 'the saved model is not available here');
    ctx.env.restore();
    ctx = await setup({ seed: {} });
    ctx.app.set({ speakVoice: { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Bella + Sky' } });
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal(ctx.server.renders[0].voice, 'af_bella(2)+af_sky(1)', "Speak's voice, blends included (app.speakVoice)");
    assert.equal(ctx.engine.state.voiceLabel, 'Bella + Sky');
  });

  it("[RE-04] a page that opens straight into the Reader (before Speak's view has run: a first visit, a deep link, a reload) plays with the voice Speak publishes at boot, its saved setup else Kokoro's Heart; the Reader adds no request on the way", async () => {
    ctx = await setup({ seed: {}, modelsKnown: false });
    const { engine, env, app, server } = ctx;
    assert.deepEqual(app.get().speakVoice, { model: '', voice: '', label: '' }, 'the real state before Speak runs');
    await startBootVoice();
    engine.load(ctx.doc);
    app.set({ models: { list: MODELS, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
    assert.deepEqual(app.get().speakVoice, { model: 'kokoro', voice: 'af_heart', label: 'Heart', kind: 'voice' }, 'published once the model list is known');
    assert.equal(engine.state.voiceLabel, 'Heart', 'the chip names what Play will use');
    const before = env.fetch.calls.length;
    engine.play();
    await settle(env);
    assert.deepEqual(server.renders[0], { from_block: 0, model: 'kokoro', voice: 'af_heart' });
    assert.ok(playingEl(), 'it plays: no "Choose a voice" prompt');
    assert.deepEqual(env.fetch.calls.slice(before).map((c) => `${c.method} ${c.path}`).slice(0, 2),
      ['POST /v1/reader/documents/doc1/render', 'PUT /v1/reader/documents/doc1/progress'], 'nothing before the render (Kokoro is loaded: the fast path)');
    assert.equal(env.fetch.callsTo('GET /v1/audio/voices').length, 0);
    assert.equal(server.saved.at(-1).voice, 'af_heart', 'the voice it played with is saved with the position');
    env.restore();
    // Speak's saved setup: a blend, or a named voice (sent as voice:<id>).
    for (const [setup_, voice] of [[{ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)' }, 'af_bella(2)+af_sky(1)'], [{ model: 'kokoro', voice: 'af_bella', identity: 'n1' }, 'voice:n1']]) {
      assert.equal(await voiceUsed({ seed: { 'open-speech-speak': JSON.stringify(setup_) }, boot: true }), voice);
      ctx.env.restore();
    }
    // Another provider: Speak selects its first voice, so its publisher asks for the list once; the Reader never does.
    const piperOnly = [{ id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'loaded' }];
    assert.equal(await voiceUsed({ seed: {}, models: piperOnly, boot: true, routes: { 'GET /v1/audio/voices': () => json(200, { voices: [{ id: 'en_US-amy-medium' }] }) } }), 'en_US-amy-medium');
    assert.deepEqual(ctx.env.fetch.callsTo('GET /v1/audio/voices').map((c) => c.query.get('model')), ['piper/en_US-amy-medium'], "one request: Speak's publisher's");
    ctx.env.restore();
    // Once Speak's view runs, its live voice replaces the boot one, and that is what Play reads with.
    ctx = await setup({ seed: {} });
    await startBootVoice();
    ctx.app.set({ speakVoice: { model: 'kokoro', voice: 'am_adam', label: 'Adam', kind: 'voice' } });
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal(ctx.server.renders[0].voice, 'am_adam');
  });

  it("[RE-04][RE-63] with no voice of its own, the Reader follows Speak's voice as it changes until it plays; that playback keeps it, so every part renders with one voice", async () => {
    ctx = await setup({ seed: {} });
    const { engine, env, app, server } = ctx;
    await startBootVoice();
    engine.load(ctx.doc);
    assert.equal(engine.state.voiceLabel, 'Heart', "Speak's voice at boot");
    app.set({ speakVoice: { model: 'kokoro', voice: 'am_adam', label: 'Adam', kind: 'voice' } });
    assert.equal(engine.state.voiceLabel, 'Adam', 'the chip follows Speak (its view ran and changed the voice)');
    engine.jumpTo(2);
    await settle(env);
    assert.equal(server.saved.at(-1).voice, 'am_adam', 'a position saved before playing records the voice in effect');
    engine.play();
    await settle(env);
    assert.equal(server.renders.at(-1).voice, 'am_adam');
    app.set({ speakVoice: { model: 'kokoro', voice: 'bf_emma', label: 'Emma', kind: 'voice' } });
    assert.equal(engine.state.voiceLabel, 'Adam', 'a playback keeps its voice');
    const el = playingEl();
    loadMeta(ctx, el);
    el.__end();
    await settle(env);
    assert.ok(server.renders.every((body) => body.voice === 'am_adam'), 'every render of this playback');
    assert.ok(server.polls.every((poll) => poll.voice === 'am_adam'), 'and every poll');
    env.restore();
    // Before any playback, choosing the voice Speak gives (the one in effect) changes nothing: the resume stays.
    ctx = await setup({ seed: {}, doc: makeDoc({ progress: { block_index: 3, offset_ms: 1500, model: null, voice: null, speed: 1 } }) });
    ctx.app.set({ speakVoice: { model: 'kokoro', voice: 'am_adam', label: 'Adam', kind: 'voice' } });
    ctx.engine.load(ctx.doc);
    assert.equal(ctx.engine.setVoice({ model: 'kokoro', voice: 'am_adam' }), false, 'the voice in effect: no change');
    assert.equal(ctx.engine.setVoice({ model: 'kokoro', voice: 'bf_emma' }), true);
  });

  it('[RE-04] after a deep link the voice waits up to 5 s for the model list, then trusts the saved voice', async () => {
    ctx = await setup({ doc: makeDoc({ progress: progress({ model: 'kokoro', voice: 'am_adam' }) }), seed: STORED, modelsKnown: false });
    const { engine, env, app, server } = ctx;
    engine.load(ctx.doc);
    engine.play();
    await settle(env, 1500);
    assert.equal(server.renders.length, 0, 'undecided while the models load');
    app.set({ models: { list: MODELS, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
    await settle(env, 200);
    assert.equal(server.renders[0].voice, 'am_adam');
    ctx.env.restore();
    ctx = await setup({ doc: makeDoc({ progress: progress({ model: 'kokoro', voice: 'am_adam' }) }), seed: STORED, modelsKnown: false });
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(ctx.env, 4900);
    assert.equal(ctx.server.renders.length, 0);
    await settle(ctx.env, 300);
    assert.equal(ctx.server.renders[0].voice, 'am_adam', 'no model list after 5 s: the saved voice is used');
  });

  it('[RE-05] no usable voice asks for one ("Choose a voice for the Reader first", the voice sheet opens, focus goes back to Play); no model at all says "No speech model is ready yet." with [Open Models]', async () => {
    // Nothing saved, and Speak reads with a raw recording (REV-09): no voice the Reader can use.
    ctx = await setup({ seed: SPEAK_RECORDING, models: [...MODELS, CHATTERBOX] });
    const { engine, env, app } = ctx;
    await startBootVoice();
    assert.equal(app.get().speakVoice.kind, 'recording', 'what Speak publishes for it');
    const needed = collect(engine, 'voice-needed');
    engine.load(ctx.doc);
    assert.equal(engine.state.voiceLabel, '', 'the chip says "Voice"');
    engine.play();
    await settle(env);
    assert.deepEqual(needed, [{ reason: 'no-voice', message: 'Choose a voice for the Reader first' }]);
    assert.equal(engine.state.playing, false);
    assert.equal(app.get().player, null, 'the session is let go');
    assert.equal(ctx.server.renders.length, 0, 'the recording name is never sent as a voice');
    ctx.env.restore();
    // A clone model with no voices of its own, and nothing else: Speak publishes nothing, and the Reader asks.
    ctx = await setup({ seed: {}, models: [CHATTERBOX], routes: { 'GET /v1/audio/voices': () => json(200, { voices: [] }) } });
    await startBootVoice();
    await settle(ctx.env);
    const none = collect(ctx.engine, 'voice-needed');
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(ctx.env);
    assert.deepEqual(none.map((d) => d.reason), ['no-voice']);
    assert.equal(ctx.env.fetch.callsTo('GET /v1/audio/voices').length, 1, "only Speak's publisher asked for the list");
    assert.equal(ctx.server.renders.length, 0);
    ctx.env.restore();
    // With the player on screen, the voice sheet opens with the note, and closing it returns focus to Play.
    ctx = await setup({ seed: SPEAK_RECORDING, models: [...MODELS, CHATTERBOX] });
    await startBootVoice();
    const { mountPlayer } = await load('views/reader/player.js');
    const router = await load('core/router.js');
    const player = mountPlayer(document.getElementById('rd-player-slot'), ctx.engine, {});
    await player.ready;
    ctx.engine.load(ctx.doc);
    await router.navigate('reader/doc1');
    const play = document.getElementById('rdp-play');
    play.focus();
    ctx.env.click(play);
    await settle(ctx.env);
    const sheet = document.getElementById('rdp-voice-sheet');
    assert.equal(sheet.open, true);
    assert.equal(document.getElementById('rdp-voice-note').hidden, false);
    assert.equal(document.getElementById('rdp-voice-note-msg').textContent, 'Choose a voice for the Reader first');
    ctx.env.key('Escape');
    await settle(ctx.env);
    assert.equal(sheet.open, false);
    assert.equal(document.activeElement.id, 'rdp-play', 'focus returns to the control that was used (WCAG 2.4.3)');
    player.destroy();
    ctx.env.restore();
    // No TTS model at all: an error toast with [Open Models] (the message doesn't repeat it), and the sheet's banner.
    ctx = await setup({ seed: {}, models: [] });
    const p2 = (await load('views/reader/player.js')).mountPlayer(document.getElementById('rd-player-slot'), ctx.engine, {});
    await p2.ready;
    ctx.engine.load(ctx.doc);
    await (await load('core/router.js')).navigate('reader/doc1');
    ctx.engine.play();
    await settle(ctx.env);
    const toastEl = document.querySelector('#app-toasts-alert .toast--error');
    const buttons = [...toastEl.querySelectorAll('button')].map((b) => b.textContent);
    assert.match(toastEl.textContent, /No speech model is ready yet\./);
    assert.equal(toastEl.textContent.split('Open Models').length - 1, 1, '"Open Models" appears once: the button');
    assert.ok(buttons.includes('Open Models'));
    assert.equal(document.getElementById('rdp-voice-banner').hidden, false);
    p2.destroy();
    ctx.env.restore();
    // Without the player (another view, before it mounted), the engine's own toast has the same action.
    ctx = await setup({ seed: {}, models: [] });
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(ctx.env);
    const alone = document.querySelector('#app-toasts-alert .toast--error');
    assert.match(alone.textContent, /^No speech model is ready yet\./);
    const open = [...alone.querySelectorAll('button')].find((b) => b.textContent === 'Open Models');
    ctx.env.click(open);
    await settle(ctx.env);
    assert.equal(ctx.app.get().route.path, 'models/tts');
  });

  it('[RE-66] a voice needs a model and a voice, is remembered for the Reader, and a change drops the resume and re-renders the paragraph when playing (else halts)', async () => {
    ctx = await setup();
    const { engine, env, server, mod } = ctx;
    assert.throws(() => engine.setVoice({ model: 'kokoro', voice: '' }), /Choose a model and a voice/);
    const voices = collect(engine, 'voice');
    const el = await startPlaying(ctx, { block: 3 });
    el.__timeupdate(2.5);
    const statuses = collect(engine, 'status');
    assert.equal(engine.setVoice({ model: 'kokoro', voice: 'bf_emma' }), true);
    assert.equal(localStorage.getItem('open-speech-reader-voice'), JSON.stringify({ model: 'kokoro', voice: 'bf_emma' }));
    assert.deepEqual(voices.at(-1), { model: 'kokoro', voice: 'bf_emma' });
    assert.equal(statuses[0]?.text, 'Voice changed. Re-rendering from here.');
    await settle(env);
    assert.deepEqual(server.renders.at(-1), { from_block: 3, model: 'kokoro', voice: 'bf_emma' });
    assert.equal(playingEl().dataset.seek, undefined, 'the paragraph restarts from its beginning');
    assert.equal(engine.state.status.text, '');
    assert.equal(engine.setVoice({ model: 'kokoro', voice: 'bf_emma' }), false, 'the same voice changes nothing');
    engine.pause();
    await settle(env);
    engine.setVoice({ model: 'kokoro', voice: 'am_adam' });
    await settle(env);
    assert.equal(playingEl(), null);
    assert.equal(server.saved.at(-1).voice, 'am_adam');
    assert.ok(mod);
  });

  it('[RE-66] the voice sheet will not apply without a voice: "Choose a model and a voice"', async () => {
    ctx = await setup({ seed: SPEAK_RECORDING, models: [...MODELS, CHATTERBOX] });
    await startBootVoice();   // Speak reads with a recording: no voice for the Reader
    const { mountPlayer } = await load('views/reader/player.js');
    const player = mountPlayer(document.getElementById('rd-player-slot'), ctx.engine, {});
    await player.ready;
    ctx.engine.load(ctx.doc);
    player.openSheet('voice');
    await settle(ctx.env);
    ctx.env.click(document.getElementById('rdp-voice-use'));
    const error = document.getElementById('rdp-voice-error');
    assert.equal(error.hidden, false);
    assert.equal(document.getElementById('rdp-voice-error-text').textContent, 'Choose a model and a voice');
    assert.equal(document.activeElement.id, 'rdp-voice-choose');
    assert.equal(document.getElementById('rdp-voice-sheet').open, true);
    player.destroy();
  });
});
