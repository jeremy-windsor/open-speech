// Speak's voice for the rest of the app (app.speakVoice): published at boot before Speak's view has run
// and while it runs, with what the voice is (`kind`). The Reader reads it for
// RE-04's last fallback and its "Same as Speak" row; a raw reference recording never reaches it (REV-09).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, resetAll } from '../harness/env.js';
import { voiceKind, bootVoice, blendLabel, savedSetup } from '../../../src/static/js/views/speak/speak-voice.js';
import { isRecordingChoice } from '../../../src/static/js/views/reader/play-logic.js';
import { mountSpeak, ready, settle, load, json, modelsPayload, routes, delay } from './helpers.js';

let t;
let env;
afterEach(() => {
  t?.env.restore();
  env?.restore();
  t = null;
  env = null;
});

const LIST = modelsPayload().models;

describe('views/speak/speak-voice (pure)', () => {
  it('[RE-04] voiceKind names what Speak sends: a named voice, a blend, a provider voice, or a reference recording alone', () => {
    assert.equal(voiceKind({ identity: 'n1', blend: true, voice: 'af_bella', reference: 'x' }), 'named');
    assert.equal(voiceKind({ blend: true, voice: 'af_bella' }), 'blend');
    assert.equal(voiceKind({ voice: 'af_bella', reference: 'demo_reference' }), 'voice', 'a clone model with a stock voice sends the voice');
    assert.equal(voiceKind({ reference: 'demo_reference' }), 'recording');
    assert.equal(voiceKind({}), '');
  });

  it('[RE-04] bootVoice: Speak\'s saved setup on the model it would pick, else that model\'s default voice', () => {
    const at = (saved, extra = {}) => bootVoice({ saved, list: LIST, configuredDefault: 'kokoro', ...extra });
    assert.deepEqual(at(null), { model: 'kokoro', voice: 'af_heart', label: 'Kokoro · Heart', kind: 'voice' }, "Kokoro's recommended voice");
    assert.deepEqual(at({ model: 'kokoro', voice: 'am_adam' }), { model: 'kokoro', voice: 'am_adam', label: 'Kokoro · Adam', kind: 'voice' });
    assert.deepEqual(at({ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)' }), { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Kokoro · Bella + Sky', kind: 'blend' });
    assert.deepEqual(at({ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', identity: 'n1' }), { model: 'kokoro', voice: 'voice:n1', label: 'Named voice', kind: 'named' });
    assert.deepEqual(at({ model: 'chatterbox', voice: 'demo_reference', reference: 'demo_reference' }),
      { model: 'chatterbox', voice: 'demo_reference', label: 'demo_reference', kind: 'recording' }, 'a reference alone is said to be one');
    assert.deepEqual(at({ model: 'chatterbox', voice: '', reference: 'demo_reference' }).kind, 'recording');
    // The saved setup's model is gone: Speak picks the configured default, and only model-independent parts move.
    assert.deepEqual(at({ model: 'xtts', voice: 'v' }), { model: 'kokoro', voice: 'af_heart', label: 'Kokoro · Heart', kind: 'voice' });
    // Another provider's default is the first voice of its list (one request, made by the publisher).
    assert.deepEqual(at({ model: 'piper/en_US-amy-medium', voice: '' }), { model: 'piper/en_US-amy-medium', voice: '', label: '', kind: '', needsList: true });
    assert.deepEqual(at(null, { list: LIST.filter((m) => m.provider === 'piper') }).needsList, true);
    // A worker that went offline is not a usable model (models.ttsModels), and no TTS model gives nothing.
    assert.deepEqual(at({ model: 'chatterbox', reference: 'demo_reference' }, { list: LIST.map((m) => (m.id === 'chatterbox' ? { ...m, provider_available: false } : m)) }).model, 'kokoro');
    assert.equal(at(null, { list: LIST.filter((m) => m.type === 'stt') }), null);
    assert.equal(at('not a setup').voice, 'af_heart');
    assert.equal(blendLabel('af_bella(1)+af_sky(1)+am_adam(1)'), 'Bella + Sky + 1');
    assert.equal(blendLabel('not a blend +'), '');
    assert.equal(savedSetup({ model: 3 }).model, undefined);
  });

  it('[REV-09] the Reader leaves out what Speak says is a recording, and keeps every other kind', () => {
    assert.equal(isRecordingChoice({ model: 'chatterbox', voice: 'demo_reference', kind: 'recording' }), true);
    for (const kind of ['voice', 'blend', 'named']) {
      assert.equal(isRecordingChoice({ model: 'chatterbox', voice: 'demo_reference', kind }, { model: 'chatterbox', reference: 'demo_reference' }), false, kind);
    }
  });
});

/** The fake browser without the Speak view: the shell's store with a model list, as main.js sees it. */
async function bootEnv({ seed = {}, extra = {}, models = modelsPayload() } = {}) {
  env = installBrowserEnv({ routes: routes(extra, { models }), seed });
  await resetAll();
  const voice = await load('views/speak/speak-voice.js');
  voice.__resetForTests();
  const { app } = await load('core/store.js');
  const setModels = (payload = models) => app.set({
    models: { list: payload.models, defaults: { tts: payload.default_tts_model || '', stt: '' }, fetchedAt: Date.now(), error: '' },
  });
  return { voice, app, setModels };
}

describe('views/speak boot voice (before Speak runs)', () => {
  it('[RE-04] main.js\'s boot publisher fills app.speakVoice once the model list is known, from the saved setup, with no request', async () => {
    const { voice, app, setModels } = await bootEnv({ seed: { 'open-speech-speak': JSON.stringify({ model: 'kokoro', voice: 'am_adam', speed: 1.2 }) } });
    const stop = voice.startBootVoice();
    assert.deepEqual(app.get().speakVoice, { model: '', voice: '', label: '' }, 'nothing before the model list');
    setModels();
    assert.deepEqual(app.get().speakVoice, { model: 'kokoro', voice: 'am_adam', label: 'Kokoro · Adam', kind: 'voice' });
    assert.deepEqual(env.fetch.calls.map((c) => c.path), [], 'no request on the way to first audio');
    // A later model list with the same answer writes nothing; a changed default model follows.
    let writes = 0;
    const off = app.select((s) => s.speakVoice, () => { writes += 1; });
    setModels();
    assert.equal(writes, 0);
    off();
    stop();
  });

  it('[RE-04] another provider\'s default is the first voice of its list: one background GET /v1/audio/voices, published when it answers', async () => {
    const piperOnly = { models: modelsPayload().models.filter((m) => m.provider === 'piper'), default_tts_model: '' };
    const { voice, app, setModels } = await bootEnv({ models: piperOnly });
    voice.startBootVoice();
    setModels(piperOnly);
    await settle(env);
    assert.deepEqual(env.fetch.calls.map((c) => `${c.method} ${c.path}`), ['GET /v1/audio/voices']);
    assert.equal(env.fetch.calls[0].query.get('model'), 'piper/en_US-amy-medium');
    assert.deepEqual(app.get().speakVoice, { model: 'piper/en_US-amy-medium', voice: 'en_US-amy-medium', label: 'Piper · en_US-amy-medium', kind: 'voice' });
    setModels(piperOnly);
    await settle(env);
    assert.equal(env.fetch.calls.length, 1, 'asked once');
  });

  it('[RE-04] a voice list that fails or never answers publishes nothing (the Reader asks for a voice then)', async () => {
    const piperOnly = { models: modelsPayload().models.filter((m) => m.provider === 'piper'), default_tts_model: '' };
    const { voice, app, setModels } = await bootEnv({ models: piperOnly, extra: { 'GET /v1/audio/voices': () => delay(60000, json(200, { voices: [{ id: 'late' }] })) } });
    voice.startBootVoice({ timeoutMs: 5000 });
    setModels(piperOnly);
    await settle(env, 5001);
    assert.equal(env.fetch.calls[0].signal.aborted, true, 'the request is bounded');
    assert.equal(app.get().speakVoice.model, '');
  });

  it('[RE-04] once Speak\'s view runs it owns app.speakVoice: the boot publisher stops, and an answer still out is dropped', async () => {
    const piperOnly = { models: modelsPayload().models.filter((m) => m.provider === 'piper'), default_tts_model: '' };
    const { voice, app, setModels } = await bootEnv({ models: piperOnly, extra: { 'GET /v1/audio/voices': () => delay(500, json(200, { voices: [{ id: 'first' }] })) } });
    voice.startBootVoice();
    setModels(piperOnly);
    await settle(env);
    voice.claim();
    await settle(env, 600);
    assert.equal(env.fetch.calls[0].signal.aborted, true);
    assert.equal(app.get().speakVoice.model, '', 'nothing published over Speak');
    setModels();
    assert.equal(app.get().speakVoice.model, '', 'later model lists change nothing');
    assert.equal(typeof voice.startBootVoice(), 'function');
    setModels();
    assert.equal(app.get().speakVoice.model, '', 'a start after Speak claimed the voice is a no-op');
  });
});

describe('views/speak publishes its voice (the view runs)', () => {
  it('[RE-04] Speak publishes its live setup with its kind: a provider voice, a blend, a named voice', async () => {
    t = await mountSpeak();
    await ready(t);
    assert.deepEqual(t.app.get().speakVoice, { model: 'kokoro', voice: 'af_heart', label: 'Kokoro · Heart', kind: 'voice' });
    t.parts.inspector.showBlend('af_bella(2)+af_sky(1)');
    await settle(t.env);
    assert.equal(t.app.get().speakVoice.kind, 'blend');
    assert.equal(t.app.get().speakVoice.voice, 'af_bella(2)+af_sky(1)');
  });

  it('[REV-09] a clone model with only a reference recording publishes it as kind "recording", so the Reader never renders with it', async () => {
    t = await mountSpeak({ seed: { 'open-speech-speak': JSON.stringify({ model: 'chatterbox', voice: 'demo_reference', reference: 'demo_reference' }) } });
    await ready(t);
    const published = t.app.get().speakVoice;
    assert.equal(published.model, 'chatterbox');
    assert.equal(published.voice, 'demo_reference', 'the string Speak sends');
    assert.equal(published.kind, 'recording');
    assert.equal(isRecordingChoice(published, null), true, 'the Reader knows without reading Speak\'s storage');
  });

  it('[RE-04] Speak\'s first setup replaces the boot guess, and the boot publisher stops for good', async () => {
    t = await mountSpeak({ seed: { 'open-speech-speak': JSON.stringify({ model: 'kokoro', voice: 'am_adam' }) } });
    const voice = await load('views/speak/speak-voice.js');
    voice.startBootVoice();   // main.js's call, after Speak claimed the voice: a no-op
    await ready(t);
    assert.deepEqual(t.app.get().speakVoice, { model: 'kokoro', voice: 'am_adam', label: 'Kokoro · Adam', kind: 'voice' });
    t.parts.inspector.showBlend('af_bella(1)+af_sky(1)');
    await settle(t.env);
    // A new model list whose boot answer would differ (no saved setup: Kokoro's default voice) changes nothing.
    localStorage.removeItem('open-speech-speak');
    t.app.set({ models: { ...t.app.get().models, fetchedAt: Date.now() } });
    assert.equal(t.app.get().speakVoice.kind, 'blend', 'no boot value over the view\'s');
    assert.equal(t.app.get().speakVoice.voice, 'af_bella(1)+af_sky(1)');
  });
});
