// Transcribe: the model select (TR-01, BUG-11) and the "No STT model" state.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountView, settle, STT, MODELS, lastSocket, begin, chooseFile, audioFile, verbose, json } from './helpers.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { delay } from '../harness/fakes.js';

let t;
afterEach(() => t?.env?.restore());

describe('transcribe/model-select: pure parts', () => {
  it('[TR-01] only faster-whisper STT models, sorted loaded > downloaded > available > missing, then id', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { sttOptions } = await importFresh('views/transcribe/model-select.js');
    const list = [
      { id: 'Systran/faster-whisper-tiny', type: 'stt', provider: 'faster-whisper', state: 'available' },
      { id: 'Systran/faster-whisper-medium', type: 'stt', provider: 'faster-whisper', state: 'provider_missing' },
      { id: 'Systran/faster-whisper-small', type: 'stt', provider: 'faster-whisper', state: 'downloaded' },
      { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'loaded', device: 'cpu' },
      { id: 'Systran/faster-whisper-large-v3', type: 'stt', provider: 'faster-whisper', state: 'downloaded' },
      { id: 'vosk-small', type: 'stt', provider: 'vosk', state: 'loaded' },
      { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded' },
    ];
    const options = sttOptions(list);
    assert.deepEqual(options.map((o) => o.id), [
      'Systran/faster-whisper-base', 'Systran/faster-whisper-large-v3', 'Systran/faster-whisper-small',
      'Systran/faster-whisper-tiny', 'Systran/faster-whisper-medium',
    ]);
    // The option text is the short name and the state in words (the old "<id> <state>").
    assert.equal(options[0].label, 'base — Loaded · CPU');
    assert.equal(options[1].label, 'large-v3 — Downloaded');
    assert.equal(options[3].label, 'tiny — Not downloaded');
    assert.equal(options[0].ready, true);
  });

  it('[TR-01] the configured default is selected until the listener picks another listed model', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { pickStt } = await importFresh('views/transcribe/model-select.js');
    const options = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
    assert.equal(pickStt(options, { configuredDefault: 'b' }), 'b');
    assert.equal(pickStt(options, { picked: 'c', configuredDefault: 'b' }), 'c');
    assert.equal(pickStt(options, { picked: 'gone', configuredDefault: 'b' }), 'b', 'a pick that vanished falls back');
    assert.equal(pickStt(options, { configuredDefault: 'missing' }), 'a', 'else the first');
    assert.equal(pickStt([], { configuredDefault: 'b' }), '');
  });

  it('[TR-01] the hint under the select says what starting will do for the chosen model', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { modelHint } = await importFresh('views/transcribe/model-select.js');
    assert.match(modelHint({ key: 'not-downloaded' }), /downloads when you start/);
    assert.match(modelHint({ key: 'downloaded' }), /loads when you start/);
    assert.equal(modelHint({ key: 'loaded' }), '');
    assert.equal(modelHint(undefined), '');
  });
});

describe('transcribe view: the model select', () => {
  it('[TR-01] the select lists the STT models with their state and selects default_stt_model', async () => {
    t = await mountView();
    const select = t.$('tr-model');
    const options = select.querySelectorAll('option');
    assert.deepEqual(options.map((o) => o.value), [STT, 'Systran/faster-whisper-small', 'Systran/faster-whisper-tiny']);
    assert.equal(select.value, STT);
    assert.equal(select.disabled, false);
    assert.equal(options[0].textContent, 'base — Loaded · CPU');
    assert.equal(t.$('tr-nomodel').hidden, true);
  });

  it('[TR-01] a model the listener picks is the one recordings and uploads use', async () => {
    t = await mountView();
    t.env.change(t.$('tr-model'), 'Systran/faster-whisper-small');
    await settle(t.env);
    assert.equal(t.ctl.state().modelId, 'Systran/faster-whisper-small');
    assert.match(t.$('tr-model-hint').textContent, /loads when you start/);
    assert.equal(t.$('tr-model-hint').hidden, false);
  });

  it('[BUG-11] the list follows app.models: a load or download done in Models shows here without a reload', async () => {
    t = await mountView();
    const select = t.$('tr-model');
    t.env.change(select, 'Systran/faster-whisper-small');
    const before = select.querySelectorAll('option');
    // Models › Load finished for "small": the inventory refresh lands in the store.
    const list = MODELS.map((m) => (m.id === 'Systran/faster-whisper-small' ? { ...m, state: 'loaded' } : m));
    t.app.set({ models: { ...t.app.get().models, list, fetchedAt: Date.now() } });
    await settle(t.env);
    const options = select.querySelectorAll('option');
    const small = options.find((o) => o.value === 'Systran/faster-whisper-small');
    assert.equal(small.textContent, 'small — Loaded · CPU');
    assert.equal(select.value, 'Systran/faster-whisper-small', 'the pick is kept');
    // An operation in flight (app.ops) shows too.
    t.app.set({ ops: { 'Systran/faster-whisper-tiny': { kind: 'download', startedAt: Date.now(), progress: 0.1 } } });
    await settle(t.env);
    const tiny = select.querySelectorAll('option').find((o) => o.value === 'Systran/faster-whisper-tiny');
    assert.equal(tiny.textContent, 'tiny — Downloading');
    // The set of models did not change, so the options were patched in place, not rebuilt (an open select
    // is never closed under the pointer).
    const after = select.querySelectorAll('option');
    assert.equal(after.length, before.length);
    after.forEach((option, i) => assert.equal(option, before[i]));
    // A new model appears: the list is rebuilt and the pick stays.
    t.app.set({ models: { ...t.app.get().models, list: [...list, { id: 'Systran/faster-whisper-medium', type: 'stt', provider: 'faster-whisper', state: 'downloaded' }] } });
    await settle(t.env);
    assert.equal(select.querySelectorAll('option').length, 4);
    assert.equal(select.value, 'Systran/faster-whisper-small');
  });

  it('[BUG-11] before the inventory arrives the select says it is loading; a failed inventory says so', async () => {
    t = await mountView({ models: [] });
    t.app.set({ models: { list: [], defaults: { tts: '', stt: '' }, fetchedAt: 0, error: '' } });
    await settle(t.env);
    const select = t.$('tr-model');
    assert.equal(select.disabled, true);
    assert.equal(select.querySelector('option').textContent, 'Loading models…');
    assert.equal(t.$('tr-nomodel').hidden, true, 'not "no model" while the list is unknown');
    t.app.set({ models: { list: [], defaults: { tts: '', stt: '' }, fetchedAt: 0, error: 'offline' } });
    await settle(t.env);
    assert.equal(select.querySelector('option').textContent, "Models didn't load");
  });

  it('[TR-01] no STT model: the empty state with Open Models; Record and the dropzone are disabled', async () => {
    t = await mountView({ models: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded' }] });
    assert.equal(t.$('tr-nomodel').hidden, false);
    assert.ok(t.$('tr-nomodel').classList.contains('empty'), 'the shared empty state');
    assert.equal(t.$('tr-nomodel').querySelector('h2').textContent, 'No transcription model is available');
    assert.equal(t.$('tr-record').disabled, true);
    assert.equal(t.$('tr-file').disabled, true);
    assert.ok(t.$('tr-drop').classList.contains('is-disabled'));
    assert.equal(t.$('tr-model').querySelector('option').textContent, 'No transcription model');
    assert.equal(t.$('tr-model-field').hidden, true, 'an empty select would only repeat the card');
    assert.equal(t.$('tr-rec-hint').textContent, 'Recording needs a transcription model.');
    assert.equal(t.$('tr-record').getAttribute('aria-describedby'), 'tr-rec-hint', 'the reason is part of the button description');
    // Open Models is a link to the STT models (middle-click and Back work like any link).
    const link = t.$('tr-nomodel').querySelector('a.btn');
    assert.equal(link.getAttribute('href'), '#models/stt');
    t.env.click(link);
    await settle(t.env);
    assert.equal(t.env.location.hash, '#models/stt');
  });

  it('[TR-01] while the model list is still loading, Record waits and says why (never "no model")', async () => {
    t = await mountView({ models: [] });
    t.app.set({ models: { list: [], defaults: { tts: '', stt: '' }, fetchedAt: 0, error: '' } });
    await settle(t.env);
    assert.equal(t.$('tr-record').disabled, true);
    assert.equal(t.$('tr-rec-hint').textContent, 'Waiting for the list of models.');
    assert.equal(t.$('tr-nomodel').hidden, true);
    t.app.set({ models: { list: MODELS, defaults: { tts: 'kokoro', stt: STT }, fetchedAt: Date.now(), error: '' } });
    await settle(t.env);
    assert.equal(t.$('tr-record').disabled, false);
    assert.equal(t.$('tr-rec-hint').textContent, 'Words appear as you speak.');
  });

  it('[BUG-11] the select follows the inventory but is off while a recording or a file runs; a refresh meanwhile leaves it off', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(3000, json(200, verbose())) } });
    const select = t.$('tr-model');
    assert.equal(select.disabled, false);
    // A recording uses the model it started with: another pick meanwhile would change nothing.
    t.env.click(t.$('tr-record'));
    assert.equal(select.disabled, true, 'off from the click on (starting)');
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    begin(ws);
    await settle(t.env);
    assert.equal(t.$('tr-record').getAttribute('aria-label'), 'Stop recording');
    assert.equal(select.disabled, true);
    const list = MODELS.map((m) => (m.id === 'Systran/faster-whisper-small' ? { ...m, state: 'loaded' } : m));
    t.app.set({ models: { ...t.app.get().models, list, fetchedAt: Date.now() } });
    await settle(t.env);
    const small = select.querySelectorAll('option').find((o) => o.value === 'Systran/faster-whisper-small');
    assert.equal(small.textContent, 'small — Loaded · CPU', 'it still follows the inventory');
    assert.equal(select.disabled, true, 'a refresh does not turn it back on');
    t.env.click(t.$('tr-record'));
    ws.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.equal(select.disabled, false);
    // The same while a file transcribes.
    chooseFile(t, audioFile());
    await settle(t.env);
    assert.equal(select.disabled, true);
    t.app.set({ ops: { 'Systran/faster-whisper-tiny': { kind: 'download', startedAt: Date.now(), progress: 0.1 } } });
    await settle(t.env);
    assert.equal(select.disabled, true);
    await settle(t.env, 3000);
    await settle(t.env);
    assert.equal(select.disabled, false);
    assert.equal(t.ctl.state().transcript.source, 'file');
  });

  it('[BUG-11] a model that appears later lifts the "no model" state at once', async () => {
    t = await mountView({ models: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded' }] });
    assert.equal(t.$('tr-nomodel').hidden, false);
    t.app.set({ models: { ...t.app.get().models, list: MODELS } });
    await settle(t.env);
    assert.equal(t.$('tr-nomodel').hidden, true);
    assert.equal(t.$('tr-model-field').hidden, false);
    assert.equal(t.$('tr-record').disabled, false);
    assert.equal(t.$('tr-file').disabled, false);
    assert.equal(t.$('tr-model').value, STT);
  });
});
