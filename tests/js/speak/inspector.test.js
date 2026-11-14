// The Speak inspector (inventory SP-03…SP-11, SP-28, SP-36, SP-38, LR-21, VL-16 (Speak half), VL-19; REV-31).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { modelGroups, modelHint, restoreLabel, showRestore, pickVoice, referenceOptions, voiceDisplay } from '../../../src/static/js/views/speak/inspector.js';
import { assertAccessibleNames } from '../harness/a11y.js';
import { mountSpeak, ready, type, settle, callsTo, json, delay, modelsPayload, CAPS, KOKORO_VOICES, RECORDINGS, NAMED as NAMED_FOR_TEST, load, answerConfirm } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

/** Settings › Developer shows Speak › Live; off, Speak has Generate only. */
const LIVE_ON = Object.freeze({ 'open-speech-prefs': JSON.stringify({ v: 1, showLive: true }) });

function fakeLive() {
  const stops = [];
  t.ctx.live = { isActive: () => true, stop: (reason) => stops.push(reason), destroy() {}, refresh() {} };
  return stops;
}

async function chooseModel(id) {
  t.env.change(t.$('sp-model'), id);
  await settle(t.env);
  await settle(t.env);
}

async function pickVoiceInPicker(label) {
  t.env.click(t.$('sp-voice'));
  await settle(t.env);
  const option = [...document.querySelectorAll('#vp-list [role=option]')].find((o) => o.querySelector('.option__label').textContent === label);
  assert.ok(option, `option ${label}`);
  t.env.click(option);
  await settle(t.env);
  await settle(t.env);
}

const pressed = (id) => t.$('sp-presets').querySelector(`[data-preset-id="${id}"]`).getAttribute('aria-pressed');

describe('views/speak/inspector (pure)', () => {
  it('[SP-03] one model select grouped by provider: Kokoro, then Piper, then the rest A–Z', () => {
    const groups = modelGroups(modelsPayload({ extra: [{ id: 'fish', type: 'tts', provider: 'fish', state: 'available' }, { id: 'xtts', type: 'tts', provider: 'xtts', state: 'provider_missing' }] }).models);
    assert.deepEqual(groups.map((g) => g.label), ['Kokoro', 'Piper', 'Chatterbox', 'Fish Audio S2']);
    assert.deepEqual(groups[1].options, [{ id: 'piper/en_US-amy-medium', text: 'Amy (US)' }], 'option text is the short name, never the state');
  });

  it('[SP-10] the model state pill reads Loaded · GPU, Loads when you generate · unloads Kokoro, or Downloads, then loads when you generate', () => {
    const list = modelsPayload().models;
    assert.deepEqual(modelHint(list[0], { list }), { tone: 'ok', text: 'Loaded · GPU' });
    assert.deepEqual(modelHint(list[1], { list }), { tone: 'idle', text: 'Loads when you generate · unloads Kokoro' });
    const remote = { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'available' };
    assert.deepEqual(modelHint(remote, { list: [remote] }), { tone: 'idle', text: 'Downloads, then loads when you generate' });
    assert.equal(modelHint({ ...list[1], state: 'loading' }, { list }).tone, 'busy');
    assert.equal(modelHint({ id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'provider_unavailable' }, { list }).text, 'Worker offline');
  });

  it('[SP-11] Restore shows when the configured default exists and another model is selected or it is not loaded', () => {
    const list = modelsPayload().models;
    assert.equal(restoreLabel('kokoro'), 'Restore Kokoro');
    assert.equal(restoreLabel('kokoro', 'piper/en_US-amy-medium'), 'Restore Kokoro', 'another model is selected');
    assert.equal(restoreLabel('kokoro', 'kokoro'), 'Load Kokoro now', 'the default is selected and only not loaded: "Restore" would read as a no-op');
    assert.equal(showRestore('kokoro', 'kokoro', list), false);
    assert.equal(showRestore('kokoro', 'piper/en_US-amy-medium', list), true);
    assert.equal(showRestore('kokoro', 'kokoro', modelsPayload({ kokoroState: 'downloaded' }).models), true);
    assert.equal(showRestore('', 'kokoro', list), false);
    assert.equal(showRestore('gone', 'kokoro', list), false, 'a default that is not usable never shows');
  });

  it('[SP-09] the provider voice is the preferred one if offered, else the current one, else the first', () => {
    assert.equal(pickVoice(['a', 'b', 'c'], { preferred: 'b', current: 'c' }), 'b');
    assert.equal(pickVoice(['a', 'b', 'c'], { preferred: 'x', current: 'c' }), 'c');
    assert.equal(pickVoice(['a', 'b', 'c'], { preferred: 'x', current: 'y' }), 'a');
    assert.equal(pickVoice([], { preferred: 'x' }), '', 'a clone model with no stock voices');
  });

  it('[SP-36] reference options say whether each recording has its transcript saved', () => {
    assert.deepEqual(referenceOptions(RECORDINGS), [
      { value: '', label: 'None' },
      { value: 'demo_reference', label: 'demo_reference · transcript saved' },
      { value: 'raw_take', label: 'raw_take · no transcript' },
    ]);
    assert.equal(voiceDisplay({ voice: 'af_heart', mix: [] }).name, 'Heart');
    assert.equal(voiceDisplay({ mix: [{ voice: 'af_bella', weight: 2 }, { voice: 'af_sky', weight: 1 }] }).name, 'Bella + Sky');
    assert.equal(voiceDisplay({ identity: 'n1', identityName: 'Narrator', mix: [] }).name, 'Narrator');
  });
});

describe('views/speak inspector (view)', () => {
  it('[SP-04] a model change stops Live ("Voice changed"), clears the named voice and preset, and reloads capabilities and voices', async () => {
    t = await mountSpeak();
    await ready(t);
    const stops = fakeLive();
    t.env.click(t.$('sp-presets').querySelector('[data-preset-id="p1"]'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(pressed('p1'), 'true');
    t.parts.inspector.commit(t.parts.inspector.state().data, { ...t.parts.inspector.currentSetup(), identity: 'n1' });
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    stops.length = 0;
    t.env.fetch.reset();
    await chooseModel('piper/en_US-amy-medium');
    assert.deepEqual(stops, ['Voice changed']);
    assert.equal(pressed('p1'), 'false', 'the preset is cleared');
    assert.equal(t.parts.inspector.state().identity, '', 'the named voice is cleared');
    assert.equal(callsTo(t, 'GET /api/tts/capabilities')[0].query.get('model'), 'piper/en_US-amy-medium');
    assert.equal(callsTo(t, 'GET /v1/audio/voices')[0].query.get('model'), 'piper/en_US-amy-medium');
    assert.equal(t.parts.inspector.voiceSpec(), 'en_US-amy-medium');
    assert.equal(t.$('sp-voice-name').textContent, 'en_US-amy-medium');
  });

  it('[SP-03] the model select has provider optgroups, a state pill and the speed range 0.5–2.0 in 0.05 steps', async () => {
    t = await mountSpeak();
    await ready(t);
    const groups = [...t.$('sp-model').querySelectorAll('optgroup')].map((g) => g.getAttribute('label'));
    assert.deepEqual(groups, ['Kokoro', 'Piper', 'Chatterbox']);
    assert.equal(t.$('sp-model').value, 'kokoro');
    assert.equal(t.$('sp-model-hint-text').textContent, 'Loaded · GPU');
    assert.equal(t.$('sp-speed').getAttribute('min'), '0.5');
    assert.equal(t.$('sp-speed').getAttribute('max'), '2');
    assert.equal(t.$('sp-speed').getAttribute('step'), '0.05');
    assertAccessibleNames(t.$('sp-inspector'));
  });

  it('[SP-05] a late answer for an earlier model never replaces the newer model\'s controls', async () => {
    t = await mountSpeak({
      routes: {
        'GET /api/tts/capabilities': (req) => {
          const model = req.query.get('model');
          const body = json(200, CAPS[model] || { capabilities: {} });
          return model === 'piper/en_US-amy-medium' ? delay(3000, body) : body;
        },
      },
    });
    await ready(t);
    t.env.change(t.$('sp-model'), 'piper/en_US-amy-medium');
    await settle(t.env);
    t.env.change(t.$('sp-model'), 'chatterbox');
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.state().model, 'chatterbox');
    assert.equal(t.$('sp-reference-field').hidden, false, 'chatterbox controls');
    await t.env.clock.tickAsync(3000);
    await settle(t.env);
    assert.equal(t.parts.inspector.state().model, 'chatterbox', 'the slow Piper answer was dropped');
    assert.equal(t.$('sp-model').value, 'chatterbox');
    assert.equal(t.parts.inspector.state().data.model, 'chatterbox');
    assert.equal(t.$('sp-reference-field').hidden, false);
    assert.equal(t.parts.inspector.limit(), 2000);
  });

  it('[SP-05] a preset that fails its checks while a model change still loads never strands the inspector', async () => {
    const directed = { id: 'pi', name: 'Directed', backend: 'kokoro', model: 'kokoro', voice: 'af_sky', instructions: 'Whisper', speed: 1, format: 'mp3', effects: [] };
    t = await mountSpeak({
      presets: { profiles: [directed], default_profile_id: null },
      routes: {
        'GET /api/tts/capabilities': (req) => {
          const model = req.query.get('model');
          const body = json(200, CAPS[model] || { capabilities: {} });
          return model === 'piper/en_US-amy-medium' ? delay(2000, body) : body;
        },
      },
    });
    await ready(t);
    t.env.change(t.$('sp-model'), 'piper/en_US-amy-medium');
    await settle(t.env);
    t.env.click(t.$('sp-presets').querySelector('[data-preset-id="pi"]'));   // Kokoro takes no instructions: refused
    await settle(t.env);
    await settle(t.env);
    assert.match(t.$('sp-notice').textContent, /Preset “Directed” can't be used/);
    await t.env.clock.tickAsync(2000);
    await settle(t.env);
    await settle(t.env);
    const s = t.parts.inspector.state();
    assert.equal(s.model, 'piper/en_US-amy-medium', 'the model change carried on');
    assert.equal(s.data.model, 'piper/en_US-amy-medium');
    assert.equal(s.loading, false);
    assert.equal(t.parts.inspector.ready(), true);
    assert.equal(t.$('sp-model').value, 'piper/en_US-amy-medium');
    type(t, 'Ready again.');
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'false', 'Generate works, no endless "Loading voices…"');
  });

  it('[SP-05] a preset applied while a model change still loads wins, and the late answer changes nothing', async () => {
    t = await mountSpeak({
      routes: {
        'GET /api/tts/capabilities': (req) => {
          const model = req.query.get('model');
          const body = json(200, CAPS[model] || { capabilities: {} });
          return model === 'piper/en_US-amy-medium' ? delay(2000, body) : body;
        },
      },
    });
    await ready(t);
    t.env.change(t.$('sp-model'), 'piper/en_US-amy-medium');
    await settle(t.env);
    t.env.click(t.$('sp-presets').querySelector('[data-preset-id="p1"]'));   // Kokoro, Bella, 1.1×
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.state().model, 'kokoro');
    assert.equal(pressed('p1'), 'true');
    await t.env.clock.tickAsync(2000);
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.state().model, 'kokoro', 'the older Piper selection was dropped');
    assert.equal(t.$('sp-model').value, 'kokoro');
    assert.equal(t.parts.inspector.ready(), true);
    assert.equal(pressed('p1'), 'true', 'the applied preset stays pressed');
    assert.equal(t.parts.inspector.state().speed, 1.1);
  });

  it('[SP-04] edits made while a model\'s data loads are kept when it lands (speed, effects, format)', async () => {
    const models = modelsPayload({ extra: [{ id: 'kokoro-int8', type: 'tts', provider: 'kokoro', state: 'downloaded', device: 'cuda:0' }] });
    t = await mountSpeak({
      models,
      routes: {
        'GET /api/tts/capabilities': (req) => {
          const model = req.query.get('model');
          const body = json(200, CAPS[model.startsWith('kokoro') ? 'kokoro' : model] || { capabilities: {} });
          return model === 'kokoro-int8' ? delay(1000, body) : body;
        },
        'GET /v1/audio/voices': (req) => json(200, { voices: String(req.query.get('model')).startsWith('kokoro') ? KOKORO_VOICES : [{ id: 'en_US-amy-medium' }] }),
      },
    });
    await ready(t);
    await pickVoiceInPicker('Sky');
    t.env.change(t.$('sp-model'), 'kokoro-int8');
    await settle(t.env);
    t.env.input(t.$('sp-speed'), '1.6');
    t.env.click(t.$('sp-effects').querySelector('[data-effect="robot"]'));
    t.env.change(t.$('sp-format'), 'flac');
    await settle(t.env);
    await t.env.clock.tickAsync(1000);
    await settle(t.env);
    await settle(t.env);
    const s = t.parts.inspector.state();
    assert.equal(s.model, 'kokoro-int8');
    assert.equal(s.speed, 1.6, 'the speed set during the load stays');
    assert.equal(t.$('sp-speed').value, '1.6');
    assert.deepEqual(t.parts.inspector.currentSetup().effects, [{ type: 'robot' }], 'so does the effect');
    assert.equal(s.format, 'flac');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_sky', 'and the voice, which the new model offers');
  });

  it('[SP-09] the voice button pressed while a model loads opens the picker for the new model once its voices are in', async () => {
    t = await mountSpeak({
      routes: {
        'GET /api/tts/capabilities': (req) => {
          const model = req.query.get('model');
          const body = json(200, CAPS[model] || { capabilities: {} });
          return model === 'piper/en_US-amy-medium' ? delay(800, body) : body;
        },
      },
    });
    await ready(t);
    t.env.change(t.$('sp-model'), 'piper/en_US-amy-medium');
    await settle(t.env);
    t.env.click(t.$('sp-voice'));
    await settle(t.env);
    assert.equal(document.getElementById('app-voice-picker').open, false, 'not on the previous model\'s voices');
    assert.equal(t.$('sp-voice').getAttribute('aria-busy'), 'true');
    await t.env.clock.tickAsync(800);
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-voice').hasAttribute('aria-busy'), false);
    assert.equal(document.getElementById('app-voice-picker').open, true);
    const labels = [...document.querySelectorAll('#vp-list [role=option] .option__label')].map((o) => o.textContent);
    assert.ok(labels.includes('en_US-amy-medium'), labels.join(', '));
    assert.equal(labels.includes('Heart'), false);
  });

  it('[SP-06] capabilities as a list of flags or an object; the limit is min(max_input_length, max_input_chars), default 4096', async () => {
    t = await mountSpeak({ routes: { 'GET /api/tts/capabilities': (req) => (req.query.get('model') === 'kokoro' ? json(200, { capabilities: { voice_blend: true } }) : json(200, CAPS[req.query.get('model')])) } });
    await ready(t);
    assert.equal(t.parts.inspector.limit(), 4096, 'no limit reported: 4096');
    await chooseModel('piper/en_US-amy-medium');
    assert.equal(t.parts.inspector.caps().streaming, true, 'a list of flags');
    assert.equal(t.$('sp-stream-wrap').hidden, false);
    await chooseModel('chatterbox');
    assert.equal(t.parts.inspector.limit(), 2000, 'min(2000, 3000)');
    assert.match(t.$('sp-counter').textContent, /\/ 2,000$/);
  });

  it('[SP-07] capabilities gate the controls: blend, instructions, reference, Stream and fixed speed', async () => {
    t = await mountSpeak();
    await ready(t);
    assert.equal(t.$('sp-blend-toggle').hidden, false, 'Kokoro blends');
    assert.equal(t.$('sp-advanced').hidden, true, 'no instructions on Kokoro');
    assert.equal(t.$('sp-reference-field').hidden, true);
    assert.equal(t.$('sp-stream-wrap').hidden, false);
    t.env.input(t.$('sp-speed'), '1.5');
    await settle(t.env);
    await chooseModel('chatterbox');
    assert.equal(t.$('sp-blend-toggle').hidden, true);
    assert.equal(t.$('sp-advanced').hidden, false, 'instructions');
    assert.equal(t.$('sp-reference-field').hidden, false, 'voice_clone');
    assert.equal(t.$('sp-stream-wrap').hidden, true, 'no streaming');
    assert.equal(t.$('sp-speed').disabled, true, 'speed_control false');
    assert.equal(t.$('sp-speed').value, '1');
    assert.equal(t.$('sp-speed-out').textContent, '1×');
    assert.equal(t.$('sp-speed-hint').hidden, false);
    assert.equal(t.$('sp-speed-hint').textContent, 'This model has a fixed speed');
    assert.equal(t.$('sp-speed').getAttribute('aria-describedby'), 'sp-speed-hint');
  });

  it('[SP-08] the Advanced panel hides without rows and a rebuild keeps the typed instructions and its open state', async () => {
    t = await mountSpeak();
    await ready(t);
    await chooseModel('chatterbox');
    t.$('sp-advanced').open = true;
    t.env.input(t.$('sp-instructions'), 'Calm and warm');
    await chooseModel('kokoro');
    assert.equal(t.$('sp-advanced').hidden, true);
    await chooseModel('chatterbox');
    assert.equal(t.$('sp-advanced').hidden, false);
    assert.equal(t.$('sp-advanced').open, true);
    assert.equal(t.$('sp-instructions').value, 'Calm and warm');
    t.parts.inspector.referenceError('');
    t.env.change(t.$('sp-reference'), 'demo_reference');
    type(t, 'With instructions.');
    t.env.click(t.$('sp-generate'));
    await answerConfirm(t, true);   // Loading Chatterbox will unload Kokoro
    assert.equal(callsTo(t, 'POST /v1/audio/speech')[0].json.instructions, 'Calm and warm');
  });

  it('[SP-09] a model change keeps the current voice when the new model offers it, else selects the first', async () => {
    const models = modelsPayload({ extra: [{ id: 'kokoro-int8', type: 'tts', provider: 'kokoro', state: 'downloaded', device: 'cuda:0' }] });
    t = await mountSpeak({
      models,
      routes: { 'GET /v1/audio/voices': (req) => json(200, { voices: String(req.query.get('model')).startsWith('kokoro') ? KOKORO_VOICES : [{ id: 'en_US-amy-medium' }] }) },
    });
    await ready(t);
    await pickVoiceInPicker('Sky');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_sky');
    await chooseModel('kokoro-int8');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_sky', 'kept: the other Kokoro offers it');
    await chooseModel('piper/en_US-amy-medium');
    assert.equal(t.parts.inspector.voiceSpec(), 'en_US-amy-medium', 'not offered: the first voice');
  });

  it('[SP-10] the state pill follows the model and updates after a load', async () => {
    t = await mountSpeak();
    await ready(t);
    await chooseModel('piper/en_US-amy-medium');
    assert.equal(t.$('sp-model-hint-text').textContent, 'Loads when you generate · unloads Kokoro');
    assert.ok(t.$('sp-model-hint').classList.contains('pill--idle'));
    const list = t.app.get().models.list.map((m) => (m.id === 'piper/en_US-amy-medium' ? { ...m, state: 'loaded' } : (m.id === 'kokoro' ? { ...m, state: 'downloaded' } : m)));
    t.app.set({ models: { ...t.app.get().models, list } });
    await settle(t.env);
    assert.equal(t.$('sp-model-hint-text').textContent, 'Loaded · CPU');
    assert.ok(t.$('sp-model-hint').classList.contains('pill--ok'));
  });

  it('[SP-11] Restore Kokoro loads the default even when the inventory says it is loaded, reselects it, disables itself meanwhile and confirms', async () => {
    let release;
    t = await mountSpeak({ routes: { 'POST /api/models/:id/load': () => new Promise((resolve) => { release = () => resolve(json(200, { status: 'loaded' })); }) } });
    await ready(t);
    assert.equal(t.$('sp-restore-default').hidden, true, 'Kokoro is the loaded default and selected');
    const stops = fakeLive();
    await chooseModel('piper/en_US-amy-medium');
    stops.length = 0;
    assert.equal(t.$('sp-restore-default').hidden, false);
    assert.equal(t.$('sp-restore-default').textContent.trim(), 'Restore Kokoro');
    t.env.click(t.$('sp-restore-default'));
    await settle(t.env);
    assert.deepEqual(stops, ['Voice changed']);
    assert.equal(t.$('sp-restore-default').disabled, true);
    assert.equal(t.$('sp-restore-default').getAttribute('aria-busy'), 'true');
    assert.equal(callsTo(t, 'POST /api/models/:id/load')[0].params.id, 'kokoro', 'always the loader, even though the inventory says loaded');
    release();
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-model').value, 'kokoro', 'reselected');
    assert.equal(t.$('sp-restore-default').disabled, false);
    assert.equal(t.$('sp-restore-default').hidden, true);
    assert.ok(document.getElementById('app-toasts').textContent.includes('Kokoro restored'));
  });

  it('[SP-11] with the default selected but not loaded, the action reads Load Kokoro now, loads it, keeps it selected and says it loaded', async () => {
    t = await mountSpeak({ models: modelsPayload({ kokoroState: 'downloaded' }) });
    await ready(t);
    assert.equal(t.$('sp-model').value, 'kokoro');
    assert.equal(t.$('sp-model-hint-text').textContent.startsWith('Loads when you generate'), true);
    const action = t.$('sp-restore-default');
    assert.equal(action.hidden, false, 'shown while the default is not loaded');
    assert.equal(action.textContent.trim(), 'Load Kokoro now', 'never "Restore Kokoro" next to the selected Kokoro');
    t.env.click(action);
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    assert.equal(callsTo(t, 'POST /api/models/:id/load')[0].params.id, 'kokoro');
    assert.equal(t.$('sp-model').value, 'kokoro');
    assert.equal(action.hidden, true, 'loaded and selected: nothing to restore');
    assert.ok(document.getElementById('app-toasts').textContent.includes('Kokoro loaded'));
    assert.ok(!document.getElementById('app-toasts').textContent.includes('Kokoro restored'));
    // Another model selected: the same action restores the default.
    await chooseModel('piper/en_US-amy-medium');
    assert.equal(action.hidden, false);
    assert.equal(action.textContent.trim(), 'Restore Kokoro');
  });

  it('[SP-28] picking a provider voice clears the blend, the named voice and the preset, and stops Live', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.click(t.$('sp-presets').querySelector('[data-preset-id="p1"]'));
    await settle(t.env);
    await settle(t.env);
    t.parts.inspector.commit(t.parts.inspector.state().data, { ...t.parts.inspector.currentSetup(), identity: 'n1' });
    assert.equal(t.$('sp-blend').hidden, false, 'the named voice is a blend');
    const stops = fakeLive();
    await pickVoiceInPicker('Emma');
    assert.equal(t.parts.inspector.voiceSpec(), 'bf_emma');
    assert.equal(t.parts.inspector.state().mix.length, 0, 'blend cleared');
    assert.equal(t.parts.inspector.state().identity, '', 'named voice cleared');
    assert.equal(t.$('sp-blend').hidden, true);
    assert.equal(pressed('p1'), 'false', 'preset cleared');
    assert.deepEqual(stops, ['Voice changed']);
    assert.equal(t.$('sp-voice-name').textContent, 'Emma');
    assert.equal(t.$('sp-voice-detail').textContent, 'British English · female');
    assert.equal(document.activeElement, t.$('sp-voice'), 'focus returns to the voice button');
  });

  it('[SP-36] clone models show the reference recording outside Advanced; the choice survives a refresh, clears the named voice and preset, and Manage recordings opens Voices', async () => {
    t = await mountSpeak();
    await ready(t);
    await chooseModel('chatterbox');
    const field = t.$('sp-reference-field');
    assert.equal(field.hidden, false);
    assert.equal(t.$('sp-advanced').contains(t.$('sp-reference')), false, 'visible outside Advanced');
    assert.deepEqual([...t.$('sp-reference').querySelectorAll('option')].map((o) => o.textContent), ['None', 'demo_reference · transcript saved', 'raw_take · no transcript']);
    const stops = fakeLive();
    t.env.change(t.$('sp-reference'), 'demo_reference');
    await settle(t.env);
    assert.deepEqual(stops, ['Voice changed']);
    assert.equal(t.parts.inspector.state().reference, 'demo_reference');
    const { emit } = await load('core/store.js');
    emit('voices:changed', { what: 'recordings' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-reference').value, 'demo_reference', 'the selection survives the refresh');
    assert.equal(field.querySelector('a.sp-link').getAttribute('href'), '#voices/recordings');
    assert.equal(field.querySelector('a.sp-link').textContent, 'Manage recordings');
    type(t, 'Cloned.');
    t.env.click(t.$('sp-generate'));
    await answerConfirm(t, true);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.json.voice_library_ref, 'demo_reference');
    assert.equal(post.json.voice, 'demo_reference', 'no stock voice: the reference name is the voice');
  });

  // With Chatterbox selected, Speak's voice picker used to look like
  // Kokoro's (named voices first, the recordings it can use last) and nothing said why. Speak passes the model's
  // label, so the picker says a cloning model has no built-in voices and lists its recordings first.
  it('the voice picker for a cloning model says it has no built-in voices and lists Recordings first; Kokoro keeps its order and gets no note', async () => {
    // As the server answers ?model=chatterbox: these named voices only have a Kokoro version.
    const forModel = (model) => ({ voices: NAMED_FOR_TEST.voices.map((v) => (model === 'kokoro' ? v : {
      ...v, available: false, realizations: v.realizations.map((r) => ({ ...r, available: false, reason: "Select this realization's model" })),
    })) });
    t = await mountSpeak({ routes: { 'GET /api/voices/identities': (req) => json(200, forModel(req.query.get('model'))) } });
    await ready(t);
    const groups = () => [...document.querySelectorAll('#vp-list .listbox__label')].map((el) => el.textContent);
    const noteShown = () => { const note = document.getElementById('vp-model-note'); return !!note && !note.hidden; };
    const picker = document.getElementById('app-voice-picker');
    t.env.click(t.$('sp-voice'));
    await settle(t.env);
    assert.equal(picker.open, true);
    assert.deepEqual(groups(), ['Named voices', 'Recommended', 'American English', 'British English']);
    assert.equal(noteShown(), false, 'Kokoro has voices of its own');
    t.env.key('Escape', {}, document.getElementById('vp-search'));
    await settle(t.env);
    assert.equal(picker.open, false);
    await chooseModel('chatterbox');
    t.env.click(t.$('sp-voice'));
    await settle(t.env);
    assert.equal(picker.open, true);
    assert.deepEqual(groups(), ['Recordings', 'Named voices with no Chatterbox version']);
    assert.equal(noteShown(), true);
    assert.equal(document.getElementById('vp-model-note-text').textContent, 'Chatterbox has no built-in voices. It speaks in the voice of a recording.');
    assert.equal(document.getElementById('vp-clone').getAttribute('href'), '#voices/clone');
    const labels = [...document.querySelectorAll('#vp-list [role=option] .option__label')].map((o) => o.textContent);
    assert.deepEqual(labels, ['demo_reference', 'raw_take', 'Narrator', 'Old clone']);
    assertAccessibleNames(picker);
    t.env.key('Escape', {}, document.getElementById('vp-search'));
    await settle(t.env);
  });

  it('[SP-38] speed 0.5–2.0 in 0.05 steps reads "1.25 times", with steppers; Format offers MP3, WAV, FLAC, Opus, AAC, M4A and PCM', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.input(t.$('sp-speed'), '1.25');
    await settle(t.env);
    assert.equal(t.$('sp-speed').getAttribute('aria-valuetext'), '1.25 times');
    assert.equal(t.$('sp-speed-out').textContent, '1.25×');
    t.env.click(t.$('sp-speed-up'));
    await settle(t.env);
    assert.equal(t.parts.inspector.state().speed, 1.3);
    assert.equal(t.$('sp-speed').getAttribute('aria-valuetext'), '1.3 times');
    t.env.input(t.$('sp-speed'), '2');
    t.env.click(t.$('sp-speed-up'));
    await settle(t.env);
    assert.equal(t.$('sp-speed-up').disabled, true, 'the + stepper stops at 2.0');
    t.env.input(t.$('sp-speed'), '0.5');
    await settle(t.env);
    assert.equal(t.$('sp-speed-down').disabled, true);
    assert.deepEqual([...t.$('sp-format').querySelectorAll('option')].map((o) => o.textContent), ['MP3', 'WAV', 'FLAC', 'Opus', 'AAC', 'M4A', 'PCM (download only)']);
    assert.match(t.$('sp-summary-text').textContent, /0\.5×$/, 'the phone summary follows');
  });

  it('[LR-21] model, voice, named voice, effects, preset, a History hand-off, Restore and an upload each stop Live with a reason', async () => {
    t = await mountSpeak();
    await ready(t);
    const stops = fakeLive();
    await chooseModel('piper/en_US-amy-medium');
    await chooseModel('kokoro');
    await pickVoiceInPicker('Bella');
    await pickVoiceInPicker('Narrator');
    t.env.click(t.$('sp-effects').querySelector('[data-effect="normalize"]'));
    t.env.click(t.$('sp-presets').querySelector('[data-preset-id="p2"]'));
    await settle(t.env);
    await settle(t.env);
    t.setIntent('speak', { type: 'apply-history', entry: { id: 'h', type: 'tts', model: 'kokoro', voice: 'af_sky', full_text: 'Again', meta: {} } });
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    await settle(t.env);
    await chooseModel('piper/en_US-amy-medium');
    t.env.click(t.$('sp-restore-default'));
    await settle(t.env);
    await settle(t.env);
    const upload = t.$('sp-upload');
    Object.defineProperty(upload, 'files', { value: [new File(['x'], 'x.txt')], configurable: true });
    upload.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(t.env);
    assert.deepEqual(stops, [
      'Voice changed', 'Voice changed', 'Voice changed', 'Voice changed', 'Voice changed', 'Preset changed',
      'Text changed', 'Voice changed', 'Voice changed', 'Text changed',
    ]);
  });

  it('[REV-31] switching Speak to Generate stops a running Live session ("Switched to Generate"), since the editor is shared', async () => {
    t = await mountSpeak({ seed: LIVE_ON });
    await ready(t);
    const stops = fakeLive();
    t.ctl.show({ view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.equal(t.$('sp-live-slot').hidden, false);
    assert.equal(t.$('sp-genbar').hidden, true, 'the Live slot replaces the Generate controls');
    assert.equal(document.querySelector('input[name="sp-mode"][value="live"]').checked, true);
    assert.deepEqual(stops, []);
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.deepEqual(stops, ['Switched to Generate']);
    assert.equal(t.$('sp-genbar').hidden, false);
    assert.equal(t.$('sp-live-slot').hidden, true);
    // The mode control navigates with focus kept.
    const live = document.querySelector('input[name="sp-mode"][value="live"]');
    live.checked = true;
    live.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(t.env);
    assert.equal(location.hash, '#speak/live');
  });

  it('[REV-31] coming back to Speak from another view (the nav, g s) keeps a Live session running in the background, in Live mode', async () => {
    t = await mountSpeak({ seed: LIVE_ON });
    await ready(t);
    const stops = fakeLive();
    t.ctl.show({ view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' }, { focus: 'heading' });
    await settle(t.env);
    t.ctl.hide();   // the listener opens the Reader; Live keeps speaking (the mini-player shows it)
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'heading' });   // "Speak" in the nav
    await settle(t.env);
    assert.deepEqual(stops, [], 'Live keeps speaking');
    assert.equal(t.$('sp-live-slot').hidden, false, 'Speak shows Live, where it runs');
    assert.equal(document.querySelector('input[name="sp-mode"][value="live"]').checked, true);
    assert.equal(location.hash, '#speak/live', 'the address follows');
    // A switch inside Speak still stops it.
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.deepEqual(stops, ['Switched to Generate']);
  });

  it('a refused key or a failed model list reads so in the phone summary, never "Loading…"', async () => {
    t = await mountSpeak({ loaded: false });
    assert.equal(t.$('sp-summary-text').textContent, 'Loading…');
    t.app.set({ connection: { ...t.app.get().connection, state: 'unauthorized' }, models: { ...t.app.get().models, error: 'This server needs an API key. Enter it in Settings › Connection.' } });
    await settle(t.env);
    assert.equal(t.$('sp-summary-text').textContent, 'API key needed');
    const banner = t.$('sp-inspector-error');
    assert.equal(banner.querySelector('.banner__title').textContent, 'API key needed');
    assert.equal(banner.querySelector('.banner__msg'), null, 'the shell banner already says it');
    assert.ok([...banner.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Open Settings'));
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'true');
    assert.equal(t.$('sp-gen-hint').textContent, 'API key needed.', 'Generate names the cause, not just "didn\'t load"');
    // Too many wrong keys: the lockout is named as such, never "API key needed".
    t.app.set({ connection: { ...t.app.get().connection, state: 'locked' }, models: { ...t.app.get().models, error: 'Too many failed API key attempts' } });
    await settle(t.env);
    assert.equal(t.$('sp-summary-text').textContent, 'Too many wrong keys');
    assert.equal(banner.querySelector('.banner__title').textContent, 'Too many wrong keys');
    assert.equal(banner.querySelector('.banner__msg'), null);
    assert.equal(t.$('sp-gen-hint').textContent, 'Too many wrong API keys.');
    t.app.set({ connection: { ...t.app.get().connection, state: 'online' }, models: { ...t.app.get().models, error: 'Something went wrong on the server' } });
    await settle(t.env);
    assert.equal(t.$('sp-summary-text').textContent, "Couldn't load the models");
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't load the models");
    assert.equal(t.$('sp-gen-hint').textContent, "The models didn't load.");
  });

  it('[SP-10] the model select is described by its state pill; the speed and blend outputs never announce themselves', async () => {
    t = await mountSpeak();
    await ready(t);
    assert.equal(t.$('sp-model').getAttribute('aria-describedby'), 'sp-model-hint');
    assert.equal(t.$('sp-model-hint').textContent.trim(), 'Loaded · GPU');
    assert.equal(t.$('sp-speed-out').getAttribute('aria-live'), 'off', 'the slider says its value itself');
    assert.equal(t.$('sp-blend-string').getAttribute('aria-live'), 'off', 'an <output> is a live region by default');
  });

  it('[VL-19] new recordings refresh the reference options without losing typed instructions or the Advanced state', async () => {
    const recordings = [...RECORDINGS];
    t = await mountSpeak({ routes: { 'GET /api/voices/library': () => json(200, recordings) } });
    await ready(t);
    await chooseModel('chatterbox');
    t.$('sp-advanced').open = true;
    t.env.input(t.$('sp-instructions'), 'Whisper softly');
    recordings.push({ name: 'fresh_take', transcript: 'Hi' });
    const { emit } = await load('core/store.js');
    emit('voices:changed', { what: 'recordings' });
    await settle(t.env);
    await settle(t.env);
    assert.ok([...t.$('sp-reference').querySelectorAll('option')].some((o) => o.value === 'fresh_take'));
    assert.equal(t.$('sp-instructions').value, 'Whisper softly');
    assert.equal(t.$('sp-advanced').open, true);
  });

  // The Voices half (Delete emits voices:changed {what:'recordings'}) is in js/voices/recordings.test.js.
  const selectedText = (select) => select.options[select.selectedIndex]?.textContent ?? '(nothing selected)';

  async function deleteSelectedReference(recordings) {
    await chooseModel('chatterbox');
    t.env.change(t.$('sp-reference'), 'demo_reference');
    await settle(t.env);
    assert.equal(t.parts.inspector.state().reference, 'demo_reference');
    recordings.splice(recordings.findIndex((r) => r.name === 'demo_reference'), 1);
    const { emit } = await load('core/store.js');
    emit('voices:changed', { what: 'recordings' });
    await settle(t.env);
    await settle(t.env);
  }

  it('[VL-16] a selected reference recording deleted in Voices is dropped from Speak: the field reads None and Generate asks for a reference instead of sending the deleted one', async () => {
    const recordings = [...RECORDINGS];
    t = await mountSpeak({ routes: { 'GET /api/voices/library': () => json(200, recordings) } });
    await ready(t);
    await deleteSelectedReference(recordings);
    const select = t.$('sp-reference');
    assert.deepEqual([...select.querySelectorAll('option')].map((o) => o.textContent), ['None', 'raw_take · no transcript']);
    assert.equal(selectedText(select), 'None', 'the field shows None, not a blank select');
    assert.equal(t.parts.inspector.state().reference, '');
    type(t, 'Clone me.');
    t.env.fetch.reset();
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(t.$('sp-reference-error').hidden, false);
    assert.equal(t.$('sp-reference-error-text').textContent, 'Choose a reference recording for this model');
    assert.equal(document.getElementById('app-confirm')?.open ?? false, false, 'no load confirm for a doomed request');
    assert.equal(t.env.fetch.calls.length, 0, 'no request names the deleted recording');
  });

  it('[VL-16] with a stock voice, the next speech request after the deletion carries no voice_library_ref', async () => {
    const recordings = [...RECORDINGS];
    const caps = { ...CAPS.chatterbox, capabilities: { ...CAPS.chatterbox.capabilities, clone_transcript_required: false } };
    t = await mountSpeak({
      routes: {
        'GET /api/voices/library': () => json(200, recordings),
        'GET /api/tts/capabilities': (req) => json(200, req.query.get('model') === 'chatterbox' ? caps : (CAPS[req.query.get('model')] || { capabilities: {} })),
        'GET /v1/audio/voices': (req) => json(200, { voices: req.query.get('model') === 'chatterbox' ? [{ id: 'Ryan', name: 'Ryan' }] : KOKORO_VOICES }),
      },
    });
    await ready(t);
    await deleteSelectedReference(recordings);
    assert.equal(selectedText(t.$('sp-reference')), 'None');
    type(t, 'Stock voice now.');
    t.env.click(t.$('sp-generate'));
    await answerConfirm(t, true);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.ok(post, 'the take is requested');
    assert.equal(post.json.voice, 'Ryan');
    assert.equal('voice_library_ref' in post.json, false, 'the deleted recording is not sent');
  });

  it('[AU-09] a new API key reloads what the server refused and keeps what the listener picked', async () => {
    t = await mountSpeak();
    await ready(t);
    await pickVoiceInPicker('Sky');
    t.env.input(t.$('sp-speed'), '1.4');
    await settle(t.env);
    t.env.fetch.reset();
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'keep', scrolled: true, reason: 'auth' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(callsTo(t, 'GET /api/profiles').length, 1, 'presets reload');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_sky');
    assert.equal(t.parts.inspector.state().speed, 1.4);
    assert.equal(t.$('sp-model').value, 'kokoro');
  });

  it('a named voice deleted elsewhere (voices:changed) is dropped from the selection, keeping its recipe', async () => {
    const named = { voices: [...NAMED_FOR_TEST.voices] };
    t = await mountSpeak({ routes: { 'GET /api/voices/identities': () => json(200, named) } });
    await ready(t);
    t.parts.inspector.commit(t.parts.inspector.state().data, { ...t.parts.inspector.currentSetup(), identity: 'n1' });
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    named.voices = named.voices.filter((v) => v.id !== 'n1');
    const { emit } = await load('core/store.js');
    emit('voices:changed', { what: 'named' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.voiceSpec(), 'af_bella(2)+af_sky(1)');
    assert.equal(t.app.get().speakVoice.voice, 'af_bella(2)+af_sky(1)', 'the Reader sees it too');
  });

  it('the phone summary toggles the inspector body (aria-expanded, aria-controls)', async () => {
    t = await mountSpeak();
    await ready(t);
    const summary = t.$('sp-summary');
    assert.equal(summary.getAttribute('aria-controls'), 'sp-inspector-body');
    assert.equal(summary.getAttribute('aria-expanded'), 'false');
    assert.equal(t.$('sp-inspector').getAttribute('data-open'), 'false');
    assert.match(t.$('sp-summary-text').textContent, /^\w+ · Kokoro · 1(\.\d+)?×$/);
    t.env.click(summary);
    assert.equal(summary.getAttribute('aria-expanded'), 'true');
    assert.equal(t.$('sp-inspector').getAttribute('data-open'), 'true');
  });
});
