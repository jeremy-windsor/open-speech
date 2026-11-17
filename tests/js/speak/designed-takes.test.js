// Speak with Breeze TTS 2: its own options (Guidance, Variation), and a take designed from the description saved as a
// designed voice (the take's Save as voice…, or Save named voice… while Speak designs).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { sampleProblem, MAX_SAMPLE_S, SAMPLE_TOO_LONG, SAMPLE_PCM } from '../../../src/static/js/views/speak/takes.js';
import { designedForm, saveNamedVoice, historySetup, NO_VOICE, DESIGNED_COPY, recordingName } from '../../../src/static/js/views/speak/presets.js';
import { designedVoiceFor } from '../../../src/static/js/views/speak/inspector.js';
import { modelOptions, optionValue, buildSpeechRequest } from '../../../src/static/js/views/speak/request.js';
import { FakeOfflineAudioContext, FakeAudioBuffer } from '../harness/fakes.js';
import { mountSpeak, ready, type, settle, callsTo, json, load, modelsPayload, CAPS, KOKORO_VOICES, NAMED } from './helpers.js';

let t;
afterEach(() => {
  FakeOfflineAudioContext.decodeResult = null;
  t?.env.restore();
});

const BREEZE = 'breeze/tts-2';
const BREEZE_OPTIONS = {
  cfg_scale: { type: 'number', label: 'Strictness', min: 1, max: 5, step: 0.5, default: null, description: 'How closely it follows the description. Above 3 can sound harsh.' },
  temperature: { type: 'number', label: 'Expressiveness', min: 0.1, max: 1.3, step: 0.05, default: 0.9, description: 'Lower is steadier, higher is livelier.' },
};
const BREEZE_CAPS = {
  capabilities: {
    voice_design: true, voice_clone: true, reference_audio: true, clone_transcript: true, clone_transcript_required: true,
    instructions: true, streaming: true, speed_control: false, live_reader: true, speakers: [], options: BREEZE_OPTIONS,
  },
  max_input_length: 4096,
};
const DESCRIPTION = 'A calm older man with a low voice';

function designedVoice(name) {
  return {
    id: 'd1', name, voice: 'voice:d1', available: true,
    realizations: [{ model: BREEZE, voice: 'designed-d1', reference_audio_id: 'designed-d1', available: true }],
    design: { description: DESCRIPTION, model: BREEZE, seed: null, options: {}, sample: 'designed-d1', transcript: 'x', duration_s: 1 },
  };
}

/** Mount Speak on Breeze; `designed` answers POST /api/voices/designed (and joins the named voices on a 201). */
async function mountBreeze({ designed = null, named = { voices: [...NAMED.voices] }, routes = {} } = {}) {
  const models = modelsPayload({ extra: [{ id: BREEZE, type: 'tts', provider: 'breeze', state: 'loaded', device: 'cuda:0', capabilities: BREEZE_CAPS.capabilities }] });
  t = await mountSpeak({
    models,
    routes: {
      ...routes,
      'GET /api/tts/capabilities': (req) => json(200, req.query.get('model') === BREEZE ? BREEZE_CAPS : (CAPS[req.query.get('model')] || { capabilities: {} })),
      'GET /v1/audio/voices': (req) => json(200, { voices: req.query.get('model') === 'kokoro' ? KOKORO_VOICES : [] }),
      'GET /api/voices/identities': () => json(200, named),
      'POST /api/voices/designed': (req) => {
        const res = designed ? designed(req) : json(201, designedVoice(req.body.get('name')));
        if (res.status === 201) named.voices.push(designedVoice(req.body.get('name')));
        return res;
      },
    },
  });
  await ready(t);
  t.env.change(t.$('sp-model'), BREEZE);
  await settle(t.env);
  await settle(t.env);
  await settle(t.env, 2000);
  t.env.change(t.$('sp-format'), 'flac');   // a whole take, no stream
  await settle(t.env);
}

async function generate(text = 'Hello from a designed voice.') {
  type(t, text);
  t.env.click(t.$('sp-generate'));
  for (let i = 0; i < 4; i += 1) await settle(t.env);
}

async function answerPrompt(value) {
  await settle(t.env);
  assert.equal(document.getElementById('app-prompt').open, true, 'the prompt is open');
  document.getElementById('app-prompt-input').value = value;
  document.getElementById('app-prompt-form').requestSubmit();
  for (let i = 0; i < 3; i += 1) await settle(t.env);
}

const takeEls = () => [...t.$('sp-takes').querySelectorAll('.sp-take')];
const noticeText = () => t.$('sp-notice').textContent.replace(/\s+/g, ' ').trim();

function saveItem(index = 0) {
  return takeEls()[index].parentElement.querySelector('[data-take-menu="save-voice"]');
}

async function chooseSave(index = 0) {
  const el = takeEls()[index];
  t.env.click(el.querySelector('[aria-haspopup="menu"]'));
  await settle(t.env);
  t.env.click(saveItem(index));
  await settle(t.env);
  await settle(t.env);
}

describe('views/speak designed takes (pure)', () => {
  it('an option whose default is Auto (null) is left out until it is set; others start at their default', () => {
    assert.equal(optionValue(BREEZE_OPTIONS.cfg_scale), undefined);
    assert.equal(optionValue(BREEZE_OPTIONS.cfg_scale, null), undefined);
    assert.equal(optionValue(BREEZE_OPTIONS.cfg_scale, 9), 5);
    assert.deepEqual(modelOptions(BREEZE_CAPS.capabilities, {}), { temperature: 0.9 });
    assert.deepEqual(modelOptions(BREEZE_CAPS.capabilities, { cfg_scale: 2.5, temperature: 0.6, steps: 8 }), { cfg_scale: 2.5, temperature: 0.6 });
    const body = buildSpeechRequest({ model: BREEZE, voice: '', input: 'Hi', caps: BREEZE_CAPS.capabilities, options: { cfg_scale: 3 } }).body;
    assert.deepEqual([body.cfg_scale, body.temperature], [3, 0.9]);
  });

  it('a take can be a voice sample unless it is PCM or longer than 30 s', () => {
    assert.equal(MAX_SAMPLE_S, 30);
    assert.equal(sampleProblem({ format: 'mp3', durationS: 12 }), '');
    assert.equal(sampleProblem({ format: 'mp3', durationS: 0 }), '', 'not known yet');
    assert.equal(sampleProblem({ format: 'wav', durationS: 30.4 }), SAMPLE_TOO_LONG);
    assert.equal(SAMPLE_TOO_LONG, 'A voice sample can be at most 30 s');
    assert.equal(sampleProblem({ format: 'pcm', durationS: 2 }), SAMPLE_PCM);
  });

  it('the designed voice form carries the sample, its words, the description, the model and the options', () => {
    const audio = new Blob([new Uint8Array([1, 2])], { type: 'audio/wav' });
    const form = designedForm(' Calm man ', { model: BREEZE, description: ` ${DESCRIPTION} `, transcript: 'Hello. ', audio, options: { cfg_scale: 2.5, temperature: 0.9 } });
    assert.deepEqual([form.get('name'), form.get('model'), form.get('description'), form.get('transcript')], ['Calm man', BREEZE, DESCRIPTION, 'Hello.']);
    assert.equal(form.get('audio').name, 'sample.wav');
    assert.deepEqual(JSON.parse(form.get('options')), { cfg_scale: 2.5, temperature: 0.9 });
    assert.equal(designedForm('x', { model: BREEZE, description: 'd', transcript: 't', audio }).has('options'), false, 'no options, no field');
  });

  it('Save named voice never sends an empty voice', async () => {
    await assert.rejects(saveNamedVoice({ model: BREEZE, recipe: '', reference: '' }), { message: NO_VOICE });
  });
});

describe('views/speak Breeze options', () => {
  it('Guidance starts at Auto and sends nothing; moved, it is sent, and Auto takes it back; Variation sends 0.9', async () => {
    await mountBreeze();
    assert.equal(t.$('sp-options').hidden, false);
    const labels = [...t.$('sp-options').querySelectorAll('label')].map((l) => l.textContent);
    assert.deepEqual(labels, ['Strictness', 'Expressiveness']);
    const outputs = () => [...t.$('sp-options').querySelectorAll('output')].map((o) => o.textContent);
    assert.deepEqual(outputs(), ['Auto', '0.90']);
    t.env.input(t.$('sp-instructions'), DESCRIPTION);
    await generate();
    let post = callsTo(t, 'POST /v1/audio/speech').at(-1);
    assert.equal('cfg_scale' in post.json, false, 'Auto sends nothing');
    assert.equal(post.json.temperature, 0.9);
    t.env.input(t.$('sp-option-cfg_scale'), '2.5');
    await settle(t.env);
    assert.deepEqual(outputs(), ['2.5', '0.90']);
    assert.deepEqual(t.parts.inspector.currentSetup().options, { cfg_scale: 2.5, temperature: 0.9 });
    await generate('Again, with guidance.');
    post = callsTo(t, 'POST /v1/audio/speech').at(-1);
    assert.equal(post.json.cfg_scale, 2.5);
    t.env.click(t.$('sp-options').querySelector('[data-auto="cfg_scale"]'));
    await settle(t.env);
    assert.deepEqual(outputs(), ['Auto', '0.90']);
    assert.equal(document.activeElement, t.$('sp-option-cfg_scale'), 'the slider keeps the focus');
    assert.deepEqual(t.parts.inspector.currentSetup().options, { temperature: 0.9 });
  });

  it('a setup with options says all of them: one it leaves out goes back to Auto or its default', async () => {
    await mountBreeze();
    t.env.input(t.$('sp-option-cfg_scale'), '4');
    t.env.input(t.$('sp-option-temperature'), '0.5');
    await settle(t.env);
    const inspector = t.parts.inspector;
    inspector.commit(inspector.state().data, { options: { temperature: 0.7 } });
    assert.deepEqual(inspector.currentSetup().options, { temperature: 0.7 });
    inspector.commit(inspector.state().data, {});
    assert.deepEqual(inspector.currentSetup().options, { temperature: 0.7 }, 'a setup without options keeps them');
  });
});

describe('views/speak Save as recording', () => {
  it("Re-generate keeps a Chatterbox take's Emotion", () => {
    const h = historySetup({ type: 'tts', model: 'chatterbox/regular', voice: 'will', full_text: 'x', meta: { exaggeration: 1.4, cfg_scale: 0.3 } }, { ttsIds: ['chatterbox/regular'] });
    assert.deepEqual(h.setup.options, { exaggeration: 1.4, cfg_scale: 0.3 });
  });

  it('names follow the voice library rule', () => {
    assert.equal(recordingName('  Will Clone-2! '), 'will_clone_2');
    assert.equal(recordingName('!!!'), '');
  });

  it("any take can be saved as a recording, as WAV with its words; a name a recording has is asked again, never replaced", async () => {
    await mountBreeze({ routes: {
      'GET /api/voices/library': () => json(200, [{ name: 'will_clone', transcript: 'An older take.' }]),
      'POST /api/voices/library': (req) => json(201, { name: req.body.get('name') }),
    } });
    const { on } = await load('core/store.js');
    const seen = [];
    const off = on('voices:changed', () => seen.push(true));
    t.env.input(t.$('sp-instructions'), DESCRIPTION);
    await generate('Hi, it is me, reading the way I talk.');
    const el = takeEls()[0];
    t.env.click(el.querySelector('[aria-haspopup="menu"]'));
    await settle(t.env);
    t.env.click(el.parentElement.querySelector('[data-take-menu="save-recording"]'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt-title').textContent, 'Save as recording');
    await answerPrompt('Will Clone');
    assert.match(document.getElementById('app-prompt').textContent, /A recording named will_clone already exists/);
    await answerPrompt('Will Clone 2');
    const calls = callsTo(t, 'POST /api/voices/library');
    assert.equal(calls.length, 1, 'nothing was sent under the name in use');
    const form = calls[0].body;
    assert.deepEqual([form.get('name'), form.get('transcript'), form.get('audio').type], ['will_clone_2', 'Hi, it is me, reading the way I talk.', 'audio/wav']);
    assert.ok(seen.length, 'Voices hears of it');
    off();
  });
});

describe('views/speak designed takes', () => {
  it('a designed take has Save as voice…, which saves its audio, words, description, model and options, then speaks with it', async () => {
    const seen = [];
    await mountBreeze();
    const { on } = await load('core/store.js');
    const off = on('voices:changed', (detail) => seen.push(detail));
    t.env.input(t.$('sp-instructions'), DESCRIPTION);
    t.env.input(t.$('sp-option-cfg_scale'), '3');
    await generate('Hello there, this is a designed voice.');
    const take = t.parts.takes.all()[0];
    assert.deepEqual(take.design, { description: DESCRIPTION, model: BREEZE, options: { cfg_scale: 3, temperature: 0.9 } });
    assert.equal(saveItem().textContent.trim(), 'Save as voice…');
    assert.equal(saveItem().getAttribute('aria-disabled'), 'false');
    await chooseSave();
    assert.equal(document.getElementById('app-prompt-title').textContent, 'Save as voice');
    assert.equal(document.getElementById('app-prompt-label').textContent, 'Voice name');
    await answerPrompt('Calm man');
    const [post] = callsTo(t, 'POST /api/voices/designed');
    const form = post.body;
    assert.deepEqual([form.get('name'), form.get('model'), form.get('description'), form.get('transcript')],
      ['Calm man', BREEZE, DESCRIPTION, 'Hello there, this is a designed voice.']);
    assert.deepEqual(JSON.parse(form.get('options')), { cfg_scale: 3, temperature: 0.9 });
    assert.equal(form.get('audio').type, 'audio/wav', 'a FLAC take is sent as WAV');
    assert.equal(callsTo(t, 'POST /api/voices/identities').length, 0);
    assert.ok(seen.some((d) => d && d.what === 'named'));
    assert.ok(document.getElementById('app-toasts-polite').textContent.includes('Saved Calm man'));
    await settle(t.env);
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:d1', 'Speak speaks with the new voice');
    assert.equal(t.$('sp-voice-name').textContent, 'Calm man');
    assert.equal(t.parts.inspector.designing(), false);
    assert.equal(t.$('sp-instructions').value, '', 'the description would steer the cloned voice: cleared');
    assert.equal(t.parts.inspector.state().instructions, '');
    assert.equal(t.$('sp-instructions-label').textContent, 'Direction');
    assert.match(document.getElementById('app-sr-status').textContent, /Voice: Calm man\. The voice description was cleared\./);
    off();
  });

  it('a name in use asks again with the server\'s message', async () => {
    let calls = 0;
    await mountBreeze({
      designed: (req) => {
        calls += 1;
        return calls === 1 ? json(409, { error: { message: 'A named voice called Narrator already exists' } }) : json(201, designedVoice(req.body.get('name')));
      },
    });
    t.env.input(t.$('sp-instructions'), DESCRIPTION);
    await generate();
    await chooseSave();
    await answerPrompt('Narrator');
    assert.equal(document.getElementById('app-prompt').open, true, 'asked again');
    assert.match(document.getElementById('app-prompt').textContent, /A named voice called Narrator already exists/);
    assert.equal(document.getElementById('app-prompt-input').value, 'Narrator');
    await answerPrompt('Narrator two');
    assert.equal(callsTo(t, 'POST /api/voices/designed').length, 2);
  });

  it('a take longer than 30 s can\'t be saved: the item says why', async () => {
    FakeOfflineAudioContext.decodeResult = new FakeAudioBuffer({ length: 31 * 8000, sampleRate: 8000 });
    await mountBreeze();
    t.env.input(t.$('sp-instructions'), DESCRIPTION);
    await generate();
    assert.equal(t.parts.takes.all()[0].durationS, 31, 'the decoded length');
    const item = saveItem();
    assert.equal(item.getAttribute('aria-disabled'), 'true');
    const note = item.querySelector('.sp-take__save-note');
    assert.equal(note.hidden, false);
    assert.equal(note.textContent, 'A voice sample can be at most 30 s');
    await chooseSave();
    assert.equal(document.getElementById('app-prompt').open, false);
    assert.equal(callsTo(t, 'POST /api/voices/designed').length, 0);
  });

  it('a take with a chosen voice has no Save as voice…', async () => {
    t = await mountSpeak();
    await ready(t);
    await generate();
    assert.equal(t.parts.takes.all()[0].design, null);
    assert.equal(saveItem(), null);
  });

  it('Save named voice… while designing: with no designed take it says so; with one it saves the newest as a designed voice', async () => {
    await mountBreeze();
    assert.equal(t.parts.inspector.designing(), true);
    t.env.click(t.$('sp-save-voice'));
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt').open, false);
    assert.match(noticeText(), /Generate a take first, then save it as a voice\./);
    t.env.input(t.$('sp-instructions'), DESCRIPTION);
    await generate('The first take.');
    await generate('The second take.');
    t.env.click(t.$('sp-save-voice'));
    await settle(t.env);
    await answerPrompt('Newest');
    const [post] = callsTo(t, 'POST /api/voices/designed');
    assert.equal(post.body.get('transcript'), 'The second take.');
    assert.equal(post.body.get('description'), DESCRIPTION);
    assert.equal(callsTo(t, 'POST /api/voices/identities').length, 0, 'never an empty voice');
    assert.equal(callsTo(t, 'POST /api/voices/identities/:id/realizations').length, 0);
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:d1');
    assert.equal(t.$('sp-instructions').value, '', 'the description is cleared here too');
  });
});

async function handOff(intent) {
  t.setIntent('speak', intent);
  t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'heading' });
  for (let i = 0; i < 4; i += 1) await settle(t.env);
}

describe('views/speak Use in Speak (use-named-voice)', () => {
  it("switches to the voice's model and selects it; a missing voice or a model it has no version for says so", async () => {
    await mountBreeze({ named: { voices: [...NAMED.voices, designedVoice('Calm man')] } });
    t.env.change(t.$('sp-model'), 'kokoro');
    await settle(t.env);
    await settle(t.env);
    await handOff({ type: 'use-named-voice', id: 'd1', model: BREEZE });
    assert.equal(t.$('sp-model').value, BREEZE);
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:d1');
    assert.equal(t.$('sp-voice-name').textContent, 'Calm man');
    assert.equal(t.$('sp-notice').hidden, true, 'no "reference recording missing" for its own sample');
    assert.equal(document.activeElement, t.$('sp-input'));
    await handOff({ type: 'use-named-voice', id: 'n1' });
    assert.equal(t.$('sp-model').value, 'kokoro', 'no model given: its first available model with a version');
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    await handOff({ type: 'use-named-voice', id: 'gone', model: BREEZE });
    assert.match(noticeText(), /Couldn't use the voice.*This named voice no longer exists\./);
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1', 'nothing changed');
    await handOff({ type: 'use-named-voice', id: 'n1', model: BREEZE });
    assert.match(noticeText(), /Couldn't use Narrator.*Narrator has no version for Breeze TTS 2\./);
    assert.equal(t.$('sp-model').value, 'kokoro', 'nothing changed');
  });
});

describe('views/speak Use in Speak and the voice description', () => {
  it('a description left from designing is cleared on Breeze; typed Instructions are kept when Speak was not designing', async () => {
    await mountBreeze({ named: { voices: [...NAMED.voices, designedVoice('Calm man')] } });
    t.env.input(t.$('sp-instructions'), DESCRIPTION);
    await settle(t.env);
    assert.equal(t.parts.inspector.designing(), true);
    await handOff({ type: 'use-named-voice', id: 'd1', model: BREEZE });
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:d1');
    assert.equal(t.$('sp-instructions').value, '');
    assert.match(document.getElementById('app-sr-status').textContent, /Voice: Calm man\. The voice description was cleared\./);
    t.env.input(t.$('sp-instructions'), 'Slow and warm');
    await settle(t.env);
    await handOff({ type: 'use-named-voice', id: 'd1', model: BREEZE });
    assert.equal(t.$('sp-instructions').value, 'Slow and warm', 'Speak was not designing: the instructions stay');
    assert.equal(t.parts.inspector.state().instructions, 'Slow and warm');
  });
});

describe('views/speak designed voices from History and in Save named voice…', () => {
  it("a History entry names its named voice (meta.named_voice), or its designed sample maps to the voice that owns it", () => {
    const h = historySetup({ type: 'tts', model: BREEZE, voice: 'designed-d1', full_text: 'x', meta: { named_voice: 'voice:d1', reference_voice: 'designed-d1' } }, { ttsIds: [BREEZE] });
    assert.deepEqual([h.setup.identity, h.setup.voice, h.referenceFallback], ['d1', '', null]);
    const row = designedVoice('Calm man');
    assert.equal(designedVoiceFor([...NAMED.voices, row], 'designed-d1', BREEZE), row);
    assert.equal(designedVoiceFor([row], 'designed-d1', 'kokoro'), null, 'no version for that model');
    assert.equal(designedVoiceFor([row], 'demo_reference', BREEZE), null);
  });

  it('Re-generate of a take spoken in a designed voice selects that voice, with no "recording missing" notice', async () => {
    await mountBreeze({ named: { voices: [...NAMED.voices, designedVoice('Calm man')] } });
    await handOff({
      type: 'apply-history',
      entry: { type: 'tts', model: BREEZE, voice: 'designed-d1', full_text: 'Said by a designed voice.', meta: { reference_voice: 'designed-d1', instructions: '', temperature: 0.9 } },
    });
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:d1');
    assert.equal(t.$('sp-voice-name').textContent, 'Calm man');
    assert.equal(t.$('sp-notice').hidden, true);
    assert.equal(t.$('sp-input').value, 'Said by a designed voice.');
    await generate('Said by a designed voice.');
    assert.equal(callsTo(t, 'POST /v1/audio/speech').at(-1).json.voice, 'voice:d1');
  });

  it('Save named voice… with a designed voice says how to copy it, with no name prompt and no save', async () => {
    await mountBreeze({ named: { voices: [...NAMED.voices, designedVoice('Calm man')] } });
    await handOff({ type: 'use-named-voice', id: 'd1', model: BREEZE });
    t.env.click(t.$('sp-save-voice'));
    await settle(t.env);
    assert.notEqual(document.getElementById('app-prompt').open, true, 'no name prompt');
    assert.match(noticeText(), new RegExp(DESIGNED_COPY.replace(/[.›]/g, '.')));
    assert.equal(callsTo(t, 'POST /api/voices/identities').length, 0);
    assert.equal(callsTo(t, 'POST /api/voices/identities/:id/realizations').length, 0);
    assert.equal(callsTo(t, 'PUT /api/voices/identities/:id/realizations/*').length, 0);
  });

  it("Update voice on a designed voice's version sends its own version, never an empty voice", async () => {
    await mountBreeze({
      named: { voices: [...NAMED.voices, designedVoice('Calm man')] },
      routes: { 'PUT /api/voices/identities/:id/realizations/*': () => json(200, designedVoice('Calm man')) },
    });
    await handOff({ type: 'edit-named-voice', id: 'd1', model: BREEZE });
    assert.equal(t.$('sp-editing').hidden, false);
    t.env.click(t.$('sp-editing-update'));
    for (let i = 0; i < 3; i += 1) await settle(t.env);
    const [put] = callsTo(t, 'PUT /api/voices/identities/:id/realizations/*');
    assert.deepEqual(put.json, { model: BREEZE, voice: 'designed-d1', reference_audio_id: 'designed-d1' });
  });
});
