// Speak with a model that designs voices: the Instructions field is the voice description, and Generate needs no voice.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { designsVoice, instructionsCopy, voiceDisplay } from '../../../src/static/js/views/speak/inspector.js';
import { mountSpeak, ready, type, settle, callsTo, json, modelsPayload, CAPS, KOKORO_VOICES } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const DESIGN = 'qwen3/1.7b-voice-design';
const BREEZE = 'breeze/tts-2';
const MODEL_CAPS = {
  [DESIGN]: { capabilities: { voice_design: true, voice_clone: false, instructions: true, streaming: true, speed_control: false, live_reader: false, speakers: [] }, max_input_length: 4096 },
  [BREEZE]: { capabilities: { voice_design: true, voice_clone: true, reference_audio: true, clone_transcript: true, clone_transcript_required: true, instructions: true, streaming: true, speed_control: false, live_reader: true, speakers: [] }, max_input_length: 4096 },
};

async function mountWith(model) {
  const models = modelsPayload({ extra: [
    { id: DESIGN, type: 'tts', provider: 'qwen3', state: 'loaded', device: 'cuda:0', capabilities: MODEL_CAPS[DESIGN].capabilities },
    { id: BREEZE, type: 'tts', provider: 'breeze', state: 'loaded', device: 'cuda:0', capabilities: MODEL_CAPS[BREEZE].capabilities },
  ] });
  t = await mountSpeak({
    models,
    routes: {
      'GET /api/tts/capabilities': (req) => json(200, MODEL_CAPS[req.query.get('model')] || CAPS[req.query.get('model')] || { capabilities: {} }),
      'GET /v1/audio/voices': (req) => json(200, { voices: req.query.get('model') === 'kokoro' ? KOKORO_VOICES : [] }),
    },
  });
  await ready(t);
  t.env.change(t.$('sp-model'), model);
  await settle(t.env);
  await settle(t.env);
  await settle(t.env, 2000);
}

const noticeTitle = () => t.$('sp-notice').querySelector('.banner__title')?.textContent || '';

describe('views/speak voice design (pure)', () => {
  it('a model that designs voices designs one only while no voice, blend, named voice or recording is chosen', () => {
    const caps = { voice_design: true };
    assert.equal(designsVoice(caps, { voice: '', mix: [], identity: '', reference: '' }), true);
    assert.equal(designsVoice(caps, { voice: '', mix: [], identity: '', reference: 'demo_reference' }), false);
    assert.equal(designsVoice(caps, { voice: '', mix: [], identity: 'n1', reference: '' }), false);
    assert.equal(designsVoice({ voice_design: false }, { voice: '', mix: [], identity: '', reference: '' }), false);
    assert.equal(instructionsCopy(true).label, 'Voice description');
    assert.equal(instructionsCopy(false).label, 'Instructions');
    assert.equal(voiceDisplay({ mix: [] }, undefined, { designing: true }).name, 'Designed voice');
  });
});

describe('views/speak voice design', () => {
  it('a design-only model labels the field "Voice description", asks for one, then generates with no voice', async () => {
    await mountWith(DESIGN);
    assert.equal(t.$('sp-advanced').hidden, false);
    assert.equal(t.$('sp-advanced').open, true, 'the description is shown at once');
    assert.equal(t.$('sp-instructions-label').textContent, 'Voice description');
    assert.equal(t.$('sp-voice-name').textContent, 'Designed voice');
    type(t, 'A designed voice.');
    t.env.fetch.reset();
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(callsTo(t, 'POST /v1/audio/speech').length, 0);
    assert.equal(noticeTitle(), 'Describe the voice first');
    assert.equal(document.activeElement, t.$('sp-instructions'));
    t.env.input(t.$('sp-instructions'), 'A calm older man with a low voice');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const posts = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(posts.length, 1);
    assert.deepEqual([posts[0].json.model, posts[0].json.voice, posts[0].json.instructions, posts[0].json.voice_library_ref],
      [DESIGN, '', 'A calm older man with a low voice', undefined]);
  });

  it('Breeze designs from the description when no recording is picked, and clones once one is', async () => {
    await mountWith(BREEZE);
    assert.equal(t.$('sp-instructions-label').textContent, 'Voice description');
    t.env.input(t.$('sp-instructions'), 'A bright young woman');
    type(t, 'Designed by Breeze.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const posts = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(posts.length, 1, 'no "Choose a reference recording" stop');
    assert.deepEqual([posts[0].json.model, posts[0].json.instructions, posts[0].json.voice_library_ref], [BREEZE, 'A bright young woman', undefined]);
    t.env.change(t.$('sp-reference'), 'demo_reference');
    await settle(t.env);
    assert.equal(t.$('sp-instructions-label').textContent, 'Instructions', 'with a recording the field is how to speak');
  });
});
