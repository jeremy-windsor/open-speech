// Voices: the 3-step design wizard at #voices/design (describe, listen, name and save), which saves a designed
// named voice: its sample, the sample's words as the transcript, and a version for each "Speak with" model.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames, accessibleName } from '../harness/a11y.js';
import { voicesEnv, MODELS, namedVoices, json, bytes, load, delay } from './helpers.js';

let ctx;
afterEach(() => {
  ctx?.env.restore();
});

const SAMPLE_TEXT = 'Hello! This is the voice you described. I can read your stories, your notes and your messages aloud.';
const DESIGN_CAPS = { voice_design: true, voice_clone: true, clone_transcript_required: true, instructions: true };
const BREEZE_OPTIONS = {
  cfg_scale: { type: 'number', label: 'Guidance', min: 1, max: 5, step: 0.5, default: null, description: 'How closely it follows the description.' },
  temperature: { type: 'number', label: 'Variation', min: 0.1, max: 1.3, step: 0.05, default: 0.9, description: 'Lower is steadier, higher is livelier.' },
};
const WITH_BREEZE = {
  ...MODELS,
  models: [...MODELS.models, { id: 'breeze/tts-2', type: 'tts', provider: 'breeze', state: 'loaded', device: 'external', provider_available: true, capabilities: DESIGN_CAPS }],
};
const WITH_BREEZE_OPTIONS = {
  ...MODELS,
  models: [...MODELS.models, { id: 'breeze/tts-2', type: 'tts', provider: 'breeze', state: 'loaded', device: 'external', provider_available: true, capabilities: { ...DESIGN_CAPS, options: BREEZE_OPTIONS } }],
};
/** A designed named voice, as GET /api/voices/identities lists it. */
const DESIGNED = Object.freeze({
  id: 'n7', name: 'Warm narrator', voice: 'voice:n7', preset_count: 0,
  realizations: [{ id: 'r7', model: 'breeze/tts-2', voice: 'designed_n7', reference_audio_id: 'designed_n7' }],
  design: { description: 'A warm narrator', model: 'breeze/tts-2', seed: 1234, options: { cfg_scale: 2.5 }, sample: 'designed_n7', transcript: 'Words of the sample.', duration_s: 4.2 },
});
const $ = (id) => document.getElementById(id);
const text = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
const status = () => text($('vd-status'));

/** A 16-bit mono WAV of `seconds` at `rate`, as the server writes it. */
function wav(seconds = 2, rate = 24000) {
  const data = Math.round(seconds * rate) * 2;
  const out = new Uint8Array(44 + data);
  const view = new DataView(out.buffer);
  const tag = (offset, s) => { for (let i = 0; i < 4; i += 1) out[offset + i] = s.charCodeAt(i); };
  tag(0, 'RIFF'); view.setUint32(4, 36 + data, true); tag(8, 'WAVE');
  tag(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  tag(36, 'data'); view.setUint32(40, data, true);
  return out;
}

/** POST /api/voices/design answering with a new seed each time (41, 42, …), as the server does without one. */
function seededSamples(seconds = 2) {
  let next = 40;
  return (req) => {
    const seed = Number.isInteger(req.json.seed) ? req.json.seed : (next += 1);
    return bytes(200, wav(seconds), 'audio/wav', { 'X-Voice-Seed': String(seed) });
  };
}

async function wizard({ models = WITH_BREEZE, routes = {}, ...options } = {}) {
  ctx = await voicesEnv({
    sub: 'design',
    models,
    routes: { 'POST /api/voices/design': seededSamples(), ...routes },
    ...options,
  });
  return ctx;
}

async function describeVoice(words = 'A warm, calm narrator with a slight rasp') {
  ctx.env.input($('vd-description'), words);
  await ctx.settle();
  ctx.env.click($('vd-next-1'));
  await ctx.settle();
}

async function generate() {
  ctx.env.click($('vd-generate'));
  await ctx.settle();
  await ctx.settle();
  await ctx.settle();
}

async function toSave() {
  ctx.env.click($('vd-next-2'));
  await ctx.settle();
}

async function click(id) {
  ctx.env.click($(id));
  await ctx.settle();
  await ctx.settle();
  await ctx.settle();
  await ctx.settle();
}

/** The named voices list, with POST /api/voices/designed adding to it. */
function designedRoutes(seed = []) {
  const state = { voices: [...namedVoices().voices, ...seed] };
  return {
    state,
    routes: {
      'GET /api/voices/identities': () => json(200, { voices: state.voices }),
      'POST /api/voices/designed': (req) => {
        const speak = req.body.getAll('speak_with').filter(Boolean);
        const made = {
          id: 'n8', name: req.body.get('name'), voice: 'voice:n8', preset_count: 0,
          realizations: speak.map((model, i) => ({ id: `r8${i}`, model, voice: 'designed_n8', reference_audio_id: 'designed_n8' })),
          design: { description: req.body.get('description'), model: req.body.get('model'), seed: Number(req.body.get('seed')), options: {}, sample: 'designed_n8', transcript: req.body.get('transcript'), duration_s: 2 },
        };
        state.voices = [...state.voices, made];
        return json(201, made);
      },
    },
  };
}

describe('views/voices design wizard: word chips', () => {
  it('a chip adds its words to the description, after a comma', async () => {
    const { addWords } = await load('views/voices/design.js');
    assert.equal(addWords('', 'warm'), 'warm');
    assert.equal(addWords('A man ', 'deep'), 'A man, deep');
    assert.equal(addWords('A man,', 'deep'), 'A man, deep');
    await wizard();
    ctx.env.input($('vd-description'), 'An older man');
    ctx.env.click(ctx.env.document.querySelector('#vd-chips [data-chip="Southern"]'));
    await ctx.settle();
    assert.equal($('vd-description').value, 'An older man, Southern');
    const groups = [...ctx.env.document.querySelectorAll('#vd-chips [role=group]')].map((g) => g.getAttribute('aria-label'));
    assert.deepEqual(groups, ['Age', 'Pitch', 'Texture', 'Accent', 'Pace and mood']);
  });
});

describe('views/voices design wizard', () => {
  it('Design a voice opens the wizard with focus on its heading, the header buttons hide, and every control is named', async () => {
    ctx = await voicesEnv({ models: WITH_BREEZE });
    const design = $('vo-design');
    assert.equal(text(design), 'Design a voice');
    assert.ok(design.classList.contains('btn--secondary'), 'Clone a voice stays the one primary header action');
    ctx.env.click(design);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices/design');
    assert.equal(design.hidden, true);
    assert.equal($('vo-clone').hidden, true);
    assert.equal($('vo-design-wizard').hidden, false);
    assert.equal($('vo-wizard').hidden, true, 'the clone wizard stays hidden');
    assert.equal(document.activeElement?.id, 'vd-title');
    assert.equal(text(document.activeElement), 'Design a voice');
    assertAccessibleNames(ctx.section);
  });

  it('the wizard is remembered as Voices, and its back button returns there', async () => {
    await wizard();
    const prefs = await load('core/prefs.js');
    assert.equal(prefs.get().lastRoute, 'voices');
    const back = ctx.$('#vo-design-wizard .vo-back');
    assert.equal(text(back), 'Voices');
    ctx.env.click(back);
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices');
  });

  it('Describe lists only the models that design a voice; Next needs a description and says so on the field', async () => {
    await wizard();
    const options = [...$('vd-model').querySelectorAll('option')].map((o) => [o.value, text(o)]);
    assert.deepEqual(options, [['breeze/tts-2', 'Breeze TTS 2']], 'Kokoro and Chatterbox cannot design a voice');
    assert.equal($('vd-options').hidden, true, 'a model without options shows none');
    assert.equal($('vd-step-2').hidden, true);
    assert.equal(ctx.$('#vo-design-wizard [data-step="2"]').getAttribute('aria-disabled'), 'true');
    ctx.env.click($('vd-next-1'));
    await ctx.settle();
    assert.equal($('vd-step-1').hidden, false, 'still on Describe');
    assert.equal($('vd-description').getAttribute('aria-invalid'), 'true');
    assert.equal(text($('vd-description-error')), 'Describe the voice you want.');
    assert.equal(document.activeElement?.id, 'vd-description');
    await describeVoice();
    assert.equal($('vd-step-2').hidden, false);
    assert.equal(document.activeElement?.id, 'vd-step-2-title');
    assert.equal($('vd-text').value, SAMPLE_TEXT, 'the sample text starts filled in and can be edited');
    assert.equal($('vd-description').getAttribute('aria-invalid'), null);
  });

  it('without a model that designs a voice, the select says so, the hint links to Models and Next is unavailable', async () => {
    await wizard({ models: MODELS });
    const select = $('vd-model');
    assert.equal(select.disabled, true);
    assert.equal(text(select.querySelector('option')), 'No model that can design a voice');
    assert.ok(text($('vd-model-hint')).startsWith('No model that can design a voice is running.'));
    assert.equal($('vd-model-hint').querySelector('a').getAttribute('href'), '#models/tts');
    assert.equal($('vd-next-1').getAttribute('aria-disabled'), 'true');
  });

  it('Generate loads the model, POSTs the description and sample text, and offers the sample to listen to', async () => {
    let loaded = false;
    await wizard({ models: { ...WITH_BREEZE, models: WITH_BREEZE.models.map((m) => (m.id === 'breeze/tts-2' ? { ...m, state: 'downloaded' } : m)) },
      routes: {
        'GET /api/models/:id/status': () => json(200, { id: 'breeze/tts-2', type: 'tts', provider: 'breeze', state: loaded ? 'loaded' : 'downloaded' }),
        'POST /api/models/:id/load': () => { loaded = true; return json(200, { status: 'loaded' }); },
      } });
    await describeVoice('  A bright young woman, quick and cheerful  ');
    assert.equal(text($('vd-generate')), 'Generate sample');
    assert.ok($('vd-generate').classList.contains('btn--primary'));
    assert.equal($('vd-next-2').getAttribute('aria-disabled'), 'true');
    await generate();
    assert.equal(ctx.confirmText().title, 'Switch speech model?', 'the other loaded speech models make way, as for a clone test');
    await ctx.answerConfirm(true);
    await ctx.settle();
    await ctx.settle();
    const post = ctx.calls('POST /api/voices/design')[0];
    assert.ok(post, 'the sample was asked for');
    assert.deepEqual(post.json, { model: 'breeze/tts-2', description: 'A bright young woman, quick and cheerful', text: SAMPLE_TEXT });
    assert.equal(ctx.calls('POST /api/models/:id/load')[0].path, `/api/models/${encodeURIComponent('breeze/tts-2')}/load`);
    assert.equal($('vd-sample').hidden, false);
    assert.ok($('vd-audio').getAttribute('src') || $('vd-audio').src, 'the player has the sample');
    assert.equal(text($('vd-sample-meta')), '2.0 s · 24 kHz · mono · 96 KB · Breeze TTS 2');
    assert.equal(status(), 'Sample ready. Listen to it, generate again, or go on to name it.');
    assert.equal(text($('vd-generate')), 'Generate again');
    assert.ok($('vd-next-2').classList.contains('btn--primary'), 'one primary action: Next once there is a sample');
    assert.equal($('vd-next-2').getAttribute('aria-disabled'), null);
    assertAccessibleNames(ctx.section);
  });

  it("the design model's options (Guidance, Variation) are offered on Describe and sent with each sample", async () => {
    await wizard({ models: WITH_BREEZE_OPTIONS });
    assert.equal($('vd-options').hidden, false);
    const guidance = $('vd-option-cfg_scale');
    const variation = $('vd-option-temperature');
    assert.equal(accessibleName(guidance), 'Guidance');
    assert.equal(accessibleName(variation), 'Variation');
    assert.equal(text(ctx.$('#vd-options [data-option="cfg_scale"] output')), 'Auto', 'a null default starts at Auto');
    await describeVoice();
    await generate();
    assert.deepEqual(ctx.calls('POST /api/voices/design')[0].json.options, { temperature: 0.9 }, 'Auto sends nothing; the default Variation is sent');
    await ctx.go('voices/design');
    ctx.env.click(ctx.$('#vo-design-wizard [data-step="1"]'));
    await ctx.settle();
    ctx.env.input(guidance, '3.5');
    ctx.env.input(variation, '0.6');
    await ctx.settle();
    ctx.env.click(ctx.$('#vo-design-wizard [data-step="2"]'));
    await ctx.settle();
    await generate();
    assert.deepEqual(ctx.calls('POST /api/voices/design')[1].json.options, { cfg_scale: 3.5, temperature: 0.6 });
    assertAccessibleNames(ctx.section);
  });

  it('the seed of the sample being listened to is kept: new words speak the same voice, Generate again makes a new one', async () => {
    await wizard();
    await describeVoice();
    await generate();
    assert.equal(ctx.calls('POST /api/voices/design')[0].json.seed, undefined, 'the first sample is a new voice');
    assert.equal(ctx.controller.__internals.designWizard.state().sample.seed, 41, 'X-Voice-Seed is kept');
    ctx.env.input($('vd-text'), 'Some other words.');
    await ctx.settle();
    await generate();
    assert.equal(ctx.calls('POST /api/voices/design')[1].json.seed, 41, 'new words: the same voice again');
    assert.equal(text($('vd-generate')), 'Generate again');
    await generate();
    assert.equal(ctx.calls('POST /api/voices/design')[2].json.seed, undefined, 'Generate again: a new voice');
    assert.equal(ctx.controller.__internals.designWizard.state().sample.seed, 42);
  });

  it('new sample words drop the sample, so the saved transcript is always what it says', async () => {
    await wizard();
    await describeVoice();
    await generate();
    assert.equal($('vd-sample').hidden, false);
    ctx.env.input($('vd-text'), 'Other words entirely.');
    await ctx.settle();
    assert.equal($('vd-sample').hidden, true);
    assert.equal($('vd-next-2').getAttribute('aria-disabled'), 'true');
    assert.equal(status(), 'The sample text changed. Generate the sample again.');
    ctx.env.input($('vd-text'), '');
    await ctx.settle();
    await generate();
    assert.equal(ctx.calls('POST /api/voices/design').length, 1, 'no request without words');
    assert.equal(text($('vd-text-error')), 'Type the words the sample speaks.');
    assert.equal($('vd-text').getAttribute('aria-invalid'), 'true');
  });

  it('a failed sample shows the server\'s reason inline and keeps the wizard on Listen', async () => {
    await wizard({ routes: { 'POST /api/voices/design': () => json(503, { detail: { code: 'provider_unavailable', message: "TTS provider 'breeze' is unavailable" } }) } });
    await describeVoice();
    await generate();
    const banner = $('vd-generate-error').querySelector('[role="alert"]');
    assert.ok(banner, 'an alert banner');
    assert.match(text(banner), /TTS provider 'breeze' is unavailable/);
    assert.equal($('vd-sample').hidden, true);
    assert.equal($('vd-step-2').hidden, false);
    assert.equal(text($('vd-generate')), 'Generate sample');
  });

  it('an answer that arrives after the words changed is dropped: it speaks the old words', async () => {
    await wizard({ routes: { 'POST /api/voices/design': () => delay(3000, bytes(200, wav(2), 'audio/wav')) } });
    await describeVoice();
    ctx.env.click($('vd-generate'));
    await ctx.settle();
    assert.equal($('vd-generate').getAttribute('aria-busy'), 'true');
    assert.equal(text($('vd-generate')), 'Generating…');
    ctx.env.input($('vd-text'), 'Changed while it was generating.');
    await ctx.settle(3500);
    await ctx.settle();
    assert.equal($('vd-sample').hidden, true);
    assert.equal($('vd-next-2').getAttribute('aria-disabled'), 'true');
    assert.equal(status(), 'The sample text changed. Generate the sample again.');
    assert.equal(text($('vd-generate')), 'Generate sample');
  });

  it('a sample that arrives after leaving the wizard is dropped and never plays', async () => {
    await wizard({ routes: { 'POST /api/voices/design': () => delay(3000, bytes(200, wav(2), 'audio/wav')) } });
    await describeVoice();
    ctx.env.click($('vd-generate'));
    await ctx.settle();
    assert.equal(text($('vd-generate')), 'Generating…');
    ctx.env.click(ctx.$('#vo-design-wizard .vo-back'));
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices');
    await ctx.settle(3500);
    await ctx.settle();
    assert.equal($('vd-audio').paused, true, 'nothing starts playing outside the wizard');
    assert.equal($('vd-sample').hidden, true);
    assert.equal(text($('vd-generate')), 'Generate sample', 'the wizard is not left busy');
  });

  it('the sample text is capped at 300 characters; a sample over 30 s says it can\'t be saved', async () => {
    await wizard({ routes: { 'POST /api/voices/design': () => bytes(200, wav(31.2), 'audio/wav', { 'X-Voice-Seed': '7' }) } });
    assert.equal($('vd-text').maxLength, 300);
    await describeVoice();
    await generate();
    const warn = $('vd-sample-long');
    assert.equal(warn.hidden, false);
    assert.equal(text(warn), 'A voice sample can be at most 30 s (this one is 31.2 s). Shorten the sample text, then generate again.');
    assert.match(status(), /at most 30 s/);
    await toSave();
    ctx.env.input($('vd-name'), 'Long one');
    await ctx.settle();
    assert.equal($('vd-save').getAttribute('aria-disabled'), 'true');
    assert.equal(text($('vd-save-hint')), 'A voice sample can be at most 30 s (this one is 31.2 s). Shorten the sample text, then generate again.');
    ctx.env.click($('vd-save'));
    await ctx.settle();
    assert.equal(ctx.calls('POST /api/voices/designed').length, 0);
  });

  it('Save POSTs a designed voice (name, model, description, sample, its words, seed, options, Speak with), then focuses its card in Voices', async () => {
    const data = designedRoutes();
    await wizard({ models: WITH_BREEZE_OPTIONS, routes: data.routes });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await describeVoice();
    await generate();
    await toSave();
    assert.equal(document.activeElement?.id, 'vd-step-3-title');
    assert.equal($('vd-save').getAttribute('aria-disabled'), 'true', 'a name first');
    assert.equal(text($('vd-save-hint')), 'Enter a name for the voice.');
    assert.equal(text($('vd-save')), 'Save voice');
    assert.equal($('vd-replace').hidden, true, 'Replace is for Edit description only');
    const boxes = ctx.$$('#vd-speak-with-list input[type="checkbox"]');
    assert.deepEqual(boxes.map((b) => [accessibleName(b), b.checked]), [['Breeze TTS 2', true], ['Chatterbox', false]], 'the cloning models; the design model clones, so it is checked');
    ctx.env.input($('vd-name'), '  Warm Narrator 2 ');
    await ctx.settle();
    assert.match(text($('vd-summary')), /TranscriptHello! This is the voice you described/);
    assert.match(text($('vd-summary')), /VersionsBreeze TTS 2/);
    await click('vd-save');
    const post = ctx.calls('POST /api/voices/designed')[0];
    assert.ok(post, 'the designed voice was saved');
    assert.equal(post.body.get('name'), 'Warm Narrator 2');
    assert.equal(post.body.get('model'), 'breeze/tts-2');
    assert.equal(post.body.get('description'), 'A warm, calm narrator with a slight rasp');
    assert.equal(post.body.get('transcript'), SAMPLE_TEXT);
    assert.equal(post.body.get('audio').type, 'audio/wav');
    assert.equal(post.body.get('seed'), '41');
    assert.deepEqual(JSON.parse(post.body.get('options')), { temperature: 0.9 });
    assert.deepEqual(post.body.getAll('speak_with'), ['breeze/tts-2']);
    assert.equal(ctx.calls('POST /api/voices/library').length, 0, 'not a recording');
    assert.ok(ctx.toasts().includes('Saved Warm Narrator 2'));
    assert.deepEqual(events, ['named']);
    assert.equal(ctx.app.get().route.path, 'voices');
    assert.equal(document.activeElement?.id, 'vo-named-n8-title', 'the new card is focused');
    await ctx.go('voices/design');
    assert.equal($('vd-description').value, '', 'the wizard starts afresh');
    assert.equal($('vd-step-1').hidden, false);
  });

  it('Speak with: the models the listener checks are sent; none checked sends an empty choice, so no version is made', async () => {
    const data = designedRoutes();
    await wizard({ routes: data.routes });
    await describeVoice();
    await generate();
    await toSave();
    ctx.env.input($('vd-name'), 'Quiet');
    ctx.env.change($('vd-speak-breeze_tts-2'), false);
    await ctx.settle();
    assert.match(text($('vd-summary')), /VersionsNone/);
    await click('vd-save');
    assert.deepEqual(ctx.calls('POST /api/voices/designed')[0].body.getAll('speak_with'), ['']);
    await ctx.go('voices/design');
    await describeVoice();
    await generate();
    await toSave();
    ctx.env.input($('vd-name'), 'Both');
    ctx.env.change($('vd-speak-chatterbox'), true);
    await ctx.settle();
    await click('vd-save');
    assert.deepEqual(ctx.calls('POST /api/voices/designed')[1].body.getAll('speak_with'), ['breeze/tts-2', 'chatterbox']);
  });

  it('Speak with defaults as the API does: the design model when it clones, else the installed cloning models of its provider', async () => {
    const { defaultSpeakWith, designForm } = await load('views/voices/design.js');
    const list = [
      { id: 'qwen3/1.7b-voicedesign', type: 'tts', provider: 'qwen3', state: 'loaded', capabilities: { voice_design: true } },
      { id: 'qwen3/1.7b-base', type: 'tts', provider: 'qwen3', state: 'downloaded', capabilities: { voice_clone: true } },
      { id: 'qwen3/0.6b-base', type: 'tts', provider: 'qwen3', state: 'available', capabilities: { voice_clone: true } },
      { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded', capabilities: { voice_clone: true } },
      { id: 'breeze/tts-2', type: 'tts', provider: 'breeze', state: 'loaded', capabilities: DESIGN_CAPS },
    ];
    assert.deepEqual(defaultSpeakWith('breeze/tts-2', list), ['breeze/tts-2']);
    assert.deepEqual(defaultSpeakWith('qwen3/1.7b-voicedesign', list), ['qwen3/1.7b-base'], 'installed only, same provider');
    const form = designForm({ sample: { model: 'm', description: 'd', text: 't', blob: new Blob([new Uint8Array(4)]), seed: null, options: {} } });
    assert.deepEqual([...form.keys()], ['model', 'description', 'transcript', 'audio'], 'a replace sends no name, seed, options or Speak with');
  });

  it('a name in use says so on the field, before saving and when the server refuses it', async () => {
    const data = designedRoutes();
    data.routes['POST /api/voices/designed'] = () => json(409, { detail: 'Voice name already exists' });
    await wizard({ routes: data.routes });
    await describeVoice();
    await generate();
    await toSave();
    ctx.env.input($('vd-name'), 'narrator');
    await ctx.settle();
    assert.equal(text($('vd-name-error')), 'A named voice with this name already exists. Choose another name.', 'matched ignoring case');
    assert.equal($('vd-name').getAttribute('aria-invalid'), 'true');
    assert.equal(text($('vd-save-hint')), 'A named voice with this name already exists. Choose another name.');
    assert.equal($('vd-save').getAttribute('aria-disabled'), 'true');
    ctx.env.click($('vd-save'));
    await ctx.settle();
    assert.equal(ctx.calls('POST /api/voices/designed').length, 0);
    ctx.env.input($('vd-name'), 'Somebody new');
    await ctx.settle();
    assert.equal($('vd-name').getAttribute('aria-invalid'), null);
    await click('vd-save');
    assert.equal(ctx.calls('POST /api/voices/designed').length, 1);
    assert.equal(text($('vd-name-error')), 'A named voice with this name already exists. Choose another name.', 'a 409 from the server too');
    assert.equal(document.activeElement?.id, 'vd-name');
    assert.equal(ctx.app.get().route.path, 'voices/design');
  });

  it('Edit description opens the wizard filled in; Replace PUTs the new sample in place, Save as new voice POSTs one with its own name', async () => {
    const data = designedRoutes([DESIGNED]);
    data.routes['PUT /api/voices/designed/:id'] = (req) => json(200, { ...DESIGNED, design: { ...DESIGNED.design, description: req.body.get('description') } });
    await wizard({ models: WITH_BREEZE_OPTIONS, routes: data.routes });
    await ctx.go('voices');
    ctx.env.click($('vo-named-n7-redesign'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices/design');
    assert.equal(text($('vd-title')), 'Edit Warm narrator');
    assert.equal(document.activeElement?.id, 'vd-title');
    assert.equal($('vd-description').value, 'A warm narrator');
    assert.equal($('vd-model').value, 'breeze/tts-2');
    assert.equal(text(ctx.$('#vd-options [data-option="cfg_scale"] output')), '2.5', 'its options');
    assert.equal($('vd-text').value, 'Words of the sample.', 'its sample text');
    ctx.env.input($('vd-description'), 'A warm narrator, a little slower');
    await ctx.settle();
    ctx.env.click($('vd-next-1'));
    await ctx.settle();
    await generate();
    assert.equal(ctx.calls('POST /api/voices/design')[0].json.seed, 1234, 'the first sample speaks with its seed');
    await toSave();
    assert.equal(text($('vd-step-3-title')), 'Replace Warm narrator, or save a new voice');
    assert.equal($('vd-replace').hidden, false);
    assert.equal(text($('vd-replace')), 'Replace Warm narrator');
    assert.equal($('vd-replace').getAttribute('aria-disabled'), null, 'Replace needs no name');
    assert.equal(text($('vd-save')), 'Save as new voice');
    assert.equal($('vd-save').getAttribute('aria-disabled'), 'true', 'a new voice needs its own name');
    ctx.env.click($('vd-replace'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.confirmText().title, 'Replace Warm narrator?');
    await ctx.answerConfirm(true);
    await ctx.settle();
    await ctx.settle();
    const put = ctx.calls('PUT /api/voices/designed/:id')[0];
    assert.equal(put.path, '/api/voices/designed/n7');
    assert.equal(put.body.get('name'), null);
    assert.equal(put.body.get('description'), 'A warm narrator, a little slower');
    assert.equal(put.body.get('transcript'), 'Words of the sample.');
    assert.equal(put.body.get('seed'), '1234');
    assert.deepEqual(JSON.parse(put.body.get('options')), { cfg_scale: 2.5, temperature: 0.9 });
    assert.equal(put.body.getAll('speak_with').length, 0, 'its versions stay');
    assert.ok(ctx.toasts().includes('Replaced Warm narrator'));
    assert.equal(ctx.app.get().route.path, 'voices');
    assert.equal(document.activeElement?.id, 'vo-named-n7-title');

    ctx.env.click($('vo-named-n7-redesign'));
    await ctx.settle();
    await ctx.settle();
    ctx.env.click($('vd-next-1'));
    await ctx.settle();
    await generate();
    await toSave();
    ctx.env.input($('vd-name'), 'Warm narrator two');
    await ctx.settle();
    await click('vd-save');
    const post = ctx.calls('POST /api/voices/designed')[0];
    assert.equal(post.body.get('name'), 'Warm narrator two');
    assert.deepEqual(post.body.getAll('speak_with'), ['breeze/tts-2']);
    assert.equal(document.activeElement?.id, 'vo-named-n8-title');
  });

  it('Design a voice after Edit description starts a new design', async () => {
    await wizard({ routes: designedRoutes([DESIGNED]).routes });
    await ctx.go('voices');
    ctx.env.click($('vo-named-n7-redesign'));
    await ctx.settle();
    await ctx.settle();
    assert.equal($('vd-description').value, 'A warm narrator');
    await ctx.go('voices');
    ctx.env.click($('vo-design'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(text($('vd-title')), 'Design a voice');
    assert.equal($('vd-description').value, '');
    assert.equal($('vd-text').value, SAMPLE_TEXT);
  });

  it('a link, Back or Forward to #voices/design after Edit description starts a new design', async () => {
    await wizard({ routes: designedRoutes([DESIGNED]).routes });
    await ctx.go('voices');
    ctx.env.click($('vo-named-n7-redesign'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(text($('vd-title')), 'Edit Warm narrator');
    await ctx.go('voices');
    await ctx.go('voices/design');
    assert.equal(text($('vd-title')), 'Design a voice');
    assert.equal($('vd-description').value, '');
    ctx.env.click(ctx.$('#vo-design-wizard .vo-back'));
    await ctx.settle();
    ctx.env.click($('vo-named-n7-redesign'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(text($('vd-title')), 'Edit Warm narrator', 'Edit description still fills it in');
    assert.equal($('vd-description').value, 'A warm narrator');
  });

  it('a completed step shows its check and says "done"; a step that can\'t be reached yet is unavailable', async () => {
    await wizard();
    const step = (n) => ctx.$(`#vo-design-wizard [data-step="${n}"]`);
    assert.equal(step(1).getAttribute('aria-current'), 'step');
    assert.equal(step(3).getAttribute('aria-disabled'), 'true');
    await describeVoice();
    assert.equal(step(2).getAttribute('aria-current'), 'step');
    assert.ok(step(1).closest('li').classList.contains('is-done'));
    assert.equal(text(step(1).querySelector('.vo-steps__done')), ', done');
    assert.equal(step(3).getAttribute('aria-disabled'), 'true', 'Name & save needs a sample');
    await generate();
    assert.equal(step(3).getAttribute('aria-disabled'), null);
    ctx.env.click(step(1));
    await ctx.settle();
    assert.equal($('vd-step-1').hidden, false);
  });
});
