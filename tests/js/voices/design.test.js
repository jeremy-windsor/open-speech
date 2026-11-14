// Voices: the 3-step design wizard at #voices/design (describe, listen, name and save), which saves a designed
// voice as a recording with its words as the transcript.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames } from '../harness/a11y.js';
import { voicesEnv, MODELS, recordings, json, bytes, load, delay } from './helpers.js';

let ctx;
afterEach(() => {
  ctx?.env.restore();
});

const SAMPLE_TEXT = 'Hello! This is the voice you described. I can read your stories, your notes and your messages aloud.';
const DESIGN_CAPS = { voice_design: true, voice_clone: true, clone_transcript_required: true, instructions: true };
const WITH_BREEZE = {
  ...MODELS,
  models: [...MODELS.models, { id: 'breeze/tts-2', type: 'tts', provider: 'breeze', state: 'loaded', device: 'external', provider_available: true, capabilities: DESIGN_CAPS }],
};
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

async function wizard({ models = WITH_BREEZE, routes = {}, ...options } = {}) {
  ctx = await voicesEnv({
    sub: 'design',
    models,
    routes: { 'POST /api/voices/design': () => bytes(200, wav(2), 'audio/wav'), ...routes },
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

describe('views/voices design wizard', () => {
  it('Design a voice opens the wizard with focus on its heading, the header buttons hide, and every control is named', async () => {
    ctx = await voicesEnv({ sub: 'recordings', models: WITH_BREEZE });
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

  it('the wizard is remembered as Recordings, and its back button returns there', async () => {
    await wizard();
    const prefs = await load('core/prefs.js');
    assert.equal(prefs.get().lastRoute, 'voices/recordings');
    ctx.env.click(ctx.$('#vo-design-wizard [data-action="wizard-close"]'));
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices/recordings');
  });

  it('Describe lists only the models that design a voice; Next needs a description and says so on the field', async () => {
    await wizard();
    const options = [...$('vd-model').querySelectorAll('option')].map((o) => [o.value, text(o)]);
    assert.deepEqual(options, [['breeze/tts-2', 'Breeze TTS 2']], 'Kokoro and Chatterbox cannot design a voice');
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
    ctx.env.click(ctx.$('#vo-design-wizard [data-action="wizard-close"]'));
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices/recordings');
    await ctx.settle(3500);
    await ctx.settle();
    assert.equal($('vd-audio').paused, true, 'nothing starts playing outside the wizard');
    assert.equal($('vd-sample').hidden, true);
    assert.equal(text($('vd-generate')), 'Generate sample', 'the wizard is not left busy');
  });

  it('the sample text is capped at 300 characters, so a saved sample stays within 30 seconds', async () => {
    await wizard();
    assert.equal($('vd-text').maxLength, 300);
  });

  it('Save checks the name, POSTs name, <name>.wav and the sample text as the transcript, then shows the card in Recordings', async () => {
    let library = recordings();
    await wizard({
      routes: {
        'GET /api/voices/library': () => json(200, library),
        'GET /api/voices/library/:name': () => json(404, { detail: "Voice 'x' not found" }),
        'POST /api/voices/library': (req) => {
          library = [...library, { name: req.body.get('name'), duration_s: 2, sample_rate: 24000, channels: 1, transcript: req.body.get('transcript') }];
          return json(201, { name: req.body.get('name') });
        },
      },
    });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await describeVoice();
    await generate();
    ctx.env.click($('vd-next-2'));
    await ctx.settle();
    assert.equal(document.activeElement?.id, 'vd-step-3-title');
    assert.equal($('vd-save').getAttribute('aria-disabled'), 'true', 'a name first');
    assert.equal(text($('vd-save-hint')), 'Enter a voice name with at least one letter or number.');
    ctx.env.input($('vd-name'), 'Warm Narrator');
    await ctx.settle();
    assert.equal(text($('vd-name-preview')), 'Saves as: warm_narrator');
    assert.match(text($('vd-summary')), /TranscriptHello! This is the voice you described/);
    ctx.env.click($('vd-save'));
    await ctx.settle();
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('GET /api/voices/library/:name')[0].path, '/api/voices/library/warm_narrator');
    const post = ctx.calls('POST /api/voices/library')[0];
    assert.equal(post.body.get('name'), 'warm_narrator');
    assert.equal(post.body.get('audio').name, 'warm_narrator.wav');
    assert.equal(post.body.get('audio').type, 'audio/wav');
    assert.equal(post.body.get('transcript'), SAMPLE_TEXT);
    assert.ok(ctx.toasts().includes('Saved recording warm_narrator'));
    assert.deepEqual(events, ['recordings']);
    assert.equal(ctx.app.get().route.path, 'voices/recordings');
    assert.equal(document.activeElement?.id, 'vo-rec-warm_narrator-title', 'the new card is focused');
  });

  it('an existing name asks before replacing it, and Cancel saves nothing', async () => {
    await wizard({ routes: { 'GET /api/voices/library/:name': () => json(200, { name: 'demo_reference' }), 'POST /api/voices/library': () => json(201, {}) } });
    await describeVoice();
    await generate();
    ctx.env.click($('vd-next-2'));
    await ctx.settle();
    ctx.env.input($('vd-name'), 'demo reference');
    await ctx.settle();
    assert.equal(text($('vd-name-preview')), 'Saves as: demo_reference · replaces the recording with this name');
    ctx.env.click($('vd-save'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.confirmText().title, 'Replace the existing recording “demo_reference”?');
    await ctx.answerConfirm(false);
    assert.equal(ctx.calls('POST /api/voices/library').length, 0);
    assert.equal(status(), 'Not saved. Choose another name to keep both.');
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
