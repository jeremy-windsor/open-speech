import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, mountPlayer, showReading, load, json, $, MODELS } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

const sheet = await load('views/reader/voice-sheet.js');
const MANY = [
  ...MODELS,
  { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded', provider_available: true },
  { id: 'piper/en_GB-alan-medium', type: 'tts', provider: 'piper', state: 'downloaded' },
];
const RECORDINGS = { 'GET /api/voices/library': () => json(200, [{ name: 'demo_reference', transcript: 'Hello there' }]) };

async function open(options = {}) {
  ctx = await setup({ shell: true, routes: RECORDINGS, ...options });
  await mountPlayer(ctx);
  ctx.engine.load(ctx.doc);
  await showReading(ctx);
  ctx.env.click($('rdp-voice'));
  await settle(ctx.env);
  return ctx;
}

const groups = () => [...document.querySelectorAll('#vp-list [role="group"]')].map((g) => document.getElementById(g.getAttribute('aria-labelledby'))?.textContent || '');

describe('reader-player/voice sheet', () => {
  it('[RE-64] models are grouped by provider with Kokoro first; the picker shows named voices and the provider voices, plus "Same as Speak" when Speak uses this model', async () => {
    assert.deepEqual(sheet.modelGroups(MANY).map((g) => [g.label, g.options.map((o) => o.id)]), [
      ['Kokoro', ['kokoro']],
      ['Piper', ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium']],
      ['Chatterbox', ['chatterbox']],
    ]);
    assert.deepEqual(sheet.sameAsSpeakRow({ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Bella + Sky' }, 'kokoro'),
      { id: 'same-as-speak', label: 'Same as Speak · Bella + Sky', detail: 'af_bella(2)+af_sky(1)', voice: 'af_bella(2)+af_sky(1)', group: 'Suggested' });
    assert.equal(sheet.sameAsSpeakRow({ model: 'piper/en_US-amy-medium', voice: 'x' }, 'kokoro'), null);
    await open({ models: MANY });
    const select = $('rdp-voice-model');
    assert.deepEqual([...select.querySelectorAll('optgroup')].map((g) => g.getAttribute('label')), ['Kokoro', 'Piper', 'Chatterbox']);
    assert.equal(select.value, 'kokoro');
    assert.equal($('rdp-voice-name').textContent, 'Kokoro · Heart');
    assert.equal($('rdp-voice-avatar').textContent, 'HE', "the voice's own initials, not the model's");
    assert.equal($('rdp-voice-detail').textContent, 'American English · female', 'the dot never starts a wrapped line on a phone');
    ctx.app.set({ speakVoice: { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Bella + Sky' } });
    ctx.env.click($('rdp-voice-choose'));
    await settle(ctx.env);
    assert.equal($('app-voice-picker').open, true);
    assert.deepEqual(groups().slice(0, 3), ['Suggested', 'Named voices', 'Kokoro · Recommended']);
    const same = [...document.querySelectorAll('#vp-list [role="option"]')].find((o) => o.textContent.includes('Same as Speak'));
    assert.ok(same, 'the Same as Speak row');
    ctx.env.click(same);
    await settle(ctx.env);
    assert.equal($('app-voice-picker').open, false);
    assert.equal($('rdp-voice-name').textContent, 'Bella + Sky', "Speak's own label");
    assert.equal($('rdp-voice-detail').textContent, 'af_bella(2)+af_sky(1)');
    assert.equal($('rdp-voice-detail').classList.contains('mono'), true, 'a blend shows its canonical string in mono');
    ctx.env.click($('rdp-voice-use'));
    await settle(ctx.env);
    assert.equal($('rdp-voice-sheet').open, false);
    assert.deepEqual(ctx.engine.state.voice, { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)' });
    // Another model: no Same as Speak row, and no voice chosen until one is picked.
    ctx.env.click($('rdp-voice'));
    ctx.env.change($('rdp-voice-model'), 'piper/en_US-amy-medium');
    assert.equal($('rdp-voice-name').textContent, 'No voice chosen');
    ctx.env.click($('rdp-voice-choose'));
    await settle(ctx.env);
    assert.equal(groups().includes('Suggested'), false);
  });

  it('[REV-09] raw recordings are never offered: the picker leaves them out (Reader renders send only a model and a voice) and the hint points to named voices', async () => {
    assert.deepEqual({ ...sheet.PICKER_INCLUDE }, { named: true, recordings: false, voices: true });
    await open();
    ctx.env.click($('rdp-voice-choose'));
    await settle(ctx.env);
    assert.equal(groups().includes('Recordings'), false);
    assert.equal(ctx.env.fetch.callsTo('GET /api/voices/library').length, 0, 'recordings are not even fetched');
    assert.ok(![...document.querySelectorAll('#vp-list [role="option"]')].some((o) => o.textContent.includes('demo_reference')));
    assert.match($('rdp-voice-sheet').textContent, /To read with a cloned voice, save it as a named voice in Voices\./);
    assert.match($('rdp-voice-sheet').textContent, /Audio renders on the server at normal speed; speed changes playback only\./);
    // A named voice reaches the Reader as voice:<id>.
    const named = [...document.querySelectorAll('#vp-list [role="option"]')].find((o) => o.textContent.includes('Narrator'));
    ctx.env.click(named);
    await settle(ctx.env);
    ctx.env.click($('rdp-voice-use'));
    await startPlaying(ctx);
    assert.equal(ctx.server.renders.at(-1).voice, 'voice:n1');
    assert.ok(playingEl());
  });

  it('[REV-09] "Same as Speak" is not offered while Speak reads with a raw recording (its name would be sent as a voice), nor used as the fallback voice', async () => {
    // What Speak publishes for a clone model reading with its reference recording alone (kind).
    const recording = { model: 'chatterbox', voice: 'demo_reference', label: 'demo_reference', kind: 'recording' };
    assert.equal(sheet.sameAsSpeakRow(recording, 'chatterbox'), null);
    assert.equal(sheet.sameAsSpeakRow({ model: 'chatterbox', voice: 'voice:n1', label: 'Narrator', kind: 'named' }, 'chatterbox').voice, 'voice:n1', 'a named voice is fine');
    assert.equal(sheet.sameAsSpeakRow({ ...recording, kind: 'voice' }, 'chatterbox').voice, 'demo_reference', "Speak's kind decides: a stock voice of that name is a voice");
    await open({ models: MANY, seed: {} });   // no Reader voice yet: Speak's would be the fallback
    ctx.app.set({ speakVoice: recording });
    ctx.env.change($('rdp-voice-model'), 'chatterbox');
    ctx.env.click($('rdp-voice-choose'));
    await settle(ctx.env);
    assert.equal($('app-voice-picker').open, true);
    assert.equal(groups().includes('Suggested'), false, 'no "Same as Speak" row');
    assert.ok(![...document.querySelectorAll('#vp-list [role="option"]')].some((o) => o.textContent.includes('demo_reference')));
    ctx.env.key('Escape');
    await settle(ctx.env);
    ctx.env.key('Escape');
    await settle(ctx.env);
    // Nor is it the voice Play falls back to: the Reader asks instead of sending the recording's name.
    assert.equal(ctx.engine.state.voice?.voice || '', '', "Speak's model, but no voice from it");
    assert.equal($('rdp-voice-label').textContent, 'Voice');
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal(ctx.server.renders.length, 0);
    assert.equal($('rdp-voice-sheet').open, true, 'it asks for a voice');
  });

  // The Reader's picker shows the cloning-model note too, as Speak's does.
  // The Reader offers no raw recordings, so the note is its first sentence, without Clone a voice….
  it('a model with no voices of its own: the note says so, the named voices with a version for it come first and the others last, disabled', async () => {
    const routes = {
      ...RECORDINGS,
      'GET /v1/audio/voices': (req) => json(200, { voices: req.query.get('model') === 'chatterbox' ? [] : [{ id: 'af_heart' }] }),
      'GET /api/voices/identities': () => json(200, { voices: [
        { id: 'n1', name: 'Narrator', available: false, realizations: [{ model: 'kokoro', voice: 'af_bella', available: false, reason: "Select this realization's model" }] },
        { id: 'b1', name: 'Butler', available: true, realizations: [{ model: 'chatterbox', voice: 'butler_take', available: true }] },
      ] }),
    };
    await open({ models: MANY, routes });
    ctx.env.change($('rdp-voice-model'), 'chatterbox');
    ctx.env.click($('rdp-voice-choose'));
    await settle(ctx.env);
    await settle(ctx.env);
    assert.equal($('app-voice-picker').open, true);
    assert.equal($('vp-model-note').hidden, false, 'the note shows');
    assert.equal($('vp-model-note-text').textContent, 'Chatterbox has no built-in voices.');
    assert.equal($('vp-clone').hidden, true, 'no raw recordings here: no Clone a voice…');
    assert.deepEqual(groups(), ['Named voices', 'Named voices with no Chatterbox version']);
    const narrator = [...document.querySelectorAll('#vp-list [role="option"]')].find((o) => o.textContent.includes('Narrator'));
    assert.equal(narrator.getAttribute('aria-disabled'), 'true');
    assert.match(narrator.textContent, /No Chatterbox version/);
    // Kokoro, which has voices of its own: no note, the usual order.
    ctx.env.key('Escape');
    await settle(ctx.env);
    ctx.env.change($('rdp-voice-model'), 'kokoro');
    ctx.env.click($('rdp-voice-choose'));
    await settle(ctx.env);
    await settle(ctx.env);
    assert.equal($('vp-model-note').hidden, true);
  });
});
