// Hand-offs into Speak (inventory HI-09; SE-03 for the edit-named-voice intent).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { historySetup } from '../../../src/static/js/views/speak/presets.js';
import { mountSpeak, ready, settle, callsTo, answerConfirm, json, delay, CAPS as CAPS_FOR_TEST } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const TTS_IDS = ['kokoro', 'piper/en_US-amy-medium', 'chatterbox'];
const notice = () => t.$('sp-notice');

function fakeLive() {
  const stops = [];
  t.ctx.live = { isActive: () => true, stop: (reason) => stops.push(reason), destroy() {}, refresh() {} };
  return stops;
}

async function handOff(intent) {
  t.setIntent('speak', intent);
  t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'heading' });
  await settle(t.env);
  await settle(t.env);
  await settle(t.env);
}

describe('views/speak history hand-off (pure)', () => {
  it('[HI-09] an entry becomes a setup: refusals for other types and missing models, the meta honoured', () => {
    assert.deepEqual(historySetup({ type: 'stt' }, { ttsIds: TTS_IDS }), { error: 'Only speech entries can be re-generated.' });
    assert.deepEqual(historySetup({ type: 'tts', model: 'xtts' }, { ttsIds: TTS_IDS }), { error: "This entry's model XTTS v2 is unavailable." });
    const h = historySetup({
      type: 'tts', model: 'chatterbox', voice: 'Ryan', speed: 1.2, format: 'wav', full_text: 'Hello',
      meta: { reference_voice: 'demo_reference', instructions: 'calm', effects: [{ type: 'robot' }] },
    }, { ttsIds: TTS_IDS });
    assert.equal(h.text, 'Hello');
    assert.deepEqual(h.setup, { model: 'chatterbox', voice: 'Ryan', identity: '', reference: 'demo_reference', instructions: 'calm', speed: 1.2, format: 'wav', effects: [{ type: 'robot' }] });
    assert.equal(h.referenceFallback, null);
    const old = historySetup({ type: 'tts', model: 'chatterbox', voice: 'demo_reference', full_text: 'x' }, { ttsIds: TTS_IDS });
    assert.equal(old.setup.reference, undefined, 'an older entry has no meta');
    assert.equal(old.referenceFallback, 'demo_reference');
    assert.equal(old.setup.instructions, '', 'never what is typed now');
    assert.equal('effects' in old.setup, false, 'effects only when recorded');
    const named = historySetup({ type: 'tts', model: 'kokoro', voice: 'voice:n1', full_text: 'x' }, { ttsIds: TTS_IDS });
    assert.equal(named.setup.identity, 'n1');
    const steps = historySetup({ type: 'tts', model: 'kokoro', voice: 'M1', full_text: 'x', meta: { steps: 16 } }, { ttsIds: TTS_IDS });
    assert.deepEqual(steps.setup.options, { steps: 16 }, "Supertonic's quality steps");
    assert.equal('options' in old.setup, false);
  });
});

describe('views/speak hand-offs (view)', () => {
  it('[HI-09] Open in Speak restores text, model, voice (a blend rebuilt), speed and format, clears the preset and stops Live', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.click(t.$('sp-presets').querySelector('[data-preset-id="p1"]'));
    await settle(t.env);
    await settle(t.env);
    const stops = fakeLive();
    t.env.input(t.$('sp-input'), 'Typed before.');
    await handOff({
      type: 'apply-history',
      entry: { id: 'h1', type: 'tts', model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', speed: 1.3, format: 'flac', full_text: 'Down the rabbit hole.', meta: { effects: [{ type: 'normalize' }] } },
    });
    assert.deepEqual(stops, ['Text changed']);
    assert.equal(t.$('sp-input').value, 'Down the rabbit hole.');
    assert.match(t.$('sp-counter').textContent, /^21 \//);
    assert.equal(t.parts.inspector.voiceSpec(), 'af_bella(2)+af_sky(1)');
    assert.equal(t.$('sp-blend').hidden, false, 'the blend is rebuilt in the editor');
    assert.equal(t.parts.inspector.state().speed, 1.3);
    assert.equal(t.$('sp-format').value, 'flac');
    assert.equal(t.$('sp-effects').querySelector('[data-effect="normalize"]').getAttribute('aria-pressed'), 'true');
    assert.equal(t.$('sp-presets').querySelector('[data-preset-id="p1"]').getAttribute('aria-pressed'), 'false');
  });

  it('[HI-09] a clone entry restores its reference and instructions from meta; a missing reference is cleared with a banner', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-model'), 'chatterbox');
    await settle(t.env);
    await settle(t.env);
    t.env.input(t.$('sp-instructions'), 'Typed now');
    await handOff({
      type: 'apply-history',
      entry: { id: 'h2', type: 'tts', model: 'chatterbox', voice: 'demo_reference', full_text: 'Clone', meta: { reference_voice: 'demo_reference', instructions: 'Slow' } },
    });
    assert.equal(t.$('sp-reference').value, 'demo_reference');
    assert.equal(t.$('sp-instructions').value, 'Slow', "the entry's instructions, never what is typed now");
    await handOff({
      type: 'apply-history',
      entry: { id: 'h3', type: 'tts', model: 'chatterbox', voice: 'gone_take', full_text: 'Old', meta: { reference_voice: 'gone_take' } },
    });
    assert.equal(t.$('sp-reference').value, '');
    assert.equal(notice().querySelector('.banner__msg').textContent, 'The reference recording gone_take this entry used is missing. Choose one before generating.');
    assert.equal(t.$('sp-instructions').value, '', 'no meta instructions: empty, never the typed ones');
    // An older entry without meta: on a clone model with no stock voices, its voice was the reference.
    await handOff({ type: 'apply-history', entry: { id: 'h4', type: 'tts', model: 'chatterbox', voice: 'raw_take', full_text: 'Older' } });
    assert.equal(t.$('sp-reference').value, 'raw_take');
  });

  it('[HI-09] other entry types and unavailable models are refused with a banner and change nothing', async () => {
    t = await mountSpeak();
    await ready(t);
    const stops = fakeLive();
    t.env.input(t.$('sp-input'), 'Keep me.');
    await handOff({ type: 'apply-history', entry: { id: 's1', type: 'stt', full_text: 'A transcript' } });
    assert.equal(notice().querySelector('.banner__msg').textContent, 'Only speech entries can be re-generated.');
    assert.equal(t.$('sp-input').value, 'Keep me.');
    await handOff({ type: 'apply-history', entry: { id: 'h9', type: 'tts', model: 'xtts', voice: 'x', full_text: 'Gone' } });
    assert.equal(notice().querySelector('.banner__msg').textContent, "This entry's model XTTS v2 is unavailable.");
    assert.equal(t.$('sp-input').value, 'Keep me.');
    assert.equal(t.$('sp-model').value, 'kokoro');
    assert.deepEqual(stops, []);
  });

  it('[HI-09] an entry whose model fails to load while a model change still loads says so, and the change carries on', async () => {
    let failClone = false;
    t = await mountSpeak({
      routes: {
        'GET /api/tts/capabilities': (req) => {
          const model = req.query.get('model');
          if (model === 'chatterbox' && failClone) return json(503, { error: { message: 'Worker busy' } });
          const body = json(200, (CAPS_FOR_TEST[model]) || { capabilities: {} });
          return model === 'piper/en_US-amy-medium' ? delay(2000, body) : body;
        },
      },
    });
    await ready(t);
    failClone = true;
    t.env.change(t.$('sp-model'), 'piper/en_US-amy-medium');
    await settle(t.env);
    await handOff({ type: 'apply-history', entry: { id: 'h', type: 'tts', model: 'chatterbox', voice: 'demo_reference', full_text: 'x', meta: {} } });
    assert.equal(notice().querySelector('.banner__title').textContent, "Couldn't load this entry's model");
    await t.env.clock.tickAsync(2000);
    await settle(t.env);
    await settle(t.env);
    const s = t.parts.inspector.state();
    assert.equal(s.model, 'piper/en_US-amy-medium');
    assert.equal(s.data.model, 'piper/en_US-amy-medium');
    assert.equal(t.parts.inspector.ready(), true, 'never stuck on "Loading voices…"');
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'false');
  });

  it('[HI-09] the hand-off waits for the first setup, so a fresh page applies the entry over the Default preset', async () => {
    t = await mountSpeak({
      loaded: false,
      beforeShow: ({ setIntent }) => setIntent('speak', { type: 'apply-history', entry: { id: 'h1', type: 'tts', model: 'piper/en_US-amy-medium', voice: 'en_US-amy-medium', full_text: 'Early', speed: 1 } }),
    });
    const { modelsPayload } = await import('./helpers.js');
    const models = modelsPayload();
    t.app.set({ models: { list: models.models, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
    await ready(t);
    await settle(t.env);
    assert.equal(t.$('sp-model').value, 'piper/en_US-amy-medium');
    assert.equal(t.$('sp-input').value, 'Early');
  });

  it('[SE-03] Edit in Speak selects the model, applies the named voice\'s version and explains how to update it', async () => {
    t = await mountSpeak();
    await ready(t);
    const stops = fakeLive();
    await handOff({ type: 'edit-named-voice', id: 'n1', model: 'kokoro' });
    assert.deepEqual(stops, ['Voice changed']);
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    // The editing bar explains it, with Update voice (tests/js/speak/editing.test.js).
    assert.equal(t.$('sp-editing-title').textContent, 'Editing Narrator for Kokoro');
    assert.equal(t.$('sp-editing-hint').textContent, 'Change the voice, blend or recording, then Update voice. Presets that use it follow.');
    assert.equal(t.$('sp-editing').getAttribute('role'), 'group', 'a bar, not an alert');
    // Editing the blend clears the selection, but Save named voice still offers the voice being edited.
    t.env.click(t.$('sp-blend-equal'));
    await settle(t.env);
    t.env.click(t.$('sp-save-voice'));
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt-input').value, 'Narrator');
    t.env.click(document.getElementById('app-prompt-cancel'));
    await settle(t.env);
    await handOff({ type: 'edit-named-voice', id: 'n1', model: 'xtts' });
    assert.equal(notice().querySelector('.banner__msg').textContent, 'This model is unavailable.');
  });

  it('use-recording selects the clone model, sets the reference and focuses the editor', async () => {
    t = await mountSpeak();
    await ready(t);
    await handOff({ type: 'use-recording', name: 'demo_reference', model: 'chatterbox' });
    assert.equal(t.$('sp-model').value, 'chatterbox');
    assert.equal(t.$('sp-reference').value, 'demo_reference');
    assert.equal(document.activeElement, t.$('sp-input'));
    t.env.input(t.$('sp-input'), 'Cloned voice.');
    t.env.click(t.$('sp-generate'));
    await answerConfirm(t, true);
    assert.equal(callsTo(t, 'POST /v1/audio/speech')[0].json.voice_library_ref, 'demo_reference');
  });
});
