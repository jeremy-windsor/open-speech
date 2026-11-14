// Presets and named voices (inventory SP-30…SP-35; TE-09).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { presetPayload, presetProblem, presetSetup, presetNameProblem } from '../../../src/static/js/views/speak/presets.js';
import { mountSpeak, ready, settle, callsTo, json, PRESETS, NAMED, answerConfirm, load } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const chip = (id) => t.$('sp-presets').querySelector(`[data-preset-id="${id}"]`);
const notice = () => t.$('sp-notice');

function fakeLive() {
  const stops = [];
  t.ctx.live = { isActive: () => true, stop: (reason) => stops.push(reason), destroy() {}, refresh() {} };
  return stops;
}

async function applyChip(id) {
  t.env.click(chip(id));
  await settle(t.env);
  await settle(t.env);
  await settle(t.env);
}

/** Answer the shell's prompt (#app-prompt) with a value. */
async function answerPrompt(value) {
  await settle(t.env);
  assert.equal(document.getElementById('app-prompt').open, true, 'the prompt is open');
  document.getElementById('app-prompt-input').value = value;
  document.getElementById('app-prompt-form').requestSubmit();
  await settle(t.env);
  await settle(t.env);
}

function presetsWith(extra, defaultId = null) {
  return { profiles: [...PRESETS.profiles, ...extra], default_profile_id: defaultId };
}

describe('views/speak/presets (pure)', () => {
  it('[SP-35] the preset payload is today\'s shape', () => {
    assert.deepEqual(presetPayload(' Night reading ', {
      model: 'kokoro', voice: 'af_bella', blend: 'af_bella(2)+af_sky(1)', identity: '', reference: '', speed: 1.1, format: 'mp3',
      effects: [{ type: 'normalize' }], instructions: '  ',
    }), {
      name: 'Night reading', backend: 'kokoro', provider: 'kokoro', model: 'kokoro', voice: 'af_bella', speed: 1.1, format: 'mp3',
      blend: 'af_bella(2)+af_sky(1)', reference_audio_id: null, voice_identity_id: null, instructions: null, effects: [{ type: 'normalize' }],
      options: {},
    });
    assert.deepEqual(presetPayload('x', { model: 'supertonic', voice: 'M1', options: { steps: 4 } }).options, { steps: 4 });
    assert.equal(presetPayload('x', { model: 'piper/en_US-amy-medium', reference: 'r1' }).backend, 'piper');
    assert.equal(presetPayload('x', { model: 'chatterbox', reference: 'r1' }).voice, 'r1', 'a clone model with only a reference');
    assert.equal(presetNameProblem(''), 'Enter a name for the preset.');
    assert.equal(presetNameProblem('x'.repeat(101)), 'Use 100 characters or fewer.');
    assert.equal(presetNameProblem('x'.repeat(100)), null);
  });

  it('[SP-33] the checks run in today\'s order: model, reference, named voice, speed, instructions', () => {
    const ttsIds = ['kokoro', 'chatterbox'];
    assert.match(presetProblem({ model: 'xtts', reference_audio_id: 'gone' }, { ttsIds, referenceMissing: true }), /^the model XTTS v2 is unavailable$/);
    assert.match(presetProblem({ model: 'kokoro', reference_audio_id: 'gone', voice_identity_id: 'n2' }, { ttsIds, referenceMissing: true, named: NAMED.voices }), /reference recording gone is missing/);
    assert.match(presetProblem({ model: 'kokoro', voice_identity_id: 'n2', speed: 1.2 }, { ttsIds, named: NAMED.voices, caps: { speed_control: false } }), /named voice is unavailable/);
    assert.match(presetProblem({ model: 'chatterbox', speed: 1.2, instructions: 'x' }, { ttsIds, caps: { speed_control: false } }), /fixed speed/);
    assert.match(presetProblem({ model: 'kokoro', instructions: 'calm' }, { ttsIds, caps: { instructions: false } }), /doesn't take instructions/);
    assert.equal(presetProblem({ model: 'kokoro', speed: 1.2 }, { ttsIds, caps: { speed_control: true } }), '');
    assert.deepEqual(presetSetup({ backend: 'kokoro', voice: 'af_sky', speed: 1.2, effects: null }), {
      model: 'kokoro', voice: 'af_sky', identity: '', reference: '', speed: 1.2, format: 'mp3', effects: [], instructions: '',
      options: {},
    });
    assert.deepEqual(presetSetup({ model: 'supertonic', voice: 'M1', options: { steps: 16 } }).options, { steps: 16 });
  });
});

describe('views/speak presets and named voices (view)', () => {
  it('[SP-33] applying a preset resolves it, then changes model, voice, speed, format and effects together and stops Live', async () => {
    t = await mountSpeak();
    await ready(t);
    const stops = fakeLive();
    t.env.fetch.reset();
    await applyChip('p2');
    assert.equal(callsTo(t, 'GET /api/profiles/:id/resolve')[0].params.id, 'p2');
    assert.deepEqual(stops, ['Preset changed']);
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'true');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_heart');
    assert.equal(t.$('sp-effects').querySelector('[data-effect="podcast_eq"]').getAttribute('aria-pressed'), 'true');
    await applyChip('p1');
    assert.equal(chip('p1').getAttribute('aria-pressed'), 'true');
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'false');
    assert.equal(t.parts.inspector.state().speed, 1.1);
    assert.equal(t.$('sp-effects').querySelector('[data-effect="podcast_eq"]').getAttribute('aria-pressed'), 'false', 'the preset\'s effects replace the old ones');
    // Any manual change clears it.
    t.env.change(t.$('sp-format'), 'wav');
    await settle(t.env);
    assert.equal(chip('p1').getAttribute('aria-pressed'), 'false');
  });

  it('[SP-33] applying a preset from the keyboard keeps the focus on its chip (the row is never rebuilt for it)', async () => {
    t = await mountSpeak();
    await ready(t);
    const p2 = chip('p2');
    p2.focus();
    t.env.click(p2);
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    assert.equal(p2.isConnected, true, 'the same chip');
    assert.equal(p2.getAttribute('aria-pressed'), 'true');
    assert.equal(document.activeElement, p2, 'the focus stays on it, never <body>');
    const p1 = chip('p1');
    p1.focus();
    t.env.click(p1);
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    assert.equal(document.activeElement, p1);
    assert.equal(p2.getAttribute('aria-pressed'), 'false');
  });

  it('[SP-32] a refresh that brings the same presets rebuilds nothing; one that changes them keeps the focus on the same chip', async () => {
    const presets = presetsWith([]);
    t = await mountSpeak({ presets });
    await ready(t);
    const { emit } = await load('core/store.js');
    const p2 = chip('p2');
    p2.focus();
    emit('voices:changed', { what: 'presets' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(chip('p2'), p2, 'the same element');
    assert.equal(document.activeElement, p2);
    presets.profiles.push({ id: 'p5', name: 'Evening', backend: 'kokoro', model: 'kokoro', voice: 'af_sky', speed: 1, format: 'mp3', effects: [] });
    emit('voices:changed', { what: 'presets' });
    await settle(t.env);
    await settle(t.env);
    assert.ok(chip('p5'), 'the new preset shows');
    assert.equal(document.activeElement, chip('p2'), 'the focus follows its chip into the new row');
  });

  it('[SP-33] a preset that can\'t be used says why and changes nothing: model, reference, named voice, speed, instructions', async () => {
    const extra = [
      { id: 'pm', name: 'Old XTTS', backend: 'xtts', model: 'xtts', voice: 'v', speed: 1, format: 'mp3', effects: [] },
      { id: 'pr', name: 'Lost clone', backend: 'chatterbox', model: 'chatterbox', voice: '', reference_audio_id: 'gone', speed: 1, format: 'wav', effects: [] },
      { id: 'pn', name: 'Old narrator', backend: 'kokoro', model: 'kokoro', voice: 'af_sky', voice_identity_id: 'n2', speed: 1, format: 'mp3', effects: [] },
      { id: 'ps', name: 'Fast clone', backend: 'chatterbox', model: 'chatterbox', voice: '', reference_audio_id: 'demo_reference', speed: 1.3, format: 'wav', effects: [] },
      { id: 'pi', name: 'Directed', backend: 'kokoro', model: 'kokoro', voice: 'af_sky', instructions: 'Whisper', speed: 1, format: 'mp3', effects: [] },
    ];
    t = await mountSpeak({ presets: presetsWith(extra) });
    await ready(t);
    const stops = fakeLive();
    const before = JSON.stringify(t.parts.inspector.currentSetup());
    const cases = [
      ['pm', 'Preset “Old XTTS” can\'t be used', 'The model XTTS v2 is unavailable.'],
      ['pr', 'Preset “Lost clone” can\'t be used', 'The reference recording gone is missing.'],
      ['pn', 'Preset “Old narrator” can\'t be used', 'Its named voice is unavailable for this model.'],
      ['ps', 'Preset “Fast clone” can\'t be used', 'Chatterbox has a fixed speed.'],
      ['pi', 'Preset “Directed” can\'t be used', "Kokoro doesn't take instructions."],
    ];
    for (const [id, title, message] of cases) {
      await applyChip(id);
      assert.equal(notice().hidden, false, id);
      assert.equal(notice().querySelector('.banner__title').textContent, title);
      assert.equal(notice().querySelector('.banner__msg').textContent, message);
      assert.equal(notice().querySelector('[role="alert"]') !== null, true);
      assert.equal(JSON.stringify(t.parts.inspector.currentSetup()), before, `${id} changed nothing (no fallback to Kokoro)`);
      assert.equal(chip(id).getAttribute('aria-pressed'), 'false');
    }
    assert.deepEqual(stops, [], 'Live keeps running when nothing changes');
  });

  it('[SP-33] a preset the server can\'t resolve shows its message, and Details carry the message and its code, never the JSON envelope', async () => {
    t = await mountSpeak({
      routes: {
        'GET /api/profiles/:id/resolve': (req) => (req.params.id === 'p1'
          ? json(500, { error: { message: 'Profile store is locked', code: 'profile_error' } })
          : json(404, { error: { message: 'Profile not found', code: 'http_error' } })),
      },
    });
    await ready(t);
    const before = JSON.stringify(t.parts.inspector.currentSetup());
    await applyChip('p1');
    assert.equal(notice().querySelector('.banner__title').textContent, 'Preset “Audiobook 1.1×” can\'t be used');
    assert.equal(notice().querySelector('.banner__msg').textContent, 'Profile store is locked.');
    assert.equal(notice().querySelector('.banner__details pre').textContent, 'Profile store is locked (profile_error)');
    assert.ok(!notice().textContent.includes('"error"'), 'never the JSON envelope');
    // The server's generic code adds nothing to the message: no Details then.
    await applyChip('p2');
    assert.equal(notice().querySelector('.banner__msg').textContent, 'Profile not found.');
    assert.equal(notice().querySelector('.banner__details'), null);
    assert.equal(JSON.stringify(t.parts.inspector.currentSetup()), before, 'nothing changed');
  });

  it('[SP-32] refreshing the presets keeps the applied one; a deleted preset falls back to none', async () => {
    const presets = presetsWith([]);
    t = await mountSpeak({ presets });
    await ready(t);
    await applyChip('p1');
    const { emit } = await load('core/store.js');
    emit('voices:changed', { what: 'presets' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(chip('p1').getAttribute('aria-pressed'), 'true');
    presets.profiles.splice(0, 1);
    emit('voices:changed', { what: 'presets' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(chip('p1'), null);
    assert.ok([...t.$('sp-presets').querySelectorAll('[aria-pressed="true"]')].length === 0);
  });

  it('[SP-34] the Default preset applies once per page session on load, over the last setup', async () => {
    t = await mountSpeak({ presets: presetsWith([], 'p2'), seed: { 'open-speech-speak': JSON.stringify({ model: 'kokoro', voice: 'am_adam', speed: 1.5, format: 'wav' }) } });
    await ready(t);
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'true');
    assert.equal(chip('p2').querySelector('.visually-hidden').textContent, ', Default', '"Default" is in the chip\'s name');
    assert.ok(chip('p2').querySelector('svg use').getAttribute('href').endsWith('#i-star-fill'));
    assert.equal(t.parts.inspector.voiceSpec(), 'af_heart');
    // The listener changes something; showing the view again never re-applies it.
    t.env.change(t.$('sp-format'), 'flac');
    t.ctl.hide();
    await t.env.clock.tickAsync(31000);
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'heading' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-format').value, 'flac');
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'false');
    assert.equal(callsTo(t, 'GET /api/profiles/:id/resolve').length, 1, 'applied once');
  });

  it('[SP-34] never over a preset the listener picked (an apply-preset hand-off at load)', async () => {
    t = await mountSpeak({
      presets: presetsWith([], 'p2'),
      beforeShow: ({ setIntent }) => setIntent('speak', { type: 'apply-preset', id: 'p1' }),
    });
    await ready(t);
    await settle(t.env);
    assert.equal(chip('p1').getAttribute('aria-pressed'), 'true');
    assert.deepEqual(callsTo(t, 'GET /api/profiles/:id/resolve').map((c) => c.params.id), ['p1'], 'the Default preset was not applied');
  });

  it('[SP-34] after a key fixes a refused load, the Default preset applies; a failing Default says so and keeps the last setup', async () => {
    let refuse = true;
    t = await mountSpeak({ presets: presetsWith([], 'p1'), routes: { 'GET /api/profiles': () => (refuse ? json(401, { error: { message: 'Invalid API key' } }) : json(200, presetsWith([], 'p1'))) } });
    await ready(t);
    assert.equal(callsTo(t, 'GET /api/profiles/:id/resolve').length, 0);
    refuse = false;
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'keep', scrolled: true, reason: 'auth' });
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    assert.equal(chip('p1').getAttribute('aria-pressed'), 'true');
    t.env.restore();

    const broken = [{ id: 'pb', name: 'Broken', backend: 'xtts', model: 'xtts', voice: 'v', speed: 1, format: 'mp3', effects: [] }];
    t = await mountSpeak({ presets: presetsWith(broken, 'pb'), seed: { 'open-speech-speak': JSON.stringify({ model: 'kokoro', voice: 'am_adam', speed: 1.5 }) } });
    await ready(t);
    assert.equal(notice().querySelector('.banner__title').textContent, "The default preset wasn't applied");
    assert.equal(notice().querySelector('.banner__msg').textContent, 'The model XTTS v2 is unavailable.');
    assert.equal(notice().querySelector('.banner').getAttribute('role'), 'status', 'a page-load failure is polite, not an alert');
    assert.equal(t.parts.inspector.voiceSpec(), 'am_adam', 'the last setup stays');
    assert.equal(t.parts.inspector.state().speed, 1.5);
    // A preset the listener picks and that can't be used is a failure after an action: an alert.
    t.env.click(chip('pb'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(notice().querySelector('.banner__title').textContent, 'Preset “Broken” can\'t be used');
    assert.equal(notice().querySelector('.banner').getAttribute('role'), 'alert');
  });

  it('[SP-35] Save preset… asks the name first, posts today\'s payload, re-asks on a 409, then selects the new preset', async () => {
    let conflict = true;
    const created = [];
    const presets = presetsWith([]);
    t = await mountSpeak({
      presets,
      routes: {
        'POST /api/profiles': (req) => {
          if (conflict) {
            conflict = false;
            return json(409, { detail: 'Profile name already exists' });
          }
          const row = { id: 'p9', ...req.json };
          created.push(req.json);
          presets.profiles.push(row);
          return json(201, row);
        },
      },
    });
    await ready(t);
    const changes = [];
    const { on } = await load('core/store.js');
    on('voices:changed', (d) => changes.push(d));
    t.env.fetch.reset();
    t.env.click(t.$('sp-presets').querySelector('[data-preset-act="save"]'));
    await settle(t.env);
    assert.equal(callsTo(t, 'POST /api/profiles').length, 0, 'nothing is built or sent before the name');
    assert.equal(document.getElementById('app-prompt-title').textContent, 'Save preset');
    assert.equal(document.getElementById('app-prompt-label').textContent, 'Preset name');
    await answerPrompt('Audiobook 1.1×');
    assert.equal(document.getElementById('app-prompt').open, true, 'the prompt opens again');
    assert.equal(document.getElementById('app-prompt-error').textContent, 'A preset with this name already exists');
    assert.equal(document.getElementById('app-prompt-input').value, 'Audiobook 1.1×');
    await answerPrompt('Evening');
    await settle(t.env);
    assert.equal(created.length, 1);
    const s = t.parts.inspector.state();
    assert.deepEqual(created[0], {
      name: 'Evening', backend: 'kokoro', provider: 'kokoro', model: 'kokoro', voice: s.voice, speed: s.speed, format: s.format,
      blend: null, reference_audio_id: null, voice_identity_id: null, instructions: null, effects: [], options: {},
    });
    assert.deepEqual(changes, [{ what: 'presets' }]);
    assert.equal(chip('p9').getAttribute('aria-pressed'), 'true');
  });

  it('[SP-35] saving from the row\'s "Save preset…" chip gives the focus back to that chip once the new preset shows', async () => {
    const presets = presetsWith([]);
    t = await mountSpeak({
      presets,
      routes: {
        'POST /api/profiles': (req) => {
          const row = { id: 'p9', ...req.json };
          presets.profiles.push(row);
          return json(201, row);
        },
      },
    });
    await ready(t);
    const save = t.$('sp-presets').querySelector('[data-preset-act="save"]');
    save.focus();
    t.env.click(save);
    await answerPrompt('Evening');
    await settle(t.env);
    await settle(t.env);
    assert.equal(chip('p9').getAttribute('aria-pressed'), 'true', 'the new preset is selected');
    const active = document.activeElement;
    assert.notEqual(active, document.body, 'never <body>');
    assert.equal(active.getAttribute('data-preset-act'), 'save', 'the focus is on "Save preset…" again');
    assert.equal(t.$('sp-presets').contains(active), true);
  });

  it('[SP-35] while there is no voice setup to save, "Save preset…" and "Save named voice…" are aria-disabled with the reason, and a click says why', async () => {
    t = await mountSpeak({ loaded: false });
    const rowSave = t.$('sp-presets').querySelector('[data-preset-act="save"]');
    for (const button of [rowSave, t.$('sp-save-preset'), t.$('sp-save-voice')]) {
      assert.equal(button.getAttribute('aria-disabled'), 'true', button.id || 'row chip');
      assert.equal(button.getAttribute('aria-describedby'), 'sp-gen-hint');
    }
    assert.equal(t.$('sp-gen-hint').textContent, 'Loading models…');
    t.env.click(rowSave);
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt').open, false, 'no prompt for a setup that isn\'t there');
    assert.equal(document.getElementById('app-sr-status').textContent, 'Loading models…', 'the click says why');
    const models = (await import('./helpers.js')).modelsPayload();
    t.app.set({ models: { list: models.models, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
    await ready(t);
    const again = t.$('sp-presets').querySelector('[data-preset-act="save"]');
    assert.equal(again.getAttribute('aria-disabled'), 'false');
    assert.equal(again.hasAttribute('aria-describedby'), false);
    assert.equal(t.$('sp-save-voice').getAttribute('aria-disabled'), 'false');
  });

  it('[TE-09] saving a preset confirms with a success toast', async () => {
    t = await mountSpeak({ routes: { 'POST /api/profiles': (req) => json(201, { id: 'p7', ...req.json }) } });
    await ready(t);
    t.env.click(t.$('sp-save-preset'));
    await answerPrompt('Quiet');
    const polite = document.getElementById('app-toasts-polite');
    assert.ok(polite.textContent.includes('Saved preset “Quiet”'));
  });

  it('[SP-30] picking a named voice applies its version for this model (recipe and reference), clears the preset, keeps the speed and stops Live', async () => {
    t = await mountSpeak();
    await ready(t);
    await applyChip('p1');
    const stops = fakeLive();
    t.env.click(t.$('sp-voice'));
    await settle(t.env);
    const identityCall = callsTo(t, 'GET /api/voices/identities').at(-1);
    assert.equal(identityCall.query.get('model'), 'kokoro');
    const options = [...document.querySelectorAll('#vp-list [role=option]')];
    const old = options.find((o) => o.querySelector('.option__label').textContent === 'Old clone');
    assert.equal(old.getAttribute('aria-disabled'), 'true', 'unavailable named voices are disabled');
    assert.equal(old.querySelector('.option__detail').textContent, 'A saved reference recording is required for this model');
    t.env.click(options.find((o) => o.querySelector('.option__label').textContent === 'Narrator'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    assert.equal(t.$('sp-blend-string').textContent, 'af_bella(2)+af_sky(1)', 'the recipe shows in the blend editor');
    assert.equal(t.$('sp-voice-name').textContent, 'Narrator');
    assert.equal(chip('p1').getAttribute('aria-pressed'), 'false');
    assert.equal(t.parts.inspector.state().speed, 1.1, 'the speed is kept');
    assert.deepEqual(stops, ['Voice changed']);
  });

  it('[SP-30] a named voice without a version for this model says so and changes nothing', async () => {
    const named = { voices: [...NAMED.voices, { id: 'n4', name: 'Bare', voice: 'voice:n4', available: true, realizations: [] }] };
    t = await mountSpeak({ routes: { 'GET /api/voices/identities': () => json(200, named) } });
    await ready(t);
    const before = t.parts.inspector.voiceSpec();
    t.env.click(t.$('sp-voice'));
    await settle(t.env);
    t.env.click([...document.querySelectorAll('#vp-list [role=option]')].find((o) => o.querySelector('.option__label').textContent === 'Bare'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.voiceSpec(), before);
    assert.equal(notice().querySelector('.banner__msg').textContent, 'This named voice has no available version for the selected model.');
  });

  it('[SP-31] Save named voice… creates the identity and its version for a new name, then reselects it', async () => {
    const named = { voices: [...NAMED.voices] };
    t = await mountSpeak({
      routes: {
        'GET /api/voices/identities': () => json(200, named),
        'POST /api/voices/identities': (req) => {
          const row = { id: 'n9', name: req.json.name, voice: 'voice:n9', available: true, realizations: [] };
          named.voices.push(row);
          return json(201, row);
        },
        'POST /api/voices/identities/:id/realizations': (req) => {
          const row = named.voices.find((v) => v.id === req.params.id);
          row.realizations.push({ ...req.json, available: true });
          return json(201, row);
        },
      },
    });
    await ready(t);
    t.env.click(t.$('sp-save-voice'));
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt-label').textContent, 'Voice name');
    assert.equal(document.getElementById('app-prompt-hint').textContent, "An existing name adds or updates this model's version");
    await answerPrompt('Storyteller');
    await settle(t.env);
    assert.equal(callsTo(t, 'GET /api/voices/identities').some((c) => c.query.get('check_availability') === 'false'), true, 'matched against the unchecked catalogue');
    assert.deepEqual(callsTo(t, 'POST /api/voices/identities')[0].json, { name: 'Storyteller' });
    const version = callsTo(t, 'POST /api/voices/identities/:id/realizations')[0];
    assert.equal(version.params.id, 'n9');
    assert.deepEqual(version.json, { model: 'kokoro', voice: t.parts.inspector.currentSetup().voice, reference_audio_id: null });
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n9', 'reselected');
    assert.ok(document.getElementById('app-toasts-polite').textContent.includes('Saved Storyteller for Kokoro'));
  });

  it('[SP-31] an existing name (any case) asks before replacing this model\'s version, then PUTs it', async () => {
    t = await mountSpeak({ routes: { 'PUT /api/voices/identities/:id/realizations/*': () => json(200, NAMED.voices[0]) } });
    await ready(t);
    t.parts.inspector.commit(t.parts.inspector.state().data, { ...t.parts.inspector.currentSetup(), identity: 'n1' });
    t.env.click(t.$('sp-save-voice'));
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt-input').value, 'Narrator', 'prefilled with the selected named voice');
    await answerPrompt('NARRATOR');
    assert.equal(document.getElementById('app-confirm-msg').textContent, 'Update “Narrator” for Kokoro? Linked presets will use the updated voice.');
    await answerConfirm(t, false);
    assert.equal(callsTo(t, 'PUT /api/voices/identities/:id/realizations/*').length, 0, 'cancel changes nothing');
    t.env.click(t.$('sp-save-voice'));
    await answerPrompt('narrator');
    await answerConfirm(t, true);
    const [put] = callsTo(t, 'PUT /api/voices/identities/:id/realizations/*');
    assert.equal(put.path, '/api/voices/identities/n1/realizations/kokoro');
    assert.equal(put.json.model, 'kokoro');
    assert.equal(callsTo(t, 'POST /api/voices/identities').length, 0, 'no new identity');
  });
});
