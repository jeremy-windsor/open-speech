// Speak: editing a named voice's version or a preset from Voices.
// Speak shows an editing bar: "Editing Narrator for Kokoro" with Update voice and Cancel, or
// "Editing preset “Podcast EQ”" with Update preset, Save as new… and Cancel. New preset on a voice's card opens
// Speak with that voice and Save preset… focused, its name prefilled.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountSpeak, ready, settle, callsTo, json, PRESETS, NAMED, load } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const bar = () => t.$('sp-editing');
const notice = () => t.$('sp-notice');
const text = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
const toasted = () => `${text(document.getElementById('app-toasts'))} ${text(document.getElementById('app-toasts-polite'))}`;
const chip = (id) => t.$('sp-presets').querySelector(`[data-preset-id="${id}"]`);

/** PRESETS plus two linked to Narrator: one for Kokoro (it follows the Kokoro version) and one for Piper. */
const LINKED = {
  profiles: [
    ...PRESETS.profiles,
    { id: 'p3', name: 'Narration', backend: 'kokoro', model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', voice_identity_id: 'n1', speed: 1, format: 'mp3', effects: [] },
    { id: 'p4', name: 'Narration Piper', backend: 'piper', model: 'piper/en_US-amy-medium', voice: 'en_US-amy-medium', voice_identity_id: 'n1', speed: 1, format: 'mp3', effects: [] },
  ],
  default_profile_id: null,
};

async function handOff(intent) {
  t.setIntent('speak', intent);
  t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'heading' });
  for (let i = 0; i < 4; i += 1) await settle(t.env);
}

/** A keyboard or pointer press: the button has the focus, then the click. */
async function click(id) {
  t.$(id).focus();
  t.env.click(t.$(id));
  for (let i = 0; i < 4; i += 1) await settle(t.env);
}

async function answerPrompt(value) {
  await settle(t.env);
  assert.equal(document.getElementById('app-prompt').open, true, 'the prompt is open');
  document.getElementById('app-prompt-input').value = value;
  document.getElementById('app-prompt-form').requestSubmit();
  for (let i = 0; i < 3; i += 1) await settle(t.env);
}

describe('views/speak editing bars', () => {
  it('Edit in Speak shows "Editing Narrator for Kokoro"; Update voice PUTs the version with no name prompt and says how many presets follow it', async () => {
    t = await mountSpeak({ presets: LINKED, routes: { 'PUT /api/voices/identities/:id/realizations/*': (req) => json(200, { ...NAMED.voices[0], realizations: [{ ...req.json, available: true }] }) } });
    await ready(t);
    const { on } = await load('core/store.js');
    const changes = [];
    on('voices:changed', (d) => changes.push(d.what));
    await handOff({ type: 'edit-named-voice', id: 'n1', model: 'kokoro' });
    assert.equal(bar().hidden, false);
    assert.equal(text(t.$('sp-editing-title')), 'Editing Narrator for Kokoro');
    assert.equal(bar().getAttribute('role'), 'group');
    assert.equal(bar().getAttribute('aria-labelledby'), 'sp-editing-title');
    assert.equal(text(t.$('sp-editing-update')), 'Update voice');
    assert.equal(t.$('sp-editing-new').hidden, true, 'Save as new… is for presets');
    assert.equal(text(t.$('sp-editing-cancel')), 'Cancel');
    assert.equal(notice().hidden, true, 'the bar replaces the old "Change the voice setup, then Save named voice" hint');
    assert.equal(document.getElementById('app-sr-status').textContent, 'Editing Narrator for Kokoro');
    t.env.click(t.$('sp-blend-equal'));
    await settle(t.env);
    const recipe = t.parts.inspector.currentSetup().voice;
    assert.notEqual(recipe, 'af_bella(2)+af_sky(1)', 'the blend changed');
    await click('sp-editing-update');
    assert.equal(document.getElementById('app-prompt').open, false, 'no name prompt');
    const puts = callsTo(t, 'PUT /api/voices/identities/:id/realizations/*');
    assert.equal(puts.length, 1);
    assert.equal(puts[0].path, '/api/voices/identities/n1/realizations/kokoro');
    assert.deepEqual(puts[0].json, { model: 'kokoro', voice: recipe, reference_audio_id: null });
    assert.match(toasted(), /Updated Narrator for Kokoro\. 1 preset follows it\./, 'only the Kokoro preset follows the Kokoro version');
    assert.equal(bar().hidden, true, 'the edit is done');
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1', 'the named voice stays selected');
    assert.ok(changes.includes('named'), 'Voices and the voice pickers hear about it');
    assert.equal(document.activeElement, t.$('sp-generate'), 'focus never drops to <body> when the bar goes');
  });

  it('Cancel puts Speak back to what it had before the edit, preset included', async () => {
    t = await mountSpeak({ routes: { 'PUT /api/voices/identities/:id/realizations/*': () => json(200, {}) } });
    await ready(t);
    t.env.click(chip('p2'));
    for (let i = 0; i < 3; i += 1) await settle(t.env);
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'true');
    const before = t.parts.inspector.currentSetup();
    await handOff({ type: 'edit-named-voice', id: 'n1', model: 'kokoro' });
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    await click('sp-editing-cancel');
    assert.equal(bar().hidden, true);
    assert.deepEqual(t.parts.inspector.currentSetup(), before, 'the voice, speed, format and effects are back');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_heart');
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'true', 'and the preset that was applied');
    assert.equal(callsTo(t, 'PUT /api/voices/identities/:id/realizations/*').length, 0, 'nothing is saved');
  });

  it('changing the model ends the voice editing with a note; Save named voice… still offers the name', async () => {
    t = await mountSpeak({ routes: { 'PUT /api/voices/identities/:id/realizations/*': () => json(200, {}) } });
    await ready(t);
    await handOff({ type: 'edit-named-voice', id: 'n1', model: 'kokoro' });
    t.env.change(t.$('sp-model'), 'piper/en_US-amy-medium');
    for (let i = 0; i < 3; i += 1) await settle(t.env);
    assert.equal(bar().hidden, true);
    assert.equal(text(notice().querySelector('.banner__title')), 'Stopped editing Narrator');
    assert.equal(text(notice().querySelector('.banner__msg')), 'The model changed. Save named voice… adds a version of Narrator for this model.');
    assert.ok(notice().querySelector('[role="status"]'), 'a note, not an alert');
    assert.equal(callsTo(t, 'PUT /api/voices/identities/:id/realizations/*').length, 0);
    t.env.click(t.$('sp-save-voice'));
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt-input').value, 'Narrator', 'the note\'s next step starts with the name');
    t.env.click(document.getElementById('app-prompt-cancel'));
    await settle(t.env);
  });

  it('Edit on a preset applies it and shows "Editing preset “Podcast EQ”"; Update preset PUTs the full payload from Speak\'s setup', async () => {
    const stored = JSON.parse(JSON.stringify(PRESETS));
    t = await mountSpeak({
      presets: stored,
      routes: {
        'PUT /api/profiles/:id': (req) => {
          const i = stored.profiles.findIndex((p) => p.id === req.params.id);
          stored.profiles[i] = { ...req.json, id: req.params.id };
          return json(200, stored.profiles[i]);
        },
      },
    });
    await ready(t);
    await handOff({ type: 'edit-preset', id: 'p2' });
    assert.equal(bar().hidden, false);
    assert.equal(text(t.$('sp-editing-title')), 'Editing preset “Podcast EQ”');
    assert.equal(text(t.$('sp-editing-update')), 'Update preset');
    assert.equal(t.$('sp-editing-new').hidden, false);
    assert.equal(text(t.$('sp-editing-new')), 'Save as new…');
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'true', 'the preset is applied');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_heart');
    t.env.click(t.$('sp-speed-up'));
    t.env.click(t.$('sp-effects').querySelector('[data-effect="normalize"]'));
    await settle(t.env);
    await click('sp-editing-update');
    assert.equal(callsTo(t, 'POST /api/profiles').length, 0, 'an update, not a new preset (no 409 for the same name)');
    const [put] = callsTo(t, 'PUT /api/profiles/:id');
    assert.equal(put.path, '/api/profiles/p2');
    assert.deepEqual(put.json, {
      name: 'Podcast EQ', backend: 'kokoro', provider: 'kokoro', model: 'kokoro', voice: 'af_heart', speed: 1.05, format: 'mp3',
      blend: null, reference_audio_id: null, voice_identity_id: null, instructions: null, effects: [{ type: 'normalize' }, { type: 'podcast_eq' }],
      options: {},
    });
    assert.match(toasted(), /Updated preset “Podcast EQ”/);
    assert.equal(bar().hidden, true);
    assert.equal(chip('p2').getAttribute('aria-pressed'), 'true', 'the updated preset stays applied');
  });

  it('Save as new… saves the edited setup as another preset; a failed Update keeps the editing so nothing is lost', async () => {
    t = await mountSpeak({
      routes: {
        'PUT /api/profiles/:id': () => json(404, { error: { message: 'Profile not found' } }),
        'POST /api/profiles': (req) => json(201, { ...req.json, id: 'p9' }),
      },
    });
    await ready(t);
    await handOff({ type: 'edit-preset', id: 'p2' });
    await click('sp-editing-update');
    assert.equal(bar().hidden, false, 'still editing');
    assert.equal(text(notice().querySelector('.banner__title')), "Couldn't update the preset");
    assert.match(text(notice().querySelector('.banner__msg')), /Profile not found/);
    await click('sp-editing-new');
    assert.equal(document.getElementById('app-prompt-title').textContent, 'Save preset');
    await answerPrompt('Podcast copy');
    const [post] = callsTo(t, 'POST /api/profiles');
    assert.equal(post.json.name, 'Podcast copy');
    assert.equal(post.json.voice, 'af_heart');
    assert.deepEqual(post.json.effects, [{ type: 'podcast_eq' }]);
    assert.equal(bar().hidden, true, 'the edit ends once it is saved as new');
  });

  it('Edit on a preset that can\'t be used says why and starts no editing', async () => {
    t = await mountSpeak({ presets: { profiles: [...PRESETS.profiles, { id: 'px', name: 'Gone model', backend: 'xtts', model: 'xtts', voice: 'v', speed: 1, format: 'mp3', effects: [] }], default_profile_id: null } });
    await ready(t);
    await handOff({ type: 'edit-preset', id: 'px' });
    assert.equal(bar().hidden, true);
    assert.equal(text(notice().querySelector('.banner__title')), 'Preset “Gone model” can\'t be used');
  });

  it('New preset: the voice is chosen for the model Speak is on when it has a version, else its first model with one; Save preset… is focused and its name prefilled', async () => {
    const named = { voices: [...NAMED.voices, { id: 'n3', name: 'Amy', voice: 'voice:n3', available: true, realizations: [{ model: 'piper/en_US-amy-medium', voice: 'en_US-amy-medium', reference_audio_id: null, available: true }] }] };
    t = await mountSpeak({ routes: { 'GET /api/voices/identities': () => json(200, named), 'POST /api/profiles': (req) => json(201, { ...req.json, id: 'p9' }) } });
    await ready(t);
    assert.equal(t.$('sp-model').value, 'kokoro');
    await handOff({ type: 'new-preset', id: 'n1' });
    assert.equal(t.$('sp-model').value, 'kokoro', 'Kokoro has a version: Speak stays on it');
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    assert.equal(document.activeElement, t.$('sp-save-preset'), 'Save preset… is focused');
    assert.equal(text(notice().querySelector('.banner__title')), 'New preset with Narrator');
    t.env.click(t.$('sp-save-preset'));
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt-input').value, 'Narrator', 'the name is prefilled');
    await answerPrompt('Narrator');
    const [post] = callsTo(t, 'POST /api/profiles');
    assert.equal(post.json.voice_identity_id, 'n1');
    assert.equal(post.json.name, 'Narrator');
    assert.equal(notice().hidden, true, 'the hint goes once the preset is saved');
    await handOff({ type: 'new-preset', id: 'n3' });
    assert.equal(t.$('sp-model').value, 'piper/en_US-amy-medium', 'no Kokoro version: its first model that has one');
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n3');
    await handOff({ type: 'new-preset', id: 'n2' });
    assert.equal(text(notice().querySelector('.banner__title')), "Couldn't start a preset with Old clone");
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n3', 'nothing changed');
  });

  it('the editing-voice and editing-preset screenshot states show the bars and save nothing', async () => {
    t = await mountSpeak({ route: { shot: 'editing-voice' } });
    await ready(t);
    for (let i = 0; i < 3; i += 1) await settle(t.env);
    assert.equal(text(t.$('sp-editing-title')), 'Editing Narrator for Kokoro');
    t.env.restore();
    t = await mountSpeak({ route: { shot: 'editing-preset' } });
    await ready(t);
    for (let i = 0; i < 3; i += 1) await settle(t.env);
    assert.equal(text(t.$('sp-editing-title')), 'Editing preset “Audiobook 1.1×”');
    const writes = t.env.fetch.calls.filter((c) => c.method !== 'GET');
    assert.deepEqual(writes.map((c) => `${c.method} ${c.path}`), [], 'display only');
  });
});
