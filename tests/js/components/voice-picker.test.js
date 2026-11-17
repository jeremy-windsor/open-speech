import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames } from '../harness/a11y.js';
import { bytes } from '../harness/fakes.js';
import { shellEnv, load, settle, json } from '../shell/helpers.js';

let env;
afterEach(() => env?.restore());

const VOICES = { voices: ['af_heart', 'af_bella', 'af_sky', 'am_adam', 'bf_emma', 'bm_george', 'ef_dora'].map((id) => ({ id })) };
const NAMED = { voices: [
  { id: 'n1', name: 'Narrator', voice: 'voice:n1', available: true, realizations: [{ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', available: true }] },
  { id: 'n2', name: 'Old clone', voice: 'voice:n2', available: false, realizations: [{ model: 'kokoro', voice: 'af_sky', available: false, reason: 'A saved reference recording is required for this model' }] },
  { id: 'n3', name: 'Chatter', voice: 'voice:n3', available: false, realizations: [{ model: 'chatterbox', voice: 'Ryan', available: false, reason: "Select this realization's model" }] },
] };
const RECORDINGS = [{ name: 'demo_reference', transcript: 'Hello there.' }, { name: 'raw_take' }];

async function setup({ seed = {}, routes = {} } = {}) {
  env = await shellEnv({
    seed,
    routes: {
      'GET /v1/audio/voices': () => json(200, VOICES),
      'GET /api/voices/identities': () => json(200, NAMED),
      'GET /api/voices/library': () => json(200, RECORDINGS),
      'POST /v1/audio/speech': () => bytes(200, [1, 2, 3, 4], 'audio/mpeg'),
      ...routes,
    },
  });
  const vp = await load('components/voice-picker.js');
  const { app } = await load('core/store.js');
  const trigger = document.createElement('button');
  trigger.setAttribute('type', 'button');
  trigger.textContent = 'Voice';
  document.getElementById('main').appendChild(trigger);
  trigger.focus();
  return { vp, app, trigger, dialog: document.getElementById('app-voice-picker'), search: document.getElementById('vp-search') };
}

// Returns {result}: an async function returning the picker's promise would wait for it to settle.
async function open(ctx, options = {}) {
  const result = ctx.vp.openVoicePicker({ trigger: ctx.trigger, model: 'kokoro', value: 'af_heart', ...options });
  await settle(env);
  return { result };
}

const options = () => [...document.querySelectorAll('#vp-list [role=option]')];
const optionNamed = (label) => options().find((o) => o.querySelector('.option__label').textContent === label);
const groupLabels = () => [...document.querySelectorAll('#vp-list .listbox__label')].map((el) => el.textContent);

describe('components/voice-picker', () => {
  it('[RE-65] opens with focus in the search combobox and picks a voice with a single click', async () => {
    const ctx = await setup();
    const { result: result } = await open(ctx);
    assert.equal(ctx.dialog.open, true);
    assert.equal(document.activeElement, ctx.search);
    assert.equal(ctx.search.getAttribute('role'), 'combobox');
    assert.equal(ctx.search.getAttribute('aria-controls'), 'vp-list');
    assert.equal(ctx.search.getAttribute('aria-expanded'), 'true');
    assert.equal(document.getElementById('vp-list').getAttribute('role'), 'listbox');
    const heart = optionNamed('Heart');
    assert.equal(heart.getAttribute('aria-selected'), 'true', 'the current voice is selected');
    assert.equal(ctx.search.getAttribute('aria-activedescendant'), heart.id, 'and active');
    env.click(optionNamed('Emma'));
    const choice = await result;
    assert.deepEqual(choice, { kind: 'voice', id: 'bf_emma', voice: 'bf_emma', label: 'Emma', detail: 'British English · female', model: 'kokoro' });
    await settle(env);
    assert.equal(ctx.dialog.open, false);
    assert.equal(document.activeElement, ctx.trigger, 'focus returns to the trigger');
  });

  it('[RE-65] keys in the search field: arrows, Home/End move the active option, Enter chooses, Esc closes with null', async () => {
    const ctx = await setup();
    let { result: result } = await open(ctx);
    const active = () => document.getElementById(ctx.search.getAttribute('aria-activedescendant'));
    env.key('ArrowDown');
    assert.equal(active().querySelector('.option__label').textContent, 'Bella');
    assert.equal(document.activeElement, ctx.search, 'focus stays in the search field');
    env.key('End');
    assert.equal(active(), options().at(-1));
    env.key('Home');
    assert.equal(active(), options()[0]);
    env.key('Enter');
    const choice = await result;
    assert.equal(choice.kind, 'named');
    await settle(env);
    ({ result } = await open(ctx));
    const esc = env.key('Escape');
    assert.equal(esc.defaultPrevented, true, 'Esc never only clears the search field');
    assert.equal(await result, null);
    assert.equal(ctx.dialog.open, false);
  });

  it('groups: extra rows, Named voices (unavailable ones disabled with the reason), Recordings, then the model’s own voices under its name: Recommended, then languages', async () => {
    const ctx = await setup();
    const { result: result } = await open(ctx, {
      include: { named: true, recordings: true, voices: true },
      extraRows: [{ id: 'same', label: 'Same as Speak', detail: 'Narrator', voice: 'af_bella(2)+af_sky(1)' }],
    });
    assert.deepEqual(groupLabels(), ['Suggested', 'Named voices', 'Recordings', 'Kokoro · Recommended', 'Kokoro · American English', 'Kokoro · British English', 'Kokoro · Spanish']);
    const old = optionNamed('Old clone');
    assert.equal(old.getAttribute('aria-disabled'), 'true');
    assert.match(old.textContent, /A saved reference recording is required for this model/);
    assert.match(optionNamed('Chatter').textContent, /Unavailable for this model/, 'no version for this model');
    assert.match(old.textContent, /unavailable/, 'the state word is part of its name');
    env.click(old);
    assert.equal(ctx.dialog.open, true, 'a disabled option is never chosen');
    assert.match(optionNamed('Narrator').textContent, /Blend · af_bella\(2\)\+af_sky\(1\)/);
    assert.match(optionNamed('Heart').textContent, /recommended/);
    assertAccessibleNames(ctx.dialog);
    env.click(optionNamed('demo_reference'));
    assert.deepEqual(await result, { kind: 'recording', id: 'demo_reference', voice: 'demo_reference', label: 'demo_reference', detail: 'Transcript saved', model: 'kokoro' });
    await settle(env);
    const { result: again } = await open(ctx);
    env.click(optionNamed('Narrator'));
    const named = await again;
    assert.equal(named.voice, 'voice:n1', 'a named voice is sent as voice:<id>');
    assert.equal(groupLabels().includes('Recordings'), false, 'recordings are off by default');
  });

  it('filters: search (accent-insensitive), Female/Male chips, Favourites and the language menu', async () => {
    const ctx = await setup({ seed: { 'open-speech-voice-favorites': JSON.stringify(['kokoro|bf_emma']) } });
    const { result: result } = await open(ctx);
    assert.equal(groupLabels()[0], 'Favorites');
    env.input(ctx.search, 'dóra');
    assert.deepEqual(options().map((o) => o.querySelector('.option__label').textContent), ['Dora']);
    env.input(ctx.search, '');
    env.click(document.getElementById('vp-male'));
    assert.equal(document.getElementById('vp-male').getAttribute('aria-pressed'), 'true');
    assert.deepEqual(options().map((o) => o.querySelector('.option__label').textContent), ['Adam', 'George']);
    env.click(document.getElementById('vp-male'));
    env.click(document.getElementById('vp-favs'));
    assert.deepEqual(groupLabels(), ['Favorites']);
    assert.equal(options().length, 1);
    env.click(document.getElementById('vp-favs'));
    env.click(document.getElementById('vp-lang'));
    const spanish = [...document.querySelectorAll('#vp-lang-menu [role=menuitemradio]')].find((el) => el.textContent.includes('Spanish'));
    env.click(spanish);
    assert.equal(document.getElementById('vp-lang-label').textContent, 'Spanish');
    assert.deepEqual(options().map((o) => o.querySelector('.option__label').textContent), ['Dora']);
    env.input(ctx.search, 'zzz');
    assert.equal(document.getElementById('vp-empty').hidden, false);
    assert.equal(document.getElementById('vp-empty').textContent, 'No voices match.');
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('a search or filter change announces the number of matches once it pauses, and "No voices match." when none do (WCAG 4.1.3)', async () => {
    // Typing that narrowed or emptied the list was silent for a screen reader.
    const ctx = await setup({ seed: { 'open-speech-voice-favorites': JSON.stringify(['kokoro|bf_emma']) } });
    const { result } = await open(ctx);
    const said = () => document.getElementById('app-sr-status').textContent;
    await settle(env, 1000);
    const before = said();
    env.input(ctx.search, 'd');
    await settle(env, 200);
    env.input(ctx.search, 'do');
    await settle(env, 200);
    env.input(ctx.search, 'dor');
    await settle(env, 200);
    assert.equal(said(), before, 'nothing is said while the listener types');
    await settle(env, 600);
    assert.equal(said(), '1 voice', 'Dora only');
    env.input(ctx.search, 'zzz');
    await settle(env, 1200);
    assert.equal(said(), 'No voices match.');
    env.input(ctx.search, '');
    await settle(env, 1200);
    assert.equal(said(), '10 voices', '7 voices and 3 named voices: the favourite listed twice (Favourites and its language) counts once');
    env.click(document.getElementById('vp-male'));
    await settle(env, 1200);
    assert.equal(said(), '2 voices', 'a filter chip announces too');
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
    env.input(ctx.search, 'x');
    await settle(env, 1200);
    assert.equal(said(), '2 voices', 'nothing is announced after the picker closed');
  });

  it('[REV-18] no Alt+F favourite key; the Favourite button (one Tab away) and the star glyph toggle and persist it', async () => {
    const ctx = await setup();
    const { result: result } = await open(ctx);
    const fav = document.getElementById('vp-fav');
    assert.equal(fav.getAttribute('aria-pressed'), 'false');
    env.key('f', { altKey: true });
    env.key('ƒ');
    assert.equal(fav.getAttribute('aria-pressed'), 'false', 'Alt+F (ƒ on macOS) does nothing');
    assert.equal(env.localStorage.getItem('open-speech-voice-favorites'), null);
    env.click(fav);
    assert.equal(fav.getAttribute('aria-pressed'), 'true');
    assert.deepEqual(JSON.parse(env.localStorage.getItem('open-speech-voice-favorites')), ['kokoro|af_heart']);
    assert.equal(groupLabels()[0], 'Favorites');
    assert.match(optionNamed('Heart').textContent, /favorite/);
    env.click(optionNamed('Emma').querySelector('[data-act="fav"]'));
    assert.deepEqual(JSON.parse(env.localStorage.getItem('open-speech-voice-favorites')), ['kokoro|af_heart', 'kokoro|bf_emma']);
    assert.equal(ctx.dialog.open, true, 'the star glyph never chooses the voice');
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('previews with Shift+Enter or Preview: an ephemeral clip, without X-History; disabled while recording', async () => {
    const ctx = await setup();
    const { result: result } = await open(ctx);
    env.key('Enter', { shiftKey: true });
    await settle(env);
    const calls = env.fetch.callsTo('POST /v1/audio/speech');
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0].json, { model: 'kokoro', voice: 'af_heart', input: 'Hello! This is how I sound.', response_format: 'mp3' });
    assert.equal(calls[0].headers['x-history'], undefined, 'previews stay out of History');
    assert.equal(ctx.app.get().player.ephemeral, true);
    assert.equal(ctx.app.get().player.owner, 'preview');
    assert.equal(ctx.dialog.open, true, 'previewing does not choose');
    ctx.app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    const preview = document.getElementById('vp-preview');
    assert.equal(preview.getAttribute('aria-disabled'), 'true');
    assert.equal(document.getElementById('vp-note').hidden, false);
    assert.equal(preview.getAttribute('aria-describedby'), 'vp-note');
    env.click(preview);
    await settle(env);
    assert.equal(env.fetch.callsTo('POST /v1/audio/speech').length, 1, 'no preview while the microphone is open');
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('closing the picker (Use, Esc) stops its preview, which has no other control, and revokes a recording preview only after the element let go', async () => {
    // The preview used to play on with no mini-player after the sheet closed, and a recording
    // preview's object URL was revoked while the element still read it.
    const ctx = await setup({ seed: { 'open-speech-api-key': 'k1' }, routes: { 'GET /api/voices/library/:name/audio': () => bytes(200, [1, 2, 3], 'audio/wav') } });
    (await load('core/auth.js')).init();
    const { result } = await open(ctx, { include: { recordings: true } });
    const el = document.getElementById('app-audio');
    env.key('Enter', { shiftKey: true });              // preview Heart
    await settle(env);
    assert.equal(ctx.app.get().player.owner, 'preview');
    assert.equal(el.paused, false);
    env.click(document.getElementById('vp-use'));
    assert.equal(await result.then((c) => c.voice), 'af_heart');
    assert.equal(el.paused, true, 'Use stops the preview');
    assert.equal(ctx.app.get().player, null);

    const again = (await open(ctx, { include: { recordings: true } })).result;
    env.click(optionNamed('demo_reference').querySelector('[data-act="preview"]'));
    await settle(env);
    await settle(env);
    const src = el.getAttribute('src');
    assert.match(src, /^blob:/, 'with a key the recording plays from an object URL');
    assert.equal(env.revokedUrls.includes(src), false, 'not revoked while it plays');
    const order = [];
    const revoke = URL.revokeObjectURL;
    URL.revokeObjectURL = (u) => { order.push(['revoke', u, el.paused]); revoke(u); };
    env.key('Escape');
    URL.revokeObjectURL = revoke;
    assert.equal(await again, null);
    assert.equal(el.paused, true, 'Esc stops it too');
    assert.deepEqual(order, [['revoke', src, true]], 'revoked once, after the element was paused and released');
  });

  it('a preview that was still starting when the picker closed is stopped as soon as it starts', async () => {
    const ctx = await setup({ routes: { 'POST /v1/audio/speech': () => new Promise((resolve) => setTimeout(() => resolve(bytes(200, [1, 2], 'audio/mpeg')), 300)) } });
    const { result } = await open(ctx);
    env.key('Enter', { shiftKey: true });
    await settle(env, 50);
    env.key('Escape');
    assert.equal(await result, null);
    await settle(env, 400);
    assert.equal(ctx.app.get().player, null, 'nothing is left playing');
    assert.equal(document.getElementById('app-audio').paused, true);
  });

  it('blend-add mode lists provider voices only, without the excluded ones, and says Add voice', async () => {
    const ctx = await setup();
    const { result: result } = await open(ctx, { mode: 'blend-add', title: 'Add a voice to the blend', exclude: ['af_bella', 'af_sky'], extraRows: [{ id: 'x', label: 'X', voice: 'af_heart' }] });
    assert.equal(document.getElementById('vp-title').textContent, 'Add a voice to the blend');
    assert.equal(document.querySelector('#vp-use .btn__label').textContent, 'Add voice');
    assert.equal(groupLabels().includes('Named voices'), false);
    assert.equal(groupLabels().includes('Suggested'), false);
    assert.equal(optionNamed('Bella'), undefined);
    assert.equal(env.fetch.callsTo('GET /api/voices/identities').length, 0);
    env.click(document.getElementById('vp-use'));
    assert.equal((await result).voice, 'af_heart', 'Use acts on the active option');
  });

  it('pickerGroups and selectedKey are pure', async () => {
    const ctx = await setup();
    const rows = ctx.vp.pickerRows({ voices: VOICES.voices, named: NAMED.voices }, { model: 'kokoro', favorites: ['kokoro|voice:n1'] });
    assert.equal(ctx.vp.selectedKey(rows, 'voice:n1'), 'n|n1');
    assert.equal(ctx.vp.selectedKey(rows, 'af_sky'), 'v|af_sky');
    assert.equal(ctx.vp.selectedKey(rows, 'nope'), null);
    const groups = ctx.vp.pickerGroups(rows, {});
    assert.deepEqual(groups.map((g) => g.label), ['Favorites', 'Named voices', 'Recommended', 'American English', 'British English', 'Spanish']);
    assert.deepEqual(ctx.vp.pickerGroups(rows, { query: 'narr', favsOnly: true }).map((g) => g.label), ['Favorites']);
  });

  it("a model's own voices are headed with its name, so voices of two models never look alike (Supertonic · Voices)", async () => {
    const ctx = await setup();
    const kokoro = ctx.vp.pickerRows({ voices: VOICES.voices }, { model: 'kokoro' });
    assert.deepEqual(ctx.vp.pickerGroups(kokoro, { modelName: 'Kokoro' }).map((g) => g.label), ['Kokoro · Recommended', 'Kokoro · American English', 'Kokoro · British English', 'Kokoro · Spanish']);
    const supertonic = [{ id: 'M1', name: 'M1', gender: 'male' }, { id: 'F2', name: 'F2', gender: 'female' }];
    const rows = ctx.vp.pickerRows({ voices: supertonic }, { model: 'supertonic' });
    const groups = ctx.vp.pickerGroups(rows, { modelName: 'Supertonic' });
    assert.deepEqual(groups.map((g) => g.label), ['Supertonic · Voices']);
    assert.deepEqual(groups[0].options.map((o) => [o.label, o.detail, o.gender]), [['M1', 'male', 'male'], ['F2', 'female', 'female']]);
    assert.deepEqual(ctx.vp.pickerGroups(rows, { modelName: 'Supertonic', male: true }).flatMap((g) => g.options.map((o) => o.id)), ['M1']);
  });

  it('a failed load (refused key, offline, a 5xx) shows the cause with Retry, never "no voices"; Retry loads again', async () => {
    let fail = true;
    const ctx = await setup({ routes: { 'GET /v1/audio/voices': () => (fail ? json(500, { error: { message: 'Kokoro crashed' } }) : json(200, VOICES)) } });
    const { result } = await open(ctx);
    const failed = document.getElementById('vp-failed');
    assert.equal(failed.hidden, false);
    assert.match(document.getElementById('vp-failed-msg').textContent, /^Couldn't load the voices\. /);
    assert.equal(document.getElementById('vp-failed-msg').getAttribute('role'), 'alert');
    assert.equal(document.getElementById('vp-empty').hidden, true, 'no misleading empty state');
    assert.equal(document.getElementById('vp-list').hidden, true);
    assert.equal(document.getElementById('vp-use').disabled, true);
    fail = false;
    env.click(document.getElementById('vp-retry'));
    await settle(env);
    assert.equal(failed.hidden, true);
    assert.ok(optionNamed('Heart'), 'the voices are there after Retry');
    env.key('Escape', {}, ctx.search);
    assert.equal(await result, null);
  });

  it('a refused key says how to fix it (Settings › Connection), not that the model has no voices', async () => {
    const ctx = await setup({ routes: { 'GET /v1/audio/voices': () => json(401, { error: { message: 'Invalid API key' } }) } });
    await open(ctx);
    assert.match(document.getElementById('vp-failed-msg').textContent, /Settings › Connection/);
    assert.doesNotMatch(document.getElementById('app-voice-picker').textContent, /no voices to choose from/);
  });

  it('when only named voices fail, the voices still show and a note says what is missing', async () => {
    const ctx = await setup({ routes: { 'GET /api/voices/identities': () => json(500, { error: { message: 'boom' } }) } });
    await open(ctx);
    assert.equal(document.getElementById('vp-failed').hidden, true);
    assert.ok(optionNamed('Heart'));
    const note = document.getElementById('vp-error');
    assert.equal(note.hidden, false);
    assert.match(note.textContent, /^Named voices didn't load\./);
  });

  it('opens at the top when the chosen voice is in the first screenful (no half-cut row, its group label visible)', async () => {
    const ctx = await setup();
    const list = document.getElementById('vp-list');
    list.__rect = { top: 200, height: 400, scrollHeight: 1200 };   // the list scrolls itself
    list.scrollTop = 120;
    await open(ctx);
    assert.equal(list.scrollTop, 0, 'Heart (selected, the first rows) leaves the list at the top');
  });

  it('the Favourites filter chip swaps its star for a filled one while pressed (not colour alone)', async () => {
    const ctx = await setup();
    await open(ctx);
    const chip = document.getElementById('vp-favs');
    assert.ok(chip.querySelector('.chip__idle use').getAttribute('href').endsWith('#i-star'));
    assert.ok(chip.querySelector('.chip__check use').getAttribute('href').endsWith('#i-star-fill'), 'the filled star shows when aria-pressed=true (CSS)');
  });
});

// With a cloning model such as Chatterbox, which has no voices of its
// own, Speak's picker looked like Kokoro's (named voices first, the recordings it can use last) and nothing said
// why. An opener that passes the model's label (Speak) now gets, for a model whose own voice list is empty: a note
// with Clone a voice…, Recordings first, the named voices that have a version for the model, and the named voices
// that have none in a last group, disabled with the reason. A model with voices of its own keeps its order.
describe('components/voice-picker: a model with no voices of its own', () => {
  // As the server answers /api/voices/identities?model=chatterbox/regular for voices that only have a Kokoro version.
  const kokoroOnly = (id, name, voice) => ({ id, name, voice: `voice:${id}`, available: false, realizations: [{ model: 'kokoro', voice, available: false, reason: "Select this realization's model" }] });
  const KOKORO_ONLY = { voices: [kokoroOnly('k1', 'Narrator', 'af_bella(2)+af_sky(1)'), kokoroOnly('k2', 'Storyteller', 'af_heart')] };
  const cloning = (routes = {}) => ({
    'GET /v1/audio/voices': () => json(200, { voices: [] }),
    'GET /api/voices/identities': () => json(200, KOKORO_ONLY),
    'GET /api/voices/library': () => json(200, [{ name: 'my_voice', transcript: 'Hello.' }]),
    ...routes,
  });
  const CHATTERBOX = Object.freeze({ model: 'chatterbox/regular', value: '', modelLabel: 'Chatterbox regular', include: { named: true, recordings: true, voices: true } });
  // "with no <Model> version", which reads well before a vowel sound too.
  const LAST = 'Named voices with no Chatterbox regular version';
  const NOTE = 'Chatterbox regular has no built-in voices. It speaks in the voice of a recording.';
  const note = () => document.getElementById('vp-model-note');
  const noteShown = () => !!note() && !note().hidden;
  const noteText = () => (document.getElementById('vp-model-note-text') || { textContent: '' }).textContent;
  const labels = () => options().map((o) => o.querySelector('.option__label').textContent);
  const inGroup = (name) => {
    const group = [...document.querySelectorAll('#vp-list [role=group]')].find((g) => g.querySelector('.listbox__label').textContent === name);
    return group ? [...group.querySelectorAll('[role=option] .option__label')].map((el) => el.textContent) : null;
  };

  it('lists Recordings first and the named voices without a version for the model in a last group, disabled with the reason', async () => {
    const ctx = await setup({ routes: cloning() });
    const { result } = await open(ctx, CHATTERBOX);
    assert.deepEqual(groupLabels(), ['Recordings', LAST]);
    assert.deepEqual(labels(), ['my_voice', 'Narrator', 'Storyteller']);
    for (const name of ['Narrator', 'Storyteller']) {
      const row = optionNamed(name);
      assert.equal(row.getAttribute('aria-disabled'), 'true', name);
      assert.match(row.textContent, /No Chatterbox regular version/, 'the reason is its detail');
      assert.match(row.textContent, /unavailable/, 'the state word is part of its name');
      env.click(row);
      assert.equal(ctx.dialog.open, true, 'a voice the model cannot speak is never chosen');
    }
    assertAccessibleNames(ctx.dialog);
    env.click(optionNamed('my_voice'));
    assert.deepEqual(await result, { kind: 'recording', id: 'my_voice', voice: 'my_voice', label: 'my_voice', detail: 'Transcript saved', model: 'chatterbox/regular' });
  });

  it('opens with a note that the model has no built-in voices and speaks in the voice of a recording, with Clone a voice…', async () => {
    const ctx = await setup({ routes: cloning() });
    const { result } = await open(ctx, CHATTERBOX);
    assert.equal(noteShown(), true, 'the note shows');
    assert.equal(noteText(), NOTE);
    const clone = document.getElementById('vp-clone');
    assert.equal(clone.localName, 'a');
    assert.equal(clone.hidden, false);
    assert.equal(clone.textContent, 'Clone a voice…');
    assert.equal(clone.getAttribute('href'), '#voices/clone');
    const body = [...document.getElementById('vp-list').parentElement.children];
    assert.ok(body.includes(note()) && body.indexOf(note()) < body.indexOf(document.getElementById('vp-list')), 'the note sits above the list');
    assert.equal(document.getElementById('vp-empty').hidden, true);
    await settle(env, 1000);
    assert.equal(document.getElementById('app-sr-status').textContent, NOTE, 'a screen reader hears it once the list is in');
    assert.equal(document.activeElement, ctx.search, 'focus stays in the search field');
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('with no recordings the note ends "No recordings yet." and Clone a voice… closes the picker and opens Voices › Clone', async () => {
    const ctx = await setup({ routes: cloning({ 'GET /api/voices/library': () => json(200, []), 'GET /api/voices/identities': () => json(200, { voices: [] }) }) });
    const { result } = await open(ctx, CHATTERBOX);
    assert.equal(noteText(), `${NOTE} No recordings yet.`);
    assert.equal(options().length, 0);
    assert.equal(document.getElementById('vp-empty').hidden, true, 'the note says it: no second "no voices" line');
    assert.equal(document.getElementById('vp-use').disabled, true);
    let prevented = null;
    const seen = (event) => { prevented = event.defaultPrevented; };
    document.addEventListener('click', seen);
    document.getElementById('vp-clone').click();
    document.removeEventListener('click', seen);
    assert.equal(prevented, false, "the link's own navigation does the routing");
    assert.equal(await result, null, 'no voice was chosen');
    assert.equal(ctx.dialog.open, false);
    assert.equal(location.hash, '#voices/clone');
  });

  it('a search that matches nothing still says "No voices match." under the note', async () => {
    const ctx = await setup({ routes: cloning() });
    const { result } = await open(ctx, CHATTERBOX);
    env.input(ctx.search, 'zzz');
    assert.equal(document.getElementById('vp-empty').hidden, false);
    assert.equal(document.getElementById('vp-empty').textContent, 'No voices match.');
    assert.equal(noteShown(), true);
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  // With nothing to list, a search announced the hidden "no voices" line.
  it('with nothing to list, a search announces "No voices match.", never the hidden "no voices" line', async () => {
    const ctx = await setup({ routes: cloning({ 'GET /api/voices/library': () => json(200, []), 'GET /api/voices/identities': () => json(200, { voices: [] }) }) });
    const { result } = await open(ctx, CHATTERBOX);
    assert.equal(options().length, 0);
    const said = () => document.getElementById('app-sr-status').textContent;
    env.input(ctx.search, 'a');
    await settle(env, 1200);
    assert.equal(document.getElementById('vp-empty').hidden, true, 'the note is the explanation');
    assert.equal(said(), 'No voices match.');
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('named voices with a version for the model sit between Recordings and the last group; a favorite the model cannot use is listed only in the last group', async () => {
    const named = { voices: [
      ...KOKORO_ONLY.voices,
      { id: 'c1', name: 'My clone', voice: 'voice:c1', available: true, realizations: [{ model: 'chatterbox/regular', voice: 'my_voice', reference_audio_id: 'my_voice', available: true }] },
      { id: 'c2', name: 'Lost clone', voice: 'voice:c2', available: false, realizations: [{ model: 'chatterbox/regular', voice: 'gone', available: false, reason: 'A saved reference recording is required for this model' }] },
    ] };
    const ctx = await setup({
      seed: { 'open-speech-voice-favorites': JSON.stringify(['chatterbox/regular|voice:k1', 'chatterbox/regular|my_voice']) },
      routes: cloning({ 'GET /api/voices/identities': () => json(200, named) }),
    });
    const { result } = await open(ctx, CHATTERBOX);
    assert.deepEqual(groupLabels(), ['Favorites', 'Recordings', 'Named voices', LAST]);
    assert.deepEqual(inGroup('Favorites'), ['my_voice'], 'a favorite the model can use keeps its place at the top');
    assert.deepEqual(inGroup('Recordings'), ['my_voice']);
    assert.deepEqual(inGroup('Named voices'), ['My clone', 'Lost clone']);
    assert.deepEqual(inGroup(LAST), ['Narrator', 'Storyteller'], 'the favorite it cannot use is here only');
    const lost = optionNamed('Lost clone');
    assert.equal(lost.getAttribute('aria-disabled'), 'true');
    assert.match(lost.textContent, /A saved reference recording is required for this model/, 'a version that cannot be used keeps its own reason');
    assert.match(optionNamed('Narrator').textContent, /favorite/, 'and it still reads as a favorite');
    env.click(document.getElementById('vp-favs'));
    assert.deepEqual(groupLabels(), ['Favorites', LAST], 'the Favorites filter follows the same rule');
    assert.deepEqual(inGroup('Favorites'), ['my_voice']);
    assert.deepEqual(inGroup(LAST), ['Narrator']);
    env.click(document.getElementById('vp-favs'));
    env.click(optionNamed('My clone'));
    assert.equal((await result).voice, 'voice:c1');
  });

  it('pickerRows and pickerGroups take the label of a model with no voices of its own (pure)', async () => {
    const ctx = await setup();
    const data = { voices: [], named: KOKORO_ONLY.voices, recordings: [{ name: 'my_voice' }] };
    const opts = { model: 'chatterbox/regular', include: { named: true, recordings: true, voices: true } };
    const rows = ctx.vp.pickerRows(data, { ...opts, noBuiltIn: 'Chatterbox regular' });
    assert.deepEqual(rows.filter((r) => r.kind === 'named').map((r) => [r.label, r.detail, r.disabled]), [
      ['Narrator', 'No Chatterbox regular version', true], ['Storyteller', 'No Chatterbox regular version', true],
    ]);
    assert.deepEqual(ctx.vp.pickerGroups(rows, { noBuiltIn: 'Chatterbox regular' }).map((g) => g.label), ['Recordings', LAST]);
    const plain = ctx.vp.pickerRows(data, opts);
    assert.deepEqual(plain.filter((r) => r.kind === 'named').map((r) => r.detail), ['Unavailable for this model', 'Unavailable for this model']);
    assert.deepEqual(ctx.vp.pickerGroups(plain, {}).map((g) => g.label), ['Named voices', 'Recordings'], 'without the label: the usual rows and order');
  });

  it('the last group reads "Named voices with no <Model> version" (never "without a F5 TTS version"); the row detail is "No <Model> version"', async () => {
    const ctx = await setup();
    const rows = ctx.vp.pickerRows({ voices: [], named: KOKORO_ONLY.voices }, { model: 'f5-tts', noBuiltIn: 'F5 TTS' });
    assert.deepEqual(ctx.vp.pickerGroups(rows, { noBuiltIn: 'F5 TTS' }).map((g) => g.label), ['Named voices with no F5 TTS version']);
    assert.deepEqual(rows.map((r) => r.detail), ['No F5 TTS version', 'No F5 TTS version']);
  });

  it('a model that cannot clone and has no voices gets the first sentence only, without Clone a voice…', async () => {
    const ctx = await setup({ routes: cloning() });
    const { result } = await open(ctx, { model: 'pocket-tts', value: '', modelLabel: 'Pocket TTS' });   // recordings are not offered
    assert.equal(noteShown(), true);
    assert.equal(noteText(), 'Pocket TTS has no built-in voices.');
    assert.equal(document.getElementById('vp-clone').hidden, true);
    assert.deepEqual(groupLabels(), ['Named voices with no Pocket TTS version']);
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('a model with voices of its own keeps its order and shows no note, also when the opener passes its label', async () => {
    const ctx = await setup();
    const { result } = await open(ctx, { modelLabel: 'Kokoro', include: { named: true, recordings: true, voices: true } });
    assert.deepEqual(groupLabels(), ['Named voices', 'Recordings', 'Kokoro · Recommended', 'Kokoro · American English', 'Kokoro · British English', 'Kokoro · Spanish']);
    assert.equal(noteShown(), false);
    assert.match(optionNamed('Chatter').textContent, /Unavailable for this model/);
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('an opener that passes no label (the blend editor; every voice picker of a view passes one) keeps the list it had for a model with no voices', async () => {
    const ctx = await setup({ routes: cloning() });
    const { result } = await open(ctx, { ...CHATTERBOX, modelLabel: undefined });
    assert.deepEqual(groupLabels(), ['Named voices', 'Recordings']);
    assert.match(optionNamed('Narrator').textContent, /Unavailable for this model/);
    assert.equal(noteShown(), false);
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('and with nothing to list, that opener still reads "This model has no voices to choose from."', async () => {
    const ctx = await setup({ routes: cloning({ 'GET /api/voices/library': () => json(200, []), 'GET /api/voices/identities': () => json(200, { voices: [] }) }) });
    const { result } = await open(ctx, { ...CHATTERBOX, modelLabel: undefined });
    assert.equal(document.getElementById('vp-empty').hidden, false);
    assert.equal(document.getElementById('vp-empty').textContent, 'This model has no voices to choose from.');
    assert.equal(noteShown(), false);
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });

  it('the note goes away when the picker opens next for a model with voices, and when the voices fail to load', async () => {
    let fail = false;
    let voiceList = { voices: [] };
    const ctx = await setup({ routes: cloning({ 'GET /v1/audio/voices': () => (fail ? json(500, { error: { message: 'boom' } }) : json(200, voiceList)) }) });
    const first = await open(ctx, CHATTERBOX);
    assert.equal(noteShown(), true);
    ctx.vp.__resetForTests();
    assert.equal(await first.result, null);
    voiceList = VOICES;
    const second = await open(ctx, { modelLabel: 'Kokoro' });
    assert.equal(noteShown(), false, 'Kokoro has voices of its own');
    ctx.vp.__resetForTests();
    assert.equal(await second.result, null);
    fail = true;
    const third = await open(ctx, CHATTERBOX);
    assert.equal(document.getElementById('vp-failed').hidden, false);
    assert.equal(noteShown(), false, 'a list that did not load is not an empty list');
    ctx.vp.__resetForTests();
    assert.equal(await third.result, null);
  });

  // A named voice whose version for Chatterbox clones a recording, and one with a Kokoro voice.
  const CLONED = { voices: [
    { id: 'c1', name: 'Butler', voice: 'voice:c1', available: true, realizations: [{ model: 'chatterbox/regular', voice: 'butler_take', reference_audio_id: 'butler_take', available: true, reason: null }] },
    { id: 'k1', name: 'Narrator', voice: 'voice:k1', available: true, realizations: [{ model: 'kokoro', voice: 'af_sky', reference_audio_id: null, available: true, reason: null }] },
  ] };

  it('a cloned named voice is offered like any other (every view can speak in it)', async () => {
    const ctx = await setup();
    const named = (rows) => rows.filter((r) => r.kind === 'named').map((r) => [r.label, r.detail, r.disabled]);
    assert.deepEqual(named(ctx.vp.pickerRows({ voices: [], named: CLONED.voices }, { model: 'chatterbox/regular', noBuiltIn: 'Chatterbox regular' })).slice(0, 1),
      [['Butler', 'butler_take', false]]);
    const models = [{ id: 'chatterbox/regular' }, { id: 'kokoro' }];
    assert.equal(ctx.vp.voiceOffered({ model: 'chatterbox/regular', voice: 'voice:c1' }, { models, named: CLONED.voices }), true);
  });

  it("a named voice the server also lists in /v1/audio/voices (kind 'named_voice') shows once, in the cloning model's order", async () => {
    const listed = { voices: [{ id: 'voice:c1', name: 'Butler', model: 'chatterbox/regular', kind: 'named_voice' }] };
    const ctx = await setup({ routes: cloning({ 'GET /v1/audio/voices': () => json(200, listed), 'GET /api/voices/identities': () => json(200, CLONED) }) });
    const { result } = await open(ctx, CHATTERBOX);
    assert.deepEqual(labels().filter((l) => l === 'Butler'), ['Butler']);
    assert.deepEqual(groupLabels(), ['Recordings', 'Named voices']);
    assert.equal(noteShown(), true, 'the model still has no built-in voices');
    ctx.vp.__resetForTests();
    assert.equal(await result, null);
  });
});


describe('components/voice-picker: the Piper voice models as voices', () => {
  const AMY = 'piper/en_US-amy-medium';
  const ARCTIC = 'piper/en_US-arctic-medium';
  const LESSAC = ['piper/en_US-lessac-low', 'piper/en_US-lessac-high'];
  const PIPERS = [AMY, ARCTIC, ...LESSAC];
  const PIPER_NAMED = { voices: [
    { id: 'n1', name: 'Narrator', voice: 'voice:n1', realizations: [{ model: 'kokoro', voice: 'af_bella' }] },
    { id: 'n4', name: 'Calm', voice: 'voice:n4', realizations: [{ model: 'kokoro', voice: 'af_sky' }, { model: 'piper/en_US-lessac-high', voice: 'piper/en_US-lessac-high' }] },
  ] };
  const routes = (speakers) => ({
    'GET /api/voices/identities': () => json(200, PIPER_NAMED),
    'GET /v1/audio/voices': (req) => {
      const model = req.query.get('model');
      if (model === ARCTIC) return speakers();
      // The server could not fetch this voice model's config.
      if (model === LESSAC[1]) return json(503, { error: { message: 'Hub unreachable' } });
      return json(200, { voices: [{ id: model, name: model.slice(6) }] });
    },
  });
  const arctic = () => json(200, { voices: [{ id: `${ARCTIC}/awb`, name: 'awb' }, { id: `${ARCTIC}/bdl`, name: 'bdl' }] });
  const piperOpen = (ctx, options = {}) => open(ctx, { model: 'piper', modelLabel: 'Piper', value: AMY, voiceModels: PIPERS, ...options });

  it('pickerRows: each voice model is a row with its model; a named voice uses its version for one of them, else is unavailable; favourites are per voice model', async () => {
    const ctx = await setup();
    const voices = (await load('core/voices.js')).modelVoices(PIPERS);
    const rows = ctx.vp.pickerRows({ voices, named: PIPER_NAMED.voices }, { model: 'piper', voiceModels: PIPERS, favorites: [`${AMY}|${AMY}`, 'piper/en_US-lessac-high|voice:n4'], extraRows: [{ id: 'same-as-speak', voice: AMY, label: 'Same as Speak · Piper · Amy (US)', model: AMY }] });
    assert.deepEqual(rows.map((r) => [r.kind, r.voice, r.model || '', r.disabled, r.favourite]), [
      ['extra', AMY, AMY, false, false],
      ['named', 'voice:n1', '', true, false],
      ['named', 'voice:n4', 'piper/en_US-lessac-high', false, true],
      ['voice', AMY, AMY, false, true],
      ['voice', ARCTIC, ARCTIC, false, false],
      ['voice', LESSAC[0], LESSAC[0], false, false],
      ['voice', LESSAC[1], LESSAC[1], false, false],
    ]);
    assert.equal(rows[1].detail, 'Unavailable for this model');
    assert.deepEqual(ctx.vp.pickerGroups(rows, { modelName: 'Piper' }).map((g) => g.label), ['Suggested', 'Favorites', 'Named voices', 'Piper · Voices']);
    assert.equal(ctx.vp.selectedKey(rows, `${ARCTIC}/awb`), `v|${ARCTIC}`, "a speaker selects its voice model's row");
    assert.equal(ctx.vp.selectedKey(rows, 'piper/en_US-gone-medium/x'), null);
    const plain = ctx.vp.pickerRows({ voices: VOICES.voices }, { model: 'kokoro' });
    assert.equal(plain.some((r) => 'model' in r), false, 'rows of one model carry no model of their own');
  });

  it('lists the voice models with no request, previews and stars each with its own model, and a choice carries that model', async () => {
    const ctx = await setup({ routes: routes(arctic) });
    const { result } = await piperOpen(ctx);
    assert.equal(env.fetch.callsTo('GET /v1/audio/voices').length, 0);
    assert.equal(env.fetch.callsTo('GET /api/voices/identities')[0].query.get('check_availability'), 'false', 'named voices of every Piper voice model');
    assert.equal(optionNamed('Amy (US)').getAttribute('aria-selected'), 'true');
    assert.equal(optionNamed('Calm').getAttribute('aria-disabled'), null, 'its Lessac (US) version speaks');
    env.key('Enter', { shiftKey: true });
    await settle(env);
    assert.deepEqual(env.fetch.callsTo('POST /v1/audio/speech')[0].json.model, AMY, 'the preview speaks with that voice model');
    env.click(document.getElementById('vp-fav'));
    assert.deepEqual(JSON.parse(env.localStorage.getItem('open-speech-voice-favorites')), [`${AMY}|${AMY}`], 'the key a Piper voice has always had');
    env.click(optionNamed('Calm'));
    assert.deepEqual(await result, { kind: 'named', id: 'n4', voice: 'voice:n4', label: 'Calm', detail: 'Lessac (US)', model: 'piper/en_US-lessac-high' });
  });

  it("choosing a voice model asks for its speakers once: one with a single voice is the choice, one with several lists them, a failed check chooses the model's own voice", async () => {
    const ctx = await setup({ routes: routes(arctic) });
    let { result } = await piperOpen(ctx);
    env.click(options().find((o) => o.textContent.includes('Low quality')));
    await settle(env);
    assert.deepEqual(await result, { kind: 'voice', id: LESSAC[0], voice: LESSAC[0], label: 'Lessac (US)', detail: 'Low quality', model: LESSAC[0] });
    assert.equal(env.fetch.callsTo('GET /v1/audio/voices')[0].query.get('model'), LESSAC[0]);
    ({ result } = await piperOpen(ctx, { value: `${ARCTIC}/bdl` }));
    env.click(optionNamed('Arctic (US)'));
    await settle(env);
    assert.equal(ctx.dialog.open, true);
    assert.equal(document.getElementById('vp-title').textContent, 'Arctic (US) speakers');
    assert.deepEqual(options().map((o) => o.querySelector('.option__label').textContent), ['awb', 'bdl']);
    assert.equal(optionNamed('bdl').getAttribute('aria-selected'), 'true', 'the speaker in use');
    assert.equal(document.activeElement, ctx.search);
    env.key('ArrowUp');
    env.key('Enter');
    await settle(env);
    assert.deepEqual(await result, { kind: 'voice', id: `${ARCTIC}/awb`, voice: `${ARCTIC}/awb`, label: 'awb', detail: 'Arctic (US) · Medium quality', model: ARCTIC });
    // Esc on the speakers closes with no choice.
    ({ result } = await piperOpen(ctx));
    env.click(optionNamed('Arctic (US)'));
    await settle(env);
    env.key('Escape');
    assert.equal(await result, null);
    assert.equal(env.fetch.callsTo('GET /v1/audio/voices').filter((c) => c.query.get('model') === ARCTIC).length, 1, 'its speakers are kept');
    // A voice model whose check fails: its own voice, the model's default speaker.
    ({ result } = await piperOpen(ctx));
    env.click(options().find((o) => o.textContent.includes('High quality')));
    await settle(env);
    await settle(env);
    assert.deepEqual(await result, { kind: 'voice', id: LESSAC[1], voice: LESSAC[1], label: 'Lessac (US)', detail: 'High quality', model: LESSAC[1] });
    assert.ok(env.fetch.callsTo('GET /v1/audio/voices').some((c) => c.query.get('model') === LESSAC[1]), 'it was asked');
  });

  it('a named voice with versions for several voice models uses the one for the first of them (the model in use)', async () => {
    const ctx = await setup();
    const both = { id: 'n5', name: 'Both', voice: 'voice:n5', realizations: [{ model: AMY, voice: AMY }, { model: LESSAC[1], voice: LESSAC[1] }] };
    const rowsFor = (voiceModels) => ctx.vp.pickerRows({ named: [both] }, { model: 'piper', voiceModels, favorites: [`${LESSAC[1]}|voice:n5`] })[0];
    assert.deepEqual([rowsFor([LESSAC[1], AMY, ARCTIC]).model, rowsFor([LESSAC[1], AMY, ARCTIC]).favourite], [LESSAC[1], true]);
    assert.deepEqual([rowsFor([AMY, LESSAC[1]]).model, rowsFor([AMY, LESSAC[1]]).favourite], [AMY, false]);
  });

  it('choosing a voice model with several speakers with Use this voice moves focus to the search field', async () => {
    const ctx = await setup({ routes: routes(arctic) });
    const { result } = await piperOpen(ctx);
    env.key('ArrowDown');
    while (document.getElementById(ctx.search.getAttribute('aria-activedescendant')).querySelector('.option__label').textContent !== 'Arctic (US)') env.key('ArrowDown');
    const use = document.getElementById('vp-use');
    use.focus();
    env.click(use);
    await settle(env);
    assert.equal(document.getElementById('vp-title').textContent, 'Arctic (US) speakers');
    assert.equal(document.activeElement, ctx.search);
    env.key('Escape');
    assert.equal(await result, null);
  });
});
