// Voices: the recordings panel (inventory VL-11…VL-18).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames, accessibleName } from '../harness/a11y.js';
import { voicesEnv, MODELS, MODELS_NO_CLONE, recordings, namedVoices, json, bytes, load, delay } from './helpers.js';

let ctx;
afterEach(() => ctx?.env.restore());

const TEST_SENTENCE = 'This is an Open Speech voice-cloning test using the saved recording.';
const cardOf = (name) => document.getElementById(`vo-rec-${name}`);
const part = (name, p) => document.getElementById(`vo-rec-${name}-${p}`);
const text = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');

const WITH_FISH = {
  ...MODELS,
  models: [...MODELS.models, { id: 'fish', type: 'tts', provider: 'fish', state: 'downloaded', provider_available: true, capabilities: { voice_clone: true, speed: true } }],
};

describe('views/voices recordings', () => {
  it('[VL-11] each recording is a card: name, length · kHz · mono, the transcript or "No transcript", clone-model badges and actions', async () => {
    ctx = await voicesEnv({ sub: 'recordings' });
    const demo = cardOf('demo_reference');
    assert.equal(text(demo.querySelector('h2')), 'demo_reference');
    assert.equal(text(demo.querySelector('.vo-card__meta')), '6.0 s · 24 kHz · mono');
    assert.equal(text(demo.querySelector('.vo-rec__transcript')), 'The quick brown fox jumps over the lazy dog.');
    assert.deepEqual([...demo.querySelectorAll('.vo-rec__compat .badge')].map(text), ['Chatterbox']);
    assert.ok(demo.querySelector('.vo-rec__compat .badge--ok'), 'a compatible model is a success badge');
    const raw = cardOf('raw_take');
    assert.equal(text(raw.querySelector('.vo-card__meta')), '8.4 s · 48 kHz · mono');
    const missing = raw.querySelector('.vo-rec__transcript--missing');
    assert.match(text(missing), /^No transcript\./, 'warn tone with words, never colour alone');
    assert.deepEqual([...raw.querySelectorAll('.vo-rec__compat .badge')].map(text), ['Chatterbox · needs a transcript'], 'greyed with the reason');
    assert.deepEqual(['preview', 'use', 'test'].map((p) => text(part('demo_reference', p))), ['Preview', 'Use in Speak', 'Test clone']);
    const menu = document.getElementById(part('demo_reference', 'more').getAttribute('aria-controls'));
    assert.deepEqual([...menu.querySelectorAll('[role="menuitem"]')].map(text), ['Edit transcript…', 'Save as preset…', 'Delete…']);
    assert.equal(accessibleName(part('demo_reference', 'more')), 'More actions for demo_reference');
    assertAccessibleNames(ctx.section);
  });

  it('[VL-11] with no recordings the panel says so and offers the clone wizard', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'GET /api/voices/library': () => json(200, []) } });
    assert.equal(text(ctx.$('#vo-rec-list .empty__title')), 'No recordings yet');
    const start = ctx.$('#vo-rec-list [data-action="clone"]');
    assert.equal(text(start), 'Clone your first voice');
    ctx.env.click(start);
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices/clone');
  });

  it('one primary action per page: an empty list that offers its own demotes the header Clone a voice, as the Library does', async () => {
    let recs = [];
    let named = { voices: [] };
    ctx = await voicesEnv({
      sub: 'recordings',
      routes: {
        'GET /api/voices/library': () => json(200, recs),
        'GET /api/voices/identities': () => json(200, named),
        'GET /api/profiles': () => json(200, { profiles: [], default_profile_id: null }),
      },
    });
    const header = document.getElementById('vo-clone');
    // The filled buttons on screen (a closed sheet's Save and the hidden wizard's are not).
    const primaries = () => ctx.$$('.btn--primary').filter((b) => !b.closest('[hidden]') && !(b.closest('dialog') && !b.closest('dialog').open)).map(text);
    assert.deepEqual(primaries(), ['Clone your first voice'], 'Recordings: the empty state\'s button, not the same action twice');
    assert.ok(header.classList.contains('btn--secondary') && !header.classList.contains('btn--primary'));
    await ctx.go('voices');
    assert.deepEqual(primaries(), ['Open Speak'], 'Voices (named voices and presets, one list): never two filled buttons on one screen');
    assert.ok(header.classList.contains('btn--secondary'));
    // Named voices arrive (Speak saved one): a list with items, so the header is the primary there again.
    named = namedVoices();
    ctx.store.emit('voices:changed', { what: 'named' });
    await ctx.settle();
    await ctx.settle();
    assert.deepEqual(primaries(), ['Clone a voice']);
    // A recording is saved: the empty state goes, and so does the demotion.
    await ctx.go('voices/recordings');
    assert.deepEqual(primaries(), ['Clone your first voice']);
    recs = recordings();
    ctx.store.emit('voices:changed', { what: 'recordings' });
    await ctx.settle();
    await ctx.settle();
    assert.ok(cardOf('demo_reference'));
    assert.deepEqual(primaries(), ['Clone a voice']);
    // A search with no match offers only Clear search (secondary): the header stays the primary.
    ctx.env.input(document.getElementById('vo-search'), 'zzz');
    await ctx.settle(200);
    assert.ok(ctx.$('#vo-rec-list .vo-nomatch'));
    assert.deepEqual(primaries(), ['Clone a voice']);
  });

  it('[VL-12] Preview plays the recording itself through api.mediaUrl as an ephemeral clip; Stop ends it and revokes the URL', async () => {
    ctx = await voicesEnv({
      sub: 'recordings', seed: { 'open-speech-api-key': 'k-123' },
      routes: { 'GET /api/voices/library/:name/audio': () => bytes(200, [82, 73, 70, 70], 'audio/wav') },
    });
    (await load('core/auth.js')).init();
    const audio = await load('core/audio.js');
    ctx.env.click(part('demo_reference', 'preview'));
    await ctx.settle();
    await ctx.settle();
    const get = ctx.calls('GET /api/voices/library/:name/audio');
    assert.equal(get.length, 1, 'with a key the audio is fetched with the Authorization header');
    assert.equal(get[0].headers.authorization || get[0].headers.Authorization, 'Bearer k-123');
    const session = audio.current();
    assert.equal(session.id, 'preview');
    assert.equal(session.ephemeral, true, 'previews never show in the mini-player');
    const src = audio.element().src;
    assert.match(src, /^blob:/);
    assert.equal(text(part('demo_reference', 'preview')), 'Stop');
    assert.equal(accessibleName(part('demo_reference', 'preview')), 'Stop preview: demo_reference');
    ctx.env.click(part('demo_reference', 'preview'));
    await ctx.settle();
    assert.equal(audio.current(), null);
    assert.ok(ctx.env.revokedUrls.includes(src), 'the object URL is revoked when the preview ends');
    assert.equal(text(part('demo_reference', 'preview')), 'Preview');
  });

  it('[VL-12] a preview stops when its panel is left (it has no mini-player); a clone test keeps playing', async () => {
    ctx = await voicesEnv({
      sub: 'recordings',
      routes: {
        'GET /api/voices/library/:name/audio': () => bytes(200, [82, 73, 70, 70], 'audio/wav'),
        'POST /v1/audio/speech': () => bytes(200, [1, 2, 3], 'audio/wav'),
      },
    });
    const audio = await load('core/audio.js');
    ctx.env.click(part('demo_reference', 'preview'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(audio.current().id, 'preview');
    await ctx.go('voices');
    assert.equal(audio.current(), null, 'switching to Voices stopped the preview');
    await ctx.go('voices/recordings');
    assert.equal(text(part('demo_reference', 'preview')), 'Preview');
    ctx.env.click(part('demo_reference', 'preview'));
    await ctx.settle();
    await ctx.settle();
    await ctx.go('speak');
    assert.equal(audio.current(), null, 'leaving Voices stopped the preview');
    await ctx.go('voices/recordings');
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    await ctx.go('speak');
    assert.equal(audio.current().id, 'clone-test', 'a clone test is a normal clip: it plays on, in the mini-player');
  });

  it('[VL-12] while the microphone records, Preview and Test clone are unavailable and say to stop recording first', async () => {
    ctx = await voicesEnv({
      sub: 'recordings',
      routes: {
        'GET /api/voices/library/:name/audio': () => bytes(200, [82, 73, 70, 70], 'audio/wav'),
        'POST /v1/audio/speech': () => bytes(200, [1, 2, 3], 'audio/wav'),
      },
    });
    const audio = await load('core/audio.js');
    const mic = await load('core/mic.js');
    // A Speak take plays, then Transcribe starts recording in the background (opening the microphone pauses it).
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', homeRoute: 'speak', src: 'blob:take1' });
    const capture = await mic.open({ owner: 'transcribe' });
    await ctx.settle();
    const note = document.getElementById('vo-rec-note');
    const preview = part('demo_reference', 'preview');
    const test = part('demo_reference', 'test');
    assert.equal(note.hidden, false);
    assert.equal(text(note), 'Stop recording to listen.');
    assert.equal(preview.getAttribute('aria-disabled'), 'true');
    assert.equal(preview.getAttribute('aria-describedby'), 'vo-rec-note', 'the visible note says why');
    assert.equal(test.getAttribute('aria-disabled'), 'true');
    assert.equal(test.getAttribute('aria-describedby'), 'vo-rec-note');
    assert.equal(part('demo_reference', 'use').getAttribute('aria-disabled'), null, 'Use in Speak plays nothing, so it stays available');
    ctx.env.click(test);
    ctx.env.click(preview);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/speech').length, 0, 'no synthesis that could not be heard');
    assert.equal(ctx.calls('GET /api/voices/library/:name/audio').length, 0);
    assert.equal(audio.current()?.id, 'take:1', 'the paused take keeps its place in the mini-player (prime() never ran)');
    capture.stop();
    await ctx.settle();
    assert.equal(note.hidden, true);
    assert.equal(preview.getAttribute('aria-disabled'), null);
    assert.equal(preview.getAttribute('aria-describedby'), null);
    assert.equal(test.getAttribute('aria-disabled'), null);
    assert.equal(part('demo_reference', 'preview'), preview, 'the buttons were updated in place, not rebuilt');
  });

  it('[VL-12] with an API key the last Preview pressed wins, and a second press while its audio loads downloads nothing more', async () => {
    ctx = await voicesEnv({
      sub: 'recordings',
      seed: { 'open-speech-api-key': 'k-123' },
      routes: {
        'GET /api/voices/library/:name/audio': (req) => (req.params.name === 'demo_reference'
          ? delay(2000, bytes(200, [82, 73, 70, 70], 'audio/wav'))
          : delay(100, bytes(200, [82, 73, 70, 71], 'audio/wav'))),
      },
    });
    (await load('core/auth.js')).init();
    const audio = await load('core/audio.js');
    await (await load('core/api.js')).get('/api/voices/library');   // the key is confirmed: requests don't wait on each other
    ctx.env.click(part('demo_reference', 'preview'));
    await ctx.settle();
    assert.equal(part('demo_reference', 'preview').getAttribute('aria-busy'), 'true', 'busy while its audio downloads');
    ctx.env.click(part('demo_reference', 'preview'));
    ctx.env.click(part('raw_take', 'preview'));
    await ctx.settle(150);
    await ctx.settle();
    assert.equal(audio.current()?.title, 'raw_take');
    assert.equal(part('demo_reference', 'preview').getAttribute('aria-busy'), null);
    await ctx.settle(2000);
    await ctx.settle();
    assert.equal(audio.current()?.title, 'raw_take', 'the earlier press, answered later, never replaces the later one');
    assert.equal(text(part('raw_take', 'preview')), 'Stop');
    assert.equal(text(part('demo_reference', 'preview')), 'Preview');
    assert.equal(ctx.calls('GET /api/voices/library/:name/audio').length, 2, 'the second press on the same card downloaded nothing');
  });

  it('[VL-12] a preview that ends while a card menu is open leaves the menu open, and a reload opens it again on the same item', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'GET /api/voices/library/:name/audio': () => bytes(200, [82, 73, 70, 70], 'audio/wav') } });
    const audio = await load('core/audio.js');
    ctx.env.click(part('demo_reference', 'preview'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(text(part('demo_reference', 'preview')), 'Stop');
    const more = part('raw_take', 'more');
    more.focus();
    ctx.env.key('Enter', {}, more);
    await ctx.settle();
    ctx.env.key('ArrowDown');
    const menu = document.getElementById(more.getAttribute('aria-controls'));
    const item = document.activeElement;
    assert.equal(text(item), 'Save as preset…');
    audio.element().__end();
    await ctx.settle();
    await ctx.settle();
    assert.equal(text(part('demo_reference', 'preview')), 'Preview', 'the card says the preview is over');
    assert.equal(menu.hidden, false, 'the open menu stays open');
    assert.equal(document.activeElement, item, 'and focus stays on its item, never <body>');
    assert.equal(part('raw_take', 'more'), more, 'nothing was rebuilt');
    // Another view changed the recordings: the list reloads and re-renders under the open menu.
    ctx.store.emit('voices:changed', { what: 'recordings' });
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('GET /api/voices/library').length, 2);
    const reopened = document.getElementById('vo-rec-raw_take-menu');
    assert.equal(reopened.hidden, false, 'the menu that had focus is open again');
    assert.ok(reopened.contains(document.activeElement));
    assert.equal(text(document.activeElement), 'Save as preset…', 'on the same item');
  });

  it('[VL-13] the clone-model select lists the models that can clone, keeps the choice across refreshes, and says when there is none', async () => {
    ctx = await voicesEnv({ sub: 'recordings', models: WITH_FISH });
    const select = document.getElementById('vo-clone-model');
    assert.deepEqual([...select.querySelectorAll('option')].map((o) => o.value), ['chatterbox', 'fish'], 'Kokoro and Piper cannot clone');
    ctx.env.change(select, 'fish');
    await ctx.settle();
    await ctx.modelsApi.refresh();
    ctx.app.set({ models: { ...ctx.app.get().models, list: [...WITH_FISH.models.slice().reverse()] } });
    await ctx.settle();
    assert.equal(select.value, 'fish', 'the choice survives a model refresh');
    ctx.env.restore();
    ctx = await voicesEnv({ sub: 'recordings', models: MODELS_NO_CLONE });
    const none = document.getElementById('vo-clone-model');
    assert.equal(none.disabled, true);
    assert.equal(text(none.querySelector('option')), 'No clone model available');
    assert.ok(document.querySelector('#vo-clone-model-hint a[href="#models/tts"]'), 'a link to Models');
    assert.equal(part('demo_reference', 'use').getAttribute('aria-disabled'), 'true');
    assert.equal(part('demo_reference', 'test').getAttribute('aria-disabled'), 'true');
    const menu = document.getElementById(part('demo_reference', 'more').getAttribute('aria-controls'));
    assert.equal(menu.querySelector('[data-menu-action="preset"]').getAttribute('aria-disabled'), 'true');
  });

  it('[VL-14] Test clone POSTs /v1/audio/speech with voice and voice_library_ref set to the name, WAV and no speed, then plays it', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'POST /v1/audio/speech': () => bytes(200, [1, 2, 3], 'audio/wav') } });
    const audio = await load('core/audio.js');
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('GET /api/models/:id/status').length, 0, 'a model known to be loaded needs no status request (the fast path)');
    const posts = ctx.calls('POST /v1/audio/speech');
    assert.equal(posts.length, 1);
    assert.deepEqual(posts[0].json, { model: 'chatterbox', voice: 'demo_reference', input: TEST_SENTENCE, response_format: 'wav', voice_library_ref: 'demo_reference' });
    assert.equal('speed' in posts[0].json, false, 'no speed: the server default passes models without speed control');
    assert.equal(posts[0].headers['x-history'] || posts[0].headers['X-History'], undefined, 'tests stay out of History');
    const session = audio.current();
    assert.equal(session.id, 'clone-test');
    assert.equal(session.ephemeral, false, 'a clone test can be stopped from the mini-player');
    assert.equal(session.homeRoute, 'voices/recordings');
    assert.equal(text(part('demo_reference', 'status')), 'Clone test complete with Chatterbox.');
  });

  it('[VL-14] while a test runs its button is busy and the other Test clone buttons wait', async () => {
    ctx = await voicesEnv({
      sub: 'recordings',
      models: WITH_FISH,
      routes: {
        'POST /v1/audio/speech': () => delay(1000, bytes(200, [1, 2, 3], 'audio/wav')),
        'GET /api/models/:id/status': () => json(200, { state: 'loaded' }),
      },
    });
    ctx.env.change(document.getElementById('vo-clone-model'), 'fish');
    await ctx.settle();
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(part('demo_reference', 'test').getAttribute('aria-busy'), 'true');
    assert.equal(text(part('demo_reference', 'test')), 'Testing…');
    assert.equal(accessibleName(part('demo_reference', 'test')), 'Testing: demo_reference', 'the name starts with what the button shows');
    assert.equal(part('raw_take', 'test').getAttribute('aria-disabled'), 'true', 'one clone test at a time');
    assert.equal(text(part('demo_reference', 'status')), 'Generating a clone test with Fish Audio S2…');
    await ctx.settle(1100);
    await ctx.settle();
    assert.equal(part('raw_take', 'test').getAttribute('aria-disabled'), null);
    assert.equal(text(part('demo_reference', 'test')), 'Stop test', 'while the test clip plays, its button stops it');
    assert.equal(accessibleName(part('demo_reference', 'test')), 'Stop test: demo_reference');
  });

  it('[VL-14] Stop test ends the clip; a clip the browser left paused offers Play test, which plays it inside the click', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'POST /v1/audio/speech': () => bytes(200, [1, 2, 3], 'audio/wav') } });
    const audio = await load('core/audio.js');
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(audio.current().id, 'clone-test');
    assert.equal(text(part('demo_reference', 'test')), 'Stop test');
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    assert.equal(audio.current(), null, 'Stop test released the clip');
    assert.equal(text(part('demo_reference', 'test')), 'Test clone');
    assert.equal(ctx.calls('POST /v1/audio/speech').length, 1, 'stopping sends nothing');
    // Autoplay refused (a browser without a recent gesture): the session waits, paused.
    audio.element().__playPolicy = 'NotAllowedError';
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(audio.current().id, 'clone-test');
    assert.equal(text(part('demo_reference', 'test')), 'Play test', 'never a Stop for a clip that is not playing');
    audio.element().__playPolicy = 'resolve';
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    assert.equal(audio.element().paused, false, 'Play test plays the clip that is already loaded');
    assert.equal(text(part('demo_reference', 'test')), 'Stop test');
    assert.equal(ctx.calls('POST /v1/audio/speech').length, 2, 'no second synthesis for Play test');
  });

  it('[VL-14] a model that needs the exact words can\'t test a recording without a transcript, and says why', async () => {
    ctx = await voicesEnv({ sub: 'recordings' });
    const button = part('raw_take', 'test');
    assert.equal(button.getAttribute('aria-disabled'), 'true');
    const hint = document.getElementById(button.getAttribute('aria-describedby'));
    assert.equal(text(hint), 'Chatterbox requires an exact reference transcript');
    ctx.env.click(button);
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/speech').length, 0);
  });

  it('[VL-14] a failed clone test shows the server\'s reason on the card; declining the model switch cancels it', async () => {
    let state = 'loaded';
    ctx = await voicesEnv({
      sub: 'recordings',
      routes: {
        'POST /v1/audio/speech': () => json(400, { error: { message: 'Reference audio is too short for this model', code: 'invalid_reference' } }),
        'GET /api/models/:id/status': () => json(200, { state }),
      },
    });
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    const banner = part('demo_reference', 'error').querySelector('.banner');
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(text(banner.querySelector('.banner__title')), 'Clone test failed');
    assert.equal(text(banner.querySelector('.banner__msg')), 'Reference audio is too short for this model');
    assert.equal(banner.querySelector('details pre').textContent, 'Reference audio is too short for this model (invalid_reference)', 'Details read as the server message and code, never the JSON envelope');
    assert.equal(part('demo_reference', 'status').hidden, true, 'the failure is said once, by the alert banner, not again by the status line');
    assert.equal(text(part('demo_reference', 'test')), 'Test clone', 'never stuck on Testing…');
    // A model that isn't loaded while Kokoro is: the switch is confirmed first.
    state = 'downloaded';
    ctx.app.set({ models: { ...ctx.app.get().models, list: ctx.app.get().models.list.map((m) => (m.id === 'chatterbox' ? { ...m, state: 'downloaded' } : m)) } });
    await ctx.settle();
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.confirmText().title, 'Switch speech model?');
    await ctx.answerConfirm(false);
    assert.equal(text(part('demo_reference', 'status')), 'Clone test canceled.');
    assert.equal(ctx.calls('POST /v1/audio/speech').length, 1, 'nothing more is sent');
  });

  it("[VL-14] a card's error and status survive a re-render (a model refresh) until dismissed", async () => {
    let fail = false;
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'POST /v1/audio/speech': () => (fail ? json(500, { error: { message: 'Worker crashed' } }) : bytes(200, [1, 2, 3], 'audio/wav')) } });
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(text(part('demo_reference', 'status')), 'Clone test complete with Chatterbox.');
    ctx.app.set({ models: { ...ctx.app.get().models, list: WITH_FISH.models } });
    await ctx.settle();
    assert.equal(text(part('demo_reference', 'status')), 'Clone test complete with Chatterbox.', 'the status line survives the re-render');
    ctx.env.click(part('demo_reference', 'test'));   // Stop test
    await ctx.settle();
    fail = true;
    ctx.env.click(part('demo_reference', 'test'));
    await ctx.settle();
    await ctx.settle();
    assert.match(text(part('demo_reference', 'error')), /Worker crashed/);
    ctx.app.set({ models: { ...ctx.app.get().models, list: MODELS.models } });
    await ctx.settle();
    assert.equal(document.querySelectorAll('#vo-clone-model option').length, 1, 'the list re-rendered for the changed clone models');
    assert.match(text(part('demo_reference', 'error')), /Worker crashed/, 'the error is still on the card');
    ctx.app.set({ models: { ...ctx.app.get().models, list: WITH_FISH.models } });
    await ctx.settle();
    ctx.env.click(part('demo_reference', 'error').querySelector('[data-action="dismiss-banner"]'));
    await ctx.settle();
    assert.equal(part('demo_reference', 'error').hidden, true);
    ctx.app.set({ models: { ...ctx.app.get().models, list: MODELS.models } });
    await ctx.settle();
    assert.equal(part('demo_reference', 'error').hidden, true, 'a dismissed error stays dismissed');
  });

  it('[VL-15][TE-09] Edit transcript opens a sheet with the words; Save PATCHes {transcript} (empty sends null) and tells the other views; Cancel sends nothing', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'PATCH /api/voices/library/:name': (req) => json(200, { name: req.params.name, transcript: req.json.transcript }) } });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    const sheet = document.getElementById('vo-transcript-sheet');
    await ctx.menu('vo-rec-demo_reference-more', 'transcript');
    assert.equal(sheet.open, true);
    const area = document.getElementById('vo-transcript-edit');
    assert.equal(area.value, 'The quick brown fox jumps over the lazy dog.');
    assert.equal(document.activeElement?.id, 'vo-transcript-edit');
    assert.equal(text(document.getElementById('vo-transcript-for')), 'demo_reference');
    ctx.env.click(sheet.querySelector('[data-close]'));
    await ctx.settle();
    assert.equal(sheet.open, false);
    assert.equal(ctx.calls('PATCH /api/voices/library/:name').length, 0);
    assert.deepEqual(events, [], 'Cancel changes nothing anywhere');
    assert.equal(document.activeElement?.id, 'vo-rec-demo_reference-more', 'focus returns to the menu button');
    await ctx.menu('vo-rec-demo_reference-more', 'transcript');
    ctx.env.input(area, '  The quick brown fox.  ');
    ctx.env.click(document.getElementById('vo-transcript-save'));
    await ctx.settle();
    await ctx.settle();
    let patches = ctx.calls('PATCH /api/voices/library/:name');
    assert.equal(patches[0].path, '/api/voices/library/demo_reference');
    assert.deepEqual(patches[0].json, { transcript: 'The quick brown fox.' });
    assert.equal(sheet.open, false);
    assert.ok(ctx.toasts().includes('Updated the transcript for demo_reference'));
    assert.equal(ctx.calls('GET /api/voices/library').length, 2, 'the list reloads');
    assert.deepEqual(events, ['recordings'], "Speak hears of it: its reference options say whether a transcript is saved");
    await ctx.menu('vo-rec-demo_reference-more', 'transcript');
    ctx.env.input(area, '   ');
    ctx.env.click(document.getElementById('vo-transcript-save'));
    await ctx.settle();
    await ctx.settle();
    patches = ctx.calls('PATCH /api/voices/library/:name');
    assert.deepEqual(patches[1].json, { transcript: null }, 'an emptied transcript is removed');
    assert.ok(ctx.toasts().includes('Removed the transcript for demo_reference'));
    assert.deepEqual(events, ['recordings', 'recordings']);
    // Saving words that did not change sends nothing and just closes the sheet.
    await ctx.menu('vo-rec-raw_take-more', 'transcript');
    assert.equal(area.value, '', 'a recording without a transcript opens empty');
    ctx.env.click(document.getElementById('vo-transcript-save'));
    await ctx.settle();
    assert.equal(sheet.open, false);
    assert.equal(ctx.calls('PATCH /api/voices/library/:name').length, 2);
    assert.deepEqual(events, ['recordings', 'recordings'], 'unchanged words: no event either');
  });

  it('[VL-15] a transcript save that ends after the sheet was closed still says how it went, by a toast', async () => {
    let fail = true;
    ctx = await voicesEnv({
      sub: 'recordings',
      routes: {
        'PATCH /api/voices/library/:name': (req) => delay(500, fail
          ? json(500, { error: { message: 'disk full' } })
          : json(200, { name: req.params.name, transcript: req.json.transcript })),
      },
    });
    const sheet = document.getElementById('vo-transcript-sheet');
    await ctx.menu('vo-rec-demo_reference-more', 'transcript');
    ctx.env.input(document.getElementById('vo-transcript-edit'), 'New words.');
    ctx.env.click(document.getElementById('vo-transcript-save'));
    await ctx.settle();
    ctx.env.click(sheet.querySelector('[data-close]'));   // Cancel while it saves
    await ctx.settle();
    assert.equal(sheet.open, false);
    await ctx.settle(600);
    await ctx.settle();
    assert.ok(ctx.toasts().includes("Couldn't save the transcript for demo_reference: disk full"), 'the failure is said where the listener is');
    const toastEl = [...document.querySelectorAll('#app-toasts .toast')].find((t) => /disk full/.test(t.textContent));
    assert.ok(toastEl.classList.contains('toast--error'));
    assert.equal(document.getElementById('vo-transcript-error').hidden, true, 'never a banner inside a closed sheet');
    // A save that succeeds after Esc closed the sheet did save: the toast says so and the list reloads.
    fail = false;
    await ctx.menu('vo-rec-demo_reference-more', 'transcript');
    ctx.env.input(document.getElementById('vo-transcript-edit'), 'Newer words.');
    ctx.env.click(document.getElementById('vo-transcript-save'));
    await ctx.settle();
    ctx.env.key('Escape');
    await ctx.settle();
    assert.equal(sheet.open, false);
    const loads = ctx.calls('GET /api/voices/library').length;
    await ctx.settle(600);
    await ctx.settle();
    assert.ok(ctx.toasts().includes('Updated the transcript for demo_reference'));
    assert.equal(ctx.calls('GET /api/voices/library').length, loads + 1);
  });

  it('[VL-15] a failed transcript save keeps the sheet open with the reason', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'PATCH /api/voices/library/:name': () => json(422, { error: { message: 'Reference transcript is too long. Max: 2000 characters' } }) } });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await ctx.menu('vo-rec-demo_reference-more', 'transcript');
    ctx.env.input(document.getElementById('vo-transcript-edit'), 'x'.repeat(2100));
    ctx.env.click(document.getElementById('vo-transcript-save'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(document.getElementById('vo-transcript-sheet').open, true);
    assert.match(text(document.getElementById('vo-transcript-error')), /too long/);
    assert.deepEqual(events, [], 'nothing changed: Speak is not told otherwise');
  });

  it('[VL-16] Delete names the presets that use the recording, then DELETEs it, tells the other views and moves focus on', async () => {
    let data = recordings();
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'GET /api/voices/library': () => json(200, data) } });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await ctx.menu('vo-rec-demo_reference-more', 'delete');
    const c = ctx.confirmText();
    assert.equal(c.title, 'Delete recording demo_reference?');
    assert.match(c.message, /^Presets using it: demo clone\./);
    data = data.filter((r) => r.name !== 'demo_reference');
    await ctx.answerConfirm(true);
    assert.equal(ctx.calls('DELETE /api/voices/library/:name')[0].path, '/api/voices/library/demo_reference');
    assert.equal(cardOf('demo_reference'), null);
    assert.ok(ctx.toasts().includes('Deleted recording demo_reference'));
    assert.deepEqual(events, ['recordings'], 'Speak hears of it and clears its reference when it was this recording');
    assert.equal(ctx.calls('GET /api/voices/library').length, 2, 'the list reloads once: the view ignores its own event');
    assert.equal(document.activeElement?.id, 'vo-rec-raw_take-title');
    await ctx.menu('vo-rec-raw_take-more', 'delete');
    assert.equal(ctx.confirmText().message, 'Its audio and transcript are removed.', 'no presets use it');
    await ctx.answerConfirm(false);
    assert.equal(ctx.calls('DELETE /api/voices/library/:name').length, 1, 'Cancel sends nothing');
    assert.deepEqual(events, ['recordings'], 'and tells no one');
  });

  it('[VL-16] a Delete the server refuses keeps the card, shows why on it and tells no other view', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'DELETE /api/voices/library/:name': () => json(500, { error: { message: 'disk is read-only' } }) } });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await ctx.menu('vo-rec-demo_reference-more', 'delete');
    await ctx.answerConfirm(true);
    assert.ok(cardOf('demo_reference'), 'the card stays');
    assert.match(text(part('demo_reference', 'error')), /disk is read-only/);
    assert.deepEqual(events, [], 'Speak keeps its reference: the recording still exists');
  });

  it('[VL-17] Save as preset asks for a name prefilled "<name> clone" and POSTs the clone preset payload', async () => {
    ctx = await voicesEnv({
      sub: 'recordings',
      routes: { 'POST /api/profiles': (req) => (req.json.name === 'Podcast EQ' ? json(409, { error: { message: 'Profile name already exists' } }) : json(201, { id: 'new', ...req.json })) },
    });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await ctx.menu('vo-rec-demo_reference-more', 'preset');
    assert.equal(ctx.promptState().value, 'demo_reference clone');
    await ctx.answerPrompt('Podcast EQ');
    assert.match(ctx.promptState().error, /already exists/, 'a 409 keeps the prompt open');
    await ctx.answerPrompt('My clone');
    const posts = ctx.calls('POST /api/profiles');
    assert.deepEqual(posts[1].json, {
      name: 'My clone', backend: 'chatterbox', model: 'chatterbox', voice: 'demo_reference', speed: 1, format: 'wav',
      blend: null, reference_audio_id: 'demo_reference', effects: [],
    });
    assert.ok(ctx.toasts().includes('Saved preset My clone'));
    assert.deepEqual(events, ['presets']);
  });

  it('[VL-18] Use in Speak sends use-recording with the chosen clone model and opens Speak', async () => {
    ctx = await voicesEnv({ sub: 'recordings' });
    ctx.env.click(part('demo_reference', 'use'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'speak');
    assert.deepEqual({ ...ctx.store.takeIntent('speak') }, { type: 'use-recording', name: 'demo_reference', model: 'chatterbox' });
    assert.equal(document.activeElement?.id, 'sp-title');
  });

  it('[VL-18] without a clone model Use in Speak is unavailable and explains why', async () => {
    ctx = await voicesEnv({ sub: 'recordings', models: MODELS_NO_CLONE });
    const use = part('demo_reference', 'use');
    assert.equal(use.getAttribute('aria-disabled'), 'true');
    assert.equal(use.getAttribute('aria-describedby'), 'vo-clone-model-hint');
    ctx.env.click(use);
    await ctx.settle();
    assert.equal(ctx.app.get().route.view, 'voices');
    assert.equal(ctx.store.takeIntent('speak'), null);
  });

  it('the ?shot=with-clone-models screenshot state shows stand-ins without writing them anywhere, and its actions do nothing', async () => {
    ctx = await voicesEnv({ sub: 'recordings', routes: { 'GET /api/voices/library': () => json(200, [recordings()[0]]) } });
    await ctx.go('voices/recordings?shot=with-clone-models');
    assert.ok(cardOf('sample_take'), 'a stand-in recording without a transcript shows');
    assert.deepEqual([...document.querySelectorAll('#vo-clone-model option')].map((o) => o.value), ['chatterbox', 'fish']);
    assert.deepEqual(ctx.controller.__internals.lists.recordings.items.map((r) => r.name), ['demo_reference'], 'never written to the list (the wizard\'s "Saves as" reads it)');
    ctx.env.click(part('demo_reference', 'test'));
    ctx.env.click(part('demo_reference', 'use'));
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/speech').length, 0, 'no request to a stand-in model');
    assert.equal(ctx.app.get().route.view, 'voices');
    assert.equal(ctx.store.takeIntent('speak'), null);
    await ctx.go('voices/recordings');
    assert.equal(cardOf('sample_take'), null, 'without the shot, the real state shows again');
    assert.deepEqual([...document.querySelectorAll('#vo-clone-model option')].map((o) => o.value), ['chatterbox']);
  });

  it('[VL-11] pure helpers: recordingMeta, compatibility and presetsUsing', async () => {
    const mod = await load('views/voices/recordings.js');
    assert.equal(mod.recordingMeta({ duration_s: 12.345, sample_rate: 22050, channels: 2 }), '12.3 s · 22.1 kHz · 2 channels');
    assert.equal(mod.recordingMeta({}), 'Length unknown');
    assert.deepEqual(mod.compatibility({ transcript: ' ' }, [{ id: 'chatterbox', capabilities: { clone_transcript_required: true } }, { id: 'qwen3', capabilities: {} }]), [
      { id: 'chatterbox', label: 'Chatterbox', ok: false, reason: 'needs a transcript' },
      { id: 'qwen3', label: 'Qwen3 TTS', ok: true, reason: '' },
    ]);
    assert.deepEqual(mod.presetsUsing('a', [{ name: 'x', reference_audio_id: 'a' }, { name: 'y', reference_audio_id: 'b' }]).map((p) => p.name), ['x']);
  });
});
