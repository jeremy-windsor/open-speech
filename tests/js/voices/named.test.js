// Voices: the named voices in Voices' one list, and the view's data loading (inventory SE-02…SE-04,
// SE-07). Named voices and presets share one list at #voices
// (#vo-voices-list); tests/js/voices/list.test.js covers what the one list adds.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames, accessibleName } from '../harness/a11y.js';
import { voicesEnv, namedVoices, MODELS, json, bytes, text as textBody, delay, load } from './helpers.js';

let ctx;
afterEach(() => ctx?.env.restore());

const card = (id) => document.getElementById(`vo-named-${id}`);
const text = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
/** A designed named voice: its version clones its hidden sample. */
const DESIGNED = Object.freeze({
  id: 'n7', name: 'Warm narrator', voice: 'voice:n7', preset_count: 0,
  realizations: [{ id: 'r7', model: 'breeze/tts-2', voice: 'designed_n7', reference_audio_id: 'designed_n7' }],
  design: { description: 'A warm, calm narrator with a slight rasp', model: 'breeze/tts-2', seed: 1234, options: {}, sample: 'designed_n7', transcript: 'Words of the sample.', duration_s: 4.2 },
});
const WITH_BREEZE = {
  ...MODELS,
  models: [...MODELS.models, { id: 'breeze/tts-2', type: 'tts', provider: 'breeze', state: 'loaded', device: 'external', provider_available: true, capabilities: { voice_design: true, voice_clone: true } }],
};
const withDesigned = (routes = {}) => ({
  models: WITH_BREEZE,
  routes: {
    'GET /api/voices/identities': () => json(200, { voices: [...namedVoices().voices, DESIGNED] }),
    'GET /api/voices/library/:name/audio': () => bytes(200, [82, 73, 70, 70], 'audio/wav'),
    ...routes,
  },
});

/** components.css makes every .card a size container; the fake browser's computed style reads inline style. */
const sizeContainer = (el) => el.style.setProperty('container-type', 'inline-size');

describe('views/voices named voices', () => {
  it('[SE-07] opening Voices loads named voices, presets and recordings once; later shows reload only stale lists', async () => {
    ctx = await voicesEnv();
    const named = ctx.calls('GET /api/voices/identities');
    assert.equal(named.length, 1, 'named voices load on the first show');
    assert.equal(named[0].query.get('check_availability'), 'false', 'the manager reads the catalogue without availability checks');
    assert.equal(ctx.calls('GET /api/profiles').length, 1, 'presets load with it');
    assert.equal(ctx.calls('GET /api/voices/library').length, 1, 'recordings load with it');
    await ctx.go('voices/recordings');
    await ctx.go('voices');
    assert.equal(ctx.calls('GET /api/voices/identities').length, 1, 'a fresh list is not fetched again');
    ctx.env.clock.tick(31000);
    await ctx.go('voices/recordings');
    assert.equal(ctx.calls('GET /api/voices/identities').length, 2, 'older than 30 s: reloaded on show');
    assert.equal(ctx.calls('GET /api/profiles').length, 2);
  });

  it('[SE-07] a voices:changed event while Voices is hidden reloads that list on the next show; while shown, at once', async () => {
    ctx = await voicesEnv();
    await ctx.go('speak');
    ctx.store.emit('voices:changed', { what: 'named' });
    await ctx.settle();
    assert.equal(ctx.calls('GET /api/voices/identities').length, 1, 'nothing loads while hidden');
    await ctx.go('voices');
    assert.equal(ctx.calls('GET /api/voices/identities').length, 2, 'the changed list reloads on show');
    assert.equal(ctx.calls('GET /api/profiles').length, 1, 'the others stay');
    ctx.store.emit('voices:changed', { what: 'presets' });
    await ctx.settle();
    assert.equal(ctx.calls('GET /api/profiles').length, 2, 'while shown, the list reloads at once');
  });

  it('[SE-02] each named voice is a card: versions with a model badge, the recipe in mono and a blend bar, the preset count, and actions', async () => {
    ctx = await voicesEnv();
    const narrator = card('n1');
    assert.ok(narrator, 'a card per named voice');
    assert.equal(narrator.querySelector('h2').textContent, 'Narrator');
    assert.equal(text(narrator.querySelector('.vo-card__meta')), '2 versions · 2 presets use it');
    const versions = [...narrator.querySelectorAll('.vo-version')];
    assert.equal(versions.length, 2);
    const kokoro = versions.find((v) => text(v.querySelector('.badge')) === 'Kokoro');
    assert.equal(text(kokoro.querySelector('.mono')), 'af_bella(2)+af_sky(1)');
    const bar = kokoro.querySelector('.blend-bar');
    assert.equal(bar.getAttribute('role'), 'img');
    assert.equal(bar.getAttribute('aria-label'), 'Bella 67 percent, Sky 33 percent');
    assert.equal(bar.querySelector('[data-w]').style.getPropertyValue('--w'), '67%', 'segment widths come from a custom property');
    assert.match(text(kokoro.querySelector('.vo-legend')), /Bella 67%.*Sky 33%/);
    const edit = kokoro.querySelector('[data-action="named-edit"]');
    const remove = kokoro.querySelector('[data-action="named-remove"]');
    assert.equal(accessibleName(edit), 'Edit in Speak: Narrator for Kokoro', 'the name starts with the visible label');
    assert.equal(accessibleName(remove), 'Remove version: Narrator for Kokoro');
    const more = document.getElementById('vo-named-n1-more');
    assert.equal(more.getAttribute('aria-haspopup'), 'menu');
    const menu = document.getElementById(more.getAttribute('aria-controls'));
    assert.deepEqual([...menu.querySelectorAll('[role="menuitem"]')].map((i) => text(i)), ['Rename…', 'Delete…']);
    assert.ok(narrator.contains(menu) && menu.previousElementSibling === more, 'the menu is written in its card, right after its button');
    assert.equal(text(card('n2').querySelector('.vo-card__meta')), '1 version · No presets use it');
    assertAccessibleNames(ctx.section);
  });

  it('[SE-02] with no named voices (and no presets: one list) the list says so and offers Open Speak', async () => {
    ctx = await voicesEnv({
      routes: {
        'GET /api/voices/identities': () => json(200, { voices: [] }),
        'GET /api/profiles': () => json(200, { profiles: [], default_profile_id: null }),
      },
    });
    const empty = ctx.$('#vo-voices-list .empty');
    assert.equal(text(empty.querySelector('.empty__title')), 'No named voices or presets yet');
    assert.equal(text(empty.querySelector('.empty__text')), 'In Speak, choose a voice, then Save named voice… or Save preset….');
    ctx.env.click(empty.querySelector('[data-action="open-speak"]'));
    await ctx.settle();
    assert.equal(ctx.app.get().route.view, 'speak');
  });

  it('[SE-02] a failed load shows the error state with Retry; an auth refusal says API key needed with Open Settings', async () => {
    let fail = true;
    ctx = await voicesEnv({
      routes: { 'GET /api/voices/identities': () => (fail ? json(500, { error: { message: 'database is locked', code: 'db_error' } }) : json(200, namedVoices())) },
    });
    const state = ctx.$('#vo-voices-list .empty--error');
    assert.ok(state.closest('.card'), 'the error state sits in a card, like the empty state');
    assert.equal(text(state.querySelector('.empty__title')), "Couldn't load named voices");
    assert.equal(text(state.querySelector('.empty__text')), 'database is locked', 'the plain server message');
    assert.equal(state.querySelector('details pre').textContent, 'database is locked (db_error)', 'Details read as the server message and code, never the JSON envelope');
    fail = false;
    ctx.env.click(state.querySelector('[data-action="retry-list"]'));
    await ctx.settle();
    assert.ok(card('n1'), 'Retry loads the list');
    ctx.env.restore();
    ctx = await voicesEnv({ routes: { 'GET /api/voices/identities': () => json(401, { error: { message: 'Invalid API key' } }) } });
    const auth = ctx.$('#vo-voices-list .empty--error');
    assert.equal(text(auth.querySelector('.empty__title')), 'API key needed');
    assert.ok(auth.querySelector('[data-action="open-settings"]'), 'Open Settings, not Retry');
  });

  it('[SE-02][SE-04] Details show the server\'s own words: a proxy\'s page as it came, and none when the server\'s generic code would only repeat the message', async () => {
    ctx = await voicesEnv({
      routes: {
        'GET /api/voices/identities': () => textBody(502, '<html><body>Bad gateway</body></html>', { 'content-type': 'text/html' }),
      },
    });
    const state = ctx.$('#vo-voices-list .empty--error');
    assert.equal(text(state.querySelector('.empty__text')), 'The server returned an unexpected response (HTTP 502).');
    assert.equal(state.querySelector('details pre').textContent, '<html><body>Bad gateway</body></html>', 'a page that is not the JSON envelope is shown as it came, as text');
    ctx.env.restore();
    ctx = await voicesEnv({
      routes: { 'DELETE /api/voices/identities/:id/realizations/*': () => json(404, { error: { message: 'Named voice not found', code: 'http_error' } }) },
    });
    ctx.env.click(card('n1').querySelector('[data-action="named-remove"][data-model="kokoro"]'));
    await ctx.settle();
    await ctx.answerConfirm(true);
    const banner = document.getElementById('vo-named-n1-error').querySelector('.banner');
    assert.equal(text(banner.querySelector('.banner__msg')), 'Named voice not found');
    assert.equal(banner.querySelector('details'), null, '"Named voice not found (http_error)" would add nothing to the message');
  });

  it('[SE-03] Edit in Speak sends the edit-named-voice intent and opens Speak with focus on its heading', async () => {
    ctx = await voicesEnv();
    const edit = card('n1').querySelector('[data-action="named-edit"][data-model="kokoro"]');
    ctx.env.click(edit);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'speak');
    assert.deepEqual({ ...ctx.store.takeIntent('speak') }, { type: 'edit-named-voice', id: 'n1', model: 'kokoro' });
    assert.equal(document.activeElement?.id, 'sp-title', 'focus follows to the Speak heading (inventory bug 18)');
  });

  it('[SE-03] a version whose model is unavailable says "This model is unavailable" and does not open Speak', async () => {
    ctx = await voicesEnv();
    const qwen = card('n1').querySelector('[data-action="named-edit"][data-model="qwen3"]');
    const warn = document.getElementById(qwen.getAttribute('aria-describedby'));
    assert.equal(text(warn), 'This model is unavailable');
    assert.equal(qwen.getAttribute('aria-disabled'), 'true');
    ctx.env.click(qwen);
    await ctx.settle();
    assert.equal(ctx.app.get().route.view, 'voices');
    assert.equal(ctx.store.takeIntent('speak'), null, 'no intent is left for Speak');
  });

  it('[SE-04] Remove version confirms with the linked-presets copy, DELETEs the version and refreshes named voices and presets', async () => {
    let data = namedVoices();
    ctx = await voicesEnv({ routes: { 'GET /api/voices/identities': () => json(200, data) } });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    const remove = card('n1').querySelector('[data-action="named-remove"][data-model="kokoro"]');
    ctx.env.click(remove);
    await ctx.settle();
    assert.deepEqual(ctx.confirmText(), {
      title: 'Remove Narrator for Kokoro?',
      message: 'Linked presets keep their current voice setup. Recordings are kept.',
      open: true,
    });
    assert.equal(document.activeElement?.id, 'app-confirm-cancel', 'a danger confirm focuses Cancel');
    await ctx.answerConfirm(false);
    assert.equal(ctx.calls('DELETE /api/voices/identities/:id/realizations/*').length, 0, 'Cancel sends nothing');
    ctx.env.click(remove);
    await ctx.settle();
    data = { voices: [{ ...data.voices[0], realizations: [data.voices[0].realizations[1]] }, data.voices[1]] };
    await ctx.answerConfirm(true);
    const del = ctx.calls('DELETE /api/voices/identities/:id/realizations/*');
    assert.equal(del.length, 1);
    assert.equal(del[0].path, '/api/voices/identities/n1/realizations/kokoro');
    assert.ok(ctx.toasts().includes('Removed Narrator for Kokoro'));
    assert.equal(ctx.calls('GET /api/voices/identities').length, 2, 'named voices reload once: the view ignores its own event');
    assert.equal(ctx.calls('GET /api/profiles').length, 2, 'presets reload: a linked preset keeps its setup but loses the link');
    assert.equal(card('n1').querySelectorAll('.vo-version').length, 1);
    assert.deepEqual(events, ['named', 'presets'], 'other views hear about the change (Speak refreshes its named voices)');
  });

  it('[SE-04] the action button is disabled while its request runs, and a failure shows on the card', async () => {
    ctx = await voicesEnv({ routes: { 'DELETE /api/voices/identities/:id/realizations/*': () => delay(500, json(409, { error: { message: 'Named voice is in use' } })) } });
    const remove = card('n1').querySelector('[data-action="named-remove"][data-model="kokoro"]');
    ctx.env.click(remove);
    await ctx.settle();
    await ctx.answerConfirm(true);
    assert.equal(remove.disabled, true, 'disabled while the DELETE runs');
    assert.equal(remove.getAttribute('aria-busy'), 'true');
    await ctx.settle(600);
    assert.equal(remove.disabled, false);
    const banner = document.getElementById('vo-named-n1-error').querySelector('.banner');
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(text(banner.querySelector('.banner__title')), "Couldn't delete");
    assert.equal(text(banner.querySelector('.banner__msg')), 'Named voice is in use');
  });

  it('[SE-04] an action the server refuses for the API key says "API key needed" on the card, with Open Settings and no Details', async () => {
    ctx = await voicesEnv({ routes: { 'DELETE /api/voices/identities/:id/realizations/*': () => json(401, { error: { message: 'Invalid API key', code: 'unauthorized' } }) } });
    const remove = card('n1').querySelector('[data-action="named-remove"][data-model="kokoro"]');
    ctx.env.click(remove);
    await ctx.settle();
    await ctx.answerConfirm(true);
    const banner = document.getElementById('vo-named-n1-error').querySelector('.banner');
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(text(banner.querySelector('.banner__title')), 'API key needed', 'the cause, not "Couldn\'t delete"');
    assert.match(text(banner.querySelector('.banner__msg')), /Settings › Connection/);
    assert.equal(banner.querySelector('details'), null, 'the refusal needs no Details');
    const open = banner.querySelector('[data-action="open-settings"]');
    assert.equal(text(open), 'Open Settings');
    assertAccessibleNames(banner);
    ctx.env.click(open);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.view, 'settings');
  });

  it('[SE-04] Rename is skipped when unchanged, PATCHes {name}, and a 409 keeps the prompt open with a field error', async () => {
    let conflict = true;
    ctx = await voicesEnv({
      routes: {
        'PATCH /api/voices/identities/:id': (req) => (conflict && req.json.name === 'Alex'
          ? json(409, { error: { message: 'Voice name already exists' } })
          : json(200, { id: 'n1', name: req.json.name, realizations: [] })),
      },
    });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await ctx.menu('vo-named-n1-more', 'rename');
    assert.equal(ctx.promptState().value, 'Narrator', 'prefilled with the current name');
    await ctx.answerPrompt('Narrator');
    assert.equal(ctx.calls('PATCH /api/voices/identities/:id').length, 0, 'unchanged: nothing is sent');
    await ctx.menu('vo-named-n1-more', 'rename');
    await ctx.answerPrompt('Alex');
    const p = ctx.promptState();
    assert.equal(p.open, true, 'a 409 re-opens the prompt');
    assert.equal(p.value, 'Alex');
    assert.equal(p.invalid, true);
    assert.match(p.error, /already exists/);
    await ctx.answerPrompt('Storyteller');
    const calls = ctx.calls('PATCH /api/voices/identities/:id');
    assert.deepEqual(calls.map((c) => c.json), [{ name: 'Alex' }, { name: 'Storyteller' }]);
    assert.equal(calls[1].path, '/api/voices/identities/n1');
    assert.ok(ctx.toasts().includes('Renamed Narrator to Storyteller'));
    assert.equal(ctx.calls('GET /api/voices/identities').length, 2, 'the list reloads after a rename');
    assert.deepEqual(events, ['named'], "once, for the rename that went through: Speak's named voices show the new name");
  });

  it('[SE-04] Delete confirms, DELETEs the named voice, refreshes named voices and presets everywhere and moves focus to the next card', async () => {
    let data = namedVoices();
    ctx = await voicesEnv({ routes: { 'GET /api/voices/identities': () => json(200, data) } });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await ctx.menu('vo-named-n1-more', 'delete');
    const c = ctx.confirmText();
    assert.equal(c.title, 'Delete Narrator?');
    assert.match(c.message, /Linked presets keep their current voice setup\. Recordings are kept\./);
    await ctx.answerConfirm(false);
    assert.equal(ctx.calls('DELETE /api/voices/identities/:id').length, 0, 'Cancel sends nothing');
    assert.deepEqual(events, [], 'and tells no one');
    await ctx.menu('vo-named-n1-more', 'delete');
    data = { voices: [data.voices[1]] };
    await ctx.answerConfirm(true);
    assert.equal(ctx.calls('DELETE /api/voices/identities/:id')[0].path, '/api/voices/identities/n1');
    assert.equal(card('n1'), null);
    assert.ok(ctx.toasts().includes('Deleted Narrator'));
    // Speak drops a selected named voice that no longer exists (else Generate sends voice:n1), and its preset
    // chips stop showing the link.
    assert.deepEqual(events, ['named', 'presets'], 'other views hear about the change (Speak refreshes its named voices and presets)');
    assert.equal(ctx.calls('GET /api/voices/identities').length, 2, 'named voices reload once: the view ignores its own event');
    assert.equal(ctx.calls('GET /api/profiles').length, 2, 'presets reload: a linked preset keeps its setup but loses the link');
    assert.equal(document.activeElement?.id, 'vo-named-n2-title', 'focus moves to the next card, never to <body>');
  });

  it('[SE-02] a designed voice\'s card shows "Designed" with its design model, its description, Play sample, Edit description and Use in Speak; its version clones the sample', async () => {
    ctx = await voicesEnv(withDesigned());
    const designed = card('n7');
    assert.ok(designed);
    assert.equal(text(designed.querySelector('.vo-design__head .badge')), 'Designed');
    assert.equal(text(designed.querySelector('.vo-design__model')), 'Breeze TTS 2');
    assert.equal(text(designed.querySelector('.vo-design__desc')), 'A warm, calm narrator with a slight rasp');
    assert.deepEqual([...designed.querySelectorAll('.vo-design__actions button')].map(accessibleName),
      ['Play sample: Warm narrator', 'Edit description: Warm narrator', 'Use in Speak: Warm narrator']);
    const version = designed.querySelector('.vo-version');
    assert.equal(text(version.querySelector('.vo-version__desc')), 'Clones its designed sample');
    assert.equal(version.querySelector('.vo-version__spec'), null, 'the sample\'s generated name is not shown');
    assert.equal(card('n1').querySelector('.vo-design'), null, 'a voice that was not designed has none of it');
    assertAccessibleNames(ctx.section);
  });

  it('[SE-02] the search matches a designed voice\'s description', async () => {
    const { namedMatches } = await load('views/voices/named.js');
    assert.equal(namedMatches(DESIGNED, 'rasp'), true);
    assert.equal(namedMatches(namedVoices().voices[0], 'rasp'), false);
  });

  it('[SE-03] Use in Speak hands a designed voice to Speak for its first available model (use-named-voice)', async () => {
    ctx = await voicesEnv(withDesigned());
    ctx.env.click(document.getElementById('vo-named-n7-use'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'speak');
    assert.deepEqual({ ...ctx.store.takeIntent('speak') }, { type: 'use-named-voice', id: 'n7', model: 'breeze/tts-2' });
  });

  it('[SE-03] Use in Speak is unavailable, and says why, when no model of the designed voice is available', async () => {
    ctx = await voicesEnv({ ...withDesigned(), models: MODELS });
    const use = document.getElementById('vo-named-n7-use');
    assert.equal(use.getAttribute('aria-disabled'), 'true');
    assert.equal(text(document.getElementById(use.getAttribute('aria-describedby'))), 'None of its models is available.');
    ctx.env.click(use);
    await ctx.settle();
    assert.equal(ctx.app.get().route.view, 'voices');
    assert.equal(ctx.store.takeIntent('speak'), null);
  });

  it('[SE-02] Play sample plays the designed sample as an ephemeral clip; Stop ends it; leaving the list stops it', async () => {
    ctx = await voicesEnv(withDesigned());
    const audio = await load('core/audio.js');
    const button = () => document.getElementById('vo-named-n7-sample');
    ctx.env.click(button());
    await ctx.settle();
    await ctx.settle();
    assert.match(audio.element().src, /\/api\/voices\/library\/designed_n7\/audio$/);
    assert.equal(audio.current().id, 'designed-sample');
    assert.equal(audio.current().ephemeral, true);
    assert.equal(text(button()), 'Stop');
    assert.equal(accessibleName(button()), 'Stop sample: Warm narrator');
    ctx.env.click(button());
    await ctx.settle();
    assert.equal(audio.current(), null);
    assert.equal(text(button()), 'Play sample');
    ctx.env.click(button());
    await ctx.settle();
    await ctx.settle();
    assert.equal(audio.current().id, 'designed-sample');
    await ctx.go('voices/recordings');
    assert.equal(audio.current(), null, 'no mini-player: it stops when the list is left');
  });

  it('[SE-04] Delete on a designed voice says its sample is deleted too', async () => {
    ctx = await voicesEnv(withDesigned());
    await ctx.menu('vo-named-n7-more', 'delete');
    const c = ctx.confirmText();
    assert.equal(c.title, 'Delete Warm narrator?');
    assert.equal(c.message, 'Every version of Warm narrator is removed. The sample it was designed from is deleted with it. Linked presets keep their current voice setup.');
    await ctx.answerConfirm(true);
    assert.equal(ctx.calls('DELETE /api/voices/identities/:id')[0].path, '/api/voices/identities/n7');
  });

  it('[SE-02] "/" focuses the search field, which filters the cards; no match offers Clear search', async () => {
    ctx = await voicesEnv();
    ctx.env.key('/', {}, document.body);
    const search = document.getElementById('vo-search');
    assert.equal(document.activeElement?.id, 'vo-search');
    ctx.env.input(search, 'alex');
    await ctx.settle(200);
    assert.equal(card('n1'), null);
    assert.ok(card('n2'));
    ctx.env.input(search, 'nobody');
    await ctx.settle(200);
    assert.match(text(ctx.$('#vo-voices-list .empty__title')), /No named voices or presets match “nobody”/);
    await ctx.go('voices/recordings');
    assert.match(text(ctx.$('#vo-rec-list .empty__title')), /No recordings match “nobody”/, 'the search applies to the section shown next');
    await ctx.go('voices');
    ctx.env.click(ctx.$('[data-action="clear-search"]'));
    await ctx.settle();
    assert.equal(search.value, '');
    assert.ok(card('n1') && card('n2'));
    assert.equal(document.activeElement?.id, 'vo-search');
  });

  it('[SE-02] a card menu is written in its card; open, it sits at the end of the view (a .card is a size container), at its button; Tab goes on from the button, Shift+Tab stops on it, and it goes back into the card', async () => {
    ctx = await voicesEnv();
    const narrator = card('n1');
    const more = document.getElementById('vo-named-n1-more');
    const menu = document.getElementById(more.getAttribute('aria-controls'));
    const inCard = () => narrator.contains(menu) && menu.previousElementSibling === more;
    assert.ok(inCard(), 'closed: in the card, right after its button');
    sizeContainer(narrator);
    // The harness viewport is a 390 px phone: the button sits at the card's top right, inside it.
    more.__rect = { left: 300, top: 300, width: 44, height: 44 };
    menu.__rect = { width: 180, height: 96 };
    more.focus();
    ctx.env.key('Enter', {}, more);
    await ctx.settle();
    assert.equal(menu.hidden, false);
    assert.equal(menu.parentElement, ctx.section, 'open: at the end of the Voices section, so its fixed box is placed against the viewport, not the card');
    assert.equal(ctx.section.lastElementChild, menu);
    assert.equal(menu.style.getPropertyValue('--menu-x'), `${344 - 180}px`, 'bottom-end: its end edge on the button\'s');
    assert.equal(menu.style.getPropertyValue('--menu-y'), `${344 + 4}px`, 'just under the button');
    assert.equal(text(document.activeElement), 'Rename…');
    const tab = ctx.env.key('Tab');
    assert.equal(tab.defaultPrevented, false, 'the browser\'s own Tab still runs…');
    assert.equal(document.activeElement, more, '…from the menu\'s button, so the next stop is the card\'s Edit in Speak, not the end of the view or <body>');
    assert.equal(menu.hidden, true);
    assert.equal(more.getAttribute('aria-expanded'), 'false');
    assert.ok(inCard(), 'closed: back in the card');
    ctx.env.key('Enter', {}, more);
    await ctx.settle();
    ctx.env.key('ArrowDown');
    assert.equal(text(document.activeElement), 'Delete…');
    const back = ctx.env.key('Tab', { shiftKey: true });
    assert.equal(back.defaultPrevented, true);
    assert.equal(document.activeElement, more, 'Shift+Tab stops on the button: the stop before the menu');
    assert.ok(menu.hidden && inCard());
    // An item chosen while the menu sits outside the card runs its action for that card.
    ctx.env.key('Enter', {}, more);
    await ctx.settle();
    assert.equal(menu.parentElement, ctx.section);
    ctx.env.key('ArrowDown');
    ctx.env.key('Enter');
    await ctx.settle();
    assert.equal(ctx.confirmText().title, 'Delete Narrator?');
    assert.ok(menu.hidden && inCard(), 'choosing an item closes the menu back into its card');
    await ctx.answerConfirm(false);
    assert.equal(document.activeElement, more, 'Cancel puts focus back on the button, never <body>');
    // The same for a preset's card (one without a named voice) and the recording cards.
    for (const [path, cardSel, id] of [['voices', '#vo-preset-p2', 'vo-preset-p2-more'], ['voices/recordings', '#vo-rec-demo_reference', 'vo-rec-demo_reference-more']]) {
      await ctx.go(path);
      sizeContainer(ctx.$(cardSel));
      const button = document.getElementById(id);
      const own = document.getElementById(button.getAttribute('aria-controls'));
      assert.equal(own.previousElementSibling, button, `${id}: written right after its button`);
      button.focus();
      ctx.env.key('Enter', {}, button);
      await ctx.settle();
      assert.equal(own.parentElement, ctx.section, `${id}: open at the end of the view`);
      ctx.env.key('Tab');
      assert.equal(document.activeElement, button, `${id}: Tab goes on from the button`);
      assert.ok(own.hidden && own.previousElementSibling === button, `${id}: closed, back after its button`);
    }
  });

  it('[SE-07] a reload under an open card menu (another view changed the list) opens it again on the same item, with no stale copy left in the view', async () => {
    ctx = await voicesEnv();
    sizeContainer(card('n1'));
    const more = document.getElementById('vo-named-n1-more');
    more.focus();
    ctx.env.key('Enter', {}, more);
    await ctx.settle();
    ctx.env.key('ArrowDown');
    const old = document.getElementById('vo-named-n1-menu');
    assert.equal(old.parentElement, ctx.section, 'open outside its card');
    ctx.store.emit('voices:changed', { what: 'named' });
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('GET /api/voices/identities').length, 2, 'the list reloaded and the cards were rebuilt');
    const copies = [...document.querySelectorAll('.menu')].filter((m) => m.id === 'vo-named-n1-menu');
    assert.equal(copies.length, 1, 'one menu with that id: the closed copy went with its old card');
    const [menu] = copies;
    assert.notEqual(menu, old);
    assert.equal(menu.hidden, false, 'the menu that had focus is open again');
    assert.ok(menu.contains(document.activeElement));
    assert.equal(text(document.activeElement), 'Delete…', 'on the same item');
    assert.equal(document.getElementById('vo-named-n1-more').getAttribute('aria-expanded'), 'true');
  });

  it('[SE-02] the search says how many match once typing pauses (WCAG 4.1.3), never once per key', async () => {
    ctx = await voicesEnv();
    const search = document.getElementById('vo-search');
    const said = () => document.getElementById('app-sr-status').textContent;
    const before = said();
    search.focus();
    ctx.env.input(search, 'al');
    await ctx.settle(200);
    ctx.env.input(search, 'ale');
    await ctx.settle(200);
    assert.equal(said(), before, 'nothing is said while typing');
    assert.equal(ctx.$$('#vo-voices-list .vo-card--voice').length, 1, 'the list itself filters as you type');
    await ctx.settle(800);
    assert.equal(said(), '1 named voice matches “ale”');
    ctx.env.input(search, 'nobody');
    await ctx.settle(1000);
    assert.equal(said(), 'No named voices or presets match “nobody”');
    ctx.env.click(ctx.$('[data-action="clear-search"]'));
    await ctx.settle(1000);
    assert.equal(said(), '2 named voices and 4 presets', 'clearing says what is back (one list)');
    ctx.env.input(search, 'podcast');
    await ctx.settle(1000);
    assert.equal(said(), '1 preset matches “podcast”');
    const { searchSummary } = await load('views/voices/shared.js');
    assert.equal(searchSummary('recordings', 3, ' '), '3 recordings');
    assert.equal(searchSummary('recordings', 1, 'demo'), '1 recording matches “demo”');
  });

  it('[SE-02] the sections are a segmented control that navigates with focus kept', async () => {
    ctx = await voicesEnv();
    const recordingsRadio = ctx.$('#vo-sub input[value="recordings"]');
    ctx.env.change(recordingsRadio, true);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices/recordings');
    assert.equal(ctx.$('#vo-recs').hidden, false);
    assert.equal(ctx.$('#vo-voices').hidden, true, 'Voices (named voices and presets, one list) is hidden');
    assert.equal(ctx.$('#vo-sub').getAttribute('role'), 'radiogroup');
    await ctx.go('voices/clone');
    assert.equal(ctx.$('#vo-bar').hidden, true, 'the wizard hides the sections and search');
    assert.equal(ctx.$('#vo-clone').hidden, true);
    assert.equal(ctx.$('#vo-wizard').hidden, false);
    assert.equal(ctx.app.get().prefs.lastRoute, 'voices/recordings', 'the wizard is never the last route');
  });
});
