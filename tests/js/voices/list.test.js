// Voices: one list of named voices and presets.
// A named voice is who speaks (one version per model), a preset is how it reads (speed, format, effects,
// instructions): each named voice is a card with its versions and, under them, its presets; presets that use a
// plain voice, a blend or a recording are cards of their own in a group at the end.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames, accessibleName } from '../harness/a11y.js';
import { voicesEnv, namedVoices, presets, json, load } from './helpers.js';

let ctx;
afterEach(() => ctx?.env.restore());

const $id = (id) => document.getElementById(id);
const text = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
const presetIds = (root) => [...root.querySelectorAll('.vo-preset')].map((el) => el.id);

describe('views/voices one list', () => {
  it('one Voices list: Voices | Recordings; each named voice is a card with its versions and its presets; presets without a named voice are cards in a group at the end; the default is marked where it sits', async () => {
    ctx = await voicesEnv({ sub: '' });
    const radios = ctx.$$('#vo-sub input[name="vo-sub"]');
    assert.deepEqual(radios.map((r) => r.value), ['voices', 'recordings'], 'Named voices and Presets are one section');
    assert.deepEqual(radios.map((r) => accessibleName(r)), ['Voices', 'Recordings']);
    assert.equal(ctx.$('#vo-sub input[value="voices"]').checked, true);
    assert.equal(ctx.$('#vo-voices').hidden, false);
    const list = ctx.$('#vo-voices-list');
    // Narrator: its versions, then its preset; Alex has none yet.
    const narrator = $id('vo-named-n1');
    assert.equal(narrator.querySelector('h2').textContent, 'Narrator');
    assert.equal(narrator.querySelectorAll('.vo-version').length, 2);
    assert.deepEqual(presetIds(narrator), ['vo-preset-p3'], 'a preset linked to the named voice sits on its card');
    const p3 = $id('vo-preset-p3');
    assert.equal(p3.querySelector('.vo-preset__name').tagName.toLowerCase(), 'h3', 'a preset is a level under its voice');
    assert.equal(text(p3.querySelector('.vo-preset__meta')), 'Kokoro · 1.25× · WAV · Instructions', 'the card names the voice: the summary says which version (it has two) and how it reads');
    const alex = $id('vo-named-n2');
    assert.deepEqual(presetIds(alex), []);
    assert.equal(text(alex.querySelector('.vo-card__nopresets')), 'No presets use it yet.');
    for (const card of [narrator, alex]) {
      const id = card.id.replace('vo-named-', '');
      const name = card.querySelector('h2').textContent;
      assert.equal(accessibleName($id(`vo-named-${id}-new`)), `New preset with ${name}`);
      assert.equal(accessibleName($id(`vo-named-${id}-rename`)), `Rename: ${name}`, 'Rename is a visible button on the card');
    }
    // The presets that use a plain voice, a blend or a recording: a group of cards at the end.
    const group = ctx.$('.vo-group');
    assert.ok(group && list.lastElementChild === group, 'the group comes last');
    assert.equal(text(group.querySelector('h2')), 'Presets without a named voice');
    assert.equal(group.getAttribute('aria-labelledby'), group.querySelector('h2').id);
    assert.deepEqual(presetIds(group), ['vo-preset-p1', 'vo-preset-p2', 'vo-preset-p4'], 'the default first, then by name');
    for (const id of ['p1', 'p2', 'p4']) assert.ok($id(`vo-preset-${id}`).classList.contains('card'), `${id}: a card of its own`);
    assert.equal(text($id('vo-preset-p1').querySelector('.vo-preset__meta')), 'Kokoro · Heart · 1.1× · MP3', 'with no named voice, the summary names the voice');
    assert.deepEqual(presetIds(list), ['vo-preset-p3', 'vo-preset-p1', 'vo-preset-p2', 'vo-preset-p4']);
    assert.deepEqual([...list.querySelectorAll('.vo-preset__badge')].map((b) => b.closest('.vo-preset').id), ['vo-preset-p1'], 'one "Default" badge, wherever the default sits');
    // Every preset: Use in Speak, Edit, Default in Speak, Rename (visible) and the menu (Rename…, Delete…).
    for (const id of ['p3', 'p1']) {
      const name = text($id(`vo-preset-${id}-title`));
      assert.deepEqual(
        [...$id(`vo-preset-${id}`).querySelectorAll('.vo-preset__actions > button')].map((b) => accessibleName(b)),
        [`Use in Speak: ${name}`, `Edit: ${name}`, `Default in Speak: ${name}`, `Rename: ${name}`, `More actions for ${name}`],
      );
      const menu = $id($id(`vo-preset-${id}-more`).getAttribute('aria-controls'));
      assert.deepEqual([...menu.querySelectorAll('[role="menuitem"]')].map(text), ['Rename…', 'Delete…']);
    }
    assertAccessibleNames(ctx.section);
  });

  it('the intro says what each is: a named voice is who speaks, a preset how it reads', async () => {
    ctx = await voicesEnv({ sub: '' });
    const intro = text(ctx.$('#vo-voices .vo-intro'));
    assert.match(intro, /named voice is who speaks/);
    assert.match(intro, /preset is how it reads/);
  });

  it('old links to #voices/named and #voices/presets open the one list at #voices', async () => {
    for (const old of ['named', 'presets']) {
      ctx = await voicesEnv({ sub: old });
      assert.equal(ctx.app.get().route.path, 'voices', `#voices/${old}`);
      assert.equal(location.hash, '#voices', 'the address is replaced, not kept');
      assert.equal(ctx.$('#vo-voices').hidden, false);
      assert.ok($id('vo-named-n1') && $id('vo-preset-p1'), 'named voices and presets together');
      ctx.env.restore();
      ctx = null;
    }
    ctx = await voicesEnv({ sub: 'recordings' });
    ctx.env.change(ctx.$('#vo-sub input[value="voices"]'), true);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices', 'the section control goes to #voices');
    assert.equal(ctx.$('#vo-recs').hidden, true);
  });

  it('the list\'s states: empty (no named voices and no presets), only presets (the group alone), one failed list with Retry, and both failed at once', async () => {
    const none = { 'GET /api/voices/identities': () => json(200, { voices: [] }), 'GET /api/profiles': () => json(200, { profiles: [], default_profile_id: null }) };
    ctx = await voicesEnv({ sub: '', routes: none });
    const empty = ctx.$('#vo-voices-list .empty');
    assert.equal(text(empty.querySelector('.empty__title')), 'No named voices or presets yet');
    assert.match(text(empty.querySelector('.empty__text')), /Save named voice… or Save preset…/);
    assert.ok(empty.querySelector('[data-action="open-speak"].btn--primary'));
    ctx.env.restore();
    ctx = await voicesEnv({ sub: '', routes: { 'GET /api/voices/identities': () => json(200, { voices: [] }) } });
    assert.equal(ctx.$('#vo-voices-list .empty'), null, 'presets alone are a list, not an empty state');
    assert.deepEqual(presetIds(ctx.$('#vo-voices-list')), ['vo-preset-p1', 'vo-preset-p2', 'vo-preset-p3', 'vo-preset-p4'], 'a link to a voice that is gone is no card to sit on');
    ctx.env.restore();
    let fail = true;
    ctx = await voicesEnv({
      sub: '',
      routes: { 'GET /api/voices/identities': () => (fail ? json(500, { error: { message: 'database is locked', code: 'db_error' } }) : json(200, namedVoices())) },
    });
    const state = ctx.$('#vo-voices-list .empty--error');
    assert.equal(text(state.querySelector('.empty__title')), "Couldn't load named voices");
    assert.equal(text(ctx.$('.vo-group h2')), 'Presets', 'the presets still show; which voice each belongs to is unknown, so the group is just "Presets"');
    fail = false;
    ctx.env.click(state.querySelector('[data-action="retry-list"]'));
    await ctx.settle();
    assert.deepEqual(presetIds($id('vo-named-n1')), ['vo-preset-p3'], 'Retry loads the named voices, and the linked preset moves onto its card');
    assert.equal(text(ctx.$('.vo-group h2')), 'Presets without a named voice');
    ctx.env.restore();
    ctx = await voicesEnv({ sub: '', routes: { 'GET /api/voices/identities': () => json(500, { error: { message: 'down' } }), 'GET /api/profiles': () => json(500, { error: { message: 'down' } }) } });
    const both = ctx.$$('#vo-voices-list .empty--error');
    assert.equal(both.length, 1, 'both lists failed: one error state, not two');
    assert.equal(text(both[0].querySelector('.empty__title')), "Couldn't load voices and presets");
    assert.equal(both[0].querySelector('[data-action="retry-list"]').getAttribute('data-list'), 'voices', 'its Retry reloads both');
  });

  it('Rename is a visible button: the voice\'s and each preset\'s, the same prompt as the menu\'s Rename…', async () => {
    ctx = await voicesEnv({
      sub: '',
      routes: {
        'PATCH /api/voices/identities/:id': (req) => json(200, { id: req.params.id, name: req.json.name, realizations: [] }),
        'PUT /api/profiles/:id': (req) => json(200, { ...req.json, id: req.params.id }),
      },
    });
    ctx.env.click($id('vo-named-n1-rename'));
    await ctx.settle();
    assert.equal(ctx.promptState().title, 'Rename named voice');
    assert.equal(ctx.promptState().value, 'Narrator');
    await ctx.answerPrompt('Storyteller');
    assert.deepEqual(ctx.calls('PATCH /api/voices/identities/:id').map((c) => [c.path, c.json]), [['/api/voices/identities/n1', { name: 'Storyteller' }]]);
    assert.ok(ctx.toasts().includes('Renamed Narrator to Storyteller'));
    ctx.env.click($id('vo-preset-p3-rename'));
    await ctx.settle();
    assert.equal(ctx.promptState().title, 'Rename preset');
    assert.equal(ctx.promptState().value, 'Narrator preset');
    await ctx.answerPrompt('Calm narrator');
    const put = ctx.calls('PUT /api/profiles/:id');
    assert.equal(put.length, 1);
    assert.equal(put[0].path, '/api/profiles/p3');
    assert.equal(put[0].json.name, 'Calm narrator');
    assert.equal(put[0].json.voice_identity_id, 'n1', 'the full payload: the preset stays on its voice');
  });

  it('Edit on a preset opens it in Speak for editing (edit-preset); New preset on a voice\'s card starts one with that voice (new-preset)', async () => {
    ctx = await voicesEnv({ sub: '' });
    ctx.env.click($id('vo-preset-p2-edit'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'speak');
    assert.deepEqual({ ...ctx.store.takeIntent('speak') }, { type: 'edit-preset', id: 'p2' });
    await ctx.go('voices');
    ctx.env.click($id('vo-named-n1-new'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'speak');
    assert.deepEqual({ ...ctx.store.takeIntent('speak') }, { type: 'new-preset', id: 'n1' });
  });

  it('New preset is unavailable, and says why, on a voice with no version for a model that is available', async () => {
    const data = namedVoices();
    data.voices.push({ id: 'n3', name: 'Draft', voice: 'voice:n3', preset_count: 0, realizations: [] });
    data.voices.push({ id: 'n4', name: 'Studio', voice: 'voice:n4', preset_count: 0, realizations: [{ id: 'r9', model: 'qwen3', voice: 'Ryan', reference_audio_id: null }] });
    ctx = await voicesEnv({ sub: '', routes: { 'GET /api/voices/identities': () => json(200, data) } });
    for (const [id, reason] of [['n3', 'No versions. Save this voice from Speak with a model to add one.'], ['n4', 'None of its models is available.']]) {
      const button = $id(`vo-named-${id}-new`);
      assert.equal(button.getAttribute('aria-disabled'), 'true', id);
      assert.equal(text($id(button.getAttribute('aria-describedby'))), reason, id);
      ctx.env.click(button);
      await ctx.settle();
      assert.equal(ctx.app.get().route.view, 'voices', `${id}: nothing opens`);
      assert.equal(ctx.store.takeIntent('speak'), null);
    }
    assert.equal($id('vo-named-n1-new').getAttribute('aria-disabled'), null, 'Narrator has a Kokoro version');
  });

  it('the search filters voices and presets together and says how many of each match', async () => {
    ctx = await voicesEnv({ sub: '' });
    const search = $id('vo-search');
    const said = () => $id('app-sr-status').textContent;
    search.focus();
    ctx.env.input(search, 'narr');
    await ctx.settle(1000);
    assert.ok($id('vo-named-n1') && !$id('vo-named-n2'));
    assert.deepEqual(presetIds(ctx.$('#vo-voices-list')), ['vo-preset-p3'], 'a voice that matches keeps its presets');
    assert.equal(said(), '1 named voice and 1 preset match “narr”');
    ctx.env.input(search, 'podcast');
    await ctx.settle(1000);
    assert.equal($id('vo-named-n1'), null);
    assert.deepEqual(presetIds(ctx.$('#vo-voices-list')), ['vo-preset-p2']);
    assert.equal(said(), '1 preset matches “podcast”');
    ctx.env.input(search, 'narrator preset');
    await ctx.settle(1000);
    assert.ok($id('vo-named-n1'), 'a preset that matches brings its voice\'s card…');
    assert.deepEqual(presetIds(ctx.$('#vo-voices-list')), ['vo-preset-p3'], '…with only the presets that match');
    assert.equal(said(), '1 named voice and 1 preset match “narrator preset”');
    ctx.env.input(search, 'nobody');
    await ctx.settle(1000);
    assert.match(text(ctx.$('#vo-voices-list .empty__title')), /No named voices or presets match “nobody”/);
    assert.equal(said(), 'No named voices or presets match “nobody”');
    ctx.env.click(ctx.$('[data-action="clear-search"]'));
    await ctx.settle(1000);
    assert.equal(said(), '2 named voices and 4 presets');
    const { voicesSummary } = await load('views/voices/list.js');
    assert.equal(voicesSummary(1, 0, 'je'), '1 named voice matches “je”');
    assert.equal(voicesSummary(0, 2, 'x'), '2 presets match “x”');
    assert.equal(voicesSummary(2, 1, ''), '2 named voices and 1 preset');
  });

  it('the screenshot states: empty and load-error empty both lists, named-menu opens the first card\'s menu; leaving the shot shows the real lists', async () => {
    ctx = await voicesEnv({ sub: '' });
    await ctx.go('voices?shot=empty');
    assert.equal(text(ctx.$('#vo-voices-list .empty__title')), 'No named voices or presets yet');
    await ctx.go('voices?shot=load-error');
    assert.ok(ctx.$('#vo-voices-list .empty--error'));
    await ctx.go('voices');
    assert.ok($id('vo-named-n1') && $id('vo-preset-p1'), 'the real lists again');
    await ctx.go('voices?shot=named-menu');
    assert.equal($id('vo-named-n1-menu').hidden, false);
  });

  it('presetSummary on a card leaves out the voice (the card names it), and the model when the voice has one version', async () => {
    const { presetSummary } = await load('views/voices/presets.js');
    const named = namedVoices().voices;
    const p3 = presets().profiles[2];
    assert.equal(presetSummary(p3, named, { identity: named[0] }), 'Kokoro · 1.25× · WAV · Instructions');
    const one = { ...named[0], realizations: [named[0].realizations[0]] };
    assert.equal(presetSummary(p3, named, { identity: one }), '1.25× · WAV · Instructions');
    assert.equal(presetSummary(p3, named), 'Kokoro · Narrator · 1.25× · WAV · Instructions', 'elsewhere it names the voice');
  });
});
