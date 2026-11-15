// Voices: the presets in Voices' one list (inventory SE-05). A
// preset sits on its named voice's card, or is a card of its own in "Presets without a named voice";
// tests/js/voices/list.test.js covers what the one list adds.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames, accessibleName } from '../harness/a11y.js';
import { voicesEnv, presets, json, load } from './helpers.js';

let ctx;
afterEach(() => ctx?.env.restore());

const row = (id) => document.getElementById(`vo-preset-${id}`);
const text = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
const toggle = (id) => document.getElementById(`vo-preset-${id}-default`);

describe('views/voices presets', () => {
  it('[SE-05] each preset shows its summary; the default preset comes first where it sits, with a "Default" badge and its "Default in Speak" toggle pressed', async () => {
    ctx = await voicesEnv({ sub: 'presets' });
    const rows = ctx.$$('#vo-voices-list .vo-preset');
    assert.deepEqual(rows.map((r) => r.id), ['vo-preset-p3', 'vo-preset-p1', 'vo-preset-p2', 'vo-preset-p4'], "Narrator's preset on its card, then the presets without a named voice, the default first");
    assert.equal(text(row('p1').querySelector('.vo-preset__name')), 'Audiobook 1.1×');
    assert.equal(text(row('p1').querySelector('.vo-preset__meta')), 'Kokoro · Heart · 1.1× · MP3');
    assert.equal(text(row('p2').querySelector('.vo-preset__meta')), 'Kokoro · Adam · 1× · MP3 · Podcast EQ · Normalize');
    assert.equal(text(row('p3').querySelector('.vo-preset__meta')), 'Kokoro · 1.25× · WAV · Instructions', 'on its named voice\'s card, which names the voice');
    assert.ok(document.getElementById('vo-named-n1').contains(row('p3')));
    assert.equal(text(row('p4').querySelector('.vo-preset__meta')), 'Chatterbox · Recording demo_reference · 1× · WAV');
    assert.deepEqual(rows.map((r) => text(r.querySelector('.vo-preset__badge'))), ['', 'Default', '', ''], 'only the default preset reads "Default"');
    assert.deepEqual(rows.map((r) => text(r.querySelector('.vo-default'))), Array(4).fill('Default in Speak'), 'a toggle keeps its label in both states');
    assert.equal(toggle('p1').getAttribute('aria-pressed'), 'true');
    assert.equal(toggle('p2').getAttribute('aria-pressed'), 'false');
    assert.equal(accessibleName(toggle('p2')), 'Default in Speak: Podcast EQ', 'one name for both states; aria-pressed says which');
    assert.equal(accessibleName(toggle('p1')), 'Default in Speak: Audiobook 1.1×');
    assert.equal(toggle('p2').getAttribute('data-tooltip'), 'Speak opens with this preset', 'the tooltip says what the default does');
    assert.equal(accessibleName(document.getElementById('vo-preset-p2-use')), 'Use in Speak: Podcast EQ');
    assertAccessibleNames(ctx.section);
  });

  it('[SE-05] with no presets each named voice\'s card says so and offers New preset', async () => {
    ctx = await voicesEnv({ sub: 'presets', routes: { 'GET /api/profiles': () => json(200, { profiles: [], default_profile_id: null }) } });
    assert.deepEqual(ctx.$$('#vo-voices-list .vo-card__nopresets').map(text), ['No presets use it yet.', 'No presets use it yet.']);
    assert.equal(ctx.$('#vo-voices-list .vo-group'), null, 'no group of presets without a named voice');
    assert.ok(document.getElementById('vo-named-n1-new'));
  });

  it('[SE-05] Use in Speak hands the preset to Speak with the apply-preset intent', async () => {
    ctx = await voicesEnv({ sub: 'presets' });
    ctx.env.click(document.getElementById('vo-preset-p2-use'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'speak');
    assert.deepEqual({ ...ctx.store.takeIntent('speak') }, { type: 'apply-preset', id: 'p2' });
  });

  it('[SE-05] the Default in Speak toggle POSTs /api/profiles/{id}/default and the pressed state moves; focus stays on it', async () => {
    ctx = await voicesEnv({
      sub: 'presets',
      routes: { 'POST /api/profiles/:id/default': (req) => json(200, { profiles: presets().profiles, default_profile_id: req.params.id }) },
    });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    toggle('p2').focus();
    ctx.env.click(toggle('p2'));
    await ctx.settle();
    await ctx.settle();
    const calls = ctx.calls('POST /api/profiles/:id/default');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].path, '/api/profiles/p2/default');
    assert.equal(toggle('p2').getAttribute('aria-pressed'), 'true');
    assert.equal(toggle('p1').getAttribute('aria-pressed'), 'false');
    assert.equal(ctx.$$('#vo-voices-list .vo-group .vo-preset')[0].id, 'vo-preset-p2', 'the default is listed first in its group');
    assert.deepEqual(ctx.$$('#vo-voices-list .vo-preset__badge').map((b) => b.closest('.vo-preset').id), ['vo-preset-p2'], 'the "Default" badge moved with it');
    assert.ok(ctx.toasts().includes('Speak opens with Podcast EQ from now on'));
    assert.equal(document.activeElement?.id, 'vo-preset-p2-default');
    assert.deepEqual(events, ['presets']);
    ctx.env.click(toggle('p2'));
    await ctx.settle();
    assert.equal(ctx.calls('POST /api/profiles/:id/default').length, 1, 'the default can only move, so pressing it again sends nothing');
  });

  it('[SE-05] Rename PUTs the full payload (the server replaces the preset); a 409 keeps the prompt open', async () => {
    ctx = await voicesEnv({
      sub: 'presets',
      routes: { 'PUT /api/profiles/:id': (req) => (req.json.name === 'Podcast EQ' ? json(409, { error: { message: 'Profile name already exists' } }) : json(200, { ...req.json, id: req.params.id })) },
    });
    await ctx.menu('vo-preset-p3-more', 'rename');
    assert.equal(ctx.promptState().title, 'Rename preset');
    assert.equal(ctx.promptState().value, 'Narrator preset');
    await ctx.answerPrompt('Podcast EQ');
    assert.equal(ctx.promptState().open, true);
    assert.match(ctx.promptState().error, /already exists/);
    await ctx.answerPrompt('Calm narrator');
    const puts = ctx.calls('PUT /api/profiles/:id');
    assert.equal(puts.length, 2);
    assert.equal(puts[1].path, '/api/profiles/p3');
    assert.deepEqual(puts[1].json, {
      name: 'Calm narrator', backend: 'kokoro', model: 'kokoro', voice: 'af_bella', speed: 1.25, format: 'wav', blend: null,
      reference_audio_id: null, voice_identity_id: 'n1', instructions: 'Calm', effects: [],
    });
    assert.ok(ctx.toasts().includes('Renamed Narrator preset to Calm narrator'));
  });

  it('[SE-05] Delete asks "Delete preset “N”?", then DELETEs it; Cancel sends nothing', async () => {
    let data = presets();
    ctx = await voicesEnv({ sub: 'presets', routes: { 'GET /api/profiles': () => json(200, data) } });
    await ctx.menu('vo-preset-p2-more', 'delete');
    assert.equal(ctx.confirmText().title, 'Delete preset “Podcast EQ”?');
    await ctx.answerConfirm(false);
    assert.equal(ctx.calls('DELETE /api/profiles/:id').length, 0);
    await ctx.menu('vo-preset-p2-more', 'delete');
    data = { ...data, profiles: data.profiles.filter((p) => p.id !== 'p2') };
    await ctx.answerConfirm(true);
    assert.equal(ctx.calls('DELETE /api/profiles/:id')[0].path, '/api/profiles/p2');
    assert.equal(row('p2'), null);
    assert.ok(ctx.toasts().includes('Deleted preset Podcast EQ'));
    assert.equal(document.activeElement?.id, 'vo-preset-p4-title', 'focus moves to the next preset (p3 sits on Narrator\'s card, before the group)');
  });

  it('[SE-05] deleting the default preset says Speak will open with the last setup', async () => {
    ctx = await voicesEnv({ sub: 'presets' });
    await ctx.menu('vo-preset-p1-more', 'delete');
    assert.match(ctx.confirmText().message, /default preset/);
    await ctx.answerConfirm(false);
  });

  it('[SE-05] presetSummary and renamePayload are pure', async () => {
    const mod = await load('views/voices/presets.js');
    assert.equal(mod.presetSummary({ model: 'kokoro', voice: 'af_bella', blend: 'af_bella(2)+af_sky(1)', speed: 1, format: 'opus', effects: ['robot'] }), 'Kokoro · Blend of Bella and Sky · 1× · OPUS · Robot');
    assert.equal(mod.presetSummary({ backend: 'piper', model: 'piper/en_US-amy-medium', voice: 'amy', speed: 0.9, format: 'mp3', effects: [{ type: 'reverb', room: 'small' }] }), 'Piper en_US-amy-medium · amy · 0.9× · MP3 · Reverb');
    const payload = mod.renamePayload({ id: 'x', name: 'A', backend: 'kokoro', voice: 'af_heart', speed: null, effects: null, created_at: 'now' }, 'B');
    assert.deepEqual(Object.keys(payload).sort(), [...mod.PRESET_FIELDS].sort(), 'only the payload fields are sent');
    assert.equal(payload.name, 'B');
    assert.equal(payload.speed, 1);
    assert.deepEqual(payload.effects, []);
    assert.equal(payload.format, 'mp3');
  });
});
