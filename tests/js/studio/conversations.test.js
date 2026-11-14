// Studio conversations: the list, the script editor, reordering, render and downloads.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { studioEnv, studioServer, teaParty, captureDownloads, delay, json, load, settle, presets, sameNode, noNode, MODELS } from './helpers.js';
import { assertAccessibleNames, liveRegions } from '../harness/a11y.js';

let t = null;
afterEach(() => {
  if (t) t.env.restore();
  t = null;
});

const twoConversations = () => studioServer({
  conversations: [
    teaParty({ updated_at: '2026-10-20T10:00:00Z' }),
    { id: 'conv-2', name: 'Demo dialogue', updated_at: '2026-10-21T10:00:00Z', turns: [{ id: 't-x', speaker: 'Narrator', text: 'Hello there.' }] },
  ],
});

const turnTexts = () => [...t.$('st-turns').querySelectorAll('.st-turn')].map((li) => [li.querySelector('.st-turn__speaker').textContent, li.querySelector('.st-turn__text').textContent]);

describe('pure helpers', () => {
  it('[REV-29] planReorder re-adds the tail from the first moved position, in its new order, and deletes the old tail', async () => {
    const { planReorder } = await load('views/studio/conversations.js');
    const turns = ['a', 'b', 'c', 'd'].map((id) => ({ id, speaker: id.toUpperCase(), text: `${id} text`, profile_id: id === 'b' ? 'p1' : null, effects: [] }));
    const up = planReorder(turns, 2, 1);
    assert.equal(up.k, 1);
    assert.deepEqual(up.order.map((x) => x.id), ['a', 'c', 'b', 'd']);
    assert.deepEqual(up.readd.map((x) => x.text), ['c text', 'b text', 'd text']);
    assert.deepEqual(up.readd[1], { speaker: 'B', text: 'b text', profile_id: 'p1', effects: null });
    assert.deepEqual(up.deleteIds, ['b', 'c', 'd']);
    const down = planReorder(turns, 0, 1);
    assert.deepEqual(down.readd.map((x) => x.text), ['b text', 'a text', 'c text', 'd text']);
    assert.deepEqual(down.deleteIds, ['a', 'b', 'c', 'd']);
    assert.equal(planReorder(turns, 0, 0).k, -1, 'no move');
    assert.equal(planReorder(turns, 3, 4).k, -1, 'past the end');
  });

  it('[ST-06] the render format comes from its path; renderDurationS adds the turns and the pauses between them', async () => {
    const { renderFormat, renderDurationS, SILENCE_MS } = await load('views/studio/conversations.js');
    assert.equal(renderFormat('/data/conversations/c/render.wav'), 'wav');
    assert.equal(renderFormat('/x/render.MP3'), 'mp3');
    assert.equal(renderFormat(null), '');
    const conv = { render_output_path: '/x/render.wav', turns: [{ duration_ms: 1000 }, { duration_ms: 2000 }] };
    assert.equal(renderDurationS(conv), (3000 + SILENCE_MS) / 1000);
    assert.equal(renderDurationS({ ...conv, render_output_path: null }), null);
    assert.equal(renderDurationS({ ...conv, turns: [{ duration_ms: null }] }), null);
  });
});

describe('the conversation list', () => {
  it('[ST-01] loads GET /api/conversations?limit=100&offset=0 and shows one link per conversation, by name', async () => {
    const server = twoConversations();
    server.state.convs.get('conv-2').name = '';
    t = await studioEnv({ server });
    const [call] = t.calls('GET /api/conversations');
    assert.equal(call.query.get('limit'), '100');
    assert.equal(call.query.get('offset'), '0');
    const links = [...t.$('st-list').querySelectorAll('a.row__main')];
    assert.deepEqual(links.map((a) => a.querySelector('.row__title').textContent), ['Untitled conversation', 'The Mad Tea-Party']);
    assert.deepEqual(links.map((a) => a.getAttribute('href')), ['#studio/conversations/conv-2', '#studio/conversations/conv-1']);
    assert.match(links[1].querySelector('.row__meta').textContent, /^Rendered WAV · Edited /);
    assert.equal(t.$('st-list').getAttribute('aria-busy'), 'false');
  });

  it('[BUG-16] an empty list says "No conversations yet" and offers New conversation; there is no Load or Delete for a selection', async () => {
    t = await studioEnv();
    const empty = t.$('st-list').querySelector('.empty');
    assert.equal(empty.querySelector('.empty__title').textContent, 'No conversations yet');
    assert.ok(empty.querySelector('[data-action="new-conv"]'), 'New conversation is offered');
    assert.equal(t.$('st-list-pane').getAttribute('data-empty'), 'true', 'once: below 960 px studio.css hides the pane’s own button, which would repeat it');
    const labels = t.$$('button').map((b) => b.textContent.trim());
    assert.ok(!labels.some((l) => /^Load$|^Delete$/.test(l)), labels.join('|'));
    assert.equal(t.$('st-conv-more').hidden, true, 'nothing is open, so there is nothing to delete');
  });

  it('[BUG-16] beside the draft (960 px and up) the empty list says how the draft starts one; New conversation is the pane’s secondary button, where focus goes after the last conversation is deleted', async () => {
    t = await studioEnv({ server: studioServer({ conversations: [teaParty()] }), path: 'studio/conversations/conv-1', viewport: { width: 1440, height: 900 }, media: { 'min-width: 960px': true } });
    assert.equal(t.$('st-list-pane').getAttribute('data-empty'), 'false');
    await t.menu(t.$('st-conv-more'), 'Delete conversation…');
    await t.answerConfirm(true);
    assert.equal(t.app.get().route.path, 'studio/conversations');
    assert.equal(t.$('st-conversations').getAttribute('data-open'), 'none');
    assert.equal(t.$('st-editor-title').textContent, 'New conversation', 'the editor is the draft');
    assert.equal(t.$('st-list-pane').getAttribute('data-empty'), 'true');
    const empty = t.$('st-list').querySelector('.st-list-empty');
    assert.equal(empty.querySelector('.st-list-empty__beside').textContent, 'Adding a turn starts your first one.');
    assert.ok(t.$('st-new-empty').classList.contains('st-list-empty__new'), 'the primary that studio.css hides beside the draft');
    sameNode(document.activeElement, t.$('st-new'), 'focus goes to the New conversation button this width shows');
    await t.click(t.$('st-new'));
    await t.answerPrompt('Interview');
    assert.deepEqual(t.calls('POST /api/conversations')[0].json, { name: 'Interview', turns: [] }, 'naming a conversation first stays possible');
  });

  it('[ST-08] opening a conversation from the list loads it into the editor, whose heading is its name', async () => {
    t = await studioEnv({ server: twoConversations() });
    const link = t.$('st-list').querySelector('a[data-id="conv-1"]');
    await t.click(link);
    assert.equal(t.app.get().route.path, 'studio/conversations/conv-1');
    assert.equal(t.$('st-editor-title').textContent, 'The Mad Tea-Party');
    sameNode(document.activeElement, t.$('st-editor-title'), 'focus moves to the opened script');
    assert.equal(t.$('st-conversations').getAttribute('data-open'), 'conv');
    assert.equal(t.$('st-list').querySelector('.row.is-selected a').getAttribute('data-id'), 'conv-1');
    assert.equal(t.$('st-list').querySelector('[aria-current="true"]').getAttribute('data-id'), 'conv-1');
    assert.equal(t.$$('.st-turn').length, 3);
  });

  it('[ST-08] a failed load keeps the list, says what went wrong and offers Retry', async () => {
    const server = twoConversations();
    let fail = true;
    t = await studioEnv({
      server,
      routes: { 'GET /api/conversations/:id': (req) => (fail ? server.envelope(500, 'database is locked') : server.routes['GET /api/conversations/:id'](req)) },
    });
    await t.go('studio/conversations/conv-1');
    const state = t.$('st-script').querySelector('.empty--error');
    assert.ok(state, 'an error state, never a blank editor');
    assert.equal(state.querySelector('.empty__title').textContent, "Couldn't load the conversation");
    assert.equal(t.$('st-list').querySelectorAll('a.row__main').length, 2, 'the list is still there');
    fail = false;
    await t.click(state.querySelector('[data-action="retry-load"]'));
    assert.equal(t.$('st-editor-title').textContent, 'The Mad Tea-Party');
    assert.equal(t.$$('.st-turn').length, 3);
  });

  it('[ST-08] a deep link to a conversation that no longer exists says so and replaces the route with the list', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/gone' });
    await t.settle();
    assert.equal(t.app.get().route.path, 'studio/conversations');
    assert.equal(location.hash, '#studio/conversations');
    const notice = t.$('st-conv-notice');
    assert.equal(notice.hidden, false);
    assert.equal(notice.querySelector('.banner__title').textContent, 'That conversation no longer exists');
    assert.equal(t.$('st-conversations').getAttribute('data-open'), 'none');
  });
});

describe('new conversations and turns', () => {
  it('[ST-02][TE-09] New conversation asks for a name, POSTs {name, turns: []} and opens it with nothing rendered', async () => {
    t = await studioEnv({ server: twoConversations() });
    t.env.click(t.$('st-new'));
    await t.settle();
    const input = document.getElementById('app-prompt-input');
    assert.equal(document.getElementById('app-prompt-label').textContent, 'Conversation name');
    assert.match(input.value, /^Conversation \w{3} \d{1,2}, \d{4}/, 'defaults to "Conversation <local date>"');
    await t.answerPrompt('Interview');
    const [post] = t.calls('POST /api/conversations');
    assert.deepEqual(post.json, { name: 'Interview', turns: [] });
    assert.match(t.app.get().route.path, /^studio\/conversations\/c\d+$/);
    assert.equal(t.$('st-editor-title').textContent, 'Interview');
    assert.equal(t.$('st-download').disabled, true, 'nothing rendered yet');
    assert.equal(t.$('st-render-status-text').textContent, 'Ready');
    assert.equal(t.$('st-turns-empty').hidden, false);
    assert.equal(t.$('st-turns-empty').textContent, 'Add the first turn below.');
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Created “Interview”'));
    sameNode(document.activeElement, t.$('st-text'), 'ready to write the first turn');
  });

  it('[ST-02] cancelling the name prompt creates nothing', async () => {
    t = await studioEnv({ server: twoConversations() });
    t.env.click(t.$('st-new'));
    await t.settle();
    await t.answerPrompt(null);
    assert.equal(t.calls('POST /api/conversations').length, 0);
  });

  it('[ST-03] Add turn with no conversation open creates one first, then POSTs the turn; the speaker defaults to "Speaker"', async () => {
    t = await studioEnv();
    t.env.input(t.$('st-text'), '  Once upon a time.  ');
    await t.click(t.$('st-add'));
    const [create] = t.calls('POST /api/conversations');
    assert.match(create.json.name, /^Conversation \w{3} \d{1,2}, \d{4}/);
    assert.deepEqual(create.json.turns, []);
    const [turn] = t.calls('POST /api/conversations/:id/turns');
    assert.deepEqual(turn.json, { speaker: 'Speaker', text: 'Once upon a time.', profile_id: null, effects: null });
    const order = t.env.fetch.calls.filter((c) => c.method === 'POST').map((c) => c.path);
    assert.deepEqual(order.slice(0, 2), ['/api/conversations', turn.path], 'the conversation is created before the turn');
    assert.match(t.app.get().route.path, /^studio\/conversations\/c\d+$/);
    assert.equal(t.$('st-text').value, '', 'the text is cleared');
    assert.deepEqual(turnTexts(), [['Speaker', 'Once upon a time.']]);
  });

  it('[ST-03] empty text gives the field error "Enter the turn’s text" and sends nothing', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const before = t.env.fetch.calls.length;
    t.env.input(t.$('st-text'), '   ');
    await t.click(t.$('st-add'));
    const text = t.$('st-text');
    assert.equal(t.$('st-text-error').hidden, false);
    assert.equal(t.$('st-text-error').textContent, 'Enter the turn’s text');
    assert.equal(text.getAttribute('aria-invalid'), 'true');
    assert.ok(text.getAttribute('aria-describedby').split(' ').includes('st-text-error'));
    sameNode(document.activeElement, text);
    assert.equal(t.env.fetch.calls.length, before, 'no request');
    t.env.input(text, 'Fixed');
    assert.equal(t.$('st-text-error').hidden, true, 'typing clears the error');
    assert.equal(text.hasAttribute('aria-invalid'), false);
  });

  it('[ST-03] Ctrl+Enter in the turn text adds the turn with the chosen speaker and preset, and keeps them for the next turn', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    t.env.input(t.$('st-speaker'), 'Dormouse');
    t.env.change(t.$('st-preset'), 'p2');
    t.$('st-text').focus();
    t.env.input(t.$('st-text'), 'Twinkle, twinkle.');
    t.env.key('Enter', { ctrlKey: true }, t.$('st-text'));
    await t.settle();
    await t.settle();
    const [turn] = t.calls('POST /api/conversations/:id/turns');
    assert.deepEqual(turn.json, { speaker: 'Dormouse', text: 'Twinkle, twinkle.', profile_id: 'p2', effects: null });
    assert.equal(t.$('st-speaker').value, 'Dormouse');
    assert.equal(t.$('st-preset').value, 'p2');
    assert.equal(turnTexts().length, 4);
    const speakers = [...t.$('st-speakers').querySelectorAll('option')].map((o) => o.value);
    assert.deepEqual(speakers, ['Hatter', 'Alice', 'March Hare', 'Dormouse'], 'the speakers so far are suggested');
  });

  it('[ST-04][BUG-16] the script shows each turn with its speaker, its preset and its text with line breaks kept', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const turns = t.$$('.st-turn');
    assert.equal(turns.length, 3);
    assert.deepEqual(turns.map((li) => li.querySelector('.st-turn__preset').textContent.replace(/^, /, '')), ['Audiobook 1.1×', 'Podcast EQ', 'Default voice']);
    assert.equal(turns[1].querySelector('.st-turn__text').textContent, 'You should learn not to make personal remarks.\nIt’s very rude.');
    assert.equal(turns[0].querySelector('.visually-hidden').textContent, 'Turn 1: ');
    assert.equal(t.$('st-text').tagName.toLowerCase(), 'textarea', 'turn text is multi-line');
    assert.equal(turns[0].querySelector('.avatar').getAttribute('aria-hidden'), 'true');
    assert.ok(turns[0].querySelector('.avatar').style.getPropertyValue('--tile-hue'), 'the avatar is coloured by its speaker');
    assert.equal(t.$('st-turns').getAttribute('aria-label'), 'Script');
    assertAccessibleNames(t.$('st-turns'));
  });

  it('[ST-04][TE-09] Delete turn… confirms "Delete this turn?", deletes that turn and reloads the script', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const more = t.$('st-turn-1-more');
    assert.equal(more.getAttribute('aria-label'), 'More actions for turn 2');
    t.env.click(more);
    await t.settle();
    const item = [...document.getElementById('st-turn-1-menu').querySelectorAll('[role="menuitem"]')].find((i) => i.textContent.trim() === 'Delete turn…');
    t.env.click(item);
    await t.settle();
    const shown = t.confirmText();
    assert.equal(shown.title, 'Delete this turn?');
    assert.match(shown.message, /^Alice: “You should learn/);
    await t.answerConfirm(true);
    const [del] = t.calls('DELETE /api/conversations/:id/turns/:tid');
    assert.equal(del.path, '/api/conversations/conv-1/turns/t-b');
    assert.deepEqual(turnTexts().map((x) => x[0]), ['Hatter', 'March Hare']);
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Turn deleted'));
    sameNode(document.activeElement, t.$('st-turn-1-more'), 'focus moves to the turn now in that place');
  });

  it('[ST-10] the turn preset select lists the presets and keeps its choice when they reload', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, path: 'studio/conversations/conv-1' });
    const select = t.$('st-preset');
    assert.deepEqual([...select.querySelectorAll('option')].map((o) => [o.value, o.textContent]), [['', 'None'], ['p1', 'Audiobook 1.1×'], ['p2', 'Podcast EQ']]);
    t.env.change(select, 'p2');
    server.state.presets = { profiles: [...presets().profiles, { id: 'p3', name: 'Whisper', model: 'kokoro', voice: 'af_sky', speed: 1 }], default_profile_id: 'p1' };
    t.store.emit('voices:changed', { what: 'presets' });
    await t.settle();
    assert.equal(t.calls('GET /api/profiles').length, 2, 'presets reload when they change elsewhere');
    assert.equal(select.querySelectorAll('option').length, 4);
    assert.equal(select.value, 'p2', 'the choice is kept');
    server.state.presets = { profiles: [presets().profiles[0]], default_profile_id: 'p1' };
    t.store.emit('voices:changed', { what: 'presets' });
    await t.settle();
    assert.equal(select.value, '', 'a deleted preset falls back to None');
    assert.match(t.$('st-turn-1').querySelector('.st-turn__preset').textContent, /Preset missing/, 'a turn whose preset is gone says so');
  });

  it('[ST-10] a preset saved while Studio is hidden is in the turn Preset select when Studio shows again, even within 30 s', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, path: 'studio/conversations/conv-1' });
    await t.go('history');
    server.state.presets = { profiles: [...presets().profiles, { id: 'p3', name: 'Whisper', model: 'kokoro', voice: 'af_sky', speed: 1 }], default_profile_id: 'p1' };
    t.store.emit('voices:changed', { what: 'presets' });
    await t.settle();
    assert.equal(t.calls('GET /api/profiles').length, 1, 'nothing loads while Studio is hidden');
    t.env.clock.tick(5000);
    await t.go('studio/conversations/conv-1');
    assert.equal(t.calls('GET /api/profiles').length, 2, 'the presets are 5 s old, but they changed');
    assert.equal(t.calls('GET /api/conversations').length, 1, 'the other lists are fresh and reused');
    assert.deepEqual([...t.$('st-preset').querySelectorAll('option')].map((o) => o.textContent), ['None', 'Audiobook 1.1×', 'Podcast EQ', 'Whisper']);
    t.store.emit('voices:changed', { what: 'voices' });
    await t.settle();
    assert.equal(t.calls('GET /api/profiles').length, 2, 'a change to something else leaves the presets alone');
  });
});

describe('reordering', () => {
  it('[REV-29] Move up re-adds the moved tail first, then deletes the old tail, reloads and keeps focus on the moved turn', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, path: 'studio/conversations/conv-1' });
    t.env.fetch.reset();
    const up = t.$('st-turn-2-up');
    assert.equal(up.getAttribute('aria-label'), 'Move turn 3 up');
    up.focus();
    await t.click(up);
    const writes = t.env.fetch.calls.filter((c) => c.method !== 'GET').map((c) => `${c.method} ${c.path}${c.json ? ` ${c.json.speaker}` : ''}`);
    assert.deepEqual(writes, [
      'POST /api/conversations/conv-1/turns March Hare',
      'POST /api/conversations/conv-1/turns Alice',
      'DELETE /api/conversations/conv-1/turns/t-b',
      'DELETE /api/conversations/conv-1/turns/t-c',
    ], 'add first, then delete: an interruption can duplicate a turn but never lose one');
    const readded = t.calls('POST /api/conversations/:id/turns')[1].json;
    assert.deepEqual(readded, { speaker: 'Alice', text: 'You should learn not to make personal remarks.\nIt’s very rude.', profile_id: 'p2', effects: null });
    assert.deepEqual(turnTexts().map((x) => x[0]), ['Hatter', 'March Hare', 'Alice']);
    sameNode(document.activeElement, t.$('st-turn-1-up'), 'focus follows the moved turn');
  });

  it('[REV-29] the first turn cannot move up and the last cannot move down', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    assert.equal(t.$('st-turn-0-up').disabled, true);
    assert.equal(t.$('st-turn-2-down').disabled, true);
    assert.equal(t.$('st-turn-0-down').disabled, false);
  });

  it('[REV-29] an interrupted reorder reloads the script and warns that turns may appear twice; no text is lost', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'DELETE /api/conversations/:id/turns/:tid': () => server.envelope(500, 'database is locked') } });
    await t.click(t.$('st-turn-0-down'));
    const banner = t.$('st-editor-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, 'Reordering was interrupted');
    assert.equal(banner.querySelector('.banner__msg').textContent, 'Some turns may appear twice; check the script.');
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'alert');
    const speakers = turnTexts().map((x) => x[0]);
    for (const name of ['Hatter', 'Alice', 'March Hare']) assert.ok(speakers.includes(name), `${name} is still in the script`);
    assert.equal(speakers.length, 6, 'the re-added tail landed after the old one');
  });
});

describe('render and downloads', () => {
  it('[ST-05] Render POSTs {format, sample_rate: 24000, save_turn_audio: true}; "Rendering…" never ticks in the live region; then "Rendered · m:ss"', async () => {
    const server = twoConversations();
    server.state.convs.get('conv-1').render_output_path = null;
    const original = server.routes['POST /api/conversations/:id/render'];
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': (req) => delay(5000, () => original(req)) } });
    assert.equal(t.$('st-render-status-text').textContent, 'Ready');
    const status = t.$('st-render-status');
    assert.equal(status.getAttribute('role'), 'status');
    assert.ok(liveRegions(t.section).some((r) => r.el === status));
    t.env.click(t.$('st-render'));
    await t.settle();
    const [post] = t.calls('POST /api/conversations/:id/render');
    assert.deepEqual(post.json, { format: 'wav', sample_rate: 24000, save_turn_audio: true });
    assert.equal(t.$('st-render-status-text').textContent, 'Rendering…');
    assert.equal(t.$('st-render').getAttribute('aria-busy'), 'true');
    await t.settle(3000);
    const elapsed = t.$('st-render-elapsed');
    assert.equal(elapsed.getAttribute('aria-hidden'), 'true');
    assert.match(elapsed.textContent, /0:0[23]/, 'the counter ticks outside the announced text');
    assert.equal(t.$('st-render-status-text').textContent, 'Rendering…', 'the announced text changes only with the phase');
    await t.settle(2500);
    await t.settle();
    assert.equal(t.$('st-render-status-text').textContent, 'Rendered · 0:04');
    assert.equal(elapsed.textContent, '');
    assert.equal(t.$('st-download').disabled, false);
    assert.equal(t.$('st-play').hidden, false);
  });

  it('[ST-05] a failed render shows a banner with the server’s reason and the status goes back to Ready', async () => {
    const server = twoConversations();
    server.state.convs.get('conv-1').render_output_path = null;
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': () => server.envelope(404, "Voice library entry 'gone' not found") } });
    await t.click(t.$('st-render'));
    const banner = t.$('st-render-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't render");
    assert.equal(banner.querySelector('.banner__msg').textContent, "Voice library entry 'gone' not found");
    assert.equal(t.$('st-render-status-text').textContent, 'Ready');
    assert.equal(t.$('st-download').disabled, true);
  });

  it('[ST-05] a conversation without turns cannot render: Render says why instead of failing', async () => {
    const server = studioServer({ conversations: [{ id: 'c-empty', name: 'Empty' }] });
    t = await studioEnv({ server, path: 'studio/conversations/c-empty' });
    const render = t.$('st-render');
    assert.equal(render.getAttribute('aria-disabled'), 'true');
    assert.equal(t.$('st-render-hint').textContent, 'Add a turn to render the script.');
    assert.equal(render.getAttribute('aria-describedby'), 'st-render-hint');
    t.env.click(render);
    await t.settle();
    assert.equal(t.calls('POST /api/conversations/:id/render').length, 0);
  });

  it('[ST-05] the render plays on the shared player, homed on its conversation', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    await t.click(t.$('st-play'));
    const player = t.app.get().player;
    assert.equal(player.id, 'studio:conv-1');
    assert.equal(player.homeRoute, 'studio/conversations/conv-1');
    assert.equal(player.title, 'The Mad Tea-Party');
    assert.match(document.getElementById('app-audio').src, /\/api\/conversations\/conv-1\/audio/);
    assert.equal(t.$('st-play').getAttribute('aria-label'), 'Pause render');
  });

  it('[ST-05] the newest Play press wins: a slow render that lands after the listener opened another conversation and played it is dropped; a second press while it loads downloads it once', async () => {
    const server = studioServer({ conversations: [
      teaParty({ updated_at: '2026-10-20T10:00:00Z' }),
      { id: 'conv-b', name: 'Short scene', updated_at: '2026-10-21T10:00:00Z', render_output_path: '/data/conversations/conv-b/render.wav', turns: [{ speaker: 'A', text: 'x', duration_ms: 1000 }] },
    ] });
    const audioRoute = server.routes['GET /api/conversations/:id/audio'];
    t = await studioEnv({
      server, path: 'studio/conversations/conv-1', seed: { 'open-speech-api-key': 'k-1' },
      routes: { 'GET /api/conversations/:id/audio': (req) => delay(req.params.id === 'conv-1' ? 2000 : 100, () => audioRoute(req)) },
    });
    t.env.click(t.$('st-play'));
    await t.settle(50);
    await t.go('studio/conversations/conv-b');
    t.env.click(t.$('st-play'));
    await t.settle(300);
    assert.equal(t.app.get().player.id, 'studio:conv-b');
    await t.settle(2000);
    await t.settle();
    assert.equal(t.app.get().player.id, 'studio:conv-b', 'the render of the conversation left behind never takes the player back');
    assert.equal(t.app.get().player.playing, true);
    assert.equal(t.toasts().length, 0);

    await t.go('studio/conversations/conv-1');
    t.env.fetch.reset();
    t.env.click(t.$('st-play'));
    await t.settle(50);
    t.env.click(t.$('st-play'));
    await t.settle(2100);
    await t.settle();
    assert.equal(t.calls('GET /api/conversations/:id/audio').length, 1, 'one download for two presses');
    assert.equal(t.app.get().player.id, 'studio:conv-1');
    assert.equal(t.app.get().player.playing, true);
    assert.equal(t.$('st-play').getAttribute('aria-label'), 'Pause render');
  });

  it('[ST-09] a conversation deleted while its render still downloads never plays, and its failed download says nothing', async () => {
    const server = twoConversations();
    const audioRoute = server.routes['GET /api/conversations/:id/audio'];
    t = await studioEnv({
      server, path: 'studio/conversations/conv-1', seed: { 'open-speech-api-key': 'k-1' },
      routes: { 'GET /api/conversations/:id/audio': (req) => delay(2000, () => audioRoute(req)) },
    });
    t.env.click(t.$('st-play'));
    await t.settle(50);
    await t.menu(t.$('st-conv-more'), 'Delete conversation…');
    await t.answerConfirm(true);
    assert.equal(t.calls('DELETE /api/conversations/:id').length, 1);
    await t.settle(2100);
    await t.settle();
    assert.equal(t.app.get().player, null, 'nothing plays');
    assert.deepEqual(t.toasts().filter((x) => x.kind === 'error'), []);
  });

  it('[ST-06] Download names its format: WAV saves the WAV render as conversation-<id>.wav without rendering again', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const downloads = captureDownloads(t.env);
    assert.equal(t.$('st-download').textContent.trim(), 'Download WAV');
    await t.click(t.$('st-download'));
    assert.equal(t.calls('POST /api/conversations/:id/render').length, 0);
    assert.deepEqual(downloads.map((d) => [d.href, d.download]), [['/api/conversations/conv-1/audio', 'conversation-conv-1.wav']]);
  });

  it('[ST-06] with MP3 chosen, Download renders MP3 first and then saves conversation-<id>.mp3', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const downloads = captureDownloads(t.env);
    const mp3 = t.$('st-format').querySelector('input[value="mp3"]');
    t.env.click(mp3);
    await t.settle();
    assert.equal(t.$('st-download').textContent.trim(), 'Download MP3');
    await t.click(t.$('st-download'));
    await t.settle();
    const [render] = t.calls('POST /api/conversations/:id/render');
    assert.equal(render.json.format, 'mp3');
    assert.deepEqual(downloads.map((d) => d.download), ['conversation-conv-1.mp3']);
  });

  it('[ST-06] with an API key the download is fetched with the key and saved from a blob (the link alone could not send it)', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1', seed: { 'open-speech-api-key': 'k-123' } });
    const downloads = captureDownloads(t.env);
    await t.click(t.$('st-download'));
    const audio = t.calls('GET /api/conversations/:id/audio').at(-1);
    assert.equal(audio.headers.authorization, 'Bearer k-123');
    assert.equal(downloads.length, 1);
    assert.match(downloads[0].href, /^blob:/);
    assert.equal(downloads[0].download, 'conversation-conv-1.wav');
    assert.ok(!downloads[0].href.includes('k-123'));
  });

  it('[ST-07] Download stays disabled until a render exists, and again after a turn changes', async () => {
    const server = twoConversations();
    server.state.convs.get('conv-1').render_output_path = null;
    t = await studioEnv({ server, path: 'studio/conversations/conv-1' });
    const download = t.$('st-download');
    assert.equal(download.disabled, true);
    assert.equal(t.$('st-download-hint').hidden, false);
    assert.equal(download.getAttribute('aria-describedby'), 'st-download-hint');
    await t.click(t.$('st-render'));
    assert.equal(download.disabled, false, 'a render enables it');
    assert.equal(t.$('st-download-hint').hidden, true);
    assert.equal(download.hasAttribute('aria-describedby'), false, 'no stale description once it can download');
    t.env.input(t.$('st-text'), 'One more line.');
    await t.click(t.$('st-add'));
    assert.equal(download.disabled, true, 'a new turn retires the render');
    assert.equal(t.$('st-render-status-text').textContent, 'Ready');
  });

  it('[ST-07] after deleting the open conversation, the next one starts with Download disabled', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    assert.equal(t.$('st-download').disabled, false);
    await t.menu(t.$('st-conv-more'), 'Delete conversation…');
    await t.answerConfirm(true);
    assert.equal(t.app.get().route.path, 'studio/conversations');
    t.env.click(t.$('st-new'));
    await t.settle();
    await t.answerPrompt('Fresh');
    assert.equal(t.$('st-editor-title').textContent, 'Fresh');
    assert.equal(t.$('st-download').disabled, true);
  });

  it('[ST-09][BUG-16] Delete conversation names the open conversation and deletes that one, never another', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    assert.equal(t.$('st-conv-more').getAttribute('aria-label'), 'More actions for The Mad Tea-Party');
    await t.menu(t.$('st-conv-more'), 'Delete conversation…');
    const shown = t.confirmText();
    assert.equal(shown.title, 'Delete “The Mad Tea-Party” and all of its turns?');
    assert.equal(shown.message, 'This can’t be undone.');
    await t.answerConfirm(true);
    const deletes = t.calls('DELETE /api/conversations/:id');
    assert.deepEqual(deletes.map((c) => c.path), ['/api/conversations/conv-1']);
    assert.equal(t.app.get().route.path, 'studio/conversations');
    const ids = [...t.$('st-list').querySelectorAll('a.row__main')].map((a) => a.getAttribute('data-id'));
    assert.deepEqual(ids, ['conv-2']);
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Deleted “The Mad Tea-Party”'));
  });

  it('[ST-09] cancelling the confirm deletes nothing', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    await t.menu(t.$('st-conv-more'), 'Delete conversation…');
    await t.answerConfirm(false);
    assert.equal(t.calls('DELETE /api/conversations/:id').length, 0);
    assert.equal(t.app.get().route.path, 'studio/conversations/conv-1');
  });
});

describe('the render lock and the 409', () => {
  it('[BUG-03] while a render runs, Add turn, Move, Delete turn and Delete conversation are disabled with the reason', async () => {
    const server = twoConversations();
    const original = server.routes['POST /api/conversations/:id/render'];
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': (req) => delay(4000, () => original(req)) } });
    t.env.click(t.$('st-render'));
    await t.settle();
    const hint = t.$('st-lock-hint');
    assert.equal(hint.hidden, false);
    assert.match(hint.textContent, /Rendering… Editing resumes when the render finishes\./);
    const add = t.$('st-add');
    assert.equal(add.getAttribute('aria-disabled'), 'true');
    assert.equal(add.getAttribute('aria-describedby'), 'st-lock-hint');
    for (const id of ['st-turn-0-down', 'st-turn-1-up', 'st-turn-1-down', 'st-turn-2-up']) {
      assert.equal(t.$(id).getAttribute('aria-disabled'), 'true', id);
    }
    assert.ok(t.menuItems(t.$('st-turn-0-more')).every((i) => i.getAttribute('aria-disabled') === 'true'), 'Delete turn… is disabled');
    assert.ok(t.menuItems(t.$('st-conv-more')).every((i) => i.getAttribute('aria-disabled') === 'true'), 'Delete conversation… is disabled');
    t.env.input(t.$('st-text'), 'Sneaky edit.');
    t.env.click(add);
    t.env.click(t.$('st-turn-1-up'));
    await t.settle();
    assert.equal(t.calls('POST /api/conversations/:id/turns').length, 0, 'nothing changes the script mid-render');
    assert.equal(t.calls('DELETE /api/conversations/:id/turns/:tid').length, 0);
    await t.settle(4500);
    await t.settle();
    assert.equal(hint.hidden, true);
    assert.equal(add.hasAttribute('aria-disabled'), false, 'editing resumes');
  });

  it('[BUG-03] a 409 (the script changed in another tab) says so and offers Render again', async () => {
    const server = twoConversations();
    let conflicts = 1;
    const original = server.routes['POST /api/conversations/:id/render'];
    t = await studioEnv({
      server,
      path: 'studio/conversations/conv-1',
      routes: {
        'POST /api/conversations/:id/render': (req) => {
          if (conflicts > 0) {
            conflicts -= 1;
            return server.envelope(409, 'Conversation turns changed during the render. Render it again.');
          }
          return original(req);
        },
      },
    });
    await t.click(t.$('st-render'));
    const banner = t.$('st-render-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, 'This script changed while it was rendering');
    assert.equal(banner.querySelector('.banner__msg').textContent, 'Render again to include the latest turns.');
    const again = [...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Render again');
    assert.ok(again, 'Render again is offered');
    await t.click(again);
    assert.equal(t.calls('POST /api/conversations/:id/render').length, 2);
    assert.match(t.$('st-render-status-text').textContent, /^Rendered/);
  });
});

describe('the view', () => {
  it('[ST-01] the section switch navigates between Conversations and Composer without moving focus', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const composer = t.$('st-sub').querySelector('input[value="composer"]');
    composer.focus();
    t.env.click(composer);
    await t.settle();
    assert.equal(t.app.get().route.path, 'studio/composer');
    assert.equal(t.$('st-composer').hidden, false);
    assert.equal(t.$('st-conversations').hidden, true);
    sameNode(document.activeElement, composer);
    const conversations = t.$('st-sub').querySelector('input[value="conversations"]');
    t.env.click(conversations);
    await t.settle();
    assert.equal(t.app.get().route.path, 'studio/conversations/conv-1', 'back to the conversation that was open');
  });

  it('[ST-01] the partial and the markup built at runtime name every control; Studio has its two status regions', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    assertAccessibleNames(t.section);
    const regions = liveRegions(t.section).map((r) => `${r.el.id}:${r.role}`);
    assert.ok(regions.includes('st-render-status:status'));
    assert.ok(regions.includes('st-mix-status:status'));
  });

  it('[ST-01] a list loaded more than 30 s ago reloads when the view shows again; auth:changed reloads everything', async () => {
    t = await studioEnv({ server: twoConversations() });
    await t.go('history');
    await t.go('studio/conversations');
    assert.equal(t.calls('GET /api/conversations').length, 1, 'fresh data is reused');
    await t.go('history');
    t.env.clock.tick(31000);
    await t.go('studio/conversations');
    assert.equal(t.calls('GET /api/conversations').length, 2, 'stale data reloads');
    await t.go('history');
    t.store.emit('auth:changed');
    await t.go('studio/conversations');
    assert.equal(t.calls('GET /api/conversations').length, 3);
    assert.equal(t.calls('GET /api/profiles').length, 3, 'presets and mixes reload with it');
    assert.equal(t.calls('GET /api/composer/renders').length, 3);
  });
});

/* ───────────── repair pass: work that ends while the user is elsewhere (review of Studio) ───────────── */

/** The action button of the toast whose message matches. */
const toastAction = (pattern) => {
  const el = [...document.querySelectorAll('.toast')].find((x) => pattern.test(x.querySelector('.toast__msg').textContent));
  return el ? el.querySelector('.btn') : null;
};
const focusedId = () => (document.activeElement && document.activeElement.id) || String(document.activeElement && document.activeElement.tagName);
const slowFor = (id, ms, route) => (req) => (req.params.id === id ? delay(ms, () => route(req)) : route(req));

describe('locks belong to one conversation', () => {
  it('[BUG-03] a render locks only its own script: a conversation opened meanwhile stays editable and renders, before and after it ends', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': slowFor('conv-1', 4000, server.routes['POST /api/conversations/:id/render']) } });
    t.env.click(t.$('st-render'));
    await t.settle();
    assert.equal(t.$('st-add').getAttribute('aria-disabled'), 'true', 'the rendering script is locked');
    await t.go('studio/conversations/conv-2');
    const free = (when) => {
      assert.equal(t.$('st-editor-title').textContent, 'Demo dialogue', when);
      assert.equal(t.$('st-render').getAttribute('aria-disabled'), null, `Render is enabled ${when}`);
      assert.equal(t.$('st-render').hasAttribute('aria-busy'), false, `Render is not busy ${when}`);
      assert.equal(t.$('st-add').getAttribute('aria-disabled'), null, `Add turn is enabled ${when}`);
      assert.equal(t.$('st-lock-hint').hidden, true, `no lock hint ${when}`);
      assert.equal(t.$('st-render-status-text').textContent, 'Ready', `its own status ${when}`);
      assert.equal(t.$('st-render-elapsed').textContent, '', `no counter ${when}`);
    };
    free('while The Mad Tea-Party renders');
    await t.settle(4500);
    await t.settle();
    free('after The Mad Tea-Party rendered');
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === '“The Mad Tea-Party” rendered'), 'a render that ends off screen is told');
    await t.click(t.$('st-render'));
    assert.deepEqual(t.calls('POST /api/conversations/:id/render').map((c) => c.path), ['/api/conversations/conv-1/render', '/api/conversations/conv-2/render']);
    await t.click(toastAction(/rendered$/));
    assert.equal(t.app.get().route.path, 'studio/conversations/conv-1', 'Open goes back to it');
    assert.match(t.$('st-render-status-text').textContent, /^Rendered/);
    assert.ok(document.activeElement === t.$('st-editor-title'), `focus: ${focusedId()}`);
  });

  it('[REV-29] a reorder that ends while another conversation is open leaves that one editable, its focus and announcements alone', async () => {
    const server = twoConversations();
    const del = server.routes['DELETE /api/conversations/:id/turns/:tid'];
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'DELETE /api/conversations/:id/turns/:tid': (req) => delay(1000, () => del(req)) } });
    t.env.click(t.$('st-turn-0-down'));
    await t.settle();
    assert.equal(t.$('st-add').getAttribute('aria-disabled'), 'true', 'the script being reordered is locked');
    assert.match(t.$('st-lock-hint').textContent, /Moving the turn… Editing resumes when it is in place\./);
    await t.go('studio/conversations/conv-2');
    assert.equal(t.$('st-add').getAttribute('aria-disabled'), null, 'another script is free while it moves');
    t.$('st-text').focus();
    await t.settle(5000);
    await t.settle();
    assert.equal(t.$('st-editor-title').textContent, 'Demo dialogue');
    assert.equal(t.$('st-add').getAttribute('aria-disabled'), null, 'still free once the reorder ended');
    assert.equal(t.$('st-render').getAttribute('aria-disabled'), null);
    assert.ok(document.activeElement === t.$('st-text'), `focus stays where the user is: ${focusedId()}`);
    assert.doesNotMatch(document.getElementById('app-sr-status').textContent, /moved/, 'nothing is said about another script');
    assert.deepEqual(server.state.convs.get('conv-1').turns.map((x) => x.speaker), ['Alice', 'Hatter', 'March Hare'], 'the move itself completed');
    await t.go('studio/conversations/conv-1');
    assert.deepEqual(turnTexts().map((x) => x[0]), ['Alice', 'Hatter', 'March Hare']);
    assert.equal(t.$('st-add').getAttribute('aria-disabled'), null);
  });

  it('[BUG-03] while a turn moves, Delete turn… and Delete conversation… are disabled with the reason, like Add, Move and Render', async () => {
    const server = twoConversations();
    const del = server.routes['DELETE /api/conversations/:id/turns/:tid'];
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'DELETE /api/conversations/:id/turns/:tid': (req) => delay(1000, () => del(req)) } });
    t.env.click(t.$('st-turn-1-down'));
    await t.settle();
    for (const button of [t.$('st-turn-0-more'), t.$('st-conv-more')]) {
      const items = t.menuItems(button);
      assert.ok(items.length > 0);
      for (const item of items) {
        assert.equal(item.getAttribute('aria-disabled'), 'true', item.textContent.trim());
        assert.equal(item.getAttribute('title'), 'Moving the turn…');
      }
    }
    assert.equal(t.$('st-render').getAttribute('aria-disabled'), 'true');
    assert.equal(t.$('st-turn-0-down').getAttribute('aria-disabled'), 'true');
    await t.settle(5000);
    await t.settle();
    assert.ok(t.menuItems(t.$('st-conv-more')).every((i) => !i.hasAttribute('aria-disabled')), 'free again once the script reloaded');
  });

  it('[ST-06] Download on a rendered conversation downloads while another conversation renders', async () => {
    const server = studioServer({
      conversations: [
        teaParty({ updated_at: '2026-10-20T10:00:00Z' }),
        { id: 'conv-2', name: 'Demo dialogue', updated_at: '2026-10-21T10:00:00Z', render_output_path: '/data/conversations/conv-2/render.wav', turns: [{ id: 't-x', speaker: 'Narrator', text: 'Hello there.', duration_ms: 900 }] },
      ],
    });
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': slowFor('conv-1', 4000, server.routes['POST /api/conversations/:id/render']) } });
    t.env.click(t.$('st-render'));
    await t.settle();
    await t.go('studio/conversations/conv-2');
    const downloads = captureDownloads(t.env);
    const download = t.$('st-download');
    assert.equal(download.disabled, false);
    assert.equal(download.textContent.trim(), 'Download WAV');
    await t.click(download);
    assert.deepEqual(downloads.map((d) => d.download), ['conversation-conv-2.wav'], 'the click is never silently dropped');
    assert.equal(t.calls('POST /api/conversations/:id/render').length, 1, 'a download in the format it has needs no render');
  });
});

describe('work that ends while its script is not on screen', () => {
  it('[ST-05] a render that fails while the Composer shows is told by an error toast with Open; its banner waits in the script', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': () => delay(3000, () => server.envelope(400, 'Saved conversation preset is unavailable')) } });
    t.env.click(t.$('st-render'));
    await t.settle();
    await t.go('studio/composer');
    await t.settle(3500);
    await t.settle();
    const told = t.toasts().filter((x) => x.kind === 'error');
    assert.deepEqual(told.map((x) => x.text), ["Couldn't render: “The Mad Tea-Party”. Saved conversation preset is unavailable"]);
    await t.settle(20000);
    assert.equal(t.toasts().filter((x) => x.kind === 'error').length, 1, 'an error toast stays until it is dismissed');
    await t.click(toastAction(/Couldn't render/));
    assert.equal(t.app.get().route.path, 'studio/conversations/conv-1');
    const banner = t.$('st-render-banner');
    assert.equal(banner.hidden, false, 'the banner is there on return');
    assert.equal(banner.querySelector('.banner__msg').textContent, 'Saved conversation preset is unavailable');
    assert.equal(t.$('st-render-status-text').textContent, 'Rendered · 0:04', 'the earlier render is still there');
    assert.ok(document.activeElement === t.$('st-editor-title'), `focus: ${focusedId()}`);
  });

  it('[ST-05] a render that fails while another conversation is open waits for its own script: the open one shows no banner', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': () => delay(3000, () => server.envelope(500, 'ffmpeg failed')) } });
    t.env.click(t.$('st-render'));
    await t.settle();
    await t.go('studio/conversations/conv-2');
    await t.settle(3500);
    await t.settle();
    assert.equal(t.$('st-render-banner').hidden, true, 'Demo dialogue did not fail');
    assert.ok(t.toasts().some((x) => x.kind === 'error' && /“The Mad Tea-Party”/.test(x.text)));
    await t.go('studio/conversations/conv-1');
    const banner = t.$('st-render-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't render");
    assert.ok([...banner.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Retry'));
    await t.go('studio/conversations/conv-2');
    await t.go('studio/conversations/conv-1');
    assert.equal(t.$('st-render-banner').hidden, true, 'told once: it does not come back after it was seen');
  });

  it('[ST-05] an API-key refusal on Render says "API key needed" with Open Settings', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': () => json(401, { error: { message: 'Invalid API key' } }) } });
    await t.click(t.$('st-render'));
    const banner = t.$('st-render-banner');
    assert.equal(banner.querySelector('.banner__title').textContent, 'API key needed');
    const open = [...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Open Settings');
    assert.ok(open, 'Open Settings is offered');
    assert.equal(open.getAttribute('data-action'), 'open-settings', 'run by the view’s own action map, like any other button');
    await t.click(open);
    assert.equal(t.app.get().route.view, 'settings');
  });
});

describe('answers never move the user (route changes mid-request)', () => {
  it('[ST-03] Add turn in the draft: once the user has left for another view, the new conversation is told by a toast with Open', async () => {
    const server = twoConversations();
    const create = server.routes['POST /api/conversations'];
    t = await studioEnv({ server, routes: { 'POST /api/conversations': (req) => delay(2000, () => create(req)) } });
    t.env.input(t.$('st-text'), 'First line.');
    t.env.click(t.$('st-add'));
    await t.settle();
    await t.go('history');
    await t.settle(2500);
    await t.settle();
    assert.equal(t.app.get().route.path, 'history', 'the answer never takes the user back');
    const [turn] = t.calls('POST /api/conversations/:id/turns');
    assert.equal(turn.json.text, 'First line.', 'the turn was still added');
    const made = turn.path.split('/')[3];
    assert.ok(t.toasts().some((x) => x.kind === 'success' && /^Added the turn to “Conversation /.test(x.text)));
    await t.click(toastAction(/^Added the turn to/));
    assert.equal(t.app.get().route.path, `studio/conversations/${made}`);
    assert.deepEqual(turnTexts(), [['Speaker', 'First line.']]);
  });

  it('[ST-03] Add turn in the draft: a conversation the user opened meanwhile stays open and untouched', async () => {
    const server = twoConversations();
    const create = server.routes['POST /api/conversations'];
    t = await studioEnv({ server, routes: { 'POST /api/conversations': (req) => delay(2000, () => create(req)) } });
    t.env.input(t.$('st-text'), 'First line.');
    t.env.click(t.$('st-add'));
    await t.settle();
    await t.click(t.$('st-list').querySelector('a[data-id="conv-2"]'));
    await t.settle(2500);
    await t.settle();
    assert.equal(t.app.get().route.path, 'studio/conversations/conv-2');
    assert.equal(t.$('st-editor-title').textContent, 'Demo dialogue');
    assert.deepEqual(turnTexts(), [['Narrator', 'Hello there.']], 'the turn went to the new conversation, not this one');
    assert.equal(t.calls('POST /api/conversations/:id/turns')[0].path.includes('conv-2'), false);
  });

  it('[ST-02] a conversation created after the user left is told by a toast with Open, never by taking them back', async () => {
    const server = twoConversations();
    const create = server.routes['POST /api/conversations'];
    t = await studioEnv({ server, routes: { 'POST /api/conversations': (req) => delay(2000, () => create(req)) } });
    t.env.click(t.$('st-new'));
    await t.settle();
    await t.answerPrompt('Interview');
    await t.go('history');
    await t.settle(2500);
    await t.settle();
    assert.equal(t.app.get().route.path, 'history');
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Created “Interview”'));
    await t.click(toastAction(/^Created “Interview”/));
    assert.match(t.app.get().route.path, /^studio\/conversations\/c\d+$/);
    assert.equal(t.$('st-editor-title').textContent, 'Interview');
  });

  it('[ST-09] after Delete conversation, the answer never navigates once the user has gone elsewhere', async () => {
    const server = twoConversations();
    const del = server.routes['DELETE /api/conversations/:id'];
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'DELETE /api/conversations/:id': (req) => delay(2000, () => del(req)) } });
    await t.menu(t.$('st-conv-more'), 'Delete conversation…');
    await t.answerConfirm(true);
    await t.go('history');
    await t.settle(2500);
    await t.settle();
    assert.equal(t.app.get().route.path, 'history', 'still where the user went');
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Deleted “The Mad Tea-Party”'));
    await t.go('studio/conversations');
    assert.deepEqual([...t.$('st-list').querySelectorAll('a.row__main')].map((a) => a.getAttribute('data-id')), ['conv-2']);
  });
});

describe('Details, banners, menus in cards and a clip’s length', () => {
  it('[ST-05] a failed render’s Details read the server’s message and its code, never its JSON envelope; a generic code adds none; the banner’s Retry sits under the message and renders again', async () => {
    const server = twoConversations();
    let reply = () => json(500, { error: { message: 'CUDA out of memory', code: 'tts_error' } });
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'POST /api/conversations/:id/render': () => reply() } });
    await t.click(t.$('st-render'));
    const slot = t.$('st-render-banner');
    const banner = slot.querySelector('.banner');
    assert.equal(banner.querySelector('.banner__msg').textContent, 'CUDA out of memory');
    assert.equal(banner.querySelector('.banner__details pre').textContent, 'CUDA out of memory (tts_error)', 'the code is the power user’s handle');
    assert.ok(banner.classList.contains('banner--stacked'), 'the actions go under the message at every width (components.css)');
    const retry = [...banner.querySelectorAll('.banner__actions button')].find((b) => b.textContent.trim() === 'Retry');
    assert.equal(retry.getAttribute('data-action'), 'render-again');
    reply = () => server.envelope(500, 'Render queue is full');
    await t.click(retry);
    assert.equal(t.calls('POST /api/conversations/:id/render').length, 2, 'Retry renders again');
    assert.equal(slot.querySelector('.banner__msg').textContent, 'Render queue is full');
    noNode(slot.querySelector('.banner__details'), 'the generic http_error code adds nothing to the message');
    assert.doesNotMatch(slot.textContent, /[{}]/, 'never the JSON envelope');
  });

  it('[ST-04][ST-09] a turn’s menu and the conversation’s menu are written right after their buttons inside the script card; open, a menu leaves the card (a size container) for the view section, sits under its button, and Tab goes on from the button', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const card = t.$('st-editor');
    card.style.setProperty('container-type', 'inline-size');   // .card's computed style, in the fake browser
    const more = t.$('st-turn-0-more');
    const menu = document.getElementById('st-turn-0-menu');
    const inPlace = () => menu.previousElementSibling === more && t.$('st-turn-0').contains(menu);
    assert.ok(inPlace(), 'written after its button, in its turn (the document order is the focus order)');
    const convMenu = document.getElementById('st-conv-menu');
    sameNode(convMenu.previousElementSibling, t.$('st-conv-more'));
    assert.deepEqual([...convMenu.querySelectorAll('[role="menuitem"]')].map((i) => i.textContent.trim()), ['Delete conversation…']);
    assert.equal(convMenu.getAttribute('aria-label'), 'The Mad Tea-Party');
    more.__rect = { left: 300, top: 300, width: 32, height: 32 };   // inside the default 390 px viewport
    menu.__rect = { width: 200, height: 48 };
    more.focus();
    t.env.click(more);
    await t.settle();
    assert.equal(menu.hidden, false);
    sameNode(menu.parentElement, t.section, 'open: in the view section, so its fixed box is placed against the viewport');
    assert.equal(card.contains(menu), false);
    assert.equal(menu.style.getPropertyValue('--menu-x'), `${332 - 200}px`, 'its end edge on the button’s');
    assert.equal(menu.style.getPropertyValue('--menu-y'), `${332 + 4}px`, 'right under the button');
    assert.equal(document.activeElement.textContent.trim(), 'Delete turn…');
    t.env.key('Tab');
    await t.settle();
    assert.equal(menu.hidden, true);
    assert.ok(inPlace(), 'closed: back after its button');
    sameNode(document.activeElement, more, 'Tab goes on from the button, wherever the open menu sat');
    await t.menu(more, 'Delete turn…');
    assert.equal(t.confirmText().title, 'Delete this turn?', 'an item chosen while the menu is out of the card still acts on its turn');
    await t.answerConfirm(false);
    await t.menu(t.$('st-conv-more'), 'Delete conversation…');
    assert.match(t.confirmText().title, /The Mad Tea-Party/);
    await t.answerConfirm(false);
    assert.equal(t.calls('DELETE /api/conversations/:id').length + t.calls('DELETE /api/conversations/:id/turns/:tid').length, 0);
  });

  it('[ST-05] with autoplay refused (the render waits paused: “Tap play to start listening”), the inline transport shows as soon as its metadata loads, and seeks', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1', playPolicy: 'NotAllowedError' });
    await t.click(t.$('st-play'));
    let player = t.app.get().player;
    assert.equal(player.id, 'studio:conv-1');
    assert.equal(player.playing, false);
    assert.match(player.statusText, /Tap play/);
    const transport = t.$('st-transport');
    assert.equal(transport.hidden, true, 'no length yet: a range from 0:00 to 0:00 says nothing');
    const element = document.getElementById('app-audio');
    element.__loadMetadata(72);
    await t.settle();
    player = t.app.get().player;
    assert.equal(player.playing, false, 'nothing plays');
    assert.equal(transport.hidden, false, 'the length is known once the metadata loads (core/audio playClip)');
    assert.equal(t.$('st-seek').getAttribute('max'), '72');
    assert.equal(t.$('st-seek-time').textContent, '0:00 / 1:12');
    t.env.input(t.$('st-seek'), '30', { change: true });
    await t.settle();
    assert.equal(element.currentTime, 30, 'the range seeks the waiting render');
  });
});

describe('repair pass: presets, focus, playback and the shortcut', () => {
  it('[ST-10] presets that did not load are said under the Preset select, with Retry; turns never look as if they had no preset', async () => {
    const server = twoConversations();
    let fail = true;
    t = await studioEnv({ server, path: 'studio/conversations/conv-1', routes: { 'GET /api/profiles': () => (fail ? server.envelope(500, 'database is locked') : json(200, server.state.presets)) } });
    const error = t.$('st-preset-error');
    const select = t.$('st-preset');
    assert.equal(error.hidden, false);
    assert.match(error.textContent, /Presets didn’t load, so only None is listed\./);
    assert.ok(select.getAttribute('aria-describedby').split(' ').includes('st-preset-error'));
    const presetsShown = () => t.$$('.st-turn').map((li) => (li.querySelector('.st-turn__preset') || { textContent: '' }).textContent.replace(/^, /, ''));
    assert.deepEqual(presetsShown(), ['Preset (name not loaded)', 'Preset (name not loaded)', 'Default voice']);
    fail = false;
    await t.click(error.querySelector('[data-action="retry-load"]'));
    assert.equal(error.hidden, true);
    assert.equal(select.getAttribute('aria-describedby'), 'st-preset-hint');
    assert.deepEqual([...select.querySelectorAll('option')].map((o) => o.textContent), ['None', 'Audiobook 1.1×', 'Podcast EQ']);
    assert.deepEqual(presetsShown(), ['Audiobook 1.1×', 'Podcast EQ', 'Default voice']);
  });

  it('[ST-08] a conversation that fails to open from the list takes focus to the editor, whose title says it couldn’t open', async () => {
    const server = twoConversations();
    t = await studioEnv({ server, routes: { 'GET /api/conversations/:id': () => server.envelope(500, 'database is locked') } });
    const link = t.$('st-list').querySelector('a[data-id="conv-1"]');
    link.focus();
    await t.click(link);
    assert.equal(t.$('st-conversations').getAttribute('data-open'), 'conv', 'below 960 px the list it was opened from is hidden now');
    assert.ok(document.activeElement === t.$('st-editor-title'), `focus: ${focusedId()}`);
    assert.equal(t.$('st-editor-title').textContent, "Couldn't open the conversation");
    assert.ok(t.$('st-script').querySelector('.empty--error [data-action="retry-load"]'), 'Retry is next');
  });

  it('[ST-05] while the render plays on its own route (the mini-player hides there), an inline transport shows where it is and seeks', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    const { miniPlayerState } = await load('core/audio.js');
    const transport = t.$('st-transport');
    assert.equal(transport.hidden, true, 'nothing to show before it plays');
    await t.click(t.$('st-play'));
    assert.equal(t.app.get().player.playing, true);
    assert.equal(transport.hidden, true, 'not until its length is known (a range from 0:00 to 0:00 says nothing)');
    const element = document.getElementById('app-audio');
    element.__loadMetadata(72);
    element.__timeupdate(12);
    await t.settle(300);
    assert.equal(miniPlayerState(t.app.get()).visible, false, 'the mini-player hides on the render’s own route');
    assert.equal(transport.hidden, false);
    const seek = t.$('st-seek');
    assert.equal(seek.getAttribute('type'), 'range');
    assert.equal(seek.getAttribute('aria-label'), 'Position in “The Mad Tea-Party”');
    assert.equal(seek.getAttribute('max'), '72');
    assert.equal(seek.value, '12');
    assert.equal(seek.getAttribute('aria-valuetext'), '12 seconds of 1 minute 12 seconds');
    assert.equal(t.$('st-seek-time').textContent, '0:12 / 1:12');
    assert.equal(t.$('st-seek-time').getAttribute('aria-hidden'), 'true', 'the ticking time is never read out');
    t.env.input(seek, '40', { change: true });
    await t.settle();
    assert.equal(element.currentTime, 40, 'the range seeks the render');
    await t.go('studio/conversations/conv-2');
    assert.equal(transport.hidden, true, 'another script shows no transport for this render');
    assert.equal(miniPlayerState(t.app.get()).visible, true, 'the mini-player takes over elsewhere');
  });

  it('[ST-05][REV-20] with an API key and the iPhone rule (play only inside a gesture), Play render plays: prime() runs inside the click', async () => {
    const server = twoConversations();
    const audioRoute = server.routes['GET /api/conversations/:id/audio'];
    t = await studioEnv({
      server, path: 'studio/conversations/conv-1', playPolicy: 'gesture', seed: { 'open-speech-api-key': 'k-1' },
      routes: { 'GET /api/conversations/:id/audio': (req) => delay(300, () => audioRoute(req)) },
    });
    t.env.click(t.$('st-play'));
    await t.settle(400);
    await t.settle();
    const player = t.app.get().player;
    assert.equal(player.id, 'studio:conv-1');
    assert.equal(player.playing, true, `plays after the keyed fetch (${player.statusText})`);
    assert.equal(t.calls('GET /api/conversations/:id/audio').at(-1).headers.authorization, 'Bearer k-1');
    assert.match(document.getElementById('app-audio').src, /^blob:/, 'the key never goes in the media URL');
  });

  it('[ST-03] the turn text advertises the shortcut this platform has: Control+Enter, or Meta+Enter on Apple', async () => {
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1' });
    assert.equal(t.$('st-text').getAttribute('aria-keyshortcuts'), 'Control+Enter');
    assert.equal(t.$('st-kbd-mod').textContent, 'Ctrl');
    t.env.restore();
    t = await studioEnv({ server: twoConversations(), path: 'studio/conversations/conv-1', platform: 'MacIntel' });
    assert.equal(t.$('st-text').getAttribute('aria-keyshortcuts'), 'Meta+Enter');
    assert.equal(t.$('st-kbd-mod').textContent, '⌘');
  });
});

describe('studio/conversations: a model that only designs voices', () => {
  it('a preset on a design-only model is not offered for turns, and a hint points to Design a voice', async () => {
    const server = twoConversations();
    server.state.presets.profiles.push({ id: 'p3', name: 'Designed', model: 'qwen3/1.7b-voice-design', voice: '', instructions: 'A calm voice', speed: 1, format: 'mp3', effects: [] });
    t = await studioEnv({ server, path: 'studio/conversations/conv-1' });
    const select = t.$('st-preset');
    assert.equal(select.querySelectorAll('option').length, 4, 'listed while the models are not known');
    const list = [...MODELS.models, { id: 'qwen3/1.7b-voice-design', type: 'tts', provider: 'qwen3', state: 'loaded', capabilities: { voice_design: true, voice_clone: false, speakers: [] } }];
    t.app.set({ models: { list, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
    await t.settle();
    assert.deepEqual([...select.querySelectorAll('option')].map((o) => o.textContent), ['None', 'Audiobook 1.1×', 'Podcast EQ']);
    assert.equal(t.$('st-preset-design-hint').hidden, false);
    assert.match(t.$('st-preset-design-hint').textContent, /Voices › Design a voice/);
    assert.ok(select.getAttribute('aria-describedby').split(' ').includes('st-preset-design-hint'));
  });
});
