// History: server search, filter chips, Load more, rows and their actions, Delete and Clear all.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { historyEnv, historyServer, speech, transcript, at, delay, load, json, sameNode, noNode } from './helpers.js';
import { bytes } from '../harness/fakes.js';
import { assertAccessibleNames, liveRegions } from '../harness/a11y.js';

let t = null;
afterEach(() => {
  if (t) t.env.restore();
  t = null;
});

const NOW = 1760000000000;
const minutesAgo = (m) => new Date(NOW - m * 60000).toISOString();

function many(n, make = speech) {
  return Array.from({ length: n }, (_, i) => make(i + 1, { created_at: minutesAgo(2 + i / 100) }));
}

function mixed() {
  return historyServer([
    speech(1, { created_at: minutesAgo(2), voice: 'af_heart', speed: 1.25 }),
    transcript(1, { created_at: minutesAgo(3) }),
    speech(2, { created_at: at(1, 18, 30), voice: 'af_bella(2)+af_sky(1)' }),
    speech(3, { created_at: at(3, 10, 0), voice: 'voice:n1', format: 'wav' }),
  ]);
}

const historyCalls = () => t.calls('GET /api/history').filter((c) => c.query.get('limit') !== '1');
const lastQuery = () => Object.fromEntries(historyCalls().at(-1).query);

describe('pure helpers', () => {
  it('[HI-01] historyQuery sends type only when filtered and search only when typed; limit and offset always', async () => {
    const { historyQuery } = await load('views/history/list.js');
    assert.equal(historyQuery({}), 'limit=50&offset=0');
    assert.equal(historyQuery({ type: 'tts', search: '  hobbit ', offset: 50 }), 'type=tts&search=hobbit&limit=50&offset=50');
  });

  it('[HI-05] a row’s meta reads voice · model · speed · format for speech, and the whisper model for a transcript', async () => {
    const { entryMeta, voiceText, sttModelText } = await load('views/history/list.js');
    assert.deepEqual(entryMeta(speech(1, { speed: 1.25 })), ['Heart', 'Kokoro', '1.25×', 'MP3']);
    assert.deepEqual(entryMeta(speech(1, { voice: 'af_bella(2)+af_sky(1)', model: 'piper/en_US-amy-medium', speed: null, format: null })), ['af_bella(2)+af_sky(1)', 'Piper en_US-amy-medium']);
    assert.deepEqual(entryMeta(transcript(1)), ['whisper base']);
    assert.equal(voiceText('voice:n1'), 'Named voice');
    assert.equal(voiceText('voice:n1', new Map([['n1', 'Narrator']])), 'Narrator');
    assert.equal(sttModelText('Systran/faster-distil-whisper-large-v3'), 'whisper distil-large-v3');
  });

  it('[HI-05] entries are grouped under Today, Yesterday and dated headings, newest first', async () => {
    const { groupByDay, dayLabel } = await load('views/history/list.js');
    const now = new Date(2026, 9, 29, 15, 0).getTime();
    const items = [
      { id: 'a', created_at: new Date(2026, 9, 29, 9, 0).toISOString() },
      { id: 'b', created_at: new Date(2026, 9, 29, 8, 0).toISOString() },
      { id: 'c', created_at: new Date(2026, 9, 28, 23, 0).toISOString() },
      { id: 'd', created_at: new Date(2026, 9, 26, 12, 0).toISOString() },
      { id: 'e', created_at: new Date(2025, 2, 3, 12, 0).toISOString() },
    ];
    const groups = groupByDay(items, now);
    assert.deepEqual(groups.map((g) => [g.label, g.items.map((i) => i.id).join('')]), [
      ['Today', 'ab'], ['Yesterday', 'c'], ['Monday, Oct 26', 'd'], ['Mar 3, 2025', 'e'],
    ]);
    assert.equal(dayLabel(null, now), 'Earlier');
  });

  it('[HI-05] a row says "5 min ago" today and its time of day under an older day (the heading carries the date)', async () => {
    const { rowTime } = await load('views/history/list.js');
    const now = new Date(2026, 9, 29, 15, 0).getTime();
    assert.equal(rowTime(new Date(2026, 9, 29, 14, 55).toISOString(), now), '5 min ago');
    // Some ICU builds put a narrow no-break space before PM.
    assert.equal(rowTime(new Date(2026, 9, 26, 16, 12).toISOString(), now).replace(/\u202f/g, ' '), '4:12 PM');
    assert.equal(rowTime(null, now), '');
  });

  it('[HI-07] each row’s More actions button is named for its entry: the kind and a short snippet, or the file name', async () => {
    const { moreLabel } = await load('views/history/list.js');
    assert.equal(moreLabel(speech(1)), 'More actions for speech “Speech entry number 1. In a hole in the…”');
    assert.equal(moreLabel(speech(2, { full_text: 'Hello.' })), 'More actions for speech “Hello.”');
    assert.equal(moreLabel(transcript(1)), 'More actions for transcript meeting-1.m4a');
    assert.equal(moreLabel(speech(3, { full_text: '', text_preview: '' })), 'More actions for speech without text');
  });

  it('[HI-04] the count says "Showing N of T" while there is more, else how many entries there are', async () => {
    const { countText } = await load('views/history/list.js');
    assert.equal(countText(50, 312), 'Showing 50 of 312');
    assert.equal(countText(3, 3), '3 entries');
    assert.equal(countText(1, 1), '1 entry');
    assert.equal(countText(1000, 1312), 'Showing 1,000 of 1,312');
    assert.equal(countText(0, 0), '');
  });

  it('[REV-30] the Clear all confirm names the unfiltered count, and leaves it out when the count is unknown', async () => {
    const { clearCopy } = await load('views/history/list.js');
    assert.deepEqual(clearCopy(312), { title: 'Delete all 312 history entries?', message: 'This can’t be undone.' });
    assert.equal(clearCopy(1).title, 'Delete 1 history entry?');
    assert.equal(clearCopy(null).title, 'Delete all history entries?');
  });

  it('[HI-06][REV-06] the empty states: none yet (with the OS_HISTORY_ENABLED hint), none for a filter, none for a search', async () => {
    const { emptyCopy } = await load('views/history/list.js');
    const none = emptyCopy({});
    assert.equal(none.title, 'No history yet');
    assert.equal(none.text, 'Speech you generate and files you transcribe appear here.');
    assert.equal(none.note, 'If nothing ever appears here, history may be turned off on the server (OS_HISTORY_ENABLED).');
    assert.equal(emptyCopy({ search: ' unicorn ' }).title, 'No history matches “unicorn”.');
    assert.equal(emptyCopy({ type: 'stt' }).title, 'No transcripts yet');
    assert.equal(emptyCopy({ type: 'tts' }).title, 'No speech yet');
  });
});

describe('loading, search and filters', () => {
  it('[HI-01] the first show asks GET /api/history?limit=50&offset=0 and lists the entries newest first, grouped by day', async () => {
    t = await historyEnv({ server: mixed() });
    assert.deepEqual(lastQuery(), { limit: '50', offset: '0' });
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['h1', 's1', 'h2', 'h3']);
    const days = t.$$('h2.hi-day').map((h) => h.textContent);
    assert.equal(days[0], 'Today');
    assert.equal(days[1], 'Yesterday');
    assert.equal(days.length, 3);
    assert.equal(t.$('hi-list').getAttribute('aria-busy'), 'false');
    assert.equal(t.$$('ul.hi-entries')[0].getAttribute('aria-labelledby'), t.$$('h2.hi-day')[0].id);
  });

  it('[HI-01] the filter chips map to type=tts|stt, show which is on, and persist to prefs for the next visit', async () => {
    t = await historyEnv({ server: mixed() });
    const chip = (type) => t.$('hi-filter').querySelector(`[data-type="${type}"]`);
    assert.equal(t.$('hi-filter').getAttribute('role'), 'group');
    assert.equal(chip('').getAttribute('aria-pressed'), 'true');
    await t.click(chip('stt'));
    assert.equal(lastQuery().type, 'stt');
    assert.equal(chip('stt').getAttribute('aria-pressed'), 'true');
    assert.equal(chip('').getAttribute('aria-pressed'), 'false');
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['s1']);
    assert.equal(t.prefs.get().historyFilter, 'stt');
    t.env.restore();
    t = await historyEnv({ server: mixed(), seed: { 'open-speech-prefs': JSON.stringify({ v: 1, historyFilter: 'tts' }) } });
    assert.equal(lastQuery().type, 'tts', 'the saved filter is used on the next visit');
    assert.equal(t.$('hi-filter').querySelector('[data-type="tts"]').getAttribute('aria-pressed'), 'true');
  });

  it('[HI-02] search waits for a 250 ms pause and sends one request, from offset 0', async () => {
    t = await historyEnv({ server: historyServer(many(120)) });
    await t.click(t.$('hi-more'));
    assert.equal(lastQuery().offset, '50');
    const before = historyCalls().length;
    const search = t.$('hi-search');
    t.env.input(search, 'h');
    await t.settle(100);
    t.env.input(search, 'ho');
    await t.settle(100);
    t.env.input(search, 'hobbit');
    await t.settle(249);
    assert.equal(historyCalls().length, before, 'nothing while typing');
    await t.settle(1);
    await t.settle();
    assert.equal(historyCalls().length, before + 1, 'one request once typing pauses');
    assert.deepEqual(lastQuery(), { search: 'hobbit', limit: '50', offset: '0' });
  });

  it('[HI-02] Enter runs the search at once', async () => {
    t = await historyEnv({ server: mixed() });
    const before = historyCalls().length;
    t.$('hi-search').focus();
    t.env.input(t.$('hi-search'), 'meeting');
    t.env.key('Enter', {}, t.$('hi-search'));
    await t.settle();
    assert.equal(historyCalls().length, before + 1);
    assert.equal(lastQuery().search, 'meeting');
    await t.settle(300);
    assert.equal(historyCalls().length, before + 1, 'the pending pause does not search again');
  });

  it('[HI-03] the newest request wins: an older, slower answer never replaces a newer one', async () => {
    const server = mixed();
    const base = server.routes['GET /api/history'];
    t = await historyEnv({
      server,
      routes: { 'GET /api/history': (req) => (req.query.get('type') === 'tts' ? delay(2000, () => base(req)) : base(req)) },
    });
    const chip = (type) => t.$('hi-filter').querySelector(`[data-type="${type}"]`);
    t.env.click(chip('tts'));
    await t.settle();
    t.env.click(chip('stt'));
    await t.settle();
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['s1']);
    await t.settle(2500);
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['s1'], 'the late Speech answer is dropped');
    assert.equal(chip('stt').getAttribute('aria-pressed'), 'true');
  });

  it('[HI-03] an older search that fails late never replaces the newer results with an error', async () => {
    const server = mixed();
    const base = server.routes['GET /api/history'];
    t = await historyEnv({
      server,
      routes: {
        'GET /api/history': (req) => (req.query.get('search') === 'hobbit'
          ? delay(2000, () => json(503, { error: { message: 'Busy', code: 'http_error' } }))
          : base(req)),
      },
    });
    const search = t.$('hi-search');
    t.env.input(search, 'hobbit');
    t.env.key('Enter', {}, search);
    await t.settle();
    t.env.input(search, 'meeting');
    t.env.key('Enter', {}, search);
    await t.settle();
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['s1']);
    await t.settle(2500);
    assert.equal(historyCalls().filter((c) => c.query.get('search') === 'hobbit').length, 1, 'the older search did go out');
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['s1'], 'the newer rows stay');
    assert.equal(t.$('hi-banner').hidden, true, 'no error banner for a request nobody waits for');
    assert.equal(t.$('hi-count').textContent, '1 entry');
  });

  it('[HI-03] paging uses what the controls show now: after a filter change Load more asks that type; a pending search starts again from the top', async () => {
    t = await historyEnv({ server: historyServer([...many(70), ...many(5, transcript)]) });
    await t.click(t.$('hi-filter').querySelector('[data-type="tts"]'));
    await t.click(t.$('hi-more'));
    assert.deepEqual(lastQuery(), { type: 'tts', limit: '50', offset: '50' });
    t.env.input(t.$('hi-search'), 'number 6');
    await t.click(t.$('hi-more'));
    assert.deepEqual(lastQuery(), { type: 'tts', search: 'number 6', limit: '50', offset: '0' }, 'not offset 100 of the old query');
  });

  it('[HI-04] Load more appends the next page and disappears on the last; the count follows', async () => {
    t = await historyEnv({ server: historyServer(many(120)) });
    const more = t.$('hi-more');
    const count = t.$('hi-count');
    assert.equal(count.getAttribute('role'), 'status');
    assert.equal(count.textContent, 'Showing 50 of 120');
    assert.equal(more.hidden, false);
    assert.equal(t.rows().length, 50);
    await t.click(more);
    assert.equal(t.rows().length, 100);
    assert.equal(count.textContent, 'Showing 100 of 120');
    await t.click(more);
    assert.equal(t.rows().length, 120);
    assert.equal(count.textContent, '120 entries');
    assert.equal(more.hidden, true, 'never past the end');
    assert.deepEqual(historyCalls().map((c) => c.query.get('offset')), ['0', '50', '100']);
  });
});

describe('rows and their actions', () => {
  it('[HI-05] a row shows its type, its text (or file name and text), its meta and a relative time with the exact time', async () => {
    t = await historyEnv({ server: mixed() });
    const [tts, stt] = t.rows();
    assert.equal(tts.querySelector('.hi-entry__type').getAttribute('aria-hidden'), 'true');
    const ttsText = tts.querySelector('.hi-entry__text');
    assert.equal(ttsText.querySelector('.visually-hidden').textContent, 'Speech: ');
    assert.match(ttsText.textContent, /^Speech: “Speech entry number 1\./);
    const sttText = stt.querySelector('.hi-entry__text');
    assert.equal(sttText.querySelector('.visually-hidden').textContent, 'Transcript: ');
    assert.equal(sttText.querySelector('.hi-entry__file').textContent, 'meeting-1.m4a');
    assert.match(tts.querySelector('.hi-entry__meta').textContent, /^Heart · Kokoro · 1\.25× · MP3/);
    assert.match(stt.querySelector('.hi-entry__meta').textContent, /^whisper base/);
    const time = tts.querySelector('time');
    assert.equal(time.querySelector('[aria-hidden="true"]').textContent, '2 min ago');
    assert.ok(time.getAttribute('title').length > 8);
    assert.equal(time.querySelector('.visually-hidden').textContent, time.getAttribute('title'), 'screen readers hear the exact time');
    assert.equal(time.getAttribute('datetime'), minutesAgo(2));
    assertAccessibleNames(t.$('hi-list'));
  });

  it('[HI-05] Open in Speak is offered on speech only; Copy text on every row; a menu with Delete… on every row', async () => {
    t = await historyEnv({ server: mixed() });
    const [tts, stt] = t.rows();
    assert.ok(tts.querySelector('[data-action="open-in-speak"]'));
    noNode(stt.querySelector('[data-action="open-in-speak"]'), 'only speech can be reused');
    assert.ok(stt.querySelector('[data-action="copy-text"]'));
    const ttsMenu = document.getElementById(tts.querySelector('.hi-entry__more').getAttribute('aria-controls'));
    const sttMenu = document.getElementById(stt.querySelector('.hi-entry__more').getAttribute('aria-controls'));
    assert.deepEqual([...ttsMenu.querySelectorAll('[role="menuitem"]')].map((i) => i.textContent.trim()), ['Copy text', 'Delete…']);
    assert.deepEqual([...sttMenu.querySelectorAll('[role="menuitem"]')].map((i) => i.textContent.trim()), ['Delete…']);
    const open = tts.querySelector('[data-action="open-in-speak"]');
    assert.equal(open.getAttribute('aria-describedby'), tts.querySelector('.hi-entry__text').id, 'each button says which entry it acts on');
  });

  it('[HI-07] the rows’ More actions buttons have names of their own, so a list of buttons tells them apart', async () => {
    t = await historyEnv({ server: mixed() });
    const buttons = t.rows().map((r) => r.querySelector('.hi-entry__more'));
    const names = buttons.map((b) => b.getAttribute('aria-label'));
    assert.equal(new Set(names).size, names.length, names.join(' | '));
    assert.equal(names[1], 'More actions for transcript meeting-1.m4a');
    assert.match(names[0], /^More actions for speech “Speech entry number 1\./);
    assert.equal(t.$('hi-more').classList.contains('hi-entry__more'), false, 'Load more and the row menus no longer share a class');
    assert.equal(t.$$('.hi-entry__more').length, 4);
  });

  it('[HI-05] Open in Speak hands the whole entry to Speak (the apply-history intent) and goes there', async () => {
    t = await historyEnv({ server: mixed() });
    const store = await load('core/store.js');
    await t.click(t.rows()[0].querySelector('[data-action="open-in-speak"]'));
    assert.equal(t.app.get().route.view, 'speak');
    const intent = store.takeIntent('speak');
    assert.equal(intent.type, 'apply-history');
    assert.equal(intent.entry.id, 'h1');
    assert.equal(intent.entry.full_text, speech(1).full_text);
    assert.equal(intent.entry.speed, 1.25);
  });

  it('[HI-05][TE-09] Copy text copies the full text, not the 180-character preview, and confirms', async () => {
    const long = `${'word '.repeat(60)}the end.`;
    t = await historyEnv({ server: historyServer([transcript(9, { full_text: long, created_at: minutesAgo(1) })]) });
    await t.click(t.rows()[0].querySelector('[data-action="copy-text"]'));
    assert.equal(t.env.clipboard.written.at(-1), long);
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Text copied'));
  });

  it('[HI-07][TE-09] Delete… confirms "Delete this history entry?", deletes it and reloads the list', async () => {
    t = await historyEnv({ server: mixed() });
    await t.menu(t.rows()[1].querySelector('.hi-entry__more'), 'Delete…');
    const shown = t.confirmText();
    assert.equal(shown.title, 'Delete this history entry?');
    assert.match(shown.message, /^meeting-1\.m4a: “Okay so the plan/);
    await t.answerConfirm(true);
    const [del] = t.calls('DELETE /api/history/:id');
    assert.equal(del.path, '/api/history/s1');
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['h1', 'h2', 'h3']);
    assert.equal(t.$('hi-count').textContent, '3 entries');
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'History entry deleted'));
    sameNode(document.activeElement, t.rows()[1].querySelector('.hi-entry__actions button'), 'focus moves to the entry now in its place');
  });

  it('[HI-07] deleting the only entry of the last loaded page reloads to the new last page, never past the end', async () => {
    t = await historyEnv({ server: historyServer(many(51)) });
    await t.click(t.$('hi-more'));
    assert.equal(t.rows().length, 51);
    await t.menu(t.rows()[50].querySelector('.hi-entry__more'), 'Delete…');
    await t.answerConfirm(true);
    const reload = historyCalls().at(-1);
    assert.equal(reload.query.get('offset'), '0');
    assert.equal(reload.query.get('limit'), '51', 'what was loaded is loaded again');
    assert.equal(t.rows().length, 50);
    assert.equal(t.$('hi-count').textContent, '50 entries');
    assert.equal(t.$('hi-more').hidden, true);
    sameNode(document.activeElement, t.rows()[49].querySelector('.hi-entry__actions button'), 'focus stays at the end of the list');
  });

  it('[HI-07] cancelling the confirm deletes nothing; a failed delete says "Couldn\'t delete"', async () => {
    const server = mixed();
    t = await historyEnv({ server, routes: { 'DELETE /api/history/:id': () => json(500, { error: { message: 'database is locked', code: 'http_error' } }) } });
    await t.menu(t.rows()[0].querySelector('.hi-entry__more'), 'Delete…');
    await t.answerConfirm(false);
    assert.equal(t.calls('DELETE /api/history/:id').length, 0);
    await t.menu(t.rows()[0].querySelector('.hi-entry__more'), 'Delete…');
    await t.answerConfirm(true);
    const banner = t.$('hi-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't delete");
    assert.equal(t.rows().length, 4, 'nothing disappears that was not deleted');
    noNode(banner.querySelector('.banner__details'), 'the generic http_error code adds nothing to the message');
  });

  it('[HI-07][HI-10] a failed delete’s Details read the server’s message and its code, never its JSON envelope; the History banner keeps its actions under the message', async () => {
    t = await historyEnv({ server: mixed(), routes: { 'DELETE /api/history/:id': () => json(500, { error: { message: 'database is locked', code: 'db_error' } }) } });
    await t.menu(t.rows()[0].querySelector('.hi-entry__more'), 'Delete…');
    await t.answerConfirm(true);
    const banner = t.$('hi-banner').querySelector('.banner');
    assert.equal(banner.querySelector('.banner__msg').textContent, 'database is locked');
    assert.equal(banner.querySelector('.banner__details pre').textContent, 'database is locked (db_error)');
    assert.ok(banner.classList.contains('banner--stacked'));
    assert.doesNotMatch(banner.textContent, /[{}]/);
  });

  it('[HI-07] a row’s menu is written right after its More actions button, in the row; Tab from the open menu goes on from the button', async () => {
    t = await historyEnv({ server: mixed() });
    const row = t.rows()[1];
    const more = row.querySelector('.hi-entry__more');
    const menu = document.getElementById(more.getAttribute('aria-controls'));
    sameNode(menu.previousElementSibling, more);
    assert.ok(row.contains(menu));
    more.focus();
    t.env.click(more);
    await t.settle();
    assert.equal(menu.hidden, false);
    assert.equal(document.activeElement.textContent.trim(), 'Delete…');
    t.env.key('Tab');
    await t.settle();
    assert.equal(menu.hidden, true);
    sameNode(document.activeElement, more);
    sameNode(menu.previousElementSibling, more);
  });
});

describe('clear all', () => {
  it('[HI-08][TE-09] Clear all counts every entry, confirms, DELETEs /api/history, reloads from the first page and tells other views', async () => {
    t = await historyEnv({ server: historyServer(many(120)) });
    await t.click(t.$('hi-more'));
    let changed = 0;
    t.store.on('history:changed', () => { changed += 1; });
    await t.click(t.$('hi-clear'));
    const shown = t.confirmText();
    assert.equal(shown.title, 'Delete all 120 history entries?');
    assert.equal(shown.message, 'This can’t be undone.');
    await t.answerConfirm(true);
    assert.equal(t.calls('DELETE /api/history').length, 1);
    assert.deepEqual(lastQuery(), { limit: '50', offset: '0' }, 'back to the first page');
    assert.equal(t.rows().length, 0);
    assert.equal(t.$('hi-list').querySelector('.empty__title').textContent, 'No history yet');
    assert.equal(changed, 1);
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Deleted 120 history entries'));
    assert.equal(t.$('hi-clear-wrap').hidden, true, 'nothing left to clear');
    sameNode(document.activeElement, t.$('hi-title'));
  });

  it('[REV-30] the count comes from an unfiltered request made when Clear all is pressed, not from the filtered total shown', async () => {
    t = await historyEnv({ server: historyServer([...many(7), ...many(3, transcript)]) });
    await t.click(t.$('hi-filter').querySelector('[data-type="stt"]'));
    t.env.input(t.$('hi-search'), 'meeting 2');
    await t.settle(300);
    assert.equal(t.$('hi-count').textContent, '1 entry');
    await t.click(t.$('hi-clear'));
    const count = t.calls('GET /api/history').filter((c) => c.query.get('limit') === '1').at(-1);
    assert.equal(count.query.get('type'), null);
    assert.equal(count.query.get('search'), null);
    assert.equal(t.confirmText().title, 'Delete all 10 history entries?');
    await t.answerConfirm(false);
    assert.equal(t.calls('DELETE /api/history').length, 0, 'cancel deletes nothing');
  });

  it('[REV-30] when the count request fails, the confirm leaves the number out', async () => {
    const server = historyServer(many(3));
    const base = server.routes['GET /api/history'];
    t = await historyEnv({ server, routes: { 'GET /api/history': (req) => (req.query.get('limit') === '1' ? json(500, { error: { message: 'busy', code: 'http_error' } }) : base(req)) } });
    await t.click(t.$('hi-clear'));
    assert.equal(t.confirmText().title, 'Delete all history entries?');
    await t.answerConfirm(true);
    assert.equal(t.calls('DELETE /api/history').length, 1);
  });

  it('[HI-08] Clear all history is the only clear action, set apart after the list', async () => {
    t = await historyEnv({ server: mixed() });
    const wrap = t.$('hi-clear-wrap');
    assert.equal(wrap.hidden, false);
    const clear = t.$('hi-clear');
    assert.ok(clear.classList.contains('btn--danger-ghost'));
    assert.equal(clear.textContent.trim(), 'Clear all history…');
    const order = [...t.section.children].map((el) => el.id || el.className);
    assert.ok(order.indexOf('hi-clear-wrap') > order.indexOf('hi-foot'), 'after the list and Load more');
  });
});

describe('states', () => {
  it('[HI-06][REV-06] an empty history says "No history yet" with the OS_HISTORY_ENABLED hint; no "history disabled" banner is ever guessed', async () => {
    t = await historyEnv();
    const empty = t.$('hi-list').querySelector('.empty');
    assert.equal(empty.querySelector('.empty__title').textContent, 'No history yet');
    assert.match(empty.textContent, /OS_HISTORY_ENABLED/);
    assert.equal(t.$('hi-banner').hidden, true);
    assert.equal(t.$('hi-count').textContent, '');
    assert.equal(t.$('hi-clear-wrap').hidden, true);
  });

  it('[HI-06] a search with no match says "No history matches “x”." and offers Clear search', async () => {
    t = await historyEnv({ server: mixed() });
    t.env.input(t.$('hi-search'), 'unicorn');
    await t.settle(300);
    await t.settle();
    const empty = t.$('hi-list').querySelector('.empty');
    assert.equal(empty.querySelector('.empty__title').textContent, 'No history matches “unicorn”.');
    assert.equal(document.getElementById('app-sr-status').textContent, 'No history matches “unicorn”.', 'announced');
    await t.click(empty.querySelector('[data-action="clear-search"]'));
    assert.equal(t.$('hi-search').value, '');
    assert.equal(t.rows().length, 4);
    sameNode(document.activeElement, t.$('hi-search'));
  });

  it('[HI-10] a failed filter or search shows the error banner with Retry instead of the old list; Retry loads it', async () => {
    const server = mixed();
    const base = server.routes['GET /api/history'];
    let fail = false;
    t = await historyEnv({ server, routes: { 'GET /api/history': (req) => (fail ? json(500, { error: { message: 'database is locked', code: 'http_error' } }) : base(req)) } });
    fail = true;
    await t.click(t.$('hi-filter').querySelector('[data-type="tts"]'));
    const banner = t.$('hi-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'alert');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't load history");
    assert.equal(t.rows().length, 0, 'the All list is not shown under the Speech chip');
    fail = false;
    const retry = [...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry');
    assert.equal(retry.getAttribute('data-action'), 'retry', 'run by the view’s own action map');
    await t.click(retry);
    assert.equal(banner.hidden, true);
    assert.deepEqual(t.rows().map((r) => r.getAttribute('data-id')), ['h1', 'h2', 'h3']);
    fail = true;
    t.env.input(t.$('hi-search'), 'x');
    await t.settle(300);
    assert.equal(banner.hidden, false, 'a failed search is reported too');
  });

  it('[HI-10] a failed Load more keeps the loaded rows and says so next to the button, with Retry', async () => {
    const server = historyServer(many(60));
    const base = server.routes['GET /api/history'];
    t = await historyEnv({ server, routes: { 'GET /api/history': (req) => (req.query.get('offset') === '50' ? json(503, { error: { message: 'Busy', code: 'http_error' } }) : base(req)) } });
    await t.click(t.$('hi-more'));
    assert.equal(t.rows().length, 50, 'what was loaded stays');
    const error = t.$('hi-more-error');
    assert.equal(error.hidden, false);
    assert.equal(error.querySelector('.banner__title').textContent, "Couldn't load more history");
    assert.equal(t.$('hi-more').hidden, false, 'Load more can be tried again');
  });

  it('[HI-10] an API-key refusal says "API key needed" with Open Settings', async () => {
    t = await historyEnv({ server: mixed(), routes: { 'GET /api/history': () => json(401, { error: { message: 'Invalid API key' } }) } });
    const banner = t.$('hi-banner');
    assert.equal(banner.querySelector('.banner__title').textContent, 'API key needed');
    assert.ok([...banner.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Open Settings'));
  });

  it('[HI-10] a load that fails on its own (on show) is said politely (role=status); a failure after the user’s action (Retry, a filter) is an alert', async () => {
    const server = mixed();
    const base = server.routes['GET /api/history'];
    let fail = true;
    t = await historyEnv({ server, routes: { 'GET /api/history': (req) => (fail ? json(503, { error: { message: 'Busy', code: 'http_error' } }) : base(req)) } });
    const banner = t.$('hi-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'status', 'the shell’s connection banner may already be speaking');
    await t.click([...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry'));
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'alert', 'the user asked for it');
    await t.click(t.$('hi-filter').querySelector('[data-type="stt"]'));
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'alert');
    await t.go('studio');
    t.store.emit('history:changed');
    await t.go('history');
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'status', 'a reload the view starts itself is polite again');
    fail = false;
    await t.click([...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry'));
    assert.equal(banner.hidden, true);
  });

  it('[HI-01] while the first page loads the list shows flat placeholders inside aria-busy, with a hidden "Loading history…"', async () => {
    const server = mixed();
    const base = server.routes['GET /api/history'];
    t = await historyEnv({ server, routes: { 'GET /api/history': (req) => delay(1000, () => base(req)) } });
    const list = t.$('hi-list');
    assert.equal(list.getAttribute('aria-busy'), 'true');
    assert.equal(list.querySelector('.hi-loading').getAttribute('aria-busy'), 'true');
    assert.equal(list.querySelector('.visually-hidden').textContent, 'Loading history…');
    await t.settle(1100);
    assert.equal(list.getAttribute('aria-busy'), 'false');
    assert.equal(t.rows().length, 4);
  });
});

describe('the view', () => {
  it('[HI-02] "/" focuses the search field (scope: History; single-key, so the shortcuts setting can turn it off)', async () => {
    t = await historyEnv({ server: mixed() });
    const shortcuts = await load('core/shortcuts.js');
    const entry = shortcuts.list().find((s) => s.id === 'history-search');
    assert.deepEqual([entry.scope, entry.singleKey], ['view:history', true]);
    document.body.focus();
    const event = t.env.key('/', {}, document.body);
    sameNode(document.activeElement, t.$('hi-search'));
    assert.equal(event.defaultPrevented, true, 'the slash is not typed into the field');
    assert.equal(t.$('hi-search').getAttribute('aria-keyshortcuts'), '/');
  });

  it('[HI-01] the list reloads on show when it is older than 30 s, or after history:changed or auth:changed while hidden; a take that only plays again logs nothing and reloads nothing', async () => {
    t = await historyEnv({ server: mixed() });
    const count = () => historyCalls().length;
    await t.go('studio');
    await t.go('history');
    assert.equal(count(), 1, 'fresh data is reused');
    await t.go('studio');
    t.env.clock.tick(31000);
    await t.go('history');
    assert.equal(count(), 2);
    await t.go('studio');
    t.store.emit('history:changed');
    await t.go('history');
    assert.equal(count(), 3);
    await t.go('studio');
    t.store.emit('auth:changed');
    await t.go('history');
    assert.equal(count(), 4);
    await t.go('speak');
    t.app.set({ player: { id: 'take:1', owner: 'speak', kind: 'clip', playing: true } });
    await t.go('history');
    assert.equal(count(), 4, 'playing a take again adds no entry: fresh data is reused (Speak emits history:changed for a logged take)');
  });

  it('[HI-01] a take generated in the real Speak view (the server logs it, X-History) is listed when History shows again within 30 s', async () => {
    const server = mixed();
    let logged = 0;
    t = await historyEnv({
      server,
      speak: true,
      routes: {
        'POST /v1/audio/speech': (req) => {
          // The server logs the request before it answers (src/history.py through X-History: true).
          if (req.headers['x-history'] === 'true') {
            logged += 1;
            server.state.entries.unshift(speech(100 + logged, { created_at: new Date(Date.now()).toISOString(), full_text: req.json.input }));
          }
          return bytes(200, [73, 68, 51, 4, 0, 0, 0, 0], 'audio/mpeg');
        },
      },
    });
    const count = () => historyCalls().length;
    assert.equal(count(), 1);
    await t.go('speak');
    await t.viewHost.controllerFor('speak').__ctx.setupDone;
    await t.settle();
    const speak = document.getElementById('view-speak');
    t.env.change(speak.querySelector('#sp-format'), 'wav');
    t.env.input(speak.querySelector('#sp-input'), 'A take History has not seen yet.');
    t.env.click(speak.querySelector('#sp-generate'));
    await t.settle();
    await t.settle();
    assert.equal(logged, 1, 'the server logged the take');
    t.env.clock.tick(5000);
    await t.go('history');
    assert.equal(count(), 2, 'the list is 5 s old, but a logged take makes it stale');
    assert.equal(t.rows()[0].getAttribute('data-id'), 'h101');
    assert.match(t.rows()[0].querySelector('.hi-entry__text').textContent, /A take History has not seen yet\./);
  });

  it('[HI-01] history:changed while History shows reloads it at once; its own deletes do not reload twice', async () => {
    t = await historyEnv({ server: mixed() });
    t.server.state.entries.unshift(speech(99, { created_at: minutesAgo(1) }));
    t.store.emit('history:changed');
    await t.settle();
    assert.equal(t.rows()[0].getAttribute('data-id'), 'h99');
    const before = historyCalls().length;
    await t.menu(t.rows()[0].querySelector('.hi-entry__more'), 'Delete…');
    await t.answerConfirm(true);
    assert.equal(historyCalls().length, before + 1, 'one reload after the delete');
  });

  it('[HI-01][HI-04] history:changed while History shows reloads what is loaded: the pages added with Load more stay, the new entry joins at the top and focus stays on its row', async () => {
    t = await historyEnv({ server: historyServer(many(160)) });
    await t.click(t.$('hi-more'));
    await t.click(t.$('hi-more'));
    assert.equal(t.rows().length, 150);
    const button = t.$('hi-entry-h120-open');
    button.focus();
    const before = historyCalls().length;
    t.server.state.entries.unshift(speech(999, { created_at: minutesAgo(1) }));
    t.store.emit('history:changed');
    await t.settle();
    await t.settle();
    assert.equal(historyCalls().length, before + 1, 'one request');
    assert.deepEqual(Object.fromEntries(historyCalls().at(-1).query), { limit: '200', offset: '0' }, 'what is loaded plus room for new entries');
    const ids = t.rows().map((r) => r.getAttribute('data-id'));
    assert.equal(ids.length, 151, 'the three pages stay');
    assert.equal(ids[0], 'h999');
    assert.equal(ids.at(-1), 'h150', 'nothing past what was loaded joins');
    assert.equal(t.$('hi-count').textContent, 'Showing 151 of 161');
    assert.equal(t.$('hi-more').hidden, false);
    sameNode(document.activeElement, t.$('hi-entry-h120-open'), 'focus stays on its row');
    // The focused entry was deleted elsewhere (another tab): focus goes to the row now in its place, not the top.
    t.$('hi-entry-h130-open').focus();
    t.server.state.entries.splice(t.server.state.entries.findIndex((e) => e.id === 'h130'), 1);
    t.store.emit('history:changed');
    await t.settle();
    await t.settle();
    assert.equal(t.rows().length, 150);
    noNode(t.$('hi-entry-h130'));
    sameNode(document.activeElement, t.$('hi-entry-h131-open'));
  });

  it('[HI-01][HI-07] a refresh waits for an open row menu to close, and for a delete’s confirm to be answered', async () => {
    t = await historyEnv({ server: historyServer(many(10)) });
    const more = t.rows()[3].querySelector('.hi-entry__more');
    more.focus();
    t.env.click(more);
    await t.settle();
    const menu = document.getElementById(more.getAttribute('aria-controls'));
    assert.equal(menu.hidden, false);
    const before = historyCalls().length;
    t.server.state.entries.unshift(speech(998, { created_at: minutesAgo(1) }));
    t.store.emit('history:changed');
    await t.settle(500);
    assert.equal(historyCalls().length, before, 'no reload under an open menu');
    assert.equal(menu.isConnected, true);
    assert.equal(menu.hidden, false, 'the menu stays open');
    t.env.key('Escape');
    await t.settle();
    await t.settle();
    assert.equal(historyCalls().length, before + 1, 'the refresh runs once the menu closes');
    assert.equal(t.rows()[0].getAttribute('data-id'), 'h998');
    sameNode(document.activeElement, t.$('hi-entry-h4-more'), 'focus is back on the row’s More actions button');

    await t.menu(t.$('hi-entry-h4-more'), 'Delete…');
    assert.equal(t.confirmText().open, true);
    t.server.state.entries.unshift(speech(997, { created_at: minutesAgo(0.5) }));
    t.store.emit('history:changed');
    await t.settle(500);
    assert.equal(historyCalls().length, before + 1, 'no reload under the confirm');
    await t.answerConfirm(false);
    await t.settle();
    assert.equal(historyCalls().length, before + 2, 'the refresh runs once the confirm is answered');
    assert.equal(t.rows()[0].getAttribute('data-id'), 'h997');
    assert.equal(t.rows().length, 12);
  });

  it('[HI-10] a background refresh that fails keeps the loaded pages and says so politely; Retry reloads what is loaded', async () => {
    const server = historyServer(many(120));
    const base = server.routes['GET /api/history'];
    let fail = false;
    t = await historyEnv({ server, routes: { 'GET /api/history': (req) => (fail ? json(503, { error: { message: 'Busy', code: 'http_error' } }) : base(req)) } });
    await t.click(t.$('hi-more'));
    assert.equal(t.rows().length, 100);
    fail = true;
    t.server.state.entries.unshift(speech(999, { created_at: minutesAgo(1) }));
    t.store.emit('history:changed');
    await t.settle();
    const banner = t.$('hi-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'status', 'nobody asked for this reload');
    assert.equal(t.rows().length, 100, 'what was loaded stays');
    fail = false;
    await t.click([...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry'));
    assert.equal(banner.hidden, true);
    assert.equal(historyCalls().at(-1).query.get('limit'), '150');
    assert.equal(t.rows().length, 101, 'the pages stay, with the new entry on top');
    assert.equal(t.rows()[0].getAttribute('data-id'), 'h999');
  });

  it('[HI-05] named voices read by name once the identities are known', async () => {
    t = await historyEnv({ server: mixed(), routes: { 'GET /api/voices/identities': () => json(200, { voices: [{ id: 'n1', name: 'Narrator' }] }) } });
    await t.settle();
    const row = t.rows().find((r) => r.getAttribute('data-id') === 'h3');
    assert.match(row.querySelector('.hi-entry__meta').textContent, /^Narrator · Kokoro · 1× · WAV/);
    const [call] = t.calls('GET /api/voices/identities');
    assert.equal(call.query.get('check_availability'), 'false', 'no availability checks just for names');
  });

  it('[HI-04] the count is the view’s status region and every control has a name', async () => {
    t = await historyEnv({ server: mixed() });
    const regions = liveRegions(t.section).map((r) => `${r.el.id}:${r.role}`);
    assert.ok(regions.includes('hi-count:status'));
    assertAccessibleNames(t.section);
  });
});
