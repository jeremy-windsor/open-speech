// The Reader library: loading, the states, search, filters and sort, the Continue-listening card,
// the rows and their menu (Open, Start over, Mark as finished, Delete), and opening a document.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, makeDoc, progressAt, readerRoutes, $, textOf, settle, go, json, load } from './helpers.js';
import { delay } from '../harness/fakes.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let ctx;
afterEach(() => ctx?.env.restore());

const rows = () => Array.from(document.querySelectorAll('#rd-list .rd-row'));
const rowTitles = () => rows().map((li) => textOf(li.querySelector('.rd-row__title')));
const rowFor = (id) => rows().find((li) => li.getAttribute('data-doc') === id);
const toasts = () => textOf($('app-toasts'));

async function confirmDialog(ok = true) {
  await settle(ctx.env);
  const dialog = $('app-confirm');
  assert.equal(dialog.open, true, 'a confirm is open');
  const message = textOf($('app-confirm-msg'));
  ctx.env.click($(ok ? 'app-confirm-ok' : 'app-confirm-cancel'));
  await settle(ctx.env);
  return message;
}

async function openMenu(id) {
  ctx.env.click(rowFor(id).querySelector('[data-action="row-menu"]'));
  await settle(ctx.env);
  const menu = $('rd-row-menu');
  assert.equal(menu.hidden, false, 'the row menu is open');
  return menu;
}

describe('views/reader/library', () => {
  it('[RL-01] the library is GET /v1/reader/documents; the first load shows three skeleton rows with aria-busy', async () => {
    const docs = [makeDoc()];
    ctx = await readerEnv({ docs, routes: { 'GET /v1/reader/documents': () => delay(400, json(200, { documents: docs.map((d) => d.summary()) })) } });
    const list = $('rd-list');
    assert.equal(list.getAttribute('aria-busy'), 'true');
    assert.equal(list.querySelectorAll('.skeleton--row').length, 3);
    assert.match(textOf(list), /Loading library…/);
    assert.equal($('rd-tools').hidden, true, 'no search or filters before the documents are in');
    await settle(ctx.env, 400);
    assert.equal(list.hasAttribute('aria-busy'), false);
    assert.equal(rows().length, 1);
    assert.equal(ctx.env.fetch.callsTo('GET /v1/reader/documents').length, 1);
  });

  it('[RL-02] errors show "Reader unavailable" with the status-specific text, the server message only in Details, no header Add, and Try again reloads', async () => {
    // A Reader turned off (READER_ENABLED=false) mounts no routes: it answers 404 like an older server.
    let answer = json(404, { detail: 'Not Found' });
    ctx = await readerEnv({ routes: { 'GET /v1/reader/documents': () => answer } });
    const empty = $('rd-empty');
    const details = () => empty.querySelector('details pre');
    assert.equal(empty.hidden, false);
    assert.ok(empty.classList.contains('empty--error'));
    assert.equal(textOf(empty.querySelector('.empty__title')), 'Reader unavailable');
    assert.equal(textOf(empty.querySelector('.empty__text')), "The Reader isn't available on this server. It may be turned off (READER_ENABLED) or need a newer Open Speech.");
    assert.equal(details(), null, 'a 404 has nothing to add');
    assert.equal($('rd-add').hidden, true, 'Add cannot work while the library cannot load');
    answer = json(503, { detail: 'Reader is disabled' });
    ctx.env.click(empty.querySelector('[data-action="retry"]'));
    await settle(ctx.env);
    assert.equal(textOf(empty.querySelector('.empty__text')), 'The Reader is turned off or still starting. Try again in a moment.');
    assert.equal(details().textContent, 'Reader is disabled', "the server's message is in Details, not glued to the text");
    assert.equal(textOf(empty.querySelector('details summary')), 'Details');
    answer = json(500, { detail: 'database is <locked>' });
    ctx.env.click(empty.querySelector('[data-action="retry"]'));
    await settle(ctx.env);
    assert.equal(textOf(empty.querySelector('.empty__text')), "Couldn't load the library. Try again in a moment.");
    assert.equal(details().textContent, 'database is <locked>', 'as text only');
    assert.equal(details().children.length, 0);
    assert.equal($('rd-add').hidden, true);
    answer = json(200, { documents: [makeDoc().summary()] });
    ctx.env.click(empty.querySelector('[data-action="retry"]'));
    await settle(ctx.env);
    assert.equal(empty.hidden, true);
    assert.equal(rows().length, 1);
    assert.equal($('rd-add').hidden, false, 'Add is back with the library');
    assert.equal(ctx.env.fetch.callsTo('GET /v1/reader/documents').length, 4, 'each Try again is one request');
  });

  it('[RL-02] a refused API key offers Open Settings instead of Try again', async () => {
    ctx = await readerEnv({ routes: { 'GET /v1/reader/documents': () => json(401, { detail: 'Invalid API key' }) } });
    const empty = $('rd-empty');
    assert.equal(textOf(empty.querySelector('.empty__title')), 'API key needed');
    assert.equal(empty.querySelector('details'), null, "the refusal is the connection banner's to give");
    assert.equal($('rd-add').hidden, true);
    ctx.env.click(empty.querySelector('[data-action="open-settings"]'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#settings');
  });

  it('[RL-03] an empty library says so and "Add your first document" opens the Add sheet', async () => {
    ctx = await readerEnv({ docs: [] });
    const empty = $('rd-empty');
    assert.equal(textOf(empty.querySelector('.empty__title')), 'Your library is empty');
    assert.equal(textOf(empty.querySelector('.empty__text')), 'Add an EPUB, Markdown, text or HTML file, or paste some text, and Open Speech will read it aloud.');
    assert.equal($('rd-hero').hidden, true);
    ctx.env.click(empty.querySelector('[data-action="add"]'));
    await settle(ctx.env);
    assert.equal($('rd-add-dialog').open, true);
  });

  it('[RL-04] search filters title and author case-insensitively after a 150 ms pause; the count follows and the card hides', async () => {
    ctx = await readerEnv({
      docs: [
        makeDoc({ id: 'a', title: 'Dune', author: 'Frank Herbert', progress: progressAt(3) }),
        makeDoc({ id: 'b', title: 'Emma', author: 'Jane Austen' }),
        makeDoc({ id: 'c', title: 'Persuasion', author: 'Jane Austen' }),
      ],
    });
    assert.equal(textOf($('rd-count')), '3 documents');
    assert.equal($('rd-hero').hidden, false);
    ctx.env.input($('rd-search'), 'AUSTEN');
    await settle(ctx.env, 100);
    assert.equal(rows().length, 3, 'nothing re-renders while typing');
    await settle(ctx.env, 60);
    assert.deepEqual(rowTitles().sort(), ['Emma', 'Persuasion']);
    assert.equal(textOf($('rd-count')), '2 matches of 3');
    assert.equal($('rd-hero').hidden, true, 'the card hides while searching');
    ctx.env.input($('rd-search'), 'zzz');
    await settle(ctx.env, 200);
    assert.equal(textOf($('rd-list')), 'No titles match “zzz”.');
    assert.equal(textOf($('rd-count')), '0 matches of 3');
    ctx.env.input($('rd-search'), '');
    $('rd-search').dispatchEvent(new Event('search', { bubbles: true }));
    await settle(ctx.env);
    assert.equal(rows().length, 3, 'clearing the field shows everything at once');
  });

  it('[RL-04] filter chips and sort apply at once and are kept in prefs; a filter with nothing says so', async () => {
    ctx = await readerEnv({
      docs: [
        makeDoc({ id: 'a', title: 'Zed', progress: progressAt(3, { updatedAt: '2026-10-01T00:00:00Z' }) }),
        makeDoc({ id: 'b', title: 'Alpha', updatedAt: '2026-10-05T00:00:00Z' }),
        makeDoc({ id: 'c', title: 'Mid', progress: progressAt(15, { updatedAt: '2026-10-02T00:00:00Z' }) }),
      ],
    });
    const prefs = await load('core/prefs.js');
    const chip = (f) => $('rd-filter').querySelector(`[data-filter="${f}"]`);
    ctx.env.click(chip('progress'));
    await settle(ctx.env);
    assert.equal(chip('progress').getAttribute('aria-pressed'), 'true');
    assert.equal(chip('all').getAttribute('aria-pressed'), 'false');
    assert.deepEqual(rowTitles(), ['Zed']);
    assert.equal(textOf($('rd-count')), '1 of 3 documents');
    ctx.env.click(chip('done'));
    await settle(ctx.env);
    assert.deepEqual(rowTitles(), ['Mid'], 'finished = block_index >= block_count');
    ctx.env.click(chip('new'));
    await settle(ctx.env);
    assert.deepEqual(rowTitles(), ['Alpha']);
    assert.equal($('rd-hero').hidden, true, 'no Continue-listening card under "Not started"');
    ctx.env.click(chip('all'));
    ctx.env.change($('rd-sort'), 'title');
    await settle(ctx.env);
    assert.deepEqual(rowTitles(), ['Alpha', 'Mid', 'Zed']);
    assert.equal(prefs.get().librarySort, 'title');
    assert.equal(prefs.get().libraryFilter, 'all');
    ctx.env.click(chip('progress'));
    const stored = JSON.parse(localStorage.getItem('open-speech-prefs'));
    assert.equal(stored.libraryFilter, 'progress');
    assert.equal(stored.librarySort, 'title');
  });

  it('[BUG-21] the card title is a focusable button that opens without playing; typing never re-renders per keystroke', async () => {
    ctx = await readerEnv();
    const open = $('rd-hero').querySelector('.rd-hero__open');
    assert.equal(open.localName, 'button');
    assert.equal(open.closest('h2').className, 'rd-hero__title');
    open.focus();
    assert.equal(document.activeElement, open, 'keyboard users can reach the title');
    ctx.env.click(open);
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader/alice');
    assert.deepEqual(ctx.engine.calls.filter((c) => c[0] === 'play' || c[0] === 'unlock'), [], 'opening from the title never plays');
    await go(ctx, 'reader');
    const before = rows()[0];
    ctx.env.input($('rd-search'), 'a');
    ctx.env.input($('rd-search'), 'al');
    ctx.env.input($('rd-search'), 'ali');
    await settle(ctx.env, 149);
    assert.equal(rows()[0], before, 'still the same row element: no render yet');
    await settle(ctx.env, 2);
    assert.notEqual(rows()[0], before, 'one render after the pause');
  });

  it('[RL-05] the Continue-listening card is the latest progress; Resume unlocks inside the click, opens and plays', async () => {
    ctx = await readerEnv({
      docs: [
        makeDoc({ id: 'old', title: 'Old book', progress: progressAt(2, { updatedAt: '2026-10-01T00:00:00Z' }) }),
        makeDoc({ id: 'alice', progress: progressAt(12, { updatedAt: '2026-10-21T00:00:00Z' }) }),
      ],
    });
    const hero = $('rd-hero');
    assert.equal(textOf(hero.querySelector('.rd-hero__title')), "Alice's Adventures in Wonderland");
    assert.match(textOf(hero.querySelector('.rd-hero__meta')), /^Lewis Carroll · 80% listened · .+ left$/);
    const bar = hero.querySelector('progress');
    assert.equal(bar.getAttribute('value'), '80');
    assert.equal(bar.getAttribute('max'), '100');
    assert.equal(bar.getAttribute('aria-label'), '80% listened');
    const resume = hero.querySelector('[data-action="resume"]');
    assert.equal(resume.getAttribute('aria-label'), "Resume Alice's Adventures in Wonderland");
    ctx.env.click(resume);
    assert.deepEqual(ctx.engine.calls[0], ['unlock'], 'audio is unlocked synchronously inside the click (iOS)');
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader/alice');
    assert.deepEqual(ctx.engine.calls.map((c) => c[0]), ['unlock', 'load', 'play']);
  });

  it('[REV-07] the card shows no chapter position: the library summary does not carry one', async () => {
    ctx = await readerEnv();
    assert.doesNotMatch(textOf($('rd-hero')), /\bCh\.|Chapter \d+ of/);
  });

  it('[RL-06] a row: tile, title, source badge, author · chapters|paragraphs · length, a bar and the percent or "New"', async () => {
    ctx = await readerEnv({
      docs: [
        makeDoc({ id: 'book', title: 'Dune', author: 'Frank Herbert', sourceType: 'epub', chapters: 3, progress: progressAt(4) }),
        makeDoc({ id: 'note', title: 'Notes', author: '', sourceType: 'markdown', chapters: 1, perChapter: 3 }),
        makeDoc({ id: 'page', title: 'A page', sourceType: 'html', chapters: 1, perChapter: 2 }),
      ],
    });
    const book = rowFor('book');
    assert.equal(textOf(book.querySelector('.tile')), 'DU');
    assert.equal(book.querySelector('.tile').getAttribute('aria-hidden'), 'true');
    assert.equal(textOf(book.querySelector('.rd-badge')), 'Book');
    assert.match(textOf(book.querySelector('.rd-row__facts')), /^Frank Herbert · 3 chapters · /);
    assert.equal(book.querySelector('progress').getAttribute('value'), '27');
    assert.equal(textOf(book.querySelector('.rd-row__pct')), '27%');
    const main = book.querySelector('.row__main');
    assert.equal(main.getAttribute('data-action'), 'open');
    assert.equal(main.getAttribute('data-doc'), 'book');
    const more = book.querySelector('[data-action="row-menu"]');
    assert.equal(more.getAttribute('aria-label'), 'More for Dune');
    assert.equal(more.closest('.row__main'), null, 'the menu button is a sibling of the main button, never inside it');
    const note = rowFor('note');
    assert.equal(textOf(note.querySelector('.rd-badge')), 'Markdown');
    assert.match(textOf(note.querySelector('.rd-row__facts')), /^3 paragraphs · /);
    // The count and the length each keep to one line ("under a / minute" wrapped on phones), and a dot is tied
    // to the word before it, so it never starts a line.
    const facts = book.querySelector('.rd-row__facts');
    const [count, length] = Array.from(facts.querySelectorAll('.rd-row__fact')).map((el) => el.textContent);
    assert.equal(facts.querySelectorAll('.rd-row__fact').length, 2, 'the author may wrap; the count and the length may not');
    assert.equal(count, '3 chapters');
    assert.match(length, /^(~\d+ min|~\d+ h \d\d m|under a minute)$/);
    assert.equal(facts.textContent, `Frank Herbert · 3 chapters · ${length}`);
    assert.equal(textOf(note.querySelector('.rd-row__pct')), 'New');
    assert.equal(note.querySelector('progress'), null);
    assert.equal(textOf(rowFor('page').querySelector('.rd-badge')), 'HTML');
    assertAccessibleNames($('rd-library'));
  });

  it('[REV-03] Start over (after a confirm) and Mark as finished keep the saved model, voice and speed', async () => {
    const docs = [makeDoc({ id: 'alice', progress: progressAt(12, { model: 'piper/en_US-amy-medium', voice: 'amy', speed: 1.5 }) })];
    ctx = await readerEnv({ docs });
    const menu = await openMenu('alice');
    assert.equal(menu.querySelector('[data-menu="restart"]').hidden, false);
    ctx.env.click(menu.querySelector('[data-menu="restart"]'));
    const message = await confirmDialog(true);
    assert.match(message, /will start from the beginning/);
    const [put] = ctx.env.fetch.callsTo('PUT /v1/reader/documents/:id/progress');
    assert.deepEqual(put.json, { block_index: 0, offset_ms: 0, model: 'piper/en_US-amy-medium', voice: 'amy', speed: 1.5 });
    assert.equal(textOf(rowFor('alice').querySelector('.rd-row__pct')), '0%');
    assert.match(toasts(), /starts from the beginning/);
    const menu2 = await openMenu('alice');
    ctx.env.click(menu2.querySelector('[data-menu="finish"]'));
    await settle(ctx.env);
    const puts = ctx.env.fetch.callsTo('PUT /v1/reader/documents/:id/progress');
    assert.deepEqual(puts[1].json, { block_index: 15, offset_ms: 0, model: 'piper/en_US-amy-medium', voice: 'amy', speed: 1.5 });
    assert.equal(textOf(rowFor('alice').querySelector('.rd-row__pct')), 'Finished, 100%', 'a finished document shows 100%');
    const menu3 = await openMenu('alice');
    assert.equal(menu3.querySelector('[data-menu="finish"]').hidden, true, 'no "Mark as finished" once finished');
  });

  it('[REV-03] Start over on the document loaded for playback stops it first, so its next save cannot overwrite the new position', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    ctx.engine.play();
    await go(ctx, 'reader');
    const menu = await openMenu('alice');
    ctx.env.click(menu.querySelector('[data-menu="restart"]'));
    await confirmDialog(true);
    // A plain unload (Close player) keeps the document and saves its position; only {forget: true} drops it
    // without saving, as views/reader/engine.js does.
    assert.deepEqual(ctx.engine.calls.filter((c) => c[0] === 'unload'), [['unload', 'alice', { forget: true }]]);
    assert.equal(ctx.engine.state.docId, '', 'the engine forgot the document');
    assert.equal(ctx.env.fetch.callsTo('PUT /v1/reader/documents/:id/progress').length, 1, 'only the Start over PUT');
  });

  it('[REV-03] Start over and Mark as finished chosen by keyboard keep focus on the row\'s More button (WCAG 2.4.3)', async () => {
    ctx = await readerEnv();
    const more = () => rowFor('alice').querySelector('[data-action="row-menu"]');
    more().focus();
    ctx.env.key('ArrowDown');
    await settle(ctx.env);
    ctx.env.key('ArrowDown');
    assert.equal(document.activeElement.getAttribute('data-menu'), 'restart');
    ctx.env.key('Enter');
    await confirmDialog(true);
    await settle(ctx.env, 20);
    assert.equal(textOf(rowFor('alice').querySelector('.rd-row__pct')), '0%', 'the row was rendered again');
    assert.equal(document.activeElement, more(), 'focus is on the new More button of the same row');
    ctx.env.key('ArrowDown');
    await settle(ctx.env);
    ctx.env.key('ArrowDown');
    ctx.env.key('ArrowDown');
    assert.equal(document.activeElement.getAttribute('data-menu'), 'finish');
    ctx.env.key('Enter');
    await settle(ctx.env, 20);
    assert.equal(textOf(rowFor('alice').querySelector('.rd-row__pct')), 'Finished, 100%');
    assert.equal(document.activeElement, more());
  });

  it('[REV-03] when the filter drops the changed row, focus moves to its neighbour, not to the page', async () => {
    ctx = await readerEnv({
      docs: [
        makeDoc({ id: 'a', title: 'Alpha', progress: progressAt(3, { updatedAt: '2026-10-21T00:00:00Z' }) }),
        makeDoc({ id: 'b', title: 'Beta', progress: progressAt(5, { updatedAt: '2026-10-20T00:00:00Z' }) }),
      ],
    });
    ctx.env.click($('rd-filter').querySelector('[data-filter="progress"]'));
    await settle(ctx.env);
    assert.deepEqual(rowTitles(), ['Alpha', 'Beta']);
    const menu = await openMenu('a');
    ctx.env.click(menu.querySelector('[data-menu="finish"]'));
    await settle(ctx.env, 20);
    assert.deepEqual(rowTitles(), ['Beta'], 'a finished book leaves "In progress"');
    assert.equal(document.activeElement, rowFor('b').querySelector('.rd-row__main'));
    const last = await openMenu('b');
    ctx.env.click(last.querySelector('[data-menu="finish"]'));
    await settle(ctx.env, 20);
    assert.equal(textOf($('rd-list')), 'Nothing in progress yet.');
    assert.equal(document.activeElement, $('rd-filter').querySelector('[data-filter="progress"]'), 'with no row left, the active filter');
  });

  it('[RL-05] under "In progress" the card never shows a finished book the list leaves out', async () => {
    ctx = await readerEnv({
      docs: [
        makeDoc({ id: 'done', title: 'Done book', progress: progressAt(15, { updatedAt: '2026-10-22T00:00:00Z' }) }),
        makeDoc({ id: 'mid', title: 'Mid book', progress: progressAt(3, { updatedAt: '2026-10-01T00:00:00Z' }) }),
      ],
    });
    assert.equal(textOf($('rd-hero').querySelector('.rd-hero__title')), 'Done book', 'All: the latest progress, finished or not');
    ctx.env.click($('rd-filter').querySelector('[data-filter="progress"]'));
    await settle(ctx.env);
    assert.deepEqual(rowTitles(), ['Mid book']);
    assert.equal(textOf($('rd-hero').querySelector('.rd-hero__title')), 'Mid book', 'the latest unfinished one');
  });

  it('[RL-06] a saved position reported by the engine starts a "New" row and moves the card; a started row is patched in place', async () => {
    ctx = await readerEnv();
    assert.equal(textOf(rowFor('note').querySelector('.rd-row__pct')), 'New');
    const aliceRow = rowFor('alice');
    ctx.engine.emit('progress', { docId: 'alice', progress: progressAt(14, { updatedAt: '2026-10-21T09:30:00Z' }) });
    await settle(ctx.env);
    assert.equal(rowFor('alice'), aliceRow, 'same row element: patched, not re-rendered');
    assert.equal(textOf(aliceRow.querySelector('.rd-row__pct')), '93%');
    assert.match(textOf($('rd-hero').querySelector('.rd-hero__meta')), /93% listened/);
    ctx.engine.emit('progress', { docId: 'note', progress: progressAt(1, { updatedAt: '2026-10-23T00:00:00Z' }) });
    await settle(ctx.env);
    const note = rowFor('note');
    assert.equal(textOf(note.querySelector('.rd-row__pct')), '33%', 'no longer "New"');
    assert.equal(note.querySelector('progress').getAttribute('value'), '33');
    assert.equal(textOf($('rd-hero').querySelector('.rd-hero__title')), 'Notes for Tuesday', 'now the Continue-listening document');
  });

  it('[RL-10] a reload answered after Mark as finished or Delete keeps the change (its request went out before it)', async () => {
    const docs = [
      makeDoc({ progress: progressAt(3) }),
      makeDoc({ id: 'note', title: 'Notes for Tuesday', chapters: 1, perChapter: 3 }),
    ];
    const base = readerRoutes(docs);
    let slow = false;
    ctx = await readerEnv({
      docs,
      routes: {
        // The answer is read when the request arrives and delivered 400 ms later.
        'GET /v1/reader/documents': (req) => {
          const answer = base['GET /v1/reader/documents'](req);
          return slow ? delay(400, answer) : answer;
        },
      },
    });
    assert.equal(textOf(rowFor('alice').querySelector('.rd-row__pct')), '20%');
    slow = true;
    ctx.ctl.library.load();   // e.g. the 30 s staleness check
    await settle(ctx.env, 10);
    const menu = await openMenu('alice');
    ctx.env.click(menu.querySelector('[data-menu="finish"]'));
    await settle(ctx.env, 20);
    const del = await openMenu('note');
    ctx.env.click(del.querySelector('[data-menu="delete"]'));
    await confirmDialog(true);
    await settle(ctx.env, 450);
    assert.equal(ctx.env.fetch.callsTo('GET /v1/reader/documents').length, 2, 'the older reload did answer');
    assert.equal(textOf(rowFor('alice').querySelector('.rd-row__pct')), 'Finished, 100%', 'not the old 20%');
    assert.equal(rowFor('note'), undefined, 'the deleted row does not come back');
    slow = false;
    await ctx.ctl.library.load();
    await settle(ctx.env);
    assert.equal(textOf(rowFor('alice').querySelector('.rd-row__pct')), 'Finished, 100%', 'and a fresh reload agrees');
  });

  it('[RL-03] while the library is empty its own "Add your first document" is the one primary action', async () => {
    ctx = await readerEnv({ docs: [] });
    const add = $('rd-add');
    assert.equal(add.classList.contains('btn--primary'), false);
    assert.equal(add.classList.contains('btn--secondary'), true);
    const shown = (el) => { for (let n = el; n; n = n.parentElement) if (n.hidden) return false; return true; };
    const primaries = Array.from(document.querySelectorAll('#rd-library .btn--primary')).filter(shown);
    assert.deepEqual(primaries.map((b) => textOf(b)), ['Add your first document']);
    ctx.docs.push(makeDoc());
    await ctx.ctl.library.load();
    await settle(ctx.env);
    assert.equal(add.classList.contains('btn--primary'), true, 'with documents, Add is the primary action again');
    assert.equal(add.classList.contains('btn--secondary'), false);
  });

  it('[RL-10] Delete asks first, deletes, toasts, and stops playback when it is the open document', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    ctx.engine.play();
    await go(ctx, 'reader');
    let menu = await openMenu('alice');
    ctx.env.click(menu.querySelector('[data-menu="delete"]'));
    let message = await confirmDialog(false);
    assert.equal(message, "Delete “Alice's Adventures in Wonderland” from the library? Its saved position is removed too.");
    assert.equal(ctx.env.fetch.callsTo('DELETE /v1/reader/documents/:id').length, 0, 'cancel deletes nothing');
    menu = await openMenu('alice');
    ctx.env.click(menu.querySelector('[data-menu="delete"]'));
    await confirmDialog(true);
    assert.equal(ctx.env.fetch.callsTo('DELETE /v1/reader/documents/:id').length, 1);
    assert.equal(rowFor('alice'), undefined);
    assert.match(toasts(), /Deleted “Alice's Adventures in Wonderland”/);
    assert.deepEqual(ctx.engine.calls.filter((c) => c[0] === 'unload'), [['unload', 'alice', { forget: true }]],
      'the open document stops playing and is dropped without a save (a plain unload would PUT progress for it)');
    assert.equal(ctx.engine.state.docId, '');
    assert.equal(document.activeElement, rowFor('note').querySelector('.rd-row__main'), 'focus moves to the next row');
    assert.equal(ctx.ctl.state.docId, '', 'the reading surface drops it');
  });

  it('[RL-10] a failed delete toasts the reason and keeps the row', async () => {
    ctx = await readerEnv({ routes: { 'DELETE /v1/reader/documents/:id': () => json(500, { detail: 'disk full' }) } });
    const menu = await openMenu('note');
    ctx.env.click(menu.querySelector('[data-menu="delete"]'));
    await confirmDialog(true);
    assert.ok(rowFor('note'));
    assert.match(textOf($('app-toasts-alert')), /Couldn't delete/);
  });

  it('[RL-13] opening a row pushes #reader/<id>; Back to library returns to #reader', async () => {
    ctx = await readerEnv();
    const before = history.length;
    ctx.env.click(rowFor('note').querySelector('.row__main'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader/note');
    assert.equal(history.length, before + 1, 'one history entry, so the browser Back returns to the library');
    assert.equal($('rd-library').hidden, true);
    assert.equal($('rd-reading').hidden, false);
    ctx.env.click($('rd-back'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader');
    assert.equal($('rd-library').hidden, false);
  });

  it('the row menu is an APG menu: ArrowDown opens it, Escape closes it and returns focus', async () => {
    ctx = await readerEnv();
    const more = rowFor('alice').querySelector('[data-action="row-menu"]');
    more.focus();
    ctx.env.key('ArrowDown');
    await settle(ctx.env);
    const menu = $('rd-row-menu');
    assert.equal(menu.hidden, false);
    assert.equal(more.getAttribute('aria-expanded'), 'true');
    assert.equal(document.activeElement.getAttribute('data-menu'), 'open');
    ctx.env.key('Escape');
    await settle(ctx.env);
    assert.equal(menu.hidden, true);
    assert.equal(document.activeElement, more);
    ctx.env.click(more);
    await settle(ctx.env);
    ctx.env.click(menu.querySelector('[data-menu="open"]'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader/alice');
  });

  it('"/" focuses the search field while the library shows', async () => {
    ctx = await readerEnv();
    ctx.env.key('/', {}, document.body);
    assert.equal(document.activeElement, $('rd-search'));
  });
});
