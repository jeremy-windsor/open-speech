// The Reader's pure logic (views/reader/text-logic.js): paging, code-point slicing, library search/filter/sort,
// the Continue-listening pick, meta lines, the progress payloads and the Add checks. No DOM.
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as t from '../../../src/static/js/views/reader/text-logic.js';

const doc = (over = {}) => ({
  id: 'd', title: 'Dune', author: 'Frank Herbert', source_type: 'epub', chapter_count: 22, block_count: 100, char_count: 900000,
  updated_at: '2026-01-01T00:00:00Z', progress: null, ...over,
});

describe('views/reader/text-logic: paging', () => {
  it('[RE-55] a long document is split into pages of whole chapters, at most 1,500 blocks each', () => {
    const chapters = [
      { first_block: 3, block_count: 900 },     // front matter 0–2 starts the first page
      { first_block: 903, block_count: 500 },
      { first_block: 1403, block_count: 400 },
      { first_block: 1803, block_count: 4000 }, // a huge chapter becomes even slices
      { first_block: 5803, block_count: 10 },
    ];
    const pages = t.pagesFor(5813, chapters);
    assert.deepEqual(pages[0], [0, 1403], 'front matter and the first two chapters share the first page');
    assert.deepEqual(pages[1], [1403, 1803]);
    assert.deepEqual(pages.slice(2, 5), [[1803, 3136], [3136, 4470], [4470, 5803]], 'a 4,000-block chapter is three even slices');
    assert.deepEqual(pages[5], [5803, 5813]);
    for (const [start, end] of pages) assert.ok(end - start <= t.PAGE_BLOCKS);
    assert.equal(pages.reduce((n, [a, b]) => n + (b - a), 0), 5813, 'every block is on exactly one page');
    assert.equal(t.pageOf(pages, 0), 0);
    assert.equal(t.pageOf(pages, 1403), 1);
    assert.equal(t.pageOf(pages, 3200), 3);
    assert.equal(t.pageOf(pages, 5812), 5);
    assert.deepEqual(t.pagesFor(0, []), []);
    assert.deepEqual(t.pagesFor(40, []), [[0, 40]], 'a short document is one page');
  });

  it('[RE-55] page links are named after the chapter they lead to, with "continued" / "earlier part" inside a chapter', () => {
    const chapters = [{ title: 'Arrakis', first_block: 0, block_count: 10 }, { title: 'Muad’Dib', first_block: 10, block_count: 3000 }];
    const pages = t.pagesFor(3010, chapters, 1500);
    const chapterAt = t.chapterMap(3010, chapters);
    assert.deepEqual(pages, [[0, 10], [10, 1510], [1510, 3010]]);
    // `page` is the page the link leads to.
    assert.equal(t.pageLinkLabel({ pages, page: 1, next: true, chapterAt, chapters }), 'Next: Muad’Dib');
    assert.equal(t.pageLinkLabel({ pages, page: 2, next: true, chapterAt, chapters }), 'Next: Muad’Dib, continued');
    assert.equal(t.pageLinkLabel({ pages, page: 1, next: false, chapterAt, chapters }), 'Previous: Muad’Dib, earlier part');
    assert.equal(t.pageLinkLabel({ pages, page: 0, next: false, chapterAt, chapters }), 'Previous: Arrakis');
  });

  it('[RE-52] sentence pieces slice by code point, so an emoji never shifts later sentences', () => {
    const text = 'Hi 😀 there. Second one.';
    const pieces = t.sentencePieces(text, [[0, 11], [12, 23]]);
    assert.deepEqual(pieces, [{ text: 'Hi 😀 there.', s: 0 }, { text: ' ', s: null }, { text: 'Second one.', s: 1 }]);
    assert.deepEqual(t.sentencePieces('No marks', []), [{ text: 'No marks', s: null }]);
    const slice = t.codePointSlicer('a😀b');
    assert.equal(slice(1, 2), '😀');
    assert.equal(slice(2, 3), 'b');
  });
});

describe('views/reader/text-logic: library', () => {
  it('[RL-04] search matches title and author, case-insensitive; filters and sorts are pure', () => {
    const docs = [
      doc({ id: 'a', title: 'Dune', author: 'Frank Herbert' }),
      doc({ id: 'b', title: 'Emma', author: 'Jane Austen', progress: { block_index: 10, updated_at: '2026-03-01T00:00:00Z' } }),
      doc({ id: 'c', title: 'Chapter 10', author: '', block_count: 5, progress: { block_index: 5, updated_at: '2026-02-01T00:00:00Z' } }),
      doc({ id: 'd', title: 'Chapter 9', author: '', updated_at: '2026-04-01T00:00:00Z' }),
    ];
    assert.deepEqual(t.searchDocs(docs, 'HERB').map((d) => d.id), ['a']);
    assert.deepEqual(t.searchDocs(docs, '  austen ').map((d) => d.id), ['b']);
    assert.equal(t.searchDocs(docs, '').length, 4);
    assert.deepEqual(t.filterDocs(docs, 'progress').map((d) => d.id), ['b']);
    assert.deepEqual(t.filterDocs(docs, 'new').map((d) => d.id), ['a', 'd']);
    assert.deepEqual(t.filterDocs(docs, 'done').map((d) => d.id), ['c'], 'finished means block_index >= block_count');
    assert.deepEqual(t.sortDocs(docs, 'recent').map((d) => d.id), ['d', 'b', 'c', 'a'], 'progress.updated_at, else updated_at, newest first');
    assert.deepEqual(t.sortDocs(docs, 'title').map((d) => d.title), ['Chapter 9', 'Chapter 10', 'Dune', 'Emma'], 'numeric-aware title order');
    assert.deepEqual(t.sortDocs(docs, 'progress').map((d) => d.id), ['c', 'b', 'd', 'a'], 'highest percent first, "New" last');
    assert.equal(t.countLabel({ shown: 1, total: 4, query: 'du' }), '1 match of 4');
    assert.equal(t.countLabel({ shown: 3, total: 4, query: 'a' }), '3 matches of 4');
    assert.equal(t.countLabel({ shown: 4, total: 4 }), '4 documents');
    assert.equal(t.countLabel({ shown: 1, total: 4, filter: 'done' }), '1 of 4 documents');
    assert.equal(t.noMatchCopy(' zz '), 'No titles match “zz”.');
    assert.equal(t.filteredEmptyCopy('progress'), 'Nothing in progress yet.');
  });

  it('[RL-05] the Continue-listening document is the one with the latest progress.updated_at; its meta has no "left" once finished', () => {
    const docs = [
      doc({ id: 'old', progress: { block_index: 1, updated_at: '2026-01-02T00:00:00Z' } }),
      doc({ id: 'new', progress: { block_index: 34, speed: 1, updated_at: '2026-05-02T00:00:00Z' } }),
      doc({ id: 'none', updated_at: '2026-09-01T00:00:00Z' }),
    ];
    assert.equal(t.heroDoc(docs).id, 'new');
    assert.equal(t.heroDoc([doc()]), null);
    assert.match(t.heroMeta(docs[1]), /^Frank Herbert · 34% listened · ~\d+ h \d\d m left$/);
    const finished = doc({ progress: { block_index: 100, updated_at: '2026-05-02T00:00:00Z' } });
    assert.equal(t.heroMeta(finished), 'Frank Herbert · 100% listened');
  });

  it('[RL-05] the card follows the list: none while searching or under Not started / Finished, an unfinished one under In progress', () => {
    const docs = [
      doc({ id: 'done', progress: { block_index: 100, updated_at: '2026-06-01T00:00:00Z' } }),
      doc({ id: 'mid', progress: { block_index: 30, updated_at: '2026-05-01T00:00:00Z' } }),
      doc({ id: 'new' }),
    ];
    assert.equal(t.heroFor(docs).id, 'done', 'All: the latest progress, finished or not');
    assert.equal(t.heroFor(docs, { filter: 'progress' }).id, 'mid');
    assert.equal(t.heroFor(docs, { filter: 'new' }), null);
    assert.equal(t.heroFor(docs, { filter: 'done' }), null);
    assert.equal(t.heroFor(docs, { query: 'dune' }), null);
    assert.equal(t.heroFor(docs, { query: '  ' }).id, 'done', 'a blank query is no search');
    assert.equal(t.heroFor([docs[0]], { filter: 'progress' }), null);
  });

  it('[RL-10] a library answer keeps what changed here after its request went out (a finished mark, a delete)', () => {
    const shown = [
      doc({ id: 'a', progress: { block_index: 100, updated_at: '2026-06-02T00:00:00Z' } }),   // marked finished here
      doc({ id: 'b', progress: { block_index: 5, updated_at: '2026-06-01T00:00:00Z' } }),
    ];
    const answer = [
      doc({ id: 'a', progress: { block_index: 20, updated_at: '2026-06-01T00:00:00Z' } }),    // read before the PUT
      doc({ id: 'b', progress: { block_index: 6, updated_at: '2026-06-01T01:00:00Z' } }),     // moved elsewhere
      doc({ id: 'c' }),                                                                      // deleted here meanwhile
      doc({ id: 'd', title: 'Added elsewhere' }),
    ];
    const changed = new Map([['a', 1000]]);
    const removed = new Map([['c', 1005]]);
    const merged = t.mergeLibrary(answer, shown, { changed, removed, since: 1000 });
    assert.deepEqual(merged.map((d) => d.id), ['a', 'b', 'd']);
    assert.equal(merged[0].progress.block_index, 100, 'the change made after the request went out stays');
    assert.equal(merged[1].progress.block_index, 6, 'everything else is the server answer');
    const later = t.mergeLibrary(answer, shown, { changed, removed, since: 2000 });
    assert.deepEqual(later.map((d) => d.id), ['a', 'b', 'c', 'd'], 'a request sent after the changes already has them');
    assert.equal(later[0].progress.block_index, 20);
    assert.deepEqual(t.mergeLibrary(answer, [], {}).map((d) => d.id), ['a', 'b', 'c', 'd']);
  });

  it('[REV-07] nothing in the library meta names a chapter position (the summary has no current chapter)', () => {
    const d = doc({ progress: { block_index: 50, updated_at: '2026-05-02T00:00:00Z' } });
    for (const line of [t.heroMeta(d), t.docMeta(d), t.timeLeft(d)]) assert.doesNotMatch(line, /\bCh\.|chapter \d+ of/i);
    assert.equal(t.docMeta(d), 'Frank Herbert · 22 chapters · ~16 h 40 m');
    assert.equal(t.docMeta(doc({ chapter_count: 1, block_count: 3, author: '', char_count: 300 })), '3 paragraphs · under a minute');
  });

  it('[RL-06] source badges name the import type; percentOf is null for a document never started ("New")', () => {
    assert.deepEqual(t.sourceBadge('epub'), { label: 'Book', icon: 'book' });
    assert.equal(t.sourceBadge('markdown').label, 'Markdown');
    assert.equal(t.sourceBadge('html').label, 'HTML');
    assert.equal(t.sourceBadge('text').label, 'Text');
    assert.equal(t.sourceBadge('weird').label, 'Text');
    // A document the AI made (Ask AI, a book job) has a badge of its own.
    assert.deepEqual(t.sourceBadge('assistant'), { label: 'AI', icon: 'wand' });
    assert.equal(t.percentOf(doc()), null);
    assert.equal(t.percentOf(doc({ progress: { block_index: 34 } })), 34);
    assert.equal(t.percentOf(doc({ progress: { block_index: 100 } })), 100, 'finished documents show 100%');
    assert.equal(t.statusOf(doc({ progress: { block_index: 100 } })), 'done');
  });

  it('[REV-03] Start over and Mark as finished keep the saved model, voice and speed (the server would reset them)', () => {
    const d = doc({ progress: { block_index: 40, offset_ms: 1200, model: 'piper/en_US-amy-medium', voice: 'amy', speed: 1.5 } });
    assert.deepEqual(t.progressPayload(d, 0), { block_index: 0, offset_ms: 0, model: 'piper/en_US-amy-medium', voice: 'amy', speed: 1.5 });
    assert.deepEqual(t.progressPayload(d, 100), { block_index: 100, offset_ms: 0, model: 'piper/en_US-amy-medium', voice: 'amy', speed: 1.5 });
    assert.deepEqual(t.progressPayload(doc(), 0), { block_index: 0, offset_ms: 0, model: null, voice: null, speed: 1 }, 'never started: the server defaults');
  });

  it('[RL-02] the library error copy depends on the status, and an auth refusal asks for the key', () => {
    // READER_ENABLED=false mounts no Reader routes: a turned-off Reader answers 404 too, so the 404 says both.
    assert.deepEqual(t.libraryErrorCopy({ status: 404, message: 'Not Found' }), { title: 'Reader unavailable', text: "The Reader isn't available on this server. It may be turned off (READER_ENABLED) or need a newer Open Speech.", detail: '', action: 'retry' });
    // The server's message goes in Details, never glued to the plain sentence.
    assert.deepEqual(t.libraryErrorCopy({ status: 503, message: 'Reader is disabled.' }), { title: 'Reader unavailable', text: 'The Reader is turned off or still starting. Try again in a moment.', detail: 'Reader is disabled.', action: 'retry' });
    assert.deepEqual(t.libraryErrorCopy({ status: 500, message: 'boom' }), { title: 'Reader unavailable', text: "Couldn't load the library. Try again in a moment.", detail: 'boom', action: 'retry' });
    assert.equal(t.libraryErrorCopy({ status: 500, message: 'boom', detailText: 'boom (code: db_locked)' }).detail, 'boom (code: db_locked)', 'the readable detail when there is one');
    assert.equal(t.libraryErrorCopy({ kind: 'network', message: 'Failed to fetch' }).text, "Couldn't reach the server. Check that Open Speech is running.");
    assert.equal(t.libraryErrorCopy({ kind: 'network', message: 'Failed to fetch' }).detail, '');
    assert.equal(t.libraryErrorCopy({ kind: 'timeout' }).text, "The server didn't answer. It may be busy; try again.");
    // The shell's connection banner carries the refusal (and a lockout's countdown): the state never repeats it.
    const auth = { title: 'API key needed', text: 'Your library shows here once the server accepts the key.', detail: '', action: 'open-settings' };
    assert.deepEqual(t.libraryErrorCopy({ kind: 'auth', status: 401, message: 'This server needs an API key.' }), auth);
    assert.deepEqual(t.libraryErrorCopy({ kind: 'locked', status: 429, message: 'Too many wrong API keys. Try again in 52 s.' }), auth);
  });

  it('[RL-12] the Add checks: a file or pasted text is required, only the Reader types are accepted', () => {
    assert.deepEqual(t.addRequest({ mode: 'file' }), { error: { field: 'file', message: 'Choose a file first' } });
    assert.deepEqual(t.addRequest({ mode: 'paste', text: '   ' }), { error: { field: 'paste', message: 'Paste some text first' } });
    assert.deepEqual(t.addRequest({ mode: 'file', file: { name: 'x.pdf' } }).error.message, 'Choose an EPUB, TXT, Markdown or HTML file');
    for (const name of ['a.epub', 'b.TXT', 'c.md', 'd.markdown', 'e.html', 'f.htm']) assert.equal(t.acceptedFile(name), true, name);
    assert.deepEqual(t.addRequest({ mode: 'paste', text: ' Hello ', title: ' ' }), { kind: 'json', json: { type: 'text', text: 'Hello' } });
    assert.deepEqual(t.addRequest({ mode: 'paste', text: 'Hello', title: 'Notes' }).json, { type: 'text', text: 'Hello', title: 'Notes' });
    assert.deepEqual(t.FILE_TYPES, ['.epub', '.txt', '.md', '.markdown', '.html', '.htm']);
    assert.equal(t.fileHint({ size: 1400000 }), '1.4 MB · Choose another file');
  });
});

describe('views/reader/text-logic: reading', () => {
  it('[RE-54] the follow scroll: in the band nothing moves; a far jump or reduced motion is instant, a near one smooth', () => {
    const band = { bandTop: 60, bandBottom: 700, viewport: 800 };
    assert.equal(t.scrollPlan({ top: 100, bottom: 140, height: 40 }, band), null);
    const near = t.scrollPlan({ top: 900, bottom: 940, height: 40 }, band);
    assert.equal(near.behavior, 'smooth');
    assert.equal(near.delta, 900 - (60 + (640 - 40) / 2));
    assert.equal(t.scrollPlan({ top: 900, bottom: 940, height: 40 }, band, { reduced: true }).behavior, 'auto');
    assert.equal(t.scrollPlan({ top: 5000, bottom: 5040, height: 40 }, band).behavior, 'auto', 'more than 3 screens away');
    assert.ok(t.scrollPlan({ top: 100, bottom: 140, height: 40 }, band, { force: true }), 'force centres it even inside the band');
    const tall = t.scrollPlan({ top: 900, bottom: 2000, height: 1100 }, band);
    assert.equal(tall.delta, 900 - 60, 'a block taller than the band is aligned to its top');
  });

  it('[RE-62] the reading percent agrees with the library: finished is 100%', () => {
    assert.equal(t.readingPercent({ cursor: 12, count: 58 }), 21);
    assert.equal(t.readingPercent({ cursor: 57, count: 58, finished: true }), 100);
    assert.equal(t.sideMeta({ author: 'Lewis Carroll', pct: 100, finished: true }), 'Lewis Carroll · 100%');
    assert.match(t.sideMeta({ author: '', pct: 20, charCount: 11250, rate: 1.25 }), /^20% · ~8 min left$/);
  });

  it('[RE-62] one document reads the same time left in the library card, its row and the reading aside (one helper, one rounding)', () => {
    // The seeded Alice: 20% heard at 1.25×; the old aside counted the characters after the cursor and said
    // "~14 min" while the library said "~15 min".
    const doc = { author: 'Lewis Carroll', block_count: 58, char_count: 21600, progress: { block_index: 12, speed: 1.25, updated_at: '2026-10-21T09:00:00Z' } };
    const pct = t.percentOf(doc);
    assert.equal(pct, t.readingPercent({ cursor: 12, count: 58 }), 'the same percent');
    const library = t.timeLeft(doc);
    assert.equal(library, t.leftText(21600, pct, 1.25));
    assert.match(t.heroMeta(doc), new RegExp(`· ${library}$`));
    assert.equal(t.sideMeta({ author: doc.author, pct, charCount: doc.char_count, rate: 1.25 }), `Lewis Carroll · ${pct}% · ${library}`);
    assert.equal(t.leftText(0, 0), 'under a minute left');
    assert.equal(t.leftText(21600, 150, 1), 'under a minute left', 'the percent is clamped');
  });

  it('text sizes: 19 px below 960 px and 20 px above by default, steps 15–28 px', () => {
    assert.deepEqual(t.SIZES_PX, [15, 17, 19, 20, 22, 25, 28]);
    assert.equal(t.sizeLabel(t.sizeStep(null, false)), '19 px');
    assert.equal(t.sizeLabel(t.sizeStep(null, true)), '20 px');
    assert.equal(t.sizeLabel(t.sizeStep(6, false)), '28 px');
    assert.equal(t.sizeStep(9, false), 2, 'an invalid step falls back to the default');
  });
});
