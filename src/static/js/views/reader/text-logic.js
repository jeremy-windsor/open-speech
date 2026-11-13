/**
 * The Reader's pure logic: library search, filter and sort, the Continue-listening document,
 * row and hero meta, the progress payloads that keep the document's voice and speed (REV-03), paging of long
 * documents, code-point-correct sentence slices, the reading percent, the follow-scroll plan and the Add
 * checks. No DOM and no browser APIs: tests import it directly.
 * @module views/reader/text-logic
 */
import { percent, lengthLabel, plural, toMs, codePointLength, bytes } from '../../core/format.js';

/** The most blocks on screen at once: a longer document shows a few whole chapters at a time (RE-55). */
export const PAGE_BLOCKS = 1500;

/** The file types the Reader imports (RL-11); the file input's accept attribute is this list. */
export const FILE_TYPES = Object.freeze(['.epub', '.txt', '.md', '.markdown', '.html', '.htm']);

/** Reading text sizes, in px, for data-rd-size 0…6. */
export const SIZES_PX = Object.freeze([15, 17, 19, 20, 22, 25, 28]);

/** Library sort modes and filter chips, in display order (labels are the visible text). */
export const SORTS = Object.freeze([['recent', 'Recent'], ['title', 'Title'], ['progress', 'Progress']]);
export const FILTERS = Object.freeze([['all', 'All'], ['progress', 'In progress'], ['new', 'Not started'], ['done', 'Finished']]);

const SOURCE_BADGES = Object.freeze({
  epub: Object.freeze({ label: 'Book', icon: 'book' }),
  markdown: Object.freeze({ label: 'Markdown', icon: 'file' }),
  html: Object.freeze({ label: 'HTML', icon: 'file' }),
  text: Object.freeze({ label: 'Text', icon: 'paste' }),
  // A document the AI made: Ask AI or a book job saves it (source_type "assistant").
  assistant: Object.freeze({ label: 'AI', icon: 'wand' }),
});

/* ───────────────────────────── library ───────────────────────────── */

/** round(block_index / block_count × 100), clamped; null when the document was never started ("New"). */
export function percentOf(doc) {
  const p = doc && doc.progress;
  if (!p) return null;
  return percent(p.block_index, doc.block_count);
}

/** Listened to the end: the saved block is past the last one (the engine saves block_count when it finishes). */
export function isFinished(doc) {
  const p = doc && doc.progress;
  const count = Number(doc && doc.block_count) || 0;
  return !!p && count > 0 && Number(p.block_index) >= count;
}

/** 'new' (never started), 'done' (finished) or 'progress'. */
export function statusOf(doc) {
  if (!doc || !doc.progress) return 'new';
  return isFinished(doc) ? 'done' : 'progress';
}

/** The documents a filter chip keeps: 'all', 'progress', 'new' or 'done'. */
export function filterDocs(docs, filter = 'all') {
  const list = Array.isArray(docs) ? docs : [];
  if (!filter || filter === 'all') return list.slice();
  return list.filter((doc) => statusOf(doc) === filter);
}

/** Search title and author, case-insensitive (RL-04). An empty query keeps everything. */
export function searchDocs(docs, query = '') {
  const list = Array.isArray(docs) ? docs : [];
  const q = String(query || '').trim().toLocaleLowerCase();
  if (!q) return list.slice();
  return list.filter((doc) => `${doc.title || ''} ${doc.author || ''}`.toLocaleLowerCase().includes(q));
}

function recentKey(doc) {
  return toMs(doc && doc.progress && doc.progress.updated_at) || toMs(doc && doc.updated_at) || 0;
}

const collator = (() => {
  try {
    return new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
  } catch {
    return null;
  }
})();

function byTitle(a, b) {
  const x = String(a.title || '');
  const y = String(b.title || '');
  return collator ? collator.compare(x, y) : x.localeCompare(y, undefined, { numeric: true });
}

/**
 * Sort a copy: 'recent' (progress.updated_at, else updated_at, newest first), 'title' (numeric-aware) or
 * 'progress' (percent, highest first, "New" last).
 */
export function sortDocs(docs, mode = 'recent') {
  const list = Array.isArray(docs) ? docs.slice() : [];
  if (mode === 'title') return list.sort(byTitle);
  if (mode === 'progress') {
    return list.sort((a, b) => {
      const pa = percentOf(a);
      const pb = percentOf(b);
      const ka = pa === null ? -1 : pa;
      const kb = pb === null ? -1 : pb;
      return kb - ka || recentKey(b) - recentKey(a) || byTitle(a, b);
    });
  }
  return list.sort((a, b) => recentKey(b) - recentKey(a) || byTitle(a, b));
}

/** The Continue-listening document: the one with the latest progress.updated_at, or null (RL-05). */
export function heroDoc(docs) {
  let best = null;
  let bestAt = -Infinity;
  for (const doc of Array.isArray(docs) ? docs : []) {
    if (!doc || !doc.progress) continue;
    const at = toMs(doc.progress.updated_at);
    if (at > bestAt) {
      best = doc;
      bestAt = at;
    }
  }
  return best;
}

/**
 * The Continue-listening card for the list on screen (RL-05): none while searching or under Not started and
 * Finished; under In progress only an unfinished document, so the card never shows a book the list leaves out.
 */
export function heroFor(docs, { filter = 'all', query = '' } = {}) {
  if (String(query || '').trim()) return null;
  if (filter === 'progress') return heroDoc(filterDocs(docs, 'progress'));
  return !filter || filter === 'all' ? heroDoc(docs) : null;
}

/**
 * Apply a library answer without undoing what changed here while it was on its way (RL-10, REV-03). A
 * reload can be answered after a Start over, a finished mark, a position the engine saved or a delete that
 * landed once its request had gone out: those documents keep the position shown now, and a deleted one stays
 * gone. `changed` and `removed` map a document id to when its change landed; `since` is when the answer's
 * request went out (a change that landed before it is already in the answer).
 * @param {object[]} incoming the documents in the answer
 * @param {object[]} current the documents shown now
 * @param {{changed?: Map<string, number>, removed?: Map<string, number>, since?: number}} [local]
 * @returns {object[]}
 */
export function mergeLibrary(incoming, current, { changed = new Map(), removed = new Map(), since = 0 } = {}) {
  const shown = new Map((Array.isArray(current) ? current : []).map((doc) => [doc.id, doc]));
  const after = (map, id) => map.has(id) && map.get(id) >= since;
  return (Array.isArray(incoming) ? incoming : [])
    .filter((doc) => !after(removed, doc.id))
    .map((doc) => (after(changed, doc.id) && shown.has(doc.id) ? { ...doc, progress: shown.get(doc.id).progress } : doc));
}

/** The source badge: {label, icon} (Book, Markdown, HTML or Text). */
export function sourceBadge(type) {
  return SOURCE_BADGES[type] || SOURCE_BADGES.text;
}

/**
 * A row's meta parts: {author, count, length} ("Lewis Carroll", "22 chapters", "~21 h"; paragraphs when there is
 * one chapter or none). The row keeps the count and the length each on one line ("under a minute" never wraps).
 */
export function docMetaParts(doc) {
  const chapters = Number(doc.chapter_count) || 0;
  return {
    author: doc.author || '',
    count: chapters > 1 ? plural(chapters, 'chapter') : plural(Number(doc.block_count) || 0, 'paragraph'),
    length: lengthLabel(doc.char_count),
  };
}

/** A row's meta line: "Author · 22 chapters · ~21 h" (paragraphs when there is one chapter or none). */
export function docMeta(doc) {
  const { author, count, length } = docMetaParts(doc);
  return [author, count, length].filter(Boolean).join(' · ');
}

/**
 * "~5 h 02 m left": the part of a document not yet heard, from its characters and the percent heard, at a speed.
 * The library (card and rows) and the reading view's aside both use it, so one document reads the same time left
 * in both places (RE-62).
 * @param {number} charCount the document's char_count
 * @param {number} pct the percent heard (0–100)
 * @param {number} [rate] the playback speed
 */
export function leftText(charCount, pct, rate = 1) {
  const heard = Math.min(100, Math.max(0, Number(pct) || 0));
  return `${lengthLabel((Number(charCount) || 0) * (1 - heard / 100), Number(rate) || 1)} left`;
}

/**
 * The time left in a document at its saved speed: "~5 h 02 m left", '' once finished.
 * The library summary carries no chapter position (REV-07), so nothing here names a chapter.
 */
export function timeLeft(doc) {
  if (isFinished(doc)) return '';
  return leftText(doc.char_count, percentOf(doc) ?? 0, Number(doc.progress && doc.progress.speed) || 1);
}

/** The hero's meta line: "Author · 34% listened · ~5 h left" (no "left" once finished). */
export function heroMeta(doc) {
  const pct = percentOf(doc) ?? 0;
  return [doc.author || '', `${pct}% listened`, timeLeft(doc)].filter(Boolean).join(' · ');
}

/** The count line: "12 documents", "3 matches of 12" (search), "4 of 12 documents" (a filter). */
export function countLabel({ shown, total, query = '', filter = 'all' }) {
  if (String(query || '').trim()) return `${plural(shown, 'match', 'matches')} of ${total}`;
  if (filter && filter !== 'all') return `${shown} of ${plural(total, 'document')}`;
  return plural(total, 'document');
}

/** What an empty filter says. */
export function filteredEmptyCopy(filter) {
  if (filter === 'progress') return 'Nothing in progress yet.';
  if (filter === 'done') return 'Nothing finished yet.';
  if (filter === 'new') return "You've started everything in your library.";
  return 'Nothing here yet.';
}

/** The search no-match line: No titles match “x”. */
export function noMatchCopy(query) {
  return `No titles match “${String(query || '').trim()}”.`;
}

/** What the Reader's own auth states say under "API key needed": the shell's connection banner carries the
 *  refusal itself (and a lockout's countdown), so this never repeats it. */
export const AUTH_TEXT = Object.freeze({
  library: 'Your library shows here once the server accepts the key.',
  document: 'This document opens once the server accepts the key.',
});

/** What the library's error state says for each cause (RL-02). */
export const LIBRARY_ERROR_TEXT = Object.freeze({
  // READER_ENABLED=false mounts no Reader routes, so a Reader that is turned off answers 404 like an old server.
  notFound: "The Reader isn't available on this server. It may be turned off (READER_ENABLED) or need a newer Open Speech.",
  unavailable: 'The Reader is turned off or still starting. Try again in a moment.',
  network: "Couldn't reach the server. Check that Open Speech is running.",
  timeout: "The server didn't answer. It may be busy; try again.",
  other: "Couldn't load the library. Try again in a moment.",
});

/**
 * The library's error state (RL-02): {title, text, detail, action: 'retry'|'open-settings'}.
 * The text says what happened and what to do; the server's own message goes in Details (`detail`), never
 * glued into the text. A 404 is a server without the Reader or with it turned off; a 503 is one turned off or
 * still starting; an auth refusal asks for the key and leaves the refusal to the shell's banner.
 */
export function libraryErrorCopy(error) {
  const e = error || {};
  const message = String(e.message || '').trim();
  if (e.kind === 'auth' || e.kind === 'locked' || e.status === 401) {
    return { title: 'API key needed', text: AUTH_TEXT.library, detail: '', action: 'open-settings' };
  }
  const copy = (text, detail = '') => ({ title: 'Reader unavailable', text, detail, action: 'retry' });
  if (e.status === 404) return copy(LIBRARY_ERROR_TEXT.notFound);   // FastAPI's "Not Found" adds nothing
  if (e.kind === 'network') return copy(LIBRARY_ERROR_TEXT.network);
  if (e.kind === 'timeout') return copy(LIBRARY_ERROR_TEXT.timeout);
  const detail = String(e.detailText || '').trim() || message;
  return copy(e.status === 503 ? LIBRARY_ERROR_TEXT.unavailable : LIBRARY_ERROR_TEXT.other, detail);
}

/**
 * The progress body for Start over (block 0) and Mark as finished (block_count). It carries the document's
 * saved model, voice and speed: the server resets left-out fields to null and 1.0 (REV-03).
 */
export function progressPayload(doc, blockIndex) {
  const p = (doc && doc.progress) || {};
  const speed = Number(p.speed);
  return {
    block_index: Math.max(0, Math.floor(Number(blockIndex) || 0)),
    offset_ms: 0,
    model: p.model ?? null,
    voice: p.voice ?? null,
    speed: Number.isFinite(speed) && speed >= 0.25 && speed <= 4 ? speed : 1,
  };
}

/* ───────────────────────────── paging (RE-55) ───────────────────────────── */

/**
 * Split a document into pages of whole chapters, at most `max` blocks each, so a phone never lays out a whole
 * Bible at once. A longer chapter is cut into even slices; front matter before the first chapter starts the
 * first page. Returns [start, end) block ranges.
 */
export function pagesFor(count, chapters = [], max = PAGE_BLOCKS) {
  const total = Math.max(0, Math.floor(Number(count) || 0));
  if (!total) return [];
  const starts = [...new Set([0, ...(chapters || []).map((c) => Number(c.first_block))])]
    .filter((start) => Number.isInteger(start) && start >= 0 && start < total)
    .sort((a, b) => a - b);
  const pages = [];
  let page = null;
  starts.forEach((first, i) => {
    const end = i + 1 < starts.length ? starts[i + 1] : total;
    const size = end - first;
    if (size > max) {
      if (page) pages.push(page);
      page = null;
      const slices = Math.ceil(size / max);
      for (let slice = 0; slice < slices; slice += 1) {
        pages.push([first + Math.round((size * slice) / slices), first + Math.round((size * (slice + 1)) / slices)]);
      }
    } else if (page && end - page[0] <= max) {
      page[1] = end;
    } else {
      if (page) pages.push(page);
      page = [first, end];
    }
  });
  if (page) pages.push(page);
  return pages;
}

/** The page holding block `index` (binary search). */
export function pageOf(pages, index) {
  if (!pages || !pages.length) return 0;
  let low = 0;
  let high = pages.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (pages[mid][0] <= index) low = mid;
    else high = mid - 1;
  }
  return low;
}

/** Block index -> chapter position (0 for front matter or a document without chapters). */
export function chapterMap(count, chapters = []) {
  const total = Math.max(0, Math.floor(Number(count) || 0));
  const map = new Array(total).fill(0);
  (chapters || []).forEach((chapter, position) => {
    const first = Math.max(0, Number(chapter.first_block) || 0);
    const end = Math.min(total, first + Math.max(1, Number(chapter.block_count) || 0));
    for (let i = first; i < end; i += 1) map[i] = position;
  });
  return map;
}

/**
 * A page link's label, named after the chapter it leads to: "Next: The Pool of Tears, continued" /
 * "Previous: Down the Rabbit-Hole, earlier part" (the arrows are added by the caller).
 */
export function pageLinkLabel({ pages, page, next, chapterAt, chapters }) {
  const [start, end] = pages[page];
  const edge = next ? start : end - 1;
  const position = chapterAt[edge] ?? 0;
  const title = (chapters && chapters[position] && chapters[position].title) || '';
  const neighbour = next ? chapterAt[start - 1] : chapterAt[end];
  const split = neighbour === position && !!title;   // the same chapter carries on across the page edge
  if (next) return `Next: ${title || 'more text'}${split ? ', continued' : ''}`;
  return `Previous: ${title || 'earlier text'}${split ? ', earlier part' : ''}`;
}

/* ───────────────────────────── text ───────────────────────────── */

/** A slicer by code point (the server's sentence offsets count code points, so an emoji never shifts them). */
export function codePointSlicer(text) {
  const s = String(text ?? '');
  if (!/[\uD800-\uDFFF]/.test(s)) return (start, end) => s.slice(start, end);
  const points = Array.from(s);
  return (start, end) => points.slice(start, end).join('');
}

/**
 * The pieces of a block's text around its sentences: [{text, s}] where s is the sentence index, or null for
 * text between sentences. Offsets are code points (RE-52).
 */
export function sentencePieces(text, sentences) {
  const s = String(text ?? '');
  const list = Array.isArray(sentences) ? sentences : [];
  if (!list.length) return [{ text: s, s: null }];
  const slice = codePointSlicer(s);
  const out = [];
  let pos = 0;
  list.forEach((pair, i) => {
    const start = Math.max(0, Number(pair && pair[0]) || 0);
    const end = Math.max(start, Number(pair && pair[1]) || 0);
    if (start > pos) out.push({ text: slice(pos, start), s: null });
    out.push({ text: slice(Math.max(start, pos), end), s: i });
    pos = Math.max(pos, end);
  });
  const rest = slice(pos);
  if (rest) out.push({ text: rest, s: null });
  return out;
}

/** Cumulative code-point counts: offsets[i] is the characters before block i (offsets[n] is the total). */
export function charOffsets(blocks) {
  const list = Array.isArray(blocks) ? blocks : [];
  const out = new Array(list.length + 1);
  out[0] = 0;
  for (let i = 0; i < list.length; i += 1) out[i + 1] = out[i] + codePointLength(list[i] && list[i].text);
  return out;
}

/** The reading view's percent: 100 when finished (as in the library), else cursor / blocks (RE-62). */
export function readingPercent({ cursor = 0, count = 0, finished = false } = {}) {
  if (finished) return 100;
  return count > 0 ? Math.min(100, Math.max(0, Math.round((cursor / count) * 100))) : 0;
}

/**
 * The aside's meta: "Lewis Carroll · 34% · ~2 h 10 m left" (no "left" once finished). The time left comes from
 * the document's characters and this percent, as in the library (leftText), so both say the same (RE-62).
 */
export function sideMeta({ author = '', pct = 0, finished = false, charCount = 0, rate = 1 } = {}) {
  return [author || '', `${pct}%`, finished ? '' : leftText(charCount, pct, rate)].filter(Boolean).join(' · ');
}

/* ───────────────────────────── follow mode (RE-53, RE-54) ───────────────────────────── */

/** A jump longer than this many screens scrolls at once: a long smooth scroll only costs memory. */
export const FAR_SCREENS = 3;

/**
 * Where to scroll so the current sentence sits in the band between the top bar and the transport.
 * @param {{top: number, bottom: number, height: number}} rect the sentence (or paragraph) on screen
 * @param {{bandTop: number, bandBottom: number, viewport: number}} band
 * @param {{force?: boolean, reduced?: boolean}} [options] force: scroll even when it is already in the band
 * @returns {{delta: number, behavior: 'auto'|'smooth'}|null} null when nothing needs to move
 */
export function scrollPlan(rect, { bandTop, bandBottom, viewport }, { force = false, reduced = false } = {}) {
  if (!rect) return null;
  const inside = rect.top >= bandTop && rect.bottom <= bandBottom;
  if (inside && !force) return null;
  const room = Math.max(1, bandBottom - bandTop);
  const delta = rect.height > room ? rect.top - bandTop : rect.top - (bandTop + (room - rect.height) / 2);
  if (Math.abs(delta) < 1) return null;
  const far = Math.abs(delta) > FAR_SCREENS * Math.max(1, viewport);
  return { delta: Math.round(delta), behavior: far || reduced ? 'auto' : 'smooth' };
}

/* ───────────────────────────── appearance ───────────────────────────── */

/** The size step in use: the saved one, else the responsive default (19 px below 960 px, 20 px above). */
export function sizeStep(size, desktop = false) {
  return Number.isInteger(size) && size >= 0 && size < SIZES_PX.length ? size : (desktop ? 3 : 2);
}

/** "19 px" for a size step. */
export function sizeLabel(step) {
  return `${SIZES_PX[Math.max(0, Math.min(SIZES_PX.length - 1, step))]} px`;
}

/* ───────────────────────────── add ───────────────────────────── */

/** True for the file types the Reader imports (by extension, case-insensitive). */
export function acceptedFile(name) {
  const lower = String(name || '').toLowerCase();
  return FILE_TYPES.some((ext) => lower.endsWith(ext));
}

/** The dropzone hint once a file is chosen: "1.4 MB · Choose another file". */
export function fileHint(file) {
  return file ? `${bytes(file.size) || '0 B'} · Choose another file` : 'EPUB, TXT, Markdown or HTML · or choose a file';
}

/**
 * Check the Add form (RL-12): {error: {field, message}} when something is missing, else the request:
 * {kind: 'file', file} or {kind: 'json', json: {type: 'text', text, title?}}.
 */
export function addRequest({ mode = 'file', file = null, title = '', text = '' } = {}) {
  if (mode === 'paste') {
    const body = String(text || '').trim();
    if (!body) return { error: { field: 'paste', message: 'Paste some text first' } };
    const name = String(title || '').trim();
    return { kind: 'json', json: { type: 'text', text: body, ...(name ? { title: name } : {}) } };
  }
  if (!file) return { error: { field: 'file', message: 'Choose a file first' } };
  if (!acceptedFile(file.name)) return { error: { field: 'file', message: 'Choose an EPUB, TXT, Markdown or HTML file' } };
  return { kind: 'file', file };
}
