/**
 * History list (inventory HI-01…HI-08, HI-10; defects REV-06, REV-30).
 *
 * One timeline of the speech generated with X-History and the files transcribed, from the server's
 * history (GET /api/history, newest first, 50 at a time):
 * - Search (#hi-search) asks the server 250 ms after typing stops and starts again from the first page; the
 *   filter chips (#hi-filter: All, Speech, Transcripts) map to type=tts|stt and persist to prefs. Paging and
 *   search always use the type the chips show now, and the newest request wins (a request token).
 * - Rows are grouped under "Today", "Yesterday" and date headings (h2.hi-day). Speech rows have Open in
 *   Speak (the apply-history intent); every row can copy its full text and has a menu with Delete….
 * - Load more appends the next page and disappears on the last one; #hi-count (role=status) says how many
 *   are shown. Deleting reloads what is loaded, so the list never stops past its end.
 * - Another view's change while History shows (history:changed) reloads what is loaded in the background
 *   (refresh()): the pages loaded stay, focus stays on its row, and an open row menu, a delete or Clear all in
 *   progress, or a Load more on its way holds it until they are done.
 * - Clear all history… sits apart at the end. Its confirm counts every entry with an unfiltered request
 *   made when it is pressed (the visible total may be filtered).
 * @module views/history/list
 */
import { html, render, setBusy, debounce } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { announce } from '../../core/a11y.js';
import { copyText } from '../../core/clipboard.js';
import { emit, setIntent } from '../../core/store.js';
import * as prefs from '../../core/prefs.js';
import { modelLabel, shortName } from '../../core/models.js';
import { describeVoice, isBlend, listNamed } from '../../core/voices.js';
import { absoluteTime, relativeTime, speed as speedText, plural, toMs } from '../../core/format.js';
import {
  idPart, seg, errorCopy, showBanner, clearBanner, focusElement, renderKeepingFocus, menuMarkup, bindMenus,
} from '../studio/shared.js';
import { onAvailability, availability, openAskAi } from '../../components/ask-ai.js';
import { charCount } from '../../components/ask-ai-logic.js';

/** One page (the server's default and the old UI's). */
export const PAGE_SIZE = 50;
/** Search waits for this pause in typing (HI-02). */
export const SEARCH_DELAY_MS = 250;
/** The data is reloaded on show when it is older than this. */
export const STALE_MS = 30000;
/** Named-voice names are looked up again after this. */
const NAMED_TTL_MS = 60000;
export const TYPES = Object.freeze(['', 'tts', 'stt']);

/* ───────────────────────────── pure helpers ───────────────────────────── */

/**
 * PURE: the query string of GET /api/history (HI-01): type only when filtered, search only when typed.
 * @param {{type?: string, search?: string, limit?: number, offset?: number}} query
 */
export function historyQuery({ type = '', search = '', limit = PAGE_SIZE, offset = 0 } = {}) {
  const qs = new URLSearchParams();
  if (type) qs.set('type', type);
  const term = String(search || '').trim();
  if (term) qs.set('search', term);
  qs.set('limit', String(limit));
  qs.set('offset', String(Math.max(0, offset)));
  return qs.toString();
}

function startOfDay(ms) {
  const d = new Date(ms);
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}

/**
 * PURE: the day heading for a time: 'Today', 'Yesterday', 'Monday, Oct 27' this year, 'Oct 27, 2025' before.
 * @param {*} date anything format.toMs() reads
 * @param {number} [now]
 */
export function dayLabel(date, now = Date.now()) {
  const at = toMs(date);
  if (!at) return 'Earlier';
  const day = startOfDay(at);
  const today = startOfDay(now);
  if (day === today) return 'Today';
  // One calendar day back (DST-safe: the start of the day before today's start).
  if (day === startOfDay(today - 1)) return 'Yesterday';
  const d = new Date(at);
  const sameYear = d.getFullYear() === new Date(now).getFullYear();
  try {
    return sameYear
      ? d.toLocaleDateString('en', { weekday: 'long', month: 'short', day: 'numeric' })
      : d.toLocaleDateString('en', { month: 'short', day: 'numeric', year: 'numeric' });
  } catch {
    return d.toDateString();
  }
}

/**
 * PURE: the entries grouped by local day, in the order given (the server sends newest first).
 * @param {object[]} items
 * @param {number} [now]
 * @returns {{key: string, label: string, items: object[]}[]}
 */
export function groupByDay(items, now = Date.now()) {
  const groups = [];
  let last = null;
  for (const item of items || []) {
    const at = toMs(item?.created_at);
    const key = at ? String(startOfDay(at)) : 'unknown';
    if (!last || last.key !== key) {
      last = { key, label: dayLabel(item?.created_at, now), items: [] };
      groups.push(last);
    }
    last.items.push(item);
  }
  return groups;
}

/**
 * PURE: how a voice reads in a row: a Kokoro voice by its name ('Heart'), a blend as its canonical string,
 * a named voice by its name when known ('Named voice' otherwise), anything else as sent.
 * @param {string} voice
 * @param {Map<string, string>|null} [named] identity id -> name
 */
export function voiceText(voice, named = null) {
  const v = String(voice || '').trim();
  if (!v) return '';
  if (v.startsWith('voice:')) {
    const name = named && named.get(v.slice(6));
    return name || 'Named voice';
  }
  if (isBlend(v)) return v.replace(/\s+/g, '');
  return describeVoice(v).label || v;
}

/** PURE: an STT model as the status chip names it: 'whisper base'. */
export function sttModelText(model) {
  const id = String(model || '');
  if (!id) return '';
  const short = shortName({ id, type: 'stt' });
  return /whisper/i.test(id) && !/whisper/i.test(short) ? `whisper ${short}` : short;
}

/**
 * PURE: the meta parts of a row: speech is voice · model · speed · format; a transcript is its
 * model.
 * @param {object} entry a /api/history item
 * @param {Map<string, string>|null} [named]
 * @returns {string[]}
 */
export function entryMeta(entry, named = null) {
  if (!entry) return [];
  if (entry.type === 'stt') return [sttModelText(entry.model)].filter(Boolean);
  const parts = [voiceText(entry.voice, named), entry.model ? modelLabel(entry.model) : ''];
  const speed = Number(entry.speed);
  if (Number.isFinite(speed) && speed > 0) parts.push(speedText(speed));
  if (entry.format) parts.push(String(entry.format).toUpperCase());
  return parts.filter(Boolean);
}

/**
 * PURE: when a row happened, as the row shows it: relative today ('5 min ago'), else the time of day ('4:12 PM'),
 * since the day heading above it already carries the date.
 * @param {*} date
 * @param {number} [now]
 */
export function rowTime(date, now = Date.now()) {
  const at = toMs(date);
  if (!at) return '';
  if (startOfDay(at) === startOfDay(now)) return relativeTime(at, now);
  try {
    return new Date(at).toLocaleTimeString('en', { hour: 'numeric', minute: '2-digit' });
  } catch {
    return relativeTime(at, now);
  }
}

/** PURE: the text of a row: a speech entry's preview, or a transcript's file name and preview. */
export function entryText(entry) {
  const preview = String(entry?.text_preview || '').replace(/\s+/g, ' ').trim();
  const file = entry?.type === 'stt' ? String(entry?.input_filename || '').trim() : '';
  return { file, preview };
}

/**
 * PURE: the accessible name of a row's "More actions" button, so a list of buttons tells the rows apart:
 * 'More actions for speech “In a hole in the ground…”', 'More actions for transcript meeting.m4a'.
 * @param {object} entry
 * @param {number} [max] the longest snippet, in characters
 */
export function moreLabel(entry, max = 40) {
  const { file, preview } = entryText(entry);
  const kind = entry?.type === 'stt' ? 'transcript' : 'speech';
  if (file) return `More actions for ${kind} ${file}`;
  if (!preview) return `More actions for ${kind} without text`;
  const short = preview.length > max ? `${preview.slice(0, max - 1).trimEnd()}…` : preview;
  return `More actions for ${kind} “${short}”`;
}

/** PURE: what #hi-count says: 'Showing 50 of 312' while there is more, else '3 entries'. */
export function countText(shown, total) {
  const n = Math.max(0, Number(shown) || 0);
  const t = Math.max(n, Number(total) || 0);
  if (!t) return '';
  return n < t ? `Showing ${n.toLocaleString('en-US')} of ${t.toLocaleString('en-US')}` : plural(t, 'entry', 'entries');
}

/**
 * PURE: the Clear all confirm (REV-30). `n` is the unfiltered count from the request made when the button
 * was pressed; null when that request failed (the copy then omits the number).
 * @param {number|null} n
 */
export function clearCopy(n) {
  const message = 'This can’t be undone.';
  if (n === null || n === undefined || !Number.isFinite(Number(n))) return { title: 'Delete all history entries?', message };
  const count = Number(n);
  if (count === 1) return { title: 'Delete 1 history entry?', message };
  return { title: `Delete all ${count.toLocaleString('en-US')} history entries?`, message };
}

/**
 * PURE: the empty state for a query (HI-06, REV-06). The server has no "history disabled" flag, so the plain
 * empty state says where to look when nothing ever appears.
 * @param {{type?: string, search?: string}} query
 * @returns {{title: string, text: string, note: string, clearSearch: boolean}}
 */
export function emptyCopy({ type = '', search = '' } = {}) {
  const term = String(search || '').trim();
  if (term) return { title: `No history matches “${term}”.`, text: 'Search looks through the whole text of every entry and transcribed file names.', note: '', clearSearch: true };
  if (type === 'tts') return { title: 'No speech yet', text: 'Speech you generate appears here.', note: '', clearSearch: false };
  if (type === 'stt') return { title: 'No transcripts yet', text: 'Files you transcribe appear here.', note: '', clearSearch: false };
  return {
    title: 'No history yet',
    text: 'Speech you generate and files you transcribe appear here.',
    note: 'If nothing ever appears here, history may be turned off on the server (OS_HISTORY_ENABLED).',
    clearSearch: false,
  };
}

/* ───────────────────────────── markup ───────────────────────────── */

const rowId = (entry) => `hi-entry-${idPart(entry.id)}`;

function entryMarkup(entry, named, now, ask = false) {
  const base = rowId(entry);
  const tts = entry.type === 'tts';
  const { file, preview } = entryText(entry);
  const meta = entryMeta(entry, named);
  const at = entry.created_at;
  const when = rowTime(at, now);
  const exact = absoluteTime(at);
  const iso = toMs(at) ? new Date(toMs(at)).toISOString() : '';
  const hasText = !!String(entry.full_text || entry.text_preview || '').trim();
  const text = preview ? html`“${preview}”` : html`<span class="hi-entry__none">No text</span>`;
  return html`<li class="hi-entry hi-entry--${tts ? 'tts' : 'stt'}" id="${base}" data-id="${entry.id}">
    <span class="hi-entry__type" aria-hidden="true">${icon(tts ? 'speak' : 'mic')}</span>
    <div class="hi-entry__main">
      <p class="hi-entry__text" id="${base}-text"><span class="visually-hidden">${tts ? 'Speech' : 'Transcript'}: </span>${file ? html`<span class="hi-entry__file">${file}</span><span class="hi-entry__sep">: </span>` : ''}${text}</p>
      <p class="hi-entry__meta">${meta.length ? html`<span>${meta.join(' · ')}</span><span aria-hidden="true"> · </span>` : ''}${when ? html`<time class="tabular" datetime="${iso}" title="${exact}"><span aria-hidden="true">${when}</span><span class="visually-hidden">${exact}</span></time>` : ''}</p>
    </div>
    <div class="hi-entry__actions">
      ${tts
        ? html`<button type="button" class="btn btn--secondary btn--sm hi-entry__open" id="${base}-open" data-action="open-in-speak" data-id="${entry.id}" aria-describedby="${base}-text">${icon('speak')}<span class="btn__label">Open in Speak</span></button>`
        : (hasText ? html`<button type="button" class="btn btn--secondary btn--sm hi-entry__copy" id="${base}-copy" data-action="copy-text" data-id="${entry.id}" aria-describedby="${base}-text">${icon('copy')}<span class="btn__label">Copy text</span></button>` : '')}
      <button type="button" class="icon-btn hi-entry__more" id="${base}-more" data-menu-btn="${base}-menu" aria-label="${moreLabel(entry)}" data-tooltip="More actions">${icon('dots')}</button>
      ${entryMenuMarkup(entry, ask)}
    </div>
  </li>`;
}

/** A row's menu, written right after its More actions button. Ask AI while the assistant is on. */
function entryMenuMarkup(entry, ask = false) {
  const tts = entry.type === 'tts';
  const hasText = !!String(entry.full_text || entry.text_preview || '').trim();
  const options = [];
  if (ask && hasText) options.push({ cmd: 'ask-ai', arg: entry.id, label: 'Ask AI…', icon: 'wand' });
  if (tts && hasText) options.push({ cmd: 'copy-text', arg: entry.id, label: 'Copy text', icon: 'copy' });
  options.push({ cmd: 'delete-entry', arg: entry.id, label: 'Delete…', icon: 'trash', danger: true });
  return menuMarkup(`${rowId(entry)}-menu`, tts ? 'Speech entry' : 'Transcript entry', options);
}

/**
 * The list markup for the loaded entries (exported for tests).
 * @param {object[]} items
 * @param {{named?: Map<string, string>|null, now?: number, ask?: boolean}} [options] ask: each row's menu has
 *   Ask AI (the assistant is on)
 */
export function listMarkup(items, { named = null, now = Date.now(), ask = false } = {}) {
  return html`${groupByDay(items, now).map((group, i) => html`<div class="hi-group">
    <h2 class="hi-day" id="hi-day-${i}">${group.label}</h2>
    <ul class="hi-entries" aria-labelledby="hi-day-${i}">${group.items.map((entry) => entryMarkup(entry, named, now, ask))}</ul>
  </div>`)}`;
}

function loadingMarkup() {
  return html`<div class="hi-loading" aria-busy="true"><span class="visually-hidden">Loading history…</span><div class="skeleton skeleton--line hi-skeleton-label" aria-hidden="true"></div><div class="hi-skeleton-card" aria-hidden="true">${[0, 1, 2, 3].map(() => html`<div class="hi-skeleton-row"><span class="skeleton hi-skeleton-icon"></span><span class="hi-skeleton-lines"><span class="skeleton skeleton--line"></span><span class="skeleton skeleton--line hi-skeleton-short"></span></span></div>`)}</div></div>`;
}

function emptyMarkup(query) {
  const copy = emptyCopy(query);
  return html`<div class="empty hi-empty">${icon(copy.clearSearch ? 'search' : 'history', { className: 'ico empty__icon' })}<h2 class="empty__title">${copy.title}</h2><p class="empty__text">${copy.text}</p>${copy.note ? html`<p class="empty__text hi-empty__note">${copy.note}</p>` : ''}${copy.clearSearch ? html`<button type="button" class="btn btn--secondary" id="hi-clear-search" data-action="clear-search"><span class="btn__label">Clear search</span></button>` : ''}</div>`;
}

/* ───────────────────────────── the list ───────────────────────────── */

/**
 * The History list inside the view section.
 * @param {HTMLElement} root the view section
 * @param {{isShown?: () => boolean, navigate?: (path: string, options?: object) => *, openSettings?: () => *}} [ctx]
 */
export function createHistoryList(root, ctx = {}) {
  const q = (id) => root.querySelector(`#${id}`);
  const el = {
    title: q('hi-title'),
    search: q('hi-search'),
    filter: q('hi-filter'),
    banner: q('hi-banner'),
    list: q('hi-list'),
    foot: q('hi-foot'),
    count: q('hi-count'),
    more: q('hi-more'),
    moreError: q('hi-more-error'),
    clearWrap: q('hi-clear-wrap'),
    clear: q('hi-clear'),
  };
  const isShown = typeof ctx.isShown === 'function' ? ctx.isShown : () => true;

  const state = {
    type: TYPES.includes(prefs.get().historyFilter) ? prefs.get().historyFilter : '',
    query: { type: '', search: '' },   // what the loaded items answer
    items: [],
    total: 0,
    status: 'idle',                   // idle | loading | ready | error
    error: null,
    errorByUser: false,               // the failed load came from the user's own action
    more: 'idle',                     // idle | loading | error
    seq: 0,
    quietSeq: 0,                      // the seq of a background refresh still out (0: none)
    retryRefresh: false,              // the failed load was a background refresh: Retry reloads what is loaded
    at: 0,
    named: null,                      // Map(identity id -> name)
    namedAt: 0,
    namedLoading: null,
    unfilteredEmpty: false,           // the whole history is known to be empty
    ask: false,                       // the rows' menus offer Ask AI (the assistant is on)
  };
  let unbindMenus = () => {};
  const offs = [];

  const controlsQuery = () => ({ type: state.type, search: String(el.search?.value || '').trim() });
  const sameQuery = (a, b) => a.type === b.type && a.search === b.search;
  const find = (id) => state.items.find((item) => item.id === id) || null;

  /* ── rendering ── */

  function syncChips() {
    for (const chip of el.filter.querySelectorAll('[data-type]')) {
      chip.setAttribute('aria-pressed', chip.getAttribute('data-type') === state.type ? 'true' : 'false');
    }
  }

  function renderFoot() {
    const loaded = state.status === 'ready' || (state.status === 'loading' && state.at > 0);
    const text = loaded ? countText(state.items.length, state.total) : '';
    if (el.count.textContent !== text) el.count.textContent = text;
    const more = loaded && state.status === 'ready' && state.items.length < state.total;
    el.more.hidden = !more;
    if (state.more === 'loading') setBusy(el.more, true, { label: 'Loading…' });
    else setBusy(el.more, false);
    // Clear all is offered unless the whole history is known to be empty.
    el.clearWrap.hidden = state.unfilteredEmpty || (state.status !== 'ready' && !state.at);
  }

  function rowFallback(index) {
    return () => {
      const rows = el.list.querySelectorAll('.hi-entry');
      if (!rows.length) return el.title;
      const row = rows[Math.min(Math.max(0, index), rows.length - 1)];
      return row.querySelector('.hi-entry__actions button') || el.title;
    };
  }

  /** The index of the row that holds keyboard focus, or null. */
  function focusedRowIndex() {
    const active = typeof document !== 'undefined' ? document.activeElement : null;
    const row = active && active !== el.list && el.list.contains(active) && active.closest ? active.closest('.hi-entry') : null;
    if (!row) return null;
    const index = [...el.list.querySelectorAll('.hi-entry')].indexOf(row);
    return index >= 0 ? index : null;
  }

  /**
   * @param {{focusIndex?: number|null, fallbackIndex?: number|null}} [options] focusIndex: move focus to that
   *   row (after a delete); fallbackIndex: where focus goes when its control is gone (default the first row)
   */
  function renderList({ focusIndex = null, fallbackIndex = null } = {}) {
    unbindMenus();
    unbindMenus = () => {};
    const busy = state.status === 'loading';
    el.list.setAttribute('aria-busy', busy ? 'true' : 'false');
    if (state.status === 'error') showLoadError();
    else clearLoadErrorBanner();
    if (state.status === 'error' && !state.items.length) {
      render(el.list, html``);
      renderFoot();
      return;
    }
    if ((busy || state.status === 'idle') && !state.at) {
      render(el.list, loadingMarkup());
      renderFoot();
      return;
    }
    state.ask = availability().phase === 'on';
    const markup = state.items.length ? listMarkup(state.items, { named: state.named, ask: state.ask }) : emptyMarkup(state.query);
    renderKeepingFocus(el.list, markup, {
      fallback: rowFallback(focusIndex ?? fallbackIndex ?? 0),
    });
    unbindMenus = bindMenus(el.list, onMenu);
    renderFoot();
    if (focusIndex !== null) {
      const target = rowFallback(focusIndex)();
      focusElement(target);
    }
  }

  /* ── banners ── */

  let loadErrorShown = false;
  /**
   * The load error banner. It is role=alert only when the failed load came from the user's own action (a
   * filter, a search, Retry, a delete); a load the view started itself (on show, after another view's change)
   * says it politely (role=status), since the shell's connection banner may already be speaking.
   */
  function showLoadError() {
    const copy = errorCopy(state.error, "Couldn't load history");
    const actions = copy.auth ? [{ cmd: 'open-settings', label: 'Open Settings' }] : [{ cmd: 'retry', label: 'Retry' }];
    showBanner(el.banner, {
      title: copy.auth ? 'API key needed' : "Couldn't load history", message: copy.message, detail: copy.detail, actions,
      dismiss: false, prefix: 'hi', role: state.errorByUser ? 'alert' : 'status',
    });
    loadErrorShown = true;
  }
  function clearLoadErrorBanner() {
    if (loadErrorShown) {
      clearBanner(el.banner);
      loadErrorShown = false;
    }
  }
  function showActionError(error, title) {
    const copy = errorCopy(error, title);
    loadErrorShown = false;
    showBanner(el.banner, { title: copy.title, message: copy.message, detail: copy.detail, actions: copy.auth ? [{ cmd: 'open-settings', label: 'Open Settings' }] : [], prefix: 'hi' });
  }

  /* ── loading ── */

  /**
   * Load the first page for the query the controls show (fresh), the next page of the list (append), or, in the
   * background, what is loaded again (refresh: see refresh()). The newest request wins: an older answer is
   * dropped (HI-03).
   * @param {{append?: boolean, refresh?: boolean, limit?: number, focusIndex?: number|null, user?: boolean}}
   *   [options] user: the user's own action asked for it (a failure is then role=alert, else role=status)
   * @returns {Promise<boolean>} whether this answer was applied
   */
  async function load({ append = false, refresh = false, limit = PAGE_SIZE, focusIndex = null, user = false } = {}) {
    const query = append || refresh ? state.query : controlsQuery();
    const offset = append ? state.items.length : 0;
    // A refresh asks for what is loaded plus a page, so the entries added above keep every row loaded below them
    // (the answer is cut after the last of those rows).
    const known = refresh ? new Set(state.items.map((item) => item.id)) : null;
    const size = refresh ? state.items.length + PAGE_SIZE : limit;
    // A page asked for while a refresh is out replaces it: the refresh runs again after the page.
    if (append && state.quietSeq && state.quietSeq === state.seq) refreshWanted = true;
    const seq = (state.seq += 1);
    state.quietSeq = refresh ? seq : 0;
    if (append) {
      state.more = 'loading';
      clearBanner(el.moreError);
      renderFoot();
    } else if (!refresh) {
      // A fresh first page answers whatever changed meanwhile.
      refreshWanted = false;
      search.cancel();
      state.status = 'loading';
      state.more = 'idle';
      clearBanner(el.moreError);
      renderList();
    }
    let data;
    try {
      data = await api.get(`/api/history?${historyQuery({ ...query, limit: size, offset })}`, { accept: 'json' });
    } catch (error) {
      if (seq !== state.seq || (error && error.kind === 'aborted')) return false;
      state.quietSeq = 0;
      if (append) {
        state.more = 'idle';
        const copy = errorCopy(error, "Couldn't load more history");
        showBanner(el.moreError, {
          title: copy.title, message: copy.message, detail: copy.detail, prefix: 'hi',
          actions: copy.auth ? [{ cmd: 'open-settings', label: 'Open Settings' }] : [{ cmd: 'more', label: 'Retry' }],
        });
        renderFoot();
        return false;
      }
      // A refresh of the same query keeps what is shown (marked stale by the banner); a new search or filter
      // that failed must not show the old answer under the new chips.
      if (!sameQuery(query, state.query)) {
        state.items = [];
        state.total = 0;
        state.at = 0;
      }
      const fallbackIndex = refresh ? focusedRowIndex() : null;
      state.status = 'error';
      state.error = error;
      state.errorByUser = !!user;
      state.retryRefresh = refresh;
      state.query = query;
      renderList({ fallbackIndex });
      return false;
    }
    if (seq !== state.seq) return false;
    state.quietSeq = 0;
    let items = Array.isArray(data?.items) ? data.items : [];
    const total = Number.isFinite(Number(data?.total)) ? Number(data.total) : items.length;
    if (refresh) {
      let end = -1;
      items.forEach((item, i) => { if (known.has(item.id)) end = i; });
      items = end >= 0 ? items.slice(0, end + 1) : items.slice(0, Math.max(PAGE_SIZE, known.size));
    }
    const fallbackIndex = refresh ? focusedRowIndex() : null;
    if (append) {
      const known = new Set(state.items.map((item) => item.id));
      state.items = [...state.items, ...items.filter((item) => !known.has(item.id))];
      state.more = 'idle';
    } else {
      state.items = items;
      state.query = query;
    }
    state.total = total;
    state.status = 'ready';
    state.error = null;
    state.at = Date.now();
    if (!query.type && !query.search) state.unfilteredEmpty = total === 0;
    else if (total > 0) state.unfilteredEmpty = false;
    renderList({ focusIndex, fallbackIndex });
    if (!append && !refresh && !items.length && query.search && isShown()) announce(emptyCopy(query).title);
    if (append && items.length) {
      // Keyboard users continue from the first new row.
      const first = items[0] && q(`${rowId(items[0])}`);
      const button = first ? first.querySelector('.hi-entry__actions button') : null;
      if (button && document.activeElement && (document.activeElement === el.more || document.activeElement === document.body)) focusElement(button);
    }
    ensureNamed();
    return true;
  }

  /** Reload what is loaded (after a delete): the same query, from the start, at least one page (HI-07). */
  function reloadLoaded({ focusIndex = null } = {}) {
    const limit = Math.max(PAGE_SIZE, state.items.length);
    // The list answers the query it shows; a search typed meanwhile runs after.
    const typed = controlsQuery();
    if (!sameQuery(typed, state.query)) return load({ focusIndex, user: true });
    return load({ limit, focusIndex, user: true });
  }

  /* ── refresh while shown ── */

  let refreshWanted = false;   // a refresh waits for an open row menu, a delete or Clear all, or Load more
  let busyActions = 0;         // a delete or Clear all between its press and its own reload
  let stopMenuWatch = null;

  const menuOpen = () => !!el.list.querySelector('[data-menu-btn][aria-expanded="true"]');

  /**
   * Another view changed the history while this one shows (HI-01): reload what is loaded,
   * not the first page, so the pages added with Load more stay, keyboard focus stays on its row and nothing is
   * announced. An open row menu, a delete or Clear all in progress, or a Load more on its way defers it until
   * they are done (their own reload then answers it). A list that shows another query, or none, loads afresh.
   * @returns {Promise<boolean>}
   */
  function refresh() {
    const loaded = state.status === 'ready' || (state.status === 'error' && state.retryRefresh);
    if (!loaded || !sameQuery(controlsQuery(), state.query)) return load();
    if (busyActions || state.more === 'loading' || menuOpen()) {
      refreshWanted = true;
      if (menuOpen()) watchMenus();
      return Promise.resolve(false);
    }
    refreshWanted = false;
    return load({ refresh: true });
  }

  function refreshIfWanted() {
    if (refreshWanted && isShown()) refresh();
  }

  /** Run a deferred refresh once the open row menu closes (an item, Esc, Tab, a click outside). */
  function watchMenus() {
    if (stopMenuWatch) return;
    let timer = 0;
    // After the event: components/menu.js closes the menu in its own listeners, some of them later in the dispatch.
    const check = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (menuOpen()) return;
        unwatchMenus();
        refreshIfWanted();
      }, 0);
    };
    const types = ['focusin', 'pointerdown', 'click', 'keydown'];
    for (const type of types) document.addEventListener(type, check, true);
    stopMenuWatch = () => {
      clearTimeout(timer);
      for (const type of types) document.removeEventListener(type, check, true);
    };
  }

  function unwatchMenus() {
    const stop = stopMenuWatch;
    stopMenuWatch = null;
    if (stop) stop();
  }

  /** Named voices ('voice:<id>') read better by name: one lookup, cached for a minute; failures keep 'Named voice'. */
  function ensureNamed() {
    const needs = state.items.some((item) => item.type === 'tts' && String(item.voice || '').startsWith('voice:'));
    if (!needs || state.namedLoading || (state.named && Date.now() - state.namedAt < NAMED_TTL_MS)) return;
    state.namedLoading = listNamed({ check: false })
      .then((list) => {
        state.named = new Map((list || []).map((n) => [String(n.id), String(n.name || '')]));
        state.namedAt = Date.now();
        if (state.items.length && state.status === 'ready') renderList();
      })
      .catch(() => {})
      .finally(() => { state.namedLoading = null; });
  }

  const search = debounce(() => { load({ user: true }); }, SEARCH_DELAY_MS);

  /* ── actions ── */

  function setType(type) {
    if (!TYPES.includes(type)) return;
    state.type = type;
    syncChips();
    prefs.set({ historyFilter: type });
    load({ user: true });
  }

  function loadMore() {
    // Paging follows what the controls show now (HI-03): a changed search or filter starts again from the top.
    if (!sameQuery(controlsQuery(), state.query) || state.status !== 'ready') return load({ user: true });
    if (state.items.length >= state.total || state.more === 'loading') return undefined;
    return load({ append: true, user: true }).finally(refreshIfWanted);
  }

  function openInSpeak(id) {
    const entry = find(id);
    if (!entry || entry.type !== 'tts') return;
    setIntent('speak', { type: 'apply-history', entry: { ...entry, meta: { ...(entry.meta || {}) } } });
    if (typeof ctx.navigate === 'function') ctx.navigate('speak');
  }

  async function copyEntry(id) {
    const entry = find(id);
    if (!entry) return;
    const ok = await copyText(String(entry.full_text || entry.text_preview || ''));
    if (ok) toast('Text copied', { kind: 'success' });
    else toast('Couldn’t copy the text. Your browser refused the clipboard.', { kind: 'error' });
  }

  /** A delete or Clear all holds a refresh back from its press to its own reload (which answers it). */
  async function holdingRefresh(work) {
    busyActions += 1;
    try {
      return await work();
    } finally {
      busyActions -= 1;
      refreshIfWanted();
    }
  }

  function deleteEntry(id) {
    const entry = find(id);
    if (!entry) return undefined;
    return holdingRefresh(() => deleteFound(id, entry));
  }

  async function deleteFound(id, entry) {
    const { file, preview } = entryText(entry);
    const ok = await confirm({
      title: 'Delete this history entry?',
      message: `${file ? `${file}: ` : ''}“${preview.length > 90 ? `${preview.slice(0, 89).trimEnd()}…` : preview}”`,
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok) return;
    const index = state.items.findIndex((item) => item.id === id);
    clearBanner(el.banner);
    loadErrorShown = false;
    try {
      await api.del(`/api/history/${seg(id)}`);
    } catch (error) {
      if (!(error && error.status === 404)) {
        showActionError(error, "Couldn't delete");
        return;
      }
    }
    await reloadLoaded({ focusIndex: Math.max(0, index) });
    toast('History entry deleted', { kind: 'success' });
    selfEmit();
  }

  /** Clear all (HI-08, REV-30): an unfiltered count when pressed, a confirm that names it, then DELETE. */
  function clearAll(button) {
    return holdingRefresh(() => clearAllNow(button));
  }

  async function clearAllNow(button) {
    let count = null;
    setBusy(button, true);
    try {
      const data = await api.get('/api/history?limit=1', { accept: 'json' });
      const n = Number(data?.total);
      count = Number.isFinite(n) ? n : null;
    } catch {
      count = null;
    } finally {
      setBusy(button, false);
    }
    if (count === 0) {
      state.unfilteredEmpty = true;
      renderFoot();
      toast('History is already empty.', { kind: 'info' });
      focusElement(el.title);
      return;
    }
    const copy = clearCopy(count);
    const ok = await confirm({ title: copy.title, message: copy.message, confirmLabel: 'Delete all', danger: true });
    if (!ok) return;
    clearBanner(el.banner);
    loadErrorShown = false;
    setBusy(button, true, { label: 'Deleting…' });
    let deleted = count;
    try {
      const result = await api.del('/api/history', { accept: 'json' });
      if (Number.isFinite(Number(result?.deleted))) deleted = Number(result.deleted);
    } catch (error) {
      setBusy(button, false);
      showActionError(error, "Couldn't delete");
      return;
    }
    setBusy(button, false);
    state.unfilteredEmpty = true;
    await load({ user: true });
    selfEmit();
    toast(deleted === null ? 'History cleared' : `Deleted ${plural(deleted, 'history entry', 'history entries')}`, { kind: 'success' });
    focusElement(el.title);
  }

  let emitting = false;
  function selfEmit() {
    emitting = true;
    try { emit('history:changed'); } finally { emitting = false; }
  }

  /**
   * Ask AI about an entry: a transcript by its id (the server reads its full text,
   * `{kind: 'transcript', history_id}`), the text of a speech entry as typed text.
   */
  function askEntry(id, trigger) {
    const entry = find(id);
    if (!entry) return;
    const words = String(entry.full_text || entry.text_preview || '').trim();
    if (!words) return;
    const tts = entry.type === 'tts';
    const { file } = entryText(entry);
    const source = tts
      ? { id: 'entry', label: 'The text of this speech entry', input: { kind: 'text', text: words }, text: words, chars: charCount(words) }
      : { id: 'entry', label: file ? `The transcript of ${file}` : 'This transcript', input: { kind: 'transcript', history_id: String(entry.id) }, text: words, chars: charCount(words) };
    openAskAi({ trigger, sources: [source] }).catch((error) => toast(error && error.message ? error.message : 'Something went wrong.', { kind: 'error' }));
  }

  function onMenu(cmd, item, button) {
    const id = item.getAttribute('data-arg');
    if (cmd === 'copy-text') copyEntry(id);
    else if (cmd === 'delete-entry') deleteEntry(id);
    else if (cmd === 'ask-ai') askEntry(id, button);
  }

  // The rows' menus offer Ask AI only while the assistant is on: they are written again when that changes.
  offs.push(onAvailability((avail) => {
    const on = avail.phase === 'on';
    if (on !== !!state.ask && state.items.length && state.status === 'ready') renderList();
  }));

  /* ── events ── */

  const onSearchInput = () => search();
  el.search.addEventListener('input', onSearchInput);
  offs.push(() => el.search.removeEventListener('input', onSearchInput));
  // Enter runs the search at once (no need to wait for the pause).
  const onSearchKey = (event) => {
    if (event.key === 'Enter' && !event.isComposing) {
      event.preventDefault();
      search.flush();
    }
  };
  el.search.addEventListener('keydown', onSearchKey);
  offs.push(() => el.search.removeEventListener('keydown', onSearchKey));

  const actions = {
    filter: (button) => setType(button.getAttribute('data-type') || ''),
    more: () => loadMore(),
    'clear-all': (button) => clearAll(button),
    'clear-search': () => {
      el.search.value = '';
      load({ user: true }).then(() => focusElement(el.search));
    },
    'open-in-speak': (button) => openInSpeak(button.getAttribute('data-id')),
    'copy-text': (button) => copyEntry(button.getAttribute('data-id')),
    // A failed background refresh is retried as one, so the pages loaded stay.
    retry: () => load({ user: true, refresh: state.retryRefresh && state.status === 'error' && sameQuery(controlsQuery(), state.query) }),
    'open-settings': () => (typeof ctx.openSettings === 'function' ? ctx.openSettings() : undefined),
  };

  syncChips();
  renderList();

  return {
    actions,
    state,
    load,
    loadMore,
    refresh,
    /** True when the list should reload on show (never loaded, older than STALE_MS, or a refresh was deferred). */
    stale(now = Date.now()) {
      return !state.at || now - state.at > STALE_MS || refreshWanted;
    },
    /** Another view changed the history while this one listens (it emitted history:changed). */
    isSelfEmit: () => emitting,
    focusSearch() {
      focusElement(el.search);
      if (typeof el.search.select === 'function') el.search.select();
    },
    /** Screenshot states. */
    applyShot(shot) {
      if (shot === 'row-menu') {
        const button = el.list.querySelector('.hi-entry__more');
        if (button) button.click();
      } else if (shot === 'load-error') {
        state.seq += 1;
        state.items = [];
        state.at = 0;
        state.status = 'error';
        state.error = new api.ApiError("Can't reach the server.", { status: 0, code: 'network', kind: 'network' });
        renderList();
      } else if (shot === 'search-empty') {
        el.search.value = 'unicorn';
        load();
      } else if (shot === 'loading') {
        state.seq += 1;
        state.items = [];
        state.at = 0;
        state.status = 'loading';
        renderList();
      } else if (shot === 'more-error') {
        state.more = 'idle';
        showBanner(el.moreError, { title: "Couldn't load more history", message: 'Check that Open Speech is running.', prefix: 'hi', actions: [{ cmd: 'more', label: 'Retry' }] });
        renderFoot();
      }
    },
    destroy() {
      search.cancel();
      unwatchMenus();
      unbindMenus();
      for (const off of offs.splice(0)) off();
    },
  };
}
