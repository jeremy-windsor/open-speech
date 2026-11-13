/**
 * The Library: `GET /v1/reader/documents`, rendered as the Continue-listening card and one row
 * per document, with search (title and author, debounced 150 ms), filter chips and sort (both kept in prefs).
 *
 * States: loading (three skeleton rows, aria-busy on #rd-list), empty, error (404 / 503 / other, each with
 * Try again and the server's message in Details; an auth refusal offers Open Settings; the header Add hides, since
 * it can't work then), no search match and filtered empty (RL-01…RL-04).
 *
 * - The card's title is a button that opens the document without playing (BUG-21); Resume unlocks audio inside
 *   the click, opens and plays (RL-05). The library summary has no current chapter, so the card shows none
 *   (REV-07).
 * - Each row opens its document; its "More" menu offers Open, Start over, Mark as finished and Delete… (RL-06,
 *   RL-10). Start over and Mark as finished carry the saved model, voice and speed, which the server would
 *   otherwise reset (REV-03).
 * - A re-render keeps keyboard focus on the same control of the same document, or on its neighbour when the
 *   row went (WCAG 2.4.3). A reload answered after a change made here keeps that change.
 * @module views/reader/library
 */
import { html, render, setHidden, debounce, actions, nextFrame } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import * as prefs from '../../core/prefs.js';
import { toast } from '../../core/toast.js';
import { confirm } from '../../core/dialogs.js';
import { emit } from '../../core/store.js';
import { applyTile, monogram } from '../../core/tiles.js';
import { bindMenuButton } from '../../components/menu.js';
import {
  searchDocs, filterDocs, sortDocs, heroDoc, heroFor, statusOf, percentOf, isFinished, sourceBadge, docMetaParts,
  heroMeta, countLabel, filteredEmptyCopy, noMatchCopy, libraryErrorCopy, progressPayload, mergeLibrary,
} from './text-logic.js';

export const DOCUMENTS_PATH = '/v1/reader/documents';
export const SEARCH_DEBOUNCE_MS = 150;
/** A library older than this reloads when the view shows again (GS-12). */
export const STALE_MS = 30000;
/** The Settings › Reading appearance hint when there is no document to show it over. */
export const HINT_TEXT = 'Open a document, then tap Aa to change how it looks.';

const docPath = (id) => `${DOCUMENTS_PATH}/${encodeURIComponent(id)}`;
const SEPARATOR = '\u00a0· ';   // a non-breaking space before the dot, a plain one after

/** The row's meta facts: the author may wrap; the count and the length each stay on one line, and a separator
 *  never starts a line (it is tied to the word before it). */
function factsMarkup(doc) {
  const { author, count, length } = docMetaParts(doc);
  const parts = [author ? html`${author}` : '', ...[count, length].filter(Boolean).map((part) => html`<span class="rd-row__fact">${part}</span>`)].filter(Boolean);
  return parts.map((part, i) => (i ? html`${SEPARATOR}${part}` : part));
}

function rowMarkup(doc) {
  const pct = percentOf(doc);
  const badge = sourceBadge(doc.source_type);
  const title = doc.title || 'Untitled';
  return html`<li class="row rd-row" data-doc="${doc.id}">
    <button type="button" class="row__main rd-row__main" data-action="open" data-doc="${doc.id}">
      <span class="tile rd-row__tile" data-tile="${doc.id}" aria-hidden="true">${monogram(title)}</span>
      <span class="rd-row__body">
        <span class="row__title rd-row__title">${title}</span>
        <span class="row__meta rd-row__meta"><span class="badge rd-badge">${icon(badge.icon, { className: 'ico ico--sm' })}${badge.label}</span> <span class="rd-row__facts">${factsMarkup(doc)}</span></span>
        ${pct === null ? '' : html`<progress class="progress progress--thin rd-row__bar" max="100" value="${pct}" aria-hidden="true"></progress>`}
      </span>
      ${pct === null
        ? html`<span class="rd-row__pct rd-row__pct--new">New</span>`
        : html`<span class="rd-row__pct tabular">${isFinished(doc) ? html`<span class="visually-hidden">Finished, </span>` : ''}${pct}%</span>`}
    </button>
    <span class="row__trail">
      <button type="button" class="icon-btn rd-row__more" data-action="row-menu" data-doc="${doc.id}" aria-label="${`More for ${title}`}" aria-haspopup="menu" aria-expanded="false" aria-controls="rd-row-menu">${icon('dots')}</button>
    </span>
  </li>`;
}

function heroMarkup(doc) {
  const pct = percentOf(doc) ?? 0;
  const title = doc.title || 'Untitled';
  return html`<span class="tile tile--lg rd-hero__tile" data-tile="${doc.id}" aria-hidden="true">${monogram(title)}</span>
    <div class="rd-hero__body">
      <h2 class="rd-hero__title"><button type="button" class="rd-hero__open" data-action="open" data-doc="${doc.id}">${title}</button></h2>
      <p class="rd-hero__meta">${heroMeta(doc)}</p>
      <progress class="progress rd-hero__bar" max="100" value="${pct}" aria-label="${`${pct}% listened`}">${pct}%</progress>
      <div class="rd-hero__actions">
        <button type="button" class="btn btn--primary rd-hero__resume" data-action="resume" data-doc="${doc.id}" aria-label="${`Resume ${title}`}">${icon('play')}<span class="btn__label">Resume</span></button>
      </div>
    </div>`;
}

function skeletonMarkup() {
  return [
    html`<li class="visually-hidden">Loading library…</li>`,
    ...[0, 1, 2].map(() => html`<li class="rd-skel" aria-hidden="true"><span class="skeleton skeleton--row"></span></li>`),
  ];
}

/**
 * Bind the library inside the Reader's root.
 * @param {HTMLElement} root
 * @param {{
 *   onOpen?: (id: string) => void, onResume?: (id: string) => void, onAdd?: (trigger: Element) => void,
 *   onOpenSettings?: () => void, beforeChange?: (doc: object) => (void|Promise<void>),
 *   onDeleted?: (id: string) => void,
 * }} [options]
 */
export function createLibrary(root, {
  onOpen = () => {}, onResume = () => {}, onAdd = () => {}, onOpenSettings = () => {},
  beforeChange = () => {}, onDeleted = () => {},
} = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    library: $('rd-library'),
    add: $('rd-add'),
    hint: $('rd-hint'),
    tools: $('rd-tools'),
    search: $('rd-search'),
    filter: $('rd-filter'),
    sort: $('rd-sort'),
    hero: $('rd-hero'),
    heroContent: $('rd-hero-content'),
    count: $('rd-count'),
    list: $('rd-list'),
    empty: $('rd-empty'),
    menu: $('rd-row-menu'),
  };
  const st = {
    docs: [],
    loaded: false,
    loading: false,
    error: null,
    loadedAt: 0,
    token: 0,
    query: '',
    filter: prefs.get().libraryFilter,
    sort: prefs.get().librarySort,
    menu: null,        // {button, ctl, id}
    menuDoc: null,
    changed: new Map(),   // id -> when a position changed here (Start over, finished, a save by the engine)
    removed: new Map(),   // id -> when it was deleted here (or an open found it gone)
  };
  const offs = [];

  const docById = (id) => st.docs.find((doc) => doc.id === id) || null;

  /* ───────────── rendering ───────────── */

  function tiles(container) {
    for (const el of container.querySelectorAll('[data-tile]')) applyTile(el, el.getAttribute('data-tile'));
  }

  function dropMenu() {
    if (st.menu) {
      st.menu.ctl.destroy();
      st.menu = null;
    }
  }

  /** The header Add is the page's primary action, except while the empty state offers its own (one per page). */
  function setAddEmphasis(quiet) {
    els.add.classList.toggle('btn--primary', !quiet);
    els.add.classList.toggle('btn--secondary', quiet);
  }

  /** The control holding keyboard focus in the rows, the card or the row menu, so a re-render can put it back. */
  function focusSpot() {
    const active = typeof document !== 'undefined' ? document.activeElement : null;
    if (!active || active === document.body || !active.closest) return null;
    const rowIndex = (id) => Array.from(els.list.querySelectorAll('.rd-row')).findIndex((li) => li.getAttribute('data-doc') === id);
    if (els.menu.contains(active)) {
      // The menu closes with the re-render: focus goes back to its row's More button.
      return st.menu ? { area: 'list', doc: st.menu.id, action: 'row-menu', index: rowIndex(st.menu.id) } : null;
    }
    const area = els.list.contains(active) ? 'list'
      : els.heroContent.contains(active) ? 'hero'
        : els.empty.contains(active) ? 'empty' : '';
    if (!area) return null;
    const control = active.closest('[data-action]');
    const doc = control ? control.getAttribute('data-doc') || '' : '';
    return { area, doc, action: control ? control.getAttribute('data-action') : '', index: area === 'list' ? rowIndex(doc) : -1 };
  }

  /**
   * After a re-render (WCAG 2.4.3): the same control of the same document (or the same button of the empty
   * state); else, for a row that went (deleted or filtered out), its neighbour; else the state's own button, the
   * active filter or Add. Focus that is still on something is left alone.
   */
  function restoreFocus(spot) {
    const active = document.activeElement;
    const lost = !active || active === document.body || active.isConnected === false || !!(active.closest && active.closest('[hidden]'));
    if (!lost) return;
    const find = (container, action, doc = spot.doc) => Array.from(container.querySelectorAll('[data-action]'))
      .find((el) => el.getAttribute('data-action') === action && (el.getAttribute('data-doc') || '') === doc) || null;
    const areas = { list: els.list, hero: els.heroContent, empty: els.empty };
    let target = spot.action && !(spot.area === 'empty' && els.empty.hidden) ? find(areas[spot.area], spot.action) : null;
    if (!target && spot.doc) target = find(els.list, 'open');
    if (!target) {
      const mains = els.list.querySelectorAll('.rd-row__main');
      if (mains.length) target = mains[Math.min(Math.max(0, spot.index), mains.length - 1)];
    }
    if (!target && !els.empty.hidden) target = els.empty.querySelector('button');
    if (!target && !els.tools.hidden) target = els.filter.querySelector('[aria-pressed="true"]');
    if (!target) target = els.add;
    if (target && typeof target.focus === 'function') target.focus();
  }

  function renderEmpty({ kind, title, text, action, detail = '' }) {
    const iconName = kind !== 'error' ? 'book' : action === 'open-settings' ? 'key' : 'alert';
    let button = '';
    if (action === 'add') button = html`<button type="button" class="btn btn--primary" data-action="add">${icon('plus')}<span class="btn__label">Add your first document</span></button>`;
    else if (action === 'retry') button = html`<button type="button" class="btn btn--secondary" data-action="retry">${icon('retry')}<span class="btn__label">Try again</span></button>`;
    else if (action === 'open-settings') button = html`<button type="button" class="btn btn--primary" data-action="open-settings">${icon('settings')}<span class="btn__label">Open Settings</span></button>`;
    els.empty.classList.toggle('empty--error', kind === 'error');
    // The server's own message goes in Details, as text only.
    const details = detail ? html`<details class="banner__details rd-empty__details"><summary>Details</summary><pre></pre></details>` : '';
    render(els.empty, html`${icon(iconName, { className: 'ico empty__icon' })}<h2 class="empty__title">${title}</h2><p class="empty__text">${text}</p>${button}${details}`);
    const pre = els.empty.querySelector('pre');
    if (pre) pre.textContent = detail;
    setHidden(els.empty, false);
  }

  function renderList() {
    dropMenu();
    const all = st.docs;
    const filtered = filterDocs(searchDocs(all, st.query), st.filter);
    const shown = sortDocs(filtered, st.sort);
    const query = st.query.trim();
    const hero = heroFor(all, { filter: st.filter, query });
    if (hero) {
      render(els.heroContent, heroMarkup(hero));
      tiles(els.heroContent);
    } else {
      render(els.heroContent, html``);
    }
    setHidden(els.hero, !hero);
    els.count.textContent = countLabel({ shown: shown.length, total: all.length, query, filter: st.filter });
    if (shown.length) {
      render(els.list, shown.map(rowMarkup));
      tiles(els.list);
    } else {
      const line = query ? noMatchCopy(query) : filteredEmptyCopy(st.filter);
      render(els.list, html`<li class="rd-none">${line}</li>`);
    }
  }

  function paint() {
    setHidden(els.empty, true);
    els.list.removeAttribute('aria-busy');
    setHidden(els.add, !!st.error);   // Add can't work while the library can't load (a Try again included)
    if (!st.loaded && st.loading) {
      dropMenu();
      setAddEmphasis(false);
      setHidden(els.tools, true);
      setHidden(els.hero, true);
      els.count.textContent = '';
      els.list.setAttribute('aria-busy', 'true');
      render(els.list, skeletonMarkup());
      return;
    }
    if (st.error) {
      dropMenu();
      setHidden(els.tools, true);
      setHidden(els.hero, true);
      els.count.textContent = '';
      render(els.list, html``);
      // The header Add is hidden: the state's own button (Try again, Open Settings) is the one action.
      const copy = libraryErrorCopy(st.error);
      renderEmpty({ kind: 'error', title: copy.title, text: copy.text, action: copy.action, detail: copy.detail });
      return;
    }
    if (!st.docs.length) {
      dropMenu();
      setAddEmphasis(true);   // "Add your first document" is the primary action then
      setHidden(els.tools, true);
      setHidden(els.hero, true);
      els.count.textContent = '';
      render(els.list, html``);
      renderEmpty({
        kind: 'empty',
        title: 'Your library is empty',
        text: 'Add an EPUB, Markdown, text or HTML file, or paste some text, and Open Speech will read it aloud.',
        action: 'add',
      });
      return;
    }
    setAddEmphasis(false);
    setHidden(els.tools, false);
    renderList();
  }

  /** Render the current state (RL-01…RL-04), keeping keyboard focus where it was. */
  function renderNow() {
    const spot = focusSpot();
    paint();
    if (spot) restoreFocus(spot);
  }

  // Typing filters after a 150 ms pause, never on every keystroke (BUG-21).
  const searchLater = debounce(() => renderNow(), SEARCH_DEBOUNCE_MS);

  /* ───────────── loading ───────────── */

  /**
   * GET the library. Late answers from an older load are dropped (request tokens).
   * @returns {Promise<boolean>} whether this load's answer was applied without an error
   */
  async function load() {
    const token = (st.token += 1);
    const since = Date.now();
    st.loading = true;
    if (!st.loaded) renderNow();
    let docs = null;
    let error = null;
    try {
      const data = await api.get(DOCUMENTS_PATH);
      docs = Array.isArray(data && data.documents) ? data.documents : [];
    } catch (e) {
      error = e;
    }
    if (token !== st.token) return false;
    st.loading = false;
    if (error && error.kind === 'aborted') return false;
    if (error) {
      st.error = error;
    } else {
      st.error = null;
      // The answer can be older than a change made here meanwhile: keep those changes.
      st.docs = mergeLibrary(docs, st.docs, { changed: st.changed, removed: st.removed, since });
      for (const map of [st.changed, st.removed]) {
        for (const [id, at] of map) if (at < since) map.delete(id);   // already in this answer
      }
      st.loaded = true;
      st.loadedAt = Date.now();
    }
    renderNow();
    return !error;
  }

  /** The Continue-listening document on screen now ('' when the card is hidden). */
  function shownHeroId() {
    if (els.hero.hidden) return '';
    const open = els.heroContent.querySelector('[data-action="open"]');
    return open ? open.getAttribute('data-doc') || '' : '';
  }

  /** Update one row's percentage and bar, and the card's meta and bar, without re-rendering. */
  function patchRow(doc) {
    const pct = percentOf(doc);
    const row = Array.from(els.list.querySelectorAll('.rd-row')).find((li) => li.getAttribute('data-doc') === doc.id);
    if (row && pct !== null) {
      const label = row.querySelector('.rd-row__pct');
      const bar = row.querySelector('.rd-row__bar');
      if (label && bar) {
        label.classList.remove('rd-row__pct--new');
        render(label, html`${isFinished(doc) ? html`<span class="visually-hidden">Finished, </span>` : ''}${pct}%`);
        bar.setAttribute('value', String(pct));
      }
    }
    const open = els.heroContent.querySelector('[data-action="open"]');
    if (open && open.getAttribute('data-doc') === doc.id) {
      const meta = els.heroContent.querySelector('.rd-hero__meta');
      const bar = els.heroContent.querySelector('.rd-hero__bar');
      const p = pct ?? 0;
      if (meta) meta.textContent = heroMeta(doc);
      if (bar) {
        bar.setAttribute('value', String(p));
        bar.textContent = `${p}%`;
        bar.setAttribute('aria-label', `${p}% listened`);
      }
    }
  }

  /* ───────────── row actions ───────────── */

  function syncFilterChips() {
    for (const chip of els.filter.querySelectorAll('[data-filter]')) {
      chip.setAttribute('aria-pressed', chip.getAttribute('data-filter') === st.filter ? 'true' : 'false');
    }
  }

  async function saveProgressOf(doc, blockIndex, { done }) {
    await beforeChange(doc);
    try {
      const answer = await api.put(`${docPath(doc.id)}/progress`, progressPayload(doc, blockIndex));
      const progress = (answer && answer.progress) || { ...progressPayload(doc, blockIndex), updated_at: new Date().toISOString() };
      st.docs = st.docs.map((d) => (d.id === doc.id ? { ...d, progress } : d));
      st.changed.set(doc.id, Date.now());
      renderNow();   // focus stays on the row's More button (or its neighbour when the filter drops the row)
      toast(done, { kind: 'success' });
      try { emit('library:changed', { docId: doc.id }); } catch { /* bus not ready */ }
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      const copy = api.explain(error, { title: "Couldn't save the position", notFound: 'That document is no longer in the library' });
      toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
      if (error && error.status === 404) load();
    }
  }

  async function startOver(doc) {
    const ok = await confirm({
      title: 'Start over?',
      message: `“${doc.title || 'This document'}” will start from the beginning next time you play it.`,
      confirmLabel: 'Start over',
    });
    if (!ok) return;
    await saveProgressOf(doc, 0, { done: `“${doc.title || 'Document'}” starts from the beginning` });
  }

  async function markFinished(doc) {
    await saveProgressOf(doc, Number(doc.block_count) || 0, { done: `Marked “${doc.title || 'document'}” as finished` });
  }

  async function remove(doc) {
    const ok = await confirm({
      title: 'Delete document?',
      message: `Delete “${doc.title || 'this document'}” from the library? Its saved position is removed too.`,
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.del(docPath(doc.id));
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      if (!(error && error.status === 404)) {   // already gone: treat it as deleted
        const copy = api.explain(error, 'delete');
        toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
        return;
      }
    }
    try { onDeleted(doc.id); } catch (e) { if (globalThis.console) globalThis.console.error(e); }
    st.docs = st.docs.filter((d) => d.id !== doc.id);
    st.removed.set(doc.id, Date.now());
    renderNow();   // focus moves from the row's More button to the next row
    toast(`Deleted “${doc.title || 'document'}”`, { kind: 'success' });
    try { emit('library:changed', { docId: doc.id }); } catch { /* bus not ready */ }
  }

  function onMenuSelect(item) {
    const doc = st.menuDoc && docById(st.menuDoc);
    if (!doc) return;
    const act = item.getAttribute('data-menu');
    if (act === 'open') onOpen(doc.id);
    else if (act === 'restart') startOver(doc);
    else if (act === 'finish') markFinished(doc);
    else if (act === 'delete') remove(doc);
  }

  /** Point the shared row menu at one row's button (bound on first use) and open it. */
  function openRowMenu(button, focus = 'first') {
    const id = button.getAttribute('data-doc');
    const doc = docById(id);
    if (!doc) return;
    if (!st.menu || st.menu.button !== button) {
      dropMenu();
      st.menu = { button, ctl: bindMenuButton(button, els.menu, { onSelect: onMenuSelect, placement: 'bottom-end' }), id };
    }
    st.menuDoc = id;
    const restart = els.menu.querySelector('[data-menu="restart"]');
    const finish = els.menu.querySelector('[data-menu="finish"]');
    setHidden(restart, !doc.progress);
    setHidden(finish, isFinished(doc));
    els.menu.setAttribute('aria-label', `Actions for ${doc.title || 'document'}`);
    st.menu.ctl.open(focus);
  }

  offs.push(actions(els.library, {
    open: (el) => onOpen(el.getAttribute('data-doc')),
    resume: (el) => onResume(el.getAttribute('data-doc')),
    add: (el) => onAdd(el),
    retry: () => load(),
    'open-settings': () => onOpenSettings(),
    'row-menu': (el) => {
      // Once bound, the menu button toggles itself; this handles the first press.
      if (st.menu && st.menu.button === el) return;
      openRowMenu(el);
    },
  }));

  function onListKeydown(event) {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    const start = event.target;
    const button = start && start.closest ? start.closest('[data-action="row-menu"]') : null;
    if (!button || (st.menu && st.menu.button === button)) return;
    event.preventDefault();
    openRowMenu(button, event.key === 'ArrowUp' ? 'last' : 'first');
  }
  els.list.addEventListener('keydown', onListKeydown);
  offs.push(() => els.list.removeEventListener('keydown', onListKeydown));

  /* ───────────── tools ───────────── */

  function onSearchInput() {
    st.query = els.search.value;
    searchLater();
  }
  function onSearchClear() {
    // The native clear button (or Esc) fires 'search' with an empty value: show everything at once.
    st.query = els.search.value;
    if (!st.query) {
      searchLater.cancel();
      renderNow();
    }
  }
  function onFilterClick(event) {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const chip = start && start.closest ? start.closest('[data-filter]') : null;
    if (!chip) return;
    st.filter = chip.getAttribute('data-filter');
    prefs.set({ libraryFilter: st.filter });
    syncFilterChips();
    renderNow();
  }
  function onSortChange() {
    st.sort = els.sort.value;
    prefs.set({ librarySort: st.sort });
    renderNow();
  }
  els.search.addEventListener('input', onSearchInput);
  els.search.addEventListener('search', onSearchClear);
  els.filter.addEventListener('click', onFilterClick);
  els.sort.addEventListener('change', onSortChange);
  offs.push(() => {
    els.search.removeEventListener('input', onSearchInput);
    els.search.removeEventListener('search', onSearchClear);
    els.filter.removeEventListener('click', onFilterClick);
    els.sort.removeEventListener('change', onSortChange);
  });

  const hintClose = $('rd-hint-close');
  const onHintClose = () => setHidden(els.hint, true);
  hintClose.addEventListener('click', onHintClose);
  offs.push(() => hintClose.removeEventListener('click', onHintClose));

  els.sort.value = st.sort;
  syncFilterChips();

  return {
    load,
    render: renderNow,
    /** Whether the library should reload when it shows (never loaded, or older than 30 s). */
    isStale: (now = Date.now()) => !st.loaded || now - st.loadedAt > STALE_MS,
    get loaded() { return st.loaded; },
    get docs() { return st.docs.slice(); },
    docById,
    titleOf: (id) => (docById(id) || {}).title || '',
    /** The Continue-listening document, or null. */
    heroDoc: () => heroDoc(st.docs),
    /**
     * Replace one document's saved progress (the engine saved it while playing). Usually the row and the card
     * are patched in place, so focus and an open menu stay where they are. When the document changes state (a
     * "New" one is started, or one is finished) or becomes the Continue-listening document, the list renders
     * again (focus is kept).
     */
    updateProgress(id, progress) {
      const before = docById(id);
      if (!before || !progress) return;
      st.docs = st.docs.map((d) => (d.id === id ? { ...d, progress } : d));
      st.changed.set(id, Date.now());
      if (!st.loaded || st.error) return;
      const after = docById(id);
      const hero = heroFor(st.docs, { filter: st.filter, query: st.query });
      if (statusOf(before) !== statusOf(after) || (hero ? hero.id : '') !== shownHeroId()) renderNow();
      else patchRow(after);
    },
    /** Remove a document that no longer exists (an open answered 404). */
    forget(id) {
      if (!docById(id)) return;
      st.docs = st.docs.filter((d) => d.id !== id);
      st.removed.set(id, Date.now());
      renderNow();
    },
    focusSearch() {
      if (els.tools.hidden) return false;
      els.search.focus();
      if (typeof els.search.select === 'function') els.search.select();
      return true;
    },
    /** The appearance hint for a Settings hand-off with no document to show it over. */
    showHint() {
      // The banner is its own status region: its text goes in after it shows, so it is announced once.
      const msg = els.hint.querySelector('#rd-hint-msg');
      const text = HINT_TEXT;
      msg.textContent = '';
      setHidden(els.hint, false);
      nextFrame().then(() => { msg.textContent = text; });
    },
    hideHint: () => setHidden(els.hint, true),
    openRowMenu,
    closeMenu() { if (st.menu) st.menu.ctl.close({ restoreFocus: false }); },
    /** Screenshot hook: show the loading or the error state without a request (any load in flight is dropped). */
    preview(state) {
      st.token += 1;
      if (state === 'loading') {
        st.loaded = false;
        st.loading = true;
        st.error = null;
      } else if (state === 'error') {
        st.loading = false;
        st.error = { status: 404, message: 'Not Found' };   // READER_ENABLED=false: no Reader routes
      }
      renderNow();
    },
    destroy() {
      searchLater.cancel();
      dropMenu();
      for (const off of offs.splice(0)) off();
    },
  };
}
