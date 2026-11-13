/**
 * The reading surface: the document text on pages of whole chapters, the read-along highlight
 * (.rd-p.is-current and .rd-s.is-now, with sentence spans built lazily for the current paragraph only), follow
 * mode with "Back to playing", play-from-paragraph by click, Enter or Space, the top bar and the aside's meta.
 *
 * It drives playback only through the engine API: playFrom, toggle and the 'cursor' events. Commands go
 * through `ensureEngine()`, which loads the document into the engine when it holds another one (or none); reads
 * go through `getEngine()`, which never changes anything. The engine builds a fresh state snapshot on every
 * read, so the view passes in the one it already has (`applyCursor(cursor, state)`, `refreshMeta(state)`).
 * @module views/reader/reading
 */
import { html, render, setHidden, isEditable, nextFrame } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { applyTile, monogram } from '../../core/tiles.js';
import { reducedMotion } from '../../core/theme.js';
import {
  pagesFor, pageOf, chapterMap, pageLinkLabel, sentencePieces, charOffsets, readingPercent, sideMeta, scrollPlan,
  AUTH_TEXT,
} from './text-logic.js';

const BAND_MARGIN = 16;   // px kept clear above and below the spoken sentence
const SCROLL_KEYS = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End']);
// Scrolling inside these never counts as "the listener scrolled away".
const OWN_SCROLLERS = 'dialog, .rd-side, .rd-topbar, .rdp-transport, #rd-player-slot, .menu';

function px(name) {
  try {
    const value = parseFloat(document.documentElement.style.getPropertyValue(name));
    return Number.isFinite(value) ? value : 0;
  } catch {
    return 0;
  }
}

function selectionText() {
  try {
    const sel = typeof globalThis.getSelection === 'function' ? globalThis.getSelection() : null;
    return sel ? String(sel) : '';
  } catch {
    return '';
  }
}

/**
 * Bind the reading surface inside the Reader's root.
 * @param {HTMLElement} root the Reader view section
 * @param {{getEngine: () => object|null, ensureEngine?: () => object|null, autoScroll?: () => boolean,
 *   onChapter?: (position: number) => void}} options
 */
export function createReading(root, { getEngine, ensureEngine, autoScroll = () => true, onChapter = () => {} } = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    reading: $('rd-reading'),
    layout: $('rd-layout'),
    text: $('rd-text'),
    title: $('rd-doc-title'),
    chapter: $('rd-top-chapter'),
    side: $('rd-side'),
    sideTile: $('rd-side-tile'),
    sideTitle: $('rd-side-title'),
    sideMeta: $('rd-side-meta'),
    ret: $('rd-return'),
    error: $('rd-open-error'),
  };
  const st = {
    doc: null,
    pages: [],
    page: -1,
    view: [0, 0],
    blockEls: [],
    chapterAt: [],
    offsets: [0],
    cursor: 0,
    sentence: -1,
    chapter: -1,
    follow: true,
    browsed: false,   // the listener scrolled since the last jump (while paused, that leaves follow on)
    playing: false,
    visible: false,
    finishedAtOpen: false,
    moved: false,
  };
  const offs = [];

  const engine = () => (typeof getEngine === 'function' ? getEngine() : null);
  const commandEngine = () => (typeof ensureEngine === 'function' ? ensureEngine() : engine());
  const count = () => (st.doc ? st.doc.blocks.length : 0);
  /** One snapshot of the engine's state (each read of engine.state builds a new one). */
  const engineState = () => {
    const e = engine();
    return e ? e.state || null : null;
  };
  const ours = (state) => !!(state && st.doc && state.docId === st.doc.id);

  function blockEl(index) {
    const [start, end] = st.view;
    return index >= start && index < end ? st.blockEls[index - start] || null : null;
  }

  function blockMarkup(block, index) {
    if (block && block.kind === 'heading') {
      return html`<h2 class="rd-p rd-chapter-title" data-block="${index}" tabindex="-1">${block.text || ''}</h2>`;
    }
    return html`<p class="rd-p" data-block="${index}" tabindex="-1">${(block && block.text) || ''}</p>`;
  }

  function pageLink(page, next) {
    const label = pageLinkLabel({ pages: st.pages, page, next, chapterAt: st.chapterAt, chapters: st.doc.chapters });
    const cls = next ? 'rd-page-link rd-page-link--next' : 'rd-page-link rd-page-link--prev';
    return next
      ? html`<button type="button" class="btn btn--secondary ${cls}" data-page="${page}"><span class="btn__label">${label}</span>${icon('chevron-right')}</button>`
      : html`<button type="button" class="btn btn--secondary ${cls}" data-page="${page}">${icon('chevron-left')}<span class="btn__label">${label}</span></button>`;
  }

  /** Lay out one page (only one is ever in the DOM). Returns true when it replaced the page on screen. */
  function showPage(page) {
    if (!st.doc || page === st.page || !st.pages[page]) return false;
    const [start, end] = st.pages[page];
    const parts = [];
    if (page > 0) parts.push(pageLink(page - 1, false));
    for (let i = start; i < end; i += 1) parts.push(blockMarkup(st.doc.blocks[i], i));
    if (page + 1 < st.pages.length) parts.push(pageLink(page + 1, true));
    render(els.text, parts);
    st.page = page;
    st.view = [start, end];
    st.blockEls = Array.from(els.text.querySelectorAll('[data-block]'));
    const el = blockEl(st.cursor);
    if (el) {
      el.classList.add('is-current');
      el.tabIndex = 0;
      if (st.sentence >= 0) markSentence(el, st.cursor, st.sentence);
    }
    return true;
  }

  /** Build the sentence spans of one paragraph, once, from code-point offsets (RE-52). */
  function ensureSpans(el, index) {
    if (!el || el.dataset.split) return;
    el.dataset.split = '1';
    const block = st.doc && st.doc.blocks[index];
    if (!block || !Array.isArray(block.sentences) || !block.sentences.length) return;
    const pieces = sentencePieces(block.text || '', block.sentences);
    render(el, pieces.map((p) => (p.s === null ? html`${p.text}` : html`<span class="rd-s" data-s="${p.s}">${p.text}</span>`)));
  }

  function markSentence(el, index, sentence) {
    ensureSpans(el, index);
    const span = el.querySelector(`.rd-s[data-s="${sentence}"]`);
    if (span) span.classList.add('is-now');
  }

  function clearSentence(el) {
    const now = el && el.querySelector('.rd-s.is-now');
    if (now) now.classList.remove('is-now');
  }

  /** The band between the reader's top chrome and the transport, from the shell's chrome metrics. */
  function band() {
    const viewport = globalThis.innerHeight || 800;
    const top = px('--chrome-top') + BAND_MARGIN;
    const bottom = viewport - px('--chrome-bottom') - BAND_MARGIN;
    return { bandTop: top, bandBottom: Math.max(top + 40, bottom), viewport };
  }

  function targetRect() {
    const block = blockEl(st.cursor);
    if (!block) return null;
    return (block.querySelector('.rd-s.is-now') || block).getBoundingClientRect();
  }

  /**
   * After an instant jump, off-screen paragraphs that were only estimated (content-visibility: auto) get their
   * real height and push the target away: check it on the next few frames and correct it, instantly, whenever
   * it left the band (the layout can still move a frame later, so a check that finds it in place goes on).
   */
  let settleToken = 0;
  function settleScroll() {
    const token = (settleToken += 1);
    let rounds = 0;
    const step = () => {
      if (token !== settleToken || !st.visible) return;
      rounds += 1;
      const plan = scrollPlan(targetRect(), band(), { reduced: true });
      if (plan && typeof window.scrollBy === 'function') window.scrollBy({ top: plan.delta, behavior: 'auto' });
      if (rounds < 5) nextFrame().then(step);
    };
    nextFrame().then(step);
  }

  /** Scroll the spoken sentence (or its paragraph) into the band (RE-53); far jumps and reduced motion jump (RE-54). */
  function keepInView({ force = false } = {}) {
    if (!st.visible) return;
    const plan = scrollPlan(targetRect(), band(), { force, reduced: reducedMotion() });
    if (!plan || typeof window.scrollBy !== 'function') return;
    window.scrollBy({ top: plan.delta, behavior: plan.behavior });
    if (plan.behavior === 'auto') settleScroll();
  }

  /** "Back to playing" shows while playing once the listener scrolled away, or always with auto-scroll off. */
  function updateReturn() {
    const away = !st.follow || !autoScroll();
    setHidden(els.ret, !away || !st.playing || !st.visible || !st.doc);
  }

  /**
   * The aside's "Author · 34% · ~2 h left"; it agrees with the library (finished = 100%, RE-62).
   * @param {object|null} [state] the engine snapshot the caller already has (read once when left out)
   */
  function updateMeta(state) {
    if (!st.doc) return;
    const s = state === undefined ? engineState() : state;
    const mine = ours(s);
    const done = (mine && !!s.finished) || (st.finishedAtOpen && !st.moved);
    const n = count();
    const rate = (mine && Number(s.speed)) || Number(st.doc.progress && st.doc.progress.speed) || 1;
    const text = sideMeta({
      author: st.doc.author || '',
      pct: readingPercent({ cursor: st.cursor, count: n, finished: done }),
      finished: done,
      charCount: Number(st.doc.char_count) || st.offsets[n] || 0,   // the library's measure (RE-62)
      rate,
    });
    if (els.sideMeta.textContent !== text) els.sideMeta.textContent = text;
  }

  function setChapter(position) {
    if (position === st.chapter) return;
    st.chapter = position;
    const chapter = st.doc && st.doc.chapters[position];
    els.chapter.textContent = (chapter && chapter.title) || '';
    try { onChapter(position); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
  }

  /**
   * Apply an engine cursor (the engine's 'cursor' event): the current paragraph and sentence, the page, and the scroll.
   * reason 'open' | 'jump' pages and scrolls there and turns follow on; 'playback' scrolls only while following.
   * @param {{block?: number, sentence?: number, reason?: string}} cursor
   * @param {object|null} [state] the engine snapshot the caller already read (for the meta)
   */
  function applyCursor({ block = 0, sentence = -1, reason = 'playback' } = {}, state) {
    if (!st.doc || !count()) return;
    const index = Math.max(0, Math.min(count() - 1, Number(block) || 0));
    const jump = reason === 'open' || reason === 'jump';
    if (reason !== 'open') st.moved = true;
    if (jump) st.follow = true;
    const following = jump || (st.follow && autoScroll());
    if (following) st.browsed = false;   // the view goes back to the spoken sentence
    const turned = following && showPage(pageOf(st.pages, index));
    const el = blockEl(index);
    if (index !== st.cursor || (el && !el.classList.contains('is-current'))) {
      const previous = blockEl(st.cursor);
      if (previous) {
        previous.classList.remove('is-current');
        previous.tabIndex = -1;
        clearSentence(previous);
      }
      st.cursor = index;
      st.sentence = -1;
      if (el) {
        el.classList.add('is-current');
        el.tabIndex = 0;
      }
    }
    const s = Number.isInteger(sentence) ? sentence : -1;
    if (s !== st.sentence) {
      clearSentence(el);
      if (s >= 0 && el) markSentence(el, index, s);
      st.sentence = s;
    }
    setChapter(st.chapterAt[index] ?? 0);
    updateMeta(state);
    if (following) keepInView({ force: jump || turned });
    updateReturn();
  }

  /**
   * The chrome around the text changed (the transport arrived or changed size): bring the spoken sentence back
   * into the band, unless the listener has scrolled away (while paused, scrolling leaves follow on).
   */
  function keepClear() {
    if (st.follow && !st.browsed) keepInView();
  }

  /** Browse to another page without moving playback: follow stops, focus goes to the opposite edge link. */
  function turnPage(page) {
    const next = page > st.page;
    if (!showPage(page)) return;
    st.follow = false;
    updateReturn();
    if (typeof window.scrollTo === 'function') window.scrollTo(0, next ? 0 : document.documentElement.scrollHeight || 0);
    const link = els.text.querySelector(next ? '.rd-page-link--prev' : '.rd-page-link--next');
    if (link) link.focus({ preventScroll: true });
  }

  /** "Back to playing": follow again and bring the spoken sentence back into view. */
  function returnToPlaying() {
    st.follow = true;
    st.browsed = false;
    showPage(pageOf(st.pages, st.cursor));
    keepInView({ force: true });
    updateReturn();
    const el = blockEl(st.cursor);
    if (el) el.focus({ preventScroll: true });
  }

  /** The listener scrolled the page: while playing that stops following (RE-53); while paused it is only noted. */
  function noteManualScroll(event) {
    if (!st.visible) return;
    const target = event && event.target;
    if (target && target.closest && target.closest(OWN_SCROLLERS)) return;
    st.browsed = true;
    if (!st.playing || !st.follow) return;
    st.follow = false;
    updateReturn();
  }

  function playFrom(index) {
    const e = commandEngine();
    if (e && typeof e.playFrom === 'function') e.playFrom(index);
  }

  function onTextClick(event) {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    if (!start || !start.closest) return;
    const link = start.closest('[data-page]');
    if (link) {
      turnPage(Number(link.getAttribute('data-page')));
      return;
    }
    const paragraph = start.closest('[data-block]');
    if (!paragraph || selectionText().length) return;   // selecting text never starts playback
    playFrom(Number(paragraph.getAttribute('data-block')));
  }

  /**
   * Enter on a focused paragraph plays from it; Space toggles play on the current paragraph and plays from any
   * other (RE-56). Both call preventDefault, so the player's global Space shortcut does not fire too (REV-17).
   */
  function onTextKeydown(event) {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    if (!target || !target.matches || !target.matches('[data-block]')) return;
    const index = Number(target.getAttribute('data-block'));
    if (event.key === 'Enter') {
      event.preventDefault();
      playFrom(index);
    } else if (event.key === ' ' || event.key === 'Spacebar') {
      event.preventDefault();
      const e = commandEngine();
      const current = index === st.cursor && !!e && ours(e.state);
      if (current && typeof e.toggle === 'function') {
        if (typeof e.unlock === 'function') e.unlock();
        e.toggle();
      } else {
        playFrom(index);
      }
    }
  }

  function onKeydown(event) {
    if (!SCROLL_KEYS.has(event.key) || event.ctrlKey || event.metaKey || event.altKey) return;
    if (isEditable(event.target)) return;
    noteManualScroll(event);
  }

  els.text.addEventListener('click', onTextClick);
  els.text.addEventListener('keydown', onTextKeydown);
  els.ret.addEventListener('click', returnToPlaying);
  offs.push(() => {
    els.text.removeEventListener('click', onTextClick);
    els.text.removeEventListener('keydown', onTextKeydown);
    els.ret.removeEventListener('click', returnToPlaying);
  });

  let windowOffs = [];
  function bindWindow() {
    if (windowOffs.length) return;
    const opts = { passive: true };
    window.addEventListener('wheel', noteManualScroll, opts);
    window.addEventListener('touchmove', noteManualScroll, opts);
    document.addEventListener('keydown', onKeydown);
    windowOffs = [
      () => window.removeEventListener('wheel', noteManualScroll, opts),
      () => window.removeEventListener('touchmove', noteManualScroll, opts),
      () => document.removeEventListener('keydown', onKeydown),
    ];
  }
  function unbindWindow() {
    for (const off of windowOffs) off();
    windowOffs = [];
  }

  /** The loading state while a document is fetched: the top bar with the known title and a quiet skeleton. */
  function showLoading(title = '') {
    st.doc = null;
    st.page = -1;
    st.view = [0, 0];
    st.blockEls = [];
    els.title.textContent = title || 'Opening…';
    els.chapter.textContent = '';
    els.sideTitle.textContent = title || '';
    els.sideMeta.textContent = '';
    setHidden(els.error, true);
    setHidden(els.layout, false);
    els.text.setAttribute('aria-busy', 'true');
    render(els.text, html`<div class="rd-text__loading" aria-hidden="true"><span class="skeleton skeleton--title"></span><span class="skeleton skeleton--line"></span><span class="skeleton skeleton--line"></span><span class="skeleton skeleton--line rd-skel--short"></span></div>`);
    updateReturn();
  }

  /** Show a loaded document: the first page is laid out by the first cursor. */
  function setDoc(doc) {
    const blocks = Array.isArray(doc.blocks) ? doc.blocks : [];
    const chapters = Array.isArray(doc.chapters) ? doc.chapters : [];
    st.doc = { ...doc, blocks, chapters };
    st.pages = pagesFor(blocks.length, chapters);
    st.chapterAt = chapterMap(blocks.length, chapters);
    st.offsets = charOffsets(blocks);
    st.page = -1;
    st.view = [0, 0];
    st.blockEls = [];
    st.cursor = 0;
    st.sentence = -1;
    st.chapter = -1;
    st.follow = true;
    st.browsed = false;
    st.moved = false;
    const p = doc.progress;
    st.finishedAtOpen = !!p && blocks.length > 0 && Number(p.block_index) >= blocks.length;
    els.text.removeAttribute('aria-busy');
    setHidden(els.error, true);
    setHidden(els.layout, false);
    els.title.textContent = doc.title || 'Untitled';
    els.sideTitle.textContent = doc.title || 'Untitled';
    els.sideTile.textContent = monogram(doc.title);
    applyTile(els.sideTile, doc.id || doc.title || '');
    if (!blocks.length) {
      render(els.text, html`<p class="rd-text__empty">This document has no readable text.</p>`);
      els.chapter.textContent = '';
    } else {
      render(els.text, []);
      setChapter(st.chapterAt[0] ?? 0);
    }
    updateMeta();
    updateReturn();
  }

  /**
   * The open-failed state: a banner, never only a toast. A refused or locked-out API key
   * says "API key needed" with Open Settings, since Retry can't succeed until the key changes; anything
   * else offers Retry.
   * @param {{title: string, message: string, detailText: string, action?: string}} copy api.explain() output
   */
  function showError(copy, { title = '' } = {}) {
    st.doc = null;
    els.text.removeAttribute('aria-busy');
    render(els.text, []);
    setHidden(els.layout, true);   // no empty page or document card under the banner
    els.title.textContent = title || "Couldn't open the document";
    els.chapter.textContent = '';
    const auth = copy.action === 'open-settings';
    const action = auth
      ? html`<button type="button" class="btn btn--secondary btn--sm" data-action="open-settings">${icon('settings')}<span class="btn__label">Open Settings</span></button>`
      : html`<button type="button" class="btn btn--secondary btn--sm" data-action="open-retry">${icon('retry')}<span class="btn__label">Retry</span></button>`;
    // Details read as the server's message and code, never its JSON envelope. An auth refusal's
    // details (and a lockout's countdown) are the connection banner's to give.
    const detail = auth ? '' : copy.detailText || '';
    render(els.error, html`${icon(auth ? 'key' : 'alert', { className: 'ico banner__icon' })}
      <div class="banner__body">
        <p class="banner__title">${auth ? 'API key needed' : copy.title}</p>
        <p class="banner__msg">${auth ? AUTH_TEXT.document : copy.message}</p>
        ${detail ? html`<details class="banner__details"><summary>Details</summary><pre class="rd-open-error__detail"></pre></details>` : ''}
      </div>
      <div class="banner__actions">${action}</div>`);
    const pre = els.error.querySelector('pre');
    if (pre) pre.textContent = detail;   // details as text only
    setHidden(els.error, false);
    updateReturn();
  }

  return {
    showLoading,
    setDoc,
    showError,
    applyCursor,
    keepInView,
    keepClear,
    returnToPlaying,
    turnPage,
    /** The document on screen (with its blocks), or null. */
    doc: () => st.doc,
    /** Follow mode and playing state (for tests and the view). */
    get follow() { return st.follow; },
    get cursor() { return st.cursor; },
    /** The sentence of the current paragraph being read, or the last one read there; -1 while none has been (the
     *  engine cursor's `sentence`, for Ask AI's "This sentence"). */
    get sentence() { return st.sentence; },
    setPlaying(playing) {
      st.playing = !!playing;
      updateReturn();
    },
    /** The listener browses away from the spoken sentence (also the screenshot hook for the Return pill). */
    stopFollowing() {
      st.follow = false;
      updateReturn();
    },
    /** Auto-scroll was switched on or off in the appearance sheet. */
    refreshReturn: updateReturn,
    /** The meta after a speed change or the end: pass the engine snapshot in hand (else it is read once). */
    refreshMeta: (state) => updateMeta(state),
    /** Whether the listener scrolled since the last jump (for tests). */
    get browsed() { return st.browsed; },
    /** The reading view is (or is no longer) the one on screen: binds the scroll listeners only while it is. */
    setVisible(visible) {
      st.visible = !!visible;
      if (st.visible) bindWindow();
      else unbindWindow();
      updateReturn();
    },
    /** Focus the current paragraph without scrolling. */
    focusCurrent() {
      const el = blockEl(st.cursor);
      if (el) el.focus({ preventScroll: true });
    },
    clear() {
      st.doc = null;
      st.page = -1;
      st.view = [0, 0];
      st.blockEls = [];
      render(els.text, []);
      updateReturn();
    },
    destroy() {
      unbindWindow();
      for (const off of offs.splice(0)) off();
    },
  };
}
