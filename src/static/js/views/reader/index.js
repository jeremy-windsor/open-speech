/**
 * Reader: the view controller for `#reader` (the library) and `#reader/<id>` (the reading view).
 *
 * - Opening a document (RE-59…RE-61): an "Opening “T”…" notice only after 150 ms, a ticket per open so a
 *   slower earlier open never replaces a later one, two frames for the notice to paint before a long layout,
 *   404 → a toast and the hash back to #reader, an auth refusal → "API key needed" with Open Settings,
 *   any other failure → a banner with Retry. Opening the document already on screen just shows it (and plays
 *   when Resume asked). The document request never waits for the engine module.
 * - Back to library saves progress and keeps playing; the mini-player then shows the Reader. It steps
 *   back in history only when the entry before is the library (RL-13).
 * - Playback goes through the engine API (views/reader/engine.js) and the player slot is player.js's
 *   `mountPlayer()`. Both are imported lazily and injectable (tests use tests/js/reader/fake-engine.js). The
 *   engine builds a fresh state snapshot on every read, so each event handler reads it at most once.
 * - The `open-appearance` intent (from Settings) opens the appearance sheet over the last opened
 *   document, or shows the library hint when there is none.
 * - Leaving the Reader (or switching between the library and a document) closes the sheets of the part that
 *   goes away, so no modal sheet stays over another view.
 * - Below 1180 px the transport's Chapters chip is the one Chapters control: `#rd-reading`'s
 *   `data-rd-transport` says whether the transport is there ("ready") or can't be ("none": the engine or the
 *   player didn't load), and reader.css shows the top bar's Chapters button only for "none".
 * - The view definition's prefetch() warms the engine, the player and its partial
 *   on idle, so Resume can unlock audio inside its click and the transport arrives with the document.
 * - Screenshot states (`#reader…?shot=`): add, add-paste, row-menu, hint, loading, error, opening,
 *   appearance, chapters, return, highlight (the first sentence of the current paragraph as if it were spoken),
 *   play (the same highlight under the player's playing transport), and Ask AI's: the component's states plus
 *   ask-ai-sentence, ask-ai-paragraph, ask-ai-chapter and ask-ai-document, the sheet with that source chosen.
 *   Like every shot state they only show: a link to one never renders, plays or
 *   saves.
 * @module views/reader/index
 */
import { setHidden, nextFrame, actions } from '../../core/dom.js';
import * as api from '../../core/api.js';
import { on as onEvent, takeIntent } from '../../core/store.js';
import * as router from '../../core/router.js';
import { toast } from '../../core/toast.js';
import { closeSheet } from '../../core/dialogs.js';
import { register } from '../../core/shortcuts.js';
import { refreshTitle, loadPartial } from '../../shell/view-host.js';
import { refresh as refreshChrome } from '../../shell/chrome.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import { createLibrary, DOCUMENTS_PATH } from './library.js';
import { createReading } from './reading.js';
import { createAdd } from './add.js';
import { createAppearance } from './appearance.js';
import { createChapters } from './chapters.js';
import { bindAskButton, openAskAi, openSample, SHOTS as ASK_SHOTS } from '../../components/ask-ai.js';
import { readerSources, selectionRange, defaultSource } from './ask-sources.js';

/** The opening notice waits this long, so a quick open never flashes it (RE-59). */
export const OPENING_NOTICE_MS = 150;
/** The Reader's own Ask AI states: the sheet with that source chosen. */
const ASK_SOURCE_SHOTS = Object.freeze({ 'ask-ai-sentence': 'sentence', 'ask-ai-paragraph': 'paragraph', 'ask-ai-chapter': 'chapter', 'ask-ai-document': 'document' });
/** The view's own screenshot states (`?shot=`); the shell handles the app-* ones. */
export const SHOTS = Object.freeze(['add', 'add-paste', 'row-menu', 'hint', 'loading', 'error', 'opening', 'appearance', 'chapters', 'return', 'highlight', 'play', ...ASK_SHOTS, ...Object.keys(ASK_SOURCE_SHOTS)]);
const DOC_SHOTS = new Set(['opening', 'appearance', 'chapters', 'return', 'highlight', 'play', ...ASK_SHOTS, ...Object.keys(ASK_SOURCE_SHOTS)]);

const NOT_FOUND = 'That document is no longer in the library';

/** The real engine and player, imported on first use. A missing module leaves the Reader read-only. */
export const DEFAULT_DEPS = Object.freeze({
  loadEngine: () => import('./engine.js').then((m) => (m && typeof m.getEngine === 'function' ? m.getEngine() : null)),
  mountPlayer: (slot, engine, options) => import('./player.js').then((m) => (m && typeof m.mountPlayer === 'function' ? m.mountPlayer(slot, engine, options) : null)),
});

const docRoute = (id) => `reader/${encodeURIComponent(id)}`;

function twoFrames() {
  // The first frame runs before the paint, the second after it; the timer covers a background tab.
  return Promise.race([nextFrame().then(() => nextFrame()), new Promise((resolve) => setTimeout(resolve, 100))]);
}

function logError(error) {
  if (globalThis.console) globalThis.console.error(error);
}

/** The last document opened in this page load (for the appearance hand-off). */
let lastDocId = '';

/**
 * Build the Reader controller.
 * @param {HTMLElement} root the view section (with partials/reader.html rendered into it)
 * @param {{loadEngine?: () => Promise<object|null>, mountPlayer?: (slot: Element, engine: object, options: object) => *}} [deps]
 */
export function createReaderController(root, deps = {}) {
  const d = { ...DEFAULT_DEPS, ...deps };
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    library: $('rd-library'),
    reading: $('rd-reading'),
    openingCard: $('rd-opening-card'),
    openingText: $('rd-opening-text'),
    back: $('rd-back'),
    appearanceBtn: $('rd-appearance-btn'),
    chaptersBtn: $('rd-chapters-btn'),
    slot: $('rd-player-slot'),
    add: $('rd-add'),
    ask: $('rd-ask'),
    text: $('rd-text'),
  };
  const st = {
    mode: 'library',
    visible: false,
    route: null,
    engine: null,
    enginePromise: null,
    player: null,           // the mountPlayer() promise
    transport: null,        // the player's chrome element whose size is watched
    sizeWatcher: null,      // its ResizeObserver
    doc: null,              // the document on the reading surface (with its blocks)
    openId: '',
    ticket: 0,
    openTimer: null,
    opening: false,
    autoplay: '',           // a document id to play once it is open (Resume)
    libraryDirty: false,
    libraryLoad: null,
    backIsHistory: false,   // the reading view was entered from the library: Back pops that entry
    redirect: '',           // the document the appearance hand-off put in place of the library entry
    scrollY: 0,
    pendingShot: '',
    pendingAppearance: false,
    shotPlaying: false,
    chapter: -1,            // the chapter on screen (the reading surface's onChapter), for Ask AI
    askSelection: null,     // the selection as the Ask AI press began (a tap can clear it before the click)
  };
  const offs = [];

  /* ───────────── engine ───────────── */

  function attachEngine(engine) {
    st.engine = engine;
    // Every read of engine.state builds a fresh snapshot (the real engine walks the timeline for it), and
    // 'state' arrives once per animation frame while playing: each handler reads it at most once.
    const mine = (state) => !!(st.doc && state && state.docId === st.doc.id);
    offs.push(engine.on('cursor', (cursor) => {
      const state = engine.state;
      if (mine(state)) reading.applyCursor(cursor || {}, state);
    }));
    offs.push(engine.on('state', (state) => {
      const ours = mine(state);
      reading.setPlaying(st.shotPlaying || (ours && !!state.playing));
      if (ours && st.visible) reading.refreshMeta(state);
    }));
    offs.push(engine.on('finished', () => {
      const state = engine.state;
      if (mine(state)) reading.refreshMeta(state);
      st.libraryDirty = true;
    }));
    // The engine also reports each saved position: the library rows follow without a reload.
    offs.push(engine.on('progress', (detail) => {
      if (detail && detail.docId && detail.progress) library.updateProgress(detail.docId, detail.progress);
    }));
  }

  /** The engine (loaded once, lazily). Resolves null when it isn't available. */
  function engineReady() {
    if (!st.enginePromise) {
      st.enginePromise = Promise.resolve()
        .then(() => d.loadEngine())
        .then((engine) => {
          if (engine && typeof engine.on === 'function') attachEngine(engine);
          return st.engine;
        }, (error) => {
          if (globalThis.console) globalThis.console.warn('Reader playback is not available:', error);
          return null;
        });
    }
    return st.enginePromise;
  }

  /** The engine with the reading surface's document loaded in it (it may hold another one, or none). */
  function ensureEngine() {
    const engine = st.engine;
    if (!engine || !st.doc) return engine;
    try {
      const state = engine.state;
      if (!state || state.docId !== st.doc.id) engine.load(st.doc);
    } catch (error) {
      logError(error);
    }
    return engine;
  }

  /**
   * Whether the transport (and its Chapters chip) is in the slot: "ready", or "none" when it can't be (the engine
   * or the player didn't load, or the player's partial failed and shows Retry). Until then the attribute is
   * absent. Below 1180 px the top bar's Chapters button stands in only for "none" (reader.css).
   */
  function markTransport(state) {
    if (els.reading.getAttribute('data-rd-transport') !== state) els.reading.setAttribute('data-rd-transport', state);
  }

  /** The player's chrome changed size (it mounted, wrapped, showed or hid): measure it, then keep the spoken
   *  paragraph clear of it unless the listener has scrolled away. */
  let chromeTimer = null;
  function chromeChanged() {
    refreshChrome();
    if (chromeTimer !== null) return;
    chromeTimer = setTimeout(() => {
      chromeTimer = null;
      if (st.visible && st.mode === 'reading') reading.keepClear();
    }, 50);
  }

  /**
   * Watch the transport's size, never its contents: its times tick every second while playing, and a re-measure
   * forces a layout. The shell measures data-chrome elements on route changes, but the transport arrives later,
   * whenever the player's mountPlayer() inserts it.
   */
  function watchTransport() {
    const transport = els.slot.querySelector('[data-chrome]');
    if (transport) markTransport(transport.querySelector('#rdp-chapters') ? 'ready' : 'none');
    if (!transport || transport === st.transport) return;
    st.transport = transport;
    if (st.sizeWatcher) st.sizeWatcher.disconnect();
    st.sizeWatcher = typeof ResizeObserver === 'function' ? new ResizeObserver(() => chromeChanged()) : null;
    if (st.sizeWatcher) st.sizeWatcher.observe(transport);
    chromeChanged();
  }

  function mountPlayerOnce() {
    if (st.player) return;
    if (!st.engine) {
      markTransport('none');   // read-only Reader: the top bar keeps the Chapters button
      return;
    }
    if (typeof MutationObserver === 'function') {
      // The slot's own children only (the player's partial going in), not the subtree.
      const watcher = new MutationObserver(() => watchTransport());
      watcher.observe(els.slot, { childList: true });
      offs.push(() => watcher.disconnect());
    }
    offs.push(() => {
      if (st.sizeWatcher) st.sizeWatcher.disconnect();
      st.sizeWatcher = null;
    });
    st.player = Promise.resolve()
      .then(() => d.mountPlayer(els.slot, st.engine, {
        onOpenChapters: (trigger) => chapters.open(trigger),
        onOpenAppearance: (trigger) => appearance.open(trigger),
      }))
      // mountPlayer() returns {ready, …}: ready resolves once its partial (and the transport) is in the slot.
      .then((player) => (player && typeof player.then !== 'function' && player.ready ? player.ready : player))
      .then(() => watchTransport())
      .catch((error) => {
        if (!els.slot.querySelector('#rdp-chapters')) markTransport('none');
        if (globalThis.console) globalThis.console.warn('The Reader player did not load:', error);
      });
  }

  /* ───────────── parts ───────────── */

  const appearance = createAppearance(root, { onChange: () => reading.refreshReturn() });
  const chapters = createChapters(root, {
    onPick: (position, chapter) => {
      const engine = ensureEngine();
      const first = Number(chapter.first_block) || 0;
      if (engine && typeof engine.playFrom === 'function') engine.playFrom(first);
      else reading.applyCursor({ block: first, reason: 'jump' });
    },
  });
  const reading = createReading(root, {
    getEngine: () => st.engine,
    ensureEngine,
    autoScroll: () => appearance.autoScroll(),
    onChapter: (position) => {
      st.chapter = position;
      chapters.setCurrent(position);
    },
  });
  const add = createAdd(root, {
    onAdded: () => {
      st.libraryLoad = library.load();
    },
  });

  /** Stop and forget a document in the engine ({forget: true} skips its save and drops it), so a stale save
   *  can't overwrite a new position and the next open loads it fresh. */
  function forgetInEngine(id) {
    const engine = st.engine;
    if (!engine) return Promise.resolve();
    try {
      const state = engine.state;
      if (!state || state.docId !== id) return Promise.resolve();
      return Promise.resolve(engine.unload({ forget: true })).catch(logError);
    } catch (error) {
      logError(error);
      return Promise.resolve();
    }
  }

  /** Before Start over or Mark as finished: the engine forgets the document and the surface drops its copy. */
  async function releaseDoc(id) {
    await forgetInEngine(id);
    if (st.doc && st.doc.id === id) {
      st.doc = null;
      reading.clear();
    }
  }

  const library = createLibrary(root, {
    onOpen: (id) => router.navigate(docRoute(id)),
    onResume: (id) => {
      // Unlock audio inside the click (iOS), then open and play (RL-05).
      if (st.engine && typeof st.engine.unlock === 'function') {
        try { st.engine.unlock(); } catch (error) { logError(error); }
      }
      st.autoplay = id;
      if (st.route && st.route.id === id) openDoc(id, { autoplay: true });
      else router.navigate(docRoute(id));
    },
    onAdd: (trigger) => add.open(trigger),
    onOpenSettings: () => openSettings(),
    beforeChange: (doc) => releaseDoc(doc.id),
    onDeleted: (id) => {
      // Deleting the open document stops its playback (RL-10).
      forgetInEngine(id);
      if (st.doc && st.doc.id === id) {
        st.doc = null;
        reading.clear();
      }
      if (lastDocId === id) lastDocId = '';
    },
  });

  function openSettings() {
    try {
      const done = openConnectionSettings();
      if (done && typeof done.catch === 'function') done.catch(logError);
    } catch (error) {
      logError(error);
    }
  }

  /* ───────────── modes ───────────── */

  /** Close the open sheets inside `container` (all of the Reader's when none is given). */
  function closeSheets(container = root) {
    for (const dialog of container.querySelectorAll('dialog')) {
      if (dialog.open) closeSheet(dialog);
    }
  }

  function showMode(mode) {
    if (mode !== st.mode) {
      // The sheets of the part going away close with it (the Add sheet over a document makes no sense, nor the
      // appearance, chapters or player sheets over the library).
      if (mode === 'reading') {
        add.close();
        library.closeMenu();
      } else {
        appearance.close();
        chapters.close();
        closeSheets(els.reading);   // the player's sheets live in the slot
      }
    }
    st.mode = mode;
    setHidden(els.library, mode !== 'library');
    setHidden(els.reading, mode !== 'reading');
    reading.setVisible(st.visible && mode === 'reading');
  }

  function showOpening(title) {
    els.openingText.textContent = title ? `Opening “${title}”…` : 'Opening…';
    setHidden(els.openingCard, false);
  }

  function hideOpening() {
    if (st.openTimer !== null) clearTimeout(st.openTimer);
    st.openTimer = null;
    setHidden(els.openingCard, true);
    els.openingText.textContent = '';
  }

  /** Forget an open in flight (the listener went elsewhere): its answer is dropped. */
  function cancelOpen() {
    if (st.opening) st.ticket += 1;
    st.opening = false;
    hideOpening();
  }

  function startPlayback() {
    const engine = ensureEngine();
    if (!engine || typeof engine.play !== 'function') return;
    const state = engine.state;
    if (state && !state.playing) engine.play();
  }

  function cursorFromProgress(doc) {
    const count = doc.blocks.length;
    const saved = doc.progress ? Number(doc.progress.block_index) || 0 : 0;
    return Math.max(0, Math.min(Math.max(0, count - 1), saved));
  }

  /** The document is on the reading surface: apply what waited for it (Resume, the hand-off, a shot). */
  function afterOpen() {
    if (st.pendingAppearance) {
      st.pendingAppearance = false;
      appearance.open(els.appearanceBtn);
    }
    if (st.pendingShot) {
      const shot = st.pendingShot;
      st.pendingShot = '';
      applyDocShot(shot);
    }
  }

  /** The document already on the surface shows again: page to where playback is, unless the view came back. */
  function reshowDoc(engine, id, { autoplay, restore }) {
    lastDocId = id;
    mountPlayerOnce();
    const state = engine ? engine.state : null;
    const live = !!(state && state.docId === id);
    if (!restore) {
      // From the library: page and scroll to where playback is (or where it stopped).
      reading.applyCursor({ block: live ? state.cursor : reading.cursor, sentence: live ? state.sentence : -1, reason: 'open' }, state);
    } else {
      reading.refreshMeta(state);   // playback may have moved on while the view was away
    }
    if (autoplay) startPlayback();
    refreshTitle();
    afterOpen();
  }

  /**
   * Open a document on the reading surface (RE-59, RE-60, RE-61).
   * @param {string} id
   * @param {{autoplay?: boolean, force?: boolean, restore?: boolean}} [options] force: fetch again even when it
   *   is on screen (Retry); restore: the view came back to the document it showed, at the spot it left
   */
  async function openDoc(id, { autoplay = false, force = false, restore = false } = {}) {
    st.openId = id;
    if (!force && st.doc && st.doc.id === id && reading.doc()) {
      // Already open: just show it (and play when Resume asked).
      const engine = await engineReady();
      if (st.openId !== id || st.mode !== 'reading') return;
      if (st.doc && st.doc.id === id && reading.doc()) {
        reshowDoc(engine, id, { autoplay, restore });
        return;
      }
    }
    const ticket = (st.ticket += 1);
    st.opening = true;
    const title = library.titleOf(id);
    reading.showLoading(title);
    hideOpening();
    st.openTimer = setTimeout(() => {
      st.openTimer = null;
      if (ticket === st.ticket) showOpening(title);
    }, OPENING_NOTICE_MS);
    // The document and the engine module load side by side: on a cold deep link the request never queues
    // behind the module graph.
    const engineLoad = engineReady();
    let doc;
    try {
      doc = await api.get(`${DOCUMENTS_PATH}/${encodeURIComponent(id)}`);
    } catch (error) {
      if (ticket !== st.ticket) return;
      st.opening = false;
      hideOpening();
      if (error && error.kind === 'aborted') return;
      if (error && error.status === 404) {
        // A dead deep link must not reopen on every reload (RE-60).
        library.forget(id);
        if (lastDocId === id) lastDocId = '';
        toast(NOT_FOUND, { kind: 'error' });
        router.navigate('reader', { replace: true });
        return;
      }
      reading.showError(api.explain(error, { title: "Couldn't open the document", notFound: NOT_FOUND }), { title: title || "Couldn't open the document" });
      refreshTitle();
      return;
    }
    if (ticket !== st.ticket) return;
    const engine = await engineLoad;
    if (ticket !== st.ticket) return;
    // A long document takes a moment to lay out: let the notice paint first (RE-59).
    if (!els.openingCard.hidden) {
      await twoFrames();
      if (ticket !== st.ticket) return;
    }
    st.opening = false;
    doc = { ...doc, blocks: Array.isArray(doc && doc.blocks) ? doc.blocks : [], chapters: Array.isArray(doc && doc.chapters) ? doc.chapters : [] };
    st.doc = doc;
    lastDocId = id;
    chapters.setDoc(doc);   // first: setDoc below marks the first chapter through onChapter
    reading.setDoc(doc);
    if (engine) {
      try {
        engine.load(doc);
      } catch (error) {
        logError(error);
      }
    }
    if (doc.blocks.length) {
      const state = engine ? engine.state : null;
      const live = !!(state && state.docId === id);
      reading.applyCursor({ block: live ? state.cursor : cursorFromProgress(doc), sentence: live ? state.sentence : -1, reason: 'open' }, state);
      reading.setPlaying(live && !!state.playing);
    }
    hideOpening();
    mountPlayerOnce();
    refreshTitle();
    if (autoplay) startPlayback();
    afterOpen();
  }

  /* ───────────── library ───────────── */

  function loadLibrary() {
    st.libraryDirty = false;
    st.libraryLoad = library.load();
    return st.libraryLoad;
  }

  /**
   * Settings › Reading appearance: the sheet over the last opened document, else the library hint.
   * "Last opened" is the one opened in this page load, else the one loaded for playback, else the
   * Continue-listening document (prefs.lastRoute can't name it: the router stores every route there
   * before show() runs).
   */
  async function openAppearanceIntent(route) {
    if (route.id) {
      if (st.doc && st.doc.id === route.id) appearance.open(els.appearanceBtn);
      else st.pendingAppearance = true;
      return;
    }
    let id = lastDocId;
    if (!id && st.engine) {
      const state = st.engine.state;
      id = (state && state.docId) || '';
    }
    if (!id) {
      if (!library.loaded) await (st.libraryLoad || loadLibrary());
      const hero = library.heroDoc();
      id = hero ? hero.id : '';
    }
    if (!st.visible || (st.route && st.route.id)) return;
    if (id) {
      st.pendingAppearance = true;
      // The document replaces the library entry, so the entry before it is Settings: Back to library has to
      // go to the library rather than step back (RL-13).
      st.redirect = id;
      router.navigate(docRoute(id), { replace: true });
    } else {
      library.showHint();
    }
  }

  /* ───────────── screenshot states ───────────── */

  /** The Chapters control on screen: the transport's chip, else the top bar's button. */
  function chaptersTrigger() {
    const chip = els.slot.querySelector('#rdp-chapters');
    return chip && !chip.hidden ? chip : els.chaptersBtn;
  }

  function applyDocShot(shot) {
    if (shot === 'appearance') appearance.open(els.appearanceBtn);
    else if (shot === 'chapters') chapters.open(chaptersTrigger());
    else if (shot === 'opening') {
      // What a slow open shows: the loading placeholder under the notice (the document is never behind it).
      const title = (st.doc && st.doc.title) || '';
      reading.showLoading(title);
      showOpening(title);
    }
    else if (shot === 'return') {
      st.shotPlaying = true;   // the screenshot shows the playing state without starting audio
      reading.setPlaying(true);
      reading.stopFollowing();
    } else if (shot === 'highlight' || shot === 'play') {
      // 'play': the player (player.js) paints its transport as playing; nothing renders or plays.
      reading.applyCursor({ block: reading.cursor, sentence: 0, reason: 'jump' });
    } else if (ASK_SHOTS.includes(shot) || ASK_SOURCE_SHOTS[shot]) {
      // The Ask AI sheet over this document (display only: nothing is sent). The current paragraph's first
      // sentence is marked as read, as in 'highlight', and the current and next paragraph are a sample selection,
      // so every source is offered. The job state starts from the whole document; the question
      // state and the source states choose This sentence or theirs.
      reading.applyCursor({ block: reading.cursor, sentence: 0, reason: 'jump' });
      const cursor = reading.cursor;
      const selection = { first: cursor, last: Math.min(cursor + 1, Math.max(0, (st.doc.blocks.length || 1) - 1)) };
      const sources = askSources(selection);
      const source = ASK_SOURCE_SHOTS[shot] || (shot === 'ask-ai-job' ? 'document' : shot === 'ask-ai-question' ? 'sentence' : defaultSource(sources));
      openSample(ASK_SOURCE_SHOTS[shot] ? 'ask-ai' : shot, { trigger: els.ask, sources, source }).catch(logError);
    }
  }

  /* ───────────── Ask AI ───────────── */

  /** What the sheet can send: this sentence and this paragraph (the reading surface's current paragraph and the
   *  sentence read in it, as the engine cursor gives them), the selection, this chapter, the whole document. */
  function askSources(selection) {
    return st.doc ? readerSources(st.doc, { selection, chapter: st.chapter, paragraph: reading.cursor, sentence: reading.sentence }) : [];
  }

  function currentSelection() {
    const sel = typeof globalThis.getSelection === 'function' ? globalThis.getSelection() : null;
    return selectionRange(sel, els.text);
  }

  function openAsk(button) {
    if (!st.doc) return;
    const selection = currentSelection() || st.askSelection;
    st.askSelection = null;
    const sources = askSources(selection);
    if (!sources.length) {
      toast('This document has no text to send.', { kind: 'info' });
      return;
    }
    // The default: the selection just made, else this sentence while something has been read, else this chapter,
    // else the document.
    openAskAi({ trigger: button, sources, source: defaultSource(sources) }).catch((error) => toast(error && error.message ? error.message : 'Something went wrong.', { kind: 'error' }));
  }

  const ask = bindAskButton(els.ask, { onClick: (button) => openAsk(button) });
  const onAskPress = () => { st.askSelection = currentSelection(); };
  if (els.ask) els.ask.addEventListener('pointerdown', onAskPress);
  offs.push(() => {
    ask.destroy();
    if (els.ask) els.ask.removeEventListener('pointerdown', onAskPress);
  });

  async function applyShot(shot, route) {
    if (!SHOTS.includes(shot)) return;
    if (DOC_SHOTS.has(shot)) {
      if (!route.id) return;
      if (st.doc && st.doc.id === route.id && !st.opening) applyDocShot(shot);
      else st.pendingShot = shot;
      return;
    }
    if (route.id) return;
    if (shot === 'add') add.open(els.add);
    else if (shot === 'add-paste') add.open(els.add, { mode: 'paste' });
    else if (shot === 'hint') library.showHint();
    else if (shot === 'loading') library.preview('loading');
    else if (shot === 'error') library.preview('error');
    else if (shot === 'row-menu') {
      await (st.libraryLoad || loadLibrary());
      const button = root.querySelector('#rd-list [data-action="row-menu"]');
      if (button && st.visible && st.mode === 'library') library.openRowMenu(button);
    }
  }

  /* ───────────── wiring ───────────── */

  function backToLibrary() {
    const engine = st.engine;
    if (engine && st.doc && typeof engine.saveProgress === 'function') {
      try {
        const state = engine.state;
        if (state && state.docId === st.doc.id) {
          const saving = engine.saveProgress();
          if (saving && typeof saving.catch === 'function') saving.catch(() => {});
        }
      } catch (error) {
        logError(error);
      }
    }
    if (st.backIsHistory && typeof history !== 'undefined' && history.length > 1) history.back();
    else router.navigate('reader');
  }

  els.back.addEventListener('click', backToLibrary);
  const onAppearanceClick = () => appearance.open(els.appearanceBtn);
  const onChaptersClick = () => chapters.open(els.chaptersBtn);
  els.appearanceBtn.addEventListener('click', onAppearanceClick);
  els.chaptersBtn.addEventListener('click', onChaptersClick);
  offs.push(() => {
    els.back.removeEventListener('click', backToLibrary);
    els.appearanceBtn.removeEventListener('click', onAppearanceClick);
    els.chaptersBtn.removeEventListener('click', onChaptersClick);
  });
  offs.push(actions(els.reading, {
    'open-retry': () => { if (st.openId) openDoc(st.openId, { force: true }); },
    'open-settings': () => openSettings(),
  }));
  offs.push(register({
    id: 'reader-search', keys: ['/'], scope: 'view:reader', singleKey: true, description: 'Search the library',
    when: () => st.mode === 'library' && st.visible,
    handler: () => library.focusSearch(),
  }));
  offs.push(onEvent('library:changed', () => { if (!st.visible) st.libraryDirty = true; }));
  offs.push(onEvent('route:changed', (detail) => {
    const route = detail && detail.route;
    const prev = detail && detail.prev;
    if (!route) return;
    if (route.view === 'reader' && route.id) {
      // Back steps back in history only when the entry before this one is the library: not after the
      // appearance hand-off, which replaced the library entry with the document.
      const redirected = !!st.redirect && st.redirect === route.id;
      st.backIsHistory = !redirected && !!(prev && prev.view === 'reader' && !prev.id);
    }
    if (!(route.view === 'reader' && !route.id)) st.redirect = '';   // the redirect's route came (or never will)
  }));

  // Start fetching the engine now, so Resume can unlock audio inside its click.
  engineReady();

  return {
    show(route, nav = {}) {
      const prevMode = st.mode;
      const wasVisible = st.visible;
      st.route = route;
      st.visible = true;
      const reason = nav.reason || 'route';
      const intent = takeIntent('reader');
      if (st.shotPlaying) {
        // The Return screenshot's playing state lasts as long as its address: the reading view follows the
        // engine again (applyShot below sets it once more when the route is that screenshot again).
        st.shotPlaying = false;
        const state = st.engine ? st.engine.state : null;
        reading.setPlaying(!!(st.doc && state && state.docId === st.doc.id && state.playing));
      }
      if (route.id) {
        const same = !!(st.doc && st.doc.id === route.id);
        const restore = same && prevMode === 'reading' && !wasVisible;
        if (restore) nav.scrolled = true;   // back from another view: the spot we left
        const autoplay = st.autoplay === route.id;
        st.autoplay = '';
        showMode('reading');
        if (restore) {
          const y = st.scrollY;
          nextFrame().then(() => {
            if (!st.visible || st.mode !== 'reading') return;
            const state = st.engine ? st.engine.state : null;
            if (reading.follow && state && state.playing) reading.keepInView({ force: true });
            else if (typeof window.scrollTo === 'function') window.scrollTo(0, y);
          });
        }
        if (reason === 'auth' && !reading.doc()) openDoc(route.id, { force: true });
        else openDoc(route.id, { autoplay, restore });
      } else {
        cancelOpen();
        showMode('library');
        if (!library.loaded || library.isStale() || st.libraryDirty || prevMode === 'reading' || reason === 'auth') loadLibrary();
      }
      if (intent && intent.type === 'open-appearance') openAppearanceIntent(route);
      if (route.shot) applyShot(route.shot, route);
      // The Ask AI button is on the reading view only; a screenshot state asks the server nothing.
      if (route.id && !route.shot) ask.refresh();
    },
    hide() {
      if (st.mode === 'reading' && typeof window !== 'undefined') st.scrollY = window.scrollY || 0;
      st.visible = false;
      reading.setVisible(false);
      if (st.opening) cancelOpen();
      // No sheet of the Reader stays open over the next view (an upload in flight reports with a toast).
      closeSheets();
      library.closeMenu();
    },
    title(route) {
      return route && route.id && st.doc && st.doc.id === route.id ? (st.doc.title || 'Untitled') : '';
    },
    immersive(route) {
      return !!(route && route.id);
    },
    // For tests and the screenshot hook.
    get state() {
      return { mode: st.mode, docId: st.doc ? st.doc.id : '', opening: st.opening, ticket: st.ticket, engine: st.engine };
    },
    engineReady,
    library,
    reading,
    appearance,
    chapters,
    add,
    destroy() {
      cancelOpen();
      if (chromeTimer !== null) clearTimeout(chromeTimer);
      chromeTimer = null;
      for (const off of offs.splice(0)) {
        try { off(); } catch { /* already gone */ }
      }
      library.destroy();
      reading.destroy();
      appearance.destroy();
      chapters.destroy();
      add.destroy();
    },
  };
}

/** Test hook: forget the last opened document. */
export function __resetForTests() {
  lastDocId = '';
}

/**
 * Warm what the Reader loads lazily: the engine, so Resume can unlock audio inside its
 * click (iOS), and the player with its partial, so the transport arrives with the first document. Importing
 * either module has no side effects; nothing is created or started here.
 */
export function prefetch() {
  return Promise.all([
    import('./engine.js'),
    import('./player.js').then((m) => (m && m.PARTIAL ? loadPartial(m.PARTIAL) : null)),
  ]);
}

export default {
  id: 'reader',
  title: 'Reader',
  partial: '/static/partials/reader.html',
  prefetch,
  init(root) {
    return createReaderController(root);
  },
};
