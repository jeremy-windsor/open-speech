/**
 * Settings: one column of sections (Appearance, Playback, Connection, Server, AI assistant,
 * History, Shortcuts, Developer, About) with an in-page table of contents on desktop. Every control works; all
 * but the AI preset editor apply at once.
 *
 * - Appearance: the theme (System / Light / Dark), reduce motion (System / On / Off), and the reading
 *   appearance summary with "Change in the Reader", which hands off through the frozen `open-appearance`
 *   intent (REV-36): the Reader opens its appearance sheet over the last document.
 * - Playback: playback.js. Connection: connection.js (the API key). AI assistant: assistant.js (the status
 *   and the AI preset editor, with Save, Preview and Test).
 * - Server and About: read-only, from /health (app.connection) and the model inventory; Check again asks
 *   /health now. History: only working controls (SE-06, BUG-04): the server's two settings by name and a
 *   link to History, where the one Clear all lives.
 * - Shortcuts: the single-key switch (WCAG 2.1.4) and the "?" sheet.
 * - Developer: "Show Speak › Live (a streaming test for developers)" (prefs.showLive, off by default; bound with
 *   the other preference switches in playback.js). Speak follows it at once.
 * - The table of contents scrolls to a section and focuses its heading without changing the hash.
 *   While the page scrolls it marks the current section (aria-current="location"): the first section at the
 *   top of the page, else the first one crossing the band 15–45 % down the viewport. At the bottom of the
 *   page, where the last sections can never reach that band, it keeps the section the listener jumped to
 *   while that is on screen, else the last one. A jump's own scrolling never re-marks. When Settings is shown
 *   it forgets that jump and marks where the page is once the shell has scrolled it (the next frame): a window
 *   already at the top doesn't scroll, so no scroll event says so.
 * - The reading summary follows the reading appearance, and the 960 px breakpoint (the Reader's default size
 *   differs on either side). The default speed follows the Reader's speed shortcuts while Settings shows.
 * - Screenshot hook: `#settings?shot=<section>` (connection, playback, …) jumps to that section.
 * @module views/settings/index
 */
import { setBusy } from '../../core/dom.js';
import { app, setIntent } from '../../core/store.js';
import * as theme from '../../core/theme.js';
import * as prefs from '../../core/prefs.js';
import { navigate } from '../../core/router.js';
import { refreshHealth } from '../../core/status.js';
import { announce } from '../../core/a11y.js';
import { classifyKind, deviceLabel } from '../../core/models.js';
import { openShortcuts } from '../../shell/shortcuts-sheet.js';
import { loadedName } from '../models/cards.js';
import { readingSummary, STATE_WORDS, SECTIONS } from './logic.js';
import { bindConnection } from './connection.js';
import { bindPlayback } from './playback.js';
import { bindAssistant } from './assistant.js';

/** The loaded models in words: "Kokoro on GPU, whisper base on CPU", or "None". */
export function loadedModelsText(list) {
  const loaded = (list || []).filter((m) => m && m.state === 'loaded');
  if (!loaded.length) return 'None';
  const ordered = [...loaded.filter((m) => classifyKind(m) === 'tts'), ...loaded.filter((m) => classifyKind(m) === 'stt')];
  return ordered.map((m) => {
    const device = deviceLabel(m.device);
    return `${loadedName(m)}${device ? ` on ${device}` : ''}`;
  }).join(', ');
}

/** The desktop breakpoint: the table of contents shows from here, and the Reader's default size grows. */
const DESKTOP_QUERY = '(min-width: 960px)';
/** A jump to a section has ended once the page has not scrolled for this long. */
const SCROLL_QUIET_MS = 150;

function desktopQuery() {
  try {
    return typeof matchMedia === 'function' ? matchMedia(DESKTOP_QUERY) : null;
  } catch {
    return null;
  }
}

/** Listen to a MediaQueryList's change (addListener in older Safari). Returns the remover. */
function onQueryChange(query, fn) {
  if (!query) return () => {};
  if (typeof query.addEventListener === 'function') {
    query.addEventListener('change', fn);
    return () => query.removeEventListener('change', fn);
  }
  if (typeof query.addListener === 'function') {
    query.addListener(fn);
    return () => query.removeListener(fn);
  }
  return () => {};
}

function reducedMotion() {
  try {
    return theme.reducedMotion();
  } catch {
    return true;
  }
}

function createView(root) {
  const $ = (id) => root.querySelector(`#${id}`);
  const desktop = desktopQuery();
  const isDesktop = () => !!desktop && desktop.matches;
  const els = {
    readingSummary: $('se-reading-summary'),
    readingOpen: $('se-reading-open'),
    serverState: $('se-server-state'),
    serverVersion: $('se-server-version'),
    serverRevision: $('se-server-revision'),
    serverModels: $('se-server-models'),
    serverOrigin: $('se-server-origin'),
    serverSecure: $('se-server-secure'),
    serverCheck: $('se-server-check'),
    aboutVersion: $('se-about-version'),
    aboutRevision: $('se-about-revision'),
    shortcutsShow: $('se-shortcuts-show'),
    toc: $('se-toc'),
  };
  const st = {
    visible: false, offs: [], current: '',
    picked: '',          // the section the listener jumped to (kept while it is on screen at the page's end)
    jumping: false,      // a jump's own scrolling is running
    jumpTimer: null,
    frame: 0,
    tracking: null,      // the scroll/resize listeners' remover while Settings shows
  };
  const connection = bindConnection(root);
  const playback = bindPlayback(root);
  const assistant = bindAssistant(root);
  st.offs.push(() => connection.destroy(), () => playback.destroy(), () => assistant.destroy());

  const listen = (el, type, fn) => {
    if (!el) return;
    el.addEventListener(type, fn);
    st.offs.push(() => el.removeEventListener(type, fn));
  };
  const setText = (el, text) => { if (el && el.textContent !== text) el.textContent = text; };

  /* ── appearance ── */

  function syncAppearance() {
    const choices = { 'se-theme': theme.getTheme(), 'se-motion': theme.getMotion() };
    for (const [name, value] of Object.entries(choices)) {
      for (const input of root.querySelectorAll(`input[name="${name}"]`)) {
        const want = input.value === value;
        if (input.checked !== want) input.checked = want;
      }
    }
    setText(els.readingSummary, readingSummary(prefs.getReading(), { desktop: isDesktop() }));
  }

  listen(root, 'change', (event) => {
    const input = event.target;
    if (!input || input.type !== 'radio' || !input.checked) return;
    if (input.name === 'se-theme') theme.setTheme(input.value);
    else if (input.name === 'se-motion') theme.setMotion(input.value);
  });

  listen(els.readingOpen, 'click', () => {
    // The frozen hand-off: the Reader takes the intent in show() and opens its appearance sheet.
    setIntent('reader', { type: 'open-appearance' });
    navigate('reader');
  });

  /* ── server and about ── */

  function syncServer() {
    const { connection: conn, models } = app.get();
    setText(els.serverState, STATE_WORDS[conn.state] || STATE_WORDS.connecting);
    setText(els.serverVersion, conn.version ? `v${conn.version}` : 'Unknown');
    setText(els.serverRevision, conn.revision || 'Unknown');
    els.serverRevision.classList.toggle('mono', !!conn.revision);
    setText(els.serverModels, models.fetchedAt ? loadedModelsText(models.list) : 'Unknown');
    setText(els.serverOrigin, typeof location !== 'undefined' ? location.origin : '');
    setText(els.serverSecure, globalThis.isSecureContext ? 'Yes' : 'No · The microphone needs HTTPS');
    setText(els.aboutVersion, conn.version ? `v${conn.version}` : '');
    setText(els.aboutRevision, conn.revision ? `Revision ${conn.revision}` : '');
    els.aboutRevision.hidden = !conn.revision;
  }

  listen(els.serverCheck, 'click', async () => {
    if (els.serverCheck.getAttribute('aria-busy') === 'true') return;
    setBusy(els.serverCheck, true, { label: 'Checking…' });
    try {
      await refreshHealth();
    } finally {
      setBusy(els.serverCheck, false);
    }
    announce(`Server checked: ${(STATE_WORDS[app.get().connection.state] || '').replace(/…$/, '')}`);
  });

  /* ── shortcuts ── */

  listen(els.shortcutsShow, 'click', () => openShortcuts(els.shortcutsShow));

  /* ── table of contents ── */

  function markCurrent(target) {
    if (st.current === target) return;
    st.current = target;
    for (const link of root.querySelectorAll('.se-toc__link')) {
      if (link.getAttribute('data-target') === target) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }

  /** A jump's scrolling ends once the page has been still for SCROLL_QUIET_MS (smooth or instant). */
  function armJumpEnd() {
    clearTimeout(st.jumpTimer);
    st.jumpTimer = setTimeout(() => {
      st.jumpTimer = null;
      st.jumping = false;
    }, SCROLL_QUIET_MS);
  }

  /** Scroll to a section and focus its heading; the hash (the route) stays #settings. */
  function goTo(headingId) {
    const heading = $(headingId);
    if (!heading) return;
    if (!heading.hasAttribute('tabindex')) heading.setAttribute('tabindex', '-1');
    heading.focus({ preventScroll: true });
    markCurrent(headingId);
    st.picked = headingId;
    st.jumping = true;
    armJumpEnd();
    // The whole section card comes into view (scroll-padding keeps it clear of the top bar), its heading focused.
    const target = heading.closest('section') || heading;
    if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' });
  }

  listen(els.toc, 'click', (event) => {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const link = start && start.closest ? start.closest('.se-toc__link') : null;
    if (!link) return;
    event.preventDefault();
    goTo(link.getAttribute('data-target'));
  });

  const sectionList = () => SECTIONS.map((name) => $(`se-${name}-title`)).filter(Boolean)
    .map((heading) => ({ id: heading.id, el: heading.closest('section') || heading }));

  /** Mark the section the page shows now (see the module header). */
  function locateCurrent() {
    if (!st.visible || st.jumping || !isDesktop()) return;   // the contents show from 960 px only
    const list = sectionList();
    if (!list.length) return;
    const view = typeof window !== 'undefined' ? window : globalThis;
    const height = view.innerHeight || 0;
    const scroller = document.scrollingElement || document.documentElement;
    const y = view.scrollY || 0;
    if (y > 0 && y + height >= scroller.scrollHeight - 2) {
      const picked = st.picked ? list.find((s) => s.id === st.picked) : null;
      const rect = picked ? picked.el.getBoundingClientRect() : null;
      markCurrent(rect && rect.bottom > 0 && rect.top < height ? picked.id : list[list.length - 1].id);
      return;
    }
    st.picked = '';
    if (y <= 0) {
      markCurrent(list[0].id);
      return;
    }
    const hit = list.find((s) => {
      const rect = s.el.getBoundingClientRect();
      return rect.bottom > height * 0.15 && rect.top < height * 0.45;
    });
    if (hit) markCurrent(hit.id);
  }

  /** Mark the current section on the next frame (one at a time). */
  function locateSoon() {
    if (st.frame) return;
    st.frame = requestAnimationFrame(() => {
      st.frame = 0;
      locateCurrent();
    });
  }

  function onScroll() {
    if (st.jumping) {
      armJumpEnd();
      return;
    }
    locateSoon();
  }

  function startTracking() {
    if (st.tracking || typeof window === 'undefined') return;
    const options = { passive: true };
    window.addEventListener('scroll', onScroll, options);
    window.addEventListener('resize', onScroll, options);
    st.tracking = () => {
      window.removeEventListener('scroll', onScroll, options);
      window.removeEventListener('resize', onScroll, options);
      if (st.frame) cancelAnimationFrame(st.frame);
      st.frame = 0;
      clearTimeout(st.jumpTimer);
      st.jumpTimer = null;
      st.jumping = false;
    };
  }

  function stopTracking() {
    if (st.tracking) st.tracking();
    st.tracking = null;
  }

  /* ── sync from the store ── */

  st.offs.push(app.select((s) => s.prefs, () => { playback.sync(); syncAppearance(); }));
  st.offs.push(app.select((s) => s.reading, () => syncAppearance()));
  st.offs.push(app.select((s) => [s.connection, s.models], () => syncServer(), { equals: (a, b) => a[0] === b[0] && a[1] === b[1] }));
  st.offs.push(theme.onThemeChange(() => syncAppearance()));
  // Crossing 960 px changes the Reader's default size (19 px / 20 px), so the summary changes with it.
  st.offs.push(onQueryChange(desktop, () => syncAppearance()));
  // The Reader's speed controls (its sheet, the [ and ] shortcuts, from any view) write the same default
  // (KEYS.readerSpeed) and publish the rate in app.player / app.resumable: follow them while shown.
  st.offs.push(app.select((s) => [s.player ? s.player.rate : null, s.resumable ? s.resumable.rate : null], () => {
    if (st.visible) playback.syncSpeed();
  }, { equals: (a, b) => a[0] === b[0] && a[1] === b[1] }));
  markCurrent(`se-${SECTIONS[0]}-title`);

  return {
    show(route, nav) {
      st.visible = true;
      connection.sync();
      playback.sync();
      syncAppearance();
      syncServer();
      assistant.show(nav);
      startTracking();
      const shot = route && route.shot;
      if (shot && SECTIONS.includes(shot)) {
        goTo(`se-${shot}-title`);
      } else {
        // The section marked before Settings was left is stale: mark where the page is after the shell's
        // scroll to the top.
        st.picked = '';
        locateSoon();
      }
    },
    hide() {
      st.visible = false;
      stopTracking();
      connection.mask();
      assistant.hide();
    },
    /** For tests. */
    get state() {
      return { visible: st.visible, current: st.current };
    },
    goTo,
    connection,
    assistant,
    destroy() {
      stopTracking();
      for (const off of st.offs.splice(0)) off();
    },
  };
}

export default {
  id: 'settings',
  title: 'Settings',
  partial: '/static/partials/settings.html',
  init(root) {
    return createView(root);
  },
};
