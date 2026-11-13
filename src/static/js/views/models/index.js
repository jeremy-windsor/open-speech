/**
 * Models: Speech (TTS) and Transcription (STT) models as provider cards, with one primary
 * action per state, "Delete files…" in an overflow menu, install commands with Copy, and an in-place patch
 * while operations run (cards.js).
 *
 * - The sub (`#models/tts`, `#models/stt`) is a segmented control; a change navigates with focus 'keep'.
 * - `#mo-summary` (role=status) says what is loaded; its text changes only when the loaded set changes, so it
 *   speaks once per finished operation, never per poll.
 * - Polling belongs to `models.runOp` (core). While this view shows and this tab runs no operation, it
 *   refreshes /api/models every 10 s to see operations started elsewhere; hide() stops that.
 * - States: loading (skeleton cards), no models for the sub, a failed first load (Retry, or "API key needed"
 *   with Open Settings in the warning tone; "Locked out" in the error tone), and an inline banner when an
 *   explicit Refresh fails over a list already shown (cards, or the sub's empty state). The banner goes once
 *   a newer inventory arrives or the view hides. A Refresh refused for the key says "API key needed" with
 *   Open Settings, not Retry: the connection banner carries the details.
 * - An operation that fails while its row is off screen (the other sub, a collapsed card, filtered out) is
 *   toasted once (toast id `model-op-<id>`, as core does while another view shows): it never fails unseen.
 * - Screenshot hook: `#models/tts?shot=menu` opens the first row's overflow menu.
 * @module views/models/index
 */
import { html, render, setHidden, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { app } from '../../core/store.js';
import * as models from '../../core/models.js';
import { explain } from '../../core/api.js';
import { announce } from '../../core/a11y.js';
import { toast } from '../../core/toast.js';
import { navigate } from '../../core/router.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import { createCards, summaryText, loadedName, __resetForTests as resetCards } from './cards.js';

/** The view's own refresh while it shows (ops started in another tab or by the server). */
export const POLL_MS = 10000;
/** A show() refreshes the inventory when it is older than this. */
export const STALE_MS = 10000;
const TICK_MS = 1000;

const EMPTY_COPY = {
  tts: { title: 'No speech models', text: 'No speech models are available. Check BAKED_PROVIDERS or start the voice workers.' },
  stt: { title: 'No transcription models', text: 'No transcription models are available.' },
};

/**
 * PURE: what the list area shows for an inventory state.
 * @returns {'cards'|'loading'|'empty'|'auth'|'error'}
 */
export function listState({ list = [], fetchedAt = 0, error = '' } = {}, sub = 'tts', connection = 'online') {
  const forSub = list.filter((m) => m && models.classifyKind(m) === sub);
  if (forSub.length) return 'cards';
  if (list.length) return 'empty';
  if (error) return connection === 'unauthorized' || connection === 'locked' ? 'auth' : 'error';
  if (!fetchedAt) return 'loading';
  return 'empty';
}

function createView(root) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    refresh: $('mo-refresh'),
    summary: $('mo-summary'),
    summaryLine: root.querySelector('.mo-summary'),
    note: $('mo-note'),
    error: $('mo-error'),
    errorTitle: $('mo-error-title'),
    errorMsg: $('mo-error-msg'),
    errorDetails: $('mo-error-details'),
    errorDetail: $('mo-error-detail'),
    errorRetry: $('mo-error-retry'),
    errorSettings: $('mo-error-settings'),
    errorDismiss: $('mo-error-dismiss'),
    state: $('mo-state'),
    list: $('mo-list'),
    menus: $('mo-menus'),
  };
  const st = {
    visible: false,
    sub: 'tts',
    resort: true,
    shown: '',          // which list state is on screen ('cards', 'loading', …)
    summaryAt: 0,       // the inventory (fetchedAt) the summary line was written from
    refreshing: null,   // the explicit Refresh in flight
    explicitError: null,
    explicitErrorAt: 0, // when it failed: a newer inventory clears the banner
    kind: '',           // the list state painted last (listState)
    opErrors: new Map(),   // model id -> the operation error already surfaced (inline or toasted)
    pendingShot: '',
    pollTimer: null,
    tickTimer: null,
    offs: [],
  };
  const cards = createCards(els.list, { menusHost: els.menus });
  // Failures from before this view existed were toasted by core (another view was showing).
  for (const [id, op] of Object.entries(app.get().ops || {})) if (op && op.error) st.opErrors.set(id, op.error);

  const listen = (el, type, fn) => {
    if (!el) return;
    el.addEventListener(type, fn);
    st.offs.push(() => el.removeEventListener(type, fn));
  };

  /* ── rendering ── */

  function syncSeg() {
    for (const input of root.querySelectorAll('input[name="mo-sub"]')) {
      const want = input.value === st.sub;
      if (input.checked !== want) input.checked = want;
    }
  }

  function paintSummary(kind) {
    const { list, fetchedAt } = app.get().models;
    // The line follows whole inventory answers only (fetchedAt), never the per-row status merges an
    // operation makes on its way: those can briefly show two speech models loaded until the refresh that
    // ends the operation. And only a real change reaches the polite region: a poll that changes nothing
    // is never announced, a finished operation is announced once.
    if (fetchedAt && fetchedAt !== st.summaryAt) {
      st.summaryAt = fetchedAt;
      const text = summaryText(list);
      if (els.summary.textContent !== text) els.summary.textContent = text;
    }
    setHidden(els.summaryLine, kind === 'error' || kind === 'auth');
    setHidden(els.note, st.sub !== 'tts');
  }

  function stateMarkup(kind) {
    const conn = app.get().connection;
    if (kind === 'loading') {
      return html`<div class="mo-skeleton" aria-busy="true"><p class="visually-hidden">Loading models…</p>${[0, 1, 2].map(() => html`<div class="card mo-skeleton__card" aria-hidden="true"><span class="skeleton skeleton--title"></span><span class="skeleton skeleton--line"></span><span class="skeleton skeleton--line mo-skeleton__short"></span></div>`)}</div>`;
    }
    if (kind === 'auth') {
      // The shell's connection banner carries the details; this state says what it means here, in
      // the banner's tone: a missing or refused key is a warning, a lockout an error.
      const locked = conn.state === 'locked';
      const title = locked ? 'Locked out' : 'API key needed';
      const text = locked
        ? 'Too many wrong API keys came from this device. The models load once the server accepts requests again.'
        : 'The models load once the server accepts your API key.';
      return html`<div class="card mo-state__card"><div class="${locked ? 'empty empty--error' : 'empty mo-empty--warn'}">${icon(locked ? 'lock' : 'key', { className: 'ico empty__icon' })}<h2 class="empty__title">${title}</h2><p class="empty__text">${text}</p><button type="button" class="btn btn--primary" data-mo-state="open-settings"><span class="btn__label">Open Settings</span></button></div></div>`;
    }
    if (kind === 'error') {
      const offline = conn.state === 'offline';
      const title = offline ? "Can't reach the server" : "Models didn't load";
      const text = offline ? 'Check that Open Speech is running.' : (app.get().models.error || 'Try again in a moment.');
      return html`<div class="card mo-state__card"><div class="empty empty--error">${icon(offline ? 'offline' : 'alert', { className: 'ico empty__icon' })}<h2 class="empty__title">${title}</h2><p class="empty__text">${text}</p><button type="button" class="btn btn--primary" data-mo-state="retry"><span class="btn__label">Retry</span></button></div></div>`;
    }
    const copy = EMPTY_COPY[st.sub];
    return html`<div class="card mo-state__card"><div class="empty">${icon('models', { className: 'ico empty__icon' })}<h2 class="empty__title">${copy.title}</h2><p class="empty__text">${copy.text}</p></div></div>`;
  }

  /**
   * The banner of an explicit Refresh that failed: over the list (its cards or the sub's empty state; the
   * other states show the failure themselves), until dismissed, outdated by a newer inventory, or hidden.
   */
  function paintBanner() {
    if (st.explicitError && app.get().models.fetchedAt > st.explicitErrorAt) st.explicitError = null;
    const error = st.explicitError;
    const show = !!error && (st.kind === 'cards' || st.kind === 'empty');
    if (show) {
      const copy = explain(error, "Couldn't refresh the models");
      // Refused for the key, or locked out: what it means here, and Open Settings instead of a Retry that
      // would be refused again. The connection banner carries the details.
      const keyed = copy.action === 'open-settings';
      const locked = error.kind === 'locked';
      const glyph = !keyed ? 'alert' : locked ? 'lock' : 'key';
      const title = !keyed ? copy.title : locked ? 'Locked out' : 'API key needed';
      const message = !keyed ? copy.message : locked
        ? 'Too many wrong API keys came from this device. The list refreshes once the server accepts requests again.'
        : 'The list refreshes once the server accepts your API key.';
      // Details: the server's message and code, never its JSON envelope; none when it would only repeat
      // the message.
      const detail = keyed ? '' : copy.detailText;
      if (els.errorTitle.textContent !== title) els.errorTitle.textContent = title;
      if (els.errorMsg.textContent !== message) els.errorMsg.textContent = message;
      els.errorDetail.textContent = detail;
      setHidden(els.errorDetails, !detail);
      els.error.classList.toggle('banner--warning', keyed && !locked);
      els.error.classList.toggle('banner--error', !keyed || locked);
      for (const svg of els.error.querySelectorAll('[data-mo-icon]')) {
        if (svg.getAttribute('data-mo-icon') === glyph) svg.removeAttribute('hidden');
        else svg.setAttribute('hidden', '');
      }
      const moveTo = keyed && els.errorRetry.contains(document.activeElement) ? els.errorSettings
        : !keyed && els.errorSettings.contains(document.activeElement) ? els.errorRetry : null;
      setHidden(els.errorRetry, keyed);
      setHidden(els.errorSettings, !keyed);
      if (moveTo) moveTo.focus({ preventScroll: true });
    }
    if (!show && els.error.contains(document.activeElement)) els.refresh.focus({ preventScroll: true });
    setHidden(els.error, !show);
  }

  /**
   * A failed operation is surfaced once: inline (role=alert) when its row is on screen, else in a toast.
   * While another view shows, core toasts it (models.runOp); it counts as surfaced then too.
   */
  function surfaceOpErrors(ops) {
    const list = ops || {};
    for (const [id, seen] of st.opErrors) {
      if (!list[id] || list[id].error !== seen) st.opErrors.delete(id);
    }
    for (const [id, op] of Object.entries(list)) {
      if (!op || !op.error || st.opErrors.get(id) === op.error) continue;
      st.opErrors.set(id, op.error);
      if (app.get().route.view !== 'models') continue;
      if (st.visible && st.kind === 'cards' && cards.isShown(id)) continue;
      const model = app.get().models.list.find((m) => m && m.id === id) || { id };
      toast(`Couldn't ${op.kind === 'load' ? 'load' : 'download'} ${loadedName(model)}: ${op.error}`, { kind: 'error', id: `model-op-${id}` });
    }
  }

  function paint({ resort = false } = {}) {
    if (!st.visible) return;
    const state = app.get();
    const kind = listState(state.models, st.sub, state.connection.state);
    st.kind = kind;
    paintSummary(kind);
    if (kind === 'cards') {
      cards.update({ sub: st.sub, list: state.models.list, ops: state.ops, defaults: state.models.defaults }, { resort: resort || st.resort });
      st.resort = false;
      if (st.shown !== 'cards') {
        const hadFocus = els.state.contains(document.activeElement);
        render(els.state, html``);
        setHidden(els.state, true);
        setHidden(els.list, false);
        if (hadFocus) {
          const first = els.list.querySelector('.mo-card__toggle');
          if (first) first.focus({ preventScroll: true });
        }
      }
    } else {
      const signature = `${kind}|${st.sub}|${kind === 'error' || kind === 'auth' ? `${state.connection.state}|${state.connection.error}|${state.models.error}` : ''}`;
      if (st.shown !== signature) {
        const hadFocus = els.state.contains(document.activeElement);
        render(els.state, stateMarkup(kind));
        if (hadFocus) {
          const button = els.state.querySelector('button');
          (button || root.querySelector('#mo-title')).focus({ preventScroll: true });
        }
      }
      setHidden(els.state, false);
      setHidden(els.list, true);
      st.shown = signature;
    }
    if (kind === 'cards') st.shown = 'cards';
    paintBanner();
    syncTicker();
    if (st.pendingShot && kind === 'cards') applyShot(st.pendingShot);
  }

  /* ── timers ── */

  function syncTicker() {
    const need = st.visible && cards.busy();
    if (need && st.tickTimer === null) {
      st.tickTimer = setInterval(() => {
        if (!cards.tick()) syncTicker();
      }, TICK_MS);
    } else if (!need && st.tickTimer !== null) {
      clearInterval(st.tickTimer);
      st.tickTimer = null;
    }
  }

  /** True while this tab runs an operation (runOp polls the status itself, so the view need not). */
  function ownOpRunning() {
    return Object.values(app.get().ops || {}).some((op) => op && op.kind && !op.error);
  }

  function pageHidden() {
    return typeof document !== 'undefined' && !!document.hidden;
  }

  /**
   * Whether a background refresh can tell anything new: not while the key is missing or refused (every
   * try would be refused; a new key re-runs the refresh through auth:changed), locked out, or offline
   * (status.js retries /health and the view refreshes when the connection comes back).
   */
  function canAsk() {
    const state = app.get().connection.state;
    return state !== 'unauthorized' && state !== 'locked' && state !== 'offline';
  }

  function startPolling() {
    stopPolling();
    st.pollTimer = setInterval(() => {
      if (!st.visible || pageHidden() || ownOpRunning() || !canAsk()) return;
      models.refresh({ silent: true }).catch(() => {});
    }, POLL_MS);
  }

  function stopPolling() {
    if (st.pollTimer !== null) clearInterval(st.pollTimer);
    st.pollTimer = null;
  }

  function refreshIfStale() {
    const { fetchedAt } = app.get().models;
    if (!canAsk()) return;
    if (!fetchedAt || Date.now() - fetchedAt >= STALE_MS) models.refresh({ silent: true }).catch(() => {});
  }

  /* ── actions ── */

  async function refreshNow() {
    if (st.refreshing) return st.refreshing;
    setBusy(els.refresh, true, { label: 'Refreshing…' });
    st.refreshing = (async () => {
      const summaryBefore = els.summary.textContent;
      try {
        await models.refresh();
        st.explicitError = null;
        st.resort = true;
        // When what is loaded changed, #mo-summary (role=status) says so already: one announcement, not two.
        if (els.summary.textContent === summaryBefore) announce('Model list updated');
      } catch (error) {
        if (!error || error.kind !== 'aborted') {
          st.explicitError = error;
          st.explicitErrorAt = Date.now();
        }
      } finally {
        setBusy(els.refresh, false);
        st.refreshing = null;
        paint({ resort: true });
      }
    })();
    return st.refreshing;
  }

  function setSub(sub, { fromControl = false } = {}) {
    const next = sub === 'stt' ? 'stt' : 'tts';
    if (next !== st.sub) {
      st.sub = next;
      st.resort = true;
    }
    syncSeg();
    paint();
    if (fromControl) navigate(`models/${next}`, { focus: 'keep' });
  }

  function applyShot(shot) {
    st.pendingShot = '';
    if (shot === 'menu') {
      const row = cards.firstMenu();
      if (row) row.menuCtl.open('first');
      else st.pendingShot = shot;
    }
  }

  listen(els.refresh, 'click', () => { refreshNow(); });
  listen(els.errorRetry, 'click', () => { refreshNow(); });
  listen(els.errorSettings, 'click', () => { openConnectionSettings().catch(() => {}); });
  listen(els.errorDismiss, 'click', () => {
    st.explicitError = null;
    els.refresh.focus({ preventScroll: true });
    paintBanner();
  });
  listen(root, 'change', (event) => {
    const input = event.target;
    if (input && input.name === 'mo-sub' && input.checked) setSub(input.value, { fromControl: true });
  });
  listen(els.state, 'click', (event) => {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const button = start && start.closest ? start.closest('[data-mo-state]') : null;
    if (!button) return;
    const what = button.getAttribute('data-mo-state');
    if (what === 'open-settings') openConnectionSettings().catch(() => {});
    else if (what === 'retry') refreshNow();
  });
  const onVisibility = () => {
    if (st.visible && !pageHidden()) refreshIfStale();
  };
  if (typeof document !== 'undefined') listen(document, 'visibilitychange', onVisibility);

  st.offs.push(app.select((s) => [s.models, s.ops, s.connection.state], (now, before) => {
    paint();
    surfaceOpErrors(now[1]);
    // Back online (or the key was accepted): catch up at once instead of on the next 10 s tick.
    if (st.visible && now[2] === 'online' && before && before[2] !== 'online') refreshIfStale();
  }, {
    equals: (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2],
  }));

  return {
    show(route, nav = {}) {
      st.visible = true;
      setSub(route && route.sub);
      if (nav.reason === 'auth') models.refresh({ silent: true }).catch(() => {});
      else refreshIfStale();
      startPolling();
      if (route && route.shot) {
        st.pendingShot = route.shot;
        paint();
      }
    },
    hide() {
      st.visible = false;
      stopPolling();
      if (st.tickTimer !== null) clearInterval(st.tickTimer);
      st.tickTimer = null;
      st.pendingShot = '';
      // A Refresh failure is old news when the listener comes back: show() refreshes a stale inventory.
      st.explicitError = null;
      // An open overflow menu is closed by the shell (closeAllMenus() runs on every route change).
    },
    /** For tests and the screenshot hook. */
    get state() {
      return { visible: st.visible, sub: st.sub, shown: st.shown, polling: st.pollTimer !== null, ticking: st.tickTimer !== null };
    },
    cards,
    refresh: refreshNow,
    destroy() {
      stopPolling();
      if (st.tickTimer !== null) clearInterval(st.tickTimer);
      for (const off of st.offs.splice(0)) off();
      cards.destroy();
    },
  };
}

/** Test hook: the cards' module state (collapse, filters, row errors) back to its defaults. */
export function __resetForTests() {
  resetCards();
}

export default {
  id: 'models',
  title: 'Models',
  partial: '/static/partials/models.html',
  init(root) {
    return createView(root);
  },
};
