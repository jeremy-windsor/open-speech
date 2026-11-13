/**
 * Speak: Live mode controls (inventory LR-01, LR-07, LR-12, LR-19, LR-20). Speak (index.js) mounts them into
 * #sp-live-slot, under the shared editor:
 *
 *   const live = mountLive(slot, {editor, getVoiceConfig, onState});
 *   live.start({readAll}) · live.stop(reason) · live.togglePause() · live.isActive() · live.destroy()
 *   (extras: live.refresh() after a voice or model change, live.getState(), live.ready)
 *
 * - One toggle (Start / Pause / Resume) plus Stop, and a menu with "Read all from start" and "Clear text",
 *   first in the slot (where Generate's button sits in Generate mode). Start reads from the cursor; Read all
 *   from the beginning. Mod+Enter starts and Esc stops while the Live slot shows (registered here, scope
 *   view:speak).
 * - Mode, last in the slot: a segmented control (Book | Responsive | Each word = latency_mode natural |
 *   responsive | instant_word) with a one-line hint; disabled while a session runs (the chosen one stays
 *   visible).
 * - #spl-status (role=status) is the pill: it changes per phase, never per response (Speaking and Listening
 *   read the same to a screen reader). #spl-latency is the rolling median voice latency. #spl-now
 *   (aria-live=off) shows the text being spoken, a hint, or the reason for a failed response.
 * - A failed or refused session shows the inline banner #spl-error (role=alert) with Details. Its retry is the
 *   Start button right above it (in place of the usual [Retry]), so the banner never repeats it; its one action is Open
 *   Settings, when the key or ticket was refused. The pill keeps the phase (Busy, Error, Disconnected) in the
 *   banner's error style. A refused Start or Resume (no voice, the microphone records) shows as the hint
 *   under the controls. Text typed while Live is stopped waits for Start: the hint then says "Press Start to
 *   hear this live." (announced politely once per stop) and Start is emphasised (data-emphasis).
 * - Switching Speak from Live to Generate (#speak/live -> #speak) stops a running session ("Switched to
 *   Generate"); leaving Speak doesn't, and neither does coming back to Speak from another view (Speak then
 *   shows Live mode). Should the editor change while Speak shows Generate, Live stops with the same reason.
 *   A model whose capabilities say live_reader: false disables Start with the reason as a hint.
 * - Screenshot hook: #speak/live?shot=live-speaking | live-paused | live-recording (a Resume refused
 *   while the microphone records) | live-connecting | live-error | live-unavailable | live-menu | live-prompt
 *   (text typed while stopped) paints that state (visual only; the engine is untouched), and live-idle
 *   shows the real state; every live-* state scrolls the controls into view once they have a layout box.
 * @module views/speak/live-ui
 */
import { render, raw, html } from '../../core/dom.js';
import { app } from '../../core/store.js';
import * as prefs from '../../core/prefs.js';
import { announce } from '../../core/a11y.js';
import { icon } from '../../core/icons.js';
import { register, modLabel } from '../../core/shortcuts.js';
import { bindMenuButton } from '../../components/menu.js';
import { loadPartial as hostLoadPartial } from '../../shell/view-host.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import {
  createLiveEngine, snapshotConfig, latencyLabel, modeFor, DEFAULT_MODE, COPY, STATUS, TONES, spokenStatus,
} from './live-session.js';

export const PARTIAL = '/static/partials/speak-live.html';
export const TONE_CLASSES = Object.freeze(['idle', 'busy', 'ok', 'warn', 'error', 'rec']);

/** The remembered Live mode (prefs.liveMode, inside open-speech-prefs): kept across remounts and page loads. */
function rememberedMode() {
  try {
    return modeFor(prefs.get().liveMode || DEFAULT_MODE).value;
  } catch {
    return DEFAULT_MODE;
  }
}

/** PURE: the toggle's label and icon for a state ("Start" when idle, "Pause" while running, "Resume" when paused). */
export function toggleFor(state) {
  if (!state || !state.active) return { label: 'Start', key: 'start', icon: 'play' };
  if (state.paused) return { label: 'Resume', key: 'resume', icon: 'play' };
  return { label: 'Pause', key: 'pause', icon: 'pause' };
}

/** PURE: the Now line ({label: show "Now", text}) for an engine state. */
export function nowLine(state) {
  const now = (state && state.now) || { kind: 'hint', text: COPY.idle };
  if (now.kind === 'spoken') return { kind: 'spoken', label: true, text: `“${now.text}”` };
  return { kind: now.kind || 'hint', label: false, text: now.text || '' };
}

/**
 * PURE: the hint under the controls: a refused Start or Resume (alert), the model's Live gate (info), or, when
 * the text changed while Live is stopped (`typed`), the prompt to press Start (info, prompt: Start is
 * emphasised). While a session runs only a refused Resume shows (the microphone records).
 */
export function hintFor(state, gate, typed = false) {
  const refusal = (state && state.refusal) || '';
  if (state && state.active) return { text: refusal, tone: 'alert', prompt: false };
  if (refusal) return { text: refusal, tone: refusal === gate ? 'info' : 'alert', prompt: false };
  if (gate) return { text: gate, tone: 'info', prompt: false };
  return typed ? { text: COPY.startPrompt, tone: 'info', prompt: true } : { text: '', tone: 'info', prompt: false };
}

/** PURE: true when Speak shows Generate mode, where the shared editor's text is meant for a take. */
export function generateShows(route) {
  const r = route || {};
  return r.view === 'speak' && r.sub !== 'live';
}

/**
 * A screenshot state waits for the controls to get a layout box (Speak shows the slot after the route
 * changes): it looks every SHOT_POLL_MS, at most SHOT_TRIES times. Timers, not animation frames: a headless
 * capture under virtual time runs timers but may draw no frame at all.
 */
export const SHOT_POLL_MS = 16;
export const SHOT_TRIES = 90;

const DEMO_TEXT = 'In a hole in the ground there lived a hobbit.';

/** The screenshot states: canned engine states, painted without touching the engine. */
function demoState(shot) {
  const base = {
    active: true, reason: '', paused: false, background: false, latencyMs: 182, refusal: '', error: null, mode: 'natural',
  };
  const make = (phase, extra = {}) => Object.freeze({
    ...base, phase, status: STATUS[phase], spoken: spokenStatus(phase, STATUS[phase]), tone: TONES[phase], ...extra,
  });
  switch (shot) {
    case 'live-speaking': return make('speaking', { now: { kind: 'spoken', text: DEMO_TEXT } });
    case 'live-paused': return make('paused', { paused: true, now: { kind: 'spoken', text: DEMO_TEXT } });
    case 'live-recording': return make('paused', { paused: true, refusal: COPY.recording, now: { kind: 'spoken', text: DEMO_TEXT } });
    case 'live-connecting': return make('connecting', { latencyMs: null, now: { kind: 'hint', text: COPY.connecting } });
    case 'live-error': return make('busy', {
      active: false, latencyMs: null, now: { kind: 'hint', text: COPY.idle },
      error: {
        title: COPY.title, message: 'Another Live session is running. Stop it first or try again shortly.',
        detail: 'Live Reader is already in use\nAnother Live session is active', action: 'retry',
      },
    });
    case 'live-unavailable': return make('stopped', { active: false, latencyMs: null, now: { kind: 'hint', text: COPY.idle } });
    case 'live-menu': return make('stopped', { active: false, latencyMs: null, now: { kind: 'hint', text: COPY.idle } });
    // Text typed while Live is stopped: the prompt to press Start, never announced.
    case 'live-prompt': return make('stopped', { active: false, latencyMs: null, now: { kind: 'hint', text: COPY.idle }, prompt: true });
    default: return null;
  }
}

function isApple() {
  return modLabel() === '⌘';
}

/**
 * Mount the Live controls into a slot.
 * @param {HTMLElement} slot #sp-live-slot
 * @param {object} options
 * @param {HTMLTextAreaElement} options.editor the shared #sp-input
 * @param {() => {model: string, voice: string, speed: number, instructions?: string, voice_library_ref?: string,
 *   caps?: object}} options.getVoiceConfig Speak's current voice setup (snapshotted at Start)
 * @param {(state: object) => void} [options.onState] every Live state change ({active, phase, status, paused, …})
 * @param {(url: string) => Promise<string>} [options.loadPartial] default: viewHost.loadPartial
 * @param {object} [options.engine] extra createLiveEngine options (tests inject fakes)
 * @returns {{start: (o?: {readAll?: boolean}) => boolean, stop: (reason?: string) => void, togglePause: () => void,
 *   isActive: () => boolean, getState: () => object, refresh: () => void, ready: Promise<boolean>, destroy: () => void}}
 */
export function mountLive(slot, {
  editor = null, getVoiceConfig = () => ({}), onState = () => {}, loadPartial = hostLoadPartial, engine: engineOptions = {},
} = {}) {
  let els = null;
  let menu = null;
  let gate = '';
  let destroyed = false;
  let demo = null;
  let shotGate = '';   // the live-unavailable screenshot state's gate
  // The text changed while Live was stopped (it waits for Start), and whether that was said.
  let typed = false;
  let promptSaid = false;
  const offs = [];

  const engine = createLiveEngine({
    editor,
    getVoiceConfig,
    getMode: () => rememberedMode(),
    // REV-31: Speak only shows Generate with Live running when it was entered with a hand-off (it stops
    // Live then) or for a moment while it switches to Live mode; should the editor change meanwhile, the
    // text is meant for a take, not for Live.
    inputStopReason: () => (generateShows(app.get().route) ? 'Switched to Generate' : ''),
    onChange: (state) => {
      if (!demo) paint(state);
      try { onState(state); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
    },
    ...engineOptions,
  });

  /* ── what Start may do ── */

  function readGate() {
    if (shotGate) return shotGate;
    try {
      const { caps } = snapshotConfig(getVoiceConfig() || {}, rememberedMode());
      return caps.live_reader === false ? COPY.unavailable : '';
    } catch {
      return '';
    }
  }

  function updateGate() {
    gate = readGate();
    paint(current());
  }

  function current() {
    return demo || engine.getState();
  }

  /** The slot is on screen: Speak shows Live mode and nothing hides the slot. */
  function shown() {
    const route = app.get().route || {};
    if (route.view !== 'speak' || route.sub !== 'live') return false;
    return !(slot.closest && slot.closest('[hidden]'));
  }

  /** Start from a click, the menu or Mod+Enter (synchronously: the AudioContext resumes inside the gesture). */
  function startFromUi({ readAll = false } = {}) {
    if (engine.isActive()) return false;
    demo = null;
    gate = readGate();
    if (gate) {
      engine.refuse(gate);
      announce(gate);
      return false;
    }
    engine.clearError();
    const ok = engine.start({ readAll });
    if (!ok) {
      const { refusal } = engine.getState();
      if (refusal) announce(refusal);
    }
    return ok;
  }

  function stopFromUi() {
    if (engine.isActive()) engine.stop('Stopped');
  }

  /**
   * The shared editor changed. While Live is stopped and its slot shows, the text waits for Start: the hint
   * says "Press Start to hear this live." and Start is emphasised. While Live
   * runs, the engine reads the change; in Generate mode it is meant for a take.
   */
  function editorChanged() {
    if (destroyed || demo || engine.isActive() || !shown()) return;
    typed = true;
    paint(current());
  }
  if (editor && typeof editor.addEventListener === 'function') {
    editor.addEventListener('input', editorChanged);
    offs.push(() => editor.removeEventListener('input', editorChanged));
  }

  function hasText() {
    return String((editor && editor.value) ?? '').trim() !== '';
  }

  function clearText() {
    // LR-19: stop, then empty the text; the input event lets Speak's counter and draft follow.
    stopFromUi();
    if (!editor) return;
    editor.value = '';
    try { editor.dispatchEvent(new Event('input', { bubbles: true })); } catch { /* old engines */ }
    if (typeof editor.focus === 'function') editor.focus();
  }

  /* ── markup ── */

  const ready = Promise.resolve()
    .then(() => loadPartial(PARTIAL))
    .then((markup) => {
      if (destroyed) return false;
      render(slot, raw(markup));
      bind();
      updateGate();
      return true;
    })
    .catch((error) => {
      if (destroyed) return false;
      renderLoadError(error);
      return false;
    });

  function renderLoadError(error) {
    render(slot, html`<div class="banner banner--error spl-load-error" role="alert">${icon('alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">Live controls didn't load</p><p class="banner__msg">Check your connection, then try again.</p>${error && error.message ? html`<details class="banner__details"><summary>Details</summary><pre>${error.message}</pre></details>` : ''}</div><div class="banner__actions"><button type="button" class="btn btn--sm btn--secondary" data-spl-retry><span class="btn__label">Retry</span></button></div></div>`);
    const retry = slot.querySelector('[data-spl-retry]');
    if (retry) {
      retry.addEventListener('click', () => {
        Promise.resolve().then(() => loadPartial(PARTIAL)).then((markup) => {
          if (destroyed) return;
          render(slot, raw(markup));
          bind();
          updateGate();
          const toggle = slot.querySelector('#spl-start');
          if (toggle) toggle.focus();
        }).catch((err) => { if (!destroyed) renderLoadError(err); });
      });
    }
  }

  function bind() {
    const q = (id) => slot.querySelector(`#${id}`);
    els = {
      root: q('spl'),
      radios: Array.from(slot.querySelectorAll('input[name="spl-mode"]')),
      modeHint: q('spl-mode-hint'),
      toggle: q('spl-start'),
      toggleLabel: q('spl-start-label'),
      playIcon: slot.querySelector('.spl-toggle__play'),
      pauseIcon: slot.querySelector('.spl-toggle__pause'),
      kbd: q('spl-kbd'),
      stop: q('spl-stop'),
      more: q('spl-more'),
      menu: q('spl-more-menu'),
      readAll: q('spl-read-all'),
      clear: q('spl-clear'),
      hint: q('spl-start-hint'),
      hintText: q('spl-start-hint-text'),
      status: q('spl-status'),
      statusLabel: q('spl-status-label'),
      statusText: q('spl-status-text'),
      latency: q('spl-latency'),
      now: q('spl-now'),
      nowLabel: q('spl-now-label'),
      nowText: q('spl-now-text'),
      error: q('spl-error'),
      errorTitle: q('spl-error-title'),
      errorMsg: q('spl-error-msg'),
      errorDetails: q('spl-error-details'),
      errorDetail: q('spl-error-detail'),
      errorActions: q('spl-error-actions'),
      errorAction: q('spl-error-action'),
      errorActionLabel: q('spl-error-action-label'),
      errorDismiss: q('spl-error-dismiss'),
    };
    const modKey = q('spl-kbd-mod');
    if (modKey) modKey.textContent = modLabel();
    const chosen = rememberedMode();
    for (const radio of els.radios) radio.checked = radio.value === chosen;
    paintMode();

    const listen = (el, type, fn) => {
      el.addEventListener(type, fn);
      offs.push(() => el.removeEventListener(type, fn));
    };
    for (const radio of els.radios) {
      listen(radio, 'change', () => {
        if (!radio.checked) return;
        prefs.set({ liveMode: modeFor(radio.value).value });
        paintMode();
      });
    }
    // The model's capabilities may arrive after Speak's voice changed: re-read the gate when the listener
    // reaches the controls, so what Start shows is never stale (the click re-reads it anyway).
    const regate = () => { if (!demo && !engine.isActive()) updateGate(); };
    listen(els.root, 'focusin', regate);
    listen(els.root, 'pointerenter', regate);
    listen(els.toggle, 'click', () => {
      demo = null;
      shotGate = '';
      if (engine.isActive()) {
        engine.togglePause();   // LR-15, synchronously inside the click
        // A Resume waits while the microphone records: say why (the hint shows it too).
        const { refusal } = engine.getState();
        if (refusal) announce(refusal);
        return;
      }
      if (els.toggle.getAttribute('aria-disabled') === 'true') {
        announce(gate || COPY.unavailable);
        return;
      }
      startFromUi();
    });
    listen(els.stop, 'click', () => {
      demo = null;
      stopFromUi();
      paint(current());
    });
    // The banner's only action (a retry is the Start button above it): Settings › Connection.
    listen(els.errorAction, 'click', () => {
      const { error } = engine.getState();
      if (!error || error.action !== 'open-settings') return;
      engine.clearError();
      openConnectionSettings();
    });
    listen(els.errorDismiss, 'click', () => {
      demo = null;
      engine.clearError();
      paint(current());
      if (els.toggle) els.toggle.focus();
    });
    menu = bindMenuButton(els.more, els.menu, {
      placement: 'bottom-end',
      onSelect: (item) => {
        demo = null;
        if (item === els.readAll) startFromUi({ readAll: true });
        else if (item === els.clear) clearText();
      },
    });
  }

  /* ── paint ── */

  function paintMode() {
    if (!els) return;
    const mode = modeFor(rememberedMode());
    if (els.modeHint.textContent !== mode.hint) els.modeHint.textContent = mode.hint;
  }

  function setText(el, text) {
    if (el && el.textContent !== text) el.textContent = text;
  }

  /** Disable a control; focus on it moves to the toggle first, so it never drops to the page. */
  function setDisabled(el, disabled) {
    if (!el || el.disabled === disabled) return;
    if (disabled && typeof document !== 'undefined' && document.activeElement === el && els.toggle) els.toggle.focus();
    el.disabled = disabled;
  }

  function paint(state) {
    if (!els || !state) return;
    const active = !!state.active;
    els.root.setAttribute('data-phase', state.phase);

    // LR-20: while running, Start (now Pause/Resume) and Stop work; mode and Read all wait for the release.
    const t = toggleFor(state);
    els.toggle.setAttribute('data-state', t.key);
    setText(els.toggleLabel, t.label);
    // SVG elements have no `hidden` property in browsers: set the attribute itself.
    els.playIcon.toggleAttribute('hidden', t.icon !== 'play');
    els.pauseIcon.toggleAttribute('hidden', t.icon !== 'pause');
    const gated = !active && !!gate;
    els.kbd.hidden = active || gated;
    if (active) els.toggle.removeAttribute('aria-keyshortcuts');
    else els.toggle.setAttribute('aria-keyshortcuts', isApple() ? 'Meta+Enter' : 'Control+Enter');
    if (gated) els.toggle.setAttribute('aria-disabled', 'true');
    else els.toggle.removeAttribute('aria-disabled');
    setDisabled(els.stop, !active);
    for (const radio of els.radios) setDisabled(radio, active);
    if (active || gated) els.readAll.setAttribute('aria-disabled', 'true');
    else els.readAll.removeAttribute('aria-disabled');

    // Hint: a Start refusal, why this model can't run Live, or (text typed while stopped) the prompt to Start.
    if (active) {
      typed = false;          // the prompt is for the stopped state only, and is said again after a new stop
      promptSaid = false;
    }
    const hint = hintFor(state, gate, (typed && hasText()) || state.prompt === true);
    els.hint.hidden = !hint.text;
    els.hint.setAttribute('data-tone', hint.tone);
    setText(els.hintText, hint.text);
    if (hint.text) els.toggle.setAttribute('aria-describedby', 'spl-start-hint');
    else els.toggle.removeAttribute('aria-describedby');
    if (hint.prompt) els.toggle.setAttribute('data-emphasis', 'true');
    else els.toggle.removeAttribute('data-emphasis');
    if (hint.prompt && !promptSaid && !demo) {
      // Politely, once per stop: a screen reader hears it when the text first changes, not per keystroke.
      promptSaid = true;
      announce(hint.text);
    }

    // LR-07: the pill. Its visible label may say Speaking; the live region says it only per phase.
    for (const tone of TONE_CLASSES) els.status.classList.toggle(`pill--${tone}`, tone === state.tone);
    setText(els.statusLabel, state.status);
    setText(els.statusText, state.spoken || state.status);
    const latency = latencyLabel(state.latencyMs);
    els.latency.hidden = !latency;
    setText(els.latency, latency);

    // A failed session: the inline banner with Details and one action, next to the controls.
    const error = !active ? state.error : null;

    // LR-12 / BUG-20: the Now line follows the playback. Idle, it is a hint (none beside an error banner).
    const now = nowLine(state);
    if (gated && now.kind === 'hint') now.text = COPY.gated;
    els.now.setAttribute('data-kind', now.kind);
    els.now.hidden = !!error;
    els.nowLabel.hidden = !now.label;
    setText(els.nowText, now.text);

    els.error.hidden = !error;
    if (error) {
      setText(els.errorTitle, error.title || COPY.title);
      setText(els.errorMsg, error.message || '');
      setText(els.errorDetail, error.detail || '');
      els.errorDetails.hidden = !error.detail;
      // A retryable failure has no banner action: Start, right above the banner, starts again (in place of the usual Retry)
      // and clears it. Two buttons for one action, with different words, would only ask which one to use.
      const action = error.action === 'open-settings' ? 'Open Settings' : '';
      els.errorAction.hidden = !action;
      if (els.errorActions) els.errorActions.hidden = !action;   // no empty actions line under the message
      if (action) setText(els.errorActionLabel, action);
    }
  }

  /* ── store, shortcuts ── */

  // Speak's voice or the model list changed: a refusal is stale, and the Live gate may have changed.
  offs.push(app.select((st) => st.speakVoice, () => { engine.clearRefusal(); updateGate(); }));
  offs.push(app.select((st) => st.models, () => updateGate()));
  // A Start refused while the microphone recorded no longer applies once the recording ends.
  offs.push(app.select((st) => st.recording, (recording) => {
    if (!recording && engine.getState().refusal === COPY.recording) engine.clearRefusal();
  }));
  offs.push(app.select((st) => st.route, (route, prev) => {
    const r = route || {};
    const p = prev || {};
    // REV-31: switching Speak from Live to Generate stops Live, because the editor is shared
    // and Live would read edits meant for a take. Arriving on #speak from another view (the rail, the bottom
    // bar, g s) does not: the mini-player kept the session going, and Speak shows Live mode for it.
    if (generateShows(r) && p.view === 'speak' && p.sub === 'live' && engine.isActive()) {
      engine.stop('Switched to Generate');
    }
    if (r.view === 'speak' && r.sub === 'live') applyShot(r.shot || '');
  }, { immediate: true }));

  /** Entering Live mode re-reads the gate; a live-* screenshot state paints a canned state. */
  function applyShot(shot) {
    demo = shot && shot.startsWith('live-') && !engine.isActive() ? demoState(shot) : null;
    shotGate = shot === 'live-unavailable' && !engine.isActive() ? COPY.unavailable : '';
    gate = readGate();
    ready.then(() => {
      paint(current());
      if (!shot.startsWith('live-') || !els) return;
      // A screenshot of a Live state shows the controls (on a phone they sit below the editor). The route
      // changes before Speak shows the slot, so the scroll and the menu wait until the controls have a box:
      // a menu placed from an empty box would open in the corner of the screen. The scroll is instant (no
      // smooth scrolling anywhere), so the menu opens from where its button ends up.
      whenLaidOut(() => {
        if (typeof els.root.scrollIntoView === 'function') els.root.scrollIntoView({ block: 'center' });
        // The menu places itself before its first style, so it opens under its button at once.
        if (shot === 'live-menu' && menu && !menu.isOpen()) menu.open('first');
      });
    });
  }

  /** Run `fn` once the controls have a layout box (looked at every SHOT_POLL_MS, at most SHOT_TRIES times). */
  function whenLaidOut(fn, tries = SHOT_TRIES) {
    setTimeout(() => {
      if (destroyed || !els) return;
      const rect = els.more.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) fn();
      else if (tries > 1) whenLaidOut(fn, tries - 1);
    }, SHOT_POLL_MS);
  }

  offs.push(register({
    id: 'live-start', keys: ['Mod+Enter'], scope: 'view:speak', allowInEditable: true,
    description: 'Start Live (Live mode)',
    when: () => shown() && !engine.isActive(),
    handler: () => { startFromUi(); },
  }));
  offs.push(register({
    id: 'live-stop', keys: ['Escape'], scope: 'view:speak', allowInEditable: true,
    description: 'Stop Live',
    when: () => shown() && engine.isActive(),
    handler: () => stopFromUi(),
  }));

  return {
    start: (options) => startFromUi(options),
    stop: (reason = 'Stopped') => engine.stop(reason),
    togglePause: () => engine.togglePause(),
    isActive: () => engine.isActive(),
    getState: () => engine.getState(),
    refresh() {
      engine.clearRefusal();
      updateGate();
    },
    ready,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      engine.destroy();
      if (menu) menu.destroy();
      for (const off of offs.splice(0)) off();
    },
  };
}

/** Test hook (the module keeps no state of its own; the mode lives in prefs). */
export function __resetForTests() {}
