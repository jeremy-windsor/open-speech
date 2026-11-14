/**
 * Talk: talk to the AI assistant with the microphone, or type, and hear it answer in
 * a voice. The engine is session.js (GA /v1/realtime), the pure parts logic.js.
 *
 * The view follows the view-host contract (shell/view-host.js): init() binds the partial once; show() never waits (it renders at once and
 * loads GET /api/assistant, then the presets, in the background; again after 30 s or a new API key); hide()
 * leaves a conversation running (the mini-player shows it, and brings you back).
 *
 * - Off: when GET /api/assistant says the assistant is off, only the off state shows, with a way to Settings ›
 *   AI assistant › Connection.
 * - The controls: Start talking / Stop (the one main control), the microphone (on or off while a conversation
 *   runs), Stop reply, Resume sound (after a pause), New conversation; the AI preset (the voice presets, `voice`
 *   by default), the reply voice (Speak's by default, from the voice picker Speak uses) and Interrupt by
 *   talking. Changing a choice while a conversation runs sends session.update; a refused change is reported and
 *   the controls go back to what the session uses. The three choices are remembered in this browser
 *   (prefs.talk); a remembered preset or voice that no longer exists falls back to the default,
 *   quietly.
 * - The conversation: a role=log with aria-live=off (the answer is heard; a screen reader reading the same words
 *   at once would talk over it), your words, the waiting phrases and the thinking dimmed (the thinking collapsed),
 *   the answers and the errors; the turn being spoken shows under it, outside the log. #tk-status speaks only
 *   phase changes. The text box under it asks a typed question (Enter sends, Shift+Enter starts a new line).
 * - Delete chat, in the conversation's header whenever the chat has entries: while a
 *   conversation runs it is New conversation (the AI forgets, the microphone stays; a toast says so); while Talk is
 *   stopped it empties the chat on screen, with Undo in a toast for 5 s. No confirm; nothing is stored.
 *
 * - A cut-off answer: the engine puts a note right under it ("The answer was cut off:
 *   the AI reached its token limit."), styled like the other notes; the log keeps the engine's order.
 *
 * Screenshot states (?shot=, display-only): talk-conversation (a sample conversation, nothing connects),
 * talk-stopped (the same chat with Talk stopped), talk-cut-off (a conversation whose last answer was cut off, with
 * its note) and talk-off (the off state).
 * @module views/talk/index
 */
import { html, render, actions, setBusy, setHidden } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { app, on as onBus } from '../../core/store.js';
import * as api from '../../core/api.js';
import { navigate } from '../../core/router.js';
import { announce } from '../../core/a11y.js';
import { toast } from '../../core/toast.js';
import * as prefs from '../../core/prefs.js';
import { DESIGN_ONLY_HINT, designOnly, modelLabel, steadyTtsModels } from '../../core/models.js';
import { describeVoice, isBlend, listVoices, listNamed } from '../../core/voices.js';
import { openVoicePicker, voiceOffered, isClonedVoice, cantSpeakIn } from '../../components/voice-picker.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import { createTalkEngine } from './session.js';
import { voicePresets, pickPreset, talkVoice, DEFAULT_PRESET, UNDO_MS, COPY } from './logic.js';

export const SHOTS = Object.freeze(['talk-conversation', 'talk-stopped', 'talk-cut-off', 'talk-off']);
/** The toast of Delete chat: a second delete replaces it. */
const CHAT_TOAST = 'talk-chat';
/** The assistant's status and presets are loaded again on show once they are this old (GS-12). */
export const STALE_MS = 30000;
/** What the reply voice picker offers: named voices and the model's voices, no recordings. */
const PICKER_INCLUDE = Object.freeze({ named: true, recordings: false, voices: true });
/**
 * Talk can't speak in a cloned voice yet (/v1/realtime refuses a named voice whose version holds a recording):
 * its picker shows those rows disabled, "Can't speak in Talk yet".
 */
const NO_CLONES = 'Talk';

/** PURE: the key of Speak's voice when it is a named voice, the one kind that can be a cloned voice; else ''. */
function speakKey(speak) {
  return speak && speak.model && speak.kind !== 'recording' && String(speak.voice || '').startsWith('voice:') ? `${speak.model}|${speak.voice}` : '';
}

// Whether a remembered reply voice can still speak is voice-picker.js's
// voiceOffered(), which the Ask AI sheet shares.

const WHO = { you: 'You', ai: 'AI', error: 'Error' };

/** PURE: one log entry's markup. */
export function entryMarkup(entry) {
  const text = html`<p class="tk-entry__text">${entry.text}</p>`;
  switch (entry.kind) {
    case 'filler':
      return html`<p class="tk-entry__text"><span class="visually-hidden">While the AI thinks: </span>${entry.text}</p>`;
    case 'thinking':
      return html`<details class="tk-thinking"><summary>Thinking</summary>${text}</details>`;
    case 'note':
      return text;
    case 'error':
      return html`<p class="tk-entry__who">${icon('alert')}<span>${WHO.error}</span></p>${text}${entry.detail ? html`<p class="tk-entry__detail">${entry.detail}</p>` : ''}`;
    case 'ai': {
      const state = entry.state === 'interrupted' ? COPY.interrupted : entry.state === 'stopped' ? COPY.stopped : '';
      return html`<p class="tk-entry__who">${WHO.ai}</p>${text}${state ? html`<p class="tk-entry__state">${state}</p>` : ''}`;
    }
    default:
      return html`<p class="tk-entry__who">${WHO.you}</p>${text}`;
  }
}

/** PURE: the reply voice button's label ("Same as Speak · Bella", "Bella + Sky", "Server default"). */
export function voiceButtonLabel(voice) {
  if (!voice) return COPY.serverDefault;
  if (voice.source === 'speak') return `Same as Speak · ${voice.label}`;
  return voice.label || COPY.serverDefault;
}

/**
 * PURE: a sample conversation for the talk-conversation screenshot state, or with Talk stopped for talk-stopped,
 * or one whose last answer was cut off for talk-cut-off.
 * @param {{stopped?: boolean, cutOff?: boolean}} [options]
 */
export function sampleState({ stopped = false, cutOff = false } = {}) {
  const entry = (id, kind, text, extra = {}) => Object.freeze({ id, kind, text, detail: '', state: '', typed: false, ...extra });
  const log = Object.freeze([
    entry('s1', 'you', "What's a good name for a gray cat?"),
    entry('s2', 'filler', 'Hmm.'),
    entry('s3', 'thinking', 'Short, friendly names. Gray suggests smoke, stone and rain clouds.'),
    entry('s4', 'ai', 'How about Pebble? Ash and Smoky suit a gray cat too.'),
    entry('s5', 'you', 'Something less common?', { typed: true }),
    entry('s6', 'error', "I couldn't reach the AI server.", { detail: "Can't reach the AI server at 127.0.0.1:9" }),
    entry('s7', 'note', COPY.reconnected),
    entry('s8', 'ai', 'Try Nimbus, like a rain cloud.', { state: 'interrupted' }),
  ]);
  if (cutOff) {
    // A running conversation whose last answer the token limit cut off, with its note.
    const cut = Object.freeze([
      entry('c1', 'you', 'Tell me about the Moon landing.'),
      entry('c2', 'filler', 'Hmm.'),
      entry('c3', 'ai', 'Apollo 11 landed on the Moon on 20 July 1969. Neil Armstrong stepped out first, and Buzz Aldrin followed him about twenty minutes later.'),
      entry('c4', 'note', COPY.cutOffLimit, { state: 'cut-off' }),
    ]);
    return Object.freeze({
      active: true, phase: 'ready', mic: 'on', reply: 'idle', hearing: false, transcribing: false, interim: '',
      paused: false, refusal: '', stopReason: '', error: null, micProblem: null,
      typed: Object.freeze({ busy: false, error: '', sent: 0 }), choiceError: '', accepted: null, log: cut,
      status: Object.freeze({ label: 'Listening', tone: 'ok', spoken: 'Listening' }),
    });
  }
  if (stopped) {
    return Object.freeze({
      active: false, phase: 'stopped', mic: 'off', reply: 'idle', hearing: false, transcribing: false, interim: '',
      paused: false, refusal: '', stopReason: '', error: null, micProblem: null,
      typed: Object.freeze({ busy: false, error: '', sent: 0 }), choiceError: '', accepted: null, log,
      status: Object.freeze({ label: 'Stopped', tone: 'idle', spoken: 'Stopped' }),
    });
  }
  return Object.freeze({
    active: true, phase: 'ready', mic: 'on', reply: 'idle', hearing: true, transcribing: false, interim: 'And one for a black cat',
    paused: false, refusal: '', stopReason: '', error: null, micProblem: null,
    typed: Object.freeze({ busy: false, error: '', sent: 0 }), choiceError: '', accepted: null, log,
    status: Object.freeze({ label: 'Hearing you…', tone: 'rec', spoken: 'Listening' }),
  });
}

/**
 * Build the controller for a mounted partial.
 * @param {HTMLElement} root the view's section, with partials/talk.html inside
 * @param {{engine?: object, engineOptions?: object}} [deps] test seams
 */
export function createTalkView(root, deps = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const el = {
    loading: $('tk-loading'), loadError: $('tk-load-error'), off: $('tk-off'), layout: $('tk-layout'),
    start: $('tk-start'), startLabel: $('tk-start-label'), startGo: root.querySelector('.tk-start__go'), startStop: root.querySelector('.tk-start__stop'),
    state: $('tk-state'), pill: $('tk-pill'), pillLabel: $('tk-pill-label'), status: $('tk-status'),
    actions: $('tk-actions'), mic: $('tk-mic'), cancel: $('tk-cancel'), resume: $('tk-resume'), newConv: $('tk-new'),
    refusal: $('tk-refusal'), micHelp: $('tk-mic-help'), error: $('tk-error'),
    preset: $('tk-preset'), voice: $('tk-voice'), voiceName: $('tk-voice-name'), voiceHint: $('tk-voice-hint'), interrupt: $('tk-interrupt'), choiceError: $('tk-choice-error'),
    empty: $('tk-empty'), log: $('tk-log'), interim: $('tk-interim'), logTitle: $('tk-log-title'), del: $('tk-delete'),
    form: $('tk-ask'), text: $('tk-text'), send: $('tk-send'), textError: $('tk-text-error'), textErrorMsg: $('tk-text-error-msg'),
  };

  // The choices start from what this browser remembers (prefs.talk).
  const saved = prefs.get().talk || prefs.TALK_DEFAULTS;
  const st = {
    phase: 'loading',        // 'loading' | 'off' | 'on' | 'failed'
    loadSeq: 0,
    loadedAt: 0,
    stale: true,
    presets: [],             // the full preset views (for the error sentences)
    presetsLoaded: false,    // a remembered preset waits for the list before it can fall back
    preset: saved.preset || DEFAULT_PRESET,
    chosen: saved.voice ? { ...saved.voice } : null,   // the voice picked in Talk ({model, voice, label}), or null to follow Speak
    voiceCheck: saved.voice ? 'waiting' : 'done',      // a remembered voice is checked once per page (checkSavedVoice)
    voiceCheckRun: null,     // that check's promise while it runs ('checking'): Start waits for it
    // Whether Speak's named voice is a cloned voice for Speak's model: {key, cloned, run}, where
    // run is the check's promise while it runs (Start waits for it too).
    speakClone: { key: '', cloned: false, run: null },
    interrupt: saved.interrupt !== false,
    destroyed: false,
    before: null,            // the choices before the last change (a refused change goes back to them)
    shot: '',
    shown: false,
    everActive: false,
    lastSent: 0,
    lastChoiceError: '',
    banners: { error: '', mic: '', load: '' },
    undo: null,              // the Undo toast of a chat deleted while stopped
  };
  const entryEls = new Map();   // entry id -> {el, sig}
  const offs = [];

  const engine = deps.engine || createTalkEngine({
    getChoices,
    // Start waits for a running check of the remembered voice, so a deleted one is never sent, and
    // for the check of Speak's voice, so a cloned one is never sent.
    choicesReady: () => {
      const runs = [checkVoiceSoon(), checkSpeakVoice()].filter(Boolean);
      return runs.length ? Promise.all(runs) : null;
    },
    onChange: (state) => { if (!st.shot) renderSession(state); },
    ...(deps.engineOptions || {}),
  });

  /* ── choices ── */

  function currentVoice() {
    const { speakVoice, models } = app.get();
    // Speak's voice is not followed when it is a cloned voice, or on a model that only designs voices: the server
    // default speaks instead.
    const speak = speakCloned() || speakDesigns() ? null : speakVoice;
    return talkVoice({ chosen: st.chosen, speak, defaultModel: (models && models.defaults && models.defaults.tts) || '' });
  }

  /** Whether Speak is on a model that only designs voices (a new voice for every reply). */
  function speakDesigns() {
    const speak = app.get().speakVoice;
    const list = (app.get().models && app.get().models.list) || [];
    return !!speak && !!speak.model && designOnly(list.find((m) => m.id === speak.model));
  }

  /** Whether Speak's voice, as it is now, was found to be a cloned voice. */
  function speakCloned() {
    const key = speakKey(app.get().speakVoice);
    return !!key && st.speakClone.key === key && st.speakClone.cloned;
  }

  /**
   * Check, in the background, whether Speak's named voice is a cloned voice for Speak's model (its version holds a
   * recording), which Talk can't speak in yet. Once per Speak voice. One that is falls back to the
   * server default, quietly; a list that fails to load says nothing, so Talk follows Speak. `deferred`: Speak's voice
   * just changed and the reply voice waits for this check to show and send it. Returns the check's promise while it
   * runs, else null.
   */
  function checkSpeakVoice({ deferred = false } = {}) {
    const speak = app.get().speakVoice;
    const key = speakKey(speak);
    if (key === st.speakClone.key) {
      if (st.speakClone.run && deferred) st.speakClone.deferred = true;
      return st.speakClone.run;
    }
    const entry = { key, cloned: false, run: null, deferred };
    st.speakClone = entry;
    if (!key) return null;
    entry.run = (async () => {
      let named = null;
      try { named = await listNamed({ model: speak.model }); } catch { /* nothing says the voice is cloned */ }
      if (st.destroyed || st.speakClone !== entry) return;
      entry.run = null;
      entry.cloned = isClonedVoice(named, speak);
      if (!entry.cloned && !entry.deferred) return;
      renderVoice();
      if (st.chosen) return;
      if (entry.cloned) {
        remember();
        changed();
      } else {
        engine.update();
      }
    })().catch((error) => { if (globalThis.console) globalThis.console.error(error); });
    return entry.run;
  }

  function getChoices() {
    const voice = currentVoice();
    return {
      model: voice.model, voice: voice.voice, preset: st.preset, interrupt: st.interrupt,
      presetInfo: st.presets.find((p) => p.id === st.preset) || null,
    };
  }

  function remember() {
    st.before = { preset: st.preset, chosen: st.chosen, interrupt: st.interrupt };
  }

  function changed() {
    st.lastChoiceError = '';
    engine.update();
  }

  /** Remember the listener's choices for the next page (prefs.talk: only the fields given). */
  function save(patch) {
    prefs.set({ talk: patch });
  }

  function renderVoice() {
    const voice = currentVoice();
    el.voiceName.textContent = voiceButtonLabel(voice);
    el.voice.disabled = !voice.model;
    if (el.voiceHint) {
      el.voiceHint.textContent = DESIGN_ONLY_HINT;
      el.voiceHint.hidden = !(voice.source === 'default' && speakDesigns());
    }
  }

  function renderPresets() {
    const list = voicePresets(st.presets);
    // A remembered preset waits for the list; one the list no longer offers falls back to the default.
    const next = st.presetsLoaded ? pickPreset(st.presets, st.preset) : st.preset;
    if (next !== st.preset) {
      remember();
      st.preset = next;
      changed();
    }
    render(el.preset, html`${list.map((p) => html`<option value="${p.id}">${p.name}</option>`)}`);
    el.preset.value = st.preset;
  }

  /**
   * A remembered reply voice is used while it can still speak (voiceOffered). It is checked once per
   * page in the background, once the model list is known, with the voice picker's own lists. One that can't falls
   * back to following Speak, quietly, and stays remembered until the listener chooses again; a choice made while
   * the check runs wins.
   */
  async function checkSavedVoice() {
    if (st.voiceCheck !== 'waiting' || !st.chosen) return;
    const known = app.get().models;
    if (!known || !known.fetchedAt) return;   // checked once the model list is known
    st.voiceCheck = 'checking';
    const chosen = st.chosen;
    const usable = steadyTtsModels(known.list);
    const lists = {};
    if (usable.some((m) => m.id === chosen.model)) {
      try {
        if (chosen.voice.startsWith('voice:')) lists.named = await listNamed({ model: chosen.model });
        else lists.voices = await listVoices(chosen.model, { strict: true });
      } catch { /* the list didn't load: nothing says the voice is gone */ }
    }
    st.voiceCheck = 'done';
    // A remembered cloned voice is not offered either: Talk can't speak in it yet.
    if (st.destroyed || st.chosen !== chosen || voiceOffered(chosen, { models: usable, include: PICKER_INCLUDE, noClones: NO_CLONES, ...lists })) return;
    remember();
    st.chosen = null;
    renderVoice();
    changed();
  }

  async function pickVoice() {
    // A check of Speak's voice that still runs ends first, so the rows say what Talk can speak in.
    const checking = checkSpeakVoice();
    if (checking) await checking;
    if (st.destroyed) return;
    const voice = currentVoice();
    if (!voice.model) return;
    const speak = app.get().speakVoice || {};
    const extraRows = [];
    // Speak's cloned voice still shows as "Same as Speak", disabled with the reason, so the fallback is explained.
    const cloned = speakCloned();
    if ((speak.model === voice.model || cloned) && speak.voice && speak.kind !== 'recording') {
      const name = speak.label || describeVoice(speak.voice).label || speak.voice;
      const detail = cloned ? cantSpeakIn(NO_CLONES) : isBlend(speak.voice) ? speak.voice : '';
      extraRows.push({ id: 'same-as-speak', voice: speak.voice, label: `Same as Speak · ${name}`, detail, group: 'Suggested', disabled: cloned });
    }
    // modelLabel: a model with no voices of its own gets the picker's note, as in Speak.
    // noClones: the named voices whose version is a recording show disabled.
    const choice = await openVoicePicker({
      trigger: el.voice, model: voice.model, modelLabel: modelLabel(voice.model), value: voice.voice, title: 'Reply voice',
      include: PICKER_INCLUDE, extraRows, noClones: NO_CLONES,
    });
    if (!choice) return;
    remember();
    if (choice.kind === 'extra' && choice.id === 'same-as-speak') st.chosen = null;
    else st.chosen = { model: choice.model || voice.model, voice: choice.voice, label: choice.label || choice.voice };
    st.voiceCheck = 'done';
    save({ voice: st.chosen });
    renderVoice();
    changed();
  }

  /* ── loading ── */

  function renderPhase() {
    const phase = st.shot === 'talk-off' ? 'off' : st.phase;
    // A conversation that runs keeps its controls, whatever a later load says.
    const on = phase === 'on' || (phase !== 'off' && engine.isActive()) || (st.shot !== '' && st.shot !== 'talk-off');
    setHidden(el.loading, phase !== 'loading' || on);
    setHidden(el.off, phase !== 'off' || on);   // a staged conversation shows alone
    setHidden(el.layout, !on);
    setHidden(el.loadError, phase !== 'failed' || on);
  }

  function showLoadError(error) {
    let title = "Couldn't load the AI assistant";
    let message = api.explain(error).message || 'Something went wrong.';
    let action = { label: 'Retry', action: 'load-retry' };
    if (error && (error.kind === 'auth' || error.kind === 'locked')) {
      title = 'API key needed';
      message = "Enter the server's API key in Settings › Connection. Talk loads once the key is accepted.";
      action = { label: 'Open Settings', action: 'open-settings' };
    } else if (error && error.status === 404) {
      title = "This server doesn't have the AI assistant";
      message = 'It comes with a newer Open Speech.';
      action = null;
    }
    const key = JSON.stringify([title, message, action && action.action]);
    if (st.banners.load === key) return;
    st.banners.load = key;
    render(el.loadError, html`${icon('alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${title}</p><p class="banner__msg">${message}</p></div>${action ? html`<div class="banner__actions"><button type="button" class="btn btn--secondary btn--sm" data-action="${action.action}"><span class="btn__label">${action.label}</span></button></div>` : ''}`);
  }

  async function load() {
    const seq = (st.loadSeq += 1);
    st.stale = false;
    if (st.phase !== 'on') {
      st.phase = 'loading';
      renderPhase();
    }
    try {
      const status = await api.get('/api/assistant', { accept: 'json' });
      if (seq !== st.loadSeq) return;
      if (!status || !status.enabled) {
        st.phase = 'off';
        st.loadedAt = Date.now();
        renderPhase();
        return;
      }
      const data = await api.get('/api/assistant/presets', { accept: 'json' });
      if (seq !== st.loadSeq) return;
      st.presets = Array.isArray(data && data.presets) ? data.presets.filter((p) => p && typeof p.id === 'string') : [];
      st.presetsLoaded = true;
      st.phase = 'on';
      st.loadedAt = Date.now();
      st.banners.load = '';
      renderPresets();
      renderPhase();
    } catch (error) {
      if (seq !== st.loadSeq || (error && error.kind === 'aborted')) return;
      if (error && error.code === 'assistant_disabled') {
        st.phase = 'off';
        renderPhase();
        return;
      }
      st.phase = 'failed';
      st.stale = true;
      showLoadError(error);
      renderPhase();
    }
  }

  /* ── the session ── */

  function banner(target, key, slot, markup) {
    if (st.banners[slot] === key) return;
    st.banners[slot] = key;
    setHidden(target, !key);
    if (key) render(target, markup);
    else render(target, html``);
  }

  function glyph(svg, show) {
    if (!svg) return;
    if (show) svg.removeAttribute('hidden');
    else if (!svg.hasAttribute('hidden')) svg.setAttribute('hidden', '');
  }

  function renderErrorBanner(error) {
    const key = error ? JSON.stringify([error.title, error.message, error.detail, error.action]) : '';
    const actionBtn = error && error.action === 'open-settings'
      ? html`<button type="button" class="btn btn--secondary btn--sm" data-action="open-settings"><span class="btn__label">Open Settings</span></button>`
      : error && error.action === 'retry'
        ? html`<button type="button" class="btn btn--secondary btn--sm" data-action="retry"><span class="btn__label">Retry</span></button>`
        : '';
    banner(el.error, key, 'error', error ? html`${icon('alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${error.title}</p><p class="banner__msg">${error.message}</p>${error.detail ? html`<details class="banner__details"><summary>Details</summary><pre>${error.detail}</pre></details>` : ''}</div>${actionBtn ? html`<div class="banner__actions">${actionBtn}</div>` : ''}<button type="button" class="icon-btn icon-btn--sm" data-action="dismiss-error" aria-label="Dismiss">${icon('x')}</button>` : html``);
  }

  function renderMicBanner(problem, dismissable) {
    const key = problem ? JSON.stringify([problem.title, problem.message, dismissable]) : '';
    banner(el.micHelp, key, 'mic', problem ? html`${icon('mic', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${problem.title}</p><p class="banner__msg">${problem.message}</p></div>${dismissable ? html`<button type="button" class="icon-btn icon-btn--sm" data-action="dismiss-mic" aria-label="Dismiss">${icon('x')}</button>` : ''}` : html``);
  }

  function renderLog(list) {
    const ids = new Set(list.map((e) => e.id));
    for (const [id, rec] of entryEls) {
      if (!ids.has(id)) {
        rec.el.remove();
        entryEls.delete(id);
      }
    }
    const box = el.log;
    // Shown before entries join, so the scroll position below is measured on the laid-out log.
    setHidden(el.empty, list.length > 0);
    setHidden(box, list.length === 0);
    // Delete chat shows while there is a chat, running or not.
    setHidden(el.del, list.length === 0);
    const atEnd = !(Number.isFinite(box.scrollHeight) && Number.isFinite(box.clientHeight))
      || box.scrollHeight - box.scrollTop - box.clientHeight < 48;
    let added = false;
    let prev = null;   // the element of the entry before this one
    for (const entry of list) {
      const sig = `${entry.kind}|${entry.text}|${entry.detail}|${entry.state}`;
      const rec = entryEls.get(entry.id);
      if (!rec) {
        const item = document.createElement('div');
        item.className = 'tk-entry';
        item.setAttribute('data-kind', entry.kind);
        if (entry.typed) item.setAttribute('data-typed', 'true');
        render(item, entryMarkup(entry));
        // In the log's order: entries join at the end, but a cut-off answer's note goes right under it.
        box.insertBefore(item, prev ? prev.nextSibling : box.firstChild);
        entryEls.set(entry.id, { el: item, sig });
        added = true;
        prev = item;
      } else {
        if (rec.sig !== sig) {
          render(rec.el, entryMarkup(entry));
          rec.sig = sig;
        }
        prev = rec.el;
      }
    }
    // Follow the conversation only when the reader is already at its end.
    if (added && atEnd && Number.isFinite(box.scrollHeight)) box.scrollTop = box.scrollHeight;
  }

  function renderInterim(state) {
    let words = '';
    if (state.hearing) words = state.interim ? `${state.interim}…` : "You're speaking…";
    else if (state.transcribing) words = 'Writing down your words…';
    if (el.interim.textContent !== words) el.interim.textContent = words;
    setHidden(el.interim, !words);
  }

  function renderSession(state) {
    const active = !!state.active;
    if (active) st.everActive = true;
    renderPhase();
    // The main control. Its glyphs are SVG elements, which have no `hidden` property: the attribute it is.
    el.startLabel.textContent = active ? 'Stop' : 'Start talking';
    glyph(el.startGo, !active);
    glyph(el.startStop, active);
    // The status: shown once a conversation has started (before that, "Stopped" would say nothing useful).
    setHidden(el.state, !st.everActive && !state.error);
    const { label, tone, spoken } = state.status;
    if (el.pillLabel.textContent !== label) el.pillLabel.textContent = label;
    el.pill.className = `pill pill--${tone}`;
    if (st.everActive && el.status.textContent !== spoken) el.status.textContent = spoken;
    // The other controls (their row takes no room before a conversation starts).
    setHidden(el.actions, !active);
    setHidden(el.mic, !active);
    el.mic.setAttribute('aria-pressed', state.mic === 'on' || state.mic === 'starting' ? 'true' : 'false');
    el.mic.disabled = state.mic === 'unavailable';
    setHidden(el.cancel, !active || state.reply === 'idle');
    setHidden(el.resume, !active || !state.paused);
    setHidden(el.newConv, !active);
    el.newConv.disabled = state.phase !== 'ready';
    // Delete chat is New conversation while one runs, so it waits for the session as New conversation does.
    // An Undo still offered for a stopped chat ends once Talk starts again.
    el.del.disabled = active && state.phase !== 'ready';
    if (active) dropUndo();
    el.refusal.textContent = state.refusal || '';
    setHidden(el.refusal, !state.refusal);
    renderErrorBanner(state.error);
    renderMicBanner(state.micProblem, !!state.micProblem && state.mic !== 'unavailable');
    // A refused change: say so, and show the choices the session still uses.
    if (state.choiceError && state.choiceError !== st.lastChoiceError && st.before) {
      // What this browser remembers goes back too, field by field.
      const back = {};
      if (st.preset !== st.before.preset) back.preset = st.before.preset;
      if (st.chosen !== st.before.chosen) back.voice = st.before.chosen;
      if (st.interrupt !== st.before.interrupt) back.interrupt = st.before.interrupt;
      st.preset = st.before.preset;
      st.chosen = st.before.chosen;
      st.interrupt = st.before.interrupt;
      el.preset.value = st.preset;
      el.interrupt.checked = st.interrupt;
      renderVoice();
      if (Object.keys(back).length) save(back);
    }
    st.lastChoiceError = state.choiceError || '';
    el.choiceError.textContent = state.choiceError ? `That change wasn't accepted: ${state.choiceError}` : '';
    setHidden(el.choiceError, !state.choiceError);
    // The conversation.
    renderLog(state.log || []);
    renderInterim(state);
    // The text box.
    const typed = state.typed || { busy: false, error: '', sent: 0 };
    setBusy(el.send, !!typed.busy);
    if (typed.sent > st.lastSent) {
      st.lastSent = typed.sent;
      el.text.value = '';
    }
    showTypedError(typed.error);
  }

  function showTypedError(message) {
    const text = String(message || '');
    if (el.textErrorMsg.textContent !== text) el.textErrorMsg.textContent = text;
    setHidden(el.textError, !text);
    if (text) el.text.setAttribute('aria-invalid', 'true');
    else el.text.removeAttribute('aria-invalid');
  }

  /* ── actions ── */

  function submitTyped() {
    if (el.send.getAttribute('aria-busy') === 'true') return;
    const words = el.text.value;
    if (!words.trim()) return;
    engine.ask(words);
  }

  /**
   * Delete chat. While a conversation runs it is New conversation: the AI forgets it,
   * the microphone stays, and that can't be undone. While Talk is stopped the chat on screen empties, with Undo for
   * UNDO_MS. No confirm; nothing is stored. Focus moves to the conversation's heading, as the button goes.
   */
  function deleteChat() {
    if (engine.isActive()) {
      const state = engine.getState();
      if (state.phase !== 'ready' || !state.log.length) return;
      engine.newConversation();
      dropUndo();
      toast(COPY.chatDeletedRunning, { kind: 'success', id: CHAT_TOAST });
    } else {
      const removed = engine.clearLog();
      if (!removed) return;
      st.undo = toast(COPY.chatDeleted, {
        kind: 'success', id: CHAT_TOAST, timeoutMs: UNDO_MS,
        action: { label: COPY.undo, run: () => undoDelete(removed) },
      });
    }
    if (!el.logTitle.hasAttribute('tabindex')) el.logTitle.setAttribute('tabindex', '-1');
    el.logTitle.focus({ preventScroll: true });
  }

  function undoDelete(removed) {
    st.undo = null;
    if (engine.restoreLog(removed)) announce(COPY.chatRestored);
  }

  /** An Undo still offered goes (Talk started again, or the view goes). */
  function dropUndo() {
    if (!st.undo) return;
    const handle = st.undo;
    st.undo = null;
    handle.dismiss();
  }

  async function openAiSettings() {
    await navigate('settings', { focus: 'heading' });
    const target = document.getElementById('se-assistant-title');
    if (target) {
      if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
      target.focus();
      if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'start' });
    }
  }

  offs.push(actions(root, {
    start: () => {
      if (engine.isActive()) engine.stop();
      else engine.start({ mic: true });
    },
    mic: () => {
      const on = el.mic.getAttribute('aria-pressed') === 'true';
      engine.setMic(!on);
    },
    'cancel-reply': () => engine.cancelReply(),
    resume: () => engine.resume(),
    new: () => {
      engine.newConversation();
      announce('New conversation');
    },
    'delete-chat': () => deleteChat(),
    retry: () => {
      engine.clearError();
      engine.start({ mic: true });
    },
    'dismiss-error': () => engine.clearError(),
    'dismiss-mic': () => engine.clearMicProblem(),
    'open-settings': (button) => (button.id === 'tk-open-settings' ? openAiSettings() : openConnectionSettings()),
    'load-retry': () => load(),
    voice: () => pickVoice(),
  }));

  const onSubmit = (event) => {
    event.preventDefault();
    submitTyped();
  };
  const onKeydown = (event) => {
    if (event.key !== 'Enter' || event.shiftKey || event.altKey || event.ctrlKey || event.metaKey || event.isComposing) return;
    event.preventDefault();
    submitTyped();
  };
  const onInput = () => {
    engine.clearTyped();
    showTypedError('');
  };
  const onPreset = () => {
    remember();
    st.preset = el.preset.value || DEFAULT_PRESET;
    save({ preset: st.preset });
    changed();
  };
  const onInterrupt = () => {
    remember();
    st.interrupt = !!el.interrupt.checked;
    save({ interrupt: st.interrupt });
    changed();
  };
  el.form.addEventListener('submit', onSubmit);
  el.text.addEventListener('keydown', onKeydown);
  el.text.addEventListener('input', onInput);
  el.preset.addEventListener('change', onPreset);
  el.interrupt.addEventListener('change', onInterrupt);
  offs.push(() => {
    el.form.removeEventListener('submit', onSubmit);
    el.text.removeEventListener('keydown', onKeydown);
    el.text.removeEventListener('input', onInput);
    el.preset.removeEventListener('change', onPreset);
    el.interrupt.removeEventListener('change', onInterrupt);
  });
  // Speak's voice (followed by default) and the model list (the default model) change the reply voice.
  offs.push(app.select((s) => s.speakVoice, () => {
    // A new named voice is checked first; the check shows and sends it once it ends.
    if (checkSpeakVoice({ deferred: true })) return;
    renderVoice();
    if (!st.chosen) engine.update();
  }));
  offs.push(app.select((s) => s.models && s.models.defaults && s.models.defaults.tts, () => renderVoice()));
  // A remembered voice is checked once the model list is known (a page that opens on Talk may not have it yet).
  offs.push(app.select((s) => s.models && s.models.fetchedAt, () => {
    renderVoice();
    checkVoiceSoon();
  }));
  offs.push(onBus('auth:changed', () => {
    st.stale = true;
    if (st.shown) load();
  }));

  /** Start the check of the remembered voice when it can run. Returns its promise while it runs, else null. */
  function checkVoiceSoon() {
    if (st.voiceCheck === 'checking') return st.voiceCheckRun;
    const run = checkSavedVoice().catch((error) => { if (globalThis.console) globalThis.console.error(error); });
    st.voiceCheckRun = st.voiceCheck === 'checking' ? run : null;
    return st.voiceCheckRun;
  }

  el.interrupt.checked = st.interrupt;
  renderPresets();
  renderVoice();
  renderSession(engine.getState());
  checkVoiceSoon();
  checkSpeakVoice();

  function applyShot(shot) {
    st.shot = SHOTS.includes(shot) ? shot : '';
    if (st.shot === 'talk-conversation') renderSession(sampleState());
    if (st.shot === 'talk-cut-off') renderSession(sampleState({ cutOff: true }));
    if (st.shot === 'talk-stopped') {
      st.everActive = true;   // a chat left on screen after Stop: the pill says "Stopped"
      renderSession(sampleState({ stopped: true }));
    }
    renderPhase();
  }

  return {
    show(route) {
      st.shown = true;
      if (route && route.shot) applyShot(route.shot);
      if (st.stale || Date.now() - st.loadedAt > STALE_MS) load();
      else renderPhase();
    },
    hide() {
      st.shown = false;
    },
    destroy() {
      st.destroyed = true;
      dropUndo();
      for (const off of offs.splice(0)) {
        try { off(); } catch { /* gone */ }
      }
      engine.destroy();
    },
    /** Test hook: the engine. */
    engine,
  };
}

export default {
  id: 'talk',
  title: 'Talk',
  partial: '/static/partials/talk.html',
  init(root) { return createTalkView(root); },
};
