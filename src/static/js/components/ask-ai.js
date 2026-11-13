/**
 * "Ask AI": the one shared button and sheet every view uses to send
 * something to the AI assistant, built on POST /api/assistant/ask. The sheet is the shell dialog
 * #app-ask-ai; the pure parts are in ask-ai-logic.js.
 *
 *   bindAskButton(button, {onClick}) -> {refresh(), destroy()}   shown only while GET /api/assistant says on
 *   openAskAi({trigger, sources, source, title}) -> Promise<void> resolves when the sheet closes
 *   openSample(state, {trigger, sources}) -> the screenshot states ask-ai, ask-ai-answer, ask-ai-job,
 *     ask-ai-error and ask-ai-question (no request)
 *
 * A source is {id, label, input, text|detail, chars, chapters?, title?, named?} (ask-ai-logic.js). The sheet shows:
 * - What's sent: the source (a choice when a view offers more than one, such as the Reader's sentence, paragraph,
 *   selection, chapter and document), a short summary and its size, and what goes with it (`named`: the Reader's
 *   document name and chapter).
 * - Your question: one optional line that grows, sent as `question` (up to 2,000 characters), with
 *   the quick picks and the last questions sent (kept in this browser, prefs.askAi.questions).
 * - The action: the AI presets that suit the input's kind (their `label`, `inputs` and `output`), the default
 *   the server's (`book` for a document, `voice` otherwise); the last one sent for that kind in this browser.
 * - The outputs: Show the answer, Speak it in a voice (Speak's voice by default, the voice picker to change it),
 *   Save it as a document; the preset's `output` sets the first choice. A long input (more than one piece of
 *   the preset's chunk_chars) goes as a job: it is never spoken here, and its answers are always saved.
 * - The privacy line: "This is sent to <AI server host>." and, for a long input, what the Reader keeps.
 * - Send, then the answer: the reply as text (always escaped: textContent), the thinking collapsed, the time,
 *   Play (an ephemeral-free clip through core/audio.js, so the mini-player and the lock screen have it), Save as
 *   a document (or Open in the Reader once saved), Copy, and Send again. A job shows its pieces' progress with
 *   Cancel (DELETE), then its answers and its document; it is read every POLL_MS, never while the page is hidden
 *   (read at once when it shows again). Errors are plain words by the server's code, in an
 *   inline banner (role=alert); with speech asked, the preset's spoken error sentence plays. The banner and the
 *   answer are brought into view inside the sheet when they appear.
 * Closing the sheet aborts a request in flight; a job goes on on the server (a toast says so) and its answers
 * appear in the Reader. Such a job is read every WATCH_MS until it ends, never while the page is hidden and no
 * longer once the page is left; then a toast says it is done, with Open for its answers, or why it failed.
 * The choices (the preset per input kind and the outputs sent last, the answer voice, the last questions) are
 * remembered in this browser (prefs.askAi); a remembered preset or voice that no longer
 * exists falls back, quietly.
 * @module components/ask-ai
 */
import { html, render, setHidden, setBusy } from '../core/dom.js';
import { icon } from '../core/icons.js';
import * as api from '../core/api.js';
import * as audio from '../core/audio.js';
import { app, emit, on as onBus } from '../core/store.js';
import { openSheet, closeSheet } from '../core/dialogs.js';
import { announce } from '../core/a11y.js';
import { toast } from '../core/toast.js';
import { copyText } from '../core/clipboard.js';
import { navigate } from '../core/router.js';
import { reducedMotion } from '../core/theme.js';
import * as prefs from '../core/prefs.js';
import { describeVoice, RECOMMENDED_VOICES, listVoices, listNamed } from '../core/voices.js';
import { modelLabel, providerOf, ttsModels } from '../core/models.js';
import { openVoicePicker, voiceOffered } from './voice-picker.js';
import * as L from './ask-ai-logic.js';

export const ASK_PATH = '/api/assistant/ask';
export const JOBS_PATH = '/v1/reader/assistant-jobs';
export const SHOTS = Object.freeze(['ask-ai', 'ask-ai-answer', 'ask-ai-job', 'ask-ai-error', 'ask-ai-question']);
/** The sample question of the screenshot state ask-ai-question. */
const SAMPLE_QUESTION = 'Explain this to me.';
const PICKER_INCLUDE = Object.freeze({ named: true, recordings: false, voices: true });

/* ───────────── availability: GET /api/assistant, then the presets ───────────── */

let avail = { phase: 'unknown', host: '', presets: [], loadedAt: 0, error: null };
let inflight = null;
let loadSeq = 0;
const listeners = new Set();
let busOff = null;
let sheet = null;   // the open sheet's controller

function setAvail(next) {
  avail = Object.freeze({ ...avail, ...next });
  for (const fn of [...listeners]) {
    try { fn(avail); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
  }
}

/** The assistant's state: {phase: 'unknown'|'loading'|'on'|'off'|'failed', host, presets, loadedAt, error}. */
export function availability() {
  return avail;
}

/** Listen for availability changes; returns the unsubscribe. */
export function onAvailability(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function watchAuth() {
  if (busOff) return;
  busOff = onBus('auth:changed', () => {
    if (listeners.size) refreshAvailability({ force: true }).catch(() => {});
    else avail = Object.freeze({ ...avail, loadedAt: 0 });
  });
}

/**
 * Load the assistant's status and, when it is on, its presets: once per STALE_MS (`force` loads again now).
 * One request at a time; a newer one wins. Never throws: a failure is phase 'failed' (the button stays hidden).
 * @returns {Promise<object>} availability()
 */
export function refreshAvailability({ force = false } = {}) {
  watchAuth();
  const fresh = avail.loadedAt && Date.now() - avail.loadedAt < L.STALE_MS && (avail.phase === 'on' || avail.phase === 'off');
  if (!force && fresh) return Promise.resolve(avail);
  if (inflight && !force) return inflight;
  const seq = (loadSeq += 1);
  if (avail.phase === 'unknown') setAvail({ phase: 'loading' });
  const run = (async () => {
    try {
      const status = await api.get('/api/assistant', { accept: 'json' });
      if (seq !== loadSeq) return avail;
      if (!status || !status.enabled) {
        setAvail({ phase: 'off', host: '', presets: [], loadedAt: Date.now(), error: null });
        return avail;
      }
      const data = await api.get('/api/assistant/presets', { accept: 'json' });
      if (seq !== loadSeq) return avail;
      const presets = Array.isArray(data && data.presets) ? data.presets.filter((p) => p && typeof p.id === 'string') : [];
      setAvail({ phase: 'on', host: L.hostOf(status), presets, loadedAt: Date.now(), error: null });
    } catch (error) {
      if (seq !== loadSeq) return avail;
      if (error && error.code === 'assistant_disabled') setAvail({ phase: 'off', host: '', presets: [], loadedAt: Date.now(), error: null });
      else if (!(error && error.kind === 'aborted')) setAvail({ phase: 'failed', loadedAt: 0, error });
    } finally {
      if (seq === loadSeq) inflight = null;
    }
    return avail;
  })();
  inflight = run;
  return run;
}

/**
 * Bind a view's "Ask AI" button: hidden unless the assistant is on; the click calls `onClick(button, event)`.
 * The view calls refresh() from its show() where the button is on screen (it loads again once the status is
 * 30 s old, or at once after a new API key); binding sends nothing.
 * @param {HTMLButtonElement} button
 * @param {{onClick: (button: HTMLElement, event: Event) => void, onChange?: (on: boolean) => void}} options
 */
export function bindAskButton(button, { onClick, onChange } = {}) {
  if (!button) return { refresh() {}, destroy() {}, get on() { return false; } };
  let on = null;
  const sync = (state) => {
    const next = state.phase === 'on';
    setHidden(button, !next);
    if (next !== on) {
      on = next;
      if (typeof onChange === 'function') onChange(next);
    }
  };
  const off = onAvailability(sync);
  const click = (event) => {
    if (button.disabled || button.getAttribute('aria-disabled') === 'true') return;
    if (typeof onClick === 'function') onClick(button, event);
  };
  button.addEventListener('click', click);
  sync(avail);   // the status is asked for by refresh(), from the view's show(): binding sends nothing
  return {
    refresh: () => refreshAvailability().catch(() => {}),
    destroy() {
      off();
      button.removeEventListener('click', click);
    },
    get on() { return avail.phase === 'on'; },
  };
}

/* ───────────── the answer voice ───────────── */

/** PURE: the voice an answer is spoken in: the one chosen here, else Speak's (never a raw recording), else the
 *  default model's (Kokoro's recommended voice; for another model the server's default). */
export function answerVoice({ chosen = null, speak = null, defaultModel = '' } = {}) {
  if (chosen && chosen.voice) return { model: String(chosen.model || ''), voice: String(chosen.voice), label: String(chosen.label || chosen.voice), source: 'chosen' };
  const s = speak || {};
  if (s.model && s.voice && s.kind !== 'recording') {
    return { model: String(s.model), voice: String(s.voice), label: `Same as Speak · ${s.label || describeVoice(s.voice).label || s.voice}`, source: 'speak' };
  }
  const model = String(defaultModel || '');
  const voice = model && providerOf(model) === 'kokoro' ? RECOMMENDED_VOICES[0] : '';
  return { model, voice, label: voice ? describeVoice(voice).label || voice : L.COPY.serverDefault, source: 'default' };
}

/* ───────────── remembered choices (prefs.askAi) ───────────── */

/** What the listener chose last in this browser: {presets: {kind: id}, outputs, voice}. */
function remembered() {
  return prefs.get().askAi || prefs.ASK_AI_DEFAULTS;
}

/** Remember choices for the next page (only the fields given; `presets` is sent merged). */
function remember(patch) {
  prefs.set({ askAi: patch });
}

/** What a remembered voice's check found, once per page: `${model}|${voice}` -> 'checking' | 'ok' | 'gone'. */
let voiceChecks = new Map();
const voiceKey = (v) => `${v.model}|${v.voice}`;

/**
 * The remembered answer voice while it can still speak, else null (Speak's is used): its model must be a usable
 * TTS model once the model list is known, and its check must not have found it gone. A fallback writes nothing,
 * so a worker that was down brings the voice back on a later page.
 */
function rememberedVoice() {
  const v = remembered().voice;
  if (!v) return null;
  if (voiceChecks.get(voiceKey(v)) === 'gone') return null;
  const { models } = app.get();
  if (models && models.fetchedAt && !ttsModels(models.list).some((m) => m.id === v.model)) return null;
  return v;
}

/**
 * Check the remembered voice once per page, in the background, once the model list is known: the voice picker's
 * own list for its model must still offer it (voiceOffered). A list that fails to load says nothing, so the voice
 * stays. Resolves true when it turned out gone.
 */
async function checkRememberedVoice() {
  const v = remembered().voice;
  const known = app.get().models;
  if (!v || voiceChecks.has(voiceKey(v)) || !known || !known.fetchedAt) return false;
  const key = voiceKey(v);
  voiceChecks.set(key, 'checking');
  const usable = ttsModels(known.list);
  const lists = {};
  if (usable.some((m) => m.id === v.model)) {
    try {
      if (String(v.voice).startsWith('voice:')) lists.named = await listNamed({ model: v.model });
      else lists.voices = await listVoices(v.model, { strict: true });
    } catch { /* the list didn't load: nothing says the voice is gone */ }
  }
  const ok = voiceOffered(v, { models: usable, include: PICKER_INCLUDE, ...lists });
  if (voiceChecks.get(key) === 'checking') voiceChecks.set(key, ok ? 'ok' : 'gone');
  return !ok;
}

/** The remembered voice's check while it runs: {key, done} (a promise that never rejects). Send waits for it. */
let voiceCheckRun = null;

/** Start the remembered voice's check when it can run (checkRememberedVoice), keeping its promise while it runs. */
function startVoiceCheck() {
  const v = remembered().voice;
  const already = v && voiceChecks.has(voiceKey(v));
  const run = checkRememberedVoice();
  if (v && !already && voiceChecks.get(voiceKey(v)) === 'checking') voiceCheckRun = { key: voiceKey(v), done: run.then(() => {}, () => {}) };
  return run;
}

/** The running check of the remembered voice (a promise), or null when none runs. */
function runningVoiceCheck() {
  const v = remembered().voice;
  if (!v || !voiceCheckRun || voiceCheckRun.key !== voiceKey(v) || voiceChecks.get(voiceKey(v)) !== 'checking') return null;
  return voiceCheckRun.done;
}

function currentVoice() {
  const { speakVoice, models } = app.get();
  return answerVoice({ chosen: rememberedVoice(), speak: speakVoice, defaultModel: (models && models.defaults && models.defaults.tts) || '' });
}

/* ───────────── a job whose sheet was closed ───────────── */

/** The jobs whose sheet was closed while they ran: job id -> {id, timer, busy}. */
const watching = new Map();
let pageOffs = [];

const pageHidden = () => typeof document !== 'undefined' && !!document.hidden;

/** One toast per job: the "goes on" toast, then the one that says how it ended, which replaces it. */
function jobToastId(job) {
  return `ask-ai-job:${job && job.id ? job.id : ''}`;
}

/** Listen to the page while a job is watched: not read while hidden, read at once when shown, dropped when left. */
function bindPage() {
  if (pageOffs.length || typeof window === 'undefined' || !window.addEventListener || typeof document === 'undefined') return;
  const onVisibility = () => {
    if (pageHidden()) pauseWatching();
    else resumeWatching();
  };
  // Left for good: nothing more is read. A page kept in the back/forward cache only waits for its return.
  const onPagehide = (event) => {
    if (event && event.persisted) pauseWatching();
    else stopWatching();
  };
  const onPageshow = (event) => { if (event && event.persisted) resumeWatching(); };
  document.addEventListener('visibilitychange', onVisibility);
  window.addEventListener('pagehide', onPagehide);
  window.addEventListener('pageshow', onPageshow);
  pageOffs = [
    () => document.removeEventListener('visibilitychange', onVisibility),
    () => window.removeEventListener('pagehide', onPagehide),
    () => window.removeEventListener('pageshow', onPageshow),
  ];
}

function unbindPage() {
  for (const off of pageOffs.splice(0)) {
    try { off(); } catch { /* gone */ }
  }
}

function watchJob(job) {
  if (!job || !job.id || watching.has(job.id)) return;
  const entry = { id: String(job.id), timer: null, busy: false };
  watching.set(entry.id, entry);
  bindPage();
  scheduleWatch(entry);
}

function scheduleWatch(entry) {
  if (entry.timer !== null || entry.busy || watching.get(entry.id) !== entry || pageHidden()) return;
  entry.timer = setTimeout(() => {
    entry.timer = null;
    readWatched(entry);
  }, L.WATCH_MS);
}

function forgetWatched(entry) {
  if (entry.timer !== null) clearTimeout(entry.timer);
  entry.timer = null;
  if (watching.get(entry.id) === entry) watching.delete(entry.id);
  if (!watching.size) unbindPage();
}

function pauseWatching() {
  for (const entry of watching.values()) {
    if (entry.timer !== null) clearTimeout(entry.timer);
    entry.timer = null;
  }
}

function resumeWatching() {
  if (pageHidden()) return;
  for (const entry of [...watching.values()]) {
    if (entry.busy) continue;
    if (entry.timer !== null) clearTimeout(entry.timer);
    entry.timer = null;
    readWatched(entry);   // shown again: at once, so its end is told now
  }
}

function stopWatching() {
  for (const entry of [...watching.values()]) forgetWatched(entry);
}

async function readWatched(entry) {
  if (watching.get(entry.id) !== entry || entry.busy) return;
  entry.busy = true;
  let job;
  try {
    job = await api.get(`${JOBS_PATH}/${encodeURIComponent(entry.id)}`, { accept: 'json' });
  } catch (error) {
    entry.busy = false;
    if (watching.get(entry.id) !== entry) return;
    if (error && error.status === 404) forgetWatched(entry);   // deleted on the server: nothing to tell
    else scheduleWatch(entry);                                   // a dropped request: the next read tries again
    return;
  }
  entry.busy = false;
  if (watching.get(entry.id) !== entry) return;
  if (!L.jobEnded(job)) {
    scheduleWatch(entry);
    return;
  }
  forgetWatched(entry);
  tellJobEnd(job);
}

/** Done: a toast with Open for its answers (the shell's success toast). Failed: an error toast, which stays. A
 *  job cancelled elsewhere says nothing. */
function tellJobEnd(job) {
  const docId = typeof job.result_document_id === 'string' ? job.result_document_id : '';
  const action = docId ? { label: 'Open', run: () => navigate(`reader/${encodeURIComponent(docId)}`) } : null;
  if (job.status === 'done') {
    emit('library:changed', { docId });
    toast(L.jobEndText(job), { kind: 'success', id: jobToastId(job), action });
  } else if (job.status === 'failed') {
    toast(L.jobEndText(job), { kind: 'error', id: jobToastId(job), action });
  }
}

/* ───────────── the sheet ───────────── */

function elements(dialog) {
  const $ = (id) => dialog.querySelector(`#${id}`);
  return {
    dialog,
    form: $('ask-form'), title: $('ask-title'), compose: $('ask-compose'), what: $('ask-what'), sources: $('ask-sources'),
    named: $('ask-named'), question: $('ask-question'), picks: $('ask-picks'), recent: $('ask-recent'),
    preset: $('ask-preset'), presetHint: $('ask-preset-hint'), outputs: $('ask-outputs'), show: $('ask-out-show'),
    speak: $('ask-out-speak'), voice: $('ask-voice'), voiceName: $('ask-voice-name'), save: $('ask-out-save'),
    long: $('ask-long'), outError: $('ask-out-error'), privacy: $('ask-privacy-text'),
    busy: $('ask-busy'), status: $('ask-status'), elapsed: $('ask-elapsed'), progressRow: $('ask-progress-row'),
    progress: $('ask-progress'), progressText: $('ask-progress-text'),
    result: $('ask-result'), reply: $('ask-reply'), thinking: $('ask-thinking'), meta: $('ask-meta'), note: $('ask-note'),
    play: $('ask-play'), saveDoc: $('ask-save'), open: $('ask-open'), copy: $('ask-copy'),
    error: $('ask-error'), cancel: $('ask-cancel'), cancelLabel: $('ask-cancel-label'), again: $('ask-again'), send: $('ask-send'),
  };
}

function bannerMarkup(copy) {
  const buttons = [];
  if (copy.retry || copy.action === 'retry') buttons.push(html`<button type="button" class="btn btn--secondary btn--sm" data-ask="retry">${icon('retry')}<span class="btn__label">Try again</span></button>`);
  if (copy.action === 'open-ai-settings') buttons.push(html`<button type="button" class="btn btn--secondary btn--sm" data-ask="ai-settings"><span class="btn__label">Open Settings</span></button>`);
  else if (copy.action === 'open-settings') buttons.push(html`<button type="button" class="btn btn--secondary btn--sm" data-ask="settings"><span class="btn__label">Open Settings</span></button>`);
  return html`${icon('alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${copy.title}</p>${copy.message ? html`<p class="banner__msg">${copy.message}</p>` : ''}${copy.sentence ? html`<p class="banner__msg ask__said">${copy.sentence}</p>` : ''}${copy.detail ? html`<details class="banner__details"><summary>Details</summary><pre></pre></details>` : ''}</div>${buttons.length ? html`<div class="banner__actions">${buttons}</div>` : ''}`;
}

function createSheet(dialog, options) {
  const el = elements(dialog);
  const sources = (Array.isArray(options.sources) ? options.sources : []).filter((s) => s && s.input && L.KINDS.includes(s.input.kind));
  const st = {
    phase: 'compose',        // compose | busy | job | result
    sourceId: (sources.find((s) => s.id === options.source) || sources[0] || {}).id || '',
    preset: '',
    outputs: { show: true, speak: false, save: false },
    voice: currentVoice(),
    controller: null,
    job: null,
    pollTimer: null,
    polling: false,          // a read of the job is in flight
    tickTimer: null,
    startedAt: 0,
    answer: null,            // {reply, thinking, audio, documentId, ...}
    asked: null,             // {preset, choice, outputs, source} of the last Send
    url: '',                 // the answer's object URL
    playingUrl: '',          // the object URL the answer clip plays
    closed: false,
    sample: !!options.sample,
  };
  const offs = [];
  const listen = (node, type, fn) => {
    if (!node) return;
    node.addEventListener(type, fn);
    offs.push(() => node.removeEventListener(type, fn));
  };

  const source = () => sources.find((s) => s.id === st.sourceId) || sources[0] || null;
  const kind = () => (source() ? source().input.kind : 'text');
  const choices = () => L.presetChoices(avail.presets, kind());
  const choice = () => choices().find((c) => c.id === st.preset) || null;
  const long = () => L.isLong(source(), choice());

  /* ── compose ── */

  function renderSources() {
    const current = source();
    if (sources.length > 1) {
      render(el.sources, html`<div class="ask__options" role="radiogroup" aria-labelledby="ask-what-label">${sources.map((s) => html`<label class="ask__option"><input class="ask__radio" type="radio" name="ask-source" value="${s.id}"${s.id === st.sourceId ? html` checked` : ''}><span class="ask__option-text"><span class="ask__option-label">${s.label}</span><span class="ask__option-detail">${L.sizeLabel(s.chars)}</span></span></label>`)}</div>${current ? html`<p class="ask__quote">${L.summarize(current.detail || current.text || '')}</p>` : ''}`);
    } else if (current) {
      render(el.sources, html`<div class="ask__source"><p class="ask__source-label">${current.label}</p><p class="ask__quote">${L.summarize(current.detail || current.text || '')}</p><p class="ask__size tabular">${L.sizeLabel(current.chars)}</p></div>`);
    } else {
      render(el.sources, html`<p class="ask__quote">Nothing to send.</p>`);
    }
    // What goes with it (the Reader's document name and chapter), when the source says.
    const named = current && typeof current.named === 'string' ? current.named : '';
    el.named.textContent = named;
    setHidden(el.named, !named);
  }

  /* ── the question ── */

  let recentShown = [];

  /** The quick picks, then the last questions sent (those that aren't a quick pick), as chips that fill the box. */
  function renderQuestionPicks() {
    render(el.picks, html`${L.QUICK_QUESTIONS.map((q, i) => html`<button type="button" class="chip" data-pick="quick" data-index="${i}"><span class="chip__label">${q}</span></button>`)}`);
    recentShown = L.recentChoices(remembered().questions);
    render(el.recent, html`${recentShown.map((q, i) => html`<button type="button" class="chip chip--action" data-pick="recent" data-index="${i}">${icon('history')}<span class="chip__label">${L.summarize(q, L.QUESTION_CHIP_CHARS)}</span></button>`)}`);
    setHidden(el.recent, !recentShown.length);
  }

  const fieldSizing = (() => {
    try { return typeof CSS !== 'undefined' && !!CSS.supports && CSS.supports('field-sizing', 'content'); } catch { return false; }
  })();

  /** One line that grows with its text: CSS does it where field-sizing is known; else its height follows here. */
  function growQuestion() {
    if (fieldSizing || !el.question.isConnected) return;
    el.question.style.setProperty('--ask-q-h', 'auto');
    const height = el.question.scrollHeight;
    if (height > 0) el.question.style.setProperty('--ask-q-h', `${height + 2}px`);
  }

  function setQuestion(text) {
    el.question.value = String(text || '');
    growQuestion();
  }

  function pickQuestion(event) {
    const target = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const button = target && target.closest ? target.closest('[data-pick]') : null;
    if (!button) return;
    const list = button.getAttribute('data-pick') === 'recent' ? recentShown : L.QUICK_QUESTIONS;
    const q = list[Number(button.getAttribute('data-index'))];
    if (typeof q !== 'string') return;
    setQuestion(q);   // a pick is just text in the box
    el.question.focus();
    const end = el.question.value.length;
    try { el.question.setSelectionRange(end, end); } catch { /* not a text field here */ }
  }

  function renderPresets() {
    const list = choices();
    // A remembered preset that is no longer offered falls back to the server's default, quietly.
    const wanted = st.preset || remembered().presets[kind()] || '';
    st.preset = L.defaultPreset(list, kind(), wanted);
    render(el.preset, html`${list.map((c) => html`<option value="${c.id}">${c.text}</option>`)}`);
    el.preset.value = st.preset;
    el.preset.disabled = !list.length;
    el.presetHint.textContent = list.length ? '' : L.COPY.noPresets;
    setHidden(el.presetHint, !!list.length);
  }

  function applyPresetOutputs() {
    const c = choice();
    const last = remembered().outputs;   // the outputs sent last, for this kind and preset only
    if (last && last.kind === kind() && last.preset === st.preset) st.outputs = { show: last.show, speak: last.speak, save: last.save };
    else st.outputs = { ...L.defaultOutputs(c), ...(options.outputs || {}) };   // a view's own start (Speak: speak it)
  }

  function renderOutputs() {
    const isLong = long();
    el.show.checked = !!st.outputs.show;
    el.speak.checked = !!st.outputs.speak && !isLong;
    el.speak.disabled = isLong;
    el.save.checked = !!st.outputs.save || isLong;
    el.save.disabled = isLong;
    el.voice.disabled = isLong || !st.outputs.speak || !st.voice.model;
    el.voiceName.textContent = `Voice: ${st.voice.label || L.COPY.serverDefault}`;
    setHidden(el.long, !isLong);
    if (isLong) el.long.textContent = L.COPY.longNote;
    const ok = isLong || L.hasOutput(st.outputs);
    el.outError.querySelector('span').textContent = ok ? '' : L.COPY.noOutput;
    setHidden(el.outError, ok);
    el.send.disabled = !ok || !source() || !st.preset;
  }

  function renderPrivacy() {
    el.privacy.textContent = L.privacyLine(avail.host, { long: long(), kind: kind() });
  }

  function renderCompose() {
    renderSources();
    renderQuestionPicks();
    renderOutputs();
    renderPrivacy();
  }

  /* ── phases ── */

  function stopTimers() {
    if (st.pollTimer !== null) clearTimeout(st.pollTimer);
    st.pollTimer = null;
    if (st.tickTimer !== null) clearInterval(st.tickTimer);
    st.tickTimer = null;
  }

  function showPhase(phase) {
    st.phase = phase;
    setHidden(el.compose, phase !== 'compose');
    setHidden(el.busy, phase !== 'busy' && phase !== 'job');
    setHidden(el.result, phase !== 'result');
    setHidden(el.send, phase !== 'compose');
    setHidden(el.again, phase !== 'result');
    el.cancelLabel.textContent = phase === 'result' ? 'Done' : 'Cancel';
    setHidden(el.progressRow, phase !== 'job');
  }

  function tick() {
    const s = Math.max(0, Math.floor((Date.now() - st.startedAt) / 1000));
    const text = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    if (el.elapsed.textContent !== text) el.elapsed.textContent = text;
  }

  function startBusy(text) {
    st.startedAt = Date.now();
    el.status.textContent = text;
    el.elapsed.textContent = '0:00';
    if (st.tickTimer === null) st.tickTimer = setInterval(tick, 1000);
    showPhase('busy');
    clearError();
  }

  function clearError() {
    render(el.error, html``);
    setHidden(el.error, true);
  }

  /** Bring a part of the sheet into view inside it: on a short viewport the banner comes
   *  after the choices, below the fold. The nearest edge only, and no smooth scroll under reduced motion. */
  function bringIntoView(node) {
    if (!node || typeof node.scrollIntoView !== 'function') return;
    try { node.scrollIntoView({ block: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' }); } catch { /* not laid out */ }
  }

  function showError(copy) {
    render(el.error, bannerMarkup(copy));
    const pre = el.error.querySelector('details pre');
    if (pre) pre.textContent = String(copy.detail || '');   // details as text only
    setHidden(el.error, false);
    bringIntoView(el.error);
  }

  /* ── the answer ── */

  function revokeUrl(url) {
    if (!url) return;
    try { URL.revokeObjectURL(url); } catch { /* gone */ }
  }

  /** An object URL for the answer's audio (kept while it can still play; revoked once replaced and let go). */
  function audioUrl(clip) {
    if (!clip) return '';
    try {
      const bytes = L.base64ToBytes(clip.base64);
      return URL.createObjectURL(new Blob([bytes], { type: clip.contentType }));
    } catch {
      return '';
    }
  }

  function ours() {
    const cur = audio.current();
    return !!cur && cur.id === L.CLIP_ID;
  }

  function renderPlay() {
    const has = !!st.url;
    setHidden(el.play, !has);
    if (!has) return;
    const cur = audio.current();
    const playing = ours() && cur.getState && cur.getState().playing;
    render(el.play, html`${icon(playing ? 'pause' : 'play')}<span class="btn__label">${playing ? 'Pause' : 'Play'}</span>`);
  }

  async function play() {
    if (!st.url) return;
    const cur = audio.current();
    if (ours() && cur.controls && typeof cur.controls.toggle === 'function' && st.playingUrl === st.url) {
      cur.controls.toggle();
      renderPlay();
      return;
    }
    const url = st.url;
    st.playingUrl = url;
    const c = st.asked && st.asked.choice;
    try {
      await audio.playClip({
        id: L.CLIP_ID, owner: 'assistant', title: 'AI answer', subtitle: c ? c.label : '', src: url,
        onRelease: () => {
          if (st.playingUrl === url) st.playingUrl = '';
          if (url !== st.url || st.closed) revokeUrl(url);
          renderPlay();
        },
      });
    } catch (error) {
      if (!st.closed) toast(error && error.message ? error.message : "Couldn't play the answer.", { kind: 'error' });
    }
    renderPlay();
  }

  function setAnswerAudio(clip) {
    const old = st.url;
    st.url = audioUrl(clip);
    // The old clip goes now unless it is the one still playing (its release revokes it).
    if (old && !(ours() && st.playingUrl === old)) revokeUrl(old);
  }

  function showAnswer(answer, { job = null } = {}) {
    st.answer = answer;
    el.reply.textContent = answer.reply || '';   // the AI's text, always as text
    const think = answer.thinking || '';
    const pre = el.thinking.querySelector('pre');
    if (pre) pre.textContent = think;
    el.thinking.open = false;
    setHidden(el.thinking, !think);
    const c = st.asked && st.asked.choice;
    el.meta.textContent = L.metaLine({ presetText: c ? c.text : '', elapsedMs: answer.elapsedMs, finishReason: answer.finishReason });
    const notes = [answer.audioNote, answer.documentNote].filter(Boolean).join(' ');
    el.note.textContent = notes;
    setHidden(el.note, !notes);
    // Show off (only Speak or Save asked): the text stays out of the way; Copy still has it.
    setHidden(el.reply, !!st.asked && st.asked.outputs.show === false && !job);
    renderSaved();
    renderPlay();
    showPhase('result');
    announce(job ? 'The AI finished the job.' : 'The AI answered.');
    if (el.title && typeof el.title.focus === 'function') el.title.focus({ preventScroll: true });
    bringIntoView(el.result);
  }

  function renderSaved() {
    const id = st.answer ? st.answer.documentId : '';
    setHidden(el.open, !id);
    setHidden(el.saveDoc, !!id || !(st.answer && st.answer.reply));
    setHidden(el.copy, !(st.answer && st.answer.reply));
  }

  /* ── Send ── */

  async function send() {
    if (st.phase !== 'compose' || el.send.disabled) return;
    const src = source();
    const c = choice();
    if (!src || !c) return;
    const isLong = long();
    const outputs = isLong ? { show: true, speak: false, save: true } : { ...st.outputs };
    if (!L.hasOutput(outputs)) return;
    // iOS: the answer plays only after the request, so the click unlocks the shared element now.
    if (outputs.speak) audio.prime();
    const question = L.questionOf(el.question.value);   // the listener's own, when the box isn't blank
    remember({
      presets: { ...remembered().presets, [kind()]: c.id },
      outputs: { kind: kind(), preset: c.id, show: !!st.outputs.show, speak: !!st.outputs.speak, save: !!st.outputs.save },
      ...(question ? { questions: L.rememberQuestion(remembered().questions, question) } : {}),
    });
    st.asked = { preset: c.id, choice: c, outputs, source: src, question };
    const controller = typeof AbortController === 'function' ? new AbortController() : null;
    st.controller = controller;
    startBusy(L.COPY.asking);
    announce(L.COPY.asking);
    // A remembered voice still being checked (it may have been deleted): wait for the check, then take the voice
    // again, so the request never names a voice the check is about to drop.
    const checking = outputs.speak ? runningVoiceCheck() : null;
    if (checking) {
      try { await checking; } catch { /* the check says nothing: the voice stands */ }
      if (st.controller !== controller || st.closed) return;   // canceled or closed meanwhile
      st.voice = currentVoice();
    }
    const body = L.requestBody({ preset: c.id, source: src, outputs, voice: outputs.speak ? st.voice : null, title: options.title || '', question });
    let data;
    try {
      data = await api.post(ASK_PATH, body, { accept: 'json', signal: controller ? controller.signal : undefined });
    } catch (error) {
      if (st.controller !== controller || st.closed) return;
      st.controller = null;
      stopTimers();
      if (error && error.kind === 'aborted') {
        showPhase('compose');
        return;
      }
      failed(error, outputs);
      return;
    }
    if (st.controller !== controller || st.closed) return;
    st.controller = null;
    if (data && data.job) {
      startJob(data.job);
      return;
    }
    stopTimers();
    const answer = L.answerOf(data);
    setAnswerAudio(outputs.speak ? answer.audio : null);
    showAnswer(answer);
    if (st.url) play();
  }

  function failed(error, outputs) {
    const copy = L.failureCopy(error);
    // The whole parsed body: a spoken sentence's audio is far longer than the cut `detail`.
    const body = outputs && outputs.speak && error ? L.failureBody(error.body != null ? error.body : error.detail) : null;
    if (body && body.sentence) copy.sentence = body.sentence;
    showPhase('compose');
    showError(copy);
    if (body && body.audio) {
      setAnswerAudio(body.audio);
      play();
    }
  }

  /* ── a job ── */

  function renderJob(job) {
    st.job = job;
    const p = L.jobProgress(job);
    el.status.textContent = L.jobStatus(job);
    el.progress.setAttribute('max', String(Math.max(1, p.total)));
    el.progress.setAttribute('value', String(p.done));
    el.progressText.textContent = p.label;
    // Progress by piece when the job reports its pieces (a long text's one part counts them
    // too); else no bar: the status says it is working, and no fraction is made up.
    setHidden(el.progressRow, st.phase !== 'job' || !p.total);
  }

  function startJob(job) {
    st.startedAt = st.startedAt || Date.now();
    showPhase('job');
    renderJob(job);
    emit('library:changed', { docId: job.document_id || '' });   // a long text is kept in the Reader
    schedulePoll();
  }

  /** The next read of the job, POLL_MS from now; never while the page is hidden (it is read at once when
   *  the page shows again, as the closed-sheet watch does). */
  function schedulePoll() {
    if (st.closed || st.pollTimer !== null || pageHidden()) return;
    st.pollTimer = setTimeout(() => {
      st.pollTimer = null;
      poll();
    }, L.POLL_MS);
  }

  function onPageVisibility() {
    if (st.closed || st.sample || st.phase !== 'job') return;   // a screenshot state's job is never read
    if (pageHidden()) {
      if (st.pollTimer !== null) clearTimeout(st.pollTimer);
      st.pollTimer = null;
    } else if (st.pollTimer === null && !st.polling) {
      poll();   // shown again: at once, so the progress (or the end) shows now
    }
  }

  async function poll() {
    const job = st.job;
    if (!job || st.closed || st.phase !== 'job' || st.polling) return;
    let next;
    st.polling = true;
    try {
      next = await api.get(`${JOBS_PATH}/${encodeURIComponent(job.id)}`, { accept: 'json' });
    } catch (error) {
      st.polling = false;
      if (st.closed || st.phase !== 'job') return;
      if (error && error.status === 404) {
        stopTimers();
        showPhase('compose');
        showError({ title: 'The job is gone', message: 'It was deleted on the server.', action: '' });
        return;
      }
      schedulePoll();   // a dropped request: the next poll tries again
      return;
    }
    st.polling = false;
    if (st.closed || st.phase !== 'job' || !st.job || st.job.id !== job.id) return;
    endOrWait(next);
  }

  function endOrWait(job) {
    renderJob(job);
    if (!L.jobEnded(job)) {
      schedulePoll();
      return;
    }
    stopTimers();
    if (job.status === 'done') {
      emit('library:changed', { docId: job.result_document_id || '' });
      showAnswer({
        reply: L.jobText(job), thinking: L.jobThinking(job), elapsedMs: 0, finishReason: '', audio: null, audioNote: '',
        documentId: typeof job.result_document_id === 'string' ? job.result_document_id : '', documentNote: '', preset: job.preset || '',
      }, { job });
    } else if (job.status === 'cancelled') {
      showPhase('compose');
      announce(L.COPY.cancelled);
    } else {
      showPhase('compose');
      showError(L.jobFailure(job));
    }
  }

  async function cancel() {
    if (st.phase === 'busy' && st.controller) {
      st.controller.abort();
      st.controller = null;
      stopTimers();
      showPhase('compose');
      return;
    }
    if (st.phase === 'job' && st.job) {
      const job = st.job;
      setBusy(el.cancel, true, { label: 'Canceling…' });
      try {
        const next = await api.del(`${JOBS_PATH}/${encodeURIComponent(job.id)}`, { accept: 'json' });
        if (!st.closed && next) endOrWait(next);
      } catch (error) {
        if (!st.closed) showError(L.failureCopy(error));
      } finally {
        setBusy(el.cancel, false);
      }
      return;
    }
    closeSheet(dialog, 'cancel');
  }

  /* ── the answer's actions ── */

  async function saveAnswer() {
    const answer = st.answer;
    if (!answer || !answer.reply || answer.documentId) return;
    const title = L.savedTitle(st.asked && st.asked.choice ? st.asked.choice.label : '', st.asked && st.asked.source);
    setBusy(el.saveDoc, true, { label: 'Saving…' });
    try {
      const doc = await api.post('/v1/reader/documents', { type: 'text', text: answer.reply, title }, { accept: 'json' });
      if (st.closed) return;
      if (doc && typeof doc.id === 'string') {
        st.answer = { ...answer, documentId: doc.id };
        emit('library:changed', { docId: doc.id });
        toast(L.COPY.savedToast, { kind: 'success', id: 'ask-ai-save' });
      }
    } catch (error) {
      if (!st.closed) showError({ ...L.failureCopy(error), title: "Couldn't save the answer" });
    } finally {
      setBusy(el.saveDoc, false);
      renderSaved();
    }
  }

  async function copyAnswer() {
    const text = st.answer ? st.answer.reply : '';
    if (!text) return;
    if (await copyText(text)) toast(L.COPY.copied, { kind: 'success', id: 'ask-ai-copy' });
    else toast(L.COPY.copyFailed, { kind: 'error', id: 'ask-ai-copy' });
  }

  function openDocument() {
    const id = st.answer ? st.answer.documentId : '';
    if (!id) return;
    closeSheet(dialog, 'open');
    navigate(`reader/${encodeURIComponent(id)}`);
  }

  function again() {
    clearError();
    showPhase('compose');
    renderCompose();
    el.send.focus();
  }

  async function pickVoice() {
    const v = st.voice;
    if (!v.model) return;
    // modelLabel: a model with no voices of its own gets the picker's note, as in Speak.
    const choiceMade = await openVoicePicker({ trigger: el.voice, model: v.model, modelLabel: modelLabel(v.model), value: v.voice, title: 'Answer voice', include: PICKER_INCLUDE });
    if (!choiceMade || st.closed) return;
    const picked = { model: choiceMade.model || v.model, voice: choiceMade.voice, label: choiceMade.label || choiceMade.voice };
    voiceChecks.set(voiceKey(picked), 'ok');   // chosen from the picker's own list just now
    remember({ voice: picked });
    st.voice = currentVoice();
    renderOutputs();
  }

  async function openSettings(where) {
    closeSheet(dialog, 'settings');
    await navigate('settings', { focus: 'heading' });
    const target = document.getElementById(where === 'ai' ? 'se-assistant-title' : 'se-key-title') || null;
    if (target) {
      if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
      target.focus();
      if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'start' });
    }
  }

  /* ── wiring ── */

  listen(el.form, 'submit', (event) => {
    event.preventDefault();
    send();
  });
  listen(el.cancel, 'click', () => { cancel(); });
  listen(el.again, 'click', () => again());
  listen(el.voice, 'click', () => { pickVoice(); });
  listen(el.play, 'click', () => { play(); });
  listen(el.saveDoc, 'click', () => { saveAnswer(); });
  listen(el.copy, 'click', () => { copyAnswer(); });
  listen(el.open, 'click', () => openDocument());
  listen(el.picks, 'click', pickQuestion);
  listen(el.recent, 'click', pickQuestion);
  listen(el.question, 'input', () => growQuestion());
  if (typeof document !== 'undefined') listen(document, 'visibilitychange', onPageVisibility);
  listen(el.sources, 'change', (event) => {
    const target = event.target;
    if (!target || target.name !== 'ask-source') return;
    st.sourceId = target.value;
    const before = st.preset;
    renderPresets();
    if (st.preset !== before) applyPresetOutputs();
    renderCompose();
  });
  listen(el.preset, 'change', () => {
    st.preset = el.preset.value;
    applyPresetOutputs();
    renderCompose();
  });
  const onOutput = () => {
    st.outputs = { show: !!el.show.checked, speak: !!el.speak.checked, save: !!el.save.checked };
    renderOutputs();
  };
  listen(el.show, 'change', onOutput);
  listen(el.speak, 'change', onOutput);
  listen(el.save, 'change', onOutput);
  listen(el.error, 'click', (event) => {
    const target = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const button = target && target.closest ? target.closest('[data-ask]') : null;
    if (!button) return;
    const what = button.getAttribute('data-ask');
    if (what === 'retry') {
      clearError();
      send();
    } else if (what === 'ai-settings') openSettings('ai');
    else if (what === 'settings') openSettings('key');
  });
  offs.push(app.select((s) => s.player, () => renderPlay()));
  const voiceChanged = () => {
    st.voice = currentVoice();
    if (st.phase === 'compose') renderOutputs();
  };
  offs.push(app.select((s) => s.speakVoice, () => {
    if (rememberedVoice()) return;
    voiceChanged();
  }));
  // A remembered voice is checked once the model list is known; one that can't speak falls back.
  const checkVoice = () => {
    startVoiceCheck().then((gone) => { if (gone && !st.closed) voiceChanged(); }, () => {});
  };
  offs.push(app.select((s) => s.models && s.models.fetchedAt, () => {
    voiceChanged();
    checkVoice();
  }));

  // The first render. A new sheet starts with an empty question (the last ones are offered as picks).
  el.title.textContent = options.heading || L.COPY.title;
  el.question.style.removeProperty('--ask-q-h');
  el.question.value = '';
  renderPresets();
  applyPresetOutputs();
  clearError();
  showPhase('compose');
  renderCompose();
  setHidden(el.result, true);
  el.reply.textContent = '';
  setHidden(el.thinking, true);
  if (!st.sample) checkVoice();

  return {
    st,
    el,
    sources,
    renderCompose,
    showAnswer,
    renderJob,
    showPhase,
    showError,
    setAnswerAudio,
    setQuestion,
    /** The sheet closed: stop the request and the polling; a job goes on on the server. */
    finish() {
      if (st.closed) return;
      st.closed = true;
      const runningJob = st.phase === 'job' && st.job && !L.jobEnded(st.job) && !st.sample;
      if (st.controller) {
        st.controller.abort();
        st.controller = null;
      }
      stopTimers();
      for (const off of offs.splice(0)) {
        try { off(); } catch { /* gone */ }
      }
      // The answer's audio stays while it plays (the mini-player has it); its release revokes it.
      if (st.url && !(ours() && st.playingUrl === st.url)) revokeUrl(st.url);
      if (runningJob) {
        // Its end is told by a toast of the same id.
        toast(L.COPY.jobGoesOn, { kind: 'info', id: jobToastId(st.job) });
        watchJob(st.job);
      }
    },
  };
}

/**
 * Open the Ask AI sheet for what a view offers to send.
 * @param {{trigger?: HTMLElement, sources: object[], source?: string, title?: string, heading?: string,
 *   outputs?: {show?: boolean, speak?: boolean, save?: boolean}}} options
 *   title: the saved document's title (else the server's); heading: the sheet's title ("Ask AI"); outputs: the
 *   view's own first outputs over the preset's (Speak's "speak the answer")
 * @returns {Promise<void>} resolves when the sheet closes
 */
export async function openAskAi(options = {}) {
  const dialog = document.getElementById('app-ask-ai');
  if (!dialog) throw new Error('The Ask AI sheet is missing from this page.');
  if (sheet) closeSheet(dialog, 'replaced');
  if (avail.phase !== 'on' || !avail.presets.length) await refreshAvailability({ force: true });
  if (avail.phase !== 'on') {
    // Turned off (or out of reach) since the button showed: say so where the listener is; nothing opens.
    const copy = avail.phase === 'off' ? L.failureCopy({ code: 'assistant_disabled' }) : L.failureCopy(avail.error);
    toast(`${copy.title}. ${copy.message}`.trim(), { kind: 'error', id: 'ask-ai-off' });
    return;
  }
  const controller = createSheet(dialog, options);
  sheet = controller;
  return new Promise((resolve) => {
    openSheet(dialog, {
      trigger: options.trigger,
      initialFocus: controller.el.send.disabled ? controller.el.preset : controller.el.send,
      // A press on the backdrop never drops an answer or a running request.
      backdropClose: () => controller.st.phase === 'compose' && !controller.st.answer,
      onClose: () => {
        controller.finish();
        if (sheet === controller) sheet = null;
        resolve();
      },
    });
  });
}

/** Whether the sheet is open. */
export function isOpen() {
  return !!sheet;
}

/* ───────────── screenshot states (display only: nothing is sent) ───────────── */

const SAMPLE_TEXT = 'Okay, so the plan for today is to finish the reader and then look at the speak view. We should ship the reader first, because people listen on their phones every evening.';

/** PURE: the sample sources a screenshot uses when the view gives none. */
export function sampleSources() {
  return [{ id: 'transcript', label: 'The transcript', input: { kind: 'transcript', text: SAMPLE_TEXT }, text: SAMPLE_TEXT, chars: L.charCount(SAMPLE_TEXT) }];
}

/**
 * Open a screenshot state (?shot=ask-ai, ask-ai-answer, ask-ai-job, ask-ai-error, ask-ai-question): the sheet as it
 * looks, with sample content where the view has none. Nothing is sent, played or saved.
 * @param {string} state one of SHOTS
 * @param {{trigger?: HTMLElement, sources?: object[], source?: string}} [options]
 */
export async function openSample(state, { trigger, sources, source } = {}) {
  if (!SHOTS.includes(state)) return false;
  const dialog = document.getElementById('app-ask-ai');
  if (!dialog) return false;
  if (sheet) closeSheet(dialog, 'replaced');
  if (avail.phase !== 'on') await refreshAvailability({ force: true });
  if (avail.phase !== 'on') {
    // The assistant is off (or unreachable): the shot still shows the sheet, with the built-in presets.
    avail = Object.freeze({ ...avail, host: avail.host || 'ai-server.lan:8080', presets: avail.presets.length ? avail.presets : [
      { id: 'voice', name: 'Voice', label: 'Answer', inputs: ['text', 'transcript'], output: 'speech', chunk_chars: 12000 },
      { id: 'book', name: 'Book', label: 'Rewrite for listening', inputs: ['document'], output: 'document', chunk_chars: 12000 },
    ] });
  }
  const given = Array.isArray(sources) && sources.length ? sources : sampleSources();
  const controller = createSheet(dialog, { sources: given, source, sample: true });
  sheet = controller;
  openSheet(dialog, {
    trigger,
    onClose: () => {
      controller.finish();
      if (sheet === controller) sheet = null;
    },
  });
  const c = L.presetChoices(avail.presets, controller.sources[0] ? controller.sources[0].input.kind : 'text').find((x) => x.id === controller.st.preset) || null;
  if (state === 'ask-ai-answer') {
    controller.st.asked = { preset: controller.st.preset, choice: c, outputs: { show: true, speak: true, save: false }, source: controller.sources[0] };
    controller.st.url = 'blob:sample';   // shows Play; a sample never plays
    controller.showAnswer({
      reply: 'Finish the Reader today, then the Speak view.\n\nShip the Reader first: people listen on their phones every evening, so it matters most.',
      thinking: 'Two tasks, in order. The reason given is when people listen.', elapsedMs: 1840, finishReason: 'stop',
      audio: null, audioNote: '', documentId: '', documentNote: '', preset: controller.st.preset,
    });
    controller.st.url = '';
  } else if (state === 'ask-ai-job') {
    controller.st.startedAt = Date.now() - 74000;
    controller.showPhase('job');
    controller.renderJob({ id: 'sample', status: 'running', chapters: [{ pieces: 4, pieces_done: 4 }, { pieces: 5, pieces_done: 2 }, { pieces: 3, pieces_done: 0 }] });
    controller.el.elapsed.textContent = '1:14';
  } else if (state === 'ask-ai-error') {
    // A failed Send with speech asked: the server's code in plain words and the preset's
    // spoken error sentence. Nothing plays.
    controller.showError({ ...L.failureCopy({ status: 502, code: 'assistant_unreachable' }), sentence: "I couldn't reach the AI server." });
  } else if (state === 'ask-ai-question') {
    // A question of the listener's own in the box. Nothing is sent or remembered.
    controller.setQuestion(SAMPLE_QUESTION);
  }
  return true;
}

/** Test hook. */
export function __resetForTests() {
  if (sheet) {
    const dialog = sheet.el.dialog;
    try { closeSheet(dialog, 'reset'); } catch { /* gone */ }
    if (sheet) sheet.finish();
  }
  sheet = null;
  avail = { phase: 'unknown', host: '', presets: [], loadedAt: 0, error: null };
  inflight = null;
  loadSeq = 0;
  listeners.clear();
  if (busOff) busOff();
  busOff = null;
  stopWatching();
  unbindPage();
  voiceChecks = new Map();
  voiceCheckRun = null;
}
