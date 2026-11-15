/**
 * Settings › AI assistant: the assistant's status, and the AI preset
 * editor with Preview and Test. It uses only the API of src/routes/assistant.py (through core/api.js) and
 * /v1/audio/speech for Preview (through core/voices.js preview(), played by core/audio.js).
 *
 * - Status (GET /api/assistant): on, the AI server's host and the default model (assistant-logic.js
 *   statusFacts); the server-only settings in words too when the Connection area can't show them. Off, how to
 *   turn it on: under Connection, or, when the connection is locked or .env-only, OS_ASSISTANT_URL in .env and a
 *   restart. The editor stays hidden and no preset is asked for. An older server without the route says so.
 * - Connection: assistant-connection.js, shown on and off, with the saved assistants above it
 *   (assistant-list.js). A change there loads the status and the presets again (unless the preset form has
 *   unsaved changes), so a saved URL turns the assistant on with no restart.
 * - A preset's Assistant (`assistant`): "The one in use" ('') or a saved assistant, shown while any is saved; its
 *   model is the preset's default model. A save sends it only when changed; Test never sends it.
 * - The picker: the built-in presets (voice, book), then yours. New (a name, POST), Duplicate (the preset as
 *   shown, unsaved changes too, POST), and Delete for yours or Restore defaults for a built-in (DELETE), each
 *   after a confirm. A taken name asks again with the server's reason.
 * - The form: every preset field, in groups. The built-in voice preset hides the book-only piece size, the book
 *   preset the voice-only groups (history, phrases, errors); a preset of your own shows all of them. The Ask AI
 *   group edits how the preset shows in the Ask AI sheet: its action name (`label`),
 *   the input kinds it is offered for (`inputs`) and its first answer choice (`output`); none reaches the AI, a
 *   save sends them only when changed, and Test never sends them.
 * - Save (the form's submit; Enter in a one-line field too) sends PUT with the name, the preset's own fields and
 *   the changed ones (saveBody): a field that comes from .env and wasn't touched keeps following .env. A 422
 *   goes on the field it names (aria-invalid, the error slot in aria-describedby, focus), with a summary banner.
 *   "Unsaved changes" and Discard changes show while the form differs from the saved preset; choosing another
 *   preset, New and Restore ask before dropping them, and leaving the page warns (beforeunload). Leaving
 *   Settings keeps them: views stay mounted, and show() doesn't reload over them.
 * - Preview (each phrase and error sentence, as typed; an empty sentence is the default one) speaks in Speak's
 *   voice (app.speakVoice): audio.prime() in the click, voices.preview() (no X-History), then an ephemeral
 *   clip on the shared element. Without a usable voice it says why and sends nothing.
 * - Test sends POST /api/assistant/test with the preset as it is on screen as overrides, busy while it runs,
 *   then shows the reply, the thinking (collapsed) and the time, or the failure in plain words.
 * - Loads: show() loads the status and the presets unless the form has unsaved changes. The shell's auth reload
 *   (reason 'auth') loads again only when the last load was refused for the key (AU-12: Clear reloads no data).
 * @module views/settings/assistant
 */
import { html, render, setHidden, setBusy, actions } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { app } from '../../core/store.js';
import * as api from '../../core/api.js';
import * as audio from '../../core/audio.js';
import { preview as previewSpeech } from '../../core/voices.js';
import { confirm, prompt } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import * as L from './assistant-logic.js';
import { bindConnectionArea } from './assistant-connection.js';
import { bindAssistantList } from './assistant-list.js';
import { findAssistant } from '../../core/assistants.js';

/** Each field's input (the lists, fillers and error_sentences, have their own rows). */
export const FIELD_INPUTS = Object.freeze({
  name: 'se-ai-name',
  // Ask AI: the inputs' error goes in #se-ai-inputs-error (setFieldError).
  label: 'se-ai-label',
  inputs: 'se-ai-inputs-text',
  output: 'se-ai-output',
  assistant: 'se-ai-assistant',
  model: 'se-ai-model',
  system_prompt: 'se-ai-system',
  user_template: 'se-ai-template',
  thinking: 'se-ai-thinking',
  reasoning_effort: 'se-ai-effort',
  thinking_budget: 'se-ai-thinking-budget',
  show_thinking: 'se-ai-show-thinking',
  max_tokens: 'se-ai-max-tokens',
  temperature: 'se-ai-temperature',
  top_p: 'se-ai-top-p',
  timeout_s: 'se-ai-timeout',
  history_turns: 'se-ai-history',
  filler_after_ms: 'se-ai-filler-after',
  filler_every_ms: 'se-ai-filler-every',
  filler_max: 'se-ai-filler-max',
  speak_errors: 'se-ai-speak-errors',
  chunk_chars: 'se-ai-chunk',
  text: 'se-ai-test-text',
});

/** The audio session id of a preview (one at a time; the next one replaces it). */
export const PREVIEW_ID = 'settings:ai-preview';

const isAuth = (error) => !!error && (error.kind === 'auth' || error.kind === 'locked' || error.status === 401);

/**
 * Bind the AI assistant section.
 * @param {HTMLElement} root the view section
 * @returns {{show: (nav?: object) => void, hide: () => void, load: () => Promise<void>, dirty: () => boolean,
 *   destroy: () => void}}
 */
export function bindAssistant(root) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    section: $('se-assistant-title') ? $('se-assistant-title').closest('section') : null,
    facts: $('se-ai-facts'),
    note: $('se-ai-server-note'),
    off: $('se-ai-off'),
    offForm: $('se-ai-off-form'),
    offEnv: $('se-ai-off-env'),
    load: $('se-ai-load'),
    editor: $('se-ai-editor'),
    preset: $('se-ai-preset'),
    presetHint: $('se-ai-preset-hint'),
    create: $('se-ai-new'),
    duplicate: $('se-ai-duplicate'),
    remove: $('se-ai-delete'),
    actionsBanner: $('se-ai-actions-banner'),
    form: $('se-ai-form'),
    assistantField: $('se-ai-assistant-field'),
    assistantPick: $('se-ai-assistant'),
    modelHint: $('se-ai-model-hint'),
    templateHint: $('se-ai-template-hint'),
    fillers: $('se-ai-fillers'),
    fillersEmpty: $('se-ai-fillers-empty'),
    fillersError: $('se-ai-fillers-error'),
    fillerAdd: $('se-ai-filler-add'),
    voiceHint: $('se-ai-voice-hint'),
    waitingBanner: $('se-ai-waiting-banner'),
    errors: $('se-ai-errors'),
    errorsError: $('se-ai-errors-error'),
    errorsBanner: $('se-ai-errors-banner'),
    testText: $('se-ai-test-text'),
    test: $('se-ai-test'),
    testStatus: $('se-ai-test-status'),
    testBanner: $('se-ai-test-banner'),
    result: $('se-ai-test-result'),
    reply: $('se-ai-test-reply'),
    thinking: $('se-ai-test-thinking'),
    meta: $('se-ai-test-meta'),
    saveBanner: $('se-ai-save-banner'),
    save: $('se-ai-save'),
    discard: $('se-ai-discard'),
    dirty: $('se-ai-dirty'),
    status: $('se-ai-status'),
  };
  const offs = [];
  const st = {
    phase: 'idle',        // idle | loading | ready | off | missing | failed
    refusedForKey: false, // the last load was refused for the API key
    status: null,         // GET /api/assistant
    presets: [],          // the presets' API answers, built-ins first
    current: null,        // the preset the form edits (as saved)
    selected: 'voice',
    fillers: [],          // the phrases as typed (blank rows included)
    loadSeq: 0,
    busy: false,          // a save, create, duplicate or delete is running
    testing: false,
    previewSeq: 0,
    invalid: [],          // [{input, slot}] fields showing an error
  };
  if (!els.section || !els.form) return { show() {}, hide() {}, load: async () => {}, dirty: () => false, destroy() {} };

  const listen = (el, type, fn, options) => {
    if (!el) return;
    el.addEventListener(type, fn, options);
    offs.push(() => el.removeEventListener(type, fn, options));
  };
  const setText = (el, text) => { if (el && el.textContent !== text) el.textContent = text; };

  /* ── banners and status lines ── */

  /** An inline banner: an error is role=alert; an info note (an older server) is plain text, never announced. */
  function showBanner(host, { title, message = '', detail = '', actions: buttons = [], tone = 'error' }) {
    if (!host) return;
    const banner = tone === 'error'
      ? html`<div class="banner banner--error se-ai__banner" role="alert">`
      : html`<div class="banner banner--info se-ai__banner">`;
    render(host, html`${banner}${icon(tone === 'error' ? 'alert' : 'info', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${title}</p>${message ? html`<p class="banner__msg">${message}</p>` : ''}${detail ? html`<details class="banner__details"><summary>Details</summary><pre></pre></details>` : ''}</div>${buttons.length ? html`<div class="banner__actions">${buttons.map((b) => html`<button type="button" class="btn btn--secondary btn--sm" data-action="${b.action}"><span class="btn__label">${b.label}</span></button>`)}</div>` : ''}</div>`);
    const pre = host.querySelector('details pre');
    if (pre) pre.textContent = String(detail);
    setHidden(host, false);
  }

  function clearBanner(host) {
    if (!host || host.hidden) return;
    render(host, html``);
    setHidden(host, true);
  }

  const clearBanners = () => [els.actionsBanner, els.saveBanner, els.testBanner, els.waitingBanner, els.errorsBanner].forEach(clearBanner);
  const setStatus = (text) => setText(els.status, text);
  const setTestStatus = (text) => setText(els.testStatus, text);

  /* ── the status ── */

  function renderFacts(list) {
    render(els.facts, html`${list.map((f) => html`<dt>${f.term}</dt><dd class="${f.mono ? 'mono' : ''}">${f.value}</dd>`)}`);
  }

  // The Connection area and the saved assistants: a change there loads the status and the presets again.
  let list = null;
  const reload = () => { load({ refresh: true }).catch(() => {}); };
  const conn = bindConnectionArea(root, {
    showBanner, clearBanner,
    onChange: reload,
    onMode: () => syncConnectionNotes(),
    onList: () => {
      if (list) list.render();
      syncAssistantPick();
    },
  });
  list = bindAssistantList(root, { conn, showBanner, clearBanner, onChange: reload });
  list.render();

  /** The status list: short while the Connection area shows every setting with its source. */
  const renderStatus = () => renderFacts(L.statusFacts(st.status, { brief: conn.shown() }));

  /** How to turn it on, and the .env-only note, follow the Connection area's mode. */
  function syncConnectionNotes() {
    const mode = conn.mode();
    const envOnly = mode === 'locked' || mode === 'missing';
    setHidden(els.offForm, envOnly);
    setHidden(els.offEnv, !envOnly);
    setHidden(els.note, !(st.phase === 'ready' && mode === 'missing'));
    if (st.status && (st.phase === 'ready' || st.phase === 'off')) renderStatus();
  }

  function showPhase(phase) {
    st.phase = phase;
    setHidden(els.off, phase !== 'off');
    setHidden(els.editor, phase !== 'ready');
    conn.setVisible(phase === 'ready' || phase === 'off');
    syncConnectionNotes();
    if (phase === 'ready' || phase === 'off' || phase === 'loading') clearBanner(els.load);
  }

  /**
   * Load the status, the connection, then (when on) the presets. Keeps the chosen preset when it still exists.
   * `refresh` (after a connection change): the connection is already shown, nothing hides while it loads, and a
   * preset form with unsaved changes is kept.
   */
  async function load({ refresh = false } = {}) {
    const seq = (st.loadSeq += 1);
    if (st.phase !== 'ready' && !refresh) {
      renderFacts([{ term: 'Status', value: 'Checking…' }]);
      showPhase('loading');
    }
    try {
      const status = await api.get('/api/assistant', { accept: 'json' });
      if (seq !== st.loadSeq) return;
      st.status = status || {};
      st.refusedForKey = false;
      if (!refresh) {
        await conn.load();
        if (seq !== st.loadSeq) return;
      }
      renderStatus();
      if (!st.status.enabled) {
        showPhase('off');
        return;
      }
      if (refresh && st.phase === 'ready' && isDirty()) {
        syncModelHint();
        return;
      }
      const data = await api.get('/api/assistant/presets', { accept: 'json' });
      if (seq !== st.loadSeq) return;
      st.presets = Array.isArray(data && data.presets) ? data.presets.filter((p) => p && typeof p.id === 'string') : [];
      if (!st.presets.some((p) => p.id === st.selected)) st.selected = st.presets.length ? st.presets[0].id : 'voice';
      renderPicker();
      showPhase('ready');
      fill(st.presets.find((p) => p.id === st.selected) || null);
    } catch (error) {
      if (seq !== st.loadSeq) return;
      loadFailed(error);
    }
  }

  function loadFailed(error) {
    if (error && error.kind === 'aborted') return;
    if (error && error.code === 'assistant_disabled') {
      // Turned off between the two requests: the status says so on the next load.
      renderFacts(L.statusFacts({ enabled: false }));
      showPhase('off');
      return;
    }
    renderFacts([{ term: 'Status', value: 'Unknown' }]);
    showPhase('failed');
    if (isAuth(error)) {
      st.refusedForKey = true;
      showBanner(els.load, { title: 'API key needed', message: "Enter the server's API key in Settings › Connection. This section loads once the key is accepted." });
      return;
    }
    if (error && error.status === 404) {
      st.phase = 'missing';
      renderFacts([{ term: 'Status', value: 'Not available' }]);
      showBanner(els.load, { tone: 'info', title: "This server doesn't have the AI assistant", message: 'It comes with a newer Open Speech.' });
      return;
    }
    const e = api.explain(error);
    showBanner(els.load, { title: "Couldn't load the AI assistant", message: e.message || 'Something went wrong.', detail: e.detailText, actions: [{ label: 'Retry', action: 'ai-retry' }] });
  }

  /* ── the picker ── */

  function renderPicker() {
    render(els.preset, html`${st.presets.map((p) => html`<option value="${p.id}">${L.optionLabel(p)}</option>`)}`);
    if (els.preset.value !== st.selected) els.preset.value = st.selected;
  }

  const viewOf = (id) => st.presets.find((p) => p.id === id) || null;

  function replacePreset(view) {
    st.presets = st.presets.map((p) => (p.id === view.id ? view : p));
    renderPicker();
  }

  function addPreset(view) {
    st.presets = [...st.presets.filter((p) => p.id !== view.id), view];
    st.selected = view.id;
    renderPicker();
    fill(view);
  }

  /* ── the form ── */

  /** Make sure a select can show a value the server sent (an effort this page doesn't list yet). */
  function ensureOption(select, value) {
    if (!select || value === '' || Array.from(select.querySelectorAll('option')).some((o) => o.value === value)) return;
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }

  const input = (field) => $(FIELD_INPUTS[field]);

  /** Show a preset in the form (as saved), with its hints and the fields its purpose uses. */
  function fill(view) {
    const before = st.current;
    st.current = view;
    clearFieldErrors();
    if (!view || !before || before.id !== view.id) {
      // The last Test was another preset's.
      setHidden(els.result, true);
      setTestStatus('');
      clearBanner(els.testBanner);
    }
    if (!view) {
      updateDirty();
      return;
    }
    const v = L.presetValues(view);
    syncAssistantPick(v.assistant);
    for (const field of ['name', 'label', 'model', 'system_prompt', 'user_template']) input(field).value = v[field];
    input('output').value = v.output;
    for (const kind of L.INPUT_KINDS) {
      const box = $(`se-ai-inputs-${kind}`);
      if (box) box.checked = v.inputs.includes(kind);
    }
    ensureOption(input('reasoning_effort'), v.reasoning_effort);
    input('thinking').value = v.thinking;
    input('reasoning_effort').value = v.reasoning_effort;
    input('show_thinking').checked = v.show_thinking;
    input('speak_errors').checked = v.speak_errors;
    for (const field of L.NUMBER_FIELDS) input(field).value = v[field] === null ? '' : String(v[field]);
    st.fillers = v.fillers.slice();
    renderFillers();
    renderErrors(v.error_sentences);
    const use = L.presetUse(view.id);
    for (const el of root.querySelectorAll('[data-ai-use]')) {
      const want = el.getAttribute('data-ai-use');
      setHidden(el, use !== 'both' && want !== use);
    }
    setText(els.presetHint, L.presetHint(view));
    const names = L.placeholders(view.id).map((name) => `{${name}}`);
    setText(els.templateHint, view.id === 'voice'
      ? 'Wraps your words into the message. It must contain {text}, the only placeholder the built-in voice preset takes.'
      : `Wraps your words, or a piece of a book, into the message. It must contain {text}. It may also use ${names.slice(1).join(', ').replace(/, ([^,]*)$/, ' and $1')}: the book's title, author and chapter title.`);
    syncModelHint();
    render(els.remove, html`${icon(view.builtin ? 'refresh' : 'trash')}<span class="btn__label">${view.builtin ? 'Restore defaults' : 'Delete'}</span>`);
    els.remove.classList.toggle('btn--danger-ghost', !view.builtin);
    els.remove.classList.toggle('btn--ghost', !!view.builtin);
    els.remove.setAttribute('aria-disabled', view.builtin && !view.customized ? 'true' : 'false');
    updateDirty();
  }

  /**
   * The Assistant choices: "The one in use" and each saved assistant, the field shown while any is saved; one the
   * preset names that is no longer saved stays, said so, until another is chosen.
   */
  function syncAssistantPick(value = els.assistantPick ? els.assistantPick.value : '') {
    if (!els.assistantPick) return;
    const saved = conn.state().saved;
    const all = saved && !saved.locked ? saved.assistants : [];
    const gone = !!value && !all.some((a) => a.id === value);
    render(els.assistantPick, html`<option value="">The one in use</option>${all.map((a) => html`<option value="${a.id}">${a.name}</option>`)}${gone ? html`<option value="${value}">No longer saved: the one in use answers</option>` : ''}`);
    els.assistantPick.value = value;
    setHidden(els.assistantField, !all.length && !gone);
    syncModelHint();
  }

  /** The preset's model hint names its default model: its assistant's, else the connection's in use. */
  function syncModelHint() {
    const named = els.assistantPick ? findAssistant(conn.state().saved, els.assistantPick.value) : null;
    if (named) {
      const own = String(named.model || '');
      setText(els.modelHint, own ? `Empty uses the model of “${named.name}”, ${own}.` : `Empty sends no model name, so the AI server of “${named.name}” picks.`);
      return;
    }
    const model = st.status && st.status.model ? String(st.status.model) : '';
    setText(els.modelHint, model ? `Empty uses the server's default, ${model}.` : 'Empty sends no model name, so the AI server picks.');
  }

  function renderFillers({ focusIndex = -1, focusAction = '' } = {}) {
    const count = st.fillers.length;
    render(els.fillers, html`${st.fillers.map((text, i) => html`<li class="se-ai__filler">
      <input class="input se-ai__filler-input" id="se-ai-filler-${i}" type="text" maxlength="${L.FILLER_MAX_CHARS}" autocomplete="off" aria-label="Phrase ${i + 1}" data-index="${i}" value="${text}">
      <span class="se-ai__row-actions">
        <button type="button" class="icon-btn" data-action="ai-preview-filler" data-index="${i}" aria-label="Preview phrase ${i + 1}" data-tooltip="Preview">${icon('play')}</button>
        <button type="button" class="icon-btn" data-action="ai-filler-up" data-index="${i}" aria-label="Move phrase ${i + 1} up" aria-disabled="${i === 0 ? 'true' : 'false'}" data-tooltip="Move up">${icon('arrow-up')}</button>
        <button type="button" class="icon-btn" data-action="ai-filler-down" data-index="${i}" aria-label="Move phrase ${i + 1} down" aria-disabled="${i === count - 1 ? 'true' : 'false'}" data-tooltip="Move down">${icon('arrow-down')}</button>
        <button type="button" class="icon-btn icon-btn--danger" data-action="ai-filler-remove" data-index="${i}" aria-label="Remove phrase ${i + 1}" data-tooltip="Remove">${icon('trash')}</button>
      </span>
    </li>`)}`);
    setHidden(els.fillersEmpty, count > 0);
    els.fillerAdd.setAttribute('aria-disabled', count >= L.MAX_FILLERS ? 'true' : 'false');
    if (focusIndex >= 0) {
      const row = els.fillers.querySelectorAll('li')[focusIndex];
      const target = row ? (focusAction ? row.querySelector(`[data-action="${focusAction}"]`) : row.querySelector('input')) : null;
      if (target) target.focus();
    }
  }

  function renderErrors(sentences) {
    render(els.errors, html`${Object.entries(sentences).map(([code, sentence]) => {
      const label = L.ERROR_LABELS[code] || code;
      return html`<div class="field se-ai__error">
        <label class="field__label" for="se-ai-err-${code}">${label}</label>
        <span class="se-ai__line"><input class="input" id="se-ai-err-${code}" type="text" maxlength="${L.SENTENCE_MAX_CHARS}" autocomplete="off" data-code="${code}" placeholder="${L.DEFAULT_ERROR_SENTENCES[code] || ''}" value="${sentence}"><button type="button" class="icon-btn" data-action="ai-preview-error" data-code="${code}" aria-label="Preview the sentence for: ${label}" data-tooltip="Preview">${icon('play')}</button></span>
      </div>`;
    })}`);
  }

  /** A number field as typed; a browser that refused the text (badInput) gives '' for it, so say so. */
  function numberRaw(el) {
    if (!el) return '';
    if (el.validity && el.validity.badInput) return 'not a number';
    return el.value;
  }

  function readValues() {
    const raw = {
      name: input('name').value,
      label: input('label').value,
      inputs: L.INPUT_KINDS.filter((kind) => { const box = $(`se-ai-inputs-${kind}`); return !!(box && box.checked); }),
      output: input('output').value,
      assistant: els.assistantPick ? els.assistantPick.value : '',
      model: input('model').value,
      system_prompt: input('system_prompt').value,
      user_template: input('user_template').value,
      thinking: input('thinking').value,
      reasoning_effort: input('reasoning_effort').value,
      show_thinking: !!input('show_thinking').checked,
      speak_errors: !!input('speak_errors').checked,
      fillers: st.fillers.slice(),
      error_sentences: {},
    };
    for (const field of L.NUMBER_FIELDS) raw[field] = numberRaw(input(field));
    for (const el of els.errors.querySelectorAll('input[data-code]')) raw.error_sentences[el.getAttribute('data-code')] = el.value;
    return L.formValues(raw);
  }

  function isDirty() {
    if (!st.current || st.phase !== 'ready') return false;
    const { values, problems } = readValues();
    return problems.length > 0 || L.isDirty(st.current, values);
  }

  function updateDirty() {
    const dirty = isDirty();
    setHidden(els.dirty, !dirty);
    setHidden(els.discard, !dirty);
    els.save.setAttribute('aria-disabled', dirty ? 'false' : 'true');
  }

  /* ── field errors ── */

  function describedBy(el, id, on) {
    const ids = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean).filter((x) => x !== id);
    if (on) ids.push(id);
    if (ids.length) el.setAttribute('aria-describedby', ids.join(' '));
    else el.removeAttribute('aria-describedby');
  }

  function setSlot(slot, text) {
    const span = slot.querySelector('span');
    setText(span || slot, text);
    setHidden(slot, !text);
  }

  function clearFieldError(el) {
    const entry = st.invalid.find((x) => x.input === el);
    if (!entry) return;
    st.invalid = st.invalid.filter((x) => x !== entry);
    el.removeAttribute('aria-invalid');
    if (!st.invalid.some((x) => x.slot === entry.slot)) setSlot(entry.slot, '');
    if (entry.added) describedBy(el, entry.slot.id, false);
  }

  function clearFieldErrors() {
    for (const entry of st.invalid.slice()) clearFieldError(entry.input);
    for (const slot of root.querySelectorAll('.se-ai .field__error')) if (!slot.hidden) setSlot(slot, '');
    st.invalid = [];
  }

  /**
   * Show a problem on its field. `index` is the server's index among the phrases it was sent (blank rows are
   * dropped before sending), `code` an error sentence's code.
   * @returns {HTMLElement|null} the field, or null when the form has no such field
   */
  function setFieldError({ field, index = null, code = '', text }) {
    let el = null;
    let slot = null;
    if (field === 'fillers') {
      const sent = st.fillers.map((s, i) => (String(s).trim() ? i : -1)).filter((i) => i >= 0);
      const at = index !== null && index < sent.length ? sent[index] : -1;
      el = at >= 0 ? els.fillers.querySelectorAll('input')[at] : els.fillers.querySelector('input') || els.fillerAdd;
      slot = els.fillersError;
    } else if (field === 'error_sentences') {
      el = (code && els.errors.querySelector(`input[data-code="${code}"]`)) || els.errors.querySelector('input');
      slot = els.errorsError;
    } else if (field === 'inputs') {
      el = input('inputs');
      slot = $('se-ai-inputs-error');
    } else if (FIELD_INPUTS[field]) {
      el = input(field);
      slot = $(`${FIELD_INPUTS[field]}-error`);
    }
    if (!el || !slot) return null;
    setSlot(slot, text);
    el.setAttribute('aria-invalid', 'true');
    const had = (el.getAttribute('aria-describedby') || '').split(/\s+/).includes(slot.id);
    if (!had) describedBy(el, slot.id, true);
    st.invalid.push({ input: el, slot, added: !had });
    return el;
  }

  /** Field problems found here (a number that isn't one): mark them all, focus the first. */
  function showProblems(problems) {
    let first = null;
    for (const p of problems) {
      const el = setFieldError({ field: p.field, text: p.text });
      if (!first) first = el;
    }
    if (first) first.focus();
  }

  /**
   * A 422 from the server: on its field when the form has it (focused), and in the banner `host` with `title`.
   * @returns {boolean} whether it was a validation error
   */
  function showValidation(error, host, title) {
    if (!error || error.code !== 'validation_error') return false;
    const first = (error.fields || [])[0] || {};
    const problem = L.fieldProblem(first.field, first.message || error.detailText || '');
    const el = problem ? setFieldError(problem) : null;
    if (el) {
      el.focus();
      showBanner(host, { title, message: `Check the highlighted field: ${problem.text}` });
    } else {
      showBanner(host, { title, message: (problem && problem.text) || first.message || error.detailText || 'Some values were not accepted.' });
    }
    return true;
  }

  function failed(host, title, error) {
    if (!error || error.kind === 'aborted') return;
    if (showValidation(error, host, title)) return;
    if (error.code === 'assistant_disabled') {
      load().catch(() => {});
      return;
    }
    if (error.code === 'preset_not_found') {
      showBanner(host, { title, message: 'This preset no longer exists on the server. The list was loaded again.' });
      load().catch(() => {});
      return;
    }
    const e = api.explain(error);
    showBanner(host, { title, message: e.message || 'Something went wrong.', detail: e.detailText });
  }

  /* ── save, new, duplicate, delete ── */

  async function discardOk() {
    if (!isDirty()) return true;
    return confirm({ title: `Discard your changes to “${st.current.name}”?`, message: "They haven't been saved.", confirmLabel: 'Discard', cancelLabel: 'Keep editing' });
  }

  async function save() {
    if (st.busy || !st.current || els.save.getAttribute('aria-disabled') === 'true') return;
    clearBanners();
    clearFieldErrors();
    const { values, problems } = readValues();
    if (problems.length) {
      showProblems(problems);
      return;
    }
    const view = st.current;
    st.busy = true;
    setBusy(els.save, true, { label: 'Saving…' });
    setStatus(`Saving “${view.name}”…`);
    try {
      const saved = await api.put(`/api/assistant/presets/${encodeURIComponent(view.id)}`, L.saveBody(view, values), { accept: 'json' });
      setStatus('');
      if (saved && saved.id === view.id) {
        replacePreset(saved);
        if (st.current === view) fill(saved);
      }
      toast(`Saved “${(saved && saved.name) || view.name}”`, { kind: 'success' });
    } catch (error) {
      setStatus('');
      failed(els.saveBanner, `Couldn't save “${view.name}”`, error);
    } finally {
      st.busy = false;
      setBusy(els.save, false);
      updateDirty();
    }
  }

  /** Ask for a name and POST a preset; a taken name (or one the server refuses) asks again with the reason. */
  async function createFrom({ title, label, value = '', hint, confirmLabel, body }) {
    let error = '';
    let suggested = value;
    for (;;) {
      const name = await prompt({ title, label, value: suggested, hint, maxLength: 80, confirmLabel, error });
      if (name === null) return null;
      suggested = name;
      st.busy = true;
      try {
        const made = await api.post('/api/assistant/presets', body(name), { accept: 'json' });
        return made;
      } catch (failure) {
        const nameProblem = failure && failure.code === 'validation_error' && (failure.fields || [])[0] && failure.fields[0].field === 'name';
        if ((failure && failure.code === 'preset_exists') || nameProblem) {
          error = nameProblem ? L.fieldProblem('name', failure.fields[0].message).text : failure.message;
          continue;
        }
        throw failure;
      } finally {
        st.busy = false;
      }
    }
  }

  async function createNew() {
    if (st.busy) return;
    clearBanners();
    if (!(await discardOk())) return;
    try {
      const made = await createFrom({
        title: 'New AI preset', label: 'Name', confirmLabel: 'Create',
        hint: "It starts from the server's defaults. Its id is made from the name.",
        body: (name) => ({ name }),
      });
      if (!made) return;
      addPreset(made);
      toast(`Created “${made.name}”`, { kind: 'success' });
    } catch (error) {
      failed(els.actionsBanner, "Couldn't create the preset", error);
    }
  }

  async function duplicate() {
    if (st.busy || !st.current) return;
    clearBanners();
    clearFieldErrors();
    const { values, problems } = readValues();
    if (problems.length) {
      showProblems(problems);
      return;
    }
    const source = st.current;
    try {
      const made = await createFrom({
        title: `Duplicate “${source.name}”`, label: 'Name of the copy', value: `${values.name.trim() || source.name} copy`, confirmLabel: 'Duplicate',
        hint: 'The copy gets the values shown now, unsaved changes too.',
        body: (name) => L.saveBody(source, { ...values, name }, { copy: true }),
      });
      if (!made) return;
      addPreset(made);
      toast(`Created “${made.name}”`, { kind: 'success' });
    } catch (error) {
      failed(els.actionsBanner, `Couldn't duplicate “${source.name}”`, error);
    }
  }

  async function removeOrRestore() {
    const view = st.current;
    if (st.busy || !view || els.remove.getAttribute('aria-disabled') === 'true') return;
    clearBanners();
    const ok = view.builtin
      ? await confirm({
        title: `Restore the defaults of “${view.name}”?`,
        message: `Its prompts, limits, phrases and sentences go back to the server's defaults.${isDirty() ? ' Your unsaved changes are dropped too.' : ''}`,
        confirmLabel: 'Restore defaults',
      })
      : await confirm({
        title: `Delete the AI preset “${view.name}”?`,
        message: `A client that asks for it by its id, ${view.id}, gets an error instead.`,
        confirmLabel: 'Delete', danger: true,
      });
    if (!ok || st.current !== view) return;
    st.busy = true;
    setBusy(els.remove, true);
    try {
      const answer = await api.del(`/api/assistant/presets/${encodeURIComponent(view.id)}`, { accept: 'json' });
      if (view.builtin) {
        const restored = answer && answer.id === view.id ? answer : { ...view, customized: false };
        replacePreset(restored);
        fill(restored);
        toast(`Restored the defaults of “${view.name}”`, { kind: 'success' });
      } else {
        st.presets = st.presets.filter((p) => p.id !== view.id);
        st.selected = viewOf('voice') ? 'voice' : (st.presets[0] ? st.presets[0].id : '');
        renderPicker();
        fill(viewOf(st.selected));
        toast(`Deleted “${view.name}”`, { kind: 'success' });
      }
    } catch (error) {
      failed(els.actionsBanner, view.builtin ? `Couldn't restore “${view.name}”` : `Couldn't delete “${view.name}”`, error);
    } finally {
      st.busy = false;
      setBusy(els.remove, false);
      // fill() set aria-disabled from the preset; setBusy only restored `disabled`.
    }
  }

  /* ── Test ── */

  async function runTest() {
    if (st.testing || !st.current) return;
    clearBanner(els.testBanner);
    clearFieldErrors();
    const { values, problems } = readValues();
    if (problems.length) {
      showProblems(problems);
      return;
    }
    const view = st.current;
    st.testing = true;
    setBusy(els.test, true, { label: 'Testing…' });
    setTestStatus('Asking the AI…');
    setHidden(els.result, true);
    // The Test box is the preset's on screen: an answer that comes after another preset was picked is
    // that one's no longer, and fill() already cleared the box. By id, so a save of this preset still shows it.
    const stillShown = () => !!st.current && st.current.id === view.id;
    try {
      const res = await api.post('/api/assistant/test', L.testBody(view.id, els.testText.value, values), { accept: 'json' }) || {};
      if (!stillShown()) return;
      const took = L.elapsedText(res.elapsed_ms);
      setText(els.reply, String(res.reply ?? ''));
      const thinking = typeof res.thinking === 'string' ? res.thinking : '';
      els.thinking.removeAttribute('open');
      setText(els.thinking.querySelector('pre'), thinking);
      setHidden(els.thinking, !thinking);
      setText(els.meta, `Answered in ${took}${res.finish_reason ? ` · finish reason: ${res.finish_reason}` : ''}`);
      setHidden(els.result, false);
      setTestStatus(`Answered in ${took}`);
    } catch (error) {
      if (!stillShown()) return;
      testFailed(error, view);
    } finally {
      st.testing = false;
      setBusy(els.test, false);
    }
  }

  function testFailed(error, view) {
    if (!error || error.kind === 'aborted') {
      setTestStatus('');
      return;
    }
    const title = `Couldn't test “${view.name}”`;
    if (showValidation(error, els.testBanner, title)) {
      setTestStatus('');
      return;
    }
    const copy = L.testFailure(error.code);
    if (copy) {
      const elapsed = L.elapsedFrom(error.detail);
      setTestStatus(elapsed === null ? 'Failed' : `Failed after ${L.elapsedText(elapsed)}`);
      showBanner(els.testBanner, { title: copy.title, message: copy.message, detail: error.detailText || error.message });
      return;
    }
    setTestStatus('');
    failed(els.testBanner, title, error);
  }

  /* ── Preview ── */

  function syncVoiceHint() {
    const voice = L.previewVoice(app.get().speakVoice);
    setText(els.voiceHint, voice.problem ? voice.problem : `Preview speaks in Speak's voice: ${voice.label}.`);
  }

  /** Speak `text` in Speak's voice. MUST run inside the click: it primes the shared element first (iOS). */
  async function runPreview(button, text, host) {
    clearBanner(host);
    const voice = L.previewVoice(app.get().speakVoice);
    if (voice.problem) {
      showBanner(host, { title: "Can't preview", message: voice.problem });
      return;
    }
    const words = String(text || '').trim();
    if (!words) {
      showBanner(host, { title: "Can't preview", message: 'Type the phrase first.' });
      return;
    }
    audio.prime();
    const seq = (st.previewSeq += 1);
    button.setAttribute('aria-busy', 'true');
    try {
      const { url } = await previewSpeech(voice.model, voice.voice, { text: words });
      if (seq !== st.previewSeq) return;
      await audio.playClip({
        id: PREVIEW_ID, owner: 'settings', title: `“${words}”`, subtitle: `Preview · ${voice.label}`, homeRoute: 'settings',
        src: url, ephemeral: true,
      });
    } catch (error) {
      if (error instanceof audio.CaptureActiveError) {
        showBanner(host, { title: "Can't preview", message: error.message });
      } else if (!(error && error.kind === 'aborted')) {
        const e = api.explain(error);
        showBanner(host, { title: "Couldn't preview", message: e.message || 'Something went wrong.', detail: e.detailText });
      }
    } finally {
      button.removeAttribute('aria-busy');
    }
  }

  /* ── events ── */

  listen(els.form, 'submit', (event) => {
    event.preventDefault();
    save().catch(() => {});
  });

  const onEdit = (event) => {
    const target = event.target;
    if (!target || target === els.testText) {
      if (target) clearFieldError(target);
      return;
    }
    if (target.classList && target.classList.contains('se-ai__filler-input')) {
      const index = Number(target.getAttribute('data-index'));
      if (Number.isInteger(index) && index >= 0 && index < st.fillers.length) st.fillers[index] = target.value;
    }
    clearFieldError(target);
    if (target === els.assistantPick) syncModelHint();
    updateDirty();
  };
  listen(els.form, 'input', onEdit);
  listen(els.form, 'change', onEdit);

  listen(els.preset, 'change', () => {
    const next = els.preset.value;
    if (!st.current || next === st.current.id) return;
    discardOk().then((ok) => {
      if (!ok) {
        els.preset.value = st.current.id;
        return;
      }
      st.selected = next;
      clearBanners();
      fill(viewOf(next));
    }).catch(() => {});
  });

  listen(els.create, 'click', () => { createNew().catch(() => {}); });
  listen(els.duplicate, 'click', () => { duplicate().catch(() => {}); });
  listen(els.remove, 'click', () => { removeOrRestore().catch(() => {}); });
  listen(els.test, 'click', () => { runTest().catch(() => {}); });
  listen(els.discard, 'click', () => {
    clearBanners();
    fill(st.current);
    els.preset.focus();
  });
  listen(els.fillerAdd, 'click', () => {
    if (els.fillerAdd.getAttribute('aria-disabled') === 'true' || st.fillers.length >= L.MAX_FILLERS) return;
    st.fillers.push('');
    renderFillers({ focusIndex: st.fillers.length - 1 });
    updateDirty();
  });

  const indexOf = (el) => {
    const i = Number(el.getAttribute('data-index'));
    return Number.isInteger(i) && i >= 0 && i < st.fillers.length ? i : -1;
  };
  const move = (from, to, focusAction) => {
    if (from < 0 || to < 0 || to >= st.fillers.length) return;
    const [item] = st.fillers.splice(from, 1);
    st.fillers.splice(to, 0, item);
    clearFieldErrors();
    renderFillers({ focusIndex: to, focusAction });
    updateDirty();
  };
  offs.push(actions(els.section, {
    'ai-retry': () => load(),
    'ai-filler-up': (el) => { const i = indexOf(el); move(i, i - 1, 'ai-filler-up'); },
    'ai-filler-down': (el) => { const i = indexOf(el); move(i, i + 1, 'ai-filler-down'); },
    'ai-filler-remove': (el) => {
      const i = indexOf(el);
      if (i < 0) return;
      st.fillers.splice(i, 1);
      clearFieldErrors();
      const next = st.fillers.length ? Math.min(i, st.fillers.length - 1) : -1;
      renderFillers({ focusIndex: next, focusAction: 'ai-filler-remove' });
      if (next < 0) els.fillerAdd.focus();
      updateDirty();
    },
    'ai-preview-filler': (el) => {
      const i = indexOf(el);
      return runPreview(el, i >= 0 ? st.fillers[i] : '', els.waitingBanner);
    },
    'ai-preview-error': (el) => {
      const code = el.getAttribute('data-code') || '';
      const field = els.errors.querySelector(`input[data-code="${code}"]`);
      const typed = field ? field.value.trim() : '';
      return runPreview(el, typed || L.DEFAULT_ERROR_SENTENCES[code] || '', els.errorsBanner);
    },
  }, { onError: (error) => { if (globalThis.console) globalThis.console.error(error); } }));

  if (typeof window !== 'undefined') {
    // Leaving the page (a reload, a closed tab) with unsaved changes: the browser asks first.
    listen(window, 'beforeunload', (event) => {
      if (!isDirty() && !conn.dirty()) return;
      event.preventDefault();
      try { event.returnValue = ''; } catch { /* a read-only returnValue: preventDefault() is the standard way */ }
    });
  }
  offs.push(app.select((s) => s.speakVoice, () => syncVoiceHint(), { immediate: true }));

  return {
    /**
     * Settings shows (or the shell reloads it): load unless the preset form or the connection has unsaved
     * changes or a load or save is running. The auth reload (a new key) loads only when the last load was
     * refused for the key.
     */
    show(nav) {
      const reason = nav && nav.reason;
      if (reason === 'auth' && !st.refusedForKey) return;
      if (st.busy || st.phase === 'loading' || isDirty() || conn.dirty()) return;
      load().catch(() => {});
    },
    hide() {},
    load,
    dirty: () => isDirty() || conn.dirty(),
    connection: conn,
    destroy() {
      st.loadSeq += 1;
      list.destroy();
      conn.destroy();
      for (const off of offs.splice(0)) off();
    },
  };
}
