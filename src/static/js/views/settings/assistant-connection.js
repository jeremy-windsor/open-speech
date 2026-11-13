/**
 * Settings › AI assistant › Connection: the AI server's URL, model,
 * key and connection settings, edited in the web app over the server's .env. It uses only the HTTP API of
 * src/routes/assistant.py, through core/api.js; the pure parts are assistant-connection-logic.js.
 *
 * - Load (GET /api/assistant/connection): each field shows its effective value and where it comes from ("from
 *   .env", "saved here", "default"), down to "Stream replies" (the field stream, OS_ASSISTANT_STREAM, on by
 *   default), a switch like "Check TLS certificate" that only voice turns use. The
 *   key field never shows a key, only "Set" or "Not set". A field saved here
 *   has Use .env, which forgets it (PUT {field: null}; the key asks first). A load that fails says so with Retry;
 *   a server without this API (404) hides the area, and the section says the connection is .env-only.
 * - Locked (`locked`, OS_ASSISTANT_CONNECTION_LOCKED=true): a read-only list that says the connection is set in
 *   .env; Check still checks the connection in use.
 * - Check (POST …/check) sends the form as it is, unsaved: only the changed fields and a typed key, so {} checks
 *   the connection in use. It generates nothing. A reachable server's models fill "Models on the server", which
 *   puts the one chosen in Model; a typed model works too. A failure says what went wrong, with the server's
 *   message in Details. An edit of the URL, key or TLS check hides the last result, which was for the old values.
 * - Save (PUT; Enter in a one-line field too) sends only the changed fields and a typed key: every field a PUT is
 *   sent is saved, so an untouched field keeps following .env. Revert to .env (DELETE) asks first. Clear key asks,
 *   then saves an empty key. Clear key and Use .env change one field and keep the other edits on screen.
 * - The key stays with its server: while the URL field points at another origin than the one
 *   loaded and no key is typed, the key field says "Enter the key again for a new server"; saving then clears
 *   the key, and the toast says so. So does turning "Check TLS certificate" off for an https URL,
 *   in its own words.
 * - A 422 goes on the field it names (aria-invalid, the error in aria-describedby, focus) with a banner; a 403
 *   (locked since the page loaded) says so and loads the area again.
 * - After a change, `onChange(view)` lets the section load its status and presets again, so a saved URL turns
 *   the assistant on with no restart. `onMode(mode)` tells it when the area's mode changes.
 * @module views/settings/assistant-connection
 */
import { html, render, setHidden, setBusy, actions } from '../../core/dom.js';
import * as api from '../../core/api.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { elapsedText, elapsedFrom } from './assistant-logic.js';
import * as C from './assistant-connection-logic.js';

/** Each field's control; `${id}-source`, `-reset` and (for the ones a 422 can name) `-error` sit beside it. */
export const CONN_INPUTS = Object.freeze({
  url: 'se-ai-conn-url',
  api_key: 'se-ai-conn-key',
  model: 'se-ai-conn-model',
  verify_tls: 'se-ai-conn-tls',
  template_kwargs: 'se-ai-conn-template',
  max_tokens_field: 'se-ai-conn-max-field',
  extra_body: 'se-ai-conn-extra',
  stream: 'se-ai-conn-stream',
});

/** The one-line fields where Enter saves. */
const ENTER_SAVES = ['url', 'model', 'api_key'];
/** The fields whose edit makes the last Check out of date (they decide what /models answers). */
const CHECK_FIELDS = ['url', 'api_key', 'verify_tls'];

/**
 * Bind the Connection area of the AI assistant section.
 * @param {HTMLElement} root the view section
 * @param {object} hooks
 * @param {(host: HTMLElement, banner: object) => void} hooks.showBanner the section's inline banner
 * @param {(host: HTMLElement) => void} hooks.clearBanner
 * @param {(view: object) => void} [hooks.onChange] the connection changed on the server
 * @param {(mode: string) => void} [hooks.onMode] the area's mode changed
 * @returns {{load: (options?: object) => Promise<void>, mode: () => string, shown: () => boolean,
 *   setVisible: (on: boolean) => void, dirty: () => boolean, destroy: () => void}}
 */
export function bindConnectionArea(root, { showBanner, clearBanner, onChange = () => {}, onMode = () => {} } = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    area: $('se-ai-conn'),
    load: $('se-ai-conn-load'),
    locked: $('se-ai-conn-locked'),
    facts: $('se-ai-conn-facts'),
    form: $('se-ai-conn-form'),
    keyState: $('se-ai-conn-key-state'),
    keyAgain: $('se-ai-conn-key-again'),
    keyClear: $('se-ai-conn-key-clear'),
    modelsField: $('se-ai-conn-models-field'),
    models: $('se-ai-conn-models'),
    actions: $('se-ai-conn-actions'),
    check: $('se-ai-conn-check'),
    save: $('se-ai-conn-save'),
    revert: $('se-ai-conn-revert'),
    dirty: $('se-ai-conn-dirty'),
    status: $('se-ai-conn-status'),
    banner: $('se-ai-conn-banner'),
    result: $('se-ai-conn-result'),
    resultTitle: $('se-ai-conn-result-title'),
    resultMsg: $('se-ai-conn-result-msg'),
  };
  const offs = [];
  const st = {
    mode: 'loading',   // loading | editable | locked | failed | missing
    view: null,        // the connection as loaded (GET or the last answer)
    visible: false,
    busy: false,       // a save, revert, clear or reset is running
    checking: false,
    loadSeq: 0,
    checkSeq: 0,
    invalid: null,     // {input, slot} the field showing an error
  };
  if (!els.area || !els.form) {
    return { load: async () => {}, mode: () => 'missing', shown: () => false, setVisible() {}, dirty: () => false, destroy() {} };
  }

  const input = (field) => $(CONN_INPUTS[field]);
  const part = (field, suffix) => $(`${CONN_INPUTS[field]}-${suffix}`);
  const setText = (el, text) => { if (el && el.textContent !== text) el.textContent = text; };
  const setStatus = (text) => setText(els.status, text);
  const listen = (el, type, fn) => {
    if (!el) return;
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  /* ── what shows ── */

  function syncAreas() {
    const mode = st.mode;
    setHidden(els.area, !(st.visible && (mode === 'editable' || mode === 'locked' || mode === 'failed')));
    setHidden(els.form, mode !== 'editable');
    setHidden(els.locked, mode !== 'locked');
    setHidden(els.actions, mode !== 'editable' && mode !== 'locked');
    for (const el of [els.save, els.revert]) setHidden(el, mode !== 'editable');
    if (mode !== 'editable') setHidden(els.dirty, true);
  }

  function setMode(mode) {
    const changed = st.mode !== mode;
    st.mode = mode;
    syncAreas();
    if (changed) onMode(mode);
  }

  /* ── the form ── */

  function readValues() {
    return {
      url: input('url').value,
      model: input('model').value,
      verify_tls: !!input('verify_tls').checked,
      template_kwargs: !!input('template_kwargs').checked,
      max_tokens_field: input('max_tokens_field').value,
      extra_body: input('extra_body').value,
      stream: !!input('stream').checked,
    };
  }
  const typedKey = () => input('api_key').value;

  function fillInputs(values) {
    input('url').value = values.url;
    input('model').value = values.model;
    input('verify_tls').checked = values.verify_tls;
    input('template_kwargs').checked = values.template_kwargs;
    input('max_tokens_field').value = values.max_tokens_field;
    input('extra_body').value = values.extra_body;
    input('stream').checked = values.stream;
    syncModelPick();
  }

  /** The source badges, Use .env beside a saved field, and the key's state. */
  function syncSources() {
    const view = st.view;
    for (const field of C.FIELDS) {
      const source = C.sourceOf(view, field);
      setText(part(field, 'source'), C.sourceText(source));
      setHidden(part(field, 'reset'), source !== 'saved');
    }
    const set = !!(view && view.api_key_set);
    setText(els.keyState, set ? 'Set' : 'Not set');
    els.keyState.classList.toggle('badge--accent', set);
    input('api_key').setAttribute('placeholder', set ? 'Type to replace' : 'Not set');
    setHidden(els.keyClear, !set);
  }

  function isDirty() {
    return st.mode === 'editable' && !!st.view && C.isDirty(st.view, readValues(), typedKey());
  }

  function updateDirty() {
    const dirty = isDirty();
    setHidden(els.dirty, !dirty);
    els.save.setAttribute('aria-disabled', dirty ? 'false' : 'true');
    const saved = !!st.view && C.FIELDS.some((field) => C.sourceOf(st.view, field) === 'saved');
    els.revert.setAttribute('aria-disabled', saved ? 'false' : 'true');
    // Another server, or the certificate check turned off for an https one: each in its own words.
    const reason = st.mode === 'editable' ? C.keyAgainReason(st.view, readValues(), typedKey()) : '';
    if (reason) setText(els.keyAgain, C.KEY_AGAIN[reason]);
    setHidden(els.keyAgain, !reason);
  }

  function renderLocked() {
    render(els.facts, html`${C.lockedFacts(st.view).map((f) => html`<dt>${f.term}</dt><dd><span class="${f.mono ? 'mono' : ''}">${f.value}</span> <span class="badge se-ai__source">${C.sourceText(f.source)}</span></dd>`)}`);
  }

  /** Show a connection view: refill the form, or (keepEdits) keep what was edited on screen over it. */
  function apply(view, { keepEdits = false, force = '' } = {}) {
    const before = st.view;
    const values = keepEdits && before ? C.rebase(before, view, readValues(), { force }) : C.viewValues(view);
    st.view = view;
    clearBanner(els.load);
    clearFieldError();
    if (!keepEdits) input('api_key').value = '';
    if (view && view.locked) {
      renderLocked();
      setMode('locked');
    } else {
      fillInputs(values);
      setMode('editable');
    }
    syncSources();
    updateDirty();
  }

  /** Load the connection. Unsaved edits are kept (nothing loads over them) unless `force`. */
  async function load({ view = null, force = false } = {}) {
    if (view) {
      apply(view);
      return;
    }
    if (!force && isDirty()) return;
    const seq = (st.loadSeq += 1);
    try {
      const got = await api.get('/api/assistant/connection', { accept: 'json' });
      if (seq !== st.loadSeq) return;
      apply(got || {});
    } catch (error) {
      if (seq !== st.loadSeq || (error && error.kind === 'aborted')) return;
      st.view = null;
      if (error && error.status === 404) {
        clearBanner(els.load);
        setMode('missing');
        return;
      }
      const e = api.explain(error);
      showBanner(els.load, { title: "Couldn't load the connection", message: e.message || 'Something went wrong.', detail: e.detailText, actions: [{ label: 'Retry', action: 'ai-conn-retry' }] });
      setMode('failed');
    }
  }

  /* ── the models from a Check ── */

  function syncModelPick() {
    if (els.modelsField.hidden) return;
    const typed = input('model').value.trim();
    const listed = Array.from(els.models.querySelectorAll('option')).some((o) => o.value && o.value === typed);
    els.models.value = listed ? typed : '';
  }

  function setModels(models) {
    if (!models.length) {
      render(els.models, html``);
      setHidden(els.modelsField, true);
      return;
    }
    render(els.models, html`<option value="">Choose a model…</option>${models.map((id) => html`<option value="${id}">${id}</option>`)}`);
    setHidden(els.modelsField, false);
    syncModelPick();
  }

  function hideCheck() {
    st.checkSeq += 1;
    setHidden(els.result, true);
    setModels([]);
    clearBanner(els.banner);
    setStatus('');
  }

  /* ── field errors ── */

  function clearFieldError(el = null) {
    const entry = st.invalid;
    if (!entry || (el && entry.input !== el)) return;
    st.invalid = null;
    entry.input.removeAttribute('aria-invalid');
    const span = entry.slot.querySelector('span');
    setText(span || entry.slot, '');
    setHidden(entry.slot, true);
  }

  /** A 422: on its field when the form has one for it (focused), and in the banner with `title`. */
  function showValidation(error, title) {
    if (!error || error.code !== 'validation_error') return false;
    const first = (error.fields || [])[0] || {};
    const problem = C.fieldProblem(first.field, first.message || error.detailText || '');
    const el = problem && st.mode === 'editable' ? input(problem.field) : null;
    const slot = problem ? part(problem.field, 'error') : null;
    if (el && slot) {
      clearFieldError();
      const span = slot.querySelector('span');
      setText(span || slot, problem.text);
      setHidden(slot, false);
      el.setAttribute('aria-invalid', 'true');
      st.invalid = { input: el, slot };
      el.focus();
      showBanner(els.banner, { title, message: `Check the highlighted field: ${problem.text}` });
    } else {
      showBanner(els.banner, { title, message: (problem && problem.text) || first.message || error.detailText || 'Some values were not accepted.' });
    }
    return true;
  }

  function failed(title, error) {
    if (!error || error.kind === 'aborted') return;
    if (showValidation(error, title)) return;
    if (error.code === 'connection_locked') {
      showBanner(els.banner, { title: 'The connection is locked', message: 'The server now sets it only in its .env. The connection was loaded again.' });
      load({ force: true }).then(() => onChange(st.view)).catch(() => {});
      return;
    }
    const e = api.explain(error);
    showBanner(els.banner, { title, message: e.message || 'Something went wrong.', detail: e.detailText });
  }

  /* ── Check ── */

  async function check() {
    if (st.checking || !st.view || (st.mode !== 'editable' && st.mode !== 'locked')) return;
    clearBanner(els.banner);
    clearFieldError();
    setHidden(els.result, true);
    const body = st.mode === 'locked' ? {} : C.changeBody(st.view, readValues(), typedKey());
    const seq = (st.checkSeq += 1);
    st.checking = true;
    setBusy(els.check, true, { label: 'Checking…' });
    setStatus('Checking the AI server…');
    try {
      const answer = await api.post('/api/assistant/connection/check', body, { accept: 'json' });
      if (seq !== st.checkSeq) return;
      const result = C.checkResult(answer || {});
      setText(els.resultTitle, result.title);
      setText(els.resultMsg, result.message);
      els.result.setAttribute('data-tone', result.tone);
      setHidden(els.result, false);
      setModels(st.mode === 'editable' ? result.models : []);
      setStatus(result.status);
    } catch (error) {
      if (seq !== st.checkSeq) return;
      checkFailed(error);
    } finally {
      st.checking = false;
      setBusy(els.check, false);
    }
  }

  function checkFailed(error) {
    if (!error || error.kind === 'aborted') {
      setStatus('');
      return;
    }
    const copy = C.checkFailure(error.code);
    if (copy) {
      const elapsed = elapsedFrom(error.detail);
      setStatus(elapsed === null ? 'Failed' : `Failed after ${elapsedText(elapsed)}`);
      showBanner(els.banner, { title: copy.title, message: copy.message, detail: error.detailText || error.message });
      return;
    }
    setStatus('');
    failed("Couldn't check the connection", error);
  }

  /* ── Save, Revert, Clear key, Use .env ── */

  async function save() {
    if (st.busy || st.mode !== 'editable' || !isDirty()) return;
    clearBanner(els.banner);
    clearFieldError();
    const before = st.view;
    const values = readValues();
    const key = typedKey();
    const lostKey = C.keyAgain(before, values, key);
    st.busy = true;
    setBusy(els.save, true, { label: 'Saving…' });
    setStatus('Saving the connection…');
    try {
      const view = await api.put('/api/assistant/connection', C.changeBody(before, values, key), { accept: 'json' });
      setStatus('');
      apply(view || {});
      if (lostKey && !(view && view.api_key_set)) toast('Saved the AI connection. Enter the API key again for this server.', { kind: 'warning' });
      else toast('Saved the AI connection', { kind: 'success' });
      onChange(st.view);
    } catch (error) {
      setStatus('');
      failed("Couldn't save the connection", error);
    } finally {
      st.busy = false;
      setBusy(els.save, false);
      updateDirty();
    }
  }

  async function revert() {
    if (st.busy || st.mode !== 'editable' || els.revert.getAttribute('aria-disabled') === 'true') return;
    clearBanner(els.banner);
    const keySaved = C.sourceOf(st.view, 'api_key') === 'saved';
    const ok = await confirm({
      title: 'Revert the AI connection to .env?',
      message: `Everything saved here is forgotten${keySaved ? ', the API key too' : ''}, and the server uses its .env again.${isDirty() ? ' Your unsaved changes are dropped too.' : ''}`,
      confirmLabel: 'Revert to .env',
    });
    if (!ok || st.busy) return;
    st.busy = true;
    setBusy(els.revert, true);
    try {
      const view = await api.del('/api/assistant/connection', { accept: 'json' });
      hideCheck();
      apply(view || {});
      toast('The AI connection follows .env again', { kind: 'success' });
      onChange(st.view);
    } catch (error) {
      failed("Couldn't revert the connection", error);
    } finally {
      st.busy = false;
      setBusy(els.revert, false);
      updateDirty();
    }
  }

  /** Change one field on the server (Clear key, Use .env), keeping the other edits on screen. */
  async function changeOne(field, value, { button, done, title }) {
    st.busy = true;
    setBusy(button, true);
    clearBanner(els.banner);
    try {
      const view = await api.put('/api/assistant/connection', { [field]: value }, { accept: 'json' });
      if (field === 'api_key') input('api_key').value = '';
      apply(view || {}, { keepEdits: true, force: field });
      if (st.mode === 'editable') input(field).focus();
      toast(done, { kind: 'success' });
      onChange(st.view);
    } catch (error) {
      failed(title, error);
    } finally {
      st.busy = false;
      setBusy(button, false);
      updateDirty();
    }
  }

  async function clearKey() {
    if (st.busy || st.mode !== 'editable' || !st.view || !st.view.api_key_set) return;
    const fromEnv = C.sourceOf(st.view, 'api_key') === 'env';
    const ok = await confirm({
      title: "Clear the AI server's API key?",
      message: fromEnv
        ? 'This saves an empty key over the one in .env, so no key is sent. Use .env brings it back.'
        : 'No key is sent to the AI server until you enter one.',
      confirmLabel: 'Clear key', danger: true,
    });
    if (!ok || st.busy) return;
    await changeOne('api_key', '', { button: els.keyClear, done: 'Cleared the API key', title: "Couldn't clear the API key" });
  }

  async function reset(field, button) {
    if (st.busy || st.mode !== 'editable' || !C.FIELDS.includes(field) || C.sourceOf(st.view, field) !== 'saved') return;
    const label = C.LABELS[field];
    if (field === 'api_key') {
      const ok = await confirm({ title: 'Use the API key from .env?', message: 'The key saved here is forgotten.', confirmLabel: 'Use .env' });
      if (!ok || st.busy) return;
    }
    await changeOne(field, null, { button, done: `${label} follows .env again`, title: `Couldn't use .env for ${label}` });
  }

  /* ── events ── */

  const onEdit = (event) => {
    const target = event.target;
    if (!target || !target.id) return;
    if (target === els.models) {
      if (event.type === 'change' && els.models.value) input('model').value = els.models.value;
      updateDirty();
      return;
    }
    const field = Object.keys(CONN_INPUTS).find((name) => CONN_INPUTS[name] === target.id);
    if (!field) return;
    clearFieldError(target);
    if (CHECK_FIELDS.includes(field)) hideCheck();
    if (field === 'model') syncModelPick();
    updateDirty();
  };
  listen(els.form, 'input', onEdit);
  listen(els.form, 'change', onEdit);
  listen(els.form, 'keydown', (event) => {
    if (event.key !== 'Enter' || event.isComposing) return;
    const field = ENTER_SAVES.find((name) => event.target && event.target.id === CONN_INPUTS[name]);
    if (!field) return;
    event.preventDefault();
    save().catch(() => {});
  });
  listen(els.check, 'click', () => { check().catch(() => {}); });
  listen(els.save, 'click', () => { save().catch(() => {}); });
  listen(els.revert, 'click', () => { revert().catch(() => {}); });
  listen(els.keyClear, 'click', () => { clearKey().catch(() => {}); });
  offs.push(actions(els.area, {
    'ai-conn-reset': (el) => reset(el.getAttribute('data-field') || '', el),
    'ai-conn-retry': () => load({ force: true }),
  }, { onError: (error) => { if (globalThis.console) globalThis.console.error(error); } }));

  return {
    load,
    mode: () => st.mode,
    /** Whether the area shows the connection's settings (the form or the locked list). */
    shown: () => st.mode === 'editable' || st.mode === 'locked',
    setVisible(on) {
      st.visible = !!on;
      syncAreas();
    },
    dirty: isDirty,
    destroy() {
      st.loadSeq += 1;
      st.checkSeq += 1;
      for (const off of offs.splice(0)) off();
    },
  };
}
