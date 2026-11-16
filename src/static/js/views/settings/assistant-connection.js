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
 * - Saved assistants (core/assistants.js; a server without them, 404, works as above): the form edits one of
 *   them, the one in use first, with its Name, through PUT …/assistants/{id} (Check: …/{id}/check; Revert to
 *   .env: every saved field null). Add assistant (add()) empties the form to the .env values for a new one (the
 *   list's `env`): Save POSTs it with the name typed or the one offered (C.suggestName), and Check sends every
 *   field, the untouched ones null, over the connection in use (C.candidateBody).
 *   With none saved the form is that new one, the first.
 *   edit(id) and add() ask before dropping unsaved changes. A load over unsaved edits only takes the new list.
 *   `onList()` tells the section when the list, what the form edits, or what shows changed.
 * @module views/settings/assistant-connection
 */
import { html, render, setHidden, setBusy, actions } from '../../core/dom.js';
import * as api from '../../core/api.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { emit } from '../../core/store.js';
import { listAssistants, findAssistant, createAssistant, updateAssistant, checkAssistant } from '../../core/assistants.js';
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

/** A saved assistant's name (not a connection field). */
export const NAME_INPUT = 'se-ai-conn-name';

/** The one-line fields where Enter saves. */
const ENTER_SAVES = ['url', 'model', 'api_key', 'name'];
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
 * @param {() => void} [hooks.onList] the saved assistants, what the form edits, or what shows changed
 * @returns {{load: (options?: object) => Promise<void>, mode: () => string, shown: () => boolean,
 *   setVisible: (on: boolean) => void, dirty: () => boolean, state: () => object, edit: (id: string) => Promise<void>,
 *   add: () => Promise<void>, destroy: () => void}}
 */
export function bindConnectionArea(root, { showBanner, clearBanner, onChange = () => {}, onMode = () => {}, onList = () => {} } = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    area: $('se-ai-conn'),
    target: $('se-ai-conn-target'),
    nameField: $('se-ai-conn-name-field'),
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
    cancel: $('se-ai-conn-cancel'),
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
    saved: null,       // the saved assistants (listAssistants()), null on a server without them
    target: { kind: 'env' },   // what the form edits: env (the connection) | saved {id} | new {first}
  };
  if (!els.area || !els.form) {
    return {
      load: async () => {}, mode: () => 'missing', shown: () => false, setVisible() {}, dirty: () => false,
      state: () => ({ mode: 'missing', visible: false, saved: null, target: { kind: 'env' } }), edit: async () => {}, add: async () => {}, destroy() {},
    };
  }

  const idOf = (field) => (field === 'name' ? NAME_INPUT : CONN_INPUTS[field]);
  const input = (field) => $(idOf(field));
  const part = (field, suffix) => $(`${idOf(field)}-${suffix}`);
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
    const kind = st.target.kind;
    setHidden(els.area, !(st.visible && (mode === 'editable' || mode === 'locked' || mode === 'failed')));
    setHidden(els.form, mode !== 'editable');
    setHidden(els.locked, mode !== 'locked');
    setHidden(els.actions, mode !== 'editable' && mode !== 'locked');
    setHidden(els.save, mode !== 'editable');
    setHidden(els.revert, mode !== 'editable' || kind === 'new');
    setHidden(els.cancel, !(mode === 'editable' && kind === 'new' && !st.target.first));
    setHidden(els.nameField, kind === 'env');
    if (mode !== 'editable') setHidden(els.dirty, true);
    syncTarget();
    onList();
  }

  /** The saved assistant the form edits, as last listed, or null. */
  const editing = () => (st.target.kind === 'saved' ? findAssistant(st.saved, st.target.id) : null);
  const takenNames = () => (st.saved ? st.saved.assistants.filter((a) => a.id !== st.target.id).map((a) => a.name) : []);
  const typedName = () => (input('name') ? input('name').value.trim() : '');
  /** The name a new assistant is saved under: the one typed, else the one offered. */
  const newName = () => typedName() || C.suggestName(readValues(), takenNames());

  /** The line under the heading that says what the form edits, and the name offered for a new one. */
  function syncTarget() {
    const kind = st.target.kind;
    let line = '';
    if (kind === 'saved' && st.view) line = st.view.active ? `Editing “${st.view.name}”, the assistant in use.` : `Editing “${st.view.name}”.`;
    if (kind === 'new' && !st.target.first) line = 'A new assistant. What you leave as it is follows .env.';
    setText(els.target, line);
    setHidden(els.target, !line || st.mode !== 'editable');
    const name = input('name');
    if (name) name.setAttribute('placeholder', kind === 'new' ? C.suggestName(readValues(), takenNames()) : '');
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
    setHidden(els.keyClear, !set || st.target.kind === 'new');
  }

  function nameDirty() {
    if (st.target.kind === 'new') return typedName() !== '';
    if (st.target.kind === 'saved') return typedName() !== String((st.view && st.view.name) || '');
    return false;
  }

  function isDirty() {
    return st.mode === 'editable' && !!st.view && (C.isDirty(st.view, readValues(), typedKey()) || nameDirty());
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
    syncTarget();
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
    if (!keepEdits) {
      input('api_key').value = '';
      if (input('name')) input('name').value = st.target.kind === 'saved' && view ? String(view.name || '') : '';
    }
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

  /**
   * Load the saved assistants and the one the form edits (`target`: that id; else the one it edits now, else the
   * one in use), or, with none, the connection. Unsaved edits are kept (only the list loads) unless `force`.
   */
  async function load({ view = null, force = false, target = '' } = {}) {
    if (view) {
      apply(view);
      return;
    }
    const keep = !force && isDirty();
    if (keep && !st.saved) return;
    const seq = (st.loadSeq += 1);
    try {
      const saved = await listAssistants();
      if (seq !== st.loadSeq) return;
      st.saved = saved;
      const all = saved && !saved.locked ? saved.assistants : [];
      if (keep) {
        const now = editing();
        if (st.target.kind !== 'saved' || now) {
          if (now) st.view = { ...st.view, active: now.active };
          syncAreas();
          return;
        }
      }
      if (all.length) {
        const pick = findAssistant(saved, target) || editing() || findAssistant(saved, saved.active) || all[0];
        st.target = { kind: 'saved', id: pick.id };
        apply(pick);
        return;
      }
      const got = await api.get('/api/assistant/connection', { accept: 'json' });
      if (seq !== st.loadSeq) return;
      st.target = saved && !saved.locked && !(got && got.locked) ? { kind: 'new', first: true } : { kind: 'env' };
      apply(got || {});
    } catch (error) {
      if (seq !== st.loadSeq || (error && error.kind === 'aborted')) return;
      st.view = null;
      st.saved = null;
      st.target = { kind: 'env' };
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
    if (error.code === 'assistant_exists' && st.target.kind !== 'env') {
      showValidation({ code: 'validation_error', fields: [{ field: 'name', message: error.message }] }, title);
      return;
    }
    if (error.status === 404 && st.target.kind === 'saved') {
      showBanner(els.banner, { title, message: 'This assistant no longer exists on the server. The list was loaded again.' });
      load({ force: true }).then(() => onChange(st.view)).catch(() => {});
      return;
    }
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
    const kind = st.mode === 'locked' ? 'env' : st.target.kind;
    let body = {};
    if (st.mode !== 'locked') body = kind === 'new' ? newBody({ check: true }) : C.changeBody(st.view, readValues(), typedKey());
    const seq = (st.checkSeq += 1);
    st.checking = true;
    setBusy(els.check, true, { label: 'Checking…' });
    setStatus('Checking the AI server…');
    try {
      const answer = kind === 'saved'
        ? await checkAssistant(st.target.id, body)
        : await api.post('/api/assistant/connection/check', body, { accept: 'json' });
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

  /** A new assistant's fields: what changed over .env. */
  function newBody({ check = false } = {}) {
    return check ? C.candidateBody(st.view, readValues(), typedKey()) : C.changeBody(st.view, readValues(), typedKey());
  }

  /** A saved assistant's answer, with the id, name and in-use flag it had when the answer leaves them out. */
  const withIdentity = (view, before) => ({ id: before.id, name: before.name, active: before.active, ...(view || {}) });

  /** Put a saved assistant's new view in the list. */
  function replaceSaved(view) {
    if (!st.saved) return;
    st.saved = { ...st.saved, assistants: st.saved.assistants.map((a) => (a.id === view.id ? { ...view, active: a.active } : a)) };
  }

  async function save() {
    if (st.busy || st.mode !== 'editable' || !isDirty()) return;
    clearBanner(els.banner);
    clearFieldError();
    const kind = st.target.kind;
    if (kind === 'saved' && !typedName()) {
      showValidation({ code: 'validation_error', fields: [{ field: 'name', message: 'name is needed' }] }, "Couldn't save the assistant");
      return;
    }
    const before = st.view;
    const values = readValues();
    const key = typedKey();
    const lostKey = C.keyAgain(before, values, key);
    const body = C.changeBody(before, values, key);
    st.busy = true;
    setBusy(els.save, true, { label: 'Saving…' });
    setStatus('Saving the connection…');
    try {
      if (kind === 'new') {
        const name = newName();
        const made = await createAssistant({ name, ...newBody() });
        setStatus('');
        await load({ force: true, target: made && made.id ? made.id : '' });
        toast(`Saved “${(made && made.name) || name}”`, { kind: 'success' });
        emit('assistants:changed', C.FROM_SETTINGS);
        onChange(st.view);
        return;
      }
      if (kind === 'saved' && nameDirty()) body.name = typedName();
      const view = kind === 'saved'
        ? withIdentity(await updateAssistant(st.target.id, body), before)
        : await api.put('/api/assistant/connection', body, { accept: 'json' });
      setStatus('');
      if (kind === 'saved') replaceSaved(view);
      apply(view || {});
      const what = kind === 'saved' ? `“${view.name}”` : 'the AI connection';
      if (lostKey && !(view && view.api_key_set)) toast(`Saved ${what}. Enter the API key again for this server.`, { kind: 'warning' });
      else toast(`Saved ${what}`, { kind: 'success' });
      if (kind === 'saved') emit('assistants:changed', C.FROM_SETTINGS);
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
    const dropped = isDirty() ? ' Your unsaved changes are dropped too.' : '';
    const before = st.view;
    const saved = st.target.kind === 'saved';
    const ok = await confirm(saved
      ? {
        title: `Revert “${before.name}” to .env?`,
        message: `Everything saved for it is forgotten${keySaved ? ', the API key too' : ''}, and it follows the server's .env. Its name stays.${dropped}`,
        confirmLabel: 'Revert to .env',
      }
      : {
        title: 'Revert the AI connection to .env?',
        message: `Everything saved here is forgotten${keySaved ? ', the API key too' : ''}, and the server uses its .env again.${dropped}`,
        confirmLabel: 'Revert to .env',
      });
    if (!ok || st.busy) return;
    st.busy = true;
    setBusy(els.revert, true);
    try {
      const view = saved
        ? withIdentity(await updateAssistant(st.target.id, C.revertBody(before)), before)
        : await api.del('/api/assistant/connection', { accept: 'json' });
      hideCheck();
      if (saved) replaceSaved(view);
      apply(view || {});
      toast(saved ? `“${view.name}” follows .env again` : 'The AI connection follows .env again', { kind: 'success' });
      if (saved) emit('assistants:changed', C.FROM_SETTINGS);
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
      const saved = st.target.kind === 'saved';
      const view = saved
        ? withIdentity(await updateAssistant(st.target.id, { [field]: value }), st.view)
        : await api.put('/api/assistant/connection', { [field]: value }, { accept: 'json' });
      if (saved) {
        replaceSaved(view);
        emit('assistants:changed', C.FROM_SETTINGS);
      }
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

  /* ── Edit, Add assistant, Cancel ── */

  async function discardOk() {
    if (!isDirty()) return true;
    return confirm({ title: 'Discard your changes to the connection?', message: "They haven't been saved.", confirmLabel: 'Discard', cancelLabel: 'Keep editing' });
  }

  /** Show `view` in the form as what it now edits: no Check result, banner or key from the one before. */
  function switchTo(target, view) {
    st.target = target;
    hideCheck();
    clearBanner(els.banner);
    apply(view);
  }

  async function edit(id) {
    if (st.busy || !st.saved || st.mode !== 'editable') return;
    if (!(st.target.kind === 'saved' && st.target.id === id)) {
      if (!findAssistant(st.saved, id) || !(await discardOk())) return;
      const view = findAssistant(st.saved, id);
      if (!view || st.busy) return;
      switchTo({ kind: 'saved', id }, view);
    }
    input('name').focus();
  }

  async function add() {
    if (st.busy || !st.saved || st.saved.locked || st.mode !== 'editable') return;
    if (st.target.kind !== 'new') {
      if (!(await discardOk()) || st.busy) return;
      switchTo({ kind: 'new', first: !st.saved.assistants.length }, st.saved.env || {});
    }
    input('name').focus();
  }

  async function cancelNew() {
    if (st.busy || st.target.kind !== 'new' || st.target.first || !st.saved) return;
    if (!(await discardOk())) return;
    const back = findAssistant(st.saved, st.saved.active) || st.saved.assistants[0];
    if (!back || st.busy) return;
    switchTo({ kind: 'saved', id: back.id }, back);
    input('name').focus();
  }

  /* ── events ── */

  const onEdit = (event) => {
    const target = event.target;
    if (!target || !target.id) return;
    if (target.id === NAME_INPUT) {
      clearFieldError(target);
      updateDirty();
      return;
    }
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
    const field = ENTER_SAVES.find((name) => event.target && event.target.id === idOf(name));
    if (!field) return;
    event.preventDefault();
    save().catch(() => {});
  });
  listen(els.check, 'click', () => { check().catch(() => {}); });
  listen(els.save, 'click', () => { save().catch(() => {}); });
  listen(els.revert, 'click', () => { revert().catch(() => {}); });
  listen(els.keyClear, 'click', () => { clearKey().catch(() => {}); });
  listen(els.cancel, 'click', () => { cancelNew().catch(() => {}); });
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
    /** What the section needs to show the saved assistants: the mode, whether the area shows, the list, the target. */
    state: () => ({ mode: st.mode, visible: st.visible, saved: st.saved, target: st.target }),
    edit,
    add,
    destroy() {
      st.loadSeq += 1;
      st.checkSeq += 1;
      for (const off of offs.splice(0)) off();
    },
  };
}
