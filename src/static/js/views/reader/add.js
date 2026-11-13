/**
 * The Add sheet: File | Paste, the only two sources (RL-11).
 *
 * - File: the dropzone label stands in for the visually hidden file input; drag and drop works too. Only
 *   EPUB, TXT, Markdown and HTML are accepted; anything else shows a field error.
 * - Paste: an optional title and the text (required).
 * - Submit sends multipart `file` or JSON `{type: 'text', text, title?}` (RL-12). While it runs the button
 *   reads "Uploading…" or "Adding…" and the form is aria-busy. A server error shows as an inline banner in the
 *   sheet. Success closes the sheet, resets the form and toasts "Added “T”" (201) or
 *   "Already in your library: “T”" (200).
 * @module views/reader/add
 */
import { html, render, setHidden, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import { toast } from '../../core/toast.js';
import { emit } from '../../core/store.js';
import { openSheet, closeSheet } from '../../core/dialogs.js';
import { addRequest, acceptedFile, fileHint } from './text-logic.js';

export const DOCUMENTS_PATH = '/v1/reader/documents';

/**
 * Bind the Add sheet inside the Reader's root.
 * @param {HTMLElement} root
 * @param {{onAdded?: (doc: object, {created: boolean}) => void}} [options]
 */
export function createAdd(root, { onAdded = () => {} } = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    sheet: $('rd-add-dialog'),
    form: $('rd-add-form'),
    filePanel: $('rd-add-file-panel'),
    pastePanel: $('rd-add-paste-panel'),
    file: $('rd-file'),
    dropzone: $('rd-dropzone'),
    fileTitle: $('rd-file-title'),
    fileHint: $('rd-file-hint'),
    fileError: $('rd-file-error'),
    title: $('rd-paste-title'),
    text: $('rd-paste-text'),
    pasteError: $('rd-paste-error'),
    error: $('rd-add-error'),
    submit: $('rd-add-submit'),
  };
  const st = { mode: 'file', file: null, busy: false };
  const offs = [];
  const listen = (el, type, fn) => {
    if (!el) return;
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  function fieldError(field, message) {
    const isFile = field === 'file';
    const box = isFile ? els.fileError : els.pasteError;
    const control = isFile ? els.file : els.text;
    if (message) {
      render(box, html`${icon('alert')}<span>${message}</span>`);
      setHidden(box, false);
      control.setAttribute('aria-invalid', 'true');
    } else {
      setHidden(box, true);
      render(box, html``);
      control.removeAttribute('aria-invalid');
    }
    if (isFile) els.dropzone.classList.toggle('is-invalid', !!message);
  }

  function clearErrors() {
    fieldError('file', '');
    fieldError('paste', '');
    setHidden(els.error, true);
    render(els.error, html``);
  }

  function setMode(mode) {
    st.mode = mode === 'paste' ? 'paste' : 'file';
    for (const input of els.form.querySelectorAll('input[name="rd-source"]')) input.checked = input.value === st.mode;
    setHidden(els.filePanel, st.mode !== 'file');
    setHidden(els.pastePanel, st.mode !== 'paste');
    setHidden(els.error, true);
  }

  function setFile(file) {
    fieldError('file', '');
    if (file && !acceptedFile(file.name)) {
      st.file = null;
      els.dropzone.classList.remove('has-file');
      els.fileTitle.textContent = 'Drop a book or text file';
      els.fileHint.textContent = fileHint(null);
      fieldError('file', 'Choose an EPUB, TXT, Markdown or HTML file');
      return;
    }
    st.file = file || null;
    els.dropzone.classList.toggle('has-file', !!file);
    els.fileTitle.textContent = file ? file.name : 'Drop a book or text file';
    els.fileHint.textContent = fileHint(file);
  }

  function reset() {
    st.file = null;
    try { els.file.value = ''; } catch { /* some browsers refuse to clear */ }
    els.title.value = '';
    els.text.value = '';
    setFile(null);
    clearErrors();
  }

  function setBusyState(busy) {
    st.busy = busy;
    if (busy) setBusy(els.submit, true, { label: st.mode === 'file' ? 'Uploading…' : 'Adding…' });
    else setBusy(els.submit, false);
    if (busy) els.form.setAttribute('aria-busy', 'true');
    else els.form.removeAttribute('aria-busy');
  }

  function showServerError(error) {
    const copy = api.explain(error, 'reader-add');
    if (!copy.title) return;
    render(els.error, html`${icon('alert', { className: 'ico banner__icon' })}
      <div class="banner__body">
        <p class="banner__title">${copy.title}</p>
        <p class="banner__msg">${copy.message}</p>
        ${copy.detailText ? html`<details class="banner__details"><summary>Details</summary><pre></pre></details>` : ''}
      </div>`);
    // Details read as the server's message and code, never its JSON envelope; there are none when
    // they would only repeat the message. Server text goes in only as text.
    const pre = els.error.querySelector('pre');
    if (pre) pre.textContent = copy.detailText;
    setHidden(els.error, false);
  }

  async function submit(event) {
    if (event) event.preventDefault();
    if (st.busy) return;
    clearErrors();
    const req = addRequest({ mode: st.mode, file: st.file, title: els.title.value, text: els.text.value });
    if (req.error) {
      fieldError(req.error.field, req.error.message);
      (req.error.field === 'file' ? els.file : els.text).focus();
      return;
    }
    let options;
    if (req.kind === 'file') {
      const form = new FormData();
      form.append('file', req.file, req.file.name);
      options = { method: 'POST', body: form };
    } else {
      options = { method: 'POST', json: req.json };
    }
    setBusyState(true);
    let doc = null;
    let created = true;
    try {
      const res = await api.request(DOCUMENTS_PATH, { ...options, accept: 'response' });
      created = res.status !== 200;
      try {
        doc = await res.json();
      } catch {
        doc = null;
      }
    } catch (error) {
      setBusyState(false);
      if (!error || error.kind === 'aborted') return;
      // The sheet may have been closed while it ran: then there is no control nearby, so a toast.
      if (els.sheet.open) showServerError(error);
      else {
        const copy = api.explain(error, 'reader-add');
        toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
      }
      return;
    }
    setBusyState(false);
    const title = (doc && doc.title) || 'document';
    reset();
    if (els.sheet.open) closeSheet(els.sheet, 'added');
    toast(created ? `Added “${title}”` : `Already in your library: “${title}”`, { kind: 'success' });
    try { emit('library:changed', { docId: doc && doc.id }); } catch { /* bus not ready */ }
    onAdded(doc, { created });
  }

  // File input and the dropzone (a label for the visually hidden input; drag and drop too).
  listen(els.file, 'change', () => setFile(els.file.files && els.file.files[0] ? els.file.files[0] : null));
  listen(els.dropzone, 'dragover', (event) => {
    event.preventDefault();
    els.dropzone.classList.add('is-drag');
  });
  listen(els.dropzone, 'dragleave', () => els.dropzone.classList.remove('is-drag'));
  listen(els.dropzone, 'drop', (event) => {
    event.preventDefault();
    els.dropzone.classList.remove('is-drag');
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setFile(file);
  });
  listen(els.form, 'change', (event) => {
    if (event.target && event.target.name === 'rd-source') setMode(event.target.value);
  });
  listen(els.text, 'input', () => { if (els.text.getAttribute('aria-invalid')) fieldError('paste', ''); });
  listen(els.form, 'submit', submit);

  setMode('file');

  return {
    /** Open the sheet; `mode` picks File or Paste. */
    open(trigger, { mode } = {}) {
      if (els.sheet.open) return;
      if (mode) setMode(mode);
      const first = els.form.querySelector('input[name="rd-source"]:checked');
      openSheet(els.sheet, {
        trigger: trigger || document.activeElement,
        initialFocus: first || undefined,
        // A click on the backdrop doesn't throw away pasted text or a chosen file.
        backdropClose: () => !st.busy && !st.file && !els.text.value.trim(),
      });
    },
    close() {
      if (els.sheet.open) closeSheet(els.sheet);
    },
    setMode,
    setFile,
    submit,
    reset,
    get mode() { return st.mode; },
    get busy() { return st.busy; },
    destroy() {
      for (const off of offs.splice(0)) off();
    },
  };
}
