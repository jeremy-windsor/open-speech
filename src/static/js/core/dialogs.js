/**
 * Modal sheets, confirm and prompt. The only module that calls showModal().
 *
 * confirm() and prompt() reuse the shell's #app-confirm and #app-prompt <dialog>s. They expect these ids
 * inside (the shell provides them; when a dialog is missing it is built here with the same structure):
 *   #app-confirm: h2#app-confirm-title, p#app-confirm-msg, button#app-confirm-cancel, button#app-confirm-ok
 *   #app-prompt:  form#app-prompt-form > h2#app-prompt-title, label#app-prompt-label[for=app-prompt-input],
 *                 input#app-prompt-input, p#app-prompt-hint, p#app-prompt-error[hidden],
 *                 button#app-prompt-cancel (type=button), button#app-prompt-ok (type=submit)
 *
 * While a modal sheet is open, everything outside it is inert: screen readers ignore live regions there
 * and nothing there can be clicked. So the shell's announcement regions (#app-sr-status, #app-sr-alert),
 * the toast stack (#app-toasts) and the tooltip (#app-tooltip) move into the top open sheet, and back to
 * <body> when the last one closes. liveHost() is where code that creates such an element appends it.
 * @module core/dialogs
 */
import { focusFirst } from './dom.js';

let open = new Map();   // dialog -> {trigger, onClose, cleanup, backdropClose}
let stack = [];         // open dialogs, topmost last
let confirmQueue = Promise.resolve();
let promptQueue = Promise.resolve();

/** The shell elements that must stay reachable while a sheet is open (see the module comment). */
export const LIVE_HOSTED = Object.freeze(['app-sr-status', 'app-sr-alert', 'app-toasts', 'app-tooltip']);

/** @returns {HTMLElement|null} where live regions and toasts belong now: the top open sheet, else <body> */
export function liveHost() {
  if (typeof document === 'undefined') return null;
  return topSheet() || document.body || null;
}

function hostLiveRegions() {
  const host = liveHost();
  if (!host) return;
  for (const id of LIVE_HOSTED) {
    const el = document.getElementById(id);
    if (el && el.parentNode !== host) host.appendChild(el);
  }
}

function resolveEl(value, root) {
  if (!value) return null;
  if (typeof value === 'string') return root.querySelector(value);
  return value;
}

function focusable(el) {
  return !!el && el.isConnected !== false && typeof el.focus === 'function' && !el.disabled;
}

/** @returns {HTMLDialogElement|null} the topmost open sheet */
export function topSheet() {
  return stack.length ? stack[stack.length - 1] : null;
}

/** @returns {boolean} whether any modal sheet is open */
export function isModalOpen() {
  return stack.length > 0;
}

/**
 * Open a <dialog> as a modal sheet. Focus moves to `initialFocus`, else the first [data-autofocus], else the
 * heading. Esc, a backdrop click and any [data-close] button close it; focus then returns to `trigger`
 * (default: the element focused before). Adds .sheet--open for the motion classes.
 * @param {HTMLDialogElement} dialog
 * @param {{trigger?: Element, initialFocus?: Element|string, onClose?: (returnValue: string) => void,
 *   backdropClose?: boolean | (() => boolean)}} [options]
 */
export function openSheet(dialog, { trigger, initialFocus, onClose, backdropClose = true } = {}) {
  if (!dialog) throw new TypeError('openSheet() needs a dialog');
  if (open.has(dialog)) return dialog;
  const returnTo = trigger || document.activeElement;
  const onCancel = (event) => {
    event.preventDefault();
    closeSheet(dialog, 'cancel');
  };
  // A press that starts inside the sheet and ends on the backdrop (selecting text in a field, dragging a
  // slider) fires its click on the dialog: only a press that also started on the backdrop closes it.
  let press = null;
  const onPointerDown = (event) => {
    press = { onBackdrop: event.target === dialog && isBackdropClick(dialog, event) };
  };
  const onClick = (event) => {
    const started = press;
    press = null;
    const target = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const closer = target && target.closest ? target.closest('[data-close]') : null;
    if (closer && dialog.contains(closer)) {
      event.preventDefault();
      closeSheet(dialog, closer.getAttribute('data-close') || '');
      return;
    }
    if (target === dialog && isBackdropClick(dialog, event)) {
      if (started && !started.onBackdrop) return;
      const allowed = typeof backdropClose === 'function' ? backdropClose() : backdropClose;
      if (allowed) closeSheet(dialog, 'backdrop');
    }
  };
  const onNativeClose = () => finish(dialog, dialog.returnValue || '');
  if (typeof dialog.showModal === 'function') dialog.showModal();   // throws for a detached dialog: nothing is registered yet
  else dialog.setAttribute('open', '');
  dialog.addEventListener('cancel', onCancel);
  dialog.addEventListener('pointerdown', onPointerDown);
  dialog.addEventListener('click', onClick);
  dialog.addEventListener('close', onNativeClose);
  open.set(dialog, {
    trigger: returnTo,
    onClose,
    cleanup: () => {
      dialog.removeEventListener('cancel', onCancel);
      dialog.removeEventListener('pointerdown', onPointerDown);
      dialog.removeEventListener('click', onClick);
      dialog.removeEventListener('close', onNativeClose);
    },
  });
  stack.push(dialog);
  dialog.classList.add('sheet--open');
  hostLiveRegions();

  const first = resolveEl(initialFocus, dialog) || dialog.querySelector('[data-autofocus]');
  if (focusable(first)) first.focus();
  else focusFirst(dialog, 'h2, h1, h3, [role="heading"]') || focusFirst(dialog);
  return dialog;
}

function isBackdropClick(dialog, event) {
  const rect = dialog.getBoundingClientRect ? dialog.getBoundingClientRect() : null;
  if (!rect || (!rect.width && !rect.height)) return true;
  const x = event.clientX;
  const y = event.clientY;
  if (typeof x !== 'number' || typeof y !== 'number') return true;
  return x < rect.left || x > rect.right || y < rect.top || y > rect.bottom;
}

function finish(dialog, returnValue) {
  const entry = open.get(dialog);
  if (!entry) return;
  open.delete(dialog);
  stack = stack.filter((d) => d !== dialog);
  entry.cleanup();
  dialog.classList.remove('sheet--open');
  hostLiveRegions();
  if (entry.trigger && focusable(entry.trigger) && entry.trigger !== document.body) {
    entry.trigger.focus({ preventScroll: true });
  }
  if (typeof entry.onClose === 'function') {
    try {
      entry.onClose(returnValue);
    } catch (error) {
      if (globalThis.console) globalThis.console.error(error);
    }
  }
}

/**
 * Close a sheet opened with openSheet() and return focus to its trigger.
 * @param {HTMLDialogElement} dialog
 * @param {string} [returnValue]
 */
export function closeSheet(dialog, returnValue = '') {
  if (!dialog) return;
  if (dialog.open && typeof dialog.close === 'function') {
    // finish() runs now; the native 'close' event that follows finds nothing left to do.
    const entry = open.get(dialog);
    if (entry) entry.cleanup();
    dialog.close(returnValue);
  } else if (dialog.hasAttribute && dialog.hasAttribute('open')) {
    dialog.removeAttribute('open');
  }
  if (open.has(dialog)) {
    dialog.returnValue = returnValue;
    finish(dialog, returnValue);
  }
}

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === true) node.setAttribute(name, '');
    else if (value !== false && value != null) node.setAttribute(name, String(value));
  }
  if (text) node.textContent = text;
  return node;
}

function ensureConfirm() {
  let dialog = document.getElementById('app-confirm');
  if (dialog && dialog.querySelector('#app-confirm-ok')) return dialog;
  if (!dialog) {
    dialog = el('dialog', { class: 'sheet sheet--alert', id: 'app-confirm', 'aria-labelledby': 'app-confirm-title', 'aria-describedby': 'app-confirm-msg' });
    document.body.appendChild(dialog);
  }
  const head = el('div', { class: 'sheet__head' });
  head.append(el('h2', { id: 'app-confirm-title', tabindex: '-1' }));
  const body = el('div', { class: 'sheet__body' });
  body.append(el('p', { id: 'app-confirm-msg' }));
  const foot = el('div', { class: 'sheet__foot' });
  foot.append(
    el('button', { type: 'button', class: 'btn btn--secondary', id: 'app-confirm-cancel' }),
    el('button', { type: 'button', class: 'btn btn--primary', id: 'app-confirm-ok' }),
  );
  dialog.replaceChildren(head, body, foot);
  return dialog;
}

function ensurePrompt() {
  let dialog = document.getElementById('app-prompt');
  if (dialog && dialog.querySelector('#app-prompt-input')) return dialog;
  if (!dialog) {
    dialog = el('dialog', { class: 'sheet', id: 'app-prompt', 'aria-labelledby': 'app-prompt-title' });
    document.body.appendChild(dialog);
  }
  const form = el('form', { id: 'app-prompt-form', novalidate: true });
  const head = el('div', { class: 'sheet__head' });
  head.append(el('h2', { id: 'app-prompt-title', tabindex: '-1' }));
  const field = el('div', { class: 'field' });
  field.append(
    el('label', { class: 'field__label', id: 'app-prompt-label', for: 'app-prompt-input' }),
    el('input', { class: 'input', id: 'app-prompt-input', type: 'text', autocomplete: 'off' }),
    el('p', { class: 'field__hint', id: 'app-prompt-hint' }),
    el('p', { class: 'field__error', id: 'app-prompt-error', hidden: true }),
  );
  const body = el('div', { class: 'sheet__body' });
  body.append(field);
  const foot = el('div', { class: 'sheet__foot' });
  foot.append(
    el('button', { type: 'button', class: 'btn btn--secondary', id: 'app-prompt-cancel' }, 'Cancel'),
    el('button', { type: 'submit', class: 'btn btn--primary', id: 'app-prompt-ok' }),
  );
  form.append(head, body, foot);
  dialog.replaceChildren(form);
  return dialog;
}

/**
 * Ask a yes/no question in #app-confirm. Focus goes to Cancel for a danger confirm, else to the confirm
 * button. Esc, the backdrop and Cancel resolve false. Confirms are queued, one at a time.
 * @param {{title: string, message?: string, confirmLabel?: string, cancelLabel?: string, danger?: boolean}} options
 * @returns {Promise<boolean>}
 */
export function confirm({ title, message = '', confirmLabel = 'OK', cancelLabel = 'Cancel', danger = false } = {}) {
  const run = () => new Promise((resolve) => {
    const dialog = ensureConfirm();
    const ok = dialog.querySelector('#app-confirm-ok');
    const cancel = dialog.querySelector('#app-confirm-cancel');
    dialog.querySelector('#app-confirm-title').textContent = String(title || '');
    const msg = dialog.querySelector('#app-confirm-msg');
    msg.textContent = String(message || '');
    msg.hidden = !message;
    ok.textContent = confirmLabel;
    cancel.textContent = cancelLabel;
    ok.classList.toggle('btn--danger', !!danger);
    ok.classList.toggle('btn--primary', !danger);
    let answer = false;
    const onOk = () => { answer = true; closeSheet(dialog, 'ok'); };
    const onCancel = () => { answer = false; closeSheet(dialog, 'cancel'); };
    ok.addEventListener('click', onOk);
    cancel.addEventListener('click', onCancel);
    openSheet(dialog, {
      initialFocus: danger ? cancel : ok,
      onClose: () => {
        ok.removeEventListener('click', onOk);
        cancel.removeEventListener('click', onCancel);
        resolve(answer);
      },
    });
  });
  const result = confirmQueue.then(run, run);
  confirmQueue = result.then(() => undefined, () => undefined);
  return result;
}

/**
 * Ask for a short text in #app-prompt. Returns the trimmed value, or null when cancelled. A `validate`
 * error (or `required` with an empty value) shows in the field's error slot (aria-invalid,
 * aria-describedby) and keeps the dialog open. Pass `error` to open it with a message already shown,
 * for example a 409 conflict from the server on the previous attempt.
 * @param {{title: string, label: string, value?: string, hint?: string, placeholder?: string,
 *   maxLength?: number, validate?: ((value: string) => string|null)|null, confirmLabel?: string,
 *   required?: boolean, error?: string}} options
 * @returns {Promise<string|null>}
 */
export function prompt({
  title, label, value = '', hint = '', placeholder = '', maxLength = 200, validate = null,
  confirmLabel = 'Save', required = true, error = '',
} = {}) {
  const run = () => new Promise((resolve) => {
    const dialog = ensurePrompt();
    const form = dialog.querySelector('#app-prompt-form') || dialog.querySelector('form');
    const input = dialog.querySelector('#app-prompt-input');
    const hintEl = dialog.querySelector('#app-prompt-hint');
    const errorEl = dialog.querySelector('#app-prompt-error');
    const cancel = dialog.querySelector('#app-prompt-cancel');
    const ok = dialog.querySelector('#app-prompt-ok');
    dialog.querySelector('#app-prompt-title').textContent = String(title || '');
    dialog.querySelector('#app-prompt-label').textContent = String(label || '');
    hintEl.textContent = String(hint || '');
    hintEl.hidden = !hint;
    ok.textContent = confirmLabel;
    input.value = String(value ?? '');
    input.setAttribute('placeholder', placeholder || '');
    if (!placeholder) input.removeAttribute('placeholder');
    if (maxLength > 0) input.setAttribute('maxlength', String(maxLength));
    else input.removeAttribute('maxlength');
    const initial = input.value;

    const showError = (text) => {
      const describedBy = [hint ? 'app-prompt-hint' : '', text ? 'app-prompt-error' : ''].filter(Boolean).join(' ');
      if (describedBy) input.setAttribute('aria-describedby', describedBy);
      else input.removeAttribute('aria-describedby');
      if (text) {
        input.setAttribute('aria-invalid', 'true');
        errorEl.textContent = text;
        errorEl.hidden = false;
      } else {
        input.removeAttribute('aria-invalid');
        errorEl.textContent = '';
        errorEl.hidden = true;
      }
    };
    showError(error || '');

    let answer = null;
    const onSubmit = (event) => {
      event.preventDefault();
      const text = String(input.value || '').trim();
      let problem = null;
      if (required && !text) problem = 'This field is required.';
      if (!problem && typeof validate === 'function') {
        try { problem = validate(text) || null; } catch (e) { problem = e?.message || String(e); }
      }
      if (problem) {
        showError(problem);
        input.focus();
        return;
      }
      answer = text;
      closeSheet(dialog, 'ok');
    };
    const onCancel = () => { answer = null; closeSheet(dialog, 'cancel'); };
    const onInput = () => { if (input.getAttribute('aria-invalid') === 'true') showError(''); };
    form.addEventListener('submit', onSubmit);
    cancel.addEventListener('click', onCancel);
    input.addEventListener('input', onInput);
    openSheet(dialog, {
      initialFocus: input,
      backdropClose: () => input.value === initial,
      onClose: () => {
        form.removeEventListener('submit', onSubmit);
        cancel.removeEventListener('click', onCancel);
        input.removeEventListener('input', onInput);
        resolve(answer);
      },
    });
    if (typeof input.select === 'function') input.select();
  });
  const result = promptQueue.then(run, run);
  promptQueue = result.then(() => undefined, () => undefined);
  return result;
}

/** Test hook. */
export function __resetForTests() {
  for (const entry of open.values()) entry.cleanup();
  open = new Map();
  stack = [];
  try { hostLiveRegions(); } catch { /* no document */ }
  confirmQueue = Promise.resolve();
  promptQueue = Promise.resolve();
}
