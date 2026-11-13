/**
 * Toasts: info/success/warning in #app-toasts-polite (role=status), errors in #app-toasts-alert
 * (role=alert). Errors persist until dismissed; the others auto-dismiss after 6 s, paused while hovered or
 * focused. At most 3 are visible. While a modal sheet is open the stack lives inside it (core/dialogs.js
 * moves #app-toasts), so a toast stays visible, clickable and announced above the backdrop.
 * @module core/toast
 */
import { render } from './dom.js';
import { icon } from './icons.js';
import { liveHost } from './dialogs.js';

const DEFAULT_TIMEOUT_MS = 6000;
const MIN_RESUME_MS = 2000;
const MAX_VISIBLE = 3;
const KINDS = new Set(['info', 'success', 'warning', 'error']);
const ICON_FOR = { info: 'info', success: 'check', warning: 'alert', error: 'alert' };

let live = [];   // oldest first: {id, kind, el, timer, remaining, startedAt, paused, dismiss}
let counter = 0;

function container(kind) {
  const alert = kind === 'error';
  const id = alert ? 'app-toasts-alert' : 'app-toasts-polite';
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.id = id;
    el.setAttribute('role', alert ? 'alert' : 'status');
    el.setAttribute('aria-live', alert ? 'assertive' : 'polite');
    const host = document.getElementById('app-toasts') || liveHost() || document.body;
    host.appendChild(el);
  }
  return el;
}

function startTimer(entry, ms) {
  clearTimeout(entry.timer);
  entry.startedAt = Date.now();
  entry.remaining = ms;
  entry.timer = setTimeout(() => entry.dismiss(), ms);
}

function pause(entry) {
  if (entry.paused || entry.kind === 'error') return;
  entry.paused = true;
  clearTimeout(entry.timer);
  entry.timer = null;
  entry.remaining = Math.max(0, entry.remaining - (Date.now() - entry.startedAt));
}

function resume(entry) {
  if (!entry.paused || entry.kind === 'error' || entry.gone) return;
  if (entry.hovered || entry.el.contains(document.activeElement)) return;
  entry.paused = false;
  startTimer(entry, Math.max(entry.remaining, MIN_RESUME_MS));
}

function button(className, label) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = className;
  if (label) b.textContent = label;
  return b;
}

/**
 * Show a toast.
 * @param {string} message plain text (never HTML)
 * @param {{kind?: 'info'|'success'|'warning'|'error', action?: {label: string, run: () => *}|null,
 *   timeoutMs?: number, id?: string}} [options] the same `id` replaces the earlier toast
 * @returns {{dismiss: () => void, el: HTMLElement, id: string}}
 */
export function toast(message, { kind = 'info', action = null, timeoutMs, id } = {}) {
  const k = KINDS.has(kind) ? kind : 'info';
  const toastId = id ? String(id) : `t${(counter += 1)}`;
  const existing = live.find((t) => t.id === toastId);
  if (existing) existing.dismiss({ restoreFocus: false });

  const el = document.createElement('div');
  el.className = `toast toast--${k}`;
  el.setAttribute('data-toast-id', toastId);
  const iconEl = document.createElement('span');
  iconEl.className = 'toast__icon';
  render(iconEl, icon(ICON_FOR[k]));
  const msg = document.createElement('p');
  msg.className = 'toast__msg';
  msg.textContent = String(message ?? '');
  el.append(iconEl, msg);

  const entry = {
    id: toastId, kind: k, el, timer: null, remaining: 0, startedAt: 0, paused: false, gone: false,
    hovered: false, returnFocus: null,
  };
  entry.dismiss = ({ restoreFocus = true } = {}) => {
    if (entry.gone) return;
    entry.gone = true;
    clearTimeout(entry.timer);
    const hadFocus = el.contains(document.activeElement);
    el.remove();
    live = live.filter((t) => t !== entry);
    if (hadFocus && restoreFocus) {
      const target = entry.returnFocus && entry.returnFocus.isConnected ? entry.returnFocus : document.getElementById('main');
      if (target && typeof target.focus === 'function') target.focus({ preventScroll: true });
    }
  };

  if (action && action.label && typeof action.run === 'function') {
    const act = button('btn btn--sm btn--ghost', action.label);
    act.addEventListener('click', () => {
      entry.dismiss();
      try {
        const result = action.run();
        if (result && typeof result.then === 'function') {
          result.then(null, (error) => toast(error?.message || String(error), { kind: 'error' }));
        }
      } catch (error) {
        toast(error?.message || String(error), { kind: 'error' });
      }
    });
    el.append(act);
  }
  const close = button('icon-btn toast__close', '');
  close.setAttribute('aria-label', 'Dismiss notification');
  render(close, icon('x'));
  close.addEventListener('click', () => entry.dismiss());
  el.append(close);

  el.addEventListener('mouseenter', () => { entry.hovered = true; pause(entry); });
  el.addEventListener('mouseleave', () => { entry.hovered = false; resume(entry); });
  el.addEventListener('focusin', (event) => {
    if (!entry.returnFocus && event.relatedTarget && !el.contains(event.relatedTarget)) entry.returnFocus = event.relatedTarget;
    pause(entry);
  });
  el.addEventListener('focusout', (event) => {
    if (!event.relatedTarget || !el.contains(event.relatedTarget)) {
      // Defer so the new focus target is known.
      setTimeout(() => resume(entry), 0);
    }
  });

  // Make room: the oldest non-error goes first, then the oldest error.
  while (live.length >= MAX_VISIBLE) {
    const victim = live.find((t) => t.kind !== 'error') || live[0];
    victim.dismiss({ restoreFocus: false });
  }
  container(k).appendChild(el);
  live.push(entry);
  if (k !== 'error') startTimer(entry, Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_TIMEOUT_MS);
  return { dismiss: () => entry.dismiss(), el, id: toastId };
}

/** Dismiss every toast. */
export function dismissAll() {
  for (const entry of live.slice()) entry.dismiss({ restoreFocus: false });
}

/** Test hook. */
export function __resetForTests() {
  for (const entry of live) clearTimeout(entry.timer);
  live = [];
  counter = 0;
}
