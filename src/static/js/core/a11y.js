/**
 * Screen-reader announcements through the shell's two visually hidden live regions. While a modal sheet is
 * open the regions live inside it (core/dialogs.js moves them), because a modal makes the rest of the page
 * inert and screen readers then ignore live regions there.
 * @module core/a11y
 */
import { liveHost } from './dialogs.js';

const COALESCE_MS = 500;
const REGIONS = {
  polite: { id: 'app-sr-status', role: 'status', live: 'polite' },
  assertive: { id: 'app-sr-alert', role: 'alert', live: 'assertive' },
};

let state = { polite: fresh(), assertive: fresh() };

function fresh() {
  return { lastAt: -Infinity, pending: null, timer: null, frame: null };
}

function region(kind) {
  const spec = REGIONS[kind];
  let el = document.getElementById(spec.id);
  if (!el) {
    // The shell provides both regions; this fallback keeps announcements working without it.
    el = document.createElement('div');
    el.id = spec.id;
    el.className = 'visually-hidden';
    el.setAttribute('role', spec.role);
    el.setAttribute('aria-live', spec.live);
    (liveHost() || document.body).appendChild(el);
  }
  return el;
}

function write(kind, text) {
  const s = state[kind];
  const el = region(kind);
  s.lastAt = Date.now();
  el.textContent = '';
  const set = () => {
    s.frame = null;
    el.textContent = text;
  };
  if (typeof requestAnimationFrame === 'function') s.frame = requestAnimationFrame(set);
  else s.frame = setTimeout(set, 16);
}

/**
 * Announce a message: polite (#app-sr-status, role=status) or assertive (#app-sr-alert, role=alert).
 * The region is cleared, then set on the next frame. At most one message per region per 500 ms:
 * messages inside that window are coalesced and the last one wins.
 * @param {string} text
 * @param {{assertive?: boolean}} [options]
 */
export function announce(text, { assertive = false } = {}) {
  const message = String(text ?? '').trim();
  if (!message) return;
  const kind = assertive ? 'assertive' : 'polite';
  const s = state[kind];
  const wait = s.lastAt + COALESCE_MS - Date.now();
  if (wait <= 0 && s.timer === null) {
    write(kind, message);
    return;
  }
  s.pending = message;
  if (s.timer === null) {
    s.timer = setTimeout(() => {
      s.timer = null;
      const pending = s.pending;
      s.pending = null;
      if (pending) write(kind, pending);
    }, Math.max(0, wait));
  }
}

/** Test hook. */
export function __resetForTests() {
  state = { polite: fresh(), assertive: fresh() };
}
