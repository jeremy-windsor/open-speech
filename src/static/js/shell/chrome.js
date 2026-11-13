/**
 * Chrome metrics (WCAG 2.4.11 Focus Not Obscured). Measures the visible fixed and sticky chrome
 * and writes three custom properties on <html>, which base.css and shell.css use:
 *   --chrome-top          how far the top chrome reaches down (the top bar; the Reader's top bar)
 *   --chrome-bottom-base  how far the bottom chrome reaches up (bottom bar, mini-player, Reader transport)
 *   --chrome-bottom       the same plus the visible toast stack, at every width
 * Toasts are positioned from --chrome-bottom-base (so they sit above the bars), and scroll padding uses
 * --chrome-bottom (so a focused element scrolled into view is never hidden behind a toast either). On desktop
 * the stack sits bottom right over the content column, so it counts there too: an error toast stays until it
 * is dismissed, and a control Tab brings in from below would otherwise land behind it (WCAG 2.4.11).
 *
 * Sticky chrome (the desktop rail and docked mini-player, the phone top bar, the Reader's bars) is counted,
 * so content never scrolls in behind it; base.css drops the scroll padding while focus is inside chrome,
 * because a stuck bar always sits inside that padding and every Tab within it would scroll the page again.
 *
 * Chrome elements opt in with data-chrome="top" or data-chrome="bottom" (the shell's bars and mini-player
 * here; the Reader's .rd-topbar and .rdp-transport). Only fixed or sticky ones count.
 * @module shell/chrome
 */

const DESKTOP = '(min-width: 960px)';

let state = null;

function visible(el) {
  if (!el || !el.isConnected) return false;
  for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
    if (node.hidden) return false;
  }
  return true;
}

function positioned(el) {
  try {
    const pos = getComputedStyle(el).position;
    return !pos || pos === 'fixed' || pos === 'sticky';
  } catch {
    return true;
  }
}

/**
 * PURE: the chrome metrics from element rects.
 * @param {{viewportHeight: number, tops: DOMRect[], bottoms: DOMRect[], toasts?: DOMRect|null}} input
 * @returns {{top: number, bottomBase: number, bottom: number}} px
 */
export function computeMetrics({ viewportHeight, tops = [], bottoms = [], toasts = null }) {
  const vh = Number(viewportHeight) || 0;
  const clamp = (n) => Math.max(0, Math.min(vh || Infinity, Math.round(n)));
  let top = 0;
  for (const r of tops) if (r && r.height > 0 && r.top < vh / 2) top = Math.max(top, clamp(r.bottom));
  let bottomBase = 0;
  for (const r of bottoms) if (r && r.height > 0 && r.bottom > vh / 2) bottomBase = Math.max(bottomBase, clamp(vh - r.top));
  let bottom = bottomBase;
  if (toasts && toasts.height > 0) bottom = Math.max(bottom, clamp(vh - toasts.top));
  return { top, bottomBase, bottom };
}

function rectsOf(selector) {
  return Array.from(document.querySelectorAll(selector))
    .filter((el) => visible(el) && positioned(el))
    .map((el) => el.getBoundingClientRect());
}

function toastStack() {
  const host = document.getElementById('app-toasts');
  if (!host || !host.querySelector('.toast')) return null;
  if (host.closest && host.closest('dialog')) return null;   // inside an open sheet (top-anchored there)
  return host.getBoundingClientRect();
}

/** Measure now and write the custom properties. */
export function measure() {
  if (!state) return null;
  const metrics = computeMetrics({
    viewportHeight: globalThis.innerHeight || 0,
    tops: rectsOf('[data-chrome="top"]'),
    bottoms: rectsOf('[data-chrome="bottom"]'),
    toasts: toastStack(),
  });
  const root = document.documentElement;
  const write = (name, value) => {
    const text = `${value}px`;
    if (root.style.getPropertyValue(name) !== text) root.style.setProperty(name, text);
  };
  write('--chrome-top', metrics.top);
  write('--chrome-bottom-base', metrics.bottomBase);
  write('--chrome-bottom', metrics.bottom);
  state.last = metrics;
  observeAll();
  return metrics;
}

/** Measure on the next frame (coalesced). */
export function refresh() {
  if (!state || state.frame !== null) return;
  const run = () => {
    if (!state) return;
    state.frame = null;
    measure();
  };
  state.frame = typeof requestAnimationFrame === 'function' ? requestAnimationFrame(run) : setTimeout(run, 16);
}

function observeAll() {
  if (!state || !state.observer) return;
  const els = [...document.querySelectorAll('[data-chrome]'), document.getElementById('app-toasts'),
    document.getElementById('app-toasts-polite'), document.getElementById('app-toasts-alert')].filter(Boolean);
  for (const el of els) {
    if (state.observed.has(el)) continue;
    state.observed.add(el);
    state.observer.observe(el);
  }
}

/**
 * Start measuring: a ResizeObserver on every chrome element and the toast stack, plus window resize.
 * @returns {{measure: () => object, refresh: () => void, destroy: () => void}}
 */
export function initChrome() {
  destroy();
  state = { frame: null, observer: null, observed: new Set(), last: null, offs: [] };
  if (typeof ResizeObserver === 'function') state.observer = new ResizeObserver(() => refresh());
  const onResize = () => refresh();
  window.addEventListener('resize', onResize);
  state.offs.push(() => window.removeEventListener('resize', onResize));
  try {
    const mq = matchMedia(DESKTOP);
    const onChange = () => refresh();
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    state.offs.push(() => { if (mq.removeEventListener) mq.removeEventListener('change', onChange); });
  } catch { /* no matchMedia */ }
  measure();
  return { measure, refresh, destroy };
}

/** @returns {{top: number, bottomBase: number, bottom: number}|null} the last metrics */
export function current() {
  return state ? state.last : null;
}

/** Stop measuring. */
export function destroy() {
  if (!state) return;
  if (state.frame !== null) {
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(state.frame);
    clearTimeout(state.frame);
  }
  if (state.observer) state.observer.disconnect();
  for (const off of state.offs) off();
  state = null;
}

/** Test hook. */
export function __resetForTests() {
  destroy();
}
