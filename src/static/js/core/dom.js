/**
 * DOM helpers. The only module that assigns innerHTML (inside render()), and only from SafeHtml.
 * @module core/dom
 */
import { toast } from './toast.js';

/** Markup that is safe to insert: built by html`` (values escaped) or raw() (trusted constants). */
export class SafeHtml {
  constructor(markup) { this.markup = String(markup); }
  toString() { return this.markup; }
}

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/**
 * Escape text for HTML text and quoted attribute values.
 * @param {*} value
 * @returns {string}
 */
export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

function piece(value) {
  if (value instanceof SafeHtml) return value.markup;
  if (value === null || value === undefined || value === false) return '';
  if (Array.isArray(value)) return value.map(piece).join('');
  return escapeHtml(String(value));
}

/**
 * Tagged template that escapes every interpolated value. SafeHtml values are inserted as is, arrays are
 * joined, null/undefined/false render nothing. Attribute values MUST be quoted in the template.
 * @returns {SafeHtml}
 */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i += 1) out += piece(values[i]) + strings[i + 1];
  return new SafeHtml(out);
}

/**
 * Trusted markup only: partials and constants. Never pass data through it.
 * @param {string} markup
 * @returns {SafeHtml}
 */
export function raw(markup) {
  return new SafeHtml(markup);
}

/**
 * Replace an element's content with SafeHtml (or an array of SafeHtml).
 * @param {Element} el
 * @param {SafeHtml|SafeHtml[]} content
 * @throws {TypeError} for a plain string or any other value
 */
export function render(el, content) {
  let markup;
  if (content instanceof SafeHtml) markup = content.markup;
  else if (Array.isArray(content) && content.every((c) => c instanceof SafeHtml)) markup = content.map((c) => c.markup).join('');
  else throw new TypeError('render() accepts SafeHtml only; build markup with html``');
  el.innerHTML = markup;
  return el;
}

/** @returns {Element|null} */
export function $(selector, root = document) { return root.querySelector(selector); }

/** @returns {Element[]} */
export function $$(selector, root = document) { return Array.from(root.querySelectorAll(selector)); }

/** @returns {HTMLElement|null} */
export function byId(id) { return document.getElementById(id); }

/**
 * Delegated listener: `handler(event, matched)` runs when the event target is inside an element matching
 * `selector` that is inside `root`. Uses closest(), so clicks on an SVG icon inside a button still match.
 * @returns {() => void} off
 */
export function on(root, type, selector, handler, options) {
  const listener = (event) => {
    const target = event.target;
    const start = target && target.nodeType === 3 ? target.parentElement : target;
    if (!start || typeof start.closest !== 'function') return;
    const matched = start.closest(selector);
    if (!matched) return;
    if (root !== matched && typeof root.contains === 'function' && !root.contains(matched)) return;
    handler(event, matched);
  };
  root.addEventListener(type, listener, options);
  return () => root.removeEventListener(type, listener, options);
}

function defaultActionError(error) {
  if (!error || error.kind === 'aborted' || error.name === 'AbortError') return;
  toast(error.message || String(error), { kind: 'error' });
}

/**
 * Click delegation for `[data-action]` elements: `map[action](el, event)`. Disabled or aria-disabled
 * elements are ignored. Thrown errors and rejected promises go to `onError` (default: an error toast;
 * aborted requests are ignored).
 * @param {Element} root
 * @param {Record<string, (el: Element, event: Event) => *>} map
 * @param {{onError?: (error: *) => void}} [options]
 * @returns {() => void} off
 */
export function actions(root, map, { onError = defaultActionError } = {}) {
  return on(root, 'click', '[data-action]', (event, el) => {
    const name = el.getAttribute('data-action');
    const fn = Object.prototype.hasOwnProperty.call(map, name) ? map[name] : null;
    if (typeof fn !== 'function') return;
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') return;
    try {
      const result = fn(el, event);
      if (result && typeof result.then === 'function') result.then(null, onError);
    } catch (error) {
      onError(error);
    }
  });
}

const busyState = new WeakMap();   // el -> {label, disabled, refocus}

/**
 * Mark a control busy: aria-busy and disabled; with `label`, the `.btn__label` text is swapped. Going back
 * to not busy restores the label and the disabled state the control had before, and, when the control had
 * keyboard focus as it went busy (disabling it dropped focus to <body>) and nothing else took focus since,
 * puts focus back on it.
 */
export function setBusy(el, busy, { label } = {}) {
  if (!el) return;
  const labelEl = el.querySelector ? el.querySelector('.btn__label') : null;
  if (busy) {
    if (!busyState.has(el)) {
      const focused = typeof document !== 'undefined' && document.activeElement === el;
      busyState.set(el, { label: labelEl ? labelEl.textContent : null, disabled: !!el.disabled, refocus: focused, swapped: false });
    }
    const st = busyState.get(el);
    el.setAttribute('aria-busy', 'true');
    el.disabled = true;
    if (label && labelEl) {
      labelEl.textContent = label;
      st.swapped = true;
    }
  } else {
    const st = busyState.get(el);
    el.removeAttribute('aria-busy');
    el.disabled = st ? st.disabled : false;
    if (st && st.swapped && labelEl && st.label !== null) labelEl.textContent = st.label;
    busyState.delete(el);
    if (st && st.refocus && !el.disabled && el.isConnected !== false && typeof document !== 'undefined') {
      const now = document.activeElement;
      if (!now || now === document.body || now === document.documentElement) el.focus({ preventScroll: true });
    }
  }
}

const SAFE_SCHEMES = new Set(['http:', 'https:', 'blob:']);

/**
 * A URL that may go into an href or src built from data, or '' when it may not: http:, https: and
 * blob: URLs (relative URLs resolve against the page), data:audio/… with {media: true}, and
 * data:image/png|jpeg|webp|gif with {image: true}. javascript:, other data: types and anything unparsable
 * give ''. Use it in templates: html`<a href="${safeUrl(take.url)}">`.
 * @param {string} url
 * @param {{media?: boolean, image?: boolean}} [options]
 * @returns {string}
 */
export function safeUrl(url, { media = false, image = false } = {}) {
  const text = String(url ?? '').trim();
  if (!text) return '';
  if (/^data:/i.test(text)) {
    if (media && /^data:audio\/[\w.+-]+[;,]/i.test(text)) return text;
    if (image && /^data:image\/(?:png|jpeg|webp|gif)[;,]/i.test(text)) return text;
    return '';
  }
  let parsed;
  try {
    parsed = typeof location !== 'undefined' && location.href ? new URL(text, location.href) : new URL(text);
  } catch {
    // No page to resolve against (tests in plain Node): a URL without a scheme is relative to this origin.
    return typeof location === 'undefined' && !/^[a-z][a-z0-9+.-]*:/i.test(text) ? text : '';
  }
  return SAFE_SCHEMES.has(parsed.protocol) ? text : '';
}

/** Set `el.hidden`; a no-op when it is unchanged. */
export function setHidden(el, hidden) {
  if (!el) return;
  const next = !!hidden;
  if (el.hidden !== next) el.hidden = next;
}

/**
 * Debounce: runs `fn` once `ms` after the last call. The returned function has cancel() and flush().
 */
export function debounce(fn, ms) {
  let timer = null;
  let lastArgs = null;
  const debounced = (...args) => {
    lastArgs = args;
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      const a = lastArgs;
      lastArgs = null;
      fn(...a);
    }, ms);
  };
  debounced.cancel = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    lastArgs = null;
  };
  debounced.flush = () => {
    if (timer === null) return;
    clearTimeout(timer);
    timer = null;
    const a = lastArgs;
    lastArgs = null;
    fn(...a);
  };
  return debounced;
}

/** Throttle: runs at most once per `ms`, on the leading call and once more at the end with the last args. */
export function throttle(fn, ms) {
  let last = -Infinity;
  let timer = null;
  let lastArgs = null;
  const throttled = (...args) => {
    const now = Date.now();
    const wait = last + ms - now;
    if (wait <= 0) {
      last = now;
      fn(...args);
      return;
    }
    lastArgs = args;
    if (timer === null) {
      timer = setTimeout(() => {
        timer = null;
        last = Date.now();
        const a = lastArgs;
        lastArgs = null;
        fn(...a);
      }, wait);
    }
  };
  throttled.cancel = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    lastArgs = null;
  };
  return throttled;
}

/** @returns {Promise<number>} resolves on the next animation frame */
export function nextFrame() {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(resolve);
    else setTimeout(() => resolve(Date.now()), 16);
  });
}

const EDITABLE_ROLES = new Set(['combobox', 'slider', 'listbox', 'spinbutton', 'textbox', 'searchbox']);
const BUTTON_INPUTS = new Set(['button', 'submit', 'reset', 'image']);

/**
 * True when keys typed at `target` belong to it: inputs (except button-like types), textarea, select,
 * contenteditable, and the combobox/slider/listbox/spinbutton roles.
 */
export function isEditable(target) {
  if (!target || target.nodeType !== 1) return false;
  const tag = String(target.tagName || '').toLowerCase();
  if (tag === 'textarea' || tag === 'select') return true;
  if (tag === 'input') return !BUTTON_INPUTS.has(String(target.type || target.getAttribute('type') || 'text').toLowerCase());
  if (target.isContentEditable) return true;
  const ce = target.getAttribute && target.getAttribute('contenteditable');
  if (ce !== null && ce !== undefined && ce !== 'false') return true;
  const role = target.getAttribute && target.getAttribute('role');
  return !!role && EDITABLE_ROLES.has(role);
}

function isHiddenTree(el, root) {
  for (let node = el; node && node !== root; node = node.parentElement) {
    if (node.hidden || (node.hasAttribute && node.hasAttribute('inert'))) return true;
  }
  return false;
}

/**
 * Focus the first visible, enabled match inside `root`. A heading without a tabindex gets tabindex="-1".
 * @returns {Element|null} the focused element
 */
export function focusFirst(root, selector = '[data-autofocus], h1, h2, button, [href], input, select, textarea') {
  if (!root) return null;
  for (const el of root.querySelectorAll(selector)) {
    if (el.disabled || isHiddenTree(el, root)) continue;
    if (/^h[1-6]$/i.test(el.tagName) && !el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');
    el.focus({ preventScroll: false });
    return el;
  }
  return null;
}
