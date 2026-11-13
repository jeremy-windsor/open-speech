// A small fake DOM for Node behaviour tests: FakeDocument, FakeElement, FakeText, events with capture and
// bubble phases, a CSS selector subset, form controls, <dialog>, focus, and innerHTML through
// html-parser.js. It models what the UI code uses; it is not a browser.
import { parseInto } from './html-parser.js';

export const HTML_NS = 'http://www.w3.org/1999/xhtml';
export const SVG_NS = 'http://www.w3.org/2000/svg';
export const VOID_ELEMENTS = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']);
const RAW_TEXT = new Set(['script', 'style']);

let errorReporter = (error) => { uncaught.push(error); };
export const uncaught = [];
export function setErrorReporter(fn) { errorReporter = fn; }
function report(error) {
  try { errorReporter(error); } catch { /* ignore */ }
}

/* ───────────────────────────── events ───────────────────────────── */

export class FakeEvent {
  constructor(type, init = {}) {
    this.type = String(type);
    this.bubbles = !!init.bubbles;
    this.cancelable = !!init.cancelable;
    this.composed = !!init.composed;
    this.defaultPrevented = false;
    this.target = null;
    this.currentTarget = null;
    this.eventPhase = 0;
    this.isTrusted = false;
    this.timeStamp = Date.now();
    this._stop = false;
    this._stopNow = false;
    this._path = [];
  }
  preventDefault() { if (this.cancelable) this.defaultPrevented = true; }
  stopPropagation() { this._stop = true; }
  stopImmediatePropagation() { this._stop = true; this._stopNow = true; }
  get cancelBubble() { return this._stop; }
  set cancelBubble(v) { if (v) this._stop = true; }
  get returnValue() { return !this.defaultPrevented; }
  composedPath() { return this._path.slice(); }
}
FakeEvent.NONE = 0; FakeEvent.CAPTURING_PHASE = 1; FakeEvent.AT_TARGET = 2; FakeEvent.BUBBLING_PHASE = 3;

export class FakeCustomEvent extends FakeEvent {
  constructor(type, init = {}) { super(type, init); this.detail = init.detail ?? null; }
}

export class FakeUIEvent extends FakeEvent {
  constructor(type, init = {}) { super(type, init); this.detail = init.detail ?? 0; this.view = init.view ?? null; }
}

export class FakeFocusEvent extends FakeUIEvent {
  constructor(type, init = {}) { super(type, init); this.relatedTarget = init.relatedTarget ?? null; }
}

export class FakeMouseEvent extends FakeUIEvent {
  constructor(type, init = {}) {
    super(type, init);
    for (const k of ['clientX', 'clientY', 'screenX', 'screenY', 'pageX', 'pageY', 'offsetX', 'offsetY', 'button', 'buttons']) this[k] = init[k] ?? 0;
    for (const k of ['ctrlKey', 'shiftKey', 'altKey', 'metaKey']) this[k] = !!init[k];
    this.relatedTarget = init.relatedTarget ?? null;
  }
}

export class FakePointerEvent extends FakeMouseEvent {
  constructor(type, init = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
    this.pointerType = init.pointerType ?? 'mouse';
    this.isPrimary = init.isPrimary ?? true;
  }
}

export class FakeKeyboardEvent extends FakeUIEvent {
  constructor(type, init = {}) {
    super(type, init);
    this.key = init.key ?? '';
    this.code = init.code ?? '';
    this.repeat = !!init.repeat;
    this.isComposing = !!init.isComposing;
    for (const k of ['ctrlKey', 'shiftKey', 'altKey', 'metaKey']) this[k] = !!init[k];
    // EventModifierInit.modifierAltGraph: AltGr, which Windows also reports as ctrlKey + altKey.
    this.__altGraph = !!init.modifierAltGraph;
  }
  getModifierState(name) {
    return { Control: this.ctrlKey, Shift: this.shiftKey, Alt: this.altKey, Meta: this.metaKey, AltGraph: this.__altGraph }[name] || false;
  }
}

export class FakeInputEvent extends FakeUIEvent {
  constructor(type, init = {}) { super(type, init); this.data = init.data ?? null; this.inputType = init.inputType ?? ''; }
}

export class FakeEventTarget {
  constructor() { this.__listeners = new Map(); }

  addEventListener(type, fn, options) {
    if (!fn) return;
    const capture = typeof options === 'boolean' ? options : !!options?.capture;
    const once = typeof options === 'object' && !!options?.once;
    const list = this.__listeners.get(type) || [];
    if (list.some((l) => l.fn === fn && l.capture === capture)) return;
    const entry = { fn, capture, once };
    list.push(entry);
    this.__listeners.set(type, list);
    const signal = typeof options === 'object' ? options?.signal : null;
    if (signal) signal.addEventListener('abort', () => this.removeEventListener(type, fn, options), { once: true });
  }

  removeEventListener(type, fn, options) {
    const capture = typeof options === 'boolean' ? options : !!options?.capture;
    const list = this.__listeners.get(type);
    if (!list) return;
    this.__listeners.set(type, list.filter((l) => !(l.fn === fn && l.capture === capture)));
  }

  /** Test helper: how many listeners of `type` are attached (for leak checks). */
  __listenerCount(type) { return (this.__listeners.get(type) || []).length; }

  /** The propagation path above this target (overridden by nodes). */
  __parentTarget() { return null; }

  dispatchEvent(event) {
    if (!(event instanceof FakeEvent)) throw new TypeError('dispatchEvent needs a fake Event (globalThis.Event inside the harness)');
    event.target = this;
    const path = [];
    for (let t = this; t; t = t.__parentTarget()) path.push(t);
    event._path = path;
    // capture: from the top down to the target's parent
    for (let i = path.length - 1; i >= 1 && !event._stop; i -= 1) invoke(path[i], event, 1);
    if (!event._stop) invoke(this, event, 2);
    if (event.bubbles) {
      for (let i = 1; i < path.length && !event._stop; i += 1) invoke(path[i], event, 3);
    }
    event.currentTarget = null;
    event.eventPhase = 0;
    return !event.defaultPrevented;
  }
}

function invoke(target, event, phase) {
  event.currentTarget = target;
  event.eventPhase = phase;
  const list = (target.__listeners.get(event.type) || []).slice();
  for (const l of list) {
    if (event._stopNow) break;
    if (phase === 1 && !l.capture) continue;
    if (phase === 3 && l.capture) continue;
    if (l.once) target.removeEventListener(event.type, l.fn, { capture: l.capture });
    try {
      if (typeof l.fn === 'function') l.fn.call(target, event);
      else if (l.fn && typeof l.fn.handleEvent === 'function') l.fn.handleEvent(event);
    } catch (error) {
      report(error);
    }
  }
  if (phase !== 1 && !event._stopNow && (phase === 2 || event.bubbles)) {
    // Property handlers (el.onclick = fn, ws.onmessage = fn); on*= attributes are never run.
    const prop = Object.prototype.hasOwnProperty.call(target, `on${event.type}`) ? target[`on${event.type}`] : null;
    if (typeof prop === 'function') {
      try { prop.call(target, event); } catch (error) { report(error); }
    }
  }
}

/* ───────────────────────────── nodes ───────────────────────────── */

// Back-links (a node's document and parent, a token list's element, the document's window) are own but NOT
// enumerable. node:assert builds a failing assertion's message with util.inspect (depth 1000, getters, no
// custom inspect), which walks enumerable own properties only: with enumerable back-links one element
// expanded the whole document again from every descendant, so a failing assert.equal(elementA, elementB)
// allocated until the run hit its memory cap and the real failure was lost.
function backLink(obj, name, value) {
  Object.defineProperty(obj, name, { value, writable: true, enumerable: false, configurable: true });
}

export class FakeNode extends FakeEventTarget {
  constructor(doc) {
    super();
    backLink(this, 'ownerDocument', doc);
    backLink(this, 'parentNode', null);
    this.childNodes = [];
  }
  get parentElement() { return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null; }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get nextSibling() {
    if (!this.parentNode) return null;
    const s = this.parentNode.childNodes;
    return s[s.indexOf(this) + 1] || null;
  }
  get previousSibling() {
    if (!this.parentNode) return null;
    const s = this.parentNode.childNodes;
    return s[s.indexOf(this) - 1] || null;
  }
  get isConnected() {
    let n = this;
    while (n.parentNode) n = n.parentNode;
    return n.nodeType === 9;
  }
  __parentTarget() {
    if (this.parentNode) return this.parentNode;
    return null;
  }
  hasChildNodes() { return this.childNodes.length > 0; }
  getRootNode() {
    let n = this;
    while (n.parentNode) n = n.parentNode;
    return n;
  }
  contains(node) {
    for (let n = node; n; n = n.parentNode) if (n === this) return true;
    return false;
  }
  appendChild(node) { return this.insertBefore(node, null); }
  insertBefore(node, ref) {
    if (node === null || node === undefined) throw new TypeError('insertBefore needs a node');
    if (node.nodeType === 11) {
      for (const child of node.childNodes.slice()) this.insertBefore(child, ref);
      return node;
    }
    if (node.contains(this)) throw new Error('HierarchyRequestError');
    if (node.parentNode) node.parentNode.removeChild(node);
    const at = ref ? this.childNodes.indexOf(ref) : -1;
    if (ref && at < 0) throw new Error('NotFoundError: ref is not a child');
    if (at < 0) this.childNodes.push(node);
    else this.childNodes.splice(at, 0, node);
    node.parentNode = this;
    adopt(node, this.ownerDocument || this);
    return node;
  }
  removeChild(node) {
    const at = this.childNodes.indexOf(node);
    if (at < 0) throw new Error('NotFoundError: not a child');
    this.childNodes.splice(at, 1);
    node.parentNode = null;
    const doc = this.ownerDocument || (this.nodeType === 9 ? this : null);
    if (doc && doc.activeElement && node.contains(doc.activeElement)) doc.__activeElement = null;
    return node;
  }
  replaceChild(node, old) {
    this.insertBefore(node, old);
    this.removeChild(old);
    return old;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  append(...nodes) { for (const n of nodes) this.appendChild(toNode(this, n)); }
  prepend(...nodes) {
    const first = this.firstChild;
    for (const n of nodes) this.insertBefore(toNode(this, n), first);
  }
  replaceChildren(...nodes) {
    for (const child of this.childNodes.slice()) this.removeChild(child);
    this.append(...nodes);
  }
  before(...nodes) { if (this.parentNode) for (const n of nodes) this.parentNode.insertBefore(toNode(this, n), this); }
  after(...nodes) {
    if (!this.parentNode) return;
    const next = this.nextSibling;
    for (const n of nodes) this.parentNode.insertBefore(toNode(this, n), next);
  }
  replaceWith(...nodes) {
    if (!this.parentNode) return;
    const parent = this.parentNode;
    const next = this.nextSibling;
    parent.removeChild(this);
    for (const n of nodes) parent.insertBefore(toNode(parent, n), next);
  }
  get textContent() {
    if (this.nodeType === 3) return this.data;
    return this.childNodes.map((c) => (c.nodeType === 1 || c.nodeType === 3 || c.nodeType === 11 ? c.textContent : '')).join('');
  }
  set textContent(value) {
    for (const child of this.childNodes.slice()) this.removeChild(child);
    const text = value === null || value === undefined ? '' : String(value);
    if (text) this.appendChild(new FakeText(this.ownerDocument || this, text));
  }
}

function adopt(node, doc) {
  if (!doc || doc.nodeType !== 9) return;
  node.ownerDocument = doc;
  for (const c of node.childNodes) adopt(c, doc);
}

function toNode(ctx, n) {
  if (n instanceof FakeNode) return n;
  return new FakeText(ctx.ownerDocument || ctx, String(n));
}

export class FakeText extends FakeNode {
  constructor(doc, data = '') {
    super(doc);
    this.nodeType = 3;
    this.nodeName = '#text';
    this.data = String(data);
  }
  get nodeValue() { return this.data; }
  set nodeValue(v) { this.data = String(v); }
  get wholeText() { return this.data; }
  cloneNode() { return new FakeText(this.ownerDocument, this.data); }
}

export class FakeComment extends FakeNode {
  constructor(doc, data = '') {
    super(doc);
    this.nodeType = 8;
    this.nodeName = '#comment';
    this.data = String(data);
  }
  get textContent() { return ''; }
  cloneNode() { return new FakeComment(this.ownerDocument, this.data); }
}

export class FakeFragment extends FakeNode {
  constructor(doc) {
    super(doc);
    this.nodeType = 11;
    this.nodeName = '#document-fragment';
  }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  querySelector(sel) { return querySelectorAll(this, sel)[0] || null; }
  querySelectorAll(sel) { return querySelectorAll(this, sel); }
  getElementById(id) { return querySelectorAll(this, '*').find((e) => e.id === id) || null; }
}

/* ───────────────────────────── element helpers ───────────────────────────── */

const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
const kebab = (s) => s.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

class TokenList {
  constructor(el, attr) { backLink(this, 'el', el); this.attr = attr; }
  _get() { return (this.el.getAttribute(this.attr) || '').split(/\s+/).filter(Boolean); }
  _set(list) { this.el.setAttribute(this.attr, list.join(' ')); }
  get length() { return this._get().length; }
  get value() { return this.el.getAttribute(this.attr) || ''; }
  item(i) { return this._get()[i] ?? null; }
  contains(t) { return this._get().includes(t); }
  add(...tokens) {
    const list = this._get();
    for (const t of tokens) if (!list.includes(t)) list.push(t);
    this._set(list);
  }
  remove(...tokens) { this._set(this._get().filter((t) => !tokens.includes(t))); }
  toggle(token, force) {
    const has = this.contains(token);
    const want = force === undefined ? !has : !!force;
    if (want && !has) this.add(token);
    if (!want && has) this.remove(token);
    return want;
  }
  replace(a, b) {
    if (!this.contains(a)) return false;
    this._set(this._get().map((t) => (t === a ? b : t)));
    return true;
  }
  values() { return this._get().values(); }
  forEach(fn) { this._get().forEach(fn); }
  [Symbol.iterator]() { return this._get()[Symbol.iterator](); }
  toString() { return this.value; }
}

function datasetProxy(el) {
  return new Proxy({}, {
    get(_, prop) {
      if (typeof prop !== 'string') return undefined;
      const v = el.getAttribute(`data-${kebab(prop)}`);
      return v === null ? undefined : v;
    },
    set(_, prop, value) { el.setAttribute(`data-${kebab(String(prop))}`, String(value)); return true; },
    deleteProperty(_, prop) { el.removeAttribute(`data-${kebab(String(prop))}`); return true; },
    has(_, prop) { return el.hasAttribute(`data-${kebab(String(prop))}`); },
    ownKeys() {
      return el.getAttributeNames().filter((n) => n.startsWith('data-')).map((n) => camel(n.slice(5)));
    },
    getOwnPropertyDescriptor(_, prop) {
      const v = el.getAttribute(`data-${kebab(String(prop))}`);
      return v === null ? undefined : { value: v, enumerable: true, configurable: true, writable: true };
    },
  });
}

function styleProxy() {
  const props = new Map();
  const api = {
    setProperty(name, value) {
      if (value === null || value === undefined || value === '') props.delete(String(name));
      else props.set(String(name), String(value));
    },
    getPropertyValue(name) { return props.get(String(name)) ?? ''; },
    removeProperty(name) { const v = props.get(String(name)) ?? ''; props.delete(String(name)); return v; },
    item(i) { return [...props.keys()][i] ?? ''; },
    get length() { return props.size; },
    get cssText() { return [...props].map(([k, v]) => `${k}: ${v};`).join(' '); },
    set cssText(text) {
      props.clear();
      for (const decl of String(text).split(';')) {
        const i = decl.indexOf(':');
        if (i > 0) props.set(decl.slice(0, i).trim(), decl.slice(i + 1).trim());
      }
    },
  };
  return new Proxy(api, {
    get(target, prop) {
      if (prop in target) return typeof target[prop] === 'function' ? target[prop].bind(target) : target[prop];
      if (typeof prop !== 'string') return undefined;
      return props.get(kebab(prop)) ?? '';
    },
    set(target, prop, value) {
      if (prop === 'cssText') { target.cssText = value; return true; }
      api.setProperty(kebab(String(prop)), value);
      return true;
    },
  });
}

const BOOLEAN_PROPS = { hidden: 'hidden', disabled: 'disabled', required: 'required', readOnly: 'readonly', multiple: 'multiple', autofocus: 'autofocus', inert: 'inert', noValidate: 'novalidate', defaultChecked: 'checked', defaultSelected: 'selected', controls: 'controls', loop: 'loop', defaultMuted: 'muted' };
const STRING_PROPS = { id: 'id', className: 'class', title: 'title', lang: 'lang', dir: 'dir', name: 'name', placeholder: 'placeholder', alt: 'alt', htmlFor: 'for', rel: 'rel', target: 'target', download: 'download', accept: 'accept', autocomplete: 'autocomplete', inputMode: 'inputmode', role: 'role', min: 'min', max: 'max', step: 'step', pattern: 'pattern', action: 'action', enctype: 'enctype', label: 'label', preload: 'preload', defaultValue: 'value', ariaLabel: 'aria-label' };

const FOCUSABLE_TAGS = new Set(['button', 'select', 'textarea', 'summary', 'iframe']);
const FORM_CONTROLS = new Set(['input', 'button', 'select', 'textarea', 'fieldset', 'option', 'optgroup']);

export class FakeElement extends FakeNode {
  constructor(doc, localName, namespaceURI = HTML_NS) {
    super(doc);
    this.nodeType = 1;
    this.namespaceURI = namespaceURI;
    this.localName = namespaceURI === HTML_NS ? String(localName).toLowerCase() : String(localName);
    this.__attrs = new Map();
    this.__state = {};
    this.__rect = null;
    this.__scrolledIntoView = [];
    this.classList = new TokenList(this, 'class');
    this.relList = new TokenList(this, 'rel');
    this.dataset = datasetProxy(this);
    this.style = styleProxy();
    this.scrollTop = 0;
    this.scrollLeft = 0;
  }

  get tagName() { return this.namespaceURI === HTML_NS ? this.localName.toUpperCase() : this.localName; }
  get nodeName() { return this.tagName; }

  /* attributes */
  getAttribute(name) {
    const v = this.__attrs.get(this._attrName(name));
    return v === undefined ? null : v;
  }
  setAttribute(name, value) {
    const n = this._attrName(name);
    const old = this.getAttribute(n);
    this.__attrs.set(n, String(value));
    this._attrChanged(n, old, String(value));
  }
  removeAttribute(name) {
    const n = this._attrName(name);
    if (!this.__attrs.has(n)) return;
    const old = this.__attrs.get(n);
    this.__attrs.delete(n);
    this._attrChanged(n, old, null);
  }
  hasAttribute(name) { return this.__attrs.has(this._attrName(name)); }
  toggleAttribute(name, force) {
    const has = this.hasAttribute(name);
    const want = force === undefined ? !has : !!force;
    if (want && !has) this.setAttribute(name, '');
    if (!want && has) this.removeAttribute(name);
    return want;
  }
  getAttributeNames() { return [...this.__attrs.keys()]; }
  get attributes() {
    const list = [...this.__attrs].map(([name, value]) => ({ name, value, localName: name }));
    list.getNamedItem = (n) => list.find((a) => a.name === n) || null;
    return list;
  }
  _attrName(name) { return this.namespaceURI === HTML_NS ? String(name).toLowerCase() : String(name); }
  _attrChanged(name, old, value) {
    if (name === 'open' && this.localName === 'details' && (old === null) !== (value === null)) {
      queueMicrotask(() => this.dispatchEvent(new FakeEvent('toggle')));
    }
  }

  /* tree */
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0] || null; }
  get lastElementChild() { const c = this.children; return c[c.length - 1] || null; }
  get nextElementSibling() {
    if (!this.parentNode) return null;
    const s = this.parentNode.childNodes.filter((c) => c.nodeType === 1);
    return s[s.indexOf(this) + 1] || null;
  }
  get previousElementSibling() {
    if (!this.parentNode) return null;
    const s = this.parentNode.childNodes.filter((c) => c.nodeType === 1);
    return s[s.indexOf(this) - 1] || null;
  }
  querySelector(sel) { return querySelectorAll(this, sel)[0] || null; }
  querySelectorAll(sel) { return querySelectorAll(this, sel); }
  getElementsByTagName(tag) { return querySelectorAll(this, tag); }
  getElementsByClassName(cls) { return querySelectorAll(this, cls.split(/\s+/).filter(Boolean).map((c) => `.${c}`).join('')); }
  matches(sel) { return matchesSelector(this, sel, this); }
  closest(sel) {
    const parsed = parseSelector(sel);
    for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (matchParsed(n, parsed, n)) return n;
    return null;
  }
  cloneNode(deep = false) {
    const copy = this.ownerDocument.createElementNS(this.namespaceURI, this.localName);
    for (const [k, v] of this.__attrs) copy.__attrs.set(k, v);
    if (deep) for (const c of this.childNodes) copy.appendChild(c.cloneNode(true));
    return copy;
  }
  insertAdjacentElement(where, el) {
    const w = where.toLowerCase();
    if (w === 'beforebegin') this.before(el);
    else if (w === 'afterbegin') this.prepend(el);
    else if (w === 'beforeend') this.append(el);
    else if (w === 'afterend') this.after(el);
    return el;
  }

  /* markup */
  get innerHTML() { return this.childNodes.map((c) => serialize(c, this)).join(''); }
  set innerHTML(markup) {
    for (const child of this.childNodes.slice()) this.removeChild(child);
    parseInto(this, String(markup ?? ''));
    if (this.localName === 'select') this._resetSelection();
  }
  get outerHTML() { return serialize(this, this.parentNode); }
  get innerText() { return this.textContent; }
  set innerText(v) { this.textContent = v; }

  /* reflection */
  get tabIndex() {
    const v = this.getAttribute('tabindex');
    if (v !== null && /^-?\d+$/.test(v.trim())) return Number(v);
    return this._nativelyFocusable() ? 0 : -1;
  }
  set tabIndex(v) { this.setAttribute('tabindex', String(v)); }
  get href() { return this.getAttribute('href') ?? ''; }
  set href(v) { this.setAttribute('href', v); }
  get src() { return this.getAttribute('src') ?? ''; }
  set src(v) { this.setAttribute('src', v); }
  get isContentEditable() {
    for (let n = this; n && n.nodeType === 1; n = n.parentNode) {
      const ce = n.getAttribute('contenteditable');
      if (ce !== null) return ce !== 'false';
    }
    return false;
  }
  get contentEditable() { return this.getAttribute('contenteditable') ?? 'inherit'; }
  set contentEditable(v) { this.setAttribute('contenteditable', v); }
  get offsetWidth() { return this.__rect?.width ?? 0; }
  get offsetHeight() { return this.__rect?.height ?? 0; }
  get clientWidth() { return this.__rect?.width ?? 0; }
  get clientHeight() { return this.__rect?.height ?? 0; }
  get scrollHeight() { return this.__rect?.scrollHeight ?? this.__rect?.height ?? 0; }
  get offsetParent() { return this.parentElement; }

  getBoundingClientRect() {
    const r = this.__rect || {};
    const x = r.x ?? r.left ?? 0;
    const y = r.y ?? r.top ?? 0;
    const width = r.width ?? 0;
    const height = r.height ?? 0;
    return { x, y, left: x, top: y, width, height, right: x + width, bottom: y + height };
  }
  getClientRects() { return this.__rect ? [this.getBoundingClientRect()] : []; }
  scrollIntoView(options) { this.__scrolledIntoView.push(options ?? true); }
  scrollTo(x, y) {
    if (typeof x === 'object') { this.scrollTop = x.top ?? this.scrollTop; this.scrollLeft = x.left ?? this.scrollLeft; }
    else { this.scrollLeft = x; this.scrollTop = y; }
  }
  scrollBy(x, y) {
    if (typeof x === 'object') { this.scrollTop += x.top ?? 0; this.scrollLeft += x.left ?? 0; }
    else { this.scrollLeft += x; this.scrollTop += y; }
  }

  /* focus */
  _nativelyFocusable() {
    const t = this.localName;
    if (FOCUSABLE_TAGS.has(t)) return true;
    if (t === 'input') return (this.getAttribute('type') || '').toLowerCase() !== 'hidden';
    if (t === 'a' || t === 'area') return this.hasAttribute('href');
    if ((t === 'audio' || t === 'video') && this.hasAttribute('controls')) return true;
    return this.isContentEditable;
  }
  _focusable() {
    if (!this.isConnected) return false;
    if (this.disabled) return false;
    for (let n = this; n && n.nodeType === 1; n = n.parentNode) {
      if (n.hasAttribute('hidden') || n.hasAttribute('inert')) return false;
      if (n.localName === 'dialog' && !n.hasAttribute('open')) return false;
    }
    return this.hasAttribute('tabindex') || this._nativelyFocusable();
  }
  focus() {
    if (!this._focusable()) return;
    const doc = this.ownerDocument;
    const old = doc.activeElement;
    if (old === this) return;
    doc.__activeElement = this;
    if (old && old !== doc.body) {
      old.dispatchEvent(new FakeFocusEvent('blur', { relatedTarget: this }));
      old.dispatchEvent(new FakeFocusEvent('focusout', { bubbles: true, relatedTarget: this }));
    }
    this.dispatchEvent(new FakeFocusEvent('focus', { relatedTarget: old === doc.body ? null : old }));
    this.dispatchEvent(new FakeFocusEvent('focusin', { bubbles: true, relatedTarget: old === doc.body ? null : old }));
  }
  blur() {
    const doc = this.ownerDocument;
    if (doc.activeElement !== this) return;
    doc.__activeElement = null;
    this.dispatchEvent(new FakeFocusEvent('blur'));
    this.dispatchEvent(new FakeFocusEvent('focusout', { bubbles: true }));
  }

  /* activation */
  click() {
    if (this.disabled) return;
    const doc = this.ownerDocument;
    const t = this.localName;
    const type = (this.getAttribute('type') || '').toLowerCase();
    let restore = null;
    if (t === 'input' && type === 'checkbox') {
      const was = this.checked;
      this.checked = !was;
      restore = () => { this.checked = was; };
    } else if (t === 'input' && type === 'radio') {
      const was = this.checked;
      const others = radioGroup(this).filter((r) => r !== this && r.checked);
      this.checked = true;
      restore = () => { this.checked = was; for (const o of others) o.checked = true; };
      this.__radioChanged = !was;
    }
    const ok = this.dispatchEvent(new FakeMouseEvent('click', { bubbles: true, cancelable: true, composed: true, view: doc.defaultView, detail: 1 }));
    if (!ok) {
      if (restore) restore();
      return;
    }
    if (t === 'input' && (type === 'checkbox' || (type === 'radio' && this.__radioChanged))) {
      this.dispatchEvent(new FakeInputEvent('input', { bubbles: true }));
      this.dispatchEvent(new FakeEvent('change', { bubbles: true }));
      return;
    }
    activationBehaviour(this);
  }

  /* form controls */
  get disabled() {
    if (!FORM_CONTROLS.has(this.localName)) return false;
    if (this.hasAttribute('disabled')) return true;
    if (this.localName === 'option') {
      const group = this.parentElement;
      return !!(group && group.localName === 'optgroup' && group.hasAttribute('disabled'));
    }
    for (let n = this.parentElement; n; n = n.parentElement) {
      if (n.localName === 'fieldset' && n.hasAttribute('disabled')) {
        const legend = n.children.find((c) => c.localName === 'legend');
        if (!legend || !legend.contains(this)) return true;
      }
    }
    return false;
  }
  set disabled(v) { this.toggleAttribute('disabled', !!v); }
  get type() {
    const t = this.localName;
    const v = (this.getAttribute('type') || '').toLowerCase();
    if (t === 'input') return v || 'text';
    if (t === 'button') return v === 'button' || v === 'reset' ? v : 'submit';
    if (t === 'select') return this.hasAttribute('multiple') ? 'select-multiple' : 'select-one';
    if (t === 'textarea') return 'textarea';
    return v;
  }
  set type(v) { this.setAttribute('type', v); }
  get form() {
    for (let n = this.parentElement; n; n = n.parentElement) if (n.localName === 'form') return n;
    return null;
  }
  get labels() {
    const doc = this.ownerDocument;
    const out = [];
    if (this.id) out.push(...querySelectorAll(doc, 'label').filter((l) => l.getAttribute('for') === this.id));
    for (let n = this.parentElement; n; n = n.parentElement) if (n.localName === 'label' && !out.includes(n)) out.push(n);
    return out;
  }
  get value() {
    const t = this.localName;
    if (t === 'input') {
      if ('value' in this.__state) return this.__state.value;
      const v = this.getAttribute('value');
      const type = this.type;
      if (v === null) return type === 'checkbox' || type === 'radio' ? 'on' : (type === 'range' ? rangeDefault(this) : '');
      return v;
    }
    if (t === 'textarea') return 'value' in this.__state ? this.__state.value : this.textContent;
    if (t === 'select') {
      const opt = this.options[this.selectedIndex];
      return opt ? opt.value : '';
    }
    if (t === 'option') return this.getAttribute('value') ?? this.textContent.replace(/\s+/g, ' ').trim();
    if (t === 'button' || t === 'output' || t === 'data' || t === 'li' || t === 'meter' || t === 'progress' || t === 'param') {
      if (t === 'output') return this.textContent;
      return this.getAttribute('value') ?? '';
    }
    return this.__state.value ?? undefined;
  }
  set value(v) {
    const t = this.localName;
    const s = v === null || v === undefined ? '' : String(v);
    if (t === 'input') {
      let next = s;
      if (this.type === 'range') next = clampRange(this, s);
      if (this.hasAttribute('maxlength') && /^(text|search|password|tel|url|email)$/.test(this.type)) {
        // the maxlength attribute only limits typing; a script may set a longer value
      }
      this.__state.value = next;
    } else if (t === 'textarea') {
      this.__state.value = s;
    } else if (t === 'select') {
      const i = this.options.findIndex((o) => o.value === s);
      this.selectedIndex = i;
    } else if (t === 'option' || t === 'button' || t === 'data' || t === 'param') {
      this.setAttribute('value', s);
    } else if (t === 'output') {
      this.textContent = s;
    } else {
      this.__state.value = v;
    }
  }
  get valueAsNumber() { const n = Number(this.value); return this.value === '' ? NaN : n; }
  set valueAsNumber(n) { this.value = String(n); }
  get checked() {
    if ('checked' in this.__state) return this.__state.checked;
    return this.hasAttribute('checked');
  }
  set checked(v) {
    this.__state.checked = !!v;
    if (v && this.localName === 'input' && this.type === 'radio') {
      for (const r of radioGroup(this)) if (r !== this) r.__state.checked = false;
    }
  }
  get indeterminate() { return !!this.__state.indeterminate; }
  set indeterminate(v) { this.__state.indeterminate = !!v; }
  get maxLength() { const v = this.getAttribute('maxlength'); return v === null ? -1 : Number(v); }
  set maxLength(v) { this.setAttribute('maxlength', String(v)); }
  get selectionStart() { return this.__state.selStart ?? String(this.value ?? '').length; }
  set selectionStart(v) { this.__state.selStart = v; }
  get selectionEnd() { return this.__state.selEnd ?? String(this.value ?? '').length; }
  set selectionEnd(v) { this.__state.selEnd = v; }
  select() { this.__state.selStart = 0; this.__state.selEnd = String(this.value ?? '').length; this.__state.selected = true; }
  setSelectionRange(a, b) { this.__state.selStart = a; this.__state.selEnd = b; }
  setRangeText(text) { this.value = text; }
  checkValidity() { return true; }
  reportValidity() { return true; }
  setCustomValidity(msg) { this.__state.customValidity = String(msg); }
  get validationMessage() { return this.__state.customValidity || ''; }

  /* select */
  get options() {
    if (this.localName === 'select' || this.localName === 'datalist') return querySelectorAll(this, 'option');
    return undefined;
  }
  get length() { return this.localName === 'select' ? this.options.length : undefined; }
  get selectedOptions() { return this.localName === 'select' ? this.options.filter((o) => o.selected) : undefined; }
  get selectedIndex() {
    if (this.localName !== 'select') return undefined;
    const opts = this.options;
    if ('selectedIndex' in this.__state) {
      const i = this.__state.selectedIndex;
      return i >= 0 && i < opts.length ? i : (opts.length && i !== -1 ? this._defaultIndex() : -1);
    }
    return this._defaultIndex();
  }
  set selectedIndex(i) {
    if (this.localName !== 'select') return;
    this.__state.selectedIndex = Number(i);
  }
  _defaultIndex() {
    const opts = this.options;
    const marked = opts.findIndex((o) => o.hasAttribute('selected'));
    if (marked >= 0) return marked;
    if (this.hasAttribute('multiple')) return -1;
    const first = opts.findIndex((o) => !o.disabled);
    return first >= 0 ? first : (opts.length ? 0 : -1);
  }
  _resetSelection() { delete this.__state.selectedIndex; }
  get selected() {
    if (this.localName !== 'option') return undefined;
    const select = this.closest('select');
    if (!select) return this.hasAttribute('selected');
    return select.options[select.selectedIndex] === this;
  }
  set selected(v) {
    if (this.localName !== 'option') return;
    const select = this.closest('select');
    if (!select) { this.toggleAttribute('selected', !!v); return; }
    const i = select.options.indexOf(this);
    if (v) select.selectedIndex = i;
    else if (select.selectedIndex === i) select.selectedIndex = -1;
  }
  get text() { return this.localName === 'option' ? this.textContent.replace(/\s+/g, ' ').trim() : this.textContent; }
  set text(v) { this.textContent = v; }
  get index() {
    if (this.localName !== 'option') return undefined;
    const select = this.closest('select');
    return select ? select.options.indexOf(this) : 0;
  }
  add(option, before) {
    if (this.localName !== 'select') return;
    this.insertBefore(option, typeof before === 'number' ? this.options[before] || null : before || null);
  }

  /* forms */
  get elements() {
    if (this.localName !== 'form' && this.localName !== 'fieldset') return undefined;
    return querySelectorAll(this, 'input, select, textarea, button, fieldset, output');
  }
  get method() {
    const m = (this.getAttribute('method') || 'get').toLowerCase();
    return m === 'post' || m === 'dialog' ? m : 'get';
  }
  set method(v) { this.setAttribute('method', v); }
  requestSubmit(submitter = null) {
    if (this.localName !== 'form') return;
    const event = new FakeEvent('submit', { bubbles: true, cancelable: true });
    event.submitter = submitter;
    const ok = this.dispatchEvent(event);
    if (ok && this.method === 'dialog') {
      const dialog = this.closest('dialog');
      if (dialog) dialog.close(submitter ? (submitter.value ?? '') : undefined);
    }
    if (ok) this.__submitted = (this.__submitted || 0) + 1;
  }
  submit() { this.__submitted = (this.__submitted || 0) + 1; }
  reset() {
    if (this.localName !== 'form') return;
    for (const el of this.elements) {
      delete el.__state.value;
      delete el.__state.checked;
      delete el.__state.selectedIndex;
    }
    this.dispatchEvent(new FakeEvent('reset', { bubbles: true, cancelable: true }));
  }

  /* dialog */
  get open() { return this.hasAttribute('open'); }
  set open(v) { this.toggleAttribute('open', !!v); }
  get returnValue() { return this.__state.returnValue ?? ''; }
  set returnValue(v) { this.__state.returnValue = String(v); }
  show() {
    if (this.localName !== 'dialog') return;
    if (this.hasAttribute('open')) return;
    this.setAttribute('open', '');
    this.__modal = false;
  }
  showModal() {
    if (this.localName !== 'dialog') throw new TypeError('showModal is for <dialog>');
    if (this.hasAttribute('open')) {
      if (this.__modal) return;
      throw new DOMException('The dialog is already open as a non-modal dialog.', 'InvalidStateError');
    }
    if (!this.isConnected) throw new DOMException('The dialog is not connected.', 'InvalidStateError');
    this.setAttribute('open', '');
    this.__modal = true;
    const doc = this.ownerDocument;
    doc.__topLayer.push(this);
    this.__previousFocus = doc.activeElement;
  }
  close(returnValue) {
    if (this.localName !== 'dialog' || !this.hasAttribute('open')) return;
    this.removeAttribute('open');
    if (returnValue !== undefined) this.returnValue = returnValue;
    const doc = this.ownerDocument;
    doc.__topLayer = doc.__topLayer.filter((d) => d !== this);
    this.__modal = false;
    if (doc.activeElement && this.contains(doc.activeElement)) doc.__activeElement = null;
    queueMicrotask(() => this.dispatchEvent(new FakeEvent('close')));
  }
  /** Test helper: what pressing Esc does to a modal dialog (cancel event, then close unless prevented). */
  __requestClose() {
    if (!this.hasAttribute('open')) return false;
    const ok = this.dispatchEvent(new FakeEvent('cancel', { cancelable: true }));
    if (ok) this.close();
    return ok;
  }

  /* canvas */
  getContext(kind) {
    if (this.localName !== 'canvas') return null;
    if (!this.__ctx) this.__ctx = createCanvasContext(this, kind);
    return this.__ctx;
  }
  toDataURL(type = 'image/png') { return this.localName === 'canvas' ? `data:${type};base64,RkFLRQ==` : undefined; }
  get width() { return this.localName === 'canvas' || this.localName === 'img' ? Number(this.getAttribute('width') ?? (this.localName === 'canvas' ? 300 : 0)) : undefined; }
  set width(v) { this.setAttribute('width', String(v)); }
  get height() { return this.localName === 'canvas' || this.localName === 'img' ? Number(this.getAttribute('height') ?? (this.localName === 'canvas' ? 150 : 0)) : undefined; }
  set height(v) { this.setAttribute('height', String(v)); }
}

for (const [prop, attr] of Object.entries(BOOLEAN_PROPS)) {
  if (prop === 'disabled') continue;
  Object.defineProperty(FakeElement.prototype, prop, {
    get() { return this.hasAttribute(attr); },
    set(v) { this.toggleAttribute(attr, !!v); },
    configurable: true,
  });
}
for (const [prop, attr] of Object.entries(STRING_PROPS)) {
  Object.defineProperty(FakeElement.prototype, prop, {
    get() { return this.getAttribute(attr) ?? ''; },
    set(v) { this.setAttribute(attr, v === null || v === undefined ? '' : String(v)); },
    configurable: true,
  });
}

function rangeDefault(el) {
  const min = Number(el.getAttribute('min') ?? 0);
  const max = Number(el.getAttribute('max') ?? 100);
  return String(max < min ? min : min + (max - min) / 2);
}

function clampRange(el, s) {
  const n = Number(s);
  if (!Number.isFinite(n)) return rangeDefault(el);
  const min = Number(el.getAttribute('min') ?? 0);
  const max = Number(el.getAttribute('max') ?? 100);
  return String(Math.min(max, Math.max(min, n)));
}

function radioGroup(el) {
  const name = el.getAttribute('name');
  if (!name) return [el];
  const root = el.form || el.ownerDocument;
  return querySelectorAll(root, 'input').filter((r) => r.type === 'radio' && r.getAttribute('name') === name && r.form === el.form);
}

function labelControl(label) {
  const forId = label.getAttribute('for');
  if (forId) return label.ownerDocument.getElementById(forId);
  return querySelectorAll(label, 'input, select, textarea, button, meter, output, progress')[0] || null;
}

function activationBehaviour(el) {
  const t = el.localName;
  if (t === 'label') {
    const control = labelControl(el);
    if (control && control !== el && !control.disabled) {
      if (control.localName === 'input' && /^(checkbox|radio|file|submit|button|reset)$/.test(control.type)) control.click();
      else control.focus();
    }
    return;
  }
  if (t === 'button' || (t === 'input' && /^(submit|reset|image)$/.test(el.type))) {
    const form = el.form;
    if (!form) return;
    if (el.type === 'submit' || el.type === 'image') form.requestSubmit(el);
    else if (el.type === 'reset') form.reset();
    return;
  }
  if (t === 'a') {
    const href = el.getAttribute('href');
    const win = el.ownerDocument.defaultView;
    if (href && href.startsWith('#') && win && win.location) win.location.hash = href;
    return;
  }
  if (t === 'summary') {
    const details = el.parentElement;
    if (details && details.localName === 'details' && details.firstElementChild === el) details.open = !details.open;
  }
}

function createCanvasContext(canvas, kind) {
  const calls = [];
  const record = (name) => (...args) => { calls.push([name, ...args]); };
  const ctx = {
    canvas, kind, calls, fillStyle: '#000', strokeStyle: '#000', font: '', textAlign: 'start', textBaseline: 'alphabetic', lineWidth: 1, globalAlpha: 1,
    createLinearGradient: (...args) => { calls.push(['createLinearGradient', ...args]); return { addColorStop: record('addColorStop') }; },
    measureText: (text) => ({ width: String(text).length * 8 }),
    getImageData: () => ({ data: new Uint8ClampedArray(4) }),
  };
  for (const name of ['fillRect', 'clearRect', 'strokeRect', 'fillText', 'strokeText', 'beginPath', 'closePath', 'moveTo', 'lineTo', 'arc', 'rect', 'fill', 'stroke', 'save', 'restore', 'scale', 'translate', 'rotate', 'setTransform', 'drawImage', 'putImageData', 'roundRect', 'quadraticCurveTo', 'bezierCurveTo']) {
    ctx[name] = record(name);
  }
  return ctx;
}

/* ───────────────────────────── media element ───────────────────────────── */

export class FakeMediaElement extends FakeElement {
  constructor(doc, localName = 'audio') {
    super(doc, localName);
    this.__currentTime = 0;
    this.duration = NaN;
    this.readyState = 0;
    this.networkState = 0;
    this.paused = true;
    this.ended = false;
    this.playbackRate = 1;
    this.defaultPlaybackRate = 1;
    this.preservesPitch = true;
    this.volume = 1;
    this.muted = false;
    this.error = null;
    this.seeking = false;
    this.buffered = { length: 0, start: () => 0, end: () => 0 };
    this.__playPolicy = doc?.defaultView?.__mediaPlayPolicy || 'resolve';
    this.__unlocked = false;
    this.__plays = [];
    this.__pendingPlay = null;
    this.__canPlay = null;
  }
  /** Like a browser: the attribute resolved against the page URL ('' when unset). */
  get src() {
    const value = this.getAttribute('src');
    if (value === null || value === undefined || value === '') return '';
    try {
      return new URL(value, globalThis.location?.href || 'http://127.0.0.1:8100/').href;
    } catch {
      return value;
    }
  }
  set src(v) {
    const had = this.hasAttribute('src');
    if (v === null || v === undefined || v === '') this.removeAttribute('src');
    else this.setAttribute('src', String(v));
    this._resetMedia(had);
  }
  get currentSrc() { return this.src; }
  removeAttribute(name) {
    super.removeAttribute(name);
  }
  _resetMedia(had) {
    if (!this.paused) {
      this.paused = true;
      this._fire('pause');
    }
    if (this.__pendingPlay) {
      const p = this.__pendingPlay;
      this.__pendingPlay = null;
      p.reject(new DOMException('The play() request was interrupted by a new load request.', 'AbortError'));
    }
    this.__currentTime = 0;
    this.duration = NaN;
    this.readyState = 0;
    this.ended = false;
    this.error = null;
    if (had) this._fire('emptied');
  }
  load() { this._resetMedia(this.hasAttribute('src')); }
  canPlayType(type) {
    const t = String(type).split(';')[0].trim().toLowerCase();
    const table = this.__canPlay || this.ownerDocument?.defaultView?.__canPlayTypes || DEFAULT_CAN_PLAY;
    return table[t] ?? '';
  }
  play() {
    const record = { at: Date.now(), src: this.src, gesture: !!this.ownerDocument?.defaultView?.__inGesture };
    this.__plays.push(record);
    const policy = this.__playPolicy;
    if (policy === 'NotAllowedError' || (policy === 'gesture' && !record.gesture && !this.__unlocked)) {
      return Promise.reject(new DOMException('play() failed because the user did not interact with the document first.', 'NotAllowedError'));
    }
    if (policy === 'AbortError') return Promise.reject(new DOMException('The play() request was interrupted.', 'AbortError'));
    if (policy === 'NotSupportedError') return Promise.reject(new DOMException('The element has no supported sources.', 'NotSupportedError'));
    if (policy === 'pending') {
      return new Promise((resolve, reject) => {
        this.__pendingPlay = { resolve: () => { this.__pendingPlay = null; this._startPlaying(record); resolve(); }, reject };
      });
    }
    this._startPlaying(record);
    return Promise.resolve();
  }
  _startPlaying(record) {
    if (record.gesture && !this.muted) this.__unlocked = true;
    if (this.ended) { this.ended = false; this.__currentTime = 0; }
    if (this.paused) {
      this.paused = false;
      this._fire('play');
      this._fire('playing');
    }
  }
  /** Test helper: resolve a 'pending' play(). */
  __resolvePlay() { if (this.__pendingPlay) this.__pendingPlay.resolve(); }
  pause() {
    if (this.paused) return;
    this.paused = true;
    this._fire('pause');
  }
  fastSeek(t) { this.currentTime = t; }
  __loadMetadata(duration) {
    this.duration = duration;
    this.readyState = 4;
    this._fire('loadedmetadata');
    this._fire('durationchange');
    this._fire('canplay');
  }
  __timeupdate(t) {
    this.__currentTime = t;
    this._fire('timeupdate');
  }
  __end() {
    this.__currentTime = Number.isFinite(this.duration) ? this.duration : this.__currentTime;
    this.paused = true;
    this.ended = true;
    this._fire('pause');
    this._fire('ended');
  }
  __fail(code = 4) {
    this.error = { code, message: `MEDIA_ERR ${code}` };
    this._fire('error');
  }
  /** Test helper: the system paused playback (headphones out, a call). */
  __systemPause() {
    if (this.paused) return;
    this.paused = true;
    this._fire('pause');
  }
  __setRate(rate) {
    this.playbackRate = rate;
    this._fire('ratechange');
  }
  _fire(type) { this.dispatchEvent(new FakeEvent(type)); }
}

export const DEFAULT_CAN_PLAY = { 'audio/mpeg': 'probably', 'audio/mp3': 'probably', 'audio/wav': 'probably', 'audio/wave': 'probably', 'audio/x-wav': 'probably', 'audio/flac': 'maybe', 'audio/ogg': 'maybe', 'audio/opus': 'maybe', 'audio/aac': 'maybe', 'audio/mp4': 'maybe', 'audio/webm': 'maybe' };

// Assigning currentTime is a seek: with metadata loaded it fires 'seeked' (in a microtask, like a browser);
// before that the browser keeps it as the start position and fires nothing.
Object.defineProperty(FakeMediaElement.prototype, 'currentTime', {
  get() { return this.__currentTime ?? 0; },
  set(v) {
    const n = Number(v);
    this.__currentTime = Number.isFinite(n) ? n : 0;
    this.__seeks = (this.__seeks || 0) + 1;
    if (this.readyState >= 1) queueMicrotask(() => this._fire('seeked'));
  },
  configurable: true,
});

/* ───────────────────────────── document ───────────────────────────── */

export class FakeDocument extends FakeNode {
  constructor() {
    super(null);
    this.nodeType = 9;
    this.nodeName = '#document';
    this.__activeElement = null;
    this.__topLayer = [];
    backLink(this, 'defaultView', null);
    this.hidden = false;
    this.visibilityState = 'visible';
    this.readyState = 'complete';
    this.__execCommand = { result: true, calls: [] };
    this.cookie = '';
    this.referrer = '';
    this.characterSet = 'UTF-8';
    const html = this.createElement('html');
    html.appendChild(this.createElement('head'));
    html.appendChild(this.createElement('body'));
    this.appendChild(html);
  }
  get documentElement() { return this.childNodes.find((c) => c.nodeType === 1) || null; }
  get head() { return this.documentElement?.children.find((c) => c.localName === 'head') || null; }
  get body() { return this.documentElement?.children.find((c) => c.localName === 'body') || null; }
  get activeElement() {
    const a = this.__activeElement;
    if (a && a.isConnected && this.contains(a)) return a;
    return this.body;
  }
  get title() {
    const t = this.head && querySelectorAll(this.head, 'title')[0];
    return t ? t.textContent : (this.__title ?? '');
  }
  set title(v) {
    const t = this.head && querySelectorAll(this.head, 'title')[0];
    if (t) t.textContent = v;
    else this.__title = String(v);
  }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  get location() { return this.defaultView?.location; }
  __parentTarget() { return this.defaultView || null; }
  createElement(tag) {
    const name = String(tag).toLowerCase();
    if (name === 'audio' || name === 'video') return new FakeMediaElement(this, name);
    return new FakeElement(this, name, HTML_NS);
  }
  createElementNS(ns, tag) {
    if (ns === HTML_NS || ns === null) return this.createElement(tag);
    return new FakeElement(this, tag, ns);
  }
  createTextNode(text) { return new FakeText(this, text); }
  createComment(text) { return new FakeComment(this, text); }
  createDocumentFragment() { return new FakeFragment(this); }
  createEvent() { return new FakeEvent(''); }
  getElementById(id) {
    const want = String(id);
    for (const el of walk(this)) if (el.getAttribute('id') === want) return el;
    return null;
  }
  querySelector(sel) { return querySelectorAll(this, sel)[0] || null; }
  querySelectorAll(sel) { return querySelectorAll(this, sel); }
  getElementsByTagName(tag) { return querySelectorAll(this, tag); }
  getElementsByClassName(cls) { return querySelectorAll(this, cls.split(/\s+/).filter(Boolean).map((c) => `.${c}`).join('')); }
  execCommand(command) {
    this.__execCommand.calls.push(command);
    return typeof this.__execCommand.result === 'function' ? this.__execCommand.result(command) : this.__execCommand.result;
  }
  hasFocus() { return true; }
  elementFromPoint() { return this.body; }
}

function* walk(root) {
  for (const child of root.childNodes) {
    if (child.nodeType !== 1) continue;
    yield child;
    yield* walk(child);
  }
}

/* ───────────────────────────── serialisation ───────────────────────────── */

function escapeText(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/ /g, '&nbsp;');
}
function escapeAttr(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/ /g, '&nbsp;');
}

export function serialize(node, parent) {
  if (node.nodeType === 3) {
    return parent && parent.nodeType === 1 && RAW_TEXT.has(parent.localName) ? node.data : escapeText(node.data);
  }
  if (node.nodeType === 8) return `<!--${node.data}-->`;
  if (node.nodeType !== 1) return node.childNodes.map((c) => serialize(c, node)).join('');
  const name = node.localName;
  const attrs = [...node.__attrs].map(([k, v]) => (v === '' ? ` ${k}=""` : ` ${k}="${escapeAttr(v)}"`)).join('');
  if (node.namespaceURI === HTML_NS && VOID_ELEMENTS.has(name)) return `<${name}${attrs}>`;
  return `<${name}${attrs}>${node.childNodes.map((c) => serialize(c, node)).join('')}</${name}>`;
}

/* ───────────────────────────── selectors ───────────────────────────── */

const selectorCache = new Map();

export function parseSelector(text) {
  const key = String(text);
  if (selectorCache.has(key)) return selectorCache.get(key);
  const parsed = new SelectorParser(key).parseList();
  selectorCache.set(key, parsed);
  return parsed;
}

class SelectorParser {
  constructor(text) { this.s = text; this.i = 0; }
  error(msg) { return new SyntaxError(`'${this.s}' is not a valid selector: ${msg}`); }
  peek() { return this.s[this.i]; }
  ws() { let any = false; while (this.i < this.s.length && /\s/.test(this.s[this.i])) { this.i += 1; any = true; } return any; }
  ident() {
    let out = '';
    while (this.i < this.s.length) {
      const c = this.s[this.i];
      if (c === '\\') {
        const hex = /^[0-9a-fA-F]{1,6} ?/.exec(this.s.slice(this.i + 1));
        if (hex) { out += String.fromCodePoint(parseInt(hex[0], 16)); this.i += 1 + hex[0].length; } else { out += this.s[this.i + 1] || ''; this.i += 2; }
        continue;
      }
      if (/[\w\- -￿]/.test(c)) { out += c; this.i += 1; continue; }
      break;
    }
    return out;
  }
  parseList() {
    const list = [];
    for (;;) {
      this.ws();
      list.push(this.parseComplex());
      this.ws();
      if (this.i >= this.s.length) break;
      if (this.peek() === ',') { this.i += 1; continue; }
      if (this.peek() === ')') break;
      throw this.error(`unexpected '${this.peek()}'`);
    }
    return list;
  }
  parseComplex() {
    const parts = [{ combinator: null, compound: this.parseCompound() }];
    for (;;) {
      const hadWs = this.ws();
      const c = this.peek();
      if (c === undefined || c === ',' || c === ')') break;
      let combinator = ' ';
      if (c === '>' || c === '+' || c === '~') { combinator = c; this.i += 1; this.ws(); } else if (!hadWs) throw this.error(`unexpected '${c}'`);
      parts.push({ combinator, compound: this.parseCompound() });
    }
    return parts;
  }
  parseCompound() {
    const comp = { tag: null, id: null, classes: [], attrs: [], pseudos: [] };
    let any = false;
    if (this.peek() === '*') { this.i += 1; any = true; } else if (/[\w\-\\]/.test(this.peek() || '')) { comp.tag = this.ident().toLowerCase(); any = true; }
    for (;;) {
      const c = this.peek();
      if (c === '#') { this.i += 1; comp.id = this.ident(); any = true; } else if (c === '.') { this.i += 1; comp.classes.push(this.ident()); any = true; } else if (c === '[') { this.i += 1; comp.attrs.push(this.parseAttr()); any = true; } else if (c === ':') {
        this.i += 1;
        if (this.peek() === ':') throw this.error('pseudo-elements are not supported');
        const name = this.ident().toLowerCase();
        let arg = null;
        if (this.peek() === '(') {
          this.i += 1;
          if (['not', 'is', 'where', 'has'].includes(name)) {
            arg = this.parseList();
          } else {
            const end = this.s.indexOf(')', this.i);
            arg = this.s.slice(this.i, end).trim();
            this.i = end;
          }
          if (this.peek() !== ')') throw this.error('missing )');
          this.i += 1;
        }
        if (!SUPPORTED_PSEUDOS.has(name)) throw this.error(`:${name} is not supported by the harness`);
        comp.pseudos.push({ name, arg });
        any = true;
      } else break;
    }
    if (!any) throw this.error(`expected a selector at ${this.i}`);
    return comp;
  }
  parseAttr() {
    this.ws();
    const name = this.ident().toLowerCase();
    this.ws();
    let op = null;
    let value = null;
    const c = this.peek();
    if (c === ']') { this.i += 1; return { name, op, value }; }
    if (c === '=') { op = '='; this.i += 1; } else if ('~|^$*'.includes(c) && this.s[this.i + 1] === '=') { op = `${c}=`; this.i += 2; } else throw this.error('bad attribute selector');
    this.ws();
    const q = this.peek();
    if (q === '"' || q === "'") {
      const end = this.s.indexOf(q, this.i + 1);
      value = this.s.slice(this.i + 1, end);
      this.i = end + 1;
    } else {
      value = this.ident();
    }
    this.ws();
    let ci = false;
    if (/[iIsS]/.test(this.peek() || '') && this.s[this.i + 1] !== undefined && /[\s\]]/.test(this.s[this.i + 1])) { ci = /[iI]/.test(this.peek()); this.i += 1; this.ws(); }
    if (this.peek() !== ']') throw this.error('missing ]');
    this.i += 1;
    return { name, op, value, ci };
  }
}

const SUPPORTED_PSEUDOS = new Set(['not', 'is', 'where', 'has', 'checked', 'disabled', 'enabled', 'first-child', 'last-child', 'only-child', 'first-of-type', 'last-of-type', 'empty', 'scope', 'focus', 'focus-within', 'root', 'required', 'open', 'modal', 'nth-child', 'defined']);

function attrMatches(el, a) {
  const actual = el.getAttribute(a.name);
  if (actual === null) return false;
  if (!a.op) return true;
  let v = actual;
  let want = a.value;
  if (a.ci) { v = v.toLowerCase(); want = want.toLowerCase(); }
  switch (a.op) {
    case '=': return v === want;
    case '~=': return v.split(/\s+/).includes(want);
    case '|=': return v === want || v.startsWith(`${want}-`);
    case '^=': return want !== '' && v.startsWith(want);
    case '$=': return want !== '' && v.endsWith(want);
    case '*=': return want !== '' && v.includes(want);
    default: return false;
  }
}

function nthMatches(index, arg) {
  const a = String(arg).replace(/\s+/g, '').toLowerCase();
  if (a === 'odd') return index % 2 === 1;
  if (a === 'even') return index % 2 === 0;
  const m = /^([+-]?\d*)n([+-]\d+)?$/.exec(a);
  if (!m) return index === Number(a);
  const step = m[1] === '' || m[1] === '+' ? 1 : m[1] === '-' ? -1 : Number(m[1]);
  const offset = Number(m[2] || 0);
  if (step === 0) return index === offset;
  return (index - offset) / step >= 0 && (index - offset) % step === 0;
}

function compoundMatches(el, c, scope) {
  if (el.nodeType !== 1) return false;
  if (c.tag && c.tag !== '*' && el.localName.toLowerCase() !== c.tag) return false;
  if (c.id !== null && el.getAttribute('id') !== c.id) return false;
  for (const cls of c.classes) if (!el.classList.contains(cls)) return false;
  for (const a of c.attrs) if (!attrMatches(el, a)) return false;
  for (const p of c.pseudos) {
    const parentKids = el.parentNode ? el.parentNode.childNodes.filter((n) => n.nodeType === 1) : [el];
    switch (p.name) {
      case 'not': if (p.arg.some((complex) => complexMatches(el, complex, scope))) return false; break;
      case 'is': case 'where': if (!p.arg.some((complex) => complexMatches(el, complex, scope))) return false; break;
      case 'has': if (!p.arg.some((complex) => querySelectorAllParsed(el, [complex], el).length)) return false; break;
      case 'checked': if (!(el.checked === true || (el.localName === 'option' && el.selected))) return false; break;
      case 'disabled': if (!el.disabled) return false; break;
      case 'enabled': if (!FORM_CONTROLS.has(el.localName) || el.disabled) return false; break;
      case 'first-child': if (parentKids[0] !== el) return false; break;
      case 'last-child': if (parentKids[parentKids.length - 1] !== el) return false; break;
      case 'only-child': if (parentKids.length !== 1) return false; break;
      case 'first-of-type': if (parentKids.find((n) => n.localName === el.localName) !== el) return false; break;
      case 'last-of-type': if (parentKids.filter((n) => n.localName === el.localName).pop() !== el) return false; break;
      case 'nth-child': if (!nthMatches(parentKids.indexOf(el) + 1, p.arg)) return false; break;
      case 'empty': if (el.childNodes.some((n) => n.nodeType === 1 || (n.nodeType === 3 && n.data))) return false; break;
      case 'scope': if (el !== scope) return false; break;
      case 'focus': if (el.ownerDocument.activeElement !== el) return false; break;
      case 'focus-within': if (!el.contains(el.ownerDocument.activeElement)) return false; break;
      case 'root': if (el.ownerDocument.documentElement !== el) return false; break;
      case 'required': if (!el.hasAttribute('required')) return false; break;
      case 'open': if (!el.hasAttribute('open')) return false; break;
      case 'modal': if (!el.__modal) return false; break;
      case 'defined': break;
      default: return false;
    }
  }
  return true;
}

function complexMatches(el, parts, scope) {
  return matchFrom(el, parts, parts.length - 1, scope);
}

function matchFrom(el, parts, index, scope) {
  if (!compoundMatches(el, parts[index].compound, scope)) return false;
  if (index === 0) return true;
  const comb = parts[index].combinator;
  if (comb === '>') {
    const p = el.parentElement;
    return !!p && matchFrom(p, parts, index - 1, scope);
  }
  if (comb === ' ') {
    for (let p = el.parentElement; p; p = p.parentElement) if (matchFrom(p, parts, index - 1, scope)) return true;
    return false;
  }
  if (comb === '+') {
    const prev = el.previousElementSibling;
    return !!prev && matchFrom(prev, parts, index - 1, scope);
  }
  if (comb === '~') {
    for (let s = el.previousElementSibling; s; s = s.previousElementSibling) if (matchFrom(s, parts, index - 1, scope)) return true;
    return false;
  }
  return false;
}

function matchParsed(el, list, scope) {
  return list.some((complex) => complexMatches(el, complex, scope));
}

export function matchesSelector(el, selector, scope = null) {
  return matchParsed(el, parseSelector(selector), scope);
}

function querySelectorAllParsed(root, list, scope) {
  const out = [];
  for (const el of walk(root)) if (matchParsed(el, list, scope)) out.push(el);
  return out;
}

export function querySelectorAll(root, selector) {
  return querySelectorAllParsed(root, parseSelector(selector), root.nodeType === 1 ? root : null);
}

/** Every element under `root` (depth first, document order). */
export function allElements(root) {
  return [...walk(root)];
}
