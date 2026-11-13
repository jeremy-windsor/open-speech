// installBrowserEnv(): a fake browser for Node tests. It replaces the globals the UI uses (window,
// document, location, history, storage, navigator, matchMedia, timers, fetch, WebSocket, Web Audio, …)
// with fakes on a fake clock, and restore() puts Node's own back.
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { readdirSync } from 'node:fs';
import v8 from 'node:v8';
import vm from 'node:vm';
import {
  FakeDocument, FakeElement, FakeNode, FakeText, FakeFragment, FakeMediaElement, FakeEventTarget, FakeEvent,
  FakeCustomEvent, FakeUIEvent, FakeFocusEvent, FakeMouseEvent, FakePointerEvent, FakeKeyboardEvent, FakeInputEvent,
  setErrorReporter, uncaught,
} from './dom.js';
import { parseInto, REPO_ROOT, loadHtml } from './html-parser.js';
import {
  createFetch, FakeSocket, FakeMediaSession, FakeMediaMetadata, FakeAudioContext, FakeOfflineAudioContext,
  FakeAudioWorkletNode, FakeMediaSource, FakeMediaDevices, FakeResizeObserver, FakeIntersectionObserver,
  FakeMutationObserver, FakeAudioSession,
} from './fakes.js';

export { REPO_ROOT, loadHtml };
export const STATIC_JS = resolve(REPO_ROOT, 'src', 'static', 'js');

const realSetImmediate = globalThis.setImmediate;
const RealDate = Date;

/* ───────────────────────────── fake clock ───────────────────────────── */

// A timer that re-arms itself without the clock moving (or allocates on every round) is a runaway loop in
// the code under test. Advancing used to stop silently after 100 000 timers, which hid the loop and let a
// broken view allocate tens of gigabytes first; now advancing fails the test.
const RUNAWAY_TIMERS = 100000;
const RUNAWAY_BYTES = 512 * 1024 * 1024;

function checkRunaway(fired, what) {
  if (fired >= RUNAWAY_TIMERS) {
    throw new Error(`FakeClock.${what}: ${RUNAWAY_TIMERS} timers fired without reaching the target time; a timer keeps re-arming itself`);
  }
  if (fired % 256 === 255 && inUse() > RUNAWAY_BYTES) {
    collectGarbage(); // garbage from earlier tests is not a runaway
    if (inUse() > RUNAWAY_BYTES) {
      throw new Error(`FakeClock.${what}: over 512 MB in use after ${fired + 1} timers; a timer loop keeps allocating`);
    }
  }
}

function inUse() {
  const { heapUsed, arrayBuffers } = process.memoryUsage();
  return heapUsed + arrayBuffers;
}

let gc = null;
function collectGarbage() {
  if (!gc) {
    try {
      v8.setFlagsFromString('--expose-gc');
      gc = vm.runInNewContext('gc');
    } catch {
      gc = () => {};
    }
  }
  try {
    gc({ type: 'major', execution: 'sync', flavor: 'last-resort' }); // a plain gc() can leave array buffers
  } catch {
    gc();
  }
}

export class FakeClock {
  constructor(start = 1760000000000) {
    this._now = start;
    this._timers = new Map();
    this._seq = 0;
  }
  now() { return this._now; }
  setTimeout(fn, ms = 0, ...args) {
    const id = (this._seq += 1);
    this._timers.set(id, { id, at: this._now + Math.max(0, Number(ms) || 0), fn, args, every: 0 });
    return id;
  }
  setInterval(fn, ms = 0, ...args) {
    const id = (this._seq += 1);
    const every = Math.max(1, Number(ms) || 0);
    this._timers.set(id, { id, at: this._now + every, fn, args, every });
    return id;
  }
  clearTimeout(id) { if (id !== undefined && id !== null) this._timers.delete(typeof id === 'object' ? id.id : id); }
  clearInterval(id) { this.clearTimeout(id); }
  /** Timers still waiting. */
  pending() { return [...this._timers.values()].map((t) => ({ id: t.id, inMs: t.at - this._now, every: t.every })); }
  _next(limit) {
    let best = null;
    for (const t of this._timers.values()) {
      if (t.at > limit) continue;
      if (!best || t.at < best.at || (t.at === best.at && t.id < best.id)) best = t;
    }
    return best;
  }
  _run(t) {
    this._now = Math.max(this._now, t.at);
    if (t.every) t.at += t.every;
    else this._timers.delete(t.id);
    try {
      if (typeof t.fn === 'function') t.fn(...t.args);
    } catch (error) {
      uncaught.push(error);
    }
  }
  /** Advance synchronously, firing due timers in order (no microtasks run in between). */
  tick(ms = 0) {
    const target = this._now + Math.max(0, ms);
    for (let fired = 0; ; fired += 1) {
      const t = this._next(target);
      if (!t) break;
      checkRunaway(fired, 'tick');
      this._run(t);
    }
    this._now = target;
  }
  /** Advance, letting promises settle after each timer (use this for async code). */
  async tickAsync(ms = 0) {
    const target = this._now + Math.max(0, ms);
    await flushMicrotasks();
    for (let fired = 0; ; fired += 1) {
      const t = this._next(target);
      if (!t) break;
      checkRunaway(fired, 'tickAsync');
      this._run(t);
      await flushMicrotasks();
    }
    this._now = target;
    await flushMicrotasks();
  }
  /** Run every pending timer (intervals once), letting promises settle. */
  async runAll(limit = 1000) {
    for (let i = 0; i < limit; i += 1) {
      const t = this._next(Infinity);
      if (!t) break;
      if (t.every) { this._timers.delete(t.id); }
      this._run({ ...t, every: 0 });
      await flushMicrotasks();
    }
  }
}

/** Let every queued promise reaction run (several rounds of the real event loop). */
export async function flushMicrotasks(rounds = 8) {
  for (let i = 0; i < rounds; i += 1) await new Promise((r) => realSetImmediate(r));
}

/* ───────────────────────────── location and history ───────────────────────────── */

class FakeLocation {
  constructor(url, win) { this._url = new URL(url); this._win = win; }
  get href() { return this._url.href; }
  set href(v) { this._navigate(v); }
  get origin() { return this._url.origin; }
  get protocol() { return this._url.protocol; }
  get host() { return this._url.host; }
  get hostname() { return this._url.hostname; }
  get port() { return this._url.port; }
  get pathname() { return this._url.pathname; }
  get search() { return this._url.search; }
  get hash() { return this._url.hash; }
  set hash(v) {
    const next = new URL(this._url.href);
    next.hash = String(v).startsWith('#') ? String(v).slice(1) : String(v);
    this._navigate(next.href);
  }
  assign(v) { this._navigate(v); }
  replace(v) { this._navigate(v, true); }
  reload() { this._win.__reloads = (this._win.__reloads || 0) + 1; }
  toString() { return this.href; }
  _set(url) { this._url = new URL(url, this._url.href); }
  _navigate(v, replace = false) {
    const next = new URL(String(v), this._url.href);
    const sameDoc = next.origin === this._url.origin && next.pathname === this._url.pathname && next.search === this._url.search;
    if (!sameDoc) { this._win.__navigations.push(next.href); return; }
    const oldHash = this._url.hash;
    const oldUrl = this._url.href;
    if (replace) this._win.history._replace(next.href);
    else if (next.href !== this._url.href) this._win.history._push(next.href);
    if (next.hash !== oldHash) {
      const event = new FakeEvent('hashchange');
      event.oldURL = oldUrl;
      event.newURL = next.href;
      this._win.dispatchEvent(event);
    }
  }
}

class FakeHistory {
  constructor(win, url) { this._win = win; this._entries = [{ url, state: null }]; this._index = 0; }
  get length() { return this._entries.length; }
  get state() { return this._entries[this._index].state; }
  get scrollRestoration() { return 'auto'; }
  set scrollRestoration(_) { /* ignored */ }
  pushState(state, _title, url) {
    const target = url === undefined || url === null ? this._win.location.href : new URL(String(url), this._win.location.href).href;
    this._entries = this._entries.slice(0, this._index + 1);
    this._entries.push({ url: target, state: state ?? null });
    this._index += 1;
    this._win.location._set(target);
  }
  replaceState(state, _title, url) {
    const target = url === undefined || url === null ? this._win.location.href : new URL(String(url), this._win.location.href).href;
    this._entries[this._index] = { url: target, state: state ?? null };
    this._win.location._set(target);
  }
  _push(url) { this.pushState(null, '', url); }
  _replace(url) { this.replaceState(null, '', url); }
  back() { this.go(-1); }
  forward() { this.go(1); }
  go(delta = 0) {
    const next = this._index + delta;
    if (!delta || next < 0 || next >= this._entries.length) return;
    queueMicrotask(() => this._traverse(next));
  }
  _traverse(index) {
    const oldUrl = this._win.location.href;
    const oldHash = this._win.location.hash;
    this._index = index;
    const entry = this._entries[index];
    this._win.location._set(entry.url);
    const pop = new FakeEvent('popstate');
    pop.state = entry.state;
    this._win.dispatchEvent(pop);
    if (this._win.location.hash !== oldHash) {
      const event = new FakeEvent('hashchange');
      event.oldURL = oldUrl;
      event.newURL = entry.url;
      this._win.dispatchEvent(event);
    }
  }
  /** Test helper: the URLs in the history stack and the current index. */
  get __entries() { return { urls: this._entries.map((e) => e.url), index: this._index }; }
}

/* ───────────────────────────── storage ───────────────────────────── */

class MemoryStorage {
  constructor(seed = {}, mode = 'memory') {
    this._map = new Map(Object.entries(seed).map(([k, v]) => [k, String(v)]));
    this._mode = mode;
  }
  _check(write) {
    if (this._mode === 'throwing') throw new DOMException('The operation is insecure.', 'SecurityError');
    if (write && this._mode === 'quota') throw new DOMException('The quota has been exceeded.', 'QuotaExceededError');
  }
  get length() { this._check(false); return this._map.size; }
  key(i) { this._check(false); return [...this._map.keys()][i] ?? null; }
  getItem(k) { this._check(false); return this._map.has(String(k)) ? this._map.get(String(k)) : null; }
  setItem(k, v) { this._check(true); this._map.set(String(k), String(v)); }
  removeItem(k) { this._check(false); this._map.delete(String(k)); }
  clear() { this._check(false); this._map.clear(); }
  /** Test helper: the raw contents. */
  get __data() { return Object.fromEntries(this._map); }
}

/* ───────────────────────────── matchMedia ───────────────────────────── */

function normaliseQuery(q) {
  return String(q).trim().replace(/^\(|\)$/g, '').replace(/\s*:\s*/g, ': ').trim();
}

function createMatchMedia(env) {
  const lists = [];
  const evaluate = (query) => String(query).split(',').some((alt) => alt.split(/\)\s+and\s+\(|\s+and\s+/).every((partRaw) => {
    let part = partRaw.trim();
    let negate = false;
    if (/^not\s+/i.test(part)) { negate = true; part = part.replace(/^not\s+/i, ''); }
    const key = normaliseQuery(part);
    let value;
    if (Object.prototype.hasOwnProperty.call(env.media, key)) value = !!env.media[key];
    else {
      const m = /^(min|max)-width: (\d+)px$/.exec(key);
      if (m) value = m[1] === 'min' ? env.viewport.width >= Number(m[2]) : env.viewport.width <= Number(m[2]);
      else if (key === 'all' || key === 'screen') value = true;
      else value = false;
    }
    return negate ? !value : value;
  }));
  function matchMedia(query) {
    const target = new FakeEventTarget();
    const mql = {
      media: String(query),
      get matches() { return evaluate(query); },
      onchange: null,
      addEventListener: (t, fn, o) => target.addEventListener(t, fn, o),
      removeEventListener: (t, fn, o) => target.removeEventListener(t, fn, o),
      addListener: (fn) => target.addEventListener('change', fn),
      removeListener: (fn) => target.removeEventListener('change', fn),
      dispatchEvent: (e) => target.dispatchEvent(e),
      __last: evaluate(query),
      __target: target,
    };
    lists.push(mql);
    return mql;
  }
  matchMedia.__lists = lists;
  matchMedia.__changed = () => {
    for (const mql of lists) {
      const now = evaluate(mql.media);
      if (now === mql.__last) continue;
      mql.__last = now;
      const event = new FakeEvent('change');
      event.matches = now;
      event.media = mql.media;
      if (typeof mql.onchange === 'function') mql.onchange(event);
      mql.__target.dispatchEvent(event);
    }
  };
  return matchMedia;
}

/* ───────────────────────────── window ───────────────────────────── */

class FakeWindow extends FakeEventTarget {}

const GLOBALS = [
  'window', 'self', 'document', 'location', 'history', 'localStorage', 'sessionStorage', 'navigator', 'matchMedia',
  'requestAnimationFrame', 'cancelAnimationFrame', 'requestIdleCallback', 'cancelIdleCallback', 'fetch', 'WebSocket',
  'AudioContext', 'webkitAudioContext', 'OfflineAudioContext', 'AudioWorkletNode', 'MediaSource', 'ManagedMediaSource',
  'MediaMetadata', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date', 'performance',
  'isSecureContext', 'Event', 'CustomEvent', 'UIEvent', 'FocusEvent', 'MouseEvent', 'PointerEvent', 'KeyboardEvent',
  'InputEvent', 'Element', 'HTMLElement', 'SVGElement', 'Node', 'Text', 'DocumentFragment', 'Document',
  'HTMLMediaElement', 'HTMLAudioElement', 'HTMLDialogElement', 'HTMLInputElement', 'HTMLButtonElement',
  'HTMLSelectElement', 'HTMLTextAreaElement', 'HTMLCanvasElement', 'ResizeObserver', 'IntersectionObserver',
  'MutationObserver', 'getComputedStyle', 'CSS', 'innerWidth', 'innerHeight', 'devicePixelRatio', 'scrollTo',
  'scrollBy', 'scrollX', 'scrollY', 'alert', 'confirm', 'prompt', 'open', 'addEventListener',
  'removeEventListener', 'dispatchEvent',
];

let installed = null;

/**
 * Install the fake browser.
 * @param {object} [options]
 * @param {string} [options.url]
 * @param {'memory'|'throwing'|'quota'} [options.storage] throwing: every access throws SecurityError
 * @param {Record<string, string>} [options.seed] localStorage contents
 * @param {Record<string, boolean>} [options.media] matchMedia answers by feature ('prefers-color-scheme: dark')
 * @param {'ok'|'reject'|'absent'} [options.clipboard]
 * @param {boolean} [options.secure] isSecureContext
 * @param {object} [options.routes] createFetch routes
 * @param {FakeDocument} [options.document] e.g. loadHtml('src/static/index.html')
 * @param {string} [options.html] markup mounted into document.body
 * @param {string} [options.platform] navigator.platform ('MacIntel' makes Mod = Cmd)
 * @param {boolean} [options.mediaSource] false: no MediaSource (iPhone Safari)
 * @param {'resolve'|'NotAllowedError'|'AbortError'|'pending'|'gesture'} [options.playPolicy] for every media
 *   element: the ones created later, and the ones already in a supplied `document` (such as the shell's
 *   #app-audio, parsed before the env existed)
 * @param {{sampleRate?: number, state?: string, worklet?: boolean|'reject'}} [options.audio]
 * @param {object} [options.mediaDevices] FakeMediaDevices options; null removes navigator.mediaDevices
 * @param {boolean|string} [options.audioSession] Safari 17+'s navigator.audioSession: true (type 'auto') or the
 *   starting type; absent by default (every other browser). The fake is env.audioSession (`history` lists the
 *   types set).
 * @returns {object} env
 */
export function installBrowserEnv({
  url = 'http://127.0.0.1:8100/web',
  storage = 'memory',
  seed = {},
  media = { 'prefers-color-scheme: dark': false, 'prefers-reduced-motion: reduce': false, 'min-width: 960px': false },
  clipboard = 'ok',
  secure = true,
  routes = {},
  document: docOption = null,
  html = '',
  platform = 'Win32',
  mediaSource = true,
  playPolicy = 'resolve',
  audio = {},
  mediaDevices: devicesOption = {},
  viewport = { width: 390, height: 844 },
  now = 1760000000000,
  permissions = {},
  audioSession: audioSessionOption = false,
} = {}) {
  if (installed) installed.restore();
  const saved = new Map();
  for (const name of GLOBALS) saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  const urlStatics = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
  uncaught.length = 0;

  const clock = new FakeClock(now);
  const win = new FakeWindow();
  win.__navigations = [];
  win.__mediaPlayPolicy = playPolicy;
  win.__inGesture = false;
  const doc = docOption || new FakeDocument();
  doc.defaultView = win;
  // A supplied document was parsed before this window existed, so its media elements (the shell's
  // #app-audio) took the default policy: give them the env's, like the elements created from now on.
  if (docOption) for (const el of doc.querySelectorAll('audio, video')) el.__playPolicy = playPolicy;
  const env = {
    clock, window: win, document: doc, media: { ...media }, viewport: { ...viewport }, errors: uncaught,
    objectUrls: new Map(), createdUrls: [], revokedUrls: [], clipboard: { written: [], mode: clipboard },
    permissions: { microphone: 'prompt', ...permissions }, wakeLocks: [],
  };
  setErrorReporter((error) => { uncaught.push(error); });

  const location = new FakeLocation(url, win);
  const history = new FakeHistory(win, location.href);
  const local = new MemoryStorage(seed, storage);
  const session = new MemoryStorage({}, storage);
  const matchMedia = createMatchMedia(env);
  const fetch = createFetch(routes, { origin: location.origin });
  const mediaSession = new FakeMediaSession();
  const audioSession = audioSessionOption ? new FakeAudioSession(typeof audioSessionOption === 'string' ? audioSessionOption : 'auto') : null;
  const devices = devicesOption === null ? null : new FakeMediaDevices(devicesOption || {});

  let rafQueue = [];
  let rafSeq = 0;
  const requestAnimationFrame = (cb) => { rafSeq += 1; rafQueue.push({ id: rafSeq, cb }); return rafSeq; };
  const cancelAnimationFrame = (id) => { rafQueue = rafQueue.filter((r) => r.id !== id); };
  env.raf = {
    flush(ts = clock.now()) {
      const run = rafQueue;
      rafQueue = [];
      for (const r of run) {
        try { r.cb(ts); } catch (error) { uncaught.push(error); }
      }
      return run.length;
    },
    get pending() { return rafQueue.length; },
  };

  class FakeDate extends RealDate {
    constructor(...args) {
      if (args.length) super(...args);
      else super(clock.now());
    }
    static now() { return clock.now(); }
  }

  const clipboardApi = clipboard === 'absent' ? undefined : {
    async writeText(textValue) {
      if (clipboard === 'reject') throw new DOMException('Write permission denied.', 'NotAllowedError');
      env.clipboard.written.push(String(textValue));
    },
    async readText() { return env.clipboard.written[env.clipboard.written.length - 1] || ''; },
  };
  doc.__execCommand = {
    calls: [],
    result: (command) => {
      if (command !== 'copy') return false;
      if (env.execCommandResult === false) return false;
      const active = doc.activeElement;
      if (active && 'value' in active && active.localName === 'textarea') env.clipboard.written.push(active.value);
      return true;
    },
  };

  const navigator = {
    userAgent: platform === 'MacIntel' ? 'Mozilla/5.0 (Macintosh)' : 'Mozilla/5.0 (Windows NT 10.0)',
    platform,
    language: 'en-US',
    languages: ['en-US', 'en'],
    onLine: true,
    maxTouchPoints: 0,
    clipboard: clipboardApi,
    mediaDevices: devices || undefined,
    mediaSession,
    ...(audioSession ? { audioSession } : {}),
    permissions: { query: async ({ name }) => ({ state: env.permissions[name] || 'prompt', onchange: null, addEventListener() {} }) },
    wakeLock: {
      async request(type) {
        const sentinel = new FakeEventTarget();
        sentinel.type = type;
        sentinel.released = false;
        sentinel.release = async () => { sentinel.released = true; sentinel.dispatchEvent(new FakeEvent('release')); };
        env.wakeLocks.push(sentinel);
        return sentinel;
      },
    },
    sendBeacon: () => true,
    vibrate: () => true,
  };

  FakeSocket.instances = [];
  FakeSocket.onCreate = null;
  FakeAudioContext.instances = [];
  FakeAudioContext.defaults = {
    sampleRate: audio.sampleRate || 48000, state: audio.state || 'running',
    worklet: audio.worklet === undefined ? true : audio.worklet, decodeResult: audio.decodeResult || null,
  };
  FakeOfflineAudioContext.instances = [];
  FakeMediaSource.instances = [];
  FakeResizeObserver.instances = [];
  FakeIntersectionObserver.instances = [];

  const performance = {
    now: () => clock.now() - now,
    timeOrigin: now,
    mark() {}, measure() {}, getEntriesByName: () => [], getEntriesByType: () => [], clearMarks() {}, clearMeasures() {},
  };

  const values = {
    window: win, self: win, document: doc, location, history, navigator, matchMedia,
    requestAnimationFrame, cancelAnimationFrame,
    requestIdleCallback: (cb) => clock.setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 50 }), 1),
    cancelIdleCallback: (id) => clock.clearTimeout(id),
    fetch, WebSocket: FakeSocket, AudioContext: FakeAudioContext, webkitAudioContext: undefined,
    OfflineAudioContext: FakeOfflineAudioContext, AudioWorkletNode: FakeAudioWorkletNode,
    MediaSource: mediaSource ? FakeMediaSource : undefined, ManagedMediaSource: undefined, MediaMetadata: FakeMediaMetadata,
    setTimeout: (fn, ms, ...args) => clock.setTimeout(fn, ms, ...args),
    clearTimeout: (id) => clock.clearTimeout(id),
    setInterval: (fn, ms, ...args) => clock.setInterval(fn, ms, ...args),
    clearInterval: (id) => clock.clearInterval(id),
    Date: FakeDate, performance, isSecureContext: !!secure,
    Event: FakeEvent, CustomEvent: FakeCustomEvent, UIEvent: FakeUIEvent, FocusEvent: FakeFocusEvent,
    MouseEvent: FakeMouseEvent, PointerEvent: FakePointerEvent, KeyboardEvent: FakeKeyboardEvent, InputEvent: FakeInputEvent,
    Element: FakeElement, HTMLElement: FakeElement, SVGElement: FakeElement, Node: FakeNode, Text: FakeText,
    DocumentFragment: FakeFragment, Document: FakeDocument, HTMLMediaElement: FakeMediaElement,
    HTMLAudioElement: FakeMediaElement, HTMLDialogElement: FakeElement, HTMLInputElement: FakeElement,
    HTMLButtonElement: FakeElement, HTMLSelectElement: FakeElement, HTMLTextAreaElement: FakeElement,
    HTMLCanvasElement: FakeElement,
    ResizeObserver: FakeResizeObserver, IntersectionObserver: FakeIntersectionObserver, MutationObserver: FakeMutationObserver,
    getComputedStyle: (el) => ({
      getPropertyValue: (p) => (el && el.style ? el.style.getPropertyValue(p) : '') || '',
      display: el?.hidden ? 'none' : 'block',
      visibility: 'visible',
    }),
    CSS: { supports: () => true, escape: (s) => String(s).replace(/[^\w-]/g, (c) => `\\${c}`) },
    innerWidth: viewport.width, innerHeight: viewport.height, devicePixelRatio: 2,
    scrollTo: (x, y) => { win.scrollX = typeof x === 'object' ? (x.left ?? win.scrollX) : x; win.scrollY = typeof x === 'object' ? (x.top ?? win.scrollY) : y; win.__scrolls.push({ x, y }); },
    scrollBy: (x, y) => { win.scrollY += typeof x === 'object' ? (x.top ?? 0) : y; },
    scrollX: 0, scrollY: 0,
    alert: () => { throw new Error('window.alert is not allowed: use toast/dialogs'); },
    confirm: () => { throw new Error('window.confirm is not allowed: use dialogs.confirm'); },
    prompt: () => { throw new Error('window.prompt is not allowed: use dialogs.prompt'); },
    open: (...args) => { win.__opened.push(args); return null; },
    addEventListener: (...args) => win.addEventListener(...args),
    removeEventListener: (...args) => win.removeEventListener(...args),
    dispatchEvent: (e) => win.dispatchEvent(e),
  };
  win.__scrolls = [];
  win.__opened = [];
  for (const [name, value] of Object.entries(values)) {
    if (!['addEventListener', 'removeEventListener', 'dispatchEvent'].includes(name)) win[name] = value;
  }
  Object.defineProperty(win, 'localStorage', { get: () => local.__accessor(), configurable: true });
  Object.defineProperty(win, 'sessionStorage', { get: () => session.__accessor(), configurable: true });
  local.__accessor = () => { if (storage === 'throwing') throw new DOMException('Access is denied for this document.', 'SecurityError'); return local; };
  session.__accessor = () => { if (storage === 'throwing') throw new DOMException('Access is denied for this document.', 'SecurityError'); return session; };
  win.window = win;
  win.top = win;
  win.parent = win;

  for (const [name, value] of Object.entries(values)) {
    Object.defineProperty(globalThis, name, { value, configurable: true, writable: true, enumerable: false });
  }
  Object.defineProperty(globalThis, 'localStorage', { get: () => local.__accessor(), configurable: true, enumerable: false });
  Object.defineProperty(globalThis, 'sessionStorage', { get: () => session.__accessor(), configurable: true, enumerable: false });

  URL.createObjectURL = (blob) => {
    const id = `blob:${location.origin}/${env.createdUrls.length + 1}-${Math.random().toString(16).slice(2, 10)}`;
    env.objectUrls.set(id, blob);
    env.createdUrls.push(id);
    return id;
  };
  URL.revokeObjectURL = (id) => {
    env.revokedUrls.push(id);
    env.objectUrls.delete(id);
  };

  if (html) parseInto(doc.body, html);

  Object.assign(env, {
    location, history, localStorage: local, sessionStorage: session, matchMedia, fetch, mediaSession, audioSession,
    mediaDevices: devices, navigator, performance,
    get sockets() { return FakeSocket.instances; },
    get audioContexts() { return FakeAudioContext.instances; },
    flush: () => flushMicrotasks(),
    async settle() {
      await flushMicrotasks();
      env.raf.flush();
      await flushMicrotasks();
    },
    setHash(hash) {
      location.hash = hash;
    },
    /**
     * The tab goes to the background or comes back: a tab switch, the app switcher or a locked screen. A browser
     * fires visibilitychange only; pagehide comes with leaving the page (see pageHide()).
     */
    setHidden(hidden) {
      doc.hidden = !!hidden;
      doc.visibilityState = hidden ? 'hidden' : 'visible';
      doc.dispatchEvent(new FakeEvent('visibilitychange', { bubbles: true }));
    },
    /**
     * The page is left (a navigation, a reload, a closed tab): hidden first, then pagehide. persisted: true is a
     * page kept in the back/forward cache, which pageShow({persisted: true}) brings back.
     */
    pageHide({ persisted = false } = {}) {
      if (!doc.hidden) env.setHidden(true);
      const event = new FakeEvent('pagehide');
      event.persisted = !!persisted;
      win.dispatchEvent(event);
    },
    pageShow({ persisted = false } = {}) {
      const event = new FakeEvent('pageshow');
      event.persisted = !!persisted;
      win.dispatchEvent(event);
      if (doc.hidden) env.setHidden(false);
    },
    setOnline(online) {
      navigator.onLine = !!online;
      win.dispatchEvent(new FakeEvent(online ? 'online' : 'offline'));
    },
    setMedia(query, value) {
      env.media[normaliseQuery(query)] = !!value;
      matchMedia.__changed();
    },
    setViewport(width, height = viewport.height) {
      env.viewport = { width, height };
      win.innerWidth = width;
      win.innerHeight = height;
      globalThis.innerWidth = width;
      globalThis.innerHeight = height;
      matchMedia.__changed();
      win.dispatchEvent(new FakeEvent('resize'));
    },
    /** Run `fn` as a user gesture (media play() then counts as user-initiated). */
    withGesture(fn) {
      const was = win.__inGesture;
      win.__inGesture = true;
      try { return fn(); } finally { win.__inGesture = was; }
    },
    /** A user click (inside a gesture). */
    click(el) {
      if (!el) throw new Error('click(): no element');
      return env.withGesture(() => el.click());
    },
    /** Type into a field: sets the value and fires input (and change with {change: true}). */
    input(el, value, { change = false } = {}) {
      el.value = value;
      el.dispatchEvent(new FakeInputEvent('input', { bubbles: true, inputType: 'insertText', data: String(value) }));
      if (change) el.dispatchEvent(new FakeEvent('change', { bubbles: true }));
    },
    /** Change a select, checkbox or field and fire input + change. */
    change(el, value) {
      if (value !== undefined) {
        if (el.localName === 'input' && (el.type === 'checkbox' || el.type === 'radio')) el.checked = !!value;
        else el.value = value;
      }
      el.dispatchEvent(new FakeInputEvent('input', { bubbles: true }));
      el.dispatchEvent(new FakeEvent('change', { bubbles: true }));
    },
    /**
     * Press a key at `target` (default: the focused element): keydown, the browser's default action unless
     * prevented (Esc closes the top modal dialog, Enter activates buttons/links and submits forms, Space
     * activates buttons and toggles checkboxes), then keyup.
     */
    key(key, init = {}, target = doc.activeElement) {
      const t = target || doc.body;
      const down = new FakeKeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
      env.withGesture(() => t.dispatchEvent(down));
      if (!down.defaultPrevented) {
        env.withGesture(() => defaultKeyAction(doc, t, key, init));
      }
      t.dispatchEvent(new FakeKeyboardEvent('keyup', { key, bubbles: true, cancelable: true, ...init }));
      return down;
    },
    /** Mount markup into document.body (or `root`); returns the first element. */
    mount(markup, root = doc.body) {
      const before = root.children.length;
      parseInto(root, markup);
      return root.children[before] || null;
    },
    restore() {
      for (const [name, desc] of saved) {
        if (desc) Object.defineProperty(globalThis, name, desc);
        else delete globalThis[name];
      }
      URL.createObjectURL = urlStatics.create;
      URL.revokeObjectURL = urlStatics.revoke;
      FakeSocket.onCreate = null;
      setErrorReporter((error) => { uncaught.push(error); });
      if (installed === env) installed = null;
    },
  });
  installed = env;
  return env;
}

function defaultKeyAction(doc, target, key, init) {
  const tag = target.localName;
  const type = tag === 'input' ? target.type : '';
  if (key === 'Escape') {
    const top = doc.__topLayer[doc.__topLayer.length - 1];
    if (top) top.__requestClose();
    return;
  }
  if (key === 'Enter') {
    if (tag === 'button' || tag === 'summary' || (tag === 'a' && target.hasAttribute('href'))) { target.click(); return; }
    if (tag === 'input' && !/^(checkbox|radio|button|submit|reset|file|range|color)$/.test(type) && target.form) {
      const form = target.form;
      const submitter = form.querySelector('button:not([type="button"]):not([type="reset"]), input[type="submit"]');
      form.requestSubmit(submitter || null);
    }
    return;
  }
  if (key === ' ' && !init.ctrlKey && !init.metaKey) {
    if (tag === 'button' || tag === 'summary' || (tag === 'input' && /^(checkbox|radio|button|submit|reset)$/.test(type))) target.click();
  }
}

/** The core modules, in the order their test hooks are reset (the store last). */
const CORE_ORDER = ['api', 'auth', 'a11y', 'toast', 'dialogs', 'theme', 'status', 'models', 'voices', 'audio', 'mic', 'router', 'shortcuts', 'store'];

/** The shell and component modules reset by resetAll(), in this order (then the rest of shell/ and
 *  components/ alphabetically, then the core). */
const UI_ORDER = [
  'shell/index.js', 'shell/view-host.js', 'shell/nav.js', 'shell/more-sheet.js', 'shell/status-chip.js',
  'shell/banners.js', 'shell/miniplayer.js', 'shell/chrome.js', 'shell/shortcuts-sheet.js',
  'components/voice-picker.js', 'components/tooltip.js', 'components/menu.js',
];

/**
 * Import a module from src/static/js (e.g. 'core/api.js' or 'views/speak/takes.js'), reset every stateful
 * shell, component and core module (resetAll), then the imported module's own __resetForTests() when it has
 * one, so each test starts from a clean state with the current env.
 */
export async function importFresh(relPath) {
  const mod = await import(pathToFileURL(resolve(STATIC_JS, relPath)).href);
  await resetAll();
  if (typeof mod.__resetForTests === 'function') mod.__resetForTests();
  return mod;
}

/** Reset every stateful shell and component module (their singletons outlive a test), then the core. */
export async function resetAll() {
  const listed = [];
  for (const dir of ['shell', 'components']) {
    let files = [];
    try { files = readdirSync(resolve(STATIC_JS, dir)).filter((f) => f.endsWith('.js')); } catch { files = []; }
    for (const f of files.sort()) listed.push(`${dir}/${f}`);
  }
  const order = [...UI_ORDER.filter((rel) => listed.includes(rel)), ...listed.filter((rel) => !UI_ORDER.includes(rel))];
  for (const rel of order) {
    const m = await import(pathToFileURL(resolve(STATIC_JS, rel)).href);
    if (typeof m.__resetForTests === 'function') m.__resetForTests();
  }
  await resetCore();
}

/** Reset every stateful core module (store last). */
export async function resetCore() {
  const dir = resolve(STATIC_JS, 'core');
  const files = readdirSync(dir).filter((f) => f.endsWith('.js')).map((f) => f.replace(/\.js$/, ''));
  const order = [...CORE_ORDER.filter((n) => files.includes(n) && n !== 'store'), ...files.filter((n) => !CORE_ORDER.includes(n)), 'store'];
  for (const name of order) {
    if (!files.includes(name)) continue;
    const m = await import(pathToFileURL(resolve(dir, `${name}.js`)).href);
    if (typeof m.__resetForTests === 'function') m.__resetForTests();
  }
}
