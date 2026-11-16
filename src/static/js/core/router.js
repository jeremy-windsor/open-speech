/**
 * Hash routes: '#home', '#reader', '#reader/<docId>', '#speak', '#speak/live', '#voices/recordings', …
 * route = {view, sub, id, path, shot}; `path` is the canonical hash without '#', ids encodeURIComponent-ed.
 * @module core/router
 */
import { app } from './store.js';
import * as prefs from './prefs.js';

// 'talk': the Talk view, a conversation with the AI assistant.
export const VIEWS = Object.freeze(['home', 'reader', 'speak', 'transcribe', 'talk', 'voices', 'studio', 'history', 'models', 'settings']);
export const SUBS = Object.freeze({
  speak: Object.freeze(['', 'live']),
  // '#voices' is one list of named voices and presets: an old '#voices/named' or
  // '#voices/presets' link is an unknown sub, so it opens that list too.
  voices: Object.freeze(['', 'recordings', 'clone', 'design']),
  studio: Object.freeze(['conversations', 'composer']),
  models: Object.freeze(['tts', 'stt']),
});
const ALIASES = { voicelab: 'voices/recordings', tts: 'speak', stt: 'transcribe' };
const DEFAULT_PATH = 'home';

let onRouteCb = null;
let cur = null;
let latest = null;   // the newest route asked for
let latestRun = Promise.resolve(null);   // the promise of the newest application
let listening = null;

function decode(segment) {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

function buildPath(view, sub, id) {
  const parts = [view];
  if (sub) parts.push(sub);
  if (id) parts.push(encodeURIComponent(id));
  return parts.join('/');
}

/**
 * PURE: parse a hash into a route, or null for an unknown view. Legacy aliases: #voicelab ->
 * voices/recordings, #more -> lastRoute (else models), #tts -> speak, #stt -> transcribe. A bare view gets
 * its default sub; an unknown sub falls back to the default sub. '?shot=<state>' becomes route.shot.
 * @param {string} hash
 * @param {{lastRoute?: string}} [options]
 * @returns {{view: string, sub: string, id: string, path: string, shot: string}|null}
 */
export function parseHash(hash, { lastRoute = '' } = {}) {
  let text = String(hash ?? '').trim().replace(/^#/, '');
  let shot = '';
  const q = text.indexOf('?');
  if (q >= 0) {
    try { shot = new URLSearchParams(text.slice(q + 1)).get('shot') || ''; } catch { shot = ''; }
    text = text.slice(0, q);
  }
  const segments = text.split('/').filter((s) => s !== '');
  if (!segments.length) return null;
  const head = decode(segments[0]).toLowerCase();
  if (head === 'more') {
    const remembered = lastRoute && !/^#?more\b/i.test(lastRoute) ? parseHash(lastRoute) : null;
    const target = remembered || parseHash('models');
    return { ...target, shot };
  }
  if (ALIASES[head]) {
    const target = parseHash(ALIASES[head]);
    return { ...target, shot };
  }
  if (!VIEWS.includes(head)) return null;
  const rest = segments.slice(1).map(decode);
  let sub = '';
  let id = '';
  if (head === 'reader') {
    id = rest[0] || '';
  } else if (head === 'speak') {
    sub = rest[0] === 'live' ? 'live' : '';
  } else if (SUBS[head]) {
    sub = SUBS[head].includes(rest[0]) ? rest[0] : SUBS[head][0];
    if (head === 'studio' && sub === 'conversations' && rest[0] === 'conversations' && rest[1]) id = rest[1];
  }
  return { view: head, sub, id, path: buildPath(head, sub, id), shot };
}

/**
 * PURE: the hash for a route, e.g. '#reader/abc' (the id is encodeURIComponent-ed; shot is left out).
 * @param {{view: string, sub?: string, id?: string}} route
 */
export function format(route) {
  return `#${buildPath(route.view, route.sub || '', route.id || '')}`;
}

function hashFor(route) {
  return `#${route.path}${route.shot ? `?shot=${encodeURIComponent(route.shot)}` : ''}`;
}

/** Where a launch without a route lands: Home, or the last view when Settings › Appearance › Open on says so. */
function defaultRoute() {
  const last = prefs.get().startOn === 'last' ? parseHash(prefs.get().lastRoute || '') : null;
  return last || parseHash(DEFAULT_PATH);
}

/** Where a wizard is remembered: the list it ends on. */
const WIZARD_HOME = Object.freeze({ clone: 'voices/recordings', design: 'voices' });

function remember(route) {
  const wizard = route.view === 'voices' && WIZARD_HOME[route.sub];
  const stored = wizard || route.path;
  if (prefs.get().lastRoute !== stored) prefs.set({ lastRoute: stored });
}

/**
 * Apply a route now. The newest route always wins: an application never waits for an earlier one, so a
 * view that is slow to load (or whose show() awaits a request, or calls navigate() itself) never holds up
 * or deadlocks navigation. The view host drops a route that was superseded while it loaded.
 */
function apply(route, { focus }) {
  const prev = cur;
  cur = route;
  latest = route;
  app.set({ route: Object.freeze({ view: route.view, sub: route.sub, id: route.id, path: route.path, shot: route.shot }) });
  remember(route);
  const run = (async () => {
    if (onRouteCb) {
      try {
        await onRouteCb(route, prev, { focus });
      } catch (error) {
        if (globalThis.console) globalThis.console.error(error);
      }
    }
    return route;
  })();
  latestRun = run;
  return run;
}

function currentHash() {
  return typeof location !== 'undefined' ? location.hash : '';
}

function normalise(route) {
  const wanted = hashFor(route);
  if (currentHash() !== wanted) history.replaceState(history.state ?? null, '', wanted);
}

function onHashChange() {
  const parsed = parseHash(currentHash(), { lastRoute: prefs.get().lastRoute });
  const route = parsed || defaultRoute();
  normalise(route);
  if (latest && route.path === latest.path && route.shot === latest.shot) return;
  apply(route, { focus: 'heading' });
}

/**
 * Go to a route. Pushes one history entry (replace: true replaces it); navigating to the current route is
 * a no-op. focus: 'heading' (the view's h1), 'keep' (sub changes from inside a view) or 'none'.
 * @param {string} path like 'voices/recordings' or '#reader/abc'
 * @param {{replace?: boolean, focus?: 'heading'|'keep'|'none'}} [options]
 * @returns {Promise<object>} the applied route
 */
export function navigate(path, { replace = false, focus = 'heading' } = {}) {
  const route = parseHash(path, { lastRoute: prefs.get().lastRoute }) || defaultRoute();
  if (latest && route.path === latest.path && route.shot === latest.shot) {
    if (currentHash() !== hashFor(route)) normalise(route);
    return latestRun;
  }
  const url = hashFor(route);
  if (replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  return apply(route, { focus });
}

/**
 * Resolve the initial route (a valid hash, else Home, or prefs.lastRoute when prefs.startOn is 'last'; normalised with
 * replaceState) and start listening to 'hashchange'. `onRoute(route, prev, {focus})` runs for each route
 * as soon as it is applied, without waiting for the previous call to finish (the newest route wins; the
 * returned promise of each navigation resolves when its own onRoute call does). Every applied route is
 * saved as prefs.lastRoute (voices/clone as voices/recordings, voices/design as voices).
 * @param {(route: object, prev: object|null, nav: {focus: string}) => *} onRoute
 * @returns {Promise<object>} the initial route once onRoute finished
 */
export function start(onRoute) {
  onRouteCb = onRoute;
  if (!listening) {
    listening = onHashChange;
    window.addEventListener('hashchange', listening);
  }
  const route = parseHash(currentHash(), { lastRoute: prefs.get().lastRoute }) || defaultRoute();
  normalise(route);
  return apply(route, { focus: 'none' });
}

/** @returns {object|null} the current route */
export function current() {
  return cur;
}

/** Test hook. */
export function __resetForTests() {
  if (listening && typeof window !== 'undefined' && window.removeEventListener) {
    try { window.removeEventListener('hashchange', listening); } catch { /* old env */ }
  }
  listening = null;
  onRouteCb = null;
  cur = null;
  latest = null;
  latestRun = Promise.resolve(null);
}
