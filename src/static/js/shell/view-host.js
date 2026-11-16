/**
 * The view host: loads each view on its first visit (its partial, then its module, then
 * init(root)), shows one view section at a time, and on every route change, in this order:
 *   1. loads the view if needed. Past 150 ms the previous view is hidden and the new section shows a quiet
 *      "Loading…" line; a failure shows an error state with Retry
 *   2. calls hide() on the previous view and hides its section
 *   3. unhides the new section and calls show(route, nav)
 *   4. sets document.title, aria-current (through the store's route) and immersive mode
 *   5. moves focus: 'heading' -> the view's visible h1 (preventScroll, then scroll to the top unless the
 *      view restored a position by setting nav.scrolled = true or returning {scrolled: true});
 *      'keep' and 'none' leave focus alone
 *   6. drops the cross-view intents left for other views (their navigation was superseded) and emits
 *      'route:changed' {route, prev}
 *
 * The newest route always wins: a route that is superseded while its view loads is dropped (no hide/show,
 * no focus move, no event). A controller's show() is called but never awaited: a view that loads its data
 * in show() (GS-12) or navigates from it never holds up navigation. show() must set nav.scrolled
 * synchronously (before its first await) and render its own loading state; a promise it returns only has
 * its rejection logged.
 *
 * Views stay mounted once loaded. prefetch(ids) caches partials and modules without init() (and runs a view
 * definition's optional prefetch(), which warms what the view imports lazily); loadPartial(url)
 * is exported for slots. The partial fetch here is the one fetch outside core/api.js (static, auth-exempt),
 * with a 15 s timeout so a stalled request reaches the error state.
 * @module shell/view-host
 */
import { render, raw, html } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { app, emit, dropIntentsExcept } from '../core/store.js';

/** The view modules, imported on first use (never in main.js's static closure). */
export const VIEW_MODULES = Object.freeze({
  home: () => import('../views/home/index.js'),
  reader: () => import('../views/reader/index.js'),
  speak: () => import('../views/speak/index.js'),
  transcribe: () => import('../views/transcribe/index.js'),
  talk: () => import('../views/talk/index.js'),
  voices: () => import('../views/voices/index.js'),
  studio: () => import('../views/studio/index.js'),
  history: () => import('../views/history/index.js'),
  models: () => import('../views/models/index.js'),
  settings: () => import('../views/settings/index.js'),
});

const LOADING_AFTER_MS = 150;
const PARTIAL_TIMEOUT_MS = 15000;
const APP_NAME = 'Open Speech';

let registry = VIEW_MODULES;
let partials = new Map();   // url -> Promise<string>
let modules = new Map();    // view id -> Promise<view definition>
let views = new Map();      // view id -> {def, controller, section}
let loading = new Map();    // view id -> Promise<entry>
let warmed = new Set();     // view ids whose definition's prefetch() ran
let currentId = '';          // the view whose section is showing (possibly still loading)
let currentRoute = null;     // the route of the last show() that completed
let showSeq = 0;             // bumped by every show(): an older one that is still loading is dropped
let immersive = false;
let immersiveListeners = new Set();
let versionOff = null;

/**
 * Fetch a partial's markup (cached; a failed fetch is retried next time).
 * @param {string} url a /static/partials/… path
 * @returns {Promise<string>}
 */
export function loadPartial(url) {
  if (partials.has(url)) return partials.get(url);
  const task = (async () => {
    const controller = typeof AbortController === 'function' ? new AbortController() : null;
    const timer = setTimeout(() => { if (controller) controller.abort(); }, PARTIAL_TIMEOUT_MS);
    try {
      const res = await fetch(url, { cache: 'no-cache', signal: controller ? controller.signal : undefined });
      if (!res.ok) throw new Error(`The page part ${url} didn't load (HTTP ${res.status}).`);
      return await res.text();
    } catch (error) {
      if (controller && controller.signal.aborted) throw new Error(`The page part ${url} didn't load in time.`);
      throw error;
    } finally {
      clearTimeout(timer);
    }
  })();
  partials.set(url, task);
  task.catch(() => { if (partials.get(url) === task) partials.delete(url); });
  return task;
}

function sectionFor(id) {
  return document.getElementById(`view-${id}`);
}

function importView(id) {
  if (modules.has(id)) return modules.get(id);
  const loader = registry[id];
  if (typeof loader !== 'function') return Promise.reject(new Error(`Unknown view: ${id}`));
  const task = Promise.resolve().then(loader).then((mod) => {
    const def = mod && (mod.default || mod);
    if (!def || typeof def.init !== 'function' || !def.partial) throw new Error(`The ${id} view module is incomplete.`);
    return def;
  });
  modules.set(id, task);
  task.catch(() => { if (modules.get(id) === task) modules.delete(id); });
  return task;
}

function titleOf(id) {
  const entry = views.get(id);
  return (entry && entry.def.title) || id.charAt(0).toUpperCase() + id.slice(1);
}

function renderError(section, id) {
  render(section, html`<header class="view-head"><h1 class="view-head__title" data-view-heading tabindex="-1">${titleOf(id)}</h1></header>
    <div class="empty empty--error view-error">${icon('alert', { className: 'ico empty__icon' })}<h2 class="empty__title">This section didn't load</h2><p class="empty__text">Check your connection, then try again. The rest of Open Speech keeps working.</p><button type="button" class="btn btn--primary" data-view-retry><span class="btn__label">Retry</span></button></div>`);
  const retry = section.querySelector('[data-view-retry]');
  if (retry) {
    retry.addEventListener('click', () => {
      // Retry the route the app is on now; a newer navigation supersedes this show() as usual.
      const route = app.get().route;
      if (!route || route.view !== id) return;
      show(route, null, { focus: 'heading' }).catch(() => {});
    }, { once: true });
  }
}

/** Load (once) and init a view: partial + module, then init(root). */
function ensureLoaded(id) {
  if (views.has(id)) return Promise.resolve(views.get(id));
  if (loading.has(id)) return loading.get(id);
  const section = sectionFor(id);
  if (!section) return Promise.reject(new Error(`There is no section for the ${id} view.`));
  const task = (async () => {
    const def = await importView(id);
    const markup = await loadPartial(def.partial);
    render(section, raw(markup));
    const controller = def.init(section) || {};
    const entry = { def, controller, section };
    views.set(id, entry);
    return entry;
  })();
  loading.set(id, task);
  task.catch(() => { if (loading.get(id) === task) loading.delete(id); });
  return task;
}

/** Hide the view that is showing (calling its hide()) and show `id`'s section. */
function switchTo(id, section) {
  if (currentId && currentId !== id) {
    const old = views.get(currentId);
    if (old && typeof old.controller.hide === 'function') {
      try { old.controller.hide(); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
    }
  }
  for (const other of document.querySelectorAll('main > section.view')) {
    if (other !== section && !other.hidden) other.hidden = true;
  }
  if (section) section.hidden = false;
  currentId = id;
}

function visibleHeading(section) {
  const all = Array.from(section.querySelectorAll('h1'));
  const shown = all.filter((h) => {
    for (let node = h; node && node !== section; node = node.parentElement) if (node.hidden) return false;
    return true;
  });
  return shown.find((h) => h.hasAttribute('data-view-heading')) || shown[0] || null;
}

function documentTitle(entry, route) {
  const version = app.get().connection.version;
  let custom = '';
  if (entry && typeof entry.controller.title === 'function') {
    try { custom = entry.controller.title(route) || ''; } catch { custom = ''; }
  }
  if (custom) return `${custom} · ${APP_NAME}`;
  const name = entry ? entry.def.title : '';
  const base = name ? `${name} · ${APP_NAME}` : APP_NAME;
  return version ? `${base} v${version}` : base;
}

function setImmersive(on) {
  const next = !!on;
  const root = document.documentElement;
  if (next) root.setAttribute('data-immersive', '');
  else root.removeAttribute('data-immersive');
  if (next === immersive) return;
  immersive = next;
  for (const cb of [...immersiveListeners]) {
    try { cb(immersive); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
  }
}

/**
 * Show a route: the router.start() callback. Resolves with the route once it is showing, or with null when
 * a newer route superseded it while its view was loading (nothing was changed for it).
 * @param {{view: string, sub: string, id: string, path: string, shot: string}} route
 * @param {object|null} prev
 * @param {{focus?: 'heading'|'keep'|'none'}} [nav]
 * @returns {Promise<object|null>}
 */
export async function show(route, prev = null, { focus = 'heading' } = {}) {
  const seq = (showSeq += 1);
  const id = route.view;
  const section = sectionFor(id);
  let entry = views.get(id) || null;
  if (!entry) {
    // Past 150 ms the previous view goes and this section shows a quiet "Loading…" line.
    let slow = setTimeout(() => {
      slow = null;
      if (seq !== showSeq || views.has(id) || !section) return;
      render(section, html`<p class="view__loading">Loading…</p>`);
      switchTo(id, section);
    }, LOADING_AFTER_MS);
    try {
      entry = await ensureLoaded(id);
    } catch (error) {
      if (globalThis.console) globalThis.console.error(error);
      if (seq === showSeq && section) renderError(section, id);
    } finally {
      if (slow) clearTimeout(slow);
    }
  }
  if (seq !== showSeq) return null;   // a newer route took over while this view loaded
  switchTo(id, section);
  const shownPrev = currentRoute;
  currentRoute = route;
  const nav = { focus, scrolled: false, reason: 'route' };
  if (entry && typeof entry.controller.show === 'function') {
    try {
      const result = entry.controller.show(route, nav);
      // Never awaited: a view's data load (or a redirect it starts) must not hold up navigation.
      if (result && typeof result.then === 'function') {
        result.then(null, (error) => { if (globalThis.console) globalThis.console.error(error); });
      } else if (result && result.scrolled) {
        nav.scrolled = true;
      }
    } catch (error) {
      if (globalThis.console) globalThis.console.error(error);
    }
  }
  if (seq !== showSeq) return null;   // show() navigated somewhere else
  // An intent left for another view belongs to a navigation that never completed: drop it (the view shown now
  // took its own in show(); a view that failed to load keeps its intent for Retry).
  dropIntentsExcept(id);
  document.title = documentTitle(entry, route);
  let wantImmersive = false;
  if (entry && typeof entry.controller.immersive === 'function') {
    try { wantImmersive = !!entry.controller.immersive(route); } catch { wantImmersive = false; }
  }
  setImmersive(wantImmersive);
  if (focus === 'heading' && section) {
    const heading = visibleHeading(section);
    if (heading) {
      if (!heading.hasAttribute('tabindex')) heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
    }
    if (!nav.scrolled && typeof window.scrollTo === 'function') window.scrollTo(0, 0);
  }
  emit('route:changed', { route, prev: shownPrev });
  return route;
}

/**
 * Set document.title again for the view showing now. A view whose title(route) depends on data it loaded
 * (the Reader's document title) calls this once the data is in.
 */
export function refreshTitle() {
  if (!currentRoute) return;
  document.title = documentTitle(views.get(currentId), currentRoute);
}

/**
 * A view definition's optional `prefetch()` warms the modules the view imports lazily (the Reader's engine).
 * Called at most once per view, on idle; a failure is left to the real visit.
 * @param {string} id
 * @param {object} def
 * @returns {Promise<void>}
 */
function warm(id, def) {
  if (warmed.has(id) || !def || typeof def.prefetch !== 'function') return Promise.resolve();
  warmed.add(id);
  return Promise.resolve().then(() => def.prefetch()).then(() => {}, () => {});
}

/**
 * Fetch partials and import modules for later (on idle, after the first view has rendered), without init(). A view definition may carry
 * `prefetch()`, which imports what the view loads lazily (for example `() => import('./engine.js')`), so a
 * control that needs it is ready before the listener can press it; it runs here, never on a route change.
 * @param {string[]} ids
 * @returns {Promise<void>}
 */
export async function prefetch(ids) {
  await Promise.all((ids || []).map(async (id) => {
    if (views.has(id) || loading.has(id)) return;
    try {
      const def = await importView(id);
      await Promise.all([loadPartial(def.partial), warm(id, def)]);
    } catch {
      // A prefetch failure is retried by the real visit.
    }
  }));
}

/**
 * Call show() again on the current view (after auth:changed), keeping focus and scroll where they are.
 * Like a route change, it does not wait for whatever show() starts.
 * @param {string} [reason]
 */
export async function reloadCurrent(reason = 'reload') {
  const entry = views.get(currentId);
  if (!entry || !currentRoute || typeof entry.controller.show !== 'function') return;
  try {
    const result = entry.controller.show(currentRoute, { focus: 'keep', scrolled: true, reason });
    if (result && typeof result.then === 'function') {
      result.then(null, (error) => { if (globalThis.console) globalThis.console.error(error); });
    }
  } catch (error) {
    if (globalThis.console) globalThis.console.error(error);
  }
}

/** @returns {boolean} whether the current view is immersive (the Reader's reading view) */
export function isImmersive() {
  return immersive;
}

/** @returns {() => void} off; cb(immersive) runs when immersive mode changes */
export function onImmersive(cb) {
  immersiveListeners.add(cb);
  return () => immersiveListeners.delete(cb);
}

/** @returns {object|null} the controller of a loaded view */
export function controllerFor(id) {
  const entry = views.get(id);
  return entry ? entry.controller : null;
}

/** @returns {string} the id of the view showing now */
export function currentView() {
  return currentId;
}

/**
 * Configure the host. registry maps view ids to module loaders (tests pass their own).
 * @param {{registry?: Record<string, () => Promise<object>>}} [options]
 */
export function initViewHost({ registry: custom } = {}) {
  if (custom) registry = custom;
  if (versionOff) versionOff();
  // The default title carries the server version once /health has answered (GS-16).
  versionOff = app.select((s) => s.connection.version, () => {
    const entry = views.get(currentId);
    if (currentRoute) document.title = documentTitle(entry, currentRoute);
  });
}

/** Test hook. */
export function __resetForTests() {
  registry = VIEW_MODULES;
  partials = new Map();
  modules = new Map();
  views = new Map();
  loading = new Map();
  warmed = new Set();
  currentId = '';
  currentRoute = null;
  showSeq = 0;
  immersive = false;
  immersiveListeners = new Set();
  if (versionOff) versionOff();
  versionOff = null;
}
