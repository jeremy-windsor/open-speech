/**
 * App state, the event bus and cross-view intents. No DOM access.
 * @module core/store
 */

/**
 * Create a small observable store.
 * - `get()` returns the current state: a frozen shallow copy. Never mutate it.
 * - `set(patchOrFn)` shallow-merges top-level keys (a function receives the state and returns a patch) and
 *   notifies the selectors whose value changed.
 * - `select(selector, listener, {equals = Object.is, immediate = false})` calls `listener(value, prev, state)`
 *   when `selector(state)` changes, and returns an unsubscribe function.
 * @param {object} initial
 */
export function createStore(initial = {}) {
  let state = Object.freeze({ ...initial });
  const initialState = state;
  let subs = [];

  function get() { return state; }

  function set(patchOrFn) {
    const patch = typeof patchOrFn === 'function' ? patchOrFn(state) : patchOrFn;
    if (!patch || typeof patch !== 'object') return state;
    const prev = state;
    state = Object.freeze({ ...state, ...patch });
    for (const sub of subs.slice()) {
      if (!sub.active) continue;
      let value;
      try {
        value = sub.selector(state);
      } catch (error) {
        reportListenerError(error);
        continue;
      }
      if (sub.equals(value, sub.value)) continue;
      const old = sub.value;
      sub.value = value;
      try {
        sub.listener(value, old, state, prev);
      } catch (error) {
        reportListenerError(error);
      }
    }
    return state;
  }

  function select(selector, listener, { equals = Object.is, immediate = false } = {}) {
    const sub = { selector, listener, equals, value: selector(state), active: true };
    subs.push(sub);
    if (immediate) {
      try { listener(sub.value, undefined, state, state); } catch (error) { reportListenerError(error); }
    }
    return () => {
      sub.active = false;
      subs = subs.filter((s) => s !== sub);
    };
  }

  function reset() {
    state = initialState;
    subs = [];
  }

  return { get, set, select, __reset: reset };
}

function reportListenerError(error) {
  // A broken subscriber must not stop the others; surface it for developers only.
  if (globalThis.console) globalThis.console.error(error);
}

/** The single app store. */
export const app = createStore({
  route: { view: '', sub: '', id: '', path: '', shot: '' },
  connection: { state: 'connecting', version: '', revision: '', modelsLoaded: 0, checkedAt: 0, error: '', retryInMs: 0, retryAt: 0 },
  auth: { hasKey: false, refused: false },
  models: { list: [], defaults: { tts: '', stt: '' }, fetchedAt: 0, error: '' },
  ops: {},
  jobs: [],
  recording: null,
  player: null,
  resumable: null,
  speakVoice: { model: '', voice: '', label: '' },
  prefs: {},
  reading: {},
});

/** Event types (exhaustive). */
export const EVENTS = Object.freeze(['library:changed', 'voices:changed', 'history:changed', 'auth:changed', 'route:changed', 'assistants:changed']);

let listeners = new Map();

/**
 * Emit an app event. `detail` is passed to each listener.
 * @param {string} type one of EVENTS
 * @param {*} [detail]
 */
export function emit(type, detail) {
  if (!EVENTS.includes(type)) throw new TypeError(`Unknown event type: ${type}`);
  for (const listener of (listeners.get(type) || []).slice()) {
    try {
      listener(detail, type);
    } catch (error) {
      reportListenerError(error);
    }
  }
}

/**
 * Listen to an app event.
 * @param {string} type one of EVENTS
 * @param {(detail: *, type: string) => void} listener
 * @returns {() => void} off
 */
export function on(type, listener) {
  if (!EVENTS.includes(type)) throw new TypeError(`Unknown event type: ${type}`);
  const list = listeners.get(type) || [];
  list.push(listener);
  listeners.set(type, list);
  return () => {
    const current = listeners.get(type) || [];
    listeners.set(type, current.filter((l) => l !== listener));
  };
}

/**
 * The frozen cross-view intents: view -> allowed intent types. 'edit-preset' {id}, 'new-preset' {id: a named
 * voice} and 'use-named-voice' {id, model?} come from Voices' one list.
 */
export const INTENTS = Object.freeze({
  speak: Object.freeze(['apply-history', 'apply-preset', 'edit-named-voice', 'edit-preset', 'new-preset', 'use-named-voice', 'use-recording']),
  reader: Object.freeze(['open-appearance']),
});

let intents = new Map();

/**
 * Leave one pending intent for a view (a newer one replaces it). The sender then calls router.navigate().
 * @param {string} view
 * @param {{type: string}} intent
 */
export function setIntent(view, intent) {
  const allowed = INTENTS[view];
  if (!allowed || !intent || !allowed.includes(intent.type)) {
    throw new TypeError(`Unknown intent ${intent?.type} for ${view}`);
  }
  intents.set(view, Object.freeze({ ...intent }));
}

/**
 * Take (and clear) the pending intent for a view. The receiving view calls this in show().
 * @param {string} view
 * @returns {object|null}
 */
export function takeIntent(view) {
  const intent = intents.get(view) || null;
  intents.delete(view);
  return intent;
}

/**
 * Drop every pending intent except the one for `view`. The view host calls this once a route has shown (its view
 * took its own intent in show()): an intent whose navigation was superseded (Open in Speak, then Back before Speak
 * had loaded) must not apply on a later, unrelated visit to that view.
 * @param {string} view the view that is showing now
 */
export function dropIntentsExcept(view) {
  for (const key of [...intents.keys()]) if (key !== view) intents.delete(key);
}

/** Test hook: back to the initial state, no listeners, no intents. */
export function __resetForTests() {
  app.__reset();
  listeners = new Map();
  intents = new Map();
}
