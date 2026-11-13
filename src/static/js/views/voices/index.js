/**
 * Voices: the single home for named voices, presets and recordings, plus the 3-step clone
 * wizard at #voices/clone. The header holds the h1 and [Clone a voice]; #vo-sub (a segmented control)
 * switches between Voices (#voices: named voices with their presets, and the presets without one, in one
 * list) and Recordings with router.navigate(…, {focus: 'keep'}); #vo-search
 * filters the list shown ("/" focuses it). An old #voices/named or #voices/presets link opens the one list.
 *
 * Data (GS-12, SE-07): the first show() loads named voices, presets and recordings together; later shows
 * reload a list when it is older than 30 s or when a 'voices:changed' or 'auth:changed' event fired while the
 * view was hidden. Loads use request tokens. Hand-offs to Speak are intents: apply-preset,
 * edit-preset, new-preset, edit-named-voice and use-recording.
 *
 * Screenshot states (?shot=…): named-menu, transcript-sheet, with-clone-models, and load-error, empty and
 * loading (for the list shown); in the wizard clone-recording, clone-draft, clone-verify, clone-save and
 * clone-error.
 * @module views/voices/index
 */
import { actions, debounce } from '../../core/dom.js';
import { app, on as onEvent, emit } from '../../core/store.js';
import { navigate } from '../../core/router.js';
import { register } from '../../core/shortcuts.js';
import { announce } from '../../core/a11y.js';
import { ApiError } from '../../core/api.js';
import * as voices from '../../core/voices.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import { createLists, clearBanner, focusElement, searchSummary } from './shared.js';
import { createVoiceList } from './list.js';
import { createRecordings } from './recordings.js';
import { createWizard } from './clone.js';

/** The sections: 'voices' is the one list at #voices (route sub ''), 'recordings' and the wizard. */
const LIST_SUBS = ['voices', 'recordings'];
const PANEL_FOR = { voices: 'vo-voices', recordings: 'vo-recs', clone: 'vo-wizard' };
/** The data lists each section shows. */
const DATA_FOR = { voices: ['named', 'presets'], recordings: ['recordings'] };
const WHATS = ['named', 'presets', 'recordings'];

export default {
  id: 'voices',
  title: 'Voices',
  partial: '/static/partials/voices.html',
  init(root) {
    return createVoicesView(root);
  },
};

/**
 * Build the controller for the Voices section (exported for tests).
 * @param {HTMLElement} root the view section, with the partial rendered into it
 */
export function createVoicesView(root) {
  const $ = (id) => root.querySelector(`#${id}`);
  const el = {
    clone: $('vo-clone'),
    bar: $('vo-bar'),
    sub: $('vo-sub'),
    search: $('vo-search'),
    panels: Object.fromEntries(Object.entries(PANEL_FOR).map(([sub, id]) => [sub, $(id)])),
    sheet: $('vo-transcript-sheet'),
  };

  let sub = 'voices';
  let shown = false;
  let everShown = false;
  const dirty = new Set();
  let selfEmit = 0;
  let shotDone = '';
  const shotLists = new Set();   // lists a screenshot state emptied: reloaded when the shot is left
  const offs = [];

  const store = createLists((what) => renderList(what));
  const lists = store.lists;

  const query = () => (el.search ? el.search.value : '');

  function changed(what) {
    selfEmit += 1;
    try {
      emit('voices:changed', { what });
    } finally {
      selfEmit -= 1;
    }
  }

  const ctx = {
    lists,
    load: (what) => store.load(what),
    put: (what, items, extra) => store.put(what, items, extra),
    models: () => app.get().models,
    query,
    changed,
    navigate: (path, options) => navigate(path, options),
    announce: (text) => announce(text),
  };

  const list = createVoiceList(el.panels.voices, ctx);
  const recordings = createRecordings(el.panels.recordings, el.sheet, ctx);
  const wizard = createWizard(el.panels.clone, {
    recordingConfig: () => voices.recordingConfig(),
    recordings: () => lists.recordings.items,
    changed,
    afterSave: async (name) => {
      await store.load('recordings');
      // Only while the listener is still on the wizard: a save that finishes after they went elsewhere (a
      // long upload) must not pull them back or move their focus. The toast already said it was saved.
      if (!shown || sub !== 'clone' || !wizard.isShown()) return;
      await navigate('voices/recordings', { focus: 'keep' });
      if (shown && sub === 'recordings') recordings.focusCard(name);
    },
    openModels: (kind) => navigate(kind === 'stt' ? 'models/stt' : 'models/tts'),
  });

  /** [Clone a voice]: open the wizard with focus on its own heading (the header button hides there). */
  async function openWizard() {
    await navigate('voices/clone', { focus: 'keep' });
    if (!shown || sub !== 'clone') return;
    if (typeof window.scrollTo === 'function') window.scrollTo(0, 0);
    focusElement(wizard.heading());
  }

  /**
   * [Clone a voice] is the page's primary action, except while the list shown is empty and its empty state
   * offers its own primary ([Open Speak], [Clone your first voice]): one primary per page, as the Library does
   * with its header Add.
   */
  function syncCloneEmphasis() {
    if (!el.clone) return;
    const panel = el.panels[sub];
    const quiet = LIST_SUBS.includes(sub) && !!(panel && panel.querySelector('.empty .btn--primary'));
    el.clone.classList.toggle('btn--primary', !quiet);
    el.clone.classList.toggle('btn--secondary', quiet);
  }

  /** Re-render what shows a data list ('named', 'presets', 'recordings') or a section ('voices'). */
  function renderList(what) {
    if (what === 'named' || what === 'presets' || what === 'voices') list.render();
    else if (what === 'recordings') recordings.render();
    syncCloneEmphasis();
  }

  /** Load the lists that need it: every list on the first show and after auth:changed, else stale ones. */
  function loadNeeded({ all = false } = {}) {
    const now = Date.now();
    const jobs = [];
    for (const what of WHATS) {
      if (all || dirty.has(what) || store.stale(what, now)) {
        dirty.delete(what);
        jobs.push(store.load(what));
      }
    }
    return Promise.all(jobs);
  }

  function setSub(next) {
    sub = next;
    for (const [name, panel] of Object.entries(el.panels)) {
      if (panel) panel.hidden = name !== sub;
    }
    const inWizard = sub === 'clone';
    if (el.bar) el.bar.hidden = inWizard;
    if (el.clone) el.clone.hidden = inWizard;
    if (el.sub) {
      for (const input of el.sub.querySelectorAll('input[name="vo-sub"]')) input.checked = input.value === sub;
    }
    syncCloneEmphasis();
  }

  /** Screenshot states, applied once per show once the data they need is in. */
  async function applyShot(shot, loads) {
    if (!shot || shotDone === shot) return;
    if (shot.startsWith('clone-')) {
      shotDone = shot;
      wizard.applyShot(shot);
      return;
    }
    await loads;
    if (shot === 'named-menu' && sub === 'voices') {
      shotDone = shot;
      list.openFirstMenu();
    } else if (shot === 'transcript-sheet' && sub === 'recordings') {
      shotDone = shot;
      recordings.openTranscript();
    } else if (shot === 'with-clone-models' && sub === 'recordings') {
      shotDone = shot;
      recordings.applyShot(shot);
      syncCloneEmphasis();
    } else if (LIST_SUBS.includes(sub) && (shot === 'load-error' || shot === 'empty' || shot === 'loading')) {
      // The list's other states, for screenshots: the preview server always seeds the built-in presets and
      // the named voices made from them, so the empty list can't be reached by data alone. The lists load
      // again once the route shows without the shot. The one list empties both of its data lists.
      shotDone = shot;
      for (const what of DATA_FOR[sub]) {
        shotLists.add(what);
        const data = lists[what];
        data.seq += 1;   // a load still in flight can't overwrite the state
        data.items = [];
        data.at = shot === 'empty' ? Date.now() : 0;
        data.status = { 'load-error': 'error', empty: 'ready', loading: 'loading' }[shot];
        data.error = shot === 'load-error' ? new ApiError("Can't reach the server.", { status: 0, code: 'network', kind: 'network' }) : null;
      }
      renderList(sub);
    }
  }

  // ───────────── events ─────────────

  const actionMap = {
    clone: () => openWizard(),
    'wizard-close': () => navigate('voices/recordings'),
    'open-settings': () => openConnectionSettings(),
    'clear-search': () => {
      if (el.search) {
        el.search.value = '';
        renderList(sub);
        el.search.focus();
        announceSearch();
      }
    },
    'dismiss-banner': (button) => {
      const host = button.closest('.vo-slot');
      if (host) {
        host.dispatchEvent(new CustomEvent('vo-dismiss', { bubbles: true, detail: { id: host.id } }));
        clearBanner(host);
        const card = host.closest('.vo-card, .vo-preset, .vo-step, .sheet');
        const next = card ? card.querySelector('button:not([disabled]):not([aria-disabled="true"])') : null;
        if (next) next.focus();
      }
    },
    'retry-list': (button) => {
      const what = button.getAttribute('data-list');
      if (what === 'voices') return Promise.all(DATA_FOR.voices.map((w) => store.load(w)));
      if (WHATS.includes(what)) return store.load(what);
      return undefined;
    },
    ...list.actions,
    ...recordings.actions,
    ...wizard.actions,
  };
  offs.push(actions(root, actionMap));

  const onSubChange = (event) => {
    const input = event.target && event.target.closest ? event.target.closest('input[name="vo-sub"]') : null;
    if (!input || !input.checked) return;
    navigate(input.value === 'voices' ? 'voices' : `voices/${input.value}`, { focus: 'keep' });
  };
  if (el.sub) {
    el.sub.addEventListener('change', onSubChange);
    offs.push(() => el.sub.removeEventListener('change', onSubChange));
  }

  /** What the search shows in the section on screen (null while it loads or failed to load). */
  function shownSummary() {
    if (sub === 'voices') return list.searchSummary(query());
    const data = lists.recordings;
    const panel = el.panels.recordings;
    if (!panel || !data.at) return null;
    return searchSummary('recordings', panel.querySelectorAll('.vo-rec').length, query());
  }

  // A search filters the list as you type; once typing pauses, the result is said (WCAG 4.1.3): "1 named
  // voice and 2 presets match “jo”", "No recordings match “x”".
  const announceSearch = debounce(() => {
    if (!shown || !LIST_SUBS.includes(sub)) return;
    const summary = shownSummary();
    if (summary) announce(summary);
  }, 700);

  const onSearch = debounce(() => {
    if (LIST_SUBS.includes(sub)) renderList(sub);
  }, 120);
  if (el.search) {
    const handler = () => {
      onSearch();
      announceSearch();
    };
    el.search.addEventListener('input', handler);
    offs.push(() => el.search.removeEventListener('input', handler));
    offs.push(() => onSearch.cancel());
    offs.push(() => announceSearch.cancel());
  }

  offs.push(register({
    id: 'voices-search',
    keys: ['/'],
    scope: 'view:voices',
    singleKey: true,
    description: 'Search voices',
    when: () => shown && sub !== 'clone' && !!el.search,
    handler: (event) => {
      if (event && typeof event.preventDefault === 'function') event.preventDefault();
      el.search.focus();
      if (typeof el.search.select === 'function') el.search.select();
    },
  }));

  // Another view changed the lists (Speak saved a named voice or a preset): reload now when showing, else
  // on the next show. The view's own changes reload themselves.
  offs.push(onEvent('voices:changed', (detail) => {
    if (selfEmit) return;
    const what = detail && WHATS.includes(detail.what) ? [detail.what] : WHATS;
    for (const w of what) {
      if (shown) store.load(w);
      else dirty.add(w);
    }
  }));
  offs.push(onEvent('auth:changed', () => {
    if (!shown) for (const w of WHATS) dirty.add(w);
  }));
  // Model availability drives the Edit buttons, the clone-model select and the badges.
  let namedSig = '';
  offs.push(app.select((s) => s.models, (modelsState) => {
    recordings.onModels();
    const sig = `${modelsState.fetchedAt ? 1 : 0}|${(modelsState.list || []).filter((m) => m.state !== 'provider_missing' && m.provider_available !== false).map((m) => m.id).join(',')}`;
    if (sig !== namedSig) {
      namedSig = sig;
      list.render();
    }
    syncCloneEmphasis();
  }));

  // First paint: loading states.
  list.render();
  recordings.render();

  return {
    show(route, nav = {}) {
      const next = route && route.sub && PANEL_FOR[route.sub] ? route.sub : 'voices';
      const wasClone = sub === 'clone';
      shown = true;
      if (wasClone && next !== 'clone') wizard.hide();
      if (sub === 'recordings' && next !== 'recordings') {
        recordings.stopPreview();
        recordings.closeSheets();
      }
      setSub(next);
      if (!(route && route.shot)) {
        // Leaving a screenshot state: the real data again (the stand-ins and emptied lists go).
        shotDone = '';
        recordings.clearShot();   // setSub() and renderList() below sync the header button
        for (const what of shotLists) dirty.add(what);
        shotLists.clear();
      }
      // The list shown now reflects the current search text (it may have changed on another section).
      if (next === 'clone') wizard.show();
      else renderList(next);
      const loads = loadNeeded({ all: !everShown || (nav && nav.reason === 'auth') });
      everShown = true;
      if (route && route.shot) applyShot(route.shot, loads).catch(() => {});
      return undefined;
    },
    hide() {
      shown = false;
      if (sub === 'clone') wizard.hide();
      recordings.stopPreview();
      recordings.closeSheets();
      announceSearch.cancel();
    },
    /** Test hooks. */
    __internals: { lists, list, recordings, wizard, dirty, get sub() { return sub; } },
  };
}
