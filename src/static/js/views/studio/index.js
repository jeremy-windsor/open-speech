/**
 * Studio: conversations (scripts of turns for several voices, rendered as one file) and the
 * Composer (WAV renders mixed into one track). #st-sub, a segmented control, switches between the two with
 * router.navigate(…, {focus: 'keep'}); a conversation has its own route, studio/conversations/<id>.
 *
 * Data (GS-12): the first show() loads the conversations, the presets (the turn preset select, ST-10) and the
 * past mixes together; a later show reloads a list that is older than 30 s, or after 'voices:changed' (presets)
 * or 'auth:changed' fired while the view was hidden. Loads use request tokens.
 *
 * Screenshot states (?shot=…): conv-menu, turn-menu, rendering, conflict, reorder-error, add-error, not-found,
 * load-error, presets-error, playing (conversations); tracks, mix-error, mix-result, mix-playing, past-playing,
 * mixing, sources-error (composer).
 * @module views/studio/index
 */
import { actions as bindActions } from '../../core/dom.js';
import { on as onEvent } from '../../core/store.js';
import { navigate } from '../../core/router.js';
import * as api from '../../core/api.js';
import { ApiError } from '../../core/api.js';
import * as voices from '../../core/voices.js';
import { toast } from '../../core/toast.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import { refreshAvailability } from '../../components/ask-ai.js';
import { createLists, clearBanner, focusElement } from './shared.js';
import { createConversations, convPath } from './conversations.js';
import { createComposer } from './composer.js';

const SUBS = ['conversations', 'composer'];
const LISTS = ['conversations', 'presets', 'mixes'];

export default {
  id: 'studio',
  title: 'Studio',
  partial: '/static/partials/studio.html',
  init(root) {
    return createStudioView(root);
  },
};

/**
 * Build the controller for the Studio section (exported for tests).
 * @param {HTMLElement} root the view section, with the partial rendered into it
 */
export function createStudioView(root) {
  const $ = (id) => root.querySelector(`#${id}`);
  const el = {
    sub: $('st-sub'),
    conversations: $('st-conversations'),
    composer: $('st-composer'),
  };

  let sub = 'conversations';
  let shown = false;
  let everShown = false;
  const dirty = new Set();
  let shotDone = '';
  const offs = [];

  const store = createLists({
    conversations: async () => (await api.get('/api/conversations?limit=100&offset=0', { accept: 'json' }))?.items || [],
    presets: async () => (await voices.listPresets()).presets,
    mixes: async () => (await api.get('/api/composer/renders?limit=50&offset=0', { accept: 'json' }))?.items || [],
  }, (name) => onListChange(name));
  const { lists } = store;

  const ctx = {
    lists,
    load: (name) => store.load(name),
    navigate: (path, options) => navigate(path, options),
    isShown: () => shown,
    isCurrentSub: () => sub === 'conversations',
  };

  const conversations = createConversations(el.conversations, ctx);
  const composer = createComposer(el.composer, { ...ctx, isCurrentSub: () => sub === 'composer' });

  function onListChange(name) {
    if (name === 'conversations') {
      conversations.renderList();
      composer.renderSources();
    } else if (name === 'presets') {
      conversations.renderPresets();
    } else if (name === 'mixes') {
      composer.renderMixes();
      composer.renderSources();
    }
  }

  function loadNeeded({ all = false } = {}) {
    const now = Date.now();
    const jobs = [];
    for (const name of LISTS) {
      if (all || dirty.has(name) || store.stale(name, now)) {
        dirty.delete(name);
        jobs.push(store.load(name));
      }
    }
    return Promise.all(jobs);
  }

  function setSub(next) {
    sub = next;
    el.conversations.hidden = sub !== 'conversations';
    el.composer.hidden = sub !== 'composer';
    for (const input of el.sub.querySelectorAll('input[name="st-sub"]')) input.checked = input.value === sub;
  }

  /** A list that never loaded and failed (screenshot states only). */
  function failList(name) {
    const list = lists[name];
    list.seq += 1;
    list.items = [];
    list.at = 0;
    list.status = 'error';
    list.error = new ApiError("Can't reach the server.", { status: 0, code: 'network', kind: 'network' });
  }

  /** Screenshot states, applied once per show once the data they need is in. */
  async function applyShot(shot, loads, convLoad) {
    if (!shot || shotDone === shot) return;
    await loads;
    await convLoad;
    shotDone = shot;
    if (shot === 'load-error' || shot === 'sources-error') {
      failList('conversations');
      conversations.renderList();
      composer.renderSources();
      return;
    }
    if (shot === 'presets-error') {
      failList('presets');
      conversations.renderPresets();
      return;
    }
    if (sub === 'composer') composer.applyShot(shot);
    else conversations.applyShot(shot);
  }

  /* ── events ── */

  const actionMap = {
    'open-settings': () => openConnectionSettings(),
    'retry-load': (button) => {
      const name = button.getAttribute('data-list');
      if (name === 'conversation') return conversations.retry();
      if (LISTS.includes(name)) return store.load(name);
      return undefined;
    },
    'dismiss-banner': (button) => {
      const host = button.closest('.st-slot');
      if (!host) return;
      clearBanner(host);
      const region = host.closest('.st-editor, .st-mix, .st-panel') || root;
      const next = region.querySelector('h2[tabindex], .st-editor__title, .st-mix__title, #st-title');
      focusElement(next || $('st-title'));
    },
    ...conversations.actions,
    ...composer.actions,
  };
  offs.push(bindActions(root, actionMap, {
    onError: (error) => {
      if (!error || error.kind === 'aborted' || error.name === 'AbortError') return;
      toast(error.message || String(error), { kind: 'error' });
    },
  }));

  const onSubChange = (event) => {
    const input = event.target && event.target.closest ? event.target.closest('input[name="st-sub"]') : null;
    if (!input || !input.checked) return;
    if (input.value === 'conversations') {
      const last = conversations.lastOpen();
      navigate(last ? convPath(last) : 'studio/conversations', { focus: 'keep' });
    } else {
      navigate(`studio/${input.value}`, { focus: 'keep' });
    }
  };
  el.sub.addEventListener('change', onSubChange);
  offs.push(() => el.sub.removeEventListener('change', onSubChange));

  // Presets changed elsewhere (Voices, Speak): the turn preset select follows (ST-10).
  offs.push(onEvent('voices:changed', (detail) => {
    if (detail && detail.what && detail.what !== 'presets') return;
    if (shown) store.load('presets');
    else dirty.add('presets');
  }));
  offs.push(onEvent('auth:changed', () => {
    if (!shown) for (const name of LISTS) dirty.add(name);
  }));

  // First paint: loading states.
  conversations.renderList();
  conversations.renderEditor();
  composer.renderMixes();

  return {
    show(route, nav = {}) {
      const next = route && SUBS.includes(route.sub) ? route.sub : 'conversations';
      shown = true;
      setSub(next);
      const auth = !!(nav && nav.reason === 'auth');
      let convLoad = null;
      if (next === 'conversations') {
        convLoad = conversations.show(route && route.id ? route.id : '', { reload: auth });
        composer.hide();
      } else {
        conversations.hide();
        composer.show();
      }
      const loads = loadNeeded({ all: !everShown || auth });
      everShown = true;
      if (route && route.shot) applyShot(route.shot, loads, convLoad).catch(() => {});
      else {
        shotDone = '';
        // Is the AI assistant on (Ask AI about a script)? A screenshot asks nothing.
        refreshAvailability().catch(() => {});
      }
      return undefined;
    },
    hide() {
      shown = false;
      conversations.hide();
      composer.hide();
    },
    /** Test hooks. */
    __internals: { lists, conversations, composer, dirty, get sub() { return sub; }, store },
  };
}
