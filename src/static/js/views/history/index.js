/**
 * History: the server's record of generated speech and transcribed files.
 *
 * Data (GS-12): the first show() loads the first page; a later show reloads when the list is older than 30 s,
 * or after 'history:changed' or 'auth:changed' fired while the view was hidden. Speak and Transcribe emit
 * 'history:changed' when the server logged a take or a file (X-History); while History shows, that reloads
 * what is loaded in the background (list.refresh()). Loads use request tokens (views/history/list.js).
 *
 * Shortcut: "/" focuses the search field (scope view:history).
 * Screenshot states (?shot=…): row-menu, load-error, search-empty, loading, more-error.
 * @module views/history/index
 */
import { actions as bindActions } from '../../core/dom.js';
import { on as onEvent } from '../../core/store.js';
import { navigate } from '../../core/router.js';
import { register } from '../../core/shortcuts.js';
import { toast } from '../../core/toast.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import { clearBanner, focusElement } from '../studio/shared.js';
import { refreshAvailability } from '../../components/ask-ai.js';
import { createHistoryList } from './list.js';

export default {
  id: 'history',
  title: 'History',
  partial: '/static/partials/history.html',
  init(root) {
    return createHistoryView(root);
  },
};

/**
 * Build the controller for the History section (exported for tests).
 * @param {HTMLElement} root the view section, with the partial rendered into it
 */
export function createHistoryView(root) {
  let shown = false;
  let dirty = false;
  let shotDone = '';
  const offs = [];

  const list = createHistoryList(root, {
    isShown: () => shown,
    navigate: (path, options) => navigate(path, options),
    openSettings: () => openConnectionSettings(),
  });

  const actionMap = {
    ...list.actions,
    'dismiss-banner': (button) => {
      const host = button.closest('.hi-slot');
      if (host) clearBanner(host);
      focusElement(root.querySelector('#hi-title'));
    },
  };
  offs.push(bindActions(root, actionMap, {
    onError: (error) => {
      if (!error || error.kind === 'aborted' || error.name === 'AbortError') return;
      toast(error.message || String(error), { kind: 'error' });
    },
  }));

  offs.push(register({
    id: 'history-search',
    keys: ['/'],
    scope: 'view:history',
    singleKey: true,
    description: 'Search history',
    when: () => shown,
    handler: () => list.focusSearch(),
  }));

  // While History shows, another view's change reloads what is loaded (the pages, focus and an open menu stay).
  offs.push(onEvent('history:changed', () => {
    if (list.isSelfEmit()) return;
    if (shown) list.refresh();
    else dirty = true;
  }));
  offs.push(onEvent('auth:changed', () => { if (!shown) dirty = true; }));

  return {
    show(route, nav = {}) {
      shown = true;
      const auth = !!(nav && nav.reason === 'auth');
      let loading = null;
      if (dirty || auth || list.stale()) {
        dirty = false;
        loading = list.load();
      }
      if (route && route.shot && shotDone !== route.shot) {
        const shot = route.shot;
        Promise.resolve(loading).then(() => {
          shotDone = shot;
          list.applyShot(shot);
        }).catch(() => {});
      } else if (!route || !route.shot) {
        shotDone = '';
      }
      // Is the AI assistant on (the rows' Ask AI)? Once per 30 s; a screenshot state asks nothing.
      if (!route || !route.shot) refreshAvailability().catch(() => {});
      return undefined;
    },
    hide() {
      shown = false;
    },
    /** Test hooks. */
    __internals: { list, get dirty() { return dirty; } },
  };
}
