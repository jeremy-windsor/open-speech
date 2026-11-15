/**
 * Voices' one list (#voices): named voices and presets
 * together. A named voice is who speaks (one version per model); a preset is how it reads (speed, format,
 * effects, instructions). Each named voice is a card with its versions and, under them, its presets and [New
 * preset] (named.js); presets that use a plain voice, a blend or a recording are cards of their own in the group
 * "Presets without a named voice" at the end (presets.js). The default preset is marked wherever it sits.
 *
 * The list renders from the view's two data lists, `named` and `presets`, each with its own request token,
 * loading and error state: one error state when both failed, a Retry per list otherwise. Inline errors of cards
 * and presets outlive a re-render (shared.js createSlots), and so does an open card menu.
 * @module views/voices/list
 */
import { html } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { applyTile } from '../../core/tiles.js';
import { toast } from '../../core/toast.js';
import {
  matches, loadingMarkup, loadErrorMarkup, staleErrorMarkup, fillDetails, loadErrorDetail, renderKeepingFocus, bindMenus,
  openMenuFocus, restoreMenuFocus, actionError, createSlots, idPart, focusElement,
} from './shared.js';
import { createNamed, namedCardMarkup, namedMatches } from './named.js';
import { createPresets, presetItemMarkup, presetSummary } from './presets.js';

/** The group of presets that use no named voice. */
export const PLAIN_TITLE = 'Presets without a named voice';

/**
 * What the search shows, said once typing pauses: "1 named voice and 2 presets match “jo”", "1 preset matches
 * “podcast”", "No named voices or presets match “x”", "2 named voices and 4 presets".
 */
export function voicesSummary(namedCount, presetCount, query) {
  const q = String(query || '').trim();
  const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const parts = [];
  if (namedCount) parts.push(count(namedCount, 'named voice', 'named voices'));
  if (presetCount) parts.push(count(presetCount, 'preset', 'presets'));
  if (!parts.length) return q ? `No named voices or presets match “${q}”` : 'No named voices or presets';
  const subject = parts.join(' and ');
  if (!q) return subject;
  return `${subject} ${namedCount + presetCount === 1 ? 'matches' : 'match'} “${q}”`;
}

const firstLoad = (list) => !list.at && (list.status === 'loading' || list.status === 'idle');
const failed = (list) => list.status === 'error' && !list.at;

/**
 * Where everything sits for a search: the named voices' cards with the presets each shows, and the presets
 * without a named voice. A voice that matches shows all its presets; one that doesn't shows only the presets
 * that match. A preset whose named voice isn't listed (gone, or the named voices didn't load) has no card.
 */
export function arrange(namedList, presetsList, query = '') {
  const identities = namedList.at ? namedList.items || [] : [];
  const items = presetsList.at ? presetsList.items || [] : [];
  const defaultId = presetsList.defaultId;
  const own = new Map(identities.map((identity) => [identity.id, []]));
  const plain = [];
  for (const preset of items) {
    const list = preset.voice_identity_id ? own.get(preset.voice_identity_id) : null;
    if (list) list.push(preset);
    else plain.push(preset);
  }
  // The default first, then by name (the server sorts by name).
  const ordered = (list) => list.slice().sort((a, b) => (b.id === defaultId) - (a.id === defaultId));
  const cards = [];
  for (const identity of identities) {
    const presets = ordered(own.get(identity.id));
    const hit = namedMatches(identity, query);
    const shown = hit ? presets : presets.filter((p) => matches(query, p.name, presetSummary(p, identities, { identity })));
    if (hit || shown.length) cards.push({ identity, presets: shown });
  }
  return {
    cards,
    plain: ordered(plain).filter((p) => matches(query, p.name, presetSummary(p, identities))),
    total: identities.length + items.length,
    defaultId,
    identities,
  };
}

/**
 * The markup of the one list for a state (exported for tests).
 * @param {{status: string, items: object[], error: *, at: number}} namedList
 * @param {{status: string, items: object[], defaultId: string|null, error: *, at: number}} presetsList
 * @param {{query?: string, models?: object}} [options]
 */
export function voicesListMarkup(namedList, presetsList, { query = '', models } = {}) {
  if (firstLoad(namedList) || firstLoad(presetsList)) return loadingMarkup('named voices and presets', { kind: 'card', rows: 2 });
  if (failed(namedList) && failed(presetsList)) return loadErrorMarkup(namedList.error, { what: 'voices and presets', list: 'voices' });
  const top = [];
  if (failed(namedList)) top.push(loadErrorMarkup(namedList.error, { what: 'named voices', list: 'named' }));
  else if (namedList.status === 'error') top.push(staleErrorMarkup(namedList.error, { what: 'named voices', list: 'named' }));
  if (failed(presetsList)) top.push(loadErrorMarkup(presetsList.error, { what: 'presets', list: 'presets' }));
  else if (presetsList.status === 'error') top.push(staleErrorMarkup(presetsList.error, { what: 'presets', list: 'presets' }));
  const { cards, plain, total, defaultId, identities } = arrange(namedList, presetsList, query);
  if (!total) {
    if (failed(namedList) || failed(presetsList)) return html`${top}`;
    return html`${top}<div class="card vo-empty-card"><div class="empty">${icon('voices', { className: 'ico empty__icon' })}<h2 class="empty__title">No named voices or presets yet</h2><p class="empty__text">In Speak, choose a voice, then Save named voice… or Save preset….</p><button type="button" class="btn btn--primary" data-action="open-speak"><span class="btn__label">Open Speak</span></button></div></div>`;
  }
  if (!cards.length && !plain.length) {
    return html`${top}<div class="empty vo-nomatch">${icon('search', { className: 'ico empty__icon' })}<h2 class="empty__title">No named voices or presets match “${query.trim()}”</h2><button type="button" class="btn btn--secondary" data-action="clear-search"><span class="btn__label">Clear search</span></button></div>`;
  }
  const grid = cards.length
    ? html`<ul class="vo-grid" aria-label="Named voices">${cards.map((c) => namedCardMarkup(c.identity, { models, presets: c.presets, defaultId, named: identities }))}</ul>`
    : '';
  // Without the named voices, which voice a preset uses is unknown: the group is just "Presets".
  const title = namedList.at ? PLAIN_TITLE : 'Presets';
  const group = plain.length
    ? html`<section class="vo-group" aria-labelledby="vo-plain-title"><h2 class="vo-group__title" id="vo-plain-title">${title}</h2><ul class="vo-grid" aria-labelledby="vo-plain-title">${plain.map((p) => presetItemMarkup(p, { isDefault: p.id === defaultId, named: identities, card: true }))}</ul></section>`
    : '';
  return html`${top}${grid}${group}`;
}

/**
 * The one list.
 * @param {HTMLElement} panel #vo-voices
 * @param {{lists: object, load: (what: string) => Promise<boolean>, put: Function, models: () => object,
 *   query: () => string, changed: (what: string) => void, navigate: Function, announce: (text: string) => void}} ctx
 */
export function createVoiceList(panel, ctx) {
  const listEl = panel.querySelector('#vo-voices-list');
  let unbind = () => {};
  const slots = createSlots(panel);

  /**
   * Re-render the list. restore: the id of the control that had focus before an async action (focus may have
   * dropped to <body> while it was disabled); when it is gone, fallback() picks the next target. A card menu
   * that had focus (a reload from another view's change) opens again on the same item.
   */
  function render({ fallback, restore = '' } = {}) {
    const menuFocus = openMenuFocus(listEl);
    unbind();
    const { named, presets } = ctx.lists;
    listEl.setAttribute('aria-busy', named.status === 'loading' || presets.status === 'loading' ? 'true' : 'false');
    renderKeepingFocus(listEl, voicesListMarkup(named, presets, { query: ctx.query(), models: ctx.models() }), { fallback, wasFocused: restore });
    if (failed(named) && failed(presets)) fillDetails(listEl, loadErrorDetail(named.error, 'voices and presets'));
    else if (failed(named)) fillDetails(listEl, loadErrorDetail(named.error, 'named voices'));
    else if (failed(presets)) fillDetails(listEl, loadErrorDetail(presets.error, 'presets'));
    for (const tile of listEl.querySelectorAll('[data-tile]')) applyTile(tile, tile.getAttribute('data-tile'));
    for (const seg of listEl.querySelectorAll('.blend-bar__seg[data-w]')) seg.style.setProperty('--w', `${seg.getAttribute('data-w')}%`);
    slots.restore(listEl);
    cards.patchSamples();
    unbind = bindMenus(listEl, onMenu);
    restoreMenuFocus(unbind, menuFocus, listEl, fallback);
  }

  const parts = {
    ...ctx,
    listEl,
    render,
    /** An inline error in a card's or a preset's slot (a toast when the item is gone). */
    error(slot, error, title) {
      const e = actionError(error, title);
      if (listEl.querySelector(`#${slot}`)) slots.error(listEl, slot, e);
      else toast(`${e.title}: ${e.message}`, { kind: 'error' });
    },
    clearError(slot) {
      slots.clear(listEl, slot);
    },
  };
  const cards = createNamed(parts);
  const presets = createPresets(parts);

  function onMenu(action, item, button) {
    if (/^vo-named-/.test(button.id)) cards.onMenu(action, item, button);
    else if (/^vo-preset-/.test(button.id)) presets.onMenu(action, item, button);
  }

  return {
    render,
    actions: { ...cards.actions, ...presets.actions },
    /** What the search shows now (null while nothing has loaded). */
    searchSummary(query) {
      if (!ctx.lists.named.at && !ctx.lists.presets.at) return null;
      return voicesSummary(listEl.querySelectorAll('.vo-card--voice').length, listEl.querySelectorAll('.vo-preset').length, query);
    },
    /** Focus a named voice's card; false when it isn't listed (a search may hide it). */
    focusCard(id) {
      const title = listEl.querySelector(`#vo-named-${idPart(id)}-title`);
      if (title) focusElement(title);
      return !!title;
    },
    stopSample: () => cards.stopSample(),
    /** Screenshot hook: open the first card's menu. */
    openFirstMenu() {
      const button = listEl.querySelector('[data-menu-btn]');
      if (button) button.click();
      return !!button;
    },
    destroy() {
      unbind();
      cards.destroy();
      slots.destroy();
    },
  };
}
