/**
 * APG listbox, single select, optionally grouped.
 * - Driven from a combobox input (`combobox` option): the input keeps focus and points at the active option
 *   with aria-activedescendant. Standalone: roving focus on the options (one tab stop).
 * - Keys: ArrowUp/ArrowDown, Home/End, PageUp/PageDown (10), Enter selects; standalone also Space and
 *   type-ahead (700 ms buffer).
 * - Options also select on click: VoiceOver on iOS sends a click on double-tap and does not follow
 *   aria-activedescendant, so this is its path.
 * - aria-disabled options can be active (their reason is read) but are never selected.
 * Option model: {value, label, detail?, disabled?, srText?, ...anything the caller's renderOption needs}.
 * @module components/listbox
 */
import { html, render } from '../core/dom.js';
import { icon } from '../core/icons.js';

const PAGE = 10;
const TYPEAHEAD_MS = 700;

function defaultOption(option) {
  return html`<span class="option__text"><span class="option__label">${option.label}</span>${option.detail ? html`<span class="option__detail">${option.detail}</span>` : ''}${option.srText ? html`<span class="visually-hidden">${`, ${option.srText}`}</span>` : ''}</span><span class="option__check" aria-hidden="true">${icon('check')}</span>`;
}

/**
 * @param {HTMLElement} el the role=listbox container
 * @param {{groups?: {label?: string, options: object[]}[], value?: *, onActive?: (option: object|null, index: number) => void,
 *   onSelect?: (option: object, event: Event|null) => void, typeahead?: boolean, virtual?: boolean,
 *   combobox?: HTMLInputElement|null, renderOption?: (option: object) => import('../core/dom.js').SafeHtml,
 *   idPrefix?: string, ignoreClick?: string}} [options]
 */
export function createListbox(el, {
  groups = [], value = null, onActive, onSelect, typeahead = true, virtual = false, combobox = null,
  renderOption = defaultOption, idPrefix = '', ignoreClick = '[data-act]',
} = {}) {
  const prefix = idPrefix || `${el.id || 'listbox'}-opt`;
  if (!el.getAttribute('role')) el.setAttribute('role', 'listbox');
  let flat = [];
  let selected = value;
  let activeIndex = -1;
  let typed = { text: '', at: 0 };
  let currentGroups = groups;

  const optionEl = (i) => (i >= 0 ? el.querySelector(`[role="option"][data-index="${i}"]`) : null);

  function build(list) {
    flat = [];
    const named = list.filter((g) => g && (g.options || []).length);
    const markup = named.map((group, gi) => {
      const rows = group.options.map((option) => {
        const index = flat.length;
        flat.push(option);
        const isSelected = selected !== null && selected !== undefined && option.value === selected;
        return html`<div class="option" role="option" id="${`${prefix}-${index}`}" data-index="${index}" aria-selected="${isSelected ? 'true' : 'false'}"${option.disabled ? html` aria-disabled="true"` : ''}${combobox ? '' : html` tabindex="-1"`}>${renderOption(option)}</div>`;
      });
      if (!group.label) return html`<div class="listbox__group">${rows}</div>`;
      const gid = `${prefix}-g${gi}`;
      return html`<div class="listbox__group" role="group" aria-labelledby="${gid}"><div class="listbox__label" id="${gid}" role="presentation">${group.label}</div>${rows}</div>`;
    });
    render(el, markup);
    if (virtual) el.setAttribute('data-virtual', 'true');
  }

  function markActive(index, { scroll = true, focus = false } = {}) {
    const prev = optionEl(activeIndex);
    if (prev) {
      prev.classList.remove('is-active');
      if (!combobox) prev.setAttribute('tabindex', '-1');
    }
    activeIndex = index >= 0 && index < flat.length ? index : -1;
    const next = optionEl(activeIndex);
    if (combobox) {
      if (next) combobox.setAttribute('aria-activedescendant', next.id);
      else combobox.removeAttribute('aria-activedescendant');
    }
    if (next) {
      next.classList.add('is-active');
      if (!combobox) {
        next.setAttribute('tabindex', '0');
        if (focus) next.focus();
      }
      if (scroll && typeof next.scrollIntoView === 'function') next.scrollIntoView({ block: 'nearest' });
    } else if (!combobox) {
      const first = optionEl(0);
      if (first) first.setAttribute('tabindex', '0');
    }
    if (typeof onActive === 'function') onActive(activeIndex >= 0 ? flat[activeIndex] : null, activeIndex);
  }

  function indexOfValue(v) {
    return flat.findIndex((o) => o.value === v);
  }

  function setGroups(next, { value: v, keepActive = true } = {}) {
    const keep = keepActive && activeIndex >= 0 ? flat[activeIndex]?.value : undefined;
    if (v !== undefined) selected = v;
    currentGroups = next || [];
    build(currentGroups);
    activeIndex = -1;
    let start = keep !== undefined ? indexOfValue(keep) : -1;
    if (start < 0) start = indexOfValue(selected);
    if (start < 0) start = flat.length ? 0 : -1;
    markActive(start, { scroll: true });
  }

  function setValue(v) {
    selected = v;
    flat.forEach((option, i) => {
      const node = optionEl(i);
      if (node) node.setAttribute('aria-selected', option.value === v ? 'true' : 'false');
    });
  }

  function select(index, event = null) {
    const option = flat[index];
    if (!option || option.disabled) return false;
    setValue(option.value);
    if (typeof onSelect === 'function') onSelect(option, event);
    return true;
  }

  function move(to, options) {
    if (!flat.length) return;
    markActive(Math.max(0, Math.min(flat.length - 1, to)), options);
  }

  /**
   * Handle a key from the combobox (or the listbox itself). Returns true when the key was used.
   * @param {KeyboardEvent} event
   */
  function handleKey(event) {
    if (event.altKey || event.ctrlKey || event.metaKey) return false;
    const focus = !combobox;
    const key = event.key;
    let handled = true;
    if (key === 'ArrowDown') move(activeIndex < 0 ? 0 : activeIndex + 1, { focus });
    else if (key === 'ArrowUp') move(activeIndex < 0 ? flat.length - 1 : activeIndex - 1, { focus });
    else if (key === 'Home') move(0, { focus });
    else if (key === 'End') move(flat.length - 1, { focus });
    else if (key === 'PageDown') move(activeIndex + PAGE, { focus });
    else if (key === 'PageUp') move(activeIndex - PAGE, { focus });
    else if (key === 'Enter' && !event.shiftKey) {
      if (activeIndex >= 0) select(activeIndex, event);
    } else if (key === ' ' && !combobox) {
      if (activeIndex >= 0) select(activeIndex, event);
    } else if (!combobox && typeahead && key.length === 1 && key !== ' ') {
      const now = Date.now();
      typed = { text: (now - typed.at < TYPEAHEAD_MS ? typed.text : '') + key.toLowerCase(), at: now };
      // A new letter searches after the active option; more letters refine from the active one.
      const from = Math.max(0, activeIndex) + (typed.text.length > 1 ? 0 : 1);
      const order = flat.map((_, k) => (from + k) % flat.length);
      const hit = order.find((k) => String(flat[k].label || '').toLowerCase().startsWith(typed.text));
      if (hit !== undefined) move(hit, { focus });
    } else handled = false;
    if (handled) event.preventDefault();
    return handled;
  }

  function onClick(event) {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const node = start && start.closest ? start.closest('[role="option"]') : null;
    if (!node || !el.contains(node)) return;
    const index = Number(node.getAttribute('data-index'));
    if (!Number.isInteger(index)) return;
    markActive(index, { scroll: false, focus: !combobox });
    if (ignoreClick && start.closest(ignoreClick)) return;
    select(index, event);
  }
  const onOwnKey = (event) => { handleKey(event); };

  el.addEventListener('click', onClick);
  const keyTarget = combobox || el;
  keyTarget.addEventListener('keydown', onOwnKey);
  setGroups(groups, { value, keepActive: false });

  return {
    setGroups,
    setValue,
    value: () => selected,
    options: () => flat.slice(),
    activeIndex: () => activeIndex,
    activeOption: () => (activeIndex >= 0 ? flat[activeIndex] : null),
    setActive(target, options = {}) {
      const index = typeof target === 'number' ? target : indexOfValue(target);
      markActive(index, options);
    },
    optionElement: (index) => optionEl(index),
    select,
    handleKey,
    destroy() {
      el.removeEventListener('click', onClick);
      keyTarget.removeEventListener('keydown', onOwnKey);
    },
  };
}

