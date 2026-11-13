/**
 * Roving tab stop for button-based radio groups and option lists (inventory GS-14): one tab
 * stop, arrow keys move focus (and activate, by default), Home/End, wrap. The Reader's speed presets use
 * it. Native radio groups (.seg) need no helper.
 * @module components/roving
 */

const KEYS = {
  horizontal: { next: ['ArrowRight'], prev: ['ArrowLeft'] },
  vertical: { next: ['ArrowDown'], prev: ['ArrowUp'] },
  both: { next: ['ArrowRight', 'ArrowDown'], prev: ['ArrowLeft', 'ArrowUp'] },
};

function usable(el) {
  return !el.hidden && !el.disabled && el.getAttribute('aria-disabled') !== 'true';
}

function stateAttr(el) {
  return el.getAttribute('role') === 'option' ? 'aria-selected' : 'aria-checked';
}

/**
 * Bind roving focus to the items inside `container`.
 * @param {HTMLElement} container the radiogroup / listbox element
 * @param {{items?: string, orientation?: 'horizontal'|'vertical'|'both', wrap?: boolean,
 *   activateOnMove?: boolean, onChange?: (item: HTMLElement, index: number, event: Event|null) => void}} [options]
 * @returns {{refresh: () => void, select: (target: number|HTMLElement, options?: {focus?: boolean, notify?: boolean}) => void,
 *   current: () => HTMLElement|null, destroy: () => void}}
 */
export function bindRoving(container, {
  items = '[role=radio], [role=option]', orientation = 'horizontal', wrap = true, activateOnMove = true, onChange,
} = {}) {
  const keys = KEYS[orientation] || KEYS.horizontal;
  const all = () => Array.from(container.querySelectorAll(items));
  const list = () => all().filter(usable);

  function current() {
    return list().find((el) => el.getAttribute(stateAttr(el)) === 'true') || null;
  }

  function syncTabStops(focusEl) {
    const els = all();
    const enabled = els.filter(usable);
    const stop = (focusEl && enabled.includes(focusEl) ? focusEl : null) || current() || enabled[0] || null;
    for (const el of els) el.setAttribute('tabindex', el === stop ? '0' : '-1');
  }

  function activate(el, event = null, { notify = true } = {}) {
    for (const other of all()) other.setAttribute(stateAttr(other), other === el ? 'true' : 'false');
    syncTabStops(el);
    if (notify && typeof onChange === 'function') onChange(el, list().indexOf(el), event);
  }

  function itemFrom(target) {
    const start = target && target.nodeType === 3 ? target.parentElement : target;
    const el = start && typeof start.closest === 'function' ? start.closest(items) : null;
    return el && container.contains(el) && usable(el) ? el : null;
  }

  function onKeydown(event) {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const from = itemFrom(event.target);
    if (!from) return;
    const els = list();
    const index = els.indexOf(from);
    let next = -1;
    if (keys.next.includes(event.key)) next = index + 1 >= els.length ? (wrap ? 0 : index) : index + 1;
    else if (keys.prev.includes(event.key)) next = index - 1 < 0 ? (wrap ? els.length - 1 : index) : index - 1;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = els.length - 1;
    else if ((event.key === ' ' || event.key === 'Enter') && from.localName !== 'button') {
      event.preventDefault();
      activate(from, event);
      return;
    } else return;
    event.preventDefault();
    const target = els[next];
    if (!target) return;
    target.focus();
    if (activateOnMove) activate(target, event);
    else syncTabStops(target);
  }

  function onClick(event) {
    const el = itemFrom(event.target);
    if (el) activate(el, event);
  }

  container.addEventListener('keydown', onKeydown);
  container.addEventListener('click', onClick);
  syncTabStops();

  return {
    refresh: () => syncTabStops(),
    current,
    select(target, { focus = false, notify = false } = {}) {
      const el = typeof target === 'number' ? list()[target] : target;
      if (!el) return;
      activate(el, null, { notify });
      if (focus) el.focus();
    },
    destroy() {
      container.removeEventListener('keydown', onKeydown);
      container.removeEventListener('click', onClick);
    },
  };
}
