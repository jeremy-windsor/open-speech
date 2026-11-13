/**
 * APG menu button. Items are role=menuitem, menuitemradio or menuitemcheckbox.
 * - Enter, Space, a click or ArrowDown open the menu on its first item (the checked one for radios);
 *   ArrowUp opens it on the last item.
 * - In the menu: ArrowUp/ArrowDown (wrapping), Home/End, type-ahead, Enter/Space/click activate.
 * - Esc closes and returns focus to the button (inventory AX-10); inside a sheet, Esc closes only the menu.
 * - Tab closes and moves focus on from the button (focus goes to the button first and the browser's own Tab
 *   continues from there); Shift+Tab closes and stops on the button. So focus order follows the button's
 *   place in the document wherever the menu element sits (WCAG 2.4.3).
 * - A click outside closes the menu without activating what was under it (the pointerdown/click swallow).
 * - The menu opens below the button, or above it when there is no room below. Its position is set before
 *   its first style is computed, so it never starts in the corner.
 * - A menu may sit anywhere, also inside a `.card`: while it is open, a menu whose ancestor would become the
 *   containing block of its `position: fixed` box (a size container such as `.card`, a transform, a filter,
 *   a backdrop-filter, layout or paint containment) moves to the end of the nearest open `<dialog>`, else of
 *   its view section (`[data-view]`), else of `<body>`; a placeholder keeps its place and it moves back when
 *   it closes. Style a menu by its own class (`.menu`, a view-prefixed class), never by the
 *   card it is written in, and handle its items through `onSelect` (or delegation on the view section).
 * @module components/menu
 */

const ITEM = '[role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"]';
const TYPEAHEAD_MS = 700;
const GAP = 4;
const EDGE = 8;
const DEFAULT_SIZE = Object.freeze({ width: 220, height: 200 });

const openMenus = new Set();

function viewport() {
  return { width: globalThis.innerWidth || 1024, height: globalThis.innerHeight || 768 };
}

/**
 * PURE: whether a computed style makes its element the containing block of `position: fixed` descendants
 * (CSS Position 3 and CSS Containment 2), which would offset a fixed menu by that element's position and
 * paint it inside that element's stacking context.
 * @param {{getPropertyValue: (name: string) => string}|null} style a CSSStyleDeclaration (or a look-alike)
 * @returns {boolean}
 */
export function trapsFixed(style) {
  if (!style || typeof style.getPropertyValue !== 'function') return false;
  const value = (name) => String(style.getPropertyValue(name) || '').trim().toLowerCase();
  const set = (name) => { const v = value(name); return v !== '' && v !== 'none'; };
  const containerType = value('container-type');
  if (containerType && containerType !== 'normal') return true;
  if (/\b(?:layout|paint|strict|content)\b/.test(value('contain'))) return true;
  const visibility = value('content-visibility');
  if (visibility === 'auto' || visibility === 'hidden') return true;
  if (['transform', 'translate', 'rotate', 'scale', 'perspective', 'filter', 'backdrop-filter', '-webkit-backdrop-filter'].some(set)) return true;
  return /\b(?:transform|translate|rotate|scale|perspective|filter|contain)\b/.test(value('will-change'));
}

/**
 * Where an open menu must live so `position: fixed` places it against the viewport: null when it can stay
 * where it is, else the element to append it to (the nearest open dialog, else the view section, else body).
 * Only the ancestors below the menu's dialog (or body) are checked: a dialog is in the top layer.
 * @param {HTMLElement} menu
 * @returns {HTMLElement|null}
 */
export function menuHost(menu) {
  const read = globalThis.getComputedStyle;
  if (!menu || typeof read !== 'function') return null;
  const doc = menu.ownerDocument || globalThis.document;
  const stop = (menu.closest && menu.closest('dialog')) || (doc && doc.body) || null;
  let trap = null;
  for (let el = menu.parentElement; el && el !== stop; el = el.parentElement) {
    let style = null;
    try { style = read(el); } catch { style = null; }
    if (trapsFixed(style)) trap = el;   // keep going: the outermost one decides
  }
  if (!trap) return null;
  const outside = trap.parentElement;
  return (outside && outside.closest && outside.closest('dialog, section[data-view]')) || stop;
}

/**
 * PURE: the menu position for a button rect and a menu size.
 * @returns {{x: number, y: number, up: boolean, maxHeight: number}}
 */
export function placeMenu(rect, size, { placement = 'bottom-start', width, height } = {}) {
  const vw = width ?? viewport().width;
  const vh = height ?? viewport().height;
  const below = vh - rect.bottom - GAP - EDGE;
  const above = rect.top - GAP - EDGE;
  const up = placement.startsWith('top') ? above >= Math.min(size.height, 160) || above > below : size.height > below && above > below;
  const room = Math.max(120, up ? above : below);
  const h = Math.min(size.height, room);
  let x = placement.endsWith('end') ? rect.right - size.width : rect.left;
  x = Math.max(EDGE, Math.min(x, vw - EDGE - size.width));
  const y = up ? rect.top - GAP - h : rect.bottom + GAP;
  return { x: Math.round(x), y: Math.round(y), up, maxHeight: Math.floor(room) };
}

/**
 * Bind a menu button to its menu.
 * @param {HTMLElement} button
 * @param {HTMLElement} menu role=menu, hidden while closed
 * @param {{onSelect?: (item: HTMLElement, event: Event) => void, placement?: 'bottom-start'|'bottom-end'|'top-start'|'top-end'}} [options]
 * @returns {{open: (focus?: 'first'|'last'|'checked') => void, close: (options?: {restoreFocus?: boolean}) => void,
 *   isOpen: () => boolean, position: () => void, destroy: () => void}} position() re-measures and re-places an
 *   open menu (after a resize or a scroll); destroy() closes it (back in its place) and unbinds
 */
export function bindMenuButton(button, menu, { onSelect, placement = 'bottom-start' } = {}) {
  if (!menu.id) throw new TypeError('bindMenuButton() needs a menu with an id');
  button.setAttribute('aria-haspopup', 'menu');
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-controls', menu.id);
  if (!menu.getAttribute('role')) menu.setAttribute('role', 'menu');
  menu.hidden = true;

  let typed = { text: '', at: 0 };
  let swallowNextClick = false;
  let escapedAt = -Infinity;
  let lastSize = null;      // the menu's size when it last opened (the first guess for the next open)
  let placeholder = null;   // marks the menu's own place while it is re-hosted

  const items = () => Array.from(menu.querySelectorAll(ITEM)).filter((el) => !el.hidden);
  const isOpen = () => !menu.hidden;

  function focusItem(el) {
    if (!el) return;
    for (const item of items()) item.setAttribute('tabindex', item === el ? '0' : '-1');
    el.focus();
  }

  function place(rect, size) {
    menu.style.setProperty('--menu-w', `${Math.round(rect.width)}px`);
    const spot = placeMenu(rect, size, { placement });
    menu.style.setProperty('--menu-x', `${spot.x}px`);
    menu.style.setProperty('--menu-y', `${spot.y}px`);
    menu.style.setProperty('--menu-max-h', `${spot.maxHeight}px`);
    menu.setAttribute('data-placement', spot.up ? 'top' : 'bottom');
  }

  /** Measure the open menu and place it at its button (also for a caller that re-places it on resize). */
  function position() {
    const rect = button.getBoundingClientRect();
    const box = menu.getBoundingClientRect();
    // scrollHeight: the content's own height, not the height a previous --menu-max-h clipped it to.
    const height = Math.max(box.height || 0, menu.scrollHeight || 0);
    const size = { width: box.width || DEFAULT_SIZE.width, height: height || DEFAULT_SIZE.height };
    if (box.width && height) lastSize = size;
    place(rect, size);
  }

  /** Move the menu out of a containing-block trap while it is open. */
  function rehost() {
    if (placeholder) return;
    const host = menuHost(menu);
    if (!host || !menu.parentNode) return;
    placeholder = (menu.ownerDocument || document).createComment(` ${menu.id} `);
    menu.before(placeholder);
    host.append(menu);
  }

  /** Put a re-hosted menu back in its place; a menu whose place was removed (re-rendered) goes too. */
  function restore() {
    if (!placeholder) return;
    const mark = placeholder;
    placeholder = null;
    if (mark.isConnected) mark.replaceWith(menu);
    else {
      mark.remove();
      menu.remove();
    }
  }

  function open(focus = 'first') {
    for (const other of [...openMenus]) if (other !== controller) other.close({ restoreFocus: false });
    rehost();
    // Placed while still hidden: its first computed style already has its position, so no
    // transition or headless capture ever starts it at (0, 0). position() then corrects for its real size.
    place(button.getBoundingClientRect(), lastSize || DEFAULT_SIZE);
    menu.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    openMenus.add(controller);
    position();
    const list = items();
    const checked = list.find((el) => el.getAttribute('aria-checked') === 'true');
    const target = focus === 'last' ? list[list.length - 1] : (focus === 'checked' && checked) || (focus === 'first' && checked) || list[0];
    focusItem(target);
    document.addEventListener('pointerdown', onOutsideDown, true);
    document.addEventListener('click', onOutsideClick, true);
    document.addEventListener('pointercancel', onPointerCancel, true);
    document.addEventListener('focusin', onFocusIn, true);
  }

  function close({ restoreFocus = true } = {}) {
    if (!isOpen()) {
      restore();
      return;
    }
    menu.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    openMenus.delete(controller);
    document.removeEventListener('pointerdown', onOutsideDown, true);
    document.removeEventListener('pointercancel', onPointerCancel, true);
    document.removeEventListener('focusin', onFocusIn, true);
    if (!swallowNextClick) document.removeEventListener('click', onOutsideClick, true);
    if (restoreFocus) button.focus();
    restore();
  }

  function activate(item, event) {
    if (!item || item.getAttribute('aria-disabled') === 'true') return;
    const role = item.getAttribute('role');
    if (role === 'menuitemradio') {
      const group = item.closest('[role="group"]') || menu;
      for (const other of group.querySelectorAll('[role="menuitemradio"]')) other.setAttribute('aria-checked', other === item ? 'true' : 'false');
    } else if (role === 'menuitemcheckbox') {
      item.setAttribute('aria-checked', item.getAttribute('aria-checked') === 'true' ? 'false' : 'true');
    }
    close({ restoreFocus: true });
    if (typeof onSelect === 'function') onSelect(item, event);
  }

  function onOutsideDown(event) {
    if (!isOpen()) return;
    const t = event.target;
    if (menu.contains(t) || button.contains(t)) return;
    // Close only: the click that follows must not press what was under the pointer.
    swallowNextClick = true;
    close({ restoreFocus: false });
  }
  function onPointerCancel() {
    if (swallowNextClick) {
      swallowNextClick = false;
      document.removeEventListener('click', onOutsideClick, true);
    }
  }
  function onOutsideClick(event) {
    if (!swallowNextClick) return;
    swallowNextClick = false;
    document.removeEventListener('click', onOutsideClick, true);
    event.preventDefault();
    event.stopPropagation();
  }
  function onFocusIn(event) {
    if (isOpen() && !menu.contains(event.target) && event.target !== button) close({ restoreFocus: false });
  }

  function onButtonClick(event) {
    if (isOpen()) close({ restoreFocus: true });
    else open('first');
    event.preventDefault();
  }
  function onButtonKeydown(event) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      open(event.key === 'ArrowUp' ? 'last' : 'first');
    }
  }

  function onMenuKeydown(event) {
    const list = items();
    const index = list.indexOf(document.activeElement);
    const key = event.key;
    if (key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      escapedAt = Date.now();
      close({ restoreFocus: true });
      return;
    }
    if (key === 'Tab') {
      // Focus order follows the button, wherever the menu element sits (a re-hosted menu, a menu in a host
      // after a list): Tab goes on from the button (the browser's own Tab runs after this), Shift+Tab stops on
      // it, which is where both keys go from a menu written right after its button.
      if (event.altKey || event.ctrlKey || event.metaKey) {
        close({ restoreFocus: false });
        return;
      }
      if (event.shiftKey) event.preventDefault();
      button.focus({ preventScroll: true });
      close({ restoreFocus: false });
      return;
    }
    let next = null;
    if (key === 'ArrowDown') next = list[(index + 1) % list.length];
    else if (key === 'ArrowUp') next = list[(index - 1 + list.length) % list.length];
    else if (key === 'Home') next = list[0];
    else if (key === 'End') next = list[list.length - 1];
    else if (key === 'Enter' || key === ' ') {
      event.preventDefault();
      if (index >= 0) activate(list[index], event);
      return;
    } else if (key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const now = Date.now();
      typed = { text: (now - typed.at < TYPEAHEAD_MS ? typed.text : '') + key.toLowerCase(), at: now };
      const ordered = [...list.slice(index + 1), ...list.slice(0, index + 1)];
      next = ordered.find((el) => el.textContent.trim().toLowerCase().startsWith(typed.text))
        || ordered.find((el) => el.textContent.trim().toLowerCase().startsWith(typed.text.slice(-1)));
    } else return;
    event.preventDefault();
    if (next) focusItem(next);
  }

  function onMenuClick(event) {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const item = start && start.closest ? start.closest(ITEM) : null;
    if (item && menu.contains(item)) {
      event.preventDefault();
      activate(item, event);
    }
  }

  // A menu inside a sheet: Esc closes the menu, not the sheet (inventory RE-65).
  const dialog = menu.closest ? menu.closest('dialog') : null;
  function onDialogCancel(event) {
    if (isOpen() || Date.now() - escapedAt < 50) {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (isOpen()) close({ restoreFocus: true });
    }
  }

  button.addEventListener('click', onButtonClick);
  button.addEventListener('keydown', onButtonKeydown);
  menu.addEventListener('keydown', onMenuKeydown);
  menu.addEventListener('click', onMenuClick);
  if (dialog) dialog.addEventListener('cancel', onDialogCancel, true);

  const controller = {
    open,
    close,
    isOpen,
    position,
    destroy() {
      close({ restoreFocus: false });
      swallowNextClick = false;
      document.removeEventListener('click', onOutsideClick, true);
      button.removeEventListener('click', onButtonClick);
      button.removeEventListener('keydown', onButtonKeydown);
      menu.removeEventListener('keydown', onMenuKeydown);
      menu.removeEventListener('click', onMenuClick);
      if (dialog) dialog.removeEventListener('cancel', onDialogCancel, true);
    },
  };
  return controller;
}

/** Close every open menu (route changes, sheets closing). */
export function closeAllMenus() {
  for (const menu of [...openMenus]) menu.close({ restoreFocus: false });
}

/** Test hook. */
export function __resetForTests() {
  for (const menu of [...openMenus]) menu.destroy();
  openMenus.clear();
}
