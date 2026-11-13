/**
 * Tooltips for every [data-tooltip] element (WCAG 1.4.13). They show on hover after
 * 500 ms and at once on keyboard focus, stay while the pointer is over the trigger or the tooltip, and
 * close on Esc, blur or pointer-out. One #app-tooltip (role=tooltip) serves them all. Icon buttons keep
 * their aria-label; the tooltip shows the same text visually, so aria-describedby is set only when the
 * tooltip says something the accessible name doesn't.
 * Placement: data-tooltip-placement="top" (default), "bottom" or "right"; it flips when there is no room.
 * While a modal sheet is open, #app-tooltip lives inside it (core/dialogs.js moves it), so tooltips for the
 * sheet's own controls draw above the backdrop.
 * @module components/tooltip
 */
import { liveHost } from '../core/dialogs.js';

const HOVER_DELAY_MS = 500;
const HIDE_DELAY_MS = 120;
const GAP = 8;
const EDGE = 8;

let active = null;   // the controller from the most recent initTooltips()

function tooltipEl() {
  let el = document.getElementById('app-tooltip');
  if (!el) {
    el = document.createElement('div');
    el.id = 'app-tooltip';
    el.className = 'tooltip';
    el.setAttribute('role', 'tooltip');
    el.hidden = true;
    (liveHost() || document.body).appendChild(el);
  }
  return el;
}

function triggerFrom(target) {
  const start = target && target.nodeType === 3 ? target.parentElement : target;
  return start && typeof start.closest === 'function' ? start.closest('[data-tooltip]') : null;
}

function keyboardFocus(el) {
  try {
    return el.matches(':focus-visible');
  } catch {
    return true;   // no :focus-visible support: treat focus as keyboard focus
  }
}

function viewport() {
  return { width: globalThis.innerWidth || 1024, height: globalThis.innerHeight || 768 };
}

/**
 * PURE: where to put a tooltip of `size` next to `rect`, inside the viewport.
 * @returns {{x: number, y: number, placement: 'top'|'bottom'|'right'}}
 */
export function placeTooltip(rect, size, { placement = 'top', width, height } = {}) {
  const vw = width ?? viewport().width;
  const vh = height ?? viewport().height;
  let side = placement;
  if (side === 'right' && rect.right + GAP + size.width > vw - EDGE) side = 'top';
  if (side === 'top' && rect.top - GAP - size.height < EDGE) side = 'bottom';
  else if (side === 'bottom' && rect.bottom + GAP + size.height > vh - EDGE && rect.top - GAP - size.height >= EDGE) side = 'top';
  let x;
  let y;
  if (side === 'right') {
    x = rect.right + GAP;
    y = rect.top + rect.height / 2 - size.height / 2;
  } else {
    x = rect.left + rect.width / 2 - size.width / 2;
    y = side === 'top' ? rect.top - GAP - size.height : rect.bottom + GAP;
  }
  x = Math.max(EDGE, Math.min(x, vw - EDGE - size.width));
  y = Math.max(EDGE, Math.min(y, vh - EDGE - size.height));
  return { x: Math.round(x), y: Math.round(y), placement: side };
}

/**
 * Serve every [data-tooltip] inside `root` (event delegation, so elements added later work too).
 * @param {Document|HTMLElement} [root]
 * @param {{delay?: number}} [options]
 * @returns {{hide: () => void, destroy: () => void, current: () => HTMLElement|null}}
 */
export function initTooltips(root = document, { delay = HOVER_DELAY_MS } = {}) {
  if (active) active.destroy();
  const tip = tooltipEl();
  let owner = null;          // the trigger the tooltip is showing for
  let showTimer = null;
  let hideTimer = null;
  let overTip = false;
  let overOwner = false;
  let addedDescribedBy = false;

  function clearTimers() {
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    showTimer = null;
    hideTimer = null;
  }

  function show(el) {
    clearTimers();
    const text = el.getAttribute('data-tooltip') || '';
    if (!text || el.getAttribute('aria-expanded') === 'true' || el.disabled) return;
    // A tooltip that repeats a label already on screen (the wide rail) adds nothing.
    const shown = el.querySelector('.nav-link__label, .btn__label, .pill__label');
    if (shown && shown.textContent.trim() === text.trim() && shown.getBoundingClientRect().width > 2) return;
    if (owner && owner !== el) hide();
    owner = el;
    tip.textContent = text;
    tip.hidden = false;
    const label = (el.getAttribute('aria-label') || el.textContent || '').trim();
    if (text.trim() !== label) {
      const ids = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
      if (!ids.includes(tip.id)) {
        el.setAttribute('aria-describedby', [...ids, tip.id].join(' '));
        addedDescribedBy = true;
      }
    }
    position();
  }

  function position() {
    if (!owner) return;
    const rect = owner.getBoundingClientRect();
    const box = tip.getBoundingClientRect();
    const spot = placeTooltip(rect, { width: box.width || 120, height: box.height || 30 }, { placement: owner.getAttribute('data-tooltip-placement') || 'top' });
    tip.style.setProperty('--tt-x', `${spot.x}px`);
    tip.style.setProperty('--tt-y', `${spot.y}px`);
    tip.setAttribute('data-placement', spot.placement);
  }

  function hide() {
    clearTimers();
    if (owner && addedDescribedBy) {
      const ids = (owner.getAttribute('aria-describedby') || '').split(/\s+/).filter((id) => id && id !== tip.id);
      if (ids.length) owner.setAttribute('aria-describedby', ids.join(' '));
      else owner.removeAttribute('aria-describedby');
    }
    addedDescribedBy = false;
    owner = null;
    overTip = false;
    overOwner = false;
    tip.hidden = true;
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      if (!overTip && !overOwner && !(owner && owner.contains(document.activeElement) && keyboardFocus(owner))) hide();
    }, HIDE_DELAY_MS);
  }

  const onOver = (event) => {
    if (event.pointerType === 'touch') return;
    const el = triggerFrom(event.target);
    if (!el) return;
    overOwner = true;
    if (owner === el) { clearTimeout(hideTimer); return; }
    clearTimeout(showTimer);
    showTimer = setTimeout(() => show(el), delay);
  };
  const onOut = (event) => {
    const el = triggerFrom(event.target);
    if (!el) return;
    const to = event.relatedTarget;
    if (to && el.contains(to)) return;
    overOwner = false;
    clearTimeout(showTimer);
    showTimer = null;
    if (to && tip.contains(to)) return;   // moving onto the tooltip keeps it (it is hoverable)
    if (owner === el) scheduleHide();
  };
  const onFocusIn = (event) => {
    const el = triggerFrom(event.target);
    if (el && el === event.target && keyboardFocus(el)) show(el);
  };
  const onFocusOut = (event) => {
    const el = triggerFrom(event.target);
    if (el && el === owner) hide();
  };
  const onKeydown = (event) => {
    if (event.key === 'Escape' && owner) {
      // Dismissible without moving focus (WCAG 1.4.13); a sheet underneath stays open.
      event.preventDefault();
      event.stopPropagation();
      hide();
    }
  };
  const onDown = () => { if (owner) hide(); };
  const onTipEnter = () => { overTip = true; clearTimeout(hideTimer); };
  const onTipLeave = () => { overTip = false; scheduleHide(); };
  const onScroll = () => { if (owner) hide(); };

  root.addEventListener('pointerover', onOver);
  root.addEventListener('pointerout', onOut);
  root.addEventListener('focusin', onFocusIn);
  root.addEventListener('focusout', onFocusOut);
  root.addEventListener('pointerdown', onDown, true);
  document.addEventListener('keydown', onKeydown, true);
  tip.addEventListener('pointerenter', onTipEnter);
  tip.addEventListener('pointerleave', onTipLeave);
  if (typeof window !== 'undefined') window.addEventListener('scroll', onScroll, { passive: true, capture: true });

  const controller = {
    hide,
    current: () => owner,
    destroy() {
      hide();
      root.removeEventListener('pointerover', onOver);
      root.removeEventListener('pointerout', onOut);
      root.removeEventListener('focusin', onFocusIn);
      root.removeEventListener('focusout', onFocusOut);
      root.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKeydown, true);
      tip.removeEventListener('pointerenter', onTipEnter);
      tip.removeEventListener('pointerleave', onTipLeave);
      if (typeof window !== 'undefined') window.removeEventListener('scroll', onScroll, { capture: true });
      if (active === controller) active = null;
    },
  };
  active = controller;
  return controller;
}

/** Test hook. */
export function __resetForTests() {
  if (active) active.destroy();
  active = null;
}
