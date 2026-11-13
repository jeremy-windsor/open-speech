// Accessibility checks on markup built at runtime: the JS twin of the Python audits in tests/ui_support.py.
import assert from 'node:assert/strict';
import { allElements } from './dom.js';

const LABELABLE = new Set(['input', 'select', 'textarea', 'meter', 'output', 'progress']);

function hiddenFromAt(el, root) {
  for (let n = el; n && n !== root?.parentNode; n = n.parentNode) {
    if (n.nodeType !== 1) continue;
    if (n.getAttribute('aria-hidden') === 'true') return true;
  }
  return false;
}

function textOf(node) {
  if (node.nodeType === 3) return node.data;
  if (node.nodeType !== 1) return '';
  if (node.getAttribute('aria-hidden') === 'true') return '';
  if (node.localName === 'svg') {
    const label = node.getAttribute('aria-label');
    return node.getAttribute('role') === 'img' && label ? label : '';
  }
  if (node.localName === 'img') return node.getAttribute('alt') || '';
  const label = node.getAttribute('aria-label');
  if (label && label.trim()) return label;
  return node.childNodes.map(textOf).join('');
}

/** Where an element's id references resolve: its document, or the top of a detached subtree. */
function scopeOf(el) {
  const top = typeof el.getRootNode === 'function' ? el.getRootNode() : null;
  return top && typeof top.querySelectorAll === 'function' ? top : el.ownerDocument;
}

function byIds(scope, ids) {
  const find = (id) => (typeof scope.getElementById === 'function'
    ? scope.getElementById(id)
    : scope.querySelectorAll('[id]').find((n) => n.id === id) || null);
  return String(ids).split(/\s+/).filter(Boolean).map(find).filter(Boolean);
}

/**
 * The accessible name of an element (a practical subset of the accname algorithm): aria-labelledby,
 * aria-label, <label for>, a wrapping <label>, the text of buttons/links/options/summary, then title.
 * aria-hidden content is ignored; a placeholder never counts.
 * @returns {string}
 */
export function accessibleName(el) {
  const scope = scopeOf(el);
  const labelledby = el.getAttribute('aria-labelledby');
  if (labelledby) {
    const name = byIds(scope, labelledby).map((n) => textOf(n).trim()).join(' ').trim();
    if (name) return name;
  }
  const aria = el.getAttribute('aria-label');
  if (aria && aria.trim()) return aria.trim();
  const tag = el.localName;
  if (LABELABLE.has(tag) || tag === 'button') {
    if (el.id) {
      const labels = scope.querySelectorAll('label').filter((l) => l.getAttribute('for') === el.id);
      const text = labels.map((l) => textOf(l).trim()).join(' ').trim();
      if (text) return text;
    }
    for (let p = el.parentElement; p; p = p.parentElement) {
      if (p.localName === 'label') {
        const text = p.childNodes.filter((c) => c !== el).map(textOf).join('').trim();
        if (text) return text;
        break;
      }
    }
  }
  if (tag === 'button' || tag === 'a' || tag === 'summary' || tag === 'option' || ['button', 'link', 'option', 'menuitem', 'menuitemradio', 'menuitemcheckbox', 'tab', 'radio', 'checkbox', 'switch', 'heading', 'treeitem'].includes(el.getAttribute('role') || '')) {
    const text = el.childNodes.map(textOf).join('').replace(/\s+/g, ' ').trim();
    if (text) return text;
  }
  if (tag === 'input' && /^(submit|reset|button)$/.test(el.type)) {
    const v = el.getAttribute('value');
    if (v) return v;
  }
  const title = el.getAttribute('title');
  if (title && title.trim()) return title.trim();
  // A placeholder is never the label, so it does not count here.
  return '';
}

/**
 * Assert that every input/select/textarea/button in a built subtree has an accessible name, every button
 * has an explicit type, and icon-only buttons are labelled with aria-label/aria-labelledby. Elements under
 * aria-hidden="true" and hidden inputs are skipped.
 * @param {object} root a FakeElement or document
 */
export function assertAccessibleNames(root) {
  const problems = [];
  const scope = root.nodeType === 9 ? root.documentElement : root;
  const elements = [scope, ...allElements(scope)];
  for (const el of elements) {
    if (el.nodeType !== 1) continue;
    const tag = el.localName;
    if (!['input', 'select', 'textarea', 'button'].includes(tag)) continue;
    if (tag === 'input' && el.type === 'hidden') continue;
    if (hiddenFromAt(el, scope)) continue;
    const name = accessibleName(el);
    const where = describe(el);
    if (!name) problems.push(`${where} has no accessible name`);
    if (tag === 'button') {
      if (!el.hasAttribute('type')) problems.push(`${where} has no explicit type`);
      const visibleText = el.childNodes.map(textOf).join('').trim();
      if (!visibleText && !el.getAttribute('aria-label') && !el.getAttribute('aria-labelledby')) {
        problems.push(`${where} is icon-only without aria-label`);
      }
    }
  }
  assert.deepEqual(problems, [], `Accessible-name problems:\n${problems.join('\n')}`);
}

function describe(el) {
  const id = el.id ? `#${el.id}` : '';
  const cls = el.getAttribute('class') ? `.${el.getAttribute('class').split(/\s+/).join('.')}` : '';
  return `<${el.localName}${id}${cls}>`;
}

/**
 * The live regions inside a subtree: role=status/alert/log/marquee/timer or an aria-live attribute.
 * @returns {{el: object, role: string, live: string}[]}
 */
export function liveRegions(root) {
  const scope = root.nodeType === 9 ? root.documentElement : root;
  const implicit = { status: 'polite', alert: 'assertive', log: 'polite', marquee: 'off', timer: 'off' };
  const out = [];
  for (const el of [scope, ...allElements(scope)]) {
    const role = el.getAttribute('role') || '';
    const live = el.getAttribute('aria-live') || implicit[role] || '';
    if (role in implicit || el.hasAttribute('aria-live')) out.push({ el, role, live });
  }
  return out;
}

/**
 * The elements a Tab key would reach, in order: natively focusable or tabindex >= 0, not disabled and not
 * inside a hidden/inert subtree or closed dialog.
 */
export function focusables(root) {
  const scope = root.nodeType === 9 ? root.documentElement : root;
  const els = allElements(scope).filter((el) => el._focusable && el._focusable() && el.tabIndex >= 0);
  const positive = els.filter((el) => el.tabIndex > 0).sort((a, b) => a.tabIndex - b.tabIndex);
  return [...positive, ...els.filter((el) => el.tabIndex === 0)];
}
