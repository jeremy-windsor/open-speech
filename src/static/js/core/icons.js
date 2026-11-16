/**
 * The icon sprite: one <symbol id="i-NAME"> per icon in /static/icons/sprite.svg.
 * @module core/icons
 */
import { SafeHtml, escapeHtml } from './dom.js';

export const SPRITE = '/static/icons/sprite.svg';

/** Every symbol id in the sprite, without the 'i-' prefix (a test keeps them in step). */
export const ICONS = Object.freeze([
  // brand and navigation
  'logo', 'home', 'book', 'speak', 'mic', 'voices', 'studio', 'history', 'models', 'settings', 'more',
  'chevron-left', 'chevron-right', 'chevron-down', 'chevron-up', 'arrow-up', 'arrow-down', 'external',
  // playback
  'play', 'pause', 'stop', 'skip-back', 'skip-fwd', 'prev', 'next', 'speed', 'sleep', 'chapters', 'type',
  'voice', 'record', 'waveform',
  // actions
  'upload', 'download', 'copy', 'code', 'plus', 'minus', 'x', 'check', 'search', 'filter', 'sort', 'trash',
  'edit', 'star', 'star-fill', 'refresh', 'retry', 'link', 'file', 'paste', 'dots',
  // settings and system
  'sliders', 'gpu', 'cpu', 'server', 'offline', 'key', 'lock', 'eye', 'eye-off', 'sun', 'moon', 'monitor',
  'keyboard',
  // status
  'alert', 'info', 'help',
  // other
  'grip', 'wand', 'layers',
  // Talk, the AI assistant's view
  'talk',
]);

const KNOWN = new Set(ICONS);

/**
 * An inline SVG that references the sprite. Decorative by default (aria-hidden); pass `label` only for a
 * meaningful standalone icon, which then gets role="img" and that aria-label.
 * @param {string} name one of ICONS
 * @param {{className?: string, label?: string}} [options]
 * @returns {SafeHtml}
 * @throws {TypeError} for an unknown icon name
 */
export function icon(name, { className = 'ico', label = '' } = {}) {
  if (!KNOWN.has(name)) throw new TypeError(`Unknown icon: ${name}`);
  const a11y = label
    ? `role="img" aria-label="${escapeHtml(label)}"`
    : 'aria-hidden="true"';
  return new SafeHtml(
    `<svg class="${escapeHtml(className)}" ${a11y} focusable="false"><use href="${SPRITE}#i-${name}"></use></svg>`,
  );
}
