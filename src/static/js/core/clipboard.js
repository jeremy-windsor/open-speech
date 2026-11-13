/**
 * Copy text: the Clipboard API in secure contexts, else a hidden textarea and execCommand('copy') (plain
 * HTTP on a LAN). Focus returns to the element that had it.
 * @module core/clipboard
 */

function fallbackCopy(text) {
  const previous = document.activeElement;
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.setAttribute('aria-hidden', 'true');
  area.setAttribute('tabindex', '-1');
  area.className = 'visually-hidden';
  document.body.appendChild(area);
  let ok = false;
  try {
    area.focus({ preventScroll: true });
    area.select();
    if (typeof area.setSelectionRange === 'function') area.setSelectionRange(0, text.length);
    ok = typeof document.execCommand === 'function' && !!document.execCommand('copy');
  } catch {
    ok = false;
  } finally {
    area.remove();
    if (previous && previous !== document.body && typeof previous.focus === 'function') previous.focus({ preventScroll: true });
  }
  return ok;
}

/**
 * Copy text to the clipboard.
 * @param {string} text
 * @returns {Promise<boolean>} false when nothing was copied (empty text, or both paths failed)
 */
export async function copyText(text) {
  const value = String(text ?? '');
  if (!value) return false;
  const clipboard = globalThis.isSecureContext !== false ? globalThis.navigator?.clipboard : null;
  if (clipboard && typeof clipboard.writeText === 'function') {
    try {
      await clipboard.writeText(value);
      return true;
    } catch {
      // permission refused or not focused: try the fallback
    }
  }
  return fallbackCopy(value);
}
