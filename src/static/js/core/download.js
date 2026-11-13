/**
 * Save files. Server files that need the API key are fetched with it; without a key a plain link is used.
 * @module core/download
 */
import { request } from './api.js';
import { hasKey } from './auth.js';

// WebKit starts a blob download asynchronously: revoking the URL too early fails it ("WebKitBlobResource error 1").
const REVOKE_AFTER_MS = 60000;

function clickLink(href, filename) {
  const a = document.createElement('a');
  a.href = href;
  a.download = filename || '';
  a.setAttribute('download', filename || '');
  a.rel = 'noopener';
  a.className = 'visually-hidden';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/**
 * Save a Blob as a file; the object URL is revoked a minute later (iOS Safari reads it asynchronously).
 * @param {Blob} blob
 * @param {string} filename
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  clickLink(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), REVOKE_AFTER_MS);
}

/**
 * Save a server file. With a key it is fetched with the key into a Blob; without one a plain
 * <a download> link is used. Same-origin paths only.
 * @param {string} path e.g. '/api/conversations/<id>/audio'
 * @param {string} filename
 */
export async function downloadFromApi(path, filename) {
  const url = new URL(String(path), location.href);
  if (url.origin !== location.origin) throw new TypeError('cross-origin request blocked');
  if (!hasKey()) {
    clickLink(`${url.pathname}${url.search}`, filename);
    return;
  }
  const blob = await request(`${url.pathname}${url.search}`, { accept: 'blob' });
  downloadBlob(blob, filename);
}
