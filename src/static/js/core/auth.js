/**
 * The API key: kept in memory and in KEYS.apiKey, sent only as an Authorization header by api.js.
 * A key the server refused is never sent again until a key is entered: the server counts every wrong key
 * toward OS_AUTH_FAILURE_LIMIT and then locks out this address, even with the right key.
 * @module core/auth
 */
import { KEYS, read, write, remove } from './storage.js';
import { app, emit } from './store.js';
import { post } from './api.js';

let key = '';
let refusedKey = null;

function publish() {
  app.set({ auth: { hasKey: !!key, refused: isRefused() } });
}

/** Read the stored key (trimmed) into memory and app.auth. */
export function init() {
  key = String(read(KEYS.apiKey, '') || '').trim();
  refusedKey = null;
  publish();
}

/** @returns {string} the key ('' when unset) */
export function getKey() { return key; }

/** @returns {boolean} */
export function hasKey() { return !!key; }

/**
 * A header can carry only Latin-1 characters: a pasted curly quote or a newline makes the key unsendable.
 * @param {string} value
 * @returns {boolean}
 */
export function isSendable(value) {
  try {
    // eslint-disable-next-line no-new
    new Headers({ Authorization: `Bearer ${value}` });
    return true;
  } catch {
    return false;
  }
}

/**
 * Store a new key (trimmed; empty removes it), clear the refusal and emit 'auth:changed'.
 * @param {string} value
 * @returns {{ok: true} | {ok: false, reason: 'unsendable'}} an unsendable key is stored but never sent
 */
export function setKey(value) {
  key = String(value ?? '').trim();
  refusedKey = null;
  if (key) write(KEYS.apiKey, key);
  else remove(KEYS.apiKey);
  publish();
  emit('auth:changed', { hasKey: !!key });
  return key && !isSendable(key) ? { ok: false, reason: 'unsendable' } : { ok: true };
}

/**
 * @returns {Record<string, string>} {} or {Authorization: 'Bearer …'}; {} when the key is refused or unsendable
 */
export function authHeaders() {
  if (!key || isRefused() || !isSendable(key)) return {};
  return { Authorization: `Bearer ${key}` };
}

/**
 * A key got a 401: it is not sent again until setKey(). Pass the key the refused request carried: a late
 * 401 for a key the listener has replaced since never marks the new key refused.
 * @param {string} [sentKey] defaults to the current key
 * @returns {boolean} whether the current key is now refused
 */
export function markRefused(sentKey = key) {
  if (!key || String(sentKey) !== key) return false;
  refusedKey = key;
  publish();
  return true;
}

/** @returns {boolean} whether the current key was refused */
export function isRefused() {
  return !!key && refusedKey === key;
}

/** @returns {string} what to tell the listener about a missing, unsendable or refused key */
export function refusalMessage() {
  if (!key) return 'This server needs an API key. Enter it in Settings › Connection.';
  if (!isSendable(key)) return "The API key contains a character that can't be sent, such as a curly quote. Check it in Settings › Connection.";
  return "The server didn't accept the API key. Check it in Settings › Connection.";
}

/**
 * A one-time WebSocket ticket: null without a key, else a fresh POST /api/auth/ws-ticket per call.
 * @returns {Promise<string|null>}
 */
export async function wsTicket() {
  if (!key) return null;
  const data = await post('/api/auth/ws-ticket');
  const ticket = data && data.ticket;
  if (!ticket) throw new Error('The server did not issue a connection ticket.');
  return String(ticket);
}

/** Test hook. */
export function __resetForTests() {
  key = '';
  refusedKey = null;
}
