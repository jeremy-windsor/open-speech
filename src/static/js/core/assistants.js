/**
 * Saved AI assistants: each one a named AI server connection, one of them in use. The API is
 * src/routes/assistant.py:
 * - GET /api/assistant/assistants: {active: id|null, locked, assistants: [connection view + {id, name, active}],
 *   env: the .env connection's view}. A connection view never holds
 *   the key, only `api_key_set`, with `sources` and `enabled`.
 * - POST …/assistants (name and connection fields) adds one (the first one saved is in use); PUT …/{id} changes
 *   one (the connection PUT's rules: null follows .env, api_key "" clears it); DELETE …/{id}; POST …/{id}/activate
 *   puts one in use; POST …/{id}/check checks one, with the same body as the connection Check.
 * @module core/assistants
 */
import { get, post, put, del } from './api.js';

export const ASSISTANTS_PATH = '/api/assistant/assistants';

const one = (id, rest = '') => `${ASSISTANTS_PATH}/${encodeURIComponent(id)}${rest}`;
const isObject = (value) => !!value && typeof value === 'object' && !Array.isArray(value);

/** PURE: a list answer with only well-formed assistants, `active` null when it names none of them. */
export function normalizeList(data) {
  const d = isObject(data) ? data : {};
  const assistants = (Array.isArray(d.assistants) ? d.assistants : [])
    .filter((a) => isObject(a) && typeof a.id === 'string' && a.id)
    .map((a) => ({ ...a, name: typeof a.name === 'string' && a.name ? a.name : a.id }));
  const named = typeof d.active === 'string' && assistants.some((a) => a.id === d.active) ? d.active : null;
  const flagged = assistants.find((a) => a.active === true);
  const active = named || (flagged ? flagged.id : null);
  return {
    active, locked: d.locked === true, assistants: assistants.map((a) => ({ ...a, active: a.id === active })),
    env: isObject(d.env) ? d.env : null,
  };
}

/**
 * GET the saved assistants, or null from a server that doesn't have them (404).
 * @returns {Promise<{active: string|null, locked: boolean, assistants: object[]}|null>}
 */
export async function listAssistants() {
  try {
    return normalizeList(await get(ASSISTANTS_PATH, { accept: 'json' }));
  } catch (error) {
    if (error && error.status === 404) return null;
    throw error;
  }
}

export const createAssistant = (body) => post(ASSISTANTS_PATH, body, { accept: 'json' });
export const updateAssistant = (id, body) => put(one(id), body, { accept: 'json' });
export const deleteAssistant = (id) => del(one(id), { accept: 'json' });
export const activateAssistant = (id) => post(one(id, '/activate'), {}, { accept: 'json' });
export const checkAssistant = (id, body = {}) => post(one(id, '/check'), body, { accept: 'json' });

/** PURE: a URL's host ("ai-server.lan:8080"), or '' when it isn't an http(s) URL. */
export function hostOf(url) {
  try {
    const parsed = new URL(String(url || '').trim());
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.host : '';
  } catch {
    return '';
  }
}

/** PURE: the assistant with `id` in a list answer, or null. */
export function findAssistant(list, id) {
  const all = list && Array.isArray(list.assistants) ? list.assistants : [];
  return (id && all.find((a) => a.id === id)) || null;
}

/**
 * PURE: the assistant a preset uses: the one it names while it is saved, else the one in use. Null when nothing
 * is saved.
 * @param {{active: string|null, assistants: object[]}|null} list
 * @param {string} [presetAssistant] the preset's `assistant` ('' is the one in use)
 */
export function assistantForPreset(list, presetAssistant = '') {
  return findAssistant(list, presetAssistant) || findAssistant(list, list && list.active);
}
