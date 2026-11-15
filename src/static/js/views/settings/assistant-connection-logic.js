/**
 * Settings › AI assistant › Connection: the pure parts. No DOM, no
 * storage, no requests. The API they read and build for is src/routes/assistant.py with
 * src/assistant/connection.py (the key stays with the server it was set for):
 * - GET /api/assistant/connection: {enabled, locked, url, model, api_key_set, api_key_tls_checked, verify_tls,
 *   template_kwargs, max_tokens_field, extra_body, stream, sources: {field: 'env'|'saved'|'default'}}. Never the key
 *   itself. `api_key_tls_checked` (read-only): whether the key in use was set with the certificate
 *   check on, null with no key.
 *   `stream` is Stream replies (OS_ASSISTANT_STREAM, on by default).
 * - PUT takes any subset of the fields, and every field it is sent is saved: so a save sends only what changed on
 *   screen, and null puts a field back to .env. A key left out is kept for the same origin (scheme, host, port)
 *   and cleared when the URL moves to another one, or when the save turns the certificate check off for an https
 *   URL; "" clears it.
 * - POST …/check takes the same body as an unsaved candidate ({} checks the connection in use) and answers
 *   {reachable, status, models, listed, elapsed_ms}, or 502/504 with an AI error code.
 * - A 422 names its field in `param` (core/api.js puts it in ApiError.fields).
 * - Saved assistants (core/assistants.js): each is a connection view with its id and name, edited with the same
 *   rules; a new one's fields that aren't set follow .env.
 * @module views/settings/assistant-connection-logic
 */
import { elapsedText } from './assistant-logic.js';

/** The connection fields (src/assistant/connection.py FIELDS), in the order they're shown. */
export const FIELDS = Object.freeze(['url', 'api_key', 'model', 'verify_tls', 'template_kwargs', 'max_tokens_field', 'extra_body', 'stream']);
/** The names the token limit may be sent under (src/config.py ASSISTANT_MAX_TOKENS_FIELDS). */
export const MAX_TOKENS_FIELDS = Object.freeze(['max_tokens', 'max_completion_tokens']);
/** The server's limits (src/config.py ASSISTANT_URL_MAX_CHARS, presets.MODEL_MAX_CHARS, connection.py). */
export const URL_MAX_CHARS = 2000;
export const MODEL_MAX_CHARS = 200;
export const API_KEY_MAX_CHARS = 4096;
export const EXTRA_BODY_MAX_CHARS = 16000;
/** A saved assistant's name: 1 to 60 characters, unique ignoring case. */
export const NAME_MAX_CHARS = 60;

/** Each field's label, as the form shows it (field errors start with it). */
export const LABELS = Object.freeze({
  url: 'Server URL',
  api_key: 'API key',
  model: 'Model',
  verify_tls: 'Check TLS certificate',
  template_kwargs: 'Send thinking as template variables',
  max_tokens_field: 'Token limit sent as',
  extra_body: 'Extra body',
  stream: 'Stream replies',
  name: 'Name',
});

/** Where a value comes from, in words. */
export const SOURCE_WORDS = Object.freeze({ env: 'from .env', saved: 'saved here', default: 'default' });

/** The fields the form edits (the key is a field of its own: it is never shown). */
const VALUE_FIELDS = ['url', 'model', 'verify_tls', 'template_kwargs', 'max_tokens_field', 'extra_body', 'stream'];

const isObject = (value) => !!value && typeof value === 'object' && !Array.isArray(value);
const text = (value) => (typeof value === 'string' ? value : '');

/** PURE: deep equality for JSON values. */
function sameJson(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((item, i) => sameJson(item, b[i]));
  }
  if (isObject(a) && isObject(b)) {
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((key) => Object.prototype.hasOwnProperty.call(b, key) && sameJson(a[key], b[key]));
  }
  return false;
}

/** PURE: the extra body as the text area holds it: '' for none, else indented JSON. */
function extraBodyText(body) {
  return isObject(body) && Object.keys(body).length ? JSON.stringify(body, null, 2) : '';
}

/** PURE: the text area's JSON: {ok, value}; blank is {} (no extra fields), as the server reads it. */
function parseExtraBody(raw) {
  const s = text(raw).trim();
  if (!s) return { ok: true, value: {} };
  try {
    const value = JSON.parse(s);
    return isObject(value) ? { ok: true, value } : { ok: false };
  } catch {
    return { ok: false };
  }
}

/** PURE: a URL as the server keeps it: trimmed, without a trailing slash. */
const normUrl = (url) => text(url).trim().replace(/\/+$/, '');

/** PURE: whether two on-screen values of `field` mean the same thing. */
function sameField(field, a, b) {
  if (field === 'url') return normUrl(a) === normUrl(b);
  if (field === 'model') return text(a).trim() === text(b).trim();
  if (field === 'extra_body') {
    const x = parseExtraBody(a);
    const y = parseExtraBody(b);
    return x.ok && y.ok ? sameJson(x.value, y.value) : text(a).trim() === text(b).trim();
  }
  return a === b;
}

/**
 * PURE: a connection view's values as the form holds them: the URL and model as text, the switches as booleans,
 * the token field's name, and the extra body as JSON text ('' for none). Never the key.
 * @param {object} view
 */
export function viewValues(view) {
  const v = isObject(view) ? view : {};
  return {
    url: text(v.url),
    model: text(v.model),
    verify_tls: v.verify_tls !== false,
    template_kwargs: v.template_kwargs === true,
    max_tokens_field: MAX_TOKENS_FIELDS.includes(v.max_tokens_field) ? v.max_tokens_field : 'max_tokens',
    extra_body: extraBodyText(v.extra_body),
    stream: v.stream !== false,
  };
}

/** PURE: the fields (not the key) whose on-screen value differs from the view's, in the form's order. */
export function changedFields(view, values) {
  const loaded = viewValues(view);
  const now = isObject(values) ? values : {};
  return VALUE_FIELDS.filter((field) => !sameField(field, now[field], loaded[field]));
}

/** PURE: whether the form differs from the connection as loaded: a changed field, or a key typed. */
export function isDirty(view, values, key) {
  return changedFields(view, values).length > 0 || text(key).trim() !== '';
}

/**
 * PURE: the body of a save (PUT) or a Check: each field changed on screen, and the key when one is typed. A field
 * that wasn't changed is left out, so it keeps following .env (every field a PUT is sent is saved). The extra
 * body goes as its text, which the server parses ('' is none) and names in a 422 when it isn't a JSON object.
 */
export function changeBody(view, values, key) {
  const body = {};
  for (const field of changedFields(view, values)) {
    const value = values[field];
    body[field] = field === 'url' || field === 'model' || field === 'extra_body' ? text(value).trim() : value;
  }
  const typed = text(key).trim();
  if (typed) body.api_key = typed;
  return body;
}

/**
 * PURE: after one field changed on the server (Clear key, Use .env), the form's values over the new view: a
 * field still as it was loaded takes the new value, an edit stays, and `force` (the field that was changed)
 * takes the new value whatever is on screen.
 */
export function rebase(before, after, values, { force = '' } = {}) {
  const old = viewValues(before);
  const next = viewValues(after);
  const out = {};
  for (const field of VALUE_FIELDS) {
    const kept = field !== force && !sameField(field, values[field], old[field]);
    out[field] = kept ? values[field] : next[field];
  }
  return out;
}

/** PURE: a URL's origin (scheme, host and port, the default port left out), or '' when it isn't http(s). */
export function originOf(url) {
  try {
    const parsed = new URL(text(url).trim());
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.origin : '';
  } catch {
    return '';
  }
}

/** What the key field says when a save would clear the key, by keyAgainReason(). */
export const KEY_AGAIN = Object.freeze({
  origin: 'Enter the key again for a new server. The saved key is sent only to the server it was saved for, so saving without one clears it.',
  tls: 'Turning off the certificate check clears the saved key: enter it again to keep using it.',
});

/**
 * PURE: why a save would clear the key, so the key field asks for it again: '' (it wouldn't), 'origin' (the
 * URL field points at another origin) or 'tls' ("Check TLS
 * certificate", on as loaded, is turned off for an https URL). Only while a key is set and none is typed. A change of
 * path keeps the key, and so do an empty URL (the key is kept but sent nowhere), an http URL (no certificate), a
 * check that was off already (the key was set with it off) and a key the view says was set with the check off
 * (`api_key_tls_checked: false`), though the check was turned on since. A view without that field (an older server) keeps the guess from `verify_tls`. Another origin
 * wins over the check.
 * @returns {''|'origin'|'tls'}
 */
export function keyAgainReason(view, values, key) {
  if (!isObject(view) || !view.api_key_set || text(key).trim()) return '';
  const next = text(values && values.url).trim();
  if (!next) return '';
  const origin = originOf(next);
  if (origin !== originOf(view.url)) return 'origin';
  const setWithCheck = view.api_key_tls_checked !== false;
  if (setWithCheck && view.verify_tls !== false && values.verify_tls === false && origin.startsWith('https:')) return 'tls';
  return '';
}

/** PURE: whether to say "Enter the key again" (keyAgainReason() gives the words): a save would clear the key. */
export function keyAgain(view, values, key) {
  return keyAgainReason(view, values, key) !== '';
}

/** PURE: where a field's value comes from: 'env', 'saved' or 'default'. */
export function sourceOf(view, field) {
  const source = isObject(view) && isObject(view.sources) ? view.sources[field] : '';
  return Object.prototype.hasOwnProperty.call(SOURCE_WORDS, source) ? source : 'default';
}

/** PURE: a source in words. */
export function sourceText(source) {
  return SOURCE_WORDS[source] || SOURCE_WORDS.default;
}

const sentenceCase = (s) => {
  const t = String(s || '').trim();
  if (!t) return '';
  const first = t.charAt(0).toUpperCase() + t.slice(1);
  return /[.!?]$/.test(first) ? first : `${first}.`;
};

/**
 * PURE: where a 422 goes and what it says there: the field the server named (`param`), the message starting with
 * the field's label instead of its API name. A param that isn't a field of the form (body, …) gives null.
 * @returns {{field: string, text: string}|null}
 */
export function fieldProblem(param, message) {
  const field = String(param || '');
  if (!FIELDS.includes(field) && field !== 'name') return null;
  const msg = String(message || '').trim();
  const body = msg.startsWith(field) && !/[\w]/.test(msg.charAt(field.length)) ? `${LABELS[field]}${msg.slice(field.length)}` : msg;
  return { field, text: sentenceCase(body) || `${LABELS[field]} wasn't accepted.` };
}

/**
 * PURE: a Check that reached the server, in words: {tone: 'ok'|'warn', title, message, status, models}. `models`
 * are the ids the server listed (strings, once each), for the dropdown; `status` is the short status line.
 */
export function checkResult(answer) {
  const a = isObject(answer) ? answer : {};
  const took = elapsedText(a.elapsed_ms);
  const code = Number(a.status) || 0;
  const ids = Array.isArray(a.models) ? [...new Set(a.models.filter((m) => typeof m === 'string' && m.trim()))] : [];
  if (code === 200 && a.listed) {
    if (!ids.length) {
      return { tone: 'ok', title: 'Reachable', message: `It answered in ${took}, but lists no models. Type the model name in Model.`, status: `Reachable in ${took}: no models listed`, models: [] };
    }
    const n = `${ids.length} model${ids.length === 1 ? '' : 's'}`;
    return { tone: 'ok', title: 'Reachable', message: `It answered in ${took} and lists ${n}. Pick one under Models on the server, or type a name in Model.`, status: `Reachable in ${took}: ${n}`, models: ids };
  }
  if (code === 200) {
    return { tone: 'warn', title: 'Reachable', message: `It answered in ${took}, but not with a list of models. Type the model name in Model.`, status: `Reachable in ${took}`, models: [] };
  }
  const status = `Reached in ${took}: HTTP ${code}`;
  if (code === 401 || code === 403) {
    return { tone: 'warn', title: 'The AI server wants an API key', message: `It answered HTTP ${code} to the model list. Enter its key under API key, or check the one saved.`, status, models: [] };
  }
  if (code === 404) {
    return { tone: 'warn', title: 'Reachable, but no model list there', message: 'It answered HTTP 404. Check that the address ends in /v1.', status, models: [] };
  }
  return { tone: 'warn', title: `The AI server answered HTTP ${code}`, message: `It was reached in ${took}, but didn't list its models. Some servers don't: type the model name in Model.`, status, models: [] };
}

/** What a Check that failed says, by the server's error code. */
const CHECK_FAILURES = Object.freeze({
  assistant_unreachable: ["Couldn't reach the AI server", "Check the address and that the server is running. Open Speech's server makes the call, so the address must work from there."],
  assistant_timeout: ["The AI server didn't answer in time", 'Nothing came back within 10 seconds. Check the address and the port.'],
  assistant_bad_response: ["The AI server's answer can't be read", "It may not be an OpenAI-compatible server, or its answer was too large."],
});

/** PURE: {title, message} for a Check that failed with an AI error code, or null for any other error. */
export function checkFailure(code) {
  const entry = CHECK_FAILURES[String(code || '')];
  return entry ? { title: entry[0], message: entry[1] } : null;
}

/**
 * PURE: the read-only list of a locked connection: each field's value in words, with where it comes from. The key
 * only as set or not set.
 * @returns {{term: string, value: string, mono?: boolean, source: string}[]}
 */
export function lockedFacts(view) {
  const v = isObject(view) ? view : {};
  const values = viewValues(v);
  const extra = isObject(v.extra_body) && Object.keys(v.extra_body).length ? JSON.stringify(v.extra_body) : '';
  const fact = (field, value, mono = false) => ({ term: LABELS[field], value, ...(mono ? { mono: true } : {}), source: sourceOf(v, field) });
  return [
    fact('url', values.url || 'None: the assistant is off', !!values.url),
    fact('model', values.model || 'None: the AI server picks', !!values.model),
    fact('api_key', v.api_key_set ? 'Set' : 'Not set'),
    fact('verify_tls', values.verify_tls ? 'On' : 'Off'),
    fact('template_kwargs', values.template_kwargs ? 'On' : 'Off'),
    fact('max_tokens_field', values.max_tokens_field, true),
    fact('extra_body', extra || 'None', !!extra),
    fact('stream', values.stream ? 'On' : 'Off'),
  ];
}

/**
 * PURE: the body of a Check for an assistant not saved yet, over the connection in use: every field, the ones not
 * changed on screen as null (so they follow .env, as the new assistant's will), and the key as typed or null.
 */
export function candidateBody(view, values, key) {
  const body = changeBody(view, values, key);
  for (const field of FIELDS) if (!(field in body)) body[field] = null;
  return body;
}

/** PURE: the body that puts every field saved for an assistant back to .env (Revert to .env). */
export function revertBody(view) {
  const body = {};
  for (const field of FIELDS) if (sourceOf(view, field) === 'saved') body[field] = null;
  return body;
}

/**
 * PURE: the .env connection as far as the saved assistants show it, for the form of a new one: each field from an
 * assistant that doesn't save it (its value is .env's), else the default. The key counts as set when an assistant
 * uses the .env key.
 * @param {object[]} assistants
 */
export function envView(assistants) {
  const list = Array.isArray(assistants) ? assistants.filter(isObject) : [];
  const defaults = viewValues({});
  const out = { enabled: false, locked: false, api_key_set: false, api_key_tls_checked: null, sources: {} };
  for (const field of VALUE_FIELDS) {
    const from = list.find((a) => sourceOf(a, field) !== 'saved');
    if (from) {
      out[field] = from[field];
      out.sources[field] = sourceOf(from, field);
    } else {
      out[field] = field === 'extra_body' ? {} : defaults[field];
      out.sources[field] = 'default';
    }
  }
  const keyed = list.find((a) => sourceOf(a, 'api_key') === 'env' && a.api_key_set);
  out.api_key_set = !!keyed;
  out.api_key_tls_checked = keyed ? keyed.api_key_tls_checked ?? null : null;
  out.sources.api_key = keyed ? 'env' : 'default';
  out.enabled = !!text(out.url).trim();
  return out;
}

/**
 * PURE: the name offered for a new assistant: its model, else its server's host, else "Assistant"; with " 2",
 * " 3"… while another assistant has it (names are unique ignoring case).
 * @param {object} values the form's values
 * @param {string[]} taken the saved assistants' names
 */
export function suggestName(values, taken = []) {
  const v = isObject(values) ? values : {};
  let host = '';
  try { host = new URL(text(v.url).trim()).host; } catch { host = ''; }
  const base = (text(v.model).trim() || host || 'Assistant').slice(0, NAME_MAX_CHARS);
  const used = new Set((Array.isArray(taken) ? taken : []).map((n) => String(n).toLowerCase()));
  if (!used.has(base.toLowerCase())) return base;
  for (let n = 2; ; n += 1) {
    const tail = ` ${n}`;
    const name = `${base.slice(0, NAME_MAX_CHARS - tail.length)}${tail}`;
    if (!used.has(name.toLowerCase())) return name;
  }
}

/** PURE: a saved assistant's line in the list: its server's host and its model, in words. */
export function assistantMeta(assistant) {
  const a = isObject(assistant) ? assistant : {};
  let host = '';
  try { host = new URL(text(a.url).trim()).host; } catch { host = ''; }
  const model = text(a.model).trim();
  return `${host || 'No server URL'} · ${model || 'the AI server picks the model'}`;
}
