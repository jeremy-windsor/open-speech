/**
 * Settings › AI assistant: the pure parts. No DOM, no storage, no
 * requests. The API they read and build for is src/routes/assistant.py:
 * - GET /api/assistant: {enabled, host, model, server_settings: {OS_ASSISTANT_*: whether it differs from its
 *   default}, assistant: {id, name} of the saved assistant in use}. Never the URL's path or the key.
 * - A preset's answer: {id, name, builtin, customized, inherited: [fields from .env or the code], ...its
 *   effective values}. PUT replaces its editable fields: a field left out or empty comes from .env or the code.
 * - A 422 names the field in `param` (core/api.js puts it in ApiError.fields).
 * @module views/settings/assistant-logic
 */

/**
 * The fields a preset may set (src/assistant/presets.py TUNABLE_FIELDS), in the order they're shown.
 * thinking_budget goes to the AI as reasoning_budget_tokens; 0 sends none.
 */
export const TUNABLE = Object.freeze([
  'model', 'system_prompt', 'user_template', 'thinking', 'reasoning_effort', 'thinking_budget', 'show_thinking',
  'max_tokens', 'temperature', 'top_p', 'timeout_s', 'history_turns', 'fillers', 'filler_after_ms',
  'filler_every_ms', 'filler_max', 'speak_errors', 'error_sentences', 'chunk_chars',
]);

/**
 * The fields of how a preset shows in the Ask AI sheet:
 * never sent to the AI and never an override. A PUT that leaves one out keeps it; '' or [] is the default again.
 */
export const ACTION_FIELDS = Object.freeze(['label', 'inputs', 'output']);
export const INPUT_KINDS = Object.freeze(['text', 'transcript', 'document']);
/**
 * The saved assistant a preset uses (an id; '' is the one in use). Like the Ask AI fields it is never an override:
 * a save sends it only when changed, and a copy keeps it.
 */
export const ASSISTANT_FIELD = 'assistant';
const KEPT_FIELDS = [...ACTION_FIELDS, ASSISTANT_FIELD];
export const OUTPUTS = Object.freeze(['text', 'speech', 'document']);

/** Number fields and the ones that may be empty ("the server's default"). */
export const NUMBER_FIELDS = Object.freeze([
  'thinking_budget', 'max_tokens', 'temperature', 'top_p', 'timeout_s', 'history_turns', 'filler_after_ms', 'filler_every_ms', 'filler_max', 'chunk_chars',
]);
const TEXT_FIELDS = ['model', 'system_prompt', 'user_template', 'reasoning_effort'];
const BOOL_FIELDS = ['show_thinking', 'speak_errors'];
export const THINKING_MODES = Object.freeze(['default', 'on', 'off']);

/** Fields only a voice conversation uses, and only a book job. A preset of your own shows both. */
export const VOICE_FIELDS = Object.freeze(['history_turns', 'fillers', 'filler_after_ms', 'filler_every_ms', 'filler_max', 'speak_errors', 'error_sentences']);
export const BOOK_FIELDS = Object.freeze(['chunk_chars']);
/** The fields whose default differs between the voice and the book fallbacks (presets._defaults). */
const PURPOSE_FIELDS = ['system_prompt', 'timeout_s'];

export const MAX_FILLERS = 10;
export const FILLER_MAX_CHARS = 80;
export const SENTENCE_MAX_CHARS = 300;
export const PROMPT_MAX_CHARS = 8000;

/** The AI error codes in their fixed order, with their default sentences (src/assistant/errors.py). */
export const DEFAULT_ERROR_SENTENCES = Object.freeze({
  assistant_unreachable: "I couldn't reach the AI server.",
  assistant_timeout: 'The AI took too long to answer.',
  assistant_http_error: 'The AI server returned an error.',
  assistant_bad_response: "The AI didn't send back an answer.",
  assistant_empty_reply: "The AI didn't send back an answer.",
  assistant_reply_cut_off: 'The answer was cut off before it started.',
});
export const ERROR_CODES = Object.freeze(Object.keys(DEFAULT_ERROR_SENTENCES));

/** What each error code means, as the label of its sentence. */
export const ERROR_LABELS = Object.freeze({
  assistant_unreachable: "Can't reach the AI server",
  assistant_timeout: 'No answer in time',
  assistant_http_error: 'The AI server returned an error',
  assistant_bad_response: 'An answer it can\'t read',
  assistant_empty_reply: 'An empty answer',
  assistant_reply_cut_off: 'The answer was cut off',
});

/** The label of each field, as the form shows it (field errors start with it). */
export const FIELD_LABELS = Object.freeze({
  name: 'Name',
  label: 'Action name',
  inputs: 'Offered for',
  output: 'First answer choice',
  assistant: 'Assistant',
  model: 'Model',
  system_prompt: 'System prompt',
  user_template: 'User template',
  thinking: 'Thinking mode',
  reasoning_effort: 'Reasoning effort',
  thinking_budget: 'Thinking budget (tokens)',
  show_thinking: 'Show thinking',
  max_tokens: 'Max tokens',
  temperature: 'Temperature',
  top_p: 'Top P',
  timeout_s: 'Timeout (seconds)',
  history_turns: 'History turns',
  fillers: 'Phrases',
  filler_after_ms: 'First phrase after (ms)',
  filler_every_ms: 'Then every (ms)',
  filler_max: 'At most',
  speak_errors: 'Say errors out loud',
  error_sentences: 'Error sentences',
  chunk_chars: 'Piece size (characters)',
  text: 'Message',
});

const isText = (value) => typeof value === 'string';
const number = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);

/**
 * PURE: which fields a preset shows: the built-in voice preset hides the book-only ones, the book preset the
 * voice-only ones, and a preset of your own (used for either) shows both.
 * @param {string} id
 * @returns {'voice'|'book'|'both'}
 */
export function presetUse(id) {
  if (id === 'voice') return 'voice';
  if (id === 'book') return 'book';
  return 'both';
}

/** PURE: the placeholders a preset's user template may use (the built-in voice preset takes only {text}). */
export function placeholders(id) {
  return id === 'voice' ? ['text'] : ['text', 'title', 'author', 'chapter'];
}

/** PURE: the error sentences shown for a preset: every known code, then any other the server sent. */
function sentencesOf(given) {
  const map = given && typeof given === 'object' && !Array.isArray(given) ? given : {};
  const codes = [...ERROR_CODES, ...Object.keys(map).filter((code) => !ERROR_CODES.includes(code) && /^[a-z0-9_]+$/.test(code))];
  const out = {};
  for (const code of codes) {
    const sentence = isText(map[code]) ? map[code].trim() : '';
    out[code] = sentence || DEFAULT_ERROR_SENTENCES[code] || '';
  }
  return out;
}

/**
 * PURE: the Ask AI fields as the form holds them: the label as text, the input kinds a preset is
 * offered for (known ones, once each, in the form's order) and the first answer choice ('text' when unknown).
 */
function actionValues(v) {
  const given = Array.isArray(v.inputs) ? v.inputs : [];
  return {
    label: isText(v.label) ? v.label : '',
    inputs: INPUT_KINDS.filter((kind) => given.includes(kind)),
    output: OUTPUTS.includes(v.output) ? v.output : 'text',
    assistant: isText(v.assistant) ? v.assistant : '',
  };
}

/**
 * PURE: a preset's values as the form holds them: text fields as text ('' for none), numbers or null (the
 * server's default), booleans, the phrases trimmed, and every error code's sentence (the default when none).
 * @param {object} view a preset's API answer
 * @returns {object}
 */
export function presetValues(view) {
  const v = view && typeof view === 'object' ? view : {};
  const out = { name: isText(v.name) ? v.name : '', ...actionValues(v) };
  for (const field of TEXT_FIELDS) out[field] = isText(v[field]) ? v[field] : '';
  out.thinking = THINKING_MODES.includes(v.thinking) ? v.thinking : 'default';
  for (const field of BOOL_FIELDS) out[field] = v[field] !== false;
  for (const field of NUMBER_FIELDS) out[field] = number(v[field]);
  out.fillers = Array.isArray(v.fillers) ? v.fillers.filter(isText).map((s) => s.trim()).filter(Boolean) : [];
  out.error_sentences = sentencesOf(v.error_sentences);
  return out;
}

/**
 * PURE: the form's raw contents as values. Raw holds the fields' strings (numbers as typed), the switches as
 * booleans, the phrases as an array and the sentences by code. An empty number is the server's default (null);
 * one that isn't a number is a problem. Blank phrases are dropped; an empty sentence is the default one.
 * @param {object} raw
 * @returns {{values: object, problems: {field: string, text: string}[]}}
 */
export function formValues(raw) {
  const r = raw && typeof raw === 'object' ? raw : {};
  const values = { name: isText(r.name) ? r.name : '', ...actionValues(r) };
  const problems = [];
  for (const field of TEXT_FIELDS) values[field] = isText(r[field]) ? r[field] : '';
  values.thinking = THINKING_MODES.includes(r.thinking) ? r.thinking : 'default';
  for (const field of BOOL_FIELDS) values[field] = r[field] !== false;
  for (const field of NUMBER_FIELDS) {
    const text = String(r[field] ?? '').trim();
    if (!text) {
      values[field] = null;
      continue;
    }
    const n = Number(text);
    if (!Number.isFinite(n)) {
      problems.push({ field, text: `${FIELD_LABELS[field]} must be a number.` });
      values[field] = null;
    } else {
      values[field] = n;
    }
  }
  values.fillers = Array.isArray(r.fillers) ? r.fillers.map((s) => String(s ?? '').trim()).filter(Boolean) : [];
  values.error_sentences = sentencesOf(r.error_sentences);
  return { values, problems };
}

/** PURE: deep equality for the values (strings, numbers, null, booleans, arrays and plain objects). */
export function sameValue(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((item, i) => sameValue(item, b[i]));
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((key) => Object.prototype.hasOwnProperty.call(b, key) && sameValue(a[key], b[key]));
  }
  return false;
}

/** PURE: whether the form differs from the preset as it was loaded. */
export function isDirty(view, values) {
  const loaded = presetValues(view);
  return ['name', ...KEPT_FIELDS, ...TUNABLE].some((field) => !sameValue(values ? values[field] : undefined, loaded[field]));
}

/** A field's value as the API takes it: only the sentences that differ from the default are sent. */
function wire(field, value) {
  if (field !== 'error_sentences') return value;
  const out = {};
  for (const [code, sentence] of Object.entries(value || {})) {
    if (sentence && sentence !== DEFAULT_ERROR_SENTENCES[code]) out[code] = sentence;
  }
  return out;
}

/**
 * PURE: the body of a save (PUT) or of a new copy (POST): the name, plus each field that is the preset's own
 * or was changed on screen. A field that comes from .env or the code and wasn't touched is left out, so it keeps
 * following .env. A cleared field goes as '' or null, which makes it follow .env again.
 * @param {object} view the preset as loaded
 * @param {object} values the form's values
 * @param {{copy?: boolean}} [options] copy: a new preset made from this one (a preset of your own falls back
 *   to the voice defaults, so a copy of the book preset carries its book system prompt and job timeout)
 */
export function saveBody(view, values, { copy = false } = {}) {
  const loaded = presetValues(view);
  const inherited = new Set(Array.isArray(view && view.inherited) ? view.inherited : []);
  const body = { name: String((values && values.name) || '').trim() };
  // The Ask AI fields go when changed (a save that leaves one out keeps it on the server); a cleared
  // label or no kind checked is the default again ('' and []). A copy keeps where it is offered and its first
  // answer choice, and the label only when it is the preset's own: a preset of yours whose label isn't its name (a
  // built-in's label, such as the voice preset's "Answer", is its default, and a copy gets its own name's). A
  // server whose presets don't have them (an older server) would refuse them, so they go only when it has them.
  for (const field of KEPT_FIELDS) {
    if (!values || !(field in values) || !view || !(field in view)) continue;
    const changed = !sameValue(values[field], loaded[field]);
    const keep = copy && (field !== 'label' || (!view.builtin && loaded.label && loaded.label !== loaded.name));
    if (changed || keep) body[field] = field === 'label' ? String(values.label || '').trim() : values[field];
  }
  for (const field of TUNABLE) {
    const own = !inherited.has(field);
    const changed = !sameValue(values[field], loaded[field]);
    const purpose = copy && view && view.id === 'book' && PURPOSE_FIELDS.includes(field);
    if (own || changed || purpose) body[field] = wire(field, values[field]);
  }
  return body;
}

/**
 * PURE: the body of POST /api/assistant/test: the preset as it is on screen, saved or not, as overrides of the
 * saved preset: every preset field, never the name (a request can't override it; another key answers 422). A
 * field emptied on screen goes as null, so the Test tries .env's or the code's default, as the saved preset
 * would; left out, the server would try the saved value. The phrases are a value
 * ([] is none); the sentences go as the ones that differ from the default (a code left out is the default),
 * null when none does. The text only when one was typed (the server's default is "Say hello in one sentence.").
 * @param {string} id
 * @param {string} text
 * @param {object} values
 */
export function testBody(id, text, values) {
  const overrides = {};
  for (const field of TUNABLE) {
    const value = wire(field, values[field]);
    const empty = value === null || value === undefined || value === ''
      || (field === 'error_sentences' && !Object.keys(value).length);
    overrides[field] = empty ? null : value;
  }
  const body = { preset: id, overrides };
  if (String(text ?? '').trim()) body.text = text;
  return body;
}

/** How each server-only setting reads (true = it differs from its default). */
const SERVER_SETTING_WORDS = Object.freeze({
  OS_ASSISTANT_API_KEY: { term: 'API key', on: 'Set', off: 'Not set' },
  OS_ASSISTANT_VERIFY_TLS: { term: 'Certificate check', on: 'Off', off: 'On (default)' },
  OS_ASSISTANT_EXTRA_BODY: { term: 'Extra request fields', on: 'Set', off: 'None' },
  OS_ASSISTANT_MAX_TOKENS_FIELD: { term: 'Token limit sent as', on: 'max_completion_tokens', off: 'max_tokens (default)', mono: true },
  OS_ASSISTANT_TEMPLATE_KWARGS: { term: 'Thinking as template variables', on: 'On', off: 'Off (default)' },
  // On by default: true means OS_ASSISTANT_STREAM=false.
  OS_ASSISTANT_STREAM: { term: 'Stream replies', on: 'Off', off: 'On (default)' },
});

/**
 * PURE: the status list for GET /api/assistant. On: the AI server (host only), the default model, and each
 * server-only setting in words: the booleans say whether it differs from its default, so an unset
 * OS_ASSISTANT_VERIFY_TLS or OS_ASSISTANT_STREAM (false) reads as the default "On", never as off. A setting
 * this page doesn't know reads "Default" or "Changed from its default" under its own name. The URL is the AI
 * server row.
 * `brief`: only the status, the AI server and the default model, for when the Connection area
 * shows every setting with where it comes from.
 * @param {object} status
 * @param {{brief?: boolean}} [options]
 * @returns {{term: string, value: string, mono?: boolean}[]}
 */
export function statusFacts(status, { brief = false } = {}) {
  if (!status || !status.enabled) return [{ term: 'Status', value: 'Off' }];
  const inUse = status.assistant && typeof status.assistant.name === 'string' ? status.assistant.name : '';
  const facts = [
    { term: 'Status', value: 'On' },
    ...(inUse ? [{ term: 'Assistant in use', value: inUse }] : []),
    { term: 'AI server', value: status.host ? String(status.host) : 'Unknown', mono: !!status.host },
    status.model ? { term: 'Default model', value: String(status.model), mono: true } : { term: 'Default model', value: 'None set: the AI server picks' },
  ];
  if (brief) return facts;
  const given = status.server_settings && typeof status.server_settings === 'object' ? status.server_settings : {};
  const known = Object.keys(SERVER_SETTING_WORDS).filter((key) => key in given);
  const others = Object.keys(given).filter((key) => !(key in SERVER_SETTING_WORDS) && key !== 'OS_ASSISTANT_URL');
  for (const key of known) {
    const words = SERVER_SETTING_WORDS[key];
    const fact = { term: words.term, value: given[key] ? words.on : words.off };
    if (words.mono) fact.mono = true;
    facts.push(fact);
  }
  for (const key of others) facts.push({ term: key, value: given[key] ? 'Changed from its default' : 'Default', mono: true });
  return facts;
}

/** PURE: the line under the preset picker. */
export function presetHint(view) {
  const v = view || {};
  if (v.id === 'voice' || v.id === 'book') {
    const what = v.id === 'voice' ? 'voice conversations' : 'book jobs';
    return `Built in. The default for ${what}.${v.customized ? ' Changed from its defaults.' : ''}`;
  }
  return `Your own. A client chooses it by its id, ${v.id}.`;
}

/** PURE: the picker's option text. */
export function optionLabel(view) {
  const v = view || {};
  const name = v.name || v.id || '';
  return v.builtin ? `${name} (built in)` : name;
}

const sentenceCase = (text) => {
  const s = String(text || '').trim();
  if (!s) return '';
  const first = s.charAt(0).toUpperCase() + s.slice(1);
  return /[.!?]$/.test(first) ? first : `${first}.`;
};

/**
 * PURE: where a 422 goes and what it says there. `param` is the field the server named; a phrase is named by
 * its place ("fillers[2] …" is phrase 3) and an error sentence by its code ("error_sentences.<code> …"). The
 * message starts with the field's label instead of its API name. No field of the form (body, overrides,
 * preset, or one this page doesn't show) gives null: the caller shows it in its banner.
 * @param {string} param
 * @param {string} message
 * @returns {{field: string, index: number|null, code: string, text: string}|null}
 */
export function fieldProblem(param, message) {
  const field = String(param || '');
  if (!Object.prototype.hasOwnProperty.call(FIELD_LABELS, field)) return null;
  const msg = String(message || '').trim();
  const m = /^([a-z_]+)(?:\[(\d+)\]|\.([a-z0-9_]+))?/.exec(msg);
  let index = null;
  let code = '';
  let text = msg;
  if (m && m[1] === field) {
    let label = FIELD_LABELS[field];
    if (m[2] !== undefined) {
      index = Number(m[2]);
      label = `Phrase ${index + 1}`;
    } else if (m[3]) {
      code = m[3];
      label = `The sentence for “${ERROR_LABELS[code] || code}”`;
    }
    text = `${label}${msg.slice(m[0].length)}`;
  }
  return { field, index, code, text: sentenceCase(text) || `${FIELD_LABELS[field]} wasn't accepted.` };
}

/** What a failed Test says, by the server's error code. */
const TEST_FAILURES = Object.freeze({
  assistant_unreachable: ["Couldn't reach the AI server", 'Check that it is running and that the Server URL under Connection points at it.'],
  assistant_timeout: ['The AI took too long to answer', 'Nothing came back within the timeout. Raise Timeout, or lower the thinking.'],
  assistant_http_error: ['The AI server returned an error', 'It refused the request. Details show what it said.'],
  assistant_bad_response: ["The AI didn't send back an answer", "What came back isn't an answer Open Speech can read."],
  assistant_empty_reply: ['The AI sent back an empty answer', 'Nothing was left once the thinking was taken out.'],
  assistant_reply_cut_off: ['The answer was cut off', 'It stopped while still thinking. Raise Max tokens, or lower the thinking.'],
  assistant_not_started: ["The AI assistant hasn't started", 'Try again in a moment.'],
});

/** PURE: {title, message} for a Test that failed with an AI error code, or null for any other error. */
export function testFailure(code) {
  const entry = TEST_FAILURES[String(code || '')];
  return entry ? { title: entry[0], message: entry[1] } : null;
}

/** PURE: a time taken: "850 ms", "1.2 s", "61 s". */
export function elapsedText(ms) {
  const n = Math.max(0, Number(ms) || 0);
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(Math.round(n / 100) / 10).toFixed(1)} s`;
  return `${Math.round(n / 1000)} s`;
}

/** PURE: elapsed_ms from a failed Test's body (ApiError.detail), or null. */
export function elapsedFrom(detail) {
  try {
    const value = JSON.parse(String(detail || '')).elapsed_ms;
    return Number.isFinite(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * PURE: the voice Preview speaks in: Speak's (app.speakVoice), or why it can't (none chosen yet, or a raw
 * reference recording, which /v1/audio/speech with a voice name can't use).
 * @param {{model?: string, voice?: string, label?: string, kind?: string}|null} speak
 * @returns {{model: string, voice: string, label: string}|{problem: string}}
 */
export function previewVoice(speak) {
  const s = speak || {};
  if (!s.model || !s.voice) return { problem: 'Choose a voice in Speak first: Preview speaks in it.' };
  if (s.kind === 'recording') return { problem: "Speak's voice is a reference recording, which Preview can't use. Choose a voice in Speak." };
  return { model: String(s.model), voice: String(s.voice), label: String(s.label || s.voice) };
}
