// Settings › AI assistant: the status, the preset picker and editor,
// Save with field errors from a 422, Restore defaults, New, Duplicate, Delete, the unsaved-changes guard, Test
// and Preview. The API is src/routes/assistant.py; a fake of it answers here.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';
import { bytes } from '../harness/fakes.js';
import { mountSettings, settle, load, json, textOf, toasts } from './helpers.js';

let t;
afterEach(() => t?.env?.restore());

const VOICE_PROMPT = "You are a helpful voice assistant. Your replies are read aloud. Answer in short, plain spoken sentences. Don't use Markdown, lists, tables, code, links or emoji.";
const BOOK_PROMPT = "You rework book text for listening, following the user's instructions. Reply only with plain prose paragraphs, because your reply is read aloud. Don't use Markdown, lists or headings, and don't comment on what you did.";
const SENTENCES = Object.freeze({
  assistant_unreachable: "I couldn't reach the AI server.",
  assistant_timeout: 'The AI took too long to answer.',
  assistant_http_error: 'The AI server returned an error.',
  assistant_bad_response: "The AI didn't send back an answer.",
  assistant_empty_reply: "The AI didn't send back an answer.",
  assistant_reply_cut_off: 'The answer was cut off before it started.',
});
const TUNABLE = [
  'model', 'system_prompt', 'user_template', 'thinking', 'reasoning_effort', 'thinking_budget', 'show_thinking',
  'max_tokens', 'temperature', 'top_p', 'timeout_s', 'history_turns', 'fillers', 'filler_after_ms',
  'filler_every_ms', 'filler_max', 'speak_errors', 'error_sentences', 'chunk_chars',
];

/** What the server's .env and code give a preset of each purpose (src/assistant/presets.py _defaults). */
function defaults(purpose, env = {}) {
  return {
    model: 'qwen3', system_prompt: purpose === 'book' ? BOOK_PROMPT : VOICE_PROMPT, user_template: '{text}',
    thinking: 'default', reasoning_effort: 'low', thinking_budget: 0, show_thinking: true, max_tokens: 0, temperature: null, top_p: null,
    timeout_s: purpose === 'book' ? 600 : 120, history_turns: 20, fillers: ['Hmm.', 'OK, just a second.', 'Let me think about that.'],
    filler_after_ms: 1200, filler_every_ms: 4000, filler_max: 3, speak_errors: true, error_sentences: { ...SENTENCES },
    chunk_chars: 12000, ...env,
  };
}

/** A preset's API answer: its effective values plus id, name, builtin, customized and inherited. */
function viewOf(id, stored = {}, env = {}) {
  const builtin = id === 'voice' || id === 'book';
  const merged = defaults(id === 'book' ? 'book' : 'voice', env);
  for (const [name, value] of Object.entries(stored)) {
    if (name === 'name') continue;
    merged[name] = name === 'error_sentences' ? { ...merged.error_sentences, ...value } : value;
  }
  return {
    id, name: stored.name || (id === 'voice' ? 'Voice' : id === 'book' ? 'Book' : id), builtin,
    customized: Object.keys(stored).length > 0 || !builtin, inherited: TUNABLE.filter((f) => !(f in stored)), ...merged,
  };
}

describe('settings/assistant: pure parts', () => {
  it('statusFacts: on, the AI server, the default model and the server-only settings in words; an unset OS_ASSISTANT_VERIFY_TLS reads as the default "On", never as off', async () => {
    installBrowserEnv();
    const { statusFacts } = await importFresh('views/settings/assistant-logic.js');
    const off = statusFacts({ enabled: false, host: null, model: '', server_settings: { OS_ASSISTANT_URL: false } });
    assert.deepEqual(off.map((f) => [f.term, f.value]), [['Status', 'Off']]);
    // The booleans say whether each server-only setting differs from its default (src/routes/assistant.py).
    const unset = {
      OS_ASSISTANT_URL: true, OS_ASSISTANT_API_KEY: false, OS_ASSISTANT_EXTRA_BODY: false, OS_ASSISTANT_VERIFY_TLS: false,
      OS_ASSISTANT_MAX_TOKENS_FIELD: false, OS_ASSISTANT_TEMPLATE_KWARGS: false,
    };
    const on = statusFacts({ enabled: true, host: 'ai-server.lan:8080', model: 'qwen3', server_settings: unset });
    assert.deepEqual(on.map((f) => [f.term, f.value]), [
      ['Status', 'On'],
      ['AI server', 'ai-server.lan:8080'],
      ['Default model', 'qwen3'],
      ['API key', 'Not set'],
      ['Certificate check', 'On (default)'],
      ['Extra request fields', 'None'],
      ['Token limit sent as', 'max_tokens (default)'],
      ['Thinking as template variables', 'Off (default)'],
    ]);
    assert.equal(on.find((f) => f.term === 'AI server').mono, true);
    // Every setting set, plus one this page doesn't know yet (a later server), in the server's order.
    const changed = {
      OS_ASSISTANT_URL: true, OS_ASSISTANT_TEMPLATE_KWARGS: true, OS_ASSISTANT_API_KEY: true, OS_ASSISTANT_EXTRA_BODY: true,
      OS_ASSISTANT_VERIFY_TLS: true, OS_ASSISTANT_MAX_TOKENS_FIELD: true, OS_ASSISTANT_SOMETHING_NEW: true, OS_ASSISTANT_OTHER: false,
    };
    const facts = statusFacts({ enabled: true, host: '10.0.0.5', model: '', server_settings: changed });
    const all = Object.fromEntries(facts.map((f) => [f.term, f.value]));
    assert.equal(all['API key'], 'Set');
    assert.equal(all['Certificate check'], 'Off', 'true means OS_ASSISTANT_VERIFY_TLS=false was set');
    assert.equal(all['Extra request fields'], 'Set');
    assert.equal(all['Token limit sent as'], 'max_completion_tokens');
    assert.equal(all['Thinking as template variables'], 'On');
    assert.equal(all['Default model'], 'None set: the AI server picks');
    assert.equal(all.OS_ASSISTANT_SOMETHING_NEW, 'Changed from its default');
    assert.equal(all.OS_ASSISTANT_OTHER, 'Default');
    assert.equal(facts.find((f) => f.term === 'OS_ASSISTANT_SOMETHING_NEW').mono, true);
    assert.equal(facts.filter((f) => /URL/.test(f.term)).length, 0, 'the URL is the AI server row');
    assert.ok(facts.every((f) => typeof f.value === 'string' && !/^(true|false)$/i.test(f.value)), 'no raw booleans');
  });

  // GET /api/assistant's server_settings has OS_ASSISTANT_STREAM (Stream
  // replies, on by default), true only while it is off.
  it('statusFacts: OS_ASSISTANT_STREAM reads "Stream replies: On (default)", or "Off" when it differs from its default; never under its raw name', async () => {
    installBrowserEnv();
    const { statusFacts } = await importFresh('views/settings/assistant-logic.js');
    const base = {
      OS_ASSISTANT_URL: true, OS_ASSISTANT_API_KEY: false, OS_ASSISTANT_EXTRA_BODY: false, OS_ASSISTANT_VERIFY_TLS: false,
      OS_ASSISTANT_MAX_TOKENS_FIELD: false, OS_ASSISTANT_TEMPLATE_KWARGS: false,
    };
    const read = (stream) => statusFacts({ enabled: true, host: 'ai-server.lan:8080', model: 'qwen3', server_settings: { ...base, OS_ASSISTANT_STREAM: stream } });
    const on = read(false);
    assert.deepEqual(on.at(-1), { term: 'Stream replies', value: 'On (default)' });
    assert.equal(on.filter((f) => /OS_ASSISTANT_STREAM/.test(f.term)).length, 0, 'never its raw name');
    assert.deepEqual(read(true).at(-1), { term: 'Stream replies', value: 'Off' });
    assert.equal(statusFacts({ enabled: true, host: 'ai-server.lan:8080', model: 'qwen3', server_settings: { ...base, OS_ASSISTANT_STREAM: true } }, { brief: true }).length, 3, 'brief: the Connection area shows it');
  });

  it('presetValues and saveBody: an untouched field that comes from .env stays inherited; a field of the preset\'s own or one changed on screen is sent; a cleared one is sent empty', async () => {
    installBrowserEnv();
    const { presetValues, saveBody, isDirty } = await importFresh('views/settings/assistant-logic.js');
    const view = viewOf('voice', { system_prompt: 'Be brief.', max_tokens: 400 });
    const values = presetValues(view);
    assert.equal(values.name, 'Voice');
    assert.equal(values.system_prompt, 'Be brief.');
    assert.equal(values.reasoning_effort, 'low');
    assert.equal(values.temperature, null);
    assert.deepEqual(values.fillers, ['Hmm.', 'OK, just a second.', 'Let me think about that.']);
    assert.equal(isDirty(view, values), false);
    // Nothing changed: the name and the preset's own fields only.
    assert.deepEqual(saveBody(view, values), { name: 'Voice', system_prompt: 'Be brief.', max_tokens: 400 });
    // The effort changed (it came from OS_ASSISTANT_REASONING_EFFORT), the prompt was cleared, a temperature set.
    const edited = { ...values, reasoning_effort: 'medium', system_prompt: '', temperature: 0.7, max_tokens: null };
    assert.equal(isDirty(view, edited), true);
    assert.deepEqual(saveBody(view, edited), { name: 'Voice', system_prompt: '', reasoning_effort: 'medium', max_tokens: null, temperature: 0.7 });
    // Changed back to what was loaded: not dirty, and the inherited field stays inherited.
    assert.equal(isDirty(view, { ...edited, reasoning_effort: 'low', system_prompt: 'Be brief.', temperature: null, max_tokens: 400 }), false);
    assert.equal('reasoning_effort' in saveBody(view, { ...values, reasoning_effort: 'low' }), false);
    // The name is trimmed.
    assert.equal(saveBody(view, { ...values, name: '  Short answers ' }).name, 'Short answers');
  });

  it('saveBody for a copy: the fields shown are kept, and a copy of the built-in book preset carries its book system prompt and job timeout (a preset of your own falls back to the voice ones)', async () => {
    installBrowserEnv();
    const { presetValues, saveBody } = await importFresh('views/settings/assistant-logic.js');
    const book = viewOf('book');
    const body = saveBody(book, { ...presetValues(book), name: 'Summaries' }, { copy: true });
    assert.deepEqual(body, { name: 'Summaries', system_prompt: BOOK_PROMPT, timeout_s: 600 });
    const voice = viewOf('voice', { temperature: 0.3 });
    assert.deepEqual(saveBody(voice, { ...presetValues(voice), name: 'Voice copy' }, { copy: true }), { name: 'Voice copy', temperature: 0.3 });
  });

  it('error sentences: an empty one is the default sentence; only the sentences that differ from the default are saved or sent', async () => {
    installBrowserEnv();
    const { presetValues, saveBody, testBody, isDirty, formValues, DEFAULT_ERROR_SENTENCES } = await importFresh('views/settings/assistant-logic.js');
    assert.deepEqual({ ...DEFAULT_ERROR_SENTENCES }, SENTENCES);
    const view = viewOf('voice', { error_sentences: { assistant_timeout: 'Still thinking, sorry.' } });
    const values = presetValues(view);
    assert.equal(values.error_sentences.assistant_timeout, 'Still thinking, sorry.');
    assert.equal(values.error_sentences.assistant_unreachable, SENTENCES.assistant_unreachable);
    assert.deepEqual(saveBody(view, values).error_sentences, { assistant_timeout: 'Still thinking, sorry.' });
    // Clearing the field puts the default back.
    const raw = (sentences) => ({ ...rawOf(values), error_sentences: { ...rawOf(values).error_sentences, ...sentences } });
    const cleared = formValues(raw({ assistant_timeout: '   ' })).values;
    assert.equal(cleared.error_sentences.assistant_timeout, SENTENCES.assistant_timeout);
    assert.deepEqual(saveBody(view, cleared).error_sentences, {});
    // An untouched default is not dirty; a new sentence is sent.
    assert.equal(isDirty(view, formValues(raw({ assistant_timeout: '' })).values), true, 'the preset had its own sentence');
    const plain = viewOf('voice');
    const fresh = presetValues(plain);
    assert.equal(isDirty(plain, formValues({ ...rawOf(fresh), error_sentences: { ...rawOf(fresh).error_sentences, assistant_unreachable: '' } }).values), false);
    const told = formValues({ ...rawOf(fresh), error_sentences: { ...rawOf(fresh).error_sentences, assistant_unreachable: 'The AI server is down.' } }).values;
    assert.deepEqual(saveBody(plain, told).error_sentences, { assistant_unreachable: 'The AI server is down.' });
    assert.deepEqual(testBody('voice', '', told).overrides.error_sentences, { assistant_unreachable: 'The AI server is down.' });
  });

  it('testBody: the preset as it is on screen goes as overrides, every preset field and never its name; an emptied field goes as null (left out, the server would try the saved value); the text only when one was typed', async () => {
    installBrowserEnv();
    const { presetValues, testBody } = await importFresh('views/settings/assistant-logic.js');
    const values = { ...presetValues(viewOf('voice')), name: 'Renamed', system_prompt: 'Answer in French.', temperature: null, model: '', reasoning_effort: '', user_template: '' };
    const body = testBody('voice', '', values);
    assert.deepEqual(Object.keys(body).sort(), ['overrides', 'preset']);
    assert.equal(body.preset, 'voice');
    assert.deepEqual(Object.keys(body.overrides), TUNABLE, 'every preset field, and only those (another key answers 422)');
    assert.equal('name' in body.overrides, false, 'a request can\'t override the name');
    assert.equal(body.overrides.system_prompt, 'Answer in French.');
    // Emptied on screen: null, so the Test tries .env's or the code's default, as the saved preset would.
    assert.equal(body.overrides.temperature, null);
    assert.equal(body.overrides.model, null);
    assert.equal(body.overrides.reasoning_effort, null);
    assert.equal(body.overrides.user_template, null);
    assert.equal(testBody('voice', '', { ...values, system_prompt: '' }).overrides.system_prompt, null);
    assert.equal(testBody('voice', '', { ...values, max_tokens: null }).overrides.max_tokens, null);
    assert.equal(body.overrides.show_thinking, true, 'a switch is never empty');
    assert.deepEqual(body.overrides.fillers, ['Hmm.', 'OK, just a second.', 'Let me think about that.']);
    assert.equal(body.overrides.error_sentences, null, 'every sentence is the default one: the default map');
    assert.equal(testBody('voice', '  ', values).text, undefined);
    assert.equal(testBody('voice', 'Hello there', values).text, 'Hello there');
    assert.deepEqual(testBody('x', '', { ...values, fillers: [] }).overrides.fillers, [], 'no phrases turns them off (a value, not an empty field)');
  });

  it('formValues: numbers come from their fields (empty is the default); anything else is a field problem; blank phrases are dropped and the rest trimmed', async () => {
    installBrowserEnv();
    const { presetValues, formValues } = await importFresh('views/settings/assistant-logic.js');
    const base = rawOf(presetValues(viewOf('voice')));
    const { values, problems } = formValues({ ...base, max_tokens: '512', temperature: '', top_p: '0.9', timeout_s: ' 30 ', fillers: ['  Mm. ', '', '   ', 'One moment.'] });
    assert.deepEqual(problems, []);
    assert.equal(values.max_tokens, 512);
    assert.equal(values.temperature, null);
    assert.equal(values.top_p, 0.9);
    assert.equal(values.timeout_s, 30);
    assert.deepEqual(values.fillers, ['Mm.', 'One moment.']);
    const bad = formValues({ ...base, max_tokens: 'lots', filler_max: '?' });
    assert.deepEqual(bad.problems.map((p) => p.field), ['max_tokens', 'filler_max']);
    assert.match(bad.problems[0].text, /number/);
  });

  it('fieldProblem: a 422 names its field (param); a phrase and an error sentence by their place; no field (body, overrides, preset) gives null', async () => {
    installBrowserEnv();
    const { fieldProblem } = await importFresh('views/settings/assistant-logic.js');
    assert.deepEqual(fieldProblem('user_template', 'user_template must contain {text}'), { field: 'user_template', index: null, code: '', text: 'User template must contain {text}.' });
    assert.deepEqual(fieldProblem('fillers', 'fillers[2] must be a phrase of 1 to 80 characters'), { field: 'fillers', index: 2, code: '', text: 'Phrase 3 must be a phrase of 1 to 80 characters.' });
    const sentence = fieldProblem('error_sentences', 'error_sentences.assistant_timeout must be text of up to 300 characters');
    assert.equal(sentence.code, 'assistant_timeout');
    assert.match(sentence.text, /^The sentence for “No answer in time” must be text of up to 300 characters\.$/);
    assert.equal(fieldProblem('max_tokens', 'max_tokens must be a whole number from 0 to 32768').text, 'Max tokens must be a whole number from 0 to 32768.');
    assert.equal(fieldProblem('text', 'text must be text of up to 8000 characters').text, 'Message must be text of up to 8000 characters.');
    assert.equal(fieldProblem('name', 'name must be 1 to 80 characters').field, 'name');
    for (const param of ['body', 'overrides', 'preset', 'nope', '']) assert.equal(fieldProblem(param, 'x'), null, param);
  });

  it('presetUse, placeholders, presetHint and optionLabel: what each preset is for and may use', async () => {
    installBrowserEnv();
    const { presetUse, placeholders, presetHint, optionLabel } = await importFresh('views/settings/assistant-logic.js');
    assert.deepEqual(['voice', 'book', 'mine'].map(presetUse), ['voice', 'book', 'both']);
    assert.deepEqual(placeholders('voice'), ['text']);
    assert.deepEqual(placeholders('book'), ['text', 'title', 'author', 'chapter']);
    assert.deepEqual(placeholders('mine'), ['text', 'title', 'author', 'chapter']);
    assert.equal(presetHint(viewOf('voice')), 'Built in. The default for voice conversations.');
    assert.equal(presetHint(viewOf('book', { max_tokens: 9 })), 'Built in. The default for book jobs. Changed from its defaults.');
    assert.equal(presetHint(viewOf('short-answers', { name: 'Short answers' })), 'Your own. A client chooses it by its id, short-answers.');
    assert.equal(optionLabel(viewOf('voice')), 'Voice (built in)');
    assert.equal(optionLabel(viewOf('short-answers', { name: 'Short answers' })), 'Short answers');
  });

  it('testFailure and the time taken: each AI error code in plain words; elapsed_ms read from the error body', async () => {
    installBrowserEnv();
    const { testFailure, elapsedText, elapsedFrom, ERROR_CODES } = await importFresh('views/settings/assistant-logic.js');
    assert.deepEqual([...ERROR_CODES], Object.keys(SENTENCES));
    for (const code of [...ERROR_CODES, 'assistant_not_started']) {
      const copy = testFailure(code);
      assert.ok(copy && copy.title && copy.message, code);
    }
    assert.equal(testFailure('assistant_unreachable').title, "Couldn't reach the AI server");
    assert.match(testFailure('assistant_reply_cut_off').message, /Max tokens/);
    assert.equal(testFailure('preset_not_found'), null);
    assert.equal(elapsedText(850), '850 ms');
    assert.equal(elapsedText(1234), '1.2 s');
    assert.equal(elapsedText(61000), '61 s');
    assert.equal(elapsedFrom('{"error": {"message": "x", "code": "assistant_timeout"}, "elapsed_ms": 120004}'), 120004);
    assert.equal(elapsedFrom('not json'), null);
    assert.equal(elapsedFrom('{"error": {}}'), null);
  });

  it('previewVoice: Speak\'s voice, or why Preview can\'t use it (none chosen yet, or a raw reference recording)', async () => {
    installBrowserEnv();
    const { previewVoice } = await importFresh('views/settings/assistant-logic.js');
    assert.deepEqual(previewVoice({ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Bella + Sky', kind: 'blend' }), { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Bella + Sky' });
    assert.deepEqual(previewVoice({ model: 'kokoro', voice: 'af_heart', label: '' }), { model: 'kokoro', voice: 'af_heart', label: 'af_heart' });
    assert.match(previewVoice({ model: '', voice: '', label: '' }).problem, /Speak/);
    assert.match(previewVoice(null).problem, /Speak/);
    assert.match(previewVoice({ model: 'chatterbox', voice: 'demo_reference', label: 'demo_reference', kind: 'recording' }).problem, /recording/);
  });

  it('TUNABLE is the server\'s list of preset fields, in its order', async () => {
    installBrowserEnv();
    const logic = await importFresh('views/settings/assistant-logic.js');
    assert.deepEqual([...logic.TUNABLE], TUNABLE);
    assert.equal(logic.MAX_FILLERS, 10);
    for (const field of ['name', ...TUNABLE, 'text']) assert.ok(logic.FIELD_LABELS[field], field);
  });
});

/** The raw form strings for a values object (what the fields hold). */
function rawOf(values) {
  const out = { ...values };
  for (const f of ['thinking_budget', 'max_tokens', 'temperature', 'top_p', 'timeout_s', 'history_turns', 'filler_after_ms', 'filler_every_ms', 'filler_max', 'chunk_chars']) {
    out[f] = values[f] === null || values[f] === undefined ? '' : String(values[f]);
  }
  out.fillers = [...values.fillers];
  out.error_sentences = { ...values.error_sentences };
  return out;
}

/* ───────────────────────────── the section on screen ───────────────────────────── */

const BUILTINS = ['voice', 'book'];
const disabledAnswer = () => json(404, { error: { message: "The AI assistant is off: set the AI server's URL in Settings or OS_ASSISTANT_URL", code: 'assistant_disabled' } });
const slug = (name) => String(name).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40);

/** The fields a body sets, as the server keeps them (presets.validate_fields): read-only, empty and null are dropped. */
function ownFields(body) {
  const out = {};
  for (const [name, value] of Object.entries(body || {})) {
    if (['id', 'builtin', 'customized', 'inherited'].includes(name) || value === null || value === '') continue;
    if (name === 'error_sentences') {
      const kept = Object.fromEntries(Object.entries(value).filter(([, sentence]) => String(sentence).trim()));
      if (Object.keys(kept).length) out[name] = kept;
      continue;
    }
    out[name] = value;
  }
  return out;
}

/**
 * A fake of src/routes/assistant.py with a preset store: built-ins first, then yours; PUT replaces the editable
 * fields; DELETE restores a built-in (200) or deletes yours (204); a 422 names its field in `param`.
 */
function fakeAssistant({ enabled = true, model = 'qwen3', host = 'ai-server.lan:8080', settings = {}, stored = {}, env = {} } = {}) {
  const store = new Map(Object.entries(stored));
  const state = { store, puts: [], posts: [], deletes: [], tests: [], speech: [], testAnswer: null };
  const list = () => [...BUILTINS, ...[...store.keys()].filter((id) => !BUILTINS.includes(id))].map((id) => viewOf(id, store.get(id) || {}, env));
  const invalid = (param, message) => json(422, { error: { message, code: 'validation_error', param } });
  const validate = (body) => {
    if (typeof body.user_template === 'string' && body.user_template.trim() && !body.user_template.includes('{text}')) return invalid('user_template', 'user_template must contain {text}');
    if ('name' in body && !String(body.name || '').trim()) return invalid('name', 'name must be 1 to 80 characters');
    const long = Array.isArray(body.fillers) ? body.fillers.findIndex((f) => f.length > 80) : -1;
    if (long >= 0) return invalid('fillers', `fillers[${long}] must be a phrase of 1 to 80 characters`);
    if (typeof body.max_tokens === 'number' && body.max_tokens > 32768) return invalid('max_tokens', 'max_tokens must be a whole number from 0 to 32768');
    if (typeof body.thinking_budget === 'number' && body.thinking_budget > 1000000) return invalid('thinking_budget', 'thinking_budget must be a whole number from 0 to 1000000');
    return null;
  };
  const serverSettings = () => ({
    OS_ASSISTANT_URL: enabled, OS_ASSISTANT_API_KEY: false, OS_ASSISTANT_EXTRA_BODY: false, OS_ASSISTANT_VERIFY_TLS: false,
    OS_ASSISTANT_MAX_TOKENS_FIELD: false, OS_ASSISTANT_TEMPLATE_KWARGS: true, OS_ASSISTANT_STREAM: false, ...settings,
  });
  const routes = {
    'GET /api/assistant': () => json(200, { enabled, host: enabled ? host : null, model, server_settings: serverSettings() }),
    // The connection from .env (the Connection area; tested in assistant-connection.test.js).
    'GET /api/assistant/connection': () => {
      const s = serverSettings();
      const from = (set) => (set ? 'env' : 'default');
      return json(200, {
        enabled, locked: false, url: enabled ? `http://${host}/v1` : '', model, api_key_set: !!s.OS_ASSISTANT_API_KEY,
        verify_tls: !s.OS_ASSISTANT_VERIFY_TLS, template_kwargs: !!s.OS_ASSISTANT_TEMPLATE_KWARGS,
        max_tokens_field: s.OS_ASSISTANT_MAX_TOKENS_FIELD ? 'max_completion_tokens' : 'max_tokens', extra_body: s.OS_ASSISTANT_EXTRA_BODY ? { top_k: 20 } : {},
        stream: !s.OS_ASSISTANT_STREAM,
        sources: {
          url: from(enabled), api_key: from(s.OS_ASSISTANT_API_KEY), model: from(!!model), verify_tls: from(s.OS_ASSISTANT_VERIFY_TLS),
          template_kwargs: from(s.OS_ASSISTANT_TEMPLATE_KWARGS), max_tokens_field: from(s.OS_ASSISTANT_MAX_TOKENS_FIELD), extra_body: from(s.OS_ASSISTANT_EXTRA_BODY),
          stream: from(s.OS_ASSISTANT_STREAM),
        },
      });
    },
    'GET /api/assistant/presets': () => (enabled ? json(200, { presets: list() }) : disabledAnswer()),
    'POST /api/assistant/presets': (req) => {
      state.posts.push(req.json);
      const bad = validate(req.json);
      if (bad) return bad;
      const id = slug(req.json.name);
      if (BUILTINS.includes(id) || store.has(id)) return json(409, { error: { message: `A preset with the id '${id}' already exists`, code: 'preset_exists' } });
      store.set(id, ownFields(req.json));
      return json(201, viewOf(id, store.get(id), env));
    },
    'PUT /api/assistant/presets/:id': (req) => {
      const id = req.params.id;
      state.puts.push({ id, body: req.json });
      if (!BUILTINS.includes(id) && !store.has(id)) return json(404, { error: { message: `AI preset '${id}' not found`, code: 'preset_not_found' } });
      const bad = validate(req.json);
      if (bad) return bad;
      const fields = ownFields(req.json);
      if (BUILTINS.includes(id) && fields.name === (id === 'voice' ? 'Voice' : 'Book')) delete fields.name;
      if (Object.keys(fields).length) store.set(id, fields);
      else store.delete(id);
      return json(200, viewOf(id, fields, env));
    },
    'DELETE /api/assistant/presets/:id': (req) => {
      const id = req.params.id;
      state.deletes.push(id);
      if (BUILTINS.includes(id)) {
        store.delete(id);
        return json(200, viewOf(id, {}, env));
      }
      if (!store.has(id)) return json(404, { error: { message: `AI preset '${id}' not found`, code: 'preset_not_found' } });
      store.delete(id);
      return json(204);
    },
    'POST /api/assistant/test': (req) => {
      state.tests.push(req.json);
      if (state.testAnswer) return state.testAnswer(req);
      return json(200, { reply: 'Hello! How can I help?', thinking: 'The user greets me.', finish_reason: 'stop', elapsed_ms: 1234 });
    },
    'POST /v1/audio/speech': (req) => {
      state.speech.push(req.json);
      return bytes(200, [1, 2, 3], 'audio/mpeg');
    },
  };
  return { routes, state };
}

const SPEAK_VOICE = Object.freeze({ model: 'kokoro', voice: 'af_heart', label: 'Heart', kind: 'voice' });

/** Settings with the fake AI server, Speak's voice set, and the section loaded. */
async function mountAssistant({ server = {}, routes = {}, seed = {}, speakVoice = SPEAK_VOICE, env } = {}) {
  const fake = fakeAssistant(server);
  t = await mountSettings({ routes: { ...fake.routes, ...routes }, seed, env });
  t.server = fake.state;
  t.app.set({ speakVoice });
  await settle(t.env);
  await settle(t.env);
  return t;
}

const $ = (id) => t.$(id);
const facts = () => {
  const dl = $('se-ai-facts');
  const terms = dl.querySelectorAll('dt').map(textOf);
  const values = dl.querySelectorAll('dd').map(textOf);
  return Object.fromEntries(terms.map((term, i) => [term, values[i]]));
};
const options = () => $('se-ai-preset').querySelectorAll('option').map((o) => [o.value, textOf(o)]);
const shown = (id) => {
  for (let el = $(id); el && el !== t.root; el = el.parentElement) if (el.hidden) return false;
  return true;
};
const fillerInputs = () => $('se-ai-fillers').querySelectorAll('input');
async function pick(id) {
  t.env.change($('se-ai-preset'), id);
  await settle(t.env);
}
async function answerConfirm(ok) {
  await settle(t.env);
  assert.equal($('app-confirm').open, true, 'a confirm dialog is open');
  t.env.click($(ok ? 'app-confirm-ok' : 'app-confirm-cancel'));
  await settle(t.env);
}
async function answerPrompt(value) {
  await settle(t.env);
  assert.equal($('app-prompt').open, true, 'a prompt dialog is open');
  t.env.input($('app-prompt-input'), value);
  t.env.click($('app-prompt-ok'));
  await settle(t.env);
}

describe('settings/assistant: the section', () => {
  it('is listed in the contents after Server, with its heading; the screenshot hook ?shot=assistant jumps to it', async () => {
    await mountAssistant();
    const links = $('se-toc').querySelectorAll('.se-toc__link');
    const link = links.find((l) => l.getAttribute('data-target') === 'se-assistant-title');
    assert.ok(link, 'a contents link');
    assert.equal(textOf(link), 'AI assistant');
    assert.equal(links.indexOf(link), links.findIndex((l) => l.getAttribute('data-target') === 'se-server-title') + 1);
    const heading = $('se-assistant-title');
    assert.equal(heading.localName, 'h2');
    assert.equal(textOf(heading), 'AI assistant');
    t.env.click(link);
    assert.equal(t.doc.activeElement, heading);
    t.env.restore();
    t = await mountSettings({ routes: fakeAssistant().routes, shot: 'assistant' });
    assert.equal(t.doc.activeElement, t.$('se-assistant-title'));
  });

  it('off: the status says so and how to turn it on, which is the Connection form now (no restart); the presets are hidden and none is asked for', async () => {
    await mountAssistant({ server: { enabled: false } });
    assert.deepEqual(facts(), { Status: 'Off' });
    assert.equal(shown('se-ai-off'), true);
    assert.match(textOf($('se-ai-off-form')), /Connection/);
    assert.doesNotMatch(textOf($('se-ai-off-form')), /restart/i);
    assert.equal(shown('se-ai-off-env'), false, 'the .env way is for a locked connection');
    assert.equal(shown('se-ai-conn'), true);
    assert.equal(shown('se-ai-editor'), false);
    assert.equal(shown('se-ai-server-note'), false);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/presets').length, 0);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant').length, 1);
  });

  it('on: the AI server and the default model; the server-only settings show in the Connection form, each with where it comes from, so the status list stays short', async () => {
    await mountAssistant();
    assert.deepEqual(facts(), { Status: 'On', 'AI server': 'ai-server.lan:8080', 'Default model': 'qwen3' });
    assert.equal(shown('se-ai-off'), false);
    assert.equal(shown('se-ai-editor'), true);
    assert.equal(shown('se-ai-server-note'), false, 'the connection is no longer .env-only');
    assert.equal($('se-ai-conn-tls').checked, true, 'an unset OS_ASSISTANT_VERIFY_TLS is the default: on');
    assert.equal(textOf($('se-ai-conn-tls-source')), 'default');
    assert.equal($('se-ai-conn-template').checked, true);
    assert.equal(textOf($('se-ai-conn-template-source')), 'from .env');
    assert.equal(textOf($('se-ai-conn-key-state')), 'Not set');
    assert.equal($('se-ai-conn-max-field').value, 'max_tokens');
  });

  // docs/AI-ASSISTANT.md (its intro and Security and privacy) says the
  // listener's words and books go to the AI server, which may be another machine or a cloud API.
  it('on: Settings › About does not say that nothing leaves this server without naming the AI server the assistant sends text to', async () => {
    await mountAssistant();
    const sectionOf = (headingId) => {
      let el = $(headingId);
      assert.ok(el, `#${headingId} exists`);
      while (el && el.localName !== 'section') el = el.parentElement;
      assert.ok(el, `#${headingId} is inside a <section>`);
      return el;
    };
    // The state under test, as the same page shows it: the assistant is on and its AI server is another host.
    assert.equal(facts().Status, 'On', 'precondition: Settings › AI assistant shows the assistant on');
    assert.equal(facts()['AI server'], 'ai-server.lan:8080', 'precondition: Settings › AI assistant names the other machine');
    assert.match(textOf(sectionOf('se-assistant-title')), /send what you say, or a book, to your own AI server/,
      'precondition: the same page says words and books go to the AI server');
    const about = textOf(sectionOf('se-about-title'));
    const claimsNothingLeaves = /nothing leaves this server/i.test(about);
    const namesTheAiServer = /\bAI\b/.test(about) || about.includes('ai-server.lan:8080');
    assert.ok(!claimsNothingLeaves || namesTheAiServer,
      `Settings › About says nothing leaves this server while the assistant sends text to ai-server.lan:8080: ${JSON.stringify(about)}`);
  });

  it('an older server without the route says it has no AI assistant; a failed load says so with Retry, which loads again', async () => {
    await mountAssistant({ routes: { 'GET /api/assistant': () => json(404, { detail: 'Not Found' }) } });
    assert.match(textOf($('se-ai-load')), /doesn't have the AI assistant/);
    assert.equal(shown('se-ai-editor'), false);
    t.env.restore();
    let fail = true;
    const fake = fakeAssistant();
    await mountAssistant({ routes: { 'GET /api/assistant': (req) => (fail ? json(500, { error: { message: 'database is locked', code: 'internal_error' } }) : fake.routes['GET /api/assistant'](req)) } });
    const banner = $('se-ai-load');
    assert.equal(shown('se-ai-load'), true);
    assert.match(textOf(banner), /Couldn't load the AI assistant/);
    const retry = banner.querySelector('[data-action="ai-retry"]');
    assert.ok(retry, 'Retry');
    fail = false;
    t.env.click(retry);
    await settle(t.env);
    await settle(t.env);
    assert.equal(facts().Status, 'On');
    assert.equal(shown('se-ai-load'), false);
    assert.equal(shown('se-ai-editor'), true);
  });

  it('the picker lists the built-in presets first, then yours; the form shows the chosen one; the voice preset hides the book-only piece size and the book preset the voice-only groups', async () => {
    await mountAssistant({ server: { stored: { 'short-answers': { name: 'Short answers', max_tokens: 200 } } } });
    assert.deepEqual(options(), [['voice', 'Voice (built in)'], ['book', 'Book (built in)'], ['short-answers', 'Short answers']]);
    assert.equal($('se-ai-preset').value, 'voice');
    assert.equal(textOf($('se-ai-preset-hint')), 'Built in. The default for voice conversations.');
    assert.equal($('se-ai-name').value, 'Voice');
    assert.equal($('se-ai-system').value, VOICE_PROMPT);
    assert.equal($('se-ai-template').value, '{text}');
    assert.equal($('se-ai-thinking').value, 'default');
    assert.equal($('se-ai-effort').value, 'low');
    assert.equal($('se-ai-show-thinking').checked, true);
    assert.equal($('se-ai-max-tokens').value, '0');
    assert.equal($('se-ai-temperature').value, '');
    assert.equal($('se-ai-timeout').value, '120');
    assert.equal($('se-ai-history').value, '20');
    assert.deepEqual(fillerInputs().map((i) => i.value), ['Hmm.', 'OK, just a second.', 'Let me think about that.']);
    assert.equal($('se-ai-filler-after').value, '1200');
    assert.equal($('se-ai-speak-errors').checked, true);
    assert.equal($('se-ai-err-assistant_timeout').value, SENTENCES.assistant_timeout);
    assert.match(textOf($('se-ai-model-hint')), /qwen3/);
    assert.match(textOf($('se-ai-template-hint')), /\{text\}/);
    assert.doesNotMatch(textOf($('se-ai-template-hint')), /\{title\}/, 'the voice preset takes only {text}');
    assert.equal(shown('se-ai-chunk'), false);
    assert.equal(shown('se-ai-history'), true);
    assert.equal(shown('se-ai-fillers'), true);
    assert.equal(shown('se-ai-speak-errors'), true);
    await pick('book');
    assert.equal($('se-ai-system').value, BOOK_PROMPT);
    assert.equal($('se-ai-timeout').value, '600');
    assert.equal(shown('se-ai-chunk'), true);
    assert.equal($('se-ai-chunk').value, '12000');
    assert.equal(shown('se-ai-history'), false);
    assert.equal(shown('se-ai-fillers'), false);
    assert.equal(shown('se-ai-speak-errors'), false);
    assert.match(textOf($('se-ai-template-hint')), /\{title\}.*\{author\}.*\{chapter\}/);
    await pick('short-answers');
    assert.equal($('se-ai-max-tokens').value, '200');
    assert.equal(shown('se-ai-chunk'), true);
    assert.equal(shown('se-ai-fillers'), true);
    assert.equal(textOf($('se-ai-preset-hint')), 'Your own. A client chooses it by its id, short-answers.');
    assert.equal(textOf($('se-ai-delete')), 'Delete');
    await pick('voice');
    assert.equal(textOf($('se-ai-delete')), 'Restore defaults');
    assertAccessibleNames(t.root);
  });

  it('editing marks the preset unsaved; Save sends the name, the preset\'s own fields and the changed ones (an untouched .env field stays inherited), then shows what the server saved', async () => {
    await mountAssistant({ server: { stored: { voice: { max_tokens: 300 } } } });
    const save = $('se-ai-save');
    assert.equal(shown('se-ai-dirty'), false);
    assert.equal(save.getAttribute('aria-disabled'), 'true', 'nothing to save yet');
    t.env.input($('se-ai-system'), 'Answer in one sentence.');
    t.env.change($('se-ai-thinking'), 'off');
    t.env.input($('se-ai-temperature'), '0.4');
    assert.equal(shown('se-ai-dirty'), true);
    assert.equal(textOf($('se-ai-dirty')), 'Unsaved changes');
    assert.equal(save.getAttribute('aria-disabled'), 'false');
    t.env.click(save);
    assert.equal(save.getAttribute('aria-busy'), 'true');
    await settle(t.env);
    assert.equal(t.server.puts.length, 1);
    assert.equal(t.server.puts[0].id, 'voice');
    assert.deepEqual(t.server.puts[0].body, { name: 'Voice', system_prompt: 'Answer in one sentence.', thinking: 'off', max_tokens: 300, temperature: 0.4 });
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'Saved “Voice”' }]);
    assert.equal(shown('se-ai-dirty'), false);
    assert.equal(save.getAttribute('aria-disabled'), 'true');
    assert.equal(save.hasAttribute('aria-busy'), false);
    assert.equal(textOf($('se-ai-preset-hint')), 'Built in. The default for voice conversations. Changed from its defaults.');
    // The next save keeps the saved fields and still leaves the .env ones out.
    t.env.input($('se-ai-name'), 'Talk');
    t.env.click(save);
    await settle(t.env);
    assert.deepEqual(t.server.puts[1].body, { name: 'Talk', system_prompt: 'Answer in one sentence.', thinking: 'off', max_tokens: 300, temperature: 0.4 });
    assert.deepEqual(options()[0], ['voice', 'Talk (built in)']);
    // Enter in a one-line field saves too (the form's submit).
    t.env.input($('se-ai-model'), 'qwen3-mini');
    $('se-ai-model').focus();
    t.env.key('Enter');
    await settle(t.env);
    assert.equal(t.server.puts.length, 3);
    assert.equal(t.server.puts[2].body.model, 'qwen3-mini');
  });

  it('a 422 shows on the field it names: aria-invalid, its error text tied with aria-describedby, focus on it, a summary banner; typing there clears it; a phrase is named by its place', async () => {
    await mountAssistant();
    const template = $('se-ai-template');
    t.env.input(template, 'Please answer: ');
    t.env.click($('se-ai-save'));
    await settle(t.env);
    const error = $('se-ai-template-error');
    assert.equal(error.hidden, false);
    assert.equal(textOf(error), 'User template must contain {text}.');
    assert.equal(template.getAttribute('aria-invalid'), 'true');
    assert.ok(template.getAttribute('aria-describedby').split(' ').includes('se-ai-template-error'));
    assert.equal(t.doc.activeElement, template);
    const banner = $('se-ai-save-banner');
    assert.equal(banner.hidden, false);
    assert.ok(banner.querySelector('[role="alert"]'));
    assert.match(textOf(banner), /Couldn't save “Voice”/);
    assert.match(textOf(banner), /User template/);
    assert.equal(shown('se-ai-dirty'), true, 'nothing was saved');
    t.env.input(template, 'Please answer: {text}');
    assert.equal(error.hidden, true);
    assert.equal(template.hasAttribute('aria-invalid'), false);
    // A phrase that is too long: the server names it by its place, fillers[1].
    t.env.input(fillerInputs()[1], 'x'.repeat(81));
    t.env.click($('se-ai-save'));
    await settle(t.env);
    const second = fillerInputs()[1];
    assert.equal(second.getAttribute('aria-invalid'), 'true');
    assert.equal(textOf($('se-ai-fillers-error')), 'Phrase 2 must be a phrase of 1 to 80 characters.');
    assert.ok(second.getAttribute('aria-describedby').split(' ').includes('se-ai-fillers-error'));
    assert.equal(t.doc.activeElement, second);
    // A number this page can't read is caught before anything is sent.
    const puts = t.server.puts.length;
    t.env.input(fillerInputs()[1], 'Wait.');
    t.env.input($('se-ai-max-tokens'), 'lots');
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.equal(t.server.puts.length, puts);
    assert.equal(textOf($('se-ai-max-tokens-error')), 'Max tokens must be a number.');
    assert.equal(t.doc.activeElement, $('se-ai-max-tokens'));
  });

  it('Restore defaults on a changed built-in asks first, then gives the defaults back; Delete on a preset of your own asks, deletes it and shows the voice preset', async () => {
    await mountAssistant({ server: { stored: { voice: { system_prompt: 'Be terse.' }, mine: { name: 'Mine', temperature: 0.2 } } } });
    assert.equal($('se-ai-system').value, 'Be terse.');
    t.env.click($('se-ai-delete'));
    await settle(t.env);
    assert.match(textOf($('app-confirm-title')), /Restore the defaults of “Voice”\?/);
    await answerConfirm(false);
    assert.deepEqual(t.server.deletes, [], 'Cancel changes nothing');
    t.env.click($('se-ai-delete'));
    await answerConfirm(true);
    assert.deepEqual(t.server.deletes, ['voice']);
    assert.equal($('se-ai-system').value, VOICE_PROMPT);
    assert.deepEqual(toasts(t).map((x) => x.text), ['Restored the defaults of “Voice”']);
    assert.equal($('se-ai-delete').getAttribute('aria-disabled'), 'true', 'nothing left to restore');
    await pick('mine');
    t.env.click($('se-ai-delete'));
    await settle(t.env);
    assert.match(textOf($('app-confirm-title')), /Delete the AI preset “Mine”\?/);
    assert.equal($('app-confirm-ok').classList.contains('btn--danger'), true);
    await answerConfirm(true);
    assert.deepEqual(t.server.deletes, ['voice', 'mine']);
    assert.deepEqual(options().map(([id]) => id), ['voice', 'book']);
    assert.equal($('se-ai-preset').value, 'voice');
    assert.equal(toasts(t).at(-1).text, 'Deleted “Mine”');
  });

  it('New asks for a name and creates the preset (a taken name asks again with the reason); Duplicate copies the preset as shown, unsaved changes too', async () => {
    await mountAssistant({ server: { stored: { 'translate-to-spanish': { name: 'Translate to Spanish' } } } });
    t.env.click($('se-ai-new'));
    await answerPrompt('Translate to Spanish');
    assert.equal($('app-prompt').open, true, 'asked again');
    assert.match(textOf($('app-prompt-error')), /already exists/);
    t.env.input($('app-prompt-input'), 'Short answers');
    t.env.click($('app-prompt-ok'));
    await settle(t.env);
    assert.deepEqual(t.server.posts.at(-1), { name: 'Short answers' });
    assert.equal($('se-ai-preset').value, 'short-answers');
    assert.equal($('se-ai-name').value, 'Short answers');
    assert.deepEqual(toasts(t).map((x) => x.text), ['Created “Short answers”']);
    // Duplicate the book preset with an unsaved template: the copy carries it, and the book fallbacks.
    await pick('book');
    t.env.input($('se-ai-template'), 'Summarise "{chapter}":\n\n{text}');
    t.env.click($('se-ai-duplicate'));
    await settle(t.env);
    assert.equal($('app-prompt-input').value, 'Book copy');
    await answerPrompt('Chapter summaries');
    assert.deepEqual(t.server.posts.at(-1), { name: 'Chapter summaries', system_prompt: BOOK_PROMPT, user_template: 'Summarise "{chapter}":\n\n{text}', timeout_s: 600 });
    assert.equal($('se-ai-preset').value, 'chapter-summaries');
    assert.equal($('se-ai-template').value, 'Summarise "{chapter}":\n\n{text}');
    assert.equal(shown('se-ai-dirty'), false);
    assert.equal(t.server.store.has('book'), false, 'the book preset itself is unchanged');
  });

  it('unsaved changes: choosing another preset asks first (Keep editing keeps them), Discard changes puts the saved values back, and leaving the page warns', async () => {
    await mountAssistant();
    const unload = () => {
      const event = new Event('beforeunload', { cancelable: true });
      window.dispatchEvent(event);
      return event.defaultPrevented;
    };
    assert.equal(unload(), false, 'no warning without changes');
    t.env.input($('se-ai-system'), 'Edited.');
    assert.equal(unload(), true, 'the browser asks before leaving');
    t.env.change($('se-ai-preset'), 'book');
    await settle(t.env);
    assert.match(textOf($('app-confirm-title')), /Discard your changes to “Voice”\?/);
    await answerConfirm(false);
    assert.equal($('se-ai-preset').value, 'voice', 'the picker goes back');
    assert.equal($('se-ai-system').value, 'Edited.');
    t.env.change($('se-ai-preset'), 'book');
    await answerConfirm(true);
    assert.equal($('se-ai-preset').value, 'book');
    assert.equal($('se-ai-system').value, BOOK_PROMPT);
    assert.equal(unload(), false);
    t.env.input($('se-ai-timeout'), '30');
    assert.equal(shown('se-ai-discard'), true);
    t.env.click($('se-ai-discard'));
    await settle(t.env);
    assert.equal($('se-ai-timeout').value, '600');
    assert.equal(shown('se-ai-dirty'), false);
    assert.equal(shown('se-ai-discard'), false);
    // Leaving Settings keeps the edits (views stay mounted), and coming back doesn't reload over them.
    t.env.input($('se-ai-timeout'), '45');
    t.ctl.hide();
    t.ctl.show(t.route, { focus: 'none' });
    await settle(t.env);
    assert.equal($('se-ai-timeout').value, '45');
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/presets').length, 1, 'no reload over the edits');
  });

  it('Test sends the preset as on screen, unsaved, with the message; the button is busy while it runs; the reply, the thinking (collapsed) and the time show', async () => {
    await mountAssistant();
    let release;
    t.server.testAnswer = () => new Promise((resolve) => { release = () => resolve(json(200, { reply: 'Bonjour !', thinking: 'French, short.', finish_reason: 'stop', elapsed_ms: 1234 })); });
    t.env.input($('se-ai-system'), 'Answer in French.');
    t.env.input($('se-ai-test-text'), 'Hello');
    const test = $('se-ai-test');
    t.env.click(test);
    await settle(t.env);
    assert.equal(test.getAttribute('aria-busy'), 'true');
    assert.equal(test.disabled, true);
    assert.equal(textOf($('se-ai-test-status')), 'Asking the AI…');
    t.env.click(test);
    await settle(t.env);
    assert.equal(t.server.tests.length, 1, 'one test at a time');
    const body = t.server.tests[0];
    assert.equal(body.preset, 'voice');
    assert.equal(body.text, 'Hello');
    assert.equal(body.overrides.system_prompt, 'Answer in French.');
    assert.equal('name' in body.overrides, false);
    release();
    await settle(t.env);
    assert.equal(test.hasAttribute('aria-busy'), false);
    assert.equal(shown('se-ai-test-result'), true);
    assert.equal(textOf($('se-ai-test-reply')), 'Bonjour !');
    const thinking = $('se-ai-test-thinking');
    assert.equal(thinking.localName, 'details');
    assert.equal(thinking.hasAttribute('open'), false, 'collapsed');
    assert.equal(thinking.hidden, false);
    assert.equal(textOf(thinking.querySelector('pre')), 'French, short.');
    assert.equal(textOf($('se-ai-test-meta')), 'Answered in 1.2 s · finish reason: stop');
    assert.equal(textOf($('se-ai-test-status')), 'Answered in 1.2 s');
    assert.equal(t.server.puts.length, 0, 'testing saves nothing');
    // No thinking (the server didn't send it, or Show thinking is off): no disclosure.
    t.server.testAnswer = () => json(200, { reply: 'Hi.', thinking: null, finish_reason: 'stop', elapsed_ms: 300 });
    t.env.input($('se-ai-test-text'), '');
    t.env.click(test);
    await settle(t.env);
    assert.equal(t.server.tests[1].text, undefined, 'an empty message uses the server\'s default');
    assert.equal($('se-ai-test-thinking').hidden, true);
    assert.equal(textOf($('se-ai-test-meta')), 'Answered in 300 ms · finish reason: stop');
    // Another preset: the last reply was this one's, so it goes.
    t.env.click($('se-ai-discard'));
    await pick('book');
    assert.equal(shown('se-ai-test-result'), false);
    assert.equal(textOf($('se-ai-test-status')), '');
  });

  it('a Test that fails says what went wrong in plain words, with the server\'s message in Details and the time taken; a 422 lands on the field', async () => {
    await mountAssistant();
    t.server.testAnswer = () => json(502, { error: { message: 'Connection refused (ai-server.lan:8080)', code: 'assistant_unreachable' }, elapsed_ms: 340 });
    t.env.click($('se-ai-test'));
    await settle(t.env);
    const banner = $('se-ai-test-banner');
    assert.equal(banner.hidden, false);
    assert.ok(banner.querySelector('[role="alert"]'));
    assert.equal(textOf(banner.querySelector('.banner__title')), "Couldn't reach the AI server");
    assert.match(textOf(banner.querySelector('.banner__msg')), /Server URL under Connection/, 'the address is set in the form now');
    assert.match(textOf(banner.querySelector('details pre')), /Connection refused \(ai-server\.lan:8080\)/);
    assert.equal(textOf($('se-ai-test-status')), 'Failed after 340 ms');
    assert.equal(shown('se-ai-test-result'), false);
    assert.equal($('se-ai-test').hasAttribute('aria-busy'), false);
    // The answer was cut off: the advice names the field to change.
    t.server.testAnswer = () => json(502, { error: { message: 'finish_reason length', code: 'assistant_reply_cut_off' }, elapsed_ms: 9000 });
    t.env.click($('se-ai-test'));
    await settle(t.env);
    assert.match(textOf($('se-ai-test-banner')), /Raise Max tokens/);
    // The on-screen values are refused: the field shows it.
    t.server.testAnswer = () => json(422, { error: { message: 'user_template has the unknown placeholder {title}; it may use {text}', code: 'validation_error', param: 'user_template' } });
    t.env.input($('se-ai-template'), '{title}: {text}');
    t.env.click($('se-ai-test'));
    await settle(t.env);
    assert.equal(textOf($('se-ai-template-error')), 'User template has the unknown placeholder {title}; it may use {text}.');
    assert.equal($('se-ai-template').getAttribute('aria-invalid'), 'true');
    assert.match(textOf($('se-ai-test-banner')), /Check the highlighted field/);
  });

  // The Test box shows the preset on screen (fill() hides another
  // preset's result). A Test that answers after another preset was picked shows nothing there, reply or failure; a
  // save of the same preset while it runs still shows its answer.
  it('a Test that answers after another preset was picked shows neither its reply nor its failure under that preset', async () => {
    await mountAssistant();
    let release;
    const held = (answer) => () => new Promise((resolve) => { release = () => resolve(answer); });
    const testVoiceThenPickBook = async (answer) => {
      t.server.testAnswer = held(answer);
      if ($('se-ai-preset').value !== 'voice') await pick('voice');
      t.env.click($('se-ai-test'));
      await settle(t.env);
      assert.equal(t.server.tests.at(-1).preset, 'voice', 'the Test is of Voice');
      await pick('book');
      assert.equal($('se-ai-name').value, 'Book', 'the form shows Book');
      release();
      await settle(t.env);
      await settle(t.env);
    };
    await testVoiceThenPickBook(json(200, { reply: 'Reply of the Voice preset.', thinking: '', finish_reason: 'stop', elapsed_ms: 30000 }));
    assert.equal(shown('se-ai-test-result'), false, "Voice's reply is not shown as Book's");
    assert.equal(textOf($('se-ai-test-status')), '', "Book's Test status claims no answer");
    assert.equal($('se-ai-test').hasAttribute('aria-busy'), false, 'the button is free again');
    await testVoiceThenPickBook(json(502, { error: { message: 'Connection refused (ai-server.lan:8080)', code: 'assistant_unreachable' }, elapsed_ms: 340 }));
    assert.equal($('se-ai-test-banner').hidden, true, "Voice's failure is not shown under Book");
    assert.equal(textOf($('se-ai-test-status')), '', 'no "Failed after" under Book');
    // The same preset saved while its Test runs: still the preset on screen, so its answer shows.
    await pick('voice');
    t.server.testAnswer = held(json(200, { reply: 'Saved and answered.', thinking: '', finish_reason: 'stop', elapsed_ms: 500 }));
    t.env.input($('se-ai-system'), 'Answer briefly.');
    t.env.click($('se-ai-test'));
    await settle(t.env);
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.equal(t.server.puts.length, 1, 'saved while the Test runs');
    release();
    await settle(t.env);
    assert.equal(shown('se-ai-test-result'), true);
    assert.equal(textOf($('se-ai-test-reply')), 'Saved and answered.');
  });

  it('Preview speaks a phrase or an error sentence, as on screen, in Speak\'s voice through the audio coordinator: primed in the click, then an ephemeral clip', async () => {
    await mountAssistant();
    const audio = await load('core/audio.js');
    t.env.input(fillerInputs()[0], 'Mm, one moment.');
    const button = $('se-ai-fillers').querySelectorAll('[data-action="ai-preview-filler"]')[0];
    assert.equal(button.getAttribute('aria-label'), 'Preview phrase 1');
    t.env.click(button);
    assert.match(audio.element().src, /^data:audio\/wav/, 'primed inside the click (iOS)');
    await settle(t.env);
    assert.deepEqual(t.server.speech, [{ model: 'kokoro', voice: 'af_heart', input: 'Mm, one moment.', response_format: 'mp3' }]);
    const player = t.app.get().player;
    assert.equal(player.id, 'settings:ai-preview');
    assert.equal(player.owner, 'settings');
    assert.equal(player.ephemeral, true, 'a preview stays out of the mini-player and the lock screen');
    assert.equal(player.playing, true);
    assert.match(audio.element().src, /^blob:/);
    assert.match(textOf($('se-ai-voice-hint')), /Heart/);
    // An empty error sentence previews the default one.
    t.env.input($('se-ai-err-assistant_timeout'), '');
    t.env.click($('se-ai-errors').querySelector('[data-action="ai-preview-error"][data-code="assistant_timeout"]'));
    await settle(t.env);
    assert.equal(t.server.speech.at(-1).input, SENTENCES.assistant_timeout);
    assert.equal(t.server.speech.length, 2);
  });

  it('Preview without a voice in Speak (or with a raw recording) says why and sends nothing; a failed synthesis shows next to the phrases', async () => {
    await mountAssistant({ speakVoice: { model: '', voice: '', label: '' } });
    assert.match(textOf($('se-ai-voice-hint')), /Choose a voice in Speak/);
    t.env.click($('se-ai-fillers').querySelectorAll('[data-action="ai-preview-filler"]')[0]);
    await settle(t.env);
    assert.equal(t.server.speech.length, 0);
    assert.match(textOf($('se-ai-waiting-banner')), /Choose a voice in Speak/);
    t.app.set({ speakVoice: { model: 'chatterbox', voice: 'demo_reference', label: 'demo_reference', kind: 'recording' } });
    await settle(t.env);
    assert.match(textOf($('se-ai-voice-hint')), /reference recording/);
    t.app.set({ speakVoice: SPEAK_VOICE });
    t.env.fetch.route('POST /v1/audio/speech', () => json(503, { error: { message: 'Kokoro is loading', code: 'model_loading' } }));
    t.env.click($('se-ai-fillers').querySelectorAll('[data-action="ai-preview-filler"]')[0]);
    await settle(t.env);
    assert.match(textOf($('se-ai-waiting-banner')), /Couldn't preview/);
    assert.match(textOf($('se-ai-waiting-banner')), /Kokoro is loading/);
  });

  it('phrases: Add (up to 10, focusing the new one), Remove, Move up and down; the edits are saved in their order; every control has a name', async () => {
    await mountAssistant();
    t.env.click($('se-ai-filler-add'));
    await settle(t.env);
    assert.equal(fillerInputs().length, 4);
    assert.equal(t.doc.activeElement, fillerInputs()[3]);
    t.env.input(fillerInputs()[3], 'One moment.');
    const row = (i) => $('se-ai-fillers').querySelectorAll('li')[i];
    t.env.click(row(3).querySelector('[data-action="ai-filler-up"]'));
    await settle(t.env);
    assert.deepEqual(fillerInputs().map((i) => i.value), ['Hmm.', 'OK, just a second.', 'One moment.', 'Let me think about that.']);
    assert.equal(t.doc.activeElement, row(2).querySelector('[data-action="ai-filler-up"]'), 'focus follows the moved phrase');
    assert.equal(row(0).querySelector('[data-action="ai-filler-up"]').getAttribute('aria-disabled'), 'true');
    assert.equal(row(3).querySelector('[data-action="ai-filler-down"]').getAttribute('aria-disabled'), 'true');
    t.env.click(row(0).querySelector('[data-action="ai-filler-remove"]'));
    await settle(t.env);
    assert.deepEqual(fillerInputs().map((i) => i.value), ['OK, just a second.', 'One moment.', 'Let me think about that.']);
    assert.equal(row(0).querySelector('[data-action="ai-filler-remove"]').getAttribute('aria-label'), 'Remove phrase 1');
    for (let i = 0; i < 7; i += 1) t.env.click($('se-ai-filler-add'));
    await settle(t.env);
    assert.equal(fillerInputs().length, 10);
    assert.equal($('se-ai-filler-add').getAttribute('aria-disabled'), 'true');
    t.env.click($('se-ai-filler-add'));
    assert.equal(fillerInputs().length, 10, 'at most 10');
    assertAccessibleNames(t.root);
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.deepEqual(t.server.puts[0].body.fillers, ['OK, just a second.', 'One moment.', 'Let me think about that.'], 'blank phrases are dropped');
    // Removing every phrase turns them off.
    while (fillerInputs().length) {
      t.env.click(row(0).querySelector('[data-action="ai-filler-remove"]'));
      await settle(t.env);
    }
    assert.match(textOf($('se-ai-fillers-empty')), /No phrases/);
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.deepEqual(t.server.puts[1].body.fillers, []);
  });

  it('a load the server refused runs again after a new key (the shell\'s auth reload); one that worked is not repeated (AU-12: Clear reloads no data)', async () => {
    let refuse = true;
    const fake = fakeAssistant();
    await mountAssistant({ routes: { 'GET /api/assistant': (req) => (refuse ? json(401, { error: { message: 'Invalid or missing API key', code: 'unauthorized' } }) : fake.routes['GET /api/assistant'](req)) } });
    assert.match(textOf($('se-ai-load')), /API key/);
    refuse = false;
    t.ctl.show(t.route, { focus: 'keep', scrolled: true, reason: 'auth' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(facts().Status, 'On');
    const before = t.env.fetch.callsTo('GET /api/assistant').length;
    t.ctl.show(t.route, { focus: 'keep', scrolled: true, reason: 'auth' });
    await settle(t.env);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant').length, before);
  });

  it('Test of a saved field emptied on screen sends it as null, so the AI gets the default the screen shows; a field left as saved goes as shown', async () => {
    await mountAssistant({ server: { stored: { voice: { reasoning_effort: 'medium', temperature: 0.7, system_prompt: 'Talk like a pirate.', max_tokens: 300 } } } });
    assert.equal($('se-ai-effort').value, 'medium');
    t.env.change($('se-ai-effort'), '');
    t.env.input($('se-ai-temperature'), '');
    t.env.input($('se-ai-system'), '');
    t.env.click($('se-ai-test'));
    await settle(t.env);
    const { overrides } = t.server.tests[0];
    assert.equal(overrides.reasoning_effort, null, 'not left out: the server would try the saved "medium"');
    assert.equal(overrides.temperature, null);
    assert.equal(overrides.system_prompt, null);
    assert.equal(overrides.max_tokens, 300, 'the saved value, as on screen');
    assert.equal(t.server.puts.length, 0, 'nothing is saved');
  });

  // The server takes `xhigh`, so the select offers it after
  // `high`, and a preset saved with it comes back with it.
  it('Reasoning effort offers xhigh after high; a preset saved with it round-trips', async () => {
    await mountAssistant();
    const values = () => $('se-ai-effort').querySelectorAll('option').map((o) => o.getAttribute('value'));
    assert.deepEqual(values(), ['', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'], 'listed in the page, not added for a value the server sent');
    t.env.change($('se-ai-effort'), 'xhigh');
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.equal(t.server.puts.length, 1);
    assert.equal(t.server.puts[0].body.reasoning_effort, 'xhigh');
    assert.equal($('se-ai-effort').value, 'xhigh', 'what the server saved');
    const stored = Object.fromEntries(t.server.store);
    t.env.restore();
    await mountAssistant({ server: { stored } });
    assert.equal($('se-ai-effort').value, 'xhigh', 'the next page shows it');
    assert.deepEqual(values(), ['', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'], 'listed once');
  });

  it('every control in the section has a name, and its live regions are its four status lines (Assistants, Connection, Test, Save)', async () => {
    await mountAssistant();
    const section = $('se-assistant-title').closest('section');
    assertAccessibleNames(section);
    const regions = section.querySelectorAll('[role="status"], [role="alert"], [aria-live]').map((el) => el.id);
    assert.deepEqual(regions, ['se-ai-assts-status', 'se-ai-conn-status', 'se-ai-test-status', 'se-ai-status']);
  });
});

/* ───────────── the Ask AI fields ───────────── */

/** The action fields the server gives a preset (src/assistant/presets.py _action_defaults). */
const ACTION_DEFAULTS = Object.freeze({
  voice: { label: 'Answer', inputs: ['text', 'transcript'], output: 'speech' },
  book: { label: 'Rewrite for listening', inputs: ['document'], output: 'document' },
});
function withActions(view, stored = {}) {
  const base = ACTION_DEFAULTS[view.id] || { label: view.name, inputs: ['text', 'transcript', 'document'], output: 'text' };
  const own = {};
  for (const field of ['label', 'inputs', 'output']) {
    const value = stored[field];
    if (value !== undefined && value !== null && value !== '' && !(Array.isArray(value) && !value.length)) own[field] = value;
  }
  return { ...view, ...base, ...own };
}

/** The fake server, with presets that carry the action fields; a PUT that leaves one out keeps it. */
function actionRoutes(fake, kept) {
  const routes = { ...fake.routes };
  const list = routes['GET /api/assistant/presets'];
  routes['GET /api/assistant/presets'] = (req) => list(req).json().then((data) => json(200, { presets: data.presets.map((p) => withActions(p, kept[p.id] || {})) }));
  const put = routes['PUT /api/assistant/presets/:id'];
  routes['PUT /api/assistant/presets/:id'] = async (req) => {
    const answer = await put(req);
    if (answer.status !== 200) return answer;
    const view = await answer.json();
    const mine = kept[req.params.id] || (kept[req.params.id] = {});
    for (const field of ['label', 'inputs', 'output']) if (field in req.json) mine[field] = req.json[field];
    return json(200, withActions(view, mine));
  };
  const post = routes['POST /api/assistant/presets'];
  routes['POST /api/assistant/presets'] = async (req) => {
    const answer = await post(req);
    if (answer.status !== 201) return answer;
    const view = await answer.json();
    const mine = {};
    for (const field of ['label', 'inputs', 'output']) if (field in req.json) mine[field] = req.json[field];
    kept[view.id] = mine;
    return json(201, withActions(view, mine));
  };
  return routes;
}

async function mountWithActions({ stored = {}, kept = {} } = {}) {
  const fake = fakeAssistant({ stored });
  t = await mountSettings({ routes: actionRoutes(fake, kept) });
  t.server = fake.state;
  t.kept = kept;
  t.app.set({ speakVoice: SPEAK_VOICE });
  await settle(t.env);
  await settle(t.env);
  return t;
}

const inputBoxes = () => ['text', 'transcript', 'document'].map((kind) => $(`se-ai-inputs-${kind}`).checked);

describe('settings/assistant: the Ask AI fields', () => {
  it('pure: presetValues and saveBody carry label, inputs and output only when changed (or kept on a copy), never for an older server, and never in a Test', async () => {
    installBrowserEnv();
    const { presetValues, saveBody, isDirty, testBody, FIELD_LABELS } = await importFresh('views/settings/assistant-logic.js');
    const view = withActions(viewOf('voice'));
    const values = presetValues(view);
    assert.deepEqual([values.label, values.inputs, values.output], ['Answer', ['text', 'transcript'], 'speech']);
    assert.deepEqual(saveBody(view, values), { name: 'Voice' }, 'unchanged: left out, so the server keeps them');
    const edited = { ...values, label: '  Reply ', inputs: ['transcript', 'document'], output: 'text' };
    assert.equal(isDirty(view, edited), true);
    assert.deepEqual(saveBody(view, edited), { name: 'Voice', label: 'Reply', inputs: ['transcript', 'document'], output: 'text' });
    assert.deepEqual(saveBody(view, { ...values, label: '', inputs: [] }), { name: 'Voice', label: '', inputs: [] }, 'cleared: the default again');
    // A copy keeps where it is offered and its first answer; the built-in's label is its own default, not copied.
    assert.deepEqual(saveBody(view, { ...values, name: 'Voice copy' }, { copy: true }), { name: 'Voice copy', inputs: ['text', 'transcript'], output: 'speech' });
    const mine = withActions(viewOf('summaries'), { label: 'Summarize' });
    assert.equal(saveBody(mine, { ...presetValues(mine), name: 'Copy' }, { copy: true }).label, 'Summarize', 'a label of its own is copied');
    // An older server's preset has no such fields: none is ever sent.
    const old = viewOf('voice');
    assert.deepEqual(saveBody(old, { ...presetValues(old), label: 'X', output: 'document' }), { name: 'Voice' });
    const overrides = testBody('voice', '', edited).overrides;
    assert.ok(!('label' in overrides) && !('inputs' in overrides) && !('output' in overrides), 'a Test never sends them');
    for (const field of ['label', 'inputs', 'output']) assert.ok(FIELD_LABELS[field], field);
  });

  it('the form shows them, marks a change unsaved, and Save sends only the changed ones; the server keeps the rest', async () => {
    await mountWithActions();
    assert.equal($('se-ai-label').value, 'Answer');
    assert.equal($('se-ai-output').value, 'speech');
    assert.deepEqual(inputBoxes(), [true, true, false]);
    assert.equal(shown('se-ai-dirty'), false);
    t.env.input($('se-ai-label'), 'Reply');
    t.env.change($('se-ai-inputs-document'), true);
    assert.equal(shown('se-ai-dirty'), true);
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.deepEqual(t.server.puts[0].body, { name: 'Voice', label: 'Reply', inputs: ['text', 'transcript', 'document'] });
    assert.equal($('se-ai-label').value, 'Reply');
    assert.deepEqual(inputBoxes(), [true, true, true]);
    assert.equal(shown('se-ai-dirty'), false);
    // Another save that leaves them alone sends none: the server keeps them.
    t.env.input($('se-ai-system'), 'Be brief.');
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.deepEqual(t.server.puts[1].body, { name: 'Voice', system_prompt: 'Be brief.' });
    assert.equal(t.kept.voice.label, 'Reply');
    await pick('book');
    assert.equal($('se-ai-label').value, 'Rewrite for listening');
    assert.equal($('se-ai-output').value, 'document');
    assert.deepEqual(inputBoxes(), [false, false, true]);
  });

  it('a 422 on one of them lands on its field', async () => {
    const fake = fakeAssistant();
    const routes = actionRoutes(fake, {});
    routes['PUT /api/assistant/presets/:id'] = () => json(422, { error: { message: 'inputs must be a list of text, transcript or document', code: 'validation_error', param: 'inputs' } });
    t = await mountSettings({ routes });
    t.server = fake.state;
    await settle(t.env);
    await settle(t.env);
    t.env.change($('se-ai-inputs-document'), true);
    t.env.click($('se-ai-save'));
    await settle(t.env);
    const box = $('se-ai-inputs-text');
    assert.equal(box.getAttribute('aria-invalid'), 'true');
    assert.match(textOf($('se-ai-inputs-error')), /^Offered for must be a list/);
    assert.ok(box.getAttribute('aria-describedby').split(' ').includes('se-ai-inputs-error'));
  });
});

/* ───────────── the thinking budget ───────────── */

// Presets › Thinking › "Thinking budget (tokens)", the preset field thinking_budget
// (0 to 1,000,000; 0 sends none), which the server sends as reasoning_budget_tokens. And the Max tokens hint says that
// the limit cuts the answer off without telling the model.
describe('settings/assistant: the thinking budget', () => {
  it('pure: a number field after the effort, with its label; shown, saved and tested like Max tokens (empty is .env\'s, 0 sends none)', async () => {
    installBrowserEnv();
    const { presetValues, formValues, saveBody, testBody, isDirty, fieldProblem, FIELD_LABELS, NUMBER_FIELDS, TUNABLE: fields } = await importFresh('views/settings/assistant-logic.js');
    assert.equal(fields.indexOf('thinking_budget'), fields.indexOf('reasoning_effort') + 1, 'after the effort, as the server lists it');
    assert.ok(NUMBER_FIELDS.includes('thinking_budget'));
    assert.equal(FIELD_LABELS.thinking_budget, 'Thinking budget (tokens)');
    // From .env: shown, and left out of a save while untouched.
    const inherited = viewOf('voice', {}, { thinking_budget: 2048 });
    const values = presetValues(inherited);
    assert.equal(values.thinking_budget, 2048);
    assert.equal('thinking_budget' in saveBody(inherited, values), false);
    // Changed on screen: sent.
    const typed = formValues({ ...rawOf(values), thinking_budget: ' 512 ' }).values;
    assert.equal(typed.thinking_budget, 512);
    assert.equal(isDirty(inherited, typed), true);
    assert.deepEqual(saveBody(inherited, typed), { name: 'Voice', thinking_budget: 512 });
    // The preset's own: in every save (0 is a value: no budget); cleared, sent as null, so .env's applies again.
    const own = viewOf('voice', { thinking_budget: 0 });
    assert.deepEqual(saveBody(own, presetValues(own)), { name: 'Voice', thinking_budget: 0 });
    const cleared = formValues({ ...rawOf(presetValues(own)), thinking_budget: '' }).values;
    assert.equal(cleared.thinking_budget, null);
    assert.deepEqual(saveBody(own, cleared), { name: 'Voice', thinking_budget: null });
    // Test: as on screen; null when emptied.
    assert.equal(testBody('voice', '', typed).overrides.thinking_budget, 512);
    assert.equal(testBody('voice', '', presetValues(own)).overrides.thinking_budget, 0);
    assert.equal(testBody('voice', '', cleared).overrides.thinking_budget, null);
    // Not a number: a problem on the field before anything is sent; the server's 422 reads with the label.
    assert.deepEqual(formValues({ ...rawOf(values), thinking_budget: 'lots' }).problems.map((p) => p.field), ['thinking_budget']);
    assert.equal(fieldProblem('thinking_budget', 'thinking_budget must be a whole number from 0 to 1000000').text, 'Thinking budget (tokens) must be a whole number from 0 to 1000000.');
  });

  it('on screen: in the Thinking group with its hint; the preset\'s value shows; a change is unsaved, tested and saved; a 422 lands on it; the Max tokens hint says it cuts the answer off', async () => {
    await mountAssistant({ server: { stored: { voice: { thinking_budget: 1500 } } } });
    const budget = $('se-ai-thinking-budget');
    assert.ok(budget, 'the field is there');
    assert.equal(budget.closest('[role="group"]').getAttribute('aria-labelledby'), 'se-ai-thinking-title');
    assert.equal(budget.value, '1500');
    const hint = textOf($('se-ai-thinking-budget-hint'));
    assert.match(hint, /reasoning_budget_tokens/);
    assert.match(hint, /0 sends none/);
    assert.match(hint, /ignore or refuse/);
    assert.match(hint, /Test shows whether yours takes it/);
    const limit = textOf($('se-ai-max-tokens-hint'));
    assert.match(limit, /cuts the answer off without telling the model/);
    assert.match(limit, /thinking counts towards it/);
    assert.match(limit, /prompt/);
    assert.match(limit, /safety net/);
    assert.equal(shown('se-ai-dirty'), false);
    t.env.input(budget, '2048');
    assert.equal(shown('se-ai-dirty'), true);
    t.env.click($('se-ai-test'));
    await settle(t.env);
    assert.equal(t.server.tests[0].overrides.thinking_budget, 2048, 'Test sends it as on screen');
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.deepEqual(t.server.puts[0].body, { name: 'Voice', thinking_budget: 2048 });
    assert.equal(budget.value, '2048');
    assert.equal(shown('se-ai-dirty'), false);
    // Too big: the server's 422 lands on the field.
    t.env.input(budget, '1000001');
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.equal(budget.getAttribute('aria-invalid'), 'true');
    assert.equal(textOf($('se-ai-thinking-budget-error')), 'Thinking budget (tokens) must be a whole number from 0 to 1000000.');
    assert.ok(budget.getAttribute('aria-describedby').split(' ').includes('se-ai-thinking-budget-error'));
    assert.equal(t.doc.activeElement, budget);
    // Cleared: sent as null, and the field shows the default again (0, none).
    t.env.input(budget, '');
    t.env.click($('se-ai-save'));
    await settle(t.env);
    assert.equal(t.server.puts.at(-1).body.thinking_budget, null);
    assert.equal(budget.value, '0');
    assertAccessibleNames(t.root);
  });
});
