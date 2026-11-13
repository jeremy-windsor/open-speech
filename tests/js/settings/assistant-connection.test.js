// Settings › AI assistant › Connection: the form that sets the AI
// server's URL, model, key and the other connection settings in the web app. Loading, Check (success, an error,
// no models), Save, Revert to .env, Clear key, Use .env, the locked state, field errors, and the key that stays
// with its server (a new origin needs the key again). The API is src/routes/assistant.py and
// src/assistant/connection.py; the fake below answers with the same save rules and messages.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';
import { mountSettings, settle, json, textOf, toasts } from './helpers.js';

let t;
afterEach(() => t?.env?.restore());

// The server's fields (src/assistant/connection.py FIELDS).
const FIELDS = ['url', 'api_key', 'model', 'verify_tls', 'template_kwargs', 'max_tokens_field', 'extra_body', 'stream'];
const DEFAULTS = Object.freeze({ url: '', api_key: '', model: '', verify_tls: true, template_kwargs: false, max_tokens_field: 'max_tokens', extra_body: {}, stream: true });
const AI_SERVER = 'http://ai-server.lan:8080/v1';
const OTHER = 'http://other.lan:9000/v1';
const TUNABLE = [
  'model', 'system_prompt', 'user_template', 'thinking', 'reasoning_effort', 'thinking_budget', 'show_thinking',
  'max_tokens', 'temperature', 'top_p', 'timeout_s', 'history_turns', 'fillers', 'filler_after_ms',
  'filler_every_ms', 'filler_max', 'speak_errors', 'error_sentences', 'chunk_chars',
];

const originOf = (url) => {
  try {
    const u = new URL(url);
    return u.protocol === 'http:' || u.protocol === 'https:' ? u.origin : '';
  } catch {
    return '';
  }
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const invalid = (param, message) => json(422, { error: { message, code: 'validation_error', param } });
const lockedAnswer = () => json(403, { error: { message: 'The AI server connection is locked: it is set only in .env (OS_ASSISTANT_CONNECTION_LOCKED)', code: 'connection_locked' } });
const disabledAnswer = () => json(404, { error: { message: "The AI assistant is off: set the AI server's URL in Settings or OS_ASSISTANT_URL", code: 'assistant_disabled' } });

/** connection.validate_fields: [changes, null] or [null, the 422]. The messages are the server's. */
function validate(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return [null, invalid('body', 'body must be a JSON object')];
  const changes = {};
  for (const [name, value] of Object.entries(body)) {
    if (['enabled', 'locked', 'api_key_set', 'api_key_tls_checked', 'sources'].includes(name)) continue;
    if (!FIELDS.includes(name)) return [null, invalid(name, `${name} is not a connection field; the fields are ${FIELDS.join(', ')}`)];
    if (value === null) {
      changes[name] = null;
    } else if (name === 'url') {
      const text = typeof value === 'string' ? value.trim() : null;
      if (text === null || (text && !/^https?:\/\/[^\s/?#@]+(\/[^\s?#]*)?$/.test(text))) return [null, invalid('url', 'url must be an http:// or https:// base URL such as http://ai-server.lan:8080/v1')];
      changes.url = text.replace(/\/+$/, '');
    } else if (name === 'api_key' || name === 'model') {
      if (typeof value !== 'string') return [null, invalid(name, `${name} must be text`)];
      changes[name] = value.trim();
    } else if (name === 'verify_tls' || name === 'template_kwargs' || name === 'stream') {
      if (typeof value !== 'boolean') return [null, invalid(name, `${name} must be true or false`)];
      changes[name] = value;
    } else if (name === 'max_tokens_field') {
      if (!['max_tokens', 'max_completion_tokens'].includes(value)) return [null, invalid(name, 'max_tokens_field must be one of max_tokens, max_completion_tokens')];
      changes[name] = value;
    } else {
      let parsed = value;
      if (typeof value === 'string') {
        try {
          parsed = value.trim() ? JSON.parse(value) : {};
        } catch (error) {
          return [null, invalid(name, `extra_body must be a JSON object such as {"top_k": 20}: ${error.message}`)];
        }
      }
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return [null, invalid(name, 'extra_body must be a JSON object such as {"top_k": 20}, not list')];
      changes[name] = parsed;
    }
  }
  return [changes, null];
}

/** A built-in preset's answer (the fields the editor reads), with the connection's model as its default. */
function presetView(id, model) {
  return {
    id, name: id === 'voice' ? 'Voice' : 'Book', builtin: true, customized: false, inherited: [...TUNABLE],
    model, system_prompt: 'Be brief.', user_template: '{text}', thinking: 'default', reasoning_effort: '', thinking_budget: 0, show_thinking: true,
    max_tokens: 0, temperature: null, top_p: null, timeout_s: id === 'book' ? 600 : 120, history_turns: 20, fillers: ['Hmm.'],
    filler_after_ms: 1200, filler_every_ms: 4000, filler_max: 3, speak_errors: true, error_sentences: {}, chunk_chars: 12000,
  };
}

/**
 * A fake of the connection API over `.env` (`env`) and a saved file (`saved`), with connection.py's rules: a saved
 * field overrides .env; null forgets it; the key is write-only and goes only to the origin it was saved for (the
 * .env key only to OS_ASSISTANT_URL's), so a save that moves the URL to another origin without a key clears it.
 * And a key set with the certificate check on goes to no https URL with the check off, so a save
 * that turns the check off for one without a key clears a saved key; the .env key counts as set with .env's check.
 */
function fakeServer({ env = {}, saved = {}, locked = false } = {}) {
  const ENV = { ...DEFAULTS, ...env };
  const state = { saved: { ...saved }, locked, puts: [], deletes: 0, checks: [], checkAnswer: null };
  if (state.saved.api_key && !('api_key_origin' in state.saved)) state.saved.api_key_origin = originOf(state.saved.url ?? ENV.url);
  const uncheckedHttps = (origin, verifyTls) => origin.startsWith('https://') && !verifyTls;

  function resolve(stored) {
    const values = {};
    const sources = {};
    for (const name of FIELDS) {
      if (!state.locked && name in stored) {
        values[name] = stored[name];
        sources[name] = 'saved';
      } else {
        values[name] = ENV[name];
        sources[name] = same(ENV[name], DEFAULTS[name]) ? 'default' : 'env';
      }
    }
    const here = originOf(values.url);
    const keyOrigin = sources.api_key === 'saved' ? (stored.api_key_origin ?? here) : originOf(ENV.url);
    const keyTlsChecked = sources.api_key === 'saved' ? (stored.api_key_tls_checked ?? values.verify_tls) : ENV.verify_tls;
    if (values.api_key && (keyOrigin !== here || (keyTlsChecked && uncheckedHttps(here, values.verify_tls)))) {
      values.api_key = '';
      sources.api_key = 'default';
    }
    // Whether the key in use was set with the check on, read-only; null with no key.
    return { values, sources, keyTlsChecked: values.api_key ? !!keyTlsChecked : null };
  }
  function apply(stored, changes) {
    const before = originOf(stored.url ?? ENV.url);
    if (stored.api_key && !('api_key_origin' in stored)) stored.api_key_origin = before;
    if (stored.api_key && !('api_key_tls_checked' in stored)) stored.api_key_tls_checked = stored.verify_tls ?? ENV.verify_tls;
    for (const [name, value] of Object.entries(changes)) {
      if (value === null) delete stored[name];
      else stored[name] = value;
    }
    const after = originOf(stored.url ?? ENV.url);
    const afterTls = stored.verify_tls ?? ENV.verify_tls;
    if ('api_key' in changes) {
      delete stored.api_key_origin;
      delete stored.api_key_tls_checked;
      if (stored.api_key) {
        stored.api_key_origin = after;
        stored.api_key_tls_checked = afterTls;
      }
    } else if (stored.api_key && ((after && stored.api_key_origin !== after) || (stored.api_key_tls_checked && uncheckedHttps(after, afterTls)))) {
      delete stored.api_key;
      delete stored.api_key_origin;
      delete stored.api_key_tls_checked;
    }
  }
  function view() {
    const { values, sources, keyTlsChecked } = resolve(state.saved);
    return {
      enabled: !!values.url, locked: state.locked, url: values.url, model: values.model, api_key_set: !!values.api_key,
      api_key_tls_checked: keyTlsChecked, verify_tls: values.verify_tls, template_kwargs: values.template_kwargs, max_tokens_field: values.max_tokens_field,
      extra_body: structuredClone(values.extra_body), stream: values.stream, sources,
    };
  }
  const routes = {
    'GET /api/assistant': () => {
      const { values, sources } = resolve(state.saved);
      return json(200, {
        enabled: !!values.url, host: values.url ? new URL(values.url).host : null, model: values.model,
        server_settings: {
          OS_ASSISTANT_URL: !!values.url, OS_ASSISTANT_API_KEY: !!values.api_key, OS_ASSISTANT_EXTRA_BODY: Object.keys(values.extra_body).length > 0,
          OS_ASSISTANT_VERIFY_TLS: values.verify_tls !== true, OS_ASSISTANT_MAX_TOKENS_FIELD: values.max_tokens_field !== 'max_tokens',
          OS_ASSISTANT_TEMPLATE_KWARGS: !!values.template_kwargs, OS_ASSISTANT_STREAM: values.stream !== true,
        },
        locked: state.locked,
        sources: { OS_ASSISTANT_URL: sources.url, OS_ASSISTANT_API_KEY: sources.api_key, OS_ASSISTANT_MODEL: sources.model, OS_ASSISTANT_VERIFY_TLS: sources.verify_tls, OS_ASSISTANT_TEMPLATE_KWARGS: sources.template_kwargs, OS_ASSISTANT_MAX_TOKENS_FIELD: sources.max_tokens_field, OS_ASSISTANT_EXTRA_BODY: sources.extra_body, OS_ASSISTANT_STREAM: sources.stream },
      });
    },
    'GET /api/assistant/presets': () => {
      const { values } = resolve(state.saved);
      return values.url ? json(200, { presets: [presetView('voice', values.model), presetView('book', values.model)] }) : disabledAnswer();
    },
    'GET /api/assistant/connection': () => json(200, view()),
    'PUT /api/assistant/connection': (req) => {
      state.puts.push(req.json);
      if (state.locked) return lockedAnswer();
      const [changes, bad] = validate(req.json);
      if (bad) return bad;
      apply(state.saved, changes);
      return json(200, view());
    },
    'DELETE /api/assistant/connection': () => {
      state.deletes += 1;
      if (state.locked) return lockedAnswer();
      state.saved = {};
      return json(200, view());
    },
    'POST /api/assistant/connection/check': (req) => {
      state.checks.push(req.json);
      const [changes, bad] = validate(req.json ?? {});
      if (bad) return bad;
      if (state.locked && Object.keys(changes).length) return lockedAnswer();
      const trial = structuredClone(state.saved);
      apply(trial, changes);
      const { values } = resolve(trial);
      if (!values.url) return invalid('url', "url is needed: the AI server's base URL, such as http://ai-server.lan:8080/v1");
      if (state.checkAnswer) return state.checkAnswer(values);
      return json(200, { reachable: true, status: 200, models: ['qwen3-30b', 'qwen3-4b'], listed: true, elapsed_ms: 42 });
    },
  };
  return { routes, state };
}

/** Settings with the fake connection API, the AI assistant section loaded. */
async function mountConnection({ env, saved, locked, routes = {} } = {}) {
  const fake = fakeServer({ env, saved, locked });
  t = await mountSettings({ routes: { ...fake.routes, ...routes } });
  t.server = fake.state;
  await settle(t.env);
  await settle(t.env);
  return t;
}

const $ = (id) => t.$(id);
const shown = (id) => {
  for (let el = $(id); el && el !== t.root; el = el.parentElement) if (el.hidden) return false;
  return !!$(id);
};
const facts = (id = 'se-ai-facts') => {
  const dl = $(id);
  const terms = dl.querySelectorAll('dt').map(textOf);
  const values = dl.querySelectorAll('dd').map(textOf);
  return Object.fromEntries(terms.map((term, i) => [term, values[i]]));
};
/** Each field's source badge, by field. */
const IDS = { url: 'url', api_key: 'key', model: 'model', verify_tls: 'tls', template_kwargs: 'template', max_tokens_field: 'max-field', extra_body: 'extra', stream: 'stream' };
const sources = () => Object.fromEntries(FIELDS.map((f) => [f, textOf($(`se-ai-conn-${IDS[f]}-source`))]));
const resets = () => FIELDS.filter((f) => shown(`se-ai-conn-${IDS[f]}-reset`));
const describedBy = (id) => ($(id).getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
async function answerConfirm(ok) {
  await settle(t.env);
  assert.equal($('app-confirm').open, true, 'a confirm dialog is open');
  t.env.click($(ok ? 'app-confirm-ok' : 'app-confirm-cancel'));
  await settle(t.env);
  await settle(t.env);
}
async function click(id) {
  t.env.click($(id));
  await settle(t.env);
  await settle(t.env);
}

/* ───────────────────────────── the pure parts ───────────────────────────── */

describe('settings/assistant-connection: pure parts', () => {
  const VIEW = Object.freeze({
    enabled: true, locked: false, url: AI_SERVER, model: 'qwen3', api_key_set: true, verify_tls: true, template_kwargs: false,
    max_tokens_field: 'max_tokens', extra_body: { top_k: 20 }, stream: true,
    sources: { url: 'env', api_key: 'saved', model: 'env', verify_tls: 'default', template_kwargs: 'default', max_tokens_field: 'default', extra_body: 'saved', stream: 'default' },
  });

  it('viewValues and changedFields: the form holds the view\'s values (extra body as JSON text); only real changes count (a trailing slash, spaces, the JSON\'s layout don\'t)', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const values = C.viewValues(VIEW);
    assert.deepEqual(Object.keys(values).sort(), ['extra_body', 'max_tokens_field', 'model', 'stream', 'template_kwargs', 'url', 'verify_tls']);
    assert.equal(values.url, AI_SERVER);
    assert.deepEqual(JSON.parse(values.extra_body), { top_k: 20 });
    assert.equal(C.viewValues({ ...VIEW, extra_body: {} }).extra_body, '', 'no extra body is an empty field');
    assert.deepEqual(C.changedFields(VIEW, values), []);
    assert.deepEqual(C.changedFields(VIEW, { ...values, url: ` ${AI_SERVER}/ `, model: ' qwen3 ', extra_body: '{"top_k":20}' }), []);
    assert.deepEqual(C.changedFields(VIEW, { ...values, url: OTHER, verify_tls: false, extra_body: '{"top_k": 40}' }), ['url', 'verify_tls', 'extra_body']);
    assert.deepEqual(C.changedFields(VIEW, { ...values, extra_body: '{oops' }), ['extra_body'], 'text that isn\'t JSON is a change (the server says why)');
    assert.deepEqual(C.changedFields({ ...VIEW, extra_body: {} }, { ...values, extra_body: '  ' }), []);
    assert.equal(C.isDirty(VIEW, values, ''), false);
    assert.equal(C.isDirty(VIEW, values, '  '), false);
    assert.equal(C.isDirty(VIEW, values, 'sk-new'), true, 'a typed key is a change');
    assert.equal(C.isDirty(VIEW, { ...values, max_tokens_field: 'max_completion_tokens' }, ''), true);
  });

  it('changeBody: only the fields changed on screen (every field a PUT sends is saved), the key only when one is typed; a Check sends the same body ({} for none)', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const values = C.viewValues(VIEW);
    assert.deepEqual(C.changeBody(VIEW, values, ''), {});
    assert.deepEqual(C.changeBody(VIEW, { ...values, model: ' qwen3-30b ', template_kwargs: true }, ''), { model: 'qwen3-30b', template_kwargs: true });
    assert.deepEqual(C.changeBody(VIEW, { ...values, url: ` ${OTHER} ` }, ' sk-other '), { url: OTHER, api_key: 'sk-other' });
    assert.deepEqual(C.changeBody(VIEW, { ...values, extra_body: '' }, ''), { extra_body: '' }, 'an emptied extra body is sent as "" (no extra fields)');
    assert.deepEqual(C.changeBody(VIEW, { ...values, extra_body: '{"top_k": 40}' }, ''), { extra_body: '{"top_k": 40}' });
    assert.deepEqual(C.changeBody(VIEW, { ...values, url: '' }, ''), { url: '' }, 'an emptied URL turns the assistant off');
    assert.equal('api_key' in C.changeBody(VIEW, values, ''), false, 'an empty key field never clears the key');
  });

  it('originOf and keyAgain: "Enter the key again" when the URL moves to another origin (scheme, host, port; 80 and 443 filled in), a key was set and none is typed; a path change keeps the key', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    assert.equal(C.originOf('http://AI-Server.LAN:8080/v1'), 'http://ai-server.lan:8080');
    assert.equal(C.originOf('http://ai-server.lan:80/v1'), C.originOf('http://ai-server.lan/api'));
    assert.equal(C.originOf('https://ai.example.com:443/v1'), C.originOf('https://ai.example.com/v1'));
    assert.notEqual(C.originOf('http://ai-server.lan/v1'), C.originOf('https://ai-server.lan/v1'));
    for (const bad of ['', 'ai-server:8080', 'ftp://ai-server/v1', 'not a url']) assert.equal(C.originOf(bad), '', bad);
    const values = C.viewValues(VIEW);
    assert.equal(C.keyAgain(VIEW, values, ''), false);
    assert.equal(C.keyAgain(VIEW, { ...values, url: 'http://AI-SERVER.lan:8080/api/v1' }, ''), false, 'a change of path keeps the key');
    assert.equal(C.keyAgain(VIEW, { ...values, url: OTHER }, ''), true);
    assert.equal(C.keyAgain(VIEW, { ...values, url: 'https://ai-server.lan:8080/v1' }, ''), true, 'another scheme is another origin');
    assert.equal(C.keyAgain(VIEW, { ...values, url: OTHER }, 'sk-other'), false, 'a key is given');
    assert.equal(C.keyAgain(VIEW, { ...values, url: '' }, ''), false, 'no URL: the key is kept, sent nowhere');
    assert.equal(C.keyAgain({ ...VIEW, api_key_set: false }, { ...values, url: OTHER }, ''), false, 'no key to lose');
  });

  // A save that turns "Check TLS certificate" off for an https URL
  // without a key clears a key set with the check on, so "Enter the key again" shows then too, in words that fit.
  it('keyAgainReason: "origin" for another server, "tls" for turning the certificate check off for an https URL (a key set, none typed), each with its words; keyAgain is true for both', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const HTTPS = Object.freeze({ ...VIEW, url: 'https://ai.example.com/v1' });
    const values = C.viewValues(HTTPS);
    const off = { ...values, verify_tls: false };
    assert.equal(C.keyAgainReason(HTTPS, values, ''), '');
    assert.equal(C.keyAgainReason(HTTPS, off, ''), 'tls');
    assert.equal(C.keyAgain(HTTPS, off, ''), true, 'the save clears the key: the helper shows');
    assert.equal(C.keyAgainReason(HTTPS, { ...off, url: 'https://AI.example.com:443/other/v1' }, ''), 'tls', 'a change of path, the check off');
    assert.equal(C.keyAgainReason(HTTPS, off, ' sk-new '), '', 'a key is given in the same save');
    assert.equal(C.keyAgainReason({ ...HTTPS, api_key_set: false }, off, ''), '', 'no key to lose');
    assert.equal(C.keyAgainReason({ ...HTTPS, verify_tls: false }, off, ''), '', 'the check was off already: the key was set with it off');
    assert.equal(C.keyAgainReason(VIEW, { ...C.viewValues(VIEW), verify_tls: false }, ''), '', 'an http URL has no certificate');
    assert.equal(C.keyAgainReason(HTTPS, { ...off, url: '' }, ''), '', 'no URL: the key is kept, sent nowhere');
    assert.equal(C.keyAgainReason(VIEW, { ...C.viewValues(VIEW), url: 'https://ai-server.lan:8080/v1', verify_tls: false }, ''), 'origin', 'another server as well: its words');
    assert.equal(C.keyAgainReason(VIEW, { ...C.viewValues(VIEW), url: OTHER }, ''), 'origin');
    assert.equal(C.KEY_AGAIN.tls, 'Turning off the certificate check clears the saved key: enter it again to keep using it.');
    assert.match(C.KEY_AGAIN.origin, /^Enter the key again for a new server\. /);
  });

  // A key set with the check off, then the check turned on, is kept when the check is
  // turned off again; the view says so (api_key_tls_checked), and the form believes it.
  it('keyAgainReason: "tls" only for a key set with the check on (api_key_tls_checked); a key set with it off is kept, so nothing asks for it; a server without the field keeps the old guess', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const HTTPS = Object.freeze({ ...VIEW, url: 'https://ai.example.com/v1' });
    const off = { ...C.viewValues(HTTPS), verify_tls: false };
    assert.equal(C.keyAgainReason({ ...HTTPS, api_key_tls_checked: false }, off, ''), '', 'set with the check off: the save keeps it');
    assert.equal(C.keyAgain({ ...HTTPS, api_key_tls_checked: false }, off, ''), false);
    assert.equal(C.keyAgainReason({ ...HTTPS, api_key_tls_checked: true }, off, ''), 'tls', 'set with the check on: the save clears it');
    assert.equal(C.keyAgainReason(HTTPS, off, ''), 'tls', 'no field (an older server): the guess from verify_tls');
    assert.equal(C.keyAgainReason({ ...HTTPS, api_key_tls_checked: false }, { ...off, url: 'https://other.example.com/v1' }, ''), 'origin', 'another server still clears it');
  });

  it('rebase: after a single-field change (Clear key, Use .env), fields untouched on screen take the new values and edits stay; a forced field takes the new value', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const after = { ...VIEW, model: 'qwen3-env', url: 'http://ai-server.lan:8080/api/v1', sources: { ...VIEW.sources, model: 'env' } };
    const edited = { ...C.viewValues(VIEW), url: OTHER };
    const out = C.rebase(VIEW, after, edited);
    assert.equal(out.url, OTHER, 'the edit stays');
    assert.equal(out.model, 'qwen3-env', 'untouched: the new value');
    assert.equal(C.rebase(VIEW, after, edited, { force: 'url' }).url, 'http://ai-server.lan:8080/api/v1');
  });

  it('sourceOf, sourceText and the labels: where each value comes from in words; every field has a label', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    assert.deepEqual(FIELDS.map((f) => C.sourceText(C.sourceOf(VIEW, f))), ['from .env', 'saved here', 'from .env', 'default', 'default', 'default', 'saved here', 'default']);
    assert.equal(C.sourceOf({}, 'url'), 'default');
    assert.equal(C.sourceOf({ sources: { url: 'nonsense' } }, 'url'), 'default');
    assert.deepEqual([...C.FIELDS], FIELDS);
    for (const field of FIELDS) assert.ok(C.LABELS[field], field);
    assert.deepEqual([...C.MAX_TOKENS_FIELDS], ['max_tokens', 'max_completion_tokens']);
  });

  it('fieldProblem: a 422 names its field (param); the message starts with the field\'s label; no field of the form (body, enabled) gives null', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    assert.deepEqual(C.fieldProblem('url', 'url must be an http:// or https:// base URL such as http://ai-server.lan:8080/v1'), { field: 'url', text: 'Server URL must be an http:// or https:// base URL such as http://ai-server.lan:8080/v1.' });
    assert.equal(C.fieldProblem('url', "url is needed: the AI server's base URL, such as http://ai-server.lan:8080/v1").text, "Server URL is needed: the AI server's base URL, such as http://ai-server.lan:8080/v1.");
    assert.equal(C.fieldProblem('extra_body', 'extra_body must be a JSON object such as {"top_k": 20}: Expecting value').text, 'Extra body must be a JSON object such as {"top_k": 20}: Expecting value.');
    assert.equal(C.fieldProblem('api_key', 'api_key must be printable ASCII of up to 4096 characters').text, 'API key must be printable ASCII of up to 4096 characters.');
    assert.equal(C.fieldProblem('model', '').text, "Model wasn't accepted.");
    for (const param of ['body', 'enabled', 'nope', '']) assert.equal(C.fieldProblem(param, 'x'), null, param);
  });

  it('checkResult: reachable with models, with none, not a list, a refused key and a 404 in plain words; checkFailure: unreachable and timeout', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const ok = C.checkResult({ reachable: true, status: 200, models: ['qwen3-30b', 'qwen3-4b', 'qwen3-30b', 7], listed: true, elapsed_ms: 1234 });
    assert.equal(ok.tone, 'ok');
    assert.equal(ok.title, 'Reachable');
    assert.deepEqual(ok.models, ['qwen3-30b', 'qwen3-4b'], 'ids only, once each');
    assert.match(ok.message, /1\.2 s/);
    assert.match(ok.message, /2 models/);
    assert.equal(ok.status, 'Reachable in 1.2 s: 2 models');
    const none = C.checkResult({ reachable: true, status: 200, models: [], listed: true, elapsed_ms: 80 });
    assert.deepEqual(none.models, []);
    assert.match(none.message, /lists no models/);
    assert.match(none.message, /Type the model name/);
    const notList = C.checkResult({ reachable: true, status: 200, models: [], listed: false, elapsed_ms: 80 });
    assert.match(notList.message, /not with a list of models/);
    const refused = C.checkResult({ reachable: true, status: 401, models: [], listed: false, elapsed_ms: 30 });
    assert.equal(refused.tone, 'warn');
    assert.match(refused.title, /API key/);
    assert.match(refused.message, /HTTP 401/);
    const missing = C.checkResult({ reachable: true, status: 404, models: [], listed: false, elapsed_ms: 30 });
    assert.match(missing.message, /\/v1/);
    assert.match(C.checkResult({ reachable: true, status: 500, models: [], listed: false, elapsed_ms: 30 }).title, /HTTP 500/);
    assert.equal(C.checkFailure('assistant_unreachable').title, "Couldn't reach the AI server");
    assert.match(C.checkFailure('assistant_unreachable').message, /Open Speech's server/);
    assert.match(C.checkFailure('assistant_timeout').title, /in time/);
    assert.equal(C.checkFailure('validation_error'), null);
  });

  it('lockedFacts: the read-only list, each value in words with where it comes from; the key only as set or not set', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const list = C.lockedFacts({ ...VIEW, locked: true, verify_tls: false, sources: { ...VIEW.sources, verify_tls: 'env', api_key: 'env', extra_body: 'env' } });
    assert.deepEqual(list.map((f) => [f.term, f.value, f.source]), [
      ['Server URL', AI_SERVER, 'env'],
      ['Model', 'qwen3', 'env'],
      ['API key', 'Set', 'env'],
      ['Check TLS certificate', 'Off', 'env'],
      ['Send thinking as template variables', 'Off', 'default'],
      ['Token limit sent as', 'max_tokens', 'default'],
      ['Extra body', '{"top_k":20}', 'env'],
      ['Stream replies', 'On', 'default'],
    ]);
    const empty = C.lockedFacts({ ...VIEW, url: '', model: '', api_key_set: false, extra_body: {} });
    assert.match(empty[0].value, /None/);
    assert.match(empty[1].value, /AI server picks/);
    assert.equal(empty[2].value, 'Not set');
    assert.equal(empty[6].value, 'None');
    assert.ok(list.every((f) => typeof f.value === 'string' && !/sk-/.test(f.value)), 'never a key');
  });

  // Stream replies is the connection field stream
  // (OS_ASSISTANT_STREAM, on by default), the last in the server's order.
  it('Stream replies: on unless the view says false; sent only when changed, as true or false; labelled; "On" or "Off" in the locked list', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    assert.equal(C.FIELDS.at(-1), 'stream', 'last, in the server\'s order');
    assert.equal(C.LABELS.stream, 'Stream replies');
    const values = C.viewValues(VIEW);
    assert.equal(values.stream, true);
    assert.equal(C.viewValues({ ...VIEW, stream: false }).stream, false);
    const { stream: _left, ...without } = VIEW;
    assert.equal(C.viewValues(without).stream, true, 'on by default');
    assert.deepEqual(C.changedFields(VIEW, values), []);
    assert.deepEqual(C.changedFields(VIEW, { ...values, stream: false }), ['stream']);
    assert.equal(C.isDirty(VIEW, { ...values, stream: false }, ''), true);
    assert.deepEqual(C.changeBody(VIEW, { ...values, stream: false }, ''), { stream: false });
    const off = { ...VIEW, stream: false, sources: { ...VIEW.sources, stream: 'env' } };
    assert.deepEqual(C.changeBody(off, { ...C.viewValues(off), stream: true }, ''), { stream: true });
    assert.equal(C.rebase(VIEW, off, values).stream, false, 'untouched on screen: the new value');
    assert.equal(C.rebase(off, VIEW, { ...C.viewValues(off), stream: true }).stream, true, 'an edit stays');
    assert.deepEqual(C.lockedFacts(off).at(-1), { term: 'Stream replies', value: 'Off', source: 'env' });
    assert.deepEqual(C.lockedFacts(VIEW).at(-1), { term: 'Stream replies', value: 'On', source: 'default' });
    assert.equal(C.fieldProblem('stream', 'stream must be true or false').text, 'Stream replies must be true or false.');
  });
});

/* ───────────────────────────── the area on screen ───────────────────────────── */

describe('settings/assistant-connection: the area', () => {
  it('loads the connection at the top of the section: each field shows its value and where it comes from; the key only as set or not set; Use .env only beside a saved field; nothing is checked or saved', async () => {
    await mountConnection({ env: { url: AI_SERVER, api_key: 'sk-env', model: 'qwen3', template_kwargs: true }, saved: { model: 'qwen3-30b', extra_body: { top_k: 20 } } });
    assert.equal(shown('se-ai-conn'), true);
    assert.equal(textOf($('se-ai-conn-title')), 'Connection');
    const section = $('se-assistant-title').closest('section');
    const order = section.querySelectorAll('#se-ai-conn, #se-ai-editor').map((el) => el.id);
    assert.deepEqual(order, ['se-ai-conn', 'se-ai-editor'], 'the connection comes before the presets');
    assert.equal(shown('se-ai-conn-form'), true);
    assert.equal(shown('se-ai-conn-locked'), false);
    assert.equal($('se-ai-conn-url').value, AI_SERVER);
    assert.equal($('se-ai-conn-model').value, 'qwen3-30b');
    assert.equal($('se-ai-conn-key').type, 'password');
    assert.equal($('se-ai-conn-key').value, '', 'a stored key is never shown');
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    assert.equal($('se-ai-conn-tls').checked, true);
    assert.equal($('se-ai-conn-template').checked, true);
    assert.equal($('se-ai-conn-max-field').value, 'max_tokens');
    assert.deepEqual(JSON.parse($('se-ai-conn-extra').value), { top_k: 20 });
    assert.deepEqual(sources(), {
      url: 'from .env', api_key: 'from .env', model: 'saved here', verify_tls: 'default', template_kwargs: 'from .env',
      max_tokens_field: 'default', extra_body: 'saved here', stream: 'default',
    });
    assert.deepEqual(resets(), ['model', 'extra_body']);
    for (const [field, id] of Object.entries({ url: 'url', api_key: 'key', model: 'model', max_tokens_field: 'max-field', extra_body: 'extra' })) {
      assert.ok(describedBy(`se-ai-conn-${id}`).includes(`se-ai-conn-${id}-source`), `${field}: its source is in its description`);
    }
    assert.ok(describedBy('se-ai-conn-key').includes('se-ai-conn-key-state'), 'set or not set is read with the key field');
    assert.match(textOf($('se-ai-conn-help')), /calls this address/);
    assert.match(textOf($('se-ai-conn-help')), /OS_API_KEY/);
    assert.equal(shown('se-ai-conn-dirty'), false);
    assert.equal($('se-ai-conn-save').getAttribute('aria-disabled'), 'true', 'nothing to save yet');
    assert.equal($('se-ai-conn-revert').getAttribute('aria-disabled'), 'false', 'something is saved');
    // The status list is the short summary of what is in use; the form shows the rest.
    assert.deepEqual(facts(), { Status: 'On', 'AI server': 'ai-server.lan:8080', 'Default model': 'qwen3-30b' });
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/connection').length, 1);
    assert.equal(t.env.fetch.callsTo('POST /api/assistant/connection/check').length, 0, 'loading checks nothing');
    assert.equal(t.server.puts.length, 0);
    assertAccessibleNames($('se-ai-conn'));
  });

  it('Check sends only what changed on screen ({} when nothing did), unsaved; busy while it runs; a reachable server\'s models fill a dropdown that fills Model; a typed model works too', async () => {
    await mountConnection({ env: { url: AI_SERVER, model: 'qwen3' } });
    let release;
    t.server.checkAnswer = () => new Promise((resolve) => { release = () => resolve(json(200, { reachable: true, status: 200, models: ['qwen3-30b', 'qwen3-4b'], listed: true, elapsed_ms: 1234 })); });
    const check = $('se-ai-conn-check');
    assert.equal(shown('se-ai-conn-models-field'), false, 'no list before a Check');
    t.env.click(check);
    await settle(t.env);
    assert.equal(check.getAttribute('aria-busy'), 'true');
    assert.equal(textOf($('se-ai-conn-status')), 'Checking the AI server…');
    t.env.click(check);
    await settle(t.env);
    assert.equal(t.server.checks.length, 1, 'one Check at a time');
    assert.deepEqual(t.server.checks[0], {}, 'nothing changed: the connection in use');
    release();
    await settle(t.env);
    assert.equal(check.hasAttribute('aria-busy'), false);
    assert.equal(shown('se-ai-conn-result'), true);
    assert.equal(textOf($('se-ai-conn-result-title')), 'Reachable');
    assert.match(textOf($('se-ai-conn-result-msg')), /2 models/);
    assert.equal(textOf($('se-ai-conn-status')), 'Reachable in 1.2 s: 2 models');
    assert.equal(shown('se-ai-conn-models-field'), true);
    const models = $('se-ai-conn-models');
    assert.deepEqual(models.querySelectorAll('option').map((o) => o.value), ['', 'qwen3-30b', 'qwen3-4b']);
    t.env.change(models, 'qwen3-4b');
    await settle(t.env);
    assert.equal($('se-ai-conn-model').value, 'qwen3-4b', 'choosing a model fills Model');
    assert.equal(shown('se-ai-conn-dirty'), true);
    assert.equal($('se-ai-conn-save').getAttribute('aria-disabled'), 'false');
    t.env.input($('se-ai-conn-model'), 'my-own-model');
    assert.equal(models.value, '', 'a typed name is not in the list');
    // Another address: the last result and its list were for the old one.
    t.env.input($('se-ai-conn-url'), OTHER);
    assert.equal(shown('se-ai-conn-result'), false);
    assert.equal(shown('se-ai-conn-models-field'), false);
    t.server.checkAnswer = null;
    await click('se-ai-conn-check');
    assert.deepEqual(t.server.checks[1], { url: OTHER, model: 'my-own-model' }, 'the form as it is, unsaved');
    assert.equal(t.server.puts.length, 0, 'Check saves nothing');
    assert.equal(t.env.fetch.callsTo('POST /api/assistant/test').length, 0, 'and generates nothing');
  });

  it('Check of a server that lists no models, or answers something else, says so in plain words and offers no list; a refused key and a missing /v1 are named', async () => {
    await mountConnection({ env: { url: AI_SERVER } });
    t.server.checkAnswer = () => json(200, { reachable: true, status: 200, models: [], listed: true, elapsed_ms: 80 });
    await click('se-ai-conn-check');
    assert.match(textOf($('se-ai-conn-result-msg')), /lists no models/);
    assert.equal(shown('se-ai-conn-models-field'), false);
    t.server.checkAnswer = () => json(200, { reachable: true, status: 200, models: [], listed: false, elapsed_ms: 80 });
    await click('se-ai-conn-check');
    assert.match(textOf($('se-ai-conn-result-msg')), /not with a list of models/);
    t.server.checkAnswer = () => json(200, { reachable: true, status: 401, models: [], listed: false, elapsed_ms: 30 });
    await click('se-ai-conn-check');
    assert.match(textOf($('se-ai-conn-result-title')), /API key/);
    t.server.checkAnswer = () => json(200, { reachable: true, status: 404, models: [], listed: false, elapsed_ms: 30 });
    await click('se-ai-conn-check');
    assert.match(textOf($('se-ai-conn-result-msg')), /\/v1/);
    assert.equal(shown('se-ai-conn-banner'), false, 'a server that answered is not an error banner');
  });

  it('a Check that fails says what went wrong in plain words, with the server\'s message in Details and the time; a check with no URL marks the URL field', async () => {
    await mountConnection({ env: { url: AI_SERVER } });
    t.server.checkAnswer = () => json(502, { error: { message: "Couldn't connect to ai-server.lan:8080", code: 'assistant_unreachable' }, reachable: false, elapsed_ms: 340 });
    await click('se-ai-conn-check');
    const banner = $('se-ai-conn-banner');
    assert.equal(shown('se-ai-conn-banner'), true);
    assert.ok(banner.querySelector('[role="alert"]'));
    assert.equal(textOf(banner.querySelector('.banner__title')), "Couldn't reach the AI server");
    assert.match(textOf(banner.querySelector('.banner__msg')), /Open Speech's server/);
    assert.match(textOf(banner.querySelector('details pre')), /Couldn't connect to ai-server\.lan:8080/);
    assert.equal(textOf($('se-ai-conn-status')), 'Failed after 340 ms');
    assert.equal(shown('se-ai-conn-result'), false);
    assert.equal($('se-ai-conn-check').hasAttribute('aria-busy'), false);
    t.server.checkAnswer = () => json(504, { error: { message: 'The AI server did not answer within 10 s', code: 'assistant_timeout' }, reachable: false, elapsed_ms: 10004 });
    await click('se-ai-conn-check');
    assert.match(textOf($('se-ai-conn-banner')), /didn't answer in time/);
    assert.equal(textOf($('se-ai-conn-status')), 'Failed after 10.0 s');
    // No URL anywhere: the server names the field.
    t.server.checkAnswer = null;
    t.env.input($('se-ai-conn-url'), '');
    await click('se-ai-conn-check');
    const url = $('se-ai-conn-url');
    assert.equal(url.getAttribute('aria-invalid'), 'true');
    assert.equal(textOf($('se-ai-conn-url-error')), "Server URL is needed: the AI server's base URL, such as http://ai-server.lan:8080/v1.");
    assert.ok(describedBy('se-ai-conn-url').includes('se-ai-conn-url-error'));
    assert.equal(t.doc.activeElement, url);
    assert.match(textOf($('se-ai-conn-banner')), /Check the highlighted field/);
    t.env.input(url, AI_SERVER);
    assert.equal(url.hasAttribute('aria-invalid'), false, 'typing there clears it');
    assert.equal($('se-ai-conn-url-error').hidden, true);
  });

  it('Save sends only the changed fields and the typed key, then shows what the server saved: "saved here", the key as set (the field empty), Use .env, and the status list follows; Enter in a one-line field saves too', async () => {
    await mountConnection({ env: { url: AI_SERVER, model: 'qwen3' } });
    const save = $('se-ai-conn-save');
    t.env.input($('se-ai-conn-model'), 'qwen3-30b');
    t.env.change($('se-ai-conn-template'), true);
    t.env.input($('se-ai-conn-key'), 'sk-new');
    assert.equal(shown('se-ai-conn-dirty'), true);
    assert.equal(textOf($('se-ai-conn-dirty')), 'Unsaved changes');
    t.env.click(save);
    assert.equal(save.getAttribute('aria-busy'), 'true');
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(t.server.puts, [{ model: 'qwen3-30b', template_kwargs: true, api_key: 'sk-new' }], 'only what changed: the .env fields keep following .env');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'Saved the AI connection' }]);
    assert.equal(save.hasAttribute('aria-busy'), false);
    assert.equal($('se-ai-conn-key').value, '', 'the key field empties');
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    assert.deepEqual(sources(), {
      url: 'from .env', api_key: 'saved here', model: 'saved here', verify_tls: 'default', template_kwargs: 'saved here',
      max_tokens_field: 'default', extra_body: 'default', stream: 'default',
    });
    assert.deepEqual(resets(), ['api_key', 'model', 'template_kwargs']);
    assert.equal(shown('se-ai-conn-dirty'), false);
    assert.equal(save.getAttribute('aria-disabled'), 'true');
    assert.equal(facts()['Default model'], 'qwen3-30b', 'the status was loaded again');
    // A change of path on the same server keeps the key; Enter in the field saves.
    t.env.input($('se-ai-conn-url'), 'http://ai-server.lan:8080/api/v1');
    $('se-ai-conn-url').focus();
    t.env.key('Enter');
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(t.server.puts[1], { url: 'http://ai-server.lan:8080/api/v1' });
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    assert.equal(sources().url, 'saved here');
  });

  it('the key stays with its server: moving the URL to another origin says "Enter the key again for a new server" (not for a path change, nor once a key is typed); saving without one clears it and says so', async () => {
    await mountConnection({ env: { url: AI_SERVER }, saved: { api_key: 'sk-saved' } });
    const again = 'se-ai-conn-key-again';
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    assert.equal(shown(again), false);
    t.env.input($('se-ai-conn-url'), OTHER);
    assert.equal(shown(again), true);
    assert.match(textOf($(again)), /^Enter the key again for a new server/);
    assert.ok(describedBy('se-ai-conn-key').includes(again), 'read with the key field');
    t.env.input($('se-ai-conn-url'), 'http://AI-SERVER.lan:8080/other/v1');
    assert.equal(shown(again), false, 'the same origin keeps the key');
    t.env.input($('se-ai-conn-url'), OTHER);
    t.env.input($('se-ai-conn-key'), 'sk-other');
    assert.equal(shown(again), false, 'a key is given');
    t.env.input($('se-ai-conn-key'), '');
    assert.equal(shown(again), true);
    await click('se-ai-conn-save');
    assert.deepEqual(t.server.puts, [{ url: OTHER }]);
    assert.equal(textOf($('se-ai-conn-key-state')), 'Not set', 'the server cleared the key for the new origin');
    assert.equal(sources().api_key, 'default');
    assert.deepEqual(toasts(t), [{ kind: 'warning', text: 'Saved the AI connection. Enter the API key again for this server.' }]);
    assert.equal(shown(again), false);
    assert.equal(shown('se-ai-conn-key-clear'), false, 'no key to clear');
  });

  // A save that turns the certificate check off for an https server, without a key, clears the key; the helper
  // used to say nothing then, so the key was lost without a warning.
  it('turning "Check TLS certificate" off for an https server says the saved key is cleared, in its own words; a key typed or the check back on hides it; saving without a key clears it and says so', async () => {
    const HTTPS = 'https://ai.example.com/v1';
    await mountConnection({ env: { url: HTTPS }, saved: { api_key: 'sk-saved' } });
    const again = 'se-ai-conn-key-again';
    const TLS_WORDS = 'Turning off the certificate check clears the saved key: enter it again to keep using it.';
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    assert.equal(shown(again), false);
    t.env.change($('se-ai-conn-tls'), false);
    assert.equal(shown(again), true);
    assert.equal(textOf($(again)), TLS_WORDS);
    assert.ok(describedBy('se-ai-conn-key').includes(again), 'read with the key field');
    t.env.input($('se-ai-conn-key'), 'sk-new');
    assert.equal(shown(again), false, 'a key is given');
    t.env.input($('se-ai-conn-key'), '');
    t.env.change($('se-ai-conn-tls'), true);
    assert.equal(shown(again), false, 'the check back on keeps the key');
    t.env.input($('se-ai-conn-url'), 'https://other.example.com/v1');
    t.env.change($('se-ai-conn-tls'), false);
    assert.match(textOf($(again)), /^Enter the key again for a new server/, 'another server: those words');
    t.env.input($('se-ai-conn-url'), HTTPS);
    assert.equal(textOf($(again)), TLS_WORDS, 'the words follow the reason');
    await click('se-ai-conn-save');
    assert.deepEqual(t.server.puts, [{ verify_tls: false }]);
    assert.equal(textOf($('se-ai-conn-key-state')), 'Not set', 'the server cleared the key');
    assert.deepEqual(toasts(t), [{ kind: 'warning', text: 'Saved the AI connection. Enter the API key again for this server.' }]);
    assert.equal(shown(again), false);
    assert.deepEqual(t.env.errors, []);
  });

  // The repro. A key saved with the check off, the check then saved on; turning it off
  // in the form said "Turning off the certificate check clears the saved key", and the save kept the key.
  it('a key saved with the check off, then the check saved on: turning it off again asks for no key, and the save keeps the key', async () => {
    const HTTPS = 'https://ai.example.com/v1';
    await mountConnection({ env: { url: HTTPS }, saved: { api_key: 'sk-saved', verify_tls: false } });
    const again = 'se-ai-conn-key-again';
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    t.env.change($('se-ai-conn-tls'), true);
    await click('se-ai-conn-save');
    assert.deepEqual(t.server.puts, [{ verify_tls: true }]);
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set', 'turning the check on keeps the key');
    t.env.change($('se-ai-conn-tls'), false);
    assert.equal(shown(again), false, 'the save keeps the key, so nothing asks for it again');
    await click('se-ai-conn-save');
    assert.deepEqual(t.server.puts.at(-1), { verify_tls: false });
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set', 'the server kept it');
    assert.equal(toasts(t).some((x) => /Enter the API key again/.test(x.text)), false);
    assert.deepEqual(t.env.errors, []);
  });

  // The Stream replies switch, with its source and Use .env like
  // the other fields; a save sends it only when it changed.
  it('Stream replies: on by default with its source; turned off, Save sends {stream: false} only, then "saved here" with Use .env, which follows .env again; off in .env reads "from .env"', async () => {
    await mountConnection({ env: { url: AI_SERVER } });
    const sw = $('se-ai-conn-stream');
    assert.equal(sw.getAttribute('role'), 'switch');
    assert.equal(textOf(sw.closest('label')), 'Stream replies');
    assert.equal(sw.checked, true, 'on by default');
    assert.equal(textOf($('se-ai-conn-stream-source')), 'default');
    assert.equal(shown('se-ai-conn-stream-reset'), false);
    assert.deepEqual(describedBy('se-ai-conn-stream'), ['se-ai-conn-stream-source', 'se-ai-conn-stream-hint']);
    assert.match(textOf($('se-ai-conn-stream-hint')), /^Talk speaks the answer a sentence at a time while the AI is still writing/);
    t.env.change(sw, false);
    assert.equal(shown('se-ai-conn-dirty'), true);
    assert.equal($('se-ai-conn-save').getAttribute('aria-disabled'), 'false');
    await click('se-ai-conn-save');
    assert.deepEqual(t.server.puts, [{ stream: false }], 'only the switch');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'Saved the AI connection' }]);
    assert.equal(sw.checked, false);
    assert.equal(textOf($('se-ai-conn-stream-source')), 'saved here');
    assert.equal(shown('se-ai-conn-stream-reset'), true);
    assert.equal($('se-ai-conn-stream-reset').getAttribute('aria-label'), 'Use .env for Stream replies');
    assert.equal(shown('se-ai-conn-dirty'), false);
    // Use .env: no question (the value is on screen); the switch follows .env again.
    t.env.click($('se-ai-conn-stream-reset'));
    await settle(t.env);
    await settle(t.env);
    assert.equal($('app-confirm').open, false);
    assert.deepEqual(t.server.puts.at(-1), { stream: null });
    assert.equal(sw.checked, true);
    assert.equal(textOf($('se-ai-conn-stream-source')), 'default');
    assert.equal(toasts(t).at(-1).text, 'Stream replies follows .env again');
    // Off in .env: the switch is off, from .env; turned on, Save sends {stream: true}.
    t.env.restore();
    await mountConnection({ env: { url: AI_SERVER, stream: false } });
    assert.equal($('se-ai-conn-stream').checked, false);
    assert.equal(textOf($('se-ai-conn-stream-source')), 'from .env');
    t.env.change($('se-ai-conn-stream'), true);
    await click('se-ai-conn-save');
    assert.deepEqual(t.server.puts, [{ stream: true }]);
    assert.deepEqual(t.env.errors, []);
  });

  it('Revert to .env asks first, then forgets everything saved: the form shows .env again; with no URL in .env the assistant turns off', async () => {
    await mountConnection({ env: { model: 'qwen3' }, saved: { url: AI_SERVER, model: 'qwen3-30b', api_key: 'sk-saved' } });
    assert.equal(shown('se-ai-editor'), true, 'on, from the saved URL');
    t.env.input($('se-ai-conn-model'), 'edited');
    t.env.click($('se-ai-conn-revert'));
    await settle(t.env);
    assert.match(textOf($('app-confirm-title')), /Revert the AI connection to \.env\?/);
    assert.match(textOf($('app-confirm-msg')), /API key/);
    assert.match(textOf($('app-confirm-msg')), /unsaved changes/);
    await answerConfirm(false);
    assert.equal(t.server.deletes, 0, 'Cancel changes nothing');
    assert.equal($('se-ai-conn-model').value, 'edited');
    t.env.click($('se-ai-conn-revert'));
    await answerConfirm(true);
    assert.equal(t.server.deletes, 1);
    assert.equal($('se-ai-conn-url').value, '');
    assert.equal($('se-ai-conn-model').value, 'qwen3');
    assert.equal(textOf($('se-ai-conn-key-state')), 'Not set');
    assert.deepEqual(resets(), []);
    assert.equal(sources().model, 'from .env');
    assert.equal($('se-ai-conn-revert').getAttribute('aria-disabled'), 'true', 'nothing saved any more');
    assert.equal(shown('se-ai-conn-dirty'), false);
    assert.deepEqual(toasts(t).map((x) => x.text), ['The AI connection follows .env again']);
    assert.deepEqual(facts(), { Status: 'Off' });
    assert.equal(shown('se-ai-editor'), false);
    assert.equal(shown('se-ai-conn'), true, 'the connection stays: it is how you turn it on');
  });

  it('Clear key asks, then saves an empty key; Use .env forgets one saved field (the key after asking); edits in the other fields stay', async () => {
    await mountConnection({ env: { url: AI_SERVER, api_key: 'sk-env', model: 'qwen3' }, saved: { model: 'qwen3-30b' } });
    t.env.input($('se-ai-conn-url'), 'http://ai-server.lan:8080/api/v1');
    assert.equal(shown('se-ai-conn-key-clear'), true);
    t.env.click($('se-ai-conn-key-clear'));
    await settle(t.env);
    assert.match(textOf($('app-confirm-title')), /Clear the AI server's API key\?/);
    assert.match(textOf($('app-confirm-msg')), /\.env/, 'the key came from .env: Use .env brings it back');
    await answerConfirm(true);
    assert.deepEqual(t.server.puts, [{ api_key: '' }], 'only the key');
    assert.equal(textOf($('se-ai-conn-key-state')), 'Not set');
    assert.equal(sources().api_key, 'saved here');
    assert.equal(shown('se-ai-conn-key-clear'), false);
    assert.equal($('se-ai-conn-url').value, 'http://ai-server.lan:8080/api/v1', 'the unsaved URL stays');
    assert.equal(shown('se-ai-conn-dirty'), true);
    assert.equal(t.doc.activeElement, $('se-ai-conn-key'));
    assert.equal(toasts(t).at(-1).text, 'Cleared the API key');
    // Use .env for the key: it asks, because the saved value can't be typed back.
    t.env.click($('se-ai-conn-key-reset'));
    await settle(t.env);
    assert.match(textOf($('app-confirm-title')), /API key from \.env/);
    await answerConfirm(true);
    assert.deepEqual(t.server.puts.at(-1), { api_key: null });
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    assert.equal(sources().api_key, 'from .env');
    // Use .env for the model: no question, the value is on screen.
    t.env.click($('se-ai-conn-model-reset'));
    await settle(t.env);
    await settle(t.env);
    assert.equal($('app-confirm').open, false);
    assert.deepEqual(t.server.puts.at(-1), { model: null });
    assert.equal($('se-ai-conn-model').value, 'qwen3');
    assert.equal(sources().model, 'from .env');
    assert.equal(shown('se-ai-conn-model-reset'), false);
    assert.equal(t.doc.activeElement, $('se-ai-conn-model'));
    assert.equal(toasts(t).at(-1).text, 'Model follows .env again');
    assert.equal($('se-ai-conn-url').value, 'http://ai-server.lan:8080/api/v1', 'still the unsaved URL');
    assert.equal(facts()['Default model'], 'qwen3');
  });

  it('locked: a read-only list that says the connection is set in .env, with each value and where it comes from; no field to edit, no Save or Revert; Check still checks the connection in use', async () => {
    await mountConnection({ locked: true, env: { url: AI_SERVER, api_key: 'sk-env', model: 'qwen3', verify_tls: false } });
    assert.equal(shown('se-ai-conn'), true);
    assert.equal(shown('se-ai-conn-locked'), true);
    assert.equal(shown('se-ai-conn-form'), false);
    const note = textOf($('se-ai-conn-locked-note'));
    assert.match(note, /\.env/);
    assert.match(note, /OS_ASSISTANT_CONNECTION_LOCKED/);
    assert.match(note, /restart/);
    assert.deepEqual(facts('se-ai-conn-facts'), {
      'Server URL': `${AI_SERVER} from .env`, Model: 'qwen3 from .env', 'API key': 'Set from .env', 'Check TLS certificate': 'Off from .env',
      'Send thinking as template variables': 'Off default', 'Token limit sent as': 'max_tokens default', 'Extra body': 'None default',
      'Stream replies': 'On default',
    });
    const area = $('se-ai-conn');
    assert.equal(area.querySelectorAll('input, select, textarea').filter((el) => shown(el.id)).length, 0, 'nothing to edit');
    assert.equal(shown('se-ai-conn-save'), false);
    assert.equal(shown('se-ai-conn-revert'), false);
    assert.equal(shown('se-ai-conn-check'), true);
    await click('se-ai-conn-check');
    assert.deepEqual(t.server.checks, [{}], 'the .env connection, unchanged');
    assert.equal(textOf($('se-ai-conn-result-title')), 'Reachable');
    assert.equal(t.server.puts.length, 0);
    // Off and locked: turning it on is an .env change and a restart.
    t.env.restore();
    await mountConnection({ locked: true });
    assert.deepEqual(facts(), { Status: 'Off' });
    assert.equal(shown('se-ai-off-env'), true);
    assert.equal(shown('se-ai-off-form'), false);
    assert.match(textOf($('se-ai-off-env')), /OS_ASSISTANT_URL/);
    assert.match(textOf($('se-ai-off-env')), /restart/);
  });

  it('off: the section says to enter the address under Connection (no restart); saving a URL turns the assistant on at once and loads the presets; emptying it turns it off again', async () => {
    await mountConnection();
    assert.deepEqual(facts(), { Status: 'Off' });
    assert.equal(shown('se-ai-off-form'), true);
    assert.equal(shown('se-ai-off-env'), false);
    assert.match(textOf($('se-ai-off-form')), /Connection/);
    assert.doesNotMatch(textOf($('se-ai-off-form')), /restart/i);
    assert.equal(shown('se-ai-editor'), false);
    assert.equal(shown('se-ai-conn'), true);
    assert.equal($('se-ai-conn-url').value, '');
    assert.equal(sources().url, 'default');
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/presets').length, 0);
    t.env.input($('se-ai-conn-url'), AI_SERVER);
    await click('se-ai-conn-save');
    await settle(t.env);
    assert.deepEqual(t.server.puts, [{ url: AI_SERVER }]);
    assert.equal(facts().Status, 'On');
    assert.equal(shown('se-ai-off'), false);
    assert.equal(shown('se-ai-editor'), true, 'the presets load: no restart');
    assert.equal($('se-ai-preset').value, 'voice');
    t.env.input($('se-ai-conn-url'), '');
    await click('se-ai-conn-save');
    await settle(t.env);
    assert.deepEqual(t.server.puts.at(-1), { url: '' });
    assert.deepEqual(facts(), { Status: 'Off' });
    assert.equal(shown('se-ai-editor'), false);
    assert.equal(shown('se-ai-off-form'), true);
  });

  it('a 422 from Save shows on the field it names: aria-invalid, its error tied with aria-describedby, focus, a summary banner; typing clears it; nothing is saved', async () => {
    await mountConnection({ env: { url: AI_SERVER } });
    const extra = $('se-ai-conn-extra');
    t.env.input(extra, '{"top_k": 20');
    await click('se-ai-conn-save');
    assert.equal(extra.getAttribute('aria-invalid'), 'true');
    assert.match(textOf($('se-ai-conn-extra-error')), /^Extra body must be a JSON object such as \{"top_k": 20\}/);
    assert.ok(describedBy('se-ai-conn-extra').includes('se-ai-conn-extra-error'));
    assert.equal(t.doc.activeElement, extra);
    const banner = $('se-ai-conn-banner');
    assert.ok(banner.querySelector('[role="alert"]'));
    assert.match(textOf(banner), /Couldn't save the connection/);
    assert.equal(shown('se-ai-conn-dirty'), true, 'nothing was saved');
    t.env.input(extra, '{"top_k": 20}');
    assert.equal(extra.hasAttribute('aria-invalid'), false);
    assert.equal($('se-ai-conn-extra-error').hidden, true);
    t.env.input($('se-ai-conn-url'), 'ai-server.lan:8080/v1');
    await click('se-ai-conn-save');
    assert.equal($('se-ai-conn-url').getAttribute('aria-invalid'), 'true');
    assert.match(textOf($('se-ai-conn-url-error')), /^Server URL must be an http:\/\/ or https:\/\/ base URL/);
    assert.equal(sources().extra_body, 'default', 'the whole save was refused');
  });

  it('a connection locked after the page loaded: Save says so and the area loads again, read-only', async () => {
    await mountConnection({ env: { url: AI_SERVER } });
    t.server.locked = true;
    t.env.input($('se-ai-conn-model'), 'qwen3-30b');
    await click('se-ai-conn-save');
    await settle(t.env);
    assert.match(textOf($('se-ai-conn-banner')), /locked/);
    assert.equal(shown('se-ai-conn-locked'), true);
    assert.equal(shown('se-ai-conn-form'), false);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/connection').length, 2);
  });

  it('unsaved connection edits: leaving the page warns, and coming back to Settings doesn\'t load over them', async () => {
    await mountConnection({ env: { url: AI_SERVER } });
    const unload = () => {
      const event = new Event('beforeunload', { cancelable: true });
      window.dispatchEvent(event);
      return event.defaultPrevented;
    };
    assert.equal(unload(), false);
    t.env.input($('se-ai-conn-url'), OTHER);
    assert.equal(unload(), true, 'the browser asks before leaving');
    t.ctl.hide();
    t.ctl.show(t.route, { focus: 'none' });
    await settle(t.env);
    assert.equal($('se-ai-conn-url').value, OTHER);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/connection').length, 1, 'no reload over the edit');
    t.env.input($('se-ai-conn-url'), AI_SERVER);
    assert.equal(unload(), false, 'back to what was loaded');
  });

  it('a connection that can\'t be loaded says so in the area, with Retry; a server without the connection API keeps it .env-only and says so', async () => {
    let fail = true;
    const fake = fakeServer({ env: { url: AI_SERVER } });
    await mountConnection({ env: { url: AI_SERVER }, routes: { 'GET /api/assistant/connection': (req) => (fail ? json(500, { error: { message: 'disk I/O error', code: 'internal_error' } }) : fake.routes['GET /api/assistant/connection'](req)) } });
    assert.equal(shown('se-ai-conn'), true);
    assert.equal(shown('se-ai-conn-form'), false);
    const banner = $('se-ai-conn-load');
    assert.match(textOf(banner), /Couldn't load the connection/);
    assert.equal(shown('se-ai-editor'), true, 'the presets still load');
    fail = false;
    t.env.click(banner.querySelector('[data-action="ai-conn-retry"]'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(shown('se-ai-conn-form'), true);
    assert.equal(shown('se-ai-conn-load'), false);
    assert.equal($('se-ai-conn-url').value, AI_SERVER);
    t.env.restore();
    await mountConnection({ env: { url: AI_SERVER }, routes: { 'GET /api/assistant/connection': () => json(404, { detail: 'Not Found' }) } });
    assert.equal(shown('se-ai-conn'), false);
    assert.equal(shown('se-ai-server-note'), true);
    assert.match(textOf($('se-ai-server-note')), /\.env/);
    assert.equal(facts()['Certificate check'], 'On (default)', 'the status list keeps the settings it can\'t show in a form');
  });

  it('every control in the area has a name; its one live region is the Check and Save status line', async () => {
    await mountConnection({ env: { url: AI_SERVER, api_key: 'sk' }, saved: { model: 'x' } });
    assertAccessibleNames($('se-ai-conn'));
    const regions = $('se-ai-conn').querySelectorAll('[role="status"], [role="alert"], [aria-live]').map((el) => el.id);
    assert.deepEqual(regions, ['se-ai-conn-status']);
    assert.equal($('se-ai-conn-key').getAttribute('autocomplete'), 'new-password', 'no password manager fills in a saved login');
    assert.equal($('se-ai-conn').closest('form'), null, 'not a <form>: a password manager would offer to save the key');
    for (const id of ['se-ai-conn-url', 'se-ai-conn-model', 'se-ai-conn-key', 'se-ai-conn-extra']) {
      assert.equal($(id).getAttribute('spellcheck'), 'false', id);
    }
  });
});
