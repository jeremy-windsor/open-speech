// Settings › AI assistant › Assistants: the saved assistants (each one a named AI server connection, one in use),
// Use, Edit, Check and Delete on each, Add assistant with the Connection form and its Name, none saved, locked,
// and the preset editor's Assistant. The fake below follows the saved-assistants API of src/routes/assistant.py:
// GET/POST /api/assistant/assistants, PUT/DELETE …/{id}, POST …/{id}/activate and …/{id}/check.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';
import { mountSettings, settle, json, textOf, toasts, load } from './helpers.js';

let t;
afterEach(() => t?.env?.restore());

const FIELDS = ['url', 'api_key', 'model', 'verify_tls', 'template_kwargs', 'max_tokens_field', 'extra_body', 'stream'];
const DEFAULTS = Object.freeze({ url: '', api_key: '', model: '', verify_tls: true, template_kwargs: false, max_tokens_field: 'max_tokens', extra_body: {}, stream: true });
const HERMES = 'http://hermes.lan:8642/v1';
const STRATA = 'http://strata.lan:9000/v1';
const TUNABLE = [
  'model', 'system_prompt', 'user_template', 'thinking', 'reasoning_effort', 'thinking_budget', 'show_thinking',
  'max_tokens', 'temperature', 'top_p', 'timeout_s', 'history_turns', 'fillers', 'filler_after_ms',
  'filler_every_ms', 'filler_max', 'speak_errors', 'error_sentences', 'chunk_chars',
];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const lockedAnswer = () => json(403, { error: { message: 'The AI server connection is locked: it is set only in .env (OS_ASSISTANT_CONNECTION_LOCKED)', code: 'connection_locked' } });

/**
 * A fake of the saved assistants over `.env` (`env`): each assistant's saved fields override .env, a field left
 * out follows it; null forgets a field. Names are unique ignoring case (409). The first one saved is in use;
 * deleting the one in use puts the first one left in use. With none saved, the .env connection is used.
 */
function fakeServer({ env = {}, assistants = [], active = null, locked = false, presetAssistant = {} } = {}) {
  const ENV = { ...DEFAULTS, ...env };
  const state = {
    list: assistants.map((a) => ({ id: a.id, name: a.name, saved: { ...a.saved } })),
    active: active || (assistants[0] ? assistants[0].id : null),
    locked, posts: [], puts: [], deletes: [], activates: [], checks: [], connChecks: [], presetPuts: [],
    presetAssistant: { ...presetAssistant }, checkAnswer: null, n: 0, tests: [],
  };
  function view(stored) {
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
    return {
      enabled: !!values.url, locked: state.locked, url: values.url, model: values.model, api_key_set: !!values.api_key,
      api_key_tls_checked: values.api_key ? values.verify_tls : null, verify_tls: values.verify_tls, template_kwargs: values.template_kwargs,
      max_tokens_field: values.max_tokens_field, extra_body: structuredClone(values.extra_body), stream: values.stream, sources,
    };
  }
  const assistantView = (a) => ({ ...view(a.saved), id: a.id, name: a.name, active: a.id === state.active });
  const inUse = () => (state.locked ? null : state.list.find((a) => a.id === state.active) || null);
  const usedView = () => (inUse() ? view(inUse().saved) : view({}));
  const find = (id) => state.list.find((a) => a.id === id);
  const missing = () => json(404, { error: { message: 'No saved assistant with that id', code: 'assistant_not_found' } });
  const clash = (name, except = '') => state.list.some((a) => a.id !== except && a.name.toLowerCase() === String(name).toLowerCase());
  const clashAnswer = (name) => json(409, { error: { message: `An assistant named “${name}” already exists`, code: 'assistant_exists' } });
  function applyFields(stored, body) {
    for (const [name, value] of Object.entries(body)) {
      if (!FIELDS.includes(name)) continue;
      if (value === null) delete stored[name];
      else stored[name] = name === 'extra_body' && typeof value === 'string' ? (value.trim() ? JSON.parse(value) : {}) : value;
    }
  }
  function presetView(id) {
    const own = state.presetAssistant[id] || '';
    const named = find(own);
    const model = view((named || inUse() || { saved: {} }).saved).model;
    return {
      id, name: id === 'voice' ? 'Voice' : 'Book', builtin: true, customized: false, inherited: [...TUNABLE], assistant: own,
      model, system_prompt: 'Be brief.', user_template: '{text}', thinking: 'default', reasoning_effort: '', thinking_budget: 0, show_thinking: true,
      max_tokens: 0, temperature: null, top_p: null, timeout_s: id === 'book' ? 600 : 120, history_turns: 20, fillers: ['Hmm.'],
      filler_after_ms: 1200, filler_every_ms: 4000, filler_max: 3, speak_errors: true, error_sentences: {}, chunk_chars: 12000,
    };
  }
  const routes = {
    'GET /api/assistant': () => {
      const v = usedView();
      const a = inUse();
      return json(200, {
        enabled: v.enabled, host: v.url ? new URL(v.url).host : null, model: v.model, server_settings: {}, locked: state.locked,
        assistant: a ? { id: a.id, name: a.name } : null,
      });
    },
    'GET /api/assistant/presets': () => json(200, { presets: [presetView('voice'), presetView('book')] }),
    'PUT /api/assistant/presets/:id': (req) => {
      state.presetPuts.push(req.json);
      if ('assistant' in req.json) state.presetAssistant[req.params.id] = req.json.assistant;
      return json(200, presetView(req.params.id));
    },
    'GET /api/assistant/connection': () => json(200, usedView()),
    'POST /api/assistant/connection/check': (req) => {
      state.connChecks.push(req.json);
      return json(200, { reachable: true, status: 200, models: ['hermes-agent'], listed: true, elapsed_ms: 12 });
    },
    'GET /api/assistant/assistants': () => json(200, {
      active: state.locked ? null : state.active, locked: state.locked,
      assistants: state.locked ? [] : state.list.map(assistantView),
      env: view({}),
    }),
    'POST /api/assistant/test': (req) => {
      state.tests.push(req.json);
      return json(200, { reply: 'Hello.', thinking: '', finish_reason: 'stop', elapsed_ms: 20 });
    },
    'POST /api/assistant/assistants': (req) => {
      state.posts.push(req.json);
      if (state.locked) return lockedAnswer();
      const name = String(req.json.name || '').trim();
      if (!name) return json(422, { error: { message: 'name must be 1 to 60 characters', code: 'validation_error', param: 'name' } });
      if (clash(name)) return clashAnswer(name);
      state.n += 1;
      const made = { id: `new${state.n}`, name, saved: {} };
      applyFields(made.saved, req.json);
      state.list.push(made);
      if (!state.active) state.active = made.id;
      return json(201, assistantView(made));
    },
    'PUT /api/assistant/assistants/:id': (req) => {
      state.puts.push({ id: req.params.id, body: req.json });
      const a = find(req.params.id);
      if (!a) return missing();
      if ('name' in req.json) {
        if (clash(req.json.name, a.id)) return clashAnswer(req.json.name);
        a.name = req.json.name;
      }
      applyFields(a.saved, req.json);
      return json(200, assistantView(a));
    },
    'DELETE /api/assistant/assistants/:id': (req) => {
      state.deletes.push(req.params.id);
      if (!find(req.params.id)) return missing();
      state.list = state.list.filter((a) => a.id !== req.params.id);
      if (state.active === req.params.id) state.active = state.list[0] ? state.list[0].id : null;
      for (const [preset, id] of Object.entries(state.presetAssistant)) if (id === req.params.id) state.presetAssistant[preset] = '';
      return json(204);
    },
    'POST /api/assistant/assistants/:id/activate': (req) => {
      state.activates.push(req.params.id);
      const a = find(req.params.id);
      if (!a) return missing();
      state.active = a.id;
      return json(200, assistantView(a));
    },
    'POST /api/assistant/assistants/:id/check': (req) => {
      state.checks.push({ id: req.params.id, body: req.json });
      if (!find(req.params.id)) return missing();
      if (state.checkAnswer) return state.checkAnswer(req.params.id);
      return json(200, { reachable: true, status: 200, models: ['hermes-agent', 'qwen3'], listed: true, elapsed_ms: 42 });
    },
  };
  return { routes, state };
}

const TWO = Object.freeze([
  { id: 'a-hermes', name: 'Hermes', saved: { url: HERMES, model: 'hermes-agent', api_key: 'sk-hermes' } },
  { id: 'a-strata', name: 'Strata', saved: { url: STRATA, model: 'strata-1' } },
]);

async function mountList(options = {}) {
  const fake = fakeServer(options);
  t = await mountSettings({ routes: fake.routes });
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
const rows = () => $('se-ai-assts-list').querySelectorAll('li').map((li) => ({
  id: li.getAttribute('data-id'),
  name: textOf(li.querySelector('.se-ai__asst-title')),
  badges: li.querySelectorAll('.badge').map(textOf),
  meta: textOf(li.querySelector('.se-ai__asst-meta')),
  actions: li.querySelectorAll('button').map((b) => textOf(b)),
}));
const button = (id, action) => $('se-ai-assts-list').querySelector(`li[data-id="${id}"] [data-action="${action}"]`);
const facts = () => {
  const dl = $('se-ai-facts');
  const terms = dl.querySelectorAll('dt').map(textOf);
  const values = dl.querySelectorAll('dd').map(textOf);
  return Object.fromEntries(terms.map((term, i) => [term, values[i]]));
};
async function press(el) {
  t.env.click(el);
  await settle(t.env);
  await settle(t.env);
  await settle(t.env);
}
async function answerConfirm(ok) {
  await settle(t.env);
  assert.equal($('app-confirm').open, true, 'a confirm dialog is open');
  t.env.click($(ok ? 'app-confirm-ok' : 'app-confirm-cancel'));
  await settle(t.env);
  await settle(t.env);
  await settle(t.env);
}

/* ───────────────────────────── the pure parts ───────────────────────────── */

describe('saved assistants: pure parts', () => {
  it('core/assistants: the list answer keeps well-formed assistants and marks the one in use; hostOf; a preset uses its own while saved, else the one in use', async () => {
    installBrowserEnv();
    const A = await importFresh('core/assistants.js');
    const list = A.normalizeList({ active: 'b', locked: false, assistants: [{ id: 'a', name: 'Hermes', active: true }, { id: 'b', name: 'Strata' }, { name: 'no id' }, null] });
    assert.deepEqual(list.assistants.map((a) => [a.id, a.active]), [['a', false], ['b', true]], '`active` names the one in use');
    assert.equal(list.active, 'b');
    assert.equal(A.normalizeList({ active: 'gone', assistants: [{ id: 'a', name: 'A', active: true }] }).active, 'a', 'an unknown id falls back to the flag');
    assert.deepEqual(A.normalizeList(null), { active: null, locked: false, assistants: [], env: null });
    assert.deepEqual(A.normalizeList({ assistants: [], env: { url: HERMES } }).env, { url: HERMES }, 'the .env connection when given');
    assert.equal(A.hostOf(HERMES), 'hermes.lan:8642');
    assert.equal(A.hostOf('ftp://x/y'), '');
    assert.equal(A.hostOf(''), '');
    assert.equal(A.assistantForPreset(list, 'a').name, 'Hermes');
    assert.equal(A.assistantForPreset(list, '').name, 'Strata', '"" is the one in use');
    assert.equal(A.assistantForPreset(list, 'gone').name, 'Strata', 'one no longer saved: the one in use');
    assert.equal(A.assistantForPreset(null, 'a'), null, 'nothing saved');
  });

  it('connection logic: a new assistant\'s Check sends every field (untouched ones null); Revert nulls only the saved fields; the .env values from the list; the name offered; the row\'s line; a 422 on the name', async () => {
    installBrowserEnv();
    const C = await importFresh('views/settings/assistant-connection-logic.js');
    const base = { url: HERMES, model: '', verify_tls: true, template_kwargs: false, max_tokens_field: 'max_tokens', extra_body: {}, stream: true, api_key_set: false, sources: {} };
    const values = { ...C.viewValues(base), model: 'hermes-agent' };
    assert.deepEqual(C.candidateBody(base, values, ' sk-1 '), {
      model: 'hermes-agent', api_key: 'sk-1', url: null, verify_tls: null, template_kwargs: null, max_tokens_field: null, extra_body: null, stream: null,
    });
    assert.equal(C.candidateBody(base, C.viewValues(base), '').api_key, null, 'no key typed: the .env key');
    assert.deepEqual(C.revertBody({ sources: { url: 'saved', model: 'env', api_key: 'saved' } }), { url: null, api_key: null });
    assert.equal(C.suggestName({ model: 'hermes-agent', url: HERMES }, []), 'hermes-agent');
    assert.equal(C.suggestName({ model: '', url: HERMES }, []), 'hermes.lan:8642');
    assert.equal(C.suggestName({ model: '', url: '' }, []), 'Assistant');
    assert.equal(C.suggestName({ model: 'Hermes' }, ['hermes', 'HERMES 2']), 'Hermes 3', 'unique ignoring case');
    assert.equal(C.suggestName({ model: 'x'.repeat(80) }, []).length, 60);
    assert.equal(C.assistantMeta({ url: HERMES, model: 'hermes-agent' }), 'hermes.lan:8642 · hermes-agent');
    assert.equal(C.assistantMeta({ url: '', model: '' }), 'No server URL · the AI server picks the model');
    assert.deepEqual(C.fieldProblem('name', 'name must be 1 to 60 characters'), { field: 'name', text: 'Name must be 1 to 60 characters.' });
    assert.equal(C.NAME_MAX_CHARS, 60);
  });

  it('a preset\'s assistant ("" is the one in use): a save sends it only when changed, to a server whose presets have it; a copy keeps it; Test never sends it; the status names the one in use', async () => {
    installBrowserEnv();
    const L = await importFresh('views/settings/assistant-logic.js');
    const view = { id: 'mine', name: 'Mine', builtin: false, inherited: [...TUNABLE], assistant: 'a-strata' };
    const values = L.presetValues(view);
    assert.equal(values.assistant, 'a-strata');
    assert.equal(L.presetValues({ id: 'x' }).assistant, '');
    assert.equal(L.isDirty(view, values), false);
    assert.equal(L.isDirty(view, { ...values, assistant: '' }), true);
    assert.equal('assistant' in L.saveBody(view, values), false, 'unchanged: left out');
    assert.equal(L.saveBody(view, { ...values, assistant: '' }).assistant, '', 'back to the one in use');
    assert.equal(L.saveBody(view, values, { copy: true }).assistant, 'a-strata', 'a copy keeps it');
    const { assistant: _gone, ...older } = view;
    assert.equal('assistant' in L.saveBody(older, { ...values, assistant: 'a-hermes' }), false, 'an older server has no such field');
    assert.equal('assistant' in L.testBody('mine', '', values).overrides, false, 'not to a server without it');
    assert.equal(L.testBody('mine', '', values, { assistant: true }).overrides.assistant, 'a-strata');
    assert.equal(L.testBody('mine', '', { ...values, assistant: '' }, { assistant: true }).overrides.assistant, null, 'the one in use');
    assert.equal(L.FIELD_LABELS.assistant, 'Assistant');
    const on = { enabled: true, host: 'hermes.lan:8642', model: 'hermes-agent', assistant: { id: 'a-hermes', name: 'Hermes' } };
    assert.deepEqual(L.statusFacts(on, { brief: true }).map((f) => f.term), ['Status', 'Assistant in use', 'AI server', 'Default model']);
    assert.deepEqual(L.statusFacts({ ...on, assistant: null }, { brief: true }).map((f) => f.term), ['Status', 'AI server', 'Default model']);
  });
});

/* ───────────────────────────── the list on screen ───────────────────────────── */

describe('settings/assistant-list: the list', () => {
  it('lists the saved assistants above the Connection: name, host and model, "In use" on the one in use (no Use there), Edit, Check, Delete; the form edits the one in use with its name; nothing is checked', async () => {
    await mountList({ assistants: TWO });
    assert.equal(shown('se-ai-assts'), true);
    const section = $('se-assistant-title').closest('section');
    assert.deepEqual(section.querySelectorAll('#se-ai-assts, #se-ai-conn, #se-ai-editor').map((el) => el.id), ['se-ai-assts', 'se-ai-conn', 'se-ai-editor']);
    assert.deepEqual(rows(), [
      { id: 'a-hermes', name: 'Hermes', badges: ['In use', 'Editing'], meta: 'hermes.lan:8642 · hermes-agent', actions: ['Edit', 'Check', 'Delete'] },
      { id: 'a-strata', name: 'Strata', badges: [], meta: 'strata.lan:9000 · strata-1', actions: ['Use', 'Edit', 'Check', 'Delete'] },
    ]);
    assert.equal(button('a-strata', 'ai-asst-use').getAttribute('aria-label'), 'Use Strata');
    assert.equal(button('a-hermes', 'ai-asst-delete').getAttribute('aria-label'), 'Delete Hermes');
    assert.equal(shown('se-ai-assts-empty'), false);
    assert.equal(shown('se-ai-conn-name-field'), true);
    assert.equal($('se-ai-conn-name').value, 'Hermes');
    assert.equal($('se-ai-conn-url').value, HERMES);
    assert.equal(textOf($('se-ai-conn-target')), 'Editing “Hermes”, the assistant in use.');
    assert.equal(textOf($('se-ai-conn-key-state')), 'Set');
    assert.equal(shown('se-ai-conn-cancel'), false);
    assert.equal($('se-ai-conn-save').getAttribute('aria-disabled'), 'true');
    assert.equal(facts()['Assistant in use'], 'Hermes');
    assert.equal(t.server.checks.length + t.server.connChecks.length, 0, 'loading checks nothing');
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/connection').length, 0, 'the list holds the connection');
    assertAccessibleNames($('se-ai-assts'));
  });

  it('Use puts another in use: POST …/activate, the list and the status load again, a toast says so, and focus stays on that row', async () => {
    await mountList({ assistants: TWO });
    const store = await load('core/store.js');
    let told = 0;
    const off = store.on('assistants:changed', () => { told += 1; });
    const before = t.env.fetch.callsTo('GET /api/assistant').length;
    await press(button('a-strata', 'ai-asst-use'));
    off();
    assert.deepEqual(t.server.activates, ['a-strata']);
    assert.deepEqual(rows().map((r) => [r.name, r.badges]), [['Hermes', ['Editing']], ['Strata', ['In use']]]);
    assert.equal(textOf($('se-ai-conn-target')), 'Editing “Hermes”.', 'the form still edits Hermes');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: '“Strata” is in use' }]);
    assert.ok(t.env.fetch.callsTo('GET /api/assistant').length > before, 'the status loads again');
    assert.equal(facts()['Assistant in use'], 'Strata');
    assert.equal(t.doc.activeElement, button('a-strata', 'ai-asst-edit'));
    assert.equal(told, 1, 'the other views hear of it');
  });

  it('a change made in Settings loads the list and the status once each, not again for its own assistants:changed', async () => {
    await mountList({ assistants: TWO });
    const lists = t.env.fetch.callsTo('GET /api/assistant/assistants').length;
    const statuses = t.env.fetch.callsTo('GET /api/assistant').length;
    await press(button('a-strata', 'ai-asst-use'));
    for (let i = 0; i < 3; i += 1) await settle(t.env);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/assistants').length - lists, 1);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant').length - statuses, 1);
  });

  it('Edit fills the form with that assistant and focuses its Name; Save sends only what changed (a new name too) to PUT …/{id}; a name another has is refused on the Name field', async () => {
    await mountList({ assistants: TWO });
    await press(button('a-strata', 'ai-asst-edit'));
    assert.equal($('se-ai-conn-name').value, 'Strata');
    assert.equal($('se-ai-conn-url').value, STRATA);
    assert.equal(textOf($('se-ai-conn-target')), 'Editing “Strata”.');
    assert.equal(textOf($('se-ai-conn-key-state')), 'Not set');
    assert.equal(t.doc.activeElement, $('se-ai-conn-name'));
    assert.deepEqual(rows().map((r) => r.badges), [['In use'], ['Editing']]);
    t.env.input($('se-ai-conn-model'), 'strata-2');
    t.env.input($('se-ai-conn-name'), 'Strata AI');
    assert.equal(shown('se-ai-conn-dirty'), true);
    await press($('se-ai-conn-save'));
    assert.deepEqual(t.server.puts, [{ id: 'a-strata', body: { model: 'strata-2', name: 'Strata AI' } }]);
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'Saved “Strata AI”' }]);
    assert.equal(rows()[1].name, 'Strata AI');
    assert.equal(rows()[1].meta, 'strata.lan:9000 · strata-2');
    assert.equal(shown('se-ai-conn-dirty'), false);
    t.env.input($('se-ai-conn-name'), 'hermes');
    await press($('se-ai-conn-save'));
    const name = $('se-ai-conn-name');
    assert.equal(name.getAttribute('aria-invalid'), 'true');
    assert.match(textOf($('se-ai-conn-name-error')), /already exists/);
    assert.equal(t.doc.activeElement, name);
    t.env.input(name, '');
    await press($('se-ai-conn-save'));
    assert.equal(textOf($('se-ai-conn-name-error')), 'Name is needed.', 'an empty name is never sent');
    assert.equal(t.server.puts.length, 2);
  });

  it('Edit over unsaved changes asks first: Keep editing keeps them, Discard shows the other one', async () => {
    await mountList({ assistants: TWO });
    t.env.input($('se-ai-conn-model'), 'edited');
    t.env.click(button('a-strata', 'ai-asst-edit'));
    await answerConfirm(false);
    assert.equal($('se-ai-conn-model').value, 'edited');
    assert.equal($('se-ai-conn-name').value, 'Hermes');
    t.env.click(button('a-strata', 'ai-asst-edit'));
    await answerConfirm(true);
    assert.equal($('se-ai-conn-name').value, 'Strata');
    assert.equal($('se-ai-conn-model').value, 'strata-1');
  });

  it('Check on a row checks that assistant as saved (POST …/{id}/check {}) and says how it went in the list\'s status line; a failure is a banner', async () => {
    await mountList({ assistants: TWO });
    await press(button('a-strata', 'ai-asst-check'));
    assert.deepEqual(t.server.checks, [{ id: 'a-strata', body: {} }]);
    assert.equal(textOf($('se-ai-assts-status')), '“Strata”: Reachable in 42 ms: 2 models');
    assert.equal(t.server.connChecks.length, 0);
    t.server.checkAnswer = () => json(502, { error: { message: "Couldn't connect to strata.lan:9000", code: 'assistant_unreachable' }, reachable: false, elapsed_ms: 300 });
    await press(button('a-strata', 'ai-asst-check'));
    assert.equal(textOf($('se-ai-assts-status')), '“Strata”: Failed after 300 ms');
    assert.equal(textOf($('se-ai-assts-banner').querySelector('.banner__title')), "Couldn't reach the AI server: “Strata”");
    assert.equal(button('a-strata', 'ai-asst-check').hasAttribute('aria-busy'), false);
  });

  it('the form\'s Check of a saved assistant goes to …/{id}/check with what changed on screen', async () => {
    await mountList({ assistants: TWO });
    t.env.input($('se-ai-conn-model'), 'hermes-2');
    await press($('se-ai-conn-check'));
    assert.deepEqual(t.server.checks, [{ id: 'a-hermes', body: { model: 'hermes-2' } }]);
    assert.equal(textOf($('se-ai-conn-status')), 'Reachable in 42 ms: 2 models');
  });

  it('Delete asks first; deleting the one in use puts the next in use and the form edits it; focus goes to Add assistant', async () => {
    await mountList({ assistants: TWO });
    t.env.click(button('a-hermes', 'ai-asst-delete'));
    await settle(t.env);
    assert.match(textOf($('app-confirm-msg')), /It's in use: “Strata” takes over\./);
    await answerConfirm(false);
    assert.deepEqual(t.server.deletes, []);
    t.env.click(button('a-hermes', 'ai-asst-delete'));
    await answerConfirm(true);
    assert.deepEqual(t.server.deletes, ['a-hermes']);
    assert.deepEqual(rows().map((r) => [r.name, r.badges]), [['Strata', ['In use', 'Editing']]]);
    assert.equal($('se-ai-conn-name').value, 'Strata');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'Deleted “Hermes”' }]);
    assert.equal(t.doc.activeElement, $('se-ai-asst-add'));
  });

  it('Add assistant empties the form to the .env values with a name offered; Check sends every field, untouched ones null; Save POSTs the name and what changed, then edits the new one; Cancel goes back', async () => {
    const both = TWO.map((a) => ({ ...a, saved: { ...a.saved, verify_tls: true } }));
    await mountList({ assistants: both, env: { verify_tls: false } });
    await press($('se-ai-asst-add'));
    assert.equal(textOf($('se-ai-conn-target')), 'A new assistant. What you leave as it is follows .env.');
    assert.equal($('se-ai-conn-name').value, '');
    assert.equal($('se-ai-conn-url').value, '');
    assert.equal($('se-ai-conn-tls').checked, false, '.env\'s value, though every saved one overrides it');
    assert.equal(textOf($('se-ai-conn-tls-source')), 'from .env');
    assert.equal(t.doc.activeElement, $('se-ai-conn-name'));
    assert.equal(shown('se-ai-conn-cancel'), true);
    assert.equal(shown('se-ai-conn-revert'), false);
    assert.equal(shown('se-ai-conn-key-clear'), false);
    assert.deepEqual(rows().map((r) => r.badges), [['In use'], []], 'none is being edited');
    t.env.input($('se-ai-conn-url'), 'http://third.lan:8080/v1');
    assert.equal($('se-ai-conn-name').getAttribute('placeholder'), 'third.lan:8080');
    t.env.input($('se-ai-conn-model'), 'third-model');
    assert.equal($('se-ai-conn-name').getAttribute('placeholder'), 'third-model');
    await press($('se-ai-conn-check'));
    assert.deepEqual(t.server.connChecks, [{
      url: 'http://third.lan:8080/v1', model: 'third-model', api_key: null, verify_tls: null, template_kwargs: null, max_tokens_field: null, extra_body: null, stream: null,
    }]);
    t.env.input($('se-ai-conn-name'), 'Third');
    await press($('se-ai-conn-save'));
    assert.deepEqual(t.server.posts, [{ name: 'Third', url: 'http://third.lan:8080/v1', model: 'third-model' }]);
    assert.deepEqual(rows().map((r) => [r.name, r.badges]), [['Hermes', ['In use']], ['Strata', []], ['Third', ['Editing']]]);
    assert.equal($('se-ai-conn-name').value, 'Third');
    assert.equal(textOf($('se-ai-conn-target')), 'Editing “Third”.');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'Saved “Third”' }]);
    await press($('se-ai-asst-add'));
    t.env.input($('se-ai-conn-model'), 'never');
    t.env.click($('se-ai-conn-cancel'));
    await answerConfirm(true);
    assert.equal($('se-ai-conn-name').value, 'Hermes', 'back to the one in use');
    assert.equal(t.server.posts.length, 1);
  });

  it('a new assistant saved with no name typed gets the name offered', async () => {
    await mountList({ assistants: TWO });
    await press($('se-ai-asst-add'));
    t.env.input($('se-ai-conn-model'), 'Hermes');
    assert.equal($('se-ai-conn-name').getAttribute('placeholder'), 'Hermes 2', 'names are unique ignoring case');
    await press($('se-ai-conn-save'));
    assert.deepEqual(t.server.posts, [{ name: 'Hermes 2', model: 'Hermes' }]);
  });

  it('Enter in the Name field saves', async () => {
    await mountList({ assistants: TWO });
    const name = $('se-ai-conn-name');
    t.env.input(name, 'Home server');
    name.focus();
    t.env.key('Enter');
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(t.server.puts, [{ id: 'a-hermes', body: { name: 'Home server' } }]);
  });

  it('a switch made elsewhere (assistants:changed) shows over unsaved edits: the In use badge, Use and the preset\'s Assistant choices follow; the edits stay', async () => {
    await mountList({ assistants: TWO });
    t.env.input($('se-ai-conn-model'), 'edited');
    t.env.change($('se-ai-assistant'), 'a-strata');
    t.server.active = 'a-strata';
    t.server.list.push({ id: 'a-lab', name: 'Lab', saved: { url: 'http://lab.lan/v1' } });
    const store = await load('core/store.js');
    store.emit('assistants:changed');
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(rows().map((r) => [r.name, r.badges, r.actions.includes('Use')]), [
      ['Hermes', ['Editing'], true], ['Strata', ['In use'], false], ['Lab', [], true],
    ]);
    assert.equal($('se-ai-conn-model').value, 'edited', 'the unsaved edit stays');
    assert.equal(textOf($('se-ai-conn-target')), 'Editing “Hermes”.');
    assert.deepEqual($('se-ai-assistant').querySelectorAll('option').map((o) => o.value), ['', 'a-hermes', 'a-strata', 'a-lab']);
    assert.equal($('se-ai-assistant').value, 'a-strata', 'the choice on screen stays');
    assert.equal(facts()['Assistant in use'], 'Strata');
    // Settings shown again over unsaved edits takes the new list too.
    t.server.list = t.server.list.filter((a) => a.id !== 'a-lab');
    t.ctl.show(t.route, { focus: 'none', scrolled: false, reason: 'route' });
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(rows().map((r) => r.name), ['Hermes', 'Strata']);
    assert.equal($('se-ai-conn-model').value, 'edited');
  });

  it('Revert to .env for a saved assistant asks, then forgets every field saved for it (PUT with null) and keeps its name', async () => {
    await mountList({ assistants: TWO, env: { url: 'http://env.lan/v1' } });
    t.env.click($('se-ai-conn-revert'));
    await settle(t.env);
    assert.equal(textOf($('app-confirm-title')), 'Revert “Hermes” to .env?');
    await answerConfirm(true);
    assert.deepEqual(t.server.puts, [{ id: 'a-hermes', body: { url: null, api_key: null, model: null } }]);
    assert.equal($('se-ai-conn-url').value, 'http://env.lan/v1');
    assert.equal($('se-ai-conn-name').value, 'Hermes');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: '“Hermes” follows .env again' }]);
  });

  it('with none saved the list says the connection comes from .env; the form is a new one, its name offered, and Save keeps it as the first assistant, in use', async () => {
    await mountList({ env: { url: HERMES, model: 'hermes-agent' } });
    assert.equal(shown('se-ai-assts'), true);
    assert.equal(shown('se-ai-assts-empty'), true);
    assert.equal(shown('se-ai-assts-list'), false);
    assert.equal(shown('se-ai-conn-target'), false);
    assert.equal(shown('se-ai-conn-cancel'), false, 'nothing to go back to');
    assert.equal($('se-ai-conn-url').value, HERMES, 'the .env connection');
    assert.equal($('se-ai-conn-name').getAttribute('placeholder'), 'hermes-agent');
    assert.equal($('se-ai-conn-save').getAttribute('aria-disabled'), 'true');
    t.env.input($('se-ai-conn-key'), 'sk-hermes');
    await press($('se-ai-conn-save'));
    assert.deepEqual(t.server.posts, [{ name: 'hermes-agent', api_key: 'sk-hermes' }], 'the untouched fields keep following .env');
    assert.deepEqual(rows().map((r) => [r.name, r.badges]), [['hermes-agent', ['In use', 'Editing']]]);
    assert.equal(shown('se-ai-assts-empty'), false);
  });

  it('locked: no list, the .env connection read-only with the locked note', async () => {
    await mountList({ assistants: TWO, locked: true, env: { url: HERMES } });
    assert.equal(shown('se-ai-assts'), false);
    assert.equal(shown('se-ai-conn-locked'), true);
    assert.equal(shown('se-ai-conn-form'), false);
    assert.equal(shown('se-ai-assistant-field'), false);
  });

  it('a server without saved assistants keeps the single connection: no list, no Name', async () => {
    const fake = fakeServer({ env: { url: HERMES } });
    delete fake.routes['GET /api/assistant/assistants'];
    t = await mountSettings({ routes: fake.routes });
    await settle(t.env);
    await settle(t.env);
    assert.equal(shown('se-ai-assts'), false);
    assert.equal(shown('se-ai-conn-form'), true);
    assert.equal(shown('se-ai-conn-name-field'), false);
    assert.equal(shown('se-ai-conn-target'), false);
  });
});

/* ───────────────────────────── a preset's assistant ───────────────────────────── */

describe('settings/assistant: a preset\'s Assistant', () => {
  it('the preset form offers "The one in use" and each saved assistant; choosing one names its model as the default; Save sends it', async () => {
    await mountList({ assistants: TWO });
    const pick = $('se-ai-assistant');
    assert.equal(shown('se-ai-assistant-field'), true);
    assert.deepEqual(pick.querySelectorAll('option').map((o) => [o.value, textOf(o)]), [['', 'The one in use'], ['a-hermes', 'Hermes'], ['a-strata', 'Strata']]);
    assert.equal(pick.value, '');
    assert.equal(textOf($('se-ai-model-hint')), "Empty uses the server's default, hermes-agent.");
    t.env.change(pick, 'a-strata');
    await settle(t.env);
    assert.equal(textOf($('se-ai-model-hint')), 'Empty uses the model of “Strata”, strata-1.');
    assert.equal($('se-ai-dirty').hidden, false);
    await press($('se-ai-save'));
    assert.equal(t.server.presetPuts.length, 1);
    assert.equal(t.server.presetPuts[0].assistant, 'a-strata');
    t.env.change(pick, '');
    await press($('se-ai-test'));
    assert.equal(t.server.tests.at(-1).overrides.assistant, null, 'Test uses the one on screen: the one in use');
    t.env.change(pick, 'a-strata');
    await settle(t.env);
    assert.equal(pick.value, 'a-strata', 'as saved');
    assert.equal($('se-ai-dirty').hidden, true);
  });

  it('a preset that names an assistant shows it; with none saved the field is hidden and nothing is sent for it', async () => {
    await mountList({ assistants: TWO, presetAssistant: { voice: 'a-strata' } });
    assert.equal($('se-ai-assistant').value, 'a-strata');
    t.env.restore();
    await mountList({ env: { url: HERMES } });
    assert.equal(shown('se-ai-assistant-field'), false);
    t.env.input($('se-ai-system'), 'Be very brief.');
    await press($('se-ai-save'));
    assert.equal('assistant' in t.server.presetPuts[0], false);
  });
});
