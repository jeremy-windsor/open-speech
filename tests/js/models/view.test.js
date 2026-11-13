// Models: the view. The TTS | STT switch and the summary line (MD-01), Refresh
// (MD-13), the loading, empty and error states, the view's own 10 s refresh, and the screenshot hook.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountModels, inventory, settle, load, json, text, delay, networkError, announced, textOf, rowIds, toasts, STT_DEFAULT } from './helpers.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let t;
afterEach(() => t?.env?.restore());

/** Count the writes to an element's textContent (a live region speaks on every change). */
function countWrites(el) {
  const proto = Object.getPrototypeOf(el);
  let desc = null;
  for (let p = proto; p && !desc; p = Object.getPrototypeOf(p)) desc = Object.getOwnPropertyDescriptor(p, 'textContent');
  const writes = [];
  Object.defineProperty(el, 'textContent', {
    configurable: true,
    get() { return desc.get.call(this); },
    set(value) { writes.push(String(value)); desc.set.call(this, value); },
  });
  return writes;
}

const modelsCalls = (env) => env.fetch.callsTo('GET /api/models').length;

describe('models/view: pure parts', () => {
  it('[MD-01] listState: cards, loading, empty, or an error (an auth error is its own state)', async () => {
    installBrowserEnv();
    const { listState } = await importFresh('views/models/index.js');
    const list = inventory();
    assert.equal(listState({ list, fetchedAt: 1 }, 'tts'), 'cards');
    assert.equal(listState({ list: list.filter((m) => m.type === 'stt'), fetchedAt: 1 }, 'tts'), 'empty');
    assert.equal(listState({ list: [], fetchedAt: 0 }, 'tts'), 'loading');
    assert.equal(listState({ list: [], fetchedAt: 5 }, 'stt'), 'empty');
    assert.equal(listState({ list: [], fetchedAt: 0, error: "Can't reach the server." }, 'tts', 'offline'), 'error');
    assert.equal(listState({ list: [], fetchedAt: 0, error: 'x' }, 'tts', 'unauthorized'), 'auth');
    assert.equal(listState({ list: [], fetchedAt: 0, error: 'x' }, 'tts', 'locked'), 'auth');
    assert.equal(listState({ list, fetchedAt: 1, error: 'stale' }, 'tts', 'offline'), 'cards', 'data already shown stays');
  });

  it('[MD-01] summaryText: what is loaded, speech first', async () => {
    installBrowserEnv();
    const { summaryText } = await importFresh('views/models/cards.js');
    assert.equal(summaryText(inventory()), '2 loaded · Kokoro on GPU, whisper base on CPU');
    assert.equal(summaryText([]), 'No model loaded');
    assert.equal(summaryText([{ id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'loaded', device: 'cpu' }]), '1 loaded · Piper Amy (US) on CPU');
  });
});

describe('models/view: on screen', () => {
  it('[MD-01] the TTS | STT switch is a labelled segmented control; choosing STT routes with focus kept and shows only that sub', async () => {
    t = await mountModels({ env: { url: 'http://127.0.0.1:8100/web#models/tts' } });
    const router = await load('core/router.js');
    router.start((route, prev, nav) => { if (route.view === 'models') t.ctl.show(route, { ...nav, reason: 'route' }); });
    await settle(t.env);
    const seg = t.$('mo-sub');
    assert.equal(seg.getAttribute('role'), 'radiogroup');
    assert.equal(seg.getAttribute('aria-label'), 'Model type');
    const radios = seg.querySelectorAll('input[type="radio"]');
    assert.deepEqual(radios.map((r) => [r.name, r.value, r.checked]), [['mo-sub', 'tts', true], ['mo-sub', 'stt', false]]);
    assert.ok(t.card('kokoro'));
    radios[1].focus();
    const entries = history.__entries ? history.__entries.urls.length : 0;
    t.env.change(radios[1], true);
    await settle(t.env);
    assert.equal(location.hash, '#models/stt');
    assert.equal(t.app.get().route.sub, 'stt');
    if (history.__entries) assert.equal(history.__entries.urls.length, entries + 1, 'a sub change pushes one history entry');
    assert.equal(t.doc.activeElement, radios[1], 'focus stays on the switch');
    assert.equal(t.card('kokoro'), null);
    assert.ok(t.card('faster-whisper'));
    assert.equal(t.$('mo-note').hidden, true, 'the speech-model note is for TTS only');
    // Back to TTS through the hash (Back/Forward): the switch follows the route.
    t.env.setHash('#models/tts');
    await settle(t.env);
    assert.equal(radios[0].checked, true);
    assert.ok(t.card('kokoro'));
    assert.equal(t.$('mo-note').hidden, false);
    // Only an explicit load evicts (README › Model residency): a Reader, Live or API request can leave two
    // speech models loaded, so the note never claims one at a time.
    assert.equal(textOf(t.$('mo-note')), 'Loading a speech model here unloads the others.');
  });

  it('[MD-01] #mo-summary (role=status) says what is loaded and changes once per finished operation, never on a poll', async () => {
    t = await mountModels();
    const summary = t.$('mo-summary');
    assert.equal(summary.getAttribute('role'), 'status');
    assert.equal(textOf(summary), '2 loaded · Kokoro on GPU, whisper base on CPU');
    const writes = countWrites(summary);
    await settle(t.env, 10000);
    await settle(t.env, 10000);
    assert.ok(modelsCalls(t.env) >= 2, 'the view refreshed the inventory twice');
    assert.deepEqual(writes, [], 'an unchanged inventory is never re-announced');
    t.server.loadMs = 3000;
    const run = t.models.runOp('piper/en_US-amy-medium', 'load');
    await settle(t.env, 2000);
    await settle(t.env, 2000);
    assert.equal(await run, true);
    await settle(t.env);
    assert.deepEqual(writes, ['2 loaded · Piper Amy (US) on GPU, whisper base on CPU'], 'one announcement for the finished load');
  });

  it('[MD-13] Refresh re-fetches /api/models one request at a time, patches the cards and updates app.models for Speak', async () => {
    t = await mountModels();
    t.env.fetch.route('GET /api/models', () => delay(300, json(200, { models: t.server.list, default_tts_model: 'kokoro', default_stt_model: STT_DEFAULT })));
    t.server.set('piper/en_US-arctic-medium', { state: 'downloaded' });
    const before = modelsCalls(t.env);
    const refresh = t.$('mo-refresh');
    refresh.focus();
    t.env.click(refresh);
    t.env.click(refresh);
    await settle(t.env, 10);
    assert.equal(refresh.getAttribute('aria-busy'), 'true');
    assert.equal(textOf(refresh), 'Refreshing…');
    await settle(t.env, 300);
    await settle(t.env);
    assert.equal(modelsCalls(t.env), before + 1, 'one request for two presses');
    assert.equal(refresh.hasAttribute('aria-busy'), false);
    assert.equal(textOf(refresh), 'Refresh');
    assert.equal(t.doc.activeElement, refresh, 'focus is back on Refresh');
    assert.equal(t.app.get().models.list.find((m) => m.id === 'piper/en_US-arctic-medium').state, 'downloaded', 'the store (and Speak) see the new inventory');
    const arctic = t.row('piper/en_US-arctic-medium');
    assert.equal(textOf(arctic.querySelector('.mo-row__primary')), 'Load');
    assert.equal(announced(t), 'Model list updated');
  });

  it('[MD-13] a Refresh that fails shows an inline banner with the plain cause, Details and Retry; Retry clears it', async () => {
    t = await mountModels();
    t.env.fetch.route('GET /api/models', () => json(500, { error: { message: 'database is locked', code: 'internal' } }));
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    const banner = t.$('mo-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(textOf(t.$('mo-error-title')), "Couldn't refresh the models");
    assert.equal(textOf(t.$('mo-error-msg')), 'database is locked');
    // Details: the server's message and its code, never the JSON envelope.
    assert.equal(t.$('mo-error-details').hidden, false);
    assert.equal(textOf(t.$('mo-error-detail')), 'database is locked (internal)');
    assert.doesNotMatch(textOf(banner), /[{}"]|"error"/, 'no JSON on screen');
    assert.equal(t.$('mo-list').hidden, false, 'the cards already shown stay');
    t.env.fetch.route('GET /api/models', () => json(200, { models: t.server.list, default_tts_model: 'kokoro', default_stt_model: STT_DEFAULT }));
    t.env.click(t.$('mo-error-retry'));
    await settle(t.env);
    assert.equal(banner.hidden, true);
  });

  it('[MD-13] a Refresh failure\'s Details: none when they would only repeat the message (a generic code); a proxy\'s page keeps its own text', async () => {
    t = await mountModels();
    const banner = t.$('mo-error');
    // The server's generic http_error code adds nothing to the message: no Details.
    t.env.fetch.route('GET /api/models', () => json(500, { error: { message: 'database is locked', code: 'http_error' } }));
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    assert.equal(banner.hidden, false);
    assert.equal(textOf(t.$('mo-error-msg')), 'database is locked');
    assert.equal(t.$('mo-error-details').hidden, true);
    assert.equal(textOf(t.$('mo-error-detail')), '');
    // A proxy in front of the server answers with its own page: Details show that text as it came.
    t.env.fetch.route('GET /api/models', () => text(502, 'Bad Gateway: upstream timed out', { 'Content-Type': 'text/plain' }));
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    assert.equal(banner.hidden, false);
    assert.equal(textOf(t.$('mo-error-msg')), 'The server returned an unexpected response (HTTP 502).');
    assert.equal(t.$('mo-error-details').hidden, false);
    assert.equal(textOf(t.$('mo-error-detail')), 'Bad Gateway: upstream timed out');
  });

  it('[MD-13] a Refresh failure banner goes once a newer inventory arrives (the view\'s own 10 s refresh) and when the view hides', async () => {
    t = await mountModels();
    t.env.fetch.route('GET /api/models', () => json(500, { error: { message: 'database is locked', code: 'internal' } }));
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    const banner = t.$('mo-error');
    assert.equal(banner.hidden, false);
    // The server recovers: the next background refresh brings a newer inventory and the banner goes.
    t.env.fetch.route('GET /api/models', () => json(200, { models: t.server.list, default_tts_model: 'kokoro', default_stt_model: STT_DEFAULT }));
    await settle(t.env, 10000);
    await settle(t.env);
    assert.equal(t.app.get().models.error, '');
    assert.equal(banner.hidden, true, 'no stale failure over current data');
    // A failure, then leaving Models: coming back shows no old banner.
    t.env.fetch.route('GET /api/models', () => json(500, { error: { message: 'database is locked', code: 'internal' } }));
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    assert.equal(banner.hidden, false);
    t.ctl.hide();
    await settle(t.env, 1000);
    t.ctl.show(t.route, { focus: 'none', reason: 'route' });
    await settle(t.env);
    assert.equal(banner.hidden, true);
  });

  it('[MD-13] a Refresh refused for the key says "API key needed" with Open Settings (the warning tone), not a Retry that would be refused again', async () => {
    t = await mountModels();
    t.env.fetch.route('GET /api/models', () => json(401, { error: { message: 'Invalid or missing API key', code: 'unauthorized' } }));
    const refresh = t.$('mo-refresh');
    refresh.focus();
    t.env.click(refresh);
    await settle(t.env);
    assert.equal(t.app.get().connection.state, 'unauthorized');
    const banner = t.$('mo-error');
    const icons = () => banner.querySelectorAll('[data-mo-icon]').filter((svg) => !svg.hasAttribute('hidden')).map((svg) => svg.getAttribute('data-mo-icon'));
    assert.equal(banner.hidden, false);
    assert.equal(textOf(t.$('mo-error-title')), 'API key needed');
    assert.equal(textOf(t.$('mo-error-msg')), 'The list refreshes once the server accepts your API key.');
    assert.doesNotMatch(textOf(banner), /Enter it in Settings/, 'the connection banner carries the details');
    assert.equal(t.$('mo-error-retry').hidden, true);
    assert.equal(t.$('mo-error-settings').hidden, false);
    assert.equal(t.$('mo-error-details').hidden, true);
    assert.ok(banner.classList.contains('banner--warning') && !banner.classList.contains('banner--error'), 'unauthorized is a warning');
    assert.deepEqual(icons(), ['key']);
    assert.equal(t.$('mo-list').hidden, false, 'the cards already shown stay');
    // A lockout: the error tone and the lock.
    t.env.fetch.route('GET /api/models', () => json(429, { error: { message: 'Too many failed API key attempts', code: 'auth_locked' } }, { 'Retry-After': '30' }));
    const auth = await load('core/auth.js');
    auth.setKey('another-key');
    t.env.click(refresh);
    await settle(t.env);
    assert.equal(textOf(t.$('mo-error-title')), 'Locked out');
    assert.ok(banner.classList.contains('banner--error') && !banner.classList.contains('banner--warning'));
    assert.deepEqual(icons(), ['lock']);
    t.env.click(t.$('mo-error-settings'));
    await settle(t.env);
    assert.equal(location.hash, '#settings', 'Open Settings goes to Settings › Connection');
  });

  it('[MD-13] an explicit Refresh that fails while the sub shows its empty state says so in the banner', async () => {
    t = await mountModels({ sub: 'stt', models: inventory().filter((m) => m.type === 'tts') });
    assert.match(textOf(t.$('mo-state')), /No transcription models are available/);
    t.env.fetch.route('GET /api/models', () => json(500, { error: { message: 'database is locked', code: 'internal' } }));
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    assert.equal(t.$('mo-error').hidden, false);
    assert.equal(textOf(t.$('mo-error-title')), "Couldn't refresh the models");
    assert.equal(textOf(t.$('mo-error-msg')), 'database is locked');
    assert.equal(t.$('mo-error-retry').hidden, false);
  });

  it('[MD-13] one Refresh, one announcement: the summary line speaks when what is loaded changed, "Model list updated" only when it did not', async () => {
    t = await mountModels();
    const summaryWrites = countWrites(t.$('mo-summary'));
    await settle(t.env, 50);   // a new inventory gets a new fetchedAt
    t.server.set('kokoro', { state: 'downloaded', device: null });
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    assert.deepEqual(summaryWrites, ['1 loaded · whisper base on CPU']);
    assert.equal(announced(t), '', 'the summary (role=status) already said it');
    await settle(t.env, 50);
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    assert.equal(summaryWrites.length, 1);
    assert.equal(announced(t), 'Model list updated');
  });

  it('[MD-04] an operation that fails while its row is off screen (the other sub, a collapsed card) is toasted once; on screen it is said inline only', async () => {
    t = await mountModels();
    // The server answers a load after 3 s: with a failure for Amy and Alan.
    const failures = { 'piper/en_US-amy-medium': 'CUDA out of memory', 'piper/en_GB-alan-medium': 'Worker busy' };
    t.env.fetch.route('POST /api/models/:id/load', (req) => delay(3000, json(500, { error: { message: failures[req.params.id] || 'boom', code: 'failed' } })));
    t.env.click(t.row('piper/en_US-amy-medium').querySelector('.mo-row__primary'));
    await settle(t.env, 10);
    // The listener switches to STT while the load runs.
    t.ctl.show({ ...t.route, sub: 'stt', path: 'models/stt' }, { focus: 'keep', reason: 'route' });
    await settle(t.env, 4000);
    await settle(t.env);
    assert.equal(t.app.get().ops['piper/en_US-amy-medium'].error, 'CUDA out of memory');
    assert.deepEqual(toasts(t), [{ kind: 'error', text: "Couldn't load Piper Amy (US): CUDA out of memory" }]);
    // Back on TTS the row says it inline; nothing is toasted again.
    t.ctl.show(t.route, { focus: 'keep', reason: 'route' });
    await settle(t.env);
    assert.match(textOf(t.row('piper/en_US-amy-medium').querySelector('.mo-row__error-text')), /CUDA out of memory/);
    assert.equal(toasts(t).length, 1);
    // A collapsed card hides its rows: a failure there is toasted too.
    t.env.click(t.row('piper/en_GB-alan-medium').querySelector('.mo-row__primary'));
    await settle(t.env, 10);
    t.env.click(t.card('piper').querySelector('.mo-card__toggle'));
    await settle(t.env, 4000);
    await settle(t.env);
    assert.equal(toasts(t).map((x) => x.text).at(-1), "Couldn't load Piper Alan (GB): Worker busy");
    // A failure whose row is on screen is said inline (role=alert), not toasted.
    t.env.click(t.card('piper').querySelector('.mo-card__toggle'));
    const count = toasts(t).length;
    t.server.fail['piper/en_US-arctic-medium'] = { op: 'prefetch', status: 500, message: 'Disk full' };
    t.env.click(t.row('piper/en_US-arctic-medium').querySelector('.mo-row__primary'));
    await settle(t.env, 4000);
    await settle(t.env);
    assert.match(textOf(t.row('piper/en_US-arctic-medium').querySelector('.mo-row__error-text')), /Disk full/);
    assert.equal(toasts(t).length, count);
  });

  it('the "API key needed" state takes the warning tone; "Locked out" the error tone', async () => {
    t = await mountModels({ models: null, routes: { 'GET /api/models': () => json(401, { error: { message: 'unauthorized', code: 'unauthorized' } }) } });
    await settle(t.env);
    const empty = t.$('mo-state').querySelector('.empty');
    assert.ok(empty.classList.contains('mo-empty--warn') && !empty.classList.contains('empty--error'));
    t.app.set({ connection: { ...t.app.get().connection, state: 'locked', error: 'After too many wrong API keys…' } });
    await settle(t.env);
    const locked = t.$('mo-state').querySelector('.empty');
    assert.ok(locked.classList.contains('empty--error') && !locked.classList.contains('mo-empty--warn'));
  });

  it('shows skeleton cards while the inventory loads, then the cards', async () => {
    t = await mountModels({ models: null, routes: { 'GET /api/models': () => delay(500, json(200, { models: inventory(), default_tts_model: 'kokoro', default_stt_model: STT_DEFAULT })) } });
    const state = t.$('mo-state');
    assert.equal(state.hidden, false);
    const skeleton = state.querySelector('.mo-skeleton');
    assert.equal(skeleton.getAttribute('aria-busy'), 'true');
    assert.match(textOf(skeleton), /Loading models…/);
    assert.equal(state.querySelectorAll('.skeleton').length, 9);
    assert.equal(textOf(t.$('mo-summary')), 'Checking what is loaded…');
    await settle(t.env, 500);
    await settle(t.env);
    assert.equal(state.hidden, true);
    assert.ok(t.card('kokoro'));
    assert.equal(textOf(t.$('mo-summary')), '2 loaded · Kokoro on GPU, whisper base on CPU');
  });

  it('a first load that fails shows the error state with Retry (focus moves to the cards once they load)', async () => {
    t = await mountModels({ models: null, routes: { 'GET /api/models': () => networkError() } });
    await settle(t.env);
    const state = t.$('mo-state');
    assert.match(textOf(state), /Can't reach the server/);
    assert.equal(t.root.querySelector('.mo-summary').hidden, true, 'the summary hides while the error state explains');
    assert.match(textOf(state), /Check that Open Speech is running\./);
    const retry = state.querySelector('button');
    assert.equal(textOf(retry), 'Retry');
    assertAccessibleNames(t.root);
    t.env.fetch.route('GET /api/models', () => json(200, { models: inventory(), default_tts_model: 'kokoro', default_stt_model: STT_DEFAULT }));
    retry.focus();
    t.env.click(retry);
    await settle(t.env);
    assert.equal(state.hidden, true);
    assert.ok(t.card('kokoro'));
    assert.equal(t.doc.activeElement, t.card('kokoro').querySelector('.mo-card__toggle'), 'focus moved from the removed Retry to the first card');
  });

  it('an auth failure says "API key needed" with Open Settings, which goes to Settings › Connection', async () => {
    t = await mountModels({ models: null, routes: { 'GET /api/models': () => json(401, { error: { message: 'unauthorized', code: 'unauthorized' } }) } });
    await settle(t.env);
    assert.equal(t.app.get().connection.state, 'unauthorized');
    const state = t.$('mo-state');
    assert.match(textOf(state), /API key needed/);
    assert.match(textOf(state), /The models load once the server accepts your API key\./);
    assert.doesNotMatch(textOf(state), /Enter it in Settings/, 'the connection banner carries the details; the state does not repeat them');
    // A lockout says so (the same Open Settings path).
    t.app.set({ connection: { ...t.app.get().connection, state: 'locked', error: 'After too many wrong API keys…' } });
    await settle(t.env);
    assert.match(textOf(state), /Locked out/);
    t.app.set({ connection: { ...t.app.get().connection, state: 'unauthorized', error: '' } });
    await settle(t.env);
    t.env.click(state.querySelector('button'));
    await settle(t.env);
    assert.equal(location.hash, '#settings');
  });

  it('[MD-02] the STT sub with no models says so', async () => {
    t = await mountModels({ sub: 'stt', models: inventory().filter((m) => m.type === 'tts') });
    assert.match(textOf(t.$('mo-state')), /No transcription models are available\./);
  });

  it('refreshes every 10 s while it shows and this tab runs no operation; stops on hide and while the page is hidden', async () => {
    t = await mountModels();
    const start = modelsCalls(t.env);
    assert.equal(t.ctl.state.polling, true);
    await settle(t.env, 10000);
    assert.equal(modelsCalls(t.env), start + 1);
    // An operation in this tab: runOp polls the status itself, so the view does not.
    t.app.set({ ops: { 'piper/en_US-amy-medium': { kind: 'load', startedAt: Date.now(), progress: null, error: '', phase: '' } } });
    await settle(t.env, 10000);
    assert.equal(modelsCalls(t.env), start + 1);
    t.app.set({ ops: {} });
    t.env.setHidden(true);
    await settle(t.env, 10000);
    assert.equal(modelsCalls(t.env), start + 1, 'no polling while the page is hidden');
    t.env.setHidden(false);
    await settle(t.env);
    assert.equal(modelsCalls(t.env), start + 2, 'a stale inventory is refreshed when the page shows again');
    t.ctl.hide();
    assert.equal(t.ctl.state.polling, false);
    await settle(t.env, 30000);
    assert.equal(modelsCalls(t.env), start + 2, 'hide() stops the refresh');
  });

  it('does not refresh while the key is missing or refused, locked out, or offline; catches up when the connection returns', async () => {
    t = await mountModels();
    const start = modelsCalls(t.env);
    for (const state of ['unauthorized', 'locked', 'offline']) {
      t.app.set({ connection: { ...t.app.get().connection, state } });
      await settle(t.env, 10000);
      assert.equal(modelsCalls(t.env), start, `no background refresh while ${state}`);
    }
    t.app.set({ connection: { ...t.app.get().connection, state: 'online' } });
    await settle(t.env);
    assert.equal(modelsCalls(t.env), start + 1, 'back online: one refresh at once');
    // An explicit Refresh always asks.
    t.app.set({ connection: { ...t.app.get().connection, state: 'offline' } });
    t.env.click(t.$('mo-refresh'));
    await settle(t.env);
    assert.equal(modelsCalls(t.env), start + 2);
  });

  it('a show() refreshes an inventory older than 10 s; an auth change always refreshes', async () => {
    t = await mountModels({ show: false });
    const start = modelsCalls(t.env);
    t.ctl.show(t.route, { focus: 'none', reason: 'route' });
    await settle(t.env);
    assert.equal(modelsCalls(t.env), start, 'fresh data: no request');
    t.ctl.hide();
    await settle(t.env, 11000);
    t.ctl.show(t.route, { focus: 'none', reason: 'route' });
    await settle(t.env);
    assert.equal(modelsCalls(t.env), start + 1);
    t.ctl.show(t.route, { focus: 'keep', scrolled: true, reason: 'auth' });
    await settle(t.env);
    assert.equal(modelsCalls(t.env), start + 2);
  });

  it('the screenshot hook ?shot=menu opens the first overflow menu', async () => {
    t = await mountModels({ shot: 'menu' });
    const more = t.row('kokoro').querySelector('.mo-row__more');
    assert.equal(more.getAttribute('aria-expanded'), 'true');
    assert.equal(t.$(more.getAttribute('aria-controls')).hidden, false);
  });

  it('every control the view builds has a name and a type, and the live regions are the required ones', async () => {
    t = await mountModels();
    assertAccessibleNames(t.root);
    t.ctl.show({ view: 'models', sub: 'stt', id: '', path: 'models/stt', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assertAccessibleNames(t.root);
    const regions = t.root.querySelectorAll('[role="status"], [role="alert"]').map((el) => [el.id || el.className, el.getAttribute('role')]);
    assert.ok(regions.some(([id, role]) => id === 'mo-summary' && role === 'status'));
    assert.deepEqual(rowIds(t.card('faster-whisper')).length, 5);
  });
});
