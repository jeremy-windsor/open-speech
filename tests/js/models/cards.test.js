// Models: the provider cards. Grouping and install/offline cards (MD-02, MD-03),
// one action per state (MD-04), collapse (MD-09), the Piper and STT lists (MD-10, MD-11), the Kokoro card
// (MD-12) and the in-place patch while operations run (BUG-10).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  mountModels, inventory, settle, load, json, delay, answerConfirm, announced, textOf, rowIds, toasts, STT_DEFAULT,
} from './helpers.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames, focusables } from '../harness/a11y.js';

let t;
afterEach(() => t?.env?.restore());

/** Every text written to an element's textContent, in order (an announcement clears the region first). */
function spyWrites(el) {
  let desc = null;
  for (let p = Object.getPrototypeOf(el); p && !desc; p = Object.getPrototypeOf(p)) desc = Object.getOwnPropertyDescriptor(p, 'textContent');
  const writes = [];
  Object.defineProperty(el, 'textContent', {
    configurable: true,
    get() { return desc.get.call(this); },
    set(value) { writes.push(String(value)); desc.set.call(this, value); },
  });
  return writes;
}

const primaryOf = (rowEl) => rowEl.querySelector('.mo-row__primary');
const pillOf = (rowEl) => rowEl.querySelector('.mo-row__pill');
const PIPER_FIRST_FIVE = ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium', 'piper/en_US-arctic-medium', 'piper/en_US-joe-medium', 'piper/en_US-lessac-low'];

describe('models/cards: pure helpers', () => {
  it('[MD-02] groupCards: cards with a loaded model first, then Kokoro, Piper, other providers, offline workers, then not-installed providers', async () => {
    installBrowserEnv();
    const { groupCards } = await importFresh('views/models/cards.js');
    const list = inventory();
    assert.deepEqual(groupCards(list, 'tts').map((c) => [c.provider, c.type]), [
      ['kokoro', 'models'], ['piper', 'models'], ['chatterbox', 'offline'], ['pocket-tts', 'install'],
    ]);
    // A provider with a loaded model comes first; Kokoro and Piper keep their place among the rest.
    const swapped = list.map((m) => (m.id === 'kokoro' ? { ...m, state: 'downloaded', device: null } : m.id === 'piper/en_US-amy-medium' ? { ...m, state: 'loaded', device: 'cpu' } : m));
    const withQwen = [...swapped, { id: 'qwen3/0.6b-base', type: 'tts', provider: 'qwen3', state: 'available', model_format: 'external', provider_available: true }];
    assert.deepEqual(groupCards(withQwen, 'tts').map((c) => c.provider), ['piper', 'kokoro', 'qwen3', 'chatterbox', 'pocket-tts']);
    // A worker that went offline while one of its models stayed loaded is one card with both rows.
    const mixed = [...list.filter((m) => m.provider !== 'chatterbox'),
      { id: 'chatterbox/regular', type: 'tts', provider: 'chatterbox', state: 'loaded', device: 'cuda:0', provider_available: true },
      { id: 'chatterbox/turbo', type: 'tts', provider: 'chatterbox', state: 'provider_unavailable', provider_available: false }];
    const card = groupCards(mixed, 'tts').find((c) => c.provider === 'chatterbox');
    assert.equal(card.type, 'models');
    assert.equal(card.models.length, 2);
    assert.deepEqual(groupCards(list, 'stt').map((c) => c.provider), ['faster-whisper']);
  });

  it('[MD-02] a loaded model whose worker stopped responding reads as an offline worker, never as "not configured"', async () => {
    installBrowserEnv();
    const { groupCards } = await importFresh('views/models/cards.js');
    const models = await load('core/models.js');
    // The server keeps state 'loaded' when the worker stops answering (provider_available false). The card
    // takes the row as the server sent it: core uiState() reads it as the offline worker it is.
    const stale = { id: 'chatterbox/regular', type: 'tts', provider: 'chatterbox', state: 'loaded', device: 'cuda:0', provider_available: false };
    const state = models.uiState(stale);
    assert.equal(state.key, 'worker-offline');
    assert.equal(state.label, 'Worker offline');
    assert.equal(state.action, null, 'no install command for a configured worker');
    const list = [...inventory().filter((m) => m.provider !== 'chatterbox'), stale,
      { id: 'chatterbox/turbo', type: 'tts', provider: 'chatterbox', state: 'provider_unavailable', provider_available: false }];
    assert.equal(groupCards(list, 'tts').find((c) => c.provider === 'chatterbox').type, 'offline');
    // Alone on its card, the stale row still makes an offline card, never an install card.
    const alone = [...inventory().filter((m) => m.provider !== 'chatterbox'), stale];
    assert.equal(groupCards(alone, 'tts').find((c) => c.provider === 'chatterbox').type, 'offline');
  });

  it('[MD-10] sortModels: Piper sorts loaded, downloaded, en_US before en_GB, then name and quality', async () => {
    installBrowserEnv();
    const { sortModels, rowTitle, rowMeta } = await importFresh('views/models/cards.js');
    const piper = inventory().filter((m) => m.provider === 'piper');
    assert.deepEqual(sortModels('piper', piper).map(rowTitle), ['Amy (US)', 'Alan (GB)', 'Arctic (US)', 'Joe (US)', 'Lessac (US)', 'Ryan (US)', 'Cori (GB)']);
    const lessac = ['high', 'low', 'medium'].map((q) => ({ id: `piper/en_US-lessac-${q}`, type: 'tts', provider: 'piper', state: 'available' }));
    assert.deepEqual(sortModels('piper', lessac).map((m) => m.id), ['piper/en_US-lessac-low', 'piper/en_US-lessac-medium', 'piper/en_US-lessac-high']);
    assert.equal(rowMeta(piper[0]), 'Medium quality · 35 MB');
    assert.equal(rowTitle({ id: 'piper/en_US-hfc_female-medium', provider: 'piper', type: 'tts' }), 'Hfc Female (US)');
  });

  it('[MD-11] STT rows: short names, description · source · format · size, sorted loaded > downloaded > the rest by name', async () => {
    installBrowserEnv();
    const { sortModels, rowTitle, rowMeta } = await importFresh('views/models/cards.js');
    const stt = inventory().filter((m) => m.type === 'stt');
    assert.deepEqual(sortModels('faster-whisper', stt).map(rowTitle), ['base', 'distil-small.en', 'large-v3', 'large-v3-turbo', 'medium', 'small', 'tiny']);
    assert.equal(rowMeta(stt[0]), 'Good balance for CPU · Systran · CT2 · 150 MB');
    assert.equal(rowMeta(stt.find((m) => m.source === 'deepdml')), 'Large-v3-turbo, near large-v3 accuracy at 3-4x speed · deepdml · CT2 · 1.6 GB');
  });

  it('[MD-12] features: a voice count and capability words, not a chip per voice', async () => {
    installBrowserEnv();
    const { features, rowMeta } = await importFresh('views/models/cards.js');
    const kokoro = inventory()[0];
    assert.deepEqual(features(kokoro), ['54 voices', 'Blends']);
    assert.equal(rowMeta(kokoro), '54 voices · Blends · 330 MB');
    assert.deepEqual(features({ capabilities: { speakers: [{}], streaming: true, voice_clone: true } }), ['1 voice', 'Streaming', 'Cloning']);
    assert.deepEqual(features({}), []);
  });

  it('visibleIds: the frozen order, filtered (every word), "Downloaded only", five until Show all', async () => {
    installBrowserEnv();
    const { visibleIds, sortModels, LIST_LIMIT } = await importFresh('views/models/cards.js');
    const piper = inventory().filter((m) => m.provider === 'piper');
    const order = sortModels('piper', piper).map((m) => m.id);
    const byId = new Map(piper.map((m) => [m.id, m]));
    assert.equal(LIST_LIMIT, 5);
    assert.deepEqual(visibleIds(order, byId), { ids: PIPER_FIRST_FIVE, total: 7, matched: 7, limited: true });
    assert.equal(visibleIds(order, byId, { showAll: true }).ids.length, 7);
    assert.deepEqual(visibleIds(order, byId, { filter: 'gb' }).ids, ['piper/en_GB-alan-medium', 'piper/en_GB-cori-high']);
    assert.deepEqual(visibleIds(order, byId, { filter: 'us high' }).ids, ['piper/en_US-ryan-high']);
    assert.deepEqual(visibleIds(order, byId, { onDisk: true }).ids, ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium']);
    // A download running in this tab keeps its row in "Downloaded only".
    const ops = { 'piper/en_US-joe-medium': { kind: 'prefetch', startedAt: 1, progress: null, error: '' } };
    assert.deepEqual(visibleIds(order, byId, { onDisk: true, ops }).ids, ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium', 'piper/en_US-joe-medium']);
  });

  it('[MD-10] filterSummary: the words said for a narrowed list (count and filter, the empty state, "Downloaded only")', async () => {
    installBrowserEnv();
    const { visibleIds, sortModels, filterSummary } = await importFresh('views/models/cards.js');
    const piper = inventory().filter((m) => m.provider === 'piper');
    const order = sortModels('piper', piper).map((m) => m.id);
    const byId = new Map(piper.map((m) => [m.id, m]));
    const say = (options) => filterSummary('piper', visibleIds(order, byId, options), options);
    assert.equal(say({ filter: 'gb' }), '2 voices match “gb”');
    assert.equal(say({ filter: '  us high ' }), '1 voice matches “us high”');
    assert.equal(say({ filter: 'zzz' }), 'No voices match “zzz”');
    assert.equal(say({}), 'Showing 5 of 7 voices');
    assert.equal(say({ showAll: true }), '7 voices');
    assert.equal(say({ onDisk: true }), '2 voices on disk');
    assert.equal(say({ onDisk: true, filter: 'gb' }), '1 voice matches “gb”');
    assert.equal(filterSummary('piper', { ids: [], matched: 0, limited: false }, { onDisk: true }), 'No voices are downloaded yet');
    // Every other provider lists models.
    assert.equal(filterSummary('faster-whisper', { ids: ['a', 'b'], matched: 2, limited: false }, { filter: 'en' }), '2 models match “en”');
    assert.equal(filterSummary('faster-whisper', { ids: ['a'], matched: 1, limited: false }, { filter: 'large' }), '1 model matches “large”');
  });
});

describe('models/cards: the cards on screen', () => {
  it('[MD-02] an offline worker gets an offline card that never says "not installed"; a missing provider gets an install card', async () => {
    t = await mountModels();
    const cards = t.root.querySelectorAll('.mo-card').map((c) => [c.getAttribute('data-provider'), c.getAttribute('data-type')]);
    assert.deepEqual(cards, [['kokoro', 'models'], ['piper', 'models'], ['chatterbox', 'offline'], ['pocket-tts', 'install']]);
    const offline = t.card('chatterbox');
    assert.match(textOf(offline), /Chatterbox Worker offline/);
    assert.match(textOf(offline), /The voice worker isn't responding\. Check the worker service\./);
    assert.match(textOf(offline), /Models: chatterbox\/regular, chatterbox\/turbo/);
    assert.doesNotMatch(textOf(offline), /not installed|not configured/i);
    assert.equal(offline.querySelector('.pill').className.includes('pill--error'), true);
    assert.equal(offline.querySelectorAll('.mo-row').length, 0, 'an offline worker offers no model actions');
    const install = t.card('pocket-tts');
    assert.match(textOf(install), /Pocket TTS Not installed/);
    assert.equal(install.querySelector('.pill').className.includes('pill--warn'), true);
    assertAccessibleNames(t.root);
  });

  it('[MD-02] a worker that stops answering while its models are loaded shows the offline card, never "not configured", with no model actions', async () => {
    // The server's list after the worker went away: loaded rows keep state 'loaded' (model_manager.py
    // list_all), only provider_available turns false. With every row of the worker loaded, nothing but core
    // uiState() tells this card from a worker that was never configured.
    const list = inventory().map((m) => (m.provider === 'chatterbox' ? { ...m, state: 'loaded', device: 'cuda:0' } : m));
    t = await mountModels({ models: list });
    const offline = t.card('chatterbox');
    assert.equal(offline.getAttribute('data-type'), 'offline');
    assert.match(textOf(offline), /Chatterbox Worker offline/);
    assert.match(textOf(offline), /The voice worker isn't responding\. Check the worker service\./);
    assert.doesNotMatch(textOf(offline), /not installed|not configured|Worker not configured/i);
    // The command turns on this worker's own Compose profile.
    assert.match(textOf(offline), /To turn the worker on, name its Compose profile:/, 'the command starts the worker; nothing to install');
    assert.equal(offline.querySelector('.mo-cmd__text').textContent, 'docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile chatterbox up -d --build');
    assert.equal(offline.querySelectorAll('.mo-row').length, 0, 'no Unload for a model its worker cannot hear');
    assert.equal(offline.querySelector('.pill').className.includes('pill--error'), true);
    assertAccessibleNames(t.root);
  });

  // A worker that answers /health "failed" reaches the
  // rows as provider_failure {code, message}. The card says it can't run and why, not "Worker offline".
  const REASON = 'Torch wheel has no code that runs on sm_120 (built for: sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90)';

  it("[MD-02] a worker that answers but can't run gets the offline card with its reason, never \"Worker offline\"", async () => {
    const failure = { code: 'cuda_arch_unsupported', message: REASON };
    const list = inventory().map((m) => (m.provider === 'chatterbox' ? { ...m, provider_failure: failure } : m));
    t = await mountModels({ models: list });
    const cards = t.root.querySelectorAll('.mo-card').map((c) => [c.getAttribute('data-provider'), c.getAttribute('data-type')]);
    assert.deepEqual(cards, [['kokoro', 'models'], ['piper', 'models'], ['chatterbox', 'offline'], ['pocket-tts', 'install']], 'its place among the workers that can do no work');
    const card = t.card('chatterbox');
    assert.match(textOf(card), /Chatterbox Worker can't run/);
    assert.equal(textOf(card.querySelector('.mo-card__reason')), `The voice worker can't run: ${REASON}.`);
    assert.match(textOf(card), /Models: chatterbox\/regular, chatterbox\/turbo/);
    assert.doesNotMatch(textOf(card), /Worker offline|isn't responding|not installed|not configured/i);
    assert.equal(card.querySelector('.pill').className.includes('pill--error'), true);
    assert.equal(card.querySelectorAll('.mo-row').length, 0, 'no model actions: no load can succeed');
    assertAccessibleNames(t.root);
  });

  it("[MD-02] the worker's reason is shown as text, never markup, and a poll that brings or ends a failure repaints the card in place", async () => {
    t = await mountModels();
    const card = t.card('chatterbox');
    assert.match(textOf(card), /Chatterbox Worker offline/);
    const poll = async (failure) => {
      const list = t.server.list.map((m) => (m.provider === 'chatterbox' ? { ...m, provider_failure: failure } : m));
      t.app.set({ models: { ...t.app.get().models, list, fetchedAt: Date.now() } });
      await settle(t.env);
    };
    const markup = '<img src=x onerror="window.owned=1"> & <b>bold</b>';
    await poll({ code: 'cuda_context_failed', message: markup });
    assert.equal(t.card('chatterbox'), card, 'the same card, patched');
    assert.match(textOf(card), /Chatterbox Worker can't run/);
    assert.equal(textOf(card.querySelector('.mo-card__reason')), `The voice worker can't run: ${markup}.`);
    assert.equal(card.querySelectorAll('img, b').length, 0, "the worker's text is never parsed as markup");
    await poll({ code: 'cuda_arch_unsupported', message: REASON });
    assert.equal(textOf(card.querySelector('.mo-card__reason')), `The voice worker can't run: ${REASON}.`, 'a new reason replaces the old one');
    await poll(null);
    assert.equal(t.card('chatterbox'), card);
    assert.match(textOf(card), /Chatterbox Worker offline/);
    assert.equal(textOf(card.querySelector('.mo-card__reason')), "The voice worker isn't responding. Check the worker service.");
    assert.doesNotMatch(textOf(card), /can't run/);
  });

  it('[MD-02] a sub with no models says so (TTS names BAKED_PROVIDERS and the workers)', async () => {
    t = await mountModels({ models: inventory().filter((m) => m.type === 'stt') });
    const state = t.$('mo-state');
    assert.equal(state.hidden, false);
    assert.equal(t.$('mo-list').hidden, true);
    assert.match(textOf(state), /No speech models are available\. Check BAKED_PROVIDERS or start the voice workers\./);
  });

  it('[MD-03] install hints: rebuild the image for a baked provider, turn on the worker\'s own Compose profile for a worker', async () => {
    const list = [...inventory(), { id: 'cosyvoice/2-0.5b', type: 'tts', provider: 'cosyvoice', state: 'provider_missing', provider_available: false, model_format: 'external' }];
    t = await mountModels({ models: list });
    const baked = t.card('pocket-tts');
    assert.match(textOf(baked), /To install, rebuild your image:/);
    assert.equal(baked.querySelector('.mo-cmd__text').textContent, 'docker build --build-arg BAKED_PROVIDERS=kokoro,piper,pocket-tts .');
    // Layering docker-compose.voice-models.yml alone starts no worker: each needs its profile.
    const compose = (p) => `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile ${p} up -d --build`;
    const worker = t.card('cosyvoice');
    assert.match(textOf(worker), /Worker not configured/);
    assert.match(textOf(worker), /CosyVoice runs as an isolated voice worker\. Turn it on with its Compose profile:/);
    assert.equal(worker.querySelector('.mo-cmd__text').textContent, compose('cosyvoice'));
    // The offline worker card (its profile isn't turned on) offers its own profile's command.
    const offline = t.card('chatterbox');
    assert.match(textOf(offline), /To turn the worker on, name its Compose profile:/);
    assert.doesNotMatch(textOf(offline), /To start the voice workers/);
    assert.equal(offline.querySelector('.mo-cmd__text').textContent, compose('chatterbox'));
  });

  it('[MD-03] a worker provider with no service in docker-compose.voice-models.yml gets no command to copy', async () => {
    const list = [...inventory(), { id: 'xtts/v2', type: 'tts', provider: 'xtts', state: 'provider_missing', provider_available: false, model_format: 'external' }];
    t = await mountModels({ models: list });
    const card = t.card('xtts');
    assert.equal(card.getAttribute('data-type'), 'install');
    assert.match(textOf(card), /XTTS v2 Worker not configured/);
    assert.match(textOf(card), /XTTS v2 runs as an isolated voice worker, and docker-compose\.voice-models\.yml has none for it\./);
    assert.equal(card.querySelector('.mo-cmd'), null, 'no command that cannot work');
    assert.equal(card.querySelector('.mo-cmd__copy'), null, 'no Copy command button');
    assertAccessibleNames(t.root);
  });

  it('[MD-03] Copy command copies, says "Copied" for 1.5 s and keeps focus on the button', async () => {
    t = await mountModels();
    const button = t.card('pocket-tts').querySelector('.mo-cmd__copy');
    button.focus();
    t.env.click(button);
    await settle(t.env);
    assert.deepEqual(t.env.clipboard.written, ['docker build --build-arg BAKED_PROVIDERS=kokoro,piper,pocket-tts .']);
    assert.equal(textOf(button), 'Copied');
    assert.equal(t.doc.activeElement, button);
    assert.equal(announced(t), 'Command copied');
    await settle(t.env, 1499);
    assert.equal(textOf(button), 'Copied');
    await settle(t.env, 1);
    assert.equal(textOf(button), 'Copy command');
  });

  it('[MD-03] over plain HTTP Copy uses the hidden-textarea fallback and returns focus; when nothing can be copied a toast says so', async () => {
    t = await mountModels({ env: { clipboard: 'absent', secure: false } });
    const button = t.card('chatterbox').querySelector('.mo-cmd__copy');
    button.focus();
    t.env.click(button);
    await settle(t.env);
    assert.equal(t.env.clipboard.written.length, 1);
    assert.match(t.env.clipboard.written[0], /^docker compose /);
    assert.equal(t.doc.activeElement, button, 'focus came back from the fallback textarea');
    assert.equal(textOf(button), 'Copied');
    t.env.restore();
    t = await mountModels({ env: { clipboard: 'reject' } });
    t.env.execCommandResult = false;
    t.env.click(t.card('pocket-tts').querySelector('.mo-cmd__copy'));
    await settle(t.env);
    assert.deepEqual(toasts(t), [{ kind: 'error', text: "Couldn't copy here. Select the command and copy it instead." }]);
    assert.equal(textOf(t.card('pocket-tts').querySelector('.mo-cmd__copy')), 'Copy command');
  });

  it('[MD-04] one primary action per state, with "Delete files…" in the overflow for models on disk', async () => {
    const list = [...inventory(), { id: 'qwen3/0.6b-base', type: 'tts', provider: 'qwen3', state: 'available', model_format: 'external', provider_available: true, description: 'Experimental Qwen3 0.6B voice cloning model', size_mb: 1800 }];
    t = await mountModels({ models: list });
    // Loaded -> Unload (the Kokoro card's header carries its state pill).
    const kokoro = t.row('kokoro');
    assert.equal(textOf(t.card('kokoro').querySelector('.mo-card__pill')), 'Loaded · GPU');
    assert.equal(textOf(primaryOf(kokoro)), 'Unload');
    assert.equal(primaryOf(kokoro).getAttribute('aria-label'), 'Unload Kokoro');
    assert.equal(kokoro.querySelector('.mo-row__more').hidden, false);
    // Downloaded -> Load, with the note that it unloads the loaded speech model.
    const amy = t.row('piper/en_US-amy-medium');
    assert.equal(textOf(pillOf(amy)), 'Downloaded');
    assert.equal(textOf(primaryOf(amy)), 'Load');
    assert.equal(primaryOf(amy).getAttribute('aria-label'), 'Load Amy (US)');
    assert.equal(textOf(amy.querySelector('.mo-row__note')), 'Unloads Kokoro');
    assert.equal(amy.querySelector('.mo-row__more').hidden, false);
    // Available -> Download (no overflow: nothing to delete).
    const arctic = t.row('piper/en_US-arctic-medium');
    assert.equal(textOf(pillOf(arctic)), 'Not downloaded');
    assert.equal(textOf(primaryOf(arctic)), 'Download');
    assert.equal(arctic.querySelector('.mo-row__more').hidden, true);
    // External weights -> "Downloads on first load" and Load.
    const qwen = t.row('qwen3/0.6b-base');
    assert.equal(textOf(pillOf(qwen)), 'Downloads on first load');
    assert.equal(textOf(primaryOf(qwen)), 'Load');
    // Busy (an operation running) -> no action, a busy pill; failure -> the reason inline and Retry.
    t.app.set({ ops: { 'piper/en_US-amy-medium': { kind: 'load', startedAt: Date.now(), progress: null, error: '', phase: '' } } });
    await settle(t.env);
    assert.equal(textOf(pillOf(amy)), 'Loading · 0 s');
    assert.equal(pillOf(amy).className.includes('pill--busy'), true);
    assert.equal(primaryOf(amy).hidden, true);
    assert.equal(amy.querySelector('.mo-row__more').hidden, true);
    t.app.set({ ops: { 'piper/en_US-amy-medium': { kind: 'load', startedAt: Date.now(), progress: null, error: 'Model reverted to available — load failed', phase: '' } } });
    await settle(t.env);
    assert.equal(textOf(pillOf(amy)), "Couldn't load");
    assert.equal(textOf(amy.querySelector('.mo-row__error-text')), 'Model reverted to available — load failed');
    assert.equal(amy.querySelector('.mo-row__error-text').getAttribute('role'), 'alert');
    assert.equal(amy.querySelector('.mo-row__error').hidden, false);
    assert.equal(textOf(primaryOf(amy)), 'Retry');
    // The overflow menu holds Delete files… (a menu button with its menu outside the card).
    const more = kokoro.querySelector('.mo-row__more');
    assert.equal(more.getAttribute('aria-haspopup'), 'menu');
    const menu = t.$(more.getAttribute('aria-controls'));
    assert.equal(menu.parentElement.id, 'mo-menus');
    assert.deepEqual(menu.querySelectorAll('[role="menuitem"]').map(textOf), ['Delete files…']);
    assertAccessibleNames(t.root);
  });

  it('[MD-04] Load runs the operation: busy while it runs, focus stays on the same button, then Unload', async () => {
    t = await mountModels();
    t.server.loadMs = 4000;
    const amy = t.row('piper/en_US-amy-medium');
    const button = primaryOf(amy);
    button.focus();
    t.env.click(button);
    await settle(t.env);
    const posts = t.env.fetch.callsTo('POST /api/models/:id/load');
    assert.equal(posts.length, 1);
    assert.equal(posts[0].path, '/api/models/piper%2Fen_US-amy-medium/load', 'the id is encodeURIComponent-ed');
    assert.equal(button.getAttribute('aria-busy'), 'true');
    assert.equal(button.getAttribute('aria-disabled'), 'true', 'aria-disabled, not disabled: focus stays on it');
    assert.equal(textOf(button), 'Loading…');
    assert.equal(t.doc.activeElement, button);
    assert.equal(textOf(pillOf(amy)), 'Loading · 0 s');
    await settle(t.env, 2000);
    assert.equal(textOf(pillOf(amy).querySelector('.pill__elapsed')), '· 2 s', 'the elapsed counter ticks');
    assert.equal(pillOf(amy).querySelector('.pill__elapsed').getAttribute('aria-hidden'), 'true');
    // A second click on the busy button does nothing.
    t.env.click(button);
    await settle(t.env);
    assert.equal(t.env.fetch.callsTo('POST /api/models/:id/load').length, 1);
    await settle(t.env, 4000);
    assert.equal(amy, t.row('piper/en_US-amy-medium'), 'the row was patched, not replaced');
    assert.equal(textOf(pillOf(amy)), 'Loaded · GPU');
    assert.equal(textOf(button), 'Unload');
    assert.equal(button.hasAttribute('aria-busy'), false);
    assert.equal(t.doc.activeElement, button, 'the button that had focus keeps it as it changes type');
    assert.equal(textOf(t.card('kokoro').querySelector('.mo-card__pill')), 'Downloaded', 'one speech model at a time');
  });

  it('[MD-04] a failed load shows the reason inline (no toast while Models shows); Retry repeats it and Dismiss clears it', async () => {
    t = await mountModels();
    t.server.fail['piper/en_US-amy-medium'] = { op: 'load', status: 500, message: 'CUDA out of memory' };
    const amy = t.row('piper/en_US-amy-medium');
    t.env.click(primaryOf(amy));
    await settle(t.env, 10);
    assert.equal(textOf(amy.querySelector('.mo-row__error-text')), 'CUDA out of memory');
    assert.equal(textOf(primaryOf(amy)), 'Retry');
    assert.deepEqual(toasts(t), [], 'the row says it; no toast while the Models view shows');
    delete t.server.fail['piper/en_US-amy-medium'];
    t.env.click(primaryOf(amy));
    await settle(t.env, 10);
    assert.equal(t.env.fetch.callsTo('POST /api/models/:id/load').length, 2, 'Retry repeats the load');
    assert.equal(textOf(primaryOf(amy)), 'Unload');
    assert.equal(amy.querySelector('.mo-row__error').hidden, true);
    // Dismiss clears a failure without retrying.
    t.server.fail['piper/en_GB-alan-medium'] = { op: 'load', status: 500, message: 'boom' };
    const alan = t.row('piper/en_GB-alan-medium');
    t.env.click(primaryOf(alan));
    await settle(t.env, 10);
    assert.equal(alan.querySelector('.mo-row__error').hidden, false);
    t.env.click(alan.querySelector('.mo-row__dismiss'));
    await settle(t.env);
    assert.equal(alan.querySelector('.mo-row__error').hidden, true);
    assert.equal(textOf(primaryOf(alan)), 'Load');
    assert.equal(t.doc.activeElement, primaryOf(alan));
  });

  it('[MD-04] Unload that the server refuses refreshes the row and says why inline', async () => {
    t = await mountModels();
    t.server.fail.kokoro = { op: 'unload', status: 503, message: 'The kokoro backend is busy' };
    const before = t.env.fetch.callsTo('GET /api/models').length;
    t.env.click(primaryOf(t.row('kokoro')));
    await settle(t.env);
    assert.equal(t.env.fetch.callsTo('DELETE /api/models/:id').length, 1);
    assert.ok(t.env.fetch.callsTo('GET /api/models').length > before, 'the row is refreshed from the server');
    const error = t.row('kokoro').querySelector('.mo-row__error-text');
    assert.equal(textOf(error), "Couldn't unload: The kokoro backend is busy");
    assert.equal(t.row('kokoro').querySelector('.mo-row__error').hidden, false);
    assert.equal(textOf(primaryOf(t.row('kokoro'))), 'Unload');
    delete t.server.fail.kokoro;
    t.env.click(primaryOf(t.row('kokoro')));
    await settle(t.env);
    assert.equal(t.row('kokoro').querySelector('.mo-row__error').hidden, true, 'a new attempt clears the old reason');
    assert.equal(textOf(primaryOf(t.row('kokoro'))), 'Load');
  });

  it('[MD-04] Delete files… confirms with the Delete files copy; Cancel sends nothing; OK deletes, refreshes and confirms in a toast', async () => {
    t = await mountModels();
    const amy = t.row('piper/en_US-amy-medium');
    const more = amy.querySelector('.mo-row__more');
    more.focus();
    t.env.click(more);
    const menu = t.$(more.getAttribute('aria-controls'));
    assert.equal(menu.hidden, false);
    t.env.click(menu.querySelector('[role="menuitem"]'));
    await settle(t.env);
    assert.equal(t.$('app-confirm').open, true);
    assert.equal(textOf(t.$('app-confirm-title')), 'Delete files?');
    assert.equal(textOf(t.$('app-confirm-msg')), 'Delete downloaded files for Piper en_US-amy-medium? This unloads the model and removes its cached files.');
    assert.equal(textOf(t.$('app-confirm-ok')), 'Delete files');
    assert.equal(t.doc.activeElement, t.$('app-confirm-cancel'), 'a danger confirm focuses Cancel');
    await answerConfirm(t, false);
    assert.equal(t.env.fetch.callsTo('DELETE /api/models/:id/artifacts').length, 0, 'Cancel sends nothing');
    assert.equal(t.doc.activeElement, more);
    t.env.click(more);
    t.env.click(menu.querySelector('[role="menuitem"]'));
    await settle(t.env);
    await answerConfirm(t, true);
    await settle(t.env);
    const calls = t.env.fetch.callsTo('DELETE /api/models/:id/artifacts');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].path, '/api/models/piper%2Fen_US-amy-medium/artifacts');
    assert.equal(textOf(primaryOf(amy)), 'Download');
    assert.equal(t.doc.activeElement, primaryOf(amy), 'the overflow button went away, so focus moved to the row action');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'Deleted the downloaded files for Amy (US).' }]);
  });

  it('[MD-09] a card header is a disclosure button; collapsed, the body is hidden (out of the tab order and the accessibility tree)', async () => {
    t = await mountModels();
    const card = t.card('piper');
    const toggle = card.querySelector('.mo-card__toggle');
    assert.equal(toggle.localName, 'button');
    assert.equal(toggle.parentElement.localName, 'h2');
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    const body = t.$(toggle.getAttribute('aria-controls'));
    assert.ok(card.contains(body));
    assert.equal(body.hidden, false);
    const reachable = () => focusables(t.root).filter((el) => body.contains(el)).length;
    assert.ok(reachable() > 0);
    toggle.focus();
    t.env.key('Enter');
    await settle(t.env);
    assert.equal(toggle.getAttribute('aria-expanded'), 'false');
    assert.equal(body.hidden, true);
    assert.equal(reachable(), 0, 'nothing inside a collapsed card takes focus');
    assert.equal(t.doc.activeElement, toggle);
    t.env.key(' ');
    await settle(t.env);
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(body.hidden, false);
  });

  it('[MD-10] Piper: "Name (CC)" and quality, five rows then Show all N / Show fewer, a filter and Downloaded only', async () => {
    t = await mountModels();
    const card = t.card('piper');
    assert.deepEqual(rowIds(card), PIPER_FIRST_FIVE);
    assert.deepEqual(card.querySelectorAll('.mo-row__name').map(textOf), ['Amy (US)', 'Alan (GB)', 'Arctic (US)', 'Joe (US)', 'Lessac (US)']);
    assert.equal(textOf(card.querySelector('.mo-row__meta')), 'Medium quality · 35 MB');
    assert.match(textOf(card.querySelector('.mo-card__count')), /^2 of 7 voices on disk$/);
    const more = card.querySelector('.mo-card__more');
    assert.equal(textOf(more), 'Show all 7 voices');
    assert.equal(more.getAttribute('aria-expanded'), 'false');
    assert.equal(more.getAttribute('aria-controls'), card.querySelector('.mo-rows').id);
    more.focus();
    t.env.click(more);
    await settle(t.env);
    assert.equal(rowIds(card).length, 7);
    assert.equal(textOf(more), 'Show fewer');
    assert.equal(more.getAttribute('aria-expanded'), 'true');
    assert.equal(t.doc.activeElement, more);
    t.env.click(more);
    await settle(t.env);
    assert.deepEqual(rowIds(card), PIPER_FIRST_FIVE);
    // The filter shows every match; a filter with no match says so and offers to clear it.
    const filter = card.querySelector('.mo-filter__input');
    assert.equal(filter.getAttribute('type'), 'search');
    assert.equal(t.doc.querySelector(`label[for="${filter.id}"]`).textContent, 'Filter Piper voices');
    t.env.input(filter, 'gb');
    await settle(t.env);
    assert.deepEqual(rowIds(card), ['piper/en_GB-alan-medium', 'piper/en_GB-cori-high']);
    assert.equal(more.parentElement.hidden, true, 'no Show all while filtering');
    t.env.input(filter, 'zzz');
    await settle(t.env);
    assert.deepEqual(rowIds(card), []);
    assert.equal(textOf(card.querySelector('.mo-card__empty-text')), 'No voices match “zzz”.');
    t.env.click(card.querySelector('[data-action="clear-filter"]'));
    await settle(t.env);
    assert.equal(filter.value, '');
    assert.equal(t.doc.activeElement, filter);
    assert.deepEqual(rowIds(card), PIPER_FIRST_FIVE);
    const onDisk = card.querySelector('.mo-ondisk');
    t.env.click(onDisk);
    await settle(t.env);
    assert.equal(onDisk.getAttribute('aria-pressed'), 'true');
    assert.deepEqual(rowIds(card), ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium']);
  });

  it('[MD-10] what the filter leaves on screen is announced once typing pauses (WCAG 4.1.3): the count, the empty state, "Downloaded only"; polls say nothing', async () => {
    t = await mountModels();
    const { FILTER_ANNOUNCE_MS } = await load('views/models/cards.js');
    assert.equal(FILTER_ANNOUNCE_MS, 700);
    const writes = spyWrites(t.$('app-sr-status'));
    const said = () => writes.filter(Boolean);
    const card = t.card('piper');
    const filter = card.querySelector('.mo-filter__input');
    filter.focus();
    // Typing filters at once but is said only after a pause, once, with the last text.
    t.env.input(filter, 'g');
    await settle(t.env, 300);
    t.env.input(filter, 'gb');
    await settle(t.env, 300);
    assert.deepEqual(rowIds(card), ['piper/en_GB-alan-medium', 'piper/en_GB-cori-high']);
    assert.deepEqual(said(), [], 'nothing is said while the listener types');
    await settle(t.env, 400);
    assert.deepEqual(said(), ['2 voices match “gb”']);
    assert.equal(announced(t), '2 voices match “gb”');
    t.env.input(filter, 'amy');
    await settle(t.env, 700);
    assert.equal(announced(t), '1 voice matches “amy”');
    // The empty state is said too, not only shown.
    t.env.input(filter, 'zzz');
    await settle(t.env, 700);
    assert.equal(textOf(card.querySelector('.mo-card__empty-text')), 'No voices match “zzz”.');
    assert.equal(announced(t), 'No voices match “zzz”');
    assert.equal(t.doc.activeElement, filter, 'announcing never moves focus');
    // Polls and re-renders while the filter stays are silent.
    const before = said().length;
    await settle(t.env, 10000);
    await settle(t.env, 10000);
    assert.ok(t.env.fetch.callsTo('GET /api/models').length >= 2, 'the view polled the inventory');
    assert.equal(said().length, before, 'a poll is never announced');
    // Clear filter and "Downloaded only" say what the list now shows.
    t.env.click(card.querySelector('[data-action="clear-filter"]'));
    await settle(t.env, 700);
    assert.equal(announced(t), 'Showing 5 of 7 voices');
    t.env.click(card.querySelector('.mo-ondisk'));
    await settle(t.env, 700);
    assert.equal(announced(t), '2 voices on disk');
    // A filter typed just before the listener leaves the sub is not said over the next one.
    const count = said().length;
    t.env.input(filter, 'zzz');
    t.env.change(t.root.querySelector('input[name="mo-sub"][value="stt"]'), true);
    await settle(t.env, 700);
    assert.ok(t.card('faster-whisper'));
    assert.equal(said().length, count, 'a card that left the screen says nothing');
  });

  it('[MD-11] STT: short names, source and format, Default on the configured default, five then all; only the STT sub is rendered', async () => {
    t = await mountModels({ sub: 'stt' });
    assert.equal(t.card('kokoro'), null, 'only the visible sub is rendered');
    const card = t.card('faster-whisper');
    assert.deepEqual(card.querySelectorAll('.mo-row__name').map(textOf), ['base', 'distil-small.en', 'large-v3', 'large-v3-turbo', 'medium']);
    const base = t.row(STT_DEFAULT);
    assert.equal(base.querySelector('.mo-row__default').hidden, false);
    assert.equal(textOf(base.querySelector('.mo-row__default')), 'Default');
    assert.equal(card.querySelectorAll('.mo-row__default').filter((b) => !b.hidden).length, 1);
    assert.equal(textOf(base.querySelector('.mo-row__meta')), 'Good balance for CPU · Systran · CT2 · 150 MB');
    assert.equal(textOf(pillOf(base)), 'Loaded · CPU');
    assert.equal(textOf(primaryOf(base)), 'Unload');
    assert.equal(textOf(primaryOf(t.row('Systran/faster-whisper-medium'))), 'Download');
    assert.equal(t.row('Systran/faster-distil-whisper-small.en').querySelector('.mo-row__note').hidden, true, 'an STT load unloads no speech model');
    const more = card.querySelector('.mo-card__more');
    assert.equal(textOf(more), 'Show all 7 models');
    t.env.click(more);
    await settle(t.env);
    assert.equal(rowIds(card).length, 7);
    assert.equal(t.card('faster-whisper').querySelector('.mo-filter__input').getAttribute('placeholder'), 'Filter models');
  });

  it('[MD-12] the Kokoro card shows its voice count and features, not a chip per voice, and the speed measured here', async () => {
    t = await mountModels({ env: { seed: { 'open-speech-perf': JSON.stringify({ kokoro: { ttfaMs: 140, renderMs: 400, durationS: 10, rtf: 0.04, at: 1 } }) } } });
    const card = t.card('kokoro');
    assert.equal(card.classList.contains('mo-card--solo'), true);
    assert.equal(card.querySelectorAll('.mo-row').length, 1);
    assert.equal(textOf(card.querySelector('.mo-card__desc')), 'Fast, natural speech with weighted voice blends');
    assert.equal(textOf(card.querySelector('.mo-row__meta')), '54 voices · Blends · 330 MB');
    assert.doesNotMatch(textOf(card), /voice_0|\+\d+ more/);
    assert.equal(textOf(card.querySelector('.mo-row__perf-text')), 'First audio 140 ms · RTF 0.04 (this machine)');
    assert.equal(card.querySelector('.mo-row__slow').hidden, true);
    assert.equal(card.querySelector('.mo-card__default').hidden, false, 'Kokoro is the configured default (beside its name)');
    assert.equal(card.querySelector('.mo-row__default'), null);
    t.env.restore();
    t = await mountModels({ env: { seed: { 'open-speech-perf': JSON.stringify({ kokoro: { ttfaMs: 900, renderMs: 12000, durationS: 10, rtf: 1.2, at: 1 } }) } } });
    assert.equal(t.card('kokoro').querySelector('.mo-row__slow').hidden, false);
    assert.equal(textOf(t.card('kokoro').querySelector('.mo-row__slow')), 'Slower than real time');
  });

  it('[BUG-10] polling patches only the changed rows: focus stays, collapsed cards stay collapsed, the other rows keep their elements', async () => {
    t = await mountModels();
    t.server.loadMs = 7000;
    // The listener collapsed one card and is focused on a button in another row.
    const pocket = t.card('pocket-tts').querySelector('.mo-card__toggle');
    t.env.click(pocket);
    await settle(t.env);
    const arctic = primaryOf(t.row('piper/en_US-arctic-medium'));
    arctic.focus();
    const piper = t.card('piper');
    const snapshot = new Map(piper.querySelectorAll('.mo-row').map((li) => [li.getAttribute('data-model-id'), { el: li, html: li.innerHTML }]));
    // An operation started elsewhere in the app (runOp owns the polling; the view only renders the store).
    const run = t.models.runOp('piper/en_US-amy-medium', 'load');
    for (let i = 0; i < 4; i += 1) {
      await settle(t.env, 2000);
      assert.equal(t.doc.activeElement, arctic, `focus is kept on poll ${i + 1}`);
      assert.equal(pocket.getAttribute('aria-expanded'), 'false', 'a collapsed card stays collapsed');
      assert.equal(t.$(pocket.getAttribute('aria-controls')).hidden, true);
    }
    assert.equal(await run, true);
    await settle(t.env);
    for (const li of piper.querySelectorAll('.mo-row')) {
      const id = li.getAttribute('data-model-id');
      const before = snapshot.get(id);
      assert.equal(li, before.el, `${id} kept its element`);
      // Rows whose state did not change were not touched at all.
      if (['piper/en_US-arctic-medium', 'piper/en_US-joe-medium', 'piper/en_US-lessac-low'].includes(id)) {
        assert.equal(li.innerHTML, before.html, `${id} was not re-rendered`);
      }
    }
    assert.equal(textOf(pillOf(t.row('piper/en_US-amy-medium'))), 'Loaded · GPU');
    // Alan's row was patched in place: loading it would now unload Amy, not Kokoro.
    assert.equal(textOf(t.row('piper/en_GB-alan-medium').querySelector('.mo-row__note')), 'Unloads Amy (US)');
    assert.deepEqual(rowIds(piper), PIPER_FIRST_FIVE, 'the rows keep the order they were shown in');
    assert.equal(t.doc.activeElement, arctic);
  });

  it('[BUG-10] a row that leaves "Downloaded only" while it holds focus (Delete files…) hands focus to the next row, else the previous one, else the chip; never to <body>', async () => {
    // Amy, Joe (en_US) and Alan (en_GB) are on disk.
    t = await mountModels({ models: inventory().map((m) => (m.id === 'piper/en_US-joe-medium' ? { ...m, state: 'downloaded' } : m)) });
    const card = t.card('piper');
    t.env.click(card.querySelector('.mo-ondisk'));
    await settle(t.env);
    assert.deepEqual(rowIds(card), ['piper/en_US-amy-medium', 'piper/en_US-joe-medium', 'piper/en_GB-alan-medium']);
    const deleteFiles = async (id) => {
      const more = t.row(id).querySelector('.mo-row__more');
      more.focus();
      t.env.key('Enter');
      await settle(t.env);
      t.env.key('Enter');   // the focused item: Delete files…
      await settle(t.env);
      await answerConfirm(t, true);
      await settle(t.env);
      await settle(t.env);
    };
    await deleteFiles('piper/en_US-amy-medium');
    assert.deepEqual(rowIds(card), ['piper/en_US-joe-medium', 'piper/en_GB-alan-medium']);
    assert.equal(t.doc.activeElement, primaryOf(t.row('piper/en_US-joe-medium')), 'the next row\'s action');
    await deleteFiles('piper/en_GB-alan-medium');
    assert.deepEqual(rowIds(card), ['piper/en_US-joe-medium']);
    assert.equal(t.doc.activeElement, primaryOf(t.row('piper/en_US-joe-medium')), 'the last row went: the previous row\'s action');
    await deleteFiles('piper/en_US-joe-medium');
    assert.deepEqual(rowIds(card), []);
    assert.equal(t.doc.activeElement, card.querySelector('.mo-ondisk'), 'no row left: "Downloaded only", which narrows the list');
    assert.equal(textOf(card.querySelector('.mo-card__empty-text')), 'No voices are downloaded yet.');
    assert.equal(t.env.fetch.callsTo('DELETE /api/models/:id/artifacts').length, 3);
  });

  it('[BUG-10] under "Downloaded only" a download that fails keeps its row, its error and Retry until dismissed; Dismiss then moves focus on', async () => {
    t = await mountModels();
    const card = t.card('piper');
    t.env.fetch.route('POST /api/models/:id/prefetch', () => delay(3000, json(500, { error: { message: 'Disk full', code: 'failed' } })));
    t.env.click(primaryOf(t.row('piper/en_US-arctic-medium')));
    await settle(t.env, 10);
    // A download in progress counts as on its way to disk.
    t.env.click(card.querySelector('.mo-ondisk'));
    await settle(t.env);
    assert.deepEqual(rowIds(card), ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium', 'piper/en_US-arctic-medium']);
    await settle(t.env, 3000);
    await settle(t.env);
    assert.equal(t.app.get().ops['piper/en_US-arctic-medium'].error, 'Disk full');
    assert.deepEqual(rowIds(card), ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium', 'piper/en_US-arctic-medium'], 'the failed row stays');
    const arctic = t.row('piper/en_US-arctic-medium');
    assert.match(textOf(arctic.querySelector('.mo-row__error-text')), /Disk full/);
    assert.equal(textOf(primaryOf(arctic)), 'Retry');
    assert.equal(toasts(t).length, 0, 'said inline, where it happened');
    const dismiss = arctic.querySelector('.mo-row__dismiss');
    dismiss.focus();
    t.env.click(dismiss);
    await settle(t.env);
    assert.deepEqual(rowIds(card), ['piper/en_US-amy-medium', 'piper/en_GB-alan-medium'], 'dismissed: not on disk, so it leaves');
    assert.equal(t.doc.activeElement, primaryOf(t.row('piper/en_GB-alan-medium')), 'focus went to the previous row\'s action');
  });

  it('[BUG-10] a busy action kept as a spinner while it had focus gives way to the pill alone once focus moves on', async () => {
    t = await mountModels();
    t.server.loadMs = 6000;
    const amy = t.row('piper/en_US-amy-medium');
    const button = primaryOf(amy);
    button.focus();
    t.env.click(button);
    await settle(t.env, 10);
    assert.equal(button.getAttribute('aria-busy'), 'true');
    assert.equal(button.hidden, false, 'kept while it has focus');
    primaryOf(t.row('piper/en_GB-alan-medium')).focus();
    await settle(t.env, 1000);
    assert.equal(button.hidden, true, 'focus moved on: no stale spinner beside the pill');
    assert.match(textOf(pillOf(amy)), /^Loading · 1 s$/);
    await settle(t.env, 6000);
    await settle(t.env);
    assert.equal(button.hidden, false);
    assert.equal(textOf(button), 'Unload');
  });

  it('[MD-04] the overflow menu opens towards the card from the button at the row\'s end (the Kokoro card\'s too) and lives outside the cards', async () => {
    t = await mountModels({ env: { viewport: { width: 1440, height: 900 } } });
    const more = t.card('kokoro').querySelector('.mo-row__more');
    more.__rect = { x: 1330, y: 300, width: 44, height: 44 };
    const menu = t.$(more.getAttribute('aria-controls'));
    menu.__rect = { width: 200, height: 52 };
    t.env.click(more);
    assert.equal(menu.hidden, false);
    assert.equal(menu.style.getPropertyValue('--menu-x'), `${1330 + 44 - 200}px`, 'its right edge lines up with the button\'s, inside the card');
    assert.equal(menu.parentElement, t.$('mo-menus'), 'not inside a card (a size container would trap it)');
  });

  it('[BUG-10] a worker that comes online replaces its offline card in place; collapse survives a sub switch', async () => {
    t = await mountModels();
    const order = () => t.root.querySelectorAll('.mo-card').map((c) => c.getAttribute('data-provider'));
    const piperToggle = t.card('piper').querySelector('.mo-card__toggle');
    t.env.click(piperToggle);
    const list = t.server.list.map((m) => (m.provider === 'chatterbox' ? { ...m, state: 'available', provider_available: true } : m));
    t.app.set({ models: { ...t.app.get().models, list, fetchedAt: Date.now() } });
    await settle(t.env);
    assert.deepEqual(order(), ['kokoro', 'piper', 'chatterbox', 'pocket-tts'], 'the order shown first is kept');
    assert.equal(t.card('chatterbox').getAttribute('data-type'), 'models');
    assert.equal(textOf(primaryOf(t.row('chatterbox/regular'))), 'Load');
    t.ctl.show({ view: 'models', sub: 'stt', id: '', path: 'models/stt', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.equal(t.card('piper'), null);
    t.ctl.show({ view: 'models', sub: 'tts', id: '', path: 'models/tts', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.equal(t.card('piper').querySelector('.mo-card__toggle').getAttribute('aria-expanded'), 'false', 'module state keeps the collapse');
  });
});
