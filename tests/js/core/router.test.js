import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv(options);
  const router = await importFresh('core/router.js');
  const prefs = await import('../../../src/static/js/core/prefs.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  prefs.load();
  return { router, prefs, app };
}

const urlHash = () => location.hash;

describe('core/router.parseHash (pure)', () => {
  it('[GS-10] deep links: a section, the library, a document (id decoded), subs and an unknown hash', async () => {
    const { router } = await setup();
    const p = (h, o) => router.parseHash(h, o);
    assert.deepEqual(p('#home'), { view: 'home', sub: '', id: '', path: 'home', shot: '' });
    assert.deepEqual(p('#reader'), { view: 'reader', sub: '', id: '', path: 'reader', shot: '' });
    assert.deepEqual(p('#reader/a%20b%2Fc'), { view: 'reader', sub: '', id: 'a b/c', path: 'reader/a%20b%2Fc', shot: '' });
    assert.deepEqual(p('#speak'), { view: 'speak', sub: '', id: '', path: 'speak', shot: '' });
    assert.equal(p('#speak/live').path, 'speak/live');
    assert.equal(p('#speak/nope').path, 'speak');
    // Voices' default section is the one list of named voices and presets: an old
    // #voices/named or #voices/presets link is an unknown sub and opens it.
    assert.deepEqual(p('#voices'), { view: 'voices', sub: '', id: '', path: 'voices', shot: '' });
    assert.equal(p('#voices/recordings').path, 'voices/recordings');
    assert.equal(p('#voices/named').path, 'voices');
    assert.equal(p('#voices/presets').path, 'voices');
    assert.equal(p('#voices/zzz').path, 'voices', 'an unknown sub falls back to the default sub');
    assert.equal(p('#studio').path, 'studio/conversations');
    assert.deepEqual(p('#studio/conversations/c-1'), { view: 'studio', sub: 'conversations', id: 'c-1', path: 'studio/conversations/c-1', shot: '' });
    assert.equal(p('#studio/composer').path, 'studio/composer');
    assert.equal(p('#models').path, 'models/tts');
    assert.equal(p('#models/stt').path, 'models/stt');
    for (const v of ['transcribe', 'history', 'settings']) assert.equal(p(`#${v}`).path, v);
    assert.equal(p('#nope'), null);
    assert.equal(p(''), null);
    assert.equal(p('#'), null);
    assert.equal(p('#READER').path, 'reader');
  });

  it('[GS-10] legacy aliases: #voicelab, #tts, #stt, and #more (lastRoute, else models)', async () => {
    const { router } = await setup();
    assert.equal(router.parseHash('#voicelab').path, 'voices/recordings');
    assert.equal(router.parseHash('#tts').path, 'speak');
    assert.equal(router.parseHash('#stt').path, 'transcribe');
    assert.equal(router.parseHash('#more').path, 'models/tts');
    assert.equal(router.parseHash('#more', { lastRoute: 'history' }).path, 'history');
    assert.equal(router.parseHash('#more', { lastRoute: 'more' }).path, 'models/tts');
  });

  it('a trailing ?shot= becomes route.shot (the screenshot hook)', async () => {
    const { router } = await setup();
    assert.deepEqual(router.parseHash('#reader/d1?shot=appearance'), { view: 'reader', sub: '', id: 'd1', path: 'reader/d1', shot: 'appearance' });
  });

  it('format() builds the canonical hash with the id encoded', async () => {
    const { router } = await setup();
    assert.equal(router.format({ view: 'reader', id: 'a b' }), '#reader/a%20b');
    assert.equal(router.format({ view: 'voices', sub: 'recordings' }), '#voices/recordings');
    assert.equal(router.format({ view: 'speak', sub: '' }), '#speak');
  });
});

describe('core/router navigation', () => {
  it('[GS-10] start() opens the hash, normalised with replaceState, and calls onRoute with focus none', async () => {
    const { router, app } = await setup({ url: 'http://127.0.0.1:8100/web#studio' });
    const seen = [];
    await router.start((route, prev, nav) => { seen.push([route.path, prev, nav.focus]); });
    assert.deepEqual(seen, [['studio/conversations', null, 'none']]);
    assert.equal(urlHash(), '#studio/conversations');
    assert.equal(history.length, 1, 'normalising does not add an entry');
    assert.equal(app.get().route.path, 'studio/conversations');
  });

  it('[GS-15] without a hash, start() opens Home; with Open on set to Last view it reopens prefs.lastRoute', async () => {
    const first = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, lastRoute: 'models/stt' }) } });
    await first.router.start(() => {});
    assert.equal(urlHash(), '#home', 'the remembered route alone does not decide the landing page');
    env.restore();
    const second = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, lastRoute: 'models/stt', startOn: 'last' }) } });
    await second.router.start(() => {});
    assert.equal(urlHash(), '#models/stt');
    env.restore();
    const third = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, startOn: 'last' }) } });
    await third.router.start(() => {});
    assert.equal(urlHash(), '#home', 'nothing remembered yet');
  });

  it('a deep link wins over the landing page, and Home is a route like the others', async () => {
    const { router, prefs } = await setup({ url: 'http://127.0.0.1:8100/web#speak' });
    await router.start(() => {});
    assert.equal(urlHash(), '#speak');
    await router.navigate('home');
    assert.equal(urlHash(), '#home');
    assert.equal(prefs.get().lastRoute, 'home');
  });

  it('[GS-10] an unknown hash is replaced by the default route', async () => {
    const { router } = await setup({ url: 'http://127.0.0.1:8100/web#bogus', seed: { 'open-speech-prefs': JSON.stringify({ v: 1, lastRoute: 'history' }) } });
    await router.start(() => {});
    assert.equal(urlHash(), '#home');
  });

  it('[GS-11] a section switch adds one history entry; navigating to the current route adds none', async () => {
    const { router } = await setup();
    const seen = [];
    await router.start((r, prev, nav) => seen.push([r.path, nav.focus]));
    await router.navigate('speak');
    await router.navigate('#speak');
    await router.navigate('voices/recordings', { focus: 'keep' });
    assert.deepEqual(history.__entries.urls.map((u) => new URL(u).hash), ['#home', '#speak', '#voices/recordings']);
    assert.deepEqual(seen, [['home', 'none'], ['speak', 'heading'], ['voices/recordings', 'keep']]);
  });

  it('[GS-11] opening a document pushes #reader/<id>, so Back returns to the library through hashchange', async () => {
    const { router } = await setup({ url: 'http://127.0.0.1:8100/web#reader' });
    const seen = [];
    await router.start((r) => seen.push(r.path));
    await router.navigate(`reader/${encodeURIComponent('doc 1')}`);
    assert.equal(urlHash(), '#reader/doc%201');
    history.back();
    await env.flush();
    assert.deepEqual(seen, ['reader', 'reader/doc%201', 'reader']);
    assert.equal(router.current().path, 'reader');
  });

  it('replace: true replaces the current entry', async () => {
    const { router } = await setup();
    await router.start(() => {});
    await router.navigate('models/tts', { replace: true });
    assert.equal(history.length, 1);
    assert.equal(urlHash(), '#models/tts');
  });

  it('a link or a typed hash (hashchange) is routed with focus heading', async () => {
    const { router } = await setup({ html: '<a id="l" href="#transcribe">Transcribe</a>' });
    const seen = [];
    await router.start((r, prev, nav) => seen.push([r.path, nav.focus]));
    document.getElementById('l').click();
    env.setHash('#stt');
    await env.flush();
    assert.deepEqual(seen, [['home', 'none'], ['transcribe', 'heading']]);
    assert.equal(urlHash(), '#transcribe', 'the alias was normalised');
  });

  it('[GS-15] every applied route is saved as prefs.lastRoute, except the wizards (voices/clone as voices/recordings, voices/design as voices) and the shot', async () => {
    const { router, prefs } = await setup();
    await router.start(() => {});
    await router.navigate('history');
    assert.equal(prefs.get().lastRoute, 'history');
    assert.equal(JSON.parse(env.localStorage.__data['open-speech-prefs']).lastRoute, 'history');
    await router.navigate('voices/clone');
    assert.equal(prefs.get().lastRoute, 'voices/recordings');
    await router.navigate('voices/design');
    assert.equal(prefs.get().lastRoute, 'voices');
    await router.navigate('reader/d1?shot=appearance');
    assert.equal(prefs.get().lastRoute, 'reader/d1');
    assert.equal(router.current().shot, 'appearance');
    assert.equal(urlHash(), '#reader/d1?shot=appearance');
  });

  it('the newest route wins at once: a slow callback never holds up the next route', async () => {
    const { router, app } = await setup();
    const order = [];
    const started = router.start(async (r) => {
      order.push(`start:${r.path}`);
      await new Promise((resolve) => setTimeout(resolve, r.path === 'speak' ? 100 : 0));
      order.push(`end:${r.path}`);
    });
    await env.clock.tickAsync(0);
    await started;
    const a = router.navigate('speak');
    const b = router.navigate('history');
    assert.equal(app.get().route.path, 'history', 'the store has the newest route at once');
    assert.equal(router.current().path, 'history');
    await env.clock.tickAsync(200);
    assert.deepEqual(await Promise.all([a, b]).then((r) => r.map((x) => x.path)), ['speak', 'history']);
    assert.deepEqual(order, ['start:home', 'end:home', 'start:speak', 'start:history', 'end:history', 'end:speak'], 'history did not wait for speak');
  });

  it('a callback that never settles, or that navigates itself, never blocks or deadlocks later routes', async () => {
    const { router } = await setup();
    const seen = [];
    await router.start(async (r) => {
      seen.push(r.path);
      if (r.path === 'speak') await new Promise(() => {});          // a stalled request
      if (r.path === 'reader/missing') await router.navigate('reader', { replace: true });   // a redirect
    });
    router.navigate('speak');
    await router.navigate('history');
    await router.navigate('reader/missing');
    assert.equal(location.hash, '#reader', 'the redirect replaced the entry');
    await router.navigate('models');
    assert.deepEqual(seen, ['home', 'speak', 'history', 'reader/missing', 'reader', 'models/tts']);
  });
});
