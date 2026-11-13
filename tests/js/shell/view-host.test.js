import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { delay, text } from '../harness/fakes.js';
import { shellEnv, load, settle, partialRoute } from './helpers.js';

let env;
afterEach(() => env?.restore());

const VIEW_IDS = ['reader', 'speak', 'transcribe', 'voices', 'studio', 'history', 'models', 'settings'];

function registry(log, overrides = {}) {
  const out = {};
  for (const id of VIEW_IDS) {
    out[id] = async () => {
      log.push(['import', id]);
      const extra = overrides[id] || {};
      return {
        default: {
          id, title: id.charAt(0).toUpperCase() + id.slice(1), partial: `/static/partials/${id}.html`,
          ...(extra.def || {}),
          init(root) {
            log.push(['init', id, root.id]);
            return {
              show(route, nav) {
                log.push(['show', id, route.path, nav.focus]);
                if (extra.restoreScroll) nav.scrolled = true;
                return extra.show ? extra.show(route, nav) : undefined;
              },
              hide() { log.push(['hide', id]); },
              ...(extra.controller || {}),
            };
          },
        },
      };
    };
  }
  return out;
}

async function setup({ routes = {}, overrides = {} } = {}) {
  env = await shellEnv({ routes });
  const host = await load('shell/view-host.js');
  const router = await load('core/router.js');
  const store = await load('core/store.js');
  const log = [];
  host.initViewHost({ registry: registry(log, overrides) });
  const events = [];
  store.on('route:changed', (detail) => events.push([detail.route.path, detail.prev ? detail.prev.path : null]));
  await router.start(host.show);
  await settle(env);
  return { host, router, store, log, events };
}

const shown = () => [...document.querySelectorAll('main > section.view')].filter((s) => !s.hidden).map((s) => s.id);

describe('shell/view-host', () => {
  it('[GS-09] exactly one view section is shown; the others are hidden (out of the accessibility tree)', async () => {
    const { router } = await setup();
    assert.deepEqual(shown(), ['view-reader']);
    await router.navigate('speak');
    assert.deepEqual(shown(), ['view-speak']);
    await router.navigate('voices/recordings');
    assert.deepEqual(shown(), ['view-voices']);
    assert.equal(document.getElementById('view-speak').hidden, true);
  });

  it('[GS-12] a view loads on its first visit (partial, module, init once) and shows on every visit; prefetch never inits', async () => {
    const { router, host, log } = await setup();
    await router.navigate('speak');
    await router.navigate('reader');
    await router.navigate('speak');
    assert.deepEqual(log.filter((e) => e[0] === 'init'), [['init', 'reader', 'view-reader'], ['init', 'speak', 'view-speak']]);
    assert.equal(env.fetch.callsTo('GET /static/partials/speak.html').length, 1, 'the partial is fetched once');
    assert.equal(env.fetch.callsTo('GET /static/partials/speak.html')[0].headers.authorization, undefined, 'partials are static and auth-exempt');
    assert.deepEqual(log.filter((e) => e[0] === 'show').map((e) => e[1]), ['reader', 'speak', 'reader', 'speak']);
    assert.deepEqual(log.filter((e) => e[0] === 'hide').map((e) => e[1]), ['reader', 'speak', 'reader'], 'the view left behind is hidden');
    assert.match(document.getElementById('view-speak').innerHTML, /sp-title/, 'the partial is rendered into the section');
    await host.prefetch(['models', 'history']);
    assert.equal(env.fetch.callsTo('GET /static/partials/models.html').length, 1);
    assert.equal(log.filter((e) => e[0] === 'init' && e[1] === 'models').length, 0, 'prefetch imports without init()');
    await router.navigate('models');
    assert.equal(env.fetch.callsTo('GET /static/partials/models.html').length, 1, 'the prefetched partial is reused');
    assert.equal(log.filter((e) => e[0] === 'init' && e[1] === 'models').length, 1);
  });

  it("[GS-12] prefetch() also runs a view definition's own prefetch() once, so what the view imports lazily is warm before its controls can be used; a visit never calls it and a failed warm-up changes nothing", async () => {
    const warmed = [];
    const { router, host, log } = await setup({ overrides: {
      speak: { def: { prefetch: () => { warmed.push('speak'); return Promise.resolve(); } } },
      voices: { def: { prefetch: () => { warmed.push('voices'); return Promise.reject(new Error('offline')); } } },
      history: { def: { prefetch: () => { warmed.push('history'); throw new Error('sync failure'); } } },
      studio: { def: { prefetch: () => { warmed.push('studio'); } } },
    } });
    await router.navigate('studio');
    assert.deepEqual(warmed, [], 'a route change loads the view but never runs its prefetch()');
    await host.prefetch(['speak', 'voices', 'history', 'models']);
    assert.deepEqual(warmed, ['speak', 'voices', 'history'], 'each prefetched view warms its own imports; a view without prefetch() is fine');
    assert.equal(log.filter((e) => e[0] === 'init' && ['speak', 'voices'].includes(e[1])).length, 0, 'still no init()');
    assert.equal(env.fetch.callsTo('GET /static/partials/voices.html').length, 1, 'the partial is cached even though the warm-up failed');
    await host.prefetch(['speak', 'voices', 'history']);
    assert.deepEqual(warmed, ['speak', 'voices', 'history'], 'once per view');
    await host.prefetch(['studio']);
    assert.deepEqual(warmed, ['speak', 'voices', 'history'], 'a view already loaded is not prefetched');
    await router.navigate('voices');
    assert.deepEqual(shown(), ['view-voices'], 'a failed warm-up never blocks the visit');
  });

  it('[GS-13] every route change calls show(route) (sub and id changes too) and emits route:changed {route, prev}', async () => {
    const { router, log, events } = await setup();
    // #voices is Voices' one list and #voices/recordings another section of it.
    await router.navigate('voices');
    await router.navigate('voices/recordings', { focus: 'keep' });
    await router.navigate('reader/doc%201');
    assert.deepEqual(events, [['reader', null], ['voices', 'reader'], ['voices/recordings', 'voices'], ['reader/doc%201', 'voices/recordings']]);
    const shows = log.filter((e) => e[0] === 'show');
    assert.deepEqual(shows.slice(-3), [['show', 'voices', 'voices', 'heading'], ['show', 'voices', 'voices/recordings', 'keep'], ['show', 'reader', 'reader/doc%201', 'heading']]);
    assert.equal(log.filter((e) => e[0] === 'hide' && e[1] === 'voices').length, 1, 'a sub change does not hide the view');
  });

  it('[GS-24] html[data-immersive] follows the controller (the reading view), and is cleared on leaving', async () => {
    const { router, host } = await setup({ overrides: { reader: { controller: { immersive: (route) => !!route.id } } } });
    const changes = [];
    host.onImmersive((on) => changes.push(on));
    assert.equal(document.documentElement.hasAttribute('data-immersive'), false, 'the library is not immersive');
    await router.navigate('reader/abc');
    assert.equal(document.documentElement.hasAttribute('data-immersive'), true);
    assert.equal(host.isImmersive(), true);
    await router.navigate('speak');
    assert.equal(document.documentElement.hasAttribute('data-immersive'), false);
    assert.deepEqual(changes, [true, false]);
  });

  it('[BUG-18] a programmatic route change moves focus to the view h1, scrolls to the top, and keeps focus for sub changes', async () => {
    const { router } = await setup({ overrides: { history: { restoreScroll: true } } });
    assert.equal(document.activeElement.localName, 'body', 'the first load moves no focus');
    window.scrollTo(0, 500);
    await router.navigate('speak');
    assert.equal(document.activeElement.id, 'sp-title');
    assert.equal(window.scrollY, 0, 'scrolled to the top');
    await router.navigate('voices');
    const button = document.createElement('button');
    button.setAttribute('type', 'button');
    button.id = 'vo-sub-recordings';
    document.getElementById('view-voices').appendChild(button);
    button.focus();
    await router.navigate('voices/recordings', { focus: 'keep' });
    assert.equal(document.activeElement.id, 'vo-sub-recordings', "focus: 'keep' leaves focus alone");
    window.scrollTo(0, 300);
    await router.navigate('history');
    assert.equal(document.activeElement.id, 'hi-title');
    assert.equal(window.scrollY, 300, 'a view that restored its scroll position keeps it');
    env.setHash('#models');
    await settle(env);
    assert.equal(document.activeElement.id, 'mo-title', 'Back/Forward and links (hashchange) focus the heading too');
  });

  it('shows a quiet "Loading…" after 150 ms, and an error state with Retry when a view fails to load', async () => {
    let fail = true;
    const { router, log } = await setup({
      routes: {
        'GET /static/partials/speak.html': (req) => delay(400, partialRoute()(req)),
        'GET /static/partials/models.html': (req) => (fail ? text(500, 'boom') : partialRoute()(req)),
      },
    });
    const pending = router.navigate('speak');
    await env.clock.tickAsync(100);
    assert.equal(document.querySelector('#view-speak .view__loading'), null, 'nothing before 150 ms');
    assert.deepEqual(shown(), ['view-reader'], 'the previous view stays for a fast load (no flash)');
    await env.clock.tickAsync(100);
    assert.equal(document.querySelector('#view-speak .view__loading').textContent, 'Loading…');
    assert.deepEqual(shown(), ['view-speak'], 'past 150 ms the loading line is on screen: its section shows and the old view is gone');
    assert.deepEqual(log.filter((e) => e[0] === 'hide').map((e) => e[1]), ['reader'], 'the old view was told it is hidden');
    await env.clock.tickAsync(300);
    await pending;
    assert.equal(document.querySelector('#view-speak .view__loading'), null);
    assert.equal(log.filter((e) => e[0] === 'hide' && e[1] === 'reader').length, 1, 'hide() ran once');
    await router.navigate('models');
    const section = document.getElementById('view-models');
    assert.equal(section.hidden, false);
    assert.match(section.textContent, /This section didn't load/);
    assert.equal(document.activeElement.textContent, 'Models', 'focus still lands on a heading');
    assert.equal(log.filter((e) => e[0] === 'init' && e[1] === 'models').length, 0);
    fail = false;
    env.click(section.querySelector('[data-view-retry]'));
    await settle(env);
    assert.equal(log.filter((e) => e[0] === 'init' && e[1] === 'models').length, 1, 'Retry loads it');
    assert.match(section.innerHTML, /mo-title/);
    await router.navigate('speak');
    assert.deepEqual(shown(), ['view-speak'], 'the rest of the app keeps working');
  });

  it('sets document.title from the view (with the server version) or from the controller', async () => {
    const { router, store } = await setup({ overrides: { reader: { controller: { title: (route) => (route.id ? "Alice's Adventures" : '') } } } });
    assert.equal(document.title, 'Reader · Open Speech');
    store.app.set({ connection: { ...store.app.get().connection, version: '0.8.0' } });
    assert.equal(document.title, 'Reader · Open Speech v0.8.0');
    await router.navigate('speak');
    assert.equal(document.title, 'Speak · Open Speech v0.8.0');
    await router.navigate('reader/doc1');
    assert.equal(document.title, "Alice's Adventures · Open Speech");
  });

  it('loadPartial caches per URL and retries after a failure; reloadCurrent re-shows with focus kept', async () => {
    let calls = 0;
    const { host, log } = await setup({ routes: { 'GET /static/partials/x.html': () => { calls += 1; return calls === 1 ? text(404, 'no') : text(200, '<h2>X</h2>'); } } });
    await assert.rejects(host.loadPartial('/static/partials/x.html'), /HTTP 404/);
    assert.equal(await host.loadPartial('/static/partials/x.html'), '<h2>X</h2>');
    assert.equal(await host.loadPartial('/static/partials/x.html'), '<h2>X</h2>');
    assert.equal(calls, 2);
    await host.reloadCurrent('auth');
    assert.deepEqual(log.at(-1), ['show', 'reader', 'reader', 'keep']);
    assert.equal(host.currentView(), 'reader');
    assert.ok(host.controllerFor('reader'));
  });

  it('the newest route wins: a route superseded while its view loads is dropped (no show, no focus, no event)', async () => {
    const { router, log, events } = await setup({ routes: { 'GET /static/partials/speak.html': (req) => delay(400, partialRoute()(req)) } });
    const slow = router.navigate('speak');
    await env.clock.tickAsync(50);
    await router.navigate('history');
    assert.deepEqual(shown(), ['view-history'], 'History shows at once');
    await env.clock.tickAsync(500);
    assert.equal((await slow).path, 'speak', 'the old navigation settles');
    assert.deepEqual(shown(), ['view-history'], 'Speak, landing late, does not take over');
    assert.equal(document.activeElement.id, 'hi-title');
    assert.equal(log.filter((e) => e[0] === 'show' && e[1] === 'speak').length, 0);
    assert.equal(log.filter((e) => e[0] === 'init' && e[1] === 'speak').length, 1, 'it is loaded for next time');
    assert.deepEqual(events.map((e) => e[0]), ['reader', 'history']);
    await router.navigate('speak');
    assert.deepEqual(shown(), ['view-speak']);
  });

  it('an intent whose navigation was superseded (Open in Speak, then Back before Speak loaded) is dropped, so a later visit to Speak never applies it', async () => {
    // The History entry used to replace the listener's restored draft on an unrelated visit.
    const taken = [];
    let store = null;
    const { router } = await setup({
      routes: { 'GET /static/partials/speak.html': (req) => delay(400, partialRoute()(req)) },
      overrides: { speak: { show: () => { taken.push(store.takeIntent('speak')); } } },
    });
    store = await load('core/store.js');
    await router.navigate('history');
    store.setIntent('speak', { type: 'apply-history', entry: { id: 'h1', text: 'The History entry text.' } });
    const slow = router.navigate('speak');
    await env.clock.tickAsync(50);
    await router.navigate('history');                 // Back, before Speak had loaded
    await env.clock.tickAsync(500);
    await slow;
    assert.equal(store.takeIntent('speak'), null, 'the abandoned intent is gone');
    store.setIntent('speak', { type: 'apply-preset', id: 'p1' });
    await router.navigate('speak');                   // a navigation that completes takes its intent
    assert.deepEqual(taken.map((i) => i && i.type), ['apply-preset']);
    await router.navigate('history');
    await router.navigate('speak');                   // minutes later, from the nav
    assert.deepEqual(taken.map((i) => i && i.type), ['apply-preset', null], 'nothing is applied on the later visit');
  });

  it('a partial that never answers blocks nothing, and after 15 s its view shows the error state with Retry', async () => {
    const { router } = await setup({ routes: { 'GET /static/partials/speak.html': () => new Promise(() => {}) } });
    router.navigate('speak');
    await env.clock.tickAsync(1000);
    assert.deepEqual(shown(), ['view-speak'], 'the loading line shows');
    await router.navigate('history');
    assert.deepEqual(shown(), ['view-history'], 'History is not held up by the stalled partial');
    const again = router.navigate('speak');   // joins the load still in flight
    await env.clock.tickAsync(14000);
    await again;
    assert.match(document.getElementById('view-speak').textContent, /This section didn't load/, 'the stalled fetch timed out into the error state');
    assert.equal(env.fetch.callsTo('GET /static/partials/speak.html').length, 1);
    assert.equal(env.fetch.callsTo('GET /static/partials/speak.html')[0].signal.aborted, true, 'the stalled request was cancelled');
  });

  it("a view whose show() awaits a slow request never holds up focus, title, route:changed or the next route", async () => {
    let release;
    const { router, events } = await setup({ overrides: { speak: { show: () => new Promise((r) => { release = r; }) } } });
    await router.navigate('speak');
    assert.equal(document.activeElement.id, 'sp-title');
    assert.equal(document.title, 'Speak · Open Speech');
    assert.deepEqual(events.at(-1), ['speak', 'reader']);
    await router.navigate('history');
    assert.deepEqual(shown(), ['view-history']);
    release();
  });

  it('a view that awaits router.navigate() inside show() (a redirect) does not deadlock the router', async () => {
    let routerRef;
    const { router } = await setup({ overrides: { speak: { show: async () => { await routerRef.navigate('history', { replace: true }); } } } });
    routerRef = router;
    await router.navigate('speak');
    await settle(env);
    assert.deepEqual(shown(), ['view-history'], 'the redirect target shows');
    await router.navigate('models');
    assert.deepEqual(shown(), ['view-models'], 'and later navigation still works');
  });

  it('refreshTitle() sets the title again once a view has the data its title needs', async () => {
    let name = '';
    const { router, host } = await setup({ overrides: { reader: { controller: { title: () => name } } } });
    await router.navigate('reader/doc1');
    assert.equal(document.title, 'Reader · Open Speech');
    name = "Alice's Adventures";
    host.refreshTitle();
    assert.equal(document.title, "Alice's Adventures · Open Speech");
  });
});

