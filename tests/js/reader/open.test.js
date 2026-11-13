// Opening documents: the delayed notice, request tickets, errors, re-opening, deep links, Back to
// library keeping playback, and the Settings › Reading appearance hand-off.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, makeDoc, progressAt, readerRoutes, $, textOf, settle, go, json, load } from './helpers.js';
import { delay, FakeMutationObserver, FakeResizeObserver } from '../harness/fakes.js';

let ctx;
afterEach(() => ctx?.env.restore());

const docGets = () => ctx.env.fetch.callsTo('GET /v1/reader/documents/:id');
const visibleH1s = () => Array.from(document.querySelectorAll('#view-reader h1')).filter((h) => {
  for (let n = h; n && n.id !== 'view-reader'; n = n.parentElement) if (n.hidden) return false;
  return true;
});

function slowDocs(docs, ms) {
  const base = readerRoutes(docs)['GET /v1/reader/documents/:id'];
  return (req) => delay(typeof ms === 'function' ? ms(req) : ms, () => base(req));
}

describe('views/reader/open', () => {
  it('[RE-59] the "Opening “T”…" notice appears only when an open takes more than 150 ms', async () => {
    const docs = [makeDoc({ progress: progressAt(12) }), makeDoc({ id: 'note', title: 'Notes for Tuesday', chapters: 1 })];
    ctx = await readerEnv({ docs, routes: { 'GET /v1/reader/documents/:id': slowDocs(docs, (req) => (req.params.id === 'note' ? 100 : 400)) } });
    const card = $('rd-opening-card');
    assert.equal($('rd-opening').getAttribute('role'), 'status', 'the notice lives in a status region');
    await ctx.router.navigate('reader/note');
    await settle(ctx.env, 120);
    assert.equal(card.hidden, true, 'a quick open never flashes the notice');
    await go(ctx, 'reader');
    await ctx.router.navigate('reader/alice');
    await settle(ctx.env, 149);
    assert.equal(card.hidden, true);
    await settle(ctx.env, 2);
    assert.equal(card.hidden, false);
    assert.equal(textOf($('rd-opening-text')), "Opening “Alice's Adventures in Wonderland”…");
    assert.equal($('rd-text').getAttribute('aria-busy'), 'true', 'the loading placeholder is under the notice');
    assert.equal(document.querySelectorAll('#rd-text [data-block]').length, 0, 'never the text of another document');
    await settle(ctx.env, 400);
    assert.equal(card.hidden, true);
    assert.equal(textOf($('rd-doc-title')), "Alice's Adventures in Wonderland");
  });

  it('[RE-59] the notice paints before the long layout: the text is laid out two frames after the answer', async () => {
    const docs = [makeDoc({ perChapter: 40 })];
    ctx = await readerEnv({ docs, routes: { 'GET /v1/reader/documents/:id': slowDocs(docs, 300) } });
    await ctx.router.navigate('reader/alice');
    await ctx.env.clock.tickAsync(200);
    assert.equal($('rd-opening-card').hidden, false);
    await ctx.env.clock.tickAsync(120);   // the answer is in
    assert.equal(document.querySelectorAll('#rd-text [data-block]').length, 0, 'not laid out yet: the notice paints first');
    await settle(ctx.env);
    await settle(ctx.env);
    assert.ok(document.querySelectorAll('#rd-text [data-block]').length > 0);
    assert.equal($('rd-opening-card').hidden, true);
  });

  it('[RE-59] a slower earlier open never replaces a later one (a ticket per open)', async () => {
    const docs = [makeDoc({ id: 'slow', title: 'Slow book' }), makeDoc({ id: 'fast', title: 'Fast book' })];
    ctx = await readerEnv({ docs, routes: { 'GET /v1/reader/documents/:id': slowDocs(docs, (req) => (req.params.id === 'slow' ? 800 : 100)) } });
    await ctx.router.navigate('reader/slow');
    await settle(ctx.env, 50);
    await ctx.router.navigate('reader/fast');
    await settle(ctx.env, 200);
    assert.equal(textOf($('rd-doc-title')), 'Fast book');
    await settle(ctx.env, 800);
    assert.equal(textOf($('rd-doc-title')), 'Fast book', 'the late answer for the first open is dropped');
    assert.deepEqual(ctx.engine.calls.filter((c) => c[0] === 'load'), [['load', 'fast']]);
    assert.equal($('rd-opening-card').hidden, true);
  });

  it('[RE-60] a 404 toasts "That document is no longer in the library" and replaces the hash with #reader', async () => {
    ctx = await readerEnv();
    const before = history.length;
    await go(ctx, 'reader/gone', 50);
    assert.equal(location.hash, '#reader');
    assert.equal(history.length, before + 1, 'the dead link was replaced, so Back and reload never reopen it');
    assert.match(textOf($('app-toasts-alert')), /That document is no longer in the library/);
    assert.equal($('rd-library').hidden, false);
    const prefs = await load('core/prefs.js');
    assert.equal(prefs.get().lastRoute, 'reader');
  });

  it('[RE-60] any other failure shows "Couldn\'t open the document" with Retry on the reading view', async () => {
    let fail = true;
    const docs = [makeDoc()];
    const base = readerRoutes(docs)['GET /v1/reader/documents/:id'];
    ctx = await readerEnv({ docs, routes: { 'GET /v1/reader/documents/:id': (req) => (fail ? json(500, { detail: 'disk error' }) : base(req)) } });
    await go(ctx, 'reader/alice', 50);
    const banner = $('rd-open-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(textOf(banner.querySelector('.banner__title')), "Couldn't open the document");
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'disk error');
    assert.equal(location.hash, '#reader/alice', 'the route stays, so Retry and reload work');
    assert.equal($('rd-layout').hidden, true, 'no empty page or document card under the banner');
    fail = false;
    ctx.env.click(banner.querySelector('[data-action="open-retry"]'));
    await settle(ctx.env, 50);
    assert.equal(banner.hidden, true);
    assert.equal($('rd-layout').hidden, false);
    assert.equal(textOf($('rd-doc-title')), "Alice's Adventures in Wonderland");
    assert.equal(docGets().length, 2);
  });

  it('[RE-60] a refused API key says "API key needed" with Open Settings, not a Retry that cannot work', async () => {
    let refuse = true;
    const docs = [makeDoc()];
    const base = readerRoutes(docs)['GET /v1/reader/documents/:id'];
    ctx = await readerEnv({ docs, routes: { 'GET /v1/reader/documents/:id': (req) => (refuse ? json(401, { detail: 'Invalid API key' }) : base(req)) } });
    await go(ctx, 'reader/alice', 50);
    const banner = $('rd-open-error');
    assert.equal(banner.hidden, false);
    assert.equal(textOf(banner.querySelector('.banner__title')), 'API key needed');
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'This document opens once the server accepts the key.',
      'the connection banner says why; this one never repeats it');
    const buttons = Array.from(banner.querySelectorAll('button')).map((b) => [b.getAttribute('data-action'), textOf(b)]);
    assert.deepEqual(buttons, [['open-settings', 'Open Settings']], 'no Retry: the key is refused locally until it changes');
    assert.equal(banner.querySelector('details'), null, "the connection banner carries the details");
    assert.equal($('rd-layout').hidden, true);
    ctx.env.click(banner.querySelector('[data-action="open-settings"]'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#settings');
    // With a working key the document opens again on the way back.
    refuse = false;
    const auth = await load('core/auth.js');
    auth.setKey('a-new-key');
    await settle(ctx.env, 50);
    history.back();
    await settle(ctx.env, 100);
    assert.equal(location.hash, '#reader/alice');
    assert.equal(banner.hidden, true);
    assert.equal(textOf($('rd-doc-title')), "Alice's Adventures in Wonderland");
  });

  it('[RE-60] a lockout after too many wrong keys is also "API key needed" with Open Settings; the countdown stays in the connection banner', async () => {
    ctx = await readerEnv({
      routes: { 'GET /v1/reader/documents/:id': () => json(429, { error: { message: 'Too many failed API key attempts' } }, { 'Retry-After': '30' }) },
    });
    await go(ctx, 'reader/alice', 50);
    const store = await load('core/store.js');
    assert.equal(store.app.get().connection.state, 'locked', 'the shell shows the lockout and its countdown');
    const banner = $('rd-open-error');
    assert.equal(banner.hidden, false);
    assert.equal(textOf(banner.querySelector('.banner__title')), 'API key needed');
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'This document opens once the server accepts the key.');
    const buttons = Array.from(banner.querySelectorAll('button')).map((b) => [b.getAttribute('data-action'), textOf(b)]);
    assert.deepEqual(buttons, [['open-settings', 'Open Settings']]);
    assert.equal(banner.querySelector('details'), null);
    assert.equal(location.hash, '#reader/alice', 'the route stays, so the document opens once the key is accepted');
  });

  it('[RE-59] the document request does not wait for the engine module: on a cold deep link both load side by side', async () => {
    let engineLoaded = false;
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    ctx = await readerEnv({
      hash: '#reader/alice',
      loadEngine: async () => { await gate; engineLoaded = true; return ctx.engine; },
    });
    assert.equal(engineLoaded, false);
    assert.equal(docGets().length, 1, 'GET /v1/reader/documents/alice went out while the engine module was still loading');
    assert.equal($('rd-reading').hidden, false);
    release();
    await settle(ctx.env, 50);
    assert.equal(textOf($('rd-doc-title')), "Alice's Adventures in Wonderland");
    assert.deepEqual(ctx.engine.calls.filter((c) => c[0] === 'load'), [['load', 'alice']], 'the engine gets the document once it is in');
    assert.equal(document.querySelector('.rd-p.is-current').getAttribute('data-block'), '12');
  });

  it('[RE-61] opening the document already open just shows it, and Resume plays it', async () => {
    ctx = await readerEnv();
    await go(ctx, 'reader/alice', 50);
    await go(ctx, 'reader', 50);
    ctx.env.click($('rd-hero').querySelector('.rd-hero__open'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader/alice');
    assert.equal(docGets().length, 1, 'no second request');
    assert.deepEqual(ctx.engine.calls.filter((c) => c[0] === 'load').length, 1);
    await go(ctx, 'reader', 50);
    ctx.env.click($('rd-hero').querySelector('[data-action="resume"]'));
    await settle(ctx.env, 50);
    assert.equal(docGets().length, 1);
    assert.deepEqual(ctx.engine.calls.slice(-2).map((c) => c[0]), ['unlock', 'play']);
  });

  it('a deep link #reader/<id> opens the document on boot: its title, one visible h1, immersive below 960 px', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    assert.equal($('rd-reading').hidden, false);
    assert.equal(document.title, "Alice's Adventures in Wonderland · Open Speech");
    assert.deepEqual(visibleH1s().map((h) => h.id), ['rd-doc-title']);
    assert.equal(document.documentElement.hasAttribute('data-immersive'), true);
    assert.equal(document.querySelector('.rd-topbar').getAttribute('data-chrome'), 'top');
    assert.equal(document.querySelector('.rd-p.is-current').getAttribute('data-block'), '12', 'it opens at the saved position');
    assert.equal(ctx.mounted.length, 1, 'the player is mounted once into the slot');
    assert.equal(ctx.mounted[0].slot, $('rd-player-slot'));
    assert.equal(typeof ctx.mounted[0].options.onOpenChapters, 'function');
    assert.equal(typeof ctx.mounted[0].options.onOpenAppearance, 'function');
    await go(ctx, 'reader');
    assert.deepEqual(visibleH1s().map((h) => h.id), ['rd-title']);
    assert.equal(document.title, 'Reader · Open Speech v0.8.0');
    assert.equal(document.documentElement.hasAttribute('data-immersive'), false);
  });

  it('the transport is measured as chrome when it arrives and when its size changes, never when its text ticks', async () => {
    const observed = [];
    const observe = FakeMutationObserver.prototype.observe;
    FakeMutationObserver.prototype.observe = function record(target, options) { observed.push({ target, options }); };
    try {
      ctx = await readerEnv({ hash: '#reader/alice' });
    } finally {
      FakeMutationObserver.prototype.observe = observe;
    }
    const slot = $('rd-player-slot');
    const slotWatch = observed.find((o) => o.target === slot);
    assert.ok(slotWatch, 'the slot is watched for the player partial going in');
    assert.equal(!!slotWatch.options.subtree, false, 'not its subtree: the elapsed and remaining times change every second');
    assert.equal(!!slotWatch.options.characterData, false);
    const watcher = FakeMutationObserver.instances.find((w) => observed.some((o) => o.target === slot) && w.callback);
    const transport = ctx.env.mount('<div class="rdp-transport" data-chrome="bottom"><span class="rdp-time">0:00</span></div>', slot);
    transport.__rect = { top: 844 - 185, height: 185 };
    await settle(ctx.env, 60);
    assert.notEqual(document.documentElement.style.getPropertyValue('--chrome-bottom'), '185px', 'nothing measures it by itself');
    for (const w of FakeMutationObserver.instances) w.callback([{ type: 'childList', target: slot }], w);
    await settle(ctx.env, 60);
    assert.equal(document.documentElement.style.getPropertyValue('--chrome-bottom'), '185px', 'measured once it is in');
    assert.ok(watcher);
    const sizes = FakeResizeObserver.instances.filter((ro) => ro.targets.has(transport));
    assert.ok(sizes.length >= 1, 'its size is watched');
    transport.__rect = { top: 844 - 230, height: 230 };   // a chip row wrapped
    for (const ro of sizes) ro.__trigger();
    await settle(ctx.env, 60);
    assert.equal(document.documentElement.style.getPropertyValue('--chrome-bottom'), '230px');
  });

  it('[RE-53] a change in the transport\'s size never pulls back a listener who scrolled away while paused', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    const scrolls = [];
    window.scrollBy = (opts) => { scrolls.push(opts); };
    const slot = $('rd-player-slot');
    const transport = ctx.env.mount('<div class="rdp-transport" data-chrome="bottom"></div>', slot);
    transport.__rect = { top: 844 - 185, height: 185 };
    document.querySelector('.rd-p.is-current').__rect = { top: 900, height: 120 };   // under the transport
    for (const w of FakeMutationObserver.instances) w.callback([{ type: 'childList', target: slot }], w);
    await settle(ctx.env, 60);
    assert.equal(scrolls.length, 1, 'just opened: the paragraph is brought clear of the transport');
    window.dispatchEvent(new Event('wheel'));   // paused: the listener browses ahead
    assert.equal(ctx.ctl.reading.follow, true, 'while paused, scrolling leaves follow on');
    assert.equal(ctx.ctl.reading.browsed, true);
    scrolls.length = 0;
    transport.__rect = { top: 844 - 230, height: 230 };
    for (const ro of FakeResizeObserver.instances.filter((r) => r.targets.has(transport))) ro.__trigger();
    await settle(ctx.env, 60);
    assert.equal(scrolls.length, 0, 'the page stays where the listener put it');
    ctx.engine.playFrom(12);
    assert.equal(ctx.ctl.reading.browsed, false, 'a jump brings the view back to the spoken sentence');
  });

  it('leaving the Reader closes its sheets; going back to the library closes the reading view\'s', async () => {
    ctx = await readerEnv();
    ctx.env.click($('rd-add'));
    await settle(ctx.env);
    assert.equal($('rd-add-dialog').open, true);
    await go(ctx, 'speak', 50);
    assert.equal(location.hash, '#speak');
    assert.equal($('rd-add-dialog').open, false, 'no Add sheet over Speak');
    await go(ctx, 'reader/alice', 50);
    ctx.env.click($('rd-appearance-btn'));
    await settle(ctx.env);
    assert.equal($('rd-appearance-sheet').open, true);
    history.back();   // the browser Back, Alt+Left or a mouse back button
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#speak');
    assert.equal($('rd-appearance-sheet').open, false);
    await go(ctx, 'reader/alice', 50);
    ctx.env.click($('rd-chapters-btn'));
    await settle(ctx.env);
    assert.equal($('rd-chapters-sheet').open, true);
    await go(ctx, 'reader', 50);
    assert.equal($('rd-chapters-sheet').open, false, 'no chapter list over the library');
  });

  it('Back to library saves progress and keeps playing; the mini-player then shows the Reader', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    const audio = await load('core/audio.js');
    ctx.engine.play();
    const state = { playing: true, waiting: false, position: 10, duration: 100, rate: 1, canSeek: true, statusText: '' };
    audio.claim({
      id: 'reader:alice', owner: 'reader', kind: 'long', title: 'Chapter 1', subtitle: "Alice's Adventures in Wonderland",
      homeRoute: 'reader/alice', controls: { play() {}, pause() {}, toggle() {}, stop() {} }, getState: () => state,
    });
    await settle(ctx.env);
    assert.equal($('app-miniplayer').hidden, true, 'the reading view is the player');
    ctx.env.click($('rd-back'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader');
    assert.ok(ctx.engine.calls.some((c) => c[0] === 'saveProgress'), 'progress is saved on the way out');
    assert.equal(ctx.engine.calls.some((c) => c[0] === 'pause' || c[0] === 'unload'), false, 'playback continues');
    assert.equal(ctx.engine.state.playing, true);
    assert.equal($('app-miniplayer').hidden, false, 'the mini-player follows the listener to the library');
  });

  it('a library reload after reading shows the finished document at 100%', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    ctx.engine.finish();
    ctx.docs[0].progress = progressAt(15);
    await go(ctx, 'reader', 50);
    const row = Array.from(document.querySelectorAll('#rd-list .rd-row')).find((li) => li.getAttribute('data-doc') === 'alice');
    assert.equal(textOf(row.querySelector('.rd-row__pct')), 'Finished, 100%');
    assert.match(textOf($('rd-hero').querySelector('.rd-hero__meta')), /100% listened$/);
  });

  it('[REV-36] the open-appearance intent opens the sheet over the last opened document', async () => {
    ctx = await readerEnv();
    await go(ctx, 'reader/note', 50);
    await go(ctx, 'settings', 50);
    ctx.store.setIntent('reader', { type: 'open-appearance' });
    await go(ctx, 'reader', 50);
    assert.equal(location.hash, '#reader/note');
    assert.equal($('rd-appearance-sheet').open, true);
  });

  it('[RL-13] after the Settings › Reading appearance hand-off, Back to library goes to the library and the browser Back to Settings', async () => {
    ctx = await readerEnv();
    await go(ctx, 'reader/note', 50);
    await go(ctx, 'settings', 50);
    const before = history.length;
    // What Settings does: the intent, then the Reader.
    ctx.store.setIntent('reader', { type: 'open-appearance' });
    await go(ctx, 'reader', 50);
    assert.equal(location.hash, '#reader/note');
    assert.equal(history.length, before + 1, 'the document replaced the library entry');
    ctx.env.click($('rd-appearance-sheet').querySelector('[data-close]'));
    await settle(ctx.env);
    ctx.env.click($('rd-back'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader', 'Back to library lands on the library, not on Settings');
    assert.equal($('rd-library').hidden, false);
    // The browser Back from a document entered this way goes to where the listener came from.
    await go(ctx, 'settings', 50);
    ctx.store.setIntent('reader', { type: 'open-appearance' });
    await go(ctx, 'reader', 50);
    assert.equal(location.hash, '#reader/note');
    history.back();
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#settings');
    // A document opened from the library still steps back to it.
    await go(ctx, 'reader', 50);
    ctx.env.click(document.querySelector('#rd-list .rd-row[data-doc="alice"] .row__main'));
    await settle(ctx.env, 50);
    const depth = history.length;
    ctx.env.click($('rd-back'));
    await settle(ctx.env, 50);
    assert.equal(location.hash, '#reader');
    assert.equal(history.length, depth, 'stepped back, no new entry');
  });

  it('[REV-36] with nothing opened yet, the intent uses the Continue-listening document; with no document, a hint', async () => {
    ctx = await readerEnv({ hash: '#settings' });
    ctx.store.setIntent('reader', { type: 'open-appearance' });
    await go(ctx, 'reader', 50);
    assert.equal(location.hash, '#reader/alice');
    assert.equal($('rd-appearance-sheet').open, true);
    ctx.env.restore();
    ctx = await readerEnv({ hash: '#settings', docs: [] });
    ctx.store.setIntent('reader', { type: 'open-appearance' });
    await go(ctx, 'reader', 50);
    assert.equal(location.hash, '#reader');
    assert.equal($('rd-hint').hidden, false);
    assert.equal(textOf($('rd-hint-msg')), 'Open a document, then tap Aa to change how it looks.');
  });

  it('a new API key reloads the library, keeping search, filter and sort (AU-09)', async () => {
    ctx = await readerEnv();
    ctx.env.input($('rd-search'), 'alice');
    ctx.env.change($('rd-sort'), 'title');
    await settle(ctx.env, 200);
    const gets = ctx.env.fetch.callsTo('GET /v1/reader/documents').length;
    await ctx.viewHost.reloadCurrent('auth');
    await settle(ctx.env, 50);
    assert.equal(ctx.env.fetch.callsTo('GET /v1/reader/documents').length, gets + 1);
    assert.equal($('rd-search').value, 'alice');
    assert.equal($('rd-sort').value, 'title');
    assert.equal(document.querySelectorAll('#rd-list .rd-row').length, 1);
  });

  it('[RE-60] the failure\'s Details read as the server\'s message and its code, never the JSON envelope; none when they would only repeat the message', async () => {
    let answer = json(500, { error: { message: 'database is locked', code: 'reader_store_error' } });
    const docs = [makeDoc()];
    ctx = await readerEnv({ docs, routes: { 'GET /v1/reader/documents/:id': () => answer } });
    await go(ctx, 'reader/alice', 50);
    const banner = $('rd-open-error');
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'database is locked');
    const details = banner.querySelector('details');
    assert.ok(details, 'a specific server code is the power user\'s handle: Details carry it');
    assert.equal(textOf(details.querySelector('summary')), 'Details');
    assert.equal(details.querySelector('pre').textContent, 'database is locked (reader_store_error)');
    assert.doesNotMatch(banner.textContent, /[{}"]/, 'no JSON anywhere in the banner');
    // The server's generic code (every plain HTTPException): Details would only repeat the message.
    answer = json(500, { error: { message: 'disk error', code: 'http_error' } });
    ctx.env.click(banner.querySelector('[data-action="open-retry"]'));
    await settle(ctx.env, 50);
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'disk error');
    // (Booleans and ids, not nodes: a failing assert would inspect the whole fake DOM through its getters.)
    assert.equal(!!banner.querySelector('details'), false);
    // Server text goes in as text only.
    answer = json(500, { error: { message: 'x', code: '<img src=x onerror=alert(1)>' } });
    ctx.env.click(banner.querySelector('[data-action="open-retry"]'));
    await settle(ctx.env, 50);
    assert.equal(!!banner.querySelector('img'), false);
    assert.equal(banner.querySelector('pre').textContent, 'x (<img src=x onerror=alert(1)>)');
  });

  it('[RE-57] below 1180 px there is one Chapters control: the transport\'s chip; the reading view says when the transport holds it, and the top bar\'s button stands in only when it can\'t (no engine, no player, a player that didn\'t load)', async () => {
    // The real player on the real engine: its transport has the chip, so the top bar's button steps aside.
    const engineModule = await load('views/reader/engine.js');
    engineModule.__resetForTests();
    const readerModule = await load('views/reader/index.js');
    ctx = await readerEnv({
      hash: '#reader/alice',
      loadEngine: async () => engineModule.getEngine(),
      mountPlayer: (slot, engine, options) => readerModule.DEFAULT_DEPS.mountPlayer(slot, engine, options),
    });
    await settle(ctx.env, 50);
    const reading = $('rd-reading');
    assert.equal(!!$('rdp-chapters'), true, 'the transport is in');
    assert.equal($('rdp-chapters').hidden, false, 'a document with chapters: the chip shows');
    assert.equal(reading.getAttribute('data-rd-transport'), 'ready', 'reader.css hides the top bar\'s button for it');
    assert.equal($('rd-chapters-btn').hidden, false, 'the button itself stays bound (reader.css decides by width)');
    ctx.env.click($('rdp-chapters'));
    await settle(ctx.env);
    assert.equal($('rd-chapters-sheet').open, true, 'the chip opens the chapters sheet');
    ctx.env.key('Escape');
    await settle(ctx.env);
    assert.equal(document.activeElement.id, 'rdp-chapters', 'and focus goes back to it');
    engineModule.__resetForTests();
    ctx.env.restore();
    // No engine: a read-only Reader, so no transport ever comes; the top bar keeps the button.
    ctx = await readerEnv({ hash: '#reader/alice', loadEngine: async () => null });
    await settle(ctx.env, 50);
    assert.equal($('rd-reading').getAttribute('data-rd-transport'), 'none');
    assert.equal(ctx.mounted.length, 0);
    ctx.env.restore();
    // The player's module didn't load.
    ctx = await readerEnv({ hash: '#reader/alice', mountPlayer: async () => { throw new Error('Failed to fetch dynamically imported module'); } });
    await settle(ctx.env, 50);
    assert.equal($('rd-reading').getAttribute('data-rd-transport'), 'none');
    ctx.env.restore();
    // The player's partial didn't load: its bar says so with Retry, and the top bar has the button until Retry works.
    ctx = await readerEnv({
      hash: '#reader/alice',
      mountPlayer: async (slot) => {
        slot.innerHTML = '<div class="rdp-transport rdp-failed" id="rdp-failed" data-chrome="bottom"><button type="button" id="rdp-retry">Retry</button></div>';
        return { ready: new Promise(() => {}) };   // the real player resolves ready only once its transport is in
      },
    });
    const slot = $('rd-player-slot');
    const notify = () => { for (const w of FakeMutationObserver.instances) w.callback([{ type: 'childList', target: slot }], w); };
    notify();
    await settle(ctx.env, 50);
    assert.equal(!!$('rdp-failed'), true);
    assert.equal($('rd-reading').getAttribute('data-rd-transport'), 'none');
    slot.innerHTML = '<div class="rdp-transport" id="rdp-transport" data-chrome="bottom"><button type="button" class="chip" id="rdp-chapters">Chapters</button></div>';
    notify();
    await settle(ctx.env, 50);
    assert.equal($('rd-reading').getAttribute('data-rd-transport'), 'ready', 'Retry brought the transport and its chip');
  });

  it('[GS-12] the Reader view\'s prefetch() warms the engine, the player and the player\'s partial on idle, so the transport arrives with the first document; it creates and starts nothing', async () => {
    ctx = await readerEnv({ hash: '#settings' });
    await settle(ctx.env, 50);
    const viewModule = await load('views/reader/index.js');
    assert.equal(typeof viewModule.default.prefetch, 'function');
    const partials = () => ctx.env.fetch.callsTo('GET /static/partials/:name').map((c) => c.params.name);
    assert.ok(partials().includes('reader-player.html'), "the shell's idle prefetch ran the Reader's own");
    assert.equal(partials().filter((n) => n === 'reader-player.html').length, 1);
    assert.equal(!!document.getElementById('rd-audio-a'), false, 'no engine was created (importing it has no side effects)');
    assert.equal(ctx.env.fetch.callsTo('GET /v1/reader/documents/:id').length, 0);
    // The player finds its partial in the cache: nothing more is fetched when it mounts.
    const viewHost = await load('shell/view-host.js');
    const player = await load('views/reader/player.js');
    await viewHost.loadPartial(player.PARTIAL);
    assert.equal(partials().filter((n) => n === 'reader-player.html').length, 1);
  });
});
