import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { json, networkError } from '../harness/fakes.js';
import { shellEnv, load, settle, MODELS } from './helpers.js';

let env;
afterEach(() => env?.restore());

const never = () => new Promise(() => {});
const toasts = () => document.querySelectorAll('#app-toasts .toast').length;

async function setup(options = {}) {
  env = await shellEnv(options);
  const shell = await load('shell/index.js');
  const { app, emit } = await load('core/store.js');
  return { shell, app, emit };
}

describe('shell boot', () => {
  it('[REV-32] the first view mounts without waiting for /health or /api/models', async () => {
    const { shell, app } = await setup({ routes: { 'GET /health': never, 'GET /api/models': never } });
    const ok = await shell.boot();
    await settle(env);
    assert.equal(ok, true);
    assert.equal(document.getElementById('view-reader').hidden, false);
    assert.equal(document.getElementById('rd-title').textContent, 'Library');
    assert.equal(app.get().connection.state, 'connecting', 'the server has not answered yet');
    assert.equal(document.getElementById('app-status').querySelector('.pill__label').textContent, 'Connecting…');
    assert.equal(env.fetch.callsTo('GET /api/models').length, 1, 'the request is on its way in the background');
  });

  it('[GS-18] boot sends one GET /api/models and one /health, fills the store, then prefetches Speak and Reader on idle', async () => {
    const { shell, app } = await setup();
    await shell.boot();
    await settle(env);
    assert.equal(env.fetch.callsTo('GET /api/models').length, 1);
    assert.equal(env.fetch.callsTo('GET /health').length, 1);
    assert.deepEqual(app.get().models.list.map((m) => m.id), MODELS.models.map((m) => m.id));
    assert.equal(app.get().models.defaults.tts, 'kokoro');
    assert.equal(app.get().connection.state, 'online');
    assert.equal(env.fetch.callsTo('GET /static/partials/speak.html').length, 0, 'nothing else before idle');
    await settle(env, 1100);
    assert.equal(env.fetch.callsTo('GET /static/partials/speak.html').length, 1, 'Speak is prefetched on idle');
    assert.equal(document.getElementById('view-speak').innerHTML, '', '…without init()');
  });

  it('[GS-18] a new API key (auth:changed) re-runs the status polling and the current view reloads, keeping focus', async () => {
    const shows = [];
    const registry = {
      reader: async () => ({ default: { id: 'reader', title: 'Reader', partial: '/static/partials/reader.html', init: () => ({ show: (route, nav) => { shows.push(nav.focus); }, hide() {} }) } }),
    };
    let refused = true;
    const { shell, app } = await setup({ routes: { 'GET /api/models': () => (refused ? json(401, { error: { message: 'Invalid API key' } }) : json(200, MODELS)) } });
    await shell.boot({ registry });
    await settle(env);
    assert.equal(app.get().connection.state, 'unauthorized');
    const auth = await load('core/auth.js');
    refused = false;
    auth.setKey('right-key');   // emits auth:changed, as Settings does
    await settle(env);
    assert.equal(env.fetch.callsTo('GET /api/models').length, 2);
    assert.equal(env.fetch.callsTo('GET /api/models')[1].headers.authorization, 'Bearer right-key');
    assert.equal(app.get().connection.state, 'online');
    assert.deepEqual(shows, ['none', 'keep'], 'the current view loads its data again without moving focus');
    const banner = document.getElementById('app-conn-banner');
    assert.equal(banner.textContent.trim(), '', 'online: the banner is empty');
    assert.equal(banner.classList.contains('visually-hidden'), true, '…and takes no space');
  });

  it('[GS-16] the server version from /health goes into the title (and the status sheet); a failing /health stays silent', async () => {
    const { shell } = await setup();
    await shell.boot();
    await settle(env);
    assert.equal(document.title, 'Reader · Open Speech v0.8.0');
    env.restore();
    const second = await setup({ routes: { 'GET /health': () => networkError(), 'GET /api/models': () => networkError() } });
    await second.shell.boot();
    await settle(env);
    assert.equal(document.title, 'Reader · Open Speech');
    assert.equal(toasts(), 0, 'no toast for it');
    assert.equal(document.getElementById('app-boot-error').hidden, true, 'and boot still succeeds');
  });

  it('[AU-10] a refused page load (401 on /api/models) shows exactly one notice: the connection banner', async () => {
    const { shell } = await setup({ routes: { 'GET /api/models': () => json(401, { error: { message: 'Invalid API key' } }) } });
    await shell.boot();
    await settle(env);
    const banner = document.getElementById('app-conn-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'status');
    assert.match(banner.textContent, /This server needs an API key\./);
    assert.match(banner.textContent, /Settings › Connection/);
    assert.equal(banner.textContent.match(/This server needs an API key/g).length, 1, 'said once');
    assert.deepEqual([...banner.querySelectorAll('svg[data-icon]')].filter((svg) => !svg.hasAttribute('hidden')).map((svg) => svg.getAttribute('data-icon')), ['key'], 'one static icon shows (resolved at page load, so it also shows offline)');
    assert.equal(banner.querySelector('[data-banner-action="settings"]').textContent.trim(), 'Open Settings');
    assert.equal(toasts(), 0, 'never repeated as a toast');
    assert.equal(document.getElementById('app-status').querySelector('.pill__label').textContent, 'API key needed');
    env.click(banner.querySelector('[data-banner-action="settings"]'));
    await settle(env);
    assert.equal(location.hash, '#settings');
  });

  it('[AU-10] a refused key: the banner title says the key was not accepted (never that one is needed), each sentence said once', async () => {
    const { shell } = await setup({
      seed: { 'open-speech-api-key': 'wrong-key' },
      routes: { 'GET /api/models': () => json(401, { error: { message: 'Invalid API key' } }) },
    });
    await shell.boot();
    await settle(env);
    const banner = document.getElementById('app-conn-banner');
    assert.equal(banner.querySelector('.banner__title').textContent, "The server didn't accept the API key.");
    assert.equal(banner.querySelector('.banner__msg').textContent, 'Check it in Settings › Connection.');
    assert.doesNotMatch(banner.textContent, /needs an API key/, 'a key was sent: the title must not contradict the line under it');
    assert.equal(banner.textContent.match(/didn't accept/g).length, 1, 'said once');
    assert.equal(banner.querySelector('[data-banner-action="settings"]').textContent.trim(), 'Open Settings');
    assert.equal(toasts(), 0);
  });

  it('[AU-10] the refusal copy: first sentence as the title, the rest under it, for a missing, refused or unsendable key', async () => {
    await setup();
    const banners = await load('shell/banners.js');
    const auth = await load('core/auth.js');
    const copy = (error) => { const m = banners.bannerModel({ state: 'unauthorized', error }); return [m.title, m.message]; };
    assert.deepEqual(copy(auth.refusalMessage()), ['This server needs an API key.', 'Enter it in Settings › Connection.'], 'no key');
    auth.setKey('wrong');
    assert.deepEqual(copy(auth.refusalMessage()), ["The server didn't accept the API key.", 'Check it in Settings › Connection.'], 'a refused key');
    auth.setKey('curly“key');
    const [title, message] = copy(auth.refusalMessage());
    assert.match(title, /^The API key contains a character that can't be sent/);
    assert.equal(message, 'Check it in Settings › Connection.');
    assert.deepEqual(copy(''), ['This server needs an API key.', 'Enter it in Settings › Connection.'], 'no message at all');
    assert.deepEqual(copy("The server didn't accept the API key."), ["The server didn't accept the API key.", 'Check it in Settings › Connection.'], 'one sentence');
    auth.setKey('');
  });

  it('[AU-10] a lockout (429 about the API key) shows the locked banner with an aria-hidden countdown', async () => {
    const { shell } = await setup({ routes: { 'GET /api/models': () => json(429, { error: { message: 'Too many failed API key attempts' } }, { 'Retry-After': '60' }) } });
    await shell.boot();
    await settle(env);
    const banner = document.getElementById('app-conn-banner');
    assert.match(banner.textContent, /Too many wrong API keys from this device\./);
    const countdown = banner.querySelector('.banner__countdown');
    assert.equal(countdown.closest('[aria-hidden="true"]').textContent.startsWith('The server will accept requests again in'), true, 'the visible countdown sentence is hidden from screen readers');
    assert.equal(banner.querySelector('.visually-hidden').textContent, 'The server will accept requests again shortly.');
    assert.equal(countdown.textContent, '60 s.');
    const before = banner.querySelector('.banner__body').innerHTML.replace(/<span class="banner__countdown[^<]*<\/span>/, '');
    env.clock.tick(5000);
    assert.equal(countdown.textContent, '55 s.');
    assert.equal(banner.querySelector('.banner__body').innerHTML.replace(/<span class="banner__countdown[^<]*<\/span>/, ''), before, 'only the hidden counter ticks');
    assert.equal(toasts(), 0);
  });

  it('offline: "Can\'t reach Open Speech." with Retry now', async () => {
    const { shell } = await setup({ routes: { 'GET /health': () => networkError() } });
    await shell.boot();
    await settle(env);
    const banner = document.getElementById('app-conn-banner');
    assert.match(banner.textContent, /Can't reach Open Speech\./);
    env.click(banner.querySelector('[data-banner-action="retry"]'));
    await settle(env);
    assert.equal(env.fetch.callsTo('GET /health').length, 2);
  });

  it('[GS-17] boot order: storage, theme and the API key come first; a failure shows the boot error banner, never a blank page or a toast', async () => {
    const { shell, app } = await setup({ seed: { 'open-speech-theme': 'dark', 'open-speech-api-key': 'secret' } });
    const seen = {};
    const ok = await shell.boot({
      startRouter: async () => {
        seen.theme = document.documentElement.getAttribute('data-theme');
        seen.hasKey = app.get().auth.hasKey;
        seen.prefs = 'shortcuts' in app.get().prefs;
        seen.navReady = document.getElementById('app-theme-toggle').getAttribute('aria-label');
        throw new Error('router exploded');
      },
    });
    await settle(env);
    assert.equal(ok, false);
    assert.deepEqual(seen, { theme: 'dark', hasKey: true, prefs: true, navReady: 'Theme: Dark. Change theme' });
    const box = document.getElementById('app-boot-error');
    assert.equal(box.hidden, false);
    assert.equal(box.getAttribute('role'), 'alert');
    assert.match(box.textContent, /Open Speech didn't start: router exploded/);
    assert.equal(box.querySelector('#app-boot-reload').textContent.trim(), 'Reload');
    env.click(box.querySelector('#app-boot-reload'));
    assert.equal(window.__reloads, 1);
    assert.equal(toasts(), 0, 'no "Init failed" toast');
    assert.equal(env.fetch.callsTo('GET /api/models')[0].headers.authorization, 'Bearer secret', 'the stored key is used from the first request');
  });

  it('global shortcuts: "?" opens the shortcuts sheet; g then a letter goes to a view', async () => {
    const { shell } = await setup();
    await shell.boot();
    await settle(env);
    document.getElementById('main').focus();
    env.key('?', { shiftKey: true });
    const sheet = document.getElementById('app-shortcuts-sheet');
    assert.equal(sheet.open, true);
    assert.match(sheet.textContent, /Go to Speak/);
    assert.match(sheet.textContent, /Single-key shortcuts can be turned off/);
    env.key('Escape');
    assert.equal(sheet.open, false);
    document.getElementById('main').focus();
    env.key('g');
    env.key('s');
    await settle(env);
    assert.equal(location.hash, '#speak');
    env.key('g');
    env.key(',');
    await settle(env);
    assert.equal(location.hash, '#settings');
  });

  it('screenshot hook: #route?shot=app-more opens the More sheet over that view', async () => {
    env = await shellEnv({ url: 'http://127.0.0.1:8100/web#history?shot=app-more' });
    const shell = await load('shell/index.js');
    await shell.boot();
    await settle(env);
    assert.equal(document.getElementById('view-history').hidden, false);
    assert.equal(document.getElementById('app-more-sheet').open, true);
    const prefs = await load('core/prefs.js');
    assert.equal(prefs.get().lastRoute, 'history', 'a shot state is never remembered');
  });

  it('the connection banner stays in the accessibility tree while online (empty, visually hidden, never display:none), so its first message is announced', async () => {
    const { shell, app } = await setup();
    await shell.boot();
    await settle(env);
    const banner = document.getElementById('app-conn-banner');
    assert.equal(banner.hidden, false, 'not display:none');
    assert.equal(banner.getAttribute('role'), 'status');
    assert.equal(banner.classList.contains('visually-hidden'), true);
    assert.equal(banner.textContent.trim(), '');
    app.set({ connection: { ...app.get().connection, state: 'offline', error: "Can't reach the server." } });
    assert.equal(banner.classList.contains('visually-hidden'), false, 'shown');
    assert.equal(banner.getAttribute('data-state'), 'offline');
    assert.match(banner.textContent, /Can't reach Open Speech\./);
    app.set({ connection: { ...app.get().connection, state: 'online', error: '' } });
    assert.equal(banner.classList.contains('visually-hidden'), true);
    assert.equal(banner.textContent.trim(), '');
    assert.equal(banner.hidden, false);
  });
});

