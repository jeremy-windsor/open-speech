import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { shellEnv, load, settle } from './helpers.js';

let env;
afterEach(() => env?.restore());

async function boot(options = {}) {
  env = await shellEnv(options);
  const shell = await load('shell/index.js');
  await shell.boot();
  await settle(env, 5);
  return { shell, router: await load('core/router.js') };
}

const currentLinks = () => [...document.querySelectorAll('.nav-link[aria-current="page"]')].map((a) => `${a.closest('nav, dialog').id}:${a.getAttribute('data-view')}`);

describe('shell/nav', () => {
  it('navigation is links in two <nav aria-label="Main"> (rail and bottom bar), not tabs; the current link has aria-current="page"', async () => {
    await boot();
    const navs = [...document.querySelectorAll('nav')];
    assert.deepEqual(navs.map((n) => [n.id, n.getAttribute('aria-label')]), [['app-rail', 'Main'], ['app-bottombar', 'Main']]);
    assert.equal(document.querySelectorAll('[role="tablist"], [role="tab"]').length, 0);
    for (const link of document.querySelectorAll('nav .nav-link[data-view]')) {
      assert.equal(link.localName, 'a');
      assert.equal(link.getAttribute('href'), `#${link.getAttribute('data-view')}`);
    }
    assert.deepEqual(currentLinks(), ['app-rail:reader', 'app-bottombar:reader']);
    env.setHash('#history');
    await settle(env);
    assert.deepEqual(currentLinks(), ['app-rail:history', 'app-more-sheet:history']);
    const more = document.getElementById('nav-more');
    assert.ok(more.classList.contains('is-current'), 'More shows the current style');
    assert.equal(more.getAttribute('aria-label'), 'More, current: History');
    env.setHash('#speak');
    await settle(env);
    assert.equal(more.getAttribute('aria-label'), 'More');
    assert.equal(more.classList.contains('is-current'), false);
  });

  it('[BUG-19] arrow keys on the nav never navigate, run loaders or add history entries', async () => {
    await boot();
    const before = history.__entries.urls.length;
    const link = document.querySelector('#app-bottombar .nav-link[data-view="reader"]');
    link.focus();
    for (const key of ['ArrowRight', 'ArrowRight', 'ArrowLeft', 'End', 'Home', 'ArrowDown']) env.key(key);
    await settle(env);
    assert.equal(location.hash, '#reader');
    assert.equal(history.__entries.urls.length, before);
    assert.equal(document.activeElement, link, 'no roving focus between nav items');
    assert.deepEqual([...document.querySelectorAll('main > section.view')].filter((s) => !s.hidden).map((s) => s.id), ['view-reader'], 'no other view was shown');
    env.click(document.querySelector('#app-bottombar .nav-link[data-view="speak"]'));
    await settle(env);
    assert.equal(location.hash, '#speak', 'a click on a link navigates');
    assert.equal(history.__entries.urls.length, before + 1, 'with one history entry');
  });

  it('More opens the More sheet; its links close the sheet and route; Back returns', async () => {
    await boot();
    const more = document.getElementById('nav-more');
    more.focus();
    env.click(more);
    const sheet = document.getElementById('app-more-sheet');
    assert.equal(sheet.open, true);
    assert.equal(more.getAttribute('aria-controls'), 'app-more-sheet');
    env.click(sheet.querySelector('a[data-view="models"]'));
    await settle(env);
    assert.equal(sheet.open, false);
    assert.equal(location.hash, '#models/tts');
    assert.equal(document.activeElement.id, 'mo-title', 'focus lands on the view heading');
    history.back();
    await settle(env);
    assert.equal(location.hash, '#reader');
    assert.equal(document.getElementById('view-reader').hidden, false);
  });

  it('the theme controls: the rail button cycles System → Light → Dark; the More sheet has the segmented control', async () => {
    await boot();
    const theme = await load('core/theme.js');
    const toggle = document.getElementById('app-theme-toggle');
    assert.equal(toggle.getAttribute('aria-label'), 'Theme: System. Change theme');
    env.click(toggle);
    assert.equal(theme.getTheme(), 'light');
    assert.equal(document.documentElement.getAttribute('data-theme'), 'light');
    assert.equal(toggle.getAttribute('aria-label'), 'Theme: Light. Change theme');
    env.click(toggle);
    assert.equal(theme.getTheme(), 'dark');
    env.click(toggle);
    assert.equal(theme.getTheme(), 'system');
    env.click(document.getElementById('nav-more'));
    const dark = document.querySelector('#app-more-sheet input[value="dark"]');
    assert.equal(document.querySelector('#app-more-sheet input[value="system"]').checked, true);
    env.click(dark);
    assert.equal(theme.getTheme(), 'dark');
    assert.equal(env.localStorage.getItem('open-speech-theme'), 'dark');
  });

  it('the skip link is the first focusable element and moves focus to <main> without touching the route', async () => {
    await boot();
    const first = document.body.querySelector('a, button, input, [tabindex]');
    assert.equal(first.className, 'skip-link');
    assert.equal(first.getAttribute('href'), '#main');
    env.click(first);
    await settle(env);
    assert.equal(document.activeElement.id, 'main');
    assert.equal(location.hash, '#reader');
  });
});
