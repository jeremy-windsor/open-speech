import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { FakeDocument } from '../harness/dom.js';
import { parseInto } from '../harness/html-parser.js';

let env;
afterEach(() => env?.restore());

function docWithMetas() {
  const doc = new FakeDocument();
  parseInto(doc.head, '<meta name="color-scheme" content="light dark"><meta name="theme-color" content="#F7F7F4" media="(prefers-color-scheme: light)"><meta name="theme-color" content="#0E0F13" media="(prefers-color-scheme: dark)">');
  return doc;
}

async function setup(options = {}) {
  env = installBrowserEnv({ document: docWithMetas(), ...options });
  const theme = await importFresh('core/theme.js');
  return theme;
}

const metas = () => [...document.querySelectorAll('meta[name="theme-color"]')].map((m) => m.getAttribute('content'));

describe('core/theme', () => {
  it('[TH-01] the choice persists in open-speech-theme and shows as data-theme on <html>', async () => {
    const theme = await setup();
    theme.init();
    assert.equal(theme.getTheme(), 'system');
    assert.equal(document.documentElement.hasAttribute('data-theme'), false);
    theme.setTheme('dark');
    assert.equal(env.localStorage.__data['open-speech-theme'], 'dark');
    assert.equal(document.documentElement.getAttribute('data-theme'), 'dark');
    theme.setTheme('system');
    assert.equal(env.localStorage.__data['open-speech-theme'], 'system');
    assert.equal(document.documentElement.hasAttribute('data-theme'), false);
    assert.throws(() => theme.setTheme('sepia'), TypeError);
  });

  it("[TH-01] the old UI's stored 'light' and 'dark' are honoured", async () => {
    const theme = await setup({ seed: { 'open-speech-theme': 'light' } });
    theme.init();
    assert.equal(theme.getTheme(), 'light');
    assert.equal(document.documentElement.getAttribute('data-theme'), 'light');
    assert.equal(theme.effectiveTheme(), 'light');
  });

  it('[BUG-23] by default the theme follows the OS, and listeners hear OS changes while on System', async () => {
    const theme = await setup();
    theme.init();
    assert.equal(theme.effectiveTheme(), 'light');
    const heard = [];
    theme.onThemeChange((effective, preference) => heard.push([effective, preference]));
    env.setMedia('(prefers-color-scheme: dark)', true);
    assert.equal(theme.effectiveTheme(), 'dark');
    assert.deepEqual(heard, [['dark', 'system']]);
    theme.setTheme('light');
    env.setMedia('(prefers-color-scheme: dark)', false);
    assert.deepEqual(heard, [['dark', 'system'], ['light', 'light']], 'a forced theme ignores the OS');
  });

  it('[BUG-23] both theme-color metas follow a forced theme and go back to per-scheme colours on System', async () => {
    const theme = await setup();
    theme.init();
    assert.deepEqual(metas(), ['#F7F7F4', '#0E0F13']);
    theme.setTheme('dark');
    assert.deepEqual(metas(), ['#0E0F13', '#0E0F13']);
    theme.setTheme('light');
    assert.deepEqual(metas(), ['#F7F7F4', '#F7F7F4']);
    theme.setTheme('system');
    assert.deepEqual(metas(), ['#F7F7F4', '#0E0F13']);
    assert.equal(document.querySelector('meta[name="color-scheme"]').getAttribute('content'), 'light dark');
  });

  it('a ?theme= or ?motion= query wins for this page load only (screenshots)', async () => {
    const theme = await setup({ url: 'http://127.0.0.1:8100/web?theme=dark&motion=reduce', seed: { 'open-speech-theme': 'light' } });
    theme.init();
    assert.equal(theme.getTheme(), 'dark');
    assert.equal(document.documentElement.getAttribute('data-theme'), 'dark');
    assert.equal(theme.getMotion(), 'reduce');
    assert.equal(env.localStorage.__data['open-speech-theme'], 'light', 'storage is untouched');
  });

  it('motion: setMotion stores prefs.reduceMotion and data-motion; reducedMotion() reads both', async () => {
    const theme = await setup();
    theme.init();
    assert.equal(theme.getMotion(), 'system');
    assert.equal(theme.reducedMotion(), false);
    env.setMedia('(prefers-reduced-motion: reduce)', true);
    assert.equal(theme.reducedMotion(), true);
    theme.setMotion('full');
    assert.equal(document.documentElement.getAttribute('data-motion'), 'full');
    assert.equal(theme.reducedMotion(), false);
    theme.setMotion('reduce');
    assert.equal(theme.reducedMotion(), true);
    assert.equal(JSON.parse(env.localStorage.__data['open-speech-prefs']).reduceMotion, 'reduce');
    theme.setMotion('system');
    assert.equal(document.documentElement.hasAttribute('data-motion'), false);
  });

  it('survives storage that throws', async () => {
    const theme = await setup({ storage: 'throwing' });
    assert.doesNotThrow(() => theme.init());
    assert.equal(theme.getTheme(), 'system');
    assert.doesNotThrow(() => theme.setTheme('dark'));
    assert.equal(document.documentElement.getAttribute('data-theme'), 'dark');
  });
});
