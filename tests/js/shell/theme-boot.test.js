import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import vm from 'node:vm';
import { installBrowserEnv, loadHtml, REPO_ROOT } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const SOURCE = readFileSync(resolve(REPO_ROOT, 'src/static/js/theme-boot.js'), 'utf8');

function run({ query = '', seed = {}, storage = 'memory' } = {}) {
  env = installBrowserEnv({ url: `http://127.0.0.1:8100/web${query}`, seed, storage, document: loadHtml('src/static/index.html') });
  vm.runInThisContext(SOURCE, { filename: 'theme-boot.js' });
  const root = document.documentElement;
  return {
    theme: root.getAttribute('data-theme'),
    motion: root.getAttribute('data-motion'),
    spacing: root.getAttribute('data-spacing'),
    metas: [...document.querySelectorAll('meta[name="theme-color"]')].map((m) => m.getAttribute('content')),
  };
}

describe('theme-boot.js (classic script, before first paint)', () => {
  it('follows the system by default: no data-theme, both theme-color metas per scheme', () => {
    assert.deepEqual(run(), { theme: null, motion: null, spacing: null, metas: ['#F7F7F4', '#0E0F13'] });
  });

  it('applies the stored theme (legacy light/dark values too) and the stored motion choice', () => {
    assert.deepEqual(run({ seed: { 'open-speech-theme': 'dark', 'open-speech-prefs': JSON.stringify({ reduceMotion: 'reduce' }) } }),
      { theme: 'dark', motion: 'reduce', spacing: null, metas: ['#0E0F13', '#0E0F13'] });
    env.restore();
    assert.equal(run({ seed: { 'open-speech-theme': 'light' } }).theme, 'light');
    env.restore();
    assert.equal(run({ seed: { 'open-speech-theme': 'purple' } }).theme, null, 'unknown values mean System');
  });

  it('the query overrides for this page load only: ?theme=, ?motion=, ?spacing=wcag', () => {
    const out = run({ query: '?theme=light&motion=reduce&spacing=wcag', seed: { 'open-speech-theme': 'dark' } });
    assert.deepEqual(out, { theme: 'light', motion: 'reduce', spacing: 'wcag', metas: ['#F7F7F4', '#F7F7F4'] });
    assert.equal(env.localStorage.getItem('open-speech-theme'), 'dark', 'nothing is written');
    env.restore();
    assert.equal(run({ query: '?theme=system', seed: { 'open-speech-theme': 'dark' } }).theme, null);
  });

  it('survives unavailable storage (private mode)', () => {
    assert.deepEqual(run({ storage: 'throwing', query: '?motion=full' }), { theme: null, motion: 'full', spacing: null, metas: ['#F7F7F4', '#0E0F13'] });
  });

  it('is ES5 with no imports, wrapped in try/catch', () => {
    assert.doesNotMatch(SOURCE, /\b(?:let|const|class|import|export)\s/);
    assert.doesNotMatch(SOURCE, /=>|`/);
    assert.match(SOURCE, /try\s*\{/);
  });
});
