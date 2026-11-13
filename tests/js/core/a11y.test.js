import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const SHELL = '<div id="app-sr-status" class="visually-hidden" role="status" aria-live="polite"></div><div id="app-sr-alert" class="visually-hidden" role="alert"></div>';

describe('core/a11y.announce', () => {
  it('clears the polite region, then sets the text on the next frame', async () => {
    env = installBrowserEnv({ html: SHELL });
    const { announce } = await importFresh('core/a11y.js');
    const region = document.getElementById('app-sr-status');
    region.textContent = 'old';
    announce('Kokoro loaded');
    assert.equal(region.textContent, '');
    env.raf.flush();
    assert.equal(region.textContent, 'Kokoro loaded');
  });

  it('assertive messages go to #app-sr-alert', async () => {
    env = installBrowserEnv({ html: SHELL });
    const { announce } = await importFresh('core/a11y.js');
    announce('Recording failed', { assertive: true });
    env.raf.flush();
    assert.equal(document.getElementById('app-sr-alert').textContent, 'Recording failed');
    assert.equal(document.getElementById('app-sr-status').textContent, '');
  });

  it('coalesces: at most one message per region per 500 ms, and the last one wins', async () => {
    env = installBrowserEnv({ html: SHELL });
    const { announce } = await importFresh('core/a11y.js');
    const region = document.getElementById('app-sr-status');
    announce('one');
    env.raf.flush();
    announce('two');
    announce('three');
    env.raf.flush();
    assert.equal(region.textContent, 'one');
    env.clock.tick(500);
    env.raf.flush();
    assert.equal(region.textContent, 'three');
  });

  it('creates the region when the shell does not have it, and ignores empty text', async () => {
    env = installBrowserEnv();
    const { announce } = await importFresh('core/a11y.js');
    announce('   ');
    assert.equal(document.getElementById('app-sr-status'), null);
    announce('Saved');
    env.raf.flush();
    const region = document.getElementById('app-sr-status');
    assert.equal(region.getAttribute('role'), 'status');
    assert.equal(region.textContent, 'Saved');
  });
});
