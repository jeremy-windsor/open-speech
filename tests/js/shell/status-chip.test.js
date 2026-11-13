import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { liveRegions } from '../harness/a11y.js';
import { shellEnv, load, settle } from './helpers.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = await shellEnv(options);
  const chip = await load('shell/status-chip.js');
  const { app } = await load('core/store.js');
  const status = await load('core/status.js');
  const router = await load('core/router.js');
  chip.initStatusChip();
  return { chip, app, status, router, button: document.getElementById('app-status'), rail: document.getElementById('app-status-rail') };
}

const pillOf = (button) => button.querySelector('.pill');

describe('shell/status-chip', () => {
  it('renders status.chipModel in every chip: tone, label and a full-sentence accessible name', async () => {
    const { app, button, rail } = await setup();
    assert.equal(pillOf(button).className, 'pill pill--idle');
    assert.equal(button.querySelector('.pill__label').textContent, 'Connecting…');
    app.set({ connection: { ...app.get().connection, state: 'online' }, models: { ...app.get().models, list: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' }] } });
    for (const b of [button, rail, document.getElementById('app-more-status')]) {
      assert.equal(pillOf(b).className, 'pill pill--ok');
      assert.equal(b.querySelector('.pill__label').textContent, 'Kokoro · GPU');
      assert.equal(b.getAttribute('aria-label'), 'Kokoro · GPU. Status: connected. Loaded: Kokoro on GPU. Open status.', 'the visible label comes first (WCAG 2.5.3)');
    }
    assert.equal(rail.getAttribute('data-tooltip'), 'Kokoro · GPU', 'the icon-rail chip shows its label as a tooltip');
    app.set({ connection: { ...app.get().connection, state: 'unauthorized' } });
    assert.equal(pillOf(button).className, 'pill pill--warn');
    assert.equal(button.querySelector('.pill__label').textContent, 'API key needed');
  });

  it('the elapsed counter ticks in an aria-hidden span only while a job or a recording runs; the name never ticks', async () => {
    const { status, app, button } = await setup();
    app.set({ connection: { ...app.get().connection, state: 'online' } });
    const end = status.beginJob('model:kokoro', 'Loading Kokoro…');
    const elapsed = button.querySelector('.pill__elapsed');
    assert.equal(elapsed.getAttribute('aria-hidden'), 'true');
    assert.equal(elapsed.textContent, '0 s');
    const name = button.getAttribute('aria-label');
    env.clock.tick(6000);
    assert.equal(elapsed.textContent, '6 s');
    assert.equal(button.getAttribute('aria-label'), name, 'the accessible name does not change every second');
    assert.equal(pillOf(button).className, 'pill pill--busy');
    end();
    assert.equal(elapsed.textContent, '');
    assert.equal(env.clock.pending().filter((t) => t.every).length, 0, 'no ticking timer while idle');
    status.setRecording({ owner: 'transcribe', startedAt: Date.now() });
    assert.equal(pillOf(button).className, 'pill pill--rec');
    assert.ok(button.querySelector('.pill__dot').classList.contains('rec-dot'), 'the recording dot pulses');
    env.clock.tick(42000);
    assert.equal(elapsed.textContent, '0:42');
    for (const region of liveRegions(document)) assert.equal(region.el.contains(elapsed), false, 'never inside a live region');
  });

  it('a chip opens the status sheet: server state, version, address, secure connection, loaded models and activity', async () => {
    const { app, button, status, router } = await setup();
    app.set({
      connection: { ...app.get().connection, state: 'online', version: '0.8.0', revision: 'abc1234' },
      models: { ...app.get().models, list: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' }, { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'loaded', device: 'cpu' }] },
    });
    button.focus();
    env.click(button);
    const sheet = document.getElementById('app-status-sheet');
    assert.equal(sheet.open, true);
    const body = document.getElementById('app-status-body').textContent;
    for (const text of ['Connected', 'v0.8.0', 'abc1234', 'http://127.0.0.1:8100', 'Yes', 'Kokoro', 'Text to speech', 'whisper base', 'Speech to text', 'GPU', 'CPU', 'Nothing is running.']) {
      assert.ok(body.includes(text), text);
    }
    assert.equal(document.getElementById('app-status-retry').hidden, true);
    const end = status.beginJob('model:piper', 'Loading Piper…', { detail: 'Unloads Kokoro' });
    assert.match(document.getElementById('app-status-body').textContent, /Loading Piper…\s*Unloads Kokoro/);
    end();
    env.click(document.getElementById('app-status-models'));
    await settle(env);
    assert.equal(sheet.open, false);
    assert.equal(router.current().view, 'models');
  });

  it('offline: the sheet offers Retry now; Connection settings opens Settings', async () => {
    const { app, button, router } = await setup();
    app.set({ connection: { ...app.get().connection, state: 'offline', error: "Can't reach the server." } });
    env.click(button);
    assert.equal(document.getElementById('app-status-retry').hidden, false);
    env.click(document.getElementById('app-status-retry'));
    await settle(env);
    assert.equal(env.fetch.callsTo('GET /health').length, 1);
    env.click(document.getElementById('app-status-settings'));
    await settle(env);
    assert.equal(router.current().view, 'settings');
  });
});
