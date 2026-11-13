import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv(options);
  return importFresh('core/perf.js');
}

describe('core/perf', () => {
  it('record computes rtf = renderMs / 1000 / durationS and stores it per model', async () => {
    const perf = await setup();
    const entry = perf.record('kokoro', { ttfaMs: 140.4, renderMs: 400, durationS: 10 });
    assert.equal(entry.rtf, 0.04);
    assert.equal(entry.ttfaMs, 140);
    assert.equal(entry.at, Date.now());
    assert.deepEqual(perf.forModel('kokoro'), entry);
    assert.equal(perf.forModel('piper/x'), null);
    assert.deepEqual(JSON.parse(env.localStorage.__data['open-speech-perf']).kokoro.rtf, 0.04);
  });

  it('label and the slow tag', async () => {
    const perf = await setup();
    assert.equal(perf.label({ ttfaMs: 140, rtf: 0.04 }), 'First audio 140 ms · RTF 0.04');
    assert.equal(perf.label({ rtf: 1.5 }), 'RTF 1.50');
    assert.equal(perf.label(null), '');
    assert.equal(perf.SLOW_RTF, 1);
    assert.equal(perf.isSlow({ rtf: 1 }), true);
    assert.equal(perf.isSlow({ rtf: 0.2 }), false);
  });

  it('keeps the 50 most recent models', async () => {
    const perf = await setup();
    for (let i = 0; i < 55; i += 1) {
      env.clock.tick(1000);
      perf.record(`m${i}`, { renderMs: 100, durationS: 1 });
    }
    const stored = JSON.parse(env.localStorage.__data['open-speech-perf']);
    assert.equal(Object.keys(stored).length, 50);
    assert.equal(stored.m0, undefined);
    assert.ok(stored.m54);
  });

  it('survives unavailable storage', async () => {
    const perf = await setup({ storage: 'throwing' });
    assert.doesNotThrow(() => perf.record('kokoro', { renderMs: 1, durationS: 1 }));
    assert.equal(perf.forModel('kokoro'), null);
  });
});
