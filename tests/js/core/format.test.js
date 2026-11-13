import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup() {
  env = installBrowserEnv();
  return importFresh('core/format.js');
}

describe('core/format', () => {
  it('[RL-07] percent = round(n / d × 100), clamped to 0–100; no progress gives null (New)', async () => {
    const f = await setup();
    assert.equal(f.percent(12, 40), 30);
    assert.equal(f.percent(1, 3), 33);
    assert.equal(f.percent(2, 3), 67);
    assert.equal(f.percent(45, 40), 100);
    assert.equal(f.percent(40, 40), 100, 'finished saves block_index = block_count');
    assert.equal(f.percent(-3, 40), 0);
    assert.equal(f.percent(0, 40), 0);
    assert.equal(f.percent(undefined, 40), null);
    assert.equal(f.percent(null, 40), null);
    assert.equal(f.percent(5, 0), null);
  });

  it('[RL-08] the length estimate at 15 chars/s: under a minute, ~18 min, ~2 h 05 m', async () => {
    const f = await setup();
    assert.equal(f.lengthLabel(0), 'under a minute');
    assert.equal(f.lengthLabel(400), 'under a minute');
    assert.equal(f.lengthLabel(18 * 60 * 15), '~18 min');
    assert.equal(f.lengthLabel(125 * 60 * 15), '~2 h 05 m');
    assert.equal(f.lengthLabel(60 * 60 * 15, 2), '~30 min', 'divided by the playback rate');
  });

  it('clock and spokenDuration', async () => {
    const f = await setup();
    assert.equal(f.clock(243), '4:03');
    assert.equal(f.clock(3723), '1:02:03');
    assert.equal(f.clock(0), '0:00');
    assert.equal(f.clock(-5), '0:00');
    assert.equal(f.clock(NaN), '0:00');
    assert.equal(f.spokenDuration(243), '4 minutes 3 seconds');
    assert.equal(f.spokenDuration(3720), '1 hour 2 minutes');
    assert.equal(f.spokenDuration(1), '1 second');
    assert.equal(f.spokenDuration(0), '0 seconds');
  });

  it('speed and spokenSpeed', async () => {
    const f = await setup();
    assert.deepEqual([1, 1.25, 1.1, 0.75, 2.5, 1.006].map(f.speed), ['1×', '1.25×', '1.1×', '0.75×', '2.5×', '1.01×']);
    assert.equal(f.spokenSpeed(1.25), '1.25 times');
  });

  it('sizeMb (today\'s formatSize) and bytes', async () => {
    const f = await setup();
    assert.equal(f.sizeMb(82), '82 MB');
    assert.equal(f.sizeMb(1200), '1.2 GB');
    assert.equal(f.sizeMb(0), '');
    assert.equal(f.sizeMb(null), '');
    assert.deepEqual([820, 14000, 1400000, 2100000000].map(f.bytes), ['820 B', '14 KB', '1.4 MB', '2.1 GB']);
  });

  it('relativeTime and absoluteTime', async () => {
    const f = await setup();
    const now = new Date(2026, 9, 22, 15, 0, 0).getTime();
    assert.equal(f.relativeTime(now - 10000, now), 'just now');
    assert.equal(f.relativeTime(now - 5 * 60000, now), '5 min ago');
    assert.equal(f.relativeTime(now - 3 * 3600000, now), '3 h ago');
    assert.equal(f.relativeTime(new Date(2026, 9, 21, 20, 0).getTime(), now), 'Yesterday');
    assert.equal(f.relativeTime(new Date(2026, 2, 3, 9, 0).toISOString(), now), 'Mar 3');
    assert.equal(f.relativeTime(new Date(2025, 2, 3, 9, 0).getTime() / 1000, now), 'Mar 3, 2025', 'seconds are accepted');
    assert.equal(f.relativeTime('', now), '');
    assert.match(f.absoluteTime(now), /2026/);
  });

  it('plural and code points', async () => {
    const f = await setup();
    assert.equal(f.plural(1, 'chapter'), '1 chapter');
    assert.equal(f.plural(3, 'chapter'), '3 chapters');
    assert.equal(f.plural(1204, 'char'), '1,204 chars');
    assert.equal(f.plural(2, 'entry', 'entries'), '2 entries');
    assert.equal(f.codePointLength('héllo'), 5);
    assert.equal(f.codePointLength('a😀b'), 3);
    assert.equal(f.sliceCodePoints('a😀b😀c', 3), 'a😀b');
    assert.equal(f.sliceCodePoints('abc', 2), 'ab');
  });
});
