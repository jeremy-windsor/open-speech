import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { FakeClock } from '../harness/env.js';

// The harness clock used to stop silently after 100 000 timers: a view whose timer re-armed itself
// passed its test, after allocating whatever each round allocated (tens of gigabytes for audio buffers).

describe('harness FakeClock', () => {
  it('a timer that keeps re-arming itself at the same time fails tick() instead of passing silently', () => {
    const clock = new FakeClock();
    let rounds = 0;
    const again = () => { rounds += 1; clock.setTimeout(again, 0); };
    clock.setTimeout(again, 0);
    assert.throws(() => clock.tick(1000), /timers fired without reaching the target time/);
    assert.equal(rounds, 100000);
  });

  it('a timer loop that keeps allocating fails tickAsync() long before it can starve the host', async () => {
    const clock = new FakeClock();
    const kept = [];
    clock.setInterval(() => { kept.push(new Float32Array(256 * 1024)); }, 10); // 1 MB per round
    await assert.rejects(clock.tickAsync(60 * 60 * 1000), /a timer loop keeps allocating/);
    assert.ok(kept.length < 1024, `stopped after ${kept.length} rounds`);
    kept.length = 0;
  });

  it('a long advance with an ordinary interval still runs every round', async () => {
    const clock = new FakeClock();
    let rounds = 0;
    clock.setInterval(() => { rounds += 1; }, 1000);
    clock.tick(60 * 60 * 1000);
    assert.equal(rounds, 3600);
    await clock.tickAsync(10 * 1000);
    assert.equal(rounds, 3610);
  });
});
