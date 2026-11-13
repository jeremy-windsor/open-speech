import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, mountPlayer, showReading, $ } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

const radio = (value) => document.querySelector(`#rdp-sleep-options input[value="${value}"]`);

describe('reader-player/sleep sheet', () => {
  it('a choice applies at once: minutes count down live (in plain text) on the chip and in the sheet; the stepper sets "End of N chapters"; Fade out is a pref', async () => {
    ctx = await setup();
    const { engine, env, prefs } = ctx;
    await mountPlayer(ctx);
    engine.load(ctx.doc);
    await showReading(ctx);
    await startPlaying(ctx, { block: 1 });
    env.click($('rdp-sleep'));
    await settle(env);
    assert.equal(radio('off').checked, true);
    assert.equal(document.activeElement, radio('off'));
    assert.equal($('rdp-sleep-remaining').textContent, 'Off');
    assert.equal($('rdp-sleep-remaining').getAttribute('role'), null, 'the live countdown is not a live region');
    env.change(radio('15'), true);
    await settle(env);
    assert.equal(engine.state.sleep.mode, 'minutes');
    assert.equal($('rdp-sleep-label').textContent, '15:00');
    assert.equal($('rdp-sleep').getAttribute('aria-label'), 'Sleep timer: 15 minutes left. Change sleep timer');
    assert.equal($('rdp-sleep-remaining').textContent, 'Stops in 15:00');
    await settle(env, 700);
    assert.match(document.getElementById('app-sr-status').textContent, /Sleep timer set: 15 minutes/);
    env.click($('rdp-sleep-n-inc'));
    await settle(env);
    assert.equal($('rdp-sleep-n').textContent, '3');
    assert.equal(radio('chapters').checked, true, 'the stepper chooses End of N chapters');
    assert.equal(engine.state.sleep.mode, 'chapters');
    assert.equal(engine.state.sleep.chapters, 3);
    assert.equal($('rdp-sleep-label').textContent, 'End of 3 ch.');
    env.click($('rdp-sleep-n-dec'));
    env.click($('rdp-sleep-n-dec'));
    assert.equal($('rdp-sleep-n').textContent, '2');
    assert.equal($('rdp-sleep-n-dec').getAttribute('aria-disabled'), 'true');
    env.change($('rdp-sleep-fade'), false);
    assert.equal(prefs.get().sleepFade, false);
    env.change(radio('off'), true);
    await settle(env);
    assert.equal(engine.state.sleep.mode, 'off');
    assert.equal($('rdp-sleep-label').textContent, 'Sleep');
  });

  it('End of chapter stops before the next chapter starts, with "Sleep timer ended" and the position at that chapter', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    await mountPlayer(ctx);
    engine.load(ctx.doc);
    await showReading(ctx);
    await startPlaying(ctx, { block: 3 });
    engine.setSleep({ chapters: 1 });
    let el = playingEl();
    el.__end();   // 3 -> 4
    await settle(env);
    el = playingEl();
    loadMeta(ctx, el);
    el.__end();   // the end of chapter 1: block 5 starts chapter 2
    await settle(env);
    assert.equal(engine.state.playing, false);
    assert.equal(engine.state.status.text, 'Sleep timer ended');
    assert.equal($('rdp-status').textContent, 'Sleep timer ended');
    assert.equal(engine.state.cursor, 5);
    assert.equal(server.saved.at(-1).block_index, 5);
    assert.equal(engine.state.sleep.mode, 'off');
  });

  it('the minutes timer fades the last 10 s, holds while paused, and pauses at 0', async () => {
    ctx = await setup();
    const { engine, env, server } = ctx;
    server.override.set(2, { duration_ms: 120000, sentences: Array.from({ length: 60 }, (_, i) => ({ block: 2, index: i, start_ms: i * 2000 })) });
    const el = await startPlaying(ctx, { block: 2 });
    engine.setSleep({ minutes: 1 });
    for (let i = 1; i <= 100; i += 1) { await env.clock.tickAsync(250); el.__timeupdate(i / 4); }
    engine.pause();
    await env.clock.tickAsync(600000);
    assert.equal(engine.state.sleep.mode, 'minutes', 'paused for ten minutes: still running');
    assert.ok(engine.state.sleep.remainingMs > 30000);
    engine.play();
    await settle(env);
    for (let i = 101; i <= 280 && engine.state.playing; i += 1) {
      await env.clock.tickAsync(250);
      el.__timeupdate(i / 4);
      if (engine.state.sleep.remainingMs > 0 && engine.state.sleep.remainingMs <= 5000) assert.ok(el.volume < 0.6, 'fading');
    }
    assert.equal(engine.state.playing, false);
    assert.equal(engine.state.status.text, 'Sleep timer ended');
    assert.equal(el.volume, 1, 'the volume is put back');
  });
});
