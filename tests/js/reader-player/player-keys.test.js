import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { setup, settle, playingEl, startPlaying, loadMeta, mountPlayer, showReading, load, $ } from './helpers.js';

let ctx = null;
afterEach(() => { ctx?.env.restore(); ctx = null; });

async function reading(options = {}) {
  ctx = await setup({ prefs: { snapToSentence: false }, ...options });
  const marks = Array.from({ length: 30 }, (_, i) => ({ block: 2, index: i, start_ms: i * 2000 }));
  ctx.server.override.set(2, { duration_ms: 60000, sentences: marks });
  await mountPlayer(ctx);
  ctx.engine.load(ctx.doc);
  await showReading(ctx);
  document.body.focus();
  return ctx;
}

const press = (key, init = {}, target = document.body) => ctx.env.key(key, init, target);

describe('reader-player/player keys', () => {
  it('[RE-58] in the reading view: Space and k toggle play, ←/→ move by paragraph, Shift+←/→ by sentence, j/l skip, [ and ] change the speed', async () => {
    await reading();
    const { engine, env } = ctx;
    press(' ');
    await settle(env);
    assert.equal(engine.state.playing, true, 'Space plays');
    const el = playingEl();
    loadMeta(ctx, el);
    press('k');
    assert.equal(engine.state.playing, false, 'k pauses');
    press('k');
    assert.equal(engine.state.playing, true);
    press('ArrowRight');
    await settle(env);
    assert.equal(engine.state.cursor, 1);
    press('ArrowLeft');
    await settle(env);
    assert.equal(engine.state.cursor, 0);
    engine.playFrom(2);
    await settle(env);
    const long = playingEl();
    loadMeta(ctx, long);
    long.__timeupdate(20.5);
    press('ArrowRight', { shiftKey: true });
    assert.equal(engine.state.sentence, 11);
    assert.equal(long.currentTime, 22);
    press('ArrowLeft', { shiftKey: true });
    assert.equal(engine.state.sentence, 10);
    press('l');
    assert.equal(long.currentTime, 50);
    press('j');
    assert.equal(long.currentTime, 35);
    press(']');
    assert.equal(engine.state.speed, 1.05);
    press('[');
    press('[');
    assert.equal(engine.state.speed, 0.95);
    await settle(env, 600);
    assert.match(document.getElementById('app-sr-status').textContent, /Speed 0\.95 times/);
  });

  it('[RE-58] the keys stay out of the way: not with Ctrl/Alt/Meta, not in a field (the scrubber keeps its arrows), not when a handler took the key, not under a sheet, single keys off when the setting is off, and nothing outside the reading view', async () => {
    await reading();
    const { engine, env, prefs } = ctx;
    await startPlaying(ctx, { block: 1 });
    press('ArrowRight', { ctrlKey: true });
    press('ArrowRight', { altKey: true });
    press(' ', { metaKey: true });
    await settle(env);
    assert.equal(engine.state.cursor, 1);
    assert.equal(engine.state.playing, true);
    const scrub = $('rdp-scrub');
    scrub.focus();
    press('ArrowRight', {}, scrub);
    await settle(env);
    assert.equal(engine.state.cursor, 1, 'the arrows belong to the slider there');
    const play = $('rdp-play');
    play.focus();
    press(' ', {}, play);   // Space on a focused button activates the button itself, once
    await settle(env);
    assert.equal(engine.state.playing, false);
    document.body.focus();
    const stopper = (event) => { if (event.key === 'k') event.preventDefault(); };
    document.body.addEventListener('keydown', stopper);
    press('k');
    assert.equal(engine.state.playing, false, 'a handler that prevented the key wins');
    document.body.removeEventListener('keydown', stopper);
    env.click($('rdp-speed'));
    press(' ', {}, document.activeElement);
    assert.equal(engine.state.playing, false, 'no player keys under a sheet');
    env.key('Escape');
    document.body.focus();
    prefs.set({ shortcuts: false });
    press('k');
    press(']');
    assert.equal(engine.state.playing, false);
    assert.equal(engine.state.speed, 1);
    press('ArrowRight');
    await settle(env);
    assert.equal(engine.state.cursor, 2, 'non-character keys are not single-key shortcuts');
    prefs.set({ shortcuts: true });
    const router = await load('core/router.js');
    await router.navigate('reader');
    await settle(env);
    press('ArrowRight');
    await settle(env);
    assert.equal(engine.state.cursor, 2, 'the library is not the reading view');
    const shortcuts = await load('core/shortcuts.js');
    const ids = shortcuts.list().filter((s) => s.scope === 'view:reader').map((s) => s.id);
    assert.deepEqual(ids.sort(), ['reader-back', 'reader-faster', 'reader-forward', 'reader-next-paragraph', 'reader-next-sentence', 'reader-prev-paragraph', 'reader-prev-sentence', 'reader-slower', 'reader-toggle']);
  });

  it('[RE-58] [ and ] typed with AltGr (German, French, Nordic layouts; Windows reports Ctrl+Alt) change the speed in the reading view and from another view', async () => {
    await reading();
    const { engine, env } = ctx;
    await startPlaying(ctx, { block: 1 });
    document.body.focus();
    const altGr = { ctrlKey: true, altKey: true, modifierAltGraph: true };
    press(']', altGr);
    assert.equal(engine.state.speed, 1.05, 'AltGr+9 types ] on a German keyboard');
    press('[', altGr);
    press('[', altGr);
    assert.equal(engine.state.speed, 0.95);
    const router = await load('core/router.js');
    await router.navigate('speak');
    await settle(env);
    press(']', altGr);
    assert.equal(engine.state.speed, 1, 'the player-scope keys too');
    press(']', { ctrlKey: true, altKey: true });
    assert.equal(engine.state.speed, 1, 'a real Ctrl+Alt chord changes nothing');
  });

  it('[RE-58] [ and ] belong to the Reader session (the "player (Reader)" shortcut scope): they change its speed from the library or another view while it plays or waits paused, once per press, and never while another player is the session', async () => {
    await reading();
    const { engine, env, audio } = ctx;
    await startPlaying(ctx, { block: 1 });
    document.body.focus();
    press(']');
    assert.equal(engine.state.speed, 1.05, 'in the reading view: once, not twice');
    const router = await load('core/router.js');
    await router.navigate('reader');   // Back to library: it keeps playing, the mini-player shows
    await settle(env);
    assert.equal(engine.state.playing, true);
    press(']');
    assert.equal(engine.state.speed, 1.1, 'from the library');
    await router.navigate('speak');
    await settle(env);
    press('[');
    press('[');
    assert.equal(engine.state.speed, 1, 'from Speak');
    await settle(env, 600);
    assert.match(document.getElementById('app-sr-status').textContent, /Speed 1 times/);
    engine.pause();
    press(']');
    assert.equal(engine.state.speed, 1.05, 'paused: the Reader is still the session the player keys act on');
    // A Speak take takes over: the player keys act on the take, so [ and ] leave the Reader alone.
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:take' });
    press(']');
    assert.equal(engine.state.speed, 1.05);
    const shortcuts = await load('core/shortcuts.js');
    const player = shortcuts.list().filter((s) => s.scope === 'player' && s.id.startsWith('reader-')).map((s) => [s.id, s.keys.join(' '), s.description]);
    assert.deepEqual(player, [['reader-slower-player', '[', 'Reader slower by 0.05×'], ['reader-faster-player', ']', 'Reader faster by 0.05×']], 'listed in the "?" sheet under "While audio plays"');
  });
});
