// The Reader's screenshot states are display only: a link to one must never write to the server.
// The page's requests carry the stored key, so the server's cross-site guard can't tell a link from the listener.
// ?shot=play and ?shot=miniplayer used to send POST /v1/reader/documents/<id>/render (GPU work, a render-cache write, then progress saves).
// These run the whole Reader: its view (views/reader/index.js) on the real engine and real player.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, makeDoc, progressAt, $, textOf, settle, go, load } from '../reader/helpers.js';

const KEY = 'sk-test-reader-shot';
const SEED = {
  'open-speech-api-key': KEY,
  // A usable Reader voice and a loaded Kokoro (the shell's /api/models): a start would render at once, so a
  // state that started playback would show up as a write here.
  'open-speech-reader-voice': JSON.stringify({ model: 'kokoro', voice: 'af_heart' }),
};
const DOC_SHOTS = ['opening', 'appearance', 'chapters', 'return', 'highlight', 'play', 'speed', 'sleep', 'voice', 'miniplayer'];
const LIBRARY_SHOTS = ['add', 'add-paste', 'row-menu', 'hint', 'loading', 'error'];

let ctx = null;
let engineModule = null;
afterEach(() => {
  if (engineModule) engineModule.__resetForTests();
  engineModule = null;
  ctx?.env.restore();
  ctx = null;
});

/** Boot the shell on `hash` with the real engine and the real player in the Reader. */
async function boot(hash, { docs } = {}) {
  engineModule = await load('views/reader/engine.js');
  engineModule.__resetForTests();
  const { mountPlayer } = await load('views/reader/player.js');
  ctx = await readerEnv({
    hash,
    seed: SEED,
    docs: docs || [makeDoc({ progress: progressAt(12) })],
    loadEngine: async () => engineModule.getEngine(),
    mountPlayer: (slot, engine, options) => mountPlayer(slot, engine, options),
  });
  // Let everything the state starts run out: the opening, the player's partial, frames, timers.
  for (let i = 0; i < 8; i += 1) await settle(ctx.env, 250);
  return ctx;
}

/** Every request that is not a read. */
function writes() {
  return ctx.env.fetch.calls.filter((call) => !['GET', 'HEAD'].includes(call.method)).map((call) => `${call.method} ${call.path}`);
}

/** The audio elements that play (the Reader's two and the shared one). */
function playingElements() {
  return Array.from(document.querySelectorAll('audio')).filter((el) => !el.paused).map((el) => el.id || 'audio');
}

const engine = () => engineModule.getEngine();
const miniplayer = () => $('app-miniplayer');

describe('Reader screenshot states never write', () => {
  for (const shot of [...DOC_SHOTS, ...LIBRARY_SHOTS]) {
    const hash = DOC_SHOTS.includes(shot) ? `#reader/alice?shot=${shot}` : `#reader?shot=${shot}`;
    it(`${hash} sends no POST, PUT, PATCH or DELETE, renders nothing and plays nothing, also when the page is left`, async () => {
      await boot(hash);
      assert.deepEqual(writes(), [], 'no write: no render request, no progress save, no model load');
      assert.equal(ctx.env.fetch.callsTo('* /v1/reader/documents/:id/render').length, 0, 'no render request');
      assert.equal(ctx.env.sockets.length, 0, 'no socket');
      assert.deepEqual(playingElements(), [], 'no audio element plays');
      assert.equal(engine().state.playing, false, 'the engine does not play');
      const audio = await load('core/audio.js');
      const session = audio.current();
      if (shot === 'miniplayer') assert.equal(session && session.id, 'shot:reader:alice', 'only the display-only session');
      else assert.equal(session, null, 'no audio session claimed');
      assert.ok(ctx.env.fetch.calls.some((call) => call.method === 'GET' && call.path.startsWith('/v1/reader/documents')), 'the Reader did load its data (the check is not vacuous)');
      // Leaving the page (a tab close or a reload) saves only a position that changed: none did.
      window.dispatchEvent(new Event('pagehide'));
      await settle(ctx.env, 1000);
      assert.deepEqual(writes(), [], 'no keepalive progress save either');
    });
  }

  it('the play state paints the transport as playing over the highlighted sentence, with nothing playing; its Pause shows it paused', async () => {
    await boot('#reader/alice?shot=play');
    assert.equal(ctx.store.app.get().route.path, 'reader/alice', 'it stays on the document');
    const play = $('rdp-play');
    assert.equal(play.getAttribute('aria-label'), 'Pause', 'the transport shows the playing state');
    assert.equal(play.querySelector('use').getAttribute('href'), '/static/icons/sprite.svg#i-pause');
    assert.equal(play.hasAttribute('aria-busy'), false, 'no preparing ring: nothing is being prepared');
    const current = document.querySelector('#rd-text .rd-p.is-current');
    assert.equal(current && current.getAttribute('data-block'), '12', 'the saved paragraph is the current one');
    assert.equal(document.querySelectorAll('#rd-text .rd-s.is-now').length, 1, 'its first sentence shows as spoken');
    assert.equal(textOf(document.querySelector('#rd-text .rd-s.is-now')), 'Paragraph 2 of chapter 3.');
    // Pause on the picture: nothing was playing, so it shows paused and asks for nothing.
    ctx.env.click(play);
    await settle(ctx.env, 500);
    assert.equal(play.getAttribute('aria-label'), 'Play');
    assert.deepEqual(writes(), []);
    assert.deepEqual(playingElements(), []);
    // The next Play is the listener's own: it starts real playback (a render), as on any other address.
    ctx.env.click(play);
    await settle(ctx.env, 50);
    assert.deepEqual(writes(), ['POST /v1/reader/documents/alice/render'], 'Play after the picture is a real start');
  });

  it('the play state gives way to real playback: once the listener starts and it stops, the transport says Play', async () => {
    await boot('#reader/alice?shot=play');
    assert.equal($('rdp-play').getAttribute('aria-label'), 'Pause');
    // The listener starts from a paragraph (the reading view's tap); this test server has no render route, so the
    // start fails and playback stops at once.
    engine().playFrom(13);
    await settle(ctx.env, 500);
    assert.ok(writes().includes('POST /v1/reader/documents/alice/render'), "the listener's own start (it saves the new position, then renders)");
    assert.equal(engine().state.playing, false, 'playback stopped');
    assert.equal($('rdp-play').getAttribute('aria-label'), 'Play', 'the picture does not come back over the real state');
  });

  it('the play state ends with its address: the plain document route shows the real, paused transport', async () => {
    await boot('#reader/alice?shot=play');
    assert.equal($('rdp-play').getAttribute('aria-label'), 'Pause');
    await go(ctx, 'reader', 250);
    await go(ctx, 'reader/alice', 250);
    assert.equal($('rdp-play').getAttribute('aria-label'), 'Play', 'nothing plays, so the transport says Play again');
    assert.deepEqual(writes(), []);
  });

  it('the return state shows the Back to playing pill over a paused document only on its own address', async () => {
    await boot('#reader/alice?shot=return');
    assert.equal($('rd-return').hidden, false, 'the pill shows, as if playback went on while the listener browsed');
    assert.equal($('rdp-play').getAttribute('aria-label'), 'Pause', 'and the transport agrees: it shows the playing state');
    assert.equal(engine().state.playing, false, 'with nothing playing');
    await go(ctx, 'reader', 250);
    await go(ctx, 'reader/alice', 250);
    assert.equal($('rdp-play').getAttribute('aria-label'), 'Play', 'off its address the transport shows the real state');
    // Paused, a scroll only notes the browse: following stays on and no pill offers to go back to playing.
    window.dispatchEvent(new Event('wheel'));
    await settle(ctx.env, 50);
    assert.equal(ctx.ctl.reading.follow, true, 'the reading view knows nothing plays');
    assert.equal($('rd-return').hidden, true);
    assert.deepEqual(writes(), []);
  });

  it('the miniplayer state shows a playing Reader session in the mini-player on #speak, with no audio and no request behind it', async () => {
    await boot('#reader/alice?shot=miniplayer');
    const { app } = ctx.store;
    assert.equal(app.get().route.path, 'speak', 'it shows the mini-player away from the Reader');
    assert.equal(miniplayer().hidden, false, 'the mini-player shows');
    assert.equal(miniplayer().getAttribute('data-kind'), 'long', 'as the Reader');
    assert.equal(textOf($('mp-title')), 'Chapter 3', 'the chapter of the saved position');
    assert.equal(textOf($('mp-sub')), "Alice's Adventures in Wonderland");
    assert.equal($('mp-toggle').getAttribute('aria-label'), 'Pause', 'playing');
    assert.equal($('mp-close').getAttribute('aria-label'), 'Close player');
    assert.equal($('mp-skip').hidden, false, 'with the forward skip of the Reader');
    assert.equal($('mp-open').getAttribute('aria-disabled'), null, 'and the way back to the document');
    assert.equal(app.get().player.id, 'shot:reader:alice');
    assert.notEqual(app.get().player.id, 'reader:alice', 'the engine never claimed its session');
    // Its controls change only the picture.
    ctx.env.click($('mp-toggle'));
    await settle(ctx.env, 50);
    assert.equal($('mp-toggle').getAttribute('aria-label'), 'Play');
    ctx.env.click($('mp-toggle'));
    await settle(ctx.env, 50);
    assert.equal($('mp-toggle').getAttribute('aria-label'), 'Pause');
    ctx.env.click($('mp-skip'));
    await settle(ctx.env, 50);
    assert.deepEqual(writes(), [], 'Pause, Play and the skip asked the server for nothing');
    assert.deepEqual(playingElements(), []);
    ctx.env.click($('mp-close'));
    await settle(ctx.env, 50);
    assert.equal(miniplayer().hidden, true, 'Close player takes it away');
    const audio = await load('core/audio.js');
    assert.equal(audio.current(), null);
    assert.equal(app.get().resumable, null, 'nothing waits to be resumed');
    assert.deepEqual(writes(), []);
  });

  it('the miniplayer session goes when the listener leaves #speak or another player starts; it never waits in app.resumable', async () => {
    await boot('#reader/alice?shot=miniplayer');
    const { app } = ctx.store;
    const audio = await load('core/audio.js');
    assert.equal(app.get().player.id, 'shot:reader:alice');
    // Another player starts: the picture is displaced and released, not kept as a paused Reader.
    const stops = [];
    const other = audio.claim({
      id: 'take-1', owner: 'speak', kind: 'clip', title: 'A take',
      controls: { play() {}, pause() {}, toggle() {}, stop: () => stops.push('take') },
      getState: () => ({ playing: true, waiting: false, position: 0, duration: 3, rate: 1, canSeek: false, statusText: '' }),
    });
    await settle(ctx.env, 10);
    assert.equal(app.get().player.id, 'take-1');
    assert.equal(app.get().resumable, null, 'the picture did not become a resumable Reader');
    other.release();
    await settle(ctx.env, 10);
    assert.equal(app.get().player, null);
    assert.equal(miniplayer().hidden, true);
    // Leaving #speak takes a picture still on screen away with it.
    ctx.env.restore();
    engineModule.__resetForTests();
    await boot('#reader/alice?shot=miniplayer');
    assert.equal(ctx.store.app.get().player.id, 'shot:reader:alice');
    await go(ctx, 'history', 250);
    assert.equal(ctx.store.app.get().player, null, 'gone with the address it was shown on');
    assert.equal(ctx.store.app.get().resumable, null);
    assert.deepEqual(writes(), []);
  });

  it('the miniplayer state never displaces audio that is already there: the mini-player shows that instead', async () => {
    await boot('#reader');
    const audio = await load('core/audio.js');
    const paused = [];
    audio.claim({
      id: 'take-2', owner: 'speak', kind: 'clip', title: 'A take',
      controls: { play() {}, pause: () => paused.push('take-2'), toggle() {}, stop() {} },
      getState: () => ({ playing: true, waiting: false, position: 0, duration: 3, rate: 1, canSeek: false, statusText: '' }),
    });
    await go(ctx, 'reader/alice?shot=miniplayer', 250);
    for (let i = 0; i < 4; i += 1) await settle(ctx.env, 250);
    assert.equal(ctx.store.app.get().route.path, 'speak');
    assert.equal(ctx.store.app.get().player.id, 'take-2', 'the take still plays');
    assert.deepEqual(paused, [], 'and was never paused');
    assert.deepEqual(writes(), []);
  });

  it('the miniplayer state followed in a tab that already holds the document shows Speak, not the Reader under #speak', async () => {
    await boot('#reader/alice');
    assert.equal(engine().state.docId, 'alice', 'the engine already holds the document');
    await go(ctx, 'reader/alice?shot=miniplayer', 250);
    for (let i = 0; i < 4; i += 1) await settle(ctx.env, 250);
    const { app } = ctx.store;
    assert.equal(app.get().route.path, 'speak');
    assert.equal($('view-reader').hidden, true, 'the Reader is not left on screen under #speak');
    assert.equal(document.documentElement.hasAttribute('data-immersive'), false, 'nor its immersive reading mode');
    assert.equal(app.get().player.id, 'shot:reader:alice', 'the mini-player shows the picture');
    await go(ctx, 'history', 250);
    assert.equal(app.get().player, null, 'and the picture goes with its address');
    assert.deepEqual(writes(), []);
  });

  it('the miniplayer state never takes the place of a Reader session waiting to be resumed', async () => {
    await boot('#reader');
    const audio = await load('core/audio.js');
    const { app } = ctx.store;
    const stopped = [];
    // A Reader session that a take displaced: it waits in app.resumable once the take ends.
    audio.claim({
      id: 'reader:alice', owner: 'reader', kind: 'long', title: 'Chapter 3', ephemeral: false,
      controls: { play() {}, pause() {}, toggle() {}, stop: () => stopped.push('reader') },
      getState: () => ({ playing: true, waiting: false, position: 10, duration: 600, rate: 1, canSeek: true, statusText: '' }),
    });
    const take = audio.claim({
      id: 'take-3', owner: 'speak', kind: 'clip', title: 'A take',
      controls: { play() {}, pause() {}, toggle() {}, stop() {} },
      getState: () => ({ playing: true, waiting: false, position: 0, duration: 3, rate: 1, canSeek: false, statusText: '' }),
    });
    take.release();
    await settle(ctx.env, 10);
    assert.equal(app.get().resumable && app.get().resumable.id, 'reader:alice', 'the Reader waits to be resumed');
    await go(ctx, 'reader/alice?shot=miniplayer', 250);
    for (let i = 0; i < 4; i += 1) await settle(ctx.env, 250);
    assert.notEqual(app.get().player && app.get().player.id, 'shot:reader:alice', 'no picture over the waiting Reader');
    // Another player starts and ends: the real Reader is still the one waiting.
    const next = audio.claim({
      id: 'take-4', owner: 'speak', kind: 'clip', title: 'Another take',
      controls: { play() {}, pause() {}, toggle() {}, stop() {} },
      getState: () => ({ playing: true, waiting: false, position: 0, duration: 3, rate: 1, canSeek: false, statusText: '' }),
    });
    next.release();
    await settle(ctx.env, 10);
    assert.equal(app.get().resumable && app.get().resumable.id, 'reader:alice', 'the listener can still resume the Reader');
    assert.deepEqual(stopped, [], 'it was never released');
    assert.deepEqual(writes(), []);
  });

  it('the sleep state arms its 30-minute timer only on its address: leaving it puts back the timer the listener had', async () => {
    await boot('#reader/alice?shot=sleep');
    assert.equal(engine().state.sleep.mode, 'minutes', 'the sheet shows a running timer');
    await go(ctx, 'reader', 250);
    await go(ctx, 'reader/alice', 250);
    assert.equal(engine().state.sleep.mode, 'off', 'the link left no timer behind: the next Play would not stop after 30 minutes');
    // A timer the listener set before following the link comes back.
    engine().setSleep({ chapters: 1 });
    await go(ctx, 'reader/alice?shot=sleep', 250);
    for (let i = 0; i < 2; i += 1) await settle(ctx.env, 250);
    assert.equal(engine().state.sleep.mode, 'minutes');
    await go(ctx, 'reader/alice', 250);
    assert.equal(engine().state.sleep.mode, 'chapters', "the listener's own timer is back");
    // A timer the listener picks in the sheet on the address is theirs: it stays.
    await go(ctx, 'reader/alice?shot=sleep', 250);
    for (let i = 0; i < 2; i += 1) await settle(ctx.env, 250);
    const fifteen = document.querySelector('input[name="rdp-sleep"][value="15"]');
    fifteen.checked = true;
    fifteen.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(ctx.env, 50);
    await go(ctx, 'reader/alice', 250);
    assert.equal(engine().state.sleep.mode, 'minutes');
    assert.equal(engine().state.sleep.minutes, 15, 'the 15 minutes picked on the address stay');
    assert.deepEqual(writes(), []);
  });
});
