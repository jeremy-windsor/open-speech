import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const SHELL = '<main id="main"></main><div id="app-audio-host" hidden><audio id="app-audio" preload="auto"></audio></div>';

async function setup(options = {}) {
  env = installBrowserEnv({ html: SHELL, ...options });
  const audio = await importFresh('core/audio.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  const prefs = await import('../../../src/static/js/core/prefs.js');
  prefs.load();
  return { audio, app };
}

/** A fake owner session (the Reader engine, a Live session) that records control calls. */
function fakeSession(id, kind, extra = {}) {
  const log = [];
  const state = { playing: true, waiting: false, position: 10, duration: 100, rate: 1.25 };
  const session = {
    id, owner: extra.owner || id.split(':')[0], kind, title: extra.title || id, subtitle: extra.subtitle || '', homeRoute: extra.homeRoute || '',
    controls: {
      play: () => { log.push('play'); state.playing = true; },
      pause: () => { log.push('pause'); state.playing = false; },
      toggle: () => log.push('toggle'),
      stop: (reason) => { log.push(reason ? `stop:${reason}` : 'stop'); state.playing = false; },
      ...(extra.seek ? { seekBy: (s) => log.push(`seekBy:${s}`), seekTo: (s) => log.push(`seekTo:${s}`) } : {}),
      ...(extra.tracks ? { prev: () => log.push('prev'), next: () => log.push('next') } : {}),
    },
    getState: () => ({ ...state }),
    ...(extra.mediaMetadata ? { mediaMetadata: extra.mediaMetadata } : {}),
    ephemeral: !!extra.ephemeral,
  };
  return { session, log, state };
}

const reader = (extra = {}) => fakeSession('reader:d1', 'long', { owner: 'reader', homeRoute: 'reader/d1', title: 'Chapter 3', subtitle: 'The Hobbit', seek: true, tracks: true, ...extra });

describe('core/audio: displacement', () => {
  it('[GS-22][BUG-05] a take pauses the Reader, which becomes resumable; Reader play pauses and releases a take', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    const take = await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', homeRoute: 'speak', src: 'blob:x/1' });
    assert.deepEqual(r.log, ['pause']);
    assert.equal(app.get().player.id, 'take:1');
    assert.equal(app.get().resumable.id, 'reader:d1');
    const el = audio.element();
    assert.equal(el.paused, false);
    audio.claim(r.session);
    assert.equal(el.paused, true, 'the take paused');
    assert.equal(take.active, false, 'and was released');
    assert.equal(app.get().player.id, 'reader:d1');
    assert.equal(app.get().resumable, null);
  });

  it('[REV-02] Live is stopped (not paused) when another player starts, with the reason', async () => {
    const { audio, app } = await setup();
    const live = fakeSession('live', 'live', { owner: 'live', homeRoute: 'speak/live' });
    audio.claim(live.session);
    audio.claim(reader().session);
    assert.deepEqual(live.log, ['stop:Another player started']);
    assert.equal(app.get().resumable, null, 'Live is never resumable');
  });

  it('[BUG-05] Live pauses the Reader (resumable) instead of playing over it', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    audio.claim(fakeSession('live', 'live', { owner: 'live' }).session);
    assert.deepEqual(r.log, ['pause']);
    assert.equal(app.get().resumable.id, 'reader:d1');
    assert.equal(app.get().player.id, 'live');
  });

  it('[REV-23] when the clip ends the mini-player offers the Reader again, paused; nothing auto-resumes', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    let ended = 0;
    await audio.playClip({ id: 'take:2', owner: 'speak', title: 'Take 2', homeRoute: 'speak', src: 'blob:x/2', onEnded: () => { ended += 1; } });
    audio.element().__end();
    assert.equal(ended, 1);
    assert.equal(app.get().player, null);
    assert.equal(app.get().resumable.id, 'reader:d1');
    assert.deepEqual(r.log, ['pause'], 'not resumed');
    app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    const mini = audio.miniPlayerState(app.get());
    assert.deepEqual([mini.visible, mini.source, mini.session.id], [true, 'resumable', 'reader:d1']);
    assert.equal(env.mediaSession.handlers.play !== undefined, true, 'the lock screen can resume the Reader');
    env.mediaSession.action('play');
    assert.deepEqual(r.log, ['pause', 'play']);
    audio.claim(r.session);
    assert.equal(app.get().player.id, 'reader:d1');
    assert.equal(app.get().resumable, null);
  });

  it('[GS-21] leaving the Reader view does not stop it; the mini-player shows off its home route', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    const route = (path) => app.set({ route: { view: path.split('/')[0], sub: '', id: '', path, shot: '' } });
    route('reader/d1');
    assert.equal(audio.miniPlayerState(app.get()).visible, false, 'hidden on its home route');
    route('voices/named');
    assert.deepEqual(r.log, [], 'navigation never pauses');
    assert.equal(app.get().player.playing, true);
    const mini = audio.miniPlayerState(app.get());
    assert.deepEqual([mini.visible, mini.source, mini.session.title, mini.session.subtitle], [true, 'player', 'Chapter 3', 'The Hobbit']);
    assert.equal(audio.miniPlayerState(app.get(), { immersive: true }).visible, false);
    assert.equal(mini.session.has.seekBy, true);
  });

  it('claiming the current id again returns its handle and changes nothing else', async () => {
    const { audio } = await setup();
    const r = reader();
    const h1 = audio.claim(r.session);
    const h2 = audio.claim(r.session);
    assert.equal(h1, h2);
    assert.deepEqual(r.log, []);
  });
});

describe('core/audio: Media Session', () => {
  it('installs metadata and handlers for the current session and clears them on release', async () => {
    const { audio } = await setup();
    const r = reader({ mediaMetadata: () => ({ title: 'Chapter 3', artist: 'J. R. R. Tolkien', album: 'The Hobbit', artwork: [{ src: 'data:image/png;base64,AA', sizes: '512x512', type: 'image/png' }] }) });
    const handle = audio.claim(r.session);
    const ms = env.mediaSession;
    assert.equal(ms.metadata.title, 'Chapter 3');
    assert.equal(ms.metadata.artist, 'J. R. R. Tolkien');
    assert.equal(ms.metadata.album, 'The Hobbit');
    assert.deepEqual(Object.keys(ms.handlers).sort(), ['nexttrack', 'pause', 'play', 'previoustrack', 'seekbackward', 'seekforward', 'seekto', 'stop']);
    ms.action('seekbackward', {});
    ms.action('seekforward', { seekOffset: 10 });
    ms.action('seekto', { seekTime: 42 });
    ms.action('previoustrack');
    assert.deepEqual(r.log, ['seekBy:-15', 'seekBy:10', 'seekTo:42', 'prev']);
    assert.equal(ms.playbackState, 'playing');
    handle.release();
    assert.deepEqual(ms.handlers, {});
    assert.equal(ms.metadata, null);
    assert.equal(ms.playbackState, 'none');
  });

  it('notify() mirrors playbackState and sets the position only when it is valid', async () => {
    const { audio } = await setup();
    const r = reader();
    const handle = audio.claim(r.session);
    const ms = env.mediaSession;
    assert.deepEqual(ms.positionStates.at(-1), { duration: 100, position: 10, playbackRate: 1.25 });
    const before = ms.positionStates.length;
    r.state.position = 150;
    handle.notify();
    r.state.position = 5;
    r.state.duration = 0;
    handle.notify();
    assert.equal(ms.positionStates.length, before, 'invalid states are skipped');
    r.state.playing = false;
    r.state.duration = 100;
    handle.notify();
    assert.equal(ms.playbackState, 'paused');
  });

  it('a session without seek or track controls gets no such handlers; the lock-screen stop closes it', async () => {
    const { audio, app } = await setup();
    const live = fakeSession('live', 'live', { owner: 'live' });
    audio.claim(live.session);
    assert.deepEqual(Object.keys(env.mediaSession.handlers).sort(), ['pause', 'play', 'stop']);
    env.mediaSession.action('stop');
    assert.deepEqual(live.log, ['stop']);
    assert.equal(app.get().player, null);
  });

  it('handle.update(meta) changes the title and subtitle in app.player and on the lock screen', async () => {
    const { audio, app } = await setup();
    const r = reader();
    const handle = audio.claim(r.session);
    handle.update({ title: 'Chapter 4', subtitle: 'The Hobbit', homeRoute: 'reader/d1' });
    assert.deepEqual([app.get().player.title, app.get().player.subtitle], ['Chapter 4', 'The Hobbit']);
    assert.equal(env.mediaSession.metadata.title, 'Chapter 4');
    assert.equal(env.mediaSession.metadata.artist, 'The Hobbit');
    handle.release();
    handle.update({ title: 'ignored' });
    assert.equal(app.get().player, null, 'a released handle changes nothing');
  });

  it('ephemeral previews get no Media Session and no mini-player', async () => {
    const { audio, app } = await setup();
    await audio.playClip({ id: 'preview', owner: 'preview', title: 'Heart', src: 'blob:p/1', ephemeral: true });
    assert.equal(env.mediaSession.metadata, null);
    assert.equal(audio.miniPlayerState(app.get()).visible, false);
  });
});

describe('core/audio: the shared element', () => {
  it('[REV-20] prime() inside the click unlocks #app-audio, so the later non-gesture play() of the result works', async () => {
    const { audio, app } = await setup({ playPolicy: 'gesture' });
    env.withGesture(() => audio.prime());
    await env.flush();
    const el = audio.element();
    assert.match(el.__plays[0].src, /^data:audio\/wav;base64,/);
    assert.equal(el.paused, true, 'the silent clip is paused again');
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', homeRoute: 'speak', src: 'blob:t/1' });
    assert.equal(el.paused, false);
    assert.equal(app.get().player.statusText, '');
  });

  it('[REV-20] without prime() a refused autoplay leaves the take paused with "Tap play to start listening"', async () => {
    const { audio, app } = await setup({ playPolicy: 'gesture' });
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', homeRoute: 'speak', src: 'blob:t/1' });
    assert.equal(audio.element().paused, true);
    assert.equal(app.get().player.statusText, 'Tap play to start listening');
    assert.equal(app.get().player.id, 'take:1', 'the session stays so the mini-player can play it');
  });

  it('a pending prime() never pauses the real clip that replaced it', async () => {
    const { audio } = await setup({ playPolicy: 'pending' });
    const el = audio.element();
    env.withGesture(() => audio.prime());
    el.__playPolicy = 'resolve';
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:t/1' });
    await env.flush();
    assert.equal(el.paused, false);
  });

  it('the same id with new audio is a fresh session: the old one leaves no listeners and its onEnded never runs', async () => {
    const { audio, app } = await setup();
    const el = audio.element();
    const ended = [];
    const first = await audio.playClip({ id: 'preview', owner: 'preview', title: 'Heart', src: 'blob:p/1', ephemeral: true, onEnded: () => ended.push('heart') });
    const second = await audio.playClip({ id: 'preview', owner: 'preview', title: 'Bella', src: 'blob:p/2', ephemeral: true, onEnded: () => ended.push('bella') });
    assert.notEqual(first, second);
    assert.equal(first.active, false);
    assert.equal(app.get().player.title, 'Bella');
    el.__end();
    assert.deepEqual(ended, ['bella'], 'only the current clip ends');
    for (const type of ['ended', 'play', 'pause', 'timeupdate', 'error']) {
      assert.equal(el.__listenerCount(type), 0, `no ${type} listener left behind`);
    }
    const replay = await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:t/1' });
    el.pause();
    assert.equal(await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:t/1' }), replay, 'the same id and src resumes');
    assert.equal(el.paused, false);
  });

  it('prime() ends a clip playing on the shared element (its source is replaced) but keeps the Reader', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    let ended = 0;
    const take = await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', homeRoute: 'speak', src: 'blob:t/1', onEnded: () => { ended += 1; } });
    env.withGesture(() => audio.prime());
    await env.flush();
    assert.equal(take.active, false);
    assert.equal(app.get().player, null);
    assert.equal(app.get().resumable.id, 'reader:d1', 'the Reader stays resumable');
    audio.element().__end();
    assert.equal(ended, 0, 'the silent clip ending is not the take ending');
    assert.deepEqual(r.log, ['pause']);
  });

  it('the shared element has preservesPitch and playClip mirrors its events into app.player', async () => {
    const { audio, app } = await setup();
    const handle = await audio.playClip({ id: 'mix:1', owner: 'studio', title: 'Mix', subtitle: 'Composer', homeRoute: 'studio/composer', src: 'blob:m/1' });
    const el = audio.element();
    assert.equal(el.preservesPitch, true);
    assert.equal(el.id, 'app-audio');
    el.__loadMetadata(30);
    el.__timeupdate(12);
    handle.notify();
    assert.equal(app.get().player.position, 12);
    assert.equal(app.get().player.duration, 30);
    el.pause();
    assert.equal(app.get().player.playing, false);
    audio.current().controls.play();
    assert.equal(app.get().player.playing, true);
  });

  it('[REV-02] otherPlayerStarted (a native <audio controls> started) displaces like a claim: Live stops, the Reader stays resumable, a take is released', async () => {
    const { audio, app } = await setup();
    const live = fakeSession('live', 'live', { owner: 'live', homeRoute: 'speak/live' });
    audio.claim(live.session);
    audio.otherPlayerStarted();
    assert.deepEqual(live.log, ['stop:Another player started'], 'stopped, never paused');
    assert.equal(app.get().player, null);
    assert.equal(app.get().resumable, null, 'Live is never resumable');

    const r = reader();
    audio.claim(r.session);
    audio.otherPlayerStarted();
    assert.deepEqual(r.log, ['pause']);
    assert.equal(app.get().player, null);
    assert.equal(app.get().resumable.id, 'reader:d1');
    assert.equal(env.mediaSession.metadata.title, 'Chapter 3', 'the lock screen offers the paused Reader again');
    assert.equal(env.mediaSession.playbackState, 'paused');

    const take = await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', homeRoute: 'speak', src: 'blob:x/1' });
    audio.otherPlayerStarted();
    assert.equal(audio.element().paused, true, 'the take paused');
    assert.equal(take.active, false, 'and was released');
    assert.equal(app.get().resumable.id, 'reader:d1');
    assert.deepEqual(r.log, ['pause'], 'nothing auto-resumes');
    audio.otherPlayerStarted();   // nothing current: a no-op
    assert.equal(app.get().player, null);
  });

  it('pauseAll pauses the current session (before the microphone opens)', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    audio.pauseAll();
    assert.deepEqual(r.log, ['pause']);
    assert.equal(app.get().player.playing, false);
  });

  it('closeCurrent stops and releases the current session, or the resumable Reader when nothing else plays', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:t/1' });
    audio.closeCurrent();
    assert.equal(app.get().player, null);
    assert.equal(app.get().resumable.id, 'reader:d1');
    audio.closeCurrent();
    assert.deepEqual(r.log, ['pause', 'stop']);
    assert.equal(app.get().resumable, null);
  });

  it('canPlay asks the shared element', async () => {
    const { audio } = await setup();
    assert.equal(audio.canPlay('audio/mpeg'), true);
    assert.equal(audio.canPlay('audio/x-unknown'), false);
  });

  it('nothing in the coordinator routes media through an AudioContext', async () => {
    const { audio } = await setup();
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:t/1' });
    assert.equal(env.audioContexts.length, 0);
  });
});

describe('core/audio: release callbacks and the capture guard (foundation repair)', () => {
  it('playClip onRelease reports every way a clip ends: ended, displaced, closed, replaced, released', async () => {
    const { audio } = await setup();
    const reasons = [];
    const el = document.getElementById('app-audio');
    const play = (id, src) => audio.playClip({ id, owner: 'speak', title: id, src, onRelease: (r) => reasons.push(`${id}:${r}`), onEnded: () => reasons.push(`${id}:onEnded`) });
    await play('take:1', 'blob:a');
    el.__end();
    await play('take:2', 'blob:b');
    audio.claim(reader().session);                    // the Reader displaces it
    await play('take:3', 'blob:c');
    audio.closeCurrent();                              // the mini-player's Stop
    await play('preview', 'blob:d');
    await play('preview', 'blob:e');                   // the next preview replaces it
    const h = await play('take:4', 'blob:f');
    h.release();
    assert.deepEqual(reasons, ['take:1:onEnded', 'take:1:ended', 'take:2:displaced', 'take:3:closed', 'preview:replaced', 'preview:displaced', 'take:4:released']);
  });

  it('while the microphone is open nothing plays on the shared element: playClip rejects and play() refuses', async () => {
    const { audio, app } = await setup();
    const el = document.getElementById('app-audio');
    const handle = await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:a' });
    el.pause();
    app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    await assert.rejects(audio.playClip({ id: 'take:2', owner: 'speak', title: 'Take 2', src: 'blob:b' }), { name: 'CaptureActiveError', message: 'Stop recording to listen.' });
    const plays = el.__plays.length;
    await audio.current().controls.play();
    assert.equal(el.__plays.length, plays, 'the mini-player Play does not start it either');
    assert.equal(app.get().player.statusText, 'Stop recording to listen');
    app.set({ recording: null });
    await audio.current().controls.play();
    assert.equal(el.__plays.length, plays + 1);
    assert.equal(handle.active, true);
  });
});

describe('core/audio: a clip that fails mid-play, prime() during a capture', () => {
  it("a clip whose media fails mid-play keeps its session with \"Couldn't play\"; the next Play loads the source again and resumes where it stopped", async () => {
    const { audio, app } = await setup();
    const el = audio.element();
    const reasons = [];
    const handle = await audio.playClip({ id: 'studio:c1', owner: 'studio', title: 'Render', homeRoute: 'studio', src: '/api/conversations/c1/audio?v=1', onRelease: (r) => reasons.push(r) });
    el.__loadMetadata(120);
    el.__timeupdate(42);
    el.__fail(2);                                   // MEDIA_ERR_NETWORK: the server restarted mid-stream
    assert.equal(app.get().player.statusText, "Couldn't play");
    el.dispatchEvent(new Event('waiting'));        // the buffer ran out: nothing more can arrive
    assert.equal(el.paused, true, 'a stalled failed clip stops, so the mini-player offers Play');
    assert.equal(app.get().player.playing, false);
    assert.equal(app.get().player.statusText, "Couldn't play", 'not "Preparing audio…" forever');
    const plays = el.__plays.length;
    audio.current().controls.toggle();              // Studio's Play, the mini-player's Play
    assert.equal(el.__plays.length, plays + 1);
    assert.equal(el.error, null, 'the source was loaded again (play() alone never reloads an errored element)');
    assert.equal(el.src, 'http://127.0.0.1:8100/api/conversations/c1/audio?v=1');
    assert.equal(el.currentTime, 42, 'it continues where it stopped');
    assert.equal(app.get().player.statusText, '');
    el.__loadMetadata(120);
    assert.equal(el.currentTime, 42);
    assert.equal(el.paused, false);
    assert.equal(handle.active, true, 'the same session');
    assert.deepEqual(reasons, [], 'the owner never saw a release');
  });

  it('a clip that fails again after the reload says so again; Play with the same id and src (the owner playing it anew) reloads too', async () => {
    const { audio, app } = await setup();
    const el = audio.element();
    const handle = await audio.playClip({ id: 'composer:m1', owner: 'studio', title: 'Mix', src: '/api/composer/render/m1/audio' });
    el.__loadMetadata(60);
    el.__timeupdate(10);
    el.pause();
    el.__fail(2);
    assert.equal(app.get().player.statusText, "Couldn't play");
    audio.current().controls.play();
    el.__fail(4);
    assert.equal(app.get().player.statusText, "Couldn't play", 'a second failure is reported, nothing loops');
    const plays = el.__plays.length;
    const again = await audio.playClip({ id: 'composer:m1', owner: 'studio', title: 'Mix', src: '/api/composer/render/m1/audio' });
    assert.equal(again, handle);
    assert.equal(el.error, null);
    assert.equal(el.__plays.length, plays + 1);
    assert.equal(el.paused, false);
  });

  it('a media error while the buffered audio still plays reports it but lets the buffer play out', async () => {
    const { audio, app } = await setup();
    const el = audio.element();
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:t/1' });
    el.__loadMetadata(30);
    el.__fail(2);
    assert.equal(el.paused, false, 'readyState is still HAVE_ENOUGH_DATA: the buffer plays on');
    assert.equal(app.get().player.statusText, "Couldn't play");
    audio.current().controls.toggle();
    assert.equal(el.paused, true, 'the toggle still pauses while it plays');
  });

  it('[REV-40] prime() while the microphone is open does nothing: the paused clip stays in the mini-player and no silence plays during the capture', async () => {
    const { audio, app } = await setup();
    const el = audio.element();
    const reasons = [];
    await audio.playClip({ id: 'studio:c1', owner: 'studio', title: 'Render', homeRoute: 'studio', src: 'blob:r/1', onRelease: (r) => reasons.push(r) });
    audio.pauseAll();                               // what mic.open() does
    app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    const plays = el.__plays.length;
    env.withGesture(() => audio.prime());           // Generate in Speak while Transcribe records
    await env.flush();
    assert.deepEqual(reasons, [], 'the paused render is not released');
    assert.equal(app.get().player.id, 'studio:c1');
    assert.equal(el.__plays.length, plays, 'no silent clip played during the capture');
    assert.equal(el.src, 'blob:r/1');
    app.set({ recording: null });
    env.withGesture(() => audio.prime());
    assert.deepEqual(reasons, ['replaced'], 'after the capture prime() works as before');
  });
});

describe('core/audio: lock screen after a re-offer, clip length, page audio session', () => {
  it('[RE-47][RE-50] closing the resumable Reader a finished clip re-offered on the lock screen clears the lock screen (handlers, metadata, state none)', async () => {
    const { audio, app } = await setup();
    const r = reader();
    audio.claim(r.session);
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', homeRoute: 'speak', src: 'blob:t/1' });
    audio.element().__end();
    const ms = env.mediaSession;
    assert.equal(typeof ms.handlers.nexttrack, 'function', 'the displaced Reader is offered again');
    assert.equal(ms.metadata.title, 'Chapter 3');
    assert.equal(ms.playbackState, 'paused', 'offered again, paused');
    audio.closeCurrent();   // the mini-player's Close: there is no current session, so the resumable Reader closes
    assert.deepEqual(r.log, ['pause', 'stop']);
    assert.equal(app.get().resumable, null);
    assert.deepEqual(ms.handlers, {}, 'no Reader handlers left on the lock screen');
    assert.equal(ms.metadata, null);
    assert.equal(ms.playbackState, 'none');
  });

  it("[RE-47][RE-50] the lock screen's Stop on the re-offered Reader clears it too; the Reader's own release() does the same", async () => {
    const { audio } = await setup();
    const r = reader();
    const handle = audio.claim(r.session);
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:t/1' });
    audio.closeCurrent();   // the take is closed; the Reader is re-offered
    env.mediaSession.action('stop');
    assert.deepEqual(env.mediaSession.handlers, {});
    assert.equal(env.mediaSession.playbackState, 'none');
    // Again, with the Reader engine releasing its own (resumable, not active) handle.
    const r2 = reader();
    const h2 = audio.claim(r2.session);
    await audio.playClip({ id: 'take:2', owner: 'speak', title: 'Take 2', src: 'blob:t/2' });
    audio.element().__end();
    assert.equal(h2.active, false);
    assert.equal(typeof env.mediaSession.handlers.play, 'function');
    h2.release();
    assert.deepEqual(env.mediaSession.handlers, {});
    assert.equal(env.mediaSession.metadata, null);
    assert.equal(handle.active, false);
  });

  it("[RE-47] releasing the displaced Reader while a clip plays leaves the clip's lock screen alone", async () => {
    const { audio, app } = await setup();
    const r = reader();
    const h = audio.claim(r.session);
    await audio.playClip({ id: 'mix:1', owner: 'studio', title: 'Mix 1', subtitle: 'Composer', src: 'blob:m/1' });
    h.release();
    assert.equal(app.get().resumable, null);
    assert.equal(env.mediaSession.metadata.title, 'Mix 1', "the clip still owns the lock screen");
    assert.equal(typeof env.mediaSession.handlers.pause, 'function');
    assert.equal(env.mediaSession.handlers.nexttrack, undefined);
  });

  it('[RE-48][RE-50] while re-offered, the Reader keeps its lock-screen metadata, state and position current', async () => {
    const { audio } = await setup();
    const r = reader();
    const handle = audio.claim(r.session);
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:t/1' });
    audio.element().__end();
    r.state.playing = false;
    r.state.position = 42;
    handle.update({ title: 'Chapter 4' });
    assert.equal(env.mediaSession.metadata.title, 'Chapter 4');
    assert.equal(env.mediaSession.playbackState, 'paused');
    assert.deepEqual(env.mediaSession.positionStates.at(-1), { duration: 100, position: 42, playbackRate: 1.25 });
  });

  it('[BUG-06] a clip reports its length once its metadata loads, before anything plays (a refused autoplay waits paused)', async () => {
    const { audio, app } = await setup({ playPolicy: 'NotAllowedError' });
    await audio.playClip({ id: 'mix:1', owner: 'studio', title: 'Mix', homeRoute: 'studio/composer', src: 'blob:m/1' });
    const el = audio.element();
    assert.deepEqual([app.get().player.statusText, app.get().player.duration], ['Tap play to start listening', 0]);
    el.__loadMetadata(8);
    assert.equal(app.get().player.duration, 8, 'loadedmetadata reaches app.player without a play, pause or timeupdate');
    assert.equal(app.get().player.canSeek, true);
    assert.equal(app.get().player.playing, false);
    assert.deepEqual(env.mediaSession.positionStates.at(-1), { duration: 8, position: 0, playbackRate: 1 });
    el.duration = 9.5;
    el.dispatchEvent(new Event('durationchange'));
    assert.equal(app.get().player.duration, 9.5, 'a duration that changes (a stream that ends) is followed too');
    audio.closeCurrent();
    for (const type of ['loadedmetadata', 'durationchange']) assert.equal(el.__listenerCount(type), 0, `no ${type} listener left behind`);
  });

  it('without navigator.audioSession (every browser but Safari 17+) nothing changes and the capture hold is a no-op', async () => {
    const { audio } = await setup();
    assert.equal(audio.audioSessionType(), '');
    audio.claim(reader().session);
    const release = audio.holdCaptureAudioSession();
    assert.equal(typeof release, 'function');
    release();
    assert.equal(navigator.audioSession, undefined);
  });

  it("Safari's audio session: a non-ephemeral session makes the page a 'playback' page (Live is audible with the ringer off); a preview alone changes nothing", async () => {
    const { audio } = await setup({ audioSession: true });
    await audio.playClip({ id: 'preview', owner: 'preview', title: 'Heart', src: 'blob:p/1', ephemeral: true });
    assert.equal(audio.audioSessionType(), 'auto', 'an ephemeral preview is not a reason to take over the audio session');
    audio.claim(fakeSession('live', 'live', { owner: 'live', homeRoute: 'speak/live' }).session);
    assert.equal(env.audioSession.type, 'playback');
    audio.claim(reader().session);
    assert.deepEqual(env.audioSession.history, ['playback'], 'set once, not on every claim');
  });

  it("Safari's audio session: a capture holds 'play-and-record' and gives the earlier type back; a claim during it is kept for afterwards", async () => {
    const { audio } = await setup({ audioSession: true });
    const first = audio.holdCaptureAudioSession();
    assert.equal(env.audioSession.type, 'play-and-record');
    first();
    assert.equal(env.audioSession.type, 'auto', 'nothing had played: back to auto');
    first();
    assert.equal(env.audioSession.type, 'auto', 'the release runs once');
    audio.claim(reader().session);
    assert.equal(env.audioSession.type, 'playback');
    const second = audio.holdCaptureAudioSession();
    assert.equal(env.audioSession.type, 'play-and-record');
    second();
    assert.equal(env.audioSession.type, 'playback', 'back to playback after the capture');
    const third = audio.holdCaptureAudioSession();
    audio.claim(fakeSession('live', 'live', { owner: 'live' }).session);
    assert.equal(env.audioSession.type, 'play-and-record', 'a claim never breaks a running capture');
    third();
    assert.equal(env.audioSession.type, 'playback');
  });
});
