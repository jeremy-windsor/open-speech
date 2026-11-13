import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { shellEnv, load, settle, readerRoutes } from './helpers.js';

let env;
afterEach(() => env?.restore());

async function setup() {
  // The real Reader opens #reader/doc1 (a 404 would send it back to #reader, RE-60).
  env = await shellEnv({ routes: readerRoutes() });
  const shell = await load('shell/index.js');
  await shell.boot();
  await settle(env, 5);
  const audio = await load('core/audio.js');
  const { app } = await load('core/store.js');
  const router = await load('core/router.js');
  return { audio, app, router, root: document.getElementById('app-miniplayer') };
}

function readerSession(log) {
  const state = { playing: true, waiting: false, position: 30, duration: 120, rate: 1.25 };
  return {
    id: 'reader:doc1', owner: 'reader', kind: 'long', title: 'Chapter 1 · Down the Rabbit-Hole', subtitle: "Alice's Adventures in Wonderland",
    homeRoute: 'reader/doc1',
    controls: {
      play: () => { log.push('play'); state.playing = true; },
      pause: () => { log.push('pause'); state.playing = false; },
      toggle: () => { if (state.playing) { log.push('pause'); state.playing = false; } else { log.push('play'); state.playing = true; } },
      stop: () => { log.push('stop'); state.playing = false; },
      seekBy: (s) => { log.push(`seek ${s}`); },
    },
    getState: () => ({ ...state, canSeek: true, statusText: '' }),
  };
}

const label = (id) => document.getElementById(id).getAttribute('aria-label');

describe('shell/miniplayer', () => {
  it('[BUG-06] a persistent mini-player shows the playing session on every route except its own, and never for previews', async () => {
    const { audio, router, root } = await setup();
    assert.equal(root.hidden, true);
    await router.navigate('reader/doc1');
    const log = [];
    audio.claim(readerSession(log));
    await settle(env);
    assert.equal(root.hidden, true, 'hidden on the session home route (the reading view is the player there)');
    await router.navigate('speak');
    assert.equal(root.hidden, false, 'it follows the listener to other views');
    assert.equal(document.getElementById('mp-title').textContent, 'Chapter 1 · Down the Rabbit-Hole');
    assert.equal(document.getElementById('mp-sub').textContent, "Alice's Adventures in Wonderland");
    assert.equal(label('mp-open'), "Open player: Chapter 1 · Down the Rabbit-Hole, Alice's Adventures in Wonderland");
    assert.equal(label('mp-toggle'), 'Pause');
    assert.equal(label('mp-skip'), 'Forward 30 seconds');
    assert.equal(root.style.getPropertyValue('--progress'), '0.25');
    env.click(document.getElementById('mp-open'));
    await settle(env);
    assert.equal(location.hash, '#reader/doc1');
    await router.navigate('speak');
    await audio.playClip({ id: 'preview', owner: 'preview', title: 'Preview: Heart', src: 'blob:x', ephemeral: true });
    await settle(env);
    assert.equal(root.hidden, true, 'an ephemeral preview never shows');
  });

  it('an instruction or error on the second line ("Tap play to start listening") marks the player data-status, and phones drop the artwork for it', async () => {
    // On a 390 px phone the status used to be cut to "Tap play to start liste…".
    const { audio, router, root } = await setup();
    await router.navigate('speak');
    let statusText = 'Tap play to start listening';
    const session = { ...readerSession([]), getState: () => ({ playing: false, waiting: false, position: 0, duration: 120, rate: 1, canSeek: true, statusText }) };
    const handle = audio.claim(session);
    await settle(env);
    assert.equal(document.getElementById('mp-sub').textContent, 'Tap play to start listening');
    assert.equal(root.hasAttribute('data-status'), true);
    statusText = '';
    handle.notify();
    await settle(env);
    assert.equal(document.getElementById('mp-sub').textContent, "Alice's Adventures in Wonderland");
    assert.equal(root.hasAttribute('data-status'), false, 'the artwork comes back with the subtitle');
    const css = readFileSync('src/static/css/shell.css', 'utf8');
    assert.match(css, /@media \(max-width: 479px\) \{\s*\.miniplayer\[data-status\] \.miniplayer__art \{ display: none; \}/);
  });

  it('[REV-22] every session can be closed: "Close player" stops and releases the Reader; clips say "Stop"', async () => {
    const { audio, app, router, root } = await setup();
    await router.navigate('speak');
    const log = [];
    audio.claim(readerSession(log));
    await settle(env);
    assert.equal(label('mp-close'), 'Close player');
    document.getElementById('mp-close').focus();
    env.click(document.getElementById('mp-close'));
    await settle(env);
    assert.deepEqual(log, ['stop']);
    assert.equal(app.get().player, null);
    assert.equal(app.get().resumable, null);
    assert.equal(root.hidden, true);
    assert.equal(document.activeElement.id, 'main', 'focus does not get lost with the player');
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:take', homeRoute: '' });
    await settle(env);
    assert.equal(label('mp-close'), 'Stop');
    assert.equal(document.getElementById('mp-open').getAttribute('aria-disabled'), 'true', 'no home route to open');
    env.click(document.getElementById('mp-close'));
    assert.equal(app.get().player, null);
  });

  it('a clip displaces the Reader; when it ends the paused Reader comes back in the mini-player, one tap from resuming', async () => {
    const { audio, app, router } = await setup();
    await router.navigate('history');
    const log = [];
    audio.claim(readerSession(log));
    const handle = await audio.playClip({ id: 'take:2', owner: 'speak', title: 'Take 2', src: 'blob:take2' });
    await settle(env);
    assert.deepEqual(log, ['pause']);
    assert.equal(document.getElementById('mp-title').textContent, 'Take 2');
    handle.release();
    await settle(env);
    assert.equal(app.get().player, null);
    assert.equal(document.getElementById('mp-title').textContent, 'Chapter 1 · Down the Rabbit-Hole');
    assert.equal(label('mp-toggle'), 'Play');
    env.click(document.getElementById('mp-toggle'));
    assert.deepEqual(log, ['pause', 'play'], "Play calls the Reader's controls synchronously inside the click");
  });

  it('Play/Pause and skip act on the session at once; skip shows only when the session can seek', async () => {
    const { audio } = await setup();
    const log = [];
    const handle = audio.claim(readerSession(log));
    handle.notify();
    await settle(env);
    env.click(document.getElementById('mp-toggle'));
    assert.deepEqual(log, ['pause']);
    handle.notify();
    assert.equal(label('mp-toggle'), 'Play');
    env.click(document.getElementById('mp-skip'));
    assert.deepEqual(log, ['pause', 'seek 30']);
    const session = readerSession(log);
    delete session.controls.seekBy;
    session.id = 'live';
    session.kind = 'live';
    audio.claim(session);
    await settle(env);
    assert.equal(document.getElementById('mp-skip').hidden, true);
    assert.equal(label('mp-close'), 'Stop');
  });

  it('player shortcuts: Space or k toggles, j and l skip by the intervals; never while typing', async () => {
    const { audio } = await setup();
    const log = [];
    const handle = audio.claim(readerSession(log));
    handle.notify();
    await settle(env);
    document.getElementById('main').focus();
    env.key('k');
    env.key('l');
    env.key('j');
    assert.deepEqual(log, ['pause', 'seek 30', 'seek -15']);
    const input = document.createElement('input');
    input.setAttribute('aria-label', 'Text');
    document.getElementById('view-reader').appendChild(input);
    input.focus();
    env.key('k');
    assert.deepEqual(log, ['pause', 'seek 30', 'seek -15'], 'single keys belong to the text field');
  });

  it('is hidden in immersive mode (the Reader transport is the player there)', async () => {
    const { audio, router, root } = await setup();
    await router.navigate('speak');
    audio.claim(readerSession([]));
    await settle(env);
    assert.equal(root.hidden, false);
    const host = await load('shell/view-host.js');
    const ctl = host.controllerFor('speak');
    ctl.immersive = () => true;
    await router.navigate('speak/live');
    await settle(env);
    assert.equal(document.documentElement.hasAttribute('data-immersive'), true);
    assert.equal(root.hidden, true);
  });

  it('Play/Pause calls controls.toggle() when the session has one: a stale published state never sends the wrong command', async () => {
    const { audio } = await setup();
    const log = [];
    const session = readerSession(log);
    audio.claim(session);
    await settle(env);
    session.controls.pause();   // the owner paused on its own (waiting for a part) and has not notified yet
    log.length = 0;
    env.click(document.getElementById('mp-toggle'));
    assert.deepEqual(log, ['play'], 'the owner knows it is paused: toggle plays');
    document.getElementById('main').focus();
    env.key('k');
    assert.deepEqual(log, ['play', 'pause'], 'the k shortcut toggles the same way');
  });
});

