/**
 * The shell: wires the navs, sheets, status chip, connection banner, mini-player, chrome metrics, tooltips
 * and global shortcuts, and runs the boot sequence:
 *   1. prefs.load(), theme.init(), auth.init()
 *   2. init(): the shell
 *   3. status.start(), NOT awaited: /health and /api/models fill the store in the background
 *   4. router.start(): the first view mounts without waiting for the server (REV-32)
 *   5. on idle, prefetch the Speak and Reader views (and Home, where a tap on the brand goes)
 * A failure in steps 1–4 shows the boot error banner, never a blank page (GS-17).
 *
 * Screenshot hook: a route's ?shot=<state> also drives these shell states, after the view shows:
 *   app-more, app-status, app-shortcuts, app-voice-picker, app-miniplayer, app-toasts.
 * @module shell/index
 */
import * as prefs from '../core/prefs.js';
import * as theme from '../core/theme.js';
import * as auth from '../core/auth.js';
import * as status from '../core/status.js';
import * as router from '../core/router.js';
import * as audio from '../core/audio.js';
import { app, on as onEvent } from '../core/store.js';
import { register } from '../core/shortcuts.js';
import { toast } from '../core/toast.js';
import { encodeWav } from '../core/pcm.js';
import { initTooltips } from '../components/tooltip.js';
import { openVoicePicker } from '../components/voice-picker.js';
import { closeAllMenus } from '../components/menu.js';
import * as viewHost from './view-host.js';
import { initNav } from './nav.js';
import { initMoreSheet, openMore } from './more-sheet.js';
import { openShortcuts } from './shortcuts-sheet.js';
import { initStatusChip, openStatusSheet } from './status-chip.js';
import { initBanners, showBootError } from './banners.js';
import { initMiniplayer, refresh as refreshMiniplayer } from './miniplayer.js';
import { initChrome, refresh as refreshChrome } from './chrome.js';

export const SHELL_SHOTS = Object.freeze(['app-more', 'app-status', 'app-shortcuts', 'app-voice-picker', 'app-miniplayer', 'app-toasts']);
// g a: Talk, to the AI assistant.
const GO_KEYS = { g: 'home', r: 'reader', s: 'speak', t: 'transcribe', a: 'talk', v: 'voices', d: 'studio', h: 'history', m: 'models', ',': 'settings' };
const GO_NAMES = { home: 'Home', reader: 'Reader', speak: 'Speak', transcribe: 'Transcribe', talk: 'Talk', voices: 'Voices', studio: 'Studio', history: 'History', models: 'Models', settings: 'Settings' };

let st = null;

function registerShortcuts() {
  const offs = [];
  offs.push(register({
    id: 'shortcuts-sheet', keys: ['?'], scope: 'global', singleKey: true, description: 'Show keyboard shortcuts',
    handler: () => openShortcuts(document.activeElement),
  }));
  for (const [key, view] of Object.entries(GO_KEYS)) {
    offs.push(register({
      id: `go-${view}`, keys: [`g ${key}`], scope: 'global', singleKey: true, description: `Go to ${GO_NAMES[view]}`,
      handler: () => router.navigate(view),
    }));
  }
  return offs;
}

function demoClip() {
  const rate = 22050;
  const samples = new Float32Array(rate * 3);
  for (let i = 0; i < samples.length; i += 1) samples[i] = 0.05 * Math.sin((2 * Math.PI * 220 * i) / rate) * Math.min(1, i / 2000, (samples.length - i) / 2000);
  return URL.createObjectURL(encodeWav(samples, rate));
}

/**
 * Apply a shell screenshot state (?shot=…). Unknown states are ignored (views handle their own).
 * @param {string} shot
 */
export async function applyShot(shot) {
  if (!SHELL_SHOTS.includes(shot)) return false;
  if (shot === 'app-more') openMore(document.getElementById('nav-more'));
  else if (shot === 'app-status') openStatusSheet(document.getElementById('app-status'));
  else if (shot === 'app-shortcuts') openShortcuts();
  else if (shot === 'app-voice-picker') {
    const model = app.get().models.defaults.tts || 'kokoro';
    openVoicePicker({ model, value: 'af_heart', include: { named: true, recordings: true, voices: true } }).catch(() => {});
  } else if (shot === 'app-miniplayer') {
    await audio.playClip({
      id: 'shot:miniplayer', owner: 'speak', title: 'Chapter 3 · A Caucus-Race', subtitle: "Alice's Adventures in Wonderland",
      homeRoute: '', src: demoClip(),
    }).catch(() => {});
  } else if (shot === 'app-toasts') {
    toast('Preset saved.', { kind: 'success' });
    toast("Couldn't load the model. Kokoro reverted to available.", { kind: 'error', action: { label: 'Retry', run: () => {} } });
  }
  return true;
}

async function onRoute(route, prev, nav) {
  closeAllMenus();
  const shown = await viewHost.show(route, prev, nav);
  if (!shown) return;   // a newer route superseded this one while its view loaded
  refreshChrome();
  if (route.shot) await applyShot(route.shot);
}

/**
 * Wire the shell (boot step 2). Safe to call once per page.
 * @param {{registry?: Record<string, () => Promise<object>>}} [options] registry: view module loaders (tests)
 */
export function init({ registry } = {}) {
  destroy();
  st = { offs: [] };
  viewHost.initViewHost({ registry });
  const parts = [
    initNav({ openMore }),
    initMoreSheet({ openShortcuts }),
    initStatusChip(),
    initBanners(),
    initMiniplayer({ isImmersive: viewHost.isImmersive, onChange: refreshChrome }),
    initChrome(),
    initTooltips(document),
  ];
  for (const part of parts) st.offs.push(() => part.destroy());
  st.offs.push(viewHost.onImmersive(() => { refreshMiniplayer(); refreshChrome(); }));
  st.offs.push(...registerShortcuts());
  // A new API key: re-run the status polling and reload the current view's data (AU-09).
  st.offs.push(onEvent('auth:changed', () => {
    status.start();
    viewHost.reloadCurrent('auth');
  }));
  st.offs.push(app.select((s) => s.player, () => refreshChrome()));
}

function idle(fn) {
  if (typeof requestIdleCallback === 'function') requestIdleCallback(() => fn(), { timeout: 2000 });
  else setTimeout(fn, 1000);
}

/**
 * The boot sequence. main.js calls it once.
 * @param {{registry?: object, startRouter?: (onRoute: Function) => Promise<object>}} [options] for tests
 * @returns {Promise<boolean>} false when boot failed (the boot error banner shows)
 */
export async function boot({ registry, startRouter = router.start } = {}) {
  const fail = (error) => {
    if (globalThis.console) globalThis.console.error(error);
    showBootError(error);
    return false;
  };
  try {
    prefs.load();
    theme.init();
    auth.init();
  } catch (error) {
    return fail(error);
  }
  try {
    init({ registry });
  } catch (error) {
    return fail(error);
  }
  try {
    status.start();   // not awaited: the first view never waits for /health or /api/models
  } catch (error) {
    return fail(error);
  }
  try {
    await startRouter(onRoute);
  } catch (error) {
    return fail(error);
  }
  idle(() => { viewHost.prefetch(['speak', 'reader', 'home']); });
  return true;
}

/** Undo init(). */
export function destroy() {
  if (!st) return;
  for (const off of st.offs.reverse()) {
    try { off(); } catch { /* already gone */ }
  }
  st = null;
}

/** Test hook. */
export function __resetForTests() {
  destroy();
}
