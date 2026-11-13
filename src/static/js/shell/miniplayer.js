/**
 * The persistent mini-player (#app-miniplayer). It shows the current audio
 * session (or the displaced Reader, paused) whenever the session is not ephemeral, the page isn't its home
 * route and the view isn't immersive. Every session can be closed from here: "Stop" for clips and Live,
 * "Close player" for the Reader, which pauses, saves its progress and releases the session (REV-22).
 * Play/Pause calls the session's controls synchronously inside the click (iOS).
 * @module shell/miniplayer
 */
import { html, render, safeUrl } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { app } from '../core/store.js';
import * as audio from '../core/audio.js';
import { navigate } from '../core/router.js';
import { register } from '../core/shortcuts.js';
import { applyTile, monogram } from '../core/tiles.js';

let st = null;

function skip() {
  const prefs = app.get().prefs || {};
  return { back: Number(prefs.skipBack) || 15, forward: Number(prefs.skipForward) || 30 };
}

/**
 * PURE: the mini-player's labels for a session's public state.
 * @param {object} session app.player or app.resumable
 * @param {{skipForward?: number}} [options]
 */
export function labels(session, { skipForward = 30 } = {}) {
  const s = session || {};
  const subtitle = s.waiting ? (s.statusText || 'Preparing audio…') : (s.statusText || s.subtitle || '');
  return {
    open: `Open player: ${[s.title, s.subtitle].filter(Boolean).join(', ')}`.replace(/: $/, ''),
    toggle: s.playing ? 'Pause' : 'Play',
    skip: `Forward ${skipForward} seconds`,
    close: s.kind === 'long' ? 'Close player' : 'Stop',
    subtitle,
    progress: s.duration > 0 ? Math.max(0, Math.min(1, s.position / s.duration)) : 0,
  };
}

function controlsFor(source) {
  const session = source === 'resumable' ? audio.resumableSession() : audio.current();
  return session ? session.controls || {} : {};
}

function setIcon(button, name) {
  const use = button.querySelector('use');
  const href = `/static/icons/sprite.svg#i-${name}`;
  if (use && use.getAttribute('href') === href) return;
  render(button, html`${icon(name)}${button.id === 'mp-skip' ? html`<span class="tbtn__num" aria-hidden="true">${skip().forward}</span>` : ''}`);
}

function setLabel(button, text) {
  if (button.getAttribute('aria-label') !== text) button.setAttribute('aria-label', text);
  if (button.getAttribute('data-tooltip') !== text) button.setAttribute('data-tooltip', text);
}

function paint() {
  if (!st || !st.root) return;
  const view = audio.miniPlayerState(app.get(), { immersive: st.immersive() });
  const { root, el } = st;
  const wasVisible = !root.hidden;
  if (!view.visible) {
    if (wasVisible) {
      const hadFocus = root.contains(document.activeElement);
      root.hidden = true;
      if (hadFocus) {
        const main = document.getElementById('main');
        if (main) main.focus({ preventScroll: true });
      }
      st.onChange();
    }
    st.source = '';
    return;
  }
  const s = view.session;
  st.source = view.source;
  const text = labels(s, { skipForward: skip().forward });
  if (el.title.textContent !== s.title) el.title.textContent = s.title || 'Now playing';
  if (el.sub.textContent !== text.subtitle) el.sub.textContent = text.subtitle;
  setLabel(el.open, text.open);
  if (s.homeRoute) el.open.removeAttribute('aria-disabled');
  else el.open.setAttribute('aria-disabled', 'true');
  const artKey = `${s.id}|${s.artwork}`;
  if (st.artKey !== artKey) {
    st.artKey = artKey;
    // Artwork is a generated tile (a PNG data URL) or an object URL; anything else falls back to the monogram.
    if (/^(?:data:image\/(?:png|jpeg|webp);base64,|blob:)/.test(s.artwork || '')) render(el.art, html`<img src="${safeUrl(s.artwork, { image: true })}" alt="">`);
    else {
      render(el.art, html`${monogram(s.title || s.id)}`);
      applyTile(el.art, s.id);
    }
  }
  setIcon(el.toggle, s.playing ? 'pause' : 'play');
  setLabel(el.toggle, text.toggle);
  if (s.waiting && s.playing) el.toggle.setAttribute('aria-busy', 'true');
  else el.toggle.removeAttribute('aria-busy');
  el.skip.hidden = !(s.has && s.has.seekBy);
  if (!el.skip.hidden) {
    setLabel(el.skip, text.skip);
    const num = el.skip.querySelector('.tbtn__num');
    if (num && num.textContent !== String(skip().forward)) num.textContent = String(skip().forward);
  }
  setLabel(el.close, text.close);
  root.style.setProperty('--progress', String(Math.round(text.progress * 1000) / 1000));
  root.setAttribute('data-kind', s.kind || '');
  // An instruction or an error on the second line: phones drop the artwork so it fits (shell.css).
  if (s.statusText) root.setAttribute('data-status', '');
  else root.removeAttribute('data-status');
  if (!wasVisible) {
    root.hidden = false;
    st.onChange();
  }
}

/** controls.toggle() when the session has it (its owner knows the real state), else play/pause. */
function toggleSession(c, session) {
  if (typeof c.toggle === 'function') c.toggle();
  else if (session && session.playing) { if (c.pause) c.pause(); } else if (c.play) c.play();
}

function onToggle() {
  if (!st || !st.source) return;
  const c = controlsFor(st.source);
  const session = st.source === 'resumable' ? app.get().resumable : app.get().player;
  // Synchronously inside the click: iOS only lets a user gesture start playback.
  toggleSession(c, session);
}

function onSkip(seconds) {
  if (!st || !st.source) return;
  const c = controlsFor(st.source);
  if (typeof c.seekBy === 'function') c.seekBy(seconds);
}

function onClose() {
  audio.closeCurrent();
}

function onOpen() {
  if (!st) return;
  const view = audio.miniPlayerState(app.get(), { immersive: st.immersive() });
  if (view.session && view.session.homeRoute) navigate(view.session.homeRoute);
}

/**
 * Start the mini-player.
 * @param {{isImmersive?: () => boolean, onChange?: () => void}} [options] onChange runs when it shows or
 *   hides (the chrome metrics re-measure)
 * @returns {{refresh: () => void, destroy: () => void}}
 */
export function initMiniplayer({ isImmersive = () => false, onChange = () => {} } = {}) {
  destroy();
  const root = document.getElementById('app-miniplayer');
  st = { root, el: null, source: '', artKey: '', immersive: isImmersive, onChange, offs: [] };
  if (!root) return { refresh: paint, destroy };
  st.el = {
    open: root.querySelector('#mp-open'), art: root.querySelector('#mp-art'), title: root.querySelector('#mp-title'),
    sub: root.querySelector('#mp-sub'), toggle: root.querySelector('#mp-toggle'), skip: root.querySelector('#mp-skip'),
    close: root.querySelector('#mp-close'),
  };
  const bind = (el, fn) => {
    el.addEventListener('click', fn);
    st.offs.push(() => el.removeEventListener('click', fn));
  };
  bind(st.el.toggle, onToggle);
  bind(st.el.skip, () => onSkip(skip().forward));
  bind(st.el.close, onClose);
  bind(st.el.open, onOpen);
  st.offs.push(app.select((s) => [s.player, s.resumable, s.route && s.route.path, s.prefs], paint, {
    equals: (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3],
  }));
  // Player shortcuts: they act on whatever plays, also when the mini-player itself is hidden.
  const playing = () => app.get().player || app.get().resumable;
  const sourceNow = () => (app.get().player ? 'player' : 'resumable');
  st.offs.push(register({
    id: 'player-toggle', keys: ['Space', 'k'], scope: 'player', singleKey: true, description: 'Play or pause',
    when: () => !!playing(),
    handler: () => toggleSession(controlsFor(sourceNow()), playing()),
  }));
  st.offs.push(register({
    id: 'player-back', keys: ['j'], scope: 'player', singleKey: true, description: 'Back by the skip interval',
    when: () => !!(playing() && playing().has && playing().has.seekBy),
    handler: () => { const c = controlsFor(sourceNow()); if (c.seekBy) c.seekBy(-skip().back); },
  }));
  st.offs.push(register({
    id: 'player-forward', keys: ['l'], scope: 'player', singleKey: true, description: 'Forward by the skip interval',
    when: () => !!(playing() && playing().has && playing().has.seekBy),
    handler: () => { const c = controlsFor(sourceNow()); if (c.seekBy) c.seekBy(skip().forward); },
  }));
  paint();
  return { refresh: paint, destroy };
}

/** Re-render (the view host calls it when immersive mode changes). */
export function refresh() {
  paint();
}

/** Stop the mini-player. */
export function destroy() {
  if (!st) return;
  for (const off of st.offs) off();
  st = null;
}

/** Test hook. */
export function __resetForTests() {
  destroy();
}
