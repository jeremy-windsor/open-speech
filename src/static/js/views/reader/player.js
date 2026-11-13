/**
 * The Reader player: the transport in #rd-player-slot and its sheets. The Reader view mounts it with
 * mountPlayer(slot, engine, {onOpenChapters, onOpenAppearance}); it loads its own partial
 * (viewHost.loadPartial) and drives the engine only through its API.
 *
 * - Transport: back and forward by the skip prefs (the number shows on the button), previous and next by
 *   prefs.stepUnit, a 64 px play button whose name swaps Play/Pause (no aria-pressed, RE-69) and which shows a
 *   busy ring while a part is prepared, and the chips for speed, sleep, voice and chapters (narrow layouts).
 * - Scrubber #rdp-scrub: a native range (REV-13: iOS VoiceOver can't adjust a custom role=slider) over the
 *   chapter timeline; input previews, change commits; PageUp/PageDown ±60 s, Home the chapter start,
 *   End the start of its last paragraph. Elapsed and remaining are aria-hidden; aria-valuetext carries them.
 *   The thumb follows the exact position; only while the keyboard or a screen reader is on the slider does it
 *   take the skip interval as its step (capped for short chapters), so the arrows move by it and a focused
 *   slider isn't re-announced every second. A change that no pointer and no key made (assistive technology that
 *   adjusts the slider without focusing it) moves at least one such step. A step (an arrow, PageUp/PageDown, an
 *   assistive increment) always moves: forward inside a long paragraph that isn't rendered yet, it goes on to the
 *   next paragraph instead of back to the start of this one (engine.seekChapterMs {step}).
 * - #rdp-status (role=status) is the one status line (RE-29); it never holds a counter. It sits inside the
 *   transport, so the chrome metrics count it and it never covers the text.
 * - Player shortcuts in the reading view (RE-58): Space/k, j/l, ←/→, Shift+←/→, [ and ]; [ and ] also work
 *   wherever the Reader is the player (the shortcut scope "player (Reader)").
 * - Every click, key or touch in the Reader unlocks the audio elements until both are (iOS, RE-26).
 * - Paints run once per frame while playing: every DOM write is skipped when its value is unchanged, so
 *   nothing (the Reader view's MutationObserver, the accessibility tree) hears about a frame in which nothing changed.
 * - When the partial doesn't load, the slot shows "The player didn't load" with Retry.
 * - Screenshot states: #reader/<id>?shot=speed|sleep|voice|play|return|miniplayer. They only show a state: a
 *   link to one never renders, plays or saves anything. `play` and `return` (the reading view's Back to
 *   playing pill) paint the transport as playing while nothing plays; `miniplayer` puts a display-only Reader session (no audio) in the mini-player on #speak;
 *   `sleep`'s 30-minute timer lasts only while its address does, unless the listener picks a timer there.
 * @module views/reader/player
 */
import { html, raw, render, actions, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { app } from '../../core/store.js';
import * as audio from '../../core/audio.js';
import { navigate } from '../../core/router.js';
import { register } from '../../core/shortcuts.js';
import { announce } from '../../core/a11y.js';
import { toast } from '../../core/toast.js';
import { isModalOpen } from '../../core/dialogs.js';
import { speed as speedText, spokenSpeed } from '../../core/format.js';
import { applyTile, monogram, artworkDataUrl } from '../../core/tiles.js';
import { bindRange } from '../../components/range.js';
import { loadPartial } from '../../shell/view-host.js';
import { COPY, SPEED_STEP } from './play-logic.js';
import { display, spokenAt, scrubPosition, percentOf, assistedTarget } from './timeline.js';
import { chipLabel, chipName } from './sleep-timer.js';
import { metadata as lockMetadata, sessionText, positionState, ARTWORK_SIZES } from './media-session.js';
import { bindSpeedSheet } from './speed-sheet.js';
import { bindSleepSheet } from './sleep-sheet.js';
import { bindVoiceSheet } from './voice-sheet.js';

export const PARTIAL = '/static/partials/reader-player.html';
export const PAGE_MS = 60000;   // PageUp / PageDown on the scrubber
export const SHOTS = Object.freeze(['speed', 'sleep', 'voice', 'play', 'return', 'miniplayer']);
export const LOAD_FAILED = "The player didn't load. Check your connection, then try again.";
const GESTURES = ['click', 'keydown', 'touchend'];

/** Set (or, with null/false, remove) an attribute only when it differs: a same-value write still queues a
 *  mutation record in a browser. */
function setAttr(node, name, value) {
  if (!node) return;
  if (value === null || value === undefined || value === false) {
    if (node.hasAttribute(name)) node.removeAttribute(name);
    return;
  }
  const text = value === true ? '' : String(value);
  if (node.getAttribute(name) !== text) node.setAttribute(name, text);
}

/** Set a custom property on an element's inline style only when it differs. */
function setVar(node, name, value) {
  if (node && node.style.getPropertyValue(name) !== value) node.style.setProperty(name, value);
}

function skips() {
  const prefs = app.get().prefs || {};
  return { back: Number(prefs.skipBack) || 15, forward: Number(prefs.skipForward) || 30 };
}

/** PURE: the transport's labels for the prefs. */
export function transportLabels(prefs = {}) {
  const back = Number(prefs.skipBack) || 15;
  const forward = Number(prefs.skipForward) || 30;
  const unit = prefs.stepUnit === 'sentence' ? 'sentence' : 'paragraph';
  return {
    back: `Back ${back} seconds`,
    forward: `Forward ${forward} seconds`,
    prev: `Previous ${unit}`,
    next: `Next ${unit}`,
    backNum: String(back),
    forwardNum: String(forward),
    step: back * 1000,
  };
}

/** PURE: the chips' visible text and accessible names for an engine state (RE-63). */
export function chipText(state) {
  const s = state || {};
  const rate = Number(s.speed) || 1;
  const voice = s.voiceLabel || 'Voice';
  return {
    speed: speedText(rate),
    speedName: `Playback speed ${spokenSpeed(rate)}. Change speed`,
    sleep: chipLabel(s.sleep),
    sleepName: chipName(s.sleep),
    voice,
    voiceName: `Reading voice: ${voice}. Change voice`,
  };
}

/** PURE: the scrubber target for PageUp/PageDown/Home/End, or null for other keys. */
export function scrubKeyTarget(key, { elapsedMs = 0, durationMs = 0, lastBlockMs = 0 } = {}) {
  if (key === 'PageUp') return Math.min(durationMs, elapsedMs + PAGE_MS);
  if (key === 'PageDown') return Math.max(0, elapsedMs - PAGE_MS);
  if (key === 'Home') return 0;
  if (key === 'End') return Math.max(0, lastBlockMs);
  return null;
}

/**
 * Mount the player into the slot.
 * @param {HTMLElement} slot #rd-player-slot
 * @param {object} engine getEngine()
 * @param {{onOpenChapters?: (trigger: Element) => void, onOpenAppearance?: (trigger: Element) => void}} [options]
 *   onOpenAppearance is accepted but unused: the appearance button lives in the Reader's top bar.
 * @returns {{ready: Promise<void>, refresh: () => void, openSheet: (name: string, trigger?: Element) => void, destroy: () => void}}
 *   ready resolves once the transport is in the slot (after a Retry, when the first load failed); it never rejects.
 */
export function mountPlayer(slot, engine, { onOpenChapters = null } = {}) {
  let destroyed = false;
  let el = null;
  let sheets = null;
  let scrub = null;
  let dragging = false;
  let keyboardScrub = false;   // the keyboard (or a screen reader) is on the scrubber: it steps by the skip interval
  let pointerScrub = false;    // the scrubber's focus came from a pointer, which drags to the exact position
  let scrubSource = '';        // 'pointer' or 'key' while one of them is moving the scrubber; '' is assistive tech
  let shownValue = 0;          // the scrubber value last painted (ms)
  let lastState = null;
  let shotDone = '';
  let shotPlay = '';           // the ?shot=play (or return) address whose transport shows the playing state while nothing plays
  let shotSession = null;      // ?shot=miniplayer: {session, handle} of the display-only session in the mini-player
  let shotSleep = null;        // ?shot=sleep: {key, before}, the address that armed its timer and the timer it replaced
  let markReady = () => {};
  const ready = new Promise((resolve) => { markReady = resolve; });
  const offs = [];
  const listen = (target, type, fn, options) => {
    target.addEventListener(type, fn, options);
    offs.push(() => target.removeEventListener(type, fn, options));
  };

  function readingShown() {
    const route = app.get().route || {};
    const docId = engine.state.docId;
    return !!docId && route.view === 'reader' && route.id === docId;
  }

  function setLabel(button, text) {
    setAttr(button, 'aria-label', text);
    if (button.hasAttribute('data-tooltip')) setAttr(button, 'data-tooltip', text);
  }

  function setText(node, text) {
    if (node && node.textContent !== text) node.textContent = text;
  }

  function paintPrefs() {
    if (!el) return;
    const labels = transportLabels(app.get().prefs || {});
    setLabel(el.back, labels.back);
    setLabel(el.fwd, labels.forward);
    setLabel(el.prev, labels.prev);
    setLabel(el.next, labels.next);
    setText(el.backNum, labels.backNum);
    setText(el.fwdNum, labels.forwardNum);
    if (lastState) paintTimeline(lastState);   // the scrubber's keyboard step is the skip interval
  }

  function paintPlay(state) {
    const playing = !!state.playing || !!shotPlay;
    const name = playing ? 'Pause' : 'Play';
    if (el.play.getAttribute('aria-label') !== name) {
      el.play.setAttribute('aria-label', name);
      render(el.play, html`${icon(playing ? 'pause' : 'play')}`);
    }
    setAttr(el.play, 'aria-busy', state.busy && playing ? 'true' : null);
  }

  function paintTimeline(state) {
    const t = state.timeline || {};
    const rate = Number(state.speed) || 1;
    const duration = Math.max(0, Math.round(t.durationMs || 0));
    setAttr(el.transport, 'data-estimated', !!t.estimated);
    if (el.scrub.max !== String(duration)) el.scrub.max = String(duration);
    const disabled = !state.canSeek;
    if (el.scrub.disabled !== disabled) el.scrub.disabled = disabled;
    setVar(el.scrub, '--buffer', percentOf(t.bufferedMs, duration));
    // No step while nothing is on the slider, so the thumb follows the exact position (a 9 s note included);
    // the skip interval while the keyboard or a screen reader is.
    const pos = scrubPosition(t, { keyboard: keyboardScrub, skipMs: skips().back * 1000 });
    setAttr(el.scrub, 'step', pos.step);
    if (dragging) return;
    const shown = display(t, rate);
    setText(el.elapsed, shown.elapsed);
    setText(el.remaining, shown.remaining);
    const value = String(pos.value);
    if (el.scrub.value !== value) el.scrub.value = value;
    shownValue = pos.value;
    setVar(el.scrub, '--fill', percentOf(pos.value, duration));
    // The spoken value says where the thumb is: with the keyboard on the slider it moves only when the stepped
    // value does, so a focused slider is not re-announced every second.
    setAttr(el.scrub, 'aria-valuetext', keyboardScrub ? spokenAt(pos.value, t, rate) : shown.valuetext);
  }

  function paintNow(state) {
    const doc = engine.doc;
    setText(el.chapter, state.chapterTitle || state.title || '');
    setText(el.doc, state.chapterTitle ? state.title : (state.author || ''));
    if (doc && el.tile.getAttribute('data-doc') !== doc.id) {
      el.tile.setAttribute('data-doc', doc.id);
      applyTile(el.tile, doc.id);
      el.tile.textContent = monogram(doc.title);
    }
  }

  function paintChips(state) {
    const chips = chipText(state);
    setText(el.speedLabel, chips.speed);
    setLabel(el.speed, chips.speedName);
    setText(el.sleepLabel, chips.sleep);
    setLabel(el.sleep, chips.sleepName);
    setAttr(el.sleep, 'data-on', state.sleep && state.sleep.mode !== 'off' ? 'true' : 'false');
    setText(el.voiceLabel, chips.voice);
    setLabel(el.voice, chips.voiceName);
    const hide = !(state.chapterCount > 1) || typeof onOpenChapters !== 'function';
    if (el.chapters.hidden !== hide) el.chapters.hidden = hide;
  }

  function paintStatus(status) {
    if (!el) return;
    const s = status || { text: '', kind: '' };
    setText(el.status, s.text);
    setAttr(el.status, 'data-kind', s.text ? (s.kind || 'plain') : '');
    el.status.classList.toggle('is-error', s.kind === 'error');
  }

  function paint(state = engine.state) {
    if (!el || destroyed) return;
    lastState = state;
    if (shotPlay && state.playing) shotPlay = '';   // real playback took over from the screenshot state
    const has = !!state.docId;
    setAttr(el.transport, 'data-empty', !has);
    for (const button of [el.play, el.back, el.fwd, el.prev, el.next, el.speed, el.sleep]) {
      setAttr(button, 'aria-disabled', has ? null : 'true');
    }
    paintPlay(state);
    paintTimeline(state);
    paintNow(state);
    paintChips(state);
    if (sheets) {
      if (sheets.speed.isOpen()) sheets.speed.paint(state);
      if (sheets.sleep.isOpen()) sheets.sleep.paint(state);
    }
    checkShot(state);
  }

  function openSheet(name, trigger) {
    if (!sheets) return;
    if (name === 'speed') sheets.speed.open(trigger || el.speed);
    else if (name === 'sleep') sheets.sleep.open(trigger || el.sleep);
    else if (name === 'voice') sheets.voice.open(trigger || el.voice);
  }

  // Where focus goes back to when the voice sheet opened because a start found no voice: the control the
  // listener used (Play, a paragraph), never the voice chip they didn't touch (WCAG 2.4.3).
  function returnTarget() {
    const active = typeof document !== 'undefined' ? document.activeElement : null;
    const section = slot.closest ? slot.closest('[data-view="reader"]') : null;
    if (active && active !== document.body && section && section.contains(active)) return active;
    return el.play;
  }

  // A start without a usable voice (RE-05): the voice sheet opens here, or a toast says so elsewhere.
  function onVoiceNeeded({ reason, message } = {}) {
    if (reason === 'no-model') {
      toast(message, { kind: 'error', id: 'reader-voice', action: { label: COPY.openModels, run: () => navigate('models/tts') } });
      if (readingShown() && sheets && !isModalOpen()) sheets.voice.open(returnTarget());
      return;
    }
    if (readingShown() && sheets && !isModalOpen()) {
      sheets.voice.open(returnTarget(), { message });
      return;
    }
    const docId = engine.state.docId;
    toast(message, {
      kind: 'error',
      id: 'reader-voice',
      action: docId ? { label: 'Choose voice', run: () => navigate(`reader/${encodeURIComponent(docId)}`).then(() => openSheet('voice')) } : null,
    });
  }

  // Screenshot states: applied once per route, when the engine holds the routed document. A link to one
  // only shows the state: it never asks the server to render, never plays and never saves. The page's
  // requests carry the stored key, so the server can't tell a link from the listener.
  function checkShot(state = null) {
    const route = app.get().route || {};
    if (!sheets || !route.shot || !SHOTS.includes(route.shot)) return;
    const s = state || engine.state;
    if (route.view !== 'reader' || !route.id || s.docId !== route.id) return;
    const key = `${route.path}?${route.shot}`;
    if (shotDone === key) return;
    shotDone = key;
    if (route.shot === 'sleep') {
      shotSleep = { key, before: s.sleep };
      engine.setSleep({ minutes: 30 });
    }
    if (route.shot === 'play' || route.shot === 'return') {
      if (!s.playing) {
        shotPlay = key;
        paintPlay(s);
      }
      return;
    }
    if (route.shot === 'miniplayer') {
      showShotSession(s);
      // Once the router has finished with this route: from inside its route subscription, a navigate would be
      // overtaken by the Reader route still being applied, leaving the Reader on screen under #speak.
      setTimeout(() => {
        const now = app.get().route || {};
        if (`${now.path}?${now.shot}` === key) navigate('speak', { replace: true });
      }, 0);
      return;
    }
    setTimeout(() => openSheet(route.shot), 0);
  }

  // The route moved on: the screenshot's playing transport, its timer and its mini-player session go with it,
  // and following the same link again shows its state again.
  function onRoute() {
    const route = app.get().route || {};
    const key = `${route.path}?${route.shot}`;
    if (shotDone && shotDone !== key) shotDone = '';
    if (shotPlay && shotPlay !== key) {
      shotPlay = '';
      if (el && lastState) paintPlay(lastState);
    }
    if (shotSleep && shotSleep.key !== key) {
      const { before } = shotSleep;
      shotSleep = null;
      engine.setSleep(sleepSpec(before));
    }
    if (shotSession && route.view !== 'speak') endShotSession();
    checkShot();
  }

  // Play/Pause (the button, Space and k). On the play screenshot nothing was playing: Pause shows it paused.
  function toggle() {
    if (shotPlay) {
      shotPlay = '';
      if (el && lastState) paintPlay(lastState);
      return;
    }
    engine.toggle();
  }

  /** The setSleep() spec that brings back a timer from engine.state.sleep (null for none). */
  function sleepSpec(sleep) {
    if (sleep && sleep.mode === 'minutes' && sleep.remainingMs > 0) return { minutes: sleep.remainingMs / 60000 };
    if (sleep && sleep.mode === 'chapters' && sleep.chapters > 0) return { chapters: sleep.chapters };
    return null;
  }

  /**
   * ?shot=miniplayer: a Reader session in the mini-player, playing at the saved position, with no audio and no
   * request behind it. Its controls only change the picture. It never displaces audio that is already there,
   * playing or waiting in app.resumable (the mini-player shows that instead), and when another player
   * displaces it, it goes rather than waiting in app.resumable.
   */
  function showShotSession(state) {
    const doc = engine.doc;
    if (!doc || shotSession || audio.current() || app.get().resumable) return;
    const t = state.timeline || {};
    const rate = Number(state.speed) || 1;
    const durationMs = Math.max(0, Number(t.durationMs) || 0);
    let elapsedMs = Math.min(durationMs, Math.max(0, Number(t.elapsedMs) || 0));
    let playing = true;
    let handle = null;
    const text = sessionText(lockMetadata({ doc, chapterIndex: state.chapter, voiceLabel: state.voiceLabel }));
    const changed = () => { if (handle) handle.notify(); };
    const session = {
      id: `shot:reader:${doc.id}`,
      owner: 'reader',
      kind: 'long',
      title: text.title,
      subtitle: text.subtitle,
      artwork: artworkDataUrl(doc.id, doc.title, ARTWORK_SIZES[1]),   // the document's tile, as the engine's session shows it
      homeRoute: `reader/${encodeURIComponent(doc.id)}`,
      ephemeral: false,
      controls: {
        play: () => { playing = true; changed(); },
        pause: () => {
          playing = false;
          changed();
          // Paused by a claim (another player started): this picture isn't the Reader, so it doesn't wait.
          Promise.resolve().then(() => {
            if (shotSession && shotSession.session === session && audio.current() !== session) endShotSession();
          });
        },
        toggle: () => { playing = !playing; changed(); },
        stop: () => endShotSession(),
        seekBy: (seconds) => {
          elapsedMs = Math.min(durationMs, Math.max(0, elapsedMs + (Number(seconds) || 0) * 1000));
          changed();
        },
      },
      getState: () => {
        const pos = positionState({ durationMs, elapsedMs, rate });
        return { playing, waiting: false, position: pos ? pos.position : 0, duration: pos ? pos.duration : 0, rate, canSeek: !!pos, statusText: '' };
      },
    };
    handle = audio.claim(session);
    shotSession = { session, handle };
  }

  function endShotSession() {
    const shot = shotSession;
    shotSession = null;
    if (shot) shot.handle.release();
  }

  function setKeyboardScrub(on) {
    if (keyboardScrub === on) return;
    keyboardScrub = on;
    if (lastState) paintTimeline(lastState);
  }

  // While the keyboard or a screen reader is on the slider its value sits on the step grid: a forward step must
  // land at least half a grid step on, or the value (and what a screen reader says) would not change.
  function stepFloor(step) {
    const grid = Number(el.scrub.getAttribute('step')) || 0;
    return keyboardScrub && step > 0 && grid > 0 ? shownValue + grid / 2 : 0;
  }

  function bindScrub() {
    scrub = bindRange(el.scrub, {
      format: (value) => display({ ...(lastState?.timeline || {}), elapsedMs: value }, lastState?.speed || 1).elapsed,
      spoken: (value) => spokenAt(value, lastState?.timeline || {}, lastState?.speed || 1),
    });
    offs.push(() => scrub.destroy());
    // A pointer drags to the exact position; the keyboard (Tab, or a screen reader moving focus) gets the step.
    listen(el.scrub, 'pointerdown', () => {
      pointerScrub = true;
      scrubSource = 'pointer';
      setKeyboardScrub(false);
    });
    // A press on the thumb that moved nothing fires no change: the next change may come from elsewhere.
    listen(el.scrub, 'click', () => { if (scrubSource === 'pointer') scrubSource = ''; });
    listen(el.scrub, 'focus', () => { if (!pointerScrub) setKeyboardScrub(true); });
    listen(el.scrub, 'input', () => {
      dragging = true;
      const t = lastState?.timeline || {};
      const shown = display({ ...t, elapsedMs: Number(el.scrub.value) }, lastState?.speed || 1);
      setText(el.elapsed, shown.elapsed);
      setText(el.remaining, shown.remaining);
    });
    listen(el.scrub, 'change', () => {
      dragging = false;
      const value = Number(el.scrub.value);
      // No pointer and no key: an assistive technology adjusted the slider without focusing it, by the browser's
      // 1 ms default step or a few percent. It moves one key step, as the arrows do.
      const target = scrubSource ? value : assistedTarget(value, shownValue, { skipMs: skips().back * 1000, durationMs: Number(el.scrub.max) });
      // A key or an assistive increment is a step: it always moves, also inside a long paragraph whose audio isn't
      // rendered yet (where a position maps back to the paragraph's start). A pointer goes where it was let go.
      const step = scrubSource === 'pointer' ? 0 : Math.sign(target - shownValue);
      scrubSource = '';
      engine.unlock();
      engine.seekChapterMs(target, { step, atLeastMs: stepFloor(step) });
    });
    // A cancelled touch may still end in a change: it stays the pointer's until that change, a click or blur.
    listen(el.scrub, 'pointercancel', () => { dragging = false; paint(); });
    listen(el.scrub, 'blur', () => {
      pointerScrub = false;
      dragging = false;
      keyboardScrub = false;
      scrubSource = '';
      paint();
    });
    // The browser's arrow step fires input and change inside the keydown, so the key is cleared on keyup.
    listen(el.scrub, 'keyup', () => { if (scrubSource === 'key') scrubSource = ''; });
    listen(el.scrub, 'keydown', (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      scrubSource = 'key';
      setKeyboardScrub(true);   // before the browser's own arrow step, which reads the step attribute
      const target = scrubKeyTarget(event.key, engine.state.timeline || {});
      if (target === null) return;
      event.preventDefault();
      dragging = false;
      const step = event.key === 'PageUp' ? 1 : event.key === 'PageDown' ? -1 : 0;   // Home and End go where they say
      engine.seekChapterMs(target, { step, atLeastMs: stepFloor(step) });
    });
  }

  function bindShortcuts() {
    const when = () => readingShown();
    const reg = (spec) => offs.push(register({ scope: 'view:reader', when, ...spec }));
    reg({ id: 'reader-toggle', keys: ['Space', 'k'], singleKey: true, description: 'Play or pause', handler: () => { engine.unlock(); toggle(); } });
    reg({ id: 'reader-back', keys: ['j'], singleKey: true, description: 'Back by the skip interval', handler: () => engine.seekBy(-skips().back) });
    reg({ id: 'reader-forward', keys: ['l'], singleKey: true, description: 'Forward by the skip interval', handler: () => engine.seekBy(skips().forward) });
    reg({ id: 'reader-prev-paragraph', keys: ['ArrowLeft'], description: 'Previous paragraph', handler: () => engine.skipBlock(-1) });
    reg({ id: 'reader-next-paragraph', keys: ['ArrowRight'], description: 'Next paragraph', handler: () => engine.skipBlock(1) });
    reg({ id: 'reader-prev-sentence', keys: ['Shift+ArrowLeft'], description: 'Previous sentence', handler: () => engine.skipSentence(-1) });
    reg({ id: 'reader-next-sentence', keys: ['Shift+ArrowRight'], description: 'Next sentence', handler: () => engine.skipSentence(1) });
    const nudge = (delta) => {
      const next = engine.setSpeed((Number(engine.state.speed) || 1) + delta * SPEED_STEP);
      announce(`Speed ${spokenSpeed(next)}`);
    };
    reg({ id: 'reader-slower', keys: ['['], singleKey: true, description: 'Slower by 0.05×', handler: () => nudge(-1) });
    reg({ id: 'reader-faster', keys: [']'], singleKey: true, description: 'Faster by 0.05×', handler: () => nudge(1) });
    // The shortcut table scopes [ and ] to the Reader session ("player (Reader)"): they also work from the library or another
    // view while the Reader is the session the player keys act on (playing, or paused and resumable). In the
    // reading view the view's own pair wins (a view scope outranks the player scope), so a key fires once.
    const readerIsPlayer = () => {
      const session = app.get().player || app.get().resumable;
      const doc = engine.doc;
      return !!(session && doc && session.id === `reader:${doc.id}`);
    };
    const anywhere = (spec) => offs.push(register({ scope: 'player', singleKey: true, when: readerIsPlayer, ...spec }));
    anywhere({ id: 'reader-slower-player', keys: ['['], description: 'Reader slower by 0.05×', handler: () => nudge(-1) });
    anywhere({ id: 'reader-faster-player', keys: [']'], description: 'Reader faster by 0.05×', handler: () => nudge(1) });
  }

  // iOS: unlock both elements inside the first gestures anywhere in the Reader (capture phase), then stop
  // listening once both have played (RE-26).
  function bindUnlock() {
    const section = slot.closest ? slot.closest('[data-view="reader"]') || slot : slot;
    const onGesture = () => {
      engine.unlock();
      if (engine.allUnlocked()) for (const type of GESTURES) section.removeEventListener(type, onGesture, true);
    };
    for (const type of GESTURES) listen(section, type, onGesture, true);
  }

  function bind() {
    const $ = (id) => slot.querySelector(`#${id}`);
    el = {
      transport: $('rdp-transport'),
      status: $('rdp-status'),
      tile: $('rdp-tile'),
      chapter: $('rdp-chapter'),
      doc: $('rdp-doc'),
      elapsed: $('rdp-elapsed'),
      remaining: $('rdp-remaining'),
      scrub: $('rdp-scrub'),
      back: $('rdp-back'),
      backNum: $('rdp-back-num'),
      prev: $('rdp-prev'),
      play: $('rdp-play'),
      next: $('rdp-next'),
      fwd: $('rdp-fwd'),
      fwdNum: $('rdp-fwd-num'),
      speed: $('rdp-speed'),
      speedLabel: $('rdp-speed-label'),
      sleep: $('rdp-sleep'),
      sleepLabel: $('rdp-sleep-label'),
      voice: $('rdp-voice'),
      voiceLabel: $('rdp-voice-label'),
      chapters: $('rdp-chapters'),
    };
    sheets = {
      speed: bindSpeedSheet(slot, engine),
      // A timer the listener picks in the sheet is theirs, also on a ?shot=sleep address.
      sleep: bindSleepSheet(slot, {
        get state() { return engine.state; },
        setSleep: (spec) => { shotSleep = null; return engine.setSleep(spec); },
      }),
      voice: bindVoiceSheet(slot, engine, { onOpenModels: () => navigate('models/tts') }),
    };
    offs.push(() => { for (const sheet of Object.values(sheets)) sheet.destroy(); });
    // Every handler runs synchronously inside the click: play() unlocks the elements in the gesture (iOS).
    offs.push(actions(slot, {
      play: () => toggle(),
      back: () => engine.seekBy(-skips().back),
      fwd: () => engine.seekBy(skips().forward),
      prev: () => engine.step(-1),
      next: () => engine.step(1),
      speed: (button) => openSheet('speed', button),
      sleep: (button) => openSheet('sleep', button),
      voice: (button) => openSheet('voice', button),
      chapters: (button) => { if (typeof onOpenChapters === 'function') onOpenChapters(button); },
    }));
    bindScrub();
    bindShortcuts();
    bindUnlock();
    offs.push(engine.on('state', (state) => paint(state)));
    offs.push(engine.on('status', paintStatus));
    offs.push(engine.on('voice', () => paint()));
    offs.push(engine.on('voice-needed', onVoiceNeeded));
    offs.push(app.select((s) => s.prefs, paintPrefs));
    offs.push(app.select((s) => s.route, onRoute));
    offs.push(app.select((s) => s.speakVoice, () => paint()));
    offs.push(app.select((s) => s.models, () => paint()));
    paintPrefs();
    paintStatus(engine.state.status);
    paint();
  }

  // The partial didn't load (offline, a restart, the 15 s timeout): a small bar where the transport goes says
  // so, with Retry. It counts as bottom chrome like the transport, so it covers nothing.
  function showFailed(error, { focus = false } = {}) {
    if (globalThis.console) globalThis.console.warn('The Reader player did not load:', error);
    render(slot, html`<div class="rdp-transport rdp-failed" id="rdp-failed" data-chrome="bottom" role="region" aria-label="Player"><div class="rdp-failed__row">${icon('alert', { className: 'ico rdp-failed__icon' })}<p class="rdp-failed__msg" id="rdp-failed-msg">${LOAD_FAILED}</p><button type="button" class="btn btn--secondary btn--sm rdp-failed__retry" id="rdp-retry" aria-describedby="rdp-failed-msg"><span class="btn__label">Retry</span></button></div></div>`);
    announce(LOAD_FAILED);
    const retry = slot.querySelector('#rdp-retry');
    if (!retry) return;
    retry.addEventListener('click', () => {
      const hadFocus = typeof document !== 'undefined' && document.activeElement === retry;
      setBusy(retry, true);
      load({ focus: hadFocus });
    }, { once: true });
    if (focus) retry.focus();
  }

  function load({ focus = false } = {}) {
    loadPartial(PARTIAL)
      .then((markup) => {
        if (destroyed) return;
        render(slot, raw(markup));
        bind();
        if (focus && el.play) el.play.focus();   // Retry had focus: it moves to Play, never to the page
        markReady();
      })
      .catch((error) => {
        if (!destroyed) showFailed(error, { focus });
      });
  }

  load();

  return {
    ready,
    refresh: () => paint(),
    openSheet,
    destroy() {
      destroyed = true;
      endShotSession();
      for (const off of offs.splice(0)) {
        try { off(); } catch { /* the store was reset */ }
      }
    },
  };
}
