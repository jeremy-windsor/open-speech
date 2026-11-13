/**
 * The transcript on the page. #tr-log is role=log, aria-live=polite, and holds only
 * p.tr-seg items, so a screen reader hears each segment once, when it is appended:
 * - microphone segments are appended one at a time; their time is plain text (span.tr-ts), because the
 *   microphone audio is not kept (REV-39: the time is never a button that does nothing). That time is
 *   aria-hidden, so each segment is read out as its words, not "zero colon zero three" first (it stays on
 *   screen and in every export);
 * - a file result is put in while the log is aria-live="off", and a one-line summary is announced instead
 *   of the whole file (REV-24); its times are buttons ("Play from 0:03") that play the upload from there;
 * - the pending utterance and the interim hypothesis live in #tr-interim, outside the log (aria-live=off).
 * @module views/transcribe/log
 */
import { html, render } from '../../core/dom.js';
import { clock } from '../../core/format.js';
import { announce } from '../../core/a11y.js';

/** Extra time after the second frame before the log is polite again (the platform layer catches up). */
export const QUIET_MS = 100;

/** PURE: the markup of one segment. `playable` makes its time a button that plays the file from there. */
export function segmentHtml(seg, index, { playable = false } = {}) {
  const time = clock(seg.start);
  const label = 'Play from ' + time;
  const ts = playable
    ? html`<button type="button" class="tr-ts" data-action="play-from" data-start="${String(seg.start)}" aria-label="${label}">${time}</button>`
    : html`<span class="tr-ts" aria-hidden="true">${time}</span>`;
  return html`<p class="tr-seg" data-index="${String(index)}">${ts} <span class="tr-seg__text">${seg.text}</span></p>`;
}

/**
 * @param {{log: HTMLElement, interim: HTMLElement}} els
 * @returns {object} the log controller
 */
export function createLog({ log, interim }) {
  let rendered = [];      // the segment objects in the log, in order
  let playable = false;
  let current = -1;
  let restoreFrame = null;
  let restoreTimer = null;

  const frame = (fn) => (typeof requestAnimationFrame === 'function' ? requestAnimationFrame(fn) : setTimeout(fn, 16));
  const cancelFrame = (id) => {
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(id);
    else clearTimeout(id);
  };

  function cancelRestore() {
    if (restoreFrame !== null) cancelFrame(restoreFrame);
    if (restoreTimer !== null) clearTimeout(restoreTimer);
    restoreFrame = null;
    restoreTimer = null;
  }

  /**
   * Insert without reading it out (REV-24). The browser takes DOM changes into its accessibility tree in the
   * next rendering update, and decides then whether they are live: the log stays aria-live="off" until two
   * frames have passed (and a moment more), so that update sees the insertion in a region that is off. A
   * zero-delay timer usually runs before that frame, and the whole file would be read out after all.
   */
  function quietly(fn) {
    cancelRestore();
    log.setAttribute('aria-live', 'off');
    fn();
    restoreFrame = frame(() => {
      restoreFrame = frame(() => {
        restoreFrame = null;
        restoreTimer = setTimeout(() => {
          restoreTimer = null;
          log.setAttribute('aria-live', 'polite');
        }, QUIET_MS);
      });
    });
  }

  function renderAll(segments) {
    render(log, html`${segments.map((s, i) => segmentHtml(s, i, { playable }))}`);
    rendered = segments.slice();
    current = -1;
  }

  // Removing segments is never read out (a log announces additions), so it needs no quiet window.
  function trimTo(count) {
    while (rendered.length > count) {
      const last = log.lastElementChild;
      if (last) last.remove();
      rendered.pop();
    }
    if (current >= count) current = -1;
  }

  const leads = (segments) => segments.length <= rendered.length && segments.every((seg, i) => seg === rendered[i]);
  const grows = (segments) => segments.length >= rendered.length && rendered.every((seg, i) => segments[i] === seg);

  return {
    /** Append the segments past the ones shown (microphone finals: each is announced by the log). */
    sync(segments) {
      if (leads(segments)) {
        trimTo(segments.length);
        return;
      }
      if (!grows(segments)) {
        quietly(() => renderAll(segments));
        return;
      }
      for (let i = rendered.length; i < segments.length; i += 1) {
        const holder = document.createElement('div');
        render(holder, segmentHtml(segments[i], i, { playable }));
        const p = holder.firstElementChild;
        if (p) log.appendChild(p);
        rendered.push(segments[i]);
      }
    },
    /**
     * Show a whole transcript without reading it out (a file result, a restore, Clear). What is already on
     * screen stays (a recording that continues the transcript, a Cancel that drops the last session's
     * segments); only an insertion goes through the quiet window.
     */
    replace(segments, { file = false, summary = '' } = {}) {
      const mode = !!file;
      if (mode === playable && leads(segments)) {
        trimTo(segments.length);
      } else if (!segments.length) {
        playable = mode;
        renderAll(segments);
      } else {
        playable = mode;
        quietly(() => renderAll(segments));
      }
      if (summary) announce(summary);
    },
    /** The provisional line: confirmed words in text colour, the hypothesis beyond them muted. */
    pending({ confirmed = '', interim: hyp = '', listening = false } = {}) {
      if (!confirmed && !hyp) {
        if (listening) render(interim, html`<span class="tr-interim__hint">Listening…</span>`);
        else render(interim, html``);
        return;
      }
      render(interim, html`${confirmed ? html`<span class="tr-interim__confirmed">${confirmed}</span>` : ''}${confirmed && hyp ? ' ' : ''}${hyp ? html`<span class="tr-interim__hyp">${hyp}</span>` : ''}`);
    },
    /** Timestamps on or off (compact transcript). */
    setCompact(compact) {
      log.classList.toggle('is-compact', !!compact);
    },
    /** Mark the segment that is playing (-1: none). */
    setCurrent(index) {
      if (index === current) return;
      const old = current >= 0 ? log.querySelector(`[data-index="${current}"]`) : null;
      if (old) old.classList.remove('is-current');
      current = index;
      const next = index >= 0 ? log.querySelector(`[data-index="${index}"]`) : null;
      if (next) next.classList.add('is-current');
    },
    /** Whether the log is in its quiet window (aria-live="off"). */
    get quiet() { return restoreFrame !== null || restoreTimer !== null; },
    get count() { return rendered.length; },
    get playable() { return playable; },
  };
}
