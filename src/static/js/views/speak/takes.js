/**
 * Speak: takes. The newest take first, at most 10 per page session; each one has a play/pause
 * button that claims the shared audio element through audio.playClip (homeRoute 'speak'), a static
 * waveform drawn once from peaks, its time, a download and a menu (Use these settings, Copy request,
 * Save as voice… for a take whose voice was designed from the description, Remove). Object URLs are revoked
 * when a take is removed.
 *
 * Also here: receive(), which turns a speech response into the take's audio, through Media Source
 * Extensions while it streams (the take's blob is built from the same chunks, so Download saves the audio
 * that played; SP-19) or as one blob.
 * @module views/speak/takes
 */
import { html, render } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { clock, plural, speed as speedText } from '../../core/format.js';
import { label as perfLabel, SLOW_RTF } from '../../core/perf.js';
import * as audio from '../../core/audio.js';
import { downloadBlob } from '../../core/download.js';
import { decodeToMono, encodeWav } from '../../core/pcm.js';
import { bindMenuButton } from '../../components/menu.js';
import { takeFilename, formatLabel } from './request.js';

export const MAX_TAKES = 10;
export const PEAK_COUNT = 96;
/** The longest take that can become a designed voice's sample (POST /api/voices/designed). */
export const MAX_SAMPLE_S = 30;
export const SAMPLE_TOO_LONG = `A voice sample can be at most ${MAX_SAMPLE_S} s`;
export const SAMPLE_PCM = "A PCM take can't be saved as a voice";

/**
 * PURE: `count` peaks (0–1) for a waveform: the largest absolute sample in each of `count` equal buckets,
 * scaled so the loudest bucket is 1. Silence (or no samples) gives zeros.
 * @param {Float32Array|number[]} samples
 * @param {number} [count]
 * @returns {number[]}
 */
export function peaks(samples, count = PEAK_COUNT) {
  const n = Math.max(1, Math.floor(count));
  const out = new Array(n).fill(0);
  const len = samples ? samples.length : 0;
  if (!len) return out;
  for (let b = 0; b < n; b += 1) {
    const start = Math.floor((b * len) / n);
    const end = Math.max(start + 1, Math.floor(((b + 1) * len) / n));
    let max = 0;
    for (let i = start; i < end && i < len; i += 1) {
      const a = Math.abs(samples[i]);
      if (a > max) max = a;
    }
    out[b] = max;
  }
  const top = Math.max(...out);
  return top > 0 ? out.map((v) => Math.round((v / top) * 1000) / 1000) : out;
}

/** PURE: the take's title, the start of its text in quotes ("“In a hole in the ground…”"). */
export function excerpt(text, max = 90) {
  const clean = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (!clean) return 'Untitled take';
  const chars = Array.from(clean);
  return `“${chars.length > max ? `${chars.slice(0, max - 1).join('').trimEnd()}…` : clean}”`;
}

/** PURE: "Heart · Kokoro · 1× · MP3 · 1,204 chars". */
export function metaLine(take) {
  const t = take || {};
  const parts = [t.voiceLabel, t.modelLabel, speedText(t.speed ?? 1), formatLabel(t.format), plural(t.chars || 0, 'char')];
  return parts.filter(Boolean).join(' · ');
}

/** PURE: "RTF 0.04"; a factor below 0.005 reads "RTF < 0.01" rather than a misleading 0.00. */
export function rtfText(rtf) {
  if (!Number.isFinite(rtf)) return '';
  if (rtf < 0.005) return 'RTF < 0.01';
  return perfLabel({ rtf });
}

/** PURE: "First audio 140 ms · RTF 0.04 · GPU" (the parts measured so far). */
export function perfLine(take) {
  const t = take || {};
  const parts = [];
  if (Number.isFinite(t.ttfaMs)) parts.push(perfLabel({ ttfaMs: t.ttfaMs }));
  if (Number.isFinite(t.rtf)) parts.push(rtfText(t.rtf));
  if (t.device) parts.push(t.device);
  return parts.join(' · ');
}

/**
 * PURE: why a designed take can't become a voice ('' when it can, or when its length isn't known yet).
 * @param {{format?: string, durationS?: number}} take
 */
export function sampleProblem(take) {
  const t = take || {};
  if (t.format === 'pcm') return SAMPLE_PCM;
  if (t.durationS > MAX_SAMPLE_S) return SAMPLE_TOO_LONG;
  return '';
}

/**
 * A take's audio as the WAV sample a designed voice keeps: a WAV take as it came, any other format decoded
 * and written as 16-bit WAV.
 * @param {{blob: Blob, format: string}} take
 * @returns {Promise<{blob: Blob, durationS: number}>}
 */
export async function sampleWav(take) {
  const { samples, sampleRate, durationS } = await decodeToMono(take.blob);
  return { blob: take.format === 'wav' ? take.blob : encodeWav(samples, sampleRate), durationS };
}

/** PURE: '0:12', or '0:03 / 0:12' once playback has moved. */
export function timeText(position, duration) {
  if (!(duration > 0)) return position > 0 ? clock(position) : '';
  return position > 0.05 ? `${clock(position)} / ${clock(duration)}` : clock(duration);
}

/**
 * A body that failed while the audio arrived (the connection dropped): a network error api.explain() words
 * plainly ("Can't reach the server"), with the browser's own text in Details. A Stop stays an abort.
 */
function bodyError(error, signal) {
  if ((signal && signal.aborted) || (error && error.name === 'AbortError')) return error;
  const out = new Error('The connection dropped while the audio arrived.');
  out.kind = 'network';
  out.detail = String((error && error.message) || error || '');
  return out;
}

/** Resolve once any of the [target, type] events fires; every listener is removed then. */
function firstEvent(pairs) {
  return new Promise((resolve) => {
    const done = () => {
      for (const [target, type] of pairs) target.removeEventListener(type, done);
      resolve();
    };
    for (const [target, type] of pairs) target.addEventListener(type, done);
  });
}

/**
 * Turn a speech response into audio. Streaming (`stream` and a body): a MediaSource is attached through
 * `onSource(url, source)` (the caller starts playback on it) and the chunks are appended as they arrive;
 * otherwise the whole body is read as one blob. Resolves with the blob of every byte received.
 *
 * The element may let go of the stream: another clip takes the shared element (the source is detached and
 * its SourceBuffer removed), the player is closed, or playback never starts (a recording runs, so the
 * source never opens). The caller then calls `source.release()`, and a detached source or a full buffer is
 * noticed here too: appending stops, the rest of the body is still read, and the take keeps every byte.
 * @param {Response} res
 * @param {{stream: boolean, type?: string, signal?: AbortSignal,
 *   onSource?: (url: string, source: {release: () => void}) => void, onChunk?: (bytes: number) => void}} options
 * @returns {Promise<Blob>}
 */
export async function receive(res, { stream = false, type = 'audio/mpeg', signal, onSource, onChunk } = {}) {
  const MS = globalThis.MediaSource;
  if (!stream || !res.body || !MS) {
    let blob;
    try {
      blob = await res.blob();
    } catch (error) {
      throw bodyError(error, signal);
    }
    if (typeof onChunk === 'function') onChunk(blob.size);
    return blob;
  }
  const ms = new MS();
  const url = URL.createObjectURL(ms);
  let held = true;   // false once the element let go of the source (or never took it)
  let letGo = () => {};
  const released = new Promise((resolve) => { letGo = resolve; });
  const source = {
    release() {
      held = false;
      letGo(false);
    },
  };
  const opened = new Promise((resolve, reject) => {
    ms.addEventListener('sourceopen', () => resolve(true), { once: true });
    if (signal) signal.addEventListener('abort', () => reject(new DOMException('Canceled', 'AbortError')), { once: true });
  });
  opened.catch(() => {});   // a Stop after the element let go is seen by the body read below
  if (typeof onSource === 'function') onSource(url, source);
  let open = false;
  try {
    open = await Promise.race([opened, released]);
  } finally {
    URL.revokeObjectURL(url);   // attached now (or abandoned): the element keeps the source
  }
  let sb = null;
  if (open && held) {
    try {
      sb = ms.addSourceBuffer(type);
    } catch {
      sb = null;   // the source closed meanwhile, or refuses the type: the take plays when it's ready
    }
  }
  const attached = () => {
    if (!sb || !held || ms.readyState !== 'open') return false;
    const list = ms.sourceBuffers;
    return !list || Array.from(list).includes(sb);
  };
  const idle = () => {
    if (!sb || !sb.updating) return Promise.resolve();
    return firstEvent([[sb, 'updateend'], [sb, 'abort'], [sb, 'error'], [ms, 'sourceclose']]);
  };
  const reader = res.body.getReader();
  const chunks = [];
  for (;;) {
    let step;
    try {
      step = await reader.read();
    } catch (error) {
      throw bodyError(error, signal);
    }
    const { done, value } = step;
    if (done) break;
    chunks.push(value);
    if (typeof onChunk === 'function') onChunk(value.byteLength || value.length || 0);
    if (!attached()) continue;
    await idle();
    if (!attached()) continue;
    try {
      sb.appendBuffer(value);
    } catch {
      // InvalidStateError (detached meanwhile) or QuotaExceededError (the buffer is full): stop feeding
      // the element and keep reading, so the take still gets the whole audio.
      held = false;
      continue;
    }
    await idle();
  }
  try {
    // Whatever was fed plays to its end (also after a full buffer); a detached source is left alone.
    if (sb && ms.readyState === 'open' && !sb.updating) ms.endOfStream();
  } catch { /* the element was reset meanwhile */ }
  return new Blob(chunks, { type });
}

/** PURE: at most `max` bars from peaks (the loudest of each group), so a narrow canvas stays legible. */
export function fitPeaks(values, max) {
  const list = values || [];
  if (list.length <= max) return list.slice();
  const out = [];
  for (let b = 0; b < max; b += 1) {
    const start = Math.floor((b * list.length) / max);
    const end = Math.max(start + 1, Math.floor(((b + 1) * list.length) / max));
    let top = 0;
    for (let i = start; i < end; i += 1) if (list[i] > top) top = list[i];
    out.push(top);
  }
  return out;
}

/**
 * Draw a waveform on a canvas: bars from `values` (0–1), the played part (`progress`, 0–1) in the
 * --sp-wave-played colour and the rest in the canvas's `color`; no values draws a flat line.
 */
export function drawWave(canvas, values, progress = 0) {
  if (!canvas || typeof canvas.getContext !== 'function') return;
  const rect = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : { width: 0, height: 0 };
  const width = Math.max(1, Math.round(rect.width || 0));
  const height = Math.max(1, Math.round(rect.height || 0));
  if (width < 4 || height < 4) return;
  const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
  const w = Math.round(width * dpr);
  const h = Math.round(height * dpr);
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  let rest = '';
  let played = '';
  try {
    const style = getComputedStyle(canvas);
    rest = style.color || '';
    played = (style.getPropertyValue('--sp-wave-played') || '').trim() || rest;
  } catch { /* no styles: the defaults below */ }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const mid = height / 2;
  if (!values || !values.length || !values.some((v) => v > 0)) {
    ctx.fillStyle = rest;
    ctx.fillRect(0, mid - 1, width, 2);
    return;
  }
  // Bars of about 2 px with a 2 px gap whatever the width: fewer peaks on a narrow card.
  const bars = fitPeaks(values, Math.max(8, Math.floor(width / 4)));
  const bar = width / bars.length;
  const gap = bar > 3 ? Math.min(3, bar / 2) : (bar > 2 ? 1 : 0);
  const cut = Math.max(0, Math.min(1, progress)) * width;
  for (let i = 0; i < bars.length; i += 1) {
    const x = i * bar;
    const barH = Math.max(2, bars[i] * (height - 4));
    ctx.fillStyle = x + bar / 2 <= cut ? played : rest;
    ctx.fillRect(x, mid - barH / 2, Math.max(1, bar - gap), barH);
  }
}

// The menu sits next to the card, not inside it. The card is a size container (container queries), which
// would trap the fixed-position menu; the menu component now re-hosts such a menu while it is open,
// so this placement is simply the simpler one, and Tab from the open menu goes through its button.
// Each tooltip equals its button's name, so assistive technology hears the name once (no description).
// The time is plain text (no live region), so the take's length can be read.
function takeMarkup(take) {
  const n = take.n;
  return html`<article class="sp-take" aria-labelledby="sp-take-${n}-title">
    <div class="sp-take__row">
      <button type="button" class="icon-btn icon-btn--lg sp-take__play" data-take-act="play" aria-label="Play take ${n}" data-tooltip="Play take ${n}">${icon('play')}</button>
      <div class="sp-take__wave"><canvas class="sp-wave" aria-hidden="true"></canvas></div>
      <span class="sp-take__time tabular"></span>
      <div class="sp-take__actions">
        <button type="button" class="icon-btn" data-take-act="download" aria-label="Download take ${n}" data-tooltip="Download take ${n}">${icon('download')}</button>
        <button type="button" class="icon-btn" id="sp-take-${n}-more" aria-label="More actions for take ${n}" data-tooltip="More actions for take ${n}" aria-haspopup="menu" aria-expanded="false" aria-controls="sp-take-${n}-menu">${icon('dots')}</button>
      </div>
    </div>
    <div class="sp-take__text">
      <h3 class="sp-take__title" id="sp-take-${n}-title"></h3>
      <p class="sp-take__meta"></p>
      <p class="sp-take__perf"></p>
      <p class="sp-take__hint" hidden></p>
    </div>
  </article>
  <div class="menu" id="sp-take-${n}-menu" role="menu" aria-label="Take ${n}" hidden>
    <button type="button" class="menu__item" role="menuitem" data-take-menu="use">${icon('sliders')}Use these settings</button>
    <button type="button" class="menu__item" role="menuitem" data-take-menu="copy">${icon('code')}Copy request (curl)</button>
    <button type="button" class="menu__item" role="menuitem" data-take-menu="save-recording">${icon('mic')}Save as recording…</button>
    ${take.design ? html`<button type="button" class="menu__item sp-take__save" role="menuitem" data-take-menu="save-voice" aria-disabled="true">${icon('voice')}<span class="sp-take__save-text"><span>Save as voice…</span><span class="sp-take__save-note" hidden></span></span></button>` : ''}
    <div role="separator" class="menu__sep"></div>
    <button type="button" class="menu__item menu__item--danger" role="menuitem" data-take-menu="remove">${icon('trash')}Remove</button>
  </div>`;
}

/**
 * The takes list.
 * @param {{list: HTMLElement, empty?: HTMLElement}} els
 * @param {{onUse?: Function, onCopy?: Function, onSaveVoice?: Function, onSaveRecording?: Function, onRemove?: Function,
 *   onPlayError?: Function}} hooks
 */
export function createTakes({ list, empty }, hooks = {}) {
  const takes = [];   // newest first
  const views = new Map();   // n -> {item, play, canvas, time, download, title, meta, perf, hint, save, saveNote, menu}

  function viewFor(take) { return views.get(take.n); }

  function paint(take) {
    const v = viewFor(take);
    if (!v) return;
    v.title.textContent = excerpt(take.text);
    v.meta.textContent = metaLine(take);
    const perf = perfLine(take);
    const badges = [];
    if (take.cached) badges.push(html`<span class="badge">Cached</span>`);
    if (Number.isFinite(take.rtf) && take.rtf >= SLOW_RTF) badges.push(html`<span class="badge badge--warn">Slower than real time</span>`);
    if (take.state === 'streaming') badges.push(html`<span class="badge badge--accent">Streaming</span>`);
    render(v.perf, html`${perf ? html`<span>${perf}</span>` : ''}${badges}`);
    v.perf.hidden = !perf && !badges.length;
    v.hint.textContent = take.hint || '';
    v.hint.hidden = !take.hint;
    const playable = take.plan === 'play' && !!take.src;
    v.play.disabled = !playable;
    v.play.hidden = take.plan !== 'play';
    v.item.classList.toggle('is-download', take.plan !== 'play');
    v.download.disabled = !take.blob;
    if (v.save) {
      const problem = sampleProblem(take);
      v.save.setAttribute('aria-disabled', String(!!problem || !take.blob));
      v.saveNote.textContent = problem;
      v.saveNote.hidden = !problem;
    }
    paintTime(take);
  }

  function paintTime(take, position = take.position || 0) {
    const v = viewFor(take);
    if (!v) return;
    v.time.textContent = timeText(position, take.durationS || 0);
  }

  function paintPlaying(take, playing) {
    const v = viewFor(take);
    if (!v || take.playingShown === playing) return;
    take.playingShown = playing;
    render(v.play, icon(playing ? 'pause' : 'play'));
    const label = `${playing ? 'Pause' : 'Play'} take ${take.n}`;
    v.play.setAttribute('aria-label', label);
    v.play.setAttribute('data-tooltip', label);
    v.item.classList.toggle('is-playing', playing);
  }

  function redraw(take) {
    const v = viewFor(take);
    if (!v) return;
    const progress = take.durationS > 0 ? (take.position || 0) / take.durationS : 0;
    drawWave(v.canvas, take.peaks, progress);
  }

  function syncEmpty() {
    if (empty) empty.hidden = takes.length > 0;
    list.hidden = takes.length === 0;
  }

  function toggle(take) {
    const cur = audio.current();
    if (cur && cur.id === take.id) {
      cur.controls.toggle();
      return;
    }
    if (take.plan !== 'play' || !take.src) return;
    play(take).catch((error) => { if (typeof hooks.onPlayError === 'function') hooks.onPlayError(error, take); });
  }

  /**
   * Start a take on the shared element (synchronously inside a click: playClip calls play() first).
   * `onRelease(reason)` runs when the element lets go of this session (displaced, replaced, closed, ended).
   */
  function play(take, { src = take.src, onRelease = null } = {}) {
    return audio.playClip({
      id: take.id, owner: 'speak', title: excerpt(take.text, 60), subtitle: `Speak · ${take.voiceLabel || take.modelLabel || ''}`.replace(/ · $/, ''),
      homeRoute: 'speak', src, ...(typeof onRelease === 'function' ? { onRelease } : {}),
    });
  }

  function add(take) {
    takes.unshift(take);
    const item = document.createElement('li');
    item.className = 'sp-takes__item';
    item.setAttribute('data-take', String(take.n));
    render(item, takeMarkup(take));
    list.insertBefore(item, list.firstChild);
    const q = (sel) => item.querySelector(sel);
    const v = {
      item, play: q('[data-take-act="play"]'), canvas: q('canvas'), time: q('.sp-take__time'),
      download: q('[data-take-act="download"]'), title: q('.sp-take__title'), meta: q('.sp-take__meta'),
      perf: q('.sp-take__perf'), hint: q('.sp-take__hint'), save: q('[data-take-menu="save-voice"]'),
      saveNote: q('.sp-take__save-note'),
    };
    v.menu = bindMenuButton(q(`#sp-take-${take.n}-more`), q(`#sp-take-${take.n}-menu`), {
      placement: 'bottom-end',
      onSelect: (menuItem) => {
        const act = menuItem.getAttribute('data-take-menu');
        if (act === 'use' && hooks.onUse) hooks.onUse(take);
        else if (act === 'copy' && hooks.onCopy) hooks.onCopy(take);
        else if (act === 'save-voice' && hooks.onSaveVoice) hooks.onSaveVoice(take);
        else if (act === 'save-recording' && hooks.onSaveRecording) hooks.onSaveRecording(take);
        else if (act === 'remove') removeTake(take.n, { focus: true });
      },
    });
    views.set(take.n, v);
    paint(take);
    syncEmpty();
    while (takes.length > MAX_TAKES) removeTake(takes[takes.length - 1].n);
    return take;
  }

  function removeTake(n, { focus = false } = {}) {
    const index = takes.findIndex((t) => t.n === n);
    if (index < 0) return;
    const [take] = takes.splice(index, 1);
    const cur = audio.current();
    if (cur && cur.id === take.id) audio.closeCurrent();
    const v = views.get(n);
    views.delete(n);
    if (v) {
      v.menu.destroy();
      v.item.remove();
    }
    if (take.url) URL.revokeObjectURL(take.url);
    take.url = '';
    take.src = '';
    take.removed = true;
    syncEmpty();
    if (typeof hooks.onRemove === 'function') hooks.onRemove(take, { focus, next: takes[index] || takes[index - 1] || null });
  }

  /** The take's audio arrived: keep the blob (Download) and an object URL (Play). */
  function setAudio(take, blob) {
    if (take.removed) return;
    take.blob = blob;
    if (take.url) URL.revokeObjectURL(take.url);
    take.url = URL.createObjectURL(blob);
    take.src = take.url;
    paint(take);
  }

  /** Decode the take's audio for its waveform (and duration) after playback started. */
  async function decode(take) {
    if (!take.blob || take.format === 'pcm' || take.removed) return null;
    try {
      const { samples, durationS } = await decodeToMono(take.blob);
      if (take.removed) return null;
      take.peaks = peaks(samples);
      // The decoded length is the take's own: it wins over what the element reported (RTF, perf records).
      if (Number.isFinite(durationS) && durationS > 0) take.durationS = durationS;
    } catch {
      take.peaks = [];   // a flat line
    }
    paint(take);
    redraw(take);
    return take.peaks;
  }

  function download(take) {
    if (!take.blob) return;
    downloadBlob(take.blob, takeFilename(take.format, take.at));
  }

  const onClick = (event) => {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const button = start && start.closest ? start.closest('[data-take-act]') : null;
    if (!button || !list.contains(button) || button.disabled) return;
    const item = button.closest('[data-take]');
    const take = takes.find((t) => String(t.n) === item?.getAttribute('data-take'));
    if (!take) return;
    const act = button.getAttribute('data-take-act');
    if (act === 'play') toggle(take);
    else if (act === 'download') download(take);
  };
  list.addEventListener('click', onClick);

  /**
   * Whether the shared element holds this take's audio now. Right after the take's session is claimed it
   * still holds the 0.1 s clip Generate primed, whose duration must never become the take's.
   */
  function holds(take) {
    try {
      const el = audio.element();
      const src = (el && (el.currentSrc || el.src)) || '';
      return !!src && (src === take.src || src === take.streamSrc);
    } catch {
      return false;
    }
  }

  /** Mirror app.player into the take that is playing: icon, time and the played part of the waveform. */
  function sync(player) {
    for (const take of takes) {
      const mine = !!player && player.id === take.id;
      paintPlaying(take, mine && player.playing);
      const position = mine ? player.position : 0;
      if (mine && player.duration > 0 && !(take.durationS > 0) && holds(take)) take.durationS = player.duration;
      if (position !== (take.position || 0)) {
        take.position = position;
        paintTime(take);
        redraw(take);
      }
    }
  }

  syncEmpty();

  return {
    add,
    remove: removeTake,
    setAudio,
    decode,
    download,
    play,
    paint,
    redraw,
    redrawAll: () => { for (const take of takes) redraw(take); },
    sync,
    get: (n) => takes.find((t) => t.n === n) || null,
    all: () => takes.slice(),
    focusPlay(take) {
      const v = viewFor(take);
      const target = v && !v.play.hidden && !v.play.disabled ? v.play : v?.download;
      if (target) target.focus();
    },
    destroy() {
      list.removeEventListener('click', onClick);
      for (const take of takes.slice()) removeTake(take.n);
    },
    /** Test hook: the markup a take renders (for the accessible-name checks). */
    __markup: takeMarkup,
  };
}
