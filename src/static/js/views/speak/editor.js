/**
 * Speak: the editor. One textarea (#sp-input) shared with Live, a code-point counter
 * (#sp-counter, aria-live=off), the input-limit field error (#sp-limit-hint), the unsent draft
 * (KEYS.speakDraft, saved 500 ms after typing stops), .txt upload and auto-grow up to 60vh.
 * @module views/speak/editor
 */
import { codePointLength, lengthLabel, plural, speed as speedText } from '../../core/format.js';
import { debounce } from '../../core/dom.js';
import { KEYS, read, write, remove } from '../../core/storage.js';
import { DEFAULT_LIMIT } from './request.js';

const DRAFT_MS = 500;
const n = (value) => Number(value).toLocaleString('en-US');

/**
 * "Generate accepts up to 4,096 characters. For longer text, use the Reader." While Settings shows Speak › Live
 * (`live`), "… Use Live for longer text." Live is a developer tool, hidden by default.
 */
export function limitMessage(limit, { live = false } = {}) {
  const next = live ? 'Use Live for longer text.' : 'For longer text, use the Reader.';
  return `Generate accepts up to ${n(limit || DEFAULT_LIMIT)} characters. ${next}`;
}

export const EMPTY_MESSAGE = 'Enter some text to speak first.';

/**
 * The counter: code points / limit · an estimate at the current speed ("1,204 / 4,096 · ~1 min at 1.25×").
 * An emoji counts as one character, as on the server (SP-14, SP-15). `limit: 0` is Live mode, which has no
 * input limit: the count alone ("5,000 characters · ~6 min at 1×"), so a long text never reads as an error.
 */
export function counterText(text, { limit = DEFAULT_LIMIT, speed = 1 } = {}) {
  const count = codePointLength(text);
  const base = limit ? `${n(count)} / ${n(limit)}` : plural(count, 'character');
  if (!count) return base;
  return `${base} · ${estimate(count, speed)} at ${speedText(speed)}`;
}

/** Today's ~15 characters a second: "~8 s" under a minute, else format.lengthLabel ("~5 min"). */
export function estimate(count, speed = 1) {
  const rate = Number(speed) > 0 ? Number(speed) : 1;
  const seconds = count / 15 / rate;
  if (seconds < 59.5) return `~${Math.max(1, Math.round(seconds))} s`;
  return lengthLabel(count, rate);
}

/** '' when the text can be generated, else 'empty' or 'over' (more code points than the limit). */
export function checkInput(text, limit = DEFAULT_LIMIT) {
  const trimmed = String(text ?? '').trim();
  if (!trimmed) return 'empty';
  return codePointLength(trimmed) > (limit || DEFAULT_LIMIT) ? 'over' : '';
}

/**
 * Wire the editor.
 * @param {{input: HTMLTextAreaElement, counter: HTMLElement, hint: HTMLElement, hintText: HTMLElement,
 *   upload: HTMLInputElement}} els
 * @param {{getLimit: () => number, getSpeed: () => number, enforceLimit: () => boolean, liveShown?: () => boolean,
 *   onChange?: (text: string, source: string) => void, onLimit?: (over: boolean) => void,
 *   onUpload?: (text: string) => void, onError?: (error: Error) => void}} hooks
 */
export function createEditor(els, hooks) {
  const { input, counter, hint, hintText, upload } = els;
  let shownError = '';   // '' | 'over' | 'empty'
  const saveDraft = debounce(() => {
    const text = input.value;
    if (text) write(KEYS.speakDraft, text);
    else remove(KEYS.speakDraft);
  }, DRAFT_MS);

  const fieldSizing = (() => {
    try { return typeof CSS !== 'undefined' && CSS.supports && CSS.supports('field-sizing', 'content'); } catch { return false; }
  })();
  let growFrame = 0;
  function grow() {
    if (fieldSizing || growFrame || typeof requestAnimationFrame !== 'function') return;
    growFrame = requestAnimationFrame(() => {
      growFrame = 0;
      if (!input.isConnected || input.offsetParent === null) return;
      input.style.setProperty('--sp-input-h', 'auto');
      const max = Math.round((globalThis.innerHeight || 800) * 0.6);
      const height = Math.min(max, input.scrollHeight + 2);
      input.style.setProperty('--sp-input-h', `${height}px`);
    });
  }

  function describe() {
    const ids = ['sp-counter'];
    if (shownError) ids.push(hint.id);
    input.setAttribute('aria-describedby', ids.join(' '));
  }

  function showError(kind) {
    shownError = kind;
    if (kind) {
      const live = typeof hooks.liveShown === 'function' && !!hooks.liveShown();
      hintText.textContent = kind === 'over' ? limitMessage(hooks.getLimit(), { live }) : EMPTY_MESSAGE;
      input.setAttribute('aria-invalid', 'true');
      hint.hidden = false;
    } else {
      hintText.textContent = '';
      input.removeAttribute('aria-invalid');
      hint.hidden = true;
    }
    describe();
  }

  /** Re-check the counter and the limit (after typing, a model change, a speed change, a mode change). */
  function refresh() {
    const limit = hooks.getLimit() || DEFAULT_LIMIT;
    const text = input.value;
    const enforce = hooks.enforceLimit();
    counter.textContent = counterText(text, { limit: enforce ? limit : 0, speed: hooks.getSpeed() });
    const over = enforce && checkInput(text, limit) === 'over';
    if (over) showError('over');
    else if (shownError === 'over' || (shownError === 'empty' && text.trim())) showError('');
    if (typeof hooks.onLimit === 'function') hooks.onLimit(over);
    return over;
  }

  /**
   * Replace the text (upload, a history entry, Clear), as if typed. A screenshot state's sample text
   * (`source` 'shot') is display only: it never replaces the listener's saved draft.
   */
  function set(text, source = 'set') {
    input.value = String(text ?? '');
    refresh();
    if (source !== 'shot') saveDraft();
    grow();
    if (typeof hooks.onChange === 'function') hooks.onChange(input.value, source);
  }

  const onInput = () => {
    refresh();
    saveDraft();
    grow();
    if (typeof hooks.onChange === 'function') hooks.onChange(input.value, 'input');
  };
  input.addEventListener('input', onInput);

  const onUpload = async () => {
    const file = upload.files && upload.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      if (typeof hooks.onUpload === 'function') hooks.onUpload(text);
      set(text, 'upload');
    } catch (error) {
      if (typeof hooks.onError === 'function') hooks.onError(error);
    } finally {
      try { upload.value = ''; } catch { /* some browsers refuse; harmless */ }
    }
  };
  upload.addEventListener('change', onUpload);

  /** Restore the unsent draft (only into an empty editor). */
  function restoreDraft() {
    if (input.value) return;
    const draft = read(KEYS.speakDraft, '');
    if (draft) {
      input.value = draft;
      refresh();
      grow();
    }
  }

  return {
    value: () => input.value,
    set,
    refresh,
    grow,
    restoreDraft,
    showError,
    error: () => shownError,
    flushDraft: () => saveDraft.flush(),
    destroy() {
      saveDraft.flush();
      input.removeEventListener('input', onInput);
      upload.removeEventListener('change', onUpload);
    },
  };
}
