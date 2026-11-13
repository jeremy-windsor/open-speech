/**
 * File transcription: one file at a time, with a busy state, elapsed time and Cancel
 * (BUG-09). The model is made ready first (models.ensureReady, which downloads and loads it when needed),
 * then POST /v1/audio/transcriptions as multipart {file, model, response_format: 'verbose_json'} with
 * X-History: true (the server's History is the one record) and an AbortController for Cancel.
 * verbose_json is always asked for, so segments with times exist for the timestamps and the SRT/VTT
 * exports (TR-18: the old format select is gone).
 * @module views/transcribe/upload
 */
import * as apiModule from '../../core/api.js';
import * as modelsModule from '../../core/models.js';
import { emit as emitEvent } from '../../core/store.js';
import { EMPTY_FILE_ERROR } from '../../core/pcm.js';

export const TRANSCRIBE_PATH = '/v1/audio/transcriptions';

/**
 * Read a transcription answer once (TR-03): a JSON content type is parsed, anything else is taken as the
 * text. Already-parsed JSON (what api.request returns for a JSON answer) and plain strings pass through.
 * @param {Response|object|string|null} answer
 * @returns {Promise<object>} at least {text}
 */
export async function parseTranscription(answer) {
  if (answer === null || answer === undefined) return { text: '' };
  if (typeof answer === 'string') return { text: answer };
  if (typeof answer.text === 'function' && answer.headers && typeof answer.headers.get === 'function') {
    const type = String(answer.headers.get('content-type') || '').toLowerCase();
    if (type.includes('json')) return (await answer.json()) || { text: '' };
    return { text: await answer.text() };
  }
  return typeof answer === 'object' ? answer : { text: String(answer) };
}

const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

/**
 * The segments of a verbose_json answer as {text, start, end}; a plain-text answer becomes one segment.
 * @param {object} data
 * @returns {{text: string, start: number, end: number}[]}
 */
export function toSegments(data) {
  const d = data || {};
  const list = Array.isArray(d.segments) ? d.segments : [];
  const segments = list
    .map((s) => ({ text: String(s?.text ?? '').trim(), start: Math.max(0, num(s?.start)), end: Math.max(0, num(s?.end)) }))
    .filter((s) => s.text);
  if (segments.length) return segments;
  const text = String(d.text ?? '').trim();
  return text ? [{ text, start: 0, end: Math.max(0, num(d.duration)) }] : [];
}

/**
 * @param {object} [deps]
 * @param {object} [deps.api] core/api.js
 * @param {object} [deps.models] core/models.js
 * @param {(state: object) => void} [deps.onChange]
 * @param {() => number} [deps.now] ms clock for the processing time (performance.now)
 * @returns {{start: (file: File, options: {model: string}) => Promise<object|null>, cancel: () => boolean,
 *   readonly state: object, readonly busy: boolean}}
 */
export function createUploader({ api = apiModule, models = modelsModule, onChange, now } = {}) {
  const clockNow = now || (() => (globalThis.performance && typeof performance.now === 'function' ? performance.now() : Date.now()));
  let state = Object.freeze({ phase: 'idle', fileName: '', startedAt: 0, error: null });
  let run = null;   // {controller, file, model}

  function set(patch) {
    state = Object.freeze({ ...state, ...patch });
    if (typeof onChange === 'function') {
      try { onChange(state); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
    }
  }

  const busy = () => state.phase === 'preparing' || state.phase === 'uploading';

  /**
   * Transcribe one file. Refused (null) while another is busy: one file at a time.
   * @param {File|Blob} file
   * @param {{model: string}} options
   * @returns {Promise<object|null>} {text, segments, duration, model, processingMs, fileName, language} on
   *   success; null when refused, cancelled or failed (the error is in state.error)
   */
  async function start(file, { model } = {}) {
    if (busy() || !file) return null;
    const fileName = String(file.name || 'audio');
    if (!file.size) {
      set({ phase: 'error', fileName, error: { title: "Couldn't transcribe", message: EMPTY_FILE_ERROR, detail: '', retryable: false, action: '' } });
      return null;
    }
    const my = { controller: typeof AbortController === 'function' ? new AbortController() : null, file, model: String(model || '') };
    run = my;
    set({ phase: 'preparing', fileName, startedAt: Date.now(), error: null });
    try {
      await models.ensureReady(my.model, { kind: 'stt' });
      if (run !== my) return null;
      set({ phase: 'uploading' });
      const form = new FormData();
      form.append('file', file, fileName);
      form.append('model', my.model);
      form.append('response_format', 'verbose_json');
      const t0 = clockNow();
      const answer = await api.request(TRANSCRIBE_PATH, {
        method: 'POST', body: form, history: true, signal: my.controller ? my.controller.signal : undefined, accept: 'auto',
      });
      const data = await parseTranscription(answer);
      if (run !== my) return null;
      const processingMs = Math.max(0, clockNow() - t0);
      run = null;
      models.markUsed(my.model);
      emitEvent('history:changed');
      const segments = toSegments(data);
      const result = {
        text: String(data.text ?? '').trim(),
        segments,
        duration: num(data.duration) || (segments.length ? segments[segments.length - 1].end : 0),
        model: my.model,
        processingMs,
        fileName,
        language: String(data.language || ''),
      };
      set({ phase: 'done', error: null });
      return result;
    } catch (error) {
      if (run !== my) return null;
      run = null;
      const ex = api.explain(error, 'transcribe');
      if (!ex.title || error?.name === 'AbortError') {
        set({ phase: 'idle', error: null });
        return null;
      }
      set({ phase: 'error', error: { title: ex.title, message: ex.message, detail: ex.detailText, retryable: ex.retryable, action: ex.action } });
      return null;
    }
  }

  /** Cancel the file in progress (the request is aborted). */
  function cancel() {
    const my = run;
    if (!my) return false;
    run = null;
    if (my.controller) my.controller.abort();
    set({ phase: 'idle', error: null });
    return true;
  }

  /** Forget a shown error. */
  function clearError() {
    if (state.error) set({ phase: 'idle', error: null });
  }

  return {
    start,
    cancel,
    clearError,
    get state() { return state; },
    get busy() { return busy(); },
  };
}
