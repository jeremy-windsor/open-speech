/**
 * Transcribe view copy and markup, pure: what each recorder phase shows, the
 * microphone help states, the meta line and the inline error banners. index.js applies them to the page.
 * @module views/transcribe/ui
 */
import { html } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { clock } from '../../core/format.js';
import { shortName } from '../../core/models.js';

/** The microphone help copy. */
export const MIC_HELP = Object.freeze({
  insecure: 'The microphone needs a secure connection. Open this page over HTTPS (Open Speech can make a self-signed certificate).',
  denied: "Microphone access is blocked. Allow it in your browser's site settings, then try again.",
  'no-device': 'No microphone found.',
  unsupported: "This browser can't capture audio.",
  background: 'Recording stopped when the app went to the background.',
});

/**
 * What the view says when a recording hears nothing: while it runs, in the microphone
 * help banner; after it ended, where the transcript would be. 'silence': no sound came in at all; 'no-speech':
 * sound, but nothing the server called speech.
 */
export const NOTHING_HEARD = Object.freeze({
  'no-speech': Object.freeze({ title: 'No speech heard yet', message: 'Check the microphone and its level.' }),
  silence: Object.freeze({
    title: 'The microphone sends silence',
    message: "No sound is coming in. Check that it isn't muted or switched off, or stop and choose another microphone.",
  }),
});
export const NOTHING_HEARD_ENDED = Object.freeze({
  'no-speech': Object.freeze({ title: 'No speech was heard.', message: 'Check the microphone and its level, then record again.' }),
  silence: Object.freeze({
    title: 'The microphone sent only silence.',
    message: "Check that it isn't muted or switched off, or choose another microphone, then record again.",
  }),
});

/**
 * PURE: the nothing-heard copy for a recorder state, or null. While the recording runs (or
 * finishes) the present tense; once it is done, the past.
 * @param {{phase?: string, nothingHeard?: string}|null} rec the recorder state
 * @returns {{reason: string, title: string, message: string}|null}
 */
export function nothingHeard(rec) {
  const kind = rec && rec.nothingHeard;
  if (!kind || !NOTHING_HEARD[kind]) return null;
  if (rec.phase === 'recording' || rec.phase === 'stopping') return { reason: kind, ...NOTHING_HEARD[kind] };
  if (rec.phase === 'done') return { reason: kind, ...NOTHING_HEARD_ENDED[kind] };
  return null;
}

/** Seconds of recording after which Cancel asks first. */
export const CONFIRM_CANCEL_AFTER_S = 30;

/**
 * PURE: the microphone help state, or null when there is nothing to say. `blocking` means Record can't work
 * at all on this page (insecure or unsupported), so it is disabled.
 * @param {{support?: {ok: boolean, reason?: string}, permission?: string, devices?: object[]|null,
 *   error?: {kind?: string, reason?: string}|null}} facts
 * @returns {{reason: string, message: string, blocking: boolean}|null}
 */
export function micHelp({ support = { ok: true }, permission = 'unknown', devices = null, error = null } = {}) {
  if (support && support.ok === false) {
    const reason = support.reason === 'insecure' ? 'insecure' : 'unsupported';
    return { reason, message: MIC_HELP[reason], blocking: true };
  }
  if (error && error.kind === 'mic' && MIC_HELP[error.reason]) {
    return { reason: error.reason, message: MIC_HELP[error.reason], blocking: false };
  }
  if (permission === 'denied') return { reason: 'denied', message: MIC_HELP.denied, blocking: false };
  if (Array.isArray(devices) && devices.length === 0) return { reason: 'no-device', message: MIC_HELP['no-device'], blocking: false };
  return null;
}

/** The phases in which Record shows as busy (the first await has started). */
const BUSY = new Set(['requesting', 'starting', 'stopping']);

/** The hint beside an idle Record button. */
export const IDLE_HINT = 'Words appear as you speak.';

/** Why Record is off while idle, in words (the button's aria-describedby points at the hint). */
const BLOCKED_HINT = Object.freeze({
  upload: 'Finish or cancel the file first.',
  model: 'Recording needs a transcription model.',
  'models-loading': 'Waiting for the list of models.',
  mic: "Recording isn't available on this page.",
});

/**
 * PURE: what the capture area shows for a recorder state. The record button's accessible name is
 * "Record" or "Stop recording" and never carries aria-pressed (REV-12).
 * @param {{phase: string}} rec the recorder state
 * @param {{jobLabel?: string, blockedBy?: ''|'upload'|'model'|'models-loading'|'mic'}} [context]
 *   jobLabel: the status job for the chosen model ("Loading whisper base…"); blockedBy: why Record is off
 * @returns {{phase: string, buttonLabel: string, busy: boolean, disabled: boolean, recording: boolean,
 *   active: boolean, showTimer: boolean, hint: string}}
 */
export function captureView(rec, { jobLabel = '', blockedBy = '' } = {}) {
  const phase = rec?.phase || 'idle';
  const recording = phase === 'recording';
  const active = recording || BUSY.has(phase);
  let text = 'Record';
  let hint = IDLE_HINT;
  if (phase === 'requesting') {
    text = 'Starting…';
    hint = 'Allow the microphone if your browser asks.';
  } else if (phase === 'starting') {
    text = jobLabel || 'Starting…';
    // Audio captured while the model gets ready is kept (the last 20 s), which covers a load but not a
    // download of several minutes.
    hint = /^Downloading/.test(jobLabel)
      ? 'The model downloads first, which can take a few minutes. Speak once recording starts.'
      : 'What you say now is kept and sent once the model is ready.';
  } else if (recording) {
    text = 'Recording';
    hint = 'Press Stop when you are done.';
  } else if (phase === 'stopping') {
    text = 'Finishing…';
    hint = 'Getting the last words.';
  } else if (BLOCKED_HINT[blockedBy]) {
    hint = BLOCKED_HINT[blockedBy];
  }
  return {
    phase: text,
    buttonLabel: recording || phase === 'stopping' ? 'Stop recording' : 'Record',
    busy: BUSY.has(phase),
    disabled: BUSY.has(phase) || (!active && !!blockedBy),
    recording,
    active,
    showTimer: recording || phase === 'stopping',
    hint,
  };
}

/** PURE: the VAD pill text (TR-09): words, never colour alone. */
export function vadLabel(vad) {
  return vad === 'speech' ? 'Speech' : 'Silence';
}

/** PURE: a model id as people say it ("whisper base"). */
export function modelName(modelId) {
  if (!modelId) return '';
  return `whisper ${shortName({ id: modelId, type: 'stt', provider: 'faster-whisper' })}`;
}

/**
 * PURE: the meta line under the transcript: "Duration 2:13 · Processed in 4.1 s · whisper base" for a file,
 * "Recorded 0:42 · whisper base" for the microphone. Durations are formatted, never a raw float (BUG-17).
 * @param {object} transcript the reducer state
 * @returns {string}
 */
export function metaLine(transcript) {
  const t = transcript || {};
  const parts = [];
  if (t.source === 'file') {
    if (t.duration > 0) parts.push(`Duration ${clock(t.duration)}`);
    if (Number.isFinite(t.processingMs) && t.processingMs !== null) parts.push(`Processed in ${(t.processingMs / 1000).toFixed(1)} s`);
  } else if (t.source === 'mic' && t.duration > 0) {
    parts.push(`Recorded ${clock(t.duration)}`);
  }
  if (t.model) parts.push(modelName(t.model));
  return parts.join(' · ');
}

/** The Details text of an error: what its producer put there, trimmed. */
const detailOf = (e) => String(e.detail || '').trim();

/**
 * PURE: the inline error banner's content: an icon, the title, the plain message, the server text in
 * Details (escaped: html`` turns it into text), and the actions, each a button whose data-action is its
 * command (the view's dom.actions map).
 *
 * `detail` is shown as it comes. Its producers already make it readable: an HTTP failure's is
 * api.explain().detailText (the server's "message (code)", never its JSON envelope, and '' when it would only
 * repeat the message), a stream's is the close reason, and the view's own errors have none.
 * @param {{title: string, message?: string, detail?: string}} error
 * @param {{actions?: {action: string, label: string}[], tone?: 'error'|'warning'|'info', dismissAction?: string}} [options] dismissAction '' leaves the Dismiss button out
 */
export function bannerHtml(error, { actions = [], tone = 'error', dismissAction = 'dismiss-error' } = {}) {
  const e = error || {};
  const detail = detailOf(e);
  return html`${icon(tone === 'info' ? 'info' : 'alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${e.title || ''}</p>${e.message ? html`<p class="banner__msg">${e.message}</p>` : ''}${detail ? html`<details class="banner__details"><summary>Details</summary><pre>${detail}</pre></details>` : ''}</div>${actions.length ? html`<div class="banner__actions">${actions.map((a) => html`<button type="button" class="btn btn--secondary btn--sm" data-action="${a.action}"><span class="btn__label">${a.label}</span></button>`)}</div>` : ''}${dismissAction ? html`<button type="button" class="icon-btn icon-btn--sm" data-action="${dismissAction}" aria-label="Dismiss">${icon('x')}</button>` : ''}`;
}

/**
 * PURE: what a banner says, as one comparable value. A banner is written only when this changes, so a models
 * poll or a VAD change never rebuilds it under the keyboard focus, closes its Details or repeats its alert.
 * @param {{title?: string, message?: string, detail?: string}|null} error
 * @param {{actions?: {action: string, label: string}[], tone?: string, dismissAction?: string}} [options]
 *   the same options as bannerHtml
 * @returns {string} '' for no banner
 */
export function bannerKey(error, { actions = [], tone = 'error', dismissAction = 'dismiss-error' } = {}) {
  if (!error) return '';
  const e = error;
  return JSON.stringify([tone, e.title || '', e.message || '', detailOf(e), actions.map((a) => [a.action, a.label]), dismissAction]);
}

/** The label of the button that repeats what failed: "Retry", the one word every view uses. */
export const RETRY_LABEL = 'Retry';

/**
 * PURE: the actions a recording or upload error offers: Open Settings for a key problem (an API-key close
 * code included), Retry when retrying can help. A notice (a server error while the recording goes on)
 * offers nothing but Dismiss: Retry next to a running recording could only stop it.
 * @param {{action?: string, retryable?: boolean, code?: number, kind?: string}} error
 * @param {string} retryAction the data-action of Retry
 */
export function errorActions(error, retryAction) {
  const e = error || {};
  const out = [];
  if (e.kind === 'notice') return out;
  const keyProblem = e.action === 'open-settings' || e.code === 4001 || (e.code === 1008 && /api key/i.test(String(e.detail || '')));
  if (keyProblem) out.push({ action: 'open-settings', label: 'Open Settings' });
  // A refusal a retry can't change (an address not allowed; api.closeAction).
  else if (e.kind === 'close' && e.action === '') return out;
  else if (e.retryable || e.action === 'retry' || e.kind === 'close' || e.kind === 'stream') out.push({ action: retryAction, label: RETRY_LABEL });
  return out;
}

/**
 * PURE: an error as one toast line, for when it lands while another view shows: the title and what
 * happened ("Part of the recording wasn't transcribed. Transcription failed: CUDA out of memory"), the message
 * alone when it already starts with the title.
 * @param {{title?: string, message?: string}|null} error
 * @returns {string}
 */
export function toastLine(error) {
  const e = error || {};
  const title = String(e.title || '').trim();
  const message = String(e.message || '').trim();
  if (!message) return title || 'Recording stopped';
  if (!title || message.toLowerCase().startsWith(title.toLowerCase())) return message;
  return `${title}. ${message}`;
}
