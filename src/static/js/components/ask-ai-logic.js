/**
 * "Ask AI": the pure parts of the shared component. No DOM, no
 * storage, no requests. The API they read and build for is POST /api/assistant/ask
 * (src/assistant/ask.py), GET /api/assistant, GET /api/assistant/presets and the Reader's job routes
 * (GET and DELETE /v1/reader/assistant-jobs/{id}).
 *
 * A source is what a view offers to send:
 *   {id, label, input: {kind: 'text'|'transcript'|'document', ...}, text?, chars, chapters?: number,
 *    title?, detail?, named?}
 * `input` goes into the request as it is; `text` (or `detail`) gives the sheet's short summary; `chars` its size;
 * `chapters` how many chapters a document part spans (more than one is always a job); `named` what goes with it
 * (the Reader's line naming the document and its chapter).
 * @module components/ask-ai-logic
 */

export const KINDS = Object.freeze(['text', 'transcript', 'document']);
/** The assistant's status and presets are loaded again once they are this old (GS-12). */
export const STALE_MS = 30000;
/** A job is read again this often while it runs. */
export const POLL_MS = 1500;
/** A job whose sheet was closed is read this often, only to tell when it ends. */
export const WATCH_MS = 10000;
/** The summary of what is sent: its first characters. */
export const SUMMARY_CHARS = 180;
/** The piece size when a preset doesn't say (src/assistant/presets.py's default, OS_ASSISTANT_CHUNK_CHARS). */
export const DEFAULT_CHUNK_CHARS = 12000;
/** The audio session of a spoken answer (one at a time; the next replaces it). */
export const CLIP_ID = 'ask-ai:answer';
/** A question of the listener's own: the server takes up to 2,000 characters, counted
 *  as Python counts them (code points; the box's maxlength counts UTF-16 units, so it always fits). */
export const QUESTION_MAX_CHARS = 2000;
/** How many of the last questions are offered again (kept in this browser, prefs.askAi.questions). */
export const RECENT_QUESTIONS = 3;
/** The quick picks under the question box: just text in it. */
export const QUICK_QUESTIONS = Object.freeze(['Explain this', 'Summarize', 'Define the hard words']);
/** A recent question's chip shows its first characters. */
export const QUESTION_CHIP_CHARS = 48;

export const COPY = Object.freeze({
  title: 'Ask AI',
  send: 'Send',
  asking: 'Asking the AI…',
  queued: 'Waiting for the AI…',
  running: 'The AI is working through it…',
  answered: 'Answered',
  cancelled: 'Canceled.',
  keptLong: 'A long text is kept in the Reader, where you can delete it.',
  keptAnswers: 'Its answers are saved in the Reader as a new document, where you can delete it.',
  longNote: "It's longer than one piece, so it goes as a job, one piece at a time. The answers are saved in the Reader as a new document; they aren't spoken here.",
  noOutput: 'Choose at least one: Show, Speak or Save.',
  copied: 'Answer copied',
  copyFailed: "Couldn't copy here: select the answer and copy it instead",
  savedToast: 'Saved in the Reader',
  serverDefault: 'Server default',
  noPresets: 'No AI preset suits this. Make one in Settings › AI assistant.',
  jobGoesOn: 'The AI job goes on. A notice says when it is done, and its answers are saved in the Reader.',
});

const nf = (n) => {
  try {
    return Number(n).toLocaleString('en-US');
  } catch {
    return String(n);
  }
};

/** PURE: the number of code points (the server counts characters as Python does). */
export function charCount(text) {
  let n = 0;
  for (const _ of String(text ?? '')) n += 1;   // eslint-disable-line no-unused-vars
  return n;
}

/** PURE: "1 character", "12,345 characters". */
export function sizeLabel(chars) {
  const n = Math.max(0, Math.floor(Number(chars) || 0));
  return `${nf(n)} ${n === 1 ? 'character' : 'characters'}`;
}

/** PURE: the first SUMMARY_CHARS characters of a text, its white space collapsed, with "…" when cut. */
export function summarize(text, max = SUMMARY_CHARS) {
  const words = String(text ?? '').replace(/\s+/g, ' ').trim();
  const points = Array.from(words);
  if (points.length <= max) return words;
  return `${points.slice(0, Math.max(1, max - 1)).join('').trimEnd()}…`;
}

/** PURE: the host part of a URL-ish string for the privacy line ("ai-server.lan:8080"), or '' when none. */
export function hostOf(status) {
  const host = status && typeof status.host === 'string' ? status.host.trim() : '';
  return host;
}

/**
 * PURE: the preset choices for an input kind, in the server's order: the presets whose `inputs` include the
 * kind; when none does, every preset (so there is always something to send with).
 * @param {object[]} presets GET /api/assistant/presets' `presets`
 * @param {string} kind
 * @returns {{id: string, label: string, name: string, output: string, chunk: number, text: string,
 *   showThinking: boolean}[]}
 */
export function presetChoices(presets, kind) {
  const list = (Array.isArray(presets) ? presets : []).filter((p) => p && typeof p.id === 'string' && p.id);
  const suits = (p) => Array.isArray(p.inputs) && p.inputs.includes(kind);
  const chosen = list.some(suits) ? list.filter(suits) : list;
  return chosen.map((p) => {
    const name = String(p.name || p.id);
    const label = String(p.label || name).trim() || name;
    return {
      id: p.id,
      label,
      name,
      output: ['text', 'speech', 'document'].includes(p.output) ? p.output : 'text',
      chunk: Number.isFinite(p.chunk_chars) && p.chunk_chars > 0 ? p.chunk_chars : DEFAULT_CHUNK_CHARS,
      text: label === name ? label : `${label} · ${name}`,
      showThinking: p.show_thinking !== false,
    };
  });
}

/**
 * PURE: the preset a sheet starts with: `wanted` while it is offered; else the server's own default for the
 * kind (`book` for a document, `voice` otherwise) when offered; else the first.
 * @param {{id: string}[]} choices presetChoices()
 * @param {string} kind
 * @param {string} [wanted]
 * @returns {string} '' when there is none
 */
export function defaultPreset(choices, kind, wanted = '') {
  const ids = (choices || []).map((c) => c.id);
  if (wanted && ids.includes(wanted)) return wanted;
  const server = kind === 'document' ? 'book' : 'voice';
  if (ids.includes(server)) return server;
  return ids[0] || '';
}

/** PURE: the outputs a preset starts with: Show always; Speak for `speech`, Save for `document`. */
export function defaultOutputs(choice) {
  const output = choice ? choice.output : 'text';
  return { show: true, speak: output === 'speech', save: output === 'document' };
}

/**
 * PURE: whether the source goes as a job (the server's rule: more than one piece of the preset's chunk_chars;
 * a document part over more than one chapter is more than one piece). The server decides; this is what the
 * sheet says before sending.
 */
export function isLong(source, choice) {
  if (!source) return false;
  const chunk = choice && choice.chunk > 0 ? choice.chunk : DEFAULT_CHUNK_CHARS;
  if (source.input && source.input.kind === 'document' && Number(source.chapters) > 1) return true;
  return Number(source.chars) > chunk;
}

/**
 * PURE: the privacy line: where it goes, and, for a long input, what the Reader keeps (by design,
 * a long text or transcript is kept as a Reader document with its answers).
 * @param {string} host
 * @param {{long?: boolean, kind?: string}} [options]
 */
export function privacyLine(host, { long = false, kind = 'text' } = {}) {
  const where = host ? `This is sent to ${host}.` : 'This is sent to the AI server.';
  if (!long) return where;
  return `${where} ${kind === 'document' ? COPY.keptAnswers : COPY.keptLong}`;
}

/**
 * PURE: the question as sent: the box's text without the white space around it, cut to
 * QUESTION_MAX_CHARS code points (never inside a pair); '' when it is blank (the server refuses a blank one).
 */
export function questionOf(text) {
  const q = String(text ?? '').trim();
  if (!q) return '';
  const points = Array.from(q);
  return points.length > QUESTION_MAX_CHARS ? points.slice(0, QUESTION_MAX_CHARS).join('').trimEnd() : q;
}

/** PURE: the recent questions once `question` was sent: it first, then the others, the newest RECENT_QUESTIONS. */
export function rememberQuestion(recent, question) {
  const list = (Array.isArray(recent) ? recent : []).filter((q) => typeof q === 'string' && q);
  if (!question) return list.slice(0, RECENT_QUESTIONS);
  return [question, ...list.filter((q) => q !== question)].slice(0, RECENT_QUESTIONS);
}

/** PURE: the recent questions the sheet offers: those that aren't a quick pick already. */
export function recentChoices(recent) {
  return (Array.isArray(recent) ? recent : []).filter((q) => typeof q === 'string' && q.trim() && !QUICK_QUESTIONS.includes(q));
}

/**
 * PURE: the body of POST /api/assistant/ask.
 * @param {{preset: string, source: object, outputs: {show: boolean, speak: boolean, save: boolean},
 *   voice?: {model?: string, voice?: string}|null, title?: string, question?: string}} options
 *   question: the listener's own, sent only when it isn't blank
 */
export function requestBody({ preset, source, outputs, voice = null, title = '', question = '' }) {
  const body = { input: { ...source.input } };
  if (preset) body.preset = preset;
  const asked = questionOf(question);
  if (asked) body.question = asked;
  const out = {};
  const o = outputs || { show: true };
  if (!o.show && (o.speak || o.save)) out.text = false;
  if (o.speak) {
    const speech = { format: 'mp3' };
    if (voice && voice.model) speech.model = String(voice.model);
    if (voice && voice.voice) speech.voice = String(voice.voice);
    out.speech = speech;
  }
  if (o.save) out.document = title ? { title: String(title).slice(0, 200) } : {};
  if (Object.keys(out).length) body.output = out;
  return body;
}

/** PURE: whether a body asks for something (Show, Speak or Save). */
export function hasOutput(outputs) {
  return !!(outputs && (outputs.show || outputs.speak || outputs.save));
}

const AI_FAILURES = Object.freeze({
  assistant_unreachable: ["Couldn't reach the AI server", 'Check that it is running, and that Settings › AI assistant › Connection points at it.'],
  assistant_timeout: ['The AI took too long to answer', 'Nothing came back in time. Try a shorter text, or raise the preset\'s timeout in Settings › AI assistant.'],
  assistant_http_error: ['The AI server returned an error', 'It refused the request.'],
  assistant_bad_response: ["The AI didn't send back an answer", "What came back isn't an answer Open Speech can read."],
  assistant_empty_reply: ['The AI sent back an empty answer', 'Nothing was left once the thinking was taken out.'],
  assistant_reply_cut_off: ['The answer was cut off', "It stopped while still thinking. Raise the preset's Max tokens, or lower its thinking."],
  assistant_not_started: ["The AI assistant hasn't started", 'Try again in a moment.'],
  assistant_disabled: ['The AI assistant is off', 'Turn it on in Settings › AI assistant.'],
  interrupted: ['The job stopped when the server restarted', 'Send it again: the pieces it finished are kept.'],
  document_not_found: ['That document is no longer in the library', 'It was deleted. Open another one.'],
  history_not_found: ['That history entry is gone', 'It was deleted.'],
  speech_unavailable: ["This server can't speak the answer", 'Turn off “Speak in a voice”, or update Open Speech.'],
});

const FIELD_TITLES = Object.freeze({
  'output.speech': "That voice wasn't accepted",
  'output.document': "The answer can't be saved here",
  'input.text': "The text wasn't accepted",
  'input.history_id': "The text wasn't accepted",
  'input.kind': "This can't be sent here",
  preset: "That preset wasn't accepted",
});

function sentence(text) {
  const s = String(text || '').trim();
  if (!s) return '';
  const first = s.charAt(0).toUpperCase() + s.slice(1);
  return /[.!?…]$/.test(first) ? first : `${first}.`;
}

/**
 * PURE: what a failed Send (or job) says, in plain words, by the server's code. The server's own message goes in
 * the details only when it adds something (never the AI server's answer: the server never echoes it).
 * @param {{status?: number, code?: string, kind?: string, message?: string, fields?: object[], detailText?: string}} error
 * @returns {{title: string, message: string, detail: string, action: ''|'retry'|'open-settings'|'open-ai-settings',
 *   retry: boolean}} `action` is the banner's settings button (or Try again); `retry` adds Try again beside it
 */
export function failureCopy(error) {
  const copy = failureCopyOf(error || {});
  return { ...copy, retry: copy.action === 'retry' || !!copy.retry };
}

function failureCopyOf(e) {
  const code = String(e.code || '');
  if (AI_FAILURES[code]) {
    const [title, message] = AI_FAILURES[code];
    if (code === 'assistant_unreachable') return { title, message, detail: '', action: 'open-ai-settings', retry: true };
    if (code === 'assistant_disabled') return { title, message, detail: '', action: 'open-ai-settings' };
    if (code === 'document_not_found' || code === 'history_not_found') return { title, message, detail: '', action: '' };
    return { title, message, detail: '', action: 'retry' };
  }
  if (e.kind === 'auth' || e.status === 401) {
    return { title: 'API key needed', message: e.message || "Enter the server's API key in Settings › Connection.", detail: '', action: 'open-settings' };
  }
  if (e.kind === 'locked') return { title: 'Too many wrong keys', message: e.message || '', detail: '', action: 'open-settings' };
  if (e.kind === 'network') return { title: "Can't reach Open Speech", message: 'Check that it is running, then try again.', detail: '', action: 'retry' };
  if (e.kind === 'timeout') return { title: "The server didn't answer", message: 'It may be busy. Try again.', detail: '', action: 'retry' };
  if (code === 'validation_error') {
    const first = (Array.isArray(e.fields) ? e.fields : [])[0] || {};
    const param = String(first.field || '');
    const key = Object.keys(FIELD_TITLES).find((p) => param === p || param.startsWith(`${p}.`));
    const title = key ? FIELD_TITLES[key] : (param.startsWith('input.') ? "That part can't be sent" : 'Some values were not accepted');
    return { title, message: sentence(first.message || e.detailText || '') || 'Some values were not accepted.', detail: '', action: '' };
  }
  if (e.status === 404) return { title: 'Not found', message: sentence(e.message) || 'It is no longer on the server.', detail: '', action: '' };
  if (e.status === 413) return { title: 'Too large', message: sentence(e.message), detail: '', action: '' };
  if (e.status >= 500 || e.status === 0) {
    return { title: 'Something went wrong on the server', message: 'Try again in a moment.', detail: String(e.detailText || e.message || ''), action: 'retry' };
  }
  return { title: 'Something went wrong', message: sentence(e.message) || 'Try again.', detail: '', action: 'retry' };
}

const AUDIO_NOTES = Object.freeze({
  speech_too_long: 'The answer is too long to speak here. Save it as a document to listen to it in the Reader.',
  speech_empty: 'The answer had nothing to speak.',
  speech_invalid: "The answer couldn't be spoken.",
  tts_disabled: 'Speech is turned off on this server.',
  provider_missing: "That voice's model isn't available on this server.",
  insufficient_gpu_memory: "There wasn't enough GPU memory to speak it. Try again once other models are unloaded.",
});

/** PURE: why a spoken answer has no sound (`audio_error: {code, message}`), in plain words. */
export function audioNote(audioError) {
  if (!audioError) return '';
  const code = String(audioError.code || '');
  if (AUDIO_NOTES[code]) return AUDIO_NOTES[code];
  const message = sentence(audioError.message);
  // The server's words as written: a first word is often a name or an acronym ("Kokoro's", "TTS", "CUDA").
  return message ? `Couldn't speak the answer: ${message}` : "Couldn't speak the answer.";
}

/** PURE: why a saved answer has no document (`document_error`). */
export function documentNote(documentError) {
  if (!documentError) return '';
  return "The answer had nothing to save as a document.";
}

/**
 * PURE: a failed answer's body (ApiError.body of a 502 or 504, parsed and never cut): the
 * spoken error sentence and its audio, when the preset speaks errors, and the AI's time. A JSON text is parsed
 * too (a hand-made error); a cut one doesn't parse, and gives null.
 * @param {object|string|null} parsed
 * @returns {{sentence: string, audio: object|null, audioError: object|null, elapsedMs: number|null}|null}
 */
export function failureBody(parsed) {
  let body = parsed;
  if (typeof parsed === 'string') {
    try {
      body = JSON.parse(parsed);
    } catch {
      return null;
    }
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) return null;
  return {
    sentence: typeof body.sentence === 'string' ? body.sentence : '',
    audio: audioOf(body.audio),
    audioError: body.audio_error && typeof body.audio_error === 'object' ? body.audio_error : null,
    elapsedMs: Number.isFinite(body.elapsed_ms) ? body.elapsed_ms : null,
  };
}

/** PURE: a usable `audio` field ({base64, content_type, format}) or null. */
export function audioOf(audio) {
  if (!audio || typeof audio !== 'object' || typeof audio.base64 !== 'string' || !audio.base64) return null;
  const type = typeof audio.content_type === 'string' && /^audio\/[\w.+-]+$/.test(audio.content_type) ? audio.content_type : 'audio/mpeg';
  return { base64: audio.base64, contentType: type, format: String(audio.format || ''), bytes: Number(audio.bytes) || 0 };
}

/** PURE: base64 to bytes (atob, then a byte per char); throws on bad input. */
export function base64ToBytes(b64) {
  const binary = atob(String(b64));
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

/**
 * PURE: a 200 answer as the sheet shows it.
 * @returns {{reply: string, thinking: string, elapsedMs: number, finishReason: string, audio: object|null,
 *   audioNote: string, documentId: string, documentNote: string, preset: string}}
 */
export function answerOf(data) {
  const d = data && typeof data === 'object' ? data : {};
  return {
    reply: typeof d.reply === 'string' ? d.reply : '',
    thinking: typeof d.thinking === 'string' ? d.thinking : '',
    elapsedMs: Number.isFinite(d.elapsed_ms) ? d.elapsed_ms : 0,
    finishReason: typeof d.finish_reason === 'string' ? d.finish_reason : '',
    audio: audioOf(d.audio),
    audioNote: audioNote(d.audio_error),
    documentId: typeof d.document_id === 'string' ? d.document_id : '',
    documentNote: documentNote(d.document_error),
    preset: typeof d.preset === 'string' ? d.preset : '',
  };
}

/** PURE: a job's pieces done and in all: {done, total, label} ("3 of 12 pieces"). */
export function jobProgress(job) {
  const chapters = job && Array.isArray(job.chapters) ? job.chapters : [];
  let done = 0;
  let total = 0;
  for (const c of chapters) {
    const pieces = Math.max(0, Number(c && c.pieces) || 0);
    total += pieces;
    done += Math.min(pieces, Math.max(0, Number(c && c.pieces_done) || 0));
  }
  return { done, total, label: total ? `${nf(done)} of ${nf(total)} ${total === 1 ? 'piece' : 'pieces'}` : '' };
}

/** PURE: whether a job has ended. */
export function jobEnded(job) {
  return !!job && ['done', 'failed', 'cancelled'].includes(job.status);
}

/** PURE: a finished job's answer: each chapter's text, joined with blank lines. */
export function jobText(job) {
  const chapters = job && Array.isArray(job.chapters) ? job.chapters : [];
  return chapters.map((c) => (c && typeof c.text === 'string' ? c.text.trim() : '')).filter(Boolean).join('\n\n');
}

/** PURE: a job's thinking, joined the same way (when the preset shows it). */
export function jobThinking(job) {
  const chapters = job && Array.isArray(job.chapters) ? job.chapters : [];
  return chapters.map((c) => (c && typeof c.thinking === 'string' ? c.thinking.trim() : '')).filter(Boolean).join('\n\n');
}

/** PURE: what a running job's status line says. */
export function jobStatus(job) {
  if (!job) return COPY.asking;
  if (job.status === 'queued') return COPY.queued;
  if (job.status === 'running') return COPY.running;
  if (job.status === 'done') return COPY.answered;
  if (job.status === 'cancelled') return COPY.cancelled;
  return 'Stopped';
}

/** PURE: a failed job's copy (its `error` code and `error_message`). */
export function jobFailure(job) {
  const copy = failureCopy({ code: job && job.error, message: job && job.error_message, status: 502 });
  return copy;
}

/**
 * PURE: what the toast says when a job whose sheet was closed ends, by the job's
 * own name: done ("Its answers are in the Reader."), or failed in plain words by its code (never the AI server's
 * own message).
 */
export function jobEndText(job) {
  const title = job && typeof job.title === 'string' ? job.title.trim() : '';
  const named = title ? `“${title}”` : '';
  if (job && job.status === 'done') return `The AI finished ${named || 'the job'}. Its answers are in the Reader.`;
  const copy = jobFailure(job);
  const why = sentence(copy.title);
  return `${named ? `The AI job ${named}` : 'The AI job'} stopped: ${why}${copy.message ? ` ${copy.message}` : ''}`;
}

/** PURE: a time taken: "850 ms", "1.2 s", "61 s". */
export function elapsedText(ms) {
  const n = Math.max(0, Number(ms) || 0);
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(Math.round(n / 100) / 10).toFixed(1)} s`;
  return `${Math.round(n / 1000)} s`;
}

/** PURE: the answer's line under it ("Answer · 1.8 s"). */
export function metaLine({ presetText = '', elapsedMs = 0, finishReason = '' } = {}) {
  const parts = [];
  if (presetText) parts.push(presetText);
  if (elapsedMs) parts.push(elapsedText(elapsedMs));
  if (finishReason === 'length') parts.push('cut off at the token limit');
  return parts.join(' · ');
}

/** PURE: the name a saved answer gets when Save comes after the answer ("Answer: the first words…"). */
export function savedTitle(presetText, source) {
  const head = summarize((source && (source.title || source.text || source.label)) || '', 60);
  const name = String(presetText || 'AI answer');
  return (head ? `${name}: ${head}` : name).slice(0, 200);
}
