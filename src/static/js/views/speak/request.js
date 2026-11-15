/**
 * Speak: request building. Pure: no DOM, no network, so tests import it directly.
 *
 *   buildSpeechRequest(s) -> {url, body, headers, stream}
 *
 * The body keeps today's shape (inventory SP-17): {model, voice, input, speed, response_format, effects},
 * then `instructions` only when non-empty, `voice_library_ref` when a reference recording is chosen, and each
 * option the model declares (capabilities.options: Supertonic's `steps`, Breeze's `cfg_scale` and `temperature`).
 * A request streams only when every condition holds; with an effect it never does, because the server
 * answers 400 to stream=true with effects (REV-01).
 * @module views/speak/request
 */

/** The formats the server accepts (VALID_TTS_RESPONSE_FORMATS), in menu order. */
export const FORMATS = Object.freeze([
  Object.freeze({ value: 'mp3', label: 'MP3' }),
  Object.freeze({ value: 'wav', label: 'WAV' }),
  Object.freeze({ value: 'flac', label: 'FLAC' }),
  Object.freeze({ value: 'opus', label: 'Opus' }),
  Object.freeze({ value: 'aac', label: 'AAC' }),
  Object.freeze({ value: 'm4a', label: 'M4A' }),
  Object.freeze({ value: 'pcm', label: 'PCM (download only)' }),
]);

/** What to ask canPlayType() for each format (PCM is raw samples: no browser plays it). */
export const FORMAT_MIME = Object.freeze({
  mp3: 'audio/mpeg', wav: 'audio/wav', flac: 'audio/flac', opus: 'audio/ogg; codecs=opus', aac: 'audio/aac',
  m4a: 'audio/mp4', pcm: '',
});

/** The effect types in today's payload order (normalize, podcast_eq, reverb, pitch, robot). */
export const EFFECT_TYPES = Object.freeze(['normalize', 'podcast_eq', 'reverb', 'pitch', 'robot']);
export const EFFECT_LABELS = Object.freeze({ normalize: 'Normalize', podcast_eq: 'Podcast EQ', reverb: 'Reverb', pitch: 'Pitch', robot: 'Robot' });
export const REVERB_ROOMS = Object.freeze(['small', 'medium', 'large']);
export const PITCH_MIN = -12;
export const PITCH_MAX = 12;

export const SPEED_MIN = 0.5;
export const SPEED_MAX = 2;
export const SPEED_STEP = 0.05;
export const DEFAULT_LIMIT = 4096;

/** The hints the Stream switch shows when streaming can't work. */
export const STREAM_HINTS = Object.freeze({
  format: 'Streaming plays MP3 only',
  effects: 'Effects are applied after the full render',
  browser: "This browser can't stream; the take plays when it's ready",
});

/** A fresh, empty effects state. */
export function emptyEffects() {
  return { normalize: false, podcast_eq: false, reverb: { on: false, room: 'small' }, pitch: { on: false, semitones: 0 }, robot: false };
}

function clampSemitones(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return 0;
  return Math.min(PITCH_MAX, Math.max(PITCH_MIN, n));
}

/** Whether an effect is switched on in an effects state. */
export function effectOn(effects, type) {
  const value = effects ? effects[type] : false;
  return value && typeof value === 'object' ? !!value.on : !!value;
}

/**
 * The `effects` payload: an array in EFFECT_TYPES order, or null when none is on (today's
 * buildEffectsPayload). Reverb carries `room`, pitch an integer `semitones` (−12…12).
 * @param {object} effects an effects state (emptyEffects())
 * @returns {object[]|null}
 */
export function effectsPayload(effects) {
  const out = [];
  for (const type of EFFECT_TYPES) {
    if (!effectOn(effects, type)) continue;
    const fx = { type };
    if (type === 'reverb') fx.room = REVERB_ROOMS.includes(effects.reverb.room) ? effects.reverb.room : 'small';
    if (type === 'pitch') fx.semitones = clampSemitones(effects.pitch.semitones);
    out.push(fx);
  }
  return out.length ? out : null;
}

/**
 * The effects state for a saved list (a preset's or a history entry's `effects`): the inverse of
 * effectsPayload(). Strings ('normalize') and objects ({type, room, semitones}) are both accepted; unknown
 * types are ignored.
 * @param {Array<object|string>|null} list
 */
export function effectsState(list) {
  const state = emptyEffects();
  for (const item of Array.isArray(list) ? list : []) {
    const type = typeof item === 'string' ? item : item && item.type;
    if (!EFFECT_TYPES.includes(type)) continue;
    if (type === 'reverb') state.reverb = { on: true, room: REVERB_ROOMS.includes(item?.room) ? item.room : 'small' };
    else if (type === 'pitch') state.pitch = { on: true, semitones: clampSemitones(item?.semitones ?? 0) };
    else state[type] = true;
  }
  return state;
}

/** The slider's speed range: 0.5–2.0, narrowed to the model's own (capabilities.speed_range, Supertonic's 0.7–1.9). */
export function speedRange(caps) {
  const range = (caps && caps.speed_range) || {};
  const low = Number(range.min);
  const high = Number(range.max);
  return {
    min: Number.isFinite(low) && range.min !== null ? Math.max(SPEED_MIN, low) : SPEED_MIN,
    max: Number.isFinite(high) && range.max !== null ? Math.min(SPEED_MAX, high) : SPEED_MAX,
  };
}

/** Round a speed to the slider's 0.05 grid inside 0.5–2.0, or inside the model's range when `caps` has one. */
export function clampSpeed(value, caps) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 1;
  const { min, max } = speedRange(caps);
  const stepped = Math.round(n / SPEED_STEP) * SPEED_STEP;
  return Math.round(Math.min(max, Math.max(min, stepped)) * 100) / 100;
}

const given = (value) => value !== null && value !== undefined && value !== '';

/**
 * An option a model declares (capabilities.options[name]: {type, label, min, max, default}), held to its range;
 * its default when `value` isn't a number, and undefined for an option whose default is null (Auto: not sent).
 * An integer option is rounded.
 */
export function optionValue(spec, value) {
  const fallback = spec && given(spec.default) ? Number(spec.default) : NaN;
  let n = given(value) ? Number(value) : NaN;
  if (!Number.isFinite(n)) n = fallback;
  if (!Number.isFinite(n)) return undefined;
  if (spec && spec.type === 'integer') n = Math.round(n);
  if (Number.isFinite(Number(spec?.min))) n = Math.max(Number(spec.min), n);
  if (Number.isFinite(Number(spec?.max))) n = Math.min(Number(spec.max), n);
  return n;
}

/** The values of the options a model declares, from `values` (each held to its range) or their defaults; Auto ones are left out. */
export function modelOptions(caps, values = {}) {
  const declared = caps && caps.options && typeof caps.options === 'object' ? caps.options : {};
  const out = {};
  for (const [name, spec] of Object.entries(declared)) {
    const value = optionValue(spec, values ? values[name] : undefined);
    if (value !== undefined) out[name] = value;
  }
  return out;
}

/**
 * The string Speak sends in `voice` (today's selectedSpeechVoice): a named voice is `voice:<id>`, a blend
 * its canonical recipe, else the provider voice, else the reference recording's name (a clone model with no
 * stock voices).
 * @param {{identity?: string, blendSpec?: string, voice?: string, reference?: string}} s
 */
export function voiceSpec({ identity = '', blendSpec = '', voice = '', reference = '' } = {}) {
  if (identity) return `voice:${identity}`;
  if (blendSpec) return blendSpec;
  return voice || reference || '';
}

/**
 * Why a request can't stream ('' when it can): 'unsupported' (the model has no streaming, or the switch is
 * off), 'format', 'effects' or 'browser'. The switch's hint shows the first three reasons it knows.
 * @param {{caps?: object, format?: string, effects?: object[]|null, mse?: boolean, on?: boolean}} s
 */
export function streamBlocker({ caps = {}, format = 'mp3', effects = null, mse = false, on = true } = {}) {
  if (!caps || caps.streaming !== true) return 'unsupported';
  if (format !== 'mp3') return 'format';
  if (effects && effects.length) return 'effects';
  if (!mse) return 'browser';
  if (!on) return 'off';
  return '';
}

/** @returns {boolean} whether this browser can stream MP3 through Media Source Extensions */
export function mseSupported(MediaSourceCtor = globalThis.MediaSource) {
  try {
    return !!MediaSourceCtor && typeof MediaSourceCtor.isTypeSupported === 'function' && MediaSourceCtor.isTypeSupported('audio/mpeg');
  } catch {
    return false;
  }
}

/**
 * Build the POST /v1/audio/speech request for a settings snapshot.
 * @param {{model: string, voice: string, input: string, speed?: number, format?: string,
 *   effects?: object[]|null, instructions?: string, reference?: string, caps?: object, stream?: boolean,
 *   mse?: boolean, options?: object}} s `voice` is voiceSpec(); `effects` is effectsPayload(); `stream` is the switch;
 *   `options` the model's own options ({steps: 8}, {cfg_scale: 2.5}), sent only when the model declares them
 * @returns {{url: string, body: object, headers: Record<string, string>, stream: boolean}}
 */
export function buildSpeechRequest(s) {
  const caps = s.caps || {};
  const format = s.format || 'mp3';
  const effects = Array.isArray(s.effects) && s.effects.length ? s.effects : null;
  const reference = s.reference || '';
  const body = {
    model: s.model,
    voice: s.voice || reference || '',
    input: String(s.input ?? '').trim(),
    speed: caps.speed_control === false ? 1 : clampSpeed(s.speed ?? 1, caps),
    response_format: format,
    effects,
  };
  const instructions = String(s.instructions ?? '').trim();
  if (instructions && caps.instructions !== false) body.instructions = instructions;
  if (reference) body.voice_library_ref = reference;
  Object.assign(body, modelOptions(caps, s.options));
  const stream = streamBlocker({ caps, format, effects, mse: !!s.mse, on: s.stream !== false }) === '';
  return {
    url: stream ? '/v1/audio/speech?stream=true' : '/v1/audio/speech',
    body,
    headers: { 'Content-Type': 'application/json', 'X-History': 'true' },
    stream,
  };
}

/**
 * What happens to a finished take: 'play'; 'download-pcm' (raw PCM is never played); 'download' (this
 * browser can't play the format, REV-33).
 * @param {string} format
 * @param {(mime: string) => boolean} canPlay
 */
export function playbackPlan(format, canPlay) {
  if (format === 'pcm') return 'download-pcm';
  const mime = FORMAT_MIME[format] || '';
  if (!mime) return 'download';
  try {
    return canPlay(mime) ? 'play' : 'download';
  } catch {
    return 'download';
  }
}

/** The hint for a take that was downloaded instead of played. */
export function downloadHint(format) {
  if (format === 'pcm') return 'PCM is raw audio, so it was downloaded.';
  return `This browser can't play ${formatLabel(format)}, so it was downloaded.`;
}

/** 'open-speech-<timestamp>.<format>' (SP-21: the format of the audio generated, not the one picked since). */
export function takeFilename(format, at) {
  return `open-speech-${Math.round(Number(at) || 0)}.${format || 'mp3'}`;
}

/** The format's label ('MP3', 'PCM'). */
export function formatLabel(format) {
  return format === 'pcm' ? 'PCM' : (FORMATS.find((f) => f.value === format)?.label || String(format || '').toUpperCase());
}
