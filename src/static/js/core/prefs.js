/**
 * User preferences (KEYS.prefs) and reading appearance (KEYS.reading). Every field is validated on its own:
 * a bad or missing field falls back to its default, never the whole object.
 * @module core/prefs
 */
import { KEYS, readJson, writeJson } from './storage.js';
import { app } from './store.js';

/**
 * Talk's choices, one object inside the preferences, versioned and validated field by
 * field like them. A remembered preset or voice that no longer exists is Talk's to fall back from: it stays stored
 * until the listener chooses again.
 */
export const TALK_DEFAULTS = Object.freeze({
  v: 1,
  preset: 'voice',                 // the AI preset's id (GET /api/assistant/presets)
  voice: null,                     // the reply voice {model, voice, label}, or null: Speak's
  interrupt: true,                 // Interrupt by talking (turn_detection.interrupt_response)
});

/**
 * The Ask AI sheet's choices, one object inside the preferences like Talk's. `presets`
 * and `questions` are replaced whole by a patch (the sheet sends the merged map and list). A remembered preset or
 * voice that no longer exists is the sheet's to fall back from: it stays stored until the listener chooses again.
 */
export const ASK_AI_DEFAULTS = Object.freeze({
  v: 1,
  presets: Object.freeze({}),      // the preset id last sent per input kind: {text?, transcript?, document?}
  outputs: null,                   // the outputs last sent {kind, preset, show, speak, save}, or null: the preset's own
  voice: null,                     // the answer voice {model, voice, label}, or null: Speak's
  questions: Object.freeze([]),    // the last questions sent, newest first
});

/** The recent questions kept, each at most the server's 2,000 characters. */
const ASK_RECENT_QUESTIONS = 3;
const ASK_QUESTION_MAX_CHARS = 2000;

export const PREF_DEFAULTS = Object.freeze({
  v: 1,
  reduceMotion: 'system',          // 'system' | 'reduce' | 'full'
  shortcuts: true,                 // single-key shortcuts on (WCAG 2.1.4 toggle)
  lastRoute: '',                   // e.g. 'speak/live', 'reader/<id>'
  skipBack: 15,                    // seconds: one of 5, 10, 15, 30, 60
  skipForward: 30,                 // seconds: one of 10, 15, 30, 60
  stepUnit: 'paragraph',           // prev/next buttons: 'paragraph' | 'sentence'
  trackUnit: 'paragraph',          // lock-screen previous/next track: 'paragraph' | 'chapter'
  snapToSentence: true,            // time skips land on a sentence start
  sleepFade: true,                 // fade the last 10 s of a sleep timer
  wakeLock: false,                 // keep the screen on while the reading view plays
  showInterim: true,
  compactTranscript: false,
  echoCancellation: true,
  noiseSuppression: true,
  librarySort: 'recent',           // 'recent' | 'title' | 'progress'
  libraryFilter: 'all',            // 'all' | 'progress' | 'new' | 'done'
  historyFilter: '',               // '' | 'tts' | 'stt'
  liveMode: 'natural',             // Live's latency_mode: 'natural' | 'responsive' | 'instant_word'
  showLive: false,                 // Speak › Live, a streaming test for developers (Settings › Developer)
  talk: TALK_DEFAULTS,             // Talk's AI preset, reply voice and Interrupt by talking
  askAi: ASK_AI_DEFAULTS,          // Ask AI's preset per input kind, outputs and answer voice
});

export const READING_DEFAULTS = Object.freeze({
  v: 1,
  size: null,                      // null = responsive default, else a step 0..6
  face: 'serif',                   // 'serif' | 'sans' | 'legible'
  leading: 'normal',               // 'compact' | 'normal' | 'relaxed'
  measure: 'medium',               // 'narrow' | 'medium' | 'wide'
  paper: 'auto',                   // 'auto' | 'paper' | 'sepia' | 'night' | 'dusk'
  justify: false,
  highlight: 'both',               // 'tint' | 'underline' | 'both' | 'off'
  autoScroll: true,
});

const oneOf = (...values) => (v) => values.includes(v);
const isBool = (v) => typeof v === 'boolean';
const isPlainObject = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
const isText = (max) => (v) => typeof v === 'string' && v.length > 0 && v.length <= max;

const isPresetId = isText(64);     // the server's preset ids are at most 40 characters
/** A chosen voice {model, voice, label?}, or null (follow Speak's). */
const isVoiceChoice = (v) => v === null || (isPlainObject(v) && isText(512)(v.model) && isText(1024)(v.voice)
  && (v.label === undefined || (typeof v.label === 'string' && v.label.length <= 512)));
/** A voice as kept: only {model, voice, label} (the label defaults to the voice), frozen. */
const tidyVoice = (v) => Object.freeze({ model: v.model, voice: v.voice, label: v.label || v.voice });

const TALK_RULES = {
  v: (v) => v === 1,
  preset: isPresetId,
  voice: isVoiceChoice,
  interrupt: isBool,
};

/** Talk's object as kept: frozen, its voice only {model, voice, label} (the label defaults to the voice). */
function tidyTalk(talk) {
  const out = { ...talk };
  if (out.voice) out.voice = tidyVoice(out.voice);
  return Object.freeze(out);
}

const ASK_KINDS = Object.freeze(['text', 'transcript', 'document']);   // POST /api/assistant/ask's input kinds

const ASK_AI_RULES = {
  v: (v) => v === 1,
  presets: isPlainObject,          // then kind by kind (tidyAskAi): a bad kind or id is dropped alone
  outputs: (v) => v === null || (isPlainObject(v) && ASK_KINDS.includes(v.kind) && isPresetId(v.preset)
    && isBool(v.show) && isBool(v.speak) && isBool(v.save)),
  voice: isVoiceChoice,
  questions: Array.isArray,        // then one by one (tidyAskAi): a bad entry is dropped alone
};

/** PURE: the recent questions as kept: text of 1 to 2,000 characters that isn't blank, the first of a repeat, the
 *  newest three. */
function tidyQuestions(list) {
  const out = [];
  for (const q of Array.isArray(list) ? list : []) {
    if (typeof q !== 'string' || !q.trim() || q.length > ASK_QUESTION_MAX_CHARS || out.includes(q)) continue;
    out.push(q);
    if (out.length === ASK_RECENT_QUESTIONS) break;
  }
  return Object.freeze(out);
}

/** Ask AI's object as kept: frozen, its preset map only valid kinds and ids, its outputs and voice only their fields,
 *  its questions only valid ones. */
function tidyAskAi(ask) {
  const out = { ...ask };
  const presets = {};
  for (const kind of ASK_KINDS) if (out.presets && isPresetId(out.presets[kind])) presets[kind] = out.presets[kind];
  out.presets = Object.freeze(presets);
  out.questions = tidyQuestions(out.questions);
  if (out.outputs) {
    const o = out.outputs;
    out.outputs = Object.freeze({ kind: o.kind, preset: o.preset, show: o.show, speak: o.speak, save: o.save });
  }
  if (out.voice) out.voice = tidyVoice(out.voice);
  return Object.freeze(out);
}

/** Fields holding an object of their own, validated field by field with its own defaults and rules. */
const PREF_OBJECTS = {
  talk: { defaults: TALK_DEFAULTS, rules: TALK_RULES, tidy: tidyTalk },
  askAi: { defaults: ASK_AI_DEFAULTS, rules: ASK_AI_RULES, tidy: tidyAskAi },
};

const PREF_RULES = {
  v: (v) => v === 1,
  reduceMotion: oneOf('system', 'reduce', 'full'),
  shortcuts: isBool,
  lastRoute: (v) => typeof v === 'string' && v.length <= 512,
  skipBack: oneOf(5, 10, 15, 30, 60),
  skipForward: oneOf(10, 15, 30, 60),
  stepUnit: oneOf('paragraph', 'sentence'),
  trackUnit: oneOf('paragraph', 'chapter'),
  snapToSentence: isBool,
  sleepFade: isBool,
  wakeLock: isBool,
  showInterim: isBool,
  compactTranscript: isBool,
  echoCancellation: isBool,
  noiseSuppression: isBool,
  librarySort: oneOf('recent', 'title', 'progress'),
  libraryFilter: oneOf('all', 'progress', 'new', 'done'),
  historyFilter: oneOf('', 'tts', 'stt'),
  liveMode: oneOf('natural', 'responsive', 'instant_word'),   // the server's LiveTextSegmenter.VALID_MODES
  showLive: isBool,
  talk: isPlainObject,             // then field by field (PREF_OBJECTS)
  askAi: isPlainObject,            // then field by field (PREF_OBJECTS)
};

const READING_RULES = {
  v: (v) => v === 1,
  size: (v) => v === null || (Number.isInteger(v) && v >= 0 && v <= 6),
  face: oneOf('serif', 'sans', 'legible'),
  leading: oneOf('compact', 'normal', 'relaxed'),
  measure: oneOf('narrow', 'medium', 'wide'),
  paper: oneOf('auto', 'paper', 'sepia', 'night', 'dusk'),
  justify: isBool,
  highlight: oneOf('tint', 'underline', 'both', 'off'),
  autoScroll: isBool,
};

function clean(source, defaults, rules, objects = {}) {
  const out = { ...defaults };
  if (!isPlainObject(source)) return out;
  for (const key of Object.keys(rules)) {
    if (!(key in source) || !rules[key](source[key])) continue;
    const nested = objects[key];
    out[key] = nested ? nested.tidy(clean(source[key], nested.defaults, nested.rules)) : source[key];
  }
  return out;
}

function applyPatch(current, patch, rules, objects = {}) {
  const next = { ...current };
  if (!isPlainObject(patch)) return next;
  for (const [key, value] of Object.entries(patch)) {
    if (key === 'v') continue;
    if (!rules[key] || !rules[key](value)) continue;
    const nested = objects[key];
    // An object field takes a patch of its own: its valid fields merge into what it holds.
    next[key] = nested ? nested.tidy(applyPatch(current[key], value, nested.rules)) : value;
  }
  return next;
}

/** Read both objects from storage into app.prefs and app.reading, validating each field. */
export function load() {
  const prefs = Object.freeze(clean(readJson(KEYS.prefs, null, isPlainObject), PREF_DEFAULTS, PREF_RULES, PREF_OBJECTS));
  const reading = Object.freeze(clean(readJson(KEYS.reading, null, isPlainObject), READING_DEFAULTS, READING_RULES));
  app.set({ prefs, reading });
  return prefs;
}

function ensureLoaded() {
  const { prefs, reading } = app.get();
  if (!prefs || !('v' in prefs) || !reading || !('v' in reading)) load();
}

/** @returns {typeof PREF_DEFAULTS} the current preferences (frozen). */
export function get() {
  ensureLoaded();
  return app.get().prefs;
}

/**
 * Merge valid fields into the preferences and persist them. Invalid fields are ignored.
 * @param {Partial<typeof PREF_DEFAULTS>} patch
 * @returns {typeof PREF_DEFAULTS}
 */
export function set(patch) {
  const next = Object.freeze(applyPatch(get(), patch, PREF_RULES, PREF_OBJECTS));
  writeJson(KEYS.prefs, next);
  app.set({ prefs: next });
  return next;
}

/** @returns {typeof READING_DEFAULTS} the reading appearance (frozen). */
export function getReading() {
  ensureLoaded();
  return app.get().reading;
}

/**
 * Merge valid fields into the reading appearance and persist them.
 * @param {Partial<typeof READING_DEFAULTS>} patch
 * @returns {typeof READING_DEFAULTS}
 */
export function setReading(patch) {
  const next = Object.freeze(applyPatch(getReading(), patch, READING_RULES));
  writeJson(KEYS.reading, next);
  app.set({ reading: next });
  return next;
}

/** Restore the reading appearance defaults (the appearance sheet's Reset). */
export function resetReading() {
  const next = Object.freeze({ ...READING_DEFAULTS });
  writeJson(KEYS.reading, next);
  app.set({ reading: next });
  return next;
}
