/**
 * The only module (with theme-boot.js) that touches localStorage. Every call survives storage that is
 * missing or throws (private mode, blocked site data, quota), so no feature breaks because of it.
 * @module core/storage
 */

/** Every localStorage key the UI may use (exactly these). */
export const KEYS = Object.freeze({
  theme: 'open-speech-theme',              // 'system' | 'light' | 'dark'
  apiKey: 'open-speech-api-key',           // string
  micDevice: 'open-speech-mic-device',     // opaque deviceId only
  readerSpeed: 'open-speech-reader-speed', // '1.25': the default reading speed for new documents
  readerVoice: 'open-speech-reader-voice', // JSON {model, voice}
  prefs: 'open-speech-prefs',              // JSON, see prefs.js
  reading: 'open-speech-reading',          // JSON reading appearance, see prefs.js
  favorites: 'open-speech-voice-favorites',// JSON string[] of "model|voiceId"
  perf: 'open-speech-perf',                // JSON {[modelId]: {ttfaMs, renderMs, durationS, rtf, at}}
  speak: 'open-speech-speak',              // JSON last Speak setup
  speakDraft: 'open-speech-speak-draft',   // string, the unsent editor text
});

const KNOWN = new Set(Object.values(KEYS));

function store() {
  // Reading the property itself throws a SecurityError when site data is blocked.
  return globalThis.localStorage || null;
}

/**
 * Read a string value.
 * @param {string} key one of KEYS
 * @param {string|null} [fallback]
 * @returns {string|null} the stored string, or `fallback` when missing or unreadable. Never throws.
 */
export function read(key, fallback = null) {
  if (!KNOWN.has(key)) return fallback;
  try {
    const value = store()?.getItem(key);
    return value == null ? fallback : value;
  } catch {
    return fallback;
  }
}

/**
 * Write a string value.
 * @param {string} key one of KEYS
 * @param {*} value stored as String(value)
 * @returns {boolean} false when the key is unknown or storage refused the write. Never throws.
 */
export function write(key, value) {
  if (!KNOWN.has(key)) return false;
  try {
    const s = store();
    if (!s) return false;
    s.setItem(key, String(value));
    return true;
  } catch {
    return false;
  }
}

/**
 * Remove a key. Never throws.
 * @param {string} key one of KEYS
 */
export function remove(key) {
  if (!KNOWN.has(key)) return;
  try {
    store()?.removeItem(key);
  } catch {
    // unavailable storage: nothing to remove
  }
}

/**
 * Read a JSON value.
 * @param {string} key one of KEYS
 * @param {*} fallback returned when the key is missing, corrupt or invalid
 * @param {(value: *) => boolean} [validate] a corrupt or invalid value is removed and `fallback` returned
 * @returns {*}
 */
export function readJson(key, fallback, validate) {
  const raw = read(key, null);
  if (raw == null) return fallback;
  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    remove(key);
    return fallback;
  }
  if (typeof validate === 'function') {
    let ok = false;
    try { ok = !!validate(value); } catch { ok = false; }
    if (!ok) {
      remove(key);
      return fallback;
    }
  }
  return value;
}

/**
 * Write a JSON value.
 * @param {string} key one of KEYS
 * @param {*} value
 * @returns {boolean}
 */
export function writeJson(key, value) {
  let text;
  try {
    text = JSON.stringify(value);
  } catch {
    return false;
  }
  if (text === undefined) return false;
  return write(key, text);
}
