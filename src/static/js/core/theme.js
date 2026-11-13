/**
 * Theme (System / Light / Dark, following the OS by default) and motion (System / Reduce / Full).
 * theme-boot.js applies both before first paint; this module keeps them in step afterwards.
 * @module core/theme
 */
import { KEYS, read, write } from './storage.js';
import * as prefs from './prefs.js';

/** The page backgrounds the theme-color metas use (tokens.css --bg). */
export const THEME_COLORS = Object.freeze({ light: '#F7F7F4', dark: '#0E0F13' });

const THEMES = ['system', 'light', 'dark'];
const MOTIONS = ['system', 'reduce', 'full'];

let listeners = new Set();
let darkQuery = null;
let onSchemeChange = null;
let override = { theme: '', motion: '' };   // ?theme= / ?motion= for this page load only

function queryOverride() {
  const out = { theme: '', motion: '' };
  try {
    const params = new URLSearchParams(location.search);
    const t = params.get('theme');
    const m = params.get('motion');
    if (THEMES.includes(t)) out.theme = t;
    if (MOTIONS.includes(m)) out.motion = m;
  } catch {
    // no location: nothing to override
  }
  return out;
}

function storedTheme() {
  const value = read(KEYS.theme, '');
  return THEMES.includes(value) ? value : 'system';
}

function prefersDark() {
  try {
    return !!(typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches);
  } catch {
    return false;
  }
}

function applyTheme(choice) {
  const root = document.documentElement;
  if (choice === 'light' || choice === 'dark') root.setAttribute('data-theme', choice);
  else root.removeAttribute('data-theme');
  const metas = document.querySelectorAll('meta[name="theme-color"]');
  for (const meta of metas) {
    const media = meta.getAttribute('media') || '';
    const schemeOfMeta = /dark/.test(media) ? 'dark' : 'light';
    meta.setAttribute('content', THEME_COLORS[choice === 'system' ? schemeOfMeta : choice]);
  }
}

function applyMotion(choice) {
  const root = document.documentElement;
  if (choice === 'reduce' || choice === 'full') root.setAttribute('data-motion', choice);
  else root.removeAttribute('data-motion');
}

function notify() {
  const effective = effectiveTheme();
  const preference = getTheme();
  for (const cb of [...listeners]) {
    try { cb(effective, preference); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
  }
}

/** Apply the saved (or query-forced) theme and motion, and follow OS colour-scheme changes. */
export function init() {
  override = queryOverride();
  applyTheme(getTheme());
  applyMotion(getMotion());
  if (!darkQuery && typeof matchMedia === 'function') {
    try {
      darkQuery = matchMedia('(prefers-color-scheme: dark)');
      onSchemeChange = () => { if (getTheme() === 'system') notify(); };
      if (typeof darkQuery.addEventListener === 'function') darkQuery.addEventListener('change', onSchemeChange);
      else if (typeof darkQuery.addListener === 'function') darkQuery.addListener(onSchemeChange);
    } catch {
      darkQuery = null;
    }
  }
}

/** @returns {'system'|'light'|'dark'} the theme choice (a ?theme= override wins for this page load) */
export function getTheme() {
  return override.theme || storedTheme();
}

/**
 * Save and apply a theme choice: KEYS.theme, <html data-theme> and both theme-color metas.
 * @param {'system'|'light'|'dark'} theme
 */
export function setTheme(theme) {
  if (!THEMES.includes(theme)) throw new TypeError(`Unknown theme: ${theme}`);
  override.theme = '';
  write(KEYS.theme, theme);
  applyTheme(theme);
  notify();
}

/** @returns {'light'|'dark'} what is showing now */
export function effectiveTheme() {
  const choice = getTheme();
  if (choice === 'light' || choice === 'dark') return choice;
  return prefersDark() ? 'dark' : 'light';
}

/**
 * Call `cb(effectiveTheme, preference)` when the theme or (under System) the OS scheme changes.
 * @returns {() => void} off
 */
export function onThemeChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/** @returns {'system'|'reduce'|'full'} */
export function getMotion() {
  if (override.motion) return override.motion;
  const value = prefs.get().reduceMotion;
  return MOTIONS.includes(value) ? value : 'system';
}

/**
 * Save and apply the motion choice (prefs.reduceMotion and <html data-motion>).
 * @param {'system'|'reduce'|'full'} motion
 */
export function setMotion(motion) {
  if (!MOTIONS.includes(motion)) throw new TypeError(`Unknown motion setting: ${motion}`);
  override.motion = '';
  prefs.set({ reduceMotion: motion });
  applyMotion(motion);
}

/** @returns {boolean} true under data-motion=reduce, or under System when the OS asks for reduced motion */
export function reducedMotion() {
  const attr = typeof document !== 'undefined' ? document.documentElement.getAttribute('data-motion') : null;
  if (attr === 'reduce') return true;
  if (attr === 'full') return false;
  try {
    return !!(typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
  } catch {
    return false;
  }
}

/** Test hook. */
export function __resetForTests() {
  if (darkQuery && onSchemeChange) {
    try {
      if (typeof darkQuery.removeEventListener === 'function') darkQuery.removeEventListener('change', onSchemeChange);
    } catch { /* old env */ }
  }
  darkQuery = null;
  onSchemeChange = null;
  listeners = new Set();
  override = { theme: '', motion: '' };
}
