/**
 * Document tiles and voice monograms: a hue from the id, two gradient stops whose lightness keeps white
 * text at 4.5:1 or better, and the monogram letters. tokens.css owns the gradient (--tile-bg); JS sets only
 * --tile-hue, --tile-l1 and --tile-l2.
 * @module core/tiles
 */

const SMALL_WORDS = new Set(['a', 'an', 'and', 'of', 'or', 'the', 'to', 'in', 'on', 'for', 'at', 'by']);
const TILE_SATURATION = [52, 58];

/** A stable hue (0–359) from any text. */
export function hueFor(text) {
  let hue = 0;
  for (const ch of String(text || '')) hue = (hue * 31 + ch.codePointAt(0)) % 360;
  return hue;
}

// A word keeps an apostrophe inside it ("Alice's", "Don’t", "O'Brien"), so its tail is never a word of its own.
const WORD = /[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu;
const APOSTROPHES = /['’]/g;

/**
 * Two letters: the first letters of the first two significant words (a leading small word is kept),
 * digits allowed; one word gives its first two letters; nothing gives '?'. An apostrophe inside a word
 * keeps it one word: "Alice's Adventures in Wonderland" gives "AA", "Don’t Panic" "DP".
 */
export function monogram(title) {
  const words = String(title || '').match(WORD) || [];
  const picked = words.filter((word, i) => i === 0 || !SMALL_WORDS.has(word.toLowerCase()));
  if (!picked.length) return '?';
  const letters = picked.length > 1
    ? Array.from(picked[0])[0] + Array.from(picked[1])[0]
    : Array.from(picked[0].replace(APOSTROPHES, '')).slice(0, 2).join('');
  return letters.toUpperCase();
}

/** HSL (degrees, percent, percent) -> sRGB in 0..1. */
export function hslToRgb(hue, saturation, lightness) {
  const s = saturation / 100;
  const l = lightness / 100;
  const a = s * Math.min(l, 1 - l);
  const channel = (n) => {
    const k = (n + hue / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [channel(0), channel(8), channel(4)];
}

/** WCAG contrast of white text on an sRGB colour (0..1 channels). */
export function contrastWithWhite(rgb) {
  const linear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const luminance = 0.2126 * linear(rgb[0]) + 0.7152 * linear(rgb[1]) + 0.0722 * linear(rgb[2]);
  return 1.05 / (luminance + 0.05);
}

/** The lightest lightness (at most `preferred`) that keeps white text at 4.5:1 or better. */
export function tileLightness(hue, saturation, preferred) {
  let lightness = preferred;
  while (lightness > 5 && contrastWithWhite(hslToRgb(hue, saturation, lightness)) < 4.5) lightness -= 1;
  return lightness;
}

/**
 * The tile colours for a seed (a document or voice id): the second stop is hue + 40°.
 * @returns {{hue: number, hue2: number, light: number, dark: number}} light/dark are the two lightnesses (%)
 */
export function tileColors(seed) {
  const hue = hueFor(seed);
  const hue2 = (hue + 40) % 360;
  return { hue, hue2, light: tileLightness(hue, TILE_SATURATION[0], 46), dark: tileLightness(hue2, TILE_SATURATION[1], 34) };
}

/** Set the tile custom properties on an element (the only styling JS does for tiles). */
export function applyTile(el, seed) {
  const c = tileColors(seed);
  el.style.setProperty('--tile-hue', String(c.hue));
  el.style.setProperty('--tile-l1', `${c.light}%`);
  el.style.setProperty('--tile-l2', `${c.dark}%`);
  return c;
}

/**
 * A square PNG data URL of the tile with its monogram, for Media Session artwork (512, 256 or 96 px).
 * @returns {string} '' when a canvas isn't available
 */
export function artworkDataUrl(seed, title, size = 512) {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    const c = tileColors(seed);
    const gradient = ctx.createLinearGradient(0, 0, size, size);
    gradient.addColorStop(0, `hsl(${c.hue} ${TILE_SATURATION[0]}% ${c.light}%)`);
    gradient.addColorStop(1, `hsl(${c.hue2} ${TILE_SATURATION[1]}% ${c.dark}%)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `600 ${Math.round(size * 0.38)}px system-ui, -apple-system, 'Segoe UI', sans-serif`;
    ctx.fillText(monogram(title || seed), size / 2, size / 2 + size * 0.02);
    return canvas.toDataURL('image/png') || '';
  } catch {
    return '';
  }
}
