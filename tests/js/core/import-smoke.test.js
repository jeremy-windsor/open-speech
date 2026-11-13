// Imports every module under src/static/js (except main.js and theme-boot.js) into this bare Node process,
// with NO fake browser installed: importing must have no side effects on the DOM or browser APIs.
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, statSync } from 'node:fs';
import { resolve, relative, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'src', 'static', 'js');
const SKIP = new Set(['main.js', 'theme-boot.js']);
const BROWSER_GLOBALS = ['window', 'document', 'localStorage', 'sessionStorage', 'location', 'history', 'matchMedia', 'AudioContext', 'WebSocket', 'requestAnimationFrame'];

function modules(dir) {
  const out = [];
  for (const name of readdirSync(dir).sort()) {
    const path = resolve(dir, name);
    if (statSync(path).isDirectory()) out.push(...modules(path));
    else if (name.endsWith('.js') && !(dir === ROOT && SKIP.has(name))) out.push(path);
  }
  return out;
}

describe('import smoke', () => {
  it('[GS-07] every module imports in a bare Node process: the old globals are explicit imports, with no side effects', async () => {
    const before = Object.fromEntries(BROWSER_GLOBALS.map((g) => [g, Object.getOwnPropertyDescriptor(globalThis, g)]));
    const files = modules(ROOT);
    assert.ok(files.length >= 25, 'the core modules are there');
    const failures = [];
    for (const file of files) {
      try {
        await import(pathToFileURL(file).href);
      } catch (error) {
        failures.push(`${relative(ROOT, file)}: ${error?.stack || error}`);
      }
    }
    assert.deepEqual(failures, []);
    for (const g of BROWSER_GLOBALS) {
      assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis, g), before[g], `importing defined or changed globalThis.${g}`);
    }
  });

  it("[GS-07] the globals the old Reader script took from the old app script now come from core modules", async () => {
    const core = async (name) => import(pathToFileURL(resolve(ROOT, 'core', `${name}.js`)).href);
    const dom = await core('dom');
    const toast = await core('toast');
    const api = await core('api');
    const storage = await core('storage');
    const models = await core('models');
    const voices = await core('voices');
    const status = await core('status');
    const store = await core('store');
    const router = await core('router');
    const replacements = {
      byId: dom.byId, esc: dom.escapeHtml, showToast: toast.toast, api: api.request, readStorage: storage.read,
      writeStorage: storage.write, getTTSModels: models.ttsModels, formatModelName: models.shortName, modelLabel: models.modelLabel,
      showModelNotice: status.beginJob, otherLoadedTts: models.otherLoadedTts, refreshModels: models.refresh,
      fetchVoices: voices.listVoices, describeVoice: voices.describeVoice, providerFromModel: models.providerOf,
      PROVIDER_DISPLAY: models.PROVIDER_DISPLAY, state: store.app, handleTabKeydown: router.navigate,
    };
    for (const [old, now] of Object.entries(replacements)) assert.ok(now, `${old} has a replacement`);
  });
});
