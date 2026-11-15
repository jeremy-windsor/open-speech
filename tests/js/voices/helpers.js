// Shared setup for the Voices tests: the real shell (index.html) and the real voices partial, the real
// view host and router, and fake server routes for the voices APIs. Not a test file (no .test.js suffix).
import { shellEnv, load, settle, json, text } from '../shell/helpers.js';
import { bytes, delay } from '../harness/fakes.js';

export { load, settle, json, text, bytes, delay };

export const KOKORO_CAPS = { voice_blend: true, speed: true, streaming: false, voice_clone: false };
export const CLONE_CAPS = { voice_clone: true, clone_transcript_required: true, speed: false };

export const MODELS = {
  models: [
    { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0', capabilities: KOKORO_CAPS },
    { id: 'piper/en_US-amy-medium', type: 'tts', provider: 'piper', state: 'downloaded', device: 'cpu', capabilities: {} },
    { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded', device: 'cuda:0', provider_available: true, capabilities: CLONE_CAPS },
    { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'downloaded', device: 'cpu' },
  ],
  default_tts_model: 'kokoro',
  default_stt_model: 'Systran/faster-whisper-base',
};

/** The inventory without a usable clone model (the worker is offline). */
export const MODELS_NO_CLONE = {
  ...MODELS,
  models: MODELS.models.map((m) => (m.id === 'chatterbox' ? { ...m, state: 'provider_unavailable', provider_available: false } : m)),
};

export function namedVoices() {
  return {
    voices: [
      {
        id: 'n1', name: 'Narrator', voice: 'voice:n1', preset_count: 2,
        realizations: [
          { id: 'r1', model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', reference_audio_id: null },
          { id: 'r2', model: 'qwen3', voice: 'Ryan', reference_audio_id: null },
        ],
      },
      { id: 'n2', name: 'Alex', voice: 'voice:n2', preset_count: 0, realizations: [{ id: 'r3', model: 'chatterbox', voice: 'demo_reference', reference_audio_id: 'demo_reference' }] },
    ],
  };
}

export function presets() {
  return {
    profiles: [
      { id: 'p1', name: 'Audiobook 1.1×', backend: 'kokoro', model: 'kokoro', voice: 'af_heart', speed: 1.1, format: 'mp3', blend: null, reference_audio_id: null, voice_identity_id: null, instructions: null, effects: [], is_default: true },
      { id: 'p2', name: 'Podcast EQ', backend: 'kokoro', model: 'kokoro', voice: 'am_adam', speed: 1, format: 'mp3', blend: null, reference_audio_id: null, voice_identity_id: null, instructions: null, effects: [{ type: 'podcast_eq' }, { type: 'normalize' }], is_default: false },
      { id: 'p3', name: 'Narrator preset', backend: 'kokoro', model: 'kokoro', voice: 'af_bella', speed: 1.25, format: 'wav', blend: null, reference_audio_id: null, voice_identity_id: 'n1', instructions: 'Calm', effects: [], is_default: false },
      { id: 'p4', name: 'demo clone', backend: 'chatterbox', model: 'chatterbox', voice: 'demo_reference', speed: 1, format: 'wav', blend: null, reference_audio_id: 'demo_reference', voice_identity_id: null, instructions: null, effects: [], is_default: false },
    ],
    default_profile_id: 'p1',
  };
}

export function recordings() {
  return [
    { name: 'demo_reference', duration_s: 6, sample_rate: 24000, channels: 1, size_bytes: 288044, transcript: 'The quick brown fox jumps over the lazy dog.' },
    { name: 'raw_take', duration_s: 8.43, sample_rate: 48000, channels: 1, size_bytes: 809000 },
  ];
}

const SPEAK_STUB = '<header class="view-head"><h1 id="sp-title" class="view-head__title" data-view-heading tabindex="-1">Speak</h1></header>';
const MODELS_STUB = '<header class="view-head"><h1 id="mo-title" class="view-head__title" data-view-heading tabindex="-1">Models</h1></header>';

function stubView(id, title, partial) {
  return async () => ({ default: { id, title, partial, init: () => ({ show() {}, hide() {} }) } });
}

/**
 * Boot the fake browser on #voices/<sub> with the voices view mounted through the real view host and router.
 * @param {{sub?: string, routes?: object, models?: object, seed?: object, mediaDevices?: object, audio?: object,
 *   secure?: boolean}} [options]
 */
export async function voicesEnv({ sub = 'named', routes = {}, models = MODELS, ...options } = {}) {
  const env = await shellEnv({
    routes: {
      'GET /api/models': () => json(200, models),
      'GET /api/voices/identities': () => json(200, namedVoices()),
      'GET /api/profiles': () => json(200, presets()),
      'GET /api/voices/library': () => json(200, recordings()),
      'GET /api/voices/library-config': () => json(200, { max_seconds: 60 }),
      'DELETE /api/voices/identities/:id/realizations/*': () => json(204),
      'DELETE /api/voices/identities/:id': () => json(200, { moved_to: null }),
      'DELETE /api/profiles/:id': () => json(204),
      'DELETE /api/voices/library/:name': () => json(204),
      'GET /static/partials/speak.html': () => text(200, SPEAK_STUB, { 'content-type': 'text/html' }),
      'GET /static/partials/models.html': () => text(200, MODELS_STUB, { 'content-type': 'text/html' }),
      ...routes,
    },
    ...options,
  });
  const viewHost = await load('shell/view-host.js');
  const router = await load('core/router.js');
  const modelsApi = await load('core/models.js');
  const store = await load('core/store.js');
  viewHost.initViewHost({
    registry: {
      voices: () => load('views/voices/index.js'),
      speak: stubView('speak', 'Speak', '/static/partials/speak.html'),
      models: stubView('models', 'Models', '/static/partials/models.html'),
    },
  });
  await modelsApi.refresh({ silent: true });
  env.setHash(`#voices/${sub}`);
  await router.start(viewHost.show);
  await settle(env);
  await settle(env);
  const section = document.getElementById('view-voices');
  const ctx = {
    env, router, viewHost, store, app: store.app, modelsApi, section,
    controller: viewHost.controllerFor('voices'),
    $: (sel) => section.querySelector(sel),
    $$: (sel) => [...section.querySelectorAll(sel)],
    calls: (pattern) => env.fetch.callsTo(pattern),
    settle: (ms = 0) => settle(env, ms),
    /** Navigate within the app and let the view settle. */
    async go(path) {
      await router.navigate(path);
      await settle(env);
      await settle(env);
    },
    /** Answer the open confirm dialog. */
    async answerConfirm(yes = true) {
      const dialog = document.getElementById('app-confirm');
      if (!dialog.open) throw new Error('no confirm is open');
      env.click(document.getElementById(yes ? 'app-confirm-ok' : 'app-confirm-cancel'));
      await settle(env);
      await settle(env);
    },
    confirmText() {
      return {
        title: document.getElementById('app-confirm-title').textContent,
        message: document.getElementById('app-confirm-msg').textContent,
        open: document.getElementById('app-confirm').open,
      };
    },
    /** Answer the open prompt (null cancels). */
    async answerPrompt(value) {
      const dialog = document.getElementById('app-prompt');
      if (!dialog.open) throw new Error('no prompt is open');
      if (value === null) {
        env.click(document.getElementById('app-prompt-cancel'));
      } else {
        const input = document.getElementById('app-prompt-input');
        input.value = value;
        env.click(document.getElementById('app-prompt-ok'));
      }
      await settle(env);
      await settle(env);
    },
    promptState() {
      const input = document.getElementById('app-prompt-input');
      const error = document.getElementById('app-prompt-error');
      return {
        open: document.getElementById('app-prompt').open,
        title: document.getElementById('app-prompt-title').textContent,
        value: input.value,
        error: error.hidden ? '' : error.textContent,
        invalid: input.getAttribute('aria-invalid') === 'true',
      };
    },
    /** Open a card's menu and choose an item. */
    async menu(buttonId, cmd) {
      const button = document.getElementById(buttonId);
      if (!button) throw new Error(`no menu button ${buttonId}`);
      env.click(button);
      await settle(env);
      const menu = document.getElementById(button.getAttribute('aria-controls'));
      const item = menu.querySelector(`[data-menu-action="${cmd}"]`);
      if (!item) throw new Error(`no menu item ${cmd}`);
      env.click(item);
      await settle(env);
      await settle(env);
    },
    toasts() {
      return [...document.querySelectorAll('#app-toasts .toast__msg')].map((el) => el.textContent);
    },
  };
  return ctx;
}
