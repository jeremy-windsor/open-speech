# Design: Remove Provider Management — Bake-First Architecture

## Decision
Providers (Python packages) are baked into the image at build time.
The "Install Provider" UI and runtime pip machinery are removed entirely.
Model management (Load/Unload/Cache) is unchanged — VRAM lifecycle stays.

## Why
- B6, B26, B30, B32 were all install-path failures. Runtime `pip install` into a
  running container is fragile, a security smell, and now unnecessary.
- With 3 supported backends (kokoro, qwen3, faster-whisper) all prebaked by default,
  there is nothing left for users to install.
- The mental model should be: users think about *models*, not *packages*.

---

## What Gets Removed

### Backend — `src/main.py`
- `POST /api/providers/install` endpoint + handler
- `GET /api/providers/install/{job_id}` polling endpoint
- `_provider_install_jobs` dict + `_provider_install_jobs_lock`
- `_update_provider_install_job()` function
- `_run_provider_install_job()` async function
- Startup sys.path manipulation block (lines ~33–41) that adds `os_providers_dir`
  to `sys.path` so pip-target-installed packages could be imported

### Backend — `src/model_manager.py`
- `PROVIDER_INSTALL_SPECS` dict (the pip install manifest)
- `install_provider()` method entirely
- `_provider_avail_cache` dict
- `_is_provider_available()` function
- `_check_provider()` function
- `_clear_provider_cache()` function
- `PROVIDER_IMPORTS` dict (only used by availability checks — remove with them)

### Keep in model_manager.py
- `ModelState.PROVIDER_MISSING` — keep as defensive fallback state; a model can still
  report this if somehow run without the baked image (wrong build). Don't crash, just
  show a helpful message.
- `ModelState.PROVIDER_INSTALLED` — rename or repurpose to just `available`; the
  distinction between "package installed, weights not downloaded" and "downloaded" is
  still meaningful. Actually keep the state name for API compat but the UI just shows
  "Available" for both `provider_installed` and `available`.

### Frontend — `src/static/index.html`
- Remove the entire "Providers" `<section>` (the tab panel with `stt-providers-list`
  and `tts-providers-list` divs)
- Remove `<li>` tab nav item for "Providers" if present in tab bar
- Remove the legend text: "Note: Kokoro is pre-baked... Other providers install quickly"
  Replace with: "kokoro and qwen3 are pre-installed. Model weights download on first Load."

### Frontend — `src/static/app.js`
Remove:
- `installProvider()` function
- `renderProviderRow()` function
- `_provider_install_jobs` polling logic
- `providerInstallOps` from state object
- References to `state.providerInstallOps`
- The `provider_missing` auto-install-then-retry flow in load handlers
  (if model is `provider_missing`, show a clear error message instead:
  "Provider not installed — rebuild image with BAKED_PROVIDERS=<provider>")
- The `loadTTSProviders()` function's provider-availability filtering
  (all providers are always available; remove the `provider_missing` filter)

Keep:
- `providerFromModel()` — still needed to map model IDs to provider names
- Provider dropdown in TTS Speak tab — users still select between kokoro/qwen3
- `state.ttsPreferredProvider` — still needed for provider selection
- `renderModelRow()` — update to remove Install button; only show Load/Unload/Cache
- Model state badge rendering — update labels:
  - `provider_missing` → "⚠ Not available (rebuild image)" — no Install button
  - `provider_installed` / `available` → "○ Ready (load to use)" — Cache + Load buttons
  - `downloaded` → "○ Cached — Load" — Load button
  - `loaded` → "● Loaded" — Unload button

### Frontend — `src/static/app.css`
- Remove `.provider-row`, `.providers-list`, `.providers-group` styles
- Remove provider install button/spinner styles if isolated

---

## What Stays Unchanged

### API endpoints (keep all of these)
- `GET /v1/models` — OpenAI compat
- `GET /v1/audio/models` — full model list with state
- `POST /v1/audio/models/load`
- `POST /v1/audio/models/unload`
- `POST /v1/audio/models/prefetch` (download weights)
- `GET /api/ps` — loaded model status
- `GET /api/tts/capabilities`
- `GET /api/tts/voices`
- All speech/transcription endpoints

### UI tabs (keep all)
- Speak (TTS)
- Transcribe (STT)
- Studio
- Models (updated — no Install buttons, just Load/Unload/Cache)
- Composer
- History

### Model management flow (unchanged)
1. User opens Models tab
2. Sees available models grouped by backend
3. Model states: Ready → (Cache) → Cached → Load → Loaded → Unload
4. VRAM managed by Load/Unload

---

## Model Tab After Change

```
┌─────────────────────────────────────────────────────┐
│ Models                                              │
│                                                     │
│ Speech-to-Text                                      │
│ ┌─────────────────────────────────────────────────┐ │
│ │ faster-whisper-large-v3-turbo  ● Loaded  [Unload]│ │
│ │ faster-whisper-base            ○ Ready   [Load]  │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Text-to-Speech                                      │
│ ┌─────────────────────────────────────────────────┐ │
│ │ kokoro                ● Loaded  [Unload]         │ │
│ │ qwen3-tts/1.7B        ○ Ready   [Cache] [Load]   │ │
│ │ pocket-tts            ○ Ready   [Load]           │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ kokoro and qwen3 are pre-installed.                 │
│ Weights download automatically on first Load.       │
└─────────────────────────────────────────────────────┘
```

No "Install Provider" button anywhere.

---

## FIXES.md
Add new entry:
```
| B33 | Remove provider management UI and runtime pip install machinery | 🟢 | <commit> |
```

---

## Testing Checklist After Implementation
- [ ] Models tab loads with no Providers section
- [ ] kokoro Load/Unload works
- [ ] qwen3 Load/Unload works (weights download on first Load)
- [ ] faster-whisper Load/Unload works
- [ ] `provider_missing` model shows error message, no Install button
- [ ] No JS errors in browser console related to provider install
- [ ] `/api/providers/install` returns 404 (removed)
- [ ] TTS Speak tab: provider dropdown still shows kokoro/qwen3
- [ ] Voice generation still works for both providers
