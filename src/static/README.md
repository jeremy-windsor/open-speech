# Web UI (`src/static`)

A static single-page app: plain files, native ES modules, no build step, no CDN, no dependencies. It runs
under the web-UI CSP (`script-src 'self'`; see `src/middleware.py` `WEB_UI_CSP`).

The shell is `index.html`, served at `/web` (read from disk with `Cache-Control: no-cache`) and at
`/static/index.html`. Each section has a hash route, such as `#reader/<id>` or `#speak`, so bookmarks work;
without one the app opens `#home`.

## Layout

```
index.html           the shell; every asset path is absolute (/static/…)
manifest.webmanifest icons/  sprite.svg (the one icon sprite) and the app icons
partials/<view>.html one HTML fragment per view or slot, injected on first visit
css/                 tokens.css base.css components.css shell.css, views/<view>.css (@layer views)
fonts/               Atkinson Hyperlegible Next woff2 (the Reader's Legible face, @font-face in tokens.css) and OFL.txt
js/theme-boot.js     classic script in <head>: applies theme and motion before first paint
js/main.js           entry module: the boot sequence
js/core/             shared modules (below)
js/components/       listbox, menu, roving, tooltip, meter, range, voice-picker
js/shell/            nav, view host, chrome metrics, status chip, mini-player, sheets, banners
js/views/<view>/     index.js (the view contract) plus that view's modules
```

## Rules every module follows

- `.js` files, relative imports with an explicit `.js`. **Importing a module has no side effects**: nothing
  touches `document`, `window`, `localStorage`, `fetch`, `navigator` or `AudioContext` at module top level
  (only `main.js` and `theme-boot.js` do). Nothing is assigned to `window`.
- One owner per browser API, enforced by `tests/test_ui_core.py`:
  `fetch`/`WebSocket` → `core/api.js` (plus the partial fetch in `shell/view-host.js`) ·
  `localStorage` → `core/storage.js` (every key is in `KEYS`) · `innerHTML` → `core/dom.js` `render()` ·
  `showModal` → `core/dialogs.js` · `navigator.mediaSession` and `navigator.audioSession` (the page's
  audio session type, Safari 17+) → `core/audio.js`.
- Markup is built with `` dom.html`…` ``, which escapes every value; `raw()` is for partials and constants only.
  Attribute values are always quoted. A URL from data goes through `dom.safeUrl(url, {media, image})`
  (http, https, blob; data:audio or data:image only when asked). The lint enforces all three (unquoted
  attributes, `raw()` of a template or concatenation, `href="${…}"`/`setAttribute('href', …)` without
  `safeUrl`). No `on*=` handlers and no `style=` in markup: delegate with
  `dom.actions(root, {name: fn})` on `[data-action]` (it uses `closest()`, so clicks on an icon still find
  the button) and set custom properties with `el.style.setProperty('--x', …)`.
- No `alert`/`confirm`/`prompt`: use `toast.toast`, `dialogs.confirm`, `dialogs.prompt`. No `eval`.
- Long-form audio is a plain `<audio>` element, never routed through an `AudioContext`
  (`createMediaElementSource` is forbidden). The microphone uses the static worklet
  `/static/js/core/mic-worklet.js` (the CSP refuses `blob:` workers).
- Errors: an inline `.banner--error` next to the control that failed (`api.explain(error, 'speak')` gives
  the title, the plain message, the raw `detail` and `detailText`, the server's message and code in readable
  form, which is what Details shows), a persistent error toast when no control is near. Live regions never
  tick: elapsed counters sit in `aria-hidden` spans.
- Menus (`components/menu.js`) may be written anywhere, also inside a `.card`: while open, a menu whose
  ancestors would trap a fixed box is re-hosted at the end of its view section (or open dialog) and goes
  back on close. Tab from an open menu continues from its button. Style a menu by its own class, never by
  the card it is written in.
- Components the views asked for: `.check` (a labeled checkbox row with `.check__input` and `.check__hint`),
  `.banner--stacked` (a banner's actions under its message at every width; below 600 px every banner does
  this), and `.chip[aria-disabled="true"]` styled like a disabled chip. A view sheet may end with
  `@layer utilities { @media (forced-colors: active) { … } }` for its own scoped forced-colors rules.
- Ids carry the view prefix (`hm-`, `rd-`, `rdp-`, `sp-`, `spl-`, `tr-`, `vo-`, `st-`, `hi-`, `mo-`, `se-`; shell
  `app-`, `nav-`, `mp-`, `vp-`) and are unique across the shell and every partial (views stay mounted).

## Core modules (`js/core/`)

| Module | What it is |
|---|---|
| `api.js` | `request/get/post/put/patch/del`, `openSocket` (one-time ticket per socket), `mediaUrl`, `ApiError` (with `detailText`), `explain`, `explainClose` (a socket that never opened says so), `readableDetail` |
| `auth.js` | the API key (never re-sends a refused key), `wsTicket()` |
| `storage.js` / `prefs.js` | `KEYS`, safe read/write; validated preferences and reading appearance |
| `store.js` | `app` store, `select()`, the event bus (`emit/on`), cross-view intents (`setIntent/takeIntent`; the view host drops an intent whose navigation was superseded) |
| `router.js` | hash routes (`#reader/<id>`, `#voices/presets`…), `navigate`, `start`, `parseHash` (pure) |
| `dom.js` / `a11y.js` | `html`, `render`, `actions`, `setBusy`…; `announce()` |
| `toast.js` / `dialogs.js` | toasts; `openSheet`, `confirm`, `prompt` |
| `theme.js` / `icons.js` | System/Light/Dark and motion; `icon(name)` from the sprite |
| `status.js` | health polling, status jobs (`beginJob(key, label, {detail})`), the pure `chipModel` |
| `models.js` / `voices.js` | inventory, `uiState`, `ensureReady` (with the no-request fast path), `runOp`; voices, blends, previews |
| `audio.js` | the one audible session: `claim`, `playClip`, `prime` (iOS unlock), mini-player model, Media Session, `audioSessionType`/`holdCaptureAudioSession` |
| `mic.js` / `mic-worklet.js` / `pcm.js` | one capture at a time, devices; WAV and level helpers |
| `format.js` / `tiles.js` / `perf.js` / `clipboard.js` / `download.js` / `shortcuts.js` | the rest |

Every exported function carries JSDoc, which is its contract; this README is the summary.

## Shell contracts the core relies on

- Live regions: `#app-sr-status` (role=status), `#app-sr-alert` (role=alert), `#app-toasts-polite`
  (role=status) and `#app-toasts-alert` (role=alert), inside `#app-toasts`. While a modal sheet is open,
  `dialogs.js` moves them (and `#app-tooltip`) into the top sheet, because a modal makes the rest of the
  page inert: announcements, toasts and their buttons keep working over a sheet. Code that creates such an
  element appends it to `dialogs.liveHost()`.
- `#app-conn-banner` (role=status) is never `display: none`: while online it is empty and visually hidden,
  so its first message is announced.
- `#app-audio-host` holds `<audio id="app-audio" preload="auto">`.
- `#app-confirm` holds `h2#app-confirm-title`, `p#app-confirm-msg`, `button#app-confirm-cancel`,
  `button#app-confirm-ok`. `#app-prompt` holds `form#app-prompt-form` with `h2#app-prompt-title`,
  `label#app-prompt-label[for=app-prompt-input]`, `input#app-prompt-input`, `p#app-prompt-hint`,
  `p#app-prompt-error[hidden]`, `button#app-prompt-cancel` (type=button) and `button#app-prompt-ok`
  (type=submit). When they are missing, `dialogs.js` builds them with that structure.
- Sheets close through `[data-close]` buttons, Esc and a backdrop click (a press that started inside the
  sheet, such as a text selection or a slider drag ending outside, does not close it).
- Chrome metrics: elements marked `data-chrome="top"` or `data-chrome="bottom"` (visible, fixed or sticky)
  are measured into `--chrome-top`, `--chrome-bottom-base` (toasts sit on it) and `--chrome-bottom` (scroll
  padding; it includes the visible toast stack at every width). The Reader's `.rd-topbar` and `.rdp-transport`
  must carry the attribute. While focus is inside chrome (`[data-chrome]`, the rail, the toasts) base.css drops
  the scroll padding: a stuck sticky bar sits inside it, so every Tab within the bar would scroll the page.
- Shell screenshot states (`#<route>?shot=…`): `app-more`, `app-status`, `app-shortcuts`,
  `app-voice-picker`, `app-miniplayer`, `app-toasts`. Views handle their own states.
- The status chip appears three times (`#app-status`, `#app-status-rail`, `#app-more-status`).

## Adding a view

`js/views/<id>/index.js` exports `default {id, title, partial: '/static/partials/<id>.html', init(root)}`,
plus an optional `prefetch()` that the shell's idle prefetch calls once (the Reader warms its engine and
player modules with it; it must not start anything). `init` returns a controller with `show(route, nav)`,
`hide()` and optionally `title(route)` and `immersive(route)`. The partial has exactly one `h1` with `data-view-heading`, `id="<prefix>-title"` and
`tabindex="-1"` (slot partials have none). Views load their data in the first `show()`, reload when stale
or after a relevant event, and keep audio playing when hidden.

Routing never waits for a view: the newest route wins at once, and `show()` is called but not awaited. So
`show()` starts its loads (rendering a loading state, using request tokens), and must set
`nav.scrolled = true` synchronously, before its first `await`, when it restores a scroll position. It may
call `router.navigate()` (a redirect) freely. A title that needs loaded data calls `viewHost.refreshTitle()`
once the data is in.

## Tests

- `tests/js/**` — Node behavior tests (`node --test`), run from pytest. `tests/js/harness/` is a small
  fake browser: `installBrowserEnv()` (fake clock, DOM, fetch routes, sockets, media, Web Audio, devices),
  `importFresh('views/speak/takes.js')` (imports the module, resets every shell, component and core module
  and then the module's own `__resetForTests()`), `loadHtml(file)`, and `assertAccessibleNames(root)` for
  markup a view builds at runtime. Like a browser, the fake fetch errors a body whose request was aborted
  after the headers, `WebSocket.close()` accepts only 1000 or 3000–4999, and a media element's `src` reads
  back as an absolute URL.
- `tests/test_ui_core.py` — the core JS tests plus the global audits over every partial, CSS file and JS
  module (names, ids, live regions, CSP, external URLs, the architecture lint, the view contract).
- `tests/test_ui_shell.py` and one `tests/test_ui_<view>.py` per view run that area's Node tests and static
  checks. `tests/test_ui_swap.py` checks that `/web` serves this shell, that everything it names resolves
  with its content type, and that the previous UI is gone. `tests/test_ui_coverage.py` requires every
  must-preserve id and defect in `tests/ui_must_preserve.json` to lead a test title in its assigned file.
- `python -m tests.ui_coverage [--owner <area>]` lists the inventory ids and defects still without an
  `[ID]`-titled test that runs (commented-out, skipped and todo tests, and titles inside strings, do not
  count).
