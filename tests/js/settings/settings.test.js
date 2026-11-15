// Settings: the Reader appearance hand-off (REV-36), Appearance, Playback, Server, History,
// Shortcuts, Developer, About, and the in-page table of contents.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountSettings, settle, load, json, textOf, announced } from './helpers.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let t;
afterEach(() => t?.env?.restore());

const checked = (root, name) => (root.querySelectorAll(`input[name="${name}"]`).find((i) => i.checked) || {}).value;

/** A desktop window (the contents show from 960 px). */
const DESKTOP = Object.freeze({
  url: 'http://127.0.0.1:8100/web#settings',
  viewport: { width: 1440, height: 900 },
  media: { 'prefers-color-scheme: dark': false, 'prefers-reduced-motion: reduce': false, 'min-width: 960px': true },
});

/** The marked contents link's text. */
const current = (links) => (links.find((l) => l.getAttribute('aria-current') === 'location') || { textContent: '' }).textContent.trim();

/**
 * Lay the nine section cards out down a 3700 px page in a 900 px window (tops 100, 450, 800, 1250, 1600 for
 * the tall AI assistant card, 2650, 2950, 3250 for the short Developer card, 3450;
 * About ends at 3650), and scroll it: the cards' rects follow window.scrollY and each scroll fires the window's
 * scroll event and its frame. `offset` moves every card that much further down (a taller page head).
 */
function layoutPage(t, { offset = 0 } = {}) {
  const sections = t.root.querySelectorAll('.se-section');
  const tops = [100, 450, 800, 1250, 1600, 2650, 2950, 3250, 3450].map((top) => top + offset);
  const heights = [300, 300, 400, 300, 1000, 250, 250, 150, 200];
  assert.equal(sections.length, tops.length, 'one place per section card');
  const height = 900;
  const scrollHeight = 3700 + offset;
  t.doc.documentElement.__rect = { width: 1440, height, scrollHeight };
  const place = () => sections.forEach((el, i) => { el.__rect = { x: 300, y: tops[i] - t.env.window.scrollY, width: 720, height: heights[i] }; });
  place();
  return {
    max: scrollHeight - height,
    scrollTo(y) {
      t.env.window.scrollY = y;
      place();
      t.env.window.dispatchEvent(new Event('scroll'));
      t.env.raf.flush();
    },
  };
}

describe('settings: pure parts', () => {
  it('readingSummary: face, size (19 px below 960 px, 20 px above when unset), width and paper', async () => {
    installBrowserEnv();
    const { readingSummary } = await importFresh('views/settings/logic.js');
    assert.equal(readingSummary({}), 'Serif · 19 px · Medium width · Match app');
    assert.equal(readingSummary({}, { desktop: true }), 'Serif · 20 px · Medium width · Match app');
    assert.equal(readingSummary({ face: 'legible', size: 6, measure: 'wide', paper: 'sepia' }), 'Legible · 28 px · Wide width · Sepia');
    assert.equal(readingSummary({ face: 'bogus', size: 99, paper: 'night' }), 'Serif · 19 px · Medium width · Night');
  });

  it('storedSpeed and clampSpeed: 0.05 steps within 0.5–3.5; anything else reads as 1', async () => {
    installBrowserEnv();
    const { storedSpeed, clampSpeed, presetIndex, SPEED_PRESETS } = await importFresh('views/settings/logic.js');
    assert.equal(storedSpeed('1.25'), 1.25);
    assert.equal(storedSpeed(null), 1);
    assert.equal(storedSpeed('fast'), 1);
    assert.equal(storedSpeed('9'), 1);
    assert.equal(clampSpeed(1.234), 1.25);
    assert.equal(clampSpeed(0.1), 0.5);
    assert.equal(clampSpeed(5), 3.5);
    assert.equal(clampSpeed(''), null);
    assert.deepEqual(SPEED_PRESETS, [0.75, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3]);
    assert.equal(presetIndex(1.25), 3);
    assert.equal(presetIndex(1.3), -1);
  });
});

describe('settings: on screen', () => {
  it('[REV-36] "Change in the Reader" hands off through the frozen open-appearance intent and goes to the Reader', async () => {
    t = await mountSettings();
    const store = await load('core/store.js');
    assert.ok(store.INTENTS.reader.includes('open-appearance'), 'the intent is part of the frozen contract');
    const button = t.$('se-reading-open');
    assert.equal(textOf(button), 'Change in the Reader');
    assert.equal(button.getAttribute('aria-describedby'), 'se-reading-summary');
    assert.equal(textOf(t.$('se-reading-summary')), 'Serif · 19 px · Medium width · Match app');
    t.env.click(button);
    await settle(t.env);
    assert.equal(location.hash, '#reader');
    assert.equal(t.app.get().route.view, 'reader');
    assert.deepEqual(store.takeIntent('reader'), { type: 'open-appearance' });
    assert.equal(store.takeIntent('reader'), null, 'taken once');
  });

  it('[REV-36] the reading summary follows the reading appearance wherever it changes (one global setting)', async () => {
    t = await mountSettings();
    t.prefs.setReading({ face: 'sans', size: 4, measure: 'narrow', paper: 'dusk' });
    await settle(t.env);
    assert.equal(textOf(t.$('se-reading-summary')), 'Sans · 22 px · Narrow width · Dusk');
  });

  it('[REV-36] the reading summary follows the 960 px breakpoint while Settings shows (the Reader\'s default size is 19 px below it, 20 px from it)', async () => {
    t = await mountSettings();
    assert.equal(textOf(t.$('se-reading-summary')), 'Serif · 19 px · Medium width · Match app');
    t.env.setMedia('(min-width: 960px)', true);   // a window widened or a tablet rotated
    assert.equal(textOf(t.$('se-reading-summary')), 'Serif · 20 px · Medium width · Match app');
    t.env.setMedia('(min-width: 960px)', false);
    assert.equal(textOf(t.$('se-reading-summary')), 'Serif · 19 px · Medium width · Match app');
    t.prefs.setReading({ size: 1 });
    t.env.setMedia('(min-width: 960px)', true);
    assert.equal(textOf(t.$('se-reading-summary')), 'Serif · 17 px · Medium width · Match app', 'a size of its own is the same everywhere');
  });

  it('Theme: System | Light | Dark applies at once, persists, and follows a change made elsewhere (the rail, the More sheet)', async () => {
    t = await mountSettings();
    assert.equal(checked(t.root, 'se-theme'), 'system');
    const dark = t.root.querySelector('input[name="se-theme"][value="dark"]');
    t.env.change(dark, true);
    await settle(t.env);
    assert.equal(document.documentElement.getAttribute('data-theme'), 'dark');
    assert.equal(t.env.localStorage.getItem('open-speech-theme'), 'dark');
    t.theme.setTheme('light');
    await settle(t.env);
    assert.equal(checked(t.root, 'se-theme'), 'light');
    const group = t.$('se-theme');
    assert.equal(group.getAttribute('role'), 'radiogroup');
    assert.equal(t.$(group.getAttribute('aria-labelledby')).textContent, 'Theme');
  });

  it('Reduce motion: System | On | Off sets prefs.reduceMotion and data-motion', async () => {
    t = await mountSettings();
    assert.equal(checked(t.root, 'se-motion'), 'system');
    t.env.change(t.root.querySelector('input[name="se-motion"][value="reduce"]'), true);
    await settle(t.env);
    assert.equal(document.documentElement.getAttribute('data-motion'), 'reduce');
    assert.equal(t.prefs.get().reduceMotion, 'reduce');
    t.env.change(t.root.querySelector('input[name="se-motion"][value="full"]'), true);
    assert.equal(document.documentElement.getAttribute('data-motion'), 'full');
    t.env.change(t.root.querySelector('input[name="se-motion"][value="system"]'), true);
    assert.equal(document.documentElement.hasAttribute('data-motion'), false);
    assert.equal(JSON.parse(t.env.localStorage.getItem('open-speech-prefs')).reduceMotion, 'system');
  });

  it('Default reading speed: preset chips (roving radios), a 0.5–3.5× slider and ±0.05 steppers write the global default', async () => {
    t = await mountSettings({ seed: { 'open-speech-reader-speed': '1.25' } });
    const chips = t.$('se-speed-presets').querySelectorAll('[role="radio"]');
    assert.equal(chips.length, 9);
    assert.deepEqual(chips.filter((c) => c.getAttribute('aria-checked') === 'true').map(textOf), ['1.25×']);
    assert.deepEqual(chips.map((c) => c.getAttribute('tabindex')), ['-1', '-1', '-1', '0', '-1', '-1', '-1', '-1', '-1'], 'one tab stop');
    const range = t.$('se-speed');
    assert.equal(range.value, '1.25');
    assert.equal(range.getAttribute('aria-valuetext'), '1.25 times');
    assert.equal(textOf(t.$('se-speed-value')), '1.25×');
    // Arrow keys move and choose (GS-14).
    chips[3].focus();
    t.env.key('ArrowRight');
    await settle(t.env);
    assert.equal(t.env.localStorage.getItem('open-speech-reader-speed'), '1.5');
    assert.equal(t.doc.activeElement, chips[4]);
    t.env.click(chips[8]);
    assert.equal(t.env.localStorage.getItem('open-speech-reader-speed'), '3');
    t.env.input(range, '1.3');
    assert.equal(t.env.localStorage.getItem('open-speech-reader-speed'), '1.3');
    assert.equal(chips.filter((c) => c.getAttribute('aria-checked') === 'true').length, 0, 'between presets no chip is checked');
    t.env.click(t.$('se-speed-inc'));
    assert.equal(t.env.localStorage.getItem('open-speech-reader-speed'), '1.35');
    t.env.input(range, '3.5');
    assert.equal(t.$('se-speed-inc').getAttribute('aria-disabled'), 'true');
    t.env.click(t.$('se-speed-inc'));
    assert.equal(t.env.localStorage.getItem('open-speech-reader-speed'), '3.5', 'nothing past 3.5×');
    t.env.click(t.$('se-speed-dec'));
    assert.equal(t.env.localStorage.getItem('open-speech-reader-speed'), '3.45');
    // The Reader changed the default (its speed sheet writes the same key): Settings shows it on show().
    t.env.localStorage.setItem('open-speech-reader-speed', '0.75');
    t.ctl.show(t.route, { focus: 'none' });
    assert.equal(range.value, '0.75');
    assert.equal(chips[0].getAttribute('aria-checked'), 'true');
  });

  it('Default reading speed follows the Reader while Settings shows (its [ and ] shortcuts and speed sheet change the same default) and another tab', async () => {
    t = await mountSettings({ seed: { 'open-speech-reader-speed': '1.25' } });
    const range = t.$('se-speed');
    const chips = t.$('se-speed-presets').querySelectorAll('[role="radio"]');
    assert.equal(range.value, '1.25');
    // What the Reader engine's setSpeed() does: it writes the default and publishes its rate.
    t.env.localStorage.setItem('open-speech-reader-speed', '1.3');
    t.app.set({ player: { id: 'reader:doc1', owner: 'reader', kind: 'long', playing: true, rate: 1.3 } });
    assert.equal(range.value, '1.3');
    assert.equal(textOf(t.$('se-speed-value')), '1.3×');
    assert.equal(chips.filter((c) => c.getAttribute('aria-checked') === 'true').length, 0);
    t.env.click(t.$('se-speed-inc'));
    assert.equal(t.env.localStorage.getItem('open-speech-reader-speed'), '1.35', 'the stepper steps from what is shown');
    // The Reader paused (resumable) and its rate changed again.
    t.env.localStorage.setItem('open-speech-reader-speed', '1.5');
    t.app.set({ player: null, resumable: { id: 'reader:doc1', owner: 'reader', kind: 'long', playing: false, rate: 1.5 } });
    assert.equal(range.value, '1.5');
    assert.equal(chips[4].getAttribute('aria-checked'), 'true');
    // Another tab changed it.
    t.env.localStorage.setItem('open-speech-reader-speed', '2');
    const event = new Event('storage');
    event.key = 'open-speech-reader-speed';
    window.dispatchEvent(event);
    assert.equal(range.value, '2');
    assert.equal(chips[6].getAttribute('aria-checked'), 'true');
  });

  it('Skip back/forward, the step unit and the lock-screen unit are segmented controls bound to the preferences', async () => {
    t = await mountSettings();
    assert.equal(checked(t.root, 'se-skip-back'), '15');
    assert.equal(checked(t.root, 'se-skip-forward'), '30');
    assert.equal(checked(t.root, 'se-step'), 'paragraph');
    assert.equal(checked(t.root, 'se-track'), 'paragraph');
    t.env.change(t.root.querySelector('input[name="se-skip-back"][value="5"]'), true);
    t.env.change(t.root.querySelector('input[name="se-skip-forward"][value="60"]'), true);
    t.env.change(t.root.querySelector('input[name="se-step"][value="sentence"]'), true);
    t.env.change(t.root.querySelector('input[name="se-track"][value="chapter"]'), true);
    const p = t.prefs.get();
    assert.deepEqual([p.skipBack, p.skipForward, p.stepUnit, p.trackUnit], [5, 60, 'sentence', 'chapter']);
    // A change made elsewhere shows here.
    t.prefs.set({ skipBack: 30, trackUnit: 'paragraph' });
    await settle(t.env);
    assert.equal(checked(t.root, 'se-skip-back'), '30');
    assert.equal(checked(t.root, 'se-track'), 'paragraph');
    // The options are spoken as seconds.
    const five = t.root.querySelector('input[name="se-skip-back"][value="5"]').parentElement;
    assert.match(textOf(five.querySelector('.visually-hidden')), /seconds/);
  });

  it('The switches (snap to sentences, sleep fade, wake lock, single-key shortcuts, Speak › Live) are bound to the preferences', async () => {
    t = await mountSettings();
    const ids = { 'se-snap': 'snapToSentence', 'se-fade': 'sleepFade', 'se-wake': 'wakeLock', 'se-shortcuts': 'shortcuts', 'se-show-live': 'showLive' };
    for (const [id, field] of Object.entries(ids)) {
      const input = t.$(id);
      assert.equal(input.getAttribute('role'), 'switch');
      assert.equal(input.checked, t.prefs.get()[field], `${id} shows the preference`);
      t.env.change(input, !input.checked);
      assert.equal(t.prefs.get()[field], input.checked, `${id} writes the preference`);
    }
    assert.match(textOf(t.$('se-fade-hint')), /Not available on iPhone/);
    assert.equal(textOf(t.$('se-wake-hint')), 'Only while the Reader is open and playing.');
  });

  // Speak › Live, a streaming test for developers, is hidden unless
  // this switch shows it. It is off by default, kept in prefs like the other UI preferences, and Speak follows it
  // at once (tests/js/speak/live-switch.test.js).
  it('Developer: "Show Speak › Live (a streaming test for developers)" is off by default and sets prefs.showLive', async () => {
    t = await mountSettings();
    const input = t.$('se-show-live');
    assert.equal(input.getAttribute('role'), 'switch');
    assert.equal(input.checked, false, 'off by default');
    assert.equal(t.prefs.get().showLive, false);
    assert.equal(textOf(input.closest('label')), 'Show Speak › Live (a streaming test for developers)');
    assert.equal(input.getAttribute('aria-describedby'), 'se-show-live-hint');
    assert.match(textOf(t.$('se-show-live-hint')), /Off, Speak shows Generate only\.$/);
    assert.equal(input.closest('section'), t.$('se-developer-title').closest('section'), 'in the Developer section');
    t.env.change(input, true);
    assert.equal(t.prefs.get().showLive, true);
    const stored = JSON.parse(localStorage.getItem('open-speech-prefs'));
    assert.equal(stored.showLive, true, 'kept in this browser with the other preferences');
    t.prefs.set({ showLive: false });     // changed elsewhere (another view, a test): the switch follows
    await settle(t.env);
    assert.equal(input.checked, false);
  });

  it('The wake-lock hint says when this browser cannot keep the screen on', async () => {
    t = await mountSettings({ show: false });
    Object.defineProperty(navigator, 'wakeLock', { value: undefined, configurable: true });
    t.ctl.show(t.route, { focus: 'none' });
    assert.match(textOf(t.$('se-wake-hint')), /can't keep the screen on/);
  });

  it('Single-key shortcuts off stops single-key shortcuts (WCAG 2.1.4); "Show all shortcuts" opens the ? sheet', async () => {
    t = await mountSettings();
    const shortcuts = await load('core/shortcuts.js');
    let fired = 0;
    shortcuts.register({ id: 'test-single', keys: ['x'], singleKey: true, description: 'Test', handler: () => { fired += 1; } });
    t.env.key('x', {}, document.body);
    assert.equal(fired, 1);
    t.env.change(t.$('se-shortcuts'), false);
    t.env.key('x', {}, document.body);
    assert.equal(fired, 1, 'off: the single-key shortcut no longer fires');
    t.env.click(t.$('se-shortcuts-show'));
    await settle(t.env);
    assert.equal(t.$('app-shortcuts-sheet').open, true);
    assert.match(textOf(t.$('app-shortcuts-body')), /Single-key shortcuts are off/);
  });

  it('Server and About read /health and the inventory; Check again asks /health and says what it found', async () => {
    t = await mountSettings({ routes: { 'GET /health': () => json(200, { status: 'ok', version: '0.9.1', models_loaded: 1, revision: 'f00ba12' }) } });
    assert.equal(textOf(t.$('se-server-state')), 'Connected');
    assert.equal(textOf(t.$('se-server-version')), 'v0.8.0');
    assert.equal(textOf(t.$('se-server-revision')), 'abc1234');
    assert.equal(t.$('se-server-revision').classList.contains('mono'), true, 'a revision is code; "Unknown" is not');
    assert.equal(textOf(t.$('se-server-models')), 'Kokoro on GPU, whisper base on CPU');
    assert.equal(textOf(t.$('se-server-origin')), 'http://127.0.0.1:8100');
    assert.equal(textOf(t.$('se-server-secure')), 'Yes');
    assert.equal(textOf(t.$('se-about-version')), 'v0.8.0');
    assert.equal(textOf(t.$('se-about-revision')), 'Revision abc1234');
    // The API description on this server, with no script: never /docs or /redoc, which load an unpinned CDN
    // script on the origin that keeps the API key.
    const docs = t.$('se-server-docs');
    assert.equal(docs.getAttribute('href'), '/openapi.json');
    assert.equal(docs.getAttribute('target'), '_blank');
    assert.equal(docs.getAttribute('rel'), 'noopener');
    assert.match(textOf(docs), /^API schema \(JSON\) \(opens in a new tab\)$/);
    assert.equal(t.root.querySelectorAll('a[href="/docs"], a[href="/redoc"], a[href^="/docs#"], a[href^="/docs?"]').length, 0);
    const check = t.$('se-server-check');
    t.env.click(check);
    assert.equal(check.getAttribute('aria-busy'), 'true');
    await settle(t.env);
    assert.equal(t.env.fetch.callsTo('GET /health').length, 1);
    assert.equal(textOf(t.$('se-server-version')), 'v0.9.1');
    assert.equal(textOf(t.$('se-about-revision')), 'Revision f00ba12');
    assert.equal(check.hasAttribute('aria-busy'), false);
    assert.equal(announced(t), 'Server checked: Connected');
    t.app.set({ connection: { ...t.app.get().connection, state: 'offline' } });
    await settle(t.env);
    assert.equal(textOf(t.$('se-server-state')), "Can't reach the server");
  });

  it('Over plain HTTP the server section says the microphone needs HTTPS', async () => {
    t = await mountSettings({ env: { secure: false, url: 'http://192.168.1.20:8100/web' } });
    assert.equal(textOf(t.$('se-server-secure')), 'No · The microphone needs HTTPS');
    assert.equal(textOf(t.$('se-server-origin')), 'http://192.168.1.20:8100');
  });

  it('[SE-06] History shows the server settings by name and a link to History (where Clear all lives), and no controls', async () => {
    t = await mountSettings();
    const section = t.$('se-history-title').closest('section');
    assert.match(textOf(section), /OS_HISTORY_ENABLED/);
    assert.match(textOf(section), /OS_HISTORY_MAX_ENTRIES/);
    assert.equal(section.querySelectorAll('input, select, textarea, button').length, 0, 'no toggles for settings only the server can change');
    const link = t.$('se-history-link');
    assert.equal(link.getAttribute('href'), '#history');
    assert.equal(textOf(link), 'Manage history');
  });

  it('The table of contents scrolls to a section and focuses its heading without changing the hash; scrolling marks the current one', async () => {
    t = await mountSettings({ env: DESKTOP });
    const page = layoutPage(t);
    const links = t.$('se-toc').querySelectorAll('.se-toc__link');
    assert.deepEqual(links.map(textOf), ['Appearance', 'Playback', 'Connection', 'Server', 'AI assistant', 'History', 'Shortcuts', 'Developer', 'About']);
    assert.equal(t.$('se-toc').getAttribute('aria-label'), 'Settings sections');
    assert.equal(current(links), 'Appearance');
    const connection = links[2];
    const click = new MouseEvent('click', { bubbles: true, cancelable: true });
    connection.dispatchEvent(click);
    assert.equal(click.defaultPrevented, true, 'the link never changes the route');
    assert.equal(location.hash, '#settings');
    const heading = t.$('se-connection-title');
    assert.equal(t.doc.activeElement, heading);
    assert.equal(heading.getAttribute('tabindex'), '-1');
    assert.equal(heading.closest('section').__scrolledIntoView.length, 1, 'the whole section card scrolls into view');
    assert.equal(connection.getAttribute('aria-current'), 'location');
    assert.equal(links.filter((l) => l.hasAttribute('aria-current')).length, 1);
    // The jump's own scrolling (frames passing other sections) never re-marks.
    page.scrollTo(400);
    page.scrollTo(900);
    assert.equal(current(links), 'Connection');
    await settle(t.env, 200);
    // The listener scrolls: the first section crossing the band 15–45 % down the viewport is the current one.
    page.scrollTo(1200);
    assert.equal(current(links), 'Server');
    page.scrollTo(0);
    assert.equal(current(links), 'Appearance');
  });

  it('The table of contents at the end of the page: a jump to a late section keeps it marked; scrolling to the end marks the last section', async () => {
    t = await mountSettings({ env: DESKTOP });
    const page = layoutPage(t);
    const links = t.$('se-toc').querySelectorAll('.se-toc__link');
    // About is at the very end: the page can't scroll its card to the top, so it never reaches the band.
    assert.equal(textOf(links[8]), 'About');
    t.env.click(links[8]);
    page.scrollTo(page.max);
    await settle(t.env, 200);
    page.scrollTo(page.max);   // a late scroll event at the end (layout settling) after the jump
    assert.equal(current(links), 'About', 'the section the listener chose stays marked');
    assert.equal(t.doc.activeElement, t.$('se-about-title'));
    // History is on screen at the end too.
    page.scrollTo(1000);
    await settle(t.env, 200);
    assert.equal(textOf(links[5]), 'History');
    t.env.click(links[5]);
    page.scrollTo(page.max);
    await settle(t.env, 200);
    page.scrollTo(page.max);
    assert.equal(current(links), 'History');
    // Scrolling away and back to the end by hand marks the last section.
    page.scrollTo(1200);
    page.scrollTo(page.max);
    assert.equal(current(links), 'About');
    // Below 960 px the contents are hidden: nothing is tracked.
    t.env.setMedia('(min-width: 960px)', false);
    page.scrollTo(0);
    assert.equal(current(links), 'About');
  });

  // Leaving Settings from a lower section (AI assistant) and coming back showed the top of Settings with that
  // section still highlighted. The shell
  // scrolls a view to the top when it shows it, but a window that is already at the top doesn't scroll, so no
  // scroll event comes. So when Settings is shown, the highlight follows where the page is (the
  // first section at the top, else the section in view), a remembered jump is forgotten, and a ?shot= jump still
  // marks its section.
  it('Coming back to Settings marks where the page is: the first section at the top, else the one in view; a remembered jump is forgotten', async () => {
    t = await mountSettings({ env: DESKTOP });
    const page = layoutPage(t);
    const links = t.$('se-toc').querySelectorAll('.se-toc__link');
    const shown = { focus: 'heading', scrolled: false, reason: 'route' };
    /** Another view on the left, then Settings again, the way the view host does it (shell/view-host.js). */
    const leaveAndComeBack = async (y, route = t.route) => {
      t.ctl.hide();
      page.scrollTo(y);                  // the other view's scrolling: Settings no longer listens
      t.ctl.show(route, shown);
      window.scrollTo(0, y);             // the host's scroll to the top: from the top it moves nothing, no event
      await settle(t.env);
    };
    // AI assistant, jumped to from the contents and read: Talk, then Settings, at the top.
    t.env.click(links[4]);
    page.scrollTo(1500);
    await settle(t.env, 200);
    assert.equal(current(links), 'AI assistant');
    await leaveAndComeBack(0);
    assert.equal(current(links), 'Appearance', 'at the top, the first section is marked');
    assert.equal(links.filter((l) => l.hasAttribute('aria-current')).length, 1);
    // Scrolled there by hand: the same.
    page.scrollTo(1500);
    assert.equal(current(links), 'AI assistant');
    await leaveAndComeBack(0);
    assert.equal(current(links), 'Appearance');
    // Shown where the page isn't at the top (the first load keeps the browser's position): the section in view.
    await leaveAndComeBack(1200);
    assert.equal(current(links), 'Server');
    // A jump remembered at the end of the page is forgotten: back at the end, the last section is marked.
    page.scrollTo(1000);
    await settle(t.env, 200);
    t.env.click(links[5]);
    page.scrollTo(page.max);
    await settle(t.env, 200);
    assert.equal(current(links), 'History');
    await leaveAndComeBack(page.max);
    assert.equal(current(links), 'About', 'the jump to History is not kept');
    // A ?shot= jump still marks its section, through the frames after it.
    await leaveAndComeBack(0, { ...t.route, shot: 'assistant' });
    assert.equal(current(links), 'AI assistant');
    assert.equal(t.doc.activeElement, t.$('se-assistant-title'));
    await settle(t.env, 200);
    assert.equal(current(links), 'AI assistant');
  });

  it('At the top of the page the first section is marked, even before any card reaches the band', async () => {
    t = await mountSettings({ env: DESKTOP });
    const page = layoutPage(t, { offset: 400 });   // the first card starts below the band 15–45 % down
    const links = t.$('se-toc').querySelectorAll('.se-toc__link');
    page.scrollTo(1200);
    assert.equal(current(links), 'Connection');
    page.scrollTo(0);
    assert.equal(current(links), 'Appearance');
  });

  it('The screenshot hook ?shot=connection jumps to the Connection section', async () => {
    t = await mountSettings({ shot: 'connection' });
    assert.equal(t.doc.activeElement, t.$('se-connection-title'));
  });

  it('every control has a name and a type; the live regions are the key status line and the AI assistant\'s four status lines', async () => {
    t = await mountSettings();
    assertAccessibleNames(t.root);
    const regions = t.root.querySelectorAll('[role="status"], [role="alert"], [aria-live]').map((el) => el.id);
    // The AI assistant's are its Assistants, Connection, Test and Save lines.
    // This server has no /api/assistant, which the section says in a plain note, not a live region.
    assert.deepEqual(regions, ['se-key-status', 'se-ai-assts-status', 'se-ai-conn-status', 'se-ai-test-status', 'se-ai-status']);
  });
});
