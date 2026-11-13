/**
 * The global status chip and the status sheet. The chip renders status.chipModel(app state) in
 * every place it appears (the top bar, the rail and the More sheet); its accessible name is the full
 * sentence without the counter, and the elapsed counter ("6 s", "0:42") is an aria-hidden span that ticks
 * only while a job or a recording runs. Clicking a chip opens #app-status-sheet.
 * @module shell/status-chip
 */
import { html, render } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { app } from '../core/store.js';
import { chipModel, refreshHealth } from '../core/status.js';
import { openSheet, closeSheet } from '../core/dialogs.js';
import { classifyKind, deviceLabel, modelLabel, shortName } from '../core/models.js';
import { navigate } from '../core/router.js';
import { clock } from '../core/format.js';

const CHIP_IDS = ['app-status', 'app-status-rail', 'app-more-status'];
const STATE_WORDS = { connecting: 'Connecting', online: 'Connected', offline: "Can't reach the server", unauthorized: 'API key needed', locked: 'Locked out' };

let st = null;

/** PURE: the elapsed text for a chip model ('0:42' while recording, '6 s' for a job), '' when none. */
export function elapsedText(model, now = Date.now()) {
  if (!model || !model.elapsedFrom) return '';
  const seconds = Math.max(0, Math.floor((now - model.elapsedFrom) / 1000));
  return model.row === 'recording' ? clock(seconds) : `${seconds} s`;
}

function sameModel(a, b) {
  return !!a && !!b && a.row === b.row && a.tone === b.tone && a.label === b.label && a.ariaLabel === b.ariaLabel
    && a.detail === b.detail && a.elapsedFrom === b.elapsedFrom;
}

function paintChip(button, model) {
  const pill = button.querySelector('.pill');
  if (!pill) return;
  pill.className = `pill pill--${model.tone}`;
  const dot = pill.querySelector('.pill__dot');
  if (dot) dot.className = model.tone === 'rec' ? 'pill__dot rec-dot' : 'pill__dot';
  const label = pill.querySelector('.pill__label');
  if (label && label.textContent !== model.label) label.textContent = model.label;
  if (button.getAttribute('aria-label') !== model.ariaLabel) button.setAttribute('aria-label', model.ariaLabel);
  if (button.hasAttribute('data-tooltip')) button.setAttribute('data-tooltip', model.detail ? `${model.label} · ${model.detail}` : model.label);
  button.setAttribute('data-row', model.row);
}

function tick() {
  if (!st) return;
  const text = elapsedText(st.model);
  for (const button of st.chips) {
    const el = button.querySelector('.pill__elapsed');
    if (el && el.textContent !== text) el.textContent = text;
  }
  if (st.sheetOpen) paintJobsElapsed();
}

function syncTicker() {
  const needed = !!(st && st.model && st.model.elapsedFrom) || !!(st && st.sheetOpen && app.get().jobs.length);
  if (needed && st.timer === null) st.timer = setInterval(tick, 1000);
  if (!needed && st && st.timer !== null) {
    clearInterval(st.timer);
    st.timer = null;
  }
  tick();
}

function onModel(model) {
  if (!st) return;
  st.model = model;
  for (const button of st.chips) paintChip(button, model);
  syncTicker();
}

/* ───────────── status sheet ───────────── */

function modelRows(list) {
  return list.filter((m) => m && m.state === 'loaded').map((m) => {
    const kind = classifyKind(m);
    const name = kind === 'stt' ? `whisper ${shortName(m)}` : modelLabel(m.id);
    return { id: m.id, name, kind: kind === 'stt' ? 'Speech to text' : 'Text to speech', device: deviceLabel(m.device) };
  });
}

function secure() {
  try {
    return !!globalThis.isSecureContext;
  } catch {
    return false;
  }
}

function sheetMarkup(state) {
  const conn = state.connection || {};
  const loaded = modelRows((state.models && state.models.list) || []);
  const jobs = state.jobs || [];
  const origin = typeof location !== 'undefined' ? location.origin : '';
  const isSecure = secure();
  return html`
    <section class="status-sheet__section" aria-labelledby="app-status-server">
      <h3 class="section-label" id="app-status-server">Server</h3>
      <dl class="kv">
        <dt>State</dt><dd>${STATE_WORDS[conn.state] || 'Connecting'}${conn.state === 'offline' && conn.error ? html` · <span class="status-sheet__empty">${conn.error}</span>` : ''}</dd>
        <dt>Version</dt><dd class="tabular">${conn.version ? `v${conn.version}` : 'Unknown'}</dd>
        ${conn.revision ? html`<dt>Revision</dt><dd class="mono">${conn.revision}</dd>` : ''}
        <dt>Address</dt><dd>${origin}</dd>
        <dt>Secure connection</dt><dd>${isSecure ? 'Yes' : html`No · <span class="status-sheet__empty">the microphone needs HTTPS</span>`}</dd>
      </dl>
    </section>
    <section class="status-sheet__section" aria-labelledby="app-status-models-title">
      <h3 class="section-label" id="app-status-models-title">Loaded models</h3>
      ${loaded.length ? html`<ul class="rows">${loaded.map((m) => html`<li class="row row--dense"><span class="row__lead">${icon(m.device === 'GPU' ? 'gpu' : 'cpu')}</span><span class="row__main"><span class="row__title">${m.name}</span><span class="row__meta">${m.kind}</span></span>${m.device ? html`<span class="row__trail"><span class="badge">${m.device}</span></span>` : ''}</li>`)}</ul>` : html`<p class="status-sheet__empty">No model is loaded. Speech loads the first time you use it.</p>`}
    </section>
    <section class="status-sheet__section" aria-labelledby="app-status-activity">
      <h3 class="section-label" id="app-status-activity">Activity</h3>
      ${jobs.length ? html`<ul class="rows">${jobs.map((job) => html`<li class="row row--dense" aria-busy="true"><span class="row__lead"><span class="spinner spinner--sm" aria-hidden="true"></span></span><span class="row__main"><span class="row__title">${job.label}</span>${job.detail ? html`<span class="row__meta">${job.detail}</span>` : ''}</span><span class="row__trail"><span class="pill__elapsed tabular" data-started="${job.startedAt}" aria-hidden="true"></span></span></li>`)}</ul>` : html`<p class="status-sheet__empty">Nothing is running.</p>`}
    </section>`;
}

function paintJobsElapsed() {
  const body = document.getElementById('app-status-body');
  if (!body) return;
  for (const el of body.querySelectorAll('[data-started]')) {
    const started = Number(el.getAttribute('data-started'));
    const text = started ? `${Math.max(0, Math.floor((Date.now() - started) / 1000))} s` : '';
    if (el.textContent !== text) el.textContent = text;
  }
}

function paintSheet() {
  const body = document.getElementById('app-status-body');
  if (!body) return;
  const state = app.get();
  render(body, sheetMarkup(state));
  paintJobsElapsed();
  const retry = document.getElementById('app-status-retry');
  if (retry) retry.hidden = state.connection.state !== 'offline';
}

/**
 * Open the status sheet.
 * @param {HTMLElement} [trigger] focus returns here when it closes
 */
export function openStatusSheet(trigger) {
  const dialog = document.getElementById('app-status-sheet');
  if (!dialog || !st) return;
  st.sheetOpen = true;
  paintSheet();
  if (dialog.open) return;
  openSheet(dialog, {
    trigger,
    onClose: () => {
      if (!st) return;
      st.sheetOpen = false;
      syncTicker();
    },
  });
  syncTicker();
}

function focusAfterRoute(id) {
  // The route change focuses the view heading; a specific target (Settings › Connection) wins when present.
  const target = document.getElementById(id);
  if (target) {
    if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
    target.focus();
    if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'start' });
  }
}

/** Open Settings and focus its Connection section (the banner and the status sheet use it). */
export async function openConnectionSettings() {
  await navigate('settings', { focus: 'heading' });
  focusAfterRoute('se-connection-title');
}

/**
 * Start the chips and the sheet.
 * @returns {{destroy: () => void}}
 */
export function initStatusChip() {
  destroy();
  st = { chips: CHIP_IDS.map((id) => document.getElementById(id)).filter(Boolean), model: null, timer: null, sheetOpen: false, offs: [] };
  const onChipClick = (event) => {
    const button = event.currentTarget;
    if (button.id === 'app-more-status') {
      const more = document.getElementById('app-more-sheet');
      if (more && more.open) closeSheet(more);
      openStatusSheet(document.getElementById('nav-more') || button);
      return;
    }
    openStatusSheet(button);
  };
  for (const button of st.chips) {
    button.addEventListener('click', onChipClick);
    st.offs.push(() => button.removeEventListener('click', onChipClick));
  }
  st.offs.push(app.select((s) => chipModel(s), onModel, { equals: sameModel, immediate: true }));
  st.offs.push(app.select((s) => [s.connection, s.models, s.jobs], () => { if (st && st.sheetOpen) paintSheet(); syncTicker(); }, {
    equals: (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2],
  }));
  const bind = (id, fn) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('click', fn);
    st.offs.push(() => el.removeEventListener('click', fn));
  };
  const closeIt = () => {
    const dialog = document.getElementById('app-status-sheet');
    if (dialog && dialog.open) closeSheet(dialog);
  };
  bind('app-status-retry', () => { refreshHealth().catch(() => {}); });
  bind('app-status-models', () => { closeIt(); navigate('models'); });
  bind('app-status-settings', () => { closeIt(); openConnectionSettings(); });
  return { destroy };
}

/** Stop the chips. */
export function destroy() {
  if (!st) return;
  if (st.timer !== null) clearInterval(st.timer);
  for (const off of st.offs) off();
  st = null;
}

/** Test hook. */
export function __resetForTests() {
  destroy();
}
