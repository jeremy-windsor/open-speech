/**
 * Studio conversations (inventory ST-01…ST-10, defects BUG-03, BUG-16, REV-29).
 *
 * - The list (`#st-list`) holds one link per conversation to its route, studio/conversations/<id>. Phones show
 *   the list or the editor; from 960 px both sit side by side.
 * - The editor (`#st-editor`, its h2 the conversation's name) is a script of turns: a speaker avatar, the
 *   preset and the text, with Move up / Move down buttons and a menu with Delete…. With no conversation open it
 *   is a draft: adding the first turn creates the conversation (ST-03).
 * - Reordering has no server endpoint: the tail from the first moved position is re-added in its new order,
 *   then the old tail is deleted (planReorder). Adding before deleting means an interruption can duplicate
 *   turns but never lose text (REV-29).
 * - Render (`#st-render`) locks the script (Add, Move, Delete and Delete conversation are aria-disabled) until
 *   it finishes, so this tab can no longer cause the server's 409; a 409 from another tab shows a banner with
 *   Render again (BUG-03). Downloads wait for a render, and a download in another format renders first.
 * - Every lock belongs to one conversation (or the draft): a render or a reorder of one never locks another,
 *   and each end re-renders whatever is open. Work that ends while its conversation is not on screen (another
 *   conversation, the Composer, another view) is told by a toast with Open, and its banner waits until that
 *   conversation opens again. An answer never navigates once the user has gone elsewhere.
 * - While the render plays, an inline transport (a seek range and the time) stands in for the mini-player,
 *   which hides on the render's own route.
 * @module views/studio/conversations
 */
import { html, render, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import { confirm, prompt } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { announce } from '../../core/a11y.js';
import * as audio from '../../core/audio.js';
import { app } from '../../core/store.js';
import { DESIGN_ONLY_HINT, designOnly } from '../../core/models.js';
import { downloadFromApi } from '../../core/download.js';
import { register, modLabel } from '../../core/shortcuts.js';
import { applyTile, monogram } from '../../core/tiles.js';
import { absoluteTime, relativeTime, clock, plural } from '../../core/format.js';
import {
  idPart, seg, errorCopy, bannerActions, showBanner, clearBanner, fillDetails, loadingMarkup, loadErrorMarkup,
  focusElement, hideKeepingFocus, renderKeepingFocus, menuButtonMarkup, menuMarkup, menuItemsMarkup, bindMenus,
  setPlayButton, playerKey, bindTransport, startPress,
} from './shared.js';
import { onAvailability, availability, openAskAi } from '../../components/ask-ai.js';
import { charCount } from '../../components/ask-ai-logic.js';

/**
 * PURE: a conversation's script as one text for Ask AI: "Speaker: words", a line
 * per turn, in their order. A turn is left out only when its own text is empty or only spaces:
 * words that end with a colon are words.
 */
export function scriptText(conv) {
  const turns = conv && Array.isArray(conv.turns) ? conv.turns : [];
  return turns
    .map((t) => ({ speaker: String((t && t.speaker) || 'Speaker'), words: String((t && t.text) ?? '').trim() }))
    .filter((t) => t.words)
    .map((t) => `${t.speaker}: ${t.words}`)
    .join('\n');
}

/** The pause the server puts between two turns of a render (src/conversation.py SILENCE_MS). */
export const SILENCE_MS = 500;
/** The render sample rate (the old UI's, and the server's default). */
export const SAMPLE_RATE = 24000;
export const NAME_MAX = 200;

/** From this width the list sits beside the editor (studio.css). */
const WIDE = '(min-width: 960px)';

/** The route of a conversation (the id encoded as the router encodes it). */
export const convPath = (id) => `studio/conversations/${seg(id)}`;

/* ───────────────────────────── pure helpers ───────────────────────────── */

/** "Conversation Oct 23, 2026, 10:41 AM": the default name of a new conversation (ST-02). */
export function defaultConversationName(now = Date.now()) {
  return `Conversation ${absoluteTime(now)}`;
}

/** The POST …/turns body that re-creates a turn as it is (speaker, text, preset and effects). */
export function turnPayload(turn) {
  const effects = Array.isArray(turn?.effects) && turn.effects.length ? turn.effects : null;
  return { speaker: String(turn?.speaker || 'Speaker'), text: String(turn?.text ?? ''), profile_id: turn?.profile_id || null, effects };
}

/**
 * PURE: the tail rewrite that moves turn `from` to position `to`. The server appends new turns and
 * re-indexes after a delete, so the tail from k = min(from, to) is re-added in its new order (it lands after
 * the old tail), then the old tail is deleted.
 * @param {{id: string}[]} turns in their current order
 * @param {number} from
 * @param {number} to
 * @returns {{k: number, order: object[], readd: object[], deleteIds: string[]}} k is -1 for a no-op
 */
export function planReorder(turns, from, to) {
  const list = Array.isArray(turns) ? turns : [];
  const n = list.length;
  if (!Number.isInteger(from) || !Number.isInteger(to) || from === to || from < 0 || to < 0 || from >= n || to >= n) {
    return { k: -1, order: list.slice(), readd: [], deleteIds: [] };
  }
  const order = list.slice();
  const [moved] = order.splice(from, 1);
  order.splice(to, 0, moved);
  const k = Math.min(from, to);
  return { k, order, readd: order.slice(k).map(turnPayload), deleteIds: list.slice(k).map((t) => t.id) };
}

/** The format of a render from its path ('…/render.wav' -> 'wav'); '' when there is none. */
export function renderFormat(path) {
  const m = /\.([a-z0-9]+)$/i.exec(String(path || ''));
  return m ? m[1].toLowerCase() : '';
}

/**
 * The length of the conversation's render in seconds: the turns' durations plus the pauses between them, or
 * null when it can't be known (no render, or a turn without a duration).
 */
export function renderDurationS(conv) {
  if (!conv || !conv.render_output_path) return null;
  const turns = conv.turns || [];
  if (!turns.length || turns.some((t) => !Number.isFinite(Number(t.duration_ms)) || t.duration_ms === null)) return null;
  const ms = turns.reduce((sum, t) => sum + Number(t.duration_ms), 0) + SILENCE_MS * (turns.length - 1);
  return ms / 1000;
}

/** The speakers of a script, in order of first appearance (the Speaker field's suggestions). */
export function speakersOf(turns) {
  const seen = new Set();
  const out = [];
  for (const t of turns || []) {
    const name = String(t?.speaker || '').trim();
    if (name && !seen.has(name.toLowerCase())) {
      seen.add(name.toLowerCase());
      out.push(name);
    }
  }
  return out;
}

/**
 * The preset label of a turn: its preset's name, "Default voice" without one, or "Preset missing" for a preset
 * that was deleted (the server refuses to render that turn). While the presets are not loaded it is '', and
 * "Preset (name not loaded)" once their load failed, so a turn never looks as if it had no preset.
 */
export function presetLabel(profileId, presets, { loaded = true, failed = false } = {}) {
  if (!profileId) return { label: 'Default voice', missing: false };
  const preset = (presets || []).find((p) => p.id === profileId);
  if (preset) return { label: preset.name || 'Preset', missing: false };
  if (loaded) return { label: 'Preset missing', missing: true };
  return { label: failed ? 'Preset (name not loaded)' : '', missing: false };
}

/**
 * PURE: why the open script is locked ('' when it isn't): a render, a reorder, a turn being added or something
 * being deleted. The disabled menu items carry it as their reason.
 * @param {{rendering?: boolean, busy?: string}} lock
 */
export function lockReason({ rendering = false, busy = '' } = {}) {
  if (rendering) return 'Rendering…';
  if (busy === 'reorder') return 'Moving the turn…';
  if (busy === 'add') return 'Adding the turn…';
  if (busy === 'delete-turn' || busy === 'delete-conv') return 'Deleting…';
  return '';
}

/** A conversation as the server sends it, with its turns in order. */
export function normaliseConversation(conv) {
  const turns = Array.isArray(conv?.turns) ? conv.turns.slice() : [];
  turns.sort((a, b) => (Number(a.turn_index) || 0) - (Number(b.turn_index) || 0));
  return { ...conv, turns: turns.map((t) => ({ ...t, effects: Array.isArray(t.effects) ? t.effects : [] })) };
}

/** The first `max` characters of a text, on one line. */
export function snippet(text, max = 80) {
  const one = String(text || '').replace(/\s+/g, ' ').trim();
  return one.length > max ? `${one.slice(0, max - 1).trimEnd()}…` : one;
}

/* ───────────────────────────── markup ───────────────────────────── */

function listRowMarkup(conv, openId) {
  const current = conv.id === openId;
  const rendered = conv.render_output_path ? `Rendered ${renderFormat(conv.render_output_path).toUpperCase()} · ` : '';
  const edited = relativeTime(conv.updated_at || conv.created_at);
  return html`<li class="row row--dense st-conv-row${current ? ' is-selected' : ''}"><a class="row__main" id="st-conv-${idPart(conv.id)}" href="#studio/conversations/${seg(conv.id)}" data-action="open-conv" data-id="${conv.id}"${current ? html` aria-current="true"` : ''}><span class="row__title">${conv.name || 'Untitled conversation'}</span><span class="row__meta">${rendered}${edited ? html`Edited <span title="${absoluteTime(conv.updated_at || conv.created_at)}">${edited}</span>` : ''}</span></a><span class="row__trail st-conv-row__chev" aria-hidden="true">${icon('chevron-right')}</span></li>`;
}

/**
 * The list pane for a state (exported for tests). The empty state carries New conversation (primary) where the
 * list shows alone (phones, tablets); from 960 px it sits beside the draft editor, which is already a new
 * conversation, so studio.css hides that button (the pane keeps its own secondary one, as with conversations) and
 * shows the line that says how the draft starts one.
 * @param {{status: string, items: object[], error: *, at: number}} list
 * @param {string} openId the open conversation
 */
export function listMarkup(list, openId) {
  if (!list.at && (list.status === 'loading' || list.status === 'idle')) return loadingMarkup('conversations', { rows: 3 });
  if (list.status === 'error' && !list.at) return loadErrorMarkup(list.error, { what: 'conversations', list: 'conversations', level: 3 });
  const items = list.items || [];
  if (!items.length) {
    return html`<div class="empty st-list-empty">${icon('studio', { className: 'ico empty__icon' })}<h3 class="empty__title">No conversations yet</h3><p class="empty__text">Write a script for several voices, then render it as one file.</p><p class="empty__text st-list-empty__beside">Adding a turn starts your first one.</p><button type="button" class="btn btn--primary st-list-empty__new" id="st-new-empty" data-action="new-conv">${icon('plus')}<span class="btn__label">New conversation</span></button></div>`;
  }
  const stale = list.status === 'error'
    ? html`<p class="status-text status-text--error st-list-stale">${icon('alert', { className: 'ico ico--sm' })}<span>Couldn't refresh the list.</span><button type="button" class="btn btn--ghost btn--sm" data-action="retry-load" data-list="conversations"><span class="btn__label">Retry</span></button></p>`
    : '';
  return html`${stale}<ul class="rows st-conv-list" aria-labelledby="st-list-title">${items.map((c) => listRowMarkup(c, openId))}</ul>`;
}

function turnMarkup(turn, i, n, { locked, described, reason, presets, presetsLoaded, presetsFailed }) {
  const num = i + 1;
  const preset = presetLabel(turn.profile_id, presets, { loaded: presetsLoaded, failed: presetsFailed });
  const lock = locked ? html` aria-disabled="true"${described ? html` aria-describedby="st-lock-hint"` : ''}` : '';
  return html`<li class="st-turn" id="st-turn-${i}" data-index="${i}">
    <span class="avatar st-turn__avatar" data-seed="${turn.speaker || 'Speaker'}" aria-hidden="true">${monogram(turn.speaker || 'Speaker')}</span>
    <p class="st-turn__who"><span class="visually-hidden">Turn ${num}: </span><span class="st-turn__speaker">${turn.speaker || 'Speaker'}</span>${preset.label ? html`<span class="st-turn__preset${preset.missing ? ' is-missing' : ''}"><span class="visually-hidden">, </span>${preset.missing ? icon('alert', { className: 'ico ico--sm' }) : ''}${preset.label}</span>` : ''}</p>
    <p class="st-turn__text">${turn.text}</p>
    <div class="st-turn__actions">
      <button type="button" class="icon-btn icon-btn--sm" id="st-turn-${i}-up" data-action="move-turn" data-index="${i}" data-dir="up" aria-label="Move turn ${num} up" data-tooltip="Move turn ${num} up"${i === 0 ? html` disabled` : lock}>${icon('arrow-up')}</button>
      <button type="button" class="icon-btn icon-btn--sm" id="st-turn-${i}-down" data-action="move-turn" data-index="${i}" data-dir="down" aria-label="Move turn ${num} down" data-tooltip="Move turn ${num} down"${i === n - 1 ? html` disabled` : lock}>${icon('arrow-down')}</button>
      ${menuButtonMarkup(`st-turn-${i}-more`, `st-turn-${i}-menu`, `turn ${num}`, { className: 'icon-btn--sm st-more' })}
      ${menuMarkup(`st-turn-${i}-menu`, `Turn ${num}`, [
        { cmd: 'delete-turn', arg: String(i), label: 'Delete turn…', icon: 'trash', danger: true, disabled: reason },
      ])}
    </div>
  </li>`;
}

/* ───────────────────────────── the panel ───────────────────────────── */

/**
 * The conversations panel.
 * @param {HTMLElement} panel #st-conversations
 * @param {{lists: object, load: (name: string) => Promise<boolean>, navigate: Function, isShown: () => boolean,
 *   isCurrentSub: () => boolean}} ctx
 */
export function createConversations(panel, ctx) {
  const q = (id) => panel.querySelector(`#${id}`);
  const el = {
    notice: q('st-conv-notice'),
    listPane: q('st-list-pane'),
    list: q('st-list'),
    editor: q('st-editor'),
    title: q('st-editor-title'),
    meta: q('st-editor-meta'),
    more: q('st-conv-more'),
    convMenu: q('st-conv-menu'),
    banner: q('st-editor-banner'),
    lockHint: q('st-lock-hint'),
    script: q('st-script'),
    turns: q('st-turns'),
    empty: q('st-turns-empty'),
    form: q('st-add-form'),
    speaker: q('st-speaker'),
    speakers: q('st-speakers'),
    preset: q('st-preset'),
    presetError: q('st-preset-error'),
    presetDesign: q('st-preset-design-hint'),
    text: q('st-text'),
    textError: q('st-text-error'),
    addError: q('st-add-error'),
    add: q('st-add'),
    renderSection: q('st-render-section'),
    format: q('st-format'),
    render: q('st-render'),
    renderHint: q('st-render-hint'),
    play: q('st-play'),
    statusText: q('st-render-status-text'),
    elapsed: q('st-render-elapsed'),
    download: q('st-download'),
    downloadHint: q('st-download-hint'),
    renderBanner: q('st-render-banner'),
    transport: q('st-transport'),
    seek: q('st-seek'),
    seekTime: q('st-seek-time'),
    kbdMod: q('st-kbd-mod'),
  };
  el.lockHintText = el.lockHint ? el.lockHint.querySelector('span') : null;

  const state = {
    openId: '',
    conv: null,
    convStatus: 'none',   // none | loading | ready | error
    convError: null,
    convAt: 0,
    convSeq: 0,
    busy: new Map(),      // editor key ('' = the draft, else a conversation id) -> add | reorder | delete-turn | delete-conv
    rendering: new Map(), // conversation id -> {format, startedAt}
    downloads: new Map(), // conversation id -> 'render' (a render in the asked format first) | 'fetch'
    renders: {},          // convId -> {path, durationMs} from the last render answer
    pending: new Map(),   // conversation id -> {banner?, renderBanner?, addError?}: failures told while it wasn't open
    notice: null,
    lastConvId: '',
    shownId: null,        // the id the last show() rendered (null before the first)
    focusEditor: false,
    askable: false,       // the conversation's menu offers Ask AI about the script
  };
  const playing = { version: '' };
  const pendingPlay = new Map();   // session id -> the Play press whose render is still downloading (startPress)
  let presetSig = null;   // the presets the select was last built from
  let presetErrorText = '';
  let elapsedTimer = null;
  let unbindMenus = () => {};
  const offs = [];

  const presetsList = () => ctx.lists.presets;
  const convList = () => ctx.lists.conversations;
  const conv = () => (state.conv && state.conv.id === state.openId ? state.conv : null);
  /** The editor's key: the open conversation's id, or '' for the draft. */
  const editorKey = () => state.openId || '';
  const busyKind = (key = editorKey()) => state.busy.get(key) || '';
  const renderingThis = () => !!(conv() && state.rendering.has(conv().id));
  const locked = () => !!busyKind() || renderingThis();
  const routeNow = () => String((app.get().route || {}).path || '');
  /** Whether `key`'s editor is what the user sees now: the view, its Conversations section and that editor. */
  const onScreen = (key) => !!(ctx.isShown() && ctx.isCurrentSub() && editorKey() === key);
  const selectedFormat = () => {
    const checked = el.format ? el.format.querySelector('input[name="st-format"]:checked') : null;
    return checked ? checked.value : 'wav';
  };
  const setFormat = (fmt) => {
    if (!el.format) return;
    for (const input of el.format.querySelectorAll('input[name="st-format"]')) input.checked = input.value === fmt;
  };

  // The shortcut is Mod+Enter: Ctrl off Apple platforms, Cmd on them (assistive technology announces this).
  if (el.kbdMod) el.kbdMod.textContent = modLabel();
  if (el.text) el.text.setAttribute('aria-keyshortcuts', modLabel() === 'Ctrl' ? 'Control+Enter' : 'Meta+Enter');

  const transport = el.seek ? bindTransport(el.seek, el.seekTime, { sessionId: () => (conv() ? `studio:${conv().id}` : '') }) : null;

  function setBusyKey(key, kind) {
    state.busy.set(key, kind);
    renderEditor();
  }

  function clearBusyKey(key) {
    state.busy.delete(key);
    renderEditor();
  }

  /** Open a conversation from a toast: its editor's title takes focus once it is in. */
  function openFromToast(id) {
    if (routeNow() === convPath(id) && conv() && conv().id === id) {
      focusElement(el.title);
      return undefined;
    }
    state.focusEditor = true;
    return ctx.navigate(convPath(id), { focus: 'keep' });
  }
  const openAction = (id) => (id ? { label: 'Open', run: () => openFromToast(id) } : null);

  /**
   * Tell a failure of `key`'s work. Its banner shows in the editor while the editor holds that
   * conversation (seen now or on return), or waits until it opens again; when it isn't on screen a persistent
   * error toast with Open says it too.
   */
  function report(key, slot, banner, { toastText = '' } = {}) {
    if (editorKey() === key) showBanner(el[slot], banner);
    else if (key) state.pending.set(key, { ...(state.pending.get(key) || {}), [slot]: banner });
    if (!onScreen(key)) toast(toastText || `${banner.title}: ${banner.message}`, { kind: 'error', action: openAction(key) });
  }

  /* ── the list pane ── */

  /**
   * The visible New conversation button: the pane's, or, below 960 px, the empty state's while the list is empty
   * (studio.css shows one of them: data-empty on the pane).
   */
  function newButton() {
    const wide = typeof matchMedia === 'function' && matchMedia(WIDE).matches;
    return (!wide && q('st-new-empty')) || q('st-new');
  }

  function renderList() {
    const list = convList();
    el.list.setAttribute('aria-busy', list.status === 'loading' && !list.at ? 'true' : 'false');
    // Below 960 px the empty state carries New conversation itself, so the pane's button would only repeat it.
    const empty = !!list.at && !(list.items || []).length;
    if (el.listPane) el.listPane.setAttribute('data-empty', empty ? 'true' : 'false');
    renderKeepingFocus(el.list, listMarkup(list, state.openId), {
      fallback: () => el.list.querySelector('.row__main') || newButton(),
    });
    if (list.status === 'error' && !list.at) fillDetails(el.list, errorCopy(list.error, "Couldn't load conversations").detail);
  }

  /* ── the editor ── */

  function setText(node, text) {
    if (node && node.textContent !== text) node.textContent = text;
  }

  function editorTitle() {
    const c = conv();
    if (c) return c.name || 'Untitled conversation';
    if (state.openId) return state.convStatus === 'error' ? "Couldn't open the conversation" : 'Loading conversation…';
    return 'New conversation';
  }

  /** The lock's hint under the title: while it renders, or while a turn moves (several requests). */
  function lockHint() {
    if (renderingThis()) return 'Rendering… Editing resumes when the render finishes.';
    if (busyKind() === 'reorder') return 'Moving the turn… Editing resumes when it is in place.';
    return '';
  }

  function renderHead() {
    const c = conv();
    setText(el.title, editorTitle());
    let meta = '';
    if (c) {
      const n = c.turns.length;
      const edited = relativeTime(c.updated_at || c.created_at);
      meta = `${plural(n, 'turn')}${edited ? ` · Edited ${edited}` : ''}`;
    } else if (!state.openId) {
      meta = 'Adding the first turn saves it.';
    }
    setText(el.meta, meta);
    el.meta.hidden = !meta;
    el.more.hidden = !c;
    if (c) el.more.setAttribute('aria-label', `More actions for ${c.name || 'this conversation'}`);
    const hint = lockHint();
    if (hint) setText(el.lockHintText, hint);
    el.lockHint.hidden = !hint;
  }

  function renderTurns() {
    const c = conv();
    const loading = state.openId && state.convStatus === 'loading' && !c;
    const failed = state.openId && state.convStatus === 'error' && !c;
    el.script.setAttribute('aria-busy', loading || busyKind() === 'reorder' ? 'true' : 'false');
    for (const holder of el.script.querySelectorAll('.st-script__state')) holder.remove();
    if (loading || failed) {
      unbindMenus();
      unbindMenus = () => {};
      render(el.turns, html``);
      el.turns.hidden = true;
      el.empty.hidden = true;
      const holder = document.createElement('div');
      holder.className = 'st-script__state';
      if (loading) {
        render(holder, loadingMarkup('the conversation', { rows: 3 }));
      } else {
        render(holder, loadErrorMarkup(state.convError, { what: 'the conversation', list: 'conversation', level: 3 }));
        fillDetails(holder, errorCopy(state.convError, "Couldn't load the conversation").detail);
      }
      el.script.append(holder);
      render(el.convMenu, html``);
      return;
    }
    const turns = c ? c.turns : [];
    const presets = presetsList();
    // Every lock disables the menus' items with its reason (a reorder too: it locks the editor first).
    const reason = lockReason({ rendering: renderingThis(), busy: busyKind() });
    const options = {
      locked: locked(), described: !!lockHint(), reason, presets: presets.items, presetsLoaded: !!presets.at,
      presetsFailed: presets.status === 'error' && !presets.at,
    };
    unbindMenus();
    renderKeepingFocus(el.turns, html`${turns.map((t, i) => turnMarkup(t, i, turns.length, options))}`, {
      fallback: () => el.turns.querySelector('.st-turn__actions .st-more') || el.text,
    });
    el.turns.hidden = !turns.length;
    el.empty.hidden = !!turns.length;
    for (const avatar of el.turns.querySelectorAll('[data-seed]')) applyTile(avatar, avatar.getAttribute('data-seed'));
    // The conversation's menu sits after its button in the editor's head (studio.html).
    el.convMenu.setAttribute('aria-label', c ? c.name || 'Conversation' : 'Conversation');
    // Ask AI about the script while the assistant is on and the script has words.
    const askable = !!c && availability().phase === 'on' && !!scriptText(c);
    state.askable = askable;
    render(el.convMenu, c ? menuItemsMarkup([
      ...(askable ? [{ cmd: 'ask-ai', label: 'Ask AI about the script…', icon: 'wand' }] : []),
      { cmd: 'delete-conv', label: 'Delete conversation…', icon: 'trash', danger: true, disabled: reason },
    ]) : html``);
    unbindMenus = bindMenus(el.editor, onMenu);
  }

  function renderAddForm() {
    const c = conv();
    const failed = state.openId && !c;
    el.form.hidden = !!failed;
    render(el.speakers, html`${speakersOf(c ? c.turns : []).map((name) => html`<option value="${name}"></option>`)}`);
    renderPresetSelect();
    // The button is shared by every editor: it is busy only while this editor's turn is being added.
    const kind = busyKind();
    if (kind === 'add') setBusy(el.add, true, { label: 'Adding…' });
    else setBusy(el.add, false);
    const lock = locked() && kind !== 'add';
    if (lock) el.add.setAttribute('aria-disabled', 'true');
    else el.add.removeAttribute('aria-disabled');
    describe(el.add, el.lockHint, lock && !!lockHint());
  }

  /**
   * The preset select keeps its choice across reloads (ST-10); a deleted preset falls back to None. Its options
   * are rebuilt only when the presets change, so a load that lands while the list is open never closes it. A
   * failed load says so under it, with Retry, instead of looking like a server without presets.
   */
  function renderPresetSelect() {
    const presets = presetsList();
    const failed = presets.status === 'error';
    if (el.presetError) {
      const text = failed ? (presets.at ? "Couldn't refresh the presets." : 'Presets didn’t load, so only None is listed.') : '';
      if (failed && (text !== presetErrorText || el.presetError.hidden)) {
        render(el.presetError, html`${icon('alert', { className: 'ico ico--sm' })}<span>${text}</span><button type="button" class="btn btn--ghost btn--sm" data-action="retry-load" data-list="presets"><span class="btn__label">Retry</span></button>`);
      }
      presetErrorText = text;
      hideKeepingFocus(el.presetError, !failed, el.preset);
      describe(el.preset, el.presetError, failed);
    }
    // A preset on a model that only designs voices would give every turn a new voice, so it isn't offered.
    const list = (app.get().models && app.get().models.list) || [];
    const all = presets.items || [];
    const items = all.filter((p) => !designOnly(list.find((m) => m.id === p.model)));
    const sig = JSON.stringify(items.map((p) => [p.id, p.name || '']));
    if (el.presetDesign) {
      el.presetDesign.textContent = DESIGN_ONLY_HINT;
      el.presetDesign.hidden = items.length === all.length;
      describe(el.preset, el.presetDesign, items.length !== all.length);
    }
    if (sig === presetSig) return;
    presetSig = sig;
    const chosen = el.preset.value;
    render(el.preset, html`<option value="">None</option>${items.map((p) => html`<option value="${p.id}">${p.name || p.id}</option>`)}`);
    el.preset.value = items.some((p) => p.id === chosen) ? chosen : '';
  }

  function stopElapsed() {
    if (elapsedTimer !== null) clearInterval(elapsedTimer);
    elapsedTimer = null;
    setText(el.elapsed, '');
  }

  function tickElapsed() {
    const c = conv();
    const job = c ? state.rendering.get(c.id) : null;
    if (!job) {
      setText(el.elapsed, '');
      return;
    }
    setText(el.elapsed, ` ${clock((Date.now() - job.startedAt) / 1000)}`);
  }

  function statusText() {
    const c = conv();
    if (!c) return 'Ready';
    if (renderingThis()) return 'Rendering…';
    if (c.render_output_path) {
      const known = state.renders[c.id];
      const seconds = known && known.path === c.render_output_path && Number.isFinite(known.durationMs) ? known.durationMs / 1000 : renderDurationS(c);
      return seconds !== null && seconds !== undefined ? `Rendered · ${clock(seconds)}` : 'Rendered';
    }
    return 'Ready';
  }

  function renderRenderSection() {
    const c = conv();
    el.renderSection.hidden = !c;
    if (!c) {
      stopElapsed();
      syncTransport();
      return;
    }
    const mine = renderingThis();
    const busy = !!busyKind();
    const noTurns = !c.turns.length;
    // Render: busy while this conversation renders; aria-disabled (focus stays) with a reason otherwise. Another
    // conversation's render never locks this one.
    if (mine) {
      setBusy(el.render, true, { label: 'Rendering…' });
      el.render.removeAttribute('aria-disabled');
      setText(el.renderHint, '');
      el.renderHint.hidden = true;
      describe(el.render, el.renderHint, false);
      describe(el.render, el.lockHint, false);
    } else {
      setBusy(el.render, false);
      const reason = noTurns ? 'Add a turn to render the script.' : '';
      if (reason || busy) el.render.setAttribute('aria-disabled', 'true');
      else el.render.removeAttribute('aria-disabled');
      setText(el.renderHint, reason);
      el.renderHint.hidden = !reason;
      describe(el.render, el.renderHint, !!reason);
      describe(el.render, el.lockHint, !reason && busy && !!lockHint());
    }
    setText(el.statusText, statusText());
    if (mine) {
      tickElapsed();
      if (elapsedTimer === null && onScreen(c.id)) elapsedTimer = setInterval(tickElapsed, 1000);
    } else {
      stopElapsed();
    }
    // Downloads wait for a render (ST-07); the label names the format the download will have (ST-06). The button
    // is shared by every editor: it is busy only while this conversation's download runs (setBusy keeps keyboard
    // focus on it through a render in the asked format).
    const fmt = selectedFormat();
    const label = el.download.querySelector('.btn__label');
    const hasRender = !!c.render_output_path;
    const downloading = state.downloads.get(c.id) || '';
    if (downloading) {
      setBusy(el.download, true, downloading === 'fetch' ? { label: 'Downloading…' } : {});
    } else {
      setBusy(el.download, false);
      if (label) setText(label, `Download ${fmt.toUpperCase()}`);
      el.download.disabled = !hasRender || mine || busy;
    }
    el.downloadHint.hidden = hasRender || mine || !!downloading;
    describe(el.download, el.downloadHint, !el.downloadHint.hidden);
    syncPlayButton();
    syncTransport();
  }

  /** Point a control at its hint only while the hint shows (a hidden description is still read out). */
  function describe(control, hint, on) {
    if (!control || !hint) return;
    const ids = (control.getAttribute('aria-describedby') || '').split(/\s+/).filter((id) => id && id !== hint.id);
    if (on) ids.push(hint.id);
    if (ids.length) control.setAttribute('aria-describedby', ids.join(' '));
    else control.removeAttribute('aria-describedby');
  }

  function renderEditor() {
    panel.setAttribute('data-open', state.openId ? 'conv' : 'none');
    renderHead();
    renderTurns();
    renderAddForm();
    renderRenderSection();
  }

  /** The render's session (app.player) when it belongs to the open conversation, else null. */
  function ownPlayer(player = app.get().player) {
    const c = conv();
    return c && player && player.id === `studio:${c.id}` ? player : null;
  }

  /**
   * Play/Pause: shown while there is a render to play, and while the render's session lasts (its script may have
   * changed since, which retires the render but not the audio already playing).
   */
  function syncPlayButton() {
    const c = conv();
    if (!c) return;
    const player = ownPlayer();
    hideKeepingFocus(el.play, renderingThis() || !(c.render_output_path || player), el.render);
    if (el.play.hidden) return;
    setPlayButton(el.play, { playing: !!(player && player.playing), label: 'render' });
  }

  /**
   * The inline transport: while the render's session is the current player (the mini-player hides here) and its
   * length is known (a range from 0:00 to 0:00 says nothing). core/audio's playClip reports the length as soon
   * as the clip's metadata loads, so the transport also shows while a refused autoplay waits paused.
   */
  function syncTransport(player = app.get().player) {
    if (!el.transport || !transport) return;
    const mine = el.renderSection.hidden || el.play.hidden ? null : ownPlayer(player);
    const own = mine && Number(mine.duration) > 0 ? mine : null;
    hideKeepingFocus(el.transport, !own, el.play);
    if (!own) return;
    const name = `Position in “${conv().name || 'the render'}”`;
    if (el.seek.getAttribute('aria-label') !== name) el.seek.setAttribute('aria-label', name);
    transport.paint(own);
  }

  /* ── loading ── */

  function showNotice() {
    if (!state.notice) return;
    showBanner(el.notice, { tone: 'warning', role: 'status', ...state.notice });
    announce(state.notice.title);
  }

  /**
   * GET the conversation. Only the open one updates the editor: an answer for a conversation that is no
   * longer open (the user moved on while it rendered) is returned without touching the editor's state.
   */
  async function loadConversation(id, { quiet = false } = {}) {
    if (state.openId !== id) {
      try {
        return normaliseConversation(await api.get(`/api/conversations/${seg(id)}`, { accept: 'json' }));
      } catch {
        return null;
      }
    }
    const seq = (state.convSeq += 1);
    if (!quiet || !state.conv || state.conv.id !== id) {
      if (!state.conv || state.conv.id !== id) state.conv = null;
      state.convStatus = 'loading';
      renderEditor();
    }
    try {
      const data = await api.get(`/api/conversations/${seg(id)}`, { accept: 'json' });
      if (seq !== state.convSeq || state.openId !== id) return normaliseConversation(data);
      const fresh = !state.conv || state.conv.id !== id;
      state.conv = normaliseConversation(data);
      state.convStatus = 'ready';
      state.convError = null;
      state.convAt = Date.now();
      if (fresh) {
        const fmt = renderFormat(state.conv.render_output_path);
        if (fmt === 'wav' || fmt === 'mp3') setFormat(fmt);
      }
      renderEditor();
      if (state.focusEditor) {
        state.focusEditor = false;
        if (onScreen(id)) focusElement(el.title);
      }
      return state.conv;
    } catch (error) {
      if (seq !== state.convSeq || state.openId !== id) return null;
      if (error && error.status === 404) {
        // A deep link (or a list row) to a conversation that no longer exists.
        state.conv = null;
        state.convAt = 0;
        state.convStatus = 'none';
        state.focusEditor = false;
        state.pending.delete(id);
        if (state.lastConvId === id) state.lastConvId = '';
        ctx.load('conversations');
        // Only while its route still shows: an answer never pulls the user back from where they went.
        if (routeNow() === convPath(id)) {
          state.notice = { title: 'That conversation no longer exists', message: 'It may have been deleted, here or in another tab.' };
          ctx.navigate('studio/conversations', { replace: true, focus: 'keep' });
        }
        return null;
      }
      if (error && error.kind === 'aborted') {
        state.focusEditor = false;
        return null;
      }
      state.convStatus = 'error';
      state.convError = error;
      renderEditor();
      if (state.focusEditor) {
        // The user opened it: focus goes to the editor's title, which now says it couldn't open. Below
        // 960 px the list it was opened from is hidden by now, so focus would otherwise fall to <body>.
        state.focusEditor = false;
        if (onScreen(id)) focusElement(el.title);
      }
      return null;
    }
  }

  /* ── actions ── */

  function fieldError(text) {
    if (text) {
      el.textError.textContent = text;
      el.textError.hidden = false;
      el.text.setAttribute('aria-invalid', 'true');
      el.text.setAttribute('aria-describedby', 'st-text-hint st-text-error');
    } else {
      el.textError.textContent = '';
      el.textError.hidden = true;
      el.text.removeAttribute('aria-invalid');
      el.text.setAttribute('aria-describedby', 'st-text-hint');
    }
  }

  /** The draft becomes the conversation just created. */
  function adopt(made) {
    state.conv = normaliseConversation(made);
    state.convStatus = 'ready';
    state.convError = null;
    state.convAt = Date.now();
    state.openId = made.id;
    state.lastConvId = made.id;
  }

  /**
   * Add turn: creates the conversation first when none is open (ST-03), then POSTs the turn. The draft opens
   * the new conversation only when the user is still on it; otherwise a toast with Open says where it went.
   */
  async function addTurn() {
    const key = editorKey();
    // While a routed conversation is still loading (or failed), a turn must never start a new conversation.
    if (locked() || (key && !conv())) return;
    const text = el.text.value.trim();
    if (!text) {
      fieldError('Enter the turn’s text');
      el.text.focus();
      return;
    }
    fieldError('');
    clearBanner(el.addError);
    const speaker = el.speaker.value.trim() || 'Speaker';
    const profileId = el.preset.value || null;
    const typed = el.text.value;
    const from = routeNow();
    const name = conv() ? (conv().name || 'the conversation') : '';
    setBusyKey(key, 'add');
    let id = key;
    let made = null;
    try {
      if (!id) {
        made = await api.post('/api/conversations', { name: defaultConversationName(), turns: [] }, { accept: 'json' });
        id = made.id;
      }
      await api.post(`/api/conversations/${seg(id)}/turns`, { speaker, text, profile_id: profileId, effects: null }, { accept: 'json' });
    } catch (error) {
      clearBusyKey(key);
      const copy = errorCopy(error, "Couldn't add the turn");
      const banner = { title: copy.title, message: copy.message, detail: copy.detail, actions: bannerActions(copy) };
      if (!made) {
        report(key, 'addError', banner);
        return;
      }
      // The conversation exists now, without the turn: it opens with the error while the user is still here.
      ctx.load('conversations');
      if (editorKey() === '' && routeNow() === from) {
        adopt(made);
        await ctx.navigate(convPath(id), { focus: 'keep' });
        showBanner(el.addError, banner);
      } else {
        toast(`${copy.title}: ${copy.message}`, { kind: 'error', action: openAction(id) });
      }
      return;
    }
    if (el.text.value === typed) el.text.value = '';
    ctx.load('conversations');
    if (made) {
      clearBusyKey(key);
      if (editorKey() !== '' || routeNow() !== from) {
        toast(`Added the turn to “${made.name || 'the new conversation'}”`, { kind: 'success', action: openAction(id) });
        return;
      }
      adopt(made);
      await loadConversation(id, { quiet: true });
      if (routeNow() === from) await ctx.navigate(convPath(id), { focus: 'keep' });
    } else {
      // The script stays locked until it has reloaded, so nothing acts on turns that are out of date.
      if (editorKey() === id) await loadConversation(id, { quiet: true });
      clearBusyKey(key);
    }
    if (!onScreen(id)) {
      toast(`Turn added to “${made ? made.name || 'the new conversation' : name}”`, { kind: 'success', action: openAction(id) });
      return;
    }
    const n = conv() ? conv().turns.length : 0;
    announce(`Turn ${n} added`);
    focusElement(el.text);
  }

  /** New conversation: a name (prompt), POST, then its route; nothing is rendered yet (ST-02). */
  async function newConversation(button) {
    const name = await prompt({
      title: 'New conversation', label: 'Conversation name', value: defaultConversationName(), maxLength: NAME_MAX,
      confirmLabel: 'Create',
    });
    if (name === null) return;
    const trigger = button && button.isConnected !== false ? button : newButton();
    const from = routeNow();
    setBusy(trigger, true);
    let made;
    try {
      made = await api.post('/api/conversations', { name, turns: [] }, { accept: 'json' });
    } catch (error) {
      setBusy(trigger, false);
      const copy = errorCopy(error, "Couldn't create the conversation");
      if (routeNow() === from && ctx.isShown() && ctx.isCurrentSub()) {
        showBanner(el.notice, { title: copy.title, message: copy.message, detail: copy.detail, actions: bannerActions(copy) });
      } else {
        toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
      }
      return;
    }
    setBusy(trigger, false);
    ctx.load('conversations');
    const label = made.name || name;
    if (routeNow() !== from || !ctx.isShown() || !ctx.isCurrentSub()) {
      // The user went elsewhere meanwhile: the answer never takes them back.
      toast(`Created “${label}”`, { kind: 'success', action: openAction(made.id) });
      return;
    }
    adopt(made);
    setFormat('wav');
    clearBanner(el.notice);
    state.notice = null;
    await ctx.navigate(convPath(made.id), { focus: 'keep' });
    toast(`Created “${label}”`, { kind: 'success' });
    if (onScreen(made.id)) focusElement(el.text);
  }

  /** Move turn `index` one place up or down: add the new tail first, then delete the old one (REV-29). */
  async function moveTurn(index, dir) {
    const c = conv();
    if (!c || locked()) return;
    const to = dir === 'up' ? index - 1 : index + 1;
    const plan = planReorder(c.turns, index, to);
    if (plan.k < 0) return;
    clearBanner(el.banner);
    const id = c.id;
    const name = c.name || 'the conversation';
    setBusyKey(id, 'reorder');
    let failed = null;
    try {
      for (const payload of plan.readd) await api.post(`/api/conversations/${seg(id)}/turns`, payload, { accept: 'json' });
      for (const turnId of plan.deleteIds) await api.del(`/api/conversations/${seg(id)}/turns/${seg(turnId)}`);
    } catch (error) {
      failed = error;
    }
    // Locked until the script has reloaded: a second move must plan from the turns as they are now.
    if (editorKey() === id) await loadConversation(id, { quiet: true });
    clearBusyKey(id);
    ctx.load('conversations');
    if (failed) {
      const copy = errorCopy(failed, 'Reordering was interrupted');
      report(id, 'banner', {
        title: 'Reordering was interrupted', message: 'Some turns may appear twice; check the script.',
        detail: copy.detail || copy.message, actions: bannerActions(copy),
      }, { toastText: `Reordering “${name}” was interrupted. Some turns may appear twice; check the script.` });
      if (onScreen(id)) {
        const active = document.activeElement;
        if (!active || active === document.body || !panel.contains(active)) focusElement(el.title);
      }
      return;
    }
    // Focus follows the moved turn only while its script is on screen (never into another conversation's).
    if (!onScreen(id)) return;
    announce(`Turn ${index + 1} moved to position ${to + 1}`);
    const same = q(`st-turn-${to}-${dir}`);
    const other = q(`st-turn-${to}-${dir === 'up' ? 'down' : 'up'}`);
    focusElement(same && !same.disabled ? same : other);
  }

  async function deleteTurn(index) {
    const c = conv();
    const turn = c ? c.turns[index] : null;
    if (!turn || locked()) return;
    const ok = await confirm({
      title: 'Delete this turn?', message: `${turn.speaker || 'Speaker'}: “${snippet(turn.text)}”`, confirmLabel: 'Delete', danger: true,
    });
    if (!ok || locked() || !conv() || conv().id !== c.id) return;
    clearBanner(el.banner);
    const id = c.id;
    setBusyKey(id, 'delete-turn');
    try {
      await api.del(`/api/conversations/${seg(id)}/turns/${seg(turn.id)}`);
    } catch (error) {
      if (error && error.status === 404) {
        // Already gone (another tab): the script reloads without it.
        if (editorKey() === id) await loadConversation(id, { quiet: true });
        clearBusyKey(id);
        return;
      }
      clearBusyKey(id);
      const copy = errorCopy(error, "Couldn't delete");
      report(id, 'banner', { title: copy.title, message: copy.message, detail: copy.detail, actions: bannerActions(copy) });
      return;
    }
    if (editorKey() === id) await loadConversation(id, { quiet: true });
    clearBusyKey(id);
    ctx.load('conversations');
    const here = onScreen(id);
    toast('Turn deleted', { kind: 'success', action: here ? null : openAction(id) });
    if (!here) return;
    const left = conv() ? conv().turns.length : 0;
    const next = left ? q(`st-turn-${Math.min(index, left - 1)}-more`) : null;
    focusElement(next || el.text);
  }

  /** Delete the conversation that is open, by name (BUG-16: never a different one). */
  async function deleteConversation() {
    const c = conv();
    if (!c || locked()) return;
    const name = c.name || 'this conversation';
    const ok = await confirm({ title: `Delete “${name}” and all of its turns?`, message: 'This can’t be undone.', confirmLabel: 'Delete', danger: true });
    if (!ok || locked() || !conv() || conv().id !== c.id) return;
    clearBanner(el.banner);
    const id = c.id;
    setBusyKey(id, 'delete-conv');
    try {
      await api.del(`/api/conversations/${seg(id)}`);
    } catch (error) {
      if (!(error && error.status === 404)) {
        clearBusyKey(id);
        const copy = errorCopy(error, "Couldn't delete");
        report(id, 'banner', { title: copy.title, message: copy.message, detail: copy.detail, actions: bannerActions(copy) });
        return;
      }
    }
    state.busy.delete(id);
    const current = audio.current();
    if (current && current.id === `studio:${id}`) audio.closeCurrent();
    if (pendingPlay.has(`studio:${id}`)) pendingPlay.get(`studio:${id}`).cancel();   // its render is gone: it never plays
    delete state.renders[id];
    state.pending.delete(id);
    if (state.conv && state.conv.id === id) state.conv = null;
    if (state.lastConvId === id) state.lastConvId = '';
    const list = convList();
    list.items = (list.items || []).filter((item) => item.id !== id);
    ctx.load('conversations');
    // Back to the list only while the deleted conversation's route still shows; its entry leaves the history.
    const here = routeNow() === convPath(id);
    if (here) await ctx.navigate('studio/conversations', { replace: true, focus: 'keep' });
    else renderEditor();
    toast(`Deleted “${name}”`, { kind: 'success' });
    if (here && onScreen('')) focusElement(el.list.querySelector('.row__main') || newButton());
  }

  /**
   * Render the conversation in `format`. Resolves true when a render of the current turns exists afterwards.
   * Its script is locked meanwhile (BUG-03); a 409 from another tab's edit shows Render again. Other
   * conversations stay free to edit and render.
   */
  async function renderConversation(format = selectedFormat()) {
    const c = conv();
    if (!c || state.rendering.has(c.id) || busyKind() || !c.turns.length) return false;
    clearBanner(el.renderBanner);
    const id = c.id;
    const name = c.name || 'The conversation';
    state.rendering.set(id, { format, startedAt: Date.now() });
    renderEditor();
    let ok = false;
    try {
      const data = await api.post(`/api/conversations/${seg(id)}/render`, { format, sample_rate: SAMPLE_RATE, save_turn_audio: true }, { accept: 'json' });
      state.rendering.delete(id);
      const fresh = await loadConversation(id, { quiet: true });
      if (fresh) state.renders[id] = { path: fresh.render_output_path, durationMs: Number(data?.duration_ms) };
      ok = !!(fresh && fresh.render_output_path);
      renderEditor();
      ctx.load('conversations');
      if (!onScreen(id)) toast(`“${name}” rendered`, { kind: 'success', action: openAction(id) });
    } catch (error) {
      state.rendering.delete(id);
      renderEditor();
      if (error && error.kind === 'aborted') return false;
      const conflict = error && error.status === 409;
      const copy = errorCopy(error, "Couldn't render");
      const banner = conflict
        ? { title: 'This script changed while it was rendering', message: 'Render again to include the latest turns.', detail: copy.detail, actions: [{ cmd: 'render-again', label: 'Render again' }] }
        : { title: copy.title, message: copy.message, detail: copy.detail, actions: bannerActions(copy, { cmd: 'render-again', label: 'Retry' }) };
      if (editorKey() === id) await loadConversation(id, { quiet: true });
      report(id, 'renderBanner', banner, { toastText: `${conflict ? `“${name}” changed while it was rendering` : `${banner.title}: “${name}”`}. ${banner.message}` });
      renderEditor();
    }
    return ok;
  }

  /** Download: renders first when the last render has another format (ST-06); a failure reports (ST-05). */
  async function download() {
    const c = conv();
    if (!c || !c.render_output_path || renderingThis() || busyKind() || state.downloads.has(c.id)) return;
    const fmt = selectedFormat();
    const id = c.id;
    clearBanner(el.renderBanner);
    try {
      if (renderFormat(c.render_output_path) !== fmt) {
        state.downloads.set(id, 'render');
        const rendered = await renderConversation(fmt);
        if (!rendered) return;
      }
      state.downloads.set(id, 'fetch');
      if (editorKey() === id) renderRenderSection();
      await downloadFromApi(`/api/conversations/${seg(id)}/audio`, `conversation-${id}.${fmt}`);
    } catch (error) {
      const copy = errorCopy(error, "Couldn't download");
      report(id, 'renderBanner', { title: copy.title, message: copy.message, detail: copy.detail, actions: bannerActions(copy) });
    } finally {
      state.downloads.delete(id);
      if (editorKey() === id) renderRenderSection();
    }
  }

  /**
   * Play the render on the shared element (prime() first: the source arrives after an await). The newest press
   * wins (shared.js startPress()): a render that lands after another Play press (this conversation's, another's,
   * a mix's) or after other audio started is dropped; a repeat press while it loads makes that press the newest
   * again, with one download.
   */
  async function playRender() {
    const c = conv();
    if (!c) return;
    const sessionId = `studio:${c.id}`;
    const waiting = pendingPlay.get(sessionId);
    if (waiting) {
      waiting.renew();
      return;
    }
    const version = `${c.render_output_path}|${c.updated_at || ''}`;
    const current = audio.current();
    if (current && current.id === sessionId && (playing.version === version || !c.render_output_path)) {
      current.controls.toggle();
      return;
    }
    if (!c.render_output_path) return;
    audio.prime();
    const press = startPress();
    pendingPlay.set(sessionId, press);
    let media = null;
    let started = false;
    try {
      media = await api.mediaUrl(`/api/conversations/${seg(c.id)}/audio?v=${encodeURIComponent(c.updated_at || String(Date.now()))}`);
      if (pendingPlay.get(sessionId) === press) pendingPlay.delete(sessionId);
      if (!press.current()) {
        media.revoke();
        return;
      }
      started = true;
      playing.version = version;
      await audio.playClip({
        id: sessionId, owner: 'studio', title: c.name || 'Conversation', subtitle: 'Studio render',
        homeRoute: convPath(c.id), src: media.url,
        onRelease: () => { if (media) media.revoke(); },
      });
    } catch (error) {
      if (pendingPlay.get(sessionId) === press) pendingPlay.delete(sessionId);
      if (media) media.revoke();
      if (error && error.kind === 'aborted') return;
      if (!started && !press.current()) return;   // a download nobody waits for any more failed: nothing to say
      const copy = errorCopy(error, "Couldn't play");
      toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
    }
  }

  /* ── menus, form and shortcuts ── */

  /** The script as a `text` input of POST /api/assistant/ask. */
  function askAboutScript(trigger) {
    const c = conv();
    const text = scriptText(c);
    if (!text) return;
    const source = { id: 'script', label: `The script of ${c.name || 'this conversation'}`, input: { kind: 'text', text }, text, chars: charCount(text) };
    openAskAi({ trigger, sources: [source] }).catch((error) => toast(error && error.message ? error.message : 'Something went wrong.', { kind: 'error' }));
  }

  function onMenu(cmd, item, button) {
    if (cmd === 'delete-turn') deleteTurn(Number(item.getAttribute('data-arg')));
    else if (cmd === 'delete-conv') deleteConversation();
    else if (cmd === 'ask-ai') askAboutScript(button);
  }

  // The conversation's menu offers Ask AI only while the assistant is on: it is written again when that changes.
  offs.push(onAvailability((avail) => {
    const want = !!conv() && avail.phase === 'on' && !!scriptText(conv());
    if (want !== !!state.askable && conv()) renderTurns();
  }));

  const onSubmit = (event) => {
    event.preventDefault();
    addTurn();
  };
  el.form.addEventListener('submit', onSubmit);
  offs.push(() => el.form.removeEventListener('submit', onSubmit));
  const onText = () => { if (el.text.getAttribute('aria-invalid') === 'true' && el.text.value.trim()) fieldError(''); };
  el.text.addEventListener('input', onText);
  offs.push(() => el.text.removeEventListener('input', onText));
  const onFormat = () => renderRenderSection();
  el.format.addEventListener('change', onFormat);
  offs.push(() => el.format.removeEventListener('change', onFormat));

  offs.push(register({
    id: 'studio-add-turn',
    keys: ['Mod+Enter'],
    scope: 'view:studio',
    allowInEditable: true,
    description: 'Add the turn (in the turn text)',
    when: () => ctx.isShown() && ctx.isCurrentSub() && document.activeElement === el.text,
    handler: (event) => {
      if (event && typeof event.preventDefault === 'function') event.preventDefault();
      addTurn();
    },
  }));

  // The play button follows which session plays and whether it does (not its 4 Hz position updates); the
  // transport follows a Studio render's position only.
  offs.push(app.select(playerKey, () => syncPlayButton()));
  offs.push(app.select((s) => s.models && s.models.list, () => renderPresetSelect()));
  offs.push(app.select((s) => (s.player && String(s.player.id).startsWith('studio:') ? s.player : null), (player) => syncTransport(player)));
  if (transport) offs.push(() => transport.destroy());

  const actions = {
    'new-conv': (button) => newConversation(button),
    'open-conv': (link, event) => {
      if (event && (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || (event.button && event.button !== 0))) return;
      if (event && typeof event.preventDefault === 'function') event.preventDefault();
      const id = link.getAttribute('data-id');
      state.focusEditor = true;
      if (state.conv && state.conv.id === id && state.openId === id) {
        state.focusEditor = false;
        focusElement(el.title);
        return;
      }
      ctx.navigate(convPath(id), { focus: 'keep' });
    },
    'back-to-list': (link, event) => {
      if (event && typeof event.preventDefault === 'function') event.preventDefault();
      const id = state.openId;
      ctx.navigate('studio/conversations', { focus: 'keep' }).then(() => {
        if (!onScreen('')) return;
        const row = id ? q(`st-conv-${idPart(id)}`) : null;
        focusElement(row || q('st-list-title'));
      });
    },
    'move-turn': (button) => moveTurn(Number(button.getAttribute('data-index')), button.getAttribute('data-dir')),
    render: () => renderConversation(selectedFormat()),
    'render-again': () => renderConversation(selectedFormat()),
    download: () => download(),
    'play-render': () => playRender(),
  };

  return {
    actions,
    renderList,
    renderEditor,
    /** Presets changed (ST-10): the select keeps its choice and the turns show the new names. */
    renderPresets() {
      renderPresetSelect();
      if (conv()) renderTurns();
    },
    /**
     * Show the route's conversation ('' = the list, with the draft editor beside it from 960 px).
     * @param {string} id
     * @param {{reload?: boolean}} [options] reload: after auth:changed or when the data is stale
     * @returns {Promise<object|null>} the conversation once it loaded (for screenshot states)
     */
    show(id, { reload = false } = {}) {
      const changed = (id || '') !== state.shownId;
      state.shownId = id || '';
      state.openId = id || '';
      if (id) {
        state.lastConvId = id;
        if (state.notice) {
          state.notice = null;
          clearBanner(el.notice);
        }
      } else if (state.notice) {
        showNotice();
      }
      if (changed) {
        clearBanner(el.banner);
        clearBanner(el.renderBanner);
        clearBanner(el.addError);
        fieldError('');
        // A failure told by a toast while this conversation wasn't open shows its banner now.
        const waiting = id ? state.pending.get(id) : null;
        if (waiting) {
          state.pending.delete(id);
          for (const [slot, banner] of Object.entries(waiting)) showBanner(el[slot], banner);
        }
      }
      renderList();
      let loading = Promise.resolve(conv());
      let started = false;
      if (id && (changed || reload || !state.conv || state.conv.id !== id || state.convStatus === 'error' || Date.now() - state.convAt > 30000)) {
        const have = state.conv && state.conv.id === id;
        loading = loadConversation(id, { quiet: have });
        started = true;
      }
      renderEditor();
      if (!started && state.focusEditor) {
        state.focusEditor = false;
        if (conv()) focusElement(el.title);
      }
      if (renderingThis() && elapsedTimer === null) elapsedTimer = setInterval(tickElapsed, 1000);
      return loading;
    },
    hide() {
      stopElapsed();
    },
    retry() {
      if (state.openId) loadConversation(state.openId);
    },
    /** The conversation to return to from the Composer, if it still exists. */
    lastOpen() {
      const id = state.lastConvId;
      return id && (convList().items || []).some((c) => c.id === id) ? id : '';
    },
    focusTitle() { focusElement(el.title); },
    /** Screenshot states. */
    applyShot(shot) {
      const c = conv();
      if (shot === 'conv-menu' && c) {
        el.more.click();
      } else if (shot === 'turn-menu' && c && c.turns.length) {
        const button = q('st-turn-0-more');
        if (button) button.click();
      } else if (shot === 'rendering' && c && c.turns.length) {
        state.rendering.set(c.id, { format: selectedFormat(), startedAt: Date.now() - 12000 });
        renderEditor();
        stopElapsed();
        setText(el.elapsed, ' 0:12');
      } else if (shot === 'conflict' && c) {
        showBanner(el.renderBanner, { title: 'This script changed while it was rendering', message: 'Render again to include the latest turns.', actions: [{ cmd: 'render-again', label: 'Render again' }] });
      } else if (shot === 'reorder-error' && c) {
        showBanner(el.banner, { title: 'Reordering was interrupted', message: 'Some turns may appear twice; check the script.' });
      } else if (shot === 'add-error') {
        fieldError('Enter the turn’s text');
      } else if (shot === 'not-found') {
        state.notice = { title: 'That conversation no longer exists', message: 'It may have been deleted, here or in another tab.' };
        showNotice();
      } else if (shot === 'playing' && c && c.render_output_path) {
        playRender();
      }
    },
    destroy() {
      unbindMenus();
      stopElapsed();
      for (const off of offs.splice(0)) off();
    },
    __state: state,
  };
}
