/**
 * Settings › Connection (inventory AU-08…AU-12, SE-01): the API key.
 *
 * - `#se-api-key` is a password field (autocomplete off, no spellcheck, paste allowed). The key lives in the
 *   field's value property only: never in an attribute, a URL, a log or other markup.
 * - Save, Enter in the field and the field's change event save it: "API key saved in this browser", or "API
 *   key removed" when the field is empty (AU-08). A key with a character a header cannot carry is refused at
 *   the field (aria-invalid + the error text) and never sent (AU-07).
 * - An unchanged key the server refused is never sent again by Enter or the change event (every wrong key
 *   counts toward the server's lockout, AU-05): the status line says it was refused. Save is the one way to
 *   try the same key again ("Trying the key again…", no "saved" toast); the change event fired by pressing
 *   Save (the field's blur) and the click are one action, so a typed key goes out once.
 * - `auth.setKey()` emits auth:changed: the shell re-runs the status checks (one GET /api/models probe with
 *   the new key) and reloads the current view; every view keeps the listener's picks (AU-09). The status line
 *   shows that probe's answer: this view shares the shell's request instead of sending its own. Core
 *   models.refresh() never joins a request that went out with another key (the shell's background poll, the
 *   boot request, the previous save's check), so the answer shown is always the new key's.
 * - The status line then follows the connection: a later refusal or lockout of the key in use, or the server
 *   accepting it again, updates it, so it never contradicts the connection banner.
 * - Show/Hide swaps type password|text, its label Show|Hide and aria-pressed (AU-11).
 * - Clear empties the field, removes the stored key and says so; it reloads no data (AU-12).
 * - Test asks GET /api/models (through models.refresh, so it shares a request in flight) and says what the
 *   server answered in `#se-key-status` (role=status). A refused key is never re-sent by Test (AU-05).
 * @module views/settings/connection
 */
import { html, render, setHidden, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { app } from '../../core/store.js';
import * as auth from '../../core/auth.js';
import * as models from '../../core/models.js';
import { toast } from '../../core/toast.js';
import { keyCheckMessage, KEY_COPY } from './logic.js';

export const UNSENDABLE = "The API key contains a character that can't be sent, such as a curly quote.";

/** A press on Save whose blur saved the field counts as that save for this long (a slow click). */
const PRESS_MS = 5000;

/** One GET /api/models through core (it shares a request in flight only if it carried the same key). */
async function ask() {
  try {
    await models.refresh();
    return { ok: true };
  } catch (error) {
    return { ok: false, error };
  }
}

/**
 * Bind the Connection section.
 * @param {HTMLElement} root the view section
 * @returns {{sync: () => void, mask: () => void, check: () => Promise<void>, destroy: () => void}}
 */
export function bindConnection(root) {
  const $ = (id) => root.querySelector(`#${id}`);
  const input = $('se-api-key');
  const show = $('se-key-show');
  const save = $('se-key-save');
  const test = $('se-key-test');
  const clear = $('se-key-clear');
  const status = $('se-key-status');
  const error = $('se-api-key-error');
  const errorText = $('se-api-key-error-text');
  const offs = [];
  let dirty = false;
  let checkSeq = 0;
  let checking = false;
  let shownKind = '';   // the keyCheckMessage kind the status line shows ('' while empty or checking)
  let press = null;     // a press on Save in progress: {at, committed}

  const listen = (el, type, fn) => {
    if (!el) return;
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  const value = () => String(input.value || '').trim();

  function setError(message) {
    if (errorText.textContent !== message) errorText.textContent = message;
    setHidden(error, !message);
    if (message) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }

  function setStatus(text, tone = '', kind = '') {
    shownKind = kind;
    const glyph = tone === 'ok' ? 'check' : tone === 'error' ? 'alert' : '';
    if (status.textContent === text && status.getAttribute('data-tone') === tone) return;
    render(status, text ? html`${glyph ? icon(glyph) : ''}<span>${text}</span>` : html``);
    status.setAttribute('data-tone', tone);
    status.classList.toggle('status-text--error', tone === 'error');
  }

  const keyFacts = () => {
    const hasKey = auth.hasKey();
    return { hasKey, sendable: !hasKey || auth.isSendable(auth.getKey()) };
  };

  function report(outcome) {
    const message = keyCheckMessage({ ...outcome, ...keyFacts() });
    setStatus(message.text, message.tone, message.kind);
  }

  /** Ask the server (GET /api/models) and say what it answered. */
  async function check({ retrying = false } = {}) {
    const seq = (checkSeq += 1);
    setStatus(retrying ? KEY_COPY.retrying : 'Checking…');
    setBusy(test, true, { label: 'Testing…' });
    checking = true;
    const outcome = await ask();
    if (seq !== checkSeq) return;   // a newer save or test took over
    checking = false;
    setBusy(test, false);
    if (outcome.error && outcome.error.kind === 'aborted') {
      setStatus('');
      return;
    }
    report(outcome);
  }

  /**
   * The connection changed after the check (a later request refused the key in use, a lockout began or
   * ended): the status line follows it rather than contradict the connection banner.
   */
  function follow(state) {
    if (checking || !shownKind) return;
    if (state === 'unauthorized' && (shownKind === 'accepted' || shownKind === 'open')) {
      report({ ok: false, error: { kind: 'auth' } });
    } else if (state === 'locked' && !['locked', 'unsendable'].includes(shownKind)) {
      const { retryAt } = app.get().connection;
      report({ ok: false, error: { kind: 'locked', retryAfter: retryAt ? Math.max(0, Math.ceil((retryAt - Date.now()) / 1000)) : null } });
    } else if (state === 'online' && ['refused', 'needed', 'locked'].includes(shownKind) && !auth.isRefused()) {
      // Exempt paths never clear 'unauthorized' or 'locked' (api.js): a request that needed the key passed.
      report({ ok: true });
    }
  }

  /**
   * Save the field (trimmed). Returns whether anything was saved.
   * @param {{retry?: boolean}} [options] retry: Save on an unchanged key the server refused sends it once more
   */
  function commit({ retry = false } = {}) {
    const next = value();
    if (next === auth.getKey()) {
      dirty = false;
      if (next && !auth.isSendable(next)) {
        setError(UNSENDABLE);
        return false;
      }
      if (!next || !auth.isRefused()) return false;
      if (!retry) {
        // Enter or the change event on the refused key: nothing is sent; say why nothing happened.
        if (!checking) report({ ok: false, error: { kind: 'auth' } });
        return false;
      }
      // Save on the refused key: the listener asks to try it again (the server's key may have changed).
      auth.setKey(next);
      setError('');
      check({ retrying: true }).catch(() => {});
      return true;
    }
    const result = auth.setKey(next);
    dirty = false;
    if (input.value !== next) input.value = next;   // show what was stored (trimmed)
    if (!result.ok) {
      // Stored but never sent (AU-07): say so at the field.
      setError(UNSENDABLE);
      setStatus('');
      return true;
    }
    setError('');
    toast(next ? 'API key saved in this browser' : 'API key removed', { kind: 'success' });
    check().catch(() => {});
    return true;
  }

  listen(input, 'input', () => {
    dirty = true;
    if (!error.hidden) setError('');
  });
  listen(input, 'change', () => {
    if (commit() && press) press.committed = true;
  });
  listen(input, 'keydown', (event) => {
    if (event.key !== 'Enter' || event.isComposing) return;
    event.preventDefault();
    commit();
  });
  // Pressing Save blurs the field first, which fires its change event (a save). The click that follows is
  // the same action: it neither saves nor checks again.
  listen(save, 'pointerdown', () => { press = { at: Date.now(), committed: false }; });
  listen(save, 'click', () => {
    const same = !!press && press.committed && Date.now() - press.at < PRESS_MS;
    press = null;
    if (same) return;
    // An unchanged key gets a fresh check when none is running; a refused one is tried again.
    if (!commit({ retry: true }) && !checking && error.hidden) check().catch(() => {});
  });
  listen(test, 'click', () => {
    if (test.getAttribute('aria-busy') === 'true') return;
    if (value() !== auth.getKey() || dirty) {
      if (commit()) return;   // a save runs the check itself
    }
    check().catch(() => {});
  });
  listen(show, 'click', () => {
    const reveal = input.type === 'password';
    input.type = reveal ? 'text' : 'password';
    const label = show.querySelector('.btn__label');
    if (label) label.textContent = reveal ? 'Hide' : 'Show';
    show.setAttribute('aria-pressed', reveal ? 'true' : 'false');
  });
  listen(clear, 'click', () => {
    input.value = '';
    dirty = false;
    setError('');
    if (auth.getKey()) {
      auth.setKey('');
      toast('API key removed', { kind: 'success' });
      check().catch(() => {});
    } else {
      setStatus('');
    }
    input.focus();
  });
  offs.push(app.select((s) => s.connection.state, (state) => follow(state)));

  return {
    /** Prefill the field with the stored key, unless the listener is editing it. */
    sync() {
      if (dirty || (typeof document !== 'undefined' && document.activeElement === input)) return;
      const key = auth.getKey();
      if (input.value !== key) input.value = key;
      if (key && !auth.isSendable(key)) setError(UNSENDABLE);
    },
    /** Hide the key again (leaving Settings). */
    mask() {
      if (input.type !== 'password') {
        input.type = 'password';
        const label = show.querySelector('.btn__label');
        if (label) label.textContent = 'Show';
        show.setAttribute('aria-pressed', 'false');
      }
    },
    check,
    destroy() {
      for (const off of offs.splice(0)) off();
    },
  };
}
