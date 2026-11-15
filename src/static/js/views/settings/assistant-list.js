/**
 * Settings › AI assistant › Assistants: the saved assistants, each with its name, its server's host, its model
 * and "In use" on the one in use, and Use (POST …/{id}/activate), Edit (the Connection form edits it), Check
 * (POST …/{id}/check as saved; the result in the status line, a failure in the banner) and Delete (asks first).
 * Add assistant empties the Connection form for a new one. With none saved the list says the connection comes
 * from .env. Hidden on a server without saved assistants and while the connection is locked (the Connection area
 * shows the .env connection read-only then). The list and what the form edits are the Connection area's
 * (assistant-connection.js state()); after a change it loads again and `onChange()` lets the section load its
 * status and presets.
 * @module views/settings/assistant-list
 */
import { html, render, setHidden, setBusy, actions } from '../../core/dom.js';
import * as api from '../../core/api.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { emit } from '../../core/store.js';
import { findAssistant, activateAssistant, checkAssistant, deleteAssistant } from '../../core/assistants.js';
import { elapsedText, elapsedFrom } from './assistant-logic.js';
import * as C from './assistant-connection-logic.js';

/**
 * Bind the Assistants list of the AI assistant section.
 * @param {HTMLElement} root the view section
 * @param {object} hooks
 * @param {object} hooks.conn the Connection area (bindConnectionArea)
 * @param {(host: HTMLElement, banner: object) => void} hooks.showBanner
 * @param {(host: HTMLElement) => void} hooks.clearBanner
 * @param {() => void} [hooks.onChange] the saved assistants changed on the server
 * @returns {{render: () => void, destroy: () => void}}
 */
export function bindAssistantList(root, { conn, showBanner, clearBanner, onChange = () => {} }) {
  const $ = (id) => root.querySelector(`#${id}`);
  const els = {
    group: $('se-ai-assts'),
    list: $('se-ai-assts-list'),
    empty: $('se-ai-assts-empty'),
    add: $('se-ai-asst-add'),
    status: $('se-ai-assts-status'),
    banner: $('se-ai-assts-banner'),
  };
  const offs = [];
  const st = { busy: false, checkSeq: 0 };
  if (!els.group || !els.list) return { render() {}, destroy() {} };

  const setStatus = (text) => { if (els.status.textContent !== text) els.status.textContent = text; };

  function rowMarkup(a, target) {
    const editing = target.kind === 'saved' && target.id === a.id;
    return html`<li class="se-ai__asst" data-id="${a.id}">
      <div class="se-ai__asst-main">
        <p class="se-ai__asst-name"><span class="se-ai__asst-title">${a.name}</span>${a.active ? html`<span class="badge badge--accent">In use</span>` : ''}${editing ? html`<span class="badge">Editing</span>` : ''}</p>
        <p class="se-ai__asst-meta mono">${C.assistantMeta(a)}</p>
      </div>
      <div class="se-ai__asst-actions">
        ${a.active ? '' : html`<button type="button" class="btn btn--secondary btn--sm" data-action="ai-asst-use" data-id="${a.id}" aria-label="Use ${a.name}"><span class="btn__label">Use</span></button>`}
        <button type="button" class="btn btn--ghost btn--sm" data-action="ai-asst-edit" data-id="${a.id}" aria-label="Edit ${a.name}"><span class="btn__label">Edit</span></button>
        <button type="button" class="btn btn--ghost btn--sm" data-action="ai-asst-check" data-id="${a.id}" aria-label="Check ${a.name}"><span class="btn__label">Check</span></button>
        <button type="button" class="btn btn--danger-ghost btn--sm" data-action="ai-asst-delete" data-id="${a.id}" aria-label="Delete ${a.name}"><span class="btn__label">Delete</span></button>
      </div>
    </li>`;
  }

  /** Draw the list from the Connection area's state; focus inside it stays on the same action of the same row. */
  function draw() {
    const { mode, visible, saved, target } = conn.state();
    const show = visible && mode === 'editable' && !!saved && !saved.locked;
    setHidden(els.group, !show);
    if (!show) return;
    const all = saved.assistants;
    const focused = typeof document !== 'undefined' ? document.activeElement : null;
    const was = focused && els.list.contains(focused) && focused.getAttribute
      ? { action: focused.getAttribute('data-action'), id: focused.getAttribute('data-id') }
      : null;
    render(els.list, html`${all.map((a) => rowMarkup(a, target))}`);
    setHidden(els.list, !all.length);
    setHidden(els.empty, all.length > 0);
    if (was && was.id) {
      const row = Array.from(els.list.querySelectorAll('li')).find((li) => li.getAttribute('data-id') === was.id);
      const again = row && (row.querySelector(`[data-action="${was.action}"]`) || row.querySelector('[data-action="ai-asst-edit"]'));
      if (again) again.focus();
    }
  }

  const nameOf = (id) => {
    const a = findAssistant(conn.state().saved, id);
    return a ? a.name : '';
  };

  function failed(title, error) {
    if (!error || error.kind === 'aborted') return;
    if (error.status === 404 || error.code === 'connection_locked') {
      const message = error.code === 'connection_locked'
        ? 'The server now sets the connection only in its .env. The list was loaded again.'
        : 'This assistant no longer exists on the server. The list was loaded again.';
      showBanner(els.banner, { title, message });
      conn.load({ force: !conn.dirty() }).then(() => onChange()).catch(() => {});
      return;
    }
    const e = api.explain(error);
    showBanner(els.banner, { title, message: e.message || 'Something went wrong.', detail: e.detailText });
  }

  /** Load the list again after a change (keeping unsaved edits), then tell the section. */
  async function changed({ force = false } = {}) {
    await conn.load({ force });
    emit('assistants:changed');
    onChange();
  }

  async function use(id, button) {
    const name = nameOf(id);
    if (st.busy || !name) return;
    clearBanner(els.banner);
    st.busy = true;
    setBusy(button, true);
    try {
      await activateAssistant(id);
      await changed();
      toast(`“${name}” is in use`, { kind: 'success' });
      // Its Use button is gone: focus stays on its row.
      const row = Array.from(els.list.querySelectorAll('li')).find((li) => li.getAttribute('data-id') === id);
      const edit = row && row.querySelector('[data-action="ai-asst-edit"]');
      if (edit) edit.focus();
    } catch (error) {
      failed(`Couldn't use “${name}”`, error);
    } finally {
      st.busy = false;
      setBusy(button, false);
    }
  }

  async function check(id, button) {
    const name = nameOf(id);
    if (!name) return;
    clearBanner(els.banner);
    const seq = (st.checkSeq += 1);
    setBusy(button, true, { label: 'Checking…' });
    setStatus(`Checking “${name}”…`);
    try {
      const result = C.checkResult(await checkAssistant(id, {}) || {});
      if (seq !== st.checkSeq) return;
      setStatus(result.tone === 'ok' ? `“${name}”: ${result.status}` : `“${name}”: ${result.title}. ${result.message}`);
    } catch (error) {
      if (seq !== st.checkSeq) return;
      const copy = C.checkFailure(error && error.code);
      if (copy) {
        const elapsed = elapsedFrom(error.detail);
        setStatus(`“${name}”: ${elapsed === null ? 'Failed' : `Failed after ${elapsedText(elapsed)}`}`);
        showBanner(els.banner, { title: `${copy.title}: “${name}”`, message: copy.message, detail: error.detailText || error.message });
      } else {
        setStatus('');
        failed(`Couldn't check “${name}”`, error);
      }
    } finally {
      setBusy(button, false);
    }
  }

  async function remove(id, button) {
    const saved = conn.state().saved;
    const a = findAssistant(saved, id);
    if (st.busy || !a) return;
    clearBanner(els.banner);
    const others = saved.assistants.filter((x) => x.id !== id);
    let next = '';
    if (a.active) next = others.length ? ` It's in use: “${others[0].name}” takes over.` : " It's in use: Open Speech goes back to the connection in its .env.";
    const ok = await confirm({
      title: `Delete the assistant “${a.name}”?`,
      message: `Its connection settings and key are forgotten.${next} AI presets that name it use the one in use instead.`,
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok || st.busy) return;
    const editingIt = conn.state().target.kind === 'saved' && conn.state().target.id === id;
    st.busy = true;
    setBusy(button, true);
    try {
      await deleteAssistant(id);
      await changed({ force: editingIt });
      toast(`Deleted “${a.name}”`, { kind: 'success' });
      els.add.focus();
    } catch (error) {
      failed(`Couldn't delete “${a.name}”`, error);
    } finally {
      st.busy = false;
      setBusy(button, false);
    }
  }

  offs.push(actions(els.group, {
    'ai-asst-use': (el) => use(el.getAttribute('data-id') || '', el),
    'ai-asst-edit': (el) => conn.edit(el.getAttribute('data-id') || ''),
    'ai-asst-check': (el) => check(el.getAttribute('data-id') || '', el),
    'ai-asst-delete': (el) => remove(el.getAttribute('data-id') || '', el),
  }, { onError: (error) => { if (globalThis.console) globalThis.console.error(error); } }));
  const onAdd = () => { conn.add().catch(() => {}); };
  els.add.addEventListener('click', onAdd);
  offs.push(() => els.add.removeEventListener('click', onAdd));

  return {
    render: draw,
    destroy() {
      st.checkSeq += 1;
      for (const off of offs.splice(0)) off();
    },
  };
}
