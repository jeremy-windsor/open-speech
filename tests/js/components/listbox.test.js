import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let env;
afterEach(() => env?.restore());

const GROUPS = [
  { label: 'Recommended', options: [{ value: 'af_heart', label: 'Heart', detail: 'American English · female' }, { value: 'af_bella', label: 'Bella' }] },
  { label: 'British English', options: [
    { value: 'bf_emma', label: 'Emma' }, { value: 'bm_george', label: 'George', disabled: true, detail: 'Unavailable' },
    ...Array.from({ length: 12 }, (_, i) => ({ value: `bx_${i}`, label: `Voice ${String.fromCharCode(65 + i)}` })),
  ] },
];

async function setup({ combobox = false, value = 'af_bella' } = {}) {
  env = installBrowserEnv({ html: '<input id="q" type="text" role="combobox" aria-label="Search" aria-expanded="true" aria-controls="lb"><div id="lb" role="listbox" aria-label="Voices"></div>' });
  const { createListbox } = await importFresh('components/listbox.js');
  const events = { active: [], selected: [] };
  const el = document.getElementById('lb');
  const input = document.getElementById('q');
  const lb = createListbox(el, {
    groups: GROUPS, value, combobox: combobox ? input : null,
    onActive: (o) => events.active.push(o && o.value), onSelect: (o) => events.selected.push(o.value),
  });
  return { lb, el, input, events, option: (v) => el.querySelectorAll('[role=option]').find((o) => lb.options()[Number(o.getAttribute('data-index'))].value === v) };
}

describe('components/listbox', () => {
  it('renders grouped options (role=group with aria-labelledby) with aria-selected and accessible names', async () => {
    const { el, option } = await setup();
    const groups = el.querySelectorAll('[role=group]');
    assert.equal(groups.length, 2);
    const label = document.getElementById(groups[0].getAttribute('aria-labelledby'));
    assert.equal(label.textContent, 'Recommended');
    assert.equal(option('af_bella').getAttribute('aria-selected'), 'true');
    assert.equal(option('af_heart').getAttribute('aria-selected'), 'false');
    assert.equal(option('bm_george').getAttribute('aria-disabled'), 'true');
    assertAccessibleNames(el);
    assert.match(option('af_heart').textContent, /Heart.*American English · female/);
  });

  it('standalone: roving focus, arrows, Home/End, PageUp/PageDown (10), Enter and Space select', async () => {
    const { lb, el, events, option } = await setup();
    const active = () => lb.activeOption().value;
    assert.equal(active(), 'af_bella', 'the selected option starts active');
    assert.equal(option('af_bella').getAttribute('tabindex'), '0', 'one tab stop');
    option('af_bella').focus();
    env.key('ArrowDown');
    assert.equal(document.activeElement, option('bf_emma'));
    env.key('ArrowDown');
    assert.equal(active(), 'bm_george', 'a disabled option can be active (its reason is read)…');
    env.key('Enter');
    assert.deepEqual(events.selected, [], '…but is never selected');
    env.key('PageDown');
    assert.equal(active(), 'bx_9', 'ten options on');
    env.key('PageUp');
    assert.equal(active(), 'bm_george');
    env.key('End');
    assert.equal(active(), 'bx_11');
    env.key('Home');
    assert.equal(active(), 'af_heart');
    env.key(' ');
    assert.deepEqual(events.selected, ['af_heart']);
    assert.equal(option('af_heart').getAttribute('aria-selected'), 'true');
    assert.equal(el.querySelectorAll('[aria-selected="true"]').length, 1);
  });

  it('type-ahead jumps by label with a 700 ms buffer', async () => {
    const { lb, option } = await setup();
    option('af_bella').focus();
    env.key('e');
    assert.equal(lb.activeOption().value, 'bf_emma', 'a first letter searches after the active option');
    env.key('m');
    assert.equal(lb.activeOption().value, 'bf_emma', 'more letters refine from the active one');
    env.clock.tick(800);
    for (const k of 'voice') env.key(k);
    assert.equal(lb.activeOption().label, 'Voice A', 'keys within 700 ms build one search');
    env.clock.tick(800);
    env.key('h');
    assert.equal(lb.activeOption().value, 'af_heart', 'after the pause a new search starts, wrapping to the top');
  });

  it('a click selects (VoiceOver on iOS clicks on double-tap); clicks on [data-act] only make it active', async () => {
    env = installBrowserEnv({ html: '<div id="lb" role="listbox" aria-label="Voices"></div>' });
    const { createListbox } = await importFresh('components/listbox.js');
    const { html } = await importFresh('core/dom.js');
    const picked = [];
    const lb = createListbox(document.getElementById('lb'), {
      groups: [{ options: [{ value: 'a', label: 'A' }, { value: 'b', label: 'B' }] }],
      renderOption: (o) => html`<span>${o.label}</span><span data-act="preview" aria-hidden="true">play</span>`,
      onSelect: (o) => picked.push(o.value),
    });
    const [a, b] = document.querySelectorAll('[role=option]');
    env.click(b.querySelector('[data-act]'));
    assert.deepEqual(picked, []);
    assert.equal(lb.activeOption().value, 'b');
    env.click(a);
    assert.deepEqual(picked, ['a']);
    assert.equal(document.querySelectorAll('[role=group]').length, 0, 'an unlabelled group renders flat');
  });

  it('combobox mode: the input keeps focus and points at the active option with aria-activedescendant', async () => {
    const { lb, input, events, option } = await setup({ combobox: true });
    input.focus();
    assert.equal(input.getAttribute('aria-activedescendant'), option('af_bella').id);
    env.key('ArrowUp');
    assert.equal(document.activeElement, input, 'focus stays in the search field');
    assert.equal(input.getAttribute('aria-activedescendant'), option('af_heart').id);
    assert.ok(option('af_heart').classList.contains('is-active'));
    assert.equal(option('af_heart').getAttribute('tabindex'), null, 'no roving tab stops with a combobox');
    env.key('Enter', { shiftKey: true });
    assert.deepEqual(events.selected, [], 'Shift+Enter is left to the caller (preview)');
    env.key('Enter');
    assert.deepEqual(events.selected, ['af_heart']);
    lb.setGroups([], {});
    assert.equal(input.getAttribute('aria-activedescendant'), null, 'no options, no active descendant');
  });

  it('setGroups keeps the active option when it is still there; setValue moves the selection', async () => {
    const { lb } = await setup();
    lb.setActive('bf_emma');
    lb.setGroups([{ label: 'British English', options: GROUPS[1].options }]);
    assert.equal(lb.activeOption().value, 'bf_emma');
    lb.setValue('bx_2');
    assert.equal(lb.value(), 'bx_2');
    assert.equal(lb.optionElement(lb.options().findIndex((o) => o.value === 'bx_2')).getAttribute('aria-selected'), 'true');
  });
});
