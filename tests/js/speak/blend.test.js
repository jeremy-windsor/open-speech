// The blend editor (inventory SP-26, SP-29).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import * as blend from '../../../src/static/js/views/speak/blend.js';
import { assertAccessibleNames } from '../harness/a11y.js';
import { mountSpeak, ready, type, settle, callsTo, json, bytes, speechBytes } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const rows = () => [...document.querySelectorAll('#sp-blend-rows [data-voice]')];
const row = (voice) => document.querySelector(`#sp-blend-rows [data-voice="${voice}"]`);

async function openBlend() {
  t.env.click(t.$('sp-blend-toggle'));
  await settle(t.env);
}

async function pickInPicker(label) {
  await settle(t.env);
  const option = [...document.querySelectorAll('#vp-list [role=option]')].find((o) => o.querySelector('.option__label').textContent === label);
  assert.ok(option, `option ${label}`);
  t.env.click(option);
  await settle(t.env);
  await settle(t.env);
}

describe('views/speak/blend (pure)', () => {
  it('[SP-26] weights step by 0.1 within 0.1–3.0, at most 6 voices, no duplicates; the last voice stays', () => {
    let items = blend.fromSpec('af_bella');
    items = blend.addVoice(items, 'af_sky');
    items = blend.addVoice(items, 'af_sky');
    assert.deepEqual(items.map((i) => i.voice), ['af_bella', 'af_sky'], 'no duplicates');
    for (const v of ['am_adam', 'bf_emma', 'bm_george', 'af_heart', 'ef_dora']) items = blend.addVoice(items, v);
    assert.equal(items.length, blend.MAX_VOICES, 'at most 6 voices');
    items = blend.setWeight(items, 'af_bella', 9);
    assert.equal(items[0].weight, 3);
    items = blend.stepWeight(items, 'af_bella', 1);
    assert.equal(items[0].weight, 3, 'not past 3.0');
    items = blend.setWeight(items, 'af_sky', '0.04');
    assert.equal(items[1].weight, 0.1, 'not under 0.1');
    items = blend.stepWeight(items, 'af_sky', -1);
    assert.equal(items[1].weight, 0.1);
    assert.deepEqual(blend.setWeight(items, 'af_sky', 'abc'), items, 'text that is not a number changes nothing');
    assert.equal(blend.clampWeight('1,5'), 1.5, 'a decimal comma is accepted');
    const two = blend.fromSpec('af_bella(2)+af_sky(1)');
    assert.equal(blend.toSpec(two), 'af_bella(2)+af_sky(1)');
    assert.deepEqual(blend.segments(two, (v) => v.slice(3)).map((s) => [s.label, s.percent, s.showLabel]), [['bella', 67, true], ['sky', 33, true]]);
    assert.equal(blend.barLabel(two, (v) => v === 'af_bella' ? 'Bella' : 'Sky'), 'Bella 67 percent, Sky 33 percent');
    assert.equal(blend.removeVoice(blend.removeVoice(two, 'af_sky'), 'af_bella').length, 1, 'one voice left is kept');
    assert.equal(blend.isMix(blend.removeVoice(two, 'af_sky')), false, 'one voice is not a blend');
    assert.deepEqual(blend.equalize(two).map((i) => i.weight), [1, 1]);
    assert.equal(blend.validateRecipe('af_bella(2)+nope', { voices: ['af_bella'] }).error, 'This model has no voice named nope.');
    assert.match(blend.validateRecipe('af_bella(0)').error, /more than 0/);
    assert.deepEqual(blend.validateRecipe('af_bella+af_sky(0.5)', { voices: ['af_bella', 'af_sky'] }).items, [{ voice: 'af_bella', weight: 1 }, { voice: 'af_sky', weight: 0.5 }]);
  });
});

describe('views/speak blend editor (view)', () => {
  it('[SP-26] Mix voices opens the editor; Add voice picks from the model\'s voices; rows carry named steppers, % and a Remove button', async () => {
    t = await mountSpeak();
    await ready(t);
    assert.equal(t.$('sp-blend').hidden, true, 'hidden until Mix voices');
    assert.equal(t.$('sp-blend-toggle').hidden, false, 'Kokoro blends');
    assert.equal(t.$('sp-blend-toggle').getAttribute('aria-controls'), 'sp-blend');
    await openBlend();
    assert.equal(t.$('sp-blend').hidden, false);
    assert.equal(document.activeElement, t.$('sp-blend-add'), 'focus moves into the editor');
    assert.equal(rows().length, 1, 'seeded with the current voice');
    t.env.click(t.$('sp-blend-add'));
    await pickInPicker('Sky');
    assert.deepEqual(rows().map((r) => r.getAttribute('data-voice')), [t.parts.inspector.state().mix[0].voice, 'af_sky']);
    assertAccessibleNames(t.$('sp-blend'));
    const sky = row('af_sky');
    assert.equal(sky.querySelector('[data-blend-act="weight"]').getAttribute('aria-label'), 'Weight for Sky');
    assert.equal(sky.querySelector('[data-blend-act="up"]').getAttribute('aria-label'), 'Increase weight for Sky');
    assert.equal(sky.querySelector('[data-blend-act="remove"]').getAttribute('aria-label'), 'Remove Sky');
    t.env.click(sky.querySelector('[data-blend-act="up"]'));
    t.env.click(sky.querySelector('[data-blend-act="up"]'));
    await settle(t.env);
    assert.equal(sky.querySelector('[data-blend-act="weight"]').value, '1.2');
    assert.equal(document.activeElement, sky.querySelector('[data-blend-act="up"]'), 'the stepper keeps focus (rows update in place)');
    assert.equal(sky.querySelector('.sp-blend__pct').textContent, '55%');
    assert.match(t.$('sp-blend-string').textContent, /\(1\)\+af_sky\(1\.2\)$/);
    assert.match(t.$('sp-blend-bar').getAttribute('aria-label'), /percent, Sky 55 percent$/);
    assert.equal(t.$('sp-blend-bar').querySelectorAll('.blend-bar__seg').length, 2);
    // Typed weights are clamped to 0.1–3.0.
    const input = sky.querySelector('[data-blend-act="weight"]');
    input.value = '7';
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(t.env);
    assert.equal(input.value, '3');
    assert.equal(t.parts.inspector.state().mix[1].weight, 3);
    assert.equal(sky.querySelector('[data-blend-act="up"]').disabled, true, 'up disabled at the maximum');
    // The blend is sent in the voice field as the canonical string.
    type(t, 'Blend me.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    assert.match(callsTo(t, 'POST /v1/audio/speech')[0].json.voice, /^af_\w+\(1\)\+af_sky\(3\)$/);
    // Remove: focus moves on, one voice left is not a blend.
    t.env.click(row('af_sky').querySelector('[data-blend-act="remove"]'));
    await settle(t.env);
    assert.equal(rows().length, 1);
    assert.equal(row(t.parts.inspector.state().voice).querySelector('[data-blend-act="remove"]').disabled, true, 'the last voice cannot be removed');
    assert.equal(t.parts.inspector.voiceSpec(), t.parts.inspector.state().voice, 'one voice left sends that voice');
  });

  it('[SP-26] editing the blend clears the named voice and the preset; Equalize and Paste blend… work; max 6 voices', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.click(t.$('sp-presets').querySelector('[data-preset-id="p1"]'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-presets').querySelector('[data-preset-id="p1"]').getAttribute('aria-pressed'), 'true');
    // A named voice with a blend version shows the editor at once.
    await t.parts.inspector.commitPrepared(await t.parts.inspector.prepared('kokoro'), { ...t.parts.inspector.currentSetup(), identity: 'n1' });
    await settle(t.env);
    assert.equal(t.parts.inspector.voiceSpec(), 'voice:n1');
    assert.equal(t.$('sp-blend').hidden, false, 'a blend voice shows the editor');
    t.env.click(t.$('sp-blend-equal'));
    await settle(t.env);
    assert.equal(t.parts.inspector.state().identity, '', 'editing clears the named voice');
    assert.equal(t.$('sp-presets').querySelector('[data-preset-id="p1"]').getAttribute('aria-pressed'), 'false', 'and the preset');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_bella(1)+af_sky(1)');
    // Paste blend… validates with the server grammar against the model's voices.
    t.env.click(t.$('sp-blend-paste'));
    await settle(t.env);
    const field = document.getElementById('app-prompt-input');
    field.value = 'af_bella(2)+unknown_voice';
    document.getElementById('app-prompt-form').requestSubmit();
    await settle(t.env);
    assert.equal(document.getElementById('app-prompt-error').hidden, false);
    assert.equal(document.getElementById('app-prompt-error').textContent, 'This model has no voice named unknown_voice.');
    field.value = 'af_heart(1.5)+am_adam+bf_emma+bm_george+af_sky+af_bella';
    document.getElementById('app-prompt-form').requestSubmit();
    await settle(t.env);
    await settle(t.env);
    assert.equal(rows().length, 6);
    assert.equal(t.$('sp-blend-add').getAttribute('aria-disabled'), 'true', 'Add voice is off at 6 voices');
    assert.equal(t.parts.inspector.voiceSpec(), 'af_heart(1.5)+am_adam(1)+bf_emma(1)+bm_george(1)+af_sky(1)+af_bella(1)');
  });

  it('[SP-29] a model without voice_blend drops the blend; returning to Kokoro offers Mix voices again', async () => {
    t = await mountSpeak();
    await ready(t);
    await openBlend();
    t.env.click(t.$('sp-blend-add'));
    await pickInPicker('Sky');
    assert.equal(t.parts.inspector.state().mix.length, 2);
    t.env.change(t.$('sp-model'), 'piper/en_US-amy-medium');
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.state().mix.length, 0, 'the blend is dropped');
    assert.equal(t.$('sp-blend').hidden, true);
    assert.equal(t.$('sp-blend-toggle').hidden, true, 'Piper has no voice_blend');
    assert.equal(t.parts.inspector.voiceSpec(), 'en_US-amy-medium');
    t.env.change(t.$('sp-model'), 'kokoro');
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-blend-toggle').hidden, false);
  });

  it('[SP-26] Preview blend plays a short sample of the canonical blend without adding a History entry', async () => {
    t = await mountSpeak();
    await ready(t);
    await openBlend();
    t.env.click(t.$('sp-blend-add'));
    await pickInPicker('Sky');
    t.env.click(t.$('sp-blend-preview'));
    await settle(t.env);
    await settle(t.env);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.json.voice, t.parts.inspector.voiceSpec());
    assert.equal(post.headers['x-history'], undefined, 'previews stay out of History');
    const { app } = await import('../../../src/static/js/core/store.js');
    assert.equal(app.get().player?.id, 'preview');
    assert.equal(app.get().player?.ephemeral, true);
  });

  it('[SP-26] a failed Preview blend says so inside the blend editor, next to the button (role=alert), not in a toast', async () => {
    let fail = true;
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => (fail ? json(500, { error: { message: 'CUDA out of memory', code: 'tts_error' } }) : bytes(200, speechBytes(), 'audio/mpeg')) } });
    await ready(t);
    await openBlend();
    t.env.click(t.$('sp-blend-add'));
    await pickInPicker('Sky');
    t.env.click(t.$('sp-blend-preview'));
    await settle(t.env);
    await settle(t.env);
    const error = t.$('sp-blend-error');
    assert.equal(error.hidden, false);
    assert.equal(error.getAttribute('role'), 'alert');
    assert.equal(t.$('sp-blend').contains(error), true);
    assert.equal(t.$('sp-blend-error-text').textContent, "Couldn't preview the blend. CUDA out of memory");
    assert.equal(document.getElementById('app-toasts').textContent.includes('preview'), false, 'no toast');
    assert.equal(t.$('sp-blend-preview').disabled, false, 'the button is usable again');
    // The next attempt (or any edit) clears it.
    fail = false;
    t.env.click(t.$('sp-blend-preview'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(error.hidden, true);
  });

  it('[SP-26] Mix voices closes the editor again while it holds one voice; a real blend keeps it open', async () => {
    t = await mountSpeak();
    await ready(t);
    const toggle = t.$('sp-blend-toggle');
    const before = t.parts.inspector.voiceSpec();
    await openBlend();
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(toggle.hidden, false, 'one voice: the toggle stays, to close the editor again');
    t.env.click(toggle);
    await settle(t.env);
    assert.equal(t.$('sp-blend').hidden, true, 'closed');
    assert.equal(toggle.getAttribute('aria-expanded'), 'false');
    assert.equal(t.parts.inspector.voiceSpec(), before, 'nothing the take uses changed');
    await openBlend();
    t.env.click(t.$('sp-blend-add'));
    await pickInPicker('Sky');
    assert.equal(toggle.hidden, true, 'a blend of two voices keeps the editor open');
    t.env.click(row('af_sky').querySelector('[data-blend-act="remove"]'));
    await settle(t.env);
    assert.equal(toggle.hidden, false, 'back to one voice: it can be closed again');
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  });
});
