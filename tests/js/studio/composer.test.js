// Studio Composer: tracks with WAV-only sources, the mix, and past mixes.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { studioEnv, studioServer, teaParty, captureDownloads, delay, load, json, sameNode, noNode } from './helpers.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let t = null;
afterEach(() => {
  if (t) t.env.restore();
  t = null;
});

function seeded() {
  return studioServer({
    conversations: [
      teaParty(),
      { id: 'conv-mp3', name: 'Rendered as MP3', render_output_path: '/data/conversations/conv-mp3/render.mp3', turns: [{ speaker: 'A', text: 'x' }] },
      { id: 'conv-new', name: 'Never rendered', turns: [{ speaker: 'A', text: 'y' }] },
    ],
    mixes: [
      { id: 'mix-1', name: 'Intro bed', created_at: '2026-10-20T10:00:00Z', render_output_path: 'data/composer/render_mix-1.wav', meta: { format: 'wav', duration_ms: 65000 } },
      { id: 'mix-2', name: 'Old MP3 mix', created_at: '2026-10-19T10:00:00Z', render_output_path: 'data/composer/render_mix-2.mp3', meta: { format: 'mp3', duration_ms: 3000 } },
    ],
  });
}

const tracks = () => t.$$('.st-track');
const lastPost = () => t.calls('POST /api/composer/render').at(-1);

describe('pure helpers', () => {
  it('[REV-04] wavSources lists only WAV renders and WAV mixes (the Composer reads WAV only)', async () => {
    const { wavSources, isWav } = await load('views/studio/composer.js');
    const src = wavSources(
      [{ name: 'A', render_output_path: '/c/a/render.wav' }, { name: 'B', render_output_path: '/c/b/render.mp3' }, { name: 'C', render_output_path: null }],
      [{ name: 'M', render_output_path: 'data/composer/render_m.wav' }, { name: 'N', render_output_path: 'data/composer/render_n.mp3' }],
    );
    assert.deepEqual(src.conversations, [{ path: '/c/a/render.wav', label: 'A' }]);
    assert.deepEqual(src.mixes, [{ path: 'data/composer/render_m.wav', label: 'M' }]);
    assert.equal(isWav('x.WAV'), true);
    assert.equal(isWav('x.mp3'), false);
  });

  it('[CO-02] offsets and volumes are clamped numbers; the payload is the server’s ComposerTrack', async () => {
    const { clampOffset, clampVolume, trackPayload, activeTracks, trackNamedIn, mixFileName, volumeSpoken } = await load('views/studio/composer.js');
    assert.equal(clampOffset('1,25'), 1.3);
    assert.equal(clampOffset('-2'), 0);
    assert.equal(clampOffset('abc'), 0);
    assert.equal(clampVolume(5), 2);
    assert.equal(clampVolume(0), 0.1);
    assert.equal(volumeSpoken(1), '1.0 times');
    assert.deepEqual(trackPayload({ source_path: ' a.wav ', offset_s: '1.5', volume: '0.7', muted: 1, solo: 0 }), { source_path: 'a.wav', offset_s: 1.5, volume: 0.7, muted: true, solo: false, effects: [] });
    assert.deepEqual(activeTracks([{ k: 1 }, { k: 2, solo: true }, { k: 3, solo: true, muted: true }]).map((x) => x.k), [2]);
    assert.equal(trackNamedIn('Track 2: unknown effect', [{}, {}]), 1);
    assert.equal(trackNamedIn('Track source not found: data/b.wav', [{ source_path: 'a.wav' }, { source_path: 'data/b.wav' }]), 1);
    assert.equal(trackNamedIn('No active tracks to render', [{}]), -1);
    assert.equal(mixFileName('mp3', 1760000000000), 'composer-mix-1760000000000.mp3');
  });
});

describe('tracks', () => {
  it('[CO-01] one empty track exists at the start, with Source, Offset, Volume 0.1–2.0, Mute, Solo and Remove, each named for its track', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    assert.equal(tracks().length, 1);
    const track = tracks()[0];
    assert.equal(track.querySelector('h3').textContent, 'Track 1');
    const src = track.querySelector('select.st-src');
    assert.equal(src.value, '', 'nothing chosen yet');
    const offset = track.querySelector('[data-field="offset"]');
    assert.equal(offset.value, '0.0');
    assert.equal(offset.getAttribute('inputmode'), 'decimal');
    const volume = track.querySelector('input[type="range"]');
    assert.deepEqual([volume.min, volume.max, volume.step, volume.value], ['0.1', '2', '0.1', '1']);
    assert.equal(volume.getAttribute('aria-valuetext'), '1.0 times');
    assert.equal(track.querySelector('output').textContent, '1.0×');
    const mute = track.querySelector('[data-field="muted"]');
    const solo = track.querySelector('[data-field="solo"]');
    assert.equal(mute.getAttribute('aria-pressed'), 'false');
    assert.equal(solo.getAttribute('aria-pressed'), 'false');
    assert.equal(mute.getAttribute('aria-label'), 'Mute track 1');
    assert.equal(track.querySelector('[data-action="remove-track"]').getAttribute('aria-label'), 'Remove track 1');
    assert.equal(track.querySelector('[data-step="-1"]').disabled, true, 'the offset cannot go below 0');
    assertAccessibleNames(t.$('st-tracks'));
  });

  it('[CO-01][REV-04] the source select lists WAV conversation renders by name and WAV mixes, plus Other server path…', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    const src = tracks()[0].querySelector('select.st-src');
    const groups = [...src.querySelectorAll('optgroup')].map((g) => [g.getAttribute('label'), [...g.querySelectorAll('option')].map((o) => o.textContent)]);
    assert.deepEqual(groups, [['Conversation renders', ['The Mad Tea-Party']], ['Previous mixes', ['Intro bed']]]);
    const texts = [...src.querySelectorAll('option')].map((o) => o.textContent);
    assert.ok(!texts.includes('Rendered as MP3') && !texts.includes('Old MP3 mix') && !texts.includes('Never rendered'), texts.join('|'));
    assert.equal(texts.at(-1), 'Other server path…');
    noNode(tracks()[0].querySelector('[id$="-src-hint"]'), 'with sources listed, the intro says why only WAV shows');
    assert.match(document.getElementById('st-mix').querySelector('.st-mix__intro').textContent, /WAV/);
  });

  it('[REV-04] with no WAV render to list, the source field says how to get one', async () => {
    t = await studioEnv({ server: studioServer({ conversations: [{ id: 'c-mp3', name: 'MP3 only', render_output_path: '/x/render.mp3', turns: [{ speaker: 'A', text: 'x' }] }] }), path: 'studio/composer' });
    const src = tracks()[0].querySelector('select.st-src');
    const hint = tracks()[0].querySelector('[id$="-src-hint"]');
    assert.equal(hint.textContent, 'Render a conversation as WAV to list it here.');
    assert.equal(src.getAttribute('aria-describedby'), hint.id);
    assert.deepEqual([...src.querySelectorAll('option')].map((o) => o.textContent), ['Choose a WAV file', 'Other server path…']);
  });

  it('[CO-01] "Other server path…" reveals a path field (mono, with the data-folder hint); a listed source hides it again', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    const track = tracks()[0];
    const field = track.querySelector('.st-track__path');
    assert.equal(field.hidden, true);
    t.env.change(track.querySelector('select.st-src'), 'other');
    assert.equal(field.hidden, false);
    const path = field.querySelector('input');
    assert.ok(path.classList.contains('mono'));
    assert.equal(field.querySelector('.field__hint').textContent, 'A WAV file inside the server’s data folder.');
    t.env.change(track.querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    assert.equal(field.hidden, true);
  });

  it('[CO-02] every edit updates the track at once: offset and volume as numbers, Mute and Solo as booleans, the path as typed', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    t.env.click(t.$('st-add-track'));
    await t.settle();
    assert.equal(tracks().length, 2);
    sameNode(document.activeElement, tracks()[1].querySelector('select.st-src'), 'a new track takes focus');
    const [one, two] = tracks();
    t.env.change(one.querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    t.env.click(one.querySelector('[data-step="1"]'));
    t.env.click(one.querySelector('[data-step="1"]'));
    t.env.change(one.querySelector('input[type="range"]'), '0.7');
    assert.equal(one.querySelector('input[type="range"]').getAttribute('aria-valuetext'), '0.7 times');
    t.env.click(one.querySelector('[data-field="muted"]'));
    assert.equal(one.querySelector('[data-field="muted"]').getAttribute('aria-pressed'), 'true');
    t.env.change(two.querySelector('select.st-src'), 'other');
    t.env.input(two.querySelector('.st-path'), 'data/voices/intro.wav');
    t.env.input(two.querySelector('[data-field="offset"]'), '1.25', { change: true });
    assert.equal(two.querySelector('[data-field="offset"]').value, '1.3', 'tidied on change');
    t.env.click(two.querySelector('[data-field="solo"]'));
    await t.click(t.$('st-mix-render'));
    assert.deepEqual(lastPost().json.tracks, [
      { source_path: '/data/conversations/conv-1/render.wav', offset_s: 0.2, volume: 0.7, muted: true, solo: false, effects: [] },
      { source_path: 'data/voices/intro.wav', offset_s: 1.3, volume: 1, muted: false, solo: true, effects: [] },
    ]);
  });

  it('[CO-01] Remove track takes it out and moves focus to the next control; with none left Render mix asks for one', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    await t.click(tracks()[0].querySelector('[data-action="remove-track"]'));
    assert.equal(tracks().length, 0);
    assert.equal(t.$('st-tracks-empty').hidden, false);
    sameNode(document.activeElement, t.$('st-add-track'));
    await t.click(t.$('st-mix-render'));
    const banner = t.$('st-mix-banner');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, 'Add at least one track');
    assert.equal(t.calls('POST /api/composer/render').length, 0);
  });
});

describe('the mix', () => {
  it('[CO-03] Render mix checks each track first: a track without a source gets a field error and nothing is sent', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    await t.click(t.$('st-mix-render'));
    const src = tracks()[0].querySelector('select.st-src');
    assert.equal(src.getAttribute('aria-invalid'), 'true');
    assert.ok(src.getAttribute('aria-describedby').includes('-src-error'));
    assert.match(tracks()[0].querySelector('[id$="-src-error"]').textContent, /Choose a source for this track/);
    sameNode(document.activeElement, src);
    assert.equal(t.calls('POST /api/composer/render').length, 0);
  });

  it('[CO-03] the POST carries {name: "Composition <date>", format, sample_rate: 24000, tracks}; the button is busy and the status says Rendering…; success shows the player and reloads past mixes', async () => {
    const server = seeded();
    const original = server.routes['POST /api/composer/render'];
    t = await studioEnv({ server, path: 'studio/composer', routes: { 'POST /api/composer/render': (req) => delay(3000, () => original(req)) } });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    t.env.click(t.$('st-mix-format').querySelector('input[value="mp3"]'));
    t.env.click(t.$('st-mix-render'));
    await t.settle();
    const body = lastPost().json;
    assert.match(body.name, /^Composition \w{3} \d{1,2}, \d{4}/);
    assert.equal(body.format, 'mp3');
    assert.equal(body.sample_rate, 24000);
    assert.equal(t.$('st-mix-render').getAttribute('aria-busy'), 'true');
    assert.equal(t.$('st-mix-status-text').textContent, 'Rendering…');
    assert.equal(t.$('st-mix-elapsed').getAttribute('aria-hidden'), 'true');
    const listed = t.calls('GET /api/composer/renders').length;
    await t.settle(3500);
    await t.settle();
    assert.equal(t.$('st-mix-status-text').textContent, 'Mixed · 0:09');
    assert.equal(t.$('st-mix-play').hidden, false);
    assert.equal(t.$('st-mix-download').hidden, false);
    assert.equal(t.$('st-mix-download').textContent.trim(), 'Download MP3');
    assert.equal(t.calls('GET /api/composer/renders').length, listed + 1, 'past mixes reload');
    const [call] = t.calls('GET /api/composer/renders');
    assert.equal(call.query.get('limit'), '50');
    assert.equal(call.query.get('offset'), '0');
  });

  it('[CO-03] a failure shows a banner with the reason; a refused path (403) or a missing source (400) shows on that track', async () => {
    const server = seeded();
    let reply = () => server.envelope(400, 'Track source not found: data/voices/missing.wav');
    t = await studioEnv({ server, path: 'studio/composer', routes: { 'POST /api/composer/render': () => reply() } });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    t.env.click(t.$('st-add-track'));
    await t.settle();
    const second = tracks()[1];
    t.env.change(second.querySelector('select.st-src'), 'other');
    t.env.input(second.querySelector('.st-path'), 'data/voices/missing.wav');
    t.$('st-mix-render').focus();   // a click focuses the button in a browser; the harness's click does not
    await t.click(t.$('st-mix-render'));
    const path = tracks()[1].querySelector('.st-path');
    assert.equal(path.getAttribute('aria-invalid'), 'true');
    assert.match(tracks()[1].querySelector('[id$="-path-error"]').textContent, /Track source not found: data\/voices\/missing\.wav/);
    assert.equal(t.$('st-mix-banner').querySelector('.banner__msg').textContent, 'Check track 2.');
    assert.ok(document.activeElement === path, `focus goes from Render mix to the field: ${document.activeElement && document.activeElement.id}`);
    reply = () => server.envelope(403, 'Track source path is outside allowed roots');
    t.env.input(path, '/etc/passwd.wav');
    await t.click(t.$('st-mix-render'));
    assert.match(tracks()[1].querySelector('[id$="-path-error"]').textContent, /doesn’t allow this path/);
    reply = () => server.envelope(500, 'ffmpeg failed');
    await t.click(t.$('st-mix-render'));
    const banner = t.$('st-mix-banner');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't render the mix");
    assert.equal(banner.querySelector('.banner').getAttribute('role'), 'alert');
    assert.equal(t.$('st-mix-status-text').textContent, 'Ready');
    assert.equal(t.$('st-mix-render').hasAttribute('aria-busy'), false);
  });

  it('[CO-03] a refused track that comes back while the user edits another field leaves their focus there', async () => {
    const server = seeded();
    t = await studioEnv({ server, path: 'studio/composer', routes: { 'POST /api/composer/render': () => delay(2000, () => server.envelope(403, 'Track source path is outside allowed roots')) } });
    const track = tracks()[0];
    t.env.change(track.querySelector('select.st-src'), 'other');
    t.env.input(track.querySelector('.st-path'), '/etc/music.wav');
    t.$('st-mix-render').focus();
    t.env.click(t.$('st-mix-render'));
    await t.settle();
    const offset = tracks()[0].querySelector('[data-field="offset"]');
    offset.focus();
    await t.settle(2500);
    await t.settle();
    assert.match(tracks()[0].querySelector('[id$="-path-error"]').textContent, /doesn’t allow this path/);
    assert.ok(document.activeElement === offset, `focus stays in the field being edited: ${document.activeElement && document.activeElement.id}`);
  });

  it('[CO-04] Download saves composer-mix-<ts>.<the format that was rendered>, fetched with the key when one is set', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer', seed: { 'open-speech-api-key': 'k-9' } });
    const downloads = captureDownloads(t.env);
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    t.env.click(t.$('st-mix-format').querySelector('input[value="mp3"]'));
    await t.click(t.$('st-mix-render'));
    t.env.click(t.$('st-mix-format').querySelector('input[value="wav"]'));
    await t.click(t.$('st-mix-download'));
    assert.equal(downloads.length, 1);
    assert.match(downloads[0].download, /^composer-mix-\d+\.mp3$/, 'the format that was rendered, not the one chosen since');
    assert.match(downloads[0].href, /^blob:/);
    const fetched = t.calls('GET /api/composer/render/:id/audio').at(-1);
    assert.equal(fetched.headers.authorization, 'Bearer k-9');
  });

  it('[CO-03] the new mix plays on the shared player, homed on the Composer', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    await t.click(t.$('st-mix-render'));
    await t.click(t.$('st-mix-play'));
    const player = t.app.get().player;
    assert.match(player.id, /^mix:m\d+$/);
    assert.equal(player.homeRoute, 'studio/composer');
    assert.equal(t.$('st-mix-play').getAttribute('aria-label'), 'Pause mix');
  });
});

describe('past mixes', () => {
  it('[CO-05] lists each mix with its name, length and date, a Play and a Download button and a menu', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    const rows = t.$$('.st-mix-row');
    assert.equal(rows.length, 2);
    assert.equal(rows[0].querySelector('.row__title').textContent, 'Intro bed');
    assert.match(rows[0].querySelector('.row__meta').textContent, /^1:05 · WAV · /);
    const when = rows[0].querySelector('.row__meta span[title]');
    assert.ok(when.getAttribute('title').length > 6, 'the exact date is in the title');
    assert.equal(rows[0].querySelector('[data-action="play-past"]').getAttribute('aria-label'), 'Play Intro bed');
    assert.equal(rows[0].querySelector('[data-action="download-past"]').getAttribute('aria-label'), 'Download Intro bed');
    assertAccessibleNames(t.$('st-mixes'));
  });

  it('[CO-05] an empty list says "No mixes yet."', async () => {
    t = await studioEnv({ server: studioServer(), path: 'studio/composer' });
    assert.equal(t.$('st-mixes').textContent.trim(), 'No mixes yet.');
  });

  it('[CO-05] Play plays the mix (fetched with the key) and Download saves composer-<id>.<ext>', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer', seed: { 'open-speech-api-key': 'k-1' } });
    const downloads = captureDownloads(t.env);
    await t.click(t.$$('.st-mix-row')[1].querySelector('[data-action="download-past"]'));
    assert.deepEqual(downloads.map((d) => d.download), ['composer-mix-2.mp3']);
    await t.click(t.$$('.st-mix-row')[0].querySelector('[data-action="play-past"]'));
    const player = t.app.get().player;
    assert.equal(player.id, 'mix:mix-1');
    assert.equal(player.homeRoute, 'studio/composer');
    assert.match(document.getElementById('app-audio').src, /^blob:/, 'keyed media is fetched, never linked');
    assert.equal(t.calls('GET /api/composer/render/:id/audio').every((c) => c.headers.authorization === 'Bearer k-1'), true);
  });

  it('[CO-05] the newest Play press wins: a slow mix that lands after another mix started is dropped; a second press while it loads downloads it once', async () => {
    const server = seeded();
    const audioRoute = server.routes['GET /api/composer/render/:id/audio'];
    t = await studioEnv({
      server, path: 'studio/composer', seed: { 'open-speech-api-key': 'k-1' },
      routes: { 'GET /api/composer/render/:id/audio': (req) => delay(req.params.id === 'mix-1' ? 2000 : 100, () => audioRoute(req)) },
    });
    const play = (id) => t.$(`st-mix-${id}`).querySelector('[data-action="play-past"]');
    const revokedBefore = t.env.revokedUrls.length;
    t.env.click(play('mix-1'));
    await t.settle(50);
    t.env.click(play('mix-2'));
    await t.settle(300);
    assert.equal(t.app.get().player.id, 'mix:mix-2');
    await t.settle(2000);
    await t.settle();
    assert.equal(t.app.get().player.id, 'mix:mix-2', 'the older press never takes the player back');
    assert.equal(t.app.get().player.playing, true);
    assert.equal(t.env.revokedUrls.length - revokedBefore, 1, 'the dropped download is released');
    assert.equal(t.toasts().length, 0);

    // Play the slow mix, then the quick one, then the slow one again while it still loads: that last press wins.
    t.env.fetch.reset();
    t.env.click(play('mix-1'));
    await t.settle(50);
    t.env.click(play('mix-2'));
    await t.settle(300);
    assert.equal(t.app.get().player.id, 'mix:mix-2');
    t.env.click(play('mix-1'));
    await t.settle(2000);
    await t.settle();
    assert.equal(t.calls('GET /api/composer/render/:id/audio').filter((c) => c.params.id === 'mix-1').length, 1, 'one download for two presses');
    assert.equal(t.app.get().player.id, 'mix:mix-1', 'the newest press plays');
    assert.equal(t.app.get().player.playing, true);

    // An older press whose audio lands first, while the newest still loads, never plays.
    t.env.click(play('mix-2'));
    t.env.click(play('mix-1'));
    await t.settle(300);
    assert.notEqual(t.app.get().player && t.app.get().player.id, 'mix:mix-2', 'the quick answer of the older press is dropped');
    await t.settle(2000);
    await t.settle();
    assert.equal(t.app.get().player.id, 'mix:mix-1');
    assert.equal(t.app.get().player.playing, true);
  });

  it('[CO-05] a mix that lands after other audio started (a take in Speak) leaves that audio playing', async () => {
    const server = seeded();
    const audioRoute = server.routes['GET /api/composer/render/:id/audio'];
    t = await studioEnv({
      server, path: 'studio/composer', seed: { 'open-speech-api-key': 'k-1' },
      routes: { 'GET /api/composer/render/:id/audio': (req) => delay(2000, () => audioRoute(req)) },
    });
    const audio = await load('core/audio.js');
    t.env.click(t.$('st-mix-mix-1').querySelector('[data-action="play-past"]'));
    await t.settle(50);
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', src: 'blob:take-1' });
    await t.settle();
    assert.equal(t.app.get().player.id, 'take:1');
    await t.settle(2100);
    await t.settle();
    assert.equal(t.app.get().player.id, 'take:1', 'the late mix never displaces it');
    assert.equal(t.app.get().player.playing, true);
  });

  it('[CO-05][TE-09] Delete… confirms "Delete the mix “N”?" and then DELETEs it', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    await t.menu(t.$$('.st-mix-row')[0].querySelector('[data-menu-btn]'), 'Delete mix…');
    assert.equal(t.confirmText().title, 'Delete the mix “Intro bed”?');
    await t.answerConfirm(true);
    const [del] = t.calls('DELETE /api/composer/render/:id');
    assert.equal(del.path, '/api/composer/render/mix-1');
    assert.deepEqual(t.$$('.st-mix-row').map((r) => r.querySelector('.row__title').textContent), ['Old MP3 mix']);
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'Deleted “Intro bed”'));
    const options = [...tracks()[0].querySelectorAll('option')].map((o) => o.textContent);
    assert.ok(!options.includes('Intro bed'), 'a deleted mix is no longer a source');
  });

  it('[CO-05] a mix deleted while its audio still downloads never plays, and its failed download says nothing', async () => {
    const server = seeded();
    const audioRoute = server.routes['GET /api/composer/render/:id/audio'];
    t = await studioEnv({
      server, path: 'studio/composer', seed: { 'open-speech-api-key': 'k-1' },
      routes: { 'GET /api/composer/render/:id/audio': (req) => delay(2000, () => audioRoute(req)) },
    });
    t.env.click(t.$('st-mix-mix-1').querySelector('[data-action="play-past"]'));
    await t.settle(50);
    await t.menu(t.$('st-mix-mix-1').querySelector('[data-menu-btn]'), 'Delete mix…');
    await t.answerConfirm(true);
    assert.equal(t.calls('DELETE /api/composer/render/:id').length, 1);
    await t.settle(2100);
    await t.settle();
    assert.equal(t.app.get().player, null, 'nothing plays');
    assert.deepEqual(t.toasts().filter((x) => x.kind === 'error'), [], 'no “Couldn’t play” for a mix the listener deleted');
  });

  it('[CO-05] a past-mixes list that fails to load says so with Retry', async () => {
    const server = seeded();
    let fail = true;
    t = await studioEnv({ server, path: 'studio/composer', routes: { 'GET /api/composer/renders': (req) => (fail ? server.envelope(500, 'boom') : server.routes['GET /api/composer/renders'](req)) } });
    const state = t.$('st-mixes').querySelector('.empty--error');
    assert.equal(state.querySelector('.empty__title').textContent, "Couldn't load mixes");
    fail = false;
    await t.click(state.querySelector('[data-action="retry-load"]'));
    assert.equal(t.$$('.st-mix-row').length, 2);
  });
});

/* ───────────── repair pass (review of Studio) ───────────── */

const toastAction = (pattern) => {
  const el = [...document.querySelectorAll('.toast')].find((x) => pattern.test(x.querySelector('.toast__msg').textContent));
  return el ? el.querySelector('.btn') : null;
};
const focusedId = () => (document.activeElement && document.activeElement.id) || String(document.activeElement && document.activeElement.tagName);

describe('a mix that ends while the Composer is not on screen', () => {
  it('[CO-03] a mix that fails while Conversations shows is told by a persistent error toast with Open; its banner waits in the Composer', async () => {
    const server = seeded();
    t = await studioEnv({ server, path: 'studio/composer', routes: { 'POST /api/composer/render': () => delay(3000, () => server.envelope(500, 'ffmpeg failed')) } });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    t.env.click(t.$('st-mix-render'));
    await t.settle();
    await t.go('studio/conversations/conv-1');
    await t.settle(3500);
    await t.settle();
    assert.deepEqual(t.toasts().filter((x) => x.kind === 'error').map((x) => x.text), ["Couldn't render the mix: ffmpeg failed"]);
    await t.settle(20000);
    assert.equal(t.toasts().filter((x) => x.kind === 'error').length, 1, 'it stays until dismissed');
    assert.equal(t.$('st-mix-banner').hidden, false, 'the banner is there for the return');
    await t.click(toastAction(/Couldn't render the mix/));
    assert.equal(t.app.get().route.path, 'studio/composer');
    assert.equal(t.$('st-composer').hidden, false);
    assert.ok(document.activeElement === t.$('st-mix-title'), `focus: ${focusedId()}`);
  });

  it('[CO-03] a mix that finishes on another view says “The mix is ready” with Open, which leads to its Play button', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer', routes: { 'POST /api/composer/render': (req) => delay(3000, () => studioServer().routes['POST /api/composer/render'](req)) } });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    t.env.click(t.$('st-mix-render'));
    await t.settle();
    await t.go('history');
    await t.settle(3500);
    await t.settle();
    assert.ok(t.toasts().some((x) => x.kind === 'success' && x.text === 'The mix is ready'));
    await t.click(toastAction(/^The mix is ready$/));
    assert.equal(t.app.get().route.path, 'studio/composer');
    assert.ok(document.activeElement === t.$('st-mix-play'), `focus: ${focusedId()}`);
    assert.equal(t.$('st-mix-status-text').textContent, 'Mixed · 0:09');
  });

  it('[CO-03] a mix that ends while the Composer shows says it in its status line, with no toast', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    await t.click(t.$('st-mix-render'));
    assert.equal(t.$('st-mix-status-text').textContent, 'Mixed · 0:09');
    assert.equal(t.toasts().length, 0);
  });

  it('[CO-03] an API-key refusal says "API key needed" in the mix banner, with Open Settings', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer', routes: { 'POST /api/composer/render': () => json(401, { error: { message: 'Invalid API key' } }) } });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    await t.click(t.$('st-mix-render'));
    const banner = t.$('st-mix-banner');
    assert.equal(banner.querySelector('.banner__title').textContent, 'API key needed');
    const open = [...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Open Settings');
    assert.ok(open);
    assert.ok(![...banner.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Retry'), 'Retry would only be refused again');
    await t.click(open);
    assert.equal(t.app.get().route.view, 'settings');
  });
});

describe('sources that did not load', () => {
  it('[CO-01] when the conversations did not load, the Composer says so above the tracks with Retry, not “render a conversation as WAV”', async () => {
    const server = studioServer({ conversations: [teaParty()] });
    const base = server.routes['GET /api/conversations'];
    let fail = true;
    t = await studioEnv({ server, path: 'studio/composer', routes: { 'GET /api/conversations': (req) => (fail ? server.envelope(500, 'database is locked') : base(req)) } });
    const error = t.$('st-sources-error');
    assert.equal(error.hidden, false);
    assert.match(error.textContent, /Conversation renders didn’t load, so they aren’t listed\./);
    const src = tracks()[0].querySelector('select.st-src');
    assert.ok(src.getAttribute('aria-describedby').split(' ').includes('st-sources-error'));
    noNode(tracks()[0].querySelector('[id$="-src-hint"]'), 'the WAV hint would blame the wrong cause');
    fail = false;
    await t.click(error.querySelector('[data-action="retry-sources"]'));
    assert.equal(error.hidden, true);
    const names = [...tracks()[0].querySelectorAll('optgroup option')].map((o) => o.textContent);
    assert.deepEqual(names, ['The Mad Tea-Party']);
    assert.equal(tracks()[0].querySelector('select.st-src').hasAttribute('aria-describedby'), false);
  });

  it('[CO-01] sourcesError names the lists that never loaded; a list that loaded once keeps its items through a failed refresh', async () => {
    const { sourcesError } = await load('views/studio/composer.js');
    const ok = { status: 'ready', at: 1 };
    const failed = { status: 'error', at: 0 };
    assert.equal(sourcesError(ok, ok), '');
    assert.equal(sourcesError(failed, ok), 'Conversation renders didn’t load, so they aren’t listed.');
    assert.equal(sourcesError(ok, failed), 'Previous mixes didn’t load, so they aren’t listed.');
    assert.equal(sourcesError(failed, failed), 'Conversation renders and previous mixes didn’t load, so they aren’t listed.');
    assert.equal(sourcesError({ status: 'error', at: 5 }, ok), '', 'a failed refresh keeps what was listed');
  });
});

describe('playback on the Composer', () => {
  it('[CO-03] while a mix plays on the Composer (where the mini-player hides), an inline transport shows its position and seeks, under the result or in its row', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    const { miniPlayerState } = await load('core/audio.js');
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    await t.click(t.$('st-mix-render'));
    assert.equal(t.$('st-mix-transport').hidden, true, 'nothing plays yet');
    await t.click(t.$('st-mix-play'));
    assert.equal(t.$('st-mix-transport').hidden, true, 'not until its length is known');
    const element = document.getElementById('app-audio');
    element.__loadMetadata(9);
    element.__timeupdate(3);
    await t.settle(300);
    assert.equal(miniPlayerState(t.app.get()).visible, false, 'the Composer is the mix’s home route');
    assert.equal(t.$('st-mix-transport').hidden, false);
    const seek = t.$('st-mix-seek');
    assert.equal(seek.getAttribute('max'), '9');
    assert.equal(seek.value, '3');
    assert.equal(seek.getAttribute('aria-valuetext'), '3 seconds of 9 seconds');
    assert.equal(t.$('st-mix-seek-time').textContent, '0:03 / 0:09');
    t.env.input(seek, '6', { change: true });
    await t.settle();
    assert.equal(element.currentTime, 6, 'the range seeks the mix');
    await t.click(t.$('st-mix-mix-1').querySelector('[data-action="play-past"]'));
    element.__loadMetadata(65);
    element.__timeupdate(10);
    await t.settle(300);
    assert.equal(t.$('st-mix-transport').hidden, true, 'the result’s transport goes with its mix');
    const rowSeek = t.$('st-mix-mix-1').querySelector('input[type="range"]');
    assert.ok(rowSeek, 'the row that plays carries the transport');
    assert.equal(rowSeek.getAttribute('aria-label'), 'Position in Intro bed');
    assert.equal(rowSeek.getAttribute('max'), '65');
    assert.equal(rowSeek.value, '10');
    assert.equal(t.$$('.st-mix-row input[type="range"]').length, 1, 'only that row');
    t.env.input(rowSeek, '30', { change: true });
    await t.settle();
    assert.equal(element.currentTime, 30);
    assertAccessibleNames(t.$('st-mixes'));
  });

  it('[CO-03][REV-20] with an API key and the iPhone rule (play only inside a gesture), Play mix and a past mix’s Play both play: prime() runs inside the click', async () => {
    const server = seeded();
    const audioRoute = server.routes['GET /api/composer/render/:id/audio'];
    t = await studioEnv({
      server, path: 'studio/composer', playPolicy: 'gesture', seed: { 'open-speech-api-key': 'k-2' },
      routes: { 'GET /api/composer/render/:id/audio': (req) => delay(300, () => audioRoute(req)) },
    });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    await t.click(t.$('st-mix-render'));
    t.env.click(t.$('st-mix-play'));
    await t.settle(400);
    await t.settle();
    let player = t.app.get().player;
    assert.match(player.id, /^mix:m\d+$/);
    assert.equal(player.playing, true, `the new mix plays (${player.statusText})`);
    t.env.click(t.$('st-mix-mix-1').querySelector('[data-action="play-past"]'));
    await t.settle(400);
    await t.settle();
    player = t.app.get().player;
    assert.equal(player.id, 'mix:mix-1');
    assert.equal(player.playing, true, `the past mix plays (${player.statusText})`);
    assert.ok(t.calls('GET /api/composer/render/:id/audio').every((c) => c.headers.authorization === 'Bearer k-2'));
  });

  it('[CO-03] with autoplay refused (the mix waits paused), the result’s transport shows as soon as the mix’s metadata loads', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer', playPolicy: 'NotAllowedError' });
    t.env.change(tracks()[0].querySelector('select.st-src'), 'p:/data/conversations/conv-1/render.wav');
    await t.click(t.$('st-mix-render'));
    await t.click(t.$('st-mix-play'));
    const player = t.app.get().player;
    assert.match(player.id, /^mix:m\d+$/);
    assert.equal(player.playing, false);
    assert.equal(t.$('st-mix-transport').hidden, true);
    document.getElementById('app-audio').__loadMetadata(9);
    await t.settle();
    assert.equal(t.$('st-mix-transport').hidden, false, 'the length is known once the metadata loads');
    assert.equal(t.$('st-mix-seek-time').textContent, '0:00 / 0:09');
  });

  it('[CO-05] a past mix’s menu is written right after its button in the Past mixes card; open, it leaves the card for the view section and comes back when it closes', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    const card = t.$('st-mixes-card');
    card.style.setProperty('container-type', 'inline-size');   // .card's computed style, in the fake browser
    const more = t.$('st-mix-mix-1-more');
    const menu = document.getElementById('st-mix-mix-1-menu');
    sameNode(menu.previousElementSibling, more);
    t.env.click(more);
    await t.settle();
    sameNode(menu.parentElement, t.section);
    t.env.key('Escape');
    await t.settle();
    sameNode(menu.previousElementSibling, more, 'back after its button');
    sameNode(document.activeElement, more, 'Esc returns focus to the button');
  });

  it('[CO-05] play buttons follow which session plays and whether it does, never its position: another player’s 4 Hz updates leave the mixes alone', async () => {
    t = await studioEnv({ server: seeded(), path: 'studio/composer' });
    const audio = await load('core/audio.js');
    const mixes = t.$('st-mixes');
    let scans = 0;
    const original = mixes.querySelectorAll.bind(mixes);
    mixes.querySelectorAll = (selector) => {
      if (String(selector).includes('play-past')) scans += 1;
      return original(selector);
    };
    await audio.playClip({ id: 'take:1', owner: 'speak', kind: 'clip', title: 'Take 1', homeRoute: 'speak', src: '/x/take.mp3' });
    await t.settle();
    const element = document.getElementById('app-audio');
    element.__loadMetadata(30);
    await t.settle();
    const before = scans;
    for (let i = 1; i <= 12; i += 1) {
      element.__timeupdate(i / 4);
      await t.settle(250);
    }
    assert.ok(t.app.get().player.position >= 2, 'the other player moved');
    assert.equal(scans, before, 'no work per position update');
    element.pause();
    await t.settle();
    assert.equal(scans, before + 1, 'a change of play state syncs the buttons once');
  });
});
