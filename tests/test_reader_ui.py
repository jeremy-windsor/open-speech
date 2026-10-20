"""Contracts for the Reader tab: its markup and CSS, plus Node runs of reader.js's pure helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path("src/static/index.html").read_text(encoding="utf-8")
JS = Path("src/static/reader.js").read_text(encoding="utf-8")
CSS = Path("src/static/reader.css").read_text(encoding="utf-8")
APP_JS = Path("src/static/app.js").read_text(encoding="utf-8")

PURE_START = "/* ── Pure helpers"
PURE_END = "/* ── Small DOM helpers ── */"


def _reader_panel() -> str:
    return HTML[HTML.index('id="panel-reader"') : HTML.index('id="panel-transcribe"')]


def _tag(element_id: str) -> str:
    match = re.search(rf'<[a-z]+ id="{element_id}"[^>]*>', HTML)
    assert match is not None, element_id
    return match.group(0)


def _pure_helpers() -> str:
    """reader.js's constants plus its pure-helper section, runnable outside a browser."""
    constants = JS[JS.index("const API =") : JS.index("const reader = {")]
    return constants + JS[JS.index(PURE_START) : JS.index(PURE_END)]


def _run_node(source: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser behavior tests")
    result = subprocess.run([node, "-e", source], timeout=10, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout or f"exit {result.returncode}"


NODE_CHECK = r"""
const check = (cond, label) => { if (!cond) { console.error(`failed: ${label}`); process.exit(1); } };
"""


def test_reader_tab_is_first_but_not_the_default() -> None:
    tab = _tag("tab-reader")
    assert 'data-tab="reader"' in tab
    assert 'aria-controls="panel-reader"' in tab
    assert 'aria-selected="false"' in tab
    assert 'tabindex="-1"' in tab
    assert HTML.index('id="tab-reader"') < HTML.index('id="tab-transcribe"')
    assert '<section id="panel-reader" class="panel" role="tabpanel" aria-labelledby="tab-reader" hidden>' in HTML
    assert '<section id="panel-transcribe" class="panel active"' in HTML


def test_reader_assets_load_after_the_app() -> None:
    assert HTML.index('href="/static/app.css"') < HTML.index('href="/static/reader.css"')
    assert '<script src="/static/reader.js" defer></script>' in HTML
    assert HTML.index('src="/static/app.js"') < HTML.index('src="/static/reader.js"')
    assert "viewport-fit=cover" in HTML


def test_reader_markup_has_library_reading_and_dialog_controls() -> None:
    for element_id in (
        "rd-library",
        "rd-search",
        "rd-add",
        "rd-hero",
        "rd-list",
        "rd-empty",
        "rd-empty-action",
        "rd-reading",
        "rd-text",
        "rd-chapter-list",
        "rd-chapter-list-sheet",
        "rd-chapters-dialog",
        "rd-status",
        "rd-return",
        "rd-back15",
        "rd-prev",
        "rd-play",
        "rd-next",
        "rd-fwd30",
        "rd-speed",
        "rd-voice",
        "rd-voice-dialog",
        "rd-voice-model",
        "rd-voice-select",
        "rd-add-dialog",
        "rd-file",
        "rd-paste-title",
        "rd-paste-text",
        "rd-add-submit",
    ):
        assert f'id="{element_id}"' in HTML, element_id
    assert '<p id="rd-status" class="rd-status" role="status" aria-live="polite"></p>' in HTML
    assert 'class="rd-chip rd-chip-chapters"' in HTML


def test_reader_icon_buttons_have_accessible_names() -> None:
    icon_buttons = re.findall(r'<button[^>]*class="(?:rd-iconbtn|rd-tbtn)[^"]*"[^>]*>', _reader_panel())
    assert len(icon_buttons) >= 9
    assert all("aria-label=" in button for button in icon_buttons)


def test_play_button_swaps_its_label_instead_of_a_pressed_state() -> None:
    play = _tag("rd-play")
    assert 'aria-label="Play"' in play
    assert "aria-pressed" not in play


def test_speed_chip_opens_a_list_instead_of_cycling() -> None:
    chip = _tag("rd-speed")
    assert 'aria-haspopup="menu"' in chip and 'aria-controls="rd-speed-menu"' in chip
    assert '<div id="rd-speed-menu" class="rd-speed-menu" role="menu"' in HTML
    assert 'role="menuitemradio"' in JS and "SPEEDS.map(" in JS
    assert "cycleSpeed" not in JS


def test_opening_a_document_shows_a_loading_notice() -> None:
    assert '<div id="rd-opening" class="rd-opening" role="status" aria-live="polite" hidden>' in HTML
    assert ".rd-opening[hidden] { display: none; }" in CSS
    opener = JS[JS.index("async function openDocument") : JS.index("function renderReading")]
    assert "setTimeout(() => showOpening(title), OPENING_NOTICE_MS)" in opener
    # A slower, earlier open never replaces the book opened after it.
    assert opener.count("if (ticket !== reader.opening) return;") == 3
    assert "hideOpening();" in opener


def test_dialogs_focus_their_first_choice() -> None:
    assert "autofocus" in _tag("rd-seg-file")
    # The voice list arrives after the dialog opens, so the chosen voice is focused in code.
    assert "fillVoiceOptions(pickedModel(), choice.voice, { focus: true })" in JS
    assert "(current || radios[0])?.focus({ preventScroll: true });" in JS


def test_voice_dialog_picks_with_one_click() -> None:
    # Native <select> popups need a press-and-hold in some desktop browsers; radio lists never do.
    dialog = HTML[HTML.index('<dialog id="rd-voice-dialog"') : HTML.index("</dialog>", HTML.index('<dialog id="rd-voice-dialog"'))]
    assert "<select" not in dialog
    assert '<div id="rd-voice-model" class="rd-pick-models"></div>' in dialog
    assert '<div id="rd-voice-select" class="rd-pick-voices"></div>' in dialog
    assert '<input type="radio" name="${name}"' in JS
    assert ".rd-pick-voices {" in CSS and "overflow: auto" in CSS[CSS.index(".rd-pick-voices {") :]


def test_voices_are_grouped_and_described() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
const heart = describeVoice({ id: 'af_heart', name: 'af_heart', gender: 'female' });
check(heart.group === 'Recommended' && heart.label === 'Heart', `heart ${JSON.stringify(heart)}`);
check(heart.detail === 'American English · female', `heart detail ${heart.detail}`);
const george = describeVoice({ id: 'bm_george' });
check(george.group === 'British English' && george.label === 'George' && george.detail === 'British English · male', `george ${JSON.stringify(george)}`);
check(describeVoice('jf_alpha').group === 'Japanese', 'plain id string');
check(describeVoice({ id: 'xf_new' }).group === 'Other languages', 'unknown Kokoro language');
const piper = describeVoice({ id: 'en_US-amy-medium', name: 'Amy', gender: 'unknown' });
check(piper.group === 'Voices' && piper.label === 'Amy' && piper.detail === '', `piper ${JSON.stringify(piper)}`);
check(RECOMMENDED_VOICES.every((id) => describeVoice(id).group === 'Recommended'), 'every recommended voice is grouped');
""")


def test_player_uses_two_audio_elements() -> None:
    assert '<audio id="rd-audio-a" preload="auto"></audio>' in HTML
    assert '<audio id="rd-audio-b" preload="auto"></audio>' in HTML


def test_add_dialog_offers_only_file_and_paste() -> None:
    dialog = HTML[HTML.index('id="rd-add-dialog"') : HTML.index("</dialog>", HTML.index('id="rd-add-dialog"'))]
    assert re.findall(r'data-seg="([a-z]+)"', dialog) == ["file", "paste"]
    assert 'accept=".epub,.txt,.md,.markdown,.html,.htm"' in dialog
    for phase_two in ("Link", "GitHub", "Bible"):
        assert phase_two not in dialog


def test_reading_view_is_calm_and_safe_area_aware() -> None:
    assert "body.is-reading .ambient, body.is-reading .voice-ribbon { display: none; }" in CSS
    assert "env(safe-area-inset-bottom)" in CSS[CSS.index(".rd-transport {") :]
    assert "#panel-reader.active > .rd-reading { animation: none; }" in CSS


def test_reader_css_reuses_app_tokens() -> None:
    assert "--rd-link" not in CSS
    assert "var(--accent-text)" in CSS
    phone = CSS[CSS.index("@media (max-width: 700px)") :]
    assert ".rd-dialog { width: 100vw;" in phone   # bottom sheet on phone
    assert "min-height: 44px" in phone


def test_finished_chapters_and_phone_percentages_stay_readable() -> None:
    assert not re.search(r"\.is-done[^{]*\{[^}]*opacity", CSS)   # dimming dropped the text below 4.5:1
    phone = CSS[CSS.index("@media (max-width: 700px)") :]
    pct = re.search(r"\.rd-pct:not\(\.rd-pct-new\) \{([^}]*)\}", phone)
    assert pct is not None
    assert "display: none" not in pct.group(1)   # visually hidden, still announced
    assert "clip-path: inset(50%)" in pct.group(1)


def test_api_errors_carry_the_http_status() -> None:
    api = APP_JS[APP_JS.index("async function api(") : APP_JS.index("function showToast(")]
    _run_node(NODE_CHECK + api + r"""
(async () => {
  globalThis.fetch = async () => ({ ok: false, status: 404, statusText: 'Not Found', json: async () => ({ detail: 'gone' }) });
  let error = null;
  try { await api('/x'); } catch (caught) { error = caught; }
  check(error && error.status === 404 && error.message === 'gone', 'status and message');
  globalThis.fetch = async () => ({ ok: true, headers: { get: () => 'application/json' }, json: async () => ({ ok: 1 }) });
  check((await api('/x')).ok === 1, 'json body');
})().catch((err) => { console.error(err); process.exit(1); });
""")


def test_reader_text_helpers_behave() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
check(monogram('Pride and Prejudice') === 'PP', 'monogram skips small words');
check(monogram('The Pragmatic Engineer') === 'TP', 'monogram keeps a leading small word');
check(monogram('John 3 (WEB)') === 'J3', 'monogram digits');
check(monogram('notes') === 'NO', 'monogram single word');
check(monogram('') === '?', 'monogram empty');
check(lengthLabel(100) === 'under a minute', 'length short');
check(lengthLabel(15 * 60 * 18) === '~18 min', 'length minutes');
check(lengthLabel(15 * 60 * 125) === '~2 h 05 m', 'length hours');
// Python sends code-point offsets; an emoji must not shift the following sentence.
const text = 'Hi 👋 there. Next one.';
const slice = codePointSlicer(text);
check(slice(0, 11) === 'Hi 👋 there.', 'slice astral');
check(slice(12, 21) === 'Next one.', 'slice after astral');
check(codePointSlicer('plain')(1, 3) === 'la', 'slice plain');
const parts = [
  {first_block: 4, last_block: 4},
  {first_block: 5, last_block: 8, sentences: [{block: 5, index: 0, start_ms: 0}, {block: 6, index: 0, start_ms: 2100}]},
];
check(locate(4, parts).part === parts[0] && locate(4, parts).offsetMs === 0, 'locate part start');
check(locate(6, parts).part === parts[1] && locate(6, parts).offsetMs === 2100, 'locate sentence mark');
check(locate(7, parts) === null, 'inside a part but no timing: never guess');
check(locate(9, parts) === null, 'outside every part');
""")


def test_long_documents_are_split_into_pages_of_whole_chapters() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
const valid = (pages, count, label) => {
  check(pages.length && pages[0][0] === 0 && pages.at(-1)[1] === count, `${label}: covers the document`);
  pages.forEach(([start, end], i) => {
    check(end > start && end - start <= PAGE_BLOCKS, `${label}: page ${i} size ${end - start}`);
    if (i) check(start === pages[i - 1][1], `${label}: page ${i} follows the last`);
  });
};
const chapters = (sizes) => {
  let first = 0;
  return sizes.map((size) => { const chapter = { first_block: first, block_count: size }; first += size; return chapter; });
};
// A novel-sized book is one page, so nothing changes for it.
check(JSON.stringify(pagesFor(1200, chapters(Array(30).fill(40)))) === '[[0,1200]]', 'short book: one page');
check(JSON.stringify(pagesFor(5, [])) === '[[0,5]]', 'no chapters: one page');
check(pagesFor(0, []).length === 0, 'empty document: no pages');
// A Bible-sized book: pages end on chapter boundaries, and a huge chapter is cut into even slices.
const sizes = [3, 1223, 1013, 859, 897, 2473, 915, 1052, 1052, 1026, 356];
const bible = chapters(sizes);
const count = sizes.reduce((a, b) => a + b, 0);
const pages = pagesFor(count, bible);
valid(pages, count, 'bible');
const starts = new Set(bible.map((chapter) => chapter.first_block));
const psalms = bible[5];
pages.forEach(([start]) => check(starts.has(start) || (start > psalms.first_block && start < psalms.first_block + psalms.block_count), `page starts at a chapter unless inside the huge one: ${start}`));
const slices = pages.filter(([start, end]) => start >= psalms.first_block && end <= psalms.first_block + psalms.block_count);
check(slices.length === 2 && Math.abs((slices[0][1] - slices[0][0]) - (slices[1][1] - slices[1][0])) <= 1, `huge chapter in even slices: ${JSON.stringify(slices)}`);
// Blocks before the first chapter heading belong to the first page.
valid(pagesFor(3000, [{ first_block: 10, block_count: 1490 }, { first_block: 1500, block_count: 1500 }]), 3000, 'front matter');
// pageOf finds the page holding any block.
for (const index of [0, 1, 2236, 5000, count - 1]) {
  const [start, end] = pages[pageOf(pages, index)];
  check(index >= start && index < end, `pageOf ${index}`);
}
""")


def test_reading_view_keeps_one_page_on_screen() -> None:
    cursor = JS[JS.index("function setCursor") : JS.index("// Scroll the current sentence")]
    assert "showPage(pageOf(reader.pages, index))" in cursor
    assert "reader.blockEls[" not in JS.replace("reader.blockEls[index - start]", "")
    assert 'data-page="${page}"' in JS
    # A jump to far-off text goes there at once instead of smooth-scrolling through it.
    assert "far || reducedMotion() ? 'auto' : 'smooth'" in JS


def test_loading_notice_wraps_a_long_title_on_a_phone() -> None:
    rule = re.search(r"\.rd-opening-card > span:last-child \{([^}]*)\}", CSS)
    assert rule is not None
    assert "nowrap" not in rule.group(1)
    assert "min-width: 0" in rule.group(1)


def test_polling_retries_with_backoff() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
check(isRetryable(new TypeError('Failed to fetch')), 'network error');
for (const status of [500, 502, 503, 408, 429]) check(isRetryable({ status }), `retry ${status}`);
for (const status of [400, 403, 404, 422]) check(!isRetryable({ status }), `final ${status}`);
check(backoffMs(0) === POLL_MS, 'no failures polls at the normal rate');
const waits = [1, 2, 3, 4, 5, 9].map(backoffMs);
check(waits.join() === [500, 1000, 2000, 4000, 4000, 4000].join(), `backoff ${waits}`);
let total = 0;
for (let n = 1; n < MAX_FAILURES; n += 1) total += backoffMs(n);
check(MAX_FAILURES >= 8 && total >= 15000 && total <= 60000, `gives up after ${MAX_FAILURES} failures, ${total} ms`);
""")


def test_a_parked_seek_never_reaches_the_next_source() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
const el = { dataset: {} };
check(takeParkedSeek(el) === null, 'nothing parked');
parkSeek(el, 4.25);
check(takeParkedSeek(el) === 4.25, 'parked seek applies once metadata loads');
check(takeParkedSeek(el) === null, 'and only once');
parkSeek(el, 12);
clearParkedSeek(el);   // what setSrc does before a new source
check(takeParkedSeek(el) === null, 'a new src drops the old seek');
parkSeek(el, 3);
parkSeek(el, 0.5);
check(takeParkedSeek(el) === 0.5, 'the latest seek wins');
""")


def test_previous_track_restarts_or_goes_back() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
check(previousTarget(5, RESTART_PARAGRAPH_MS + 500) === 5, 'well into the paragraph: restart it');
check(previousTarget(5, 800) === 4, 'just started: previous paragraph');
check(previousTarget(0, 0) === 0, 'never before the first block');
check(previousTarget(0, 9000) === 0, 'restart the first block');
""")


def test_voice_and_offset_resolve_once_models_are_known() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
const progress = { block_index: 3, offset_ms: 4000, model: 'kokoro', voice: 'af_heart' };
const stored = { model: 'piper', voice: 'amy' };
const speak = { model: 'kokoro', voice: 'bf_emma' };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
check(resolveVoice({ progress, stored, speak, modelIds: [] }) === null, 'models not loaded yet: undecided');
check(same(resolveVoice({ progress, stored, speak, modelIds: ['kokoro', 'piper'] }), { model: 'kokoro', voice: 'af_heart' }), 'saved voice first');
check(same(resolveVoice({ progress, stored, speak, modelIds: ['piper'] }), stored), 'then the Reader choice');
check(same(resolveVoice({ progress: null, stored: null, speak, modelIds: ['kokoro'] }), speak), 'then the Speak tab');
check(resolveVoice({ progress: null, stored: null, speak: { model: 'kokoro', voice: '' }, modelIds: ['kokoro'] }) === null, 'needs a voice');
const voice = { model: 'kokoro', voice: 'af_heart' };
check(resumeOffset(progress, voice, 3) === 4000, 'saved offset with the saved voice');
check(resumeOffset(progress, voice, 4) === 0, 'another paragraph starts at 0');
check(resumeOffset(progress, { model: 'kokoro', voice: 'bf_emma' }, 3) === 0, 'another voice starts at 0');
check(resumeOffset(progress, null, 3) === 0, 'no voice');
check(resumeOffset(null, voice, 3) === 0, 'no saved progress');
check(resumeOffset({ ...progress, offset_ms: -5 }, voice, 3) === 0, 'never negative');
""")


def test_monogram_tiles_keep_white_text_readable() -> None:
    _run_node(NODE_CHECK + _pure_helpers() + r"""
for (let hue = 0; hue < 360; hue += 1) {
  const light = tileLightness(hue, 52, 46);
  const dark = tileLightness(hue, 58, 34);
  check(light <= 46 && dark <= 34, `never lighter than the design at ${hue}`);
  check(contrastWithWhite(hslToRgb(hue, 52, light)) >= 4.5, `light stop at ${hue}`);
  check(contrastWithWhite(hslToRgb(hue, 58, dark)) >= 4.5, `dark stop at ${hue}`);
}
check(Math.abs(contrastWithWhite([0, 0, 0]) - 21) < 0.01, 'black on white is 21:1');
const colors = tileColors('pride-and-prejudice');
check(colors.hue2 === (colors.hue + 40) % 360, 'second stop is 40 degrees on');
""")


def test_reader_js_is_valid_javascript() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for syntax checks")
    subprocess.run([node, "--check", "src/static/reader.js"], check=True, timeout=10, capture_output=True)
