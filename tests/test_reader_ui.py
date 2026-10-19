"""Static contracts for the Reader tab (src/static/reader.js, reader.css and its markup)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path("src/static/index.html").read_text(encoding="utf-8")
JS = Path("src/static/reader.js").read_text(encoding="utf-8")
CSS = Path("src/static/reader.css").read_text(encoding="utf-8")


def _source(start: str, end: str) -> str:
    return JS[JS.index(start) : JS.index(end, JS.index(start))]


def _run_node(source: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser behavior tests")
    result = subprocess.run([node, "-e", source], timeout=5, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or f"exit {result.returncode}"


def test_reader_tab_is_first_but_not_the_default() -> None:
    tab = re.search(r'<button id="tab-reader"[^>]*>', HTML)
    assert tab is not None
    assert 'data-tab="reader"' in tab.group(0)
    assert 'aria-controls="panel-reader"' in tab.group(0)
    assert 'aria-selected="false"' in tab.group(0)
    assert 'tabindex="-1"' in tab.group(0)
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
    reader = HTML[HTML.index('id="panel-reader"') : HTML.index('id="panel-transcribe"')]
    icon_buttons = re.findall(r'<button[^>]*class="(?:rd-iconbtn|rd-tbtn)[^"]*"[^>]*>', reader)
    assert len(icon_buttons) >= 9
    assert all("aria-label=" in button for button in icon_buttons)
    assert 'aria-label="Delete ${esc(doc.title)}"' in JS


def test_player_uses_two_audio_elements() -> None:
    assert '<audio id="rd-audio-a" preload="auto"></audio>' in HTML
    assert '<audio id="rd-audio-b" preload="auto"></audio>' in HTML
    assert "new Audio(" not in JS
    assert "function otherEl(el)" in JS
    assert "el.addEventListener('ended', handleEnded)" in JS
    assert "beginPart(preloaded.part, 0, reader.token, preloaded.el)" in JS


def test_add_dialog_offers_only_file_and_paste() -> None:
    dialog = HTML[HTML.index('id="rd-add-dialog"') : HTML.index("</dialog>", HTML.index('id="rd-add-dialog"'))]
    assert re.findall(r'data-seg="([a-z]+)"', dialog) == ["file", "paste"]
    assert 'accept=".epub,.txt,.md,.markdown,.html,.htm"' in dialog
    for phase_two in ("Link", "GitHub", "Bible"):
        assert phase_two not in dialog
    assert "FILE_TYPES = ['.epub', '.txt', '.md', '.markdown', '.html', '.htm']" in JS
    assert "form.append('file', reader.addFile)" in JS
    assert "JSON.stringify({ type: 'text', text, ...(title ? { title } : {}) })" in JS


def test_reader_js_follows_the_api_contract() -> None:
    assert "const API = '/v1/reader/documents';" in JS
    assert "docUrl(reader.doc.id, '/render')" in JS
    assert "JSON.stringify({ from_block: fromBlock, model: voice.model, voice: voice.voice })" in JS
    assert "docUrl(reader.doc.id, `/parts?${qs}`)" in JS
    assert "new URLSearchParams({ from_block: String(fromBlock), model: voice.model, voice: voice.voice })" in JS
    assert "docUrl(doc.id, '/progress')" in JS
    assert "method: 'PUT'" in JS
    assert "readerApi(docUrl(id), { method: 'DELETE' })" in JS
    assert "keepalive," in JS
    assert "block_index: reader.cursor, offset_ms: offset" in JS


def test_render_polls_until_ready_and_handles_errors() -> None:
    assert "const POLL_MS = 500;" in JS
    assert "await sleep(POLL_MS);" in JS
    assert "found?.part.status === 'error'" in JS
    assert "found?.part.status === 'ready' && found.part.url" in JS
    assert "requestRender(part.first_block, voice)" in JS   # extend lookahead when a part starts
    assert "el.addEventListener('error', handleAudioError)" in JS
    assert "await refreshParts(active.part.first_block, voice);" in JS   # expired signed URL


def test_preparing_audio_appears_after_one_second() -> None:
    assert "const PREPARING_DELAY_MS = 1000;" in JS
    assert "setTimeout(() => showPreparing(true), PREPARING_DELAY_MS)" in JS
    assert "setStatus('Preparing audio…')" in JS


def test_media_session_is_feature_detected_and_complete() -> None:
    for action in ("play", "pause", "previoustrack", "nexttrack", "seekbackward", "seekforward"):
        assert re.search(rf"\b{action}: \(", JS), action
    assert "'mediaSession' in navigator" in JS
    assert "typeof MediaMetadata === 'undefined'" in JS
    assert "album: 'Open Speech'" in JS
    assert "sizes: '512x512'" in JS
    assert "canvas.toDataURL('image/png')" in JS
    assert "session?.setPositionState" in JS
    assert "navigator.mediaSession.setActionHandler(action, handler)" in JS


def test_progress_is_saved_on_pause_interval_and_page_hide() -> None:
    assert "const SAVE_EVERY_MS = 10000;" in JS
    assert "setInterval(() => saveProgress(), SAVE_EVERY_MS)" in JS
    assert "if (document.hidden) saveProgress({ keepalive: true });" in JS
    assert "window.addEventListener('pagehide', () => saveProgress({ keepalive: true }));" in JS
    pause = _source("function pause()", "function togglePlay()")
    assert "saveProgress();" in pause


def test_speed_cycles_and_is_remembered() -> None:
    assert "const SPEEDS = [0.75, 1, 1.25, 1.5, 1.75, 2];" in JS
    assert "writeStorage(SPEED_KEY, String(reader.speed))" in JS
    assert "readStorage(SPEED_KEY, '1')" in JS
    assert "el.playbackRate = reader.speed;" in JS
    assert "el.defaultPlaybackRate = reader.speed;" in JS


def test_voice_defaults_to_the_speak_tab_choice() -> None:
    assert "byId('tts-model')?.value" in JS
    assert "selectedSpeechVoice()" in JS
    assert "state.ttsVoices" in JS
    assert "fetchVoices(model)" in JS


def test_keyboard_shortcuts_skip_form_fields() -> None:
    keys = _source("function handleKeydown", "/* ── Wiring ── */")
    for key in ("case ' ':", "case 'ArrowLeft':", "case 'ArrowRight':", "case 'j': case 'J':", "case 'l': case 'L':"):
        assert key in keys
    assert "input, textarea, select" in keys
    assert "seekBy(-BACK_SECONDS)" in keys
    assert "seekBy(FORWARD_SECONDS)" in keys


def test_reading_view_is_calm_deep_linkable_and_safe_area_aware() -> None:
    assert "body.is-reading .ambient, body.is-reading .voice-ribbon { display: none; }" in CSS
    assert "document.body.classList.toggle('is-reading', readingVisible())" in JS
    assert "/^#reader(?:\\/(.+))?$/" in JS
    assert "setHash(`#reader/${encodeURIComponent(reader.doc.id)}`)" in JS
    assert "setHash('#reader');" in _source("async function showLibrary", "async function openDocument")
    assert "env(safe-area-inset-bottom)" in CSS[CSS.index(".rd-transport {") :]
    assert "#panel-reader.active > .rd-reading { animation: none; }" in CSS
    assert "prefers-reduced-motion" in JS


def test_reader_css_reuses_app_tokens() -> None:
    assert "--rd-link" not in CSS
    assert "var(--accent-text)" in CSS
    phone = CSS[CSS.index("@media (max-width: 700px)") :]
    assert ".rd-dialog { width: 100vw;" in phone   # bottom sheet on phone
    assert "min-height: 44px" in phone


def test_library_failures_show_a_friendly_state() -> None:
    assert "error.status = res.status;" in JS
    assert "status === 404" in JS
    assert "status === 503" in JS
    assert "button.dataset.action = error ? 'retry' : 'add';" in JS
    delete = _source("async function deleteDocument", "function handleLibraryClick")
    assert delete.index("window.confirm(") < delete.index("method: 'DELETE'")


def test_reader_helpers_behave() -> None:
    helpers = (
        "const SMALL_WORDS = new Set(['a', 'an', 'and', 'of', 'or', 'the', 'to', 'in', 'on', 'for', 'at', 'by']);\n"
        "const CHARS_PER_SECOND = 15;\n"
        + _source("function monogram", "function tile(")
        + _source("function lengthLabel", "function plural(")
        + _source("function codePointSlicer", "function ensureSentenceSpans")
        + _source("function locate(", "async function requestRender")
    )
    _run_node(helpers + r"""
const reader = {parts: []};
const check = (cond, code) => { if (!cond) process.exit(code); };
check(monogram('Pride and Prejudice') === 'PP', 1);
check(monogram('The Pragmatic Engineer') === 'TP', 2);
check(monogram('John 3 (WEB)') === 'J3', 3);
check(monogram('notes') === 'NO', 4);
check(monogram('') === '?', 5);
check(lengthLabel(100) === 'under a minute', 6);
check(lengthLabel(15 * 60 * 18) === '~18 min', 7);
check(lengthLabel(15 * 60 * 125) === '~2 h 05 m', 8);
// Python sends code-point offsets; an emoji must not shift the following sentence.
const text = 'Hi 👋 there. Next one.';
const slice = codePointSlicer(text);
check(slice(0, 11) === 'Hi 👋 there.', 9);
check(slice(12, 21) === 'Next one.', 10);
check(codePointSlicer('plain')(1, 3) === 'la', 11);
const parts = [
  {first_block: 4, last_block: 4},
  {first_block: 5, last_block: 8, sentences: [{block: 5, index: 0, start_ms: 0}, {block: 6, index: 0, start_ms: 2100}]},
];
check(locate(4, parts).part === parts[0] && locate(4, parts).offsetMs === 0, 12);
check(locate(6, parts).part === parts[1] && locate(6, parts).offsetMs === 2100, 13);
check(locate(7, parts) === null, 14);   // inside a part but no timing: never guess
check(locate(9, parts) === null, 15);
""")


def test_reader_js_is_valid_javascript() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for syntax checks")
    subprocess.run([node, "--check", "src/static/reader.js"], check=True, timeout=10, capture_output=True)
