"""Behavior and accessibility contracts for the dependency-free web UI."""

from __future__ import annotations

import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

HTML = Path("src/static/index.html").read_text(encoding="utf-8")
CSS = Path("src/static/app.css").read_text(encoding="utf-8")
JS = Path("src/static/app.js").read_text(encoding="utf-8")


def _source(start: str, end: str) -> str:
    return JS[JS.index(start) : JS.index(end, JS.index(start))]


def _run_node(source: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser behavior tests")
    subprocess.run([node, "-e", source], check=True, timeout=5, capture_output=True, text=True)


def test_transcription_response_parser_accepts_text_and_json() -> None:
    parser = _source(
        "async function parseTranscriptionResponse",
        "async function transcribeFile",
    )
    _run_node(parser + r"""
(async () => {
  let textReads = 0;
  let jsonReads = 0;
  const textResult = await parseTranscriptionResponse({
    headers: {get: () => 'text/plain; charset=utf-8'},
    text: async () => { textReads += 1; return 'spoken words'; },
    json: async () => { throw new Error('text response parsed as JSON'); },
  });
  if (textResult.text !== 'spoken words' || textReads !== 1) process.exit(1);

  const jsonResult = await parseTranscriptionResponse({
    headers: {get: () => 'application/json'},
    text: async () => { throw new Error('JSON response parsed as text'); },
    json: async () => { jsonReads += 1; return {text: 'json words'}; },
  });
  if (jsonResult.text !== 'json words' || jsonReads !== 1) process.exit(2);
})().catch(() => process.exit(3));
""")


def test_history_storage_recovers_from_corrupt_or_unavailable_storage() -> None:
    storage = _source("function readStorage", "function formatSize")
    history = _source("function readLocalHistory", "function pushHistory")
    _run_node(storage + history + r"""
let removed = 0;
const localStorage = {
  getItem: () => '{broken json',
  setItem: () => {},
  removeItem: () => { removed += 1; },
};
if (readLocalHistory('history').length !== 0 || removed !== 1) process.exit(1);
localStorage.getItem = () => '{"unexpected":"object"}';
if (readLocalHistory('history').length !== 0 || removed !== 2) process.exit(2);
localStorage.getItem = () => '[{"text":"kept"}]';
if (readLocalHistory('history')[0].text !== 'kept') process.exit(3);
localStorage.getItem = () => { throw new Error('SecurityError'); };
if (readLocalHistory('history').length !== 0) process.exit(4);
localStorage.setItem = () => { throw new Error('QuotaExceededError'); };
if (writeStorage('history', '[]') !== false) process.exit(5);
""")


def test_theme_initialization_survives_unavailable_storage() -> None:
    storage = _source("function readStorage", "function formatSize")
    theme = _source("function initTheme", "function tabKeyTargetIndex")
    _run_node(storage + theme + r"""
const localStorage = {
  getItem: () => { throw new Error('SecurityError'); },
  setItem: () => { throw new Error('SecurityError'); },
};
const attrs = {};
const toggle = {textContent: '', onclick: null};
const document = {documentElement: {
  setAttribute: (key, value) => { attrs[key] = value; },
  getAttribute: (key) => attrs[key],
}};
function byId(id) { if (id !== 'theme-toggle') process.exit(1); return toggle; }
initTheme();
if (attrs['data-theme'] !== 'dark' || toggle.textContent !== '☀️') process.exit(2);
toggle.onclick();
if (attrs['data-theme'] !== 'light' || toggle.textContent !== '🌙') process.exit(3);
""")


def test_model_delete_confirms_and_uses_artifact_route() -> None:
    delete_model = _source("async function deleteModel", "async function runModelOp")
    _run_node(delete_model + r"""
let confirmed = false;
let calls = [];
const window = {confirm: () => confirmed};
async function api(url, options) { calls.push([url, options]); }
(async () => {
  if (await deleteModel('piper/en_US-lessac-medium') !== false || calls.length) process.exit(1);
  confirmed = true;
  if (await deleteModel('piper/en_US-lessac-medium') !== true) process.exit(2);
  if (calls.length !== 1) process.exit(3);
  if (calls[0][0] !== '/api/models/piper%2Fen_US-lessac-medium/artifacts') process.exit(4);
  if (calls[0][1].method !== 'DELETE') process.exit(5);
})().catch(() => process.exit(6));
""")


def test_persistent_delete_actions_require_confirmation() -> None:
    delete_profile = _source("async function deleteProfile", "async function setDefaultProfile")
    delete_history = _source("async function deleteHistoryEntry", "async function clearHistory")
    delete_conversation = _source("async function deleteConversation", "async function createConversation")
    delete_turn = _source("async function deleteTurn", "async function renderConversation")
    _run_node(delete_profile + delete_history + delete_conversation + delete_turn + r"""
let confirmed = false;
let calls = [];
let reloads = 0;
const window = {confirm: () => confirmed};
const state = {
  history: {type: '', limit: 50, offset: 0},
  currentConversationId: 'conversation/1',
  currentConversation: null,
};
async function api(url, options) {
  calls.push([url, options]);
  return {id: 'conversation/1', turns: []};
}
async function loadProfiles() { reloads += 1; }
async function loadHistory() { reloads += 1; }
async function loadConversations() { reloads += 1; }
function renderStudioTurns() { reloads += 1; }
(async () => {
  if (await deleteProfile('profile/1') !== false) process.exit(1);
  if (await deleteHistoryEntry('history/1') !== false) process.exit(2);
  if (await deleteConversation('conversation/1') !== false) process.exit(3);
  if (await deleteTurn('turn/1') !== false) process.exit(4);
  if (calls.length || reloads) process.exit(5);
  confirmed = true;
  await deleteProfile('profile/1');
  await deleteHistoryEntry('history/1');
  await deleteConversation('conversation/1');
  state.currentConversationId = 'conversation/1';
  await deleteTurn('turn/1');
  const deleteUrls = calls.filter((call) => call[1]?.method === 'DELETE').map((call) => call[0]);
  if (!deleteUrls.includes('/api/profiles/profile%2F1')) process.exit(6);
  if (!deleteUrls.includes('/api/history/history%2F1')) process.exit(7);
  if (!deleteUrls.includes('/api/conversations/conversation%2F1')) process.exit(8);
  if (!deleteUrls.includes('/api/conversations/conversation%2F1/turns/turn%2F1')) process.exit(9);
})().catch(() => process.exit(10));
""")


def test_tab_keyboard_navigation_wraps_and_activates_target() -> None:
    helpers = _source("function tabKeyTargetIndex", "function initTabs")
    _run_node(helpers + r"""
if (tabKeyTargetIndex('ArrowRight', 2, 3) !== 0) process.exit(1);
if (tabKeyTargetIndex('ArrowLeft', 0, 3) !== 2) process.exit(2);
if (tabKeyTargetIndex('Home', 2, 3) !== 0) process.exit(3);
if (tabKeyTargetIndex('End', 0, 3) !== 2) process.exit(4);
if (tabKeyTargetIndex('Enter', 0, 3) !== -1) process.exit(5);
let focused = -1;
let clicked = -1;
let prevented = false;
const tabs = [0, 1, 2].map((index) => ({
  focus: () => { focused = index; },
  click: () => { clicked = index; },
}));
handleTabKeydown({
  key: 'ArrowLeft',
  currentTarget: tabs[0],
  preventDefault: () => { prevented = true; },
}, tabs);
if (!prevented || focused !== 2 || clicked !== 2) process.exit(6);
""")
    assert JS.count("t.tabIndex = active ? 0 : -1;") == 2
    assert "handleTabKeydown(event, tabs)" in JS
    assert "handleTabKeydown(event, modelTabs)" in JS


def test_provider_collapse_hides_body_from_focus_and_accessibility_tree() -> None:
    collapse = _source("function toggleProviderCard", "function bindEvents")
    _run_node(collapse + r"""
let collapsed = false;
const body = {hidden: false};
const header = {
  classList: {toggle: () => { collapsed = !collapsed; return collapsed; }},
  nextElementSibling: null,
};
let expanded = '';
const button = {
  closest: () => header,
  getAttribute: () => 'provider-test-body',
  setAttribute: (name, value) => { if (name === 'aria-expanded') expanded = value; },
};
function byId(id) { return id === 'provider-test-body' ? body : null; }
toggleProviderCard(button);
if (!body.hidden || expanded !== 'false') process.exit(1);
toggleProviderCard(button);
if (body.hidden || expanded !== 'true') process.exit(2);
""")
    assert JS.count('aria-controls="${bodyId}"') >= 6
    assert JS.count('id="${bodyId}" class="provider-card-body') >= 6


def test_missing_provider_install_command_deduplicates_builtin_providers() -> None:
    renderer = _source(
        "function renderNotInstalledCard",
        "function renderUnavailableWorkerCard",
    )
    _run_node(renderer + r"""
const esc = (value) => String(value);
const providerBodyId = (provider, variant) => `${provider}-${variant}`;
const cases = [
  ['kokoro', 'BAKED_PROVIDERS=kokoro,piper .'],
  ['piper', 'BAKED_PROVIDERS=kokoro,piper .'],
  ['qwen3', 'BAKED_PROVIDERS=kokoro,piper,qwen3 .'],
];
for (const [provider, expected] of cases) {
  const card = renderNotInstalledCard(provider, provider, 'description');
  if (!card.includes(expected)) process.exit(1);
  const providers = card.match(/BAKED_PROVIDERS=([^ ]+)/)?.[1].split(',') || [];
  if (providers.length !== new Set(providers).size) process.exit(2);
}
""")
    assert "max-height: 800px" not in CSS


class _FormLabelAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.labels_for: set[str] = set()
        self.controls: list[tuple[str, str, bool, bool]] = []

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs_list)
        inside_label = "label" in self.stack
        if tag == "label" and attrs.get("for"):
            self.labels_for.add(str(attrs["for"]))
        if tag in {"input", "select", "textarea"} and attrs.get("id"):
            hidden = attrs.get("type") == "hidden"
            aria_named = bool(attrs.get("aria-label") or attrs.get("aria-labelledby"))
            self.controls.append((str(attrs["id"]), tag, inside_label or aria_named, hidden))
        if tag not in {"input", "meta", "link", "br", "hr", "img", "source"}:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag not in self.stack:
            return
        index = len(self.stack) - 1 - self.stack[::-1].index(tag)
        del self.stack[index:]


def test_visible_form_controls_have_accessible_names() -> None:
    audit = _FormLabelAudit()
    audit.feed(HTML)
    unlabeled = [
        f"{tag}#{control_id}"
        for control_id, tag, internally_named, hidden in audit.controls
        if not hidden and not internally_named and control_id not in audit.labels_for
    ]
    assert unlabeled == []
    assert '<label for="tts-input">Text to speak or read</label>' in HTML


def test_tabs_start_with_one_roving_tab_stop_per_tablist() -> None:
    main_tabs = re.findall(r'<button[^>]*class="tab(?: [^"]*)?"[^>]*>', HTML)
    model_tabs = re.findall(r'<button[^>]*class="models-tab(?: [^"]*)?"[^>]*>', HTML)
    assert len(main_tabs) == 7
    assert sum('tabindex="0"' in tab for tab in main_tabs) == 1
    assert sum('tabindex="-1"' in tab for tab in main_tabs) == 6
    assert len(model_tabs) == 2
    assert sum('tabindex="0"' in tab for tab in model_tabs) == 1
    assert sum('tabindex="-1"' in tab for tab in model_tabs) == 1


def test_mobile_form_rows_reset_desktop_flex_basis() -> None:
    mobile = CSS[CSS.index("@media (max-width: 500px)") :]
    assert ".form-row { align-items: stretch; flex-direction: column; gap: 10px; }" in mobile
    reset = re.search(
        r"\.form-row\.two-col > \*,\s*\.mic-device-field,\s*"
        r"\.tts-reference-row select\s*\{(?P<body>[^}]+)\}",
        mobile,
    )
    assert reset is not None
    body = reset.group("body")
    for declaration in ("flex: 0 1 auto", "max-width: none", "min-width: 0", "width: 100%"):
        assert declaration in body


def test_settings_only_expose_working_controls() -> None:
    for removed_id in (
        "profile-new",
        "profile-dialog",
        "profile-form",
        "profile-save",
        "profile-name",
        "history-enabled",
        "history-max-entries",
        "history-retain-audio",
    ):
        assert f'id="{removed_id}"' not in HTML
    for variable in (
        "OS_HISTORY_ENABLED",
        "OS_HISTORY_MAX_ENTRIES",
        "OS_HISTORY_RETAIN_AUDIO",
    ):
        assert variable in HTML
    assert 'id="settings-clear-history"' in HTML
