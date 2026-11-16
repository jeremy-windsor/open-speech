"""Settings (V7): the Node behaviour tests, the partial's structure and the view CSS.

The behaviour (the API key card, the auth hand-off, appearance, playback, server, shortcuts, the table of
contents and the Reader appearance hand-off) is tested in tests/js/settings/*.test.js; this file runs them
and checks what only the static files can show, including that History offers only working controls.
"""

from __future__ import annotations

import re

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

PARTIAL = ui.PARTIALS_DIR / "settings.html"
CSS = ui.CSS_ROOT / "views" / "settings.css"
JS_DIR = ui.JS_ROOT / "views" / "settings"
SECTIONS = ("appearance", "playback", "connection", "server", "assistant", "history", "shortcuts", "developer", "about")
LOGIC = JS_DIR / "assistant-logic.js"


def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


def node(id_: str) -> ui.Node:
    el = ui.by_id(tree(), id_)
    assert el is not None, f"#{id_} is missing from settings.html"
    return el


def section(name: str) -> ui.Node:
    heading = node(f"se-{name}-title")
    host = next(a for a in heading.ancestors() if a.tag == "section")
    assert host.get("aria-labelledby") == heading.id
    return host


def decls(selector: str, chain_has: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for sel, body, chain in ui.css_rules(CSS):
        conditional = [c for c in chain if c.startswith(("@media", "@supports", "@container"))]
        wanted = any(chain_has in c for c in conditional) if chain_has else not conditional
        if wanted and any(part.strip() == selector for part in sel.split(",")):
            out.update(dict(body))
    return out


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("settings"))
def test_js(path):
    ui.run_node_test(path, timeout=240)


def test_every_v7_inventory_id_and_defect_has_a_test():
    """Every id and defect V7 owns, Models and Settings (15 ids, 3 defects), leads a test title that runs in
    its assigned file."""
    assert missing("V7") == []


# ───────────────────────────── the partial ─────────────────────────────


def test_sections_in_order_with_their_heading_ids():
    """Nine sections (the seven settings sections, the AI assistant after Server, and
    Developer before About), each a card labelled by its h2 se-<name>-title (the
    shell's Connection settings hand-off focuses #se-connection-title); one focusable h1; no placeholder marker."""
    t = tree()
    h1s = ui.select(t, "h1")
    assert len(h1s) == 1 and h1s[0].id == "se-title" and h1s[0].get("tabindex") == "-1" and h1s[0].has("data-view-heading")
    h2s = [h.id for h in ui.select(t, "h2")]
    assert h2s == [f"se-{name}-title" for name in SECTIONS]
    for name in SECTIONS:
        assert "card" in section(name).classes
    assert not ui.select(t, "[data-stub-body], [data-stub]")


def test_the_table_of_contents_lists_every_section_and_never_routes_away():
    """The in-page links point at the section headings and keep the route (#settings)."""
    toc = node("se-toc")
    assert toc.tag == "nav" and toc.get("aria-label") == "Settings sections"
    links = ui.select(toc, "a.se-toc__link")
    assert [link.get("data-target") for link in links] == [f"se-{name}-title" for name in SECTIONS]
    assert {link.get("href") for link in links} == {"#settings"}


def test_se01_the_api_key_card_has_its_field_buttons_and_notes():
    """[SE-01] Connection: a password field (autocomplete off, no spellcheck, no paste blocking) with the
    visible label "API key", its hint and error slot, Show (aria-pressed), Save, Test, Clear, the role=status
    result line and the ticket note (WebSocket tickets, not the key, in URLs)."""
    field = node("se-api-key")
    assert field.get("type") == "password"
    assert field.get("autocomplete") == "off" and field.get("spellcheck") == "false"
    assert not any(attr.startswith("on") for attr in field.attrs), "paste is never blocked"
    assert field.get("value") is None
    assert field.get("aria-describedby") == "se-api-key-hint se-api-key-error"
    assert ui.accessible_name(field, tree()) == "API key"
    assert ui.visible_text(node("se-api-key-hint")) == "Needed when the server sets OS_API_KEY. Kept in this browser only."
    assert node("se-api-key-error").has("hidden")
    assert node("se-key-show").get("aria-pressed") == "false"
    assert [ui.visible_text(node(i)) for i in ("se-key-save", "se-key-test", "se-key-clear")] == ["Save", "Test", "Clear"]
    assert node("se-key-status").get("role") == "status"
    note = ui.visible_text(section("connection"))
    assert "one-time ticket made from this key. The key itself is never put in a URL." in note
    assert ui.select(section("connection"), "form") == [], "no <form>: a password manager would offer to save the key"


def test_se06_history_shows_only_working_controls():
    """[SE-06] History names the two server settings (restart required) and links to History, where the one
    Clear all lives: no toggles for settings only the server can change, and no second Clear."""
    s = section("history")
    text = ui.visible_text(s)
    assert "OS_HISTORY_ENABLED" in text and "OS_HISTORY_MAX_ENTRIES" in text and "restart required" in text
    assert ui.select(s, "input, select, textarea, button") == []
    link = node("se-history-link")
    assert link.tag == "a" and link.get("href") == "#history" and ui.visible_text(link) == "Manage history"
    whole = ui.visible_text(tree())
    assert "Clear all" not in whole and "Clear All" not in whole


def test_bug04_os_history_retain_audio_is_gone_from_settings():
    """[BUG-04] OS_HISTORY_RETAIN_AUDIO had no effect and is no longer mentioned in Settings: not in the
    partial, the Settings modules or the stylesheet (nor in README.md and .env.example)."""
    files = [PARTIAL, CSS, *sorted(JS_DIR.glob("*.js"))]
    for path in files:
        assert "RETAIN_AUDIO" not in path.read_text(encoding="utf-8"), path.name


def test_appearance_and_playback_controls_are_labelled_groups():
    """Every segmented control is a radiogroup with a visible label; the speed presets are
    role=radio chips; the slider has the 0.5–3.5× range in 0.05 steps; the switches are role=switch."""
    t = tree()
    groups = {g.get("id") or ui.by_id(t, g.get("aria-labelledby")).text_content(): g for g in ui.select(t, '[role="radiogroup"]')}
    assert len(groups) == 8   # theme, motion, open on, speed presets, skip back, skip forward, step, track
    for group in groups.values():
        label = ui.by_id(t, group.get("aria-labelledby") or "")
        assert label is not None and ui.visible_text(label)
    names = {i.get("name") for i in ui.select(t, 'input[type="radio"]')}
    assert names == {"se-theme", "se-motion", "se-start", "se-skip-back", "se-skip-forward", "se-step", "se-track"}
    values = lambda name: [i.get("value") for i in ui.select(t, f'input[name="{name}"]')]  # noqa: E731
    assert values("se-theme") == ["system", "light", "dark"]
    assert values("se-motion") == ["system", "reduce", "full"]
    assert values("se-start") == ["home", "last"]   # Appearance › Open on (prefs.startOn)
    assert values("se-skip-back") == ["5", "10", "15", "30", "60"]
    assert values("se-skip-forward") == ["10", "15", "30", "60"]
    assert values("se-step") == ["paragraph", "sentence"]
    assert values("se-track") == ["paragraph", "chapter"]
    chips = ui.select(node("se-speed-presets"), '[role="radio"]')
    assert [c.get("data-speed") for c in chips] == ["0.75", "1", "1.1", "1.25", "1.5", "1.75", "2", "2.5", "3"]
    speed = node("se-speed")
    assert (speed.get("min"), speed.get("max"), speed.get("step")) == ("0.5", "3.5", "0.05")
    switches = [i.id for i in ui.select(t, 'input[role="switch"]')]
    assert switches == [
        "se-snap", "se-fade", "se-wake", "se-ai-conn-tls", "se-ai-conn-template",
        "se-ai-conn-stream",
        "se-ai-show-thinking", "se-ai-speak-errors", "se-shortcuts", "se-show-live",
    ]
    assert "Not available on iPhone" in ui.visible_text(node("se-fade-hint"))


def test_developer_has_the_switch_that_shows_speak_live():
    """Speak › Live is a streaming test for developers, so its
    switch has a section of its own, Developer, after Shortcuts. A switch like the others (role=switch, a visible
    label, a hint it is described by), off in the markup (the default; playback.js shows the stored preference),
    and the only control there."""
    s = section("developer")
    assert ui.visible_text(node("se-developer-title")) == "Developer"
    switch = node("se-show-live")
    assert switch.tag == "input" and switch.get("type") == "checkbox" and switch.get("role") == "switch"
    assert not switch.has("checked"), "off by default"
    assert ui.accessible_name(switch, tree()) == "Show Speak › Live (a streaming test for developers)"
    assert switch.get("aria-describedby") == "se-show-live-hint"
    hint = ui.visible_text(node("se-show-live-hint"))
    assert "streaming speech endpoint" in hint and hint.endswith("Off, Speak shows Generate only."), hint
    assert [i.id for i in ui.select(s, "input, select, textarea, button")] == ["se-show-live"], "in Developer, alone"


def test_server_and_about_are_read_only_and_the_docs_link_stays_on_this_server():
    """No external links or fetches; the API description opens on this server in a new tab.
    It is /openapi.json, never /docs or /redoc: FastAPI's Swagger and ReDoc pages load an unpinned script from
    a CDN on this origin, where the API key is stored, and are blank offline."""
    docs = node("se-server-docs")
    assert docs.get("href") == "/openapi.json" and docs.get("target") == "_blank" and docs.get("rel") == "noopener"
    assert ui.visible_text(docs).startswith("API schema (JSON)")
    for a in ui.select(tree(), "a"):
        assert (a.get("href") or "").split("#")[0].split("?")[0] not in ("/docs", "/redoc"), a.get("href")
    for a in ui.select(tree(), "a"):
        assert not re.match(r"^(?:[a-z]+:)?//", a.get("href") or ""), a.get("href")
    # The note names the AI server, where the AI assistant sends text (docs/AI-ASSISTANT.md).
    about = ui.visible_text(section("about"))
    assert "Self-hosted. Nothing leaves this server, except what you send to the AI server you set under AI assistant." in about
    assert ui.select(section("server"), "input, select, textarea") == []


# ───────────────────────────── the AI assistant section ─────────────────────────────

# Each preset field's control in settings.html (the phrases and the error sentences are rows the view builds).
AI_FIELDS = {
    "name": "se-ai-name", "model": "se-ai-model", "system_prompt": "se-ai-system", "user_template": "se-ai-template",
    "thinking": "se-ai-thinking", "reasoning_effort": "se-ai-effort", "show_thinking": "se-ai-show-thinking",
    "thinking_budget": "se-ai-thinking-budget",
    "max_tokens": "se-ai-max-tokens", "temperature": "se-ai-temperature", "top_p": "se-ai-top-p", "timeout_s": "se-ai-timeout",
    "history_turns": "se-ai-history", "filler_after_ms": "se-ai-filler-after", "filler_every_ms": "se-ai-filler-every",
    "filler_max": "se-ai-filler-max", "speak_errors": "se-ai-speak-errors", "chunk_chars": "se-ai-chunk",
    # The Ask AI group: how the preset shows in the Ask AI sheet.
    "label": "se-ai-label", "output": "se-ai-output",
    # The saved assistant the preset uses ("" is the one in use).
    "assistant": "se-ai-assistant",
}
AI_LISTS = {"fillers": "se-ai-fillers", "error_sentences": "se-ai-errors", "inputs": "se-ai-inputs"}


def js_string_list(name: str) -> list[str]:
    m = re.search(rf"export const {name} = Object\.freeze\(\[(.*?)\]\)", LOGIC.read_text(encoding="utf-8"), re.S)
    assert m, f"{name} is missing from assistant-logic.js"
    return re.findall(r"'([^']*)'", m.group(1))


def js_sentences() -> dict[str, str]:
    m = re.search(r"export const DEFAULT_ERROR_SENTENCES = Object\.freeze\(\{(.*?)\}\);", LOGIC.read_text(encoding="utf-8"), re.S)
    assert m, "DEFAULT_ERROR_SENTENCES is missing from assistant-logic.js"
    return {code: sentence for code, _quote, sentence in re.findall(r"(\w+): ([\"'])(.*?)\2,", m.group(1))}


def test_the_ai_assistant_section_has_every_preset_field_in_its_groups():
    """The status list, the off note, the preset picker with New,
    Duplicate and Delete/Restore, and every preset field in its groups (Prompts, Thinking, Limits,
    Sampling, While waiting, Errors) plus Test, after the Connection group. Off, it says to turn
    the assistant on under Connection, or, with a locked connection, with OS_ASSISTANT_URL and a restart. Each
    text, number and select field has its error slot in aria-describedby (a 422 lands there); Save submits the
    form, Test does not; the Test and Save status lines are role=status. Voice-only and book-only parts carry
    data-ai-use."""
    s = section("assistant")
    assert "card" in s.classes
    assert ui.visible_text(node("se-assistant-title")) == "AI assistant"
    assert node("se-ai-facts").tag == "dl"
    env = ui.visible_text(node("se-ai-off-env"))
    assert "OS_ASSISTANT_URL" in env and "restart" in env and node("se-ai-off-env").has("hidden")
    form_way = ui.visible_text(node("se-ai-off-form"))
    assert "Connection" in form_way and "restart" not in form_way, "no restart: the form turns it on"
    assert node("se-ai-off").has("hidden") and node("se-ai-editor").has("hidden"), "hidden until the status is known"
    assert [ui.visible_text(h) for h in ui.select(s, "h3")] == [
        "Assistants", "Connection", "Presets", "Ask AI", "Prompts", "Thinking", "Limits", "Sampling", "While waiting", "Errors", "Test",
    ], "the presets have their own heading after Connection, so the picker isn't under Connection; Ask AI follows the name"
    # The Ask AI group's three input kinds: one checkbox each, named, with the group's hint and error slot.
    boxes = [node(f"se-ai-inputs-{kind}") for kind in ("text", "transcript", "document")]
    assert [(b.get("type"), b.get("value")) for b in boxes] == [("checkbox", "text"), ("checkbox", "transcript"), ("checkbox", "document")]
    assert all(ui.accessible_name(b, tree()) for b in boxes)
    assert node("se-ai-inputs").tag == "fieldset" and "se-ai-inputs-error" in node("se-ai-inputs").get("aria-describedby").split()
    assert [o.get("value") for o in ui.select(node("se-ai-output"), "option")] == ["text", "speech", "document"]
    for group in ui.select(s, '[role="group"]'):
        assert ui.by_id(tree(), group.get("aria-labelledby") or "") is not None
    for field, id_ in AI_FIELDS.items():
        el = node(id_)
        assert ui.accessible_name(el, tree()), field
        if el.get("role") == "switch":
            continue
        assert f"{id_}-error" in (el.get("aria-describedby") or "").split(), f"{field}: the error slot is in aria-describedby"
        assert node(f"{id_}-error").has("hidden")
    for field, id_ in AI_LISTS.items():
        assert node(id_) is not None and node(f"{id_}-error").has("hidden"), field
    assert node("se-ai-preset").tag == "select" and ui.accessible_name(node("se-ai-preset"), tree()) == "Preset"
    assert [ui.visible_text(node(i)) for i in ("se-ai-new", "se-ai-duplicate", "se-ai-delete")] == ["New", "Duplicate", "Delete"]
    form = node("se-ai-form")
    assert form.tag == "form" and form.has("novalidate")
    save = node("se-ai-save")
    assert save.get("type") == "submit" and save.get("aria-disabled") == "true"
    assert any(a.id == "se-ai-form" for a in save.ancestors()), "Save submits the preset form"
    assert node("se-ai-test").get("type") == "button"
    assert node("se-ai-test-status").get("role") == "status" and node("se-ai-status").get("role") == "status"
    thinking = node("se-ai-test-thinking")
    assert thinking.tag == "details" and not thinking.has("open"), "the thinking starts collapsed"
    uses = {el.get("data-ai-use") for el in ui.select(s, "[data-ai-use]")}
    assert uses == {"voice", "book"}
    assert next(node("se-ai-chunk").ancestors()).get("data-ai-use") == "book"
    assert next(node("se-ai-history").ancestors()).get("data-ai-use") == "voice"


def test_the_ai_section_and_the_server_agree_on_fields_ranges_and_sentences():
    """The UI follows the server's preset contract (src/assistant/presets.py, errors.py, src/config.py) rather
    than a copy that can drift: the field list, the thinking modes and efforts, every number field's range,
    the text limits, the default error sentences, and a wording for every server-only setting that
    GET /api/assistant reports."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.assistant import presets
    from src.assistant.errors import ERROR_SENTENCES
    from src.config import (
        ASSISTANT_LIMITS,
        ASSISTANT_PROMPT_MAX_CHARS,
        ASSISTANT_REASONING_EFFORTS,
        ASSISTANT_THINKING_MODES,
        Settings,
    )
    from src.routes.assistant import create_router

    assert js_string_list("TUNABLE") == list(presets.TUNABLE_FIELDS)
    assert set(AI_FIELDS) | set(AI_LISTS) == {"name", "assistant", *presets.TUNABLE_FIELDS, *presets.ACTION_FIELDS}
    # The Ask AI fields: the server's names, kinds, outputs and label length.
    assert js_string_list("ACTION_FIELDS") == list(presets.ACTION_FIELDS)
    assert js_string_list("INPUT_KINDS") == list(presets.INPUT_KINDS)
    assert js_string_list("OUTPUTS") == list(presets.OUTPUT_KINDS)
    assert [o.get("value") for o in ui.select(node("se-ai-output"), "option")] == list(presets.OUTPUT_KINDS)
    assert node("se-ai-label").get("maxlength") == str(presets.LABEL_MAX_CHARS)
    assert js_string_list("THINKING_MODES") == list(ASSISTANT_THINKING_MODES)
    assert [o.get("value") for o in ui.select(node("se-ai-thinking"), "option")] == list(ASSISTANT_THINKING_MODES)
    assert [o.get("value") or "" for o in ui.select(node("se-ai-effort"), "option")] == list(ASSISTANT_REASONING_EFFORTS)
    for field, (low, high) in ASSISTANT_LIMITS.items():
        el = node(AI_FIELDS[field])
        assert el.get("type") == "number", field
        assert (float(el.get("min")), float(el.get("max"))) == (low, high), field
    for id_ in ("se-ai-system", "se-ai-template", "se-ai-test-text"):
        assert node(id_).get("maxlength") == str(ASSISTANT_PROMPT_MAX_CHARS), id_
    assert node("se-ai-name").get("maxlength") == str(presets.NAME_MAX_CHARS)
    assert node("se-ai-model").get("maxlength") == str(presets.MODEL_MAX_CHARS)
    logic = LOGIC.read_text(encoding="utf-8")
    assert re.search(rf"export const MAX_FILLERS = {presets.FILLERS_MAX_COUNT};", logic)
    assert re.search(rf"export const FILLER_MAX_CHARS = {presets.FILLER_MAX_CHARS};", logic)
    assert re.search(rf"export const SENTENCE_MAX_CHARS = {presets.ERROR_SENTENCE_MAX_CHARS};", logic)
    assert re.search(rf"export const PROMPT_MAX_CHARS = {ASSISTANT_PROMPT_MAX_CHARS};", logic)
    assert js_sentences() == dict(ERROR_SENTENCES)

    app = FastAPI()
    app.include_router(create_router(get_settings=lambda: Settings(_env_file=None, os_assistant_url="http://ai-server.lan:8080/v1")))
    status = TestClient(app).get("/api/assistant").json()
    worded = set(re.findall(r"^\s*(OS_ASSISTANT_\w+): \{ term:", logic, re.M))
    assert set(status["server_settings"]) - {"OS_ASSISTANT_URL"} == worded, "each server-only setting has its own words"


def test_the_thinking_budget_sits_with_the_thinking_and_the_hints_say_what_each_limit_does():
    """Presets › Thinking has "Thinking budget (tokens)", a whole
    number from 0 to 1,000,000 that the server sends as reasoning_budget_tokens. Its hint says 0 sends none, that
    other servers ignore or refuse the field, and that Test shows whether yours takes it. The Max tokens hint says
    the limit cuts the answer off without telling the model, that the thinking counts towards it, and that a short
    answer is asked for in the prompt, with the limit as a safety net."""
    budget = node("se-ai-thinking-budget")
    group = next(a for a in budget.ancestors() if a.get("role") == "group")
    assert group.get("aria-labelledby") == "se-ai-thinking-title"
    assert ui.accessible_name(budget, tree()) == "Thinking budget (tokens)"
    assert (budget.get("type"), budget.get("inputmode"), budget.get("min"), budget.get("max"), budget.get("step")) == (
        "number", "numeric", "0", "1000000", "1")
    assert "se-ai-thinking-budget-hint" in budget.get("aria-describedby").split()
    hint = ui.visible_text(node("se-ai-thinking-budget-hint"))
    for words in ("reasoning_budget_tokens", "0 sends none", "ignore or refuse", "Test shows whether yours takes it"):
        assert words in hint, words
    limit = ui.visible_text(node("se-ai-max-tokens-hint"))
    for words in ("cuts the answer off without telling the model", "thinking counts towards it", "prompt",
                  "safety net", "0 sends no limit"):
        assert words in limit, words


def test_the_ai_section_css_keeps_save_in_reach_and_reflows_at_every_width():
    """The Save bar sticks above the bottom chrome (a z-index token, the card's
    surface) and fields scrolled into view keep clear of it; a phrase row wraps its buttons under the field in
    a narrow card; the picker sits beside its buttons only in a wide card; empty status lines stay in the
    accessibility tree."""
    foot = decls('[data-view="settings"] .se-ai__foot')
    assert foot["position"] == "sticky" and "var(--chrome-bottom" in foot["bottom"]
    assert foot["z-index"] == "var(--z-sticky)" and foot["background"] == "var(--surface-1)"
    for control in (".input", ".textarea", ".select", ".switch__input"):
        clear = decls(f'[data-view="settings"] .se-ai__form {control}')
        assert "var(--chrome-bottom" in clear["scroll-margin-bottom"], control
    row = decls('[data-view="settings"] .se-ai__filler')
    assert row["flex-wrap"] == "wrap" and row["min-width"] == "0"
    assert decls('[data-view="settings"] .se-ai__filler-input')["min-width"] == "0"
    assert "minmax(0, 1fr)" in decls('[data-view="settings"] .se-ai__picker', "@container")["grid-template-columns"]
    # The status list's long terms: stacked in a narrow card, beside their values in a wide one.
    assert decls('[data-view="settings"] .se-ai__facts')["grid-template-columns"] == "minmax(0, 1fr)"
    assert decls('[data-view="settings"] .se-ai__facts', "@container")["grid-template-columns"] == "max-content minmax(0, 1fr)"
    empty = decls('[data-view="settings"] .se-ai__status:empty')
    assert empty.get("position") == "absolute" and "display" not in empty
    assert decls('[data-view="settings"] .se-ai__reply')["white-space"] == "pre-wrap"


# ───────────────────────────── the AI connection ─────────────────────────────

CONN_LOGIC = JS_DIR / "assistant-connection-logic.js"
# Each connection field's control (the id the view's CONN_INPUTS uses) and whether a 422 can land on it.
CONN_FIELDS = {
    "url": ("se-ai-conn-url", True), "api_key": ("se-ai-conn-key", True), "model": ("se-ai-conn-model", True),
    "verify_tls": ("se-ai-conn-tls", False), "template_kwargs": ("se-ai-conn-template", False),
    "max_tokens_field": ("se-ai-conn-max-field", True), "extra_body": ("se-ai-conn-extra", True),
    "stream": ("se-ai-conn-stream", False),
}


def conn_logic_list(name: str) -> list[str]:
    m = re.search(rf"export const {name} = Object\.freeze\(\[(.*?)\]\)", CONN_LOGIC.read_text(encoding="utf-8"), re.S)
    assert m, f"{name} is missing from assistant-connection-logic.js"
    return re.findall(r"'([^']*)'", m.group(1))


def conn_logic_number(name: str) -> int:
    m = re.search(rf"export const {name} = (\d+);", CONN_LOGIC.read_text(encoding="utf-8"))
    assert m, f"{name} is missing from assistant-connection-logic.js"
    return int(m.group(1))


def test_the_ai_connection_area_and_the_server_agree_on_fields_limits_and_the_view(tmp_path):
    """The Connection area follows the server's contract
    (src/assistant/connection.py, src/config.py, the real GET /api/assistant/connection) rather than a copy that
    can drift: the field list, the token-field names, every text limit, the view's keys and its sources, and the
    off message the JS fakes answer with."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.assistant import connection
    from src.assistant.presets import MODEL_MAX_CHARS
    from src.config import ASSISTANT_MAX_TOKENS_FIELDS, ASSISTANT_URL_MAX_CHARS, Settings
    from src.routes.assistant import create_router

    assert conn_logic_list("FIELDS") == list(connection.FIELDS)
    assert set(CONN_FIELDS) == set(connection.FIELDS)
    assert conn_logic_list("MAX_TOKENS_FIELDS") == list(ASSISTANT_MAX_TOKENS_FIELDS)
    assert [o.get("value") for o in ui.select(node("se-ai-conn-max-field"), "option")] == list(ASSISTANT_MAX_TOKENS_FIELDS)
    limits = {
        "URL_MAX_CHARS": (ASSISTANT_URL_MAX_CHARS, "se-ai-conn-url"),
        "MODEL_MAX_CHARS": (MODEL_MAX_CHARS, "se-ai-conn-model"),
        "API_KEY_MAX_CHARS": (connection.API_KEY_MAX_CHARS, "se-ai-conn-key"),
        "EXTRA_BODY_MAX_CHARS": (connection.EXTRA_BODY_MAX_CHARS, "se-ai-conn-extra"),
    }
    for name, (limit, id_) in limits.items():
        assert conn_logic_number(name) == limit, name
        assert node(id_).get("maxlength") == str(limit), id_
    view_js = (JS_DIR / "assistant-connection.js").read_text(encoding="utf-8")
    for field, (id_, _) in CONN_FIELDS.items():
        assert re.search(rf"^\s*{field}: '{id_}',$", view_js, re.M), f"CONN_INPUTS.{field} is {id_}"

    settings = Settings(_env_file=None, os_assistant_url="http://ai-server.lan:8080/v1", os_studio_db_path=str(tmp_path / "studio.db"))
    app = FastAPI()
    app.include_router(create_router(get_settings=lambda: settings))
    view = TestClient(app).get("/api/assistant/connection").json()
    # api_key_tls_checked (read-only): keyAgainReason() asks for the key again only when the key was
    # set with the certificate check on.
    assert set(view) == {"enabled", "locked", "url", "model", "api_key_set", "api_key_tls_checked", "verify_tls", "template_kwargs", "max_tokens_field", "extra_body", "stream", "sources"}
    assert view["api_key_tls_checked"] is None, "no key in use"
    assert "view.api_key_tls_checked" in CONN_LOGIC.read_text(encoding="utf-8"), "the form reads it"
    assert set(view["sources"]) == set(connection.FIELDS)
    assert set(view["sources"].values()) <= set(conn_logic_list_sources())
    assert "api_key" not in view, "the key is never in the view"
    off = (ui.REPO / "src" / "routes" / "assistant.py").read_text(encoding="utf-8")
    message = "The AI assistant is off: set the AI server's URL in Settings or OS_ASSISTANT_URL"
    assert message in off
    for test_file in ("assistant.test.js", "assistant-connection.test.js"):
        assert message in (ui.REPO / "tests" / "js" / "settings" / test_file).read_text(encoding="utf-8"), test_file


def conn_logic_list_sources() -> list[str]:
    m = re.search(r"export const SOURCE_WORDS = Object\.freeze\(\{(.*?)\}\);", CONN_LOGIC.read_text(encoding="utf-8"), re.S)
    assert m, "SOURCE_WORDS is missing from assistant-connection-logic.js"
    return re.findall(r"(\w+): '", m.group(1))


def test_the_url_hints_give_the_servers_one_example_and_the_js_fakes_copy_its_messages(tmp_path):
    """The Server
    URL hint and the .env way to turn the assistant on give the one example AI server URL that the server's own
    messages give (the real answers to a save of a URL that isn't http or https and to a check with no URL), and
    the JS fakes in tests/js/settings/assistant-connection.test.js answer with exactly those messages and name no
    other example, so neither copy drifts from the server again."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.config import Settings
    from src.routes.assistant import create_router

    settings = Settings(_env_file=None, os_assistant_url="", os_studio_db_path=str(tmp_path / "studio.db"))
    app = FastAPI()
    app.include_router(create_router(get_settings=lambda: settings))
    client = TestClient(app)
    answers = [
        client.put("/api/assistant/connection", json={"url": "ftp://other.lan/v1"}),
        client.post("/api/assistant/connection/check", json={}),
    ]
    assert [(r.status_code, r.json()["error"]["param"]) for r in answers] == [(422, "url"), (422, "url")]
    messages = [r.json()["error"]["message"] for r in answers]
    examples = {m.group(1) for m in (re.search(r"such as (\S+)$", message) for message in messages) if m}
    assert len(examples) == 1, f"the server gives one example: {messages}"
    example = examples.pop()
    for id_ in ("se-ai-conn-url-hint", "se-ai-off-env"):
        text = ui.visible_text(node(id_))
        codes = [ui.visible_text(c) for c in ui.select(node(id_), "code") if "://" in ui.visible_text(c)]
        assert codes == [example] and text.count("://") == 1, f"#{id_} gives the server's example {example}: {text}"
    fakes = (ui.REPO / "tests" / "js" / "settings" / "assistant-connection.test.js").read_text(encoding="utf-8")
    for message in messages:
        assert message in fakes, f"the JS fakes answer with the server's own text: {message}"
    named = {url.rstrip(".") for url in re.findall(r"such as (https?://[^\s'\"]+)", fakes)}
    assert named == {example}, f"the JS fakes name only the server's example: {sorted(named)}"


def test_the_ai_connection_area_has_a_labelled_field_for_each_setting_and_never_shows_the_key():
    """A Connection group at the top of the AI assistant section,
    before the presets, hidden until loaded. Each setting has a labelled control, a source badge in its
    description and a hidden Use .env button for its field; the text fields have a hidden error slot in their
    description. The key is a password field the page never fills (autocomplete new-password, no value), with
    its set/not-set badge and the "Enter the key again for a new server" hint in its description and Clear key.
    The help near the URL says the server calls this address and names OS_API_KEY. The locked note names .env,
    OS_ASSISTANT_CONNECTION_LOCKED and a restart. The area is no <form> (a password manager would offer to save
    the key); its one live region is its status line."""
    s = section("assistant")
    area = node("se-ai-conn")
    assert area.has("hidden") and area.get("role") == "group" and area.get("aria-labelledby") == "se-ai-conn-title"
    assert ui.visible_text(node("se-ai-conn-title")) == "Connection"
    order = [el.id for el in ui.select(s, "#se-ai-conn, #se-ai-editor")]
    assert order == ["se-ai-conn", "se-ai-editor"]
    assert not any(a.tag == "form" for a in area.ancestors()) and ui.select(area, "form") == []
    for field, (id_, has_error) in CONN_FIELDS.items():
        el = node(id_)
        assert ui.accessible_name(el, tree()), field
        described = (el.get("aria-describedby") or "").split()
        assert f"{id_}-source" in described, f"{field}: its source is read with it"
        assert node(f"{id_}-source").tag == "span" and "badge" in node(f"{id_}-source").classes
        reset = node(f"{id_}-reset")
        assert reset.has("hidden") and reset.get("data-action") == "ai-conn-reset" and reset.get("data-field") == field
        assert reset.get("type") == "button" and ui.accessible_name(reset, tree()).startswith("Use .env for ")
        if has_error:
            assert f"{id_}-error" in described and node(f"{id_}-error").has("hidden"), field
    key = node("se-ai-conn-key")
    assert key.get("type") == "password" and key.get("autocomplete") == "new-password" and key.get("value") is None
    assert {"se-ai-conn-key-state", "se-ai-conn-key-again"} <= set(key.get("aria-describedby").split())
    assert ui.visible_text(node("se-ai-conn-key-again")).startswith("Enter the key again for a new server")
    assert node("se-ai-conn-key-again").has("hidden")
    assert ui.visible_text(node("se-ai-conn-key-clear")) == "Clear key" and node("se-ai-conn-key-clear").has("hidden")
    url = node("se-ai-conn-url")
    assert url.get("type") == "url" and url.get("spellcheck") == "false" and url.get("autocapitalize") == "off"
    assert "se-ai-conn-help" in url.get("aria-describedby").split()
    help_text = ui.visible_text(node("se-ai-conn-help"))
    assert "calls this address" in help_text and "OS_API_KEY" in help_text
    locked = ui.visible_text(node("se-ai-conn-locked-note"))
    assert ".env" in locked and "OS_ASSISTANT_CONNECTION_LOCKED" in locked and "restart" in locked
    assert node("se-ai-conn-facts").tag == "dl"
    assert [ui.visible_text(node(i)) for i in ("se-ai-conn-check", "se-ai-conn-save", "se-ai-conn-revert")] == ["Check", "Save connection", "Revert to .env"]
    assert all(node(i).get("type") == "button" for i in ("se-ai-conn-check", "se-ai-conn-save", "se-ai-conn-revert"))
    assert node("se-ai-conn-save").get("aria-disabled") == "true"
    assert ui.accessible_name(node("se-ai-conn-models"), tree()) == "Models on the server"
    assert node("se-ai-conn-models-field").has("hidden")
    regions = [el.id for el in ui.select(area, '[role="status"], [role="alert"], [aria-live]')]
    assert regions == ["se-ai-conn-status"]


def test_the_saved_assistants_list_sits_above_the_connection_and_the_form_has_a_name_field():
    """The Assistants group comes first in the AI assistant section, before Connection, hidden until the
    saved assistants are known: its list (named by the heading), the note for none saved, Add assistant and its
    own status line. The Connection form starts with a Name field (hidden until it edits a saved or a new
    assistant, at most 60 characters, its hint and error slot in its description), a line that says what it
    edits, and Cancel beside Revert to .env for a new one. The preset form's Prompts group starts with an Assistant
    select whose first choice is "The one in use"."""
    s = section("assistant")
    order = [el.id for el in ui.select(s, "#se-ai-assts, #se-ai-conn, #se-ai-editor")]
    assert order == ["se-ai-assts", "se-ai-conn", "se-ai-editor"]
    group = node("se-ai-assts")
    assert group.has("hidden") and group.get("role") == "group" and group.get("aria-labelledby") == "se-ai-assts-title"
    assert ui.visible_text(node("se-ai-assts-title")) == "Assistants"
    lst = node("se-ai-assts-list")
    assert lst.tag == "ul" and lst.get("aria-labelledby") == "se-ai-assts-title"
    empty = node("se-ai-assts-empty")
    assert empty.has("hidden") and ".env" in ui.visible_text(empty)
    add = node("se-ai-asst-add")
    assert add.get("type") == "button" and ui.visible_text(add) == "Add assistant"
    assert node("se-ai-assts-status").get("role") == "status"
    regions = [el.id for el in ui.select(group, '[role="status"], [role="alert"], [aria-live]')]
    assert regions == ["se-ai-assts-status"]
    name = node("se-ai-conn-name")
    assert next(name.ancestors()).id == "se-ai-conn-name-field" and node("se-ai-conn-name-field").has("hidden")
    assert ui.accessible_name(name, tree()) == "Name" and name.get("maxlength") == "60"
    assert name.get("aria-describedby").split() == ["se-ai-conn-name-hint", "se-ai-conn-name-error"]
    assert node("se-ai-conn-name-error").has("hidden")
    fields = [el.id for el in ui.select(node("se-ai-conn-form"), "input, select, textarea")]
    assert fields[0] == "se-ai-conn-name", "the name comes first"
    target = node("se-ai-conn-target")
    assert target.has("hidden") and target.get("role") is None
    cancel = node("se-ai-conn-cancel")
    assert cancel.get("type") == "button" and cancel.has("hidden") and ui.visible_text(cancel) == "Cancel"
    actions = [el.id for el in ui.select(node("se-ai-conn-actions"), "button")]
    assert actions == ["se-ai-conn-check", "se-ai-conn-save", "se-ai-conn-revert", "se-ai-conn-cancel"]
    pick = node("se-ai-assistant")
    assert pick.tag == "select" and ui.accessible_name(pick, tree()) == "Assistant"
    assert [(o.get("value"), ui.visible_text(o)) for o in ui.select(pick, "option")] == [("", "The one in use")]
    assert node("se-ai-assistant-field").has("hidden")
    prompts = [el.id for el in ui.select(node("se-ai-form"), "input, select, textarea")]
    assert prompts.index("se-ai-assistant") == prompts.index("se-ai-model") - 1, "right before the model"


def test_the_assistants_list_and_the_server_agree_on_names_answers_and_the_preset_field(tmp_path):
    """The saved-assistant parts follow the server's contract (src/assistant/connection.py, presets.py, the real
    /api/assistant/assistants routes): the name's limit, the list answer's keys, a new assistant's answer (its
    id, name and in use, the first one saved in use), a name taken answering 409, and the preset field's
    name. Nothing is sent to an AI server."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.assistant import connection, presets
    from src.config import Settings
    from src.routes.assistant import create_router

    m = re.search(r"export const NAME_MAX_CHARS = (\d+);", CONN_LOGIC.read_text(encoding="utf-8"))
    assert m and int(m.group(1)) == connection.NAME_MAX_CHARS
    assert node("se-ai-conn-name").get("maxlength") == str(connection.NAME_MAX_CHARS)
    assert re.search(rf"export const ASSISTANT_FIELD = '{presets.ASSISTANT_FIELD}';", LOGIC.read_text(encoding="utf-8"))
    settings = Settings(_env_file=None, os_assistant_url="", os_studio_db_path=str(tmp_path / "studio.db"))
    app = FastAPI()
    app.include_router(create_router(get_settings=lambda: settings))
    client = TestClient(app)
    empty = client.get("/api/assistant/assistants").json()
    assert {k: empty[k] for k in ("active", "locked", "assistants")} == {"active": None, "locked": False, "assistants": []}
    # Add assistant starts from the .env connection, a connection view.
    assert {"url", "model", "api_key_set", "verify_tls", "sources"} <= set(empty["env"])
    made = client.post("/api/assistant/assistants", json={"name": "Home server", "url": "http://ai-server.lan:8080/v1"})
    assert made.status_code == 201
    view = made.json()
    assert {"id", "name", "active", "url", "model", "api_key_set", "sources", "enabled"} <= set(view)
    assert (view["name"], view["active"]) == ("Home server", True), "the first one saved is in use"
    listed = client.get("/api/assistant/assistants").json()
    assert listed["active"] == view["id"] and [a["id"] for a in listed["assistants"]] == [view["id"]]
    clash = client.post("/api/assistant/assistants", json={"name": "home SERVER"})
    assert clash.status_code == 409, "a name taken, ignoring case"
    status = client.get("/api/assistant").json()
    assert status["assistant"] == {"id": view["id"], "name": "Home server"}


def test_the_assistants_list_css_wraps_its_rows_at_narrow_widths():
    """A saved assistant's row wraps: its actions go under its name in a narrow card; names and hosts wrap
    anywhere rather than widen the card."""
    row = decls('[data-view="settings"] .se-ai__asst')
    assert row["display"] == "flex" and row["flex-wrap"] == "wrap" and row["min-width"] == "0"
    assert decls('[data-view="settings"] .se-ai__asst-actions')["flex-wrap"] == "wrap"
    assert decls('[data-view="settings"] .se-ai__asst-main')["min-width"] == "0"
    for sel in ('[data-view="settings"] .se-ai__asst-title', '[data-view="settings"] .se-ai__asst-meta'):
        assert decls(sel)["overflow-wrap"] == "anywhere", sel


def test_the_connection_has_the_stream_replies_switch_after_the_template_switch():
    """"Stream replies" (the field stream, OS_ASSISTANT_STREAM) is
    a switch in Connection right after "Send thinking as template variables" and before "Token limit sent as",
    like "Check TLS certificate": its source badge and hint in its description, a hidden Use .env, no error slot.
    The hint says what it does in Talk, when to turn it off, and that Ask AI and book jobs wait for the whole
    reply."""
    stream = node("se-ai-conn-stream")
    assert (stream.tag, stream.get("type"), stream.get("role")) == ("input", "checkbox", "switch")
    assert "switch__input" in stream.classes
    assert ui.accessible_name(stream, tree()) == "Stream replies"
    assert stream.get("aria-describedby").split() == ["se-ai-conn-stream-source", "se-ai-conn-stream-hint"]
    order = [el.id for el in ui.select(node("se-ai-conn-form"), "input, select, textarea")]
    assert order.index("se-ai-conn-stream") == order.index("se-ai-conn-template") + 1, "right after the template switch"
    assert order.index("se-ai-conn-max-field") == order.index("se-ai-conn-stream") + 1, "before the token-limit field"
    reset = node("se-ai-conn-stream-reset")
    assert reset.has("hidden") and reset.get("data-field") == "stream"
    assert ui.accessible_name(reset, tree()) == "Use .env for Stream replies"
    assert ui.by_id(tree(), "se-ai-conn-stream-error") is None, "never a 422 target"
    assert ui.visible_text(node("se-ai-conn-stream-hint")) == (
        "Talk speaks the answer a sentence at a time while the AI is still writing, so the voice starts sooner. "
        "Turn it off for a server that can't stream. Ask AI and book jobs always wait for the whole reply."
    )


def test_the_ai_connection_css_wraps_every_row_at_narrow_widths():
    """At every width (the web app runs on phones, tablets and desktops): a label row
    (label, source badge, Use .env) wraps; the key field and Clear key wrap, the field shrinking to fit; nothing
    has a fixed width wider than a 320 px card; a switch's hint lines up with its text; fields scrolled into view
    keep clear of the bottom chrome; the empty status line stays in the accessibility tree."""
    row = decls('[data-view="settings"] .se-ai__label-row')
    assert row["display"] == "flex" and row["flex-wrap"] == "wrap" and row["min-width"] == "0"
    key = decls('[data-view="settings"] .se-ai__key-row')
    assert key["flex-wrap"] == "wrap" and key["min-width"] == "0"
    field = decls('[data-view="settings"] .se-ai__key-row .input')
    assert field["min-width"] == "0" and field["flex"].startswith("1 1")
    assert decls('[data-view="settings"] .se-ai__conn-form')["min-width"] == "0"
    assert decls('[data-view="settings"] .se-ai__max-field')["max-width"] == "22rem", "a cap, never a fixed width"
    assert "42px" in decls('[data-view="settings"] .se-ai__switch-field .se-row__hint')["padding-inline-start"]
    for control in (".input", ".textarea", ".select", ".switch__input"):
        clear = decls(f'[data-view="settings"] .se-ai__conn {control}')
        assert "var(--chrome-bottom" in clear["scroll-margin-bottom"], control
    for sel, body, _chain in ui.css_rules(CSS):
        if "se-ai__conn" in sel or "se-ai__label-row" in sel or "se-ai__key" in sel:
            for prop, value in body:
                if prop in ("width", "min-width") and re.search(r"\d", value):
                    assert value in ("0", "auto") or value.endswith("%"), f"{sel} {{{prop}: {value}}}"
    empty = decls('[data-view="settings"] .se-ai__status:empty')
    assert empty.get("position") == "absolute"


# ───────────────────────────── the view CSS ─────────────────────────────


def test_one_column_with_a_sticky_table_of_contents_on_desktop():
    """The sections are one column of at most 720px; the table of contents is hidden below 960px and
    sticky beside the column from 960px."""
    assert decls('[data-view="settings"] .se-sections')["max-width"] == "720px"
    assert decls('[data-view="settings"] .se-toc')["display"] == "none"
    desk = decls('[data-view="settings"] .se-toc', "min-width: 960px")
    assert desk["position"] == "sticky" and desk["display"] == "block"
    layout = decls('[data-view="settings"] .se-layout', "min-width: 960px")
    assert "720px" in layout["grid-template-columns"]


def test_rows_reflow_by_the_card_width_and_the_status_line_stays_in_the_tree():
    """A settings row puts its control beside the label only in a wide card (a container query), so
    nothing scrolls sideways at 320 px; the empty key status line is visually hidden, never display: none, so
    its first message is announced."""
    wide = decls('[data-view="settings"] .se-row', "@container")
    assert wide["grid-template-columns"] == "minmax(0, 1fr) auto"
    empty = decls('[data-view="settings"] .se-key__status:empty')
    assert empty.get("position") == "absolute" and "display" not in empty
    checked = decls('[data-view="settings"] .se-speed__chip[aria-checked="true"]')
    assert checked["background"] == "var(--accent-soft)" and "var(--accent)" in checked["box-shadow"], "a checked chip is not colour alone"


def test_short_chips_and_segmented_options_are_44px_wide_on_touch():
    """Under pointer: coarse every control is at least 44 x 44. The components make chips
    and segmented options 44px tall; the short ones here ("1×", "2×", "3×", "5 s") were 42px wide, so the
    speed chips and the options of every Settings segmented control get a 44px minimum width too."""
    chip = decls('[data-view="settings"] .se-speed__chip', "pointer: coarse")
    assert chip["min-inline-size"] == "44px" and chip["justify-content"] == "center"
    assert decls('[data-view="settings"] .se-seg .seg__opt', "pointer: coarse")["min-inline-size"] == "44px"
    segs = ui.select(tree(), ".seg")
    assert segs and all("se-seg" in s.classes for s in segs), "every segmented control in Settings is a .se-seg"
