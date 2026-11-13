"""History (V6): the Node behaviour tests of the view, plus the partial's structure and the stylesheet's intent.
The global audits in test_ui_core.py also run over history.html and history.css."""

from __future__ import annotations

import re

import pytest

from tests import ui_support as ui

PARTIAL = ui.PARTIALS_DIR / "history.html"
CSS = ui.CSS_ROOT / "views" / "history.css"
JS = ui.JS_ROOT / "views" / "history"


@pytest.fixture(scope="module")
def tree():
    return ui.parse_html(PARTIAL)


def one(tree, css):
    found = ui.select(tree, css)
    assert len(found) == 1, f"{css}: {found}"
    return found[0]


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("history"))
def test_js(path):
    ui.run_node_test(path)


# ───────────────────────────── partial structure ─────────────────────────────


def test_history_partial_has_its_view_heading(tree):
    """[HI-01] One focusable h1 (#hi-title, the view heading); day headings are h2 (built at runtime)."""
    headings = ui.select(tree, "h1")
    assert [h.id for h in headings] == ["hi-title"]
    assert headings[0].has("data-view-heading") and headings[0].get("tabindex") == "-1"


def test_search_field_is_labelled_and_advertises_its_shortcut(tree):
    """[HI-02] #hi-search is type=search with a real label (the placeholder is not the label), a description of
    what it searches, and aria-keyshortcuts="/"."""
    search = one(tree, "#hi-search")
    assert search.get("type") == "search"
    assert ui.accessible_name(search, tree) == "Search history"
    assert search.get("aria-keyshortcuts") == "/"
    assert one(tree, "#" + search.get("aria-describedby")).text_content().startswith("Searches the whole text")


def test_filter_chips_are_a_labelled_group_of_toggles(tree):
    """[HI-01] #hi-filter is a labelled group of three toggle chips (All, Speech, Transcripts) mapping to
    type '', tts and stt; All starts pressed."""
    group = one(tree, "#hi-filter")
    assert group.get("role") == "group" and group.get("aria-label")
    chips = ui.select(group, "button")
    assert [(c.get("data-type"), c.text_content().strip(), c.get("aria-pressed")) for c in chips] == [
        ("", "All", "true"), ("tts", "Speech", "false"), ("stt", "Transcripts", "false")]
    assert all(c.get("type") == "button" and c.get("data-action") == "filter" for c in chips)


def test_toolbar_puts_the_filter_first_and_the_search_after_it(tree):
    """[HI-02] One toolbar order for the list views: what to show first (History's chips, as the Library's chips
    and Voices' sections), the search after it at the end of the row (DOM order = visual order), at most 20rem
    wide from 600 px and the whole row on phones."""
    bar = one(tree, ".hi-bar")
    kids = [el for el in bar.children if el.is_element]
    assert "hi-filter" in kids[0].classes and "hi-search" in kids[1].classes, [k.classes for k in kids]
    assert _decls('[data-view="history"] .hi-bar').get("justify-content") == "space-between"
    assert _decls('[data-view="history"] .hi-search').get("max-width") == "20rem"
    assert _decls('[data-view="history"] .hi-search', "max-width: 599px").get("flex-basis") == "100%"
    for sel, _body, _chain in ui.css_rules(CSS):
        assert "order" not in dict(_body), f"{sel}: the visual order is the DOM order"


def test_count_is_a_status_region_and_load_more_sits_with_it(tree):
    """[HI-04] #hi-count is the role=status line (a required live region) and is never hidden itself (a live region must exist
    before it speaks); Load more starts hidden next to it."""
    count = one(tree, "#hi-count")
    assert count.get("role") == "status"
    assert not count.has("hidden") and not any(a.has("hidden") for a in count.ancestors())
    more = one(tree, "#hi-more")
    assert more.has("hidden") and more.get("data-action") == "more"
    assert one(tree, "#hi-foot") in list(more.ancestors())


def test_clear_all_sits_apart_at_the_end_without_a_danger_zone(tree):
    """[HI-08] Clear all history… is a danger-ghost button after the list and Load more, with no "Danger zone"
    heading; it is the only clear action in the whole UI."""
    clear = one(tree, "#hi-clear")
    assert "btn--danger-ghost" in clear.classes and clear.text_content().strip() == "Clear all history…"
    assert "danger zone" not in PARTIAL.read_text(encoding="utf-8").lower()
    ids = [el.id for el in tree.elements() if el.id]
    assert ids.index("hi-clear-wrap") > ids.index("hi-foot") > ids.index("hi-list")
    others = [p.name for p in ui.PARTIALS_DIR.glob("*.html") if p.name != "history.html"
              and re.search(r"clear\s+(all\s+)?history", p.read_text(encoding="utf-8"), re.I)]
    assert others == [], f"only History clears history: {others}"


def test_banners_are_the_components_stacked_banner():
    """[HI-10] History's banners (a failed load, Load more, a delete) take their layout and touch size from
    components.css: actions under the message, aligned with it, and a 44 px Details summary on touch."""
    from tests.test_ui_studio import assert_banners_come_from_the_component

    assert_banners_come_from_the_component(CSS)


# ───────────────────────────── JS intent checks ─────────────────────────────


def test_history_sends_only_the_frozen_intent():
    """[HI-05] History hands entries to Speak only through setIntent('speak', {type: 'apply-history', entry}) and
    navigates; it never clicks another view's buttons."""
    sent = set()
    for path in JS.glob("*.js"):
        text = path.read_text(encoding="utf-8")
        sent |= set(re.findall(r"setIntent\(\s*'speak',\s*\{\s*type:\s*'([\w-]+)'", text))
        assert "[data-tab" not in text, path
    assert sent == {"apply-history"}


def test_no_history_disabled_flag_is_guessed():
    """[REV-06] The server has no "history disabled" flag: nothing claims history is off; the empty state names
    OS_HISTORY_ENABLED as the place to look."""
    src = ui.lex_js(JS / "list.js")
    copy = [value for _, value in src.strings + src.templates]
    assert any("OS_HISTORY_ENABLED" in value for value in copy)
    assert not any(re.search(r"history (?:is |was )?(?:disabled|turned off)\b(?! on the server \(OS_)", value, re.I) for value in copy)


# ───────────────────────────── stylesheet intent ─────────────────────────────


def _rules():
    return ui.css_rules(CSS)


def _decls(selector, chain_has=None):
    out = {}
    for sel, body, chain in _rules():
        if sel.strip() != selector:
            continue
        if chain_has is None and any(c.startswith(("@media", "@container")) for c in chain):
            continue
        if chain_has is not None and not any(chain_has in c for c in chain):
            continue
        out.update(dict(body))
    return out


def test_rows_reflow_on_phones_and_long_text_wraps():
    """[HI-05] On phones a row's actions move under its text (the text keeps the width); text and meta wrap long
    words, so nothing scrolls sideways at 320 px."""
    phone = _decls('[data-view="history"] .hi-entry', "max-width: 599px")
    assert '". actions"' in phone.get("grid-template-areas", "")
    assert _decls('[data-view="history"] .hi-entry__text').get("overflow-wrap") == "anywhere"
    assert _decls('[data-view="history"] .hi-entry__meta').get("overflow-wrap") == "anywhere"
    assert _decls('[data-view="history"] .hi-search', "max-width: 599px").get("max-width") == "none"


def test_day_groups_use_text_2_on_the_page_and_rows_sit_on_a_surface():
    """[HI-05] Day headings are --text-2 micro labels; each day's rows sit on --surface-1 with hairlines between
    them (--border), the same surface as an empty state."""
    day = _decls('[data-view="history"] .hi-day')
    assert day.get("color") == "var(--text-2)" and day.get("text-transform") == "uppercase"
    entries = _decls('[data-view="history"] .hi-entries')
    assert entries.get("background") == "var(--surface-1)"
    assert _decls('[data-view="history"] .hi-entry + .hi-entry').get("border-top") == "1px solid var(--border)"
    assert _decls('[data-view="history"] .hi-empty').get("background") == "var(--surface-1)"
