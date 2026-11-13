"""Studio (V6): the Node behaviour tests of the view, plus the partial's structure and the stylesheet's intent.
The global audits in test_ui_core.py also run over studio.html and studio.css."""

from __future__ import annotations

import pytest

from tests import ui_coverage
from tests import ui_support as ui

PARTIAL = ui.PARTIALS_DIR / "studio.html"
CSS = ui.CSS_ROOT / "views" / "studio.css"
JS = ui.JS_ROOT / "views" / "studio"


@pytest.fixture(scope="module")
def tree():
    return ui.parse_html(PARTIAL)


def one(tree, css):
    found = ui.select(tree, css)
    assert len(found) == 1, f"{css}: {found}"
    return found[0]


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("studio"))
def test_js(path):
    ui.run_node_test(path)


def test_v6_inventory_ids_and_defects_have_tests():
    """Every inventory id and defect owned by V6 (Studio and History) leads a test title in its assigned file."""
    assert ui_coverage.missing("V6") == []


# ───────────────────────────── partial structure ─────────────────────────────


def test_studio_partial_has_its_view_heading_and_section_switch(tree):
    """[ST-01] One focusable h1 (#st-title, the view heading) and #st-sub, a labelled native radio group
    Conversations | Composer (one tab stop, arrow keys)."""
    headings = ui.select(tree, "h1")
    assert [h.id for h in headings] == ["st-title"]
    assert headings[0].has("data-view-heading") and headings[0].get("tabindex") == "-1"
    sub = one(tree, "#st-sub")
    assert sub.get("role") == "radiogroup" and sub.get("aria-label")
    radios = ui.select(sub, "input")
    assert [(r.get("type"), r.get("name"), r.get("value")) for r in radios] == [
        ("radio", "st-sub", "conversations"), ("radio", "st-sub", "composer")]
    assert [ui.accessible_name(r, tree) for r in radios] == ["Conversations", "Composer"]


def test_conversations_list_and_editor(tree):
    """[ST-02][ST-09] #st-new, #st-list and the editor (#st-editor, labelled by its h2, the conversation's name)
    with the script ol#st-turns, a back link for phones, and the conversation's menu written right after its More
    actions button (the document order is the focus order; components/menu.js moves the open menu out of the
    card, a size container that would trap its fixed box)."""
    new = one(tree, "#st-new")
    assert new.get("data-action") == "new-conv" and "New conversation" in new.text_content()
    assert one(tree, "#st-list")
    editor = one(tree, "#st-editor")
    assert editor.tag == "section" and editor.get("aria-labelledby") == "st-editor-title"
    assert one(tree, "#st-editor-title").tag == "h2"
    turns = one(tree, "#st-turns")
    assert turns.tag == "ol" and turns.get("aria-label") == "Script"
    assert one(tree, "#st-back").get("href") == "#studio/conversations"
    more = one(tree, "#st-conv-more")
    assert more.get("aria-label") and more.has("hidden"), "Delete conversation is offered only when one is open"
    menu = one(tree, "#st-conv-menu")
    assert more.get("data-menu-btn") == "st-conv-menu" and menu.get("role") == "menu" and menu.has("hidden")
    siblings = [c for c in more.parent.children if c.is_element]
    assert siblings[siblings.index(more) + 1] is menu, "the menu follows its button"


def test_add_turn_form(tree):
    """[ST-03][BUG-16] Add turn: a Speaker field with a datalist of the script's speakers, a Preset select
    (None first), a multi-line Text field with its hint and error slot, and a submit button."""
    speaker = one(tree, "#st-speaker")
    assert ui.accessible_name(speaker, tree) == "Speaker"
    assert speaker.get("list") == "st-speakers" and one(tree, "datalist#st-speakers")
    preset = one(tree, "#st-preset")
    assert preset.tag == "select" and ui.accessible_name(preset, tree) == "Preset"
    assert [o.text_content() for o in ui.select(preset, "option")] == ["None"]
    text = one(tree, "#st-text")
    assert text.tag == "textarea", "a turn may hold several lines (BUG-16)"
    assert ui.accessible_name(text, tree) == "Text"
    assert "st-text-hint" in text.get("aria-describedby")
    # The default; the view sets the platform's own (Meta+Enter on Apple), tested in conversations.test.js.
    assert text.get("aria-keyshortcuts") == "Control+Enter"
    assert one(tree, "#st-text-error").has("hidden")
    add = one(tree, "#st-add")
    assert add.get("type") == "submit" and add in list(ui.select(one(tree, "#st-add-form"), "button"))


def test_render_controls_and_status(tree):
    """[ST-05][ST-06] Render: #st-format (WAV | MP3, a labelled radio group), the one primary button #st-render,
    the role=status line whose elapsed counter is aria-hidden, and #st-download (disabled until a render)."""
    fmt = one(tree, "#st-format")
    assert fmt.get("role") == "radiogroup" and fmt.get("aria-labelledby") == "st-format-label"
    assert [r.get("value") for r in ui.select(fmt, "input")] == ["wav", "mp3"]
    render = one(tree, "#st-render")
    assert "btn--primary" in render.classes and render.text_content().strip() == "Render"
    status = one(tree, "#st-render-status")
    assert status.get("role") == "status"
    elapsed = one(tree, "#st-render-elapsed")
    assert elapsed.get("aria-hidden") == "true" and status in list(elapsed.ancestors())
    download = one(tree, "#st-download")
    assert download.has("disabled") and "Download WAV" in download.text_content()
    lock = one(tree, "#st-lock-hint")
    assert lock.has("hidden") and "Rendering…" in lock.text_content()
    region = one(tree, "#st-render-section")
    primaries = [b for b in ui.select(one(tree, "#st-editor"), "button") if "btn--primary" in b.classes]
    assert [b.id for b in primaries] == ["st-render"], "one primary action per region"
    assert region.get("role") == "group"


def test_composer_controls(tree):
    """[CO-01][CO-03] The Composer: ol#st-tracks, Add track, #st-mix-format (WAV | MP3), the primary Render mix,
    the role=status line, the result's play and download, and #st-mixes (past mixes) in its own card."""
    composer = one(tree, "#st-composer")
    assert composer.has("hidden"), "Conversations shows first"
    assert one(tree, "#st-tracks").tag == "ol"
    assert one(tree, "#st-add-track").get("data-action") == "add-track"
    fmt = one(tree, "#st-mix-format")
    assert [r.get("value") for r in ui.select(fmt, "input")] == ["wav", "mp3"]
    render = one(tree, "#st-mix-render")
    assert "btn--primary" in render.classes and render.text_content().strip() == "Render mix"
    assert one(tree, "#st-mix-status").get("role") == "status"
    assert one(tree, "#st-mix-elapsed").get("aria-hidden") == "true"
    assert one(tree, "#st-mix-download").has("hidden")
    assert one(tree, "#st-mixes")


# ───────────────────────────── JS intent checks ─────────────────────────────


def test_studio_media_never_puts_the_key_in_a_url():
    """[ST-06][CO-04] Studio audio goes through api.mediaUrl / download.downloadFromApi (the key never goes in a URL): no module builds
    a URL with the key, and nothing reads the stored key directly."""
    for path in JS.glob("*.js"):
        text = path.read_text(encoding="utf-8")
        assert "api_key" not in text and "getKey(" not in text, path.name
    conversations = (JS / "conversations.js").read_text(encoding="utf-8")
    composer = (JS / "composer.js").read_text(encoding="utf-8")
    assert "downloadFromApi(" in conversations and "api.mediaUrl(" in conversations
    assert "downloadFromApi(" in composer and "api.mediaUrl(" in composer


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


def _component_banner_rules():
    """components.css: the .banner--stacked layout (actions on their own line under the message) and the coarse
    pointer's 44 px list, as (selector, declarations, chain)."""
    return [(sel.strip(), dict(body), chain) for sel, body, chain in ui.css_rules(ui.CSS_ROOT / "components.css")]


def assert_banners_come_from_the_component(view_css):
    """The view's inline banners are the component's .banner--stacked (views/studio/shared.js showBanner), whose
    actions sit under the message at every width, with a 44 px Details summary on touch;
    the view sheet re-lays out no banner of its own."""
    shared = ui.lex_js(ui.JS_ROOT / "views" / "studio" / "shared.js")
    assert any("banner--stacked" in value for _, value in shared.strings + shared.templates)
    rules = _component_banner_rules()
    actions = [d for sel, d, chain in rules if sel == ".banner--stacked .banner__actions" and not chain[1:]]
    assert actions and actions[0].get("flex-basis") == "100%", actions
    coarse = [d for sel, d, chain in rules if ".banner__details > summary" in [s.strip() for s in sel.split(",")]
              and any("pointer: coarse" in c for c in chain)]
    assert coarse and coarse[0].get("min-height") == "44px", coarse
    own = [sel for sel, _body, _chain in ui.css_rules(view_css) if "banner" in sel]
    assert own == [], f"view rules on banners: {own}"


def test_banners_are_the_components_stacked_banner():
    """[ST-05][CO-03] Studio's banners (a failed render, a mix, a delete) take their layout and touch size from
    components.css: actions under the message, aligned with it, and a 44 px Details summary on touch."""
    assert_banners_come_from_the_component(CSS)


def test_phones_show_the_list_or_the_script_and_desktop_both():
    """[ST-08] Below 960 px the list and the open script take turns (data-open), so a phone shows one at a time;
    from 960 px they sit side by side, and the back link goes."""
    phone = [(sel, dict(body)) for sel, body, chain in _rules() if any("max-width: 959px" in c for c in chain)]
    hidden = {s.strip() for sel, d in phone if d.get("display") == "none" for s in sel.split(",")}
    assert '[data-view="studio"] .st-conv[data-open="conv"] .st-list-pane' in hidden
    assert '[data-view="studio"] .st-conv[data-open="none"] .st-editor' in hidden
    grid = _decls('[data-view="studio"] .st-conv__grid', "min-width: 960px")
    assert "280px" in grid.get("grid-template-columns", "")
    assert _decls('[data-view="studio"] .st-back', "min-width: 960px").get("display") == "none"


def test_an_empty_list_offers_new_conversation_once_and_never_competes_with_the_draft():
    """[BUG-16] Below 960 px the empty list shows alone: its empty state carries New conversation (primary) and the
    pane's own button hides (data-empty). From 960 px it sits beside the draft editor, which is already a new
    conversation: the empty state's primary hides, its line on how the draft starts one shows, and the pane keeps
    its secondary New conversation (the way to name one first; the server has no rename)."""
    narrow = [(sel, dict(body)) for sel, body, chain in _rules() if any("max-width: 959px" in c for c in chain)]
    hidden = {s.strip() for sel, d in narrow if d.get("display") == "none" for s in sel.split(",")}
    assert '[data-view="studio"] .st-list-pane[data-empty="true"] .st-new' in hidden
    assert _decls('[data-view="studio"] .st-list-empty__beside').get("display") == "none"
    assert _decls('[data-view="studio"] .st-list-empty__new', "min-width: 960px").get("display") == "none"
    beside = _decls('[data-view="studio"] .st-conv[data-open="none"] .st-list-empty__beside', "min-width: 960px")
    assert beside.get("display") == "block"
    wide = [(sel, dict(body)) for sel, body, chain in _rules() if any("min-width: 960px" in c for c in chain)]
    assert not [sel for sel, d in wide if ".st-new" in sel.replace(".st-new-", "") and d.get("display") == "none"],         "the pane's New conversation stays from 960 px"


def test_turn_text_keeps_its_line_breaks_and_long_words_wrap():
    """[BUG-16] Turn text is pre-wrap (a script keeps its line breaks) and long names and words wrap at 320 px."""
    text = _decls('[data-view="studio"] .st-turn__text')
    assert text.get("white-space") == "pre-wrap"
    assert text.get("overflow-wrap") == "anywhere"
    assert _decls('[data-view="studio"] .st-editor__title').get("overflow-wrap") == "anywhere"


def test_missing_preset_colour_only_adds_to_the_words():
    """[ST-10] The warning colour of a turn whose preset was deleted only adds to the words and icon ("Preset
    missing", asserted by the Node test of ST-10)."""
    decl = _decls('[data-view="studio"] .st-turn__preset.is-missing')
    assert decl.get("color") == "var(--warning)"


def test_list_pane_scrolls_on_its_own_beside_a_long_script():
    """[ST-01] From 960 px the sticky list pane is capped to the screen between the chrome bars and scrolls on its
    own, so the last of 100 conversations is reachable while a long script is read (and never under the docked
    mini-player)."""
    pane = _decls('[data-view="studio"] .st-list-pane', "min-width: 960px")
    assert pane.get("position") == "sticky"
    assert "--chrome-bottom" in pane.get("max-height", "") and "--chrome-top" in pane.get("max-height", "")
    assert pane.get("overflow-y") == "auto" and pane.get("overscroll-behavior") == "contain"


def test_mute_and_solo_keep_their_width_when_pressed():
    """[CO-02] Mute and Solo reserve room for the check glyph, so pressing one never shortens the volume slider
    beside it (the slider is the row's flexible item)."""
    chip = _decls('[data-view="studio"] .st-track__toggles .chip')
    assert "min-width" in chip and "ch" in chip["min-width"]
    assert _decls('[data-view="studio"] .st-track__volume').get("flex", "").startswith("1 1")


def test_renders_and_mixes_have_an_inline_transport(tree):
    """[ST-05][CO-03] A render and a mix play on their own route, where the mini-player hides: each
    result has a hidden transport with a named native seek range and an aria-hidden time."""
    for box, seek, time in (("#st-transport", "#st-seek", "#st-seek-time"), ("#st-mix-transport", "#st-mix-seek", "#st-mix-seek-time")):
        holder = one(tree, box)
        assert holder.has("hidden")
        rng = one(tree, seek)
        assert rng.tag == "input" and rng.get("type") == "range" and "seek" in rng.classes
        assert rng.get("aria-label") and holder in list(rng.ancestors())
        assert one(tree, time).get("aria-hidden") == "true"


def test_preset_and_source_lists_have_a_place_to_say_they_did_not_load(tree):
    """[ST-10][CO-01] #st-preset-error (under the Preset select) and #st-sources-error (above the tracks) start
    hidden; the view fills them with the cause and Retry when a list fails."""
    assert one(tree, "#st-preset-error").has("hidden")
    assert one(tree, "#st-sources-error").has("hidden")
