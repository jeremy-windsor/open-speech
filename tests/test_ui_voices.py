"""Voices (V5): the Node behaviour tests of the view, plus the partial's structure and the stylesheet's intent.
The global audits in test_ui_core.py also run over voices.html and voices.css."""

from __future__ import annotations

import re

import pytest

from tests import ui_coverage
from tests import ui_support as ui

PARTIAL = ui.PARTIALS_DIR / "voices.html"
CSS = ui.CSS_ROOT / "views" / "voices.css"
JS = ui.JS_ROOT / "views" / "voices"


@pytest.fixture(scope="module")
def tree():
    return ui.parse_html(PARTIAL)


def one(tree, css):
    found = ui.select(tree, css)
    assert len(found) == 1, f"{css}: {found}"
    return found[0]


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("voices"))
def test_js(path):
    ui.run_node_test(path)


def test_v5_inventory_ids_and_defects_have_tests():
    """Every inventory id and defect owned by V5 leads a test title in its assigned file."""
    assert ui_coverage.missing("V5") == []


# ───────────────────────────── partial structure ─────────────────────────────


def test_voices_partial_has_its_view_heading_and_header_action(tree):
    """[SE-02] One h1 (#vo-title, focusable, the view heading) and the one primary header action, Clone a voice."""
    headings = ui.select(tree, "h1")
    assert [h.id for h in headings] == ["vo-title"]
    assert headings[0].has("data-view-heading") and headings[0].get("tabindex") == "-1"
    clone = one(tree, "#vo-clone")
    assert "btn--primary" in clone.classes and clone.get("data-action") == "clone"
    assert "Clone a voice" in clone.text_content()


def test_sections_are_a_labelled_segmented_control(tree):
    """[SE-02] #vo-sub is a native radio group (one tab stop, arrow keys) with Voices | Recordings: named voices
    and presets are one list."""
    sub = one(tree, "#vo-sub")
    assert sub.get("role") == "radiogroup" and sub.get("aria-label")
    radios = ui.select(sub, "input")
    assert [(r.get("type"), r.get("name"), r.get("value")) for r in radios] == [
        ("radio", "vo-sub", "voices"), ("radio", "vo-sub", "recordings")]
    names = [ui.accessible_name(r, tree) for r in radios]
    assert names == ["Voices", "Recordings"]


def test_each_panel_has_its_list_and_the_wizard(tree):
    """[SE-02] The one list of named voices and presets, the recordings and the wizard exist (hidden
    until routed). The card menus are written inside the cards (components/menu.js moves an open one out of the
    size-container card and back), so no panel keeps a separate menu host; tests/js/voices/named.test.js checks
    the menus' placement and focus order."""
    for gone in ("vo-named", "vo-presets", "vo-named-list", "vo-presets-list"):
        assert ui.by_id(tree, gone) is None, f"#{gone}: one list, not two tabs"
    for panel, lst in (("vo-voices", "vo-voices-list"), ("vo-recs", "vo-rec-list")):
        node = one(tree, f"#{panel}")
        assert node.has("hidden")
        assert one(node, f"#{lst}")
    assert ui.select(tree, ".vo-menus") == [], "the menus live in their cards, not in a host after the list"
    assert one(tree, "#vo-wizard").has("hidden")
    select = one(tree, "#vo-clone-model")
    assert select.tag == "select" and ui.accessible_name(select, tree) == "Clone model"
    assert select.get("aria-describedby") == "vo-clone-model-hint"


def test_search_field_is_labelled_and_advertises_its_shortcut(tree):
    """[SE-02] The search field has a real label (the placeholder is not the label) and aria-keyshortcuts="/"."""
    search = one(tree, "#vo-search")
    assert search.get("type") == "search"
    assert ui.accessible_name(search, tree) == "Search voices"
    assert search.get("aria-keyshortcuts") == "/"


def test_wizard_steps_and_capture_controls(tree):
    """[VL-01] The stepper (ol.vo-steps, aria-current="step"), Record, the optional countdown, the level meter, the
    microphone select, the upload dropzone and the draft player of step 1."""
    steps = one(tree, "#vo-wizard ol.vo-steps")
    buttons = ui.select(steps, "button")
    assert [b.get("data-step") for b in buttons] == ["1", "2", "3"]
    assert [ui.accessible_name(b, tree) for b in buttons] == ["Capture", "Verify words", "Test & save"], "the step number is aria-hidden; the list gives the order"
    assert buttons[0].get("aria-current") == "step"
    record = one(tree, "#vo-record")
    assert record.get("data-action") == "record" and record.tag == "button"
    countdown = one(tree, "#vo-countdown")
    assert countdown.get("type") == "checkbox" and ui.accessible_name(countdown, tree) == "3-2-1 countdown"
    meter = one(tree, "#vo-meter")
    assert meter.get("role") == "meter" and meter.get("aria-label") == "Input level"
    assert ui.accessible_name(one(tree, "#vo-mic"), tree) == "Microphone"
    file_input = one(tree, "#vo-file")
    assert "visually-hidden" in file_input.classes and not file_input.has("hidden")
    assert one(tree, "#vo-dropzone").get("for") == "vo-file"
    assert one(tree, "#vo-draft-audio").has("controls")


def test_ticking_numbers_are_never_announced(tree):
    """[VL-01] The timer and the countdown digits are aria-hidden and sit outside #vo-status (live regions never
    tick); #vo-status is the wizard's role=status line."""
    status = one(tree, "#vo-status")
    assert status.get("role") == "status"
    for ident in ("vo-timer", "vo-countdown-num"):
        node = one(tree, f"#{ident}")
        assert node.get("aria-hidden") == "true"
        assert status not in list(node.ancestors())


def test_script_card_matches_the_script_the_wizard_prefills():
    """[VL-05] The paragraph read aloud is the transcript a recording keeps: the partial and clone.js agree."""
    source = (JS / "clone.js").read_text(encoding="utf-8")
    script = re.search(r"export const SCRIPT = '([^']+)';", source).group(1)
    tree = ui.parse_html(PARTIAL)
    assert " ".join(one(tree, "#vo-script").text_content().split()) == script


def test_verify_and_save_controls(tree):
    """[VL-08] Step 2 has the name with its "Saves as" preview, the transcript, Suggest, the confirm and skip
    checkboxes; step 3 the test model, Test before saving and Save to library."""
    name = one(tree, "#vo-name")
    assert ui.accessible_name(name, tree) == "Voice name"
    assert "vo-name-preview" in (name.get("aria-describedby") or "")
    assert ui.accessible_name(one(tree, "#vo-transcript"), tree) == "Transcript"
    assert one(tree, "#vo-suggest").get("data-action") == "suggest"
    assert ui.accessible_name(one(tree, "#vo-confirm"), tree) == "This matches the recording word for word"
    assert ui.accessible_name(one(tree, "#vo-skip"), tree) == "Save without a transcript"
    # They confirm a statement, so they are the shared checkbox (.check: a 44 px row, the box on the
    # label's first line), not switches; the skip hint is indented to the label's text.
    for ident in ("vo-confirm", "vo-skip"):
        box = one(tree, f"#{ident}")
        assert box.get("type") == "checkbox" and "check__input" in box.classes, ident
        assert "check" in box.parent.classes and box.parent.tag == "label", ident
    assert "check__hint" in one(tree, "#vo-skip-hint").classes
    assert ui.accessible_name(one(tree, "#vo-test-model"), tree) == "Test with"
    assert one(tree, "#vo-test").get("data-action") == "test"
    save = one(tree, "#vo-save")
    assert save.get("data-action") == "save" and "btn--primary" in save.classes


def test_design_wizard_controls(tree):
    """The design wizard goes back to Voices; step 1 has a host for the model's options, step 3 the voice name, a
    "Speak with" fieldset of checkboxes (filled per model) and Replace (Edit description only) before Save voice.
    The clone wizard's step 3 has "Also save as a named voice", checked, the shared checkbox with its hint."""
    back = one(tree, "#vo-design-wizard .vo-back")
    assert back.get("data-action") == "design-close" and " ".join(back.text_content().split()) == "Voices"
    assert one(tree, "#vd-options").has("hidden")
    assert ui.accessible_name(one(tree, "#vd-name"), tree) == "Voice name"
    fieldset = one(tree, "#vd-speak-with")
    assert fieldset.tag == "fieldset" and one(fieldset, "legend").text_content().strip() == "Speak with"
    assert one(fieldset, "#vd-speak-with-list") and one(fieldset, "#vd-speak-with-hint")
    replace, save = one(tree, "#vd-replace"), one(tree, "#vd-save")
    assert replace.has("hidden") and replace.get("data-action") == "design-replace"
    assert save.get("data-action") == "design-save" and "Save voice" in save.text_content()
    foot = [b.id for b in ui.select(one(tree, "#vd-step-3"), ".vo-step__foot button")]
    assert foot.index("vd-replace") < foot.index("vd-save")
    also = one(tree, "#vo-also-named")
    assert also.get("type") == "checkbox" and also.has("checked") and "check__input" in also.classes
    assert ui.accessible_name(also, tree) == "Also save as a named voice"
    assert also.get("aria-describedby") == "vo-also-named-hint"


def test_transcript_sheet_is_a_labelled_dialog(tree):
    """[VL-15] #vo-transcript-sheet is a sheet dialog with a labelled textarea, Cancel (data-close) and Save."""
    sheet = one(tree, "#vo-transcript-sheet")
    assert sheet.tag == "dialog" and "sheet" in sheet.classes
    assert sheet.get("aria-labelledby") == "vo-transcript-sheet-title"
    assert ui.accessible_name(one(sheet, "#vo-transcript-edit"), tree) == "Transcript"
    assert one(sheet, "#vo-transcript-save").get("type") == "submit"
    assert any(b.has("data-close") for b in ui.select(sheet, "button"))


def test_stop_recording_notes_are_static_hidden_text_outside_live_regions(tree):
    """[VL-12] While the microphone records (any view) nothing plays: Preview and Test clone, Test
    before saving and the clip player point at a visible "Stop recording to listen." note. Each note is plain
    text, hidden until then, and never a live region (the button that is pressed already says why)."""
    for ident in ("vo-rec-note", "vo-test-note", "vo-draft-note"):
        note = one(tree, f"#{ident}")
        assert note.has("hidden"), ident
        assert " ".join(note.text_content().split()) == "Stop recording to listen.", ident
        assert not note.get("role") and not note.get("aria-live"), ident


# The hand-offs to Speak (store intents) and the clone-test requests (REV-05) are asserted by the JS behaviour
# tests: store.takeIntent() in named, presets and recordings.test.js, and the request bodies in clone.test.js and
# recordings.test.js. store.setIntent() itself throws for an intent outside the frozen list.


# ───────────────────────────── stylesheet intent ─────────────────────────────


def _rules():
    return ui.css_rules(CSS)


def test_cards_reflow_to_one_column_and_nothing_scrolls_sideways():
    """[SE-02] The card grid never forces a column wider than the screen (min(100%, …)), long ids wrap, and the
    sections become a full-width control on phones (WCAG 1.4.10)."""
    grid = [body for sel, body, _ in _rules() if sel.strip() == '[data-view="voices"] .vo-grid']
    assert grid and "min(100%," in dict(grid[0])["grid-template-columns"]
    titles = dict(next(body for sel, body, _ in _rules() if sel.strip() == '[data-view="voices"] .vo-card__title'))
    assert titles.get("overflow-wrap") == "anywhere"
    phone = [(sel, dict(body)) for sel, body, chain in _rules() if any("max-width: 599px" in c for c in chain)]
    assert any(sel.strip() == '[data-view="voices"] .vo-sub' and d.get("width") == "100%" for sel, d in phone)


def test_preset_actions_wrap_and_more_ends_their_line():
    """[SE-05] A preset sits on its named voice's card or is a card of its own, at most half the
    content width, so its actions wrap under its name at every width (the old full-width rows shared columns
    through a subgrid). More goes to the end of its line, so the pressed "Set as default" toggle, which is wider,
    never pushes only its own preset's More; a card of its own keeps its actions on its last line, so cards side
    by side line them up."""
    rules = {sel.strip(): dict(body) for sel, body, chain in _rules() if tuple(chain) == ("@layer views",)}
    actions = rules['[data-view="voices"] .vo-preset__actions']
    assert actions["display"] == "flex" and actions["flex-wrap"] == "wrap"
    assert rules['[data-view="voices"] .vo-preset__actions > .vo-more'].get("margin-inline-start") == "auto"
    assert not any("subgrid" in d.get("grid-template-columns", "") for d in rules.values())
    assert rules['[data-view="voices"] .vo-card--preset'].get("flex-direction") == "column"
    assert rules['[data-view="voices"] .vo-card--preset .vo-preset__actions'].get("margin-top") == "auto"


def test_section_labels_are_whole_words_at_every_width(tree):
    """[SE-02] The sections read Voices and Recordings at every width: with no "Named voices" left,
    nothing is clipped to a short label on phones."""
    assert ui.select(tree, ".vo-wide-only") == []
    assert not [sel for sel, _, _ in _rules() if "vo-wide-only" in sel]


def test_touch_targets_are_never_shrunk_below_44px():
    """[SE-05] View rules sit in a later layer than components.css, so a size set here beats its 44 px touch rule:
    the small control height may only appear inside a (pointer: fine) block, and nothing here gives a
    control a px height or min-height under 44 px. The banners' Details summary and the .check rows get their
    44 px from components.css, so nothing here may set a size on them."""
    for sel, body, chain in _rules():
        decls = dict(body)
        for prop in ("min-height", "height"):
            value = decls.get(prop, "")
            if "--control-h-sm" in value:
                assert any("pointer: fine" in c for c in chain), f"{sel} {prop}: {value} outside (pointer: fine)"
            px = re.fullmatch(r"(\d+(?:\.\d+)?)px", value.strip())
            target = re.search(r"(\.btn|\.chip|button|\.vo-default|summary|select|\.check)(?![\w-])", sel)
            if px and float(px.group(1)) < 44 and target:   # a checkbox's own box is hit through its 44 px label
                pytest.fail(f"{sel} {prop}: {value} is under the 44 px touch target")
        if re.search(r"(summary|\.check)(?![\w-])", sel):
            assert not {"height", "min-height", "padding", "padding-block"} & decls.keys(), f"{sel}: the component sizes it"
    shared = ui.css_rules(ui.CSS_ROOT / "components.css")
    coarse = [(sel, dict(body)) for sel, body, chain in shared if any("pointer: coarse" in c for c in chain)]
    assert any(".banner__details > summary" in sel and d.get("min-height") == "44px" for sel, d in coarse), \
        "a banner's Details summary (every Voices error banner has one) is 44 px on touch"


def test_wizard_steps_are_labelled_sections_with_focusable_headings(tree):
    """[VL-08] Each step is a section labelled by its own heading (focus moves there on Next/Back), and step 3
    keeps the test model select and its button on one row with the hint below."""
    for n in (1, 2, 3):
        section = one(tree, f"#vo-step-{n}")
        heading = one(tree, f"#vo-step-{n}-title")
        assert section.get("aria-labelledby") == f"vo-step-{n}-title"
        assert heading.tag == "h3" and heading.get("tabindex") == "-1" and heading.text_content().strip()
    row = one(tree, "#vo-wizard .vo-test__row")
    assert [n.id for n in ui.select(row, "select, button")] == ["vo-test-model", "vo-test"]
    assert one(tree, "#vo-test-model").get("aria-describedby") == "vo-test-model-hint"


def test_recording_state_and_warnings_are_not_colour_alone():
    """[VL-01] The recording button fills with --rec-fill (never --rec as text); a missing transcript is warn text
    with an icon and words; an unverified transcript is a dashed warning border plus the hint text."""
    decls = {sel.strip(): dict(body) for sel, body, _ in _rules()}
    rec = decls['[data-view="voices"] .vo-record.is-recording']
    assert rec["background"] == "var(--rec-fill)" and rec["color"] == "var(--on-accent)"
    assert decls['[data-view="voices"] .vo-rec__transcript--missing']["color"] == "var(--warning)"
    unverified = decls['[data-view="voices"] .vo-transcript.is-unverified']
    assert unverified["border-style"] == "dashed"
    for sel, body in decls.items():
        assert "var(--rec)" not in body.get("color", ""), sel
