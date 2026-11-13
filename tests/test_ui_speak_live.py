"""Speak: Live mode (V8): the Node behaviour tests for the session engine, the text flow and the
controls, plus the slot partial's structure and the view stylesheet's intent.

The behaviour lives in tests/js/speak-live/*.test.js (titles carry the inventory ids LR-01…LR-20, LR-22,
LR-23 and BUG-20); this file runs them and checks what a parsed partial or stylesheet can show.
"""

from __future__ import annotations

import json
import re

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

PARTIAL = ui.PARTIALS_DIR / "speak-live.html"
CSS = ui.CSS_ROOT / "views" / "speak-live.css"


def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("speak-live"))
def test_js_speak_live(path):
    ui.run_node_test(path)


def test_v8_inventory_ids_and_defects_have_tests():
    """Every inventory id and defect owned by V8 leads a running test title in its assigned file."""
    assert missing("V8") == []


# ───────────────────────────── the slot partial ─────────────────────────────


def test_slot_partial_has_no_h1_and_its_headings_start_at_h2():
    """A slot lives inside Speak's view, which owns the one h1 (the view contract)."""
    t = tree()
    assert ui.select(t, "h1") == []
    headings = [el for el in t.elements() if re.fullmatch(r"h[1-6]", el.tag)]
    assert headings and headings[0].tag == "h2"
    assert ui.stub_markers(t) == [], "the placeholder marker is gone"


def test_status_is_the_only_live_region_and_the_now_line_is_never_announced():
    """#spl-status is role=status; #spl-now is aria-live=off; the visible pill label is aria-hidden and a
    visually hidden twin carries what the region says (it changes per phase, not per response)."""
    t = tree()
    status = ui.by_id(t, "spl-status")
    assert status.get("role") == "status"
    label = ui.by_id(t, "spl-status-label")
    twin = ui.by_id(t, "spl-status-text")
    assert label.get("aria-hidden") == "true"
    assert "visually-hidden" in twin.classes
    assert status in list(twin.ancestors()) and status in list(label.ancestors())
    assert ui.by_id(t, "spl-now").get("aria-live") == "off"
    live = [el for el in t.elements() if el.get("role") in ("status", "log") or el.get("aria-live") in ("polite", "assertive")]
    assert [el.id for el in live] == ["spl-status"], "nothing else talks while Live reads"
    assert ui.by_id(t, "spl-latency").get("role") is None
    error = ui.by_id(t, "spl-error")
    assert error.get("role") == "alert" and error.has("hidden"), "the failure banner is role=alert and only shows on failure"


def test_mode_is_a_native_radio_group_with_the_three_latency_modes():
    """LR-01: Book | Responsive | Each word map to natural | responsive | instant_word; Book is the default."""
    t = tree()
    group = ui.by_id(t, "spl-mode")
    assert group.get("role") == "radiogroup"
    assert ui.by_id(t, group.get("aria-labelledby")).text_content().strip() == "Mode"
    radios = ui.select(group, "input")
    assert [(r.get("type"), r.get("name"), r.get("value")) for r in radios] == [
        ("radio", "spl-mode", "natural"), ("radio", "spl-mode", "responsive"), ("radio", "spl-mode", "instant_word"),
    ]
    assert [ui.accessible_name(r, t) for r in radios] == ["Book", "Responsive", "Each word"]
    assert radios[0].has("checked")


def test_controls_are_one_toggle_plus_stop_and_a_menu():
    """The research brief's Live pattern: Start/Pause is one toggle, Stop is separate, the rest sits in a menu."""
    t = tree()
    start = ui.by_id(t, "spl-start")
    stop = ui.by_id(t, "spl-stop")
    more = ui.by_id(t, "spl-more")
    assert "btn--primary" in start.classes and ui.accessible_name(start, t) == "Start"
    assert stop.has("disabled") and ui.accessible_name(stop, t) == "Stop"
    assert more.get("aria-label") == "More Live actions" and more.get("aria-controls") == "spl-more-menu"
    menu = ui.by_id(t, "spl-more-menu")
    assert menu.get("role") == "menu" and menu.has("hidden")
    assert [ui.accessible_name(i, t) for i in ui.select(menu, "[role=menuitem]")] == ["Read all from start", "Clear text"]
    kbd = ui.by_id(t, "spl-kbd")
    assert kbd.get("aria-hidden") == "true", "the shortcut hint is not part of the button's name"
    primaries = [el for el in t.elements() if "btn--primary" in el.classes]
    assert primaries == [start], "one primary action in the region"


def test_ids_carry_the_spl_prefix_and_buttons_have_a_type_and_a_name():
    t = tree()
    ids = [el.id for el in t.elements() if el.id]
    assert ids and all(i == "spl" or i.startswith("spl-") for i in ids)
    for button in ui.select(t, "button"):
        assert button.get("type") == "button", button.id
        assert ui.accessible_name(button, t), button.id
    for svg in ui.select(t, "svg"):
        assert svg.get("aria-hidden") == "true"


# ───────────────────────────── the stylesheet ─────────────────────────────


def _rules():
    return ui.css_rules(CSS)


FORCED = ("@layer utilities", "@media (forced-colors: active)")


def test_live_css_is_scoped_to_speak_inside_layer_views():
    """Every rule is scoped to Speak inside @layer views; the one exception is the forced-colours block, with system colours only."""
    for selector, body, chain in _rules():
        if selector.startswith("@"):
            continue
        assert chain and (chain[0] == "@layer views" or tuple(chain) == FORCED), (selector, chain)
        for part in selector.split(","):
            assert part.strip().startswith('[data-view="speak"]'), part
        if tuple(chain) == FORCED:
            assert not any("var(--" in value for _, value in body), selector


def test_the_now_card_keeps_its_edge_and_the_spoken_state_in_forced_colours():
    """Forced colours drop the Now card's surface and its tint while it shows the text being
    spoken. The card keeps a CanvasText edge, and the spoken state turns it into a Highlight ring; the "Now"
    label says the same in words (the Node test [LR-12])."""
    forced = {s: dict(b) for s, b, c in _rules() if tuple(c) == FORCED}
    assert forced['[data-view="speak"] .spl-now'].get("border") == "1px solid CanvasText"
    spoken = forced['[data-view="speak"] .spl-now[data-kind="spoken"]']
    assert spoken.get("border-color") == "Highlight" and "Highlight" in spoken.get("outline", "")


def test_the_shortcut_hint_shows_for_a_fine_pointer_only():
    """A touch screen has no Ctrl+Enter: the kbd hint hides under (pointer: coarse)."""
    hidden = [body for selector, body, chain in _rules()
              if selector == '[data-view="speak"] .spl-kbd' and any("pointer: coarse" in c for c in chain)]
    assert hidden and ("display", "none") in hidden[0]


def test_start_has_an_emphasis_state_for_text_typed_while_stopped():
    """Text typed while Live is stopped waits for Start, so Start gets an emphasis
    state (live-ui.js sets data-emphasis; the Node test [LR-20] "typing while Live is stopped…"). It is drawn from
    tokens, it changes no size (a ring, not a border or padding), and in forced colours it keeps a system-colour
    ring, since the tokens' colours drop out there."""
    selector = '[data-view="speak"] .spl-toggle[data-emphasis="true"]'
    plain = [dict(body) for s, body, chain in _rules() if s == selector and tuple(chain) == ("@layer views",)]
    assert plain, "an emphasis rule inside @layer views"
    ring = plain[0].get("box-shadow", "")
    assert "var(--" in ring and "inset" not in ring, ring
    assert not {"border", "border-width", "padding", "width", "height", "min-width"} & set(plain[0]), plain[0]
    forced = [dict(body) for s, body, chain in _rules() if s == selector and tuple(chain) == FORCED]
    assert forced and "Highlight" in forced[0].get("outline", ""), forced


def test_a_long_status_wraps_instead_of_being_cut_and_the_now_card_wraps_long_words():
    """WCAG 1.4.10 at 320 px: "Stopped: another player started" and a long unbroken word stay readable."""
    decls = {}
    for selector, body, chain in _rules():
        if not any(c.startswith("@media") for c in chain):
            decls.setdefault(selector, {}).update(dict(body))
    assert decls['[data-view="speak"] .spl-pill'].get("white-space") == "normal"
    assert decls['[data-view="speak"] .spl-pill .pill__label'].get("overflow") == "visible"
    assert decls['[data-view="speak"] .spl-now'].get("overflow-wrap") == "anywhere"
    assert decls['[data-view="speak"] .spl-hint'].get("overflow-wrap") == "anywhere"


def test_the_now_card_tint_marks_speech_in_words_too():
    """Colour carries state only, never alone: the Now card has its own tint while it shows the text being
    spoken (a token, unlike the idle card's), and the partial carries the "Now" label that says so in words
    (the Node test [LR-12] shows it exactly while the card shows spoken text)."""
    t = tree()
    label = ui.by_id(t, "spl-now-label")
    assert label.text_content().strip() == "Now" and label.has("hidden")
    assert label in list(ui.by_id(t, "spl-now").elements())
    decls = {selector: dict(body) for selector, body, chain in _rules() if not any(c.startswith("@media") for c in chain)}
    idle = decls['[data-view="speak"] .spl-now'].get("background")
    spoken = decls['[data-view="speak"] .spl-now[data-kind="spoken"]'].get("background")
    assert spoken and spoken.startswith("var(--") and spoken != idle


def test_the_slot_reads_in_one_order_on_every_screen():
    """The one line saying what Live is comes first, then the actions (where
    Generate's button sits in Generate mode, so on a phone Start is above the bottom bar), then what they do, then
    the mode; no stylesheet rule reorders them, so the focus order and a screen reader's reading order are the
    visual order (WCAG 1.3.2, 2.4.3)."""
    section = ui.by_id(tree(), "spl")
    blocks = [child for child in section.children if child.is_element and "visually-hidden" not in child.classes]
    assert [next(c for c in b.classes if c.startswith("spl-")) for b in blocks] == [
        "spl-about", "spl-actions", "spl-hint", "spl-state", "spl-error", "spl-mode",
    ]
    reordered = [selector for selector, body, _ in _rules() if any(name in ("order", "grid-row", "grid-area") for name, _ in body)
                 and any(f".{name}" in selector for name in ("spl-about", "spl-actions", "spl-hint", "spl-state", "spl-mode", "spl-now", "spl-status-row"))]
    assert reordered == []


def test_live_opens_with_one_line_saying_what_it_is():
    """Live is a streaming test for developers, shown only once
    Settings turns it on, and it opens with one line saying what it is. It is plain text, not a live
    region and not a heading; its paragraph sits flush in the grid (the gap spaces it) and wraps at 320 px."""
    about = ui.by_id(tree(), "spl-about")
    assert about is not None and about.tag == "p"
    assert ui.visible_text(about) == (
        "Reads text as it arrives, the way an AI app streams its reply to the speech endpoint. Type or paste to try it."
    )
    assert about.get("role") is None and about.get("aria-live") is None
    decls = {selector: dict(body) for selector, body, chain in _rules() if not any(c.startswith("@media") for c in chain)}
    rule = decls['[data-view="speak"] .spl-about']
    assert rule.get("margin") == "0" and rule.get("color", "").startswith("var(--"), rule
    assert rule.get("overflow-wrap") == "anywhere"


def test_a_locked_mode_keeps_the_chosen_one_visible():
    """While Live runs the mode radios are disabled (the Node test [LR-20]). The segmented control keeps the
    chosen option of a disabled group at full opacity (its fill and edge) and dims only its label, so which mode
    runs stays visible in dark mode too; the Live sheet leaves that to the component and dims nothing
    of its own."""
    components = {s: dict(b) for s, b, c in ui.css_rules(ui.CSS_ROOT / "components.css") if c == ("@layer components",)}
    assert components[".seg__input:disabled + span"].get("opacity") == ".5"
    chosen = components[".seg__input:disabled:checked + span"]
    assert chosen.get("opacity") == "1" and chosen.get("color", "").startswith("var(--text")
    assert not [s for s, _, _ in _rules() if ":disabled" in s or ".seg__input" in s], "no disabled-state rule of its own"
    assert [r.get("type") for r in ui.select(tree(), 'input[name="spl-mode"]')] == ["radio"] * 3
    assert all("seg__input" in r.classes for r in ui.select(tree(), 'input[name="spl-mode"]'))


# ───────────────────────────── the module contract ─────────────────────────────


def test_the_modules_export_what_speak_and_the_tests_use():
    """live-ui.js exports mountLive (its returned contract is exercised in the Node test "[LR-07] mountLive
    keeps the contract Speak relies on…") and the partial path; live-session.js the engine and the three latency modes."""
    out = ui.run_node_script(
        "const ui = await import('./src/static/js/views/speak/live-ui.js');\n"
        "const live = await import('./src/static/js/views/speak/live-session.js');\n"
        "console.log(JSON.stringify({mount: typeof ui.mountLive, partial: ui.PARTIAL,\n"
        "  modes: live.MODES.map((m) => [m.value, m.label]), engine: typeof live.createLiveEngine}));\n"
    )
    data = json.loads(out.strip().splitlines()[-1])
    assert data == {
        "mount": "function",
        "partial": "/static/partials/speak-live.html",
        "modes": [["natural", "Book"], ["responsive", "Responsive"], ["instant_word", "Each word"]],
        "engine": "function",
    }
