"""Models (V7): the Node behaviour tests, the partial's structure and the view CSS.

The behaviour (provider cards and their state vocabulary, collapse, the Piper and STT lists, install and
offline cards, the in-place patch while operations run, the TTS | STT switch, the summary line, Refresh and
the loading/empty/error states) is tested in tests/js/models/*.test.js; this file runs them and checks what
only the static files can show.
"""

from __future__ import annotations

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

PARTIAL = ui.PARTIALS_DIR / "models.html"
CSS = ui.CSS_ROOT / "views" / "models.css"
JS_DIR = ui.JS_ROOT / "views" / "models"


def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


def node(id_: str) -> ui.Node:
    el = ui.by_id(tree(), id_)
    assert el is not None, f"#{id_} is missing from models.html"
    return el


def rules(chain_has: str = "") -> list[tuple[str, dict[str, str]]]:
    """(selector, declarations) in models.css, outside @media/@container unless chain_has names one."""
    out = []
    for sel, body, chain in ui.css_rules(CSS):
        conditional = [c for c in chain if c.startswith(("@media", "@supports", "@container"))]
        wanted = any(chain_has in c for c in conditional) if chain_has else not conditional
        if wanted:
            out.append((sel, dict(body)))
    return out


def decls(selector: str, chain_has: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for sel, body in rules(chain_has):
        if any(part.strip() == selector for part in sel.split(",")):
            out.update(body)
    return out


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("models"))
def test_js(path):
    ui.run_node_test(path, timeout=240)


def test_every_models_inventory_id_and_defect_has_a_test():
    """Every Models id and defect V7 owns (MD-01…04, MD-09…13, BUG-10) leads a test title that runs in its
    assigned file."""
    gaps = [g for g in missing("V7") if "/models/" in g[1] or g[1].endswith("test_ui_models.py")]
    assert gaps == []


# ───────────────────────────── the partial ─────────────────────────────


def test_the_partial_carries_every_element_id_from_the_spec():
    """The ids the view, its tests and the shell rely on; the placeholder marker is gone."""
    ids = {el.id for el in tree().elements() if el.id}
    for id_ in ("mo-title", "mo-refresh", "mo-sub", "mo-summary", "mo-note", "mo-error", "mo-error-title", "mo-error-msg",
                "mo-error-details", "mo-error-detail", "mo-error-retry", "mo-error-dismiss", "mo-state", "mo-list", "mo-menus"):
        assert id_ in ids, id_
    assert not ui.select(tree(), "[data-stub-body], [data-stub]")


def test_the_view_heading_the_status_line_and_the_error_banner():
    """View contract and live regions: one focusable h1; #mo-summary is the polite status line; the Refresh failure banner is an
    alert with an icon, a title and Details (never colour alone)."""
    h1s = ui.select(tree(), "h1")
    assert len(h1s) == 1 and h1s[0].id == "mo-title" and h1s[0].get("tabindex") == "-1" and h1s[0].has("data-view-heading")
    assert node("mo-summary").get("role") == "status"
    banner = node("mo-error")
    assert banner.get("role") == "alert" and banner.has("hidden")
    assert "banner--error" in banner.classes
    assert ui.select(banner, "svg use") and ui.select(banner, "details summary")
    assert node("mo-error-dismiss").get("aria-label") == "Dismiss"


def test_md01_the_sub_switch_is_a_labelled_segmented_control_of_native_radios():
    """[MD-01] TTS | STT is a .seg radiogroup (one tab stop, arrow keys) with a label and a reference to the
    list it switches; TTS is checked by default."""
    seg = node("mo-sub")
    assert seg.get("role") == "radiogroup" and seg.get("aria-label") == "Model type"
    assert seg.get("aria-controls") == "mo-list"
    radios = ui.select(seg, "input")
    assert [(r.get("type"), r.get("name"), r.get("value")) for r in radios] == [("radio", "mo-sub", "tts"), ("radio", "mo-sub", "stt")]
    assert radios[0].has("checked") and not radios[1].has("checked")
    labels = [ui.visible_text(label) for label in ui.select(seg, "label.seg__opt")]
    assert labels == ["Speech (TTS)", "Transcription (STT)"]


def test_overflow_menus_live_outside_the_cards():
    """A card is a size container (components.css: container-type), which would trap a position: fixed
    menu: the row menus go in #mo-menus, a sibling of the card list, never inside a card (the JS [MD-04]
    tests check each menu's parent in the running view)."""
    t = tree()
    menus = ui.by_id(t, "mo-menus")
    listing = ui.by_id(t, "mo-list")
    assert listing not in list(menus.ancestors()) and menus not in list(listing.ancestors())


# ───────────────────────────── the view CSS ─────────────────────────────


def test_narrow_cards_stack_each_row():
    """Container queries: a card narrower than 34rem stacks a row's text over its state and actions, so
    nothing scrolls sideways at 320 px; wider cards put the state and actions beside the text."""
    wide = decls('[data-view="models"] .mo-row')
    assert wide["grid-template-areas"] == '"text side" "error error"'
    narrow = decls('[data-view="models"] .mo-row', "@container")
    assert narrow["grid-template-areas"] == '"text" "side" "error"'
    # The actions end the line in every card: a row's state pill starts it, and a single-model card (Kokoro),
    # whose state is in its header, keeps its actions at the end too, as on desktop.
    assert decls('[data-view="models"] .mo-row__side', "@container")["justify-content"] == "space-between"
    assert decls('[data-view="models"] .mo-row--solo .mo-row__side', "@container")["justify-content"] == "flex-end"
    assert decls('[data-view="models"] .mo-row__side')["justify-content"] == "flex-end"
    assert "34rem" in next(c for _, _, chain in ui.css_rules(CSS) for c in chain if c.startswith("@container"))


def test_the_disclosure_shows_its_state_and_keeps_its_focus_ring():
    """MD-09: the chevron turns when a card collapses (aria-expanded drives it, not a class), and the header's
    focus ring is moved inside the card edge, never removed."""
    chev = decls('[data-view="models"] .mo-card__toggle[aria-expanded="false"] .mo-card__chev')
    assert "rotate(-90deg)" in chev["transform"]
    ring = decls('[data-view="models"] .mo-card__toggle:focus-visible')
    assert ring == {"outline-offset": "-2px"}
    toggle = decls('[data-view="models"] .mo-card__toggle')
    assert toggle["min-height"] == "56px" and toggle["width"] == "100%"


def test_meta_text_wraps_and_state_never_relies_on_colour():
    """Long model ids and commands wrap (WCAG 1.4.10); errors use --danger on --danger-soft (5.45:1) with the
    reason in words; the install command is selectable in one go."""
    assert decls('[data-view="models"] .mo-row__name')["overflow-wrap"] == "anywhere"
    assert decls('[data-view="models"] .mo-cmd__text')["overflow-wrap"] == "anywhere"
    assert decls('[data-view="models"] .mo-cmd__text')["user-select"] == "all"
    err = decls('[data-view="models"] .mo-row__error')
    assert err["background"] == "var(--danger-soft)"
    assert decls('[data-view="models"] .mo-row__error-text')["color"] == "var(--danger)"


def test_a_workers_reason_wraps_in_a_narrow_card():
    """The offline card's reason line can carry the worker's own text (one line of up to
    300 characters from its /health, such as a list of GPU architectures), so it wraps anywhere rather than push a
    narrow card sideways (WCAG 1.4.10); the JS tests in tests/js/models/cards.test.js check what it says."""
    assert decls('[data-view="models"] .mo-card__reason').get("overflow-wrap") == "anywhere"
