"""Speak: Generate mode (V3): the Node behaviour tests, the partial's structure and the view CSS.

The behaviour (request building, the latency-budgeted Generate path, presets, takes, hand-offs) is tested in
tests/js/speak/*.test.js; this file runs them and checks what only the static files can show.
"""

from __future__ import annotations

import re

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

PARTIAL = ui.PARTIALS_DIR / "speak.html"
CSS = ui.CSS_ROOT / "views" / "speak.css"


def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


def node(id_: str) -> ui.Node:
    el = ui.by_id(tree(), id_)
    assert el is not None, f"#{id_} is missing from speak.html"
    return el


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("speak"))
def test_js(path):
    ui.run_node_test(path, timeout=240)


def test_every_v3_inventory_id_and_defect_has_a_test():
    """Every id and defect V3 owns appears as [ID] in a test title that runs, in its assigned file."""
    assert missing("V3") == []


# ───────────────────────────── the partial ─────────────────────────────


def test_the_partial_carries_every_element_id_from_the_spec():
    """The ids other code, tests and docs rely on."""
    ids = {el.id for el in tree().elements() if el.id}
    for id_ in (
        "sp-title", "sp-mode", "sp-more", "sp-input", "sp-counter", "sp-limit-hint", "sp-upload", "sp-presets", "sp-generate",
        "sp-stream", "sp-gen-status", "sp-gen-error", "sp-inspector", "sp-inspector-body", "sp-summary", "sp-model", "sp-model-hint",
        "sp-restore-default", "sp-voice", "sp-blend", "sp-blend-toggle", "sp-blend-bar", "sp-blend-rows", "sp-blend-string",
        "sp-blend-copy", "sp-blend-paste", "sp-blend-add", "sp-blend-equal", "sp-blend-preview", "sp-reference", "sp-speed",
        "sp-speed-out", "sp-format", "sp-effects", "sp-reverb-room", "sp-pitch", "sp-advanced", "sp-instructions", "sp-save-voice",
        "sp-save-preset", "sp-takes", "sp-live-slot",
    ):
        assert id_ in ids, id_


def test_the_view_heading_mode_and_live_regions():
    """View contract and live regions: one focusable h1; the status is role=status and the error role=alert; the counter never speaks."""
    h1s = ui.select(tree(), "h1")
    assert len(h1s) == 1 and h1s[0].id == "sp-title" and h1s[0].get("tabindex") == "-1" and h1s[0].has("data-view-heading")
    assert node("sp-gen-status").get("role") == "status"
    assert node("sp-gen-error").get("role") == "alert"
    assert node("sp-counter").get("aria-live") == "off"
    mode = node("sp-mode")
    assert mode.get("role") == "radiogroup" and mode.get("aria-label")
    assert [i.get("value") for i in ui.select(mode, "input[type=radio]")] == ["generate", "live"]
    assert node("sp-live-slot").get("data-slot") == "speak-live" and node("sp-live-slot").has("hidden")
    assert not ui.select(tree(), "[data-stub-body], [data-stub]"), "the placeholder marker is gone"


def test_live_is_a_developer_tool_hidden_until_settings_shows_it():
    """Speak shows Generate only unless Settings › Developer turns
    on Speak › Live, so the mode switch starts hidden (index.js shows it then: the Node tests in
    tests/js/speak/live-switch.test.js) and names Live "Live (test)". With no switch, a phone's header keeps no
    empty row for it: the title and the More button are its only row."""
    mode = node("sp-mode")
    assert mode.has("hidden")
    assert [o.text_content().strip() for o in ui.select(mode, ".seg__opt")] == ["Generate", "Live (test)"]
    off = _find(':not([data-show-live="true"]) .sp-head', "max-width: 599px")
    assert off and off[0][1].get("grid-template-areas") == '"title more"', off
    assert not _find(':not([data-show-live="true"]) .sp-head', "min-width: 600px"), "wide screens keep one row anyway"


def test_quiet_and_persistent_live_regions():
    """Status and announcements: the inspector's load failure is a persistent polite region (in the tree before its first
    message, never hidden); the RTF that arrives after "Done" is aria-hidden; outputs never announce
    themselves (an <output> is a live region by default); a failed blend preview is an alert next to it."""
    err = node("sp-inspector-error")
    assert err.get("role") == "status" and not err.has("hidden")
    extra = node("sp-gen-status-extra")
    assert extra.get("aria-hidden") == "true" and "sp-gen-status" in [a.id for a in extra.ancestors()]
    for id_ in ("sp-blend-string", "sp-speed-out"):
        assert node(id_).tag == "output" and node(id_).get("aria-live") == "off", id_
    blend_error = node("sp-blend-error")
    assert blend_error.get("role") == "alert" and blend_error.has("hidden")
    assert "sp-blend" in [a.id for a in blend_error.ancestors()]
    assert node("sp-model").get("aria-describedby") == "sp-model-hint"


def test_the_reference_picker_is_outside_advanced_and_the_upload_is_reachable():
    """SP-16, SP-36 (old V::test_speak_reference_selector_is_visible_outside_advanced_controls), AX-05."""
    t = tree()
    advanced = ui.by_id(t, "sp-advanced")
    reference = ui.by_id(t, "sp-reference")
    assert "sp-advanced" not in [a.id for a in reference.ancestors()]
    assert "sp-instructions" in [el.id for el in advanced.elements()]
    upload = ui.by_id(t, "sp-upload")
    assert upload.get("type") == "file" and "visually-hidden" in upload.classes and not upload.has("hidden")
    assert upload.get("accept") == ".txt,text/plain"
    labels = [l for l in ui.select(t, "label") if l.get("for") == "sp-upload"]
    assert labels and labels[0].text_content().strip() == "Open .txt"


def test_the_controls_are_named_and_the_disclosures_point_at_their_panels():
    """AX-01, AX-02, AX-06 for this partial: names for every control, aria-controls on every aria-expanded."""
    t = tree()
    for el in t.elements():
        if el.tag in ("input", "select", "textarea", "button"):
            assert ui.accessible_name(el, t).strip(), f"{el!r} has no accessible name"
        if el.has("aria-expanded"):
            assert ui.by_id(t, el.get("aria-controls") or "") is not None, f"{el!r} aria-controls"
    assert node("sp-summary").get("aria-controls") == "sp-inspector-body"
    assert node("sp-blend-toggle").get("aria-controls") == "sp-blend"
    assert node("sp-pitch").get("aria-labelledby") == "sp-pitch-label"
    speed = node("sp-speed")
    assert (speed.get("min"), speed.get("max"), speed.get("step")) == ("0.5", "2", "0.05")
    assert [o.get("value") for o in ui.select(node("sp-format"), "option")] == ["mp3", "wav", "flac", "opus", "aac", "m4a", "pcm"]
    assert [b.get("data-effect") for b in ui.select(node("sp-effects"), "button[data-effect]")] == ["normalize", "podcast_eq", "reverb", "pitch", "robot"]
    assert all(b.get("aria-pressed") == "false" for b in ui.select(node("sp-effects"), "button[data-effect]"))


def test_the_editing_bar_is_a_labelled_group_above_the_editor_with_its_three_actions():
    """Editing a named voice's version or a preset from Voices shows
    #sp-editing, the banner component as a labelled group (not a live region: index.js announces the start once),
    above the layout like #sp-notice, hidden until an edit begins. Its actions are Update (voice or preset),
    Save as new… (presets only, hidden until then) and Cancel; the JS tests are tests/js/speak/editing.test.js."""
    t = tree()
    bar = node("sp-editing")
    assert bar.has("hidden") and "banner" in bar.classes
    assert bar.get("role") == "group" and not bar.get("aria-live")
    assert bar.get("aria-labelledby") == "sp-editing-title" and bar.get("aria-describedby") == "sp-editing-hint"
    assert "sp-layout" not in [a.id for a in bar.ancestors()], "outside the layout grid, whose children are all placed"
    buttons = ui.select(bar, "button")
    assert [b.id for b in buttons] == ["sp-editing-update", "sp-editing-new", "sp-editing-cancel"]
    assert all(b.get("type") == "button" for b in buttons)
    assert [ui.accessible_name(b, t) for b in buttons] == ["Update voice", "Save as new…", "Cancel"]
    assert node("sp-editing-new").has("hidden")
    assert not [b for b in buttons if "btn--primary" in b.classes], "Generate stays the page's one primary action"


def test_phone_order_puts_the_voice_settings_between_the_editor_and_generate():
    """On phones the inspector is inline after the editor, before presets and Generate (DOM order)."""
    order = [el.id or " ".join(el.classes) for el in node("sp-layout").children if el.is_element]
    assert order.index("sp-editor") < order.index("sp-inspector") < order.index("sp-presets-row") < order.index("sp-genbar") < order.index("sp-takes-section")


# ───────────────────────────── the view CSS ─────────────────────────────


def _rules():
    return ui.css_rules(CSS)


def _find(selector_part: str, chain_has: str = ""):
    out = []
    for selector, decls, chain in _rules():
        if selector_part in selector and (not chain_has or any(chain_has in c for c in chain)):
            out.append((selector, dict(decls), chain))
    return out


def test_the_phone_inspector_body_is_collapsible_and_always_open_on_desktop():
    """Below 960 px the body hides unless the summary opened it; at 960 px and up the summary is gone."""
    phone = _find(':not([data-open="true"]) .sp-inspector__body', "max-width: 959.98px")
    assert phone and phone[0][1].get("display") == "none"
    desktop = _find(".sp-summary", "min-width: 960px")
    assert any(d.get("display") == "none" for _, d, _ in desktop)
    layout = _find(".sp-layout", "min-width: 960px")
    assert any("var(--inspector-w)" in d.get("grid-template-columns", "") for _, d, _ in layout)


def test_desktop_tab_order_follows_the_columns_and_phones_keep_the_dom_order():
    """WCAG 2.4.3: at 960 px and up the inspector is the right-hand column, after the editor in
    the DOM, so DOM order would take Tab from the text through 15 inspector controls before the presets and
    Generate under it. The desktop grid sets reading-flow: grid-columns: Tab goes down the left column (editor,
    presets, Generate, Live, takes), then the inspector, which starts the second column. Every grid item is
    placed by a named area (so the columns are what reading-flow reads), and nothing uses `order`, which would
    split the phone's focus order (the DOM order) from what it shows."""
    layout = _find(".sp-layout", "min-width: 960px")
    grid = [d for s, d, _ in layout if s.strip() == '[data-view="speak"] .sp-layout']
    assert grid and grid[0].get("display") == "grid" and grid[0].get("reading-flow") == "grid-columns", grid
    rows = [r.strip().strip('"').split() for r in re.findall(r'"[^"]+"', grid[0]["grid-template-areas"])]
    assert [r[0] for r in rows] == ["editor", "presets", "gen", "live", "takes"], rows
    assert {r[1] for r in rows} == {"inspector"}, "the inspector is the whole second column"
    areas = {s.strip(): dict(d)["grid-area"] for s, d, c in _rules()
             if "grid-area" in dict(d) and any("min-width: 960px" in x for x in c)}
    children = [el for el in node("sp-layout").children if el.is_element]
    classes = {"sp-editor": "editor", "sp-inspector": "inspector", "sp-presets-row": "presets", "sp-genbar": "gen",
               "sp-live-slot": "live", "sp-takes": "takes"}
    for child in children:
        cls = next(c for c in child.classes if c in classes)
        assert areas.get(f'[data-view="speak"] .{cls}') == classes[cls], (cls, areas)
    assert len(children) == len(classes), "every child of the layout is a placed grid item"
    assert not [s for s, d, _ in _rules() if "order" in dict(d)], "no CSS order: focus follows what the page shows"
    phone_flow = [s for s, d, c in _rules() if "reading-flow" in dict(d) and not any("min-width: 960px" in x for x in c)]
    assert not phone_flow, "phones keep the DOM order"


def test_blend_actions_wrap_instead_of_cutting_their_labels():
    """WCAG 1.4.12: in the 360 px inspector (and under text spacing) the four blend buttons wrap to
    new rows rather than shrink: a button is never narrower than its label. They come in two pairs, so a
    narrow panel shows them two by two, never three and one."""
    for selector in (".sp-blend__actions", ".sp-blend__pair"):
        rules = [d for s, d, _ in _find(selector) if s.strip().endswith(selector)]
        assert rules and rules[0].get("display") == "flex" and rules[0].get("flex-wrap") == "wrap", selector
    buttons = [d for s, d, _ in _find(".sp-blend__pair .btn")]
    assert buttons and buttons[0].get("min-width") == "max-content"
    assert not any("repeat(2" in d.get("grid-template-columns", "") for _, d, _ in _find(".sp-blend__"))
    pairs = [[b.id for b in ui.select(pair, "button")] for pair in ui.select(node("sp-blend"), ".sp-blend__pair")]
    assert pairs == [["sp-blend-add", "sp-blend-preview"], ["sp-blend-equal", "sp-blend-paste"]]


def test_a_blocked_save_preset_chip_looks_disabled():
    """"Save preset…" is aria-disabled (focusable, a click says why) while there is no setup to save
    (the Node test [SP-35] shows when). It is a `.chip.chip--action` (presets.js); the chip component dims an
    aria-disabled chip like a disabled one and gives it no hover fill, and the view leaves that to the
    component: no rule of its own undoes or repeats it."""
    components = ui.css_rules(ui.CSS_ROOT / "components.css")
    dimmed = [dict(b) for s, b, c in components if '.chip[aria-disabled="true"]' in [p.strip() for p in s.split(",")]]
    assert dimmed and dimmed[0].get("opacity") == ".5"
    hovers = [s for s, b, c in components if s.startswith((".chip:hover", ".chip--action:hover")) and any(n in ("background",) for n, _ in b)]
    assert hovers and all(':not([aria-disabled="true"])' in s for s in hovers), hovers
    assert not [s for s, _, _ in _rules() if "aria-disabled" in s and ".chip" in s], "Speak keeps no chip state rule of its own"


def test_takes_reflow_by_their_own_width_and_the_waveform_uses_tokens():
    """Takes are container-query cards; the waveform colours come from tokens (no literal colours)."""
    assert any(d.get("container-type") == "inline-size" for s, d, _ in _find(".sp-take") if s.strip().endswith(".sp-take"))
    assert _find(".sp-take__text", "@container")
    wave = _find(".sp-wave")
    assert wave and wave[0][1].get("--sp-wave-played") == "var(--accent)" and wave[0][1].get("color", "").startswith("var(--")


FORCED = ("@layer utilities", "@media (forced-colors: active)")


def test_the_view_css_is_scoped_layered_and_has_no_colour_literals():
    """View CSS scoping: every selector under [data-view="speak"], inside @layer views, tokens only. The one exception
    is the forced-colours block: @layer utilities > @media (forced-colors: active), system colours."""
    text = CSS.read_text(encoding="utf-8")
    assert "@layer views" in text
    for selector, decls, chain in _rules():
        assert chain and (chain[0] == "@layer views" or tuple(chain) == FORCED), (selector, chain)
        for part in selector.split(","):
            assert part.strip().startswith('[data-view="speak"]'), part
        for name, value in decls:
            assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", value), f"{selector} {name}: {value}"
            if tuple(chain) == FORCED:
                assert not re.search(r"var\(--", value), f"forced colours use system colours only: {selector} {name}: {value}"


def test_the_blend_bar_keeps_its_shares_in_forced_colours():
    """Forced colours drop the blend segment colours; each segment keeps a CanvasText edge, so the
    shares still show as widths in the legend's order (the legend and the bar's aria-label say them in words, the
    Node test [SP-26]). The swatches, which only repeat the colours, step aside (an outlined square would read as
    a checkbox), and the tinted blend panel keeps its outline."""
    forced = {}
    for s, b, c in _rules():
        if tuple(c) == FORCED:
            for part in s.split(","):
                forced.setdefault(part.strip(), {}).update(dict(b))
    for selector in ('[data-view="speak"] .sp-blend__bar .blend-bar__seg', '[data-view="speak"] .sp-blend'):
        assert forced.get(selector, {}).get("border") == "1px solid CanvasText", selector
    assert forced['[data-view="speak"] .sp-blend__row .swatch'].get("display") == "none"
    bar = node("sp-blend-bar")
    assert "sp-blend__bar" in bar.classes and bar.get("role") == "img"


def test_the_inspector_and_takes_keep_their_edges_in_forced_colours():
    """The inspector and the take cards draw their edges with box-shadows, which forced colours remove; each
    keeps a CanvasText border there, and the take that plays (its accent edge) a Highlight one."""
    shadowed = [s for s, b, c in _rules() if c and c[0] == "@layer views" and s.strip() in ('[data-view="speak"] .sp-inspector', '[data-view="speak"] .sp-take')
                and any(n == "box-shadow" for n, _ in b)]
    assert len(shadowed) == 2, shadowed
    forced = {}
    for s, b, c in _rules():
        if tuple(c) == FORCED:
            for part in s.split(","):
                forced.setdefault(part.strip(), {}).update(dict(b))
    assert forced['[data-view="speak"] .sp-inspector'].get("border") == "1px solid CanvasText"
    assert forced['[data-view="speak"] .sp-take'].get("border") == "1px solid CanvasText"
    playing = forced['[data-view="speak"] .sp-takes__item.is-playing .sp-take']
    assert playing.get("border-color") == "Highlight" and "Highlight" in playing.get("outline", "")


def test_main_starts_the_boot_voice_after_the_first_view_without_preloading_it():
    """Boot order and the latency budget: main.js publishes Speak's voice for the Reader before Speak's view has run, by
    importing views/speak/speak-voice.js dynamically once boot() resolved (the first view has mounted); a static
    import would put it in the preloaded boot graph. The module itself imports only preloaded core modules."""
    main = (ui.JS_ROOT / "main.js").read_text(encoding="utf-8")
    assert re.search(r"boot\(\)\.then\(", main)
    hook = main[main.index("boot().then("):]
    assert "import('./views/speak/speak-voice.js')" in hook and "startBootVoice()" in hook
    closure = {p.relative_to(ui.JS_ROOT).as_posix() for p in ui.static_import_closure(ui.JS_ROOT / "main.js")}
    assert not [p for p in closure if p.startswith("views/")], "no view module in main.js's static graph"
    voice = ui.JS_ROOT / "views" / "speak" / "speak-voice.js"
    own = {p.relative_to(ui.JS_ROOT).as_posix() for p in ui.static_import_closure(voice)} - {"views/speak/speak-voice.js"}
    assert own and own <= closure, sorted(own - closure)
