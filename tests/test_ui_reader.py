"""Reader: library and reading surface (V1). The JS behaviour tests in tests/js/reader, the partial's
structure (the ids and controls the view relies on), and the CSS intent checks for the reading surface.

The global audits in test_ui_core.py cover this partial and stylesheet too (names, ids, CSP, scoping, tokens).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

PARTIAL = ui.PARTIALS_DIR / "reader.html"
CSS = ui.CSS_ROOT / "views" / "reader.css"


def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


def by_id(id_: str) -> ui.Node:
    node = ui.by_id(tree(), id_)
    assert node is not None, f"#{id_} is missing from partials/reader.html"
    return node


def decls(selector: str, chain_has: str = "") -> dict[str, str]:
    """The declarations of one exact selector in reader.css (outside @media, or inside the one named)."""
    out: dict[str, str] = {}
    for sel, body, chain in ui.css_rules(CSS):
        conditional = [c for c in chain if c.startswith(("@media", "@supports"))]
        wanted = any(chain_has in c for c in conditional) if chain_has else not conditional
        if wanted and selector in [part.strip() for part in sel.split(",")]:
            out.update(dict(body))
    return out


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("reader"))
def test_js(path):
    ui.run_node_test(path)


def test_v1_inventory_ids_and_defects_have_tests():
    """Every inventory id and defect owned by V1 leads a test title in its assigned file."""
    assert missing("V1") == []


# ───────────────────────────── partial structure ─────────────────────────────


def test_rev25_reading_view_has_exactly_one_document_title():
    """[REV-25] There is exactly one #rd-doc-title (v1 had a phone and a desktop copy): the Reader partial has two
    view headings, the library's #rd-title and the reading view's #rd-doc-title, inside the one top bar."""
    root = tree()
    assert len([n for n in root.elements() if n.id == "rd-doc-title"]) == 1
    headings = [n.id for n in ui.select(root, "h1[data-view-heading]")]
    assert headings == ["rd-title", "rd-doc-title"]
    title = by_id("rd-doc-title")
    assert title.get("tabindex") == "-1"
    topbar = next(a for a in title.ancestors() if "rd-topbar" in a.classes)
    assert topbar.tag == "header" and topbar.get("data-chrome") == "top"
    assert ui.select_one(topbar, "#rd-top-chapter") is not None
    assert len(ui.select(root, ".rd-topbar")) == 1


def test_library_markup_has_the_e1_controls():
    """The library's controls: Add, search (labelled "Search library"), the four filter chips in a labelled
    group, the sort select with its visible label, the card, the count (a status region), the list and the empty
    state, and the shared row menu with Open, Start over, Mark as finished and Delete."""
    root = tree()
    add = by_id("rd-add")
    assert "btn--primary" in add.classes and add.get("data-action") == "add" and add.get("aria-controls") == "rd-add-dialog"
    search = by_id("rd-search")
    assert search.get("type") == "search" and search.get("aria-label") == "Search library"
    group = by_id("rd-filter")
    assert group.get("role") == "group" and group.get("aria-label")
    chips = ui.select(group, "button.chip")
    assert [c.get("data-filter") for c in chips] == ["all", "progress", "new", "done"]
    assert [ui.visible_text(c).strip() for c in chips] == ["All", "In progress", "Not started", "Finished"]
    assert all(c.get("aria-pressed") in ("true", "false") for c in chips)
    sort = by_id("rd-sort")
    assert [o.get("value") for o in ui.select(sort, "option")] == ["recent", "title", "progress"]
    assert ui.accessible_name(sort, root) == "Sort"
    assert by_id("rd-count").get("role") == "status"
    assert by_id("rd-list").tag == "ul"
    assert "empty" in by_id("rd-empty").classes
    assert by_id("rd-hero").get("aria-labelledby") == "rd-hero-kicker"
    menu = by_id("rd-row-menu")
    assert menu.get("role") == "menu" and menu.has("hidden")
    items = ui.select(menu, '[role="menuitem"]')
    assert [i.get("data-menu") for i in items] == ["open", "restart", "finish", "delete"]
    assert ui.visible_text(items[3]).strip() == "Delete…" and "menu__item--danger" in items[3].classes


def test_add_sheet_offers_file_and_paste_only():
    """[RL-11] The Add sheet: a File | Paste segmented control (one radio group, so arrows move like tabs), the file
    input (visually hidden, its label is the dropzone) accepting only the Reader types, and the paste fields."""
    root = tree()
    dialog = by_id("rd-add-dialog")
    assert dialog.tag == "dialog" and "sheet" in dialog.classes and dialog.get("aria-labelledby") == "rd-add-title"
    radios = ui.select(dialog, 'input[name="rd-source"]')
    assert [r.get("value") for r in radios] == ["file", "paste"] and radios[0].has("checked")
    seg = next(a for a in radios[0].ancestors() if a.get("role") == "radiogroup")
    assert "seg" in seg.classes and seg.get("aria-label")
    file = by_id("rd-file")
    assert file.get("accept") == ".epub,.txt,.md,.markdown,.html,.htm"
    assert "visually-hidden" in file.classes and not file.has("hidden")
    assert by_id("rd-dropzone").tag == "label" and by_id("rd-dropzone").get("for") == "rd-file"
    assert by_id("rd-paste-title").get("maxlength") == "200"
    assert by_id("rd-paste-text").tag == "textarea"
    submit = by_id("rd-add-submit")
    assert submit.get("type") == "submit" and ui.visible_text(submit).strip() == "Add to library"
    assert by_id("rd-add-error").get("role") == "alert"
    assert ui.accessible_name(by_id("rd-paste-text"), root) == "Text or Markdown"


def test_reading_markup_has_the_e1_2_elements():
    """The reading surface: #rd-reading carries the appearance attributes (highlight "both" by default), the
    top bar's back / appearance / chapters buttons, the aside, the labelled text article, Back to playing, the
    player slot, the opening notice (a status region), and the appearance and chapters sheets."""
    surface = by_id("rd-reading")
    for name, value in {"data-rd-paper": "auto", "data-rd-face": "serif", "data-rd-leading": "normal",
                        "data-rd-measure": "medium", "data-rd-justify": "false", "data-rd-hl": "both"}.items():
        assert surface.get(name) == value, name
    assert by_id("rd-back").get("aria-label") == "Back to library"
    assert by_id("rd-appearance-btn").get("aria-label") == "Reading appearance"
    assert by_id("rd-chapters-btn").get("aria-label") == "Chapters"
    assert by_id("rd-side").tag == "aside"
    assert by_id("rd-chapter-list").tag == "ol"
    text = by_id("rd-text")
    assert text.tag == "article" and text.get("aria-label") == "Document text" and not text.has("lang")
    assert ui.visible_text(by_id("rd-return")).strip() == "Back to playing"
    slot = by_id("rd-player-slot")
    assert slot.get("data-slot") == "reader-player" and not list(slot.elements())
    assert by_id("rd-opening").get("role") == "status"
    assert by_id("rd-chapters-sheet").tag == "dialog" and by_id("rd-chapters-sheet-list").tag == "ol"


def test_appearance_sheet_controls():
    """The appearance sheet: the size stepper, the Typeface / Line spacing / Width / Highlight segments, the
    five paper swatches with visible labels, the Auto-scroll and Justify switches, and Reset. Each group has a
    visible label."""
    root = tree()
    sheet = by_id("rd-appearance-sheet")
    groups = {
        "rd-face": ["serif", "sans", "legible"], "rd-leading": ["compact", "normal", "relaxed"],
        "rd-measure": ["narrow", "medium", "wide"], "rd-paper": ["auto", "paper", "sepia", "night", "dusk"],
        "rd-hl": ["tint", "underline", "both", "off"],
    }
    for name, values in groups.items():
        radios = ui.select(sheet, f'input[name="{name}"]')
        assert [r.get("value") for r in radios] == values, name
        group = next(a for a in radios[0].ancestors() if a.get("role") == "radiogroup")
        assert group.get("aria-labelledby"), name
    labels = [ui.visible_text(n).strip() for n in ui.select(sheet, ".rd-paper__label")]
    assert labels == ["Match app", "Paper", "Sepia", "Night", "Dusk"]
    assert by_id("rd-size-dec").get("aria-label") == "Smaller text" and by_id("rd-size-inc").get("aria-label") == "Larger text"
    assert by_id("rd-size-value").tag == "output"
    for id_ in ("rd-autoscroll", "rd-justify"):
        switch = by_id(id_)
        assert switch.get("role") == "switch" and switch.get("type") == "checkbox"
        assert ui.accessible_name(switch, root)
    assert ui.visible_text(by_id("rd-appearance-reset")).strip() == "Reset"


# ───────────────────────────── CSS intent ─────────────────────────────


def test_rev35_spoken_sentence_is_marked_by_tint_and_underline_and_never_moves_text():
    """[REV-35] "Both" (the default) paints the spoken sentence with the paper's highlight and a 2 px underline in its
    accent, so it is never marked by colour alone; "Tint" and "Underline" each keep one. Highlighting changes only
    the background and the underline: no property that moves text (RE-52)."""
    tint = decls('.rd-reading[data-rd-hl="both"] .rd-s.is-now')
    assert tint.get("background") == "var(--rd-hl)"
    assert tint.get("text-decoration-line") == "underline"
    assert tint.get("text-decoration-color") == "var(--rd-accent)"
    assert tint.get("text-decoration-thickness") == "2px"
    assert decls('.rd-reading[data-rd-hl="tint"] .rd-s.is-now').get("background") == "var(--rd-hl)"
    assert decls('.rd-reading[data-rd-hl="underline"] .rd-s.is-now').get("text-decoration-line") == "underline"
    assert not decls('.rd-reading[data-rd-hl="off"] .rd-s.is-now')
    moving = {"font-weight", "font-size", "padding", "margin", "border", "letter-spacing", "line-height", "display", "width"}
    for selector, body, _ in ui.css_rules(CSS):
        if ".is-now" in selector:
            assert not moving & {name for name, _ in body}, selector


def test_reading_text_is_cheap_to_lay_out_and_uses_the_paper_tokens():
    """Long documents: off-screen paragraphs skip layout (content-visibility: auto with an intrinsic size, for the latency budget), the
    text column is the reading measure at most, and the paper colours come only from the --rd-* tokens."""
    p = decls(".rd-p")
    assert p.get("content-visibility") == "auto" and "4em" in p.get("contain-intrinsic-size", "")
    text = decls(".rd-text")
    assert text.get("max-width") == "min(var(--rd-measure), 100%)" and text.get("width") == "100%"
    assert text.get("color") == "var(--rd-text)" and text.get("font-size") == "var(--rd-size)"
    sizes = {decls(f'.rd-reading[data-rd-size="{i}"]').get("--rd-size") for i in range(7)}
    assert sizes == {".9375rem", "1.0625rem", "1.1875rem", "1.25rem", "1.375rem", "1.5625rem", "1.75rem"}, "15–28 px (the type scale)"
    assert decls(".rd-reading").get("--rd-size") == "1.1875rem", "19 px below 960 px"
    assert decls(".rd-reading", "min-width: 960px").get("--rd-size") == "1.25rem", "20 px from 960 px"
    assert decls(".rd-reading::before", "max-width: 959px").get("background") == "var(--rd-bg)", "the paper fills the phone screen"
    assert decls(".rd-text", "min-width: 960px").get("background") == "var(--rd-bg)", "a paper panel on desktop"


def test_reader_top_bar_is_sticky_chrome_with_an_opaque_fallback():
    """The Reader top bar is sticky chrome (counted by the shell's metrics through data-chrome="top"), drawn from
    the tokens (the app's, or under a paper theme the ones derived from the paper, see the paper-chrome test), blurs
    only where allowed, and turns opaque without blur support or with reduced transparency."""
    bar = decls(".rd-topbar")
    assert bar.get("position") == "sticky" and bar.get("top") == "0"
    assert bar.get("background") == "var(--bar-bg)" and bar.get("color") == "var(--text)"
    assert "safe-area-inset-top" in bar.get("padding", "")
    assert decls(".rd-topbar", "backdrop-filter").get("background") == "var(--bg)"
    assert decls(".rd-topbar", "prefers-reduced-transparency").get("backdrop-filter") == "none"


def test_ax08_long_title_wraps_in_the_opening_notice_on_phones():
    """[AX-08] A long title wraps in the "Opening “T”…" notice on a phone: the card never grows past the screen minus
    the gutters and its text wraps anywhere (checked at 320 px in the screenshots too)."""
    card = decls(".rd-opening__card")
    assert card.get("position") == "fixed"
    # It hangs under the top bar (the chrome metrics' top), never in the middle of the reading column, where it
    # sliced through lines of text.
    assert card.get("top") == "calc(var(--chrome-top, 0px) + var(--sp-3))"
    assert "100vw - 2 * var(--gutter)" in card.get("max-width", "")
    assert card.get("width") == "max-content"
    text = decls(".rd-opening__text")
    assert text.get("overflow-wrap") == "anywhere" and text.get("white-space") == "normal" and text.get("min-width") == "0"


def test_rows_wrap_long_titles_and_show_their_percentage_at_every_width():
    """[RL-06] Long titles wrap (no horizontal scroll at 320 px). The percentage (or "New") shows at every width: on
    phones the percentage used to be visually hidden while "New" stayed, so a started row looked unstarted. The
    meta keeps its count and its length each on one line ("under a / minute" wrapped)."""
    title = decls(".rd-row__title")
    assert title.get("white-space") == "normal" and title.get("overflow-wrap") == "anywhere"
    for selector, body, chain in ui.css_rules(CSS):
        if "rd-row__pct" in selector:
            props = dict(body)
            assert "clip-path" not in props and props.get("position") != "absolute", (selector, chain)
            assert props.get("display") != "none" and props.get("visibility") != "hidden", (selector, chain)
    assert decls(".rd-row__pct").get("flex") == "none"
    assert decls(".rd-row__fact").get("white-space") == "nowrap"


def test_appearance_sheet_keeps_the_text_visible():
    """The appearance sheet is non-blocking in look: a scrim at half the usual strength (20%), and on desktop it sits
    to the side of the text."""
    assert decls(".rd-appearance::backdrop").get("background") == "color-mix(in srgb, var(--scrim) 50%, transparent)"
    side = decls(".rd-appearance", "min-width: 960px")
    assert side.get("margin", "").endswith("auto auto")


def test_current_chapter_and_finished_chapters_keep_their_contrast():
    """[RE-57] The current chapter is accent-soft with accent text (an inset ring, not surface-3, when hovered);
    finished chapters use --text-2, never opacity (AX-07)."""
    current = decls(".rd-ch.is-current .rd-ch__btn")
    assert current.get("background") == "var(--accent-soft)" and current.get("color") == "var(--accent)"
    hover = decls(".rd-ch.is-current .rd-ch__btn:hover")
    assert hover.get("background") == "var(--accent-soft)" and "var(--border-strong)" in hover.get("box-shadow", "")
    assert decls(".rd-ch.is-done .rd-ch__t").get("color") == "var(--text-2)"
    text = CSS.read_text(encoding="utf-8")
    assert not re.search(r"is-done[^{]*\{[^}]*opacity", text)


FORCED = ("@layer utilities", "@media (forced-colors: active)")


def forced_rules() -> dict[str, dict[str, str]]:
    """The declarations of reader.css's forced-colours block, by selector."""
    out: dict[str, dict[str, str]] = {}
    for selector, body, chain in ui.css_rules(CSS):
        if tuple(chain) == FORCED:
            for part in selector.split(","):
                out.setdefault(part.strip(), {}).update(dict(body))
    return out


PAPERS = ("paper", "sepia", "night", "dusk")
PAPER_CHROME = (".rd-topbar", ".rd-side", ".rd-text", ".rd-return", ".rdp-transport")
PAPER_SCOPE = '.rd-reading:not([data-rd-paper="auto"])'


def paper_tokens(paper: str) -> dict[str, str]:
    """A paper theme's five --rd-* tokens, from tokens.css."""
    out: dict[str, str] = {}
    for selector, body, _ in ui.css_rules(ui.CSS_ROOT / "tokens.css"):
        if selector.strip() == f'[data-rd-paper="{paper}"]':
            out.update(dict(body))
    return out


def paper_chrome() -> dict[str, dict[str, str]]:
    """The custom properties reader.css sets on each part of the Reader's chrome under a paper theme."""
    out: dict[str, dict[str, str]] = {}
    for selector, body, chain in ui.css_rules(CSS):
        if any(c.startswith(("@media", "@supports")) for c in chain):
            continue
        for part in selector.split(","):
            p = part.strip()
            if p.startswith(PAPER_SCOPE + " "):
                out.setdefault(p[len(PAPER_SCOPE) + 1:], {}).update({k: v for k, v in body if k.startswith("--")})
    return {k: v for k, v in out.items() if v}


def test_paper_themes_theme_the_readers_own_chrome_and_keep_contrast():
    """[TH-03] Unlike the first token design (which kept the app tokens on the Reader's bars under every
    paper): those bars are about 92% opaque, so the paper bled through them, a grey band over Night and a white
    frame around Sepia, and on desktop the paper was a bright strip in dark chrome. Under Paper, Sepia, Night and
    Dusk the whole page and the Reader's own chrome (top bar, aside, text panel, Back to playing, transport) take the
    paper's colours, as Books themes its bars; sheets, banners, toasts and the rail stay app chrome, and Match app
    changes nothing. Every app token those controls read is derived from the paper's own five tokens (no new
    colour), and every pair they draw is recomputed here: text and meta 4.5:1 on the bar, the raised bar, a chip
    and a hover; the accent on its soft tint 4.5:1; the play icon on its fill and hover 4.5:1; focus, the scrubber's
    track and its fill 3:1."""
    chrome = paper_chrome()
    assert set(chrome) == set(PAPER_CHROME), sorted(chrome)
    remap = chrome[".rd-topbar"]
    assert all(body == remap for body in chrome.values()), "one set of paper tokens for every part of the chrome"
    needed = {"--bg", "--raised", "--surface-1", "--surface-2", "--surface-3", "--text", "--text-2", "--text-3", "--border",
              "--border-strong", "--edge", "--hover", "--accent", "--accent-soft", "--accent-fill", "--accent-fill-hover",
              "--on-accent", "--focus", "--bar-bg", "--bar-bg-raised"}
    assert needed <= set(remap), sorted(needed - set(remap))
    for name, value in remap.items():
        refs = re.findall(r"var\((--[\w-]+)\)", value)
        assert refs and all(ref.startswith("--rd-") for ref in refs), f"{name}: {value} is not derived from the paper"
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", value), f"{name}: {value} adds a colour"
    for part in PAPER_CHROME:
        assert decls(f"{PAPER_SCOPE} {part}").get("color") == "var(--text)", f"{part}: inherited text keeps the app colour"
    page = decls(f"{PAPER_SCOPE}::before")
    assert page.get("background") == "var(--rd-bg)" and page.get("position") == "fixed" and page.get("z-index") == "-1"
    assert decls(f"{PAPER_SCOPE} .rd-text", "min-width: 960px").get("box-shadow") == "none", "the page, not a card on it"
    for paper, scheme in (("paper", "light"), ("sepia", "light"), ("night", "dark"), ("dusk", "dark")):
        for part in PAPER_CHROME:
            assert decls(f'.rd-reading[data-rd-paper="{paper}"] {part}').get("color-scheme") == scheme, (paper, part)
    assert decls(f"{PAPER_SCOPE} .rdp-status.is-error").get("color") == "var(--text)", "no paper red: the ring says error"
    problems = []
    for paper in PAPERS:
        table = paper_tokens(paper)
        assert {"--rd-bg", "--rd-text", "--rd-meta", "--rd-accent"} <= set(table), paper
        page_bg = ui.resolve_color(table["--rd-bg"], table)

        def col(name, over=page_bg, table=table):
            return ui.resolve_color(remap[name], table, over=over)

        bar, raised = col("--bar-bg"), col("--bar-bg-raised")
        chip, hover, soft = col("--surface-2", bar), col("--hover", bar), col("--accent-soft", bar)
        fill, fill_hover = col("--accent-fill"), col("--accent-fill-hover")
        pairs = [
            ("text on the bar", col("--text"), bar, 4.5),
            ("meta on the bar", col("--text-2"), bar, 4.5),
            ("text on the raised bar", col("--text"), raised, 4.5),
            ("meta on the raised bar", col("--text-2"), raised, 4.5),
            ("text on a chip", col("--text"), chip, 4.5),
            ("meta on a chip", col("--text-2"), chip, 4.5),
            ("text on a hover", col("--text"), hover, 4.5),
            ("accent on its soft tint", col("--accent"), soft, 4.5),
            ("play icon on its fill", col("--on-accent"), fill, 4.5),
            ("play icon on its hover", col("--on-accent"), fill_hover, 4.5),
            ("focus ring (UI)", col("--focus"), bar, 3.0),
            ("scrubber track (UI)", col("--border-strong"), bar, 3.0),
            ("scrubber fill (UI)", col("--accent"), bar, 3.0),
        ]
        for label, fg, bg, minimum in pairs:
            ratio = ui.contrast(fg, bg)
            if round(ratio, 2) < minimum:
                problems.append(f"{paper}: {label} {ratio:.2f} < {minimum}")
    assert problems == []


def test_one_chapters_control_below_1180_px():
    """[RE-57] Below 1180 px the transport's Chapters chip is the one Chapters control (in thumb reach
    on phones, labelled on tablets); the top bar's button shows only while the reading view has no transport to
    hold it (data-rd-transport="none", set by index.js, behaviour in tests/js/reader/open.test.js). From 1180 px
    neither shows: the aside has the list."""
    assert decls('.rd-reading:not([data-rd-transport="none"]) .rd-chapters-btn').get("display") == "none"
    assert decls(".rd-chapters-btn", "min-width: 1180px").get("display") == "none"
    player_css = ui.CSS_ROOT / "views" / "reader-player.css"
    chip = {}
    for selector, body, chain in ui.css_rules(player_css):
        if selector.strip() == ".rdp-chip--chapters" and any("min-width: 1180px" in c for c in chain):
            chip.update(dict(body))
    assert chip.get("display") == "none"
    assert by_id("rd-chapters-btn").get("aria-label") == "Chapters", "the fallback keeps its name"


def test_the_open_error_banner_uses_the_component_layout_on_phones():
    """The banner's actions wrap under the message on phones in components.css (.banner below
    600 px), so reader.css keeps no layout rule of its own for the open-failed banner."""
    for selector, body, chain in ui.css_rules(CSS):
        if "rd-open-error" in selector:
            names = {name for name, _ in body}
            assert not names & {"flex-wrap", "flex", "flex-basis", "padding-inline-start", "order"}, selector


def test_forced_colours_keep_the_selected_paper_the_highlight_choice_and_the_paragraph_marker():
    """The forced-colours block sits in @layer utilities with system colours only.
    - The paper swatches lose their colours and their box-shadow edge: each keeps a CanvasText border, the selected
      one a Highlight ring (the forced-colours selected state) beside its check glyph, a focused one the ring further out.
    - The spoken sentence keeps the Highlight choice: base.css paints Both (Highlight fill and underline); Tint
      drops the underline, Underline the fill (a CanvasText underline on the page's colours), Off both.
    - The current paragraph keeps its marker in the margin in CanvasText, instead of base.css's 3 px border, which
      would move the text each time the paragraph changes (RE-52)."""
    forced = forced_rules()
    v = '[data-view="reader"] '
    assert forced[v + ".rd-paper__dot"].get("border") == "1px solid CanvasText"
    checked = forced[v + ".rd-paper__input:checked + .rd-paper__dot"]
    assert checked.get("outline") == "2px solid Highlight" and checked.get("outline-offset") == "2px"
    focused = forced[v + ".rd-paper__input:focus-visible + .rd-paper__dot"]
    assert focused.get("outline") == "2px solid Highlight" and focused.get("outline-offset") == "4px"
    assert forced[v + '.rd-reading[data-rd-hl="tint"] .rd-s.is-now'] == {"text-decoration": "none"}
    underline = forced[v + '.rd-reading[data-rd-hl="underline"] .rd-s.is-now']
    # base.css's HighlightText is a system colour, which forced colours keep: without the fill, CanvasText again.
    assert underline == {"forced-color-adjust": "auto", "background": "none", "color": "CanvasText"}
    off = forced[v + '.rd-reading[data-rd-hl="off"] .rd-s.is-now']
    assert off == {"forced-color-adjust": "auto", "background": "none", "color": "CanvasText", "text-decoration": "none"}
    assert not any('data-rd-hl="both"' in s for s in forced), "Both is base.css's forced highlight"
    assert forced[v + ".rd-p.is-current"].get("border-inline-start") == "0"
    marker = forced[v + ".rd-p.is-current::before"]
    assert marker.get("background") == "CanvasText" and marker.get("forced-color-adjust") == "none"
    for selector, body in forced.items():
        for name, value in body.items():
            assert not re.search(r"var\(--|#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", value), f"system colours only: {selector} {name}: {value}"
        assert not {"padding", "margin", "font-weight", "letter-spacing", "line-height"} & set(body), selector


# ───────────────────────────── the bundled Legible font ─────────────────────────────
# The Reader's "Legible" face ships Atkinson
# Hyperlegible Next as self-hosted woff2 under src/static/fonts/, with its OFL licence beside it.

FONTS = ui.STATIC / "fonts"
LEGIBLE_FAMILY = "Atkinson Hyperlegible Next"
LEGIBLE_STACK = "'Atkinson Hyperlegible Next', 'Atkinson Hyperlegible', Verdana, 'Segoe UI', sans-serif"
FONT_BUDGET = 150_000  # bytes, all font files together (Latin, plus Latin Extended only near 120 KB)
CSS_URL = re.compile(r"""url\(\s*(['"]?)(.*?)\1\s*\)""", re.S)


def legible_faces(path=ui.TOKENS_PATH) -> list[dict[str, str]]:
    """The declarations of every @font-face rule for the Legible family."""
    faces = []
    for selector, body, _ in ui.css_rules(path):
        face = dict(body)
        if selector == "@font-face" and face.get("font-family", "").strip("'\" ") == LEGIBLE_FAMILY:
            faces.append(face)
    return faces


def face_weights(face: dict[str, str]) -> range:
    """The weights one @font-face covers: a single weight, or the variable font's "low high" range."""
    parts = [int(p) for p in face.get("font-weight", "400").split()]
    return range(parts[0], parts[-1] + 1)


def css_url_problems(path) -> list[str]:
    """Every url() in a stylesheet is a relative path to a file under src/static: no scheme (a CDN, data:), no
    protocol-relative // and no root path, and the file it names exists."""
    path = Path(path)
    problems = []
    for match in CSS_URL.finditer(ui._strip_css_comments(path.read_text(encoding="utf-8"))):
        url = match.group(2).strip()
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith(("/", "\\")):
            problems.append(f"{path.name}: url({url}) is not a relative same-origin path")
            continue
        target = (path.parent / url.split("#")[0].split("?")[0]).resolve()
        if not target.is_file():
            problems.append(f"{path.name}: url({url}) names a file that does not exist")
        elif ui.STATIC.resolve() not in target.parents:
            problems.append(f"{path.name}: url({url}) points outside src/static")
    return problems


def test_the_legible_face_ships_atkinson_hyperlegible_next():
    """tokens.css declares @font-face rules for 'Atkinson Hyperlegible Next': an installed copy first
    (local()), else the bundled woff2; font-display: swap, so the text shows in the fallback while the font loads;
    and the Latin subset's unicode-range, so other scripts fall back to Verdana or Segoe UI as before. Regular (400)
    and bold (700) are both covered. The --font-legible stack is unchanged, and nothing preloads the font: a
    @font-face rule loads only when text uses the family (the Reader's Legible face, or its option label)."""
    faces = legible_faces()
    assert faces, "no @font-face for 'Atkinson Hyperlegible Next' in tokens.css"
    for face in faces:
        src = face.get("src", "")
        assert re.match(rf"""local\((['"]){LEGIBLE_FAMILY}\1\)\s*,""", src), f"an installed copy comes first: {src}"
        urls = [m.group(2) for m in CSS_URL.finditer(src)]
        assert len(urls) == 1 and re.fullmatch(r"\.\./fonts/[\w.-]+\.woff2", urls[0]), src
        assert re.search(r"""url\([^)]*\)\s*format\((['"])woff2\1\)$""", src), f"woff2 only: {src}"
        assert face.get("font-display") == "swap"
        assert face.get("font-style", "normal") == "normal"
        assert re.fullmatch(r"U\+[0-9A-Fa-f?]+(?:-[0-9A-Fa-f]+)?(?:\s*,\s*U\+[0-9A-Fa-f?]+(?:-[0-9A-Fa-f]+)?)*", face.get("unicode-range", "")), face
    latin = [f for f in faces if re.search(r"U\+0000-00FF\b", f["unicode-range"], re.I)]
    assert latin, "the Latin subset (U+0000-00FF) is bundled"
    assert any(400 in face_weights(f) for f in latin) and any(700 in face_weights(f) for f in latin), "regular and bold"
    assert ui.tokens()["--font-legible"] == LEGIBLE_STACK
    shell = ui.shell_path().read_text(encoding="utf-8")
    assert ".woff2" not in shell and not re.search(r"""rel=["']preload["'][^>]*as=["']font""", shell), "never preloaded"


def test_every_font_file_the_css_names_is_a_woff2_with_its_licence():
    """Each font file the CSS names exists, is non-empty and starts with the woff2 signature wOF2;
    the folder holds only those files and the SIL Open Font License (OFL.txt) beside them, and the fonts stay in
    the size budget."""
    named = set()
    for face in legible_faces():
        for match in CSS_URL.finditer(face["src"]):
            named.add((ui.CSS_ROOT / match.group(2)).resolve())
    assert named, "the CSS names no font file"
    total = 0
    for path in sorted(named):
        assert path.parent == FONTS.resolve() and path.is_file(), f"{path} is missing"
        data = path.read_bytes()
        assert len(data) > 1000 and data[:4] == b"wOF2", f"{path.name} is not a woff2 file"
        total += len(data)
    assert total <= FONT_BUDGET, f"{total} bytes of fonts"
    licence = FONTS / "OFL.txt"
    assert licence.is_file()
    text = licence.read_text(encoding="utf-8")
    assert re.search(r"SIL OPEN FONT LICENSE Version 1\.1", text, re.I) and "Atkinson Hyperlegible" in text
    assert {p.name for p in FONTS.iterdir()} == {p.name for p in named} | {"OFL.txt"}, "no stray files"


def test_every_css_url_is_relative_and_same_origin(tmp_path):
    """Every url() in the stylesheets is a relative path to a file under src/static (no CDN): the
    check passes the real CSS and flags a CDN font, a protocol-relative and a root path, a data: URI and a missing
    file."""
    assert [p for f in ui.css_files() for p in css_url_problems(f)] == []
    css = tmp_path / "css" / "bad.css"
    css.parent.mkdir()
    css.write_text(
        "@font-face { src: url(https://fonts.gstatic.com/s/x.woff2) format('woff2'); }\n"
        "@font-face { src: url('//cdn.example/x.woff2'); }\n"
        '@font-face { src: url("/static/fonts/x.woff2"); }\n'
        ".a { background: url(data:image/png;base64,AAAA); }\n"
        ".b { background: url('../fonts/missing.woff2'); }\n",
        encoding="utf-8",
    )
    problems = css_url_problems(css)
    assert len(problems) == 5, problems
    assert sum("not a relative same-origin path" in p for p in problems) == 4
    assert "bad.css: url(../fonts/missing.woff2) names a file that does not exist" in problems


def test_the_font_is_served_as_font_woff2_under_the_csp():
    """GET /static/fonts/<file>.woff2 answers 200 with font/woff2 and the file's bytes, and the web
    UI's CSP allows a same-origin font (font-src 'self')."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.middleware import WEB_UI_CSP
    from src.routes import web

    names = sorted({m.group(2).rsplit("/", 1)[-1] for face in legible_faces() for m in CSS_URL.finditer(face["src"])})
    assert names, "the CSS names no font file"
    app = FastAPI()
    app.mount("/static", web.RevalidatingStaticFiles(directory=str(ui.STATIC)), name="static")
    client = TestClient(app)
    for name in names:
        res = client.get(f"/static/fonts/{name}")
        assert res.status_code == 200, name
        assert res.headers["content-type"].split(";")[0] == "font/woff2"
        assert res.content == (FONTS / name).read_bytes()
    assert "font-src 'self'" in WEB_UI_CSP
