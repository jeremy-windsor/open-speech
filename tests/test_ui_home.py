"""Home: the Node behaviour tests of the view, plus the partial's structure and the stylesheet's intent.
The global audits in test_ui_core.py also run over home.html and home.css."""

from __future__ import annotations

import pytest

from tests import ui_support as ui

PARTIAL = ui.PARTIALS_DIR / "home.html"
CSS = ui.CSS_ROOT / "views" / "home.css"


@pytest.fixture(scope="module")
def tree():
    return ui.parse_html(PARTIAL)


def decls(selector, chain_has=""):
    out = {}
    for sel, body, chain in ui.css_rules(CSS):
        conditional = [c for c in chain if c.startswith(("@media", "@supports"))]
        wanted = any(chain_has in c for c in conditional) if chain_has else not conditional
        if sel.strip() == selector and wanted:
            out.update(dict(body))
    return out


@pytest.mark.parametrize("path", ui.js_test_files("home"))
def test_js(path):
    ui.run_node_test(path)


def test_home_partial_has_its_heading_shortcuts_and_two_lists(tree):
    """One focusable h1 (the greeting); the four shortcuts are plain links in the tab bar's order; the two lists
    are labelled sections that start busy; the status button names itself and opens the status sheet."""
    headings = ui.select(tree, "h1")
    assert [h.id for h in headings] == ["hm-title"]
    assert headings[0].has("data-view-heading") and headings[0].get("tabindex") == "-1"
    tiles = ui.select(tree, ".hm-tiles a.hm-tile")
    assert [(a.get("href"), a.get("data-go")) for a in tiles] == [(f"#{v}", v) for v in ("reader", "speak", "transcribe", "talk")]
    assert [ui.visible_text(t) for t in ui.select(tree, ".hm-tile__title")] == ["Reader", "Speak", "Transcribe", "Talk"]
    for section in ui.select(tree, "section"):
        assert ui.by_id(tree, section.get("aria-labelledby")) is not None
    for host in ("hm-continue", "hm-recent"):
        assert ui.by_id(tree, host).get("aria-busy") == "true"
    status = ui.by_id(tree, "hm-status")
    assert status.get("aria-controls") == "app-status-sheet" and status.get("aria-label").endswith("Open status.")
    assert ui.by_id(tree, "hm-all").get("href") == "#history"
    wave = ui.select(tree, ".hm-wave")[0]
    assert wave.get("aria-hidden") == "true" and len(ui.select(wave, ".hm-wave__bar")) >= 16


def test_home_css_hero_motion_and_phone_layout():
    """The hero is the accent gradient with on-accent text; its bars are the view's one loop, moved by transform
    only; the shortcuts are two columns on phones (their descriptions hidden there) and four from 720 px; entrance
    animations fill backwards only, so no transform stays applied."""
    hero = decls('[data-view="home"] .hm-hero')
    assert "var(--accent-fill)" in hero["background"] and "var(--accent-fill-2)" in hero["background"]
    assert hero["color"] == "var(--on-accent)"
    bar = decls('[data-view="home"] .hm-wave__bar')
    assert "hm-wave" in bar["animation"] and "infinite" in bar["animation"]
    frames = {sel: dict(body) for sel, body, _ in ui.css_rules(CSS) if sel.startswith("@keyframes")}
    assert set(frames) == {"@keyframes hm-wave", "@keyframes hm-rise"}
    assert "repeat(2" in decls('[data-view="home"] .hm-tiles')["grid-template-columns"]
    assert "repeat(4" in decls('[data-view="home"] .hm-tiles', "min-width: 720px")["grid-template-columns"]
    assert decls('[data-view="home"] .hm-tile__meta', "max-width: 599px")["display"] == "none"
    for sel, body, _ in ui.css_rules(CSS):
        value = dict(body).get("animation", "")
        if "hm-rise" in value:
            assert "backwards" in value and "both" not in value and "forwards" not in value, sel
