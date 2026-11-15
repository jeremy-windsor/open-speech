"""Web UI shell: the shell and component behaviour tests, the shell's (index.html) structure, and the static
routes that serve it.

Route tests use the real app through TestClient (inside pytest only); everything else reads files.
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests import ui_support as ui
from tests.ui_coverage import missing

SHELL = ui.SHELL_PATH


def shell_tree() -> ui.Node:
    return ui.parse_html(ui.shell_path())


def head(tree: ui.Node) -> ui.Node:
    return ui.select_one(tree, "head")


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("shell"))
def test_js_shell(path):
    ui.run_node_test(path)


@pytest.mark.parametrize("path", ui.js_test_files("components"))
def test_js_components(path):
    ui.run_node_test(path)


def test_f2_inventory_ids_and_defects_have_tests():
    """Every inventory id and defect owned by F2 leads a test title in its assigned file."""
    assert missing("F2") == []


# ───────────────────────────── the shell (index.html) ─────────────────────────────


def test_viewport_keeps_safe_areas_and_zoom():
    """[GS-08] viewport-fit=cover so safe-area insets apply; zoom is never locked (WCAG 1.4.4)."""
    meta = ui.select_one(head(shell_tree()), 'meta[name="viewport"]')
    content = meta.get("content")
    assert "viewport-fit=cover" in content
    assert "width=device-width" in content
    assert "maximum-scale" not in content and "user-scalable" not in content


def test_head_has_manifest_icons_and_theme_colors():
    """[BUG-28] The shell links the manifest, the SVG icon (never /favicon.ico, which is not auth-exempt), the
    apple-touch-icon, and one theme-color per colour scheme; it declares color-scheme."""
    h = head(shell_tree())
    links = {(link.get("rel"), link.get("href")) for link in ui.select(h, "link")}
    assert ("manifest", "/static/manifest.webmanifest") in links
    assert ("icon", "/static/icons/icon.svg") in links
    assert ("apple-touch-icon", "/static/icons/apple-touch-icon.png") in links
    assert not any("favicon" in (href or "") for _, href in links)
    metas = {(m.get("media"), m.get("content")) for m in ui.select(h, 'meta[name="theme-color"]')}
    assert metas == {("(prefers-color-scheme: light)", "#F7F7F4"), ("(prefers-color-scheme: dark)", "#0E0F13")}
    assert ui.select_one(h, 'meta[name="color-scheme"]').get("content") == "light dark"
    assert ui.select_one(h, 'meta[name="apple-mobile-web-app-title"]').get("content") == "Open Speech"
    assert ui.select_one(shell_tree(), "html").get("lang") == "en"


def test_theme_boot_runs_blocking_before_the_stylesheets_and_main_is_the_only_module():
    """theme-boot.js is a classic script ahead of every stylesheet (no flash of the wrong theme); main.js is the
    one module script; the stylesheets are tokens, base, components, shell, then all eleven view sheets (Talk's
    included)."""
    h = head(shell_tree())
    order = [(el.tag, el.get("src") or el.get("href"), el.get("rel"), el.get("type")) for el in h.elements() if el.tag in ("script", "link") and el.get("rel") != "modulepreload"]
    scripts = [o for o in order if o[0] == "script"]
    assert scripts == [("script", "/static/js/theme-boot.js", None, None), ("script", "/static/js/main.js", None, "module")]
    sheets = [o[1] for o in order if o[2] == "stylesheet"]
    assert order.index(scripts[0]) < order.index(("link", sheets[0], "stylesheet", None))
    views = ["reader", "reader-player", "speak", "speak-live", "transcribe", "talk", "voices", "studio", "history", "models", "settings"]
    assert sheets == [f"/static/css/{n}.css" for n in ("tokens", "base", "components", "shell")] + [f"/static/css/views/{v}.css" for v in views]


def test_modulepreload_list_equals_the_static_import_closure_of_main():
    """[REV-37] Every module in main.js's static import closure is modulepreloaded, and nothing else (views
    are imported lazily), so a no-cache revalidation never turns into a request waterfall."""
    preload = {link.get("href") for link in ui.select(head(shell_tree()), 'link[rel="modulepreload"]')}
    closure = {"/static/js/" + p.relative_to(ui.JS_ROOT).as_posix() for p in ui.static_import_closure(ui.JS_ROOT / "main.js")}
    assert closure, "main.js imports the shell"
    assert preload == closure, (sorted(closure - preload), sorted(preload - closure))
    assert not any("/views/" in href for href in preload)


def test_shell_landmarks_navs_and_view_sections():
    """One banner (top bar), two <nav aria-label="Main"> (only one visible at a time), one <main
    tabindex=-1> with the connection banner first and one hidden section per view, the mini-player region,
    the skip link first. Talk is the ninth view: in the rail's Capture group after
    Transcribe. On narrow screens the bottom bar holds Reader, Speak, Transcribe, Talk and More; Voices, Studio,
    History, Models and Settings are in the More sheet, each with a one-line description."""
    tree = shell_tree()
    body = ui.select_one(tree, "body")
    first_focusable = next(el for el in body.elements() if el.tag in ("a", "button", "input", "select", "textarea"))
    assert first_focusable.classes == ["skip-link"] and first_focusable.get("href") == "#main"
    assert [h.id for h in ui.select(tree, "header")] == ["app-topbar"]
    assert [(n.id, n.get("aria-label")) for n in ui.select(tree, "nav")] == [("app-rail", "Main"), ("app-bottombar", "Main")]
    mains = ui.select(tree, "main")
    assert len(mains) == 1 and mains[0].id == "main" and mains[0].get("tabindex") == "-1"
    children = [el for el in mains[0].children if el.is_element]
    assert children[0].id == "app-conn-banner" and children[0].get("role") == "status"
    sections = ui.select(tree, "main > section.view")
    assert [s.id for s in sections] == [f"view-{v}" for v in ("reader", "speak", "transcribe", "talk", "voices", "studio", "history", "models", "settings")]
    assert all(s.has("hidden") and s.get("data-view") == s.id[5:] for s in sections)
    mp = ui.by_id(tree, "app-miniplayer")
    assert mp.get("role") == "region" and mp.get("aria-label") == "Now playing" and mp.has("hidden")
    assert [b.id for b in ui.select(mp, "button")] == ["mp-open", "mp-toggle", "mp-skip", "mp-close"]
    bottom = [a.get("data-view") or a.id for a in ui.select(tree, "#app-bottombar .nav-link")]
    assert bottom == ["reader", "speak", "transcribe", "talk", "nav-more"]
    assert [ui.visible_text(a) for a in ui.select(tree, "#app-bottombar .nav-link__label")] == ["Reader", "Speak", "Transcribe", "Talk", "More"]
    more = ui.by_id(tree, "nav-more")
    assert more.get("aria-haspopup") == "dialog" and more.get("aria-controls") == "app-more-sheet"
    rail = [a.get("data-view") for a in ui.select(tree, "#app-rail .nav-link")]
    assert rail == ["reader", "speak", "transcribe", "talk", "voices", "studio", "history", "models", "settings"]
    more_links = ui.select(tree, "#app-more-sheet .more-link")
    assert [a.get("data-view") for a in more_links] == ["voices", "studio", "history", "models", "settings"]
    assert all(a.get("href") == f"#{a.get('data-view')}" for a in more_links)
    metas = [ui.visible_text(m) for m in ui.select(tree, "#app-more-sheet .more-link__meta")]
    assert len(metas) == len(more_links) and all(metas)
    assert metas[0] == "Named voices, presets and recordings"
    assert [p.text_content() for p in ui.select(tree, "#app-rail .rail__group")] == ["Listen", "Capture", "Create", "Manage"]
    assert all(p.get("aria-hidden") == "true" for p in ui.select(tree, "#app-rail .rail__group"))


def test_shell_dialogs_regions_and_audio_host():
    """The shell sheets are <dialog class="sheet"> with an h2 title; #app-confirm and #app-prompt carry the ids
    core/dialogs.js uses; the toast, screen-reader and tooltip regions and the shared <audio> exist."""
    tree = shell_tree()
    for dialog_id in ("app-more-sheet", "app-status-sheet", "app-shortcuts-sheet", "app-voice-picker", "app-ask-ai", "app-confirm", "app-prompt"):
        dialog = ui.by_id(tree, dialog_id)
        assert dialog.tag == "dialog" and "sheet" in dialog.classes
        title = ui.by_id(tree, dialog.get("aria-labelledby"))
        assert title is not None and title.tag == "h2"
    for id_ in ("app-confirm-title", "app-confirm-msg", "app-confirm-cancel", "app-confirm-ok", "app-prompt-form", "app-prompt-title",
                "app-prompt-label", "app-prompt-input", "app-prompt-hint", "app-prompt-error", "app-prompt-cancel", "app-prompt-ok"):
        assert ui.by_id(tree, id_) is not None, id_
    assert ui.by_id(tree, "app-prompt-ok").get("type") == "submit"
    assert ui.by_id(tree, "app-confirm").get("aria-describedby") == "app-confirm-msg"
    assert "sheet--wide" in ui.by_id(tree, "app-voice-picker").classes and "sheet--alert" in ui.by_id(tree, "app-confirm").classes
    search = ui.by_id(tree, "vp-search")
    assert (search.get("role"), search.get("aria-expanded"), search.get("aria-controls"), search.get("aria-autocomplete")) == ("combobox", "true", "vp-list", "list")
    assert ui.by_id(tree, "vp-list").get("role") == "listbox"
    assert {b.id for b in ui.select(tree, "#app-voice-picker .sheet__foot button")} == {"vp-preview", "vp-fav", "vp-use"}
    toasts = ui.by_id(tree, "app-toasts")
    assert [(c.id, c.get("role")) for c in toasts.children if c.is_element] == [("app-toasts-polite", "status"), ("app-toasts-alert", "alert")]
    assert ui.by_id(tree, "app-sr-status").get("role") == "status" and ui.by_id(tree, "app-sr-alert").get("role") == "alert"
    assert ui.by_id(tree, "app-tooltip").get("role") == "tooltip"
    audio = ui.by_id(tree, "app-audio")
    assert audio.tag == "audio" and audio.get("preload") == "auto" and audio.parent.id == "app-audio-host"


def test_the_voice_picker_has_a_note_for_a_model_with_no_voices_of_its_own():
    """With a cloning model (Chatterbox), nothing in Speak's voice picker said that the
    model has no built-in voices. The picker's body holds a note above the list, hidden until
    components/voice-picker.js fills it for such a model, with a real link to Voices, Clone (a trailing "…": a
    dialog follows). It is a link in the body, so the footer keeps its three buttons."""
    tree = shell_tree()
    note = ui.by_id(tree, "vp-model-note")
    assert note is not None and note.tag == "p" and note.has("hidden") and "vp__model-note" in note.classes
    body = note.parent
    assert "sheet__body" in body.classes and ui.by_id(tree, "app-voice-picker") in list(note.ancestors())
    order = [c.id for c in body.children if c.is_element]
    assert order.index("vp-hint") < order.index("vp-model-note") < order.index("vp-list")
    text = ui.by_id(tree, "vp-model-note-text")
    assert text.parent is note and text.text_content() == ""
    link = ui.by_id(tree, "vp-clone")
    assert link.parent is note and link.tag == "a"
    assert (link.get("href"), link.text_content()) == ("#voices/clone", "Clone a voice…")
    assert {b.id for b in ui.select(tree, "#app-voice-picker .sheet__foot button")} == {"vp-preview", "vp-fav", "vp-use"}
    rules = {sel for sel, _decls, _chain in ui.css_rules(ui.CSS_ROOT / "shell.css")}
    assert ".vp__model-note" in rules


def test_the_voice_picker_packs_a_short_list_at_the_top():
    """The picker's list (#vp-list, a .listbox grid) fills the sheet's height as a flex
    item, and a grid's align-content (normal, which acts as stretch) shared the free height among its rows: three
    rows were about 97 px high instead of 56 to 62, and a group's label floated above them. The listbox packs its rows
    at the start, whatever its height."""
    tree = shell_tree()
    vp_list = ui.by_id(tree, "vp-list")
    assert {"listbox", "vp__list"} <= set(vp_list.classes)
    listbox = {
        name: value.strip()
        for sel, decls, chain in ui.css_rules(ui.CSS_ROOT / "components.css")
        if all(c.startswith("@layer") for c in chain) and ".listbox" in [s.strip() for s in sel.split(",")]
        for name, value in decls
    }
    assert listbox.get("display") == "grid"
    assert listbox.get("align-content") in {"start", "flex-start"}, "a short list keeps its rows' own height"
    fills = [
        decls for sel, decls, _chain in ui.css_rules(ui.CSS_ROOT / "shell.css")
        if ".vp__list" in sel and any(n == "flex" and v.strip().startswith("1") for n, v in decls)
    ]
    assert fills, "the list still fills the sheet (it scrolls inside it)"


def test_the_ask_ai_sheet_is_one_shell_dialog_every_view_opens():
    """The shared Ask AI sheet is the shell dialog #app-ask-ai (components/ask-ai.js
    fills it): a form whose Send submits; what is sent, the preset, the three outputs (Show, Speak in a voice,
    Save as a document) and the privacy line; a status line for the wait and a job's progress; the answer as a
    keyboard-scrollable region with the thinking collapsed, Play, Save, Open in the Reader and Copy; an alert for
    errors. Transcribe's and the Reader's buttons open it (aria-controls), hidden until the assistant is on."""
    tree = shell_tree()
    dialog = ui.by_id(tree, "app-ask-ai")
    assert "sheet--wide" in dialog.classes and dialog.get("aria-labelledby") == "ask-title"
    form = ui.by_id(tree, "ask-form")
    assert form.parent is dialog and form.tag == "form"
    assert ui.by_id(tree, "ask-send").get("type") == "submit"
    assert {b.id for b in ui.select(tree, "#app-ask-ai .sheet__foot button")} == {"ask-cancel", "ask-again", "ask-send"}
    for id_ in ("ask-out-show", "ask-out-speak", "ask-out-save"):
        assert (ui.by_id(tree, id_).get("type"), ui.by_id(tree, id_).get("role")) == ("checkbox", "switch"), id_
    assert ui.by_id(tree, "ask-preset").tag == "select"
    assert ui.by_id(tree, "ask-privacy-text") is not None
    assert ui.by_id(tree, "ask-status").get("role") == "status"
    assert ui.by_id(tree, "ask-error").get("role") == "alert" and ui.by_id(tree, "ask-error").has("hidden")
    reply = ui.by_id(tree, "ask-reply")
    assert (reply.get("tabindex"), reply.get("role"), reply.get("aria-labelledby")) == ("0", "region", "ask-result-title")
    assert ui.by_id(tree, "ask-thinking").tag == "details" and ui.by_id(tree, "ask-thinking").has("hidden")
    assert {b.id for b in ui.select(tree, "#ask-actions button")} == {"ask-play", "ask-save", "ask-open", "ask-copy"}
    assert ui.by_id(tree, "ask-progress").tag == "progress"
    for name, button_id in (("transcribe", "tr-ask"), ("reader", "rd-ask")):
        button = ui.by_id(ui.parse_html(ui.PARTIALS_DIR / f"{name}.html"), button_id)
        assert (button.get("aria-haspopup"), button.get("aria-controls")) == ("dialog", "app-ask-ai"), button_id
        assert button.has("hidden"), f"{button_id} shows only once the assistant says it is on"
    rules = {sel for sel, _decls, _chain in ui.css_rules(ui.CSS_ROOT / "shell.css")}
    assert ".ask__reply" in rules and ".ask .sheet__body" in rules


def test_view_titles_share_one_baseline_and_loading_states_stay_visible():
    """The view h1 sat 39 to 49 px down depending on the header's actions, so it jumped
    on navigation. Every title is centred in a row one control high, so it lands at the same height with or
    without 44 px actions or a segmented control. A still spinner (reduced motion) is a faint full ring with an
    arc, never a bare "C", and skeletons use --surface-3, which shows on the light page (--surface-2 did not)."""
    rules = ui.css_rules(ui.CSS_ROOT / "shell.css")
    decl = lambda selector: {n: v for sel, decls, chain in rules if sel == selector and not chain[1:] for n, v in decls}  # noqa: E731
    assert decl(".view-head")["align-items"] == "center"
    title = decl(".view-head__title")
    assert (title["display"], title["align-items"], title["min-height"]) == ("flex", "center", "var(--control-h)")
    assert ui.tokens("light")["--control-h"] == "44px"
    # No view moves its title off that line: its header class (sp-head, st-head, …) keeps the centring.
    heads = {"view-head"}
    for name in ui.VIEW_PARTIALS:
        for header in ui.select(ui.parse_html(ui.PARTIALS_DIR / f"{name}.html"), "header.view-head"):
            heads.update(header.classes)
    head_re = re.compile(r"\.(?:%s)(?![\w-])" % "|".join(sorted(map(re.escape, heads))))
    for path in ui.css_files():
        for selector, decls, _ in ui.css_rules(path):
            if head_re.search(selector) and "align-items" in dict(decls):
                assert dict(decls)["align-items"] == "center", (path.name, selector)
    comp = {sel: dict(decls) for sel, decls, _ in ui.css_rules(ui.CSS_ROOT / "components.css")}
    spinner = comp[".spinner"]
    assert "transparent" not in spinner.get("border-right-color", "") and "color-mix(" in spinner["border"], spinner
    assert spinner["border-top-color"] == "currentColor"
    assert comp[".skeleton"]["background"] == "var(--surface-3)"


def test_the_status_and_shortcuts_sheet_bodies_scroll_from_the_keyboard():
    """The Status and Shortcuts sheets hold no control in their bodies, which overflow at 200 % zoom or in a short
    window (455 px of content in 315 px at 640x450). Safari doesn't make a scroll area
    focusable, so each body is a named, focusable region the keyboard can scroll (WCAG 2.1.1)."""
    tree = shell_tree()
    for body_id, title_id in (("app-status-body", "app-status-title"), ("app-shortcuts-body", "app-shortcuts-title")):
        body = ui.by_id(tree, body_id)
        assert (body.get("tabindex"), body.get("role"), body.get("aria-labelledby")) == ("0", "region", title_id), body_id
        assert ui.by_id(tree, title_id).tag == "h2"
    rules = ui.css_rules(ui.CSS_ROOT / "shell.css")
    assert any(sel == '.sheet__body[tabindex="0"]:focus-visible' and dict(decls).get("outline-offset") == "-2px" for sel, decls, _ in rules)


def test_scroll_padding_follows_the_chrome_but_not_while_focus_is_inside_it():
    """[REV-21] Scroll padding is the measured chrome, so a focused element is never scrolled in behind a bar or
    the toast stack. While focus is inside chrome (a bar, the rail, the docked mini-player, the Reader's bars, the
    toasts) there is none: a stuck sticky bar always sits inside the padding, so each Tab within it scrolled the page
    by the padding again (about 380 px per Tab in the Reader's transport, the rail footer
    and the mini-player). The sticky and fixed chrome carries data-chrome (or is the rail) so the rule reaches it."""
    rules = ui.css_rules(ui.CSS_ROOT / "base.css")
    html = {name: value for selector, decls, _ in rules if selector == "html" for name, value in decls}
    assert "var(--chrome-top" in html["scroll-padding-top"] and "var(--chrome-bottom," in html["scroll-padding-bottom"]
    zero = [(selector, dict(decls)) for selector, decls, _ in rules if selector.startswith("html:has(")]
    assert len(zero) == 1, zero
    selector, decls = zero[0]
    assert decls.get("scroll-padding-block") == "0", decls
    inner = re.fullmatch(r"html:has\(:is\(([^)]*)\):focus-within\)", selector)
    assert inner, selector
    assert {part.strip() for part in inner.group(1).split(",")} == {"[data-chrome]", ".rail", ".toasts"}
    tree = shell_tree()
    assert [ui.by_id(tree, i).get("data-chrome") for i in ("app-topbar", "app-miniplayer", "app-bottombar")] == ["top", "bottom", "bottom"]
    assert "rail" in ui.by_id(tree, "app-rail").classes and "toasts" in ui.by_id(tree, "app-toasts").classes
    reader = ui.parse_html(ui.PARTIALS_DIR / "reader.html")
    player = ui.parse_html(ui.PARTIALS_DIR / "reader-player.html")
    assert ui.select_one(reader, ".rd-topbar").get("data-chrome") == "top"
    assert ui.select_one(player, ".rdp-transport").get("data-chrome") == "bottom"


def test_manifest_fields_and_icons():
    """[BUG-28] The web manifest: name, short_name, start_url and scope /web, standalone, background and theme
    colours, icons at 192 and 512 plus a maskable 512; every icon file exists with its stated size."""
    manifest = json.loads((ui.STATIC / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["name"] == "Open Speech" and manifest["short_name"] == "Open Speech"
    assert manifest["start_url"] == "/web" and manifest["scope"] == "/web"
    assert manifest["display"] == "standalone"
    assert manifest["background_color"] == "#F7F7F4" and manifest["theme_color"]
    icons = {(i["src"], i["sizes"], i.get("purpose", "any")) for i in manifest["icons"]}
    assert ("/static/icons/icon-192.png", "192x192", "any") in icons
    assert ("/static/icons/icon-512.png", "512x512", "any") in icons
    assert ("/static/icons/icon-maskable-512.png", "512x512", "maskable") in icons
    for src, sizes, _ in icons:
        path = ui.STATIC / src.removeprefix("/static/")
        assert path.exists(), src
        if src.endswith(".png"):
            data = path.read_bytes()
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
            assert f"{width}x{height}" == sizes, src
    touch = (ui.STATIC / "icons" / "apple-touch-icon.png").read_bytes()
    assert (int.from_bytes(touch[16:20], "big"), int.from_bytes(touch[20:24], "big")) == (180, 180)
    assert touch[25] in (2, 0), "the apple-touch-icon is opaque (RGB or greyscale, no alpha channel)"


def test_sprite_matches_icons_js():
    """The sprite has exactly the symbols core/icons.js lists, each on a 24px grid."""
    text = ui.SPRITE_PATH.read_text(encoding="utf-8")
    ids = set(re.findall(r'<symbol id="i-([\w-]+)" viewBox="0 0 24 24">', text))
    icons_js = (ui.JS_ROOT / "core" / "icons.js").read_text(encoding="utf-8")
    block = re.sub(r"//[^\n]*", "", re.search(r"ICONS\s*=\s*Object\.freeze\(\[(.*?)\]\)", icons_js, re.S).group(1))
    assert ids == set(re.findall(r"'([\w-]+)'", block))
    assert len(ui.sprite_ids()) == len(ids)


# ───────────────────────────── routes ─────────────────────────────


@pytest.fixture(scope="module")
def client():
    from src.main import app

    return TestClient(app)


def _assets(tree: ui.Node) -> list[str]:
    out = []
    for el in tree.elements():
        for attr in ("href", "src"):
            value = el.get(attr) or ""
            if value.startswith("/static/") and el.tag != "use":
                out.append(value)
    return sorted(set(out))


def _partials() -> list[str]:
    paths = set()
    for index in sorted((ui.JS_ROOT / "views").glob("*/index.js")):
        paths.update(re.findall(r"partial:\s*'(/static/partials/[\w-]+\.html)'", index.read_text(encoding="utf-8")))
    paths.update(f"/static/partials/{name}.html" for name in ui.SLOT_PARTIALS)
    return sorted(paths)


TYPES = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json",
         ".png": "image/png", ".html": "text/html"}


def test_the_shell_is_served_revalidated_with_the_web_ui_csp(client):
    """[GS-01] /static/index.html: 200, text/html, Cache-Control: no-cache (an ETag revalidates to 304), and the
    web-UI CSP; /web serves the same shell from disk (it replaced the old UI's shell)."""
    from src.middleware import WEB_UI_CSP

    page = client.get("/static/index.html")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["cache-control"] == "no-cache"
    assert page.headers["content-security-policy"] == WEB_UI_CSP
    again = client.get("/static/index.html", headers={"If-None-Match": page.headers["etag"]})
    assert again.status_code == 304 and again.headers["cache-control"] == "no-cache"
    web = client.get("/web")
    assert web.status_code == 200 and web.headers["cache-control"] == "no-cache"
    assert web.headers["content-security-policy"] == WEB_UI_CSP
    assert web.text == ui.shell_path().read_text(encoding="utf-8"), "/web serves the shell"
    assert '<script type="module" src="/static/js/main.js"></script>' in web.text


def test_every_shell_asset_partial_and_the_sprite_are_served_with_their_types(client):
    """[GS-01] Every href/src in the shell, every view and slot partial, and the sprite answer 200 with the
    right content type (.js as text/javascript whatever the host's MIME table says) and revalidate."""
    paths = _assets(shell_tree()) + _partials() + ["/static/icons/sprite.svg", "/static/js/main.js"]
    assert len(paths) > 50
    for path in paths:
        res = client.get(path)
        assert res.status_code == 200, path
        suffix = "." + path.rsplit(".", 1)[-1]
        assert res.headers["content-type"].split(";")[0] == TYPES[suffix], (path, res.headers["content-type"])
        assert res.headers["cache-control"] == "no-cache", path


def test_the_shell_and_its_assets_are_auth_exempt_but_the_api_is_not(client):
    """[GS-02] With OS_API_KEY set, /static/* (the shell, modules, partials, sprite, manifest, icons), /web and
    /health answer without a key; the API needs it."""
    import src.middleware as mw
    from src.config import Settings

    settings = Settings(_env_file=None, os_api_key="shell-secret")
    mw.reset_security_state_for_tests()
    try:
        with patch("src.middleware.settings", settings):
            for path in ("/static/index.html", "/static/js/main.js", "/static/js/shell/index.js", "/static/partials/speak.html",
                         "/static/icons/sprite.svg", "/static/icons/icon.svg", "/static/manifest.webmanifest",
                         "/static/icons/apple-touch-icon.png", "/web", "/health"):
                assert client.get(path).status_code == 200, path
            assert client.get("/api/models").status_code == 401
            assert client.get("/api/models", headers={"Authorization": "Bearer shell-secret"}).status_code == 200
    finally:
        mw.reset_security_state_for_tests()
