"""Transcribe (V4): the Node behaviour tests, the partial's structure and the view CSS.

The behaviour (the recording state machine and its socket protocol, the transcript reducer and the role=log,
uploads with busy and Cancel, exports and copy, the screenshot hook) is tested in tests/js/transcribe/*.test.js;
this file runs them and checks what only the static files can show.
"""

from __future__ import annotations

import json
import re

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

PARTIAL = ui.PARTIALS_DIR / "transcribe.html"
CSS = ui.CSS_ROOT / "views" / "transcribe.css"
JS_DIR = ui.JS_ROOT / "views" / "transcribe"


def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


def node(id_: str) -> ui.Node:
    el = ui.by_id(tree(), id_)
    assert el is not None, f"#{id_} is missing from transcribe.html"
    return el


def decls(selector: str, chain_has: str = "") -> dict[str, str]:
    """The declarations of one exact selector in transcribe.css (outside @media, @supports and @container
    unless chain_has names the condition)."""
    out: dict[str, str] = {}
    for sel, body, chain in ui.css_rules(CSS):
        conditional = [c for c in chain if c.startswith(("@media", "@supports", "@container"))]
        wanted = any(chain_has in c for c in conditional) if chain_has else not conditional
        if wanted and any(part.strip() == selector for part in sel.split(",")):
            out.update(dict(body))
    return out


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("transcribe"))
def test_js(path):
    ui.run_node_test(path, timeout=240)


def test_every_v4_inventory_id_and_defect_has_a_test():
    """Every id and defect V4 owns appears as [ID] in a test title that runs, in its assigned file."""
    assert missing("V4") == []


_VALUE_ASSERT = re.compile(
    r"\bassert\.(?:equal|notEqual|strictEqual|notStrictEqual|deepEqual|deepStrictEqual|notDeepEqual|notDeepStrictEqual)\("
)
# An argument that is a harness DOM node: it ends in a call or property that returns one.
_NODE_ARG = re.compile(
    r"(?:querySelector|getElementById|closest|\$)\([^()]*\)\s*$"
    r"|(?:activeElement|parentNode|parentElement|firstElementChild|lastElementChild|firstChild|lastChild"
    r"|nextElementSibling|previousElementSibling|\.body|\.documentElement)\s*$"
)


def _call_args(src: str, start: int) -> list[str]:
    """The top-level arguments of the call whose '(' ends just before `start` (strings and brackets skipped)."""
    depth, i, cur, out, quote = 1, start, [], [], None
    while i < len(src) and depth:
        c = src[i]
        if quote:
            cur.append(c)
            if c == "\\":
                cur.append(src[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "'\"`":
            quote = c
            cur.append(c)
        elif c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            if depth:
                cur.append(c)
        elif c == "," and depth == 1:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur).strip())
    return out


def _dom_node_asserts(src: str) -> list[str]:
    found = []
    for m in _VALUE_ASSERT.finditer(src):
        args = _call_args(src, m.end())[:2]
        if any(_NODE_ARG.search(a) for a in args):
            found.append(f"line {src.count(chr(10), 0, m.start()) + 1}: {m.group(0)}{', '.join(args)}")
    return found


def test_node_tests_compare_dom_nodes_with_the_node_helpers_never_assert_equal():
    """A failing assert.equal/deepEqual whose actual or expected is a harness DOM node makes node:assert inspect
    it (depth 1000, getters on). Through the node's enumerable links that walks the whole fake document, window
    and env: one such failure went past the 2 GB cap with RangeError "Array buffer allocation failed" (in
    mutation runs). The Transcribe tests compare nodes with sameNode()/noNode() from helpers.js instead."""
    planted = (
        "assert.equal(t.doc.activeElement, t.$('tr-record'), 'x');\n"
        "assert.equal(banner.querySelector('details'), null);\n"
        "assert.notEqual(el.closest('.card'), null);\n"
        "assert.equal(banner.querySelector('details pre').textContent, 'ok');\n"
        "sameNode(t.doc.activeElement, t.$('tr-record'));\n"
    )
    assert [line.split(":")[0] for line in _dom_node_asserts(planted)] == ["line 1", "line 2", "line 3"]
    offenders = {}
    for path in sorted((ui.REPO / "tests" / "js" / "transcribe").glob("*.js")):
        found = _dom_node_asserts(path.read_text(encoding="utf-8"))
        if found:
            offenders[path.name] = found
    assert offenders == {}


# ───────────────────────────── the partial ─────────────────────────────


def test_the_partial_carries_every_element_id_from_the_spec():
    """The ids other code, tests and docs rely on."""
    ids = {el.id for el in tree().elements() if el.id}
    for id_ in (
        "tr-title", "tr-record", "tr-cancel", "tr-timer", "tr-vad", "tr-meter", "tr-meter-label", "tr-mic", "tr-model",
        "tr-mic-help", "tr-dropzone", "tr-file", "tr-upload-status", "tr-upload-cancel", "tr-log", "tr-interim", "tr-meta",
        "tr-copy", "tr-export", "tr-clear", "tr-show-interim", "tr-show-ts", "tr-nomodel",
    ):
        assert id_ in ids, id_
    assert not ui.select(tree(), "[data-stub-body], [data-stub]"), "the placeholder marker is gone"


def test_the_view_heading_and_the_live_regions():
    """View contract and live regions: one focusable h1; #tr-log is role=log aria-live=polite and #tr-interim sits outside it with
    aria-live=off; #tr-upload-status is role=status; the error banners are alerts and the mic help a status."""
    t = tree()
    h1s = ui.select(t, "h1")
    assert len(h1s) == 1 and h1s[0].id == "tr-title" and h1s[0].get("tabindex") == "-1" and h1s[0].has("data-view-heading")
    log = node("tr-log")
    assert log.get("role") == "log" and log.get("aria-live") == "polite"
    interim = node("tr-interim")
    assert interim.get("aria-live") == "off"
    assert "tr-log" not in [a.id for a in interim.ancestors()]
    assert not [el for el in log.elements() if el.is_element], "the log starts empty: only p.tr-seg go in"
    assert node("tr-upload-status").get("role") == "status"
    assert node("tr-rec-error").get("role") == "alert" and node("tr-upload-error").get("role") == "alert"
    assert node("tr-mic-help").get("role") == "status"


def test_the_capture_column_banners_put_their_actions_under_the_message():
    """The three banners sit in the narrow capture column, so they use the component's
    .banner--stacked: Retry / Open Settings go on a line of their own under the message at every width, and Dismiss
    stays on the title's line. The view sheet no longer re-implements that layout (the old .tr-banner rules)."""
    for id_ in ("tr-mic-help", "tr-rec-error", "tr-upload-error"):
        classes = node(id_).classes
        assert "banner" in classes and "banner--stacked" in classes, (id_, classes)
        assert "tr-banner" not in classes, id_
    for selector, _body, _chain in ui.css_rules(CSS):
        assert "tr-banner" not in selector and "banner__actions" not in selector, selector


def test_the_nothing_heard_banner_is_a_status_under_the_level_meter():
    """A recording that hears nothing for a few seconds says so in its own banner,
    #tr-heard: a polite role=status (it is advice, not an error), stacked like the other capture banners, and
    placed after the record button and the level meter it is about, so it never pushes Stop down under the
    pointer while recording."""
    heard = node("tr-heard")
    assert heard.get("role") == "status" and heard.has("hidden")
    assert {"banner", "banner--warning", "banner--stacked"} <= set(heard.classes), heard.classes
    order = [el.id for el in tree().elements() if el.id in ("tr-record", "tr-meter", "tr-heard", "tr-cancel")]
    assert order == ["tr-record", "tr-meter", "tr-heard", "tr-cancel"], order


def test_counters_never_tick_in_a_live_region():
    """[REV-10] The recording timer and the upload's elapsed time are aria-hidden, outside every live region."""
    t = tree()
    live = [el for el in t.elements() if el.get("role") in ("status", "alert", "log") or el.get("aria-live") in ("polite", "assertive")]
    for id_ in ("tr-timer", "tr-upload-elapsed"):
        counter = ui.by_id(t, id_)
        assert ui.is_aria_hidden(counter), id_
        assert not any(counter in list(region.elements()) for region in live), f"#{id_} sits inside a live region"


def test_the_record_button_has_a_name_and_no_pressed_state():
    """[REV-12] #tr-record starts as a button named "Record", described by the hint, with no aria-pressed
    (that would read "Stop recording, pressed"). That it never gains one while its name swaps is the [REV-12]
    test in tests/js/transcribe/recorder.test.js, which drives a whole session."""
    t = tree()
    record = node("tr-record")
    assert record.tag == "button" and record.get("type") == "button"
    assert ui.accessible_name(record, t) == "Record"
    assert record.get("aria-describedby") == "tr-rec-hint" and ui.by_id(t, "tr-rec-hint") is not None
    assert not record.has("aria-pressed")


def test_the_dropzone_is_a_label_for_a_visually_hidden_audio_input():
    """AX-05: label#tr-dropzone[for=tr-file]; the file input accepts audio and is visually hidden, not
    hidden; Cancel is a sibling button, never inside the label."""
    t = tree()
    zone = node("tr-dropzone")
    assert zone.tag == "label" and zone.get("for") == "tr-file"
    file_input = node("tr-file")
    assert file_input.get("type") == "file" and file_input.get("accept") == "audio/*"
    assert "visually-hidden" in file_input.classes and not file_input.has("hidden")
    assert not [el for el in zone.elements() if el.tag in ("button", "a", "input", "select")]
    assert ui.accessible_name(file_input, t).startswith("Upload audio")
    assert "tr-dropzone" not in [a.id for a in node("tr-upload-cancel").ancestors()]


def test_export_menu_switches_and_actions():
    """BUG-17, TR-18: Copy and Export start disabled; Export is an APG menu button with TXT, SRT,
    VTT and JSON (no format select); the two display switches are role=switch with visible labels."""
    t = tree()
    assert node("tr-copy").has("disabled") and node("tr-export").has("disabled") and node("tr-clear").has("disabled")
    export = node("tr-export")
    assert export.get("aria-haspopup") == "menu" and export.get("aria-controls") == "tr-export-menu"
    menu = node("tr-export-menu")
    assert menu.get("role") == "menu" and menu.has("hidden")
    assert [b.get("data-format") for b in ui.select(menu, "[role=menuitem]")] == ["txt", "srt", "vtt", "json"]
    assert "tr-panel" not in [a.id for a in menu.ancestors()], "outside the card: a .card is a containing block for fixed menus"
    assert {s.id for s in ui.select(t, "select")} == {"tr-mic", "tr-model"}
    for id_, label in (("tr-show-interim", "Interim text"), ("tr-show-ts", "Timestamps")):
        switch = node(id_)
        assert switch.get("role") == "switch" and switch.get("type") == "checkbox"
        assert ui.accessible_name(switch, t) == label
    assert ui.accessible_name(node("tr-clear"), t) == "Clear transcript"


def test_no_model_is_the_shared_empty_state_with_a_link_to_models():
    """"No transcription model is available" uses the shared .empty component (as Speak does), with an h2
    title and Open Models as a real link to the STT models, not a scripted button."""
    box = node("tr-nomodel")
    assert "empty" in box.classes and box.has("hidden")
    titles = ui.select(box, "h2.empty__title")
    assert len(titles) == 1 and ui.visible_text(titles[0]).strip() == "No transcription model is available"
    links = ui.select(box, "a.btn")
    assert len(links) == 1 and links[0].get("href") == "#models/stt"
    assert ui.visible_text(links[0]).strip() == "Open Models"
    assert not ui.select(box, "button")


def test_the_meter_and_the_vad_pill_carry_text():
    """TR-09: the meter is role=meter with a name and a visible text label beside it; the VAD pill holds the
    word, never colour alone."""
    meter = node("tr-meter")
    assert meter.get("role") == "meter" and meter.get("aria-label") == "Input level"
    assert meter.get("aria-valuemin") == "0" and meter.get("aria-valuemax") == "100"
    assert "meter__label" in node("tr-meter-label").classes
    assert ui.visible_text(node("tr-vad")).strip() == "Silence"


def test_the_partial_uses_only_its_prefix_and_known_icons():
    """Every id carries the tr- prefix; every icon is in the sprite."""
    t = tree()
    for el in t.elements():
        if el.id:
            assert el.id.startswith("tr-"), el.id
    sprite = ui.sprite_ids() or set()
    for use in ui.select(t, "use"):
        assert use.get("href", "").split("#")[-1] in sprite


# ───────────────────────────── the view CSS ─────────────────────────────


def test_interim_text_is_muted_without_italics_or_opacity():
    """AX-07: the interim line is text-2 and upright; the hypothesis never dims with opacity (the planned
    ".75 opacity" gave way to the global no-dimmed-text audit); confirmed words use the text colour."""
    interim = decls(".tr-interim")
    assert interim.get("color") == "var(--text-2)" and interim.get("font-style") == "normal"
    assert "opacity" not in interim and "opacity" not in decls(".tr-interim__hyp")
    assert decls(".tr-interim__confirmed").get("color") == "var(--text)"


def test_the_record_button_is_an_80px_circle_that_turns_red_only_while_recording():
    """An 80 px round button; the accent fill idle, --rec-fill while recording (colour carries the
    state, and the name and the dot say it too); a transparent border that forced colours paint."""
    button = decls(".tr-rec-btn")
    assert button.get("width") == "80px" and button.get("height") == "80px" and button.get("border-radius") == "50%"
    assert button.get("background") == "var(--accent-fill)"
    assert "transparent" in button.get("border", "")
    assert decls(".tr-rec-btn.is-recording").get("background") == "var(--rec-fill)"
    busy = decls('.tr-rec-btn[aria-busy="true"]::after')
    assert "infinite" in busy.get("animation", ""), "the only loop is the busy spinner"


def test_view_css_is_scoped_layered_and_token_only():
    """View CSS scoping: every rule is in @layer views and scoped to [data-view="transcribe"] or the tr- prefix; no
    colour literal and no px font size."""
    text = CSS.read_text(encoding="utf-8")
    assert "@layer views" in text
    for selector, body, chain in ui.css_rules(CSS):
        if selector.startswith("@"):
            continue
        assert chain and chain[0] == "@layer views", selector
        for part in selector.split(","):
            p = part.strip()
            assert p.startswith('[data-view="transcribe"]') or re.match(r"[.#]tr-", p), p
        for prop, value in body:
            assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", value), f"{selector} {prop}: {value}"
            if prop == "font-size":
                assert not re.search(r"\dpx", value), f"{selector} {prop}: {value}"


def test_phones_bring_the_transcript_up_while_recording():
    """Below 960 px the upload and model fields step aside while a session runs, so the live transcript
    sits under the capture card; at 960 px and up the layout is capture | transcript."""
    for phase in ("requesting", "starting", "recording", "stopping"):
        for part in (".tr-upload", ".tr-model"):
            assert decls(f'.tr-layout[data-phase="{phase}"] {part}', "max-width: 959px").get("display") == "none"
    assert "minmax(0, 1fr)" in decls(".tr-layout", "min-width: 960px").get("grid-template-columns", "")


def test_a_narrow_transcript_panel_puts_each_time_above_its_words():
    """Reflow: the transcript panel is a .card (a size container); when it is narrow (a phone) each segment
    is one column, the time above the words, and the interim line drops the time-gutter indent, so the words are
    never squeezed into a narrow column beside a time gutter at 320 px."""
    assert "card" in node("tr-panel").classes
    assert decls(".tr-seg").get("grid-template-columns", "").startswith("4.25rem"), "a time gutter when there is room"
    narrow = [c for _, _, chain in ui.css_rules(CSS) for c in chain if c.startswith("@container")]
    assert narrow and all("max-width" in c for c in narrow)
    assert decls(".tr-seg", "@container").get("grid-template-columns") == "minmax(0, 1fr)"
    assert decls(".tr-interim", "@container").get("padding-inline-start") == "var(--sp-2)"


def test_timestamp_buttons_meet_the_touch_target():
    """On a coarse pointer the timestamp buttons are 44 px tall."""
    assert decls('[data-view="transcribe"] button.tr-ts', "pointer: coarse").get("min-height") == "44px"


def test_the_view_module_follows_the_contract():
    """View contract: imported in bare Node (no window, no document: nothing touches a browser API at import), the default
    export is {id: 'transcribe', title: 'Transcribe', partial, init()}, and its parts export what index.js uses."""
    script = """
import { pathToFileURL } from 'node:url';
const at = (f) => pathToFileURL('src/static/js/views/transcribe/' + f).href;
const view = (await import(at('index.js'))).default;
const parts = {};
for (const [file, names] of Object.entries({
  'recorder.js': ['createRecorder', 'noticeKey'], 'transcript.js': ['reduce', 'createState', 'summary'],
  'upload.js': ['createUploader', 'parseTranscription'], 'export.js': ['exportFile', 'toSrt', 'toVtt'],
  'log.js': ['createLog', 'segmentHtml'], 'model-select.js': ['bindModelSelect', 'sttOptions'],
  'ui.js': ['captureView', 'errorActions', 'bannerKey'],
})) {
  const m = await import(at(file));
  parts[file] = names.filter((n) => typeof m[n] !== 'function');
}
console.log(JSON.stringify({ id: view.id, title: view.title, partial: view.partial, init: typeof view.init,
  window: typeof globalThis.window, document: typeof globalThis.document, parts }));
"""
    r = json.loads(ui.run_node_script(script).strip().splitlines()[-1])
    assert (r["window"], r["document"]) == ("undefined", "undefined"), "bare Node"
    assert (r["id"], r["title"], r["partial"], r["init"]) == ("transcribe", "Transcribe", "/static/partials/transcribe.html", "function")
    assert (ui.STATIC / r["partial"].removeprefix("/static/")).exists()
    assert all(not missing_names for missing_names in r["parts"].values()), r["parts"]


def test_only_the_playing_segment_is_boxed_in_forced_colours():
    """No transcript segment carries an outline or border of its own (forced colours paint a transparent
    one in the text colour, boxing every line); the playing one is .is-current, which base.css outlines with
    Highlight in forced colours. A file's timestamp buttons keep a transparent border, so there they are still
    seen to be buttons."""
    for selector, body, _chain in ui.css_rules(CSS):
        if ".tr-seg" in selector:
            props = dict(body)
            assert not any(p.startswith(("outline", "border")) and not p.startswith("border-radius") for p in props), (selector, props)
    base = [(sel, dict(body)) for sel, body, chain in ui.css_rules(ui.CSS_ROOT / "base.css") if any("forced-colors" in c for c in chain)]
    assert any(".is-current" in sel and "Highlight" in body.get("outline", "") for sel, body in base)
    assert decls('[data-view="transcribe"] button.tr-ts').get("border") == "1px solid transparent"


def test_timestamp_buttons_have_a_44px_hit_area_on_a_fine_pointer_too():
    """A small control on a fine pointer keeps a 44 px hit area through an ::after inset; the button itself
    stays 32 px so the row does not grow."""
    button = decls('[data-view="transcribe"] button.tr-ts')
    assert button.get("min-height") == "32px" and button.get("position") == "relative"
    assert button.get("border") == "1px solid transparent", "the inset below is measured from inside this border"
    hit = decls('[data-view="transcribe"] button.tr-ts::after')
    assert hit.get("content") in ('""', "''") and hit.get("position") == "absolute"
    # Inside the 1px border the padding box is 30px: -7px above and below makes 44px, the button's own 32px plus
    # 6px each side; -1px sideways reaches the border's edge.
    assert hit.get("inset") == "calc((32px - var(--hit)) / 2 - 1px) -1px"
    assert decls('[data-view="transcribe"] button.tr-ts::after', "pointer: coarse").get("inset") == "0", "coarse: the 44 px button is the target"


def test_the_busy_file_name_wraps_instead_of_being_cut():
    """The busy dropzone shows which file runs: its name wraps (anywhere, for names with no spaces) and is
    never ellipsised to a few characters."""
    title = decls(".tr-drop__title")
    assert title.get("overflow-wrap") == "anywhere"
    assert "text-overflow" not in title and title.get("white-space") != "nowrap" and "overflow" not in title


def test_spacing_is_on_the_4px_scale():
    """Gaps, margins and paddings use the --sp-* scale (or 0, or rem); a px literal appears only inside a
    calc() that adds a fixed size to a token (the 96 px Cancel button)."""
    spacing = re.compile(r"^(gap|row-gap|column-gap|margin|padding)(-|$)")
    for selector, body, _chain in ui.css_rules(CSS):
        for prop, value in body:
            if not spacing.match(prop):
                continue
            outside_calc = re.sub(r"calc\((?:[^()]|\([^()]*\))*\)", "", value)
            assert not re.search(r"\d+(\.\d+)?px", outside_calc), f"{selector} {prop}: {value}"
            for calc in re.findall(r"calc\((?:[^()]|\([^()]*\))*\)", value):
                assert "var(--" in calc, f"{selector} {prop}: {value}"
    assert decls(".tr-rec__info").get("gap") == "var(--sp-1)"
    assert decls(".tr-drop__text").get("gap") == "var(--sp-1)"
    assert decls(".tr-log").get("gap") == "var(--sp-1)"
    assert decls('[data-view="transcribe"] button.tr-ts').get("padding") == "0 var(--sp-1)"
