"""Reader player (V2): the playback engine, the transport and its sheets.

The behaviour lives in the Node tests under tests/js/reader-player (one pytest case per file). This file adds
the structure of the slot partial, the architecture rules the engine must keep (plain audio elements, never an
AudioContext), and the CSS intent of the transport (fixed and safe-area padded under 960 px, docked above).
"""

from __future__ import annotations

import re
from functools import lru_cache

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

PARTIAL = ui.PARTIALS_DIR / "reader-player.html"
CSS = ui.CSS_ROOT / "views" / "reader-player.css"
READER_JS = ui.JS_ROOT / "views" / "reader"
V2_MODULES = ("engine", "play-logic", "timeline", "sleep-timer", "media-session", "player", "speed-sheet", "sleep-sheet", "voice-sheet")


@lru_cache(maxsize=1)
def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


def by_id(id_: str) -> ui.Node:
    node = ui.by_id(tree(), id_)
    assert node is not None, f"#{id_} is missing"
    return node


# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("reader-player"))
def test_js_reader_player(path):
    ui.run_node_test(path)


def test_v2_inventory_ids_and_defects_have_tests():
    """Every inventory id and defect V2 owns has an [ID]-titled test that runs."""
    assert missing("V2") == []


# ───────────────────────────── architecture ─────────────────────────────


def test_reader_audio_is_plain_audio_elements_never_an_audio_context():
    """[RE-02] Reader audio is plain <audio> elements, never routed through an AudioContext (a locked iPhone
    keeps playing HTMLAudioElement audio in the background, not Web Audio). That the engine plays through
    #rd-audio-a and #rd-audio-b inside #app-audio-host is proven by behaviour: engine-parts.test.js [RE-01]."""
    for name in V2_MODULES:
        src = ui.lex_js(READER_JS / f"{name}.js")
        for banned in ("AudioContext", "createMediaElementSource", "createScriptProcessor", "createMediaStreamSource", "MediaSource"):
            assert banned not in src.code, f"{name}.js uses {banned}"


def test_v2_modules_have_no_import_side_effects_and_leave_browser_apis_to_core():
    """The engine and player touch the browser only inside functions; navigator.mediaSession stays with
    core/audio.js, fetch with core/api.js, storage with core/storage.js."""
    for name in V2_MODULES:
        src = ui.lex_js(READER_JS / f"{name}.js")
        assert "navigator.mediaSession" not in src.code, name
        assert not re.search(r"(?<![\w.])fetch\(", src.code), name
        assert "localStorage" not in src.code, name
        assert ".innerHTML" not in src.code, name
        for line in src.code.splitlines():
            if line.startswith(("document.", "window.", "navigator.")):
                pytest.fail(f"{name}.js touches the page at import: {line.strip()}")


# ───────────────────────────── partial structure ─────────────────────────────


def test_slot_partial_has_no_h1_and_its_sheets_start_at_h2():
    """A slot partial lives inside the Reader's view: no h1; each sheet is labelled by its own h2."""
    root = tree()
    assert ui.select(root, "h1") == []
    sheets = ui.select(root, "dialog.sheet")
    assert {s.id for s in sheets} == {"rdp-speed-sheet", "rdp-sleep-sheet", "rdp-voice-sheet"}
    for sheet in sheets:
        heading = ui.by_id(root, sheet.get("aria-labelledby"))
        assert heading is not None and heading.tag == "h2", sheet.id
        assert ui.select(sheet, "[data-close]"), f"{sheet.id} has a close button"


def test_transport_counts_in_the_chrome_metrics_and_holds_the_player_controls():
    """The transport (.rdp-transport) carries data-chrome="bottom" (the measured chrome) and the player controls by id."""
    transport = by_id("rdp-transport")
    assert "rdp-transport" in transport.classes
    assert transport.get("data-chrome") == "bottom"
    root = tree()
    for id_ in ("rdp-status", "rdp-elapsed", "rdp-scrub", "rdp-remaining", "rdp-chapter", "rdp-back", "rdp-prev", "rdp-play",
                "rdp-next", "rdp-fwd", "rdp-speed", "rdp-sleep", "rdp-voice", "rdp-chapters"):
        node = ui.by_id(root, id_)
        assert node is not None and transport in list(node.ancestors()), id_
    for id_ in ("rdp-elapsed", "rdp-remaining"):
        assert by_id(id_).get("aria-hidden") == "true", "the times are spoken through the scrubber's aria-valuetext"


def test_status_line_is_the_players_one_status_region():
    """[RE-29] #rdp-status is the player's polite status line (role=status); the transport has no other."""
    status = by_id("rdp-status")
    assert status.get("role") == "status"
    assert not status.has("hidden"), "a live region must be in the tree before its first message"
    live = [n for n in by_id("rdp-transport").elements() if n.get("role") in ("status", "alert", "log") or n.has("aria-live")]
    assert live == [status]


def test_scrubber_is_a_native_range_named_chapter_position():
    """[REV-13] The scrubber is a native <input type=range class=seek> (iOS VoiceOver adjusts it with swipes),
    named "Chapter position", with a spoken value; never a custom role=slider."""
    scrub = by_id("rdp-scrub")
    assert scrub.tag == "input" and scrub.get("type") == "range"
    assert "seek" in scrub.classes
    assert scrub.get("aria-label") == "Chapter position"
    assert scrub.get("aria-valuetext")
    assert not scrub.has("role")
    assert ui.select(tree(), '[role="slider"]') == []


def test_play_button_swaps_its_name_and_never_uses_aria_pressed():
    """[RE-69] The 64 px play button is named Play (Pause while playing) and carries no aria-pressed."""
    play = by_id("rdp-play")
    assert "play-btn" in play.classes
    assert play.get("aria-label") == "Play"
    assert not play.has("aria-pressed")


def test_every_reader_player_icon_button_has_a_name():
    """[RE-70] Every icon-only button of the player (skips, previous/next, play, the sheets' close and steppers)
    has an aria-label."""
    root = tree()
    buttons = ui.select(root, ".icon-btn, .play-btn")
    assert len(buttons) >= 12
    for button in buttons:
        assert button.get("aria-label"), button
        assert button.get("type") == "button", button


def test_chips_open_their_sheets_and_say_what_they_set():
    """The speed, sleep and voice chips are dialog buttons pointing at their sheets, with spoken names."""
    for chip, sheet in (("rdp-speed", "rdp-speed-sheet"), ("rdp-sleep", "rdp-sleep-sheet"), ("rdp-voice", "rdp-voice-sheet")):
        node = by_id(chip)
        assert node.get("aria-haspopup") == "dialog"
        assert node.get("aria-controls") == sheet
    assert by_id("rdp-speed").get("aria-label") == "Playback speed 1 times. Change speed"
    assert by_id("rdp-voice").get("aria-label").startswith("Reading voice: ")
    assert by_id("rdp-chapters").get("aria-haspopup") == "dialog"


def test_speed_sheet_has_a_preset_radiogroup_a_fine_slider_and_steppers():
    """[RE-67] The speed sheet: presets 0.75–3× as one role=radiogroup of role=radio buttons, a 0.5–3.5×
    slider in 0.05 steps, ±0.05 steppers and Reset to 1×."""
    group = by_id("rdp-speed-presets")
    assert group.get("role") == "radiogroup"
    radios = ui.select(group, '[role="radio"]')
    assert [r.get("data-speed") for r in radios] == ["0.75", "1", "1.1", "1.25", "1.5", "1.75", "2", "2.5", "3"]
    assert all(r.has("aria-checked") for r in radios)
    slider = by_id("rdp-speed-range")
    assert (slider.get("type"), slider.get("min"), slider.get("max"), slider.get("step")) == ("range", "0.5", "3.5", "0.05")
    assert by_id("rdp-speed-dec").get("aria-label") == "Slower by 0.05"
    assert by_id("rdp-speed-inc").get("aria-label") == "Faster by 0.05"
    assert "Reset to 1×" in by_id("rdp-speed-reset").text_content()


def test_sleep_sheet_offers_minutes_chapters_and_fade():
    """The sleep sheet: Off, 5–60 min, End of chapter, End of N chapters with a stepper, Fade out."""
    values = [n.get("value") for n in ui.select(tree(), 'input[name="rdp-sleep"]')]
    assert values == ["off", "5", "10", "15", "30", "45", "60", "chapter", "chapters"]
    assert by_id("rdp-sleep-fade").get("role") == "switch"
    assert "Not available on iPhone" in by_id("rdp-sleep-fade-hint").text_content()
    assert by_id("rdp-sleep-remaining").get("role") is None, "the live countdown is not a live region"


def test_voice_sheet_has_a_model_select_the_current_voice_and_no_recordings():
    """[REV-09] The voice sheet: a labelled model select, the current voice with Choose voice, the hint that a
    cloned voice reaches the Reader as a named voice, and nothing about raw recordings."""
    root = tree()
    select = by_id("rdp-voice-model")
    assert select.tag == "select" and ui.accessible_name(select, root) == "Model"
    text = by_id("rdp-voice-sheet").text_content()
    assert "To read with a cloned voice, save it as a named voice in Voices." in text
    assert "Audio renders on the server at normal speed; speed changes playback only." in text
    assert "Recording" not in text
    assert by_id("rdp-voice-banner").get("role") == "alert"


# ───────────────────────────── CSS intent ─────────────────────────────


def _rules(selector: str, media: str = ""):
    out = {}
    for sel, body, chain in ui.css_rules(CSS):
        conditional = [c for c in chain if c.startswith(("@media", "@supports"))]
        wanted = any(media in c for c in conditional) if media else not conditional
        if sel.strip() == selector and wanted:
            out.update(dict(body))
    return out


def test_transport_is_fixed_and_safe_area_padded_on_phones_and_docked_on_desktop():
    """[GS-25] Below 960 px the transport is fixed at the bottom with the safe-area insets (the immersive
    reading view); from 960 px it is docked (sticky) at the bottom of the content column."""
    phone = _rules(".rdp-transport")
    assert phone.get("position") == "fixed"
    assert phone.get("bottom") == "0"
    assert "safe-area-inset-bottom" in phone.get("padding", "")
    assert "safe-area-inset-left" in phone.get("padding", "") and "safe-area-inset-right" in phone.get("padding", "")
    desktop = _rules(".rdp-transport", "min-width: 960px")
    assert desktop.get("position") == "sticky"
    assert _rules('[data-view="reader"] #rd-player-slot').get("display") == "contents"


def test_transport_blur_has_opaque_fallbacks_and_no_transformed_ancestor():
    """[GS-23] The transport may blur what scrolls under it with an opaque fallback; nothing in the
    player's CSS transforms, filters or blurs a view, main or the slot (a fixed bar would stop being fixed)."""
    assert "blur(16px)" in _rules(".rdp-transport").get("backdrop-filter", "")
    assert _rules(".rdp-transport", "prefers-reduced-transparency").get("backdrop-filter") == "none"
    assert _rules(".rdp-transport", "@supports not").get("background") == "var(--bg)"
    for sel, body, _ in ui.css_rules(CSS):
        props = dict(body)
        if re.search(r"(^|\s)(main|\.view|\[data-view=\"reader\"\]|#rd-player-slot|#rd-reading)$", sel.strip()):
            for prop in ("transform", "filter", "backdrop-filter", "will-change", "contain"):
                assert prop not in props, f"{sel} sets {prop}"


def test_status_line_stays_in_the_tree_when_empty():
    """An empty #rdp-status is clipped, not display:none, so its first message is announced."""
    empty = _rules(".rdp-status:empty")
    assert "clip-path" in empty
    assert empty.get("display") != "none" and "visibility" not in empty


def test_status_line_is_part_of_the_transport_box_so_it_never_covers_the_text():
    """[RE-29] With something to say, #rdp-status sits in the transport's own flow (not floated above it), so
    the transport's measured chrome (scroll padding, the Reader's follow band) includes it and it never covers the
    sentence being read; empty, it takes no room."""
    status = _rules(".rdp-status")
    assert status.get("position") in (None, "static", "relative"), status.get("position")
    assert "transform" not in status and "bottom" not in status
    empty = _rules(".rdp-status:empty")
    assert empty.get("position") == "absolute" and empty.get("width") == "1px" and empty.get("height") == "1px"
    transport = by_id("rdp-transport")
    assert by_id("rdp-status") in transport.children, "the status line is a direct child of the transport"


def test_the_transport_never_repeats_the_document_and_covers_the_gap_under_it():
    """The transport never shows the document's tile, title or chapter. The Reader's
    top bar names both at every width and the aside does from 1180 px, so on desktop the title showed three times;
    that room goes to the scrubber, a full row under the controls and chips. Docked on desktop, its last shadow layer
    is a band in the page's colour reaching the window's edge, so no sliced line of text shows in the gap under it."""
    assert _rules(".rdp-now").get("display") == "none"
    for sel, body, chain in ui.css_rules(CSS):
        props = dict(body)
        if ".rdp-now" in [part.strip() for part in sel.split(",")]:
            assert props.get("display") in (None, "none"), (sel, chain)   # its children never show without it
        assert "now" not in props.get("grid-template-areas", ""), (sel, chain)
    assert _rules(".rdp-bar", "min-width: 960px").get("grid-template-areas") == '"controls chips" "timeline timeline"'
    desktop = _rules(".rdp-transport", "min-width: 960px")
    assert desktop.get("bottom") == "var(--sp-4)"
    layers = [layer.strip() for layer in re.split(r",(?![^(]*\))", desktop.get("box-shadow", ""))]
    assert layers[-1] == "0 calc(var(--sp-4) + 1px) 0 0 var(--bg)", layers


def test_the_icon_only_chapters_chip_is_a_44px_target():
    """Where the Chapters chip shows only its icon (below 600 px, and on short screens below the rail) it stays
    at least 44 px wide; it measured 40 x 44 on a phone."""
    for media in ("max-width: 599px", "max-height: 520px"):
        chip = _rules(".rdp-chip--chapters", media)
        assert chip.get("min-width") == "44px" and chip.get("justify-content") == "center", media
