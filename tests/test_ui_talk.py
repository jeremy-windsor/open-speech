"""Talk: the Node behaviour tests, the partial's structure and the view CSS.

The behaviour (the GA /v1/realtime session engine against a scripted fake socket, the microphone and the audio
coordinator, typed questions, keep-alive and quiet reconnects, leak checks over repeated start and stop, the view's
off state, log, typing, errors and screenshot states, and Talk in the shell) is tested in tests/js/talk/*.test.js;
this file runs them and checks what only the static files can show.
"""

from __future__ import annotations

import pytest

from tests import ui_support as ui

PARTIAL = ui.PARTIALS_DIR / "talk.html"
CSS = ui.CSS_ROOT / "views" / "talk.css"


def tree() -> ui.Node:
    return ui.parse_html(PARTIAL)


def node(id_: str) -> ui.Node:
    el = ui.by_id(tree(), id_)
    assert el is not None, f"#{id_} is missing from talk.html"
    return el


def decls(selector: str, chain_has: str = "") -> dict[str, str]:
    """The declarations of one exact selector in talk.css (outside @media and @container unless chain_has names
    the condition)."""
    out: dict[str, str] = {}
    for sel, body, chain in ui.css_rules(CSS):
        conditional = [c for c in chain if c.startswith(("@media", "@supports", "@container"))]
        wanted = any(chain_has in c for c in conditional) if chain_has else not conditional
        if wanted and any(part.strip() == selector for part in sel.split(",")):
            out.update(dict(body))
    return out


@pytest.mark.parametrize("path", ui.js_test_files("talk"))
def test_js(path):
    """The Talk behaviour tests."""
    ui.run_node_test(path, timeout=240)


def test_the_partial_has_one_heading_the_off_state_the_controls_and_the_conversation():
    """One h1 (Talk); the off state with its way to Settings; the main control, the microphone switch,
    New conversation, Stop reply and Resume; the preset select, the voice button and the interrupt switch."""
    t = tree()
    h1s = ui.select(t, "h1")
    assert [h.id for h in h1s] == ["tk-title"] and h1s[0].text_content().strip() == "Talk"
    off = node("tk-off")
    assert off.has("hidden")
    assert "The AI assistant is off" in off.text_content()
    assert ui.by_id(off, "tk-open-settings").tag == "button"
    for id_ in ("tk-start", "tk-mic", "tk-new", "tk-cancel", "tk-resume", "tk-voice", "tk-send"):
        el = node(id_)
        assert el.tag == "button" and el.get("type") in ("button", "submit"), id_
    assert node("tk-mic").get("aria-pressed") in ("true", "false")
    assert node("tk-preset").tag == "select"
    interrupt = node("tk-interrupt")
    assert (interrupt.get("type"), interrupt.get("role"), interrupt.has("checked")) == ("checkbox", "switch", True)
    assert "tk-interrupt-hint" in (interrupt.get("aria-describedby") or "")
    assert "headphones" in node("tk-interrupt-hint").text_content()


def test_the_assistant_select_comes_first_among_the_choices_hidden_until_assistants_are_saved():
    """Who answers: an Assistant select, labelled, with its hint in its description, first in the choices
    (before the AI preset), hidden until the view knows of saved assistants."""
    field = node("tk-assistant-field")
    assert field.has("hidden")
    pick = node("tk-assistant")
    assert pick.tag == "select" and ui.accessible_name(pick, tree()) == "Assistant"
    assert pick.get("aria-describedby") == "tk-assistant-hint"
    assert "in use" in ui.visible_text(node("tk-assistant-hint"))
    order = [el.id for el in ui.select(tree(), ".tk-choices select, .tk-choices button, .tk-choices input")]
    assert order[:2] == ["tk-assistant", "tk-preset"]


def test_the_log_is_a_quiet_log_with_the_interim_outside_and_a_status_region_beside_it():
    """#tk-log is role=log with aria-live=off (the answer is heard; a screen reader reading the same
    words at once would talk over it), named by its heading and focusable so it scrolls from the keyboard; the
    words of the turn being spoken (#tk-interim) sit outside it, never announced; #tk-status is the polite
    status region; the fatal banner is an alert."""
    log = node("tk-log")
    assert (log.get("role"), log.get("aria-live"), log.get("tabindex")) == ("log", "off", "0")
    assert ui.by_id(tree(), log.get("aria-labelledby")).tag == "h2"
    interim = node("tk-interim")
    assert interim.get("aria-live") == "off"
    assert log not in list(interim.ancestors())
    assert node("tk-status").get("role") == "status"
    assert node("tk-error").get("role") == "alert"
    assert node("tk-load-error").get("role") == "alert"


def test_delete_chat_sits_in_the_conversation_header_with_an_icon_and_its_label():
    """Delete chat is a button with an icon and a visible text label in the
    conversation panel's header, beside its heading and outside the log; hidden until the chat has entries. The
    header keeps the heading and the button on one line while they fit, and wraps when they don't (320 px, text
    spacing)."""
    t = tree()

    def node(id_: str) -> ui.Node:   # one parse, so the nodes compare by identity
        el = ui.by_id(t, id_)
        assert el is not None, f"#{id_} is missing from talk.html"
        return el

    btn = node("tk-delete")
    assert btn.tag == "button" and btn.get("type") == "button"
    assert btn.has("hidden")
    assert ui.accessible_name(btn, t) == "Delete chat"
    assert not btn.get("aria-label"), "the visible label is the name"
    uses = ui.select(btn, "use")
    assert [u.get("href") for u in uses] == ["/static/icons/sprite.svg#i-trash"]
    assert ui.select(btn, "svg")[0].get("aria-hidden") == "true"
    ancestors = list(btn.ancestors())
    assert node("tk-panel") in ancestors
    assert node("tk-log") not in ancestors
    title = node("tk-log-title")
    assert title.tag == "h2" and title.parent is btn.parent, "beside the conversation's heading"
    head = decls('[data-view="talk"] .tk-panel__head')
    assert (head.get("display"), head.get("flex-wrap")) == ("flex", "wrap")
    assert "tk-panel__head" in btn.parent.classes


def test_the_empty_state_says_what_talk_is_for_and_points_to_transcribe():
    """The two uses in one line: speech to text as you talk is Transcribe; Talk sends what you
    say to the AI and reads its answer back."""
    empty = node("tk-empty")
    links = [a.get("href") for a in ui.select(empty, "a")]
    assert links == ["#transcribe"]
    text = " ".join(empty.text_content().split())
    assert "Transcribe" in text and "AI" in text


def test_the_text_box_takes_a_typed_question_of_up_to_8000_characters():
    """A labelled textarea (8,000 characters), a submit button and the hint and error it names."""
    box = node("tk-text")
    assert box.tag == "textarea" and box.get("maxlength") == "8000"
    assert ui.accessible_name(box, tree())
    described = (box.get("aria-describedby") or "").split()
    assert "tk-text-hint" in described and "tk-text-error" in described
    assert node("tk-send").get("type") == "submit"
    assert node("tk-ask").tag == "form"
    assert node("tk-text-error").has("hidden")


def test_the_css_lays_out_two_columns_on_desktop_and_keeps_the_log_scrollable():
    """One column on phones and tablets, controls beside the conversation from 960 px; the log scrolls
    inside its card with a height cap; your words sit at the end of the line, the waiting phrases and the thinking
    are dimmed."""
    desk = decls('[data-view="talk"] .tk-layout', "min-width: 960px")
    assert "minmax(0, 1fr)" in desk["grid-template-columns"]
    log = decls('[data-view="talk"] .tk-log')
    assert log["overflow-y"] == "auto" and "max-height" in log
    assert decls('[data-view="talk"] .tk-entry[data-kind="you"]')["justify-self"] == "end"
    for kind in ("filler", "thinking"):
        assert decls(f'[data-view="talk"] .tk-entry[data-kind="{kind}"]')["color"] == "var(--text-2)", kind
    assert decls('[data-view="talk"] .tk-entry__text')["overflow-wrap"] == "anywhere"
