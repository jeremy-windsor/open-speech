"""Prompts: a preset's system prompt and its user template, which wraps your words or a piece of a book.

Rendering is one pass of literal replacement over the template only, never ``str.format``: braces that aren't a
placeholder are kept as they are, and text inserted for ``{text}`` is never searched for placeholders, so a book
that contains ``{title}`` stays as written. An unknown placeholder is kept here; saving a preset refuses it
(``src.assistant.presets``).

``POST /api/assistant/ask`` may put two things before the rendered template, each followed by a blank line: the
listener's own question, then, for a Reader document, a line naming it (``source_line``). They are never part of the
template, so the text inserted for them isn't searched either.
"""

from __future__ import annotations

import re
from typing import Any

# {text}: your words, or a piece of the book. {title}, {author}, {chapter} (the chapter's title): book jobs.
PLACEHOLDERS = ("text", "title", "author", "chapter")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def placeholders_in(template: str) -> list[str]:
    """The placeholder names in ``template``, in order, known or not."""
    return _PLACEHOLDER.findall(template)


def render(template: str, **values: Any) -> str:
    """Fill the known placeholders of ``template`` in one pass; one without a value becomes empty."""

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in PLACEHOLDERS:
            return match.group(0)
        value = values.get(name)
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(fill, template)


def source_line(template: str, title: str, chapter: str | None = None) -> str:
    """The line that tells the AI what a Reader document's text is: ``From "<title>":``,
    or ``From "<title>", chapter "<chapter>":`` with a chapter. "" for a ``template`` that already uses ``{title}`` or
    ``{chapter}``, so nothing says it twice."""
    if {"title", "chapter"} & set(placeholders_in(template)):
        return ""
    return f'From "{title}", chapter "{chapter}":' if chapter else f'From "{title}":'


def lead(question: str | None = None, source: str = "") -> str:
    """What goes before the rendered template: the ``question`` as typed, then the ``source`` line, each followed
    by a blank line; "" for neither."""
    return "".join(f"{part}\n\n" for part in (question, source) if part)


def messages_for(settings, history: list[dict[str, Any]], text: str | None = None, *, question: str | None = None,
                 source: str = "", **values: Any) -> list[dict[str, Any]]:
    """The chat messages for one request: the system prompt, the history, then ``text`` in the user template.

    ``history`` is what ``Conversation.for_request`` returns (answers only, never thinking); it is copied. With
    ``text`` None no user message is added, for a history that already ends with the user's words. ``values``
    fills ``{title}``, ``{author}`` and ``{chapter}``. ``question`` and ``source`` (a ``source_line``) go before the
    rendered template, as ``lead`` joins them.
    """
    messages: list[dict[str, Any]] = []
    if settings.system_prompt:
        messages.append({"role": "system", "content": settings.system_prompt})
    messages.extend(dict(message) for message in history)
    if text is not None:
        content = lead(question, source) + render(settings.user_template, **values, text=text)
        messages.append({"role": "user", "content": content})
    return messages
