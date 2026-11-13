"""The conversation of one voice connection: in memory, so a reconnect starts fresh.

It holds the user's words (as rendered with the preset's user template) and the AI's answers, never the thinking:
Qwen3 and DeepSeek both say to drop earlier thinking from the conversation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# The most exchanges kept; history_turns goes up to 100. Older ones are dropped.
MAX_EXCHANGES = 100
_PARTIAL_WORD = re.compile(r"\S*\Z")


@dataclass
class _Message:
    role: str
    content: str
    item_id: str | None = None


class Conversation:
    """An exchange is the user's messages up to and including the answer that followed them. Messages after the
    last answer (a question waiting for its answer, or one a barge-in left unanswered) are the pending tail."""

    def __init__(self) -> None:
        self._messages: list[_Message] = []

    def add_user(self, text: str) -> None:
        self._messages.append(_Message("user", text))
        self._trim()

    def add_assistant(self, text: str, item_id: str | None = None) -> None:
        """Add an answer: ``AssistantReply.text``, never its thinking. ``item_id`` is the realtime item."""
        self._messages.append(_Message("assistant", text, item_id))
        self._trim()

    def reset(self) -> None:
        self._messages.clear()

    def for_request(self, history_turns: int) -> list[dict[str, Any]]:
        """The last ``history_turns`` exchanges, then the pending tail, as chat messages, oldest first.

        Answers only, never the thinking. The list and its messages are copies.
        """
        exchanges, tail = self._split()
        count = max(0, int(history_turns))
        chosen = exchanges[-count:] if count else []
        return [{"role": m.role, "content": m.content} for exchange in chosen for m in exchange] + [
            {"role": m.role, "content": m.content} for m in tail
        ]

    def truncate_assistant(self, item_id: str, fraction: float) -> bool:
        """Keep the part of an answer that was heard: its first ``fraction`` (0 to 1), cut at a word.

        An answer with nothing heard is removed. Returns False for an unknown item or a fraction that isn't a
        number.
        """
        if item_id is None or isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction):
            return False
        for index in range(len(self._messages) - 1, -1, -1):
            message = self._messages[index]
            if message.role == "assistant" and message.item_id == item_id:
                break
        else:
            return False
        fraction = min(max(float(fraction), 0.0), 1.0)
        if fraction >= 1.0:
            return True
        text = message.content
        kept = text[: int(len(text) * fraction)]
        if kept and len(kept) < len(text) and not text[len(kept)].isspace():
            kept = _PARTIAL_WORD.sub("", kept)  # a word cut in half wasn't heard
        kept = kept.strip()
        if kept:
            message.content = kept
        else:
            del self._messages[index]
        return True

    def _split(self) -> tuple[list[list[_Message]], list[_Message]]:
        exchanges: list[list[_Message]] = []
        current: list[_Message] = []
        for message in self._messages:
            current.append(message)
            if message.role == "assistant":
                exchanges.append(current)
                current = []
        return exchanges, current

    def _trim(self) -> None:
        exchanges, tail = self._split()
        if len(exchanges) > MAX_EXCHANGES or len(tail) > MAX_EXCHANGES:
            kept = exchanges[-MAX_EXCHANGES:]
            self._messages = [m for exchange in kept for m in exchange] + tail[-MAX_EXCHANGES:]
