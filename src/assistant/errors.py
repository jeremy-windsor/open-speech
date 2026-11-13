"""Errors of the AI assistant: each has a code and a default sentence that is spoken when it happens.

A preset's ``error_sentences`` replaces a sentence per code; a code it leaves out keeps the default here.
"""

from __future__ import annotations

from types import MappingProxyType

ASSISTANT_UNREACHABLE = "assistant_unreachable"
ASSISTANT_TIMEOUT = "assistant_timeout"
ASSISTANT_HTTP_ERROR = "assistant_http_error"
ASSISTANT_BAD_RESPONSE = "assistant_bad_response"
ASSISTANT_EMPTY_REPLY = "assistant_empty_reply"
ASSISTANT_REPLY_CUT_OFF = "assistant_reply_cut_off"

# The default sentence for each code.
ERROR_SENTENCES = MappingProxyType({
    ASSISTANT_UNREACHABLE: "I couldn't reach the AI server.",  # Connection refused, DNS or TLS failure
    ASSISTANT_TIMEOUT: "The AI took too long to answer.",  # No whole reply within timeout_s
    ASSISTANT_HTTP_ERROR: "The AI server returned an error.",  # A status other than 200
    ASSISTANT_BAD_RESPONSE: "The AI didn't send back an answer.",  # Not JSON, or no choices[0].message
    ASSISTANT_EMPTY_REPLY: "The AI didn't send back an answer.",  # Nothing left once the thinking is removed
    ASSISTANT_REPLY_CUT_OFF: "The answer was cut off before it started.",  # Empty, with finish_reason "length"
})
ERROR_CODES = tuple(ERROR_SENTENCES)
# For a code this module doesn't know (an error code added elsewhere).
FALLBACK_SENTENCE = ERROR_SENTENCES[ASSISTANT_HTTP_ERROR]


def sentence_for(code: str, sentences: dict[str, str] | None = None) -> str:
    """The sentence to speak for ``code``: the preset's own, else the default."""
    if sentences and sentences.get(code):
        return sentences[code]
    return ERROR_SENTENCES.get(code, FALLBACK_SENTENCE)


class AssistantError(Exception):
    """The AI server gave no usable answer. ``code`` is one of ERROR_CODES; ``message`` never holds the API key."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
