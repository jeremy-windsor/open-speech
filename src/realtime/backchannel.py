"""Detect short acknowledgements ("mm-hmm", "yeah okay") spoken over a reply.

While the server is talking, a listener often says "uh-huh" without meaning to
take the turn. Treating that as an interruption cuts the reply off for nothing.
The idea comes from SpeakRail's harness: tile the utterance with a small
dictionary of English backchannel phrases. English only.
"""

from __future__ import annotations

import re

_PHRASES = {
    "mm", "mmm", "mhm", "mmhm", "mmhmm", "hm", "hmm", "uh", "um", "uhm",
    "uhhuh", "uh huh", "mm hmm", "mm hm", "ah", "aha", "oh", "ooh", "wow",
    "yeah", "yep", "yup", "yes", "ya", "yea", "okay", "ok", "kay", "right",
    "sure", "alright", "all right", "cool", "nice", "great", "good", "fine",
    "true", "exactly", "totally", "absolutely", "indeed", "really", "perfect",
    "i see", "got it", "gotcha", "makes sense", "fair enough", "of course",
    "for sure", "sounds good", "i know", "understood", "correct",
    "go on", "keep going", "carry on",
    "oh okay", "oh ok", "oh right", "oh yeah", "oh i see", "oh wow", "oh really",
}
_PHRASE_TOKENS = {tuple(p.split()) for p in _PHRASES}
_LONGEST_PHRASE = max(len(p) for p in _PHRASE_TOKENS)

MAX_WORDS = 5  # anything longer is content, whatever the words


def _words(text: str) -> list[str]:
    text = text.lower().replace("-", " ").replace("'", "")
    return re.findall(r"[a-z]+", text)


def is_backchannel(text: str) -> bool:
    """True when the whole utterance is made of acknowledgement phrases."""
    words = _words(text)
    if not words or len(words) > MAX_WORDS:
        return False
    # can_end[i]: words[:i] can be split into known phrases
    can_end = [True] + [False] * len(words)
    for end in range(1, len(words) + 1):
        for size in range(1, min(_LONGEST_PHRASE, end) + 1):
            if can_end[end - size] and tuple(words[end - size:end]) in _PHRASE_TOKENS:
                can_end[end] = True
                break
    return can_end[-1]
