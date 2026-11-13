"""An AI reply cleaned and split for speech, by the same segmenter the Live Reader uses.

Emphasis markers are removed, links are read by their label, and a code block becomes "Code block skipped." A
reply is never refused for its length: it is spoken segment by segment.
"""

from __future__ import annotations

from src import config as config_module
from src.live_tts.segmenter import LiveTextSegmenter

# The segmenter takes a long text this much at a time, as it takes a streamed reply: one call with all of it costs
# time that grows with the square of its length.
FEED_CHARS = 16_384


def speakable_segments(text: str, max_chars: int | None = None, *, stop_after_chars: int | None = None) -> list[str]:
    """The reply's speakable segments, in order, none empty.

    ``max_chars`` is the model's input limit (``tts_router.max_input_chars_for(model)``), or None. As in the Live
    Reader, a segment is also at most ``TTS_LIVE_SENTENCE_MAX_CHARS`` characters and
    ``TTS_LIVE_SENTENCE_MAX_WORDS`` words.

    With ``stop_after_chars``, it stops once the segments hold more than that many characters: the list is then
    only the reply's start, and its length in characters is over ``stop_after_chars``.
    """
    settings = config_module.settings
    limit = settings.tts_live_sentence_max_chars
    if max_chars:
        limit = min(limit, int(max_chars))
    segmenter = LiveTextSegmenter(mode="natural", max_chars=limit, max_words=settings.tts_live_sentence_max_words)
    segments: list[str] = []
    spoken = 0

    def take(found) -> bool:
        nonlocal spoken
        for segment in found:
            if piece := segment.text.strip():
                segments.append(piece)
                spoken += len(piece)
        return stop_after_chars is not None and spoken > stop_after_chars

    for start in range(0, len(text), FEED_CHARS):
        if take(segmenter.append(text[start:start + FEED_CHARS])):
            return segments
    take(segmenter.flush())
    return segments
