"""Keep the container's filesystem paths out of the error messages a client sees.

An exception's text often names the file it failed on (``/root/.cache/huggingface/hub/.../encoder.onnx``). The
log keeps the full message; a client gets each absolute path cut to its last part (``encoder.onnx``).
"""

from __future__ import annotations

import re

# An absolute path: POSIX (/root/.cache/...), Windows (C:\...) or UNC (\\host\share\...), up to a space or quote.
_PATH = re.compile(r"""(?<![\w:/\\.])(?:/[^\s'"/:,;()\[\]]+){2,}/?|\b[A-Za-z]:\\[^\s'",;()\[\]]*|\\\\[^\s'",;()\[\]]+""")


def redact_paths(text: str) -> str:
    """``text`` with each absolute filesystem path replaced by its last part, for a client: the container's folders
    are no business of an API caller's, and the log keeps the full message."""
    def last_part(match: re.Match) -> str:
        parts = [part for part in re.split(r"[\\/]", match.group(0)) if part]
        return parts[-1] if parts else "..."

    return _PATH.sub(last_part, text)
