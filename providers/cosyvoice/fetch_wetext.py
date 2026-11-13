"""Fetch the FST files of CosyVoice's text normalizer at a pinned commit, for the image build.

CosyVoice reads numbers, money, times and dates through wetext 0.0.4, which on its own fetches
the whole ModelScope repository pengzhendong/wetext from its default branch at every model load
in a new container. The CosyVoice Dockerfile runs this script once instead, and the worker points
wetext at the files it writes (WETEXT_DIR in app.py). Each file is checked against the size and
SHA-256 that ModelScope's file listing gives for the pinned commit.

Usage: python fetch_wetext.py <target directory>
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY = "pengzhendong/wetext"
# The repository's head from 2025-01-07 to 2025-07-20, all of wetext 0.0.4's life: the next
# commit came with wetext 0.0.7. At this commit the repository holds only the en and zh files
# that wetext 0.0.4 reads.
REVISION = "67928a15fac1e769e23189c53648a65dea206a39"
# What CosyVoice's two Normalizers read: English and Chinese text normalization (operator "tn",
# remove_erhua=False). Size in bytes and SHA-256 at REVISION.
FILES = {
    "en/tn/tagger.fst": (5209814, "cdbc646a6c0dea245aac43ffa88372fae36e90e7c284b52fc8b60071c1da9410"),
    "en/tn/verbalizer.fst": (2121910, "c2b44c1c023df96a09c0ac8f5dd8625b139a69a13da5ab1b0c8ad765cda73266"),
    "zh/tn/tagger.fst": (3479454, "36d7affd5a4e39efca570e5b10273103e5be75615982e71a6314299ee7b8ebdd"),
    "zh/tn/verbalizer.fst": (558366, "81618a81f4eee9524fae0763152a09d7c70460d2d670043bfc9715668575a9a6"),
}
# ModelScope answers 403 "too many requests" for a while after a burst; wait it out.
RETRY_WAITS_S = (15, 30, 60, 120, 240)
RETRY_STATUS = frozenset({403, 429, 500, 502, 503, 504})


def file_url(name: str) -> str:
    query = urllib.parse.urlencode({"Revision": REVISION, "FilePath": name})
    return f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo?{query}"


def _download(
    url: str, opener: Callable[..., Any], sleep: Callable[[float], None]
) -> bytes:
    for wait in (*RETRY_WAITS_S, None):
        try:
            with opener(url, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if wait is None or exc.code not in RETRY_STATUS:
                raise
            failure: Exception = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if wait is None:
                raise
            failure = exc
        print(f"{url}: {failure}; retrying in {wait} s", flush=True)
        sleep(wait)
    raise AssertionError("unreachable")


def fetch(
    target_dir: str | Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    for name, (size, sha256) in FILES.items():
        data = _download(file_url(name), opener, sleep)
        digest = hashlib.sha256(data).hexdigest()
        if (len(data), digest) != (size, sha256):
            raise RuntimeError(
                f"{REPOSITORY}@{REVISION} {name}: got {len(data)} bytes with SHA-256 {digest}, "
                f"expected {size} bytes with {sha256}"
            )
        target = Path(target_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"{name}: {size} bytes, SHA-256 {sha256}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    fetch(sys.argv[1])
