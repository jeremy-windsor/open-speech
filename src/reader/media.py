"""Short-lived signed links for rendered Reader audio.

An HTML audio element cannot send an ``Authorization`` header, so part audio is
served from links carrying an expiry and an HMAC signature instead of the API key.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

MEDIA_PREFIX = "/v1/reader/media/"


class MediaSigner:
    def __init__(self, api_key: str = "") -> None:
        # Without an API key the links only need to be unguessable for this process.
        seed = api_key.encode("utf-8") if api_key else secrets.token_bytes(32)
        self._key = hmac.new(seed, b"open-speech-reader-media", hashlib.sha256).digest()

    def _signature(self, part_id: str, expires: int) -> str:
        message = f"{part_id}:{expires}".encode("utf-8")
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()[:32]

    def url(self, part_id: str, ttl_s: int) -> str:
        expires = int(time.time()) + max(60, ttl_s)
        return f"{MEDIA_PREFIX}{part_id}.mp3?exp={expires}&sig={self._signature(part_id, expires)}"

    def verify(self, part_id: str, expires: int, signature: str) -> bool:
        if expires < time.time():
            return False
        return hmac.compare_digest(self._signature(part_id, expires), signature)
