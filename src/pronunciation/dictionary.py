from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


class PronunciationDictionary:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._entries: dict[str, str] = {}
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self._entries = {}
            return
        text = p.read_text()
        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        self._entries = {str(k): str(v) for k, v in (data or {}).items()}

    @property
    def entries(self) -> dict[str, str]:
        return dict(self._entries)

    def apply(self, text: str) -> str:
        out = text
        for src, dst in sorted(self._entries.items(), key=lambda kv: len(kv[0]), reverse=True):
            out = re.sub(re.escape(src), dst, out)
        return out


def parse_ssml(input_text: str) -> str:
    # basic subset handling without strict XML parser fragility
    text = input_text
    text = re.sub(r"<break\s+time=\"(\d+)ms\"\s*/>", lambda m: " " + ("." * max(1, int(m.group(1)) // 250)) + " ", text)
    text = re.sub(r"</?emphasis[^>]*>", "", text)
    text = re.sub(r"<phoneme[^>]*>(.*?)</phoneme>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"</?speak[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()
