# Open Speech — Fixes & Feature Requests

Quick intake for bugs, fixes, and feature ideas. Will triages, Forge builds.

For bigger items, open a [GitHub Issue](https://github.com/will-assistant/open-speech/issues).

## Status Key
- 🔴 Open
- 🟡 In Progress (dispatched to Forge)
- 🟢 Done (commit linked)

---

## Bugs

| # | Description | Status | Commit |
|---|-------------|--------|--------|
| B1 | Mic transcription captures nothing (WebSocket/format issue) | 🟡 | c128533 (partial) |

## Fixes

| # | Description | Status | Commit |
|---|-------------|--------|--------|
| F1 | Speed slider 0.25 step → 5% increments | 🟢 | 75fb457 |
| F2 | Kokoro showing in STT dropdown | 🟢 | c128533 |
| F3 | Kokoro-82M listed as STT in Models tab | 🟢 | c128533 |
| F4 | Moonshine models show Download but provider not installed | 🟢 | c128533 |
| F5 | Version badge showed v1.0 | 🟢 | c128533 |
| F6 | Voice presets didn't match actual voices | 🟢 | 75fb457 |
| F7 | Vosk Zip Slip safe extraction + validation | 🟢 | pending |
| F8 | Realtime audio buffer limit + idle timeout protections | 🟢 | pending |
| F9 | Auth hardening (`OS_AUTH_REQUIRED`, startup warning, query-key deprecation) | 🟢 | pending |
| F10 | WS origin allowlist + Wyoming localhost default bind | 🟢 | pending |
| F11 | Voice clone upload size limit + TLS cert dir hardening | 🟢 | pending |
| F12 | Docker non-root user + cache/cert path updates | 🟢 | pending |
| F13 | Model manager concurrency locks + realtime model resolution fix | 🟢 | pending |

## Features

| # | Description | Status | Commit |
|---|-------------|--------|--------|
| T1 | TTS history — download + delete buttons | 🟢 | c128533 |
| T2 | Stream toggle tooltip | 🟢 | c128533 |
| T3 | Voice presets dropdown in Speak tab | 🟢 | 0ea4d4c |
| T4 | Voice cloning endpoint (`/v1/audio/speech/clone`) | 🟢 | 0ea4d4c |
| T5 | Qwen3-TTS backend | 🟢 | 0ea4d4c |
| T6 | Fish Speech backend | 🟢 | 0ea4d4c |

---

*To add: just tell Will in Discord. He'll add it here and batch dispatch to Forge.*
