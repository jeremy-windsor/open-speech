# Post-Phase 4 Fixes

Bugs and UX issues found during testing on Windows GPU box.

## UI Fixes

### 1. Speed slider — 10% increments
- Current: 0.25 step. Wanted: 0.1 step (0.9x, 1.0x, 1.1x...)

### 2. Recent generations — delete + download buttons
- Each history item needs delete and download buttons

### 3. Kokoro in STT dropdown (Transcribe tab)
- Bug: Kokoro appears in Transcribe model selector. It's TTS only.

### 4. Microphone transcription not working
- Mic button captures nothing, no text output. Check WebSocket, audio format, HTTPS context.

### 5. Stream toggle — add tooltip
- Explain: "Start playback before generation finishes (lower latency for long text)"

## Models Tab Fixes

### 6. Kokoro-82M showing as STT model
- `hexgrad/Kokoro-82M` appears under STT as "Downloaded" (626MB). It's TTS.

### 7. Moonshine models listed but uninstallable
- moonshine/tiny and moonshine/base show "Download & Load" but moonshine-onnx isn't installed.
- Hide models whose provider isn't installed, or show "Provider not installed" badge.

### 8. Version badge shows v1.0
- Header says v1.0. Should read from pyproject.toml (0.4.0 after Phase 4).
- `/health` also reports 0.1.0.
