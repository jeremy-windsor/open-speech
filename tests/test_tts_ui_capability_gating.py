"""Tests for TTS UI capability-based feature gating in src/static/index.html."""

from __future__ import annotations

from pathlib import Path


def _index_html() -> str:
    return Path("src/static/index.html").read_text(encoding="utf-8")


def test_tts_capability_control_groups_exist():
    html = _index_html()
    # Capability-gated controls expected in the UI
    for control_id in (
        "tts-voice-blend-group",
        "tts-voice-design-group",
        "tts-voice-clone-group",
        "tts-stream-group",
        "tts-instructions-group",
    ):
        assert f'id="{control_id}"' in html


def test_apply_tts_capabilities_gates_feature_visibility():
    html = _index_html()

    # Ensure applyTTSCapabilities computes booleans from capabilities
    for flag in (
        "const supportsVoiceBlend = currentTTSCapabilities.voice_blend === true;",
        "const supportsVoiceDesign = currentTTSCapabilities.voice_design === true;",
        "const supportsVoiceClone = currentTTSCapabilities.voice_clone === true;",
        "const supportsStreaming = currentTTSCapabilities.streaming === true;",
        "const supportsInstructions = currentTTSCapabilities.instructions === true;",
    ):
        assert flag in html

    # Ensure each UI group is toggled by its capability
    for snippet in (
        "voiceBlendGroup.style.display = supportsVoiceBlend ? '' : 'none';",
        "voiceDesignGroup.style.display = supportsVoiceDesign ? '' : 'none';",
        "voiceCloneGroup.style.display = supportsVoiceClone ? '' : 'none';",
        "streamGroup.style.display = supportsStreaming ? '' : 'none';",
        "instructionsGroup.style.display = supportsInstructions ? '' : 'none';",
    ):
        assert snippet in html
