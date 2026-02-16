"""Transcript format converters — SRT, VTT, plain text."""

from __future__ import annotations

from typing import Any


def _fmt_time_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_time_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_as_text(result: dict[str, Any]) -> str:
    """Extract plain text from transcription result."""
    return result.get("text", "").strip()


def format_as_srt(result: dict[str, Any]) -> str:
    """Format transcription result as SRT subtitles."""
    segments = result.get("segments", [])
    if not segments:
        # No segments — make a single entry from the full text
        text = result.get("text", "").strip()
        if not text:
            return ""
        duration = result.get("duration", 0.0)
        return f"1\n{_fmt_time_srt(0)} --> {_fmt_time_srt(duration)}\n{text}\n"

    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{i}")
        lines.append(f"{_fmt_time_srt(start)} --> {_fmt_time_srt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def format_as_vtt(result: dict[str, Any]) -> str:
    """Format transcription result as WebVTT subtitles."""
    segments = result.get("segments", [])
    lines = ["WEBVTT", ""]

    if not segments:
        text = result.get("text", "").strip()
        if not text:
            return "WEBVTT\n"
        duration = result.get("duration", 0.0)
        lines.append(f"{_fmt_time_vtt(0)} --> {_fmt_time_vtt(duration)}")
        lines.append(text)
        lines.append("")
        return "\n".join(lines)

    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{_fmt_time_vtt(start)} --> {_fmt_time_vtt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def format_transcription(result: dict[str, Any], response_format: str) -> tuple[str, str]:
    """Format a transcription result.
    
    Returns (content, content_type).
    """
    if response_format == "text":
        return format_as_text(result), "text/plain"
    elif response_format == "srt":
        return format_as_srt(result), "text/plain"
    elif response_format == "vtt":
        return format_as_vtt(result), "text/vtt"
    else:
        # json — return None to signal JSON response
        return "", "application/json"
