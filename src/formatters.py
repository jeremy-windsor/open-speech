"""Transcript format converters — SRT, VTT, plain text."""

from __future__ import annotations

from typing import Any


def _split_ms(seconds: float) -> tuple[int, int, int, int]:
    # Round once in whole milliseconds, as whisper.utils.format_timestamp does;
    # truncating (2.3 % 1) * 1000 gives 299 because of binary float error.
    total_ms = round(seconds * 1000)
    h, total_ms = divmod(total_ms, 3_600_000)
    m, total_ms = divmod(total_ms, 60_000)
    s, ms = divmod(total_ms, 1000)
    return h, m, s, ms


def _fmt_time_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h, m, s, ms = _split_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_time_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h, m, s, ms = _split_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _speaker_lines(segments: list[dict[str, Any]]) -> str:
    """One "SPEAKER: text" line per run of segments from the same speaker."""
    lines: list[list[str]] = []
    speaker = None
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if not lines or seg.get("speaker") != speaker:
            speaker = seg.get("speaker")
            lines.append([f"{speaker}:"] if speaker else [])
        lines[-1].append(text)
    return "\n".join(" ".join(words) for words in lines)


def _vtt_escape(text: str) -> str:
    """WebVTT cue text: "&" starts an entity and "<" a tag; "-->" becomes "--&gt;"."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cue_text(text: str, speaker: str | None, *, vtt: bool) -> str:
    # A blank line ends a cue, and a cue line holding "-->" reads as a timing line.
    text = "\n".join(line for line in text.splitlines() if line.strip())
    if vtt:
        text = _vtt_escape(text)
        # WebVTT names the speaker with a voice span.
        return f"<v {_vtt_escape(speaker)}>{text}" if speaker else text
    # SRT has no escapes (players would show "&lt;"), nor markup for the speaker.
    text = f"{speaker}: {text}" if speaker else text
    return text.replace("-->", "->")  # as whisper.utils writes it


def format_as_text(result: dict[str, Any]) -> str:
    """Extract plain text from transcription result, one line per speaker when diarized."""
    segments = result.get("segments") or []
    if any(seg.get("speaker") for seg in segments):
        return _speaker_lines(segments)
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
        return f"1\n{_fmt_time_srt(0)} --> {_fmt_time_srt(duration)}\n{_cue_text(text, None, vtt=False)}\n"

    lines = []
    cue = 0
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if not text:
            continue
        cue += 1
        lines.append(f"{cue}")
        lines.append(f"{_fmt_time_srt(start)} --> {_fmt_time_srt(end)}")
        lines.append(_cue_text(text, seg.get("speaker"), vtt=False))
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
        lines.append(_cue_text(text, None, vtt=True))
        lines.append("")
        return "\n".join(lines)

    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{_fmt_time_vtt(start)} --> {_fmt_time_vtt(end)}")
        lines.append(_cue_text(text, seg.get("speaker"), vtt=True))
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
