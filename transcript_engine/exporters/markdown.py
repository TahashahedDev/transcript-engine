"""
Markdown Exporter

Produces human-readable transcripts with:
  - Speaker block headers (## Speaker Name)
  - Paragraph detection from within-segment pauses
  - Four timestamp modes: none, speaker, paragraph, minute
  - Confidence highlighting for uncertain words ([word?])
"""

from __future__ import annotations

from typing import ClassVar, Literal

from transcript_engine.config.settings import ExportConfig
from transcript_engine.exporters.registry import register
from transcript_engine.models.transcript import Segment, Transcript, Word

TimestampMode = Literal["none", "speaker", "paragraph", "minute"]

# Pause between words (seconds) that triggers a paragraph break within a segment
_PARAGRAPH_PAUSE = 2.5

# Confidence below which a word is highlighted as uncertain
_DEFAULT_LOW_CONF = 0.65


@register
class MarkdownExporter:
    """
    Primary human-readable output format.

    Output structure:
        # Meeting Title

        ## Speaker 1

        [00:00] First paragraph of speech...

        Second paragraph after a pause...

        ## Speaker 2

        Reply text...
    """

    format: ClassVar[str] = "markdown"
    extension: ClassVar[str] = ".md"

    def export(self, transcript: Transcript, config: ExportConfig) -> bytes:
        timestamp_mode: TimestampMode = getattr(config, "timestamp_mode", "none")
        low_conf_threshold: float = getattr(
            config, "low_confidence_threshold", _DEFAULT_LOW_CONF
        )
        highlight_confidence: bool = getattr(config, "highlight_confidence", False)

        title = transcript.audio_path.stem.replace("_", " ").replace("-", " ").title()
        lines: list[str] = [f"# {title}", ""]

        # Minute-boundary tracker (seconds → already inserted)
        next_minute: float = 60.0

        for segment in transcript.segments:
            display = transcript.display_name(segment.speaker_id)

            # ── Speaker header ──────────────────────────────────────────────
            ts_str = _format_ts(segment.start) if timestamp_mode == "speaker" else ""
            header = f"## {display}"
            if ts_str:
                header += f"  {ts_str}"
            lines.append(header)
            lines.append("")

            # ── Split segment into paragraphs ───────────────────────────────
            paragraphs = _split_paragraphs(segment)

            for para_words in paragraphs:
                if not para_words:
                    continue

                para_start = para_words[0].start

                # Minute timestamp injection
                if timestamp_mode == "minute":
                    while para_start >= next_minute:
                        lines.append(f"*[{_format_ts(next_minute)}]*")
                        lines.append("")
                        next_minute += 60.0

                # Paragraph timestamp
                ts_prefix = ""
                if timestamp_mode == "paragraph":
                    ts_prefix = f"{_format_ts(para_start)} "

                # Render words, optionally highlighting low-confidence ones
                para_text = _render_words(
                    para_words, highlight_confidence, low_conf_threshold
                )
                lines.append(ts_prefix + para_text)
                lines.append("")

        content = "\n".join(lines).rstrip() + "\n"
        return content.encode("utf-8")


# ── helpers ──────────────────────────────────────────────────────────────────


def _split_paragraphs(segment: Segment) -> list[list[Word]]:
    """
    Split a segment's word list into paragraphs at long pauses.
    A pause >= _PARAGRAPH_PAUSE seconds between consecutive words starts a new paragraph.
    """
    if not segment.words:
        return []

    paragraphs: list[list[Word]] = []
    current: list[Word] = [segment.words[0]]

    for prev, word in zip(segment.words, segment.words[1:], strict=False):
        gap = word.start - prev.end
        if gap >= _PARAGRAPH_PAUSE:
            paragraphs.append(current)
            current = []
        current.append(word)

    if current:
        paragraphs.append(current)

    return paragraphs


def _render_words(
    words: list[Word],
    highlight: bool,
    threshold: float,
) -> str:
    """Render a list of words into a text string, optionally highlighting low-confidence ones."""
    parts: list[str] = []
    for word in words:
        text = word.text
        if highlight and word.confidence is not None and word.confidence < threshold:
            # Mark uncertain word: **[word?]**
            bare = text.rstrip(".,;:!?")
            suffix = text[len(bare) :]
            text = f"**[{bare}?]**{suffix}"
        parts.append(text)

    # Join words — don't add space before punctuation-only tokens
    if not parts:
        return ""
    result = [parts[0]]
    for part in parts[1:]:
        if part and part[0] in ".,;:!?":
            result.append(part)
        else:
            result.append(" " + part)
    return "".join(result).strip()


def _format_ts(seconds: float) -> str:
    """Format seconds as [MM:SS]."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"[{m:02d}:{s:02d}]"
