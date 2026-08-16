"""
Transcript Intelligence Processor

Improves transcript readability without changing meaning:
  - Sentence boundary detection and capitalization
  - Terminal punctuation insertion
  - Standalone "i" → "I" correction

This processor NEVER rewrites, summarizes, or invents content.
All changes are deterministic and rule-based.
Confidence highlighting is handled downstream in the markdown exporter.
"""

from __future__ import annotations

import re
from typing import ClassVar

from transcript_engine.logging import get_logger
from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.registry import register

logger = get_logger(__name__)

# Words that suggest a segment is a question
_QUESTION_STARTERS = frozenset(
    {
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "which",
        "whose",
        "whom",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "have",
        "has",
        "had",
    }
)

_SENTENCE_END = re.compile(r"[.!?]$")
_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")
_STANDALONE_I = re.compile(r"^i$", re.IGNORECASE)

# Filler words that should not receive terminal punctuation
_FILLERS = frozenset({"mm", "uh", "um", "hmm", "hm", "ah", "oh"})


@register
class TranscriptIntelligenceProcessor:
    """
    Rule-based readability improvements applied before speaker formatting.
    Updates Word.text values so changes survive segment merging.
    """

    name: ClassVar[str] = "transcript_intelligence"

    def __init__(self, confidence_threshold: float = 0.65) -> None:
        self._confidence_threshold = confidence_threshold

    def process(self, transcript: Transcript, ctx: ProcessorContext) -> Transcript:
        new_segments = tuple(self._process_segment(seg) for seg in transcript.segments)
        changed = sum(
            1
            for old, new in zip(transcript.segments, new_segments, strict=True)
            if old is not new
        )
        if changed:
            logger.info(
                f"Intelligence: improved {changed}/{len(new_segments)} segments"
            )
        return transcript.with_segments(new_segments)

    def _process_segment(self, segment: Segment) -> Segment:
        if not segment.words:
            return segment

        words = list(segment.words)
        words = self._fix_pronouns(words)
        words = self._fix_capitalization(words)
        words = self._ensure_terminal_punctuation(words)

        new_text = _join_words(words)

        text_changed = new_text != segment.text
        words_changed = any(
            w.text != ow.text for w, ow in zip(words, segment.words, strict=True)
        )

        if not text_changed and not words_changed:
            return segment

        return segment.model_copy(update={"words": tuple(words), "text": new_text})

    def _fix_pronouns(self, words: list[Word]) -> list[Word]:
        """Replace standalone 'i' with 'I'."""
        result = []
        for word in words:
            bare = _TRAILING_PUNCT.sub("", word.text)
            if _STANDALONE_I.match(bare) and len(word.text) <= 2:
                suffix = word.text[len(bare) :]
                result.append(word.model_copy(update={"text": "I" + suffix}))
            else:
                result.append(word)
        return result

    def _fix_capitalization(self, words: list[Word]) -> list[Word]:
        """
        Capitalize the first word of the segment and any word
        immediately following a sentence-ending token (.!?).
        """
        if not words:
            return words

        result = list(words)
        capitalize_next = True

        for i, word in enumerate(result):
            if capitalize_next:
                new_text = _capitalize_word(word.text)
                if new_text != word.text:
                    result[i] = word.model_copy(update={"text": new_text})
                capitalize_next = False

            if _SENTENCE_END.search(word.text):
                capitalize_next = True

        return result

    def _ensure_terminal_punctuation(self, words: list[Word]) -> list[Word]:
        """Add a period or question mark to the last word if it has no terminal punctuation."""
        if not words:
            return words

        last = words[-1]
        if _SENTENCE_END.search(last.text):
            return words

        bare_last = _TRAILING_PUNCT.sub("", last.text).lower()
        if bare_last in _FILLERS or len(bare_last) <= 1:
            return words

        first_bare = _TRAILING_PUNCT.sub("", words[0].text).lower()
        punct = "?" if first_bare in _QUESTION_STARTERS else "."

        new_last = last.model_copy(update={"text": last.text + punct})
        return words[:-1] + [new_last]


def _capitalize_word(text: str) -> str:
    """Capitalize the first alphabetic character, preserving any leading punctuation."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1 :]
    return text


def _join_words(words: list[Word]) -> str:
    """Join word texts with spaces, but don't add space before punctuation-only tokens."""
    if not words:
        return ""
    parts = [words[0].text]
    for word in words[1:]:
        if word.text and word.text[0] in ".,;:!?":
            parts.append(word.text)
        else:
            parts.append(" " + word.text)
    return "".join(parts).strip()
