"""
Transcript Cleanup Processor

Removes transcription artifacts without altering meaning:
  - Duplicate consecutive words ("I I I", "the the", "right right")
  - Stutter patterns ("I- I- I think")
  - Broken punctuation and extra whitespace
  - Filler artifacts that appear as duplicates

Does NOT:
  - Rewrite grammar
  - Summarize
  - Change meaning
  - Remove deliberate repetition used for emphasis
"""

from __future__ import annotations

import re
from typing import ClassVar

from transcript_engine.logging import get_logger
from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.registry import register

logger = get_logger(__name__)

_DUPLICATE_WORD = re.compile(r"\b(\w+)(\s+\1){2,}\b", re.IGNORECASE)
_DOUBLE_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
_STUTTER = re.compile(r"\b(\w+)-\s+\1\b", re.IGNORECASE)
_MULTI_SPACE = re.compile(r" {2,}")
_BROKEN_PUNCT = re.compile(r"\s+([,.!?;:])")
_LEADING_PUNCT = re.compile(r"^[,\.]+\s*")

_EMPHASIS_WORDS = frozenset({"very", "really", "no", "yes", "absolutely", "definitely"})
_MIN_DUPLICATE_LEN = 4


@register
class TranscriptCleanupProcessor:
    """
    Cleans transcription artifacts from segment text and word lists.
    Operates segment by segment; does not cross segment boundaries.
    """

    name: ClassVar[str] = "transcript_cleanup"

    def __init__(self) -> None:
        pass

    def process(self, transcript: Transcript, ctx: ProcessorContext) -> Transcript:
        cleaned_segments = tuple(
            self._clean_segment(seg) for seg in transcript.segments
        )
        return transcript.with_segments(cleaned_segments)

    def _clean_segment(self, segment: Segment) -> Segment:
        cleaned_words = self._deduplicate_words(list(segment.words))
        new_text = self._clean_text(" ".join(w.text for w in cleaned_words))

        if new_text == segment.text and len(cleaned_words) == len(segment.words):
            return segment

        return segment.model_copy(
            update={"words": tuple(cleaned_words), "text": new_text}
        )

    def _deduplicate_words(self, words: list[Word]) -> list[Word]:
        """
        Remove immediately repeated words unless they are emphasis words
        or the repetition looks intentional (single word, short).
        """
        if len(words) <= 1:
            return words

        result: list[Word] = [words[0]]
        for word in words[1:]:
            prev = result[-1]
            if self._is_duplicate(prev.text, word.text):
                logger.debug(f"Removed duplicate word: '{word.text}'")
                continue
            result.append(word)

        return result

    def _is_duplicate(self, prev: str, current: str) -> bool:
        if prev.lower() != current.lower():
            return False
        if current.lower() in _EMPHASIS_WORDS:
            return False
        return not len(current) < _MIN_DUPLICATE_LEN

    def _clean_text(self, text: str) -> str:
        text = _STUTTER.sub(r"\1", text)
        text = _DUPLICATE_WORD.sub(r"\1", text)
        text = _DOUBLE_WORD.sub(
            lambda m: (
                m.group(0) if m.group(1).lower() in _EMPHASIS_WORDS else m.group(1)
            ),
            text,
        )
        text = _BROKEN_PUNCT.sub(r"\1", text)
        text = _LEADING_PUNCT.sub("", text)
        text = _MULTI_SPACE.sub(" ", text)
        return text.strip()
