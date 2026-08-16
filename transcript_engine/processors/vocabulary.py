"""
Vocabulary Correction Processor

Performs context-aware alias → canonical term substitutions using:
  - Profile-driven vocabulary entries with confidence thresholds
  - Context words to raise or lower effective confidence
  - case-insensitive regex with word boundaries
  - rapidfuzz for fuzzy matching on entries marked fuzzy=True

Rules are loaded from a Profile object. No code changes needed to add terms.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rapidfuzz import fuzz

from transcript_engine.logging import get_logger
from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.registry import register
from transcript_engine.profiles.model import Profile, VocabularyEntry

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_APPLY_THRESHOLD = 0.8
_FUZZY_THRESHOLD = 88
_MIN_FUZZY_LENGTH = 6


@register
class VocabularyCorrectionProcessor:
    """
    Replaces known mis-transcriptions with canonical terms using a Profile.

    Context words, when defined, adjust effective confidence:
    - context defined and hits found  → confidence boosted slightly
    - context defined but none found  → confidence drops to 40% (usually below threshold)
    - no context defined              → base confidence used as-is
    """

    name: ClassVar[str] = "vocabulary_correction"

    def __init__(self, profile: Profile) -> None:
        self._profile = profile
        self._entries = list(profile.vocabulary)
        self._compiled = self._compile_patterns()
        logger.info(
            f"Vocabulary loaded: {len(self._entries)} terms from profile '{profile.name}'"
        )

    @classmethod
    def from_vocabulary_file(
        cls, vocab_file: Path, profile_name: str = "custom"
    ) -> VocabularyCorrectionProcessor:
        """Load from a raw vocabulary YAML file (legacy/test compatibility)."""
        from transcript_engine.profiles.loader import _load_vocab_entries

        entries = _load_vocab_entries(vocab_file)
        profile = Profile(name=profile_name, vocabulary=tuple(entries))
        return cls(profile)

    def process(self, transcript: Transcript, ctx: ProcessorContext) -> Transcript:
        corrected_segments = tuple(
            self._correct_segment(seg, ctx) for seg in transcript.segments
        )
        return transcript.with_segments(corrected_segments)

    def _correct_segment(self, segment: Segment, ctx: ProcessorContext) -> Segment:
        current_text = segment.text

        for entry in self._entries:
            for alias, pattern in self._compiled.get(entry.canonical, []):
                match = pattern.search(current_text)
                if not match:
                    continue

                effective_conf = self._effective_confidence(entry, current_text)
                if effective_conf < _APPLY_THRESHOLD:
                    logger.debug(
                        f"Skipped '{alias}' → '{entry.canonical}' "
                        f"(confidence={effective_conf:.2f} < {_APPLY_THRESHOLD})"
                    )
                    continue

                original_match = match.group(0)
                new_text = pattern.sub(entry.canonical, current_text)
                if new_text != current_text:
                    ctx.corrections.append(
                        CorrectionRecord(
                            original=original_match,
                            replacement=entry.canonical,
                            reason=self._correction_reason(entry, alias, current_text),
                            confidence=round(effective_conf, 3),
                            profile=self._profile.name,
                            segment_id=segment.id,
                        )
                    )
                    current_text = new_text

        if current_text == segment.text:
            return segment

        corrected_words = self._align_words(segment, current_text)
        return segment.model_copy(
            update={"text": current_text, "words": corrected_words}
        )

    def _effective_confidence(self, entry: VocabularyEntry, segment_text: str) -> float:
        if not entry.context:
            return entry.confidence
        text_lower = segment_text.lower()
        hits = sum(1 for c in entry.context if c.lower() in text_lower)
        if hits == 0:
            return entry.confidence * 0.4
        return min(1.0, entry.confidence + 0.02 * hits)

    def _correction_reason(
        self, entry: VocabularyEntry, alias: str, segment_text: str
    ) -> str:
        if entry.context:
            text_lower = segment_text.lower()
            matched_ctx = [c for c in entry.context if c.lower() in text_lower]
            if matched_ctx:
                return (
                    f"Alias '{alias}' matched with context [{', '.join(matched_ctx)}]"
                )
        return f"Alias '{alias}' matched"

    def _align_words(self, segment: Segment, new_text: str) -> tuple[Word, ...]:
        """
        Rebuild word tuple after text correction, preserving timing where possible.
        Same-count edits update word text in-place; count changes interpolate timing.
        """
        new_tokens = new_text.split()
        old_words = list(segment.words)

        if len(new_tokens) == len(old_words):
            return tuple(
                w.model_copy(update={"text": t}) if w.text != t else w
                for w, t in zip(old_words, new_tokens, strict=False)
            )

        if not old_words:
            return ()

        start = old_words[0].start
        end = old_words[-1].end
        step = (end - start) / max(len(new_tokens), 1)
        old_count = len(old_words)
        new_count = len(new_tokens)
        return tuple(
            Word(
                text=token,
                start=round(start + i * step, 3),
                end=round(start + (i + 1) * step, 3),
                speaker_id=old_words[0].speaker_id,
                # Carry confidence from the positionally-closest old word.
                confidence=old_words[
                    round(i / max(new_count - 1, 1) * max(old_count - 1, 0))
                ].confidence,
            )
            for i, token in enumerate(new_tokens)
        )

    def correct_text(self, text: str) -> str:
        """Apply all corrections to an arbitrary string (no context check, base confidence)."""
        for entry in self._entries:
            for _alias, pattern in self._compiled.get(entry.canonical, []):
                text = pattern.sub(entry.canonical, text)
        return text

    def fuzzy_correct(self, text: str) -> str:
        """Apply fuzzy matching for entries marked fuzzy=True."""
        for entry in self._entries:
            if not entry.fuzzy:
                continue
            for alias in entry.aliases:
                if len(alias) < _MIN_FUZZY_LENGTH:
                    continue
                score = fuzz.ratio(text.lower(), alias.lower())
                if score >= _FUZZY_THRESHOLD:
                    logger.debug(
                        f"Fuzzy match: '{text}' → '{entry.canonical}' (score={score})"
                    )
                    return entry.canonical
        return text

    def _compile_patterns(
        self,
    ) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
        result: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for entry in self._entries:
            patterns: list[tuple[str, re.Pattern[str]]] = []
            for alias in sorted(entry.aliases, key=len, reverse=True):
                flags = re.IGNORECASE if not entry.case_sensitive else 0
                pattern = re.compile(rf"\b{re.escape(alias)}\b", flags)
                patterns.append((alias, pattern))
            result[entry.canonical] = patterns
        return result
