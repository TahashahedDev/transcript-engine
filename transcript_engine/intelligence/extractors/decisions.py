"""Decision extractor — finds choices, agreements, and conclusions."""

from __future__ import annotations

import re
from typing import ClassVar

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.extractors.action_items import _iter_sentences
from transcript_engine.intelligence.models import Decision
from transcript_engine.models.transcript import Transcript

_PATTERNS: list[tuple[re.Pattern[str], float, str, int]] = [
    # (pattern, confidence, reason, capture_group_index)
    (
        re.compile(r"\bwe(?:'ve)?\s+decided\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        0.95,
        "Explicit decision",
        1,
    ),
    (
        re.compile(r"\bdecision\s+is\b\s*:?\s*(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        0.92,
        "Decision stated explicitly",
        1,
    ),
    (
        re.compile(r"\bwe(?:'ve)?\s+agreed\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        0.92,
        "Agreement stated",
        1,
    ),
    (
        re.compile(
            r"\bwe(?:'re)?\s+moving\s+forward\s+with\b\s+(.{5,100}?)(?=[.!?]|$)",
            re.IGNORECASE,
        ),
        0.90,
        "Decision to proceed",
        1,
    ),
    (
        re.compile(
            r"\bwe(?:'re)?\s+going\s+with\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE
        ),
        0.90,
        "Choice confirmed",
        1,
    ),
    (
        re.compile(
            r"\bwe(?:'ll)?\s+go\s+with\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE
        ),
        0.88,
        "Choice made",
        1,
    ),
    (
        re.compile(r"\blet'?s\s+go\s+with\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        0.88,
        "Choice made",
        1,
    ),
    (
        re.compile(r"\bwe(?:'ll)?\s+use\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        0.88,
        "Tool/resource chosen",
        1,
    ),
    (
        re.compile(r"\bwe(?:'re)?\s+using\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        0.82,
        "Current approach stated",
        1,
    ),
]


class DecisionExtractor:
    name: ClassVar[str] = "decisions"

    def extract(
        self, transcript: Transcript, ctx: IntelligenceContext
    ) -> list[Decision]:
        results: list[Decision] = []
        seen: set[str] = set()

        for segment in transcript.segments:
            for sentence, ts in _iter_sentences(segment):
                for pattern, confidence, reason, group in _PATTERNS:
                    m = pattern.search(sentence)
                    if not m:
                        continue
                    decision_text = m.group(group).strip().rstrip(".,;:")
                    key = decision_text.lower()
                    if key in seen or len(decision_text) < 5:
                        continue
                    seen.add(key)
                    results.append(
                        Decision(
                            text=sentence[:200],
                            speaker_id=segment.speaker_id,
                            timestamp=ts,
                            confidence=confidence,
                            reason=reason,
                            decision=decision_text,
                        )
                    )
                    break

        return results
