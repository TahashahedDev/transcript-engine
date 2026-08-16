"""Action item extractor — finds commitments and assigned tasks."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import ClassVar

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.models import ActionItem
from transcript_engine.models.transcript import Segment, Transcript

# ── Due-date patterns (applied to the sentence containing the action) ─────────

_DUE_WEEKDAY = re.compile(
    r"\bby\s+((?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b",
    re.IGNORECASE,
)
_DUE_MONTH_DATE = re.compile(
    r"\bby\s+((?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
    re.IGNORECASE,
)
_DUE_THIS_NEXT = re.compile(r"\b(this|next)\s+(week|month)\b", re.IGNORECASE)
_DUE_TOMORROW = re.compile(r"\btomorrow\b", re.IGNORECASE)
_DUE_TODAY = re.compile(r"\btoday\b", re.IGNORECASE)
_DUE_EOD = re.compile(r"\bby\s+EOD\b", re.IGNORECASE)

# ── Action trigger patterns ───────────────────────────────────────────────────

_PATTERNS: list[tuple[re.Pattern[str], str | None, float, str]] = [
    # (pattern, owner_token, confidence, reason)
    # STRONG — first-person commitments
    (
        re.compile(r"\bi'?ll\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        "SPEAKER",
        0.90,
        "First-person commitment",
    ),
    (
        re.compile(r"\bi will\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        "SPEAKER",
        0.92,
        "First-person commitment",
    ),
    (
        re.compile(r"\bi'?m going to\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        "SPEAKER",
        0.88,
        "First-person commitment",
    ),
    (
        re.compile(r"\baction item\s*:?\s*(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.95,
        "Explicit action item marker",
    ),
    # MODERATE — requests / team tasks
    (
        re.compile(r"\bcan you\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.82,
        "Request to addressee",
    ),
    (
        re.compile(r"\bwe need to\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.75,
        "Team task",
    ),
    (
        re.compile(r"\bfollow up\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        "SPEAKER",
        0.80,
        "Follow-up item",
    ),
    (
        re.compile(r"\bplease\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.78,
        "Polite request",
    ),
    # LOWER — suggestions and implied tasks
    (
        re.compile(r"\blet'?s\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.68,
        "Team suggestion",
    ),
    (
        re.compile(r"\bwe should\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.73,
        "Team recommendation",
    ),
    (
        re.compile(
            r"\bnext steps?\s*(?:is|are|:)\s*(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE
        ),
        None,
        0.85,
        "Explicit next step",
    ),
    (
        re.compile(r"\byou need to\b\s+(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.79,
        "Directed task",
    ),
    (
        re.compile(r"\bmake sure\b\s+(?:to\s+)?(.{5,100}?)(?=[.!?]|$)", re.IGNORECASE),
        None,
        0.77,
        "Reminder task",
    ),
]


class ActionItemExtractor:
    name: ClassVar[str] = "action_items"

    def extract(
        self, transcript: Transcript, ctx: IntelligenceContext
    ) -> list[ActionItem]:
        results: list[ActionItem] = []
        seen_tasks: set[str] = set()

        for segment in transcript.segments:
            for sentence, ts in _iter_sentences(segment):
                for pattern, owner_token, confidence, reason in _PATTERNS:
                    m = pattern.search(sentence)
                    if not m:
                        continue
                    task = m.group(1).strip().rstrip(".,;:")
                    task_key = task.lower()
                    if task_key in seen_tasks or len(task) < 5:
                        continue
                    seen_tasks.add(task_key)

                    owner: str | None = None
                    if owner_token == "SPEAKER":
                        owner = transcript.display_name(segment.speaker_id)

                    due_date = _extract_due_date(sentence)

                    results.append(
                        ActionItem(
                            text=sentence[:200],
                            speaker_id=segment.speaker_id,
                            timestamp=ts,
                            confidence=confidence,
                            reason=reason,
                            task=task,
                            owner=owner,
                            due_date=due_date,
                        )
                    )
                    break  # only first matching pattern per sentence

        return results


def _extract_due_date(sentence: str) -> str | None:
    if m := _DUE_WEEKDAY.search(sentence):
        return m.group(1)
    if m := _DUE_MONTH_DATE.search(sentence):
        return m.group(1)
    if m := _DUE_THIS_NEXT.search(sentence):
        return f"{m.group(1)} {m.group(2)}"
    if _DUE_EOD.search(sentence):
        return "end of day"
    if _DUE_TOMORROW.search(sentence):
        return "tomorrow"
    if _DUE_TODAY.search(sentence):
        return "today"
    return None


def _iter_sentences(segment: Segment) -> Iterator[tuple[str, float]]:
    """Yield (sentence_text, start_timestamp) for each sentence in the segment."""
    current: list[str] = []
    start = segment.start
    for word in segment.words:
        current.append(word.text)
        if re.search(r"[.!?]$", word.text):
            yield " ".join(current).strip(), start
            current = []
            start = word.end
    if current:
        yield " ".join(current).strip(), start
