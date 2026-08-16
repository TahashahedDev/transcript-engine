"""Frozen data models for meeting intelligence extraction results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractedItem:
    text: str  # original excerpt from the transcript
    speaker_id: str  # speaker who said it (internal ID)
    timestamp: float  # approximate start time in seconds
    confidence: float  # 0.0–1.0
    reason: str  # why this item was extracted


@dataclass(frozen=True)
class ActionItem(ExtractedItem):
    task: str
    owner: str | None = None  # None = unknown
    due_date: str | None = None  # None = not mentioned


@dataclass(frozen=True)
class Decision(ExtractedItem):
    decision: str
    rationale: str | None = None


@dataclass(frozen=True)
class Question(ExtractedItem):
    question: str
    is_answered: bool = False


@dataclass(frozen=True)
class TimelineEvent:
    timestamp: float
    event: str
    speaker_id: str | None = None


@dataclass(frozen=True)
class EntityCollection:
    people: tuple[str, ...] = ()
    organizations: tuple[str, ...] = ()
    acronyms: tuple[str, ...] = ()
    dollar_amounts: tuple[str, ...] = ()
    percentages: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    loan_ids: tuple[str, ...] = ()
    ticket_numbers: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()

    @property
    def total_count(self) -> int:
        return (
            len(self.people)
            + len(self.organizations)
            + len(self.acronyms)
            + len(self.dollar_amounts)
            + len(self.percentages)
            + len(self.dates)
            + len(self.loan_ids)
            + len(self.ticket_numbers)
            + len(self.urls)
            + len(self.emails)
        )


@dataclass(frozen=True)
class Topic:
    start: float
    end: float
    label: str
    keywords: tuple[str, ...] = ()


@dataclass
class IntelligenceResult:
    summary_bullets: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    entities: EntityCollection = field(default_factory=EntityCollection)
    topics: list[Topic] = field(default_factory=list)
    ai_enhanced: bool = False
