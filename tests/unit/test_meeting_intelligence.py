"""Unit tests for the Meeting Intelligence module (Phase 5)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.engine import MeetingIntelligenceEngine
from transcript_engine.intelligence.exports import (
    export_action_items,
    export_decisions,
    export_entities,
    export_metrics,
    export_questions,
    export_summary,
    export_timeline,
)
from transcript_engine.intelligence.extractors.action_items import ActionItemExtractor
from transcript_engine.intelligence.extractors.decisions import DecisionExtractor
from transcript_engine.intelligence.extractors.entities import EntityExtractor
from transcript_engine.intelligence.extractors.questions import QuestionExtractor
from transcript_engine.intelligence.extractors.summary import SummaryGenerator
from transcript_engine.intelligence.extractors.timeline import TimelineExtractor
from transcript_engine.intelligence.models import (
    ActionItem,
    EntityCollection,
    IntelligenceResult,
)
from transcript_engine.models.transcript import Segment, Transcript, Word

# ── Helpers ───────────────────────────────────────────────────────────────────


def _word(text: str, start: float = 0.0, end: float = 0.5) -> Word:
    return Word(text=text, start=start, end=end, speaker_id="SPEAKER_00")


def _seg(
    text: str, speaker: str = "SPEAKER_00", start: float = 0.0, end: float = 5.0
) -> Segment:
    words = tuple(
        Word(
            text=w, start=start + i * 0.3, end=start + i * 0.3 + 0.3, speaker_id=speaker
        )
        for i, w in enumerate(text.split())
    )
    return Segment(
        id=f"seg_{start}",
        speaker_id=speaker,
        start=start,
        end=end,
        words=words,
        text=text,
    )


def _transcript(
    segments: list[Segment], speakers: dict[str, str] | None = None
) -> Transcript:
    return Transcript(
        audio_path=Path("test.wav"),
        language="en",
        segments=tuple(segments),
        speakers=speakers or {"SPEAKER_00": "Speaker 1"},
        duration=segments[-1].end if segments else 10.0,
    )


_CTX = IntelligenceContext()


# ── Models ────────────────────────────────────────────────────────────────────


def test_intelligence_result_defaults() -> None:
    r = IntelligenceResult()
    assert r.action_items == []
    assert r.decisions == []
    assert r.questions == []
    assert r.timeline == []
    assert r.ai_enhanced is False
    assert isinstance(r.entities, EntityCollection)


def test_entity_collection_immutable() -> None:
    ec = EntityCollection(people=("John",))
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        ec.people = ("Neel",)  # type: ignore[misc]


def test_action_item_fields() -> None:
    ai = ActionItem(
        text="I'll follow up",
        speaker_id="SPEAKER_00",
        timestamp=10.0,
        confidence=0.9,
        reason="First-person commitment",
        task="follow up",
        owner="Speaker 1",
        due_date="tomorrow",
    )
    assert ai.task == "follow up"
    assert ai.owner == "Speaker 1"
    assert ai.due_date == "tomorrow"


# ── ActionItemExtractor ───────────────────────────────────────────────────────


def test_action_item_ill_follow_up() -> None:
    seg = _seg("I'll follow up on the loan application.")
    t = _transcript([seg])
    items = ActionItemExtractor().extract(t, _CTX)
    assert len(items) >= 1
    assert items[0].owner == "Speaker 1"
    assert items[0].confidence >= 0.88


def test_action_item_i_will_send() -> None:
    seg = _seg("I will send the report to the team.")
    t = _transcript([seg])
    items = ActionItemExtractor().extract(t, _CTX)
    assert len(items) >= 1
    assert items[0].confidence >= 0.88
    assert "send" in items[0].task.lower() or "report" in items[0].task.lower()


def test_action_item_can_you() -> None:
    seg = _seg("Can you review the document before Friday?")
    t = _transcript([seg])
    items = ActionItemExtractor().extract(t, _CTX)
    assert len(items) >= 1
    assert items[0].owner is None  # addressee unknown


def test_action_item_we_need_to() -> None:
    seg = _seg("We need to validate the APPL_ID before closing.")
    t = _transcript([seg])
    items = ActionItemExtractor().extract(t, _CTX)
    assert len(items) >= 1
    assert items[0].owner is None  # team task


def test_action_item_due_date_extracted() -> None:
    seg = _seg("I'll do it by Friday.")
    t = _transcript([seg])
    items = ActionItemExtractor().extract(t, _CTX)
    assert len(items) >= 1
    assert items[0].due_date is not None
    assert "friday" in items[0].due_date.lower()


def test_action_item_no_plain_statement() -> None:
    seg = _seg("The report was sent yesterday.")
    t = _transcript([seg])
    items = ActionItemExtractor().extract(t, _CTX)
    assert len(items) == 0


# ── DecisionExtractor ─────────────────────────────────────────────────────────


def test_decision_we_decided() -> None:
    seg = _seg("We decided to use the banking profile.")
    t = _transcript([seg])
    decisions = DecisionExtractor().extract(t, _CTX)
    assert len(decisions) >= 1
    assert decisions[0].confidence >= 0.90


def test_decision_lets_go_with() -> None:
    seg = _seg("Let's go with the generic profile for now.")
    t = _transcript([seg])
    decisions = DecisionExtractor().extract(t, _CTX)
    assert len(decisions) >= 1


def test_decision_we_agreed() -> None:
    seg = _seg("We've agreed to move forward with the new system.")
    t = _transcript([seg])
    decisions = DecisionExtractor().extract(t, _CTX)
    assert len(decisions) >= 1


def test_no_decision_from_might() -> None:
    seg = _seg("We might consider using a different approach.")
    t = _transcript([seg])
    decisions = DecisionExtractor().extract(t, _CTX)
    assert len(decisions) == 0


# ── QuestionExtractor ─────────────────────────────────────────────────────────


def test_question_detected() -> None:
    seg = _seg("Is the APPL_ID correct?")
    t = _transcript([seg])
    questions = QuestionExtractor().extract(t, _CTX)
    assert len(questions) >= 1
    assert questions[0].confidence == 0.95


def test_question_what_is() -> None:
    seg = _seg("What is the loan number?")
    t = _transcript([seg])
    questions = QuestionExtractor().extract(t, _CTX)
    assert len(questions) >= 1


def test_no_question_from_statement() -> None:
    seg = _seg("The loan number is LN12345.")
    t = _transcript([seg])
    questions = QuestionExtractor().extract(t, _CTX)
    assert len(questions) == 0


# ── EntityExtractor ───────────────────────────────────────────────────────────


def test_entity_dollar_amount() -> None:
    seg = _seg("The loan is for $450,000.")
    t = _transcript([seg])
    entities = EntityExtractor().extract(t, _CTX)
    assert any("450,000" in d for d in entities.dollar_amounts)


def test_entity_acronym() -> None:
    seg = _seg("We checked the APPL_ID in the system.")
    t = _transcript([seg])
    entities = EntityExtractor().extract(t, _CTX)
    assert "APPL_ID" in entities.acronyms


def test_entity_loan_id() -> None:
    seg = _seg("The file is LN12345.")
    t = _transcript([seg])
    entities = EntityExtractor().extract(t, _CTX)
    assert any("12345" in lid for lid in entities.loan_ids)


def test_entity_percentage() -> None:
    seg = _seg("The interest rate is 50%.")
    t = _transcript([seg])
    entities = EntityExtractor().extract(t, _CTX)
    assert any("50" in p for p in entities.percentages)


def test_entity_date() -> None:
    seg = _seg("The closing is on January 15th.")
    t = _transcript([seg])
    entities = EntityExtractor().extract(t, _CTX)
    assert any("January" in d for d in entities.dates)


# ── SummaryGenerator ──────────────────────────────────────────────────────────


def test_summary_returns_list() -> None:
    seg = _seg("We agreed on the approach and decided to proceed.")
    t = _transcript([seg])
    bullets = SummaryGenerator().extract(t, _CTX)
    assert isinstance(bullets, list)


def test_summary_max_10_bullets() -> None:
    segs = [
        _seg(f"We discussed topic {i}.", start=float(i), end=float(i + 1))
        for i in range(20)
    ]
    t = _transcript(segs)
    bullets = SummaryGenerator().extract(t, _CTX)
    assert len(bullets) <= 10


def test_summary_no_long_bullets() -> None:
    seg = _seg(
        "We had a very long meeting about many important topics including banking and profiles."
    )
    t = _transcript([seg])
    bullets = SummaryGenerator().extract(t, _CTX)
    for b in bullets:
        assert len(b) <= 200


# ── TimelineExtractor ─────────────────────────────────────────────────────────


def test_timeline_has_meeting_begins() -> None:
    seg = _seg("Hello everyone.", start=0.0, end=5.0)
    t = _transcript([seg])
    events = TimelineExtractor().extract(t, _CTX)
    assert any("begins" in e.event.lower() for e in events)


def test_timeline_has_meeting_ends() -> None:
    seg = _seg("Thanks bye.", start=0.0, end=5.0)
    t = _transcript([seg])
    events = TimelineExtractor().extract(t, _CTX)
    assert any("ends" in e.event.lower() for e in events)


def test_timeline_includes_decisions() -> None:
    seg = _seg("We decided to use the banking profile.", start=10.0, end=15.0)
    t = _transcript([seg])
    decisions = DecisionExtractor().extract(t, _CTX)
    events = TimelineExtractor().extract(t, _CTX, decisions=decisions)
    assert any("Decision" in e.event for e in events)


# ── Exports ───────────────────────────────────────────────────────────────────


def _make_result() -> tuple[IntelligenceResult, Transcript]:
    seg = _seg(
        "I'll follow up. We decided to use banking. Is the APPL_ID correct? $50,000 was mentioned.",
        start=0.0,
        end=30.0,
    )
    t = _transcript([seg])
    engine = MeetingIntelligenceEngine()
    return engine.analyze(t), t


def test_export_summary_header() -> None:
    result, t = _make_result()
    md = export_summary(result, t)
    assert "# Meeting Summary" in md


def test_export_action_items_table() -> None:
    result, t = _make_result()
    md = export_action_items(result, t)
    assert "# Action Items" in md


def test_export_decisions_content() -> None:
    result, t = _make_result()
    md = export_decisions(result, t)
    assert "# Decisions" in md


def test_export_questions_table() -> None:
    result, t = _make_result()
    md = export_questions(result, t)
    assert "# Open Questions" in md


def test_export_entities_valid_json() -> None:
    result, _ = _make_result()
    raw = export_entities(result)
    data = json.loads(raw)
    assert "acronyms" in data
    assert "people" in data
    assert isinstance(data["dollar_amounts"], list)


def test_export_timeline_table() -> None:
    result, t = _make_result()
    md = export_timeline(result, t)
    assert "# Meeting Timeline" in md
    assert "|" in md


def test_export_metrics_valid_json() -> None:
    result, t = _make_result()
    raw = export_metrics(t, 10.0, [], "generic", result)
    data = json.loads(raw)
    required = {
        "audio_file",
        "word_count",
        "speaker_count",
        "action_items",
        "decisions",
        "open_questions",
        "speaking_time",
        "reading_time_minutes",
        "confidence",
    }
    for key in required:
        assert key in data, f"Missing key: {key}"
