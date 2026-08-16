"""Unit tests for Phase 6: review report, search index, topic detection."""

from __future__ import annotations

import json

import pytest

from tests.conftest import make_segment, make_transcript
from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.engine import MeetingIntelligenceEngine
from transcript_engine.intelligence.extractors.topics import TopicExtractor
from transcript_engine.intelligence.models import (
    ActionItem,
    Decision,
    EntityCollection,
    IntelligenceResult,
    Question,
    Topic,
)
from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Segment, Word
from transcript_engine.review.index import generate_index
from transcript_engine.review.report import generate_review
from transcript_engine.review.search import search_index

# ── fixtures ──────────────────────────────────────────────────────────────────


def _word(
    text: str,
    start: float = 0.0,
    end: float = 0.5,
    confidence: float | None = None,
    speaker_id: str = "SPEAKER_00",
) -> Word:
    return Word(
        text=text, start=start, end=end, speaker_id=speaker_id, confidence=confidence
    )


def _seg(
    words: list[Word],
    seg_id: str = "s0",
    speaker_id: str = "SPEAKER_00",
) -> Segment:
    text = " ".join(w.text for w in words)
    return Segment(
        id=seg_id,
        speaker_id=speaker_id,
        start=words[0].start,
        end=words[-1].end,
        words=tuple(words),
        text=text,
    )


def _intel(
    *,
    action_items: list[ActionItem] | None = None,
    decisions: list[Decision] | None = None,
    questions: list[Question] | None = None,
    topics: list[Topic] | None = None,
) -> IntelligenceResult:
    return IntelligenceResult(
        summary_bullets=[],
        action_items=action_items or [],
        decisions=decisions or [],
        questions=questions or [],
        timeline=[],
        entities=EntityCollection(),
        topics=topics or [],
    )


def _correction(seg_id: str = "s0") -> CorrectionRecord:
    return CorrectionRecord(
        original="Apple ID",
        replacement="APPL_ID",
        reason="vocabulary match",
        confidence=0.95,
        profile="banking",
        segment_id=seg_id,
    )


# ── Topic model ────────────────────────────────────────────────────────────────


def test_topic_is_frozen() -> None:
    t = Topic(start=0.0, end=60.0, label="Discussion about loans")
    with pytest.raises((AttributeError, TypeError)):
        t.label = "changed"  # type: ignore[misc]


def test_topic_has_keywords() -> None:
    t = Topic(start=0.0, end=30.0, label="About APPL_ID", keywords=("appl_id", "loan"))
    assert "appl_id" in t.keywords
    assert "loan" in t.keywords


# ── TopicExtractor ─────────────────────────────────────────────────────────────


def test_topic_single_segment() -> None:
    seg = make_segment("SPEAKER_00", [("Hello", 0.0, 0.5), ("everyone", 0.5, 1.0)])
    t = make_transcript([seg])
    ctx = IntelligenceContext()
    topics = TopicExtractor().extract(t, ctx)
    assert len(topics) == 1
    assert topics[0].start == 0.0


def test_topic_long_pause_creates_two_topics() -> None:
    seg1 = _seg([_word("loan", 0.0, 0.5), _word("servicing", 0.5, 1.0)], "s1")
    # 10 second gap → new topic
    seg2 = _seg([_word("validation", 11.0, 11.5), _word("strategy", 11.5, 12.0)], "s2")
    t = make_transcript([seg1, seg2])
    ctx = IntelligenceContext()
    topics = TopicExtractor().extract(t, ctx)
    assert len(topics) == 2
    assert topics[0].start == 0.0
    assert topics[1].start == 11.0


def test_topic_label_non_empty() -> None:
    seg = make_segment(
        "SPEAKER_00", [("APPL_ID", 0.0, 0.5), ("loan", 0.5, 1.0), ("APPL_ID", 1.0, 1.5)]
    )
    t = make_transcript([seg])
    ctx = IntelligenceContext()
    topics = TopicExtractor().extract(t, ctx)
    assert topics[0].label != ""


def test_topic_keywords_extracted() -> None:
    seg = _seg(
        [
            _word("mortgage", 0.0, 0.3),
            _word("mortgage", 0.3, 0.6),
            _word("mortgage", 0.6, 0.9),
            _word("loan", 0.9, 1.2),
        ],
        "s0",
    )
    t = make_transcript([seg])
    ctx = IntelligenceContext()
    topics = TopicExtractor().extract(t, ctx)
    assert "mortgage" in topics[0].keywords


def test_intelligence_engine_populates_topics() -> None:
    seg = make_segment("SPEAKER_00", [("Hello", 0.0, 0.5), ("world", 0.5, 1.0)])
    t = make_transcript([seg])
    engine = MeetingIntelligenceEngine()
    result = engine.analyze(t)
    assert isinstance(result.topics, list)
    assert len(result.topics) >= 1


# ── generate_review ────────────────────────────────────────────────────────────


def test_review_returns_markdown() -> None:
    t = make_transcript()
    report = generate_review(t, [])
    assert "# Review Report" in report


def test_review_section_headers_present() -> None:
    t = make_transcript()
    report = generate_review(t, [])
    for section in [
        "## 1. Low-Confidence Words",
        "## 2. Low-Confidence Segments",
        "## 3. Corrections Applied",
        "## 4. Potential Proper Nouns",
        "## 5. Numbers & Codes",
        "## 6. Possible Speaker Issues",
    ]:
        assert section in report, f"Missing: {section}"


def test_review_low_conf_word_flagged() -> None:
    w_low = _word("uncertain", 0.0, 0.5, confidence=0.40)
    w_high = _word("clear", 0.5, 1.0, confidence=0.95)
    seg = _seg([w_low, w_high], "s0")
    t = make_transcript([seg])
    report = generate_review(t, [], low_conf_threshold=0.65)
    assert "uncertain" in report
    assert "40%" in report


def test_review_high_conf_word_not_in_low_section() -> None:
    w = _word("confident", 0.0, 0.5, confidence=0.90)
    seg = _seg([w], "s0")
    t = make_transcript([seg])
    report = generate_review(t, [], low_conf_threshold=0.65)
    # The word should NOT appear in the low-confidence table
    # (it still appears in segment text, so we can't just check absence)
    # Verify 0.90 doesn't appear as a confidence value
    assert "90%" not in report or "Low-Confidence Words" not in report.split("90%")[0]


def test_review_no_corrections_message() -> None:
    t = make_transcript()
    report = generate_review(t, [])
    assert "No corrections were applied" in report


def test_review_corrections_appear() -> None:
    w = _word("Apple", 0.0, 0.5)
    seg = _seg([w], "s0")
    t = make_transcript([seg])
    rec = _correction("s0")
    report = generate_review(t, [rec])
    assert "Apple ID" in report
    assert "APPL_ID" in report
    assert "banking" in report


def test_review_dollar_amount_found() -> None:
    w = _word("$50,000", 0.0, 0.5)
    seg = _seg([w], "s0")
    t = make_transcript([seg])
    report = generate_review(t, [])
    assert "$50,000" in report


def test_review_short_segment_flagged() -> None:
    # 0.4s segment → below 1.5s threshold
    w = _word("yes", 0.0, 0.4)
    seg = _seg([w], "s0")
    t = make_transcript([seg])
    report = generate_review(t, [], short_segment_threshold=1.5)
    assert "Short segment" in report


def test_review_no_issues_for_long_segments() -> None:
    seg = make_segment(
        "SPEAKER_00",
        [("Hello", 0.0, 1.0), ("everyone", 1.0, 2.0), ("welcome", 2.0, 3.0)],
    )
    t = make_transcript([seg])
    report = generate_review(t, [], short_segment_threshold=1.5)
    assert "No speaker issues detected" in report


def test_review_alternating_pair_flagged() -> None:
    # 6 alternating segments ABABAB → triggers alternating pair detection
    segs = []
    for i in range(6):
        spk = "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"
        s = float(i * 2)
        e = s + 1.9
        segs.append(_seg([_word("word", s, e, speaker_id=spk)], f"s{i}", spk))
    t = make_transcript(
        segs,
        speakers={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
    )
    report = generate_review(t, [], short_segment_threshold=0.1)
    assert "Alternating pair" in report


# ── generate_index ─────────────────────────────────────────────────────────────


def test_index_has_required_keys() -> None:
    seg = make_segment()
    t = make_transcript([seg])
    intel = _intel()
    idx = generate_index(t, intel, [], "generic")
    for key in (
        "version",
        "audio_file",
        "segments",
        "entities",
        "action_items",
        "decisions",
        "questions",
        "corrections",
        "topics",
    ):
        assert key in idx, f"Missing key: {key}"


def test_index_version_is_1() -> None:
    t = make_transcript()
    idx = generate_index(t, _intel(), [], "generic")
    assert idx["version"] == 1


def test_index_segments_match_transcript() -> None:
    segs = [
        make_segment("SPEAKER_00", [("Hello", 0.0, 0.5)]),
        make_segment("SPEAKER_01", [("Hi", 1.0, 1.5)]),
    ]
    t = make_transcript(segs, speakers={"SPEAKER_00": "A", "SPEAKER_01": "B"})
    idx = generate_index(t, _intel(), [], "generic")
    assert len(idx["segments"]) == 2


def test_index_segment_has_words() -> None:
    seg = make_segment("SPEAKER_00", [("Hello", 0.0, 0.5), ("world", 0.5, 1.0)])
    t = make_transcript([seg])
    idx = generate_index(t, _intel(), [], "generic")
    words = idx["segments"][0]["words"]
    assert len(words) == 2
    assert "text" in words[0]
    assert "start" in words[0]
    assert "end" in words[0]
    assert "confidence" in words[0]


def test_index_corrections_include_timestamp() -> None:
    w = _word("Apple", 5.0, 5.5)
    seg = _seg([w], "s0")
    t = make_transcript([seg])
    rec = _correction("s0")
    idx = generate_index(t, _intel(), [rec], "banking")
    assert idx["corrections"][0]["timestamp"] == seg.start


def test_index_serializable() -> None:
    seg = make_segment()
    t = make_transcript([seg])
    idx = generate_index(t, _intel(), [], "generic")
    dumped = json.dumps(idx)
    assert len(dumped) > 10


def test_index_topics_present() -> None:
    seg = make_segment()
    t = make_transcript([seg])
    topic = Topic(start=0.0, end=1.0, label="Test topic", keywords=("test",))
    intel = _intel(topics=[topic])
    idx = generate_index(t, intel, [], "generic")
    assert len(idx["topics"]) == 1
    assert idx["topics"][0]["label"] == "Test topic"


def test_index_speaker_names_in_segments() -> None:
    seg = make_segment("SPEAKER_00")
    t = make_transcript([seg], speakers={"SPEAKER_00": "Alice"})
    idx = generate_index(t, _intel(), [], "generic")
    assert idx["segments"][0]["speaker_name"] == "Alice"


# ── search_index ────────────────────────────────────────────────────────────────


def _make_index(text: str, speaker_name: str = "Alice") -> dict:
    return {
        "segments": [
            {
                "id": "s0",
                "speaker_id": "SPEAKER_00",
                "speaker_name": speaker_name,
                "start": 10.0,
                "end": 15.0,
                "text": text,
                "words": [],
            }
        ],
        "action_items": [],
        "decisions": [],
        "questions": [],
    }


def test_search_finds_term() -> None:
    idx = _make_index("We discussed the APPL_ID system.")
    results = search_index(idx, "APPL_ID")
    assert len(results) == 1
    assert results[0]["type"] == "segment"


def test_search_case_insensitive() -> None:
    idx = _make_index("The APPL_ID is important.")
    results = search_index(idx, "appl_id")
    assert len(results) >= 1


def test_search_no_match_returns_empty() -> None:
    idx = _make_index("Hello world.")
    results = search_index(idx, "mortgage")
    assert results == []


def test_search_special_action_items() -> None:
    idx = {
        "segments": [],
        "action_items": [
            {
                "task": "Follow up on loan",
                "owner": "Alice",
                "due_date": None,
                "timestamp": 5.0,
                "speaker_id": "SPEAKER_00",
                "confidence": 0.9,
                "reason": "test",
            },
        ],
        "decisions": [],
        "questions": [],
    }
    results = search_index(idx, "action items")
    assert len(results) == 1
    assert results[0]["type"] == "action_item"


def test_search_special_decisions() -> None:
    idx = {
        "segments": [],
        "action_items": [],
        "decisions": [
            {
                "decision": "Use banking profile",
                "rationale": None,
                "timestamp": 10.0,
                "speaker_id": "SPEAKER_00",
                "confidence": 0.95,
                "reason": "test",
            },
        ],
        "questions": [],
    }
    results = search_index(idx, "decisions")
    assert len(results) == 1
    assert results[0]["type"] == "decision"


def test_search_returns_timestamp() -> None:
    idx = _make_index("APPL_ID check complete.")
    results = search_index(idx, "APPL_ID")
    assert results[0]["timestamp"] == 10.0


def test_search_context_contains_bold_match() -> None:
    idx = _make_index("The APPL_ID must be validated.")
    results = search_index(idx, "APPL_ID")
    assert "**APPL_ID**" in results[0]["context"]


def test_search_action_items_in_results() -> None:
    idx = {
        "segments": [
            {
                "id": "s0",
                "speaker_id": "SPEAKER_00",
                "speaker_name": "Alice",
                "start": 0.0,
                "end": 1.0,
                "text": "I'll follow up on APPL_ID tomorrow.",
                "words": [],
            }
        ],
        "action_items": [
            {
                "task": "Follow up on APPL_ID",
                "owner": "Alice",
                "due_date": None,
                "timestamp": 0.0,
                "speaker_id": "SPEAKER_00",
                "confidence": 0.9,
                "reason": "test",
            }
        ],
        "decisions": [],
        "questions": [],
    }
    results = search_index(idx, "APPL_ID")
    types = {r["type"] for r in results}
    assert "segment" in types
    assert "action_item" in types


# ── IntelligenceResult backward compat ───────────────────────────────────────


def test_intelligence_result_default_topics() -> None:
    r = IntelligenceResult()
    assert r.topics == []


def test_intelligence_result_topics_assignable() -> None:
    t = Topic(start=0.0, end=10.0, label="Intro")
    r = IntelligenceResult(topics=[t])
    assert len(r.topics) == 1
