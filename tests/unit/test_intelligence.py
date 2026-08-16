"""Unit tests for TranscriptIntelligenceProcessor."""

from __future__ import annotations

import pytest

from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.intelligence import (
    TranscriptIntelligenceProcessor,
    _capitalize_word,
    _join_words,
)


def _word(
    text: str, start: float = 0.0, end: float = 0.5, confidence: float | None = None
) -> Word:
    return Word(
        text=text, start=start, end=end, speaker_id="SPEAKER_00", confidence=confidence
    )


def _seg(words: list[Word], seg_id: str = "s0") -> Segment:
    text = " ".join(w.text for w in words)
    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    return Segment(
        id=seg_id,
        speaker_id="SPEAKER_00",
        start=start,
        end=end,
        words=tuple(words),
        text=text,
    )


def _transcript(segs: list[Segment]) -> Transcript:
    from pathlib import Path

    return Transcript(
        audio_path=Path("test.wav"),
        language="en",
        segments=tuple(segs),
        speakers={"SPEAKER_00": "Speaker 1"},
        duration=10.0,
    )


@pytest.fixture
def proc() -> TranscriptIntelligenceProcessor:
    return TranscriptIntelligenceProcessor()


# ── capitalize_word helper ────────────────────────────────────────────────────


def test_capitalize_word_simple() -> None:
    assert _capitalize_word("hello") == "Hello"


def test_capitalize_word_already_caps() -> None:
    assert _capitalize_word("Hello") == "Hello"


def test_capitalize_word_with_leading_punct() -> None:
    assert _capitalize_word("'twas") == "'Twas"


def test_capitalize_word_all_caps() -> None:
    # acronyms should not be changed (already capital)
    assert _capitalize_word("AI") == "AI"


# ── join_words helper ─────────────────────────────────────────────────────────


def test_join_words_simple() -> None:
    words = [_word("hello"), _word("world")]
    assert _join_words(words) == "hello world"


def test_join_words_no_space_before_comma() -> None:
    words = [_word("yes"), _word(","), _word("sure")]
    assert _join_words(words) == "yes, sure"


# ── pronoun correction ────────────────────────────────────────────────────────


def test_fix_standalone_i(proc: TranscriptIntelligenceProcessor) -> None:
    seg = _seg([_word("i"), _word("went"), _word("there")])
    result = proc._fix_pronouns(list(seg.words))
    assert result[0].text == "I"
    assert result[1].text == "went"


def test_fix_i_with_trailing_comma(proc: TranscriptIntelligenceProcessor) -> None:
    words = [_word("i,"), _word("think")]
    result = proc._fix_pronouns(words)
    # "i," length is 2, should be fixed
    assert result[0].text == "I,"


def test_does_not_fix_long_i_word(proc: TranscriptIntelligenceProcessor) -> None:
    # "in" should not become "In"
    words = [_word("in"), _word("the")]
    result = proc._fix_pronouns(words)
    assert result[0].text == "in"


# ── capitalization ────────────────────────────────────────────────────────────


def test_first_word_capitalized(proc: TranscriptIntelligenceProcessor) -> None:
    seg = _seg([_word("hello"), _word("world")])
    result = proc.process(_transcript([seg]), ProcessorContext())
    assert result.segments[0].words[0].text == "Hello"


def test_word_after_period_capitalized(proc: TranscriptIntelligenceProcessor) -> None:
    words = [_word("okay."), _word("let"), _word("us"), _word("begin")]
    seg = _seg(words)
    result = proc.process(_transcript([seg]), ProcessorContext())
    out_words = result.segments[0].words
    assert out_words[0].text == "Okay."
    assert out_words[1].text == "Let"


def test_word_after_question_mark_capitalized(
    proc: TranscriptIntelligenceProcessor,
) -> None:
    words = [_word("done?"), _word("yes,"), _word("we"), _word("are")]
    seg = _seg(words)
    result = proc.process(_transcript([seg]), ProcessorContext())
    out_words = result.segments[0].words
    assert out_words[1].text == "Yes,"


# ── terminal punctuation ──────────────────────────────────────────────────────


def test_period_added_to_statement(proc: TranscriptIntelligenceProcessor) -> None:
    seg = _seg([_word("the"), _word("deal"), _word("closed")])
    result = proc.process(_transcript([seg]), ProcessorContext())
    last = result.segments[0].words[-1]
    assert last.text == "closed."


def test_question_mark_added_to_question(proc: TranscriptIntelligenceProcessor) -> None:
    seg = _seg([_word("is"), _word("it"), _word("ready")])
    result = proc.process(_transcript([seg]), ProcessorContext())
    last = result.segments[0].words[-1]
    assert last.text == "ready?"


def test_no_punctuation_added_if_already_present(
    proc: TranscriptIntelligenceProcessor,
) -> None:
    seg = _seg([_word("hello"), _word("world.")])
    result = proc.process(_transcript([seg]), ProcessorContext())
    last = result.segments[0].words[-1]
    assert last.text == "world."


def test_no_punctuation_on_filler(proc: TranscriptIntelligenceProcessor) -> None:
    seg = _seg([_word("mm")])
    result = proc.process(_transcript([seg]), ProcessorContext())
    last = result.segments[0].words[-1]
    # "mm" is a filler, no punctuation added
    assert last.text in ("mm", "Mm")  # capitalized but no period


# ── full segment flow ─────────────────────────────────────────────────────────


def test_empty_segment_unchanged(proc: TranscriptIntelligenceProcessor) -> None:
    seg = Segment(
        id="s0", speaker_id="SPEAKER_00", start=0.0, end=0.0, words=(), text=""
    )
    result = proc.process(_transcript([seg]), ProcessorContext())
    assert result.segments[0] is seg


def test_text_updated_from_words(proc: TranscriptIntelligenceProcessor) -> None:
    seg = _seg([_word("i"), _word("think"), _word("so")])
    result = proc.process(_transcript([seg]), ProcessorContext())
    # text should reflect word changes
    out_seg = result.segments[0]
    assert out_seg.text[0].isupper()


def test_multiple_segments_processed(proc: TranscriptIntelligenceProcessor) -> None:
    segs = [
        _seg([_word("hello"), _word("there")], "s0"),
        _seg([_word("i"), _word("agree")], "s1"),
    ]
    result = proc.process(_transcript(segs), ProcessorContext())
    assert result.segments[0].words[0].text == "Hello"
    assert result.segments[1].words[0].text == "I"


# ── stats/quality importability ───────────────────────────────────────────────


def test_export_stats_importable() -> None:
    from transcript_engine.exporters.stats import export_stats

    assert callable(export_stats)


def test_export_quality_importable() -> None:
    from transcript_engine.exporters.quality import export_quality

    assert callable(export_quality)


def test_export_stats_runs(proc: TranscriptIntelligenceProcessor) -> None:
    import json

    from transcript_engine.exporters.stats import export_stats

    words = [_word("hello", confidence=0.9), _word("world", confidence=0.6)]
    seg = _seg(words)
    t = _transcript([seg])
    result = export_stats(
        t, elapsed_seconds=10.0, corrections=[], profile_name="generic"
    )
    data = json.loads(result)
    assert data["word_count"] == 2
    assert data["corrections_applied"] == 0
    assert data["confidence"]["words_with_score"] == 2


def test_export_quality_runs(proc: TranscriptIntelligenceProcessor) -> None:
    from transcript_engine.exporters.quality import export_quality

    words = [_word("hello", confidence=0.9), _word("UNKN", confidence=0.4)]
    seg = _seg(words)
    t = _transcript([seg])
    result = export_quality(t, corrections=[], profile_name="generic")
    assert "Quality Report" in result
    assert "UNKN" in result  # should appear as unknown acronym candidate
