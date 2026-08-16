"""Unit tests for all processors."""

from __future__ import annotations

import pytest

from tests.conftest import make_segment, make_transcript
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.cleanup import TranscriptCleanupProcessor
from transcript_engine.processors.speaker_formatting import SpeakerFormattingProcessor
from transcript_engine.processors.vocabulary import VocabularyCorrectionProcessor
from transcript_engine.profiles.model import Profile


@pytest.fixture
def ctx() -> ProcessorContext:
    return ProcessorContext()


class TestVocabularyCorrectionProcessor:
    def test_exact_alias_replaced(
        self, banking_profile: Profile, ctx: ProcessorContext
    ) -> None:
        # "Apple ID" spans two Word objects — correction applies to the segment text
        seg = make_segment("SPEAKER_00", [("Apple", 0.0, 0.5), ("ID", 0.5, 1.0)])
        transcript = make_transcript([seg])
        proc = VocabularyCorrectionProcessor(banking_profile)
        result = proc.process(transcript, ctx)
        assert "APPL_ID" in result.segments[0].text

    def test_case_insensitive_match(
        self, banking_profile: Profile, ctx: ProcessorContext
    ) -> None:
        seg = make_segment("SPEAKER_00", [("cyber", 0.0, 0.5)])
        transcript = make_transcript([seg])
        proc = VocabularyCorrectionProcessor(banking_profile)
        result = proc.process(transcript, ctx)
        assert result.segments[0].words[0].text == "CBR"

    def test_no_match_unchanged(
        self, banking_profile: Profile, ctx: ProcessorContext
    ) -> None:
        seg = make_segment("SPEAKER_00", [("hello", 0.0, 0.5)])
        transcript = make_transcript([seg])
        proc = VocabularyCorrectionProcessor(banking_profile)
        result = proc.process(transcript, ctx)
        assert result.segments[0].words[0].text == "hello"

    def test_transcript_immutable(
        self, banking_profile: Profile, ctx: ProcessorContext
    ) -> None:
        seg = make_segment("SPEAKER_00", [("Lane", 0.0, 0.5)])
        transcript = make_transcript([seg])
        proc = VocabularyCorrectionProcessor(banking_profile)
        result = proc.process(transcript, ctx)
        assert transcript.segments[0].words[0].text == "Lane"
        assert result.segments[0].words[0].text == "LN"

    def test_missing_vocab_file_loads_empty(self, ctx: ProcessorContext) -> None:
        empty_profile = Profile(name="empty")
        proc = VocabularyCorrectionProcessor(empty_profile)
        seg = make_segment("SPEAKER_00", [("Lane", 0.0, 0.5)])
        transcript = make_transcript([seg])
        result = proc.process(transcript, ctx)
        assert result.segments[0].words[0].text == "Lane"

    def test_corrections_recorded(self, banking_profile: Profile) -> None:
        ctx = ProcessorContext()
        seg = make_segment("SPEAKER_00", [("cyber", 0.0, 0.5)])
        transcript = make_transcript([seg])
        proc = VocabularyCorrectionProcessor(banking_profile)
        proc.process(transcript, ctx)
        assert len(ctx.corrections) == 1
        assert ctx.corrections[0].replacement == "CBR"
        assert ctx.corrections[0].profile == "banking"


class TestTranscriptCleanupProcessor:
    def test_triplicate_removed(self, ctx: ProcessorContext) -> None:
        seg = make_segment(
            "SPEAKER_00",
            [
                ("right", 0.0, 0.3),
                ("right", 0.3, 0.6),
                ("right", 0.6, 0.9),
                ("okay", 0.9, 1.2),
            ],
        )
        transcript = make_transcript([seg])
        proc = TranscriptCleanupProcessor()
        result = proc.process(transcript, ctx)
        words = [w.text for w in result.segments[0].words]
        assert words.count("right") == 1

    def test_emphasis_not_removed(self, ctx: ProcessorContext) -> None:
        seg = make_segment(
            "SPEAKER_00",
            [("very", 0.0, 0.3), ("very", 0.3, 0.6), ("good", 0.6, 0.9)],
        )
        transcript = make_transcript([seg])
        proc = TranscriptCleanupProcessor()
        result = proc.process(transcript, ctx)
        words = [w.text for w in result.segments[0].words]
        assert "very" in words

    def test_single_word_no_change(self, ctx: ProcessorContext) -> None:
        seg = make_segment("SPEAKER_00", [("hello", 0.0, 0.5)])
        transcript = make_transcript([seg])
        proc = TranscriptCleanupProcessor()
        result = proc.process(transcript, ctx)
        assert result.segments[0].text == "hello"


class TestSpeakerFormattingProcessor:
    def test_consecutive_same_speaker_merged(self, ctx: ProcessorContext) -> None:
        segs = [
            make_segment("SPEAKER_00", [("hello", 0.0, 0.5)]),
            make_segment("SPEAKER_00", [("world", 0.6, 1.0)]),
        ]
        transcript = make_transcript(segs, {"SPEAKER_00": "Speaker 1"})
        proc = SpeakerFormattingProcessor()
        result = proc.process(transcript, ctx)
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello world"

    def test_different_speakers_not_merged(self, ctx: ProcessorContext) -> None:
        segs = [
            make_segment("SPEAKER_00", [("hello", 0.0, 0.5)]),
            make_segment("SPEAKER_01", [("hi", 0.6, 1.0)]),
        ]
        transcript = make_transcript(segs, {"SPEAKER_00": "A", "SPEAKER_01": "B"})
        proc = SpeakerFormattingProcessor()
        result = proc.process(transcript, ctx)
        assert len(result.segments) == 2

    def test_alternating_speakers_preserved(self, ctx: ProcessorContext) -> None:
        segs = [
            make_segment("SPEAKER_00", [("a", 0.0, 0.5)]),
            make_segment("SPEAKER_01", [("b", 0.6, 1.0)]),
            make_segment("SPEAKER_00", [("c", 1.1, 1.5)]),
        ]
        transcript = make_transcript(segs, {"SPEAKER_00": "A", "SPEAKER_01": "B"})
        proc = SpeakerFormattingProcessor()
        result = proc.process(transcript, ctx)
        assert len(result.segments) == 3

    def test_empty_transcript_unchanged(self, ctx: ProcessorContext) -> None:
        transcript = make_transcript(segments=[])
        proc = SpeakerFormattingProcessor()
        result = proc.process(transcript, ctx)
        assert result.segments == ()
