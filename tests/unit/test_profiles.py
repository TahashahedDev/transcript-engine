"""Unit tests for the profile system."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_segment, make_transcript
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.vocabulary import VocabularyCorrectionProcessor
from transcript_engine.profiles.loader import list_profiles, load_profile
from transcript_engine.profiles.model import Profile, VocabularyEntry


class TestProfileLoading:
    def test_load_banking_profile(self) -> None:
        profile = load_profile("banking")
        assert profile.name == "banking"
        assert len(profile.vocabulary) > 0
        assert profile.industry == "banking"

    def test_load_generic_profile(self) -> None:
        profile = load_profile("generic")
        assert profile.name == "generic"
        assert len(profile.vocabulary) == 0

    def test_load_nonexistent_profile_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent_xyz_profile")

    def test_list_profiles(self) -> None:
        profiles = list_profiles()
        assert "banking" in profiles
        assert "generic" in profiles

    def test_banking_processors_include_vocab(self) -> None:
        profile = load_profile("banking")
        assert "vocabulary_correction" in profile.processors

    def test_generic_processors_exclude_vocab(self) -> None:
        profile = load_profile("generic")
        assert "vocabulary_correction" not in profile.processors

    def test_vocabulary_entry_has_confidence(self) -> None:
        profile = load_profile("banking")
        assert all(0 < e.confidence <= 1.0 for e in profile.vocabulary)

    def test_appl_id_has_context(self) -> None:
        profile = load_profile("banking")
        appl = next((e for e in profile.vocabulary if e.canonical == "APPL_ID"), None)
        assert appl is not None
        assert "loan" in appl.context


class TestContextAwareCorrection:
    def _make_proc(
        self, entries: list[VocabularyEntry]
    ) -> VocabularyCorrectionProcessor:
        profile = Profile(name="test", vocabulary=tuple(entries))
        return VocabularyCorrectionProcessor(profile)

    def test_no_context_correction_always_applied(self) -> None:
        entry = VocabularyEntry(canonical="CBR", aliases=["Cyber"], confidence=0.90)
        proc = self._make_proc([entry])
        ctx = ProcessorContext()
        seg = make_segment("SPEAKER_00", [("Cyber", 0.0, 0.5)])
        transcript = make_transcript([seg])
        result = proc.process(transcript, ctx)
        assert result.segments[0].words[0].text == "CBR"

    def test_context_words_present_correction_applied(self) -> None:
        entry = VocabularyEntry(
            canonical="APPL_ID",
            aliases=["Apple ID"],
            confidence=0.95,
            context=["loan"],
        )
        proc = self._make_proc([entry])
        ctx = ProcessorContext()
        # Segment contains "loan" as context → correction should apply
        seg = make_segment(
            "SPEAKER_00",
            [
                ("Apple", 0.0, 0.4),
                ("ID", 0.4, 0.8),
                ("loan", 0.8, 1.1),
                ("number", 1.1, 1.5),
            ],
        )
        transcript = make_transcript([seg])
        result = proc.process(transcript, ctx)
        assert "APPL_ID" in result.segments[0].text

    def test_context_words_absent_correction_skipped(self) -> None:
        entry = VocabularyEntry(
            canonical="APPL_ID",
            aliases=["Apple ID"],
            confidence=0.95,
            context=["loan"],
        )
        proc = self._make_proc([entry])
        ctx = ProcessorContext()
        # Segment has no context words → confidence drops to 0.95 * 0.4 = 0.38 < threshold
        seg = make_segment(
            "SPEAKER_00",
            [("Apple", 0.0, 0.4), ("ID", 0.4, 0.8), ("phone", 0.8, 1.1)],
        )
        transcript = make_transcript([seg])
        result = proc.process(transcript, ctx)
        # Should NOT be corrected — Apple ID near "phone" is the Apple product
        assert "APPL_ID" not in result.segments[0].text

    def test_correction_below_threshold_skipped(self) -> None:
        entry = VocabularyEntry(canonical="CBR", aliases=["Cyber"], confidence=0.75)
        proc = self._make_proc([entry])
        ctx = ProcessorContext()
        seg = make_segment("SPEAKER_00", [("Cyber", 0.0, 0.5)])
        transcript = make_transcript([seg])
        result = proc.process(transcript, ctx)
        # confidence 0.75 < 0.80 threshold
        assert result.segments[0].words[0].text == "Cyber"


class TestCorrectionRecords:
    def test_correction_record_populated(self) -> None:
        entry = VocabularyEntry(canonical="CBR", aliases=["Cyber"], confidence=0.90)
        profile = Profile(name="banking", vocabulary=(entry,))
        proc = VocabularyCorrectionProcessor(profile)
        ctx = ProcessorContext()
        seg = make_segment("SPEAKER_00", [("Cyber", 0.0, 0.5)])
        transcript = make_transcript([seg])
        proc.process(transcript, ctx)
        assert len(ctx.corrections) == 1
        rec = ctx.corrections[0]
        assert rec.original == "Cyber"
        assert rec.replacement == "CBR"
        assert rec.profile == "banking"
        assert 0 < rec.confidence <= 1.0

    def test_no_correction_no_record(self) -> None:
        entry = VocabularyEntry(canonical="CBR", aliases=["Cyber"], confidence=0.90)
        profile = Profile(name="banking", vocabulary=(entry,))
        proc = VocabularyCorrectionProcessor(profile)
        ctx = ProcessorContext()
        seg = make_segment("SPEAKER_00", [("hello", 0.0, 0.5)])
        transcript = make_transcript([seg])
        proc.process(transcript, ctx)
        assert len(ctx.corrections) == 0

    def test_profile_isolation(self) -> None:
        """Banking corrections must NOT appear when using an empty profile."""
        empty_profile = Profile(name="generic")
        proc = VocabularyCorrectionProcessor(empty_profile)
        ctx = ProcessorContext()
        seg = make_segment("SPEAKER_00", [("Cyber", 0.0, 0.5), ("Lane", 0.5, 1.0)])
        transcript = make_transcript([seg])
        result = proc.process(transcript, ctx)
        # No banking vocab loaded — words unchanged
        assert result.segments[0].words[0].text == "Cyber"
        assert result.segments[0].words[1].text == "Lane"
        assert len(ctx.corrections) == 0


class TestProfileLoadFromFile:
    def test_from_vocabulary_file(self, vocab_file: Path) -> None:
        proc = VocabularyCorrectionProcessor.from_vocabulary_file(vocab_file)
        ctx = ProcessorContext()
        seg = make_segment("SPEAKER_00", [("Cyber", 0.0, 0.5)])
        transcript = make_transcript([seg])
        result = proc.process(transcript, ctx)
        assert result.segments[0].words[0].text == "CBR"
