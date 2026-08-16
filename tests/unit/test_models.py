"""Unit tests for the core data models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from transcript_engine.models.transcript import Segment, Transcript, Word


class TestWord:
    def test_immutable(self) -> None:
        w = Word(text="hello", start=0.0, end=0.5)
        with pytest.raises((ValidationError, TypeError)):
            w.text = "world"  # type: ignore[misc]

    def test_duration(self) -> None:
        w = Word(text="hello", start=1.0, end=1.8)
        assert abs(w.duration - 0.8) < 0.001


class TestSegment:
    def test_computed_word_count(self) -> None:
        words = (
            Word(text="a", start=0.0, end=0.5, speaker_id="SPEAKER_00"),
            Word(text="b", start=0.5, end=1.0, speaker_id="SPEAKER_00"),
        )
        seg = Segment(
            speaker_id="SPEAKER_00",
            start=0.0,
            end=1.0,
            words=words,
            text="a b",
        )
        assert seg.word_count == 2
        assert abs(seg.duration - 1.0) < 0.001


class TestTranscript:
    def _make(self) -> Transcript:
        words = (Word(text="hello", start=0.0, end=0.5, speaker_id="SPEAKER_00"),)
        seg = Segment(
            speaker_id="SPEAKER_00", start=0.0, end=0.5, words=words, text="hello"
        )
        return Transcript(
            audio_path=Path("test.mp4"),
            duration=5.0,
            language="en",
            segments=(seg,),
            speakers={"SPEAKER_00": "Speaker 1"},
        )

    def test_display_name_lookup(self) -> None:
        t = self._make()
        assert t.display_name("SPEAKER_00") == "Speaker 1"

    def test_display_name_fallback(self) -> None:
        t = self._make()
        assert t.display_name("SPEAKER_99") == "SPEAKER_99"

    def test_with_speakers_returns_new_object(self) -> None:
        t = self._make()
        t2 = t.with_speakers({"SPEAKER_00": "John"})
        assert t.speakers["SPEAKER_00"] == "Speaker 1"
        assert t2.speakers["SPEAKER_00"] == "John"

    def test_json_round_trip(self) -> None:
        t = self._make()
        serialized = t.model_dump_json()
        restored = Transcript.model_validate_json(serialized)
        assert restored.word_count == t.word_count
        assert restored.speakers == t.speakers
        assert restored.language == t.language

    def test_word_count(self) -> None:
        t = self._make()
        assert t.word_count == 1

    def test_speaker_ids_ordered(self) -> None:
        words0 = (Word(text="a", start=0.0, end=0.5, speaker_id="SPEAKER_00"),)
        words1 = (Word(text="b", start=1.0, end=1.5, speaker_id="SPEAKER_01"),)
        seg0 = Segment(
            speaker_id="SPEAKER_00", start=0.0, end=0.5, words=words0, text="a"
        )
        seg1 = Segment(
            speaker_id="SPEAKER_01", start=1.0, end=1.5, words=words1, text="b"
        )
        t = Transcript(
            audio_path=Path("test.mp4"),
            duration=2.0,
            language="en",
            segments=(seg0, seg1),
            speakers={"SPEAKER_00": "A", "SPEAKER_01": "B"},
        )
        assert t.speaker_ids == ["SPEAKER_00", "SPEAKER_01"]
