"""
Unit tests for TranscriptMerger.
All data is synthetic — no ML models loaded.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_diarization, make_raw_transcription
from transcript_engine.merger.merger import TranscriptMerger
from transcript_engine.models.audio import PreparedAudio


def _audio(duration: float = 10.0) -> PreparedAudio:
    return PreparedAudio(
        path=Path("/tmp/test.wav"),
        duration=duration,
        original_path=Path("/tmp/test.mp4"),
        original_format="mp4",
    )


class TestSpeakerAssignment:
    def test_single_speaker(self) -> None:
        transcription = make_raw_transcription(
            [("Hello", 0.0, 0.5), ("world", 0.5, 1.0)]
        )
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert all(
            w.speaker_id == "SPEAKER_00"
            for seg in transcript.segments
            for w in seg.words
        )

    def test_two_speakers(self) -> None:
        transcription = make_raw_transcription(
            [
                ("Hello", 0.0, 0.5),
                ("there", 0.5, 1.0),
                ("Hi", 2.0, 2.4),
                ("back", 2.4, 2.8),
            ]
        )
        diarization = make_diarization(
            [("SPEAKER_00", 0.0, 1.2), ("SPEAKER_01", 1.8, 3.0)]
        )
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        segments = transcript.segments
        assert len(segments) == 2
        assert segments[0].speaker_id == "SPEAKER_00"
        assert segments[1].speaker_id == "SPEAKER_01"

    def test_word_spanning_two_speakers_assigned_to_majority(self) -> None:
        transcription = make_raw_transcription([("crossover", 0.8, 1.6)])
        diarization = make_diarization(
            [("SPEAKER_00", 0.0, 1.0), ("SPEAKER_01", 1.0, 3.0)]
        )
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        word = transcript.segments[0].words[0]
        # overlap: SPEAKER_00 covers 0.2s, SPEAKER_01 covers 0.6s → SPEAKER_01 wins
        assert word.speaker_id == "SPEAKER_01"

    def test_word_before_first_diarization_segment(self) -> None:
        transcription = make_raw_transcription(
            [("early", 0.0, 0.2), ("word", 2.0, 2.5)]
        )
        diarization = make_diarization([("SPEAKER_00", 1.5, 3.0)])
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        # "early" has no diarization coverage → assigned to nearest subsequent known speaker
        all_words = [w for seg in transcript.segments for w in seg.words]
        assert all(w.speaker_id is not None for w in all_words)

    def test_gap_between_speakers_inherits_last_known(self) -> None:
        transcription = make_raw_transcription(
            [("one", 0.0, 0.5), ("gap", 1.5, 2.0), ("two", 3.0, 3.5)]
        )
        diarization = make_diarization(
            [("SPEAKER_00", 0.0, 0.8), ("SPEAKER_01", 2.5, 4.0)]
        )
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        words = [w for seg in transcript.segments for w in seg.words]
        # "gap" falls in silence — should inherit SPEAKER_00
        assert words[1].speaker_id == "SPEAKER_00"

    def test_empty_transcription(self) -> None:
        transcription = make_raw_transcription([])
        diarization = make_diarization([("SPEAKER_00", 0.0, 1.0)])
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert transcript.segments == ()
        assert transcript.word_count == 0

    def test_no_diarization_segments(self) -> None:
        transcription = make_raw_transcription([("hello", 0.0, 0.5)])
        diarization = make_diarization([])
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert len(transcript.segments) >= 1


class TestSegmentGrouping:
    def test_consecutive_same_speaker_grouped(self) -> None:
        transcription = make_raw_transcription(
            [("a", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 1.0, 1.5)]
        )
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert len(transcript.segments) == 1
        assert transcript.segments[0].word_count == 3

    def test_speaker_change_creates_new_segment(self) -> None:
        transcription = make_raw_transcription([("a", 0.0, 0.5), ("b", 1.0, 1.5)])
        diarization = make_diarization(
            [("SPEAKER_00", 0.0, 0.7), ("SPEAKER_01", 0.8, 2.0)]
        )
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert len(transcript.segments) == 2

    def test_segment_text_is_joined_words(self) -> None:
        transcription = make_raw_transcription(
            [("Hello", 0.0, 0.5), ("world", 0.5, 1.0)]
        )
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert transcript.segments[0].text == "Hello world"


class TestSpeakersDict:
    def test_speakers_initialized_in_order(self) -> None:
        transcription = make_raw_transcription(
            [("a", 0.0, 0.5), ("b", 1.0, 1.5), ("c", 2.0, 2.5)]
        )
        diarization = make_diarization(
            [
                ("SPEAKER_00", 0.0, 0.7),
                ("SPEAKER_01", 0.8, 1.8),
                ("SPEAKER_00", 1.9, 3.0),
            ]
        )
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert "SPEAKER_00" in transcript.speakers
        assert "SPEAKER_01" in transcript.speakers
        assert transcript.speakers["SPEAKER_00"] == "Speaker 1"
        assert transcript.speakers["SPEAKER_01"] == "Speaker 2"

    def test_three_speakers(self) -> None:
        transcription = make_raw_transcription(
            [("a", 0.0, 0.5), ("b", 1.0, 1.5), ("c", 2.0, 2.5)]
        )
        diarization = make_diarization(
            [
                ("SPEAKER_00", 0.0, 0.7),
                ("SPEAKER_01", 0.8, 1.8),
                ("SPEAKER_02", 1.9, 3.0),
            ]
        )
        merger = TranscriptMerger()
        transcript = merger.merge(transcription, diarization, _audio())
        assert len(transcript.speakers) == 3


class TestSpeakerNumberingWithUnknown:
    """
    Regression: display names skipped "Speaker 1" whenever any word was
    unattributed. `enumerate(seen)` let the "Unknown" bucket consume index 1,
    so a real meeting rendered as "Unknown, Speaker 2, Speaker 3". This fires
    whenever transcription starts before pyannote's first speech segment —
    a cough or "um" at t=0 is enough.
    """

    def _merge(self, words, segments):  # type: ignore[no-untyped-def]
        from pathlib import Path

        from transcript_engine.merger.merger import TranscriptMerger
        from transcript_engine.models.audio import PreparedAudio
        from transcript_engine.models.pipeline import DiarizationResult, RawTranscription

        audio = PreparedAudio(
            path=Path("a.wav"), duration=10.0,
            original_path=Path("a.wav"), original_format="wav",
        )
        return TranscriptMerger().merge(
            RawTranscription(words=tuple(words), language="en", duration=10.0),
            DiarizationResult(segments=tuple(segments), num_speakers=len({s.speaker_id for s in segments})),
            audio,
        )

    def test_speaker_one_exists_when_some_words_are_unattributed(self) -> None:
        from transcript_engine.models.pipeline import RawWord, SpeakerSegment

        words = [
            RawWord(text="Um,", start=0.0, end=0.4, confidence=None),   # before any segment
            RawWord(text="hello", start=1.2, end=1.6, confidence=None),
            RawWord(text="hi", start=3.2, end=3.6, confidence=None),
        ]
        segments = [
            SpeakerSegment(speaker_id="SPEAKER_00", start=1.0, end=2.0),
            SpeakerSegment(speaker_id="SPEAKER_01", start=3.0, end=4.0),
        ]

        transcript = self._merge(words, segments)

        assert transcript.speakers["SPEAKER_00"] == "Speaker 1"
        assert transcript.speakers["SPEAKER_01"] == "Speaker 2"
        assert transcript.speakers["SPEAKER_UNKNOWN"] == "Unknown"

    def test_numbering_is_sequential_without_unknown(self) -> None:
        from transcript_engine.models.pipeline import RawWord, SpeakerSegment

        words = [
            RawWord(text="a", start=1.2, end=1.4, confidence=None),
            RawWord(text="b", start=3.2, end=3.4, confidence=None),
        ]
        segments = [
            SpeakerSegment(speaker_id="SPEAKER_00", start=1.0, end=2.0),
            SpeakerSegment(speaker_id="SPEAKER_01", start=3.0, end=4.0),
        ]

        speakers = self._merge(words, segments).speakers

        assert sorted(speakers.values()) == ["Speaker 1", "Speaker 2"]

    def test_out_of_order_words_are_still_attributed_correctly(self) -> None:
        """
        _assign_speakers walks a pointer that only moves forward, so unsorted
        input would silently misattribute. merge() sorts defensively.
        """
        from transcript_engine.models.pipeline import RawWord, SpeakerSegment

        words = [
            RawWord(text="later", start=3.2, end=3.6, confidence=None),
            RawWord(text="earlier", start=1.2, end=1.6, confidence=None),
        ]
        segments = [
            SpeakerSegment(speaker_id="SPEAKER_00", start=1.0, end=2.0),
            SpeakerSegment(speaker_id="SPEAKER_01", start=3.0, end=4.0),
        ]

        transcript = self._merge(words, segments)
        by_text = {w.text: w.speaker_id for seg in transcript.segments for w in seg.words}

        assert by_text["earlier"] == "SPEAKER_00"
        assert by_text["later"] == "SPEAKER_01"
