"""
Shared fixtures and factory helpers for all tests.
No ML models are loaded here — all data is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcript_engine.models.pipeline import (
    DiarizationResult,
    RawTranscription,
    RawWord,
    SpeakerSegment,
)
from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.profiles.loader import _load_vocab_entries
from transcript_engine.profiles.model import Profile

# ── factories ────────────────────────────────────────────────────────────────


def make_raw_words(data: list[tuple[str, float, float]]) -> tuple[RawWord, ...]:
    return tuple(RawWord(text=w, start=s, end=e) for w, s, e in data)


def make_raw_transcription(
    data: list[tuple[str, float, float]],
    language: str = "en",
    duration: float = 0.0,
) -> RawTranscription:
    words = make_raw_words(data)
    if duration == 0.0 and words:
        duration = words[-1].end
    return RawTranscription(words=words, language=language, duration=duration)


def make_diarization(data: list[tuple[str, float, float]]) -> DiarizationResult:
    segs = tuple(SpeakerSegment(speaker_id=sp, start=s, end=e) for sp, s, e in data)
    num = len({sp for sp, _, _ in data})
    return DiarizationResult(segments=segs, num_speakers=num)


def make_word(
    text: str,
    start: float = 0.0,
    end: float = 1.0,
    speaker_id: str | None = "SPEAKER_00",
) -> Word:
    return Word(text=text, start=start, end=end, speaker_id=speaker_id)


def make_segment(
    speaker_id: str = "SPEAKER_00",
    words: list[tuple[str, float, float]] | None = None,
) -> Segment:
    if words is None:
        words = [("hello", 0.0, 0.5), ("world", 0.5, 1.0)]
    word_objs = tuple(
        Word(text=t, start=s, end=e, speaker_id=speaker_id) for t, s, e in words
    )
    text = " ".join(w.text for w in word_objs)
    return Segment(
        speaker_id=speaker_id,
        start=word_objs[0].start,
        end=word_objs[-1].end,
        words=word_objs,
        text=text,
    )


def make_transcript(
    segments: list[Segment] | None = None,
    speakers: dict[str, str] | None = None,
    audio_path: Path | None = None,
) -> Transcript:
    if segments is None:
        segments = [make_segment()]
    if speakers is None:
        speakers = {"SPEAKER_00": "Speaker 1"}
    if audio_path is None:
        audio_path = Path("test_meeting.mp4")
    duration = segments[-1].end if segments else 0.0
    return Transcript(
        audio_path=audio_path,
        duration=duration,
        language="en",
        segments=tuple(segments),
        speakers=speakers,
    )


# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def simple_transcript() -> Transcript:
    segs = [
        make_segment("SPEAKER_00", [("Hello", 0.0, 0.5), ("there", 0.5, 1.0)]),
        make_segment("SPEAKER_01", [("Hi", 1.2, 1.6), ("everyone", 1.6, 2.1)]),
    ]
    return make_transcript(
        segments=segs,
        speakers={"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"},
    )


@pytest.fixture
def vocab_file(tmp_path: Path) -> Path:
    content = """
APPL_ID:
  aliases:
    - Apple ID
    - Apple I

LN:
  aliases:
    - Lane
    - Ellen

CBR:
  aliases:
    - Cyber
    - Sidebar

McCraken:
  aliases:
    - Mac cracker
    - MacCraken
"""
    f = tmp_path / "vocabulary.yaml"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def banking_profile(vocab_file: Path) -> Profile:
    """Banking profile built from the temp vocab fixture."""
    entries = _load_vocab_entries(vocab_file)
    return Profile(name="banking", vocabulary=tuple(entries))
