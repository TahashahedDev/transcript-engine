from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


class Word(BaseModel, frozen=True):
    """
    A single recognized word with timing information.
    speaker_id holds the internal diarization label ("SPEAKER_00"),
    never a display name. Display names live in Transcript.speakers.
    """

    text: str
    start: float
    end: float
    confidence: float | None = None
    speaker_id: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class Segment(BaseModel, frozen=True):
    """
    A consecutive run of words from one speaker, bounded by a speaker change
    or a silence gap. The unit of display in most exporters.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    speaker_id: str
    start: float
    end: float
    words: tuple[Word, ...]
    text: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        return self.end - self.start

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return len(self.words)


class Transcript(BaseModel, frozen=True):
    """
    The single result object that flows through every pipeline stage.
    Processors must return a new Transcript — never mutate this one.

    speakers maps internal diarization IDs to display names:
        {"SPEAKER_00": "Speaker 1"} → later → {"SPEAKER_00": "John"}
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    audio_path: Path
    duration: float
    language: str
    segments: tuple[Segment, ...]
    speakers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pipeline_version: str = "0.1.0"

    def display_name(self, speaker_id: str) -> str:
        """Resolve a speaker_id to its display name."""
        return self.speakers.get(speaker_id, speaker_id)

    def with_speakers(self, speakers: dict[str, str]) -> Transcript:
        """Return a new Transcript with an updated speakers mapping."""
        return self.model_copy(update={"speakers": speakers})

    def with_segments(self, segments: tuple[Segment, ...]) -> Transcript:
        """Return a new Transcript with replaced segments."""
        return self.model_copy(update={"segments": segments})

    def with_metadata(self, key: str, value: Any) -> Transcript:
        """Return a new Transcript with one additional metadata key."""
        return self.model_copy(update={"metadata": {**self.metadata, key: value}})

    @property
    def speaker_ids(self) -> list[str]:
        """Ordered unique speaker IDs as they appear in the transcript."""
        seen: dict[str, None] = {}
        for seg in self.segments:
            seen[seg.speaker_id] = None
        return list(seen)

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.segments)
