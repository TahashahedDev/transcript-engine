from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Transcript


class RawWord(BaseModel, frozen=True):
    """
    A transcribed word before speaker assignment.
    Produced by the transcription engine; consumed by the merger.
    """

    text: str
    start: float
    end: float
    confidence: float | None = None


class RawTranscription(BaseModel, frozen=True):
    """Output of the transcription engine, before diarization merge."""

    words: tuple[RawWord, ...]
    language: str
    duration: float


class SpeakerSegment(BaseModel, frozen=True):
    """A contiguous time range attributed to one speaker by the diarizer."""

    speaker_id: str
    start: float
    end: float


class DiarizationResult(BaseModel, frozen=True):
    """Output of the diarization engine."""

    segments: tuple[SpeakerSegment, ...]
    num_speakers: int


class PipelineResult(BaseModel):
    """The final result returned by Pipeline.run()."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transcript: Transcript
    audio: PreparedAudio
    elapsed_seconds: float
    warnings: list[str] = []
    corrections: list[CorrectionRecord] = []
