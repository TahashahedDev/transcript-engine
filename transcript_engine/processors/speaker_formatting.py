"""
Speaker Formatting Processor

Merges consecutive segments from the same speaker into unified turns.
This is the final processing step before export.

Input:  Many small same-speaker segments (one per diarization chunk)
Output: Fewer, larger segments — one per contiguous speaker turn
"""

from __future__ import annotations

from typing import ClassVar
from uuid import uuid4

from transcript_engine.logging import get_logger
from transcript_engine.models.transcript import Segment, Transcript, Word
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.registry import register

logger = get_logger(__name__)


@register
class SpeakerFormattingProcessor:
    """
    Collapses consecutive same-speaker segments into a single segment.
    Result is one segment per contiguous speaker turn — the natural unit
    for human-readable transcripts.
    """

    name: ClassVar[str] = "speaker_formatting"

    def __init__(self) -> None:
        pass

    def process(self, transcript: Transcript, ctx: ProcessorContext) -> Transcript:
        if not transcript.segments:
            return transcript

        merged = self._merge_consecutive(list(transcript.segments))

        before = len(transcript.segments)
        after = len(merged)
        if before != after:
            logger.info(f"Speaker formatting: merged {before} → {after} segments")

        return transcript.with_segments(tuple(merged))

    def _merge_consecutive(self, segments: list[Segment]) -> list[Segment]:
        if not segments:
            return []

        merged: list[Segment] = []
        current = segments[0]

        for next_seg in segments[1:]:
            if next_seg.speaker_id == current.speaker_id:
                current = self._combine(current, next_seg)
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged

    def _combine(self, a: Segment, b: Segment) -> Segment:
        combined_words: tuple[Word, ...] = a.words + b.words
        combined_text = f"{a.text} {b.text}".strip()
        return Segment(
            id=str(uuid4()),
            speaker_id=a.speaker_id,
            start=a.start,
            end=b.end,
            words=combined_words,
            text=combined_text,
        )
