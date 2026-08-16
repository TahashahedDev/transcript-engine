"""Timeline extractor — builds a chronological event list."""

from __future__ import annotations

from typing import ClassVar

from transcript_engine.intelligence.base import IntelligenceContext
from transcript_engine.intelligence.models import (
    ActionItem,
    Decision,
    TimelineEvent,
    Topic,
)
from transcript_engine.models.transcript import Transcript


class TimelineExtractor:
    name: ClassVar[str] = "timeline"

    def extract(
        self,
        transcript: Transcript,
        ctx: IntelligenceContext,
        decisions: list[Decision] | None = None,
        action_items: list[ActionItem] | None = None,
        topics: list[Topic] | None = None,
    ) -> list[TimelineEvent]:
        if not transcript.segments:
            return []

        events: list[TimelineEvent] = []

        # Meeting start
        first_seg = transcript.segments[0]
        events.append(
            TimelineEvent(
                timestamp=first_seg.start,
                event="Meeting begins",
                speaker_id=None,
            )
        )

        # First appearance per speaker (only if multiple speakers)
        if len(transcript.speakers) > 1:
            seen_speakers: set[str] = set()
            for seg in transcript.segments:
                if seg.speaker_id not in seen_speakers:
                    seen_speakers.add(seg.speaker_id)
                    display = transcript.display_name(seg.speaker_id)
                    events.append(
                        TimelineEvent(
                            timestamp=seg.start,
                            event=f"{display} speaks",
                            speaker_id=seg.speaker_id,
                        )
                    )

        # Topic transitions
        _skip_labels = frozenset({"unknown", "other", "general", ""})
        for topic in topics or []:
            if topic.label and topic.label.lower() not in _skip_labels:
                label = topic.label[:60] + ("..." if len(topic.label) > 60 else "")
                events.append(
                    TimelineEvent(
                        timestamp=topic.start,
                        event=f"Topic: {label}",
                        speaker_id=None,
                    )
                )

        # Decisions
        for dec in decisions or []:
            label = dec.decision[:60] + ("..." if len(dec.decision) > 60 else "")
            events.append(
                TimelineEvent(
                    timestamp=dec.timestamp,
                    event=f"Decision: {label}",
                    speaker_id=dec.speaker_id,
                )
            )

        # Action items
        for item in action_items or []:
            label = item.task[:60] + ("..." if len(item.task) > 60 else "")
            events.append(
                TimelineEvent(
                    timestamp=item.timestamp,
                    event=f"Action item: {label}",
                    speaker_id=item.speaker_id,
                )
            )

        # Meeting end
        last_seg = transcript.segments[-1]
        events.append(
            TimelineEvent(
                timestamp=last_seg.end,
                event="Meeting ends",
                speaker_id=None,
            )
        )

        # Sort and deduplicate at identical timestamps
        events.sort(key=lambda e: e.timestamp)
        seen_events: set[tuple[float, str]] = set()
        unique: list[TimelineEvent] = []
        for ev in events:
            key = (ev.timestamp, ev.event)
            if key not in seen_events:
                seen_events.add(key)
                unique.append(ev)

        return unique
