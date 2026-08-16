"""Meeting search index generator — meeting.index.json

Serializes the full transcript + intelligence result into a flat JSON structure
that can be searched quickly without re-running the pipeline.
"""

from __future__ import annotations

from typing import Any

from transcript_engine.intelligence.models import IntelligenceResult
from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Transcript


def generate_index(
    transcript: Transcript,
    intelligence: IntelligenceResult,
    corrections: list[CorrectionRecord],
    profile_name: str,
) -> dict[str, Any]:
    """Return a dict suitable for json.dumps — the meeting search index."""
    seg_by_id = {seg.id: seg for seg in transcript.segments}

    segments = [
        {
            "id": seg.id,
            "speaker_id": seg.speaker_id,
            "speaker_name": transcript.display_name(seg.speaker_id),
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in seg.words
            ],
        }
        for seg in transcript.segments
    ]

    ent = intelligence.entities
    entities: dict[str, list[str]] = {
        "people": list(ent.people),
        "organizations": list(ent.organizations),
        "acronyms": list(ent.acronyms),
        "dollar_amounts": list(ent.dollar_amounts),
        "percentages": list(ent.percentages),
        "dates": list(ent.dates),
        "loan_ids": list(ent.loan_ids),
        "ticket_numbers": list(ent.ticket_numbers),
        "urls": list(ent.urls),
        "emails": list(ent.emails),
    }

    action_items = [
        {
            "task": ai.task,
            "owner": ai.owner,
            "due_date": ai.due_date,
            "timestamp": ai.timestamp,
            "speaker_id": ai.speaker_id,
            "confidence": ai.confidence,
            "reason": ai.reason,
        }
        for ai in intelligence.action_items
    ]

    decisions = [
        {
            "decision": d.decision,
            "rationale": d.rationale,
            "timestamp": d.timestamp,
            "speaker_id": d.speaker_id,
            "confidence": d.confidence,
            "reason": d.reason,
        }
        for d in intelligence.decisions
    ]

    questions = [
        {
            "question": q.question,
            "is_answered": q.is_answered,
            "timestamp": q.timestamp,
            "speaker_id": q.speaker_id,
            "confidence": q.confidence,
            "reason": q.reason,
        }
        for q in intelligence.questions
    ]

    indexed_corrections = []
    for rec in corrections:
        seg = seg_by_id.get(rec.segment_id)
        indexed_corrections.append(
            {
                "original": rec.original,
                "replacement": rec.replacement,
                "reason": rec.reason,
                "confidence": rec.confidence,
                "profile": rec.profile,
                "segment_id": rec.segment_id,
                "timestamp": seg.start if seg else None,
            }
        )

    topics = [
        {
            "start": t.start,
            "end": t.end,
            "label": t.label,
            "keywords": list(t.keywords),
        }
        for t in intelligence.topics
    ]

    return {
        "version": 1,
        "audio_file": transcript.audio_path.name,
        "duration": transcript.duration,
        "language": transcript.language,
        "profile": profile_name,
        "speakers": dict(transcript.speakers),
        "segments": segments,
        "topics": topics,
        "entities": entities,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "corrections": indexed_corrections,
    }
