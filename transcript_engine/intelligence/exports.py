"""
Export functions for meeting intelligence artifacts.

Each function returns a str (markdown or JSON).
None of these functions modify the Transcript.
"""

from __future__ import annotations

import json
from collections import defaultdict

from transcript_engine.intelligence.models import IntelligenceResult
from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Transcript


def export_summary(result: IntelligenceResult, transcript: Transcript) -> str:
    ai_label = "Yes" if result.ai_enhanced else "No"
    lines = [
        "# Meeting Summary",
        "",
        f"> Generated from transcript. AI-enhanced: {ai_label}.",
        "",
    ]
    if result.summary_bullets:
        for bullet in result.summary_bullets:
            lines.append(f"- {bullet}")
    else:
        lines.append("*No summary could be generated from this transcript.*")
    lines.append("")
    return "\n".join(lines)


def export_action_items(result: IntelligenceResult, transcript: Transcript) -> str:
    lines = [
        "# Action Items",
        "",
    ]
    if not result.action_items:
        lines.append("*No action items detected.*")
        lines.append("")
        return "\n".join(lines)

    lines += [
        "| Task | Owner | Due Date | Confidence | Timestamp | Reason |",
        "|------|-------|----------|------------|-----------|--------|",
    ]
    for item in result.action_items:
        ts = _fmt_ts(item.timestamp)
        owner = item.owner or "—"
        due = item.due_date or "—"
        conf = f"{item.confidence:.0%}"
        task = item.task.replace("|", "\\|")
        lines.append(f"| {task} | {owner} | {due} | {conf} | {ts} | {item.reason} |")

    lines.append("")
    return "\n".join(lines)


def export_decisions(result: IntelligenceResult, transcript: Transcript) -> str:
    lines = ["# Decisions", ""]
    if not result.decisions:
        lines.append("*No decisions detected.*")
        lines.append("")
        return "\n".join(lines)

    for i, dec in enumerate(result.decisions, 1):
        display = transcript.display_name(dec.speaker_id)
        lines += [
            f"## Decision {i}",
            "",
            f"**Decision:** {dec.decision}  ",
        ]
        if dec.rationale:
            lines.append(f"**Rationale:** {dec.rationale}  ")
        lines += [
            f"**Timestamp:** {_fmt_ts(dec.timestamp)}  ",
            f"**Speaker:** {display}  ",
            f"**Confidence:** {dec.confidence:.0%}",
            "",
        ]
    return "\n".join(lines)


def export_questions(result: IntelligenceResult, transcript: Transcript) -> str:
    lines = ["# Open Questions", ""]
    if not result.questions:
        lines.append("*No questions detected.*")
        lines.append("")
        return "\n".join(lines)

    lines += [
        "| Question | Speaker | Timestamp | Answered |",
        "|----------|---------|-----------|----------|",
    ]
    for q in result.questions:
        display = transcript.display_name(q.speaker_id)
        answered = "Yes" if q.is_answered else "No"
        question_text = q.question.replace("|", "\\|")
        lines.append(
            f"| {question_text} | {display} | {_fmt_ts(q.timestamp)} | {answered} |"
        )

    lines.append("")
    return "\n".join(lines)


def export_entities(result: IntelligenceResult) -> str:
    ent = result.entities
    data = {
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
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_timeline(result: IntelligenceResult, transcript: Transcript) -> str:
    lines = ["# Meeting Timeline", ""]
    if not result.timeline:
        lines.append("*No timeline events.*")
        lines.append("")
        return "\n".join(lines)

    lines += [
        "| Time | Event | Speaker |",
        "|------|-------|---------|",
    ]
    for event in result.timeline:
        ts = _fmt_ts(event.timestamp)
        speaker = transcript.display_name(event.speaker_id) if event.speaker_id else "—"
        event_text = event.event.replace("|", "\\|")
        lines.append(f"| {ts} | {event_text} | {speaker} |")

    lines.append("")
    return "\n".join(lines)


def export_metrics(
    transcript: Transcript,
    elapsed_seconds: float,
    corrections: list[CorrectionRecord],
    profile_name: str,
    result: IntelligenceResult,
    low_conf_threshold: float = 0.65,
) -> str:
    words = [w for seg in transcript.segments for w in seg.words]
    confidences = [w.confidence for w in words if w.confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    low_conf_count = sum(1 for c in confidences if c < low_conf_threshold)

    # Speaking time per speaker
    speaking_time: dict[str, float] = defaultdict(float)
    for seg in transcript.segments:
        speaking_time[seg.speaker_id] += seg.end - seg.start

    # Confidence histogram
    histogram = _build_histogram(confidences)

    open_questions = sum(1 for q in result.questions if not q.is_answered)
    entity_count = result.entities.total_count

    data = {
        "audio_file": transcript.audio_path.name,
        "audio_duration_seconds": round(transcript.duration, 2),
        "processing_time_seconds": round(elapsed_seconds, 2),
        "language": transcript.language,
        "profile": profile_name,
        "word_count": transcript.word_count,
        "speaker_count": len(transcript.speakers),
        "segment_count": len(transcript.segments),
        "corrections_applied": len(corrections),
        "action_items": len(result.action_items),
        "decisions": len(result.decisions),
        "questions_total": len(result.questions),
        "open_questions": open_questions,
        "named_entities": entity_count,
        "timeline_events": len(result.timeline),
        "speaking_time": {k: round(v, 2) for k, v in sorted(speaking_time.items())},
        "reading_time_minutes": round(transcript.word_count / 200.0, 2),
        "topics": len(result.topics),
        "ai_enhanced": result.ai_enhanced,
        "confidence": {
            "average": round(avg_conf, 4) if avg_conf is not None else None,
            "low_count": low_conf_count,
            "low_threshold": low_conf_threshold,
            "words_with_score": len(confidences),
            "histogram": histogram,
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def _fmt_ts(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"[{m:02d}:{s:02d}]"


def _build_histogram(confidences: list[float]) -> dict[str, int]:
    bins = {
        "0.0-0.5": 0,
        "0.5-0.6": 0,
        "0.6-0.7": 0,
        "0.7-0.8": 0,
        "0.8-0.9": 0,
        "0.9-1.0": 0,
    }
    for c in confidences:
        if c < 0.5:
            bins["0.0-0.5"] += 1
        elif c < 0.6:
            bins["0.5-0.6"] += 1
        elif c < 0.7:
            bins["0.6-0.7"] += 1
        elif c < 0.8:
            bins["0.7-0.8"] += 1
        elif c < 0.9:
            bins["0.8-0.9"] += 1
        else:
            bins["0.9-1.0"] += 1
    return bins
