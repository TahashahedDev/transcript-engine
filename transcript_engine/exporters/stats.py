"""
Stats Exporter — meeting.stats.json

Generates a machine-readable summary of a completed pipeline run.
"""

from __future__ import annotations

import json

from transcript_engine.models.correction import CorrectionRecord
from transcript_engine.models.transcript import Transcript


def export_stats(
    transcript: Transcript,
    elapsed_seconds: float,
    corrections: list[CorrectionRecord],
    profile_name: str,
    low_conf_threshold: float = 0.65,
) -> str:
    """Return a JSON string with pipeline run statistics."""
    words = [w for seg in transcript.segments for w in seg.words]
    confidences = [w.confidence for w in words if w.confidence is not None]

    avg_conf = sum(confidences) / len(confidences) if confidences else None
    low_conf_count = sum(1 for c in confidences if c < low_conf_threshold)

    histogram = _build_histogram(confidences)

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
        "confidence": {
            "average": round(avg_conf, 4) if avg_conf is not None else None,
            "low_count": low_conf_count,
            "low_threshold": low_conf_threshold,
            "words_with_score": len(confidences),
            "histogram": histogram,
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


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
