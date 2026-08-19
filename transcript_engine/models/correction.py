from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorrectionRecord:
    original: str
    replacement: str
    reason: str
    confidence: float
    profile: str
    segment_id: str
