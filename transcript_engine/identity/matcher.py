"""
Matches one voice embedding against a set of known SpeakerProfiles.

Deliberately compares against each profile's centroid rather than
re-clustering the whole recording from scratch (mission brief section 7):
diarization has already done the hard part of grouping this recording's
speech into per-speaker clusters, so identification only needs to answer
"which known person does this cluster's voice belong to", a lookup, not a
clustering problem.

The similarity thresholds below (HIGH_SIMILARITY_THRESHOLD /
MEDIUM_SIMILARITY_THRESHOLD) are PROPOSED starting points based on published
cosine-similarity behavior for speaker-verification embeddings in general —
they have not been tuned against this system's actual embedding model
because no embedding model is wired in yet and no multi-speaker audio exists
in this repository to test against (see IDENTITY_ARCHITECTURE.md open
questions). They must be validated against real recordings before being
trusted in production, and are exposed as module constants specifically so
that validation can override them without a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from transcript_engine.identity.profile import SpeakerProfile

Confidence = Literal["high", "medium", "unknown"]

HIGH_SIMILARITY_THRESHOLD = 0.80
MEDIUM_SIMILARITY_THRESHOLD = 0.65


@dataclass(frozen=True)
class MatchResult:
    """
    The outcome of comparing one embedding against all known profiles.
    profile_id/display_name are None when confidence is "unknown" — callers
    must not fall back to the nearest profile in that case (mission brief
    section 15: an unmatched speaker must stay Unknown, never be forced onto
    the closest known person just because they were closest).
    """

    profile_id: str | None
    display_name: str | None
    similarity: float
    confidence: Confidence


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _confidence_for(similarity: float) -> Confidence:
    if similarity >= HIGH_SIMILARITY_THRESHOLD:
        return "high"
    if similarity >= MEDIUM_SIMILARITY_THRESHOLD:
        return "medium"
    return "unknown"


def match_embedding(embedding: np.ndarray, profiles: list[SpeakerProfile]) -> MatchResult:
    """
    Compare one embedding against every profile's centroid and return the
    best match. Profiles with no embeddings yet (centroid is None — e.g. a
    profile created from a self-identification statement before any audio
    was matched to it) are skipped, not treated as a similarity of 0.
    """
    best_similarity = -1.0
    best_profile: SpeakerProfile | None = None

    for profile in profiles:
        centroid = profile.centroid
        if centroid is None:
            continue
        similarity = cosine_similarity(embedding, centroid)
        if similarity > best_similarity:
            best_similarity = similarity
            best_profile = profile

    if best_profile is None:
        return MatchResult(profile_id=None, display_name=None, similarity=0.0, confidence="unknown")

    confidence = _confidence_for(best_similarity)
    if confidence == "unknown":
        return MatchResult(profile_id=None, display_name=None, similarity=best_similarity, confidence="unknown")

    return MatchResult(
        profile_id=best_profile.profile_id,
        display_name=best_profile.display_name,
        similarity=best_similarity,
        confidence=confidence,
    )
