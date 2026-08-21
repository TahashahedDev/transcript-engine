"""
SpeakerProfile: a persistent identity, built from one or more voice
embeddings collected across meetings.

Deliberately not a database model. Section 12 of the mission brief asks
whether a vector database is justified before reaching for one: at the scale
Echo actually has today (zero speaker embeddings extracted, zero profiles
stored, an unwired Postgres instance per the repo audit), a vector DB is not
justified. A profile is a handful of ~192-512 float embeddings — trivial for
NumPy and a JSON file. See store.py for the persistence layer this feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

# Minimum speech duration for a segment to be usable as enrollment/profile
# material (section 17). This is a conservative starting heuristic, not a
# figure derived from measurement: no real speaker-embedding model has been
# run against real audio in this repo, so there is nothing to tune it
# against yet. Flagged as PROPOSED in the engineering report, not VERIFIED.
MIN_ENROLLMENT_DURATION_S = 2.0

# Below this word count a segment is disproportionately likely to be a bare
# acknowledgement ("yeah", "okay") rather than sustained speech, even if it
# clears the duration bar (e.g. a slow "um... yeah...").
MIN_ENROLLMENT_WORD_COUNT = 3


def segment_is_enrollment_quality(*, duration_s: float, word_count: int) -> bool:
    """
    Whether a diarized segment is clean enough to contribute a voice
    embedding to a profile.

    Deliberately does not check for overlapping speech: the current
    diarization output (transcript_engine/diarization/pyannote_engine.py)
    does not carry an overlap-probability or per-segment quality field, so
    that check cannot be implemented without first extending diarization's
    output — see IDENTITY_ARCHITECTURE.md, open question OQ-3.
    """
    return duration_s >= MIN_ENROLLMENT_DURATION_S and word_count >= MIN_ENROLLMENT_WORD_COUNT


@dataclass
class SpeakerProfile:
    """
    One known person, represented as a small set of voice embeddings plus
    their centroid, and the evidence that justified creating the profile.
    """

    display_name: str
    profile_id: str = field(default_factory=lambda: str(uuid4()))
    embeddings: list[np.ndarray] = field(default_factory=list)
    source_job_ids: list[str] = field(default_factory=list)
    identity_evidence: list[str] = field(default_factory=list)
    # "confirmed": a human explicitly assigned this name.
    # "inferred": assigned automatically from a self-identification statement
    #   and never corrected — still shown to users as an active guess, not a
    #   verified fact (mission brief section 9: never present an uncertain
    #   voice match as a settled identity).
    confirmation_status: str = "inferred"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_embedding(self, embedding: np.ndarray, *, job_id: str, evidence: str) -> None:
        """
        Add one embedding to the profile and recompute nothing eagerly —
        centroid is a property, computed on read, so a profile with many
        embeddings added in a loop only pays the aggregation cost once per
        read rather than once per add.
        """
        self.embeddings.append(np.asarray(embedding, dtype=np.float64))
        if job_id not in self.source_job_ids:
            self.source_job_ids.append(job_id)
        self.identity_evidence.append(evidence)
        self.updated_at = datetime.now(UTC)

    @property
    def centroid(self) -> np.ndarray | None:
        """
        Mean of all stored embeddings, L2-normalized so cosine similarity
        against it behaves consistently regardless of how many embeddings
        were averaged in. None for a profile with no embeddings yet (e.g.
        one created purely from a self-identification statement, before any
        qualifying audio has been matched to it).
        """
        if not self.embeddings:
            return None
        mean = np.mean(np.stack(self.embeddings), axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            return mean
        return mean / norm

    @property
    def sample_count(self) -> int:
        return len(self.embeddings)
