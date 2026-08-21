"""
Orchestrates the full speaker-identification vertical slice:

    Transcript (already has segments attributed to SPEAKER_00/01/...)
      + diarization segments (for embedding windows)
      + audio file
      + known SpeakerProfiles
    ↓
    self-identification evidence, per speaker
    voice embedding, per speaker (this job's occurrence)
    ↓
    match against known profiles / bootstrap a new profile from a
    self-introduction
    ↓
    Transcript.speakers[speaker_id] = display name   (only when confident)
    profile store updated                             (only when confident)

One simplification versus the generic "find the speaker active at this
timestamp" design in the brief: it is unnecessary here. VERIFIED against
transcript_engine/models/transcript.py — Segment.speaker_id is a required,
already-assigned field ("A consecutive run of words from one speaker"), so
a self-identification statement found inside a given Segment's text is
already attributed to that segment's speaker_id by construction. No separate
timestamp/diarization lookup is needed to answer "who said this".

Confidence policy (PROPOSED — this decision table is new design, not
extracted from anywhere in the repo, and is unvalidated against real audio):

  1. Voice matches a known profile at HIGH confidence
       → assign that name, confidence HIGH, profile enrolled with the new
         embeddings (section 11: only high-confidence matches may update a
         profile, to avoid poisoning it with a false match).
  2. No strong voice match, but this speaker's own segments contain a HIGH
     confidence self-identification ("I'm Neel")
       → this is the automatic identity anchor from section 8/10: create
         (or attach to an existing name-only) profile for that name from
         this speaker's own embeddings, confidence HIGH, evidence recorded
         as "self_identification:high".
  3. Voice matches a known profile at MEDIUM confidence only
       → assign that name for *this transcript only*, confidence MEDIUM.
         The profile is NOT updated (section 11: medium evidence must never
         silently enroll new material).
  4. Otherwise
       → Unknown. Transcript.speakers is left unset for that speaker_id
         rather than written with a placeholder string: the existing
         fallback behaviour (display_name() returns the raw speaker_id when
         absent from the map) is what the rest of the system already relies
         on, and overwriting it with an invented "Unknown Speaker" string
         was not verified to be compatible with every consumer of
         Transcript.speakers, so it is deliberately left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from transcript_engine.identity.embedding_extractor import (
    SpeakerEmbeddingExtractor,
    compute_embedding_windows,
)
from transcript_engine.identity.matcher import match_embedding
from transcript_engine.identity.profile import SpeakerProfile
from transcript_engine.identity.self_identification import extract_self_identifications
from transcript_engine.identity.store import SpeakerProfileStore
from transcript_engine.logging import get_logger
from transcript_engine.models.transcript import Transcript

logger = get_logger(__name__)

Confidence = Literal["high", "medium", "unknown"]


@dataclass(frozen=True)
class IdentificationResult:
    """One speaker_id's outcome, kept for logging/audit/tests — this is the
    human-readable explanation the mission brief asks confidence scoring to
    support (section 9)."""

    speaker_id: str
    display_name: str | None
    confidence: Confidence
    evidence: list[str] = field(default_factory=list)


def _mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def identify_speakers(
    transcript: Transcript,
    audio_path: Path,
    diarization_segments: list[tuple[str, float, float]],
    extractor: SpeakerEmbeddingExtractor,
    store: SpeakerProfileStore,
    job_id: str,
) -> tuple[Transcript, list[IdentificationResult]]:
    # 1. Self-identification evidence, grouped by the speaker who said it.
    #    Highest-confidence match per speaker wins if more than one is found.
    self_id_by_speaker: dict[str, tuple[str, str]] = {}  # speaker_id -> (name, confidence)
    for seg in transcript.segments:
        for candidate in extract_self_identifications(seg.text):
            existing = self_id_by_speaker.get(seg.speaker_id)
            if existing is None or (existing[1] == "medium" and candidate.confidence == "high"):
                self_id_by_speaker[seg.speaker_id] = (candidate.name, candidate.confidence)

    # 2. Embeddings for every speaker with qualifying speech, in one batch.
    windows = compute_embedding_windows(diarization_segments)
    embeddings_by_index = extractor.extract(audio_path, windows)
    embeddings_by_speaker: dict[str, list[np.ndarray]] = {}
    for i, window in enumerate(windows):
        if i not in embeddings_by_index:
            continue
        embeddings_by_speaker.setdefault(window.speaker_id, []).append(embeddings_by_index[i])

    known_profiles = store.load_all()
    new_speakers: dict[str, str] = {}
    results: list[IdentificationResult] = []

    for speaker_id in transcript.speaker_ids:
        speaker_embeddings = embeddings_by_speaker.get(speaker_id, [])
        self_id = self_id_by_speaker.get(speaker_id)
        job_centroid = _mean_embedding(speaker_embeddings) if speaker_embeddings else None

        match = match_embedding(job_centroid, known_profiles) if job_centroid is not None else None

        if match is not None and match.confidence == "high" and match.profile_id is not None:
            profile = next(p for p in known_profiles if p.profile_id == match.profile_id)
            for emb in speaker_embeddings:
                profile.add_embedding(emb, job_id=job_id, evidence=f"voice_match:high:{match.similarity:.3f}")
            store.save(profile)
            new_speakers[speaker_id] = profile.display_name
            results.append(
                IdentificationResult(
                    speaker_id, profile.display_name, "high",
                    [f"voice_match:high:{match.similarity:.3f}"],
                )
            )
            continue

        if self_id is not None and self_id[1] == "high":
            name = self_id[0]
            existing_profile = next((p for p in known_profiles if p.display_name == name), None)
            profile = existing_profile or SpeakerProfile(display_name=name, confirmation_status="inferred")
            if existing_profile is None:
                known_profiles.append(profile)
            for emb in speaker_embeddings:
                profile.add_embedding(emb, job_id=job_id, evidence="self_identification:high")
            store.save(profile)
            new_speakers[speaker_id] = name
            results.append(
                IdentificationResult(speaker_id, name, "high", ["self_identification:high"])
            )
            continue

        if match is not None and match.confidence == "medium" and match.display_name is not None:
            # Section 11: medium evidence names this transcript's speaker but
            # must never enroll — no store.save() here.
            new_speakers[speaker_id] = match.display_name
            results.append(
                IdentificationResult(
                    speaker_id, match.display_name, "medium",
                    [f"voice_match:medium:{match.similarity:.3f}"],
                )
            )
            continue

        results.append(IdentificationResult(speaker_id, None, "unknown", []))

    updated_transcript = transcript.with_speakers({**transcript.speakers, **new_speakers})
    return updated_transcript, results
