"""
Unit tests for transcript_engine.identity.matcher and .profile.

IMPORTANT: every embedding vector in this file is synthetic — hand-built
unit vectors, not output from any real speaker-embedding model (none is
wired into this repository yet; see IDENTITY_ARCHITECTURE.md). These tests
verify the matching ALGORITHM (cosine similarity, threshold tiers, centroid
aggregation, unknown-speaker fallback) is correct given some vector, not
that any specific similarity threshold is correct for real voices. The
threshold values themselves remain unvalidated against real audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from transcript_engine.identity.matcher import (
    HIGH_SIMILARITY_THRESHOLD,
    MEDIUM_SIMILARITY_THRESHOLD,
    cosine_similarity,
    match_embedding,
)
from transcript_engine.identity.profile import (
    SpeakerProfile,
    segment_is_enrollment_quality,
)


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def test_cosine_similarity_identical_vectors_is_one():
    v = _unit([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_same_speaker_synthetic_vector_matches_high_confidence():
    # A profile enrolled on a vector; a near-identical vector (same speaker,
    # slightly different recording) must match with high confidence.
    neel = SpeakerProfile(display_name="Neel")
    neel.add_embedding(_unit([1.0, 0.05, 0.02, 0.0]), job_id="job-1", evidence="synthetic")

    query = _unit([0.99, 0.06, 0.03, 0.01])  # deliberately close, not identical
    result = match_embedding(query, [neel])

    assert result.confidence == "high"
    assert result.display_name == "Neel"
    assert result.similarity >= HIGH_SIMILARITY_THRESHOLD


def test_different_speaker_synthetic_vector_is_unknown():
    neel = SpeakerProfile(display_name="Neel")
    neel.add_embedding(_unit([1.0, 0.0, 0.0, 0.0]), job_id="job-1", evidence="synthetic")

    query = _unit([0.0, 1.0, 0.0, 0.0])  # orthogonal: a clearly different voice
    result = match_embedding(query, [neel])

    assert result.confidence == "unknown"
    assert result.display_name is None
    assert result.profile_id is None


def test_unknown_speaker_never_forced_onto_nearest_known_profile():
    # Mission brief section 15: even the *nearest* known profile must not be
    # returned if similarity falls below the medium threshold.
    profiles = [
        SpeakerProfile(display_name="Neel"),
        SpeakerProfile(display_name="Mahar"),
    ]
    profiles[0].add_embedding(_unit([1.0, 0.0]), job_id="job-1", evidence="synthetic")
    profiles[1].add_embedding(_unit([0.0, 1.0]), job_id="job-1", evidence="synthetic")

    query = _unit([0.5, -0.9])  # far from both, "nearest" is still not close
    result = match_embedding(query, profiles)

    assert result.confidence == "unknown"
    assert result.display_name is None


def test_medium_confidence_band_between_thresholds():
    # Construct a query whose similarity to the profile centroid sits
    # strictly between the two thresholds.
    target_similarity = (HIGH_SIMILARITY_THRESHOLD + MEDIUM_SIMILARITY_THRESHOLD) / 2
    profile = SpeakerProfile(display_name="Sarah")
    profile.add_embedding(np.array([1.0, 0.0]), job_id="job-1", evidence="synthetic")

    theta = np.arccos(target_similarity)
    query = np.array([np.cos(theta), np.sin(theta)])
    result = match_embedding(query, [profile])

    assert result.confidence == "medium"
    assert result.display_name == "Sarah"
    assert MEDIUM_SIMILARITY_THRESHOLD <= result.similarity < HIGH_SIMILARITY_THRESHOLD


def test_profile_with_no_embeddings_has_no_centroid_and_is_skipped():
    empty_profile = SpeakerProfile(display_name="Ghost")  # e.g. name-only, no audio yet
    real_profile = SpeakerProfile(display_name="Neel")
    real_profile.add_embedding(_unit([1.0, 0.0]), job_id="job-1", evidence="synthetic")

    assert empty_profile.centroid is None

    query = _unit([1.0, 0.01])
    result = match_embedding(query, [empty_profile, real_profile])
    assert result.display_name == "Neel"


def test_centroid_is_mean_of_embeddings_l2_normalized():
    profile = SpeakerProfile(display_name="Neel")
    profile.add_embedding(np.array([1.0, 0.0]), job_id="job-1", evidence="a")
    profile.add_embedding(np.array([0.0, 1.0]), job_id="job-2", evidence="b")

    centroid = profile.centroid
    assert centroid is not None
    assert np.linalg.norm(centroid) == pytest.approx(1.0)
    assert centroid[0] == pytest.approx(centroid[1])


def test_add_embedding_records_job_and_evidence_without_duplicating_job_id():
    profile = SpeakerProfile(display_name="Neel")
    profile.add_embedding(np.array([1.0, 0.0]), job_id="job-1", evidence="self-id")
    profile.add_embedding(np.array([1.0, 0.1]), job_id="job-1", evidence="repeat meeting match")

    assert profile.source_job_ids == ["job-1"]
    assert profile.sample_count == 2
    assert profile.identity_evidence == ["self-id", "repeat meeting match"]


@pytest.mark.parametrize(
    ("duration_s", "word_count", "expected"),
    [
        (3.0, 5, True),
        (1.0, 5, False),   # too short even with enough words
        (3.0, 1, False),   # long enough but a bare "yeah"/"okay" fragment
        (2.0, 3, True),    # exactly at both floors
    ],
)
def test_segment_is_enrollment_quality(duration_s, word_count, expected):
    assert segment_is_enrollment_quality(duration_s=duration_s, word_count=word_count) is expected
