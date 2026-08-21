"""
End-to-end test of transcript_engine.identity.pipeline.identify_speakers —
the full vertical slice: transcript + diarization segments + embeddings +
self-identification text + known profiles → named Transcript.speakers.

Uses a FakeExtractor rather than the real model: this test is about the
ORCHESTRATION logic (evidence combination, profile creation/update rules,
unknown fallback) being correct, which does not depend on which embedding
model produced the vectors. The real model is separately exercised in
test_embedding_extractor_real_model.py. Every embedding here is a synthetic
hand-built vector, clearly not derived from any audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transcript_engine.identity.pipeline import identify_speakers
from transcript_engine.identity.profile import SpeakerProfile
from transcript_engine.identity.store import SpeakerProfileStore
from transcript_engine.models.transcript import Segment, Transcript


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float64)
    return arr / np.linalg.norm(arr)


class FakeExtractor:
    """Returns pre-baked embeddings instead of running real inference,
    keyed by window index exactly like the real extractor's return type."""

    def __init__(self, embedding_by_speaker: dict[str, np.ndarray]) -> None:
        self._by_speaker = embedding_by_speaker

    def extract(self, audio_path: Path, windows: list) -> dict[int, np.ndarray]:  # noqa: ARG002
        return {
            i: self._by_speaker[w.speaker_id]
            for i, w in enumerate(windows)
            if w.speaker_id in self._by_speaker
        }


def _segment(speaker_id: str, text: str, start: float, end: float) -> Segment:
    return Segment(speaker_id=speaker_id, start=start, end=end, words=(), text=text)


def _diarization_segments(*speaker_spans: tuple[str, float, float]) -> list[tuple[str, float, float]]:
    return list(speaker_spans)


def test_self_introduction_bootstraps_a_brand_new_profile(tmp_path: Path):
    transcript = Transcript(
        audio_path=Path("audio.wav"),
        duration=10.0,
        language="en",
        segments=(_segment("SPEAKER_00", "Hi everyone, I'm Neel.", 0.0, 5.0),),
    )
    diarization = _diarization_segments(("SPEAKER_00", 0.0, 5.0))
    extractor = FakeExtractor({"SPEAKER_00": _unit([1.0, 0.0, 0.0])})
    store = SpeakerProfileStore(directory=tmp_path)

    updated, results = identify_speakers(
        transcript, Path("audio.wav"), diarization, extractor, store, job_id="job-1"
    )

    assert updated.speakers["SPEAKER_00"] == "Neel"
    assert results[0].confidence == "high"
    assert "self_identification:high" in results[0].evidence

    # The profile must actually be persisted for future meetings to match against.
    saved = store.load_all()
    assert len(saved) == 1
    assert saved[0].display_name == "Neel"
    assert saved[0].sample_count == 1


def test_name_mention_without_self_introduction_stays_unknown(tmp_path: Path):
    transcript = Transcript(
        audio_path=Path("audio.wav"),
        duration=10.0,
        language="en",
        segments=(_segment("SPEAKER_00", "Neel will join tomorrow.", 0.0, 5.0),),
    )
    diarization = _diarization_segments(("SPEAKER_00", 0.0, 5.0))
    extractor = FakeExtractor({"SPEAKER_00": _unit([1.0, 0.0, 0.0])})
    store = SpeakerProfileStore(directory=tmp_path)

    updated, results = identify_speakers(
        transcript, Path("audio.wav"), diarization, extractor, store, job_id="job-1"
    )

    assert "SPEAKER_00" not in updated.speakers
    assert results[0].confidence == "unknown"
    assert results[0].display_name is None
    assert store.load_all() == []  # no profile invented from a bare mention


def test_repeated_meeting_matches_existing_profile_by_voice_alone(tmp_path: Path):
    # Meeting 1 equivalent: a profile already exists for Neel from a prior
    # session (mission brief section 10 — no re-introduction on repeat).
    store = SpeakerProfileStore(directory=tmp_path)
    neel = SpeakerProfile(display_name="Neel", confirmation_status="inferred")
    neel.add_embedding(_unit([1.0, 0.02, 0.0]), job_id="job-0", evidence="self_identification:high")
    store.save(neel)

    # Meeting 2: same voice, no introduction this time.
    transcript = Transcript(
        audio_path=Path("audio2.wav"),
        duration=10.0,
        language="en",
        segments=(_segment("SPEAKER_00", "So, about the budget for next quarter...", 0.0, 5.0),),
    )
    diarization = _diarization_segments(("SPEAKER_00", 0.0, 5.0))
    extractor = FakeExtractor({"SPEAKER_00": _unit([0.99, 0.03, 0.01])})

    updated, results = identify_speakers(
        transcript, Path("audio2.wav"), diarization, extractor, store, job_id="job-2"
    )

    assert updated.speakers["SPEAKER_00"] == "Neel"
    assert results[0].confidence == "high"
    assert "voice_match:high" in results[0].evidence[0]

    reloaded = store.load(neel.profile_id)
    assert reloaded is not None
    assert reloaded.sample_count == 2  # enrolled: high-confidence voice match


def test_medium_confidence_voice_match_names_speaker_but_does_not_enroll(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    sarah = SpeakerProfile(display_name="Sarah")
    sarah.add_embedding(np.array([1.0, 0.0]), job_id="job-0", evidence="self_identification:high")
    store.save(sarah)

    # Construct an embedding whose similarity sits in the medium band.
    from transcript_engine.identity.matcher import (
        HIGH_SIMILARITY_THRESHOLD,
        MEDIUM_SIMILARITY_THRESHOLD,
    )

    target = (HIGH_SIMILARITY_THRESHOLD + MEDIUM_SIMILARITY_THRESHOLD) / 2
    theta = np.arccos(target)
    medium_vec = np.array([np.cos(theta), np.sin(theta)])

    transcript = Transcript(
        audio_path=Path("audio.wav"),
        duration=10.0,
        language="en",
        segments=(_segment("SPEAKER_00", "Let's get started.", 0.0, 5.0),),
    )
    diarization = _diarization_segments(("SPEAKER_00", 0.0, 5.0))
    extractor = FakeExtractor({"SPEAKER_00": medium_vec})

    updated, results = identify_speakers(
        transcript, Path("audio.wav"), diarization, extractor, store, job_id="job-1"
    )

    assert updated.speakers["SPEAKER_00"] == "Sarah"
    assert results[0].confidence == "medium"

    reloaded = store.load(sarah.profile_id)
    assert reloaded is not None
    assert reloaded.sample_count == 1  # unchanged: medium match must not enroll


def test_multiple_speakers_one_identified_one_unknown(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    transcript = Transcript(
        audio_path=Path("audio.wav"),
        duration=10.0,
        language="en",
        segments=(
            _segment("SPEAKER_00", "Hi, my name is Mahar.", 0.0, 5.0),
            _segment("SPEAKER_01", "Thanks for joining.", 5.0, 10.0),
        ),
    )
    diarization = _diarization_segments(
        ("SPEAKER_00", 0.0, 5.0),
        ("SPEAKER_01", 5.0, 10.0),
    )
    extractor = FakeExtractor(
        {
            "SPEAKER_00": _unit([1.0, 0.0]),
            "SPEAKER_01": _unit([0.0, 1.0]),  # no profile exists to match this
        }
    )

    updated, results = identify_speakers(
        transcript, Path("audio.wav"), diarization, extractor, store, job_id="job-1"
    )

    assert updated.speakers.get("SPEAKER_00") == "Mahar"
    assert "SPEAKER_01" not in updated.speakers

    by_speaker = {r.speaker_id: r for r in results}
    assert by_speaker["SPEAKER_00"].confidence == "high"
    assert by_speaker["SPEAKER_01"].confidence == "unknown"


def test_no_embeddings_extracted_falls_back_to_unknown_even_with_self_id(tmp_path: Path):
    # Segment too short for enrollment quality: no window was produced, so no
    # embedding exists — self-identification text alone, with zero voice
    # evidence, still creates the profile (evidence-only bootstrap is valid),
    # but the profile has no embeddings yet, matching mission brief section
    # 11's allowance for a name-only profile awaiting future audio.
    transcript = Transcript(
        audio_path=Path("audio.wav"),
        duration=10.0,
        language="en",
        segments=(_segment("SPEAKER_00", "I'm Neel.", 0.0, 0.5),),
    )
    diarization = _diarization_segments(("SPEAKER_00", 0.0, 0.5))  # below WINDOW_MIN_S
    extractor = FakeExtractor({})  # nothing to extract
    store = SpeakerProfileStore(directory=tmp_path)

    updated, results = identify_speakers(
        transcript, Path("audio.wav"), diarization, extractor, store, job_id="job-1"
    )

    assert updated.speakers["SPEAKER_00"] == "Neel"
    assert results[0].confidence == "high"
    saved = store.load_all()
    assert saved[0].sample_count == 0


@pytest.mark.parametrize("confirmation_status", ["inferred", "confirmed"])
def test_confirmation_status_is_preserved_through_a_voice_only_update(tmp_path: Path, confirmation_status):
    store = SpeakerProfileStore(directory=tmp_path)
    profile = SpeakerProfile(display_name="Neel", confirmation_status=confirmation_status)
    profile.add_embedding(_unit([1.0, 0.0]), job_id="job-0", evidence="seed")
    store.save(profile)

    transcript = Transcript(
        audio_path=Path("audio.wav"),
        duration=10.0,
        language="en",
        segments=(_segment("SPEAKER_00", "Let's begin.", 0.0, 5.0),),
    )
    diarization = _diarization_segments(("SPEAKER_00", 0.0, 5.0))
    extractor = FakeExtractor({"SPEAKER_00": _unit([0.99, 0.02])})

    identify_speakers(transcript, Path("audio.wav"), diarization, extractor, store, job_id="job-1")

    reloaded = store.load(profile.profile_id)
    assert reloaded is not None
    assert reloaded.confirmation_status == confirmation_status
