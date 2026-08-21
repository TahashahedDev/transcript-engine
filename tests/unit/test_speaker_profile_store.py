"""
Unit tests for transcript_engine.identity.store.SpeakerProfileStore.

Verifies profiles (including their embeddings) survive a save/reload cycle —
the property persistent cross-meeting identity actually depends on, per
mission brief section 10 (repeated meetings should not require re-introduction
every time).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from transcript_engine.identity.profile import SpeakerProfile
from transcript_engine.identity.store import SpeakerProfileStore


def test_save_then_load_roundtrips_all_fields(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    profile = SpeakerProfile(display_name="Neel", confirmation_status="confirmed")
    profile.add_embedding(np.array([1.0, 2.0, 3.0]), job_id="job-1", evidence="self-id: I'm Neel")

    store.save(profile)
    reloaded = store.load(profile.profile_id)

    assert reloaded is not None
    assert reloaded.profile_id == profile.profile_id
    assert reloaded.display_name == "Neel"
    assert reloaded.confirmation_status == "confirmed"
    assert reloaded.source_job_ids == ["job-1"]
    assert reloaded.identity_evidence == ["self-id: I'm Neel"]
    assert len(reloaded.embeddings) == 1
    np.testing.assert_array_equal(reloaded.embeddings[0], np.array([1.0, 2.0, 3.0]))


def test_load_nonexistent_profile_returns_none(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    assert store.load("does-not-exist") is None


def test_load_all_returns_every_saved_profile(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    neel = SpeakerProfile(display_name="Neel")
    mahar = SpeakerProfile(display_name="Mahar")
    store.save(neel)
    store.save(mahar)

    names = {p.display_name for p in store.load_all()}
    assert names == {"Neel", "Mahar"}


def test_load_all_on_empty_or_missing_directory_returns_empty_list(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path / "does-not-exist-yet")
    assert store.load_all() == []


def test_delete_removes_profile_and_reports_result(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    profile = SpeakerProfile(display_name="Neel")
    store.save(profile)

    assert store.delete(profile.profile_id) is True
    assert store.load(profile.profile_id) is None
    assert store.delete(profile.profile_id) is False


def test_save_does_not_leave_a_tmp_file_behind(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    profile = SpeakerProfile(display_name="Neel")
    store.save(profile)

    files = list(tmp_path.iterdir())
    assert files == [tmp_path / f"{profile.profile_id}.json"]


def test_centroid_survives_roundtrip_for_matching(tmp_path: Path):
    store = SpeakerProfileStore(directory=tmp_path)
    profile = SpeakerProfile(display_name="Neel")
    profile.add_embedding(np.array([3.0, 4.0]), job_id="job-1", evidence="a")

    store.save(profile)
    reloaded = store.load(profile.profile_id)

    assert reloaded is not None
    centroid = reloaded.centroid
    assert centroid is not None
    np.testing.assert_allclose(centroid, np.array([0.6, 0.8]))
