"""
Unit tests for transcript_engine.identity.embedding_extractor.compute_embedding_windows.

Pure windowing-policy tests — no model, no audio file. Verifies the
duration policy documented in embedding_extractor.py: too-short segments are
dropped, normal segments become one window, long segments are split.
"""

from __future__ import annotations

from transcript_engine.identity.embedding_extractor import (
    WINDOW_MAX_S,
    WINDOW_MIN_S,
    WINDOW_TARGET_S,
    compute_embedding_windows,
)


def test_segment_shorter_than_minimum_is_dropped():
    segments = [("SPEAKER_00", 0.0, WINDOW_MIN_S - 0.1)]
    assert compute_embedding_windows(segments) == []


def test_segment_within_range_becomes_one_window():
    segments = [("SPEAKER_00", 10.0, 10.0 + WINDOW_TARGET_S)]
    windows = compute_embedding_windows(segments)
    assert len(windows) == 1
    assert windows[0].speaker_id == "SPEAKER_00"
    assert windows[0].start == 10.0
    assert windows[0].end == 10.0 + WINDOW_TARGET_S


def test_segment_at_exactly_minimum_duration_is_kept():
    segments = [("SPEAKER_00", 0.0, WINDOW_MIN_S)]
    assert len(compute_embedding_windows(segments)) == 1


def test_long_segment_is_split_into_target_sized_windows():
    total = WINDOW_MAX_S * 2.5  # forces at least two full-size splits
    segments = [("SPEAKER_00", 0.0, total)]
    windows = compute_embedding_windows(segments)

    assert all(w.speaker_id == "SPEAKER_00" for w in windows)
    # Every window respects the policy bounds.
    for w in windows:
        duration = w.end - w.start
        assert WINDOW_MIN_S <= duration <= WINDOW_MAX_S + WINDOW_TARGET_S
    # Windows are contiguous and cover the whole segment.
    assert windows[0].start == 0.0
    assert windows[-1].end == total
    for a, b in zip(windows, windows[1:], strict=False):
        assert a.end == b.start


def test_split_never_drops_the_tail_of_a_long_segment():
    # Whatever total duration, the last window must reach exactly the
    # segment's end — split speech is never silently truncated at a
    # boundary, regardless of how the remainder divides.
    for total in (WINDOW_MAX_S + 0.01, WINDOW_MAX_S + WINDOW_MIN_S, WINDOW_MAX_S * 3 + 1.7):
        segments = [("SPEAKER_00", 0.0, total)]
        windows = compute_embedding_windows(segments)
        assert windows[-1].end == total
        assert windows[0].start == 0.0


def test_multiple_speakers_are_windowed_independently():
    segments = [
        ("SPEAKER_00", 0.0, 5.0),
        ("SPEAKER_01", 5.0, 10.0),
        ("SPEAKER_00", 10.0, 15.0),
    ]
    windows = compute_embedding_windows(segments)
    speaker_ids = [w.speaker_id for w in windows]
    assert speaker_ids == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_empty_input_returns_empty_list():
    assert compute_embedding_windows([]) == []
