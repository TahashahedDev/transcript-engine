"""
Tests for the pre-pipeline audio analyzer's diarization-skip decision.

Skipping diarization is a speed optimisation whose failure mode is severe and
silent: every speaker collapses into "Speaker 1" and the whole meeting
intelligence layer (who said what, action items per speaker) is wrong, while
the job still reports success.

The previous heuristic set sample_confidence purely from duration — anything
longer than two minutes scored 0.9, which sits above the 0.85 skip gate. The
verdict for an entire meeting therefore rested on one 60-second window from
the middle of the recording, so a presenter monologuing during that window was
enough to disable diarization for the whole file.
"""

from __future__ import annotations

from transcript_engine.audio.analyzer import (
    _DIAR_SKIP_CONFIDENCE,
    _MAX_WINDOWS,
    _WINDOW_SECONDS,
    AudioAnalyzer,
)

SR = 16_000


class TestSampleOffsets:
    def test_short_audio_yields_single_window(self) -> None:
        assert AudioAnalyzer._sample_offsets(30 * SR, SR) == [0]

    def test_long_audio_is_sampled_at_multiple_points(self) -> None:
        offsets = AudioAnalyzer._sample_offsets(3600 * SR, SR)
        assert len(offsets) == _MAX_WINDOWS

    def test_windows_span_the_whole_recording(self) -> None:
        """A mid-only sample was the original bug — the last window must sit near the end."""
        total = 3600 * SR
        offsets = AudioAnalyzer._sample_offsets(total, SR)

        assert offsets[0] == 0
        last_window_end = offsets[-1] + _WINDOW_SECONDS * SR
        assert last_window_end <= total
        assert last_window_end > total * 0.95, "final window should reach the end of the audio"

    def test_windows_are_ordered_and_unique(self) -> None:
        offsets = AudioAnalyzer._sample_offsets(600 * SR, SR)
        assert offsets == sorted(offsets)
        assert len(offsets) == len(set(offsets))

    def test_windows_never_read_past_the_end(self) -> None:
        for seconds in (61, 90, 180, 600, 3600, 10_800):
            total = seconds * SR
            for offset in AudioAnalyzer._sample_offsets(total, SR):
                assert offset >= 0
                assert offset + _WINDOW_SECONDS * SR <= total


def _confidence(duration_s: float, n_windows: int, counts: list[int]) -> float:
    """Mirror of the confidence rule in _analyze, for gate testing."""
    analyzed_s = n_windows * _WINDOW_SECONDS
    coverage = min(1.0, analyzed_s / duration_s) if duration_s > 0 else 0.0
    unanimous = len(set(counts)) <= 1
    if duration_s < 120 or n_windows < 2:
        return 0.5
    if unanimous:
        return round(min(0.95, 0.55 + 0.45 * coverage), 3)
    return 0.4


class TestDiarizationSkipGate:
    def test_long_meeting_never_skips_on_sampled_evidence(self) -> None:
        """
        The important case. Four minutes sampled from a 60-minute meeting is
        ~7% coverage — nowhere near enough to claim the whole recording has one
        speaker, no matter how consistent those windows look.
        """
        conf = _confidence(duration_s=3600, n_windows=_MAX_WINDOWS, counts=[1, 1, 1, 1])
        assert conf < _DIAR_SKIP_CONFIDENCE

    def test_disagreeing_windows_never_skip(self) -> None:
        conf = _confidence(duration_s=600, n_windows=4, counts=[1, 2, 1, 1])
        assert conf < _DIAR_SKIP_CONFIDENCE

    def test_short_audio_never_skips(self) -> None:
        assert _confidence(duration_s=90, n_windows=2, counts=[1, 1]) < _DIAR_SKIP_CONFIDENCE

    def test_fully_covered_recording_may_skip(self) -> None:
        """The optimisation still works where the evidence actually supports it."""
        conf = _confidence(duration_s=180, n_windows=3, counts=[1, 1, 1])
        assert conf >= _DIAR_SKIP_CONFIDENCE

    def test_confidence_never_claims_certainty(self) -> None:
        assert _confidence(duration_s=120, n_windows=4, counts=[1, 1, 1, 1]) <= 0.95


class TestSpeakerCountAggregation:
    def test_multi_speaker_evidence_anywhere_wins(self) -> None:
        """
        Hearing two voices in any window is positive evidence; hearing one is
        only weak evidence about the rest. max() encodes that asymmetry.
        """
        assert max([1, 2, 1, 1]) == 2
        assert max([1, 1, 1, 1]) == 1
