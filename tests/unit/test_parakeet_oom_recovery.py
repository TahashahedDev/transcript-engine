"""
Regression tests for CUDA-OOM recovery in the Parakeet engine.

The bug these guard against: an earlier implementation responded to OOM by
truncating the chunk to its first half and discarding the rest. With a 300 s
chunk that silently deleted ~150 s of speech from the transcript while the job
still reported success. Losing audio is never an acceptable response to memory
pressure — every second of input must appear in the output.

No CUDA, no NeMo: model.transcribe is faked, and OOM is simulated by raising
RuntimeError("CUDA out of memory") for segments longer than a threshold.
"""

from __future__ import annotations

import contextlib
import glob
import os
import tempfile
import wave
from pathlib import Path

import pytest

from transcript_engine.transcription import parakeet_engine as pe
from transcript_engine.transcription.parakeet_engine import (
    _split_segment_in_half,
    _transcribe_chunk_with_retry,
    _transcribe_chunked,
)

SAMPLE_RATE = 16_000


@pytest.fixture(autouse=True)
def _cleanup_temps():  # type: ignore[no-untyped-def]
    """Remove split temps these tests create in the OS temp dir."""
    pattern = os.path.join(tempfile.gettempdir(), "*_prkt_*.wav")
    before = set(glob.glob(pattern))
    yield
    for leaked in set(glob.glob(pattern)) - before:
        with contextlib.suppress(OSError):
            os.unlink(leaked)


def _make_wav(path: Path, duration_s: float) -> None:
    n_frames = int(duration_s * SAMPLE_RATE)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * n_frames)


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


class _FakeHyp:
    """Hypothesis whose words tile the segment at 1 word/second."""

    def __init__(self, duration: float, tag: str) -> None:
        self.timestep_units = None
        self.text = ""
        n = max(1, int(round(duration)))
        self.timestamp = {
            "word": [
                {"word": f"{tag}{i}", "start": float(i), "end": float(i) + 0.9}
                for i in range(n)
            ]
        }


def _oom_above(threshold_s: float):  # type: ignore[no-untyped-def]
    """
    Fake _call_transcribe that OOMs for segments longer than `threshold_s`.

    Mirrors real behaviour: memory pressure scales with segment length, so a
    shorter segment succeeds where a longer one failed.
    """
    calls: list[float] = []

    def fake_call(model, path):  # type: ignore[no-untyped-def]
        dur = _wav_duration(path)
        calls.append(dur)
        if dur > threshold_s:
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")
        return [_FakeHyp(dur, "w")]

    fake_call.calls = calls  # type: ignore[attr-defined]
    return fake_call


class TestSplitSegmentInHalf:
    def test_returns_two_contiguous_halves_covering_whole_segment(
        self, tmp_path: Path
    ) -> None:
        wav = tmp_path / "seg.wav"
        _make_wav(wav, 100.0)

        halves = _split_segment_in_half(str(wav), chunk_i=0, depth=1)

        assert len(halves) == 2
        (p0, off0, dur0), (p1, off1, dur1) = halves
        assert off0 == 0.0
        # Second half starts exactly where the first ends — no gap, no overlap.
        assert abs(off1 - dur0) < 0.01
        # Together they account for the full original duration.
        assert abs((dur0 + dur1) - 100.0) < 0.05
        for p in (p0, p1):
            assert Path(p).exists()

    def test_halves_are_written_as_readable_wavs(self, tmp_path: Path) -> None:
        wav = tmp_path / "seg.wav"
        _make_wav(wav, 40.0)

        halves = _split_segment_in_half(str(wav), chunk_i=0, depth=1)

        for path, _, declared_duration in halves:
            assert abs(_wav_duration(path) - declared_duration) < 0.01


class TestOomKeepsAllAudio:
    def test_oom_transcribes_both_halves_not_just_the_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        THE regression test. A 100 s chunk that OOMs above 60 s must still
        produce words covering the whole 0-100 s span, not just 0-50 s.
        """
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(60.0))

        words = _transcribe_chunk_with_retry(
            model=object(),
            chunk_path=str(wav),
            wav_path=str(wav),  # equal → helper never deletes the fixture
            wav_start=0.0,
            chunk_wav_duration=100.0,
            frame_duration=0.08,
            nominal_start=0.0,
            nominal_end=float("inf"),
            max_retries=2,
            chunk_i=0,
            n_chunks=1,
            ts_sources=[],
        )

        assert words, "OOM recovery produced no words at all"
        last_word_end = max(w.end for w in words)
        assert last_word_end > 90.0, (
            f"audio after {last_word_end:.0f}s was silently dropped — the second "
            f"half of the chunk was not transcribed"
        )

    def test_second_half_timestamps_are_offset_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Words from the second half must carry absolute times, not restart at 0."""
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(60.0))

        words = _transcribe_chunk_with_retry(
            model=object(),
            chunk_path=str(wav),
            wav_path=str(wav),
            wav_start=0.0,
            chunk_wav_duration=100.0,
            frame_duration=0.08,
            nominal_start=0.0,
            nominal_end=float("inf"),
            max_retries=2,
            chunk_i=0,
            n_chunks=1,
            ts_sources=[],
        )

        starts = [w.start for w in words]
        assert starts == sorted(starts), "timestamps are not monotonically increasing"
        # No duplicated 0-based restart from the second half.
        assert starts.count(0.0) <= 1, "second half restarted its timestamps at zero"

    def test_chunk_offset_is_preserved_through_oom_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk starting at 600 s must not emit words near t=0 after splitting."""
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(60.0))

        words = _transcribe_chunk_with_retry(
            model=object(),
            chunk_path=str(wav),
            wav_path=str(wav),
            wav_start=600.0,
            chunk_wav_duration=100.0,
            frame_duration=0.08,
            nominal_start=600.0,
            nominal_end=700.0,
            max_retries=2,
            chunk_i=1,
            n_chunks=2,
            ts_sources=[],
        )

        assert words
        assert min(w.start for w in words) >= 600.0
        assert max(w.start for w in words) < 700.0

    def test_repeated_oom_splits_recursively_and_still_covers_span(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two levels of splitting (100s → 50s → 25s) must still cover the chunk."""
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(30.0))

        words = _transcribe_chunk_with_retry(
            model=object(),
            chunk_path=str(wav),
            wav_path=str(wav),
            wav_start=0.0,
            chunk_wav_duration=100.0,
            frame_duration=0.08,
            nominal_start=0.0,
            nominal_end=float("inf"),
            max_retries=2,
            chunk_i=0,
            n_chunks=1,
            ts_sources=[],
        )

        assert max(w.end for w in words) > 90.0

    def test_retry_exhaustion_raises_rather_than_returning_partial_audio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        If OOM cannot be resolved, the job must fail loudly. Returning a partial
        transcript would hide missing speech behind a 'completed' status.
        """
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(0.1))  # always OOM

        with pytest.raises(RuntimeError, match="out of memory"):
            _transcribe_chunk_with_retry(
                model=object(),
                chunk_path=str(wav),
                wav_path=str(wav),
                wav_start=0.0,
                chunk_wav_duration=100.0,
                frame_duration=0.08,
                nominal_start=0.0,
                nominal_end=float("inf"),
                max_retries=2,
                chunk_i=0,
                n_chunks=1,
                ts_sources=[],
            )

    def test_non_oom_error_is_not_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        calls = {"n": 0}

        def boom(model, path):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        monkeypatch.setattr(pe, "_call_transcribe", boom)

        with pytest.raises(RuntimeError, match="illegal memory access"):
            _transcribe_chunk_with_retry(
                model=object(),
                chunk_path=str(wav),
                wav_path=str(wav),
                wav_start=0.0,
                chunk_wav_duration=100.0,
                frame_duration=0.08,
                nominal_start=0.0,
                nominal_end=float("inf"),
                max_retries=2,
                chunk_i=0,
                n_chunks=1,
                ts_sources=[],
            )

        assert calls["n"] == 1, "a non-OOM failure must not trigger the split path"


class TestTempFileHygieneUnderOom:
    def test_no_temp_files_survive_successful_oom_recovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(60.0))

        pattern = os.path.join(tempfile.gettempdir(), "*_prkt_split_*.wav")
        before = set(glob.glob(pattern))

        _transcribe_chunk_with_retry(
            model=object(),
            chunk_path=str(wav),
            wav_path=str(wav),
            wav_start=0.0,
            chunk_wav_duration=100.0,
            frame_duration=0.08,
            nominal_start=0.0,
            nominal_end=float("inf"),
            max_retries=2,
            chunk_i=0,
            n_chunks=1,
            ts_sources=[],
        )

        assert set(glob.glob(pattern)) == before, "split temps leaked after success"

    def test_no_temp_files_survive_retry_exhaustion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wav = tmp_path / "chunk.wav"
        _make_wav(wav, 100.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(0.1))

        pattern = os.path.join(tempfile.gettempdir(), "*_prkt_split_*.wav")
        before = set(glob.glob(pattern))

        with pytest.raises(RuntimeError):
            _transcribe_chunk_with_retry(
                model=object(),
                chunk_path=str(wav),
                wav_path=str(wav),
                wav_start=0.0,
                chunk_wav_duration=100.0,
                frame_duration=0.08,
                nominal_start=0.0,
                nominal_end=float("inf"),
                max_retries=2,
                chunk_i=0,
                n_chunks=1,
                ts_sources=[],
            )

        assert set(glob.glob(pattern)) == before, "split temps leaked after failure"

    def test_original_chunk_temp_is_deleted_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When chunk_path is a real temp (not the source wav), it must be removed."""
        source = tmp_path / "source.wav"
        chunk = tmp_path / "chunk_prkt_000.wav"
        _make_wav(source, 200.0)
        _make_wav(chunk, 50.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(1000.0))  # never OOM

        _transcribe_chunk_with_retry(
            model=object(),
            chunk_path=str(chunk),
            wav_path=str(source),
            wav_start=0.0,
            chunk_wav_duration=50.0,
            frame_duration=0.08,
            nominal_start=0.0,
            nominal_end=float("inf"),
            max_retries=2,
            chunk_i=0,
            n_chunks=1,
            ts_sources=[],
        )

        assert not chunk.exists(), "chunk temp was not cleaned up"
        assert source.exists(), "source audio must never be deleted"


class TestFullChunkedPassUnderOom:
    """
    End-to-end over the chunked pass: multiple chunks, OOM in the middle.

    Guards the interaction the chunk-level tests can't see — that OOM splitting
    inside one chunk still leaves the seam with its neighbours gap-free and
    duplicate-free.
    """

    def test_multi_chunk_audio_with_oom_has_no_gap_and_no_duplicates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wav = tmp_path / "long.wav"
        _make_wav(wav, 600.0)  # 10 min → 2 chunks at 300 s
        # 300 s chunks (+overlap) exceed the 200 s threshold, so EVERY chunk
        # OOMs once and must recover by splitting.
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(200.0))

        words, _ = _transcribe_chunked(
            model=object(),
            wav_path=str(wav),
            audio_duration=600.0,
            frame_duration=0.08,
            chunk_seconds=300.0,
            overlap_seconds=0.5,
            max_retries=2,
            on_progress=None,
        )

        assert words, "no words returned from chunked pass"

        starts = [w.start for w in words]
        assert starts == sorted(starts), "words are not in chronological order"
        assert len(starts) == len(set(starts)), "duplicate word timestamps across seams"

        # Coverage: the 1 word/second fake means a gap shows up as a jump.
        assert min(starts) < 5.0, "beginning of audio missing"
        assert max(starts) > 590.0, "end of audio missing"
        biggest_gap = max(b - a for a, b in zip(starts, starts[1:], strict=False))
        assert biggest_gap < 5.0, (
            f"{biggest_gap:.0f}s gap in the transcript — audio was dropped at a "
            f"chunk seam or inside an OOM split"
        )

    def test_no_temp_files_leak_across_full_chunked_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wav = tmp_path / "long.wav"
        _make_wav(wav, 600.0)
        monkeypatch.setattr(pe, "_call_transcribe", _oom_above(200.0))

        pattern = os.path.join(tempfile.gettempdir(), "*_prkt_*.wav")
        before = set(glob.glob(pattern))

        _transcribe_chunked(
            model=object(),
            wav_path=str(wav),
            audio_duration=600.0,
            frame_duration=0.08,
            chunk_seconds=300.0,
            overlap_seconds=0.5,
            max_retries=2,
            on_progress=None,
        )

        assert set(glob.glob(pattern)) == before, "chunk/split temps leaked"
