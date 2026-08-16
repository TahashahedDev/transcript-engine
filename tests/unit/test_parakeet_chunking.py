"""
Unit tests for Parakeet chunked transcription logic.

All tests are pure Python — no CUDA, no NeMo, no ML models.
The WAV writing/reading uses the stdlib wave module only.
"""

from __future__ import annotations

import contextlib
import math
import wave
from pathlib import Path

import pytest

from transcript_engine.config.settings import PipelineConfig, TranscriptionConfig
from transcript_engine.gpu.hardware import GPU_COMPUTE_LOCK, GpuInfo
from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.pipeline import RawWord
from transcript_engine.transcription import parakeet_engine as pe
from transcript_engine.transcription.parakeet_engine import (
    ParakeetEngine,
    _distribute_text,
    _extract_words,
    split_wav_with_overlap,
)

# ---------------------------------------------------------------------------
# WAV test fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_parakeet_chunk_temps():  # type: ignore[no-untyped-def]
    """
    Delete chunk WAVs these tests leave in the OS temp dir.

    split_wav_with_overlap() writes chunks with delete=False because in
    production _transcribe_chunk_with_retry owns deleting them. Tests call the
    splitter directly, so nothing else cleans up — without this fixture a full
    suite run strands hundreds of MB of WAVs in the system temp dir (observed:
    384 files / 3.3 GB accumulated across repeated runs).
    """
    import glob
    import os as _os
    import tempfile as _tempfile

    pattern = _os.path.join(_tempfile.gettempdir(), "*_prkt_*.wav")
    before = set(glob.glob(pattern))
    yield
    for leaked in set(glob.glob(pattern)) - before:
        with contextlib.suppress(OSError):
            _os.unlink(leaked)


def _make_wav(path: Path, duration_s: float, sample_rate: int = 16_000) -> None:
    """Write a silent 16-bit mono WAV of the given duration."""
    n_frames = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# split_wav_with_overlap — structural tests
# ---------------------------------------------------------------------------


class TestSplitWavWithOverlap:
    def test_short_audio_returns_original_path(self, tmp_path: Path) -> None:
        """Audio shorter than chunk_seconds must return the original path unchanged."""
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=120.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        assert len(chunks) == 1
        chunk_path, wav_start, nominal_start = chunks[0]
        assert chunk_path == str(wav)
        assert wav_start == 0.0
        assert nominal_start == 0.0

    def test_exact_one_chunk_no_temp_files(self, tmp_path: Path) -> None:
        """Audio equal to chunk_seconds: single chunk, original file path."""
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=300.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        assert len(chunks) == 1
        assert chunks[0][0] == str(wav)

    def test_two_chunks_created_for_long_audio(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=600.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        assert len(chunks) == 2

    def test_nominal_starts_are_monotonically_increasing(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=900.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        nominal_starts = [c[2] for c in chunks]
        assert nominal_starts == sorted(nominal_starts)
        assert nominal_starts[0] == 0.0

    def test_wav_starts_are_non_negative(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=900.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        for _chunk_path, wav_start, _ in chunks:
            assert wav_start >= 0.0

    def test_first_chunk_starts_at_zero(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=600.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        # First chunk: no left overlap, wav starts at 0
        assert chunks[0][1] == 0.0

    def test_second_chunk_starts_before_nominal_due_to_overlap(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=600.0)
        overlap = 0.5

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=overlap)

        _, wav_start_1, nominal_start_1 = chunks[1]
        # wav_start of chunk 1 should be nominal_start - overlap
        assert abs(wav_start_1 - (nominal_start_1 - overlap)) < 0.01

    def test_chunk_wavs_have_correct_duration(self, tmp_path: Path) -> None:
        """Each chunk WAV should cover [wav_start, min(nominal_end+overlap, total)]."""
        wav = tmp_path / "test.wav"
        total = 600.0
        chunk_s = 300.0
        overlap = 0.5
        _make_wav(wav, duration_s=total)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=chunk_s, overlap_seconds=overlap)

        for i, (chunk_path, wav_start, _nominal_start) in enumerate(chunks):
            nominal_end = chunks[i + 1][2] if i + 1 < len(chunks) else total
            expected_end = min(nominal_end + overlap, total)
            expected_dur = expected_end - wav_start
            actual_dur = _wav_duration(chunk_path)
            assert abs(actual_dur - expected_dur) < 0.05, (
                f"Chunk {i}: expected ~{expected_dur:.2f}s, got {actual_dur:.2f}s"
            )

    def test_temp_files_have_different_paths_from_original(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=600.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        for chunk_path, _, _ in chunks:
            assert chunk_path != str(wav), "Chunk should be a temp file, not original"

    def test_many_chunks(self, tmp_path: Path) -> None:
        """60-minute audio with 5-minute chunks → 12 chunks."""
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=3600.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)

        assert len(chunks) == 12

    def test_zero_overlap_produces_no_overlap(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=600.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.0)

        # wav_start of chunk 1 should equal nominal_start_1 (no overlap)
        assert abs(chunks[1][1] - chunks[1][2]) < 0.01


# ---------------------------------------------------------------------------
# Boundary deduplication logic
# ---------------------------------------------------------------------------


class TestBoundaryDeduplication:
    """
    The deduplication rule implemented in _transcribe_chunk_with_retry:
      A word is 'owned' by chunk i if:  nominal_start[i] <= word.start < nominal_end[i]
    where nominal_end[i] = nominal_start[i+1] (or inf for last chunk).

    These tests verify the rule directly using synthetic word lists.
    """

    def _owned(
        self,
        words: list[RawWord],
        nominal_start: float,
        nominal_end: float,
    ) -> list[RawWord]:
        return [w for w in words if nominal_start <= w.start < nominal_end]

    def test_boundary_word_belongs_to_earlier_chunk(self) -> None:
        # Word starts exactly at the boundary (300.0s).
        # nominal chunk 0: [0, 300), chunk 1: [300, 600)
        word_at_boundary = RawWord(text="hello", start=300.0, end=300.4, confidence=None)
        word_before = RawWord(text="world", start=299.5, end=299.9, confidence=None)

        owned_by_0 = self._owned([word_before, word_at_boundary], 0.0, 300.0)
        owned_by_1 = self._owned([word_before, word_at_boundary], 300.0, float("inf"))

        assert word_before in owned_by_0
        assert word_at_boundary not in owned_by_0  # 300.0 is NOT < 300.0
        assert word_at_boundary in owned_by_1

    def test_overlap_word_before_boundary_stays_in_earlier_chunk(self) -> None:
        # A word at 299.8 should stay with chunk 0 even if chunk 1 also captures it
        # via overlap (because chunk 1's wav_start = 299.5).
        word = RawWord(text="hello", start=299.8, end=300.1, confidence=None)

        owned_by_0 = self._owned([word], 0.0, 300.0)
        owned_by_1 = self._owned([word], 300.0, float("inf"))

        assert word in owned_by_0
        assert word not in owned_by_1

    def test_no_word_is_counted_twice(self) -> None:
        words = [
            RawWord(text=f"word{i}", start=i * 10.0, end=i * 10.0 + 1.0, confidence=None)
            for i in range(60)
        ]
        boundary = 300.0

        owned_chunk0 = self._owned(words, 0.0, boundary)
        owned_chunk1 = self._owned(words, boundary, float("inf"))

        # No word appears in both
        ids_0 = {w.start for w in owned_chunk0}
        ids_1 = {w.start for w in owned_chunk1}
        assert not ids_0.intersection(ids_1)
        # All words accounted for
        assert len(owned_chunk0) + len(owned_chunk1) == len(words)

    def test_last_chunk_has_no_upper_bound(self) -> None:
        words = [
            RawWord(text="late", start=3590.0, end=3595.0, confidence=None),
            RawWord(text="last", start=3598.0, end=3600.0, confidence=None),
        ]
        owned = self._owned(words, 3300.0, float("inf"))
        assert len(owned) == 2


# ---------------------------------------------------------------------------
# _distribute_text
# ---------------------------------------------------------------------------


class TestDistributeText:
    def test_empty_text_returns_empty(self) -> None:
        assert _distribute_text("", 10.0) == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert _distribute_text("   ", 10.0) == []

    def test_single_word_spans_full_duration(self) -> None:
        words = _distribute_text("hello", 5.0)
        assert len(words) == 1
        assert words[0].text == "hello"
        assert abs(words[0].start - 0.0) < 0.01
        assert abs(words[0].end - 5.0) < 0.01

    def test_words_are_contiguous(self) -> None:
        words = _distribute_text("one two three four", 4.0)
        for i in range(len(words) - 1):
            assert abs(words[i].end - words[i + 1].start) < 0.01

    def test_last_word_ends_at_duration(self) -> None:
        words = _distribute_text("a b c", 3.0)
        assert abs(words[-1].end - 3.0) < 0.01

    def test_confidence_is_none(self) -> None:
        words = _distribute_text("hello world", 2.0)
        assert all(w.confidence is None for w in words)

    def test_word_count_matches_token_count(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        words = _distribute_text(text, 9.0)
        assert len(words) == 9


# ---------------------------------------------------------------------------
# _extract_words — fallback behaviour without NeMo
# ---------------------------------------------------------------------------


class TestExtractWords:
    def test_plain_string_input_distributes_evenly(self) -> None:
        words = _extract_words("hello world", 2.0, frame_duration=0.08)
        assert len(words) == 2
        assert words[0].text == "hello"
        assert words[1].text == "world"

    def test_object_with_no_timestamps_falls_back_to_distribute(self) -> None:
        class FakeHyp:
            text = "one two three"
            timestamp = None
            timestep_units = None

        words = _extract_words(FakeHyp(), 3.0, frame_duration=0.08)
        assert len(words) == 3

    def test_nemo2_timestamp_dict_is_preferred(self) -> None:
        class FakeHyp:
            text = "fallback"
            timestamp = {
                "word": [
                    {"word": "hello", "start": 0.1, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.0},
                ]
            }
            timestep_units = None

        words = _extract_words(FakeHyp(), 2.0, frame_duration=0.08)
        assert len(words) == 2
        assert words[0].text == "hello"
        assert abs(words[0].start - 0.1) < 0.001
        assert abs(words[1].start - 0.6) < 0.001

    def test_nemo2_empty_word_tokens_are_skipped(self) -> None:
        class FakeHyp:
            text = ""
            timestamp = {
                "word": [
                    {"word": "", "start": 0.0, "end": 0.1},
                    {"word": "hello", "start": 0.2, "end": 0.5},
                ]
            }
            timestep_units = None

        words = _extract_words(FakeHyp(), 1.0, frame_duration=0.08)
        assert len(words) == 1
        assert words[0].text == "hello"

    def test_nemo1_timestep_units_fallback(self) -> None:
        class FakeUnit:
            def __init__(self, word: str, start: int, end: int) -> None:
                self.word = word
                self.start_offset = start
                self.end_offset = end

        class FakeHyp:
            text = ""
            timestamp = None
            timestep_units = [FakeUnit("hi", 0, 2), FakeUnit("there", 3, 5)]

        words = _extract_words(FakeHyp(), 10.0, frame_duration=0.08)
        assert len(words) == 2
        assert words[0].text == "hi"
        assert abs(words[0].start - 0.0) < 0.001   # 0 * 0.08
        assert abs(words[0].end - 0.16) < 0.001    # 2 * 0.08

    def test_end_is_always_after_start(self) -> None:
        class FakeHyp:
            text = ""
            timestamp = {
                "word": [{"word": "x", "start": 1.0, "end": 1.0}]  # zero duration
            }
            timestep_units = None

        words = _extract_words(FakeHyp(), 5.0, frame_duration=0.08)
        assert len(words) == 1
        assert words[0].end > words[0].start


# ---------------------------------------------------------------------------
# Timestamp offset correctness
# ---------------------------------------------------------------------------


class TestTimestampOffset:
    """Verify that adding wav_start_s to relative timestamps gives correct absolute times."""

    def test_offset_shifts_all_timestamps(self) -> None:
        wav_start = 300.0
        relative_words = [
            RawWord(text="hello", start=0.1, end=0.5, confidence=None),
            RawWord(text="world", start=0.6, end=1.0, confidence=None),
        ]
        absolute_words = [
            RawWord(
                text=w.text,
                start=round(w.start + wav_start, 3),
                end=round(w.end + wav_start, 3),
                confidence=w.confidence,
            )
            for w in relative_words
        ]
        assert abs(absolute_words[0].start - 300.1) < 0.001
        assert abs(absolute_words[1].end - 301.0) < 0.001

    def test_offset_zero_is_identity(self) -> None:
        word = RawWord(text="test", start=1.23, end=1.78, confidence=None)
        offset_word = RawWord(
            text=word.text,
            start=round(word.start + 0.0, 3),
            end=round(word.end + 0.0, 3),
            confidence=word.confidence,
        )
        assert offset_word.start == word.start
        assert offset_word.end == word.end


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_short_audio_single_chunk(self, tmp_path: Path) -> None:
        wav = tmp_path / "short.wav"
        _make_wav(wav, duration_s=5.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)
        assert len(chunks) == 1

    def test_audio_slightly_longer_than_chunk(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=301.0)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)
        assert len(chunks) == 2

    def test_nominal_starts_sum_to_full_coverage(self, tmp_path: Path) -> None:
        """The nominal windows should cover the full audio without gaps."""
        total = 1800.0  # 30 min
        chunk_s = 300.0
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=total)

        chunks = split_wav_with_overlap(str(wav), chunk_seconds=chunk_s, overlap_seconds=0.5)

        for i, (_, _, nominal_start) in enumerate(chunks):
            expected_start = i * chunk_s
            assert abs(nominal_start - expected_start) < 0.01, (
                f"Chunk {i}: expected nominal_start={expected_start}, got {nominal_start}"
            )

    def test_chunk_count_formula(self, tmp_path: Path) -> None:
        """Number of chunks = ceil(total / chunk_s) for audio > chunk_s."""
        for total in [310.0, 600.0, 900.0, 1800.0, 3601.0]:
            wav = tmp_path / f"test_{total}.wav"
            _make_wav(wav, duration_s=total)
            chunks = split_wav_with_overlap(str(wav), chunk_seconds=300.0, overlap_seconds=0.5)
            expected = math.ceil(total / 300.0)
            assert len(chunks) == expected, f"total={total}: expected {expected} chunks, got {len(chunks)}"


# ---------------------------------------------------------------------------
# ParakeetEngine.transcribe() — always routes through the retry-protected path
#
# Regression coverage for the Phase 1 OOM fix: before this change, audio
# shorter than chunk_seconds took a direct, unprotected model.transcribe()
# call with no OOM retry at all. These tests never touch NeMo/CUDA — they
# monkeypatch _transcribe_chunked and the GPU-detection functions, so no
# torch install is required (mirrors the rest of this file).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_parakeet_engine_class_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate ParakeetEngine's class-level model/GPU cache across tests."""
    monkeypatch.setattr(ParakeetEngine, "_model", None)
    monkeypatch.setattr(ParakeetEngine, "_model_id", "")
    monkeypatch.setattr(ParakeetEngine, "_frame_duration", pe._FALLBACK_FRAME_DURATION_S)
    monkeypatch.setattr(ParakeetEngine, "_gpu_info", None)


def _prepared_audio(tmp_path: Path, duration: float) -> PreparedAudio:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"")  # never opened — _transcribe_chunked is monkeypatched
    return PreparedAudio(
        path=wav,
        duration=duration,
        original_path=wav,
        original_format="wav",
    )


class TestParakeetEngineAlwaysUsesRetryProtectedPath:
    def test_short_audio_goes_through_transcribe_chunked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Audio well under any VRAM-derived chunk size must still call
        _transcribe_chunked (which wraps every chunk — including a single
        whole-file "chunk" — in OOM retry-with-halving), not a bare
        model.transcribe() call.
        """
        monkeypatch.setattr(pe, "_load_nemo_model", lambda model_id: object())
        monkeypatch.setattr(pe, "detect_gpu", lambda: None)
        monkeypatch.setattr(pe, "configure_cuda_for_inference", lambda gpu: None)
        monkeypatch.setattr(pe, "log_vram_stats", lambda label: None)

        calls: list[float] = []

        def fake_transcribe_chunked(**kwargs: object) -> tuple[list[RawWord], str]:
            calls.append(kwargs["audio_duration"])  # type: ignore[arg-type]
            return (
                [RawWord(text="hi", start=0.0, end=0.4, confidence=None)],
                "distributed",
            )

        monkeypatch.setattr(pe, "_transcribe_chunked", fake_transcribe_chunked)

        engine = ParakeetEngine()
        audio = _prepared_audio(tmp_path, duration=10.0)  # far below any chunk_seconds tier

        result = engine.transcribe(
            audio=audio,
            config=TranscriptionConfig(),
            pipeline_config=PipelineConfig(),
        )

        assert calls == [10.0]
        assert len(result.words) == 1

    def test_empty_result_from_transcribe_chunked_returns_empty_transcription(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pe, "_load_nemo_model", lambda model_id: object())
        monkeypatch.setattr(pe, "detect_gpu", lambda: None)
        monkeypatch.setattr(pe, "configure_cuda_for_inference", lambda gpu: None)
        monkeypatch.setattr(pe, "log_vram_stats", lambda label: None)
        monkeypatch.setattr(pe, "_transcribe_chunked", lambda **kwargs: ([], "distributed"))

        engine = ParakeetEngine()
        audio = _prepared_audio(tmp_path, duration=5.0)

        result = engine.transcribe(
            audio=audio,
            config=TranscriptionConfig(),
            pipeline_config=PipelineConfig(),
        )

        assert result.words == ()


class TestParakeetEngineGpuInfoRefresh:
    def test_gpu_info_is_re_detected_on_every_call_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Regression test: _gpu_info used to be captured once at model-load time
        and reused for the process lifetime. Free VRAM legitimately changes
        between jobs, so it must be re-queried on every transcribe() call.
        """
        monkeypatch.setattr(pe, "_load_nemo_model", lambda model_id: object())
        monkeypatch.setattr(pe, "configure_cuda_for_inference", lambda gpu: None)
        monkeypatch.setattr(pe, "log_vram_stats", lambda label: None)
        monkeypatch.setattr(pe, "_transcribe_chunked", lambda **kwargs: ([], "distributed"))

        # A transcribe() call may query detect_gpu() more than once internally
        # (e.g. once at model-load time, once for chunk sizing) — what matters
        # is that free VRAM reflects the *current* reading, not whatever was
        # cached the first time the model was loaded. So this fake always
        # returns whatever `current_free_gb` holds right now, and the test
        # mutates it between transcribe() calls.
        current_free_gb = {"value": 6.0}

        def fake_detect_gpu() -> GpuInfo:
            return GpuInfo(
                index=0, name="RTX 3060 Ti", cuda_version="12.1",
                vram_total_gb=8.0, vram_free_gb=current_free_gb["value"],
                compute_capability=(8, 6),
            )

        monkeypatch.setattr(pe, "detect_gpu", fake_detect_gpu)

        engine = ParakeetEngine()
        audio = _prepared_audio(tmp_path, duration=5.0)

        engine.transcribe(audio=audio, config=TranscriptionConfig(), pipeline_config=PipelineConfig())
        assert ParakeetEngine._gpu_info.vram_free_gb == 6.0

        current_free_gb["value"] = 2.0  # simulate VRAM pressure from a job that ran in between
        engine.transcribe(audio=audio, config=TranscriptionConfig(), pipeline_config=PipelineConfig())
        assert ParakeetEngine._gpu_info.vram_free_gb == 2.0, (
            "second call must re-detect current VRAM, not reuse the first call's snapshot"
        )


class TestGpuComputeLock:
    def test_lock_is_shared_module_level_lock(self) -> None:
        import threading

        assert isinstance(GPU_COMPUTE_LOCK, type(threading.Lock()))

    def test_parakeet_engine_acquires_shared_lock_during_inference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ParakeetEngine must serialize its inference through GPU_COMPUTE_LOCK
        so a concurrently-running GPU-placed diarization pipeline can't overlap
        with it (see transcript_engine/diarization/pyannote_engine.py)."""
        monkeypatch.setattr(pe, "_load_nemo_model", lambda model_id: object())
        monkeypatch.setattr(pe, "detect_gpu", lambda: None)
        monkeypatch.setattr(pe, "configure_cuda_for_inference", lambda gpu: None)
        monkeypatch.setattr(pe, "log_vram_stats", lambda label: None)

        lock_held_during_call = {"value": False}

        def fake_transcribe_chunked(**kwargs: object) -> tuple[list[RawWord], str]:
            lock_held_during_call["value"] = GPU_COMPUTE_LOCK.locked()
            return ([], "distributed")

        monkeypatch.setattr(pe, "_transcribe_chunked", fake_transcribe_chunked)

        engine = ParakeetEngine()
        audio = _prepared_audio(tmp_path, duration=5.0)
        engine.transcribe(audio=audio, config=TranscriptionConfig(), pipeline_config=PipelineConfig())

        assert lock_held_during_call["value"] is True
        assert not GPU_COMPUTE_LOCK.locked()  # released after the call


# ---------------------------------------------------------------------------
# _load_nemo_model — CUDA OOM at model-load time
#
# Neither torch nor nemo_toolkit is installed in this local dev environment
# (this is deliberately checked with pytest.importorskip in other files —
# here we instead inject minimal fake modules into sys.modules so the test
# runs everywhere, including CI without a GPU/NeMo install).
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, cuda_error: Exception | None) -> None:
        self._cuda_error = cuda_error
        self.device = "unloaded"

    def eval(self) -> _FakeModel:
        return self

    def cuda(self) -> _FakeModel:
        if self._cuda_error is not None:
            raise self._cuda_error
        self.device = "cuda"
        return self

    def cpu(self) -> _FakeModel:
        self.device = "cpu"
        return self


def _install_fake_torch_and_nemo(
    monkeypatch: pytest.MonkeyPatch, cuda_error: Exception | None
) -> None:
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda: "Fake GPU",
        empty_cache=lambda: None,
    )

    class _FakeEncDecRNNTBPEModel:
        @staticmethod
        def from_pretrained(model_id: str) -> _FakeModel:
            return _FakeModel(cuda_error)

    nemo_models_mod = types.ModuleType("nemo.collections.asr.models")
    nemo_models_mod.EncDecRNNTBPEModel = _FakeEncDecRNNTBPEModel  # type: ignore[attr-defined]

    # `from nemo.collections.asr.models import X` needs every intermediate
    # package present in sys.modules, not just the leaf module.
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    monkeypatch.setitem(sys.modules, "nemo.collections", types.ModuleType("nemo.collections"))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", types.ModuleType("nemo.collections.asr"))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", nemo_models_mod)


class TestModelLoadCudaOom:
    def test_cuda_oom_during_weight_transfer_falls_back_to_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Regression test: model.cuda() used to be called with no exception
        handling at all — an OOM here (weights alone don't fit) would fail
        the model load, and therefore the whole job, with no fallback.
        """
        _install_fake_torch_and_nemo(
            monkeypatch, cuda_error=RuntimeError("CUDA out of memory. Tried to allocate 2.40 GiB")
        )

        model = pe._load_nemo_model("fake/model-id")

        assert model.device == "cpu"

    def test_non_oom_runtime_error_during_weight_transfer_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_torch_and_nemo(
            monkeypatch, cuda_error=RuntimeError("CUDA error: an illegal memory access")
        )

        with pytest.raises(RuntimeError, match="illegal memory access"):
            pe._load_nemo_model("fake/model-id")

    def test_successful_load_lands_on_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_torch_and_nemo(monkeypatch, cuda_error=None)

        model = pe._load_nemo_model("fake/model-id")

        assert model.device == "cuda"
