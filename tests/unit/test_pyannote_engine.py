"""
Unit tests for PyannoteEngine's GPU serialization and OOM recovery.

No real pyannote/torch models are loaded — the diarization pipeline is a
fake object exposing just the surface PyannoteEngine.diarize() touches
(._segmentation, ._te_on_cuda, .to(), and __call__). This mirrors the
mocking style in test_whisperx_engine.py.

The CUDA-OOM-fallback test needs `torch.device(...)`, so it is skipped when
torch is not installed (this repo's local dev environment does not have
torch installed — see test_parakeet_chunking.py for the same convention).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from transcript_engine.config.settings import DiarizationConfig
from transcript_engine.diarization.exceptions import DiarizationError, MissingHFTokenError
from transcript_engine.diarization.pyannote_engine import PyannoteEngine
from transcript_engine.gpu.hardware import GPU_COMPUTE_LOCK
from transcript_engine.models.audio import PreparedAudio


def _audio(tmp_path) -> PreparedAudio:  # type: ignore[no-untyped-def]
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"")
    return PreparedAudio(path=wav, duration=30.0, original_path=wav, original_format="wav")


class _CallableNamespace(SimpleNamespace):
    """SimpleNamespace with a __call__ method — plain SimpleNamespace can't
    have one set as an instance attribute because Python looks up dunders on
    the type, not the instance."""

    def __init__(self, call_fn, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._call_fn = call_fn

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._call_fn(*args, **kwargs)


def _engine(pipeline) -> PyannoteEngine:  # type: ignore[no-untyped-def]
    registry = MagicMock()
    registry.get_diarization_pipeline.return_value = pipeline
    return PyannoteEngine(registry)


def _empty_diarization() -> SimpleNamespace:
    return SimpleNamespace(itertracks=lambda yield_label: iter(()))


class TestGpuLockSerialization:
    def test_diarize_holds_shared_lock_when_pipeline_on_cuda(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        lock_held = {"value": False}

        def call_fn(path, **kwargs):  # type: ignore[no-untyped-def]
            lock_held["value"] = GPU_COMPUTE_LOCK.locked()
            return _empty_diarization()

        pipeline = _CallableNamespace(
            call_fn, _segmentation=SimpleNamespace(duration=10.0, step=1.0),
            _te_on_cuda=True, to=MagicMock(),
        )
        engine = _engine(pipeline)

        engine.diarize(_audio(tmp_path), DiarizationConfig())

        assert lock_held["value"] is True
        assert not GPU_COMPUTE_LOCK.locked()

    def test_diarize_does_not_touch_lock_when_pipeline_on_cpu(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        lock_held = {"value": None}

        def call_fn(path, **kwargs):  # type: ignore[no-untyped-def]
            lock_held["value"] = GPU_COMPUTE_LOCK.locked()
            return _empty_diarization()

        pipeline = _CallableNamespace(
            call_fn, _segmentation=SimpleNamespace(duration=10.0, step=1.0),
            _te_on_cuda=False, to=MagicMock(),
        )
        engine = _engine(pipeline)

        engine.diarize(_audio(tmp_path), DiarizationConfig())

        # Not held by this engine (may coincidentally be free either way, but
        # must never be True — CPU diarization must not serialize on the GPU lock).
        assert lock_held["value"] is False


class TestOomCpuFallback:
    def test_cuda_oom_falls_back_to_cpu_and_succeeds(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("torch")

        calls = {"n": 0}

        def call_fn(path, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
            return _empty_diarization()

        pipeline = _CallableNamespace(
            call_fn, _segmentation=SimpleNamespace(duration=10.0, step=1.0),
            _te_on_cuda=True, to=MagicMock(),
        )
        engine = _engine(pipeline)

        result = engine.diarize(_audio(tmp_path), DiarizationConfig())

        assert calls["n"] == 2  # first call OOM'd, retry succeeded
        assert pipeline._te_on_cuda is False  # downgraded after fallback
        pipeline.to.assert_called_once()
        assert result.num_speakers == 0

    def test_non_oom_runtime_error_is_not_retried(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        calls = {"n": 0}

        def call_fn(path, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            raise RuntimeError("some unrelated CUDA error")

        pipeline = _CallableNamespace(
            call_fn, _segmentation=SimpleNamespace(duration=10.0, step=1.0),
            _te_on_cuda=True, to=MagicMock(),
        )
        engine = _engine(pipeline)

        with pytest.raises(DiarizationError):
            engine.diarize(_audio(tmp_path), DiarizationConfig())

        assert calls["n"] == 1  # no retry for a non-OOM RuntimeError
        pipeline.to.assert_not_called()


class TestLoadFailureHandling:
    def test_lock_is_released_when_pipeline_load_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """
        Model loading is wrapped in GPU_COMPUTE_LOCK (it performs the .to(cuda)
        weight transfer). If loading raises, the lock must still be released —
        otherwise every subsequent job would deadlock waiting on it forever.
        """
        registry = MagicMock()
        registry.get_diarization_pipeline.side_effect = RuntimeError("model download failed")
        engine = PyannoteEngine(registry)

        with pytest.raises(DiarizationError):
            engine.diarize(_audio(tmp_path), DiarizationConfig())

        assert not GPU_COMPUTE_LOCK.locked()

    def test_lock_is_released_when_inference_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        def call_fn(path, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        pipeline = _CallableNamespace(
            call_fn, _segmentation=SimpleNamespace(duration=10.0, step=1.0),
            _te_on_cuda=True, to=MagicMock(),
        )
        engine = _engine(pipeline)

        with pytest.raises(DiarizationError):
            engine.diarize(_audio(tmp_path), DiarizationConfig())

        assert not GPU_COMPUTE_LOCK.locked()

    def test_missing_hf_token_surfaces_actionable_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        registry = MagicMock()
        registry.get_diarization_pipeline.side_effect = RuntimeError("401 invalid token")
        engine = PyannoteEngine(registry)

        with pytest.raises(MissingHFTokenError):
            engine.diarize(_audio(tmp_path), DiarizationConfig())


def test_gpu_compute_lock_is_a_real_lock() -> None:
    assert isinstance(GPU_COMPUTE_LOCK, type(threading.Lock()))
