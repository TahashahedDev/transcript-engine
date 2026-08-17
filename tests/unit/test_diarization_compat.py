"""
Tests for the pyannote compatibility shims.

These exist because both shims paper over API differences between pyannote
releases, and both failed silently or confusingly in ways that only showed up
when a real job ran.
"""

from __future__ import annotations

from types import SimpleNamespace

from transcript_engine.diarization.compat import (
    apply_segmentation_step,
    load_pretrained_pipeline,
)


class _Pipeline3x:
    """pyannote 3.x: the token keyword is `use_auth_token`."""

    last_call: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, checkpoint_path, hparams_file=None, use_auth_token=None, cache_dir=None):  # type: ignore[no-untyped-def]
        cls.last_call = {
            "checkpoint_path": checkpoint_path,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
        }
        return SimpleNamespace(name="3x")


class _Pipeline4x:
    """pyannote 4.x: renamed to `token`."""

    last_call: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, checkpoint_path, token=None, cache_dir=None):  # type: ignore[no-untyped-def]
        cls.last_call = {
            "checkpoint_path": checkpoint_path,
            "token": token,
            "cache_dir": cache_dir,
        }
        return SimpleNamespace(name="4x")


class _PipelineNoToken:
    """A build exposing neither keyword — must not raise."""

    @staticmethod
    def from_pretrained(checkpoint_path):  # type: ignore[no-untyped-def]
        return SimpleNamespace(name="no-token")


class TestLoadPretrainedPipeline:
    def test_uses_use_auth_token_on_pyannote_3x(self) -> None:
        result = load_pretrained_pipeline(_Pipeline3x, "pyannote/x", "hf_abc", cache_dir="/tmp/c")

        assert result.name == "3x"
        assert _Pipeline3x.last_call["use_auth_token"] == "hf_abc"
        assert _Pipeline3x.last_call["cache_dir"] == "/tmp/c"

    def test_uses_token_on_pyannote_4x(self) -> None:
        result = load_pretrained_pipeline(_Pipeline4x, "pyannote/x", "hf_abc")

        assert result.name == "4x"
        assert _Pipeline4x.last_call["token"] == "hf_abc"

    def test_omits_token_kwarg_when_no_token_given(self) -> None:
        _Pipeline3x.last_call = {}
        load_pretrained_pipeline(_Pipeline3x, "pyannote/x", None)

        assert _Pipeline3x.last_call["use_auth_token"] is None

    def test_unknown_signature_does_not_raise(self) -> None:
        # Falls back to the HF_TOKEN environment variable rather than passing a
        # keyword that would raise TypeError mid-job.
        result = load_pretrained_pipeline(_PipelineNoToken, "pyannote/x", "hf_abc")

        assert result.name == "no-token"

    def test_cache_dir_dropped_when_unsupported(self) -> None:
        result = load_pretrained_pipeline(_PipelineNoToken, "pyannote/x", None, cache_dir="/tmp/c")

        assert result.name == "no-token"


class TestApplySegmentationStep:
    def test_sets_step_as_absolute_seconds(self) -> None:
        pipeline = SimpleNamespace(_segmentation=SimpleNamespace(duration=10.0, step=None))

        applied = apply_segmentation_step(pipeline, 0.5)

        assert applied == 5.0
        assert pipeline._segmentation.step == 5.0

    def test_missing_internals_degrade_instead_of_raising(self) -> None:
        # A pyannote release that restructures internals must slow the pipeline
        # down, not fail the job.
        assert apply_segmentation_step(SimpleNamespace(), 0.5) is None

    def test_segmentation_without_duration_degrades(self) -> None:
        pipeline = SimpleNamespace(_segmentation=SimpleNamespace())

        assert apply_segmentation_step(pipeline, 0.5) is None
