"""
Compatibility shims for pyannote.audio.

pyannote has moved the two things this project touches most — how the
authentication token is passed to ``Pipeline.from_pretrained``, and where the
segmentation step lives — across its 3.x and 4.x releases. Both are handled here
so the call sites stay readable and there is one place to look when a version
bump breaks something.

Kept free of heavy imports so the subprocess workers in
``pipeline.parallel_diarizer`` can use it without pulling in the settings and
model-registry stack.
"""

from __future__ import annotations

import inspect
from typing import Any

from transcript_engine.logging import get_logger

logger = get_logger(__name__)


def load_pretrained_pipeline(
    pipeline_cls: Any,
    model_id: str,
    token: str | None,
    cache_dir: str | None = None,
) -> Any:
    """
    Call ``Pipeline.from_pretrained`` with whichever token keyword this
    installed pyannote actually accepts.

    3.x names it ``use_auth_token``; 4.x renamed it to ``token``. Passing the
    wrong one raises ``TypeError: unexpected keyword argument``, which surfaces
    only when a job runs — and, because the message contains the word "token",
    is easily mistaken for a missing-credentials error. Inspect the signature
    instead of guessing from the version string.
    """
    kwargs: dict[str, Any] = {}
    params = inspect.signature(pipeline_cls.from_pretrained).parameters

    if token:
        if "token" in params:
            kwargs["token"] = token
        elif "use_auth_token" in params:
            kwargs["use_auth_token"] = token
        else:
            # Neither name is present. HF_TOKEN is set in the environment by the
            # caller, and huggingface_hub reads it from there, so this is still
            # very likely to succeed.
            logger.warning(
                "pyannote's from_pretrained accepts neither 'token' nor "
                "'use_auth_token' (parameters: %s) — relying on the HF_TOKEN "
                "environment variable instead.",
                ", ".join(params),
            )

    if cache_dir and "cache_dir" in params:
        kwargs["cache_dir"] = cache_dir

    return pipeline_cls.from_pretrained(model_id, **kwargs)


def apply_segmentation_step(pipeline: Any, segmentation_step: float) -> float | None:
    """
    Set how far the segmentation window advances between inferences.

    pyannote exposes ``segmentation_step`` as a plain attribute that is read once
    in ``__init__``; assigning to it after the pipeline is built has no effect.
    The value that actually matters lives on the internal ``Inference`` object as
    an absolute offset in seconds, so we set that instead.

    That means reaching into a private attribute. It is guarded rather than
    assumed: a pyannote release that renames or restructures the internals would
    otherwise raise ``AttributeError`` in the middle of a job. Diarization is
    perfectly correct at pyannote's default step — it is only slower — so a
    missing attribute degrades to the default with a warning instead of failing
    the run.

    Returns the step actually applied in seconds, or None when it could not be
    set.
    """
    seg_inf = getattr(pipeline, "_segmentation", None)
    duration = getattr(seg_inf, "duration", None)
    if seg_inf is None or duration is None:
        logger.warning(
            "Could not set the diarization segmentation step: this pyannote "
            "build does not expose _segmentation.duration. Falling back to the "
            "library default (correct, but slower than step=%s).",
            segmentation_step,
        )
        return None

    step_seconds = segmentation_step * float(duration)
    seg_inf.step = step_seconds
    return step_seconds
