from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from transcript_engine.config.settings import DiarizationConfig
from transcript_engine.diarization.compat import apply_segmentation_step
from transcript_engine.diarization.exceptions import (
    DiarizationError,
    MissingHFTokenError,
)
from transcript_engine.gpu.hardware import GPU_COMPUTE_LOCK
from transcript_engine.logging import get_logger
from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.pipeline import DiarizationResult, SpeakerSegment
from transcript_engine.models.types import ProgressCallback

if TYPE_CHECKING:
    from transcript_engine.model_registry.registry import ModelRegistry

logger = get_logger(__name__)


class PyannoteEngine:
    """
    Speaker diarization engine backed by pyannote.audio.
    Wired manually — does NOT use WhisperX's built-in diarization,
    which pins incompatible pyannote versions.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def diarize(
        self,
        audio: PreparedAudio,
        config: DiarizationConfig,
        on_progress: ProgressCallback | None = None,
    ) -> DiarizationResult:
        if on_progress:
            on_progress(0.62, "Loading diarization model")

        logger.info(f"Loading diarization model: {config.model_id}")

        # ModelRegistry caches the loaded pipeline, so this only does real work
        # (including the one-time .to(cuda) weight transfer) on the first call
        # for a given config. That first-load transfer is exactly the kind of
        # GPU allocation Parakeet could be doing at the same instant in the
        # other orchestrator thread, so it's covered by the same shared lock
        # as inference — cheap once cached, and it closes the cold-start race
        # where neither engine has loaded anything onto the GPU yet.
        try:
            with GPU_COMPUTE_LOCK:
                pipeline = self._registry.get_diarization_pipeline(config)
        except Exception as exc:
            # Matching on the word "token" anywhere in the message is too broad:
            # a `TypeError: from_pretrained() got an unexpected keyword argument
            # 'token'` — a pyannote API mismatch — was being reported to the user
            # as "add your HuggingFace token", sending them to fix a setting that
            # was already correct. Require an actual auth signal, and never
            # reclassify a TypeError, which is by definition a calling bug here.
            message = str(exc).lower()
            is_auth_failure = (
                not isinstance(exc, TypeError)
                and (
                    "401" in message
                    or "unauthorized" in message
                    or "authentication" in message
                    or "gated" in message
                    or "access to model" in message
                    or "not authorized" in message
                )
            )
            if is_auth_failure:
                raise MissingHFTokenError(config.model_id) from exc
            raise DiarizationError(f"Failed to load diarization model: {exc}") from exc

        step_seconds = apply_segmentation_step(pipeline, config.segmentation_step)

        logger.info(
            "Diarizing %s (%.0fs, seg_step=%s, actual_step=%s)",
            audio.original_path.name,
            audio.duration,
            config.segmentation_step,
            f"{step_seconds:.1f}s" if step_seconds is not None else "library default",
        )
        if on_progress:
            on_progress(0.65, "Running speaker diarization")

        pipeline_kwargs: dict[str, int] = {}
        if config.min_speakers is not None:
            pipeline_kwargs["min_speakers"] = config.min_speakers
        if config.max_speakers is not None:
            pipeline_kwargs["max_speakers"] = config.max_speakers

        # When this pipeline runs on CUDA, it shares the GPU with Parakeet
        # transcription, which the orchestrator may run concurrently in another
        # thread. Serialize actual GPU compute through the shared lock so the
        # two engines never allocate at the same instant. On CPU there is no
        # GPU contention, so this is a no-op (nullcontext) and diarization runs
        # truly in parallel with transcription, as intended.
        on_cuda = getattr(pipeline, "_te_on_cuda", False)
        gpu_guard = GPU_COMPUTE_LOCK if on_cuda else contextlib.nullcontext()

        try:
            with gpu_guard:
                diarization = pipeline(str(audio.path), **pipeline_kwargs)
        except RuntimeError as exc:
            if not (on_cuda and "out of memory" in str(exc).lower()):
                raise DiarizationError(f"Diarization failed: {exc}") from exc

            # One-shot recovery: fall back to CPU rather than failing the whole
            # job. Slower (see registry._load_diarization: ~2-5 min vs ~15-30s
            # on CUDA for 90-min audio) but keeps a GPU OOM here from taking
            # down an otherwise-successful transcription.
            logger.warning("[Diarization] CUDA OOM — retrying once on CPU (%s)", exc)
            import torch  # noqa: PLC0415

            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            pipeline.to(torch.device("cpu"))
            pipeline._te_on_cuda = False
            try:
                diarization = pipeline(str(audio.path), **pipeline_kwargs)
            except Exception as retry_exc:
                raise DiarizationError(
                    f"Diarization failed on CPU fallback after CUDA OOM: {retry_exc}"
                ) from retry_exc
        except Exception as exc:
            raise DiarizationError(f"Diarization failed: {exc}") from exc

        segments = self._extract_segments(diarization)
        num_speakers = len({s.speaker_id for s in segments})

        logger.info(
            f"Diarization complete: {num_speakers} speaker(s), {len(segments)} segments"
        )
        if on_progress:
            on_progress(0.90, "Diarization complete")

        return DiarizationResult(
            segments=tuple(segments),
            num_speakers=num_speakers,
        )

    def _extract_segments(self, diarization) -> list[SpeakerSegment]:  # type: ignore[no-untyped-def]
        segments: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker_id=speaker,
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                )
            )
        return sorted(segments, key=lambda s: s.start)
