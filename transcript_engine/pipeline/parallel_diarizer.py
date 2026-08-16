"""
Parallel diarization runner.

Splits the prepared audio into N chunks and diarizes each in a separate
OS process so all available CPU cores are used concurrently.  The main
process meanwhile runs the MLX/CT2 transcription on the GPU/Metal.

Design notes
------------
- Uses ``ProcessPoolExecutor`` with the 'spawn' start method so each worker
  gets a clean Python interpreter.  This avoids fork-after-threading issues
  with PyTorch and makes it safe on macOS.
- Each worker loads pyannote independently, which takes ~20 s cold-start but
  costs no extra RAM vs the parent because the models are on disk.
- Diarization runs on CPU in workers to avoid Metal contention with the
  concurrent MLX transcription in the main process.
- Worker results carry global timestamps (chunk.global_start already added),
  ready for speaker reconciliation.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any

from transcript_engine.audio.chunker import AudioChunk
from transcript_engine.config.settings import DiarizationConfig
from transcript_engine.models.pipeline import DiarizationResult, SpeakerSegment
from transcript_engine.pipeline.speaker_reconciler import ChunkDiarization, reconcile

logger = logging.getLogger(__name__)


# ── Public entry point ───────────────────────────────────────────────────────


def diarize_parallel(
    chunks: list[AudioChunk],
    config: DiarizationConfig,
    hf_token: str | None,
    on_progress: Any | None = None,
) -> DiarizationResult:
    """
    Diarize *chunks* in parallel worker processes and merge the results.

    If *chunks* has exactly one entry, runs inline (no subprocess overhead).
    """
    if len(chunks) == 1:
        return _diarize_inline(chunks[0], config, hf_token, on_progress)

    if on_progress:
        on_progress(0.62, "Loading diarization model")

    n_workers = len(chunks)
    logger.info(
        f"Starting parallel diarization: {n_workers} workers, "
        f"step={config.segmentation_step}, "
        f"chunks={[f'{c.duration:.0f}s' for c in chunks]}"
    )

    # Use 'spawn' (the macOS default) to avoid fork-after-threading issues.
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
    ) as pool:
        futures = {
            pool.submit(
                _worker_diarize,
                str(chunk.path),
                chunk.chunk_idx,
                chunk.global_start,
                chunk.global_end,
                chunk.overlap_seconds,
                config.model_id,
                config.segmentation_step,
                config.min_speakers,
                config.max_speakers,
                hf_token,
            ): chunk
            for chunk in chunks
        }

        chunk_results: list[ChunkDiarization] = []
        for fut in concurrent.futures.as_completed(futures):
            chunk = futures[fut]
            try:
                result = fut.result()
                chunk_results.append(result)
                logger.info(
                    f"Chunk {chunk.chunk_idx} done: "
                    f"{len(result.segments)} segments, "
                    f"{len({s.speaker_id for s in result.segments})} speakers"
                )
            except Exception as exc:
                logger.error(f"Chunk {chunk.chunk_idx} diarization failed: {exc}")
                # Return a single-speaker fallback for this chunk.
                chunk_results.append(
                    ChunkDiarization(
                        chunk_idx=chunk.chunk_idx,
                        global_start=chunk.global_start,
                        global_end=chunk.global_end,
                        overlap_seconds=chunk.overlap_seconds,
                        segments=[
                            SpeakerSegment(
                                speaker_id="SPEAKER_00",
                                start=round(chunk.global_start, 3),
                                end=round(chunk.global_end, 3),
                            )
                        ],
                    )
                )

    if on_progress:
        on_progress(0.90, "Diarization complete")

    return reconcile(sorted(chunk_results, key=lambda c: c.chunk_idx))


# ── Worker function (runs in subprocess) ────────────────────���────────────────


def _worker_diarize(
    audio_path: str,
    chunk_idx: int,
    global_start: float,
    global_end: float,
    overlap_seconds: float,
    model_id: str,
    segmentation_step: float,
    min_speakers: int | None,
    max_speakers: int | None,
    hf_token: str | None,
) -> ChunkDiarization:
    """
    Worker function — runs in a separate process.

    Does NOT import transcript_engine.model_registry to keep the worker
    lean.  Loads pyannote directly with minimal setup.
    """
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore")

    # Spawned processes don't inherit the parent's QOS class — set it explicitly.
    from transcript_engine.scheduling import enter_background_mode  # noqa: PLC0415

    enter_background_mode()

    import torch  # noqa: PLC0415
    from pyannote.audio import Pipeline  # noqa: PLC0415

    # Inject the HuggingFace token if provided.
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    # PyTorch 2.6 changed torch.load to weights_only=True by default, breaking pyannote.
    # This patch must be applied in every subprocess — the main-process patch in
    # ModelRegistry.__init__ does NOT carry over to spawned workers.
    _patch_torch_load()

    # Monkey-patch hf_hub_download for pyannote/huggingface_hub compatibility.
    _patch_hf_hub()

    pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
    seg_inf = pipeline._segmentation
    seg_inf.step = segmentation_step * float(seg_inf.duration)

    # Force CPU only — main process uses Metal for transcription.
    pipeline.to(torch.device("cpu"))

    kwargs: dict[str, int] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    with torch.no_grad():
        diarization = pipeline(audio_path, **kwargs)

    # Convert to our model, adding global_start offset to timestamps.
    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                speaker_id=speaker,
                start=round(turn.start + global_start, 3),
                end=round(turn.end + global_start, 3),
            )
        )

    return ChunkDiarization(
        chunk_idx=chunk_idx,
        global_start=global_start,
        global_end=global_end,
        overlap_seconds=overlap_seconds,
        segments=sorted(segments, key=lambda s: s.start),
    )


def _diarize_inline(
    chunk: AudioChunk,
    config: DiarizationConfig,
    hf_token: str | None,
    on_progress: Any | None,
) -> DiarizationResult:
    """Single-chunk path — avoids subprocess overhead for short audio."""
    from transcript_engine.diarization.pyannote_engine import (
        PyannoteEngine,
    )  # noqa: PLC0415

    # We don't have a registry here; use a minimal shim.
    class _MinimalAudio:
        def __init__(self, path: Path, duration: float) -> None:
            self.path = path
            self.original_path = path
            self.duration = duration

    from transcript_engine.config.loader import load_settings  # noqa: PLC0415
    from transcript_engine.model_registry.registry import ModelRegistry  # noqa: PLC0415

    settings = load_settings()
    registry = ModelRegistry(settings)
    engine = PyannoteEngine(registry)
    audio = _MinimalAudio(chunk.path, chunk.global_end - chunk.global_start)
    result = engine.diarize(audio, config, on_progress)  # type: ignore[arg-type]
    return result


# ── PyTorch 2.6 compat patch (same as in ModelRegistry) ─────────────────────


def _patch_torch_load() -> None:
    """
    Restore pre-2.6 torch.load behaviour in subprocess workers.
    PyTorch 2.6 changed weights_only default to True, breaking pyannote checkpoints.
    This runs in EACH spawned worker because the main-process patch doesn't carry over.
    """
    import functools  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415

        if getattr(torch.load, "_te_patched", False):
            return
        _orig = torch.load

        @functools.wraps(_orig)
        def _patched(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("weights_only") is not True:
                kwargs["weights_only"] = False
            return _orig(*args, **kwargs)

        _patched._te_patched = True  # type: ignore[attr-defined]
        torch.load = _patched  # noqa: B010
    except (ImportError, AttributeError):
        pass


# ── HuggingFace compat patch (same as in ModelRegistry) ──────────────────────


def _patch_hf_hub() -> None:
    """Bridge pyannote 3.x with huggingface_hub >= 1.0 in worker processes."""
    import importlib  # noqa: PLC0415

    import huggingface_hub  # noqa: PLC0415

    if getattr(huggingface_hub.hf_hub_download, "_te_patched", False):
        return
    _orig = huggingface_hub.hf_hub_download

    def _patched(*args: Any, **kwargs: Any) -> Any:
        if "use_auth_token" in kwargs:
            uak = kwargs.pop("use_auth_token")
            if uak and "token" not in kwargs:
                kwargs["token"] = uak
        return _orig(*args, **kwargs)

    _patched._te_patched = True  # type: ignore[attr-defined]
    huggingface_hub.hf_hub_download = _patched  # noqa: B010

    for _mod_name in [
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
        "pyannote.audio.pipelines.speaker_verification",
    ]:
        try:
            _mod = importlib.import_module(_mod_name)
            if hasattr(_mod, "hf_hub_download"):
                setattr(_mod, "hf_hub_download", _patched)  # noqa: B010
        except (ImportError, AttributeError):
            pass
