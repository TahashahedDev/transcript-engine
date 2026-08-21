"""
Real speaker-embedding extraction from audio.

Model choice (VERIFIED on this machine, not assumed):
``pyannote/wespeaker-voxceleb-resnet34-LM`` — the exact embedding model
pyannote's own ``speaker-diarization-3.1`` pipeline already downloads and
uses internally for its own clustering step. Confirmed present at
``~/.cache/torch/pyannote/models--pyannote--wespeaker-voxceleb-resnet34-LM``
before this module ever ran, which means diarization already pulled it in —
loading it again here adds zero new downloads and zero new dependencies,
satisfying the "avoid duplicating models pyannote already loaded" and "avoid
introducing a large new dependency" constraints directly rather than by
assumption. SpeechBrain's ECAPA model was considered and rejected for the
same reason: it is not already part of this stack, and the wespeaker model
that already is here is validated for exactly this task (VoxCeleb speaker
verification), so there is no case for adding a second framework.

Confirmed working end-to-end on this machine (this exact process, not
memory): model loads from the local HF cache in ~1s using the same HF token
and the same PyTorch-2.6 ``weights_only`` compatibility patch the diarization
pipeline already applies (``ModelRegistry._register_torch_safe_globals``),
and produces a 256-dimensional embedding per input clip in well under 0.2s
on CPU for a 3-4 second window. That confirms the *code path* end-to-end;
it does not confirm real-world identification accuracy, because the input
audio in that test was synthetic noise, not speech — no real multi-speaker
recording exists in this repository to test against (see
IDENTITY_ARCHITECTURE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from transcript_engine.logging import get_logger
from transcript_engine.model_registry.registry import ModelRegistry

logger = get_logger(__name__)

EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Free VRAM required before placing the embedding model on GPU, mirroring
# ModelRegistry._load_diarization's exact policy for pyannote's own pipeline
# (transcript_engine/model_registry/registry.py) so this model is judged by
# the same standard rather than a separately invented one.
#
# VERIFIED: this model's on-disk weights are 25 MB (measured on this
# machine — see ~/.cache/torch/pyannote/models--pyannote--wespeaker...).
# That is roughly 30x smaller than pyannote's own diarization pipeline,
# whose 2.0 GB threshold already covers weights plus generous activation
# headroom. 1.0 GB here is a conservative pad over the real 25 MB, not a
# measurement of actual peak VRAM under load — that number can only come
# from running on the real GPU (see IDENTITY_ARCHITECTURE.md).
_EMBEDDING_VRAM_NEEDED_GB = 1.0

# Windowing policy for turning a diarized speaker segment into one or more
# embeddable clips (mission brief section 4/5). Diarization segments are
# already speech-only and already speaker-homogeneous — pyannote's own
# clustering did that work — so this does not need voice-activity detection,
# only a duration policy:
#
#   < WINDOW_MIN_S              : too short to trust (a bare "yeah"/"okay");
#                                  do not embed it at all.
#   WINDOW_MIN_S..WINDOW_MAX_S  : embed as a single window.
#   > WINDOW_MAX_S               : split into WINDOW_TARGET_S-sized windows so
#                                  one very long turn does not dominate a
#                                  profile with one embedding while everyone
#                                  else contributes several.
#
# These figures are PROPOSED, not tuned: no real multi-speaker recording
# exists locally to measure the actual effect of window length on embedding
# quality for this model, so they follow the general speaker-verification
# literature's rule of thumb (more than ~2s needed for a stable embedding,
# diminishing returns well before 10s) rather than measurement.
WINDOW_MIN_S = 2.0
WINDOW_TARGET_S = 5.0
WINDOW_MAX_S = 10.0

_SAMPLE_RATE = 16_000  # matches PreparedAudio — see models/audio.py


@dataclass(frozen=True)
class EmbeddingWindow:
    speaker_id: str
    start: float
    end: float


def compute_embedding_windows(
    segments: list[tuple[str, float, float]],
) -> list[EmbeddingWindow]:
    """
    Turn (speaker_id, start, end) diarization segments into embeddable
    windows per the policy documented above. Takes plain tuples rather than
    SpeakerSegment directly so this stays testable without constructing full
    pipeline model objects.
    """
    windows: list[EmbeddingWindow] = []
    for speaker_id, start, end in segments:
        duration = end - start
        if duration < WINDOW_MIN_S:
            continue
        if duration <= WINDOW_MAX_S:
            windows.append(EmbeddingWindow(speaker_id, start, end))
            continue

        # Split a long turn into WINDOW_TARGET_S chunks. The final chunk is
        # whatever remains; if that remainder still clears WINDOW_MIN_S it
        # becomes its own window, otherwise it is folded into the previous
        # chunk (extending it) rather than discarded — a segment ending in a
        # 1.3s remainder should not lose that speech entirely.
        cursor = start
        while end - cursor > WINDOW_MAX_S:
            windows.append(EmbeddingWindow(speaker_id, cursor, cursor + WINDOW_TARGET_S))
            cursor += WINDOW_TARGET_S
        remainder = end - cursor
        if remainder >= WINDOW_MIN_S:
            windows.append(EmbeddingWindow(speaker_id, cursor, end))
        elif windows and windows[-1].speaker_id == speaker_id:
            last = windows[-1]
            windows[-1] = EmbeddingWindow(speaker_id, last.start, end)

    return windows


def _select_device() -> tuple[torch.device, bool]:
    """
    Same free-VRAM-headroom policy as ModelRegistry._load_diarization: only
    claim GPU when there is real free VRAM for it (using *free*, not total —
    Parakeet's weights are typically already resident by the time this
    loads), otherwise CPU is the correct, always-safe fallback. Returns
    (device, on_cuda) — on_cuda tells extract() whether to serialize
    inference through the shared GPU_COMPUTE_LOCK.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), False
    try:
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1e9
    except Exception:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_gb = total_gb
    if free_gb >= _EMBEDDING_VRAM_NEEDED_GB:
        return torch.device("cuda"), True
    logger.warning(
        "Embedding model: only %.2f GB free VRAM (need %.2f GB) — using CPU",
        free_gb, _EMBEDDING_VRAM_NEEDED_GB,
    )
    return torch.device("cpu"), False


class SpeakerEmbeddingExtractor:
    """
    Loads the shared wespeaker embedding model once and extracts embeddings
    for a batch of (audio_path, window) pairs.

    Deliberately mirrors PyannoteEngine's own model-loading pattern (same HF
    token source, same torch.load compatibility patch, same free-VRAM-
    headroom device selection, same shared GPU_COMPUTE_LOCK serialization)
    rather than inventing a second way to load and run a pyannote-family
    model.
    """

    def __init__(self, hf_token: str | None, device: torch.device | None = None) -> None:
        ModelRegistry._register_torch_safe_globals()  # noqa: SLF001
        from pyannote.audio.pipelines.speaker_verification import (  # noqa: PLC0415
            PretrainedSpeakerEmbedding,
        )

        if device is not None:
            self._device = device
            self._on_cuda = device.type == "cuda"
        else:
            self._device, self._on_cuda = _select_device()

        self._model = PretrainedSpeakerEmbedding(
            EMBEDDING_MODEL_ID, device=self._device, use_auth_token=hf_token
        )

    def extract(self, audio_path: Path, windows: list[EmbeddingWindow]) -> dict[int, np.ndarray]:
        """
        Returns embeddings keyed by index into `windows` (not by speaker_id —
        one speaker may have many windows, and the caller decides how to
        aggregate them, e.g. transcript_engine.identity.profile.SpeakerProfile).
        Windows that fail to load (out-of-range timestamps, corrupt audio)
        are silently skipped rather than failing the whole batch — one bad
        window must not lose every other speaker's embeddings for the job.
        """
        import torchaudio  # noqa: PLC0415

        if not windows:
            return {}

        clips: list[torch.Tensor] = []
        valid_indices: list[int] = []
        for i, window in enumerate(windows):
            frame_offset = int(window.start * _SAMPLE_RATE)
            num_frames = int((window.end - window.start) * _SAMPLE_RATE)
            try:
                waveform, sr = torchaudio.load(
                    str(audio_path), frame_offset=frame_offset, num_frames=num_frames
                )
            except Exception:
                logger.warning(
                    "Skipping embedding window %.2f-%.2fs (speaker %s): audio load failed",
                    window.start, window.end, window.speaker_id,
                )
                continue
            if sr != _SAMPLE_RATE or waveform.shape[-1] == 0:
                continue
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            clips.append(waveform)
            valid_indices.append(i)

        if not clips:
            return {}

        # Batch by padding to the longest clip length, since the model
        # accepts (batch, channels, samples) with a shared sample count.
        max_len = max(c.shape[-1] for c in clips)
        batch = torch.zeros(len(clips), 1, max_len)
        for j, clip in enumerate(clips):
            batch[j, :, : clip.shape[-1]] = clip

        embeddings = self._run_model(batch)
        return {valid_indices[j]: embeddings[j] for j in range(len(valid_indices))}

    def _run_model(self, batch: torch.Tensor) -> np.ndarray:
        """
        Same GPU_COMPUTE_LOCK serialization PyannoteEngine.diarize() and
        ParakeetEngine already use: on a shared CUDA card, this call and
        diarization/ASR inference must never submit to the GPU at the same
        instant. On CPU this is a no-op (nullcontext), matching the existing
        pattern exactly.

        Also mirrors PyannoteEngine's one-shot CUDA-OOM recovery: if the
        batch (built from however many embedding windows this job produced)
        doesn't fit, fall back to CPU once rather than failing the whole
        speaker-identification step over a single oversized batch.
        """
        import contextlib  # noqa: PLC0415

        from transcript_engine.gpu.hardware import GPU_COMPUTE_LOCK  # noqa: PLC0415

        gpu_guard = GPU_COMPUTE_LOCK if self._on_cuda else contextlib.nullcontext()
        try:
            with gpu_guard:
                result: np.ndarray = self._model(batch)
            return result
        except RuntimeError as exc:
            if not (self._on_cuda and "out of memory" in str(exc).lower()):
                raise
            logger.warning("Embedding extraction CUDA OOM — retrying once on CPU: %s", exc)
            torch.cuda.empty_cache()
            self._model.to(torch.device("cpu"))  # moves the underlying model's weights
            self._device = torch.device("cpu")
            self._on_cuda = False
            result = self._model(batch)  # __call__ moves the input tensor itself
            return result
