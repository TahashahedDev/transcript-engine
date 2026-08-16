"""
Parakeet TDT transcription engine.

Uses NVIDIA NeMo's parakeet-tdt-0.6b-v2 (or any EncDecRNNTBPEModel).
Drop-in replacement for WhisperXEngine — satisfies the TranscriptionEngine protocol.

Key characteristics:
  - Word-level timestamps produced by the TDT decoder (no wav2vec2 alignment)
  - Chunked inference: long audio is split into hardware-sized pieces to cap peak VRAM
  - Overlap + boundary deduplication: no word is dropped or duplicated at chunk seams
  - torch.inference_mode() wraps every NeMo call (stronger than no_grad)
  - TF32 and cuDNN benchmark mode enabled on Ampere+ GPUs (free throughput gain)
  - VRAM telemetry logged after every chunk
  - Automatic retry with halved chunk size on CUDA OOM

Enable: TE_ASR_BACKEND=parakeet
Override model: TE_PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2
"""

from __future__ import annotations

import contextlib
import gc
import os
import tempfile
import time
import wave
from typing import Any

from transcript_engine.config.settings import PipelineConfig, TranscriptionConfig
from transcript_engine.gpu.hardware import (
    DEFAULT_OVERLAP_SECONDS,
    GPU_COMPUTE_LOCK,
    GpuInfo,
    check_gpu_compatibility,
    configure_cuda_for_inference,
    detect_gpu,
    log_vram_stats,
    optimal_chunk_seconds,
)
from transcript_engine.logging import get_logger
from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.pipeline import RawTranscription, RawWord
from transcript_engine.models.types import ProgressCallback

logger = get_logger(__name__)

_DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"

# Seconds per encoder output frame.
# 16 kHz → 10 ms mel hop (160 samples) → 8× Conformer subsampling → 80 ms/frame.
# Read from model config at load time; fall back to this constant.
_FALLBACK_FRAME_DURATION_S = 0.08

# Minimum chunk size regardless of VRAM (very short chunks hurt quality).
_MIN_CHUNK_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_nemo_model(model_id: str) -> Any:
    """Load, move to CUDA, and eval a NeMo RNNT/TDT model."""
    import torch  # noqa: PLC0415

    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModel  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "nemo_toolkit[asr] is required for the Parakeet engine.\n"
            "Install: pip install 'nemo_toolkit[asr]'\n"
            "Or set TE_ASR_BACKEND=whisper to use the default engine."
        ) from exc

    logger.info("[Parakeet] Loading model: %s", model_id)
    t0 = time.monotonic()
    model: Any = EncDecRNNTBPEModel.from_pretrained(model_id)
    model.eval()

    if torch.cuda.is_available():
        # Fail fast with an actionable message if this PyTorch build has no
        # kernels for the card, rather than dying at the first launch with
        # "no kernel image is available for execution on the device".
        gpu_for_check = detect_gpu()
        if gpu_for_check is not None:
            incompatible = check_gpu_compatibility(gpu_for_check)
            if incompatible:
                raise RuntimeError(incompatible)
        try:
            model = model.cuda()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            # The model's weight footprint (~2.5 GB for parakeet-tdt-0.6b) didn't
            # fit even before any inference started — e.g. another process (or a
            # diarization pipeline that loaded first) already used most of VRAM.
            # Model weights don't shrink like a chunk does, so there's nothing to
            # retry smaller; CPU is the only real fallback. Slow, but the job
            # completes instead of failing outright.
            logger.warning(
                "[Parakeet] CUDA OOM while loading model weights — falling back "
                "to CPU. Transcription will be much slower. (%s)", exc,
            )
            torch.cuda.empty_cache()
            model = model.cpu()
        else:
            logger.info(
                "[Parakeet] Model loaded in %.1fs on cuda:%d (%s)",
                time.monotonic() - t0,
                torch.cuda.current_device(),
                torch.cuda.get_device_name(),
            )
    else:
        logger.warning(
            "[Parakeet] CUDA not available — model on CPU. "
            "Parakeet targets NVIDIA GPUs; CPU throughput will be very slow."
        )
    return model


def _get_frame_duration(model: Any) -> float:
    try:
        hop_s: float = float(model.cfg.preprocessor.window_stride)
        subsampling: int = int(model.cfg.encoder.subsampling_factor)
        return hop_s * subsampling
    except Exception:
        return _FALLBACK_FRAME_DURATION_S


# ---------------------------------------------------------------------------
# WAV chunking
# ---------------------------------------------------------------------------


def split_wav_with_overlap(
    wav_path: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[str, float, float]]:
    """
    Split a WAV file into overlapping chunks for chunked NeMo inference.

    Each chunk covers a nominal [ns, ne) window of the original audio.  The
    actual WAV written is extended by overlap_seconds on each side so a word
    that straddles the nominal boundary is captured by at least one chunk.

    Returns a list of (chunk_path, wav_start_s, nominal_start_s):
      chunk_path     — path to the written temp WAV (or original if short audio)
      wav_start_s    — where this WAV begins in the original timeline
      nominal_start_s — ownership boundary: words with abs_start >= nominal_start_s
                        belong to this chunk (used for deduplication)

    The caller is responsible for deleting temp files (chunk_path != wav_path).
    """
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        total_frames = wf.getnframes()
        total_seconds = total_frames / sr
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

    if total_seconds <= chunk_seconds:
        # Short audio: return original, no splitting needed.
        return [(wav_path, 0.0, 0.0)]

    chunks: list[tuple[str, float, float]] = []
    n_chunks = max(1, int(total_seconds / chunk_seconds) + (1 if total_seconds % chunk_seconds > 0.1 else 0))

    # Re-open for sequential reads across chunks.
    with wave.open(wav_path, "rb") as wf:
        for i in range(n_chunks):
            nominal_start = i * chunk_seconds
            nominal_end = min((i + 1) * chunk_seconds, total_seconds)

            if nominal_start >= total_seconds:
                break  # past end of audio

            # Extend each chunk by overlap on both sides so the boundary word
            # is present in the chunk's audio.  First chunk has no left overlap;
            # last chunk has no right overlap beyond the file end.
            wav_start = max(0.0, nominal_start - overlap_seconds)
            wav_end = min(total_seconds, nominal_end + overlap_seconds)

            start_frame = int(wav_start * sr)
            end_frame = min(int(wav_end * sr) + 1, total_frames)
            n_to_read = end_frame - start_frame

            wf.setpos(start_frame)
            raw = wf.readframes(n_to_read)

            with tempfile.NamedTemporaryFile(
                suffix=f"_prkt_{i:03d}.wav", delete=False
            ) as tmp:
                tmp_name = tmp.name

            with wave.open(tmp_name, "wb") as out:
                out.setnchannels(n_channels)
                out.setsampwidth(sampwidth)
                out.setframerate(sr)
                out.writeframes(raw)

            chunks.append((tmp_name, wav_start, nominal_start))

    return chunks


def _delete_temp(path: str, original: str) -> None:
    """Silently remove a temp chunk file if it is not the original audio."""
    if path != original:
        with contextlib.suppress(Exception):
            os.unlink(path)


def _cuda_empty_cache() -> None:
    """Call torch.cuda.empty_cache() if CUDA is available.  No-op otherwise."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Timestamp extraction from NeMo Hypothesis
# ---------------------------------------------------------------------------


def _distribute_text(text: str, audio_duration: float) -> list[RawWord]:
    """Even-distribution fallback — used only when no timestamps are available."""
    tokens = text.strip().split()
    if not tokens:
        return []
    logger.debug(
        "[Parakeet] No timestamps — distributing %d tokens over %.0fs",
        len(tokens),
        audio_duration,
    )
    dur = audio_duration / len(tokens)
    return [
        RawWord(text=t, start=round(i * dur, 3), end=round((i + 1) * dur, 3), confidence=None)
        for i, t in enumerate(tokens)
    ]


def _extract_words(hyp: Any, audio_duration: float, frame_duration: float) -> list[RawWord]:
    """
    Extract RawWord list from a NeMo Hypothesis object.

    Priority order:
    1. NeMo 2.x  hyp.timestamp['word'] — preferred (accurate TDT timestamps)
    2. NeMo 1.x  hyp.timestep_units    — frame-offset integers
    3. Even distribution across audio_duration (last-resort fallback)
    """
    if isinstance(hyp, str):
        return _distribute_text(hyp, audio_duration)

    # ── NeMo 2.x: hyp.timestamp dict ─────────────────────────────────────────
    timestamp = getattr(hyp, "timestamp", None)
    if isinstance(timestamp, dict):
        word_ts = timestamp.get("word") or []
        if word_ts:
            try:
                words: list[RawWord] = []
                for wt in word_ts:
                    text = (wt.get("word") or wt.get("token") or "").strip()
                    if not text:
                        continue
                    start_s = float(wt.get("start", 0.0))
                    end_s = float(wt.get("end", start_s + 0.1))
                    words.append(
                        RawWord(
                            text=text,
                            start=round(start_s, 3),
                            end=round(max(end_s, start_s + 0.01), 3),
                            confidence=None,
                        )
                    )
                if words:
                    logger.debug("[Parakeet] %d words from NeMo 2.x timestamp dict", len(words))
                    return words
            except Exception as exc:
                logger.warning("[Parakeet] NeMo 2.x timestamp parse failed: %s", exc)

    # ── NeMo 1.x: hyp.timestep_units ─────────────────────────────────────────
    timestep_units = getattr(hyp, "timestep_units", None)
    if timestep_units:
        try:
            words = []
            for unit in timestep_units:
                word = getattr(unit, "word", None)
                start_frame = getattr(unit, "start_offset", None)
                end_frame = getattr(unit, "end_offset", None)
                if word is None or start_frame is None:
                    continue
                word = word.strip()
                if not word:
                    continue
                start_s = float(start_frame) * frame_duration
                end_s = (
                    float(end_frame) * frame_duration
                    if end_frame is not None
                    else start_s + frame_duration
                )
                words.append(
                    RawWord(
                        text=word,
                        start=round(start_s, 3),
                        end=round(max(end_s, start_s + 0.01), 3),
                        confidence=None,
                    )
                )
            if words:
                logger.debug("[Parakeet] %d words from NeMo 1.x timestep_units", len(words))
                return words
        except Exception as exc:
            logger.warning("[Parakeet] timestep_units parse failed: %s", exc)

    # ── Fallback ──────────────────────────────────────────────────────────────
    text = getattr(hyp, "text", "")
    return _distribute_text(str(text) if not isinstance(text, str) else text, audio_duration)


def _ts_source_label(hyp: Any) -> str:
    if isinstance(hyp, str):
        return "distributed"
    if (getattr(hyp, "timestamp", None) or {}).get("word"):
        return "NeMo 2.x"
    if getattr(hyp, "timestep_units", None):
        return "NeMo 1.x"
    return "distributed"


# ---------------------------------------------------------------------------
# Single-chunk inference (used by both paths)
# ---------------------------------------------------------------------------


def _call_transcribe(model: Any, chunk_path: str) -> list[Any]:
    """
    Call model.transcribe() inside inference_mode, handling NeMo API variations.

    torch.inference_mode() is strictly stronger than no_grad: it additionally
    disables autograd view tracking, reducing per-tensor overhead.  NeMo uses
    no_grad internally; nesting with inference_mode is safe and composable.
    """
    import torch  # noqa: PLC0415

    with torch.inference_mode():
        try:
            hypotheses = model.transcribe(
                audio=[chunk_path],
                batch_size=1,
                return_hypotheses=True,
                verbose=False,
            )
        except TypeError as exc:
            logger.debug("[Parakeet] transcribe() kwarg fallback: %s", exc)
            hypotheses = model.transcribe(
                [chunk_path],
                batch_size=1,
                return_hypotheses=True,
            )
    return hypotheses if hypotheses else []


def _unwrap_hypothesis(raw: Any) -> Any:
    """NeMo sometimes nests hypotheses as [[Hypothesis,...]] — unwrap if needed."""
    if isinstance(raw, (list, tuple)) and raw:
        return raw[0]
    return raw


# ---------------------------------------------------------------------------
# Chunked transcription
# ---------------------------------------------------------------------------


def _transcribe_chunked(
    model: Any,
    wav_path: str,
    audio_duration: float,
    frame_duration: float,
    chunk_seconds: float,
    overlap_seconds: float,
    max_retries: int,
    on_progress: ProgressCallback | None,
) -> tuple[list[RawWord], str]:
    """
    Transcribe a long audio file by splitting into overlapping chunks.

    Word timestamps are offset by each chunk's position in the original audio.
    A boundary deduplication step ensures words near chunk seams appear exactly
    once — the word is owned by whichever chunk's nominal window contains its
    start time.

    VRAM management per chunk:
      - gc.collect() releases Python-level tensor references
      - torch.cuda.empty_cache() returns freed pages to the CUDA allocator pool
      - Model weights remain resident; only activation memory is recycled

    Temp file safety: split_wav_with_overlap() creates all chunk temps upfront.
    If an exception propagates, the except clause below deletes any chunks that
    were not yet started so they don't accumulate in the OS temp dir.
    """
    chunks = split_wav_with_overlap(wav_path, chunk_seconds, overlap_seconds)
    n_chunks = len(chunks)
    all_words: list[RawWord] = []
    ts_sources: list[str] = []

    logger.info(
        "[Parakeet] Chunked transcription: %d chunks × %.0fs (overlap=%.1fs) for %.0fs audio",
        n_chunks,
        chunk_seconds,
        overlap_seconds,
        audio_duration,
    )
    log_vram_stats("before-chunked")

    processed_through = -1  # index of the last chunk we called _transcribe_chunk_with_retry on
    try:
        for chunk_i, (chunk_path, wav_start, nominal_start) in enumerate(chunks):
            # Nominal window owned by this chunk
            nominal_end = chunks[chunk_i + 1][2] if chunk_i + 1 < n_chunks else float("inf")
            chunk_wav_duration = min(nominal_end, audio_duration) - wav_start

            if on_progress:
                frac = 0.18 + 0.42 * (chunk_i / n_chunks)
                start_min = nominal_start / 60
                end_min = min(nominal_end, audio_duration) / 60
                on_progress(frac, f"Transcribing chunk {chunk_i + 1}/{n_chunks} ({start_min:.0f}–{end_min:.0f} min)")

            logger.info(
                "[Parakeet] Chunk %d/%d: wav=[%.1f, %.1f]s  nominal=[%.1f, %.1f]s",
                chunk_i + 1,
                n_chunks,
                wav_start,
                wav_start + chunk_wav_duration,
                nominal_start,
                min(nominal_end, audio_duration),
            )

            processed_through = chunk_i  # mark before call; retry func handles chunk_path cleanup
            words = _transcribe_chunk_with_retry(
                model=model,
                chunk_path=chunk_path,
                wav_path=wav_path,
                wav_start=wav_start,
                chunk_wav_duration=chunk_wav_duration,
                frame_duration=frame_duration,
                nominal_start=nominal_start,
                nominal_end=nominal_end,
                max_retries=max_retries,
                chunk_i=chunk_i,
                n_chunks=n_chunks,
                ts_sources=ts_sources,
            )
            all_words.extend(words)

            # Per-chunk VRAM cleanup: release activations, keep model weights.
            gc.collect()
            _cuda_empty_cache()
            log_vram_stats(f"after-chunk-{chunk_i + 1}")

    except Exception:
        # _transcribe_chunk_with_retry deletes chunks[processed_through][0] before re-raising.
        # Delete any remaining chunk temps that were created but not yet started.
        for i in range(processed_through + 1, n_chunks):
            _delete_temp(chunks[i][0], wav_path)
        raise

    overall_ts_source = ts_sources[0] if ts_sources else "distributed"
    return all_words, overall_ts_source


def _split_segment_in_half(
    seg_path: str, chunk_i: int, depth: int
) -> list[tuple[str, float, float]]:
    """
    Write the two halves of a segment to new temp WAVs.

    Returns [(path, offset_seconds_within_segment, duration_seconds), ...].
    The caller owns the returned temp files.

    A one-frame overlap is not added here: the halves are contiguous, and any
    word straddling the split is recovered by the outer chunk's own overlap
    padding. Splitting mid-word is rare and costs at most one word, whereas
    dropping half the audio (the previous behaviour) cost minutes of speech.
    """
    with wave.open(seg_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        half_frames = total_frames // 2
        wf.setpos(0)
        first_raw = wf.readframes(half_frames)
        second_raw = wf.readframes(total_frames - half_frames)

    halves: list[tuple[str, float, float]] = []
    for idx, (raw, offset_frames) in enumerate(
        ((first_raw, 0), (second_raw, half_frames))
    ):
        n_frames = len(raw) // (n_channels * sampwidth) if sampwidth else 0
        if n_frames <= 0:
            continue
        with tempfile.NamedTemporaryFile(
            suffix=f"_prkt_split_{chunk_i:03d}_{depth}_{idx}.wav", delete=False
        ) as tmp:
            half_path = tmp.name
        with wave.open(half_path, "wb") as out:
            out.setnchannels(n_channels)
            out.setsampwidth(sampwidth)
            out.setframerate(sr)
            out.writeframes(raw)
        halves.append((half_path, offset_frames / sr, n_frames / sr))

    return halves


def _transcribe_segment(
    model: Any,
    seg_path: str,
    wav_path: str,
    seg_start: float,
    seg_duration: float,
    frame_duration: float,
    retries_left: int,
    chunk_i: int,
    n_chunks: int,
    ts_sources: list[str],
) -> list[RawWord]:
    """
    Transcribe one contiguous audio segment, returning absolute-timestamped words.

    On CUDA OOM the segment is split in half and BOTH halves are transcribed
    (recursively, spending one retry per level). This is the whole point of the
    function: an earlier implementation truncated to the first half and threw
    the rest away, which silently deleted up to half a chunk — minutes of
    speech — from the transcript while still reporting success. Losing audio is
    never an acceptable response to memory pressure.

    Temp-file contract: deletes seg_path before returning or raising, unless it
    is the caller's original wav_path. Any half-segments it creates are deleted
    by the corresponding recursive call, and unstarted halves are cleaned up if
    an earlier half raises.
    """
    try:
        hypotheses = _call_transcribe(model, seg_path)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or retries_left <= 0:
            _delete_temp(seg_path, wav_path)
            raise

        logger.warning(
            "[Parakeet] CUDA OOM on chunk %d/%d (%.0fs segment, %d retr%s left) "
            "— splitting into two halves and transcribing both: %s",
            chunk_i + 1, n_chunks, seg_duration, retries_left,
            "y" if retries_left == 1 else "ies", exc,
        )
        gc.collect()
        _cuda_empty_cache()

        try:
            halves = _split_segment_in_half(seg_path, chunk_i, retries_left)
        except Exception:
            _delete_temp(seg_path, wav_path)
            raise
        _delete_temp(seg_path, wav_path)

        if not halves:
            # Segment too short to split further — nothing left to try.
            raise

        words: list[RawWord] = []
        for i, (half_path, half_offset, half_duration) in enumerate(halves):
            try:
                words.extend(
                    _transcribe_segment(
                        model=model,
                        seg_path=half_path,
                        wav_path=wav_path,
                        seg_start=seg_start + half_offset,
                        seg_duration=half_duration,
                        frame_duration=frame_duration,
                        retries_left=retries_left - 1,
                        chunk_i=chunk_i,
                        n_chunks=n_chunks,
                        ts_sources=ts_sources,
                    )
                )
            except Exception:
                # Clean up halves that were never started before propagating.
                for leftover_path, _, _ in halves[i + 1:]:
                    _delete_temp(leftover_path, wav_path)
                raise
        return words

    except Exception:
        _delete_temp(seg_path, wav_path)
        raise

    _delete_temp(seg_path, wav_path)

    if not hypotheses:
        logger.info(
            "[Parakeet] Chunk %d/%d: empty output for %.0fs segment",
            chunk_i + 1, n_chunks, seg_duration,
        )
        return []

    hyp = _unwrap_hypothesis(hypotheses[0])
    words_relative = _extract_words(hyp, seg_duration, frame_duration)
    ts_sources.append(_ts_source_label(hyp))

    # Offset timestamps from segment-relative to original-audio absolute.
    return [
        RawWord(
            text=w.text,
            start=round(w.start + seg_start, 3),
            end=round(w.end + seg_start, 3),
            confidence=w.confidence,
        )
        for w in words_relative
    ]


def _transcribe_chunk_with_retry(
    model: Any,
    chunk_path: str,
    wav_path: str,
    wav_start: float,
    chunk_wav_duration: float,
    frame_duration: float,
    nominal_start: float,
    nominal_end: float,
    max_retries: int,
    chunk_i: int,
    n_chunks: int,
    ts_sources: list[str],
) -> list[RawWord]:
    """
    Transcribe a single chunk and return only the words it owns.

    OOM handling lives in _transcribe_segment (split-and-keep-both). This
    function adds the ownership filter: a word belongs to this chunk when its
    start falls in [nominal_start, nominal_end), which deduplicates words that
    appear in the overlap zone shared with the adjacent chunk.
    """
    words_absolute = _transcribe_segment(
        model=model,
        seg_path=chunk_path,
        wav_path=wav_path,
        seg_start=wav_start,
        seg_duration=chunk_wav_duration,
        frame_duration=frame_duration,
        retries_left=max_retries,
        chunk_i=chunk_i,
        n_chunks=n_chunks,
        ts_sources=ts_sources,
    )

    words_owned = [w for w in words_absolute if nominal_start <= w.start < nominal_end]

    logger.info(
        "[Parakeet] Chunk %d/%d: %d words extracted → %d owned (%s)",
        chunk_i + 1,
        n_chunks,
        len(words_absolute),
        len(words_owned),
        ts_sources[-1] if ts_sources else "none",
    )
    return words_owned


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ParakeetEngine:
    """
    Transcription engine backed by NVIDIA NeMo Parakeet-TDT.

    Satisfies the TranscriptionEngine protocol (same interface as WhisperXEngine).
    No alignment step: TDT decoder produces word-level timestamps natively.

    Model is loaded lazily on the first transcribe() call and cached for the
    process lifetime.  All model loading and inference is serialised through
    the shared GPU_COMPUTE_LOCK — NeMo models are not thread-safe for
    simultaneous inference, and the same lock keeps this engine from running
    concurrently with GPU-placed pyannote diarization (see gpu/hardware.py).

    Hardware is re-detected on every transcribe() call (cheap — a property
    query, not an allocation) so chunk sizing reflects current free VRAM
    rather than a snapshot from whenever the model was first loaded.

    Environment:
      TE_ASR_BACKEND=parakeet         — enable this engine
      TE_PARAKEET_MODEL=nvidia/...    — override model (default: parakeet-tdt-0.6b-v2)
    """

    _model: Any = None
    _model_id: str = ""
    _frame_duration: float = _FALLBACK_FRAME_DURATION_S
    _gpu_info: GpuInfo | None = None

    def transcribe(
        self,
        audio: PreparedAudio,
        config: TranscriptionConfig,
        on_progress: ProgressCallback | None = None,
        skip_alignment: bool = False,  # ignored: TDT provides timestamps natively
        pipeline_config: PipelineConfig | None = None,
    ) -> RawTranscription:
        model_id = os.environ.get("TE_PARAKEET_MODEL", _DEFAULT_MODEL)

        # pipeline_config is not passed through the TranscriptionEngine protocol, so load
        # settings directly when it is absent. This makes ParakeetConfig env vars reachable.
        resolved_pipeline_config = pipeline_config
        if resolved_pipeline_config is None:
            try:
                from transcript_engine.config.loader import load_settings  # noqa: PLC0415
                resolved_pipeline_config = load_settings().pipeline
            except Exception as _cfg_err:
                logger.warning(
                    "[Parakeet] Could not load pipeline settings (%s) — using defaults",
                    _cfg_err,
                )
        parakeet_cfg = resolved_pipeline_config.parakeet if resolved_pipeline_config is not None else None

        # Lazy model load — lock prevents a concurrent call from loading twice,
        # and prevents this load from racing GPU-placed diarization for VRAM.
        with GPU_COMPUTE_LOCK:
            if ParakeetEngine._model is None or ParakeetEngine._model_id != model_id:
                if on_progress:
                    on_progress(0.05, "Loading Parakeet model")
                gpu_at_load = detect_gpu()
                ParakeetEngine._model = _load_nemo_model(model_id)
                ParakeetEngine._model_id = model_id
                ParakeetEngine._frame_duration = _get_frame_duration(ParakeetEngine._model)
                if gpu_at_load is not None:
                    configure_cuda_for_inference(gpu_at_load)
                logger.info(
                    "[Parakeet] Frame duration: %dms",
                    int(ParakeetEngine._frame_duration * 1000),
                )

        model = ParakeetEngine._model

        # Measure free VRAM *after* the model is resident, and on every call.
        #
        # Order matters: the weights are ~2.5 GB, so a reading taken before the
        # load reports headroom the job will not actually have. On a cold start
        # without warm-loading that overestimate is enough to pick a chunk size
        # one or two tiers too large and OOM on the very first job.
        #
        # Re-reading per call also keeps sizing honest as conditions change
        # between jobs (allocator fragmentation, a diarization pipeline that
        # loaded onto the same GPU since last time). detect_gpu() only queries
        # device properties and never allocates, so this is cheap.
        gpu = detect_gpu()
        ParakeetEngine._gpu_info = gpu

        # Resolve chunk parameters: config override → VRAM detection → safe default.
        overlap_s = (
            parakeet_cfg.overlap_seconds if parakeet_cfg is not None else DEFAULT_OVERLAP_SECONDS
        )
        max_retries = parakeet_cfg.max_chunk_retries if parakeet_cfg is not None else 2

        configured_chunk_s = (
            parakeet_cfg.chunk_seconds
            if parakeet_cfg is not None and parakeet_cfg.chunk_seconds > 0
            else 0.0
        )
        if configured_chunk_s > 0:
            chunk_s = configured_chunk_s
            logger.info("[Parakeet] Chunk size from config: %.0fs", chunk_s)
        elif gpu is not None:
            chunk_s = max(_MIN_CHUNK_SECONDS, optimal_chunk_seconds(gpu.vram_free_gb))
            logger.info(
                "[Parakeet] Chunk size from VRAM detection (%.1f GB free): %.0fs",
                gpu.vram_free_gb,
                chunk_s,
            )
        else:
            chunk_s = 300.0  # safe default for unknown hardware
            logger.info("[Parakeet] Chunk size: %.0fs (no GPU info — using safe default)", chunk_s)

        if on_progress:
            on_progress(0.18, f"Transcribing {audio.original_path.name}")

        logger.info(
            "[Parakeet] Transcribing: %s (%.0fs, %.1f min) — model=%s chunk=%.0fs overlap=%.1fs",
            audio.original_path.name,
            audio.duration,
            audio.duration / 60,
            model_id,
            chunk_s,
            overlap_s,
        )

        wav_path = str(audio.path)
        t0 = time.monotonic()

        # Always go through _transcribe_chunked, even for audio shorter than
        # chunk_s: split_wav_with_overlap() degrades to a single whole-file
        # "chunk" with no copy/split in that case, so this costs nothing extra
        # but guarantees every job — not just long ones — gets OOM
        # retry-with-halving via _transcribe_chunk_with_retry. Previously,
        # short audio used a direct, unprotected model.transcribe() call with
        # no retry path at all.
        with GPU_COMPUTE_LOCK:
            words, ts_source = _transcribe_chunked(
                model=model,
                wav_path=wav_path,
                audio_duration=audio.duration,
                frame_duration=ParakeetEngine._frame_duration,
                chunk_seconds=chunk_s,
                overlap_seconds=overlap_s,
                max_retries=max_retries,
                on_progress=on_progress,
            )

            if not words:
                if on_progress:
                    on_progress(0.60, "Transcription complete")
                return RawTranscription(
                    words=(), language=config.language or "en", duration=audio.duration
                )

        elapsed = time.monotonic() - t0
        rtf = audio.duration / elapsed if elapsed > 0 else 0.0
        log_vram_stats("post-transcription")

        logger.info(
            "[Parakeet] Done: %.0fs audio in %.1fs | RTF=%.0fx | %d words | ts=%s",
            audio.duration,
            elapsed,
            rtf,
            len(words),
            ts_source,
        )

        if on_progress:
            on_progress(0.60, "Transcription complete")

        return RawTranscription(
            words=tuple(words),
            language=config.language or "en",
            duration=audio.duration,
        )
