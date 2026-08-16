from __future__ import annotations

import importlib.util
import os
import time
from typing import TYPE_CHECKING, Any

from transcript_engine.config.settings import TranscriptionConfig
from transcript_engine.logging import get_logger
from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.pipeline import RawTranscription, RawWord
from transcript_engine.models.types import ProgressCallback

if TYPE_CHECKING:
    from transcript_engine.model_registry.registry import ModelRegistry

logger = get_logger(__name__)

# Map from faster-whisper model IDs to MLX-community repo equivalents.
# MLX is opt-in: set TE_USE_MLX=1 in .env to enable.
#
# WARNING: MLX on M1 8 GB suffers from Metal GPU thermal throttling that
# degrades token generation from ~20 tokens/s (cold) to ~4 tokens/s (warm),
# making 8-min meetings take 20+ minutes instead of ~2 minutes.
# CTranslate2 (CPU int8 + VAD + batch_size=16) is more predictable at
# ~5-8 min for 8-min meetings.  Enable MLX only on machines with active
# cooling or on recordings shorter than ~3 minutes.
_MLX_MODEL_MAP: dict[str, str] = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}

# MLX is disabled by default.  Set TE_USE_MLX=1 to enable.
_USE_MLX = os.environ.get("TE_USE_MLX", "").lower() in ("1", "true", "yes")
_HAS_MLX = _USE_MLX and importlib.util.find_spec("mlx_whisper") is not None


class WhisperXEngine:
    """
    Transcription engine backed by WhisperX (faster-whisper + forced alignment).
    On Apple Silicon, auto-selects the MLX backend for the transcription step,
    then feeds the segments through the same wav2vec2 alignment step.

    MLX runs on the Metal GPU and is ~3-4x faster than CTranslate2 int8 on CPU
    for the transcription stage on M-series Macs.

    Device and compute_type selection is delegated to ModelRegistry.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def transcribe(
        self,
        audio: PreparedAudio,
        config: TranscriptionConfig,
        on_progress: ProgressCallback | None = None,
        skip_alignment: bool = False,
    ) -> RawTranscription:
        mlx_repo = _MLX_MODEL_MAP.get(config.model_id) if _HAS_MLX else None

        if mlx_repo:
            return self._transcribe_mlx(
                audio, config, mlx_repo, on_progress, skip_alignment
            )
        return self._transcribe_ct2(audio, config, on_progress, skip_alignment)

    # ── MLX backend ─────────────────────────────────────────────────────────

    def _transcribe_mlx(
        self,
        audio: PreparedAudio,
        config: TranscriptionConfig,
        mlx_repo: str,
        on_progress: ProgressCallback | None,
        skip_alignment: bool = False,
    ) -> RawTranscription:
        import mlx_whisper  # noqa: PLC0415
        import whisperx  # noqa: PLC0415

        if on_progress:
            on_progress(0.08, "Loading Whisper model")

        logger.info(
            f"[MLX] Transcribing: {audio.original_path.name} "
            f"({audio.duration:.0f}s) via {mlx_repo}"
        )

        if on_progress:
            on_progress(0.18, f"Transcribing {audio.original_path.name}")

        t0 = time.monotonic()
        try:
            result = mlx_whisper.transcribe(
                str(audio.path),
                path_or_hf_repo=mlx_repo,
                language=config.language or None,
                verbose=False,
            )
        except (IndexError, StopIteration):
            logger.info("No speech detected in audio — returning empty transcription")
            return RawTranscription(words=(), language="en", duration=audio.duration)

        elapsed = time.monotonic() - t0
        detected_language: str = result.get("language") or config.language or "en"
        segs: list[dict[str, Any]] = result.get("segments", [])

        rtf = audio.duration / elapsed if elapsed > 0 else 0
        logger.info(
            f"[MLX] Transcription done in {elapsed:.1f}s "
            f"({rtf:.1f}x real-time), language={detected_language}"
        )

        if not segs:
            return RawTranscription(
                words=(), language=detected_language, duration=audio.duration
            )

        # Feed MLX segments into wav2vec2 alignment for precise word timestamps.
        # MLX segments have the same {text, start, end} schema whisperx expects.
        if on_progress:
            on_progress(0.45, f"Aligning timestamps (language={detected_language})")

        n_segs = len(segs)
        total_words_in_segs = sum(
            len(s.get("words", s.get("text", "").split())) for s in segs
        )
        logger.info(
            f"[MLX] Alignment input: {n_segs} segments, ~{total_words_in_segs} words, "
            f"audio={audio.duration:.0f}s"
        )

        if skip_alignment:
            logger.info(
                "[MLX] Skipping wav2vec2 alignment — extracting words from %d segments "
                "(segment-level timestamps, ±segment_duration accuracy)",
                n_segs,
            )
            words = self._extract_words({"segments": segs})
            logger.info(
                "[MLX] Segment extraction: %d segments → %d words (skip_alignment path)",
                n_segs,
                len(words),
            )
        else:
            t0 = time.monotonic()
            align_model, align_metadata = self._registry.get_alignment_model(
                detected_language, config
            )
            logger.info(f"[MLX] Alignment model loaded in {time.monotonic() - t0:.2f}s")

            torch_device = "cpu"  # must match the device used in get_alignment_model()

            try:
                actual_param = next(iter(align_model.parameters()))
                actual_device = str(actual_param.device)
            except Exception:
                actual_device = "unknown"
            logger.info(
                f"[MLX] Alignment: requested device={torch_device}, "
                f"actual model device={actual_device}"
            )

            t_align = time.monotonic()
            try:
                aligned = whisperx.align(
                    segs,
                    align_model,
                    align_metadata,
                    str(audio.path),
                    torch_device,
                    return_char_alignments=False,
                )
            except Exception as exc:
                # Alignment can fail on unusual audio; fall back to segment-level timing.
                logger.warning(
                    f"[MLX] Alignment failed ({exc}); using segment-level timestamps"
                )
                aligned = {"segments": segs}

            align_elapsed = time.monotonic() - t_align
            words = self._extract_words(aligned)
            rtf_align = audio.duration / align_elapsed if align_elapsed > 0 else 0
            logger.info(
                f"[MLX] Alignment done: elapsed={align_elapsed:.1f}s, "
                f"RTF={rtf_align:.1f}x, words_out={len(words)}, device={actual_device}"
            )
            if align_elapsed > audio.duration * 0.20:
                logger.warning(
                    f"[MLX] SLOW ALIGNMENT: {align_elapsed:.0f}s for {audio.duration:.0f}s audio "
                    f"({align_elapsed/audio.duration:.2f}x audio duration). "
                    f"Segments={n_segs}, device={actual_device}"
                )

        if on_progress:
            on_progress(0.60, "Transcription complete")

        return RawTranscription(
            words=tuple(words),
            language=detected_language,
            duration=audio.duration,
        )

    # ── CTranslate2 backend ──────────────────────────────────────────────────

    def _transcribe_ct2(
        self,
        audio: PreparedAudio,
        config: TranscriptionConfig,
        on_progress: ProgressCallback | None,
        skip_alignment: bool = False,
    ) -> RawTranscription:
        import whisperx  # noqa: PLC0415

        device = self._registry._resolve_ctranslate_device(config.device)

        if on_progress:
            on_progress(0.08, "Loading Whisper model")

        t0 = time.monotonic()
        model = self._registry.get_whisper_model(config)
        logger.info(f"Whisper model ready in {time.monotonic() - t0:.1f}s")

        if on_progress:
            on_progress(0.18, f"Transcribing {audio.original_path.name}")

        logger.info(
            f"[CT2/{device}] Transcribing: {audio.original_path.name} "
            f"({audio.duration:.0f}s)"
        )

        # batch_size>0 activates whisperx's PyTorch batched pipeline — fast on CUDA,
        # catastrophically slow on CPU (0.5x RTF vs 3-4x RTF for native streaming).
        # Use the native faster-whisper streaming path on CPU (batch_size=None).
        effective_batch = config.batch_size if device == "cuda" else None

        t0 = time.monotonic()
        try:
            result = model.transcribe(
                str(audio.path),
                batch_size=effective_batch,
                language=config.language,
                chunk_size=config.chunk_length,
            )
        except (IndexError, StopIteration):
            # whisperx raises IndexError when VAD finds no speech segments.
            # Return an empty transcription rather than crashing the pipeline.
            logger.info("No speech detected in audio — returning empty transcription")
            return RawTranscription(words=(), language="en", duration=audio.duration)
        elapsed = time.monotonic() - t0
        detected_language = result.get("language", "en")

        rtf = audio.duration / elapsed if elapsed > 0 else 0
        logger.info(
            f"Transcription done in {elapsed:.1f}s "
            f"({rtf:.1f}x real-time), language={detected_language}"
        )

        segs = result["segments"]
        n_segs = len(segs)
        total_words_in_segs = sum(
            len(s.get("words", s.get("text", "").split())) for s in segs
        )
        logger.info(
            f"Alignment input: {n_segs} segments, ~{total_words_in_segs} words, "
            f"audio={audio.duration:.0f}s"
        )

        if on_progress:
            on_progress(0.45, f"Aligning timestamps (language={detected_language})")

        if skip_alignment:
            logger.info(
                "Skipping wav2vec2 alignment — extracting words from %d CT2 segments "
                "(segment-level timestamps, ±segment_duration accuracy)",
                n_segs,
            )
            words = self._extract_words({"segments": segs})
            logger.info(
                "CT2 segment extraction: %d segments → %d words (skip_alignment path)",
                n_segs,
                len(words),
            )
        else:
            t0 = time.monotonic()
            align_model, align_metadata = self._registry.get_alignment_model(
                detected_language, config
            )
            logger.info(f"Alignment model loaded in {time.monotonic() - t0:.2f}s")

            torch_device = "cpu"  # must match the device used in get_alignment_model()

            # Verify the actual compute device the model landed on.
            try:
                actual_param = next(iter(align_model.parameters()))
                actual_device = str(actual_param.device)
            except Exception:
                actual_device = "unknown"
            logger.info(
                f"Alignment: requested device={torch_device}, "
                f"actual model device={actual_device}"
            )

            t_align = time.monotonic()
            try:
                aligned = whisperx.align(
                    segs,
                    align_model,
                    align_metadata,
                    str(audio.path),
                    torch_device,
                    return_char_alignments=False,
                )
            except Exception as exc:
                logger.warning(
                    f"Alignment failed ({exc}); using segment-level timestamps"
                )
                aligned = {"segments": segs}
            align_elapsed = time.monotonic() - t_align

            words = self._extract_words(aligned)
            rtf_align = audio.duration / align_elapsed if align_elapsed > 0 else 0
            logger.info(
                f"Alignment done: elapsed={align_elapsed:.1f}s, "
                f"RTF={rtf_align:.1f}x, "
                f"words_out={len(words)}, "
                f"device={actual_device}"
            )
            if align_elapsed > audio.duration * 0.20:
                # Alignment is consuming more than 20% of audio duration — abnormal.
                logger.warning(
                    f"SLOW ALIGNMENT: {align_elapsed:.0f}s for {audio.duration:.0f}s audio "
                    f"({align_elapsed/audio.duration:.2f}x audio duration). "
                    f"Expected <{audio.duration * 0.10:.0f}s. "
                    f"Segments={n_segs}, device={actual_device}"
                )

        if on_progress:
            on_progress(0.60, "Transcription complete")

        return RawTranscription(
            words=tuple(words),
            language=detected_language,
            duration=audio.duration,
        )

    # ── Shared word extraction ───────────────────────────────────────────────

    def _extract_words(self, aligned: dict[str, Any]) -> list[RawWord]:
        """
        Extract word-level timing from an aligned result dict.
        Works with both whisperx alignment output (key: 'score') and
        MLX segment-level fallback (key: 'probability' or absent).
        When a segment has no word-level data (raw CT2 output, skip_alignment path),
        distributes the segment text evenly across the segment duration.
        Guards against missing start/end.
        """
        words: list[RawWord] = []
        last_end = 0.0

        for segment in aligned.get("segments", []):
            word_data_list = segment.get("words", [])

            if not word_data_list:
                # Raw CT2 segments have no word-level timing (whisperx returns only
                # {text, start, end} without alignment). Distribute text tokens evenly
                # across the segment duration so the transcript has content.
                seg_text = segment.get("text", "").strip()
                if not seg_text:
                    continue
                seg_start = float(segment.get("start", last_end))
                seg_end = float(segment.get("end", seg_start + 0.1))
                if seg_end <= seg_start:
                    seg_end = seg_start + 0.05
                tokens = seg_text.split()
                if tokens:
                    dur_per_token = (seg_end - seg_start) / len(tokens)
                    for i, token in enumerate(tokens):
                        t_start = round(seg_start + i * dur_per_token, 3)
                        t_end = round(t_start + dur_per_token, 3)
                        words.append(
                            RawWord(
                                text=token,
                                start=t_start,
                                end=t_end,
                                confidence=None,
                            )
                        )
                    last_end = seg_end
                continue

            for word_data in word_data_list:
                text = word_data.get("word", "").strip()
                if not text:
                    continue

                start = word_data.get("start")
                end = word_data.get("end")

                if start is None:
                    start = segment.get("start", last_end)
                if end is None:
                    end = segment.get("end", start + 0.1)

                start = float(start)
                end = float(end)

                if end <= start:
                    end = start + 0.05

                # whisperx alignment uses 'score'; MLX uses 'probability'
                raw_conf = word_data.get("score") or word_data.get("probability")
                confidence = float(raw_conf) if raw_conf is not None else None

                words.append(
                    RawWord(
                        text=text,
                        start=round(start, 3),
                        end=round(end, 3),
                        confidence=confidence,
                    )
                )
                last_end = end

        return words
