"""
Pipeline Orchestrator

The single composition root. Coordinates all stages in order.
Every interface (CLI, REST API, Python SDK, batch runner) calls Pipeline.run().
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any

from transcript_engine.audio.chunker import cleanup_chunks, split_audio
from transcript_engine.audio.preprocessor import AudioPreprocessor
from transcript_engine.config.settings import PipelineConfig
from transcript_engine.diarization.pyannote_engine import PyannoteEngine
from transcript_engine.logging import get_logger
from transcript_engine.merger.merger import TranscriptMerger
from transcript_engine.model_registry.registry import ModelRegistry
from transcript_engine.models.pipeline import (
    DiarizationResult,
    PipelineResult,
    SpeakerSegment,
)
from transcript_engine.models.transcript import Transcript
from transcript_engine.models.types import ProgressCallback
from transcript_engine.pipeline.parallel_diarizer import diarize_parallel
from transcript_engine.processors.ai_grammar import AIGrammarProcessor
from transcript_engine.processors.base import ProcessorContext
from transcript_engine.processors.cleanup import TranscriptCleanupProcessor
from transcript_engine.processors.context_correction import ContextCorrectionProcessor
from transcript_engine.processors.intelligence import TranscriptIntelligenceProcessor
from transcript_engine.processors.registry import get_processor
from transcript_engine.processors.speaker_formatting import SpeakerFormattingProcessor
from transcript_engine.processors.vocabulary import VocabularyCorrectionProcessor
from transcript_engine.profiles.loader import load_profile
from transcript_engine.profiles.model import Profile
from transcript_engine.transcription.parakeet_engine import ParakeetEngine
from transcript_engine.transcription.whisperx_engine import WhisperXEngine


# Diagnostic word-count helper — logs how many words remain at each pipeline stage.
def _wc(label: str, obj: Any) -> None:
    """Log word count at a named pipeline stage for correctness debugging."""
    try:
        if hasattr(obj, "word_count"):
            n = obj.word_count
        elif hasattr(obj, "words"):
            n = len(obj.words)
        elif isinstance(obj, list):
            n = len(obj)
        else:
            n = -1
        logger.info("STAGE_WORDCOUNT [%s]: %d words", label, n)
    except Exception:
        pass

logger = get_logger(__name__)

# Progress weights for the Rich progress bar (must sum to 1.0)
_STAGE_WEIGHTS = {
    "preprocess": (0.00, 0.05),
    "transcribe": (0.05, 0.62),
    "diarize": (0.62, 0.88),
    "merge": (0.88, 0.92),
    "process": (0.92, 0.97),
    "export_prep": (0.97, 1.00),
}


class Pipeline:
    """
    Executes the full transcription pipeline from audio file to Transcript.

    Usage:
        pipeline = Pipeline(config, registry, profile=profile)
        result = pipeline.run(Path("meeting.mp4"), on_progress=callback)
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: ModelRegistry,
        profile: Profile | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        if profile is None:
            profile = _load_profile_for_config(config)
        self._profile = profile
        _asr_backend = os.environ.get("TE_ASR_BACKEND", "whisper").lower()
        _engine: ParakeetEngine | WhisperXEngine
        if _asr_backend == "parakeet":
            _engine = ParakeetEngine()
            logger.info("ASR backend: Parakeet (NVIDIA NeMo TDT)")
        else:
            _engine = WhisperXEngine(registry)
            logger.info(f"ASR backend: WhisperX ({_asr_backend})")
        self._transcription_engine = _engine
        self._diarization_engine = PyannoteEngine(registry)
        self._merger = TranscriptMerger()
        self._processors = self._build_processors()

    def run(
        self,
        input_path: Path,
        on_progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        pipeline_start = time.monotonic()
        warnings: list[str] = []
        timings: dict[str, float] = {}

        logger.info(f"Pipeline start: {input_path.name} | profile={self._profile.name}")
        _progress(on_progress, "preprocess", 0.0, "Preprocessing audio")

        with AudioPreprocessor(self._config.audio) as preprocessor:
            t0 = time.monotonic()
            audio = preprocessor.prepare(input_path)
            timings["preprocess"] = time.monotonic() - t0
            logger.info(
                f"Audio preprocessed in {timings['preprocess']:.1f}s "
                f"(duration={audio.duration:.1f}s)"
            )
            logger.info(
                "STAGE_AUDIO: duration=%.1fs (%.1f min), path=%s",
                audio.duration, audio.duration / 60, audio.path,
            )
            _progress(
                on_progress, "preprocess", 1.0, f"Audio ready|{audio.duration:.1f}"
            )

            # ── Fast pre-analysis: make adaptive pipeline decisions ───────────
            from transcript_engine.audio.analyzer import AudioAnalyzer  # noqa: PLC0415

            _decision = None
            _audio_profile = None
            try:
                _analyzer = AudioAnalyzer()
                _audio_profile, _decision = _analyzer.analyze(audio.path)
                for _reason in _decision.reasons:
                    logger.info("Pipeline decision: %s", _reason)
                # Forward decisions to UI via progress callback
                if on_progress and _decision.reasons:
                    import json as _json  # noqa: PLC0415

                    on_progress(
                        -1.0,
                        f"pipeline_decision|{_json.dumps({'decisions': _decision.reasons, 'savings_s': _decision.estimated_savings_s})}",
                    )
            except Exception as _ae:
                logger.warning("Audio analysis failed: %s — using full pipeline", _ae)
                _decision = None

            _skip_alignment = self._config.transcription.force_skip_alignment or (
                _decision is not None and not _decision.use_wav2vec2_alignment
            )
            _use_diarization = self._config.diarization.enabled and (
                _decision is None or _decision.use_diarization
            )

            if _use_diarization:
                # ── Parallel architecture ────────────────────────────────
                # Transcription runs in the main thread; diarization runs
                # concurrently (in-process for short audio, or in N worker
                # processes for long audio — see diarize_parallel).
                #
                # Whether this is genuinely parallel depends on the ASR
                # backend and where diarization landed:
                #   - WhisperX/CT2 (CPU) + pyannote (CPU or MPS): no GPU
                #     contention either way.
                #   - Parakeet (CUDA) + pyannote on CPU (small/no GPU): truly
                #     parallel, no contention.
                #   - Parakeet (CUDA) + pyannote on CUDA (>= 8 GB card, e.g.
                #     the RTX 3060 Ti target): both threads submit to the same
                #     GPU. gpu.hardware.GPU_COMPUTE_LOCK (acquired inside
                #     ParakeetEngine and PyannoteEngine) serializes the actual
                #     compute in that case, so this branch still launches both
                #     threads for pipelining, but the on-GPU work itself does
                #     not overlap — trading some wall-clock speed for
                #     correctness (no simultaneous VRAM allocation → no OOM).
                diar_config = self._config.diarization
                hf_token: str | None = getattr(
                    getattr(self._registry, "_settings", None), "hf_token", None
                )

                t0 = time.monotonic()
                chunks = split_audio(
                    audio.path,
                    n_workers=diar_config.num_workers,
                    overlap_seconds=diar_config.chunk_overlap_seconds,
                    out_dir=audio.path.parent,
                )

                if len(chunks) == 1:
                    # Audio is short relative to the overlap window — running
                    # subprocesses would just reload models from scratch.
                    # Use the hot registry already loaded in this process.
                    #
                    # CONCURRENCY NOTE: on Apple Silicon (WhisperX/MLX backend), both
                    # tasks share the MPS device; MPS serialises Metal kernels
                    # internally, so alignment and diarization do not truly overlap
                    # on GPU there. On CUDA (Parakeet backend), GPU_COMPUTE_LOCK
                    # (see gpu/hardware.py) provides the same serialization
                    # explicitly, only when pyannote was placed on CUDA — see the
                    # comment above this branch for the full device matrix.
                    t_parallel_start = time.monotonic()
                    logger.info(
                        f"Parallel branch (1 chunk): audio={audio.duration:.0f}s — "
                        f"launching transcription+diarization threads"
                    )
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        t_fut = pool.submit(
                            self._transcription_engine.transcribe,
                            audio,
                            self._config.transcription,
                            on_progress,
                            _skip_alignment,
                        )
                        d_fut = pool.submit(
                            self._diarization_engine.diarize,
                            audio,
                            diar_config,
                            on_progress,
                        )
                        transcription = t_fut.result()
                        t_transcription_done = time.monotonic()
                        diarization = d_fut.result()
                        t_diarization_done = time.monotonic()
                    logger.info(
                        f"Parallel wall: {t_diarization_done - t_parallel_start:.1f}s | "
                        f"transcription finished at +{t_transcription_done - t_parallel_start:.1f}s | "
                        f"diarization finished at +{t_diarization_done - t_parallel_start:.1f}s"
                    )
                else:
                    # Long audio: N subprocess workers diarize chunks on CPU
                    # (diarize_parallel always forces pipeline.to("cpu") in its
                    # workers) while the ASR backend transcribes on GPU/CPU in
                    # the main thread — genuinely parallel, no GPU contention.
                    t_parallel_start = time.monotonic()
                    logger.info(
                        f"Parallel branch ({len(chunks)} chunks): audio={audio.duration:.0f}s — "
                        f"launching parallel diarizer + transcription"
                    )
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=1
                    ) as diar_pool:
                        diar_future = diar_pool.submit(
                            diarize_parallel,
                            chunks,
                            diar_config,
                            hf_token,
                            on_progress,
                        )
                        transcription = self._transcription_engine.transcribe(
                            audio,
                            self._config.transcription,
                            on_progress,
                            _skip_alignment,
                        )
                        t_transcription_done = time.monotonic()
                        diarization = diar_future.result()
                        t_diarization_done = time.monotonic()
                    logger.info(
                        f"Parallel wall: {t_diarization_done - t_parallel_start:.1f}s | "
                        f"transcription finished at +{t_transcription_done - t_parallel_start:.1f}s | "
                        f"diarization finished at +{t_diarization_done - t_parallel_start:.1f}s"
                    )

                cleanup_chunks(chunks, audio.path)
                elapsed = time.monotonic() - t0
                timings["transcription"] = elapsed  # wall-clock (overlapped)
                timings["diarization"] = 0.0  # included in above
                logger.info(
                    f"Parallel transcription+diarization wall time: {elapsed:.1f}s "
                    f"(chunks={len(chunks)})"
                )
            else:
                t0 = time.monotonic()
                transcription = self._transcription_engine.transcribe(
                    audio,
                    self._config.transcription,
                    on_progress=on_progress,
                    skip_alignment=_skip_alignment,
                )
                timings["transcription"] = time.monotonic() - t0
                diarization = DiarizationResult(
                    segments=(
                        SpeakerSegment(
                            speaker_id="SPEAKER_00",
                            start=0.0,
                            end=audio.duration,
                        ),
                    ),
                    num_speakers=1,
                )
                timings["diarization"] = 0.0
                if _decision is not None and not _decision.use_diarization:
                    warnings.append(
                        f"Diarization skipped: {_decision.reasons[0] if _decision.reasons else 'single speaker detected'}"
                    )
                else:
                    warnings.append(
                        "Diarization disabled — all audio attributed to Speaker 1."
                    )

            # ── Diagnostic: stage word counts ────────────────────────────────
            _wc("transcription", transcription)
            logger.info(
                "STAGE_TRANSCRIPTION: words=%d, language=%s, duration=%.1fs, "
                "segments_raw=%d",
                len(transcription.words),
                getattr(transcription, "language", "?"),
                getattr(transcription, "duration", 0.0),
                0,  # raw segment count not available here; see whisperx_engine logs
            )
            logger.info(
                "STAGE_DIARIZATION: segments=%d, speakers=%d, enabled=%s",
                len(diarization.segments),
                diarization.num_speakers,
                _use_diarization,
            )
            if diarization.segments:
                first = diarization.segments[0]
                last = diarization.segments[-1]
                logger.info(
                    "STAGE_DIARIZATION_RANGE: %.1fs → %.1fs",
                    first.start, last.end,
                )

            _progress(on_progress, "merge", 0.0, "Merging speakers and words")
            t0 = time.monotonic()
            transcript = self._merger.merge(transcription, diarization, audio)
            timings["merge"] = time.monotonic() - t0
            _wc("merge", transcript)
            logger.info(
                "STAGE_MERGE: words=%d, segments=%d",
                transcript.word_count, len(transcript.segments),
            )

            _progress(on_progress, "process", 0.0, "Running processors")
            t0 = time.monotonic()
            ctx = ProcessorContext(profile_name=self._profile.name)
            transcript = self._run_processors(transcript, ctx)
            timings["processing"] = time.monotonic() - t0
            _wc("processors", transcript)
            logger.info(
                "STAGE_PROCESSORS: words=%d, segments=%d, corrections=%d",
                transcript.word_count, len(transcript.segments), len(ctx.corrections),
            )

            _progress(on_progress, "export_prep", 1.0, "Done")

        elapsed = time.monotonic() - pipeline_start
        timings["total"] = elapsed

        self._log_summary(audio.duration, timings, transcript, len(ctx.corrections))

        return PipelineResult(
            transcript=transcript,
            audio=audio,
            elapsed_seconds=elapsed,
            warnings=warnings,
            corrections=ctx.corrections,
        )

    def _run_processors(
        self, transcript: Transcript, ctx: ProcessorContext
    ) -> Transcript:
        for processor in self._processors:
            words_before = transcript.word_count
            t0 = time.monotonic()
            transcript = processor.process(transcript, ctx)
            elapsed = time.monotonic() - t0
            words_after = transcript.word_count
            delta = words_after - words_before
            logger.info(
                "Processor '%s' done in %.2fs | words: %d → %d (%+d)",
                processor.name, elapsed, words_before, words_after, delta,
            )
            if words_before > 0 and words_after < words_before * 0.95:
                logger.warning(
                    "WORD_LOSS in processor '%s': %d → %d (%.1f%% lost)",
                    processor.name, words_before, words_after,
                    100 * (words_before - words_after) / words_before,
                )
        return transcript

    def _build_processors(self) -> list[Any]:
        processor_names = self._profile.processors

        conf_threshold = self._config.processing.intelligence_confidence_threshold
        builtin_map: dict[str, Any] = {
            "vocabulary_correction": lambda: VocabularyCorrectionProcessor(
                self._profile
            ),
            "context_correction": lambda: ContextCorrectionProcessor(),
            "transcript_cleanup": lambda: TranscriptCleanupProcessor(),
            "transcript_intelligence": lambda: TranscriptIntelligenceProcessor(
                conf_threshold
            ),
            "speaker_formatting": lambda: SpeakerFormattingProcessor(),
            "ai_grammar": lambda: AIGrammarProcessor(),
        }

        result = []
        for name in processor_names:
            if name in builtin_map:
                result.append(builtin_map[name]())
            else:
                try:
                    cls = get_processor(name)
                    result.append(cls())
                except KeyError:
                    logger.warning(f"Processor '{name}' not found — skipping.")

        return result

    def _log_summary(
        self,
        audio_duration: float,
        timings: dict[str, float],
        transcript: Transcript,
        num_corrections: int,
    ) -> None:
        rtf = audio_duration / timings["total"] if timings["total"] > 0 else 0
        logger.info(
            f"Pipeline complete | "
            f"audio={audio_duration:.0f}s | "
            f"total={timings['total']:.1f}s | "
            f"RTF={rtf:.1f}x | "
            f"words={transcript.word_count} | "
            f"speakers={len(transcript.speakers)} | "
            f"segments={len(transcript.segments)} | "
            f"corrections={num_corrections}"
        )
        logger.debug(
            "Stage timings: "
            + " | ".join(f"{k}={v:.2f}s" for k, v in timings.items() if k != "total")
        )


def _load_profile_for_config(config: PipelineConfig) -> Profile:
    """Load the profile specified in ProcessingConfig, with a clear error on missing."""
    try:
        return load_profile(
            config.processing.profile,
            profiles_dir=config.processing.profiles_dir,
        )
    except FileNotFoundError:
        logger.warning(
            f"Profile '{config.processing.profile}' not found in "
            f"{config.processing.profiles_dir} — falling back to empty generic profile."
        )
        return Profile(
            name="generic",
            processors=[
                "context_correction",
                "transcript_cleanup",
                "speaker_formatting",
            ],
        )


def _progress(
    callback: ProgressCallback | None,
    stage: str,
    fraction_within_stage: float,
    message: str,
) -> None:
    if callback is None:
        return
    start, end = _STAGE_WEIGHTS.get(stage, (0.0, 1.0))
    overall = start + (end - start) * fraction_within_stage
    callback(overall, message)
