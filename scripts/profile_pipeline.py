#!/usr/bin/env python3
"""
Transcript Engine — Pipeline Performance Profiler

Instruments every stage without modifying production code.
Subclasses the key engine classes to add precise per-stage timing.

Usage:
    python scripts/profile_pipeline.py <audio_file> [--profile generic|banking]

Output:
    - Human-readable report printed to stdout
    - performance.json saved in project root
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ── Project root ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Timing collector ─────────────────────────────────────────────────────────


class Timings:
    """Thread-safe ordered accumulator for stage elapsed seconds."""

    def __init__(self) -> None:
        self._data: OrderedDict[str, float] = OrderedDict()
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = time.perf_counter() - self._starts.pop(name, time.perf_counter())
        self._data[name] = elapsed
        return elapsed

    def record(self, name: str, elapsed: float) -> None:
        self._data[name] = elapsed

    def get(self, name: str) -> float:
        return self._data.get(name, 0.0)

    def items(self) -> list[tuple[str, float]]:
        return list(self._data.items())


T = Timings()


# ── Profiling subclasses ─────────────────────────────────────────────────────


class ProfilingWhisperEngine:
    """Wraps WhisperXEngine to measure model load, transcription, and alignment."""

    def __init__(self, registry: Any) -> None:
        from transcript_engine.transcription.whisperx_engine import WhisperXEngine

        self._inner = WhisperXEngine(registry)

    def transcribe(self, audio: Any, config: Any, on_progress: Any = None) -> Any:
        import whisperx

        device = self._inner._registry._resolve_ctranslate_device(config.device)

        # ── Model load ──────────────────────────────────────────────────────
        T.start("whisper_model_load")
        model = self._inner._registry.get_whisper_model(config)
        T.stop("whisper_model_load")

        if on_progress:
            on_progress(0.18, f"Transcribing {audio.original_path.name}")

        effective_batch = config.batch_size if device == "cuda" else None

        # ── CT2 inference ───────────────────────────────────────────────────
        T.start("whisper_transcribe")
        try:
            result = model.transcribe(
                str(audio.path),
                batch_size=effective_batch,
                language=config.language,
                chunk_size=config.chunk_length,
            )
        except (IndexError, StopIteration):
            T.stop("whisper_transcribe")
            from transcript_engine.models.pipeline import RawTranscription

            return RawTranscription(words=(), language="en", duration=audio.duration)
        T.stop("whisper_transcribe")

        detected_language = result.get("language", "en")

        if on_progress:
            on_progress(0.45, f"Aligning timestamps (language={detected_language})")

        # ── Alignment model load ────────────────────────────────────────────
        T.start("alignment_model_load")
        align_model, align_metadata = self._inner._registry.get_alignment_model(
            detected_language, config
        )
        T.stop("alignment_model_load")

        torch_device = self._inner._registry._resolve_torch_device(config.device)

        # ── Alignment inference ─────────────────────────────────────────────
        T.start("alignment")
        aligned = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            str(audio.path),
            torch_device,
            return_char_alignments=False,
        )
        T.stop("alignment")

        words = self._inner._extract_words(aligned)

        if on_progress:
            on_progress(0.60, "Transcription complete")

        from transcript_engine.models.pipeline import RawTranscription

        return RawTranscription(
            words=tuple(words),
            language=detected_language,
            duration=audio.duration,
        )


class ProfilingDiarizationEngine:
    """Wraps PyannoteEngine to measure model load vs inference."""

    def __init__(self, registry: Any) -> None:
        from transcript_engine.diarization.pyannote_engine import PyannoteEngine

        self._inner = PyannoteEngine(registry)

    def diarize(self, audio: Any, config: Any, on_progress: Any = None) -> Any:
        if on_progress:
            on_progress(0.62, "Loading diarization model")

        T.start("diarization_model_load")
        pipeline = self._inner._registry.get_diarization_pipeline(config)
        T.stop("diarization_model_load")

        # Apply segmentation step
        seg_inf = pipeline._segmentation
        seg_inf.step = config.segmentation_step * float(seg_inf.duration)

        if on_progress:
            on_progress(0.65, "Running speaker diarization")

        pipeline_kwargs: dict[str, int] = {}
        if config.min_speakers is not None:
            pipeline_kwargs["min_speakers"] = config.min_speakers
        if config.max_speakers is not None:
            pipeline_kwargs["max_speakers"] = config.max_speakers

        T.start("diarization_inference")
        diarization = pipeline(str(audio.path), **pipeline_kwargs)
        T.stop("diarization_inference")

        segments = self._inner._extract_segments(diarization)
        num_speakers = len({s.speaker_id for s in segments})

        if on_progress:
            on_progress(0.90, "Diarization complete")

        from transcript_engine.models.pipeline import DiarizationResult

        return DiarizationResult(
            segments=tuple(segments),
            num_speakers=num_speakers,
        )


class ProfilingIntelligenceEngine:
    """Wraps MeetingIntelligenceEngine to time each extractor."""

    def analyze(self, transcript: Any, profile_name: str = "generic") -> Any:
        from transcript_engine.intelligence.base import IntelligenceContext
        from transcript_engine.intelligence.extractors.action_items import (
            ActionItemExtractor,
        )
        from transcript_engine.intelligence.extractors.decisions import (
            DecisionExtractor,
        )
        from transcript_engine.intelligence.extractors.entities import EntityExtractor
        from transcript_engine.intelligence.extractors.questions import (
            QuestionExtractor,
        )
        from transcript_engine.intelligence.extractors.summary import SummaryGenerator
        from transcript_engine.intelligence.extractors.timeline import TimelineExtractor
        from transcript_engine.intelligence.extractors.topics import TopicExtractor
        from transcript_engine.intelligence.models import IntelligenceResult

        ctx = IntelligenceContext(profile_name=profile_name, ai_backend=None)

        extractors: list[tuple[str, Any]] = [
            (
                "intel_action_items",
                lambda: ActionItemExtractor().extract(transcript, ctx),
            ),
            ("intel_decisions", lambda: DecisionExtractor().extract(transcript, ctx)),
            ("intel_questions", lambda: QuestionExtractor().extract(transcript, ctx)),
            ("intel_entities", lambda: EntityExtractor().extract(transcript, ctx)),
            ("intel_topics", lambda: TopicExtractor().extract(transcript, ctx)),
        ]

        results: dict[str, Any] = {}
        for key, fn in extractors:
            T.start(key)
            results[key] = fn()
            T.stop(key)

        T.start("intel_timeline")
        results["intel_timeline"] = TimelineExtractor().extract(
            transcript,
            ctx,
            decisions=results["intel_decisions"],
            action_items=results["intel_action_items"],
            topics=results["intel_topics"],
        )
        T.stop("intel_timeline")

        T.start("intel_summary")
        results["intel_summary"] = SummaryGenerator().extract(
            transcript,
            ctx,
            decisions=results["intel_decisions"],
            action_items=results["intel_action_items"],
            entities=results["intel_entities"],
        )
        T.stop("intel_summary")

        return IntelligenceResult(
            summary_bullets=results["intel_summary"],
            action_items=results["intel_action_items"],
            decisions=results["intel_decisions"],
            questions=results["intel_questions"],
            timeline=results["intel_timeline"],
            topics=results["intel_topics"],
            entities=results["intel_entities"],
        )


# ── Profiling pipeline ────────────────────────────────────────────────────────


def _run_profiling_pipeline(
    audio_path: Path,
    profile_name: str,
    beam_size: int | None = None,
    seg_step: float | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    _extra_diar: dict[str, Any] = {}
    if seg_step is not None:
        _extra_diar["segmentation_step"] = seg_step
    if num_workers is not None:
        _extra_diar["num_workers"] = num_workers
    import concurrent.futures

    from transcript_engine.audio.chunker import cleanup_chunks, split_audio
    from transcript_engine.audio.preprocessor import AudioPreprocessor
    from transcript_engine.config.loader import load_settings
    from transcript_engine.exporters.corrections import export_corrections
    from transcript_engine.exporters.quality import export_quality
    from transcript_engine.exporters.registry import get_exporter
    from transcript_engine.exporters.stats import export_stats
    from transcript_engine.intelligence.exports import (
        export_action_items,
        export_decisions,
        export_entities,
        export_metrics,
        export_questions,
        export_summary,
        export_timeline,
    )
    from transcript_engine.merger.merger import TranscriptMerger
    from transcript_engine.model_registry.registry import ModelRegistry
    from transcript_engine.models.pipeline import DiarizationResult, SpeakerSegment
    from transcript_engine.pipeline.parallel_diarizer import diarize_parallel
    from transcript_engine.processors.base import ProcessorContext
    from transcript_engine.processors.cleanup import TranscriptCleanupProcessor
    from transcript_engine.processors.context_correction import (
        ContextCorrectionProcessor,
    )
    from transcript_engine.processors.speaker_formatting import (
        SpeakerFormattingProcessor,
    )
    from transcript_engine.processors.vocabulary import VocabularyCorrectionProcessor
    from transcript_engine.profiles.loader import load_profile
    from transcript_engine.review.index import generate_index
    from transcript_engine.review.report import generate_review

    settings = load_settings()
    if beam_size is not None or _extra_diar is not None:
        diar_update = {}
        if _extra_diar:
            diar_update.update(_extra_diar)
        settings = settings.model_copy(
            update={
                "pipeline": settings.pipeline.model_copy(
                    update={
                        "transcription": (
                            settings.pipeline.transcription.model_copy(
                                update={"beam_size": beam_size}
                            )
                            if beam_size is not None
                            else settings.pipeline.transcription
                        ),
                        "diarization": (
                            settings.pipeline.diarization.model_copy(update=diar_update)
                            if diar_update
                            else settings.pipeline.diarization
                        ),
                    }
                )
            }
        )
    profile = load_profile(
        profile_name, profiles_dir=settings.pipeline.processing.profiles_dir
    )

    # Fresh registry — models loaded cold (first real-world conditions)
    registry = ModelRegistry(settings)

    whisper_engine = ProfilingWhisperEngine(registry)
    diar_engine = ProfilingDiarizationEngine(registry)
    merger = TranscriptMerger()

    cfg = settings.pipeline
    diar_cfg = cfg.diarization.model_copy(update={"enabled": bool(settings.hf_token)})
    t_cfg = cfg.transcription

    output_dir = Path("outputs") / "profiler_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_log: list[tuple[float, float, str]] = []

    def on_progress(fraction: float, message: str) -> None:
        progress_log.append((time.perf_counter(), fraction, message))

    meta: dict[str, Any] = {}

    # ── 1. PREPROCESS ─────────────────────────────────────────────────────────
    # Keep the preprocessor context open for the entire run — it owns the
    # temp WAV file that split_audio and whisperx need.
    preprocessor = AudioPreprocessor(cfg.audio)
    preprocessor.__enter__()
    try:
        T.start("preprocess")
        audio = preprocessor.prepare(audio_path)
        T.stop("preprocess")
    except Exception:
        preprocessor.__exit__(*sys.exc_info())
        raise
    meta["audio_duration_s"] = audio.duration
    meta["audio_file"] = audio_path.name

    if diar_cfg.enabled:
        # ── 2. SPLIT AUDIO ────────────────────────────────────────────────────
        T.start("audio_split")
        chunks = split_audio(
            audio.path,
            n_workers=diar_cfg.num_workers,
            overlap_seconds=diar_cfg.chunk_overlap_seconds,
            out_dir=audio.path.parent,
        )
        T.stop("audio_split")
        meta["chunks"] = len(chunks)

        if len(chunks) == 1:
            # Short audio: run transcription+diarization in parallel threads
            T.start("parallel_wall")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                t_fut = pool.submit(
                    whisper_engine.transcribe, audio, t_cfg, on_progress
                )
                d_fut = pool.submit(diar_engine.diarize, audio, diar_cfg, on_progress)
                transcription = t_fut.result()
                diarization = d_fut.result()
            T.stop("parallel_wall")
        else:
            # Long audio: diarize chunks in subprocess workers, transcribe in main thread
            T.start("diarization_parallel_wall")
            hf_token: str | None = getattr(
                getattr(registry, "_settings", None), "hf_token", None
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as diar_pool:
                diar_future = diar_pool.submit(
                    diarize_parallel, chunks, diar_cfg, hf_token, on_progress
                )
                transcription = whisper_engine.transcribe(audio, t_cfg, on_progress)
                diarization = diar_future.result()
            T.stop("diarization_parallel_wall")

        cleanup_chunks(chunks, audio.path)
    else:
        transcription = whisper_engine.transcribe(audio, t_cfg, on_progress)
        diarization = DiarizationResult(
            segments=(
                SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=audio.duration),
            ),
            num_speakers=1,
        )

    # ── 3. MERGE ──────────────────────────────────────────────────────────────
    T.start("merge")
    transcript = merger.merge(transcription, diarization, audio)
    T.stop("merge")

    # ── 4. PROCESSORS ─────────────────────────────────────────────────────────
    ctx = ProcessorContext(profile_name=profile.name)
    processor_map: dict[str, Any] = {
        "vocabulary_correction": lambda: VocabularyCorrectionProcessor(profile),
        "context_correction": lambda: ContextCorrectionProcessor(),
        "transcript_cleanup": lambda: TranscriptCleanupProcessor(),
        "speaker_formatting": lambda: SpeakerFormattingProcessor(),
    }
    for proc_name in profile.processors:
        if proc_name not in processor_map:
            continue
        proc: Any = processor_map[proc_name]()
        key = f"proc_{proc_name}"
        T.start(key)
        transcript = proc.process(transcript, ctx)
        T.stop(key)

    # ── 5. MEETING INTELLIGENCE ───────────────────────────────────────────────
    T.start("intelligence_total")
    intel_result = ProfilingIntelligenceEngine().analyze(
        transcript, profile_name=profile.name
    )
    T.stop("intelligence_total")

    # ── 6. EXPORT ─────────────────────────────────────────────────────────────
    stem = "transcript"
    conf_threshold = 0.65
    export_cfg = cfg.export.model_copy(
        update={
            "output_dir": output_dir,
            "formats": ["markdown", "json"],
        }
    )

    T.start("export_files")
    for fmt in ["markdown", "json"]:
        try:
            exporter_cls = get_exporter(fmt)
            content = exporter_cls().export(transcript, export_cfg)
            (output_dir / f"{stem}{exporter_cls.extension}").write_bytes(content)
        except Exception:
            pass

    if ctx.corrections:
        (output_dir / f"{stem}.corrections.md").write_text(
            export_corrections(ctx.corrections, profile.name, audio_path.name),
            encoding="utf-8",
        )
    (output_dir / f"{stem}.stats.json").write_text(
        export_stats(
            transcript,
            T.get("preprocess"),
            ctx.corrections,
            profile.name,
            low_conf_threshold=conf_threshold,
        ),
        encoding="utf-8",
    )
    (output_dir / f"{stem}.quality.md").write_text(
        export_quality(
            transcript, ctx.corrections, profile.name, low_conf_threshold=conf_threshold
        ),
        encoding="utf-8",
    )
    (output_dir / f"{stem}.summary.md").write_text(
        export_summary(intel_result, transcript), encoding="utf-8"
    )
    (output_dir / f"{stem}.action_items.md").write_text(
        export_action_items(intel_result, transcript), encoding="utf-8"
    )
    (output_dir / f"{stem}.decisions.md").write_text(
        export_decisions(intel_result, transcript), encoding="utf-8"
    )
    (output_dir / f"{stem}.questions.md").write_text(
        export_questions(intel_result, transcript), encoding="utf-8"
    )
    (output_dir / f"{stem}.timeline.md").write_text(
        export_timeline(intel_result, transcript), encoding="utf-8"
    )
    (output_dir / f"{stem}.entities.json").write_text(
        export_entities(intel_result), encoding="utf-8"
    )
    metrics_json = export_metrics(
        transcript,
        0.0,
        ctx.corrections,
        profile.name,
        intel_result,
        low_conf_threshold=conf_threshold,
    )
    (output_dir / f"{stem}.metrics.json").write_text(metrics_json, encoding="utf-8")
    (output_dir / f"{stem}.review.md").write_text(
        generate_review(transcript, ctx.corrections, low_conf_threshold=conf_threshold),
        encoding="utf-8",
    )
    index_data = generate_index(transcript, intel_result, ctx.corrections, profile.name)
    (output_dir / f"{stem}.index.json").write_text(
        json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    T.stop("export_files")

    # ── 7. BUNDLE ─────────────────────────────────────────────────────────────
    T.start("bundle")
    bundle_path = output_dir / "transcript_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(output_dir.iterdir()):
            if (
                f.suffix in {".md", ".json", ".txt", ".srt", ".vtt"}
                and f != bundle_path
            ):
                zf.write(f, f.name)
    T.stop("bundle")

    preprocessor.__exit__(None, None, None)
    return meta


# ── Report formatter ─────────────────────────────────────────────────────────


def _fmt_time(s: float) -> str:
    if s < 0.01:
        return "<0.01s"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s"


def _pct(elapsed: float, total: float) -> str:
    if total <= 0:
        return "—"
    return f"{100 * elapsed / total:.1f}%"


def build_report(meta: dict[str, Any]) -> dict[str, Any]:
    audio_s = meta["audio_duration_s"]
    timings = dict(T.items())

    # Sum whisper-related stages
    whisper_total = (
        timings.get("whisper_model_load", 0)
        + timings.get("whisper_transcribe", 0)
        + timings.get("alignment_model_load", 0)
        + timings.get("alignment", 0)
    )

    # Diarization: use subprocess parallel wall if available, else in-process
    diar_wall = timings.get("diarization_parallel_wall", 0.0)
    diar_model = timings.get("diarization_model_load", 0.0)
    diar_inf = timings.get("diarization_inference", 0.0)

    # Sum processor stages
    proc_stages = {k: v for k, v in timings.items() if k.startswith("proc_")}
    proc_total = sum(proc_stages.values())

    # Sum intelligence stages
    intel_stages = {k: v for k, v in timings.items() if k.startswith("intel_")}
    intel_total = timings.get("intelligence_total", sum(intel_stages.values()))

    # Compute grand total (wall clock from preprocess to bundle)
    # Use "parallel_wall" or "diarization_parallel_wall" as the transcription+diarization block
    parallel_key = (
        "diarization_parallel_wall"
        if "diarization_parallel_wall" in timings
        else "parallel_wall"
    )
    transcribe_block = timings.get(parallel_key, whisper_total)

    total = (
        timings.get("preprocess", 0)
        + timings.get("audio_split", 0)
        + transcribe_block
        + timings.get("merge", 0)
        + proc_total
        + intel_total
        + timings.get("export_files", 0)
        + timings.get("bundle", 0)
    )

    rtf = audio_s / total if total > 0 else 0.0

    return {
        "audio_file": meta["audio_file"],
        "audio_duration_s": round(audio_s, 1),
        "beam_size": meta.get("beam_size", 5),
        "total_s": round(total, 1),
        "rtf": round(rtf, 2),
        "chunks": meta.get("chunks", 1),
        "stages": {
            "preprocess": {
                "elapsed_s": round(timings.get("preprocess", 0), 2),
                "pct": (
                    round(100 * timings.get("preprocess", 0) / total, 1) if total else 0
                ),
            },
            "audio_split": {
                "elapsed_s": round(timings.get("audio_split", 0), 2),
                "pct": (
                    round(100 * timings.get("audio_split", 0) / total, 1)
                    if total
                    else 0
                ),
            },
            "whisper_model_load": {
                "elapsed_s": round(timings.get("whisper_model_load", 0), 2),
                "pct": (
                    round(100 * timings.get("whisper_model_load", 0) / total, 1)
                    if total
                    else 0
                ),
                "note": "cold load (first run); ~0s on warm API",
            },
            "whisper_transcribe": {
                "elapsed_s": round(timings.get("whisper_transcribe", 0), 2),
                "pct": (
                    round(100 * timings.get("whisper_transcribe", 0) / total, 1)
                    if total
                    else 0
                ),
            },
            "alignment_model_load": {
                "elapsed_s": round(timings.get("alignment_model_load", 0), 2),
                "pct": (
                    round(100 * timings.get("alignment_model_load", 0) / total, 1)
                    if total
                    else 0
                ),
                "note": "cold load (first run); ~0s on warm API",
            },
            "alignment": {
                "elapsed_s": round(timings.get("alignment", 0), 2),
                "pct": (
                    round(100 * timings.get("alignment", 0) / total, 1) if total else 0
                ),
            },
            "diarization_wall": {
                "elapsed_s": round(diar_wall, 2),
                "model_load_s": round(diar_model, 2),
                "inference_s": round(diar_inf, 2),
                "pct": round(100 * diar_wall / total, 1) if total else 0,
                "note": f"{meta.get('chunks', 1)} subprocess worker(s), overlaps with whisper",
            },
            "whisper_total": {
                "elapsed_s": round(whisper_total, 2),
                "pct": round(100 * whisper_total / total, 1) if total else 0,
                "note": "model_load + transcribe + alignment_load + alignment",
            },
            "transcribe_diarize_block": {
                "elapsed_s": round(transcribe_block, 2),
                "pct": round(100 * transcribe_block / total, 1) if total else 0,
                "note": "wall-clock (transcription overlaps diarization in parallel)",
            },
            "merge": {
                "elapsed_s": round(timings.get("merge", 0), 2),
                "pct": round(100 * timings.get("merge", 0) / total, 1) if total else 0,
            },
            "processors": {
                "total_s": round(proc_total, 2),
                "pct": round(100 * proc_total / total, 1) if total else 0,
                "breakdown": {
                    k.replace("proc_", ""): round(v, 3)
                    for k, v in sorted(proc_stages.items(), key=lambda x: -x[1])
                },
            },
            "intelligence": {
                "total_s": round(intel_total, 2),
                "pct": round(100 * intel_total / total, 1) if total else 0,
                "breakdown": {
                    k.replace("intel_", ""): round(v, 3)
                    for k, v in sorted(intel_stages.items(), key=lambda x: -x[1])
                    if k != "intelligence_total"
                },
            },
            "export_files": {
                "elapsed_s": round(timings.get("export_files", 0), 2),
                "pct": (
                    round(100 * timings.get("export_files", 0) / total, 1)
                    if total
                    else 0
                ),
            },
            "bundle": {
                "elapsed_s": round(timings.get("bundle", 0), 2),
                "pct": round(100 * timings.get("bundle", 0) / total, 1) if total else 0,
            },
        },
    }


def print_report(report: dict[str, Any]) -> None:
    total = report["total_s"]
    audio = report["audio_duration_s"]
    rtf = report["rtf"]
    s = report["stages"]

    w = 58
    print("\n" + "─" * w)
    print(f"  Performance Report — {report['audio_file']}")
    print(
        f"  Audio: {_fmt_time(audio)}  |  Total: {_fmt_time(total)}  |  RTF: {rtf:.2f}x"
    )
    print("─" * w)
    print(f"  {'Stage':<36} {'Time':>8}  {'%':>6}")
    print("─" * w)

    rows = [
        ("Preprocess (ffmpeg/resample)", s["preprocess"]["elapsed_s"]),
        ("Audio split (chunk prep)", s["audio_split"]["elapsed_s"]),
        ("  Whisper model load", s["whisper_model_load"]["elapsed_s"]),
        ("  Whisper CT2 transcription", s["whisper_transcribe"]["elapsed_s"]),
        ("  Alignment model load", s["alignment_model_load"]["elapsed_s"]),
        ("  Alignment (wav2vec2)", s["alignment"]["elapsed_s"]),
        (
            f"  Diarization wall ({report['chunks']}x subprocess)",
            s["diarization_wall"]["elapsed_s"],
        ),
        (" ──Transcribe+Diarize block──", s["transcribe_diarize_block"]["elapsed_s"]),
        ("Merge (speakers + words)", s["merge"]["elapsed_s"]),
    ]
    for proc, t in s["processors"]["breakdown"].items():
        rows.append((f"  Processor: {proc}", t))
    rows.append(("Processors total", s["processors"]["total_s"]))
    for name, t in s["intelligence"]["breakdown"].items():
        rows.append((f"  Intelligence: {name}", t))
    rows.append(("Meeting Intelligence total", s["intelligence"]["total_s"]))
    rows.append(("Export (14 files)", s["export_files"]["elapsed_s"]))
    rows.append(("Bundle (zip)", s["bundle"]["elapsed_s"]))

    for label, elapsed in rows:
        bar = _pct(elapsed, total)
        print(f"  {label:<36} {_fmt_time(elapsed):>8}  {bar:>6}")

    print("─" * w)
    print(f"  {'TOTAL':<36} {_fmt_time(total):>8}  {'100%':>6}")
    print(
        f"  RTF: {rtf:.2f}x  ({_fmt_time(audio)} audio processed in {_fmt_time(total)})"
    )
    print("─" * w)

    # Bottleneck analysis
    print("\n  BOTTLENECK ANALYSIS")
    print("  " + "─" * 54)
    stages_by_pct = sorted(
        [
            (k, v["elapsed_s"] if isinstance(v, dict) and "elapsed_s" in v else 0)
            for k, v in s.items()
        ],
        key=lambda x: -x[1],
    )
    for rank, (name, elapsed) in enumerate(stages_by_pct[:5], 1):
        if elapsed < 0.1:
            break
        bar = "█" * int(50 * elapsed / total)
        print(f"  #{rank} {name:<32} {_pct(elapsed, total):>6}  {bar}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the transcript-engine pipeline"
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--profile", default="generic", help="Profile name (default: generic)"
    )
    parser.add_argument(
        "--beam-size", type=int, default=None, help="Override beam_size (default: 5)"
    )
    parser.add_argument(
        "--seg-step",
        type=float,
        default=None,
        help="Override diarization segmentation_step (default: 0.5)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override diarization num_workers (default: 2)",
    )
    parser.add_argument("--output", default="performance.json", help="Output JSON path")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    params = (
        ", ".join(
            filter(
                None,
                [
                    f"beam_size={args.beam_size}" if args.beam_size else None,
                    f"seg_step={args.seg_step}" if args.seg_step else None,
                    f"num_workers={args.num_workers}" if args.num_workers else None,
                ],
            )
        )
        or "defaults"
    )
    print(f"\nProfiling: {audio_path.name}  (profile={args.profile}, {params})")
    print("Running full pipeline with per-stage instrumentation...\n")

    wall_start = time.perf_counter()
    meta = _run_profiling_pipeline(
        audio_path,
        args.profile,
        beam_size=args.beam_size,
        seg_step=args.seg_step,
        num_workers=args.num_workers,
    )
    meta["wall_total_s"] = time.perf_counter() - wall_start
    meta["beam_size"] = args.beam_size or 5
    meta["seg_step"] = args.seg_step or 0.5
    meta["num_workers"] = args.num_workers or 2

    report = build_report(meta)
    print_report(report)

    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {out.resolve()}\n")


if __name__ == "__main__":
    main()
