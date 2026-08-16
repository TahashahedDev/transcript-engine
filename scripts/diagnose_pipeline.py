"""
Pipeline diagnostic script.

Run on a real audio file to verify word counts at every stage.
Prints raw Whisper output → aligned output → merged transcript → final transcript.

Usage:
    python scripts/diagnose_pipeline.py /path/to/meeting.wav [--mode balanced]

Expected output for a 60-minute meeting: 6000–12000 words at every stage.
If any stage loses >5% of words, the output highlights it as WORD_LOSS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _wc_label(label: str, count: int) -> str:
    bar = "▓" * min(count // 100, 40)
    return f"  {label:<35} {count:>6} words  {bar}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose pipeline stage by stage")
    parser.add_argument("audio", type=Path, help="Path to audio file")
    parser.add_argument("--mode", default="balanced",
                        choices=["fast", "balanced", "high_accuracy"],
                        help="Pipeline mode (default: balanced)")
    parser.add_argument("--profile", default="generic",
                        help="Profile name (default: generic)")
    parser.add_argument("--skip-diarization", action="store_true",
                        help="Skip diarization (faster for debugging)")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/te_diagnose"),
                        help="Directory to write intermediate outputs")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Transcript Engine Pipeline Diagnostic")
    print(f"{'='*60}")
    print(f"  Audio  : {args.audio}")
    print(f"  Mode   : {args.mode}")
    print(f"  Profile: {args.profile}")
    print(f"  Out    : {args.out_dir}")
    print(f"{'='*60}\n")

    from transcript_engine.audio.preprocessor import AudioPreprocessor
    from transcript_engine.config.loader import load_settings
    from transcript_engine.config.settings import PIPELINE_MODES
    from transcript_engine.model_registry.registry import ModelRegistry

    settings = load_settings()
    mode_spec = PIPELINE_MODES.get(args.mode, PIPELINE_MODES["balanced"])

    import os
    hf_token = os.environ.get("HF_TOKEN") or settings.hf_token
    use_diarization = not args.skip_diarization and bool(hf_token)

    # Keep preprocessor context open for the entire run — it manages a temp WAV
    # file that must exist for all stages (transcription, alignment, diarization).
    with AudioPreprocessor(settings.pipeline.audio) as prep:

        # ── Stage 1: Audio preprocessing ─────────────────────────────────────
        print("STAGE 1: Audio Preprocessing")
        t0 = time.monotonic()
        audio = prep.prepare(args.audio)
        print(f"  Duration  : {audio.duration:.1f}s ({audio.duration/60:.1f} min)")
        print(f"  Path      : {audio.path}")
        print(f"  Time      : {time.monotonic()-t0:.2f}s")
        print()

        # ── Stage 2: Whisper transcription (raw, before alignment) ───────────
        print("STAGE 2: Whisper Transcription (raw)")
        t0 = time.monotonic()

        import whisperx  # noqa

        tc = settings.pipeline.transcription.model_copy(update={
            "model_id": mode_spec.model_id,
            "beam_size": mode_spec.beam_size,
            "best_of": mode_spec.best_of,
            "temperatures": mode_spec.temperatures,
        })

        registry = ModelRegistry(settings)
        model = registry.get_whisper_model(tc)

        device = registry._resolve_ctranslate_device(tc.device)
        effective_batch = tc.batch_size if device == "cuda" else None

        raw_result = model.transcribe(
            str(audio.path),
            batch_size=effective_batch,
            language=tc.language,
            chunk_size=tc.chunk_length,
        )
        elapsed_transcribe = time.monotonic() - t0

        raw_segs = raw_result.get("segments", [])
        detected_lang = raw_result.get("language", "en")
        raw_words_approx = sum(len(s.get("text", "").split()) for s in raw_segs)
        raw_has_word_timestamps = any(s.get("words") for s in raw_segs)

        print(f"  Language  : {detected_lang}")
        print(f"  Segments  : {len(raw_segs)}")
        print(f"  Words     : ~{raw_words_approx} (from segment text)")
        print(f"  Word-ts   : {'YES' if raw_has_word_timestamps else 'NO — no word-level timestamps'}")
        print(f"  Time      : {elapsed_transcribe:.1f}s")
        print(f"  RTF       : {audio.duration/elapsed_transcribe:.2f}x")
        if raw_segs:
            first_seg = raw_segs[0]
            last_seg = raw_segs[-1]
            print(f"  Timestamps: [{first_seg.get('start',0):.1f}s → {last_seg.get('end', last_seg.get('start',0)):.1f}s]")
            print(f"  First text: {first_seg.get('text','')[:120]!r}")
            print(f"  Last text : {last_seg.get('text','')[:120]!r}")

        # Write raw output
        raw_out = args.out_dir / "01_raw_whisper.json"
        with raw_out.open("w") as f:
            json.dump({"language": detected_lang, "segments": [
                {"start": s.get("start"), "end": s.get("end"), "text": s.get("text"),
                 "words": s.get("words", [])}
                for s in raw_segs
            ]}, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {raw_out}")
        print()

        # ── Stage 3: Alignment ────────────────────────────────────────────────
        skip_alignment = mode_spec.skip_alignment
        aligned_segs = raw_segs
        aligned_word_count = raw_words_approx

        if skip_alignment:
            print("STAGE 3: Alignment SKIPPED (fast mode)")
            print(f"  Using segment-level timestamps, ~{aligned_word_count} words")
            print()
        else:
            print("STAGE 3: Wav2Vec2 Alignment")
            t0 = time.monotonic()
            try:
                align_model, align_meta = registry.get_alignment_model(detected_lang, tc)
                aligned = whisperx.align(
                    raw_segs, align_model, align_meta,
                    str(audio.path), "cpu",
                    return_char_alignments=False,
                )
                elapsed_align = time.monotonic() - t0
                aligned_segs = aligned.get("segments", [])
                aligned_word_count = sum(len(s.get("words", [])) for s in aligned_segs)
                print(f"  Segments  : {len(aligned_segs)}")
                print(f"  Words     : {aligned_word_count}")
                print(f"  Time      : {elapsed_align:.1f}s")
                if aligned_segs:
                    first_a = aligned_segs[0]
                    last_a = aligned_segs[-1]
                    print(f"  Timestamps: [{first_a.get('start',0):.1f}s → {last_a.get('end', last_a.get('start',0)):.1f}s]")
                aligned_out = args.out_dir / "02_aligned.json"
                with aligned_out.open("w") as f:
                    json.dump({"segments": aligned_segs}, f, indent=2, ensure_ascii=False)
                print(f"  Saved: {aligned_out}")
            except Exception as exc:
                print(f"  ALIGNMENT FAILED: {exc}")
                print("  Falling back to raw segments")
                aligned_segs = raw_segs
                aligned_word_count = raw_words_approx
            print()

        # ── Stage 4: Extract RawWords ─────────────────────────────────────────
        print("STAGE 4: Word Extraction")
        from transcript_engine.models.pipeline import RawTranscription
        from transcript_engine.transcription.whisperx_engine import WhisperXEngine

        engine = WhisperXEngine(registry)
        raw_words = engine._extract_words({"segments": aligned_segs})
        word_extract_count = len(raw_words)

        print(f"  Words extracted: {word_extract_count}")
        if raw_words:
            first_w = raw_words[0]
            last_w = raw_words[-1]
            print(f"  First word: {first_w.text!r} @ {first_w.start:.2f}s")
            print(f"  Last  word: {last_w.text!r} @ {last_w.end:.2f}s")
            print(f"  Timestamp range: {first_w.start:.1f}s → {last_w.end:.1f}s")
        print()

        raw_transcription = RawTranscription(
            words=tuple(raw_words),
            language=detected_lang,
            duration=audio.duration,
        )

        # ── Stage 5: Diarization ──────────────────────────────────────────────
        from transcript_engine.diarization.pyannote_engine import PyannoteEngine
        from transcript_engine.models.pipeline import DiarizationResult, SpeakerSegment

        if use_diarization:
            print("STAGE 5: Speaker Diarization")
            diar_engine = PyannoteEngine(registry)
            t0 = time.monotonic()
            try:
                diarization = diar_engine.diarize(audio, settings.pipeline.diarization)
                elapsed_diar = time.monotonic() - t0
                print(f"  Speakers  : {diarization.num_speakers}")
                print(f"  Segments  : {len(diarization.segments)}")
                print(f"  Time      : {elapsed_diar:.1f}s")
                if diarization.segments:
                    ds_first = diarization.segments[0]
                    ds_last = diarization.segments[-1]
                    print(f"  Range     : {ds_first.start:.1f}s → {ds_last.end:.1f}s")
                diar_out = args.out_dir / "03_diarization.json"
                with diar_out.open("w") as f:
                    json.dump({
                        "num_speakers": diarization.num_speakers,
                        "segments": [
                            {"speaker_id": s.speaker_id, "start": s.start, "end": s.end}
                            for s in diarization.segments
                        ]
                    }, f, indent=2)
                print(f"  Saved: {diar_out}")
            except Exception as exc:
                print(f"  DIARIZATION FAILED: {exc}")
                print("  Using single-speaker fallback")
                diarization = DiarizationResult(
                    segments=(SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=audio.duration),),
                    num_speakers=1,
                )
        else:
            print("STAGE 5: Diarization SKIPPED (--skip-diarization or no HF token)")
            diarization = DiarizationResult(
                segments=(SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=audio.duration),),
                num_speakers=1,
            )
        print()

        # ── Stage 6: Merge ────────────────────────────────────────────────────
        print("STAGE 6: Transcript Merge")
        from transcript_engine.merger.merger import TranscriptMerger

        t0 = time.monotonic()
        transcript = TranscriptMerger().merge(raw_transcription, diarization, audio)
        elapsed_merge = time.monotonic() - t0

        print(f"  Words     : {transcript.word_count}")
        print(f"  Segments  : {len(transcript.segments)}")
        print(f"  Speakers  : {len(transcript.speakers)}")
        print(f"  Time      : {elapsed_merge:.2f}s")
        if transcript.segments:
            first_s = transcript.segments[0]
            last_s = transcript.segments[-1]
            print(f"  Range     : {first_s.start:.1f}s → {last_s.end:.1f}s")
            print(f"  First seg : [{first_s.speaker_id}] {first_s.text[:100]!r}")
            print(f"  Last seg  : [{last_s.speaker_id}] {last_s.text[:100]!r}")
        print()

        # ── Stage 7: Processors ───────────────────────────────────────────────
        print("STAGE 7: Processors")
        from transcript_engine.processors.base import ProcessorContext
        from transcript_engine.processors.cleanup import TranscriptCleanupProcessor
        from transcript_engine.processors.context_correction import ContextCorrectionProcessor
        from transcript_engine.processors.intelligence import TranscriptIntelligenceProcessor
        from transcript_engine.processors.speaker_formatting import SpeakerFormattingProcessor
        from transcript_engine.processors.vocabulary import VocabularyCorrectionProcessor
        from transcript_engine.profiles.loader import load_profile

        profiles_dir = Path(settings.pipeline.processing.profiles_dir)
        try:
            profile = load_profile(args.profile, profiles_dir=profiles_dir)
        except FileNotFoundError:
            from transcript_engine.profiles.model import Profile
            profile = Profile(
                name="generic",
                processors=["context_correction", "transcript_cleanup", "speaker_formatting"],
            )

        ctx = ProcessorContext(profile_name=profile.name)
        words_before_processors = transcript.word_count

        conf_threshold = settings.pipeline.processing.intelligence_confidence_threshold
        processor_map = {
            "vocabulary_correction": lambda: VocabularyCorrectionProcessor(profile),
            "context_correction": ContextCorrectionProcessor,
            "transcript_cleanup": TranscriptCleanupProcessor,
            "transcript_intelligence": lambda: TranscriptIntelligenceProcessor(conf_threshold),
            "speaker_formatting": SpeakerFormattingProcessor,
        }

        for proc_name in mode_spec.processors:
            factory = processor_map.get(proc_name)
            if factory:
                t0 = time.monotonic()
                proc: Any = factory() if callable(factory) else factory
                before = transcript.word_count
                transcript = proc.process(transcript, ctx)
                after = transcript.word_count
                delta = after - before
                elapsed_proc = time.monotonic() - t0
                warn = " *** WORD_LOSS ***" if before > 0 and after < before * 0.95 else ""
                print(f"  {proc_name:<35} {before:>6} → {after:>6} ({delta:+d})  {elapsed_proc:.1f}s{warn}")
            else:
                print(f"  {proc_name:<35} (not in map, skipped)")

        print()
        print(f"  Words after all processors: {transcript.word_count}")
        print()

        # ── Summary ───────────────────────────────────────────────────────────
        print("="*60)
        print("  PIPELINE WORD COUNT SUMMARY")
        print("="*60)
        print(_wc_label("1. Raw Whisper (segment text)", raw_words_approx))
        if not skip_alignment:
            print(_wc_label("2. After alignment", aligned_word_count))
        print(_wc_label("3. After word extraction", word_extract_count))
        print(_wc_label("4. After merge", words_before_processors))
        print(_wc_label("5. After processors (FINAL)", transcript.word_count))
        print()

        expected_min = int(audio.duration / 60 * 100)
        expected_max = int(audio.duration / 60 * 200)
        final_wc = transcript.word_count
        print(f"  Audio duration: {audio.duration/60:.1f} min")
        print(f"  Expected range: {expected_min:,} – {expected_max:,} words")
        print(f"  Final word count: {final_wc:,} words")
        if final_wc < expected_min:
            print(f"\n  *** ABNORMAL: {final_wc} words is below expected minimum ({expected_min})")
            print("      Something is dropping words. Check stages above for WORD_LOSS.")
        else:
            print("\n  OK: word count is within expected range")
        print()

        # Save final transcript
        final_out = args.out_dir / "04_final_transcript.md"
        from transcript_engine.exporters.markdown import MarkdownExporter
        exp_config = settings.pipeline.export.model_copy(update={
            "output_dir": args.out_dir,
            "timestamp_mode": "none",
            "highlight_confidence": False,
        })
        content = MarkdownExporter().export(transcript, exp_config)
        final_out.write_bytes(content)
        print(f"  Final transcript saved to: {final_out}")
        print(f"  All intermediates in     : {args.out_dir}")
        print()


if __name__ == "__main__":
    main()
