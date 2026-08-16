#!/usr/bin/env python3
"""
Alignment Investigation Script

Measures Whisper transcription, wav2vec2 alignment, and pyannote diarization
times INDEPENDENTLY and CONCURRENTLY to determine whether MPS contention
is the cause of the reported 2+ hour "Aligning Words" stage.

Usage:
    python scripts/investigate_alignment.py <audio_file> [--no-diarization]

The script:
  1. Runs Whisper CT2 to get segments (measures transcription time)
  2. Runs wav2vec2 alignment ALONE (no diarization) to get a baseline
  3. Optionally runs alignment WITH concurrent diarization to measure contention
  4. Reports what device the alignment model actually runs on
  5. Reports per-segment statistics

Findings are written to alignment_investigation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def _resolve_actual_device(model: Any) -> str:
    """Find the device the model is actually on (not just what was requested)."""
    try:
        param = next(iter(model.parameters()))
        return str(param.device)
    except Exception:
        pass
    try:
        return str(next(model.buffers()).device)
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip the concurrent diarization comparison (alignment-only run)",
    )
    parser.add_argument(
        "--output",
        default="alignment_investigation.json",
        help="Output JSON path (default: alignment_investigation.json)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    from transcript_engine.audio.preprocessor import AudioPreprocessor
    from transcript_engine.config.loader import load_settings
    from transcript_engine.model_registry.registry import ModelRegistry

    settings = load_settings()
    registry = ModelRegistry(settings)
    t_cfg = settings.pipeline.transcription
    d_cfg = settings.pipeline.diarization

    results: dict[str, Any] = {
        "audio_file": audio_path.name,
        "device_config": t_cfg.device,
    }

    print(f"\nInvestigation target: {audio_path.name}")
    print(f"Device config: {t_cfg.device}")

    preprocessor = AudioPreprocessor(settings.pipeline.audio)
    preprocessor.__enter__()
    try:
        print("\n[1/4] Preprocessing audio...")
        t0 = time.perf_counter()
        audio = preprocessor.prepare(audio_path)
        preprocess_s = time.perf_counter() - t0
        results["audio_duration_s"] = round(audio.duration, 1)
        results["preprocess_s"] = round(preprocess_s, 2)
        print(f"      {audio.duration:.0f}s audio ready in {preprocess_s:.1f}s")

        # ── Step 2: Whisper CT2 transcription ─────────────────────────────────
        print("\n[2/4] Running Whisper CT2 transcription (measuring time)...")
        import whisperx

        ct2_device = registry._resolve_ctranslate_device(t_cfg.device)
        print(f"      CT2 device: {ct2_device}")

        t0 = time.perf_counter()
        model = registry.get_whisper_model(t_cfg)
        whisper_load_s = time.perf_counter() - t0
        print(f"      Whisper model loaded in {whisper_load_s:.1f}s")

        effective_batch = t_cfg.batch_size if ct2_device == "cuda" else None

        t0 = time.perf_counter()
        try:
            result = model.transcribe(
                str(audio.path),
                batch_size=effective_batch,
                language=t_cfg.language,
                chunk_size=t_cfg.chunk_length,
            )
        except (IndexError, StopIteration):
            print("      No speech detected — aborting")
            preprocessor.__exit__(None, None, None)
            return

        transcribe_s = time.perf_counter() - t0
        segs = result["segments"]
        detected_language = result.get("language", "en")
        n_segs = len(segs)
        total_seg_words = sum(
            len(s.get("words", s.get("text", "").split())) for s in segs
        )
        rtf_ct2 = audio.duration / transcribe_s if transcribe_s > 0 else 0

        results["transcription"] = {
            "device": ct2_device,
            "elapsed_s": round(transcribe_s, 1),
            "rtf": round(rtf_ct2, 1),
            "language": detected_language,
            "segments": n_segs,
            "words_in_segments": total_seg_words,
        }
        print(f"      Transcription done: {transcribe_s:.0f}s ({rtf_ct2:.1f}x RTF)")
        print(
            f"      Segments: {n_segs}, words: {total_seg_words}, lang: {detected_language}"
        )

        # ── Step 3: Alignment alone (no diarization) ──────────────────────────
        print("\n[3/4] Running wav2vec2 alignment IN ISOLATION (no diarization)...")
        torch_device = registry._resolve_torch_device(t_cfg.device)
        print(f"      Requested torch device: {torch_device}")

        t0 = time.perf_counter()
        align_model, align_metadata = registry.get_alignment_model(
            detected_language, t_cfg
        )
        align_load_s = time.perf_counter() - t0
        print(f"      Alignment model loaded in {align_load_s:.2f}s")

        actual_device = _resolve_actual_device(align_model)
        print(f"      Actual model device: {actual_device}")

        # Try to detect if MPS is actually being used or falling back
        try:
            import torch

            mps_available = torch.backends.mps.is_available()
            mps_built = torch.backends.mps.is_built()
            print(f"      MPS available={mps_available}, built={mps_built}")
            results["mps_available"] = mps_available
        except Exception:
            pass

        t_align_alone_start = time.perf_counter()
        try:
            aligned_alone = whisperx.align(
                segs,
                align_model,
                align_metadata,
                str(audio.path),
                torch_device,
                return_char_alignments=False,
            )
            align_alone_s = time.perf_counter() - t_align_alone_start
            align_alone_ok = True
        except Exception as exc:
            align_alone_s = time.perf_counter() - t_align_alone_start
            align_alone_ok = False
            print(f"      Alignment failed: {exc}")

        words_out = 0
        if align_alone_ok:
            for seg in aligned_alone.get("segments", []):
                words_out += len(seg.get("words", []))

        rtf_align = audio.duration / align_alone_s if align_alone_s > 0 else 0
        print(f"      Alignment done: {align_alone_s:.0f}s ({rtf_align:.1f}x RTF)")
        print(f"      Words out: {words_out}")
        print(f"      Actual device: {actual_device}")

        results["alignment_alone"] = {
            "requested_device": torch_device,
            "actual_device": actual_device,
            "elapsed_s": round(align_alone_s, 1),
            "rtf": round(rtf_align, 1),
            "words_out": words_out,
            "pct_of_audio": round(100 * align_alone_s / audio.duration, 1),
        }

        if align_alone_s > audio.duration * 0.10:
            print(
                f"\n  ⚠ SLOW: alignment took {100 * align_alone_s / audio.duration:.0f}% of audio duration"
            )
            print(f"     Expected: <10% ({audio.duration * 0.10:.0f}s)")
            print(f"     Actual:    {align_alone_s:.0f}s")
            if actual_device == "cpu" and torch_device != "cpu":
                print(
                    f"     ROOT CAUSE CANDIDATE: requested {torch_device} but running on CPU"
                )
                results["alignment_alone"][
                    "warning"
                ] = f"requested {torch_device} but model is on cpu — MPS fallback"

        # ── Step 4: Alignment WITH concurrent diarization (contention test) ──
        if args.no_diarization:
            print("\n[4/4] Skipping concurrent diarization test (--no-diarization)")
        else:
            has_token = bool(settings.hf_token)
            if not has_token:
                print(
                    "\n[4/4] Skipping concurrent diarization test (no HF token configured)"
                )
                results["concurrent_test"] = {"skipped": "no hf_token"}
            else:
                print(
                    "\n[4/4] Running alignment WITH concurrent pyannote diarization..."
                )
                print("      (This measures MPS contention between the two models)")
                from transcript_engine.diarization.pyannote_engine import PyannoteEngine

                diar_engine = PyannoteEngine(registry)

                diar_timing: dict[str, Any] = {}

                def _run_diar() -> None:
                    t = time.perf_counter()
                    try:
                        diar_engine.diarize(audio, d_cfg)
                        diar_timing["elapsed_s"] = round(time.perf_counter() - t, 1)
                        diar_timing["error"] = None
                    except Exception as exc:
                        diar_timing["elapsed_s"] = round(time.perf_counter() - t, 1)
                        diar_timing["error"] = str(exc)

                diar_thread = threading.Thread(target=_run_diar, daemon=True)

                t_concurrent_start = time.perf_counter()
                diar_thread.start()

                # Run alignment concurrently with diarization
                t_align_concurrent = time.perf_counter()
                try:
                    whisperx.align(
                        segs,
                        align_model,
                        align_metadata,
                        str(audio.path),
                        torch_device,
                        return_char_alignments=False,
                    )
                    align_concurrent_s = time.perf_counter() - t_align_concurrent
                except Exception:
                    align_concurrent_s = time.perf_counter() - t_align_concurrent

                diar_thread.join()
                concurrent_wall_s = time.perf_counter() - t_concurrent_start

                contention_ratio = (
                    align_concurrent_s / align_alone_s if align_alone_s > 0 else 0
                )
                print(f"      Alignment alone:       {align_alone_s:.0f}s")
                print(f"      Alignment concurrent:  {align_concurrent_s:.0f}s")
                print(
                    f"      Diarization:           {diar_timing.get('elapsed_s', '?')}s"
                )
                print(f"      Concurrent wall:       {concurrent_wall_s:.0f}s")
                print(
                    f"      Contention ratio:      {contention_ratio:.2f}x (>1.5 indicates MPS contention)"
                )

                results["concurrent_test"] = {
                    "alignment_alone_s": round(align_alone_s, 1),
                    "alignment_concurrent_s": round(align_concurrent_s, 1),
                    "diarization_s": diar_timing.get("elapsed_s"),
                    "diarization_error": diar_timing.get("error"),
                    "concurrent_wall_s": round(concurrent_wall_s, 1),
                    "contention_ratio": round(contention_ratio, 2),
                }

                if contention_ratio > 1.5:
                    print(
                        f"\n  ⚠ MPS CONTENTION CONFIRMED: alignment is {contention_ratio:.1f}x slower when"
                    )
                    print(
                        "    running concurrently with diarization. Both are competing for the Metal GPU."
                    )
                    results["concurrent_test"][
                        "finding"
                    ] = f"MPS contention: alignment {contention_ratio:.1f}x slower when concurrent with diarization"

    finally:
        preprocessor.__exit__(None, None, None)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  INVESTIGATION SUMMARY")
    print("─" * 60)
    print(
        f"  Audio:              {results['audio_duration_s']:.0f}s ({results['audio_file']})"
    )
    print(
        f"  Whisper CT2:        {results['transcription']['elapsed_s']:.0f}s "
        f"({results['transcription']['rtf']:.1f}x RTF) on {results['transcription']['device']}"
    )
    print(
        f"  Alignment (alone):  {results['alignment_alone']['elapsed_s']:.0f}s "
        f"({results['alignment_alone']['rtf']:.1f}x RTF) on {results['alignment_alone']['actual_device']}"
    )
    print(f"    = {results['alignment_alone']['pct_of_audio']:.0f}% of audio duration")
    if (
        "concurrent_test" in results
        and "contention_ratio" in results["concurrent_test"]
    ):
        ct = results["concurrent_test"]
        print(
            f"  Concurrent align:   {ct['alignment_concurrent_s']:.0f}s "
            f"({ct['contention_ratio']:.2f}x slower than alone)"
        )

    align_pct = results["alignment_alone"]["pct_of_audio"]
    if align_pct > 10:
        print(
            f"\n  ROOT CAUSE: alignment is consuming {align_pct:.0f}% of audio duration."
        )
        align_alone = results["alignment_alone"]
        if (
            align_alone["actual_device"] == "cpu"
            and align_alone["requested_device"] != "cpu"
        ):
            print(
                f"  → wav2vec2 requested {align_alone['requested_device']} but landed on CPU."
            )
            print(
                "  → MPS may not support all wav2vec2 ops, causing silent CPU fallback."
            )
        elif "contention_ratio" in results.get("concurrent_test", {}):
            if results["concurrent_test"]["contention_ratio"] > 1.5:
                print(
                    "  → MPS contention: both wav2vec2 and pyannote compete for Metal GPU."
                )
                print("  → Running them concurrently serializes their Metal kernels.")
    else:
        print(
            f"\n  Alignment is within expected range ({align_pct:.0f}% of audio duration)."
        )

    print("─" * 60)

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Full results: {out.resolve()}")


if __name__ == "__main__":
    main()
