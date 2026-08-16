"""
Runs the transcript_engine pipeline in a background thread.
Progress events are published to a per-job asyncio.Queue for SSE consumption.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import threading
import time
import traceback as _traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.job_manager import job_manager
from api.models import JobStatus

_log = logging.getLogger(__name__)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# How long a job may emit *no* pipeline progress at all before the watchdog
# declares it stalled. Deliberately generous: a single Parakeet chunk can be up
# to ~22 min of audio, and a CPU-fallback run of that chunk is far slower than
# realtime, so a legitimately-working job can be quiet for a long stretch.
# Override with TE_API_STALL_TIMEOUT_MINUTES for slower hardware.
_STALL_TIMEOUT_S: float = (
    float(os.environ.get("TE_API_STALL_TIMEOUT_MINUTES", "45")) * 60.0
)

# Transcript formats written for every completed job. All are derived from the
# same finished Transcript, so adding a format costs milliseconds rather than
# another GPU pass. Keep in sync with _EXPECTED_ARTIFACTS below.
TRANSCRIPT_EXPORT_FORMATS: tuple[str, ...] = ("markdown", "json", "txt", "srt", "vtt")


# ── Stage timing tracker ──────────────────────────────────────────────────────


class _StageTracker:
    """Records wall-clock duration for each pipeline stage transition."""

    def __init__(self, pipeline_start: float) -> None:
        self._t0 = pipeline_start
        self._current: str | None = None
        self._current_start: float = pipeline_start
        self._completed: dict[str, dict[str, Any]] = {}

    def add_upload(self, upload_seconds: float) -> None:
        if upload_seconds > 0:
            self._completed["Upload"] = {
                "start_s": round(-upload_seconds, 2),
                "end_s": 0.0,
                "duration_s": round(upload_seconds, 2),
            }

    def transition(self, stage: str) -> None:
        if stage == self._current:
            return
        now = time.monotonic()
        if self._current is not None and self._current not in ("__done__",):
            dur = now - self._current_start
            # Update existing entry if stage revisited, else create
            prev = self._completed.get(self._current, {})
            self._completed[self._current] = {
                "start_s": prev.get(
                    "start_s", round(self._current_start - self._t0, 2)
                ),
                "end_s": round(now - self._t0, 2),
                "duration_s": round(prev.get("duration_s", 0.0) + dur, 2),
            }
        self._current = stage
        self._current_start = now
        if stage not in self._completed:
            self._completed[stage] = {
                "start_s": round(now - self._t0, 2),
                "end_s": None,
                "duration_s": 0.0,
            }

    def close(self) -> None:
        self.transition("__done__")

    def current(self) -> str:
        return self._current or "unknown"

    def report(self, total_s: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for stage, data in self._completed.items():
            if stage.startswith("__"):
                continue
            dur = data.get("duration_s", 0.0)
            out[stage] = {
                "start_s": data.get("start_s", 0.0),
                "end_s": data.get("end_s", 0.0),
                "duration_s": round(dur, 2),
                "pct": round(100 * dur / total_s, 1) if total_s > 0 else 0.0,
            }
        return out


# ── Resource monitor ──────────────────────────────────────────────────────────


def _resource_monitor(
    push: Any,
    stop: threading.Event,
    stage_ref: dict[str, str],
    samples: list[dict[str, Any]],
    interval: float = 5.0,
) -> None:
    """Sample CPU/RAM/swap every interval seconds; push SSE events and accumulate samples."""
    try:
        import psutil  # noqa: PLC0415

        process = psutil.Process()
    except ImportError:
        return
    tick = 0
    while not stop.wait(interval):
        try:
            cpu = psutil.cpu_percent(interval=None)
            rss_gb = process.memory_info().rss / 1_073_741_824
            push(
                {
                    "type": "resource",
                    "cpu_pct": round(cpu, 1),
                    "ram_gb": round(rss_gb, 2),
                }
            )
            tick += 1
            if tick % 2 == 0:  # every ~10 s
                try:
                    disk_gb = psutil.disk_usage(".").used / 1_073_741_824
                except Exception:
                    disk_gb = 0.0
                try:
                    threads = process.num_threads()
                except Exception:
                    threads = 0
                try:
                    swap_gb = psutil.swap_memory().used / 1_073_741_824
                except Exception:
                    swap_gb = 0.0
                samples.append(
                    {
                        "t_s": round(time.monotonic(), 1),
                        "stage": stage_ref.get("current", "unknown"),
                        "cpu_pct": round(cpu, 1),
                        "ram_gb": round(rss_gb, 2),
                        "swap_gb": round(swap_gb, 2),
                        "disk_used_gb": round(disk_gb, 2),
                        "threads": threads,
                    }
                )
        except Exception:
            pass


# ── Synthetic progress ticker for long transcription ─────────────────────────


def _synthetic_ticker(
    push: Any,
    stop: threading.Event,
    start_fraction: float,
    ceiling: float,
    expected_s: float,
) -> None:
    """Emit fake incremental stage events during Transcribing so the bar never freezes."""
    elapsed = 0.0
    while not stop.wait(10.0):
        elapsed += 10.0
        progress_ratio = min(elapsed / max(expected_s, 1.0), 0.95)
        frac = start_fraction + (ceiling - start_fraction) * progress_ratio
        push(
            {
                "type": "stage",
                "stage": "Transcribing",
                "fraction": round(min(frac, ceiling - 0.01), 3),
                "message": "Transcribing...",
            }
        )


# ── Stuck-job watchdog ────────────────────────────────────────────────────────


def _stall_watchdog(
    job_id: str,
    stop: threading.Event,
    last_progress: dict[str, float],
    on_stall: Any,
    timeout_s: float,
    check_interval_s: float = 60.0,
) -> None:
    """
    Fail a job that has stopped emitting *pipeline* progress entirely.

    Watches `last_progress["t"]`, which is updated only by the real pipeline
    progress callback — never by the synthetic ticker or the resource monitor,
    both of which keep ticking from their own threads even while the pipeline
    thread is wedged. That distinction is the whole point: a heartbeat that
    survives the hang can't detect the hang.

    IMPORTANT: this cannot kill the stuck worker thread — Python has no safe
    thread-kill. It marks the job failed so the API and frontend reach a
    terminal state instead of spinning forever. The wedged thread still holds
    the single-worker executor, so subsequent jobs remain queued until the
    process is restarted; that is logged explicitly as an operator signal.
    """
    while not stop.wait(check_interval_s):
        idle_s = time.monotonic() - last_progress["t"]
        if idle_s < timeout_s:
            continue
        _log.error(
            "Job %s stalled: no pipeline progress for %.0f min (limit %.0f min). "
            "Marking failed. The worker thread cannot be killed and still holds "
            "the executor — restart the API process to recover job throughput.",
            job_id,
            idle_s / 60,
            timeout_s / 60,
        )
        try:
            on_stall(idle_s)
        except Exception as exc:
            _log.warning("Stall handler failed for job %s: %s", job_id, exc)
        return


# ── Stage label mapping ───────────────────────────────────────────────────────

_STAGE_LABELS: dict[str, str] = {
    "Preprocessing audio": "Preprocessing Audio",
    "Audio ready": "Preprocessing Audio",
    "Loading Whisper model": "Loading Model",
    "Loading Parakeet model": "Loading Model",
    "Aligning timestamps": "Aligning Words",
    "Transcription complete": "Transcribing",
    "Finalizing speaker attribution": "Processing",
    "Loading diarization model": "Loading Diarization",
    "Running speaker diarization": "Speaker Diarization",
    "Diarization complete": "Speaker Diarization",
    "Merging speakers and words": "Processing",
    "Running processors": "Processing",
    "Done": "Generating Files",
}


def _stage_label(message: str) -> str:
    if message in _STAGE_LABELS:
        return _STAGE_LABELS[message]
    m = message.lower()
    if "audio" in m or "preprocess" in m:
        return "Preprocessing Audio"
    if "load" in m and ("model" in m or "whisper" in m or "parakeet" in m):
        return "Loading Model"
    if "diariz" in m and "load" in m:
        return "Loading Diarization"
    if "transcrib" in m:
        return "Transcribing"
    if "align" in m:
        return "Aligning Words"
    if "speaker attribution" in m or "finaliz" in m:
        return "Processing"
    if "diariz" in m or "speaker" in m:
        return "Speaker Diarization"
    if "merg" in m:
        return "Processing"
    if "process" in m or "vocab" in m or "clean" in m or "correct" in m:
        return "Processing"
    if "export" in m or "done" in m:
        return "Generating Files"
    return "Processing"


def _parse_chunk_progress(message: str) -> dict[str, int] | None:
    """
    Parse "Transcribing chunk N/M (start–end min)" messages.

    Returns {"current": N, "total": M} when the message matches, else None.
    Used to emit a structured chunk_progress SSE event alongside the stage event.
    """
    import re  # noqa: PLC0415

    match = re.search(r"chunk\s+(\d+)/(\d+)", message, re.IGNORECASE)
    if match:
        return {"current": int(match.group(1)), "total": int(match.group(2))}
    return None


# ── Error humanization ────────────────────────────────────────────────────────


def _humanize_error(exc: Exception) -> str:
    msg = str(exc)
    ml = msg.lower()
    if "huggingface" in ml or "hf_token" in ml or ("token" in ml and "model" in ml):
        return (
            "Speaker diarization requires a HuggingFace token with access to pyannote models. "
            "Go to the Setup page to add your token."
        )
    if "401" in msg or "unauthorized" in ml:
        return (
            "HuggingFace token is invalid or expired. "
            "Generate a new token at huggingface.co/settings/tokens and update it in Setup."
        )
    if "403" in msg or "forbidden" in ml or "license" in ml or "gated" in ml:
        return (
            "You need to accept the pyannote model license on HuggingFace. "
            "Visit huggingface.co/pyannote/speaker-diarization-3.1 and click 'Agree'."
        )
    if "cuda" in ml and "device" in ml:
        return "Device error. Try setting TE_PIPELINE__TRANSCRIPTION__DEVICE=cpu in your .env file."
    if "ffmpeg" in ml:
        return "FFmpeg not found. Install it with: brew install ffmpeg"
    if "no such file or directory" in ml:
        return "Audio file not found. Please re-upload."
    if "connectionerror" in ml or "networkerror" in ml or "timeout" in ml:
        return "Network error. Check your internet connection and try again."
    if "permission" in ml or "errno 13" in ml:
        return (
            "Permission denied writing to output directory. Check folder permissions."
        )
    if "out of memory" in ml or "oom" in ml:
        return (
            "GPU memory was insufficient for this recording, even after the engine's "
            "automatic retries with a smaller processing window. Try again — free VRAM "
            "varies between runs — or use a shorter recording or a GPU with more VRAM."
        )
    if isinstance(exc, (IndexError, StopIteration)) or "index out of range" in ml:
        return "No speech detected in the audio file. Please check the recording contains speech."
    return f"Processing failed: {msg}"


# ── Output validation ─────────────────────────────────────────────────────────

_EXPECTED_ARTIFACTS = [
    "transcript.md",
    "transcript.json",
    "transcript.txt",
    "transcript.srt",
    "transcript.vtt",
    "transcript.stats.json",
    "transcript.quality.md",
    "transcript.summary.md",
    "transcript.action_items.md",
    "transcript.decisions.md",
    "transcript.questions.md",
    "transcript.timeline.md",
    "transcript.entities.json",
    "transcript.metrics.json",
    "transcript.review.md",
    "transcript.index.json",
    "transcript_bundle.zip",
]


def _validate_outputs(output_dir: Path) -> dict[str, Any]:
    missing: list[str] = []
    artifacts_info: dict[str, Any] = {}
    for name in _EXPECTED_ARTIFACTS:
        p = output_dir / name
        if p.exists():
            artifacts_info[name] = {"status": "ok", "size_bytes": p.stat().st_size}
        else:
            artifacts_info[name] = {"status": "missing"}
            missing.append(name)
            _log.warning("Missing artifact: %s", name)
    return {
        "total": len(_EXPECTED_ARTIFACTS),
        "present": len(_EXPECTED_ARTIFACTS) - len(missing),
        "missing": missing,
        "artifacts": artifacts_info,
    }


# ── Report helpers ────────────────────────────────────────────────────────────


def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, rm = divmod(m, 60)
    return f"{h}h {rm:02d}m {sec:02d}s"


def _write_processing_report(
    output_dir: Path,
    job_id: str,
    audio_filename: str,
    audio_duration_s: float,
    started_at: str,
    completed_at: str,
    total_s: float,
    status: str,
    upload_s: float,
    stage_report: dict[str, Any],
    model_info: dict[str, str],
    output_validation: dict[str, Any],
    temp_validation: dict[str, Any],
    error: str | None,
) -> None:
    data = {
        "job_id": job_id,
        "audio_file": audio_filename,
        "audio_duration_s": round(audio_duration_s, 1),
        "started_at": started_at,
        "completed_at": completed_at,
        "total_s": round(total_s, 2),
        "status": status,
        "upload_s": round(upload_s, 2),
        "stages": stage_report,
        "models": model_info,
        "output_validation": output_validation,
        "temp_validation": temp_validation,
        "error": error,
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "processing_report.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        _log.warning("Could not write processing_report.json: %s", exc)


def _write_resource_report(
    output_dir: Path,
    job_id: str,
    samples: list[dict[str, Any]],
) -> None:
    if not samples:
        return
    cpu_vals = [s.get("cpu_pct", 0.0) for s in samples]
    ram_vals = [s.get("ram_gb", 0.0) for s in samples]
    swap_vals = [s.get("swap_gb", 0.0) for s in samples]
    summary: dict[str, Any] = {
        "cpu_peak": round(max(cpu_vals), 1),
        "cpu_avg": round(sum(cpu_vals) / len(cpu_vals), 1),
        "ram_peak_gb": round(max(ram_vals), 2),
        "ram_avg_gb": round(sum(ram_vals) / len(ram_vals), 2),
        "swap_peak_gb": round(max(swap_vals), 2),
        "swap_avg_gb": round(sum(swap_vals) / len(swap_vals), 2),
    }
    data = {
        "job_id": job_id,
        "interval_s": 10,
        "samples": samples,
        "summary": summary,
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resource_usage.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        _log.warning("Could not write resource_usage.json: %s", exc)


def _write_production_report(
    output_dir: Path,
    audio_filename: str,
    started_at: str,
    total_s: float,
    status: str,
    stage_report: dict[str, Any],
    resource_samples: list[dict[str, Any]],
    output_validation: dict[str, Any],
    temp_validation: dict[str, Any],
    pipeline_warnings: list[str],
    error: str | None,
    word_count: int,
    speaker_count: int,
    corrections_count: int,
    avg_confidence: float | None,
) -> None:
    lines: list[str] = [
        "# Production Run Report",
        "",
        f"**File**: {audio_filename}",
        f"**Date**: {started_at}",
        f"**Total**: {_fmt_dur(total_s)}",
        f"**Status**: {'COMPLETED' if status == 'completed' else 'FAILED'}",
        "",
        "## Stage Timings",
        "",
        "| Stage | Duration | % of Total |",
        "|-------|----------|------------|",
    ]
    for stage, data in sorted(
        stage_report.items(), key=lambda x: x[1].get("start_s", 0.0)
    ):
        lines.append(
            f"| {stage} | {_fmt_dur(data['duration_s'])} | {data.get('pct', 0):.1f}% |"
        )
    lines.append("")

    sorted_stages = sorted(
        stage_report.items(),
        key=lambda x: x[1].get("duration_s", 0.0),
        reverse=True,
    )
    lines += ["## Bottleneck Ranking", ""]
    for i, (stage, data) in enumerate(sorted_stages[:5], 1):
        dur = data.get("duration_s", 0.0)
        pct = data.get("pct", 0.0)
        bar = "█" * min(int(pct / 2), 30)
        lines.append(f"{i}. **{stage}** — {_fmt_dur(dur)} ({pct:.1f}%) {bar}")
    lines.append("")

    if resource_samples:
        cpu_vals = [s.get("cpu_pct", 0.0) for s in resource_samples]
        ram_vals = [s.get("ram_gb", 0.0) for s in resource_samples]
        swap_vals = [s.get("swap_gb", 0.0) for s in resource_samples]
        lines += [
            "## Resource Usage",
            "",
            "| Metric | Peak | Average |",
            "|--------|------|---------|",
            f"| CPU % | {max(cpu_vals):.0f}% | {sum(cpu_vals)/len(cpu_vals):.0f}% |",
            f"| RAM (GB) | {max(ram_vals):.2f} | {sum(ram_vals)/len(ram_vals):.2f} |",
            f"| Swap (GB) | {max(swap_vals):.2f} | {sum(swap_vals)/len(swap_vals):.2f} |",
            "",
        ]

    lines += [
        "## Transcript Statistics",
        "",
        f"- Speakers: {speaker_count}",
        f"- Words: {word_count:,}",
    ]
    if avg_confidence is not None:
        lines.append(f"- Confidence: {avg_confidence:.2f} (average)")
    lines += [f"- Corrections applied: {corrections_count}", ""]

    lines += ["## Artifacts", ""]
    for name, info in output_validation.get("artifacts", {}).items():
        if info.get("status") == "ok":
            lines.append(f"✓ {name} ({info.get('size_bytes', 0):,} bytes)")
        else:
            lines.append(f"✗ MISSING: {name}")
    lines.append("")

    audio_del = temp_validation.get("audio_deleted", False)
    temp_del = temp_validation.get("temp_dir_deleted", False)
    lines += [
        "## Temp File Cleanup",
        "",
        f"- Audio file: {'✓ deleted' if audio_del else '✗ NOT DELETED'}",
        f"- Temp dir: {'✓ deleted' if temp_del else '✗ NOT DELETED'}",
    ]
    if temp_validation.get("orphaned_chunks"):
        lines.append(f"- Orphaned chunks: {temp_validation['orphaned_chunks']}")
    lines.append("")

    lines += ["## Warnings", ""]
    if pipeline_warnings:
        for w in pipeline_warnings:
            lines.append(f"- {w}")
    else:
        lines.append("(none)")
    lines.append("")

    if error:
        lines += ["## Error", "", f"```\n{error}\n```", ""]

    if sorted_stages:
        slowest_name, slowest_data = sorted_stages[0]
        lines += [
            "## Slowest Stage",
            "",
            f"{slowest_name} — {_fmt_dur(slowest_data.get('duration_s', 0))} "
            f"({slowest_data.get('pct', 0):.1f}% of total)",
            "",
        ]

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "production_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    except Exception as exc:
        _log.warning("Could not write production_report.md: %s", exc)


# ── Pipeline runner ───────────────────────────────────────────────────────────


def _run_pipeline_sync(
    job_id: str,
    audio_path: Path,
    profile_name: str,
    mode: str,
    output_dir: Path,
    timestamps: str,
    highlight_confidence: bool,
    push_event: Any,
    upload_seconds: float = 0.0,
) -> None:
    """Executes the full pipeline synchronously (called from thread pool)."""
    from api.model_cache import get_registry
    from transcript_engine.config.loader import load_settings
    from transcript_engine.exporters.corrections import export_corrections
    from transcript_engine.exporters.quality import export_quality
    from transcript_engine.exporters.registry import get_exporter
    from transcript_engine.exporters.stats import export_stats
    from transcript_engine.intelligence.ai_backend import AIBackend
    from transcript_engine.intelligence.engine import MeetingIntelligenceEngine
    from transcript_engine.intelligence.exports import (
        export_action_items,
        export_decisions,
        export_entities,
        export_metrics,
        export_questions,
        export_summary,
        export_timeline,
    )
    from transcript_engine.pipeline.orchestrator import Pipeline
    from transcript_engine.profiles.loader import load_profile
    from transcript_engine.review.index import generate_index
    from transcript_engine.review.report import generate_review

    # Enter background mode immediately — before any CPU-intensive work.
    # Sets QOS_CLASS_UTILITY (macOS) so this thread yields to Chrome/VS Code/Cursor.
    # Sets IOPOL_THROTTLE so model reads don't compete with UI app I/O.
    from transcript_engine.scheduling import enter_background_mode  # noqa: PLC0415

    enter_background_mode()

    pipeline_start = time.monotonic()
    started_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Observability state
    tracker = _StageTracker(pipeline_start)
    tracker.add_upload(upload_seconds)
    stage_ref: dict[str, str] = {"current": "Queued"}
    resource_samples: list[dict[str, Any]] = []
    pipeline_warnings: list[str] = []
    _run_state: dict[str, Any] = {
        "result": None,
        "metrics": {},
        "error": None,
        "status": "failed",
        "audio_duration_s": 0.0,
        "word_count": 0,
        "speaker_count": 0,
        "corrections_count": 0,
        "avg_confidence": None,
        "output_validation": {"total": 0, "present": 0, "missing": [], "artifacts": {}},
        "temp_validation": {"audio_deleted": False, "temp_dir_deleted": False},
        "model_info": {},
    }

    # Synthetic ticker state
    _ticker_stop: threading.Event | None = None
    _ticker_thread: threading.Thread | None = None
    _audio_duration_ref: dict[str, float] = {"value": 0.0}
    _current_stage_for_ticker: dict[str, str] = {"value": ""}

    def _stop_ticker() -> None:
        nonlocal _ticker_stop, _ticker_thread
        if _ticker_stop is not None:
            _ticker_stop.set()
            _ticker_stop = None
        _ticker_thread = None

    def _start_ticker(current_fraction: float) -> None:
        nonlocal _ticker_stop, _ticker_thread
        _stop_ticker()
        audio_dur = _audio_duration_ref["value"]
        expected_s = audio_dur / 3.5 if audio_dur > 0 else 600.0
        stop_ev = threading.Event()
        _ticker_stop = stop_ev
        t = threading.Thread(
            target=_synthetic_ticker,
            args=(push_event, stop_ev, current_fraction, 0.62, expected_s),
            daemon=True,
        )
        t.start()
        _ticker_thread = t

    _stop_monitor = threading.Event()
    _monitor_thread = threading.Thread(
        target=_resource_monitor,
        args=(push_event, _stop_monitor, stage_ref, resource_samples),
        daemon=True,
    )
    _monitor_thread.start()

    # Stuck-job watchdog. _last_progress["t"] is bumped only from the real
    # pipeline progress callback (see on_progress below), so it reflects actual
    # pipeline advancement rather than the ticker/monitor threads' heartbeats.
    _last_progress = {"t": time.monotonic()}
    _stop_watchdog = threading.Event()

    def _on_stall(idle_s: float) -> None:
        stall_msg = (
            f"Job stopped responding — no processing progress for "
            f"{idle_s / 60:.0f} minutes. The job was marked failed; "
            f"restart the server if this repeats."
        )
        _run_state["error"] = stall_msg
        _run_state["status"] = "failed"
        job_manager.update(
            job_id,
            status=JobStatus.failed,
            completed_at=datetime.now(tz=UTC),
            error=stall_msg,
        )
        push_event({"type": "error", "message": stall_msg})

    _watchdog_thread = threading.Thread(
        target=_stall_watchdog,
        args=(job_id, _stop_watchdog, _last_progress, _on_stall, _STALL_TIMEOUT_S),
        daemon=True,
    )
    _watchdog_thread.start()

    try:
        from transcript_engine.config.settings import PIPELINE_MODES  # noqa: PLC0415

        settings = load_settings()
        profiles_dir = Path(settings.pipeline.processing.profiles_dir)
        loaded_profile = load_profile(profile_name, profiles_dir=profiles_dir)

        mode_spec = PIPELINE_MODES.get(mode, PIPELINE_MODES["balanced"])
        settings = settings.model_copy(
            update={
                "pipeline": settings.pipeline.model_copy(
                    update={
                        "transcription": settings.pipeline.transcription.model_copy(
                            update={
                                "model_id": mode_spec.model_id,
                                "beam_size": mode_spec.beam_size,
                                "best_of": mode_spec.best_of,
                                "temperatures": mode_spec.temperatures,
                                "force_skip_alignment": mode_spec.skip_alignment,
                            }
                        ),
                        "diarization": settings.pipeline.diarization.model_copy(
                            update={"enabled": bool(settings.hf_token)}
                        ),
                        "processing": settings.pipeline.processing.model_copy(
                            update={
                                "profile": profile_name,
                                "processors": mode_spec.processors,
                            }
                        ),
                        "export": settings.pipeline.export.model_copy(
                            update={
                                "formats": ["markdown", "json"],
                                "output_dir": output_dir,
                                "timestamp_mode": timestamps,
                                "highlight_confidence": highlight_confidence,
                            }
                        ),
                    }
                )
            }
        )

        registry = get_registry()
        pipeline = Pipeline(settings.pipeline, registry, profile=loaded_profile)

        # Model validation log
        import os as _os  # noqa: PLC0415
        diar_enabled = settings.pipeline.diarization.enabled
        diar_model = (
            settings.pipeline.diarization.model_id if diar_enabled else "disabled"
        )
        _asr_backend = _os.environ.get("TE_ASR_BACKEND", "whisper").lower()
        if _asr_backend == "parakeet":
            asr_model = _os.environ.get("TE_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
            resolved_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        else:
            asr_model = settings.pipeline.transcription.model_id
            resolved_device = registry._resolve_ctranslate_device(
                settings.pipeline.transcription.device
            )
        _log.info(
            "Models: asr=%s, backend=%s, device=%s, diarization=%s",
            asr_model,
            _asr_backend,
            resolved_device,
            diar_model,
        )
        _run_state["model_info"] = {
            "asr": asr_model,
            "backend": _asr_backend,
            "device": resolved_device,
            "diarization": diar_model,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        job_manager.update(job_id, status=JobStatus.running)

        # High-watermark: progress bar must never visually go backward.
        # In the parallel branch, diarization (which completes first) can push the
        # fraction to 0.90; transcription then fires at 0.60 — a visible regression.
        _fraction_hwm: dict[str, float] = {"value": 0.0}

        def on_progress(fraction: float, message: str) -> None:
            nonlocal _ticker_stop, _ticker_thread

            # Heartbeat for the stall watchdog. Deliberately only here — the
            # synthetic ticker and resource monitor run in their own threads and
            # would keep this fresh even if the pipeline itself were wedged.
            _last_progress["t"] = time.monotonic()

            # Pipeline decision messages from the adaptive analyzer
            if message.startswith("pipeline_decision|"):
                try:
                    payload = json.loads(message[len("pipeline_decision|") :])
                    push_event(
                        {
                            "type": "pipeline_decision",
                            "decisions": payload.get("decisions", []),
                            "savings_s": payload.get("savings_s", 0),
                        }
                    )
                except Exception:
                    pass
                return

            # Decode audio duration from "Audio ready|{duration}" messages
            if "|" in message:
                bare, _, extra = message.partition("|")
                with contextlib.suppress(ValueError):
                    dur = float(extra)
                    push_event({"type": "audio_info", "audio_duration": dur})
                    _audio_duration_ref["value"] = dur
                    _run_state["audio_duration_s"] = dur
                message = bare

            stage = _stage_label(message)
            prev_stage = _current_stage_for_ticker["value"]

            # Manage synthetic ticker
            if stage == "Transcribing" and prev_stage != "Transcribing":
                _start_ticker(fraction)
            elif stage != "Transcribing" and prev_stage == "Transcribing":
                _stop_ticker()

            _current_stage_for_ticker["value"] = stage
            tracker.transition(stage)
            stage_ref["current"] = stage

            # Clamp to [hwm, 0.95] — never let a late-arriving progress event
            # from a concurrent thread visually move the bar backward.
            effective = max(min(fraction, 0.95), _fraction_hwm["value"])
            _fraction_hwm["value"] = effective

            push_event(
                {
                    "type": "stage",
                    "stage": stage,
                    "fraction": effective,
                    "message": message,
                }
            )

            # Emit a structured chunk progress event for Parakeet chunked transcription.
            chunk_info = _parse_chunk_progress(message)
            if chunk_info is not None:
                push_event(
                    {
                        "type": "chunk_progress",
                        "current_chunk": chunk_info["current"],
                        "total_chunks": chunk_info["total"],
                        "stage": stage,
                        "fraction": effective,
                    }
                )

        result = pipeline.run(
            audio_path,
            on_progress=on_progress,
        )
        _stop_ticker()

        # Guard: a non-trivial audio file producing 0 words means the pipeline
        # silently failed (e.g. skip_alignment path with no word-level data).
        # Raise so the job is marked failed rather than completing with empty output.
        audio_dur = _run_state.get("audio_duration_s") or 0.0
        if result.transcript.word_count == 0 and audio_dur > 30.0:
            raise RuntimeError(
                f"Pipeline produced 0 words for {audio_dur:.0f}s of audio. "
                "Transcription likely failed silently — check server logs for the upstream error."
            )

        # Emit the final processed transcript immediately so the UI shows the
        # speaker-labeled, vocabulary-corrected version while intelligence + exports run.
        try:
            speakers = result.transcript.speakers or {}
            segments_final = []
            for seg in result.transcript.segments:
                spk_id = getattr(seg, "speaker_id", "SPEAKER_00")
                display = speakers.get(spk_id) or getattr(
                    seg, "speaker_display", "Speaker"
                )
                segments_final.append(
                    {
                        "speaker": display,
                        "text": seg.text,
                        "start": round(float(getattr(seg, "start", 0.0)), 1),
                    }
                )
            if segments_final:
                push_event(
                    {
                        "type": "partial_transcript",
                        "segments": segments_final,
                        "total_segments": len(segments_final),
                        "word_count": result.transcript.word_count,
                    }
                )
        except Exception:
            pass

        _run_state["result"] = result
        _run_state["word_count"] = result.transcript.word_count
        _run_state["speaker_count"] = len(result.transcript.speakers)
        _run_state["corrections_count"] = len(result.corrections)

        # Use max(hwm, 0.96) so these stages never visually go backwards.
        _fraction_hwm["value"] = max(_fraction_hwm["value"], 0.96)
        push_event(
            {
                "type": "stage",
                "stage": "Meeting Intelligence",
                "fraction": 0.96,
                "message": "Extracting meeting intelligence",
            }
        )
        tracker.transition("Meeting Intelligence")
        stage_ref["current"] = "Meeting Intelligence"
        ai_backend = AIBackend()
        intel_engine = MeetingIntelligenceEngine(
            ai_backend=ai_backend if ai_backend.configured else None
        )
        intel_result = intel_engine.analyze(
            result.transcript, profile_name=loaded_profile.name
        )

        _fraction_hwm["value"] = max(_fraction_hwm["value"], 0.97)
        push_event(
            {
                "type": "stage",
                "stage": "Generating Files",
                "fraction": 0.97,
                "message": "Writing output files",
            }
        )
        tracker.transition("Generating Files")
        stage_ref["current"] = "Generating Files"

        stem = "transcript"
        conf_threshold = 0.65

        # Every registered transcript format is written, not just markdown/json.
        #   markdown — human-readable, the default view
        #   json     — re-export without re-running inference (speaker renames)
        #   txt      — plain text for pasting into other tools
        #   srt/vtt  — subtitle tracks for video editors and players
        # These all derive from the same finished Transcript, so generating
        # them costs milliseconds; withholding them only forced users to
        # re-run a GPU job to get a different file extension.
        #
        # A failure here must never be silent: the job would otherwise report
        # "completed" with no transcript and nothing in the logs explaining why.
        # We still don't abort (later artifacts and reports are worth
        # producing), but the failure is logged and surfaced as a job warning.
        for fmt in TRANSCRIPT_EXPORT_FORMATS:
            try:
                exporter_cls = get_exporter(fmt)
                export_config = settings.pipeline.export
                content = exporter_cls().export(result.transcript, export_config)
                (output_dir / f"{stem}{exporter_cls.extension}").write_bytes(content)
            except Exception as exc:
                _log.error(
                    "Failed to write '%s' transcript for job %s: %s",
                    fmt, job_id, exc, exc_info=True,
                )
                pipeline_warnings.append(f"Transcript export '{fmt}' failed: {exc}")

        def _write(path: Path, content: str | bytes) -> None:
            """Write artifact, logging on failure. Never raises — a failed export
            must not abort the job after 10+ minutes of GPU processing."""
            try:
                if isinstance(content, bytes):
                    path.write_bytes(content)
                else:
                    path.write_text(content, encoding="utf-8")
            except Exception as exc:
                _log.warning("Failed to write artifact %s: %s", path.name, exc)

        if result.corrections:
            _write(
                output_dir / f"{stem}.corrections.md",
                export_corrections(result.corrections, loaded_profile.name, audio_path.name),
            )

        _write(
            output_dir / f"{stem}.stats.json",
            export_stats(
                result.transcript, result.elapsed_seconds, result.corrections,
                loaded_profile.name, low_conf_threshold=conf_threshold,
            ),
        )
        _write(
            output_dir / f"{stem}.quality.md",
            export_quality(
                result.transcript, result.corrections, loaded_profile.name,
                low_conf_threshold=conf_threshold,
            ),
        )
        _write(output_dir / f"{stem}.summary.md", export_summary(intel_result, result.transcript))
        _write(output_dir / f"{stem}.action_items.md", export_action_items(intel_result, result.transcript))
        _write(output_dir / f"{stem}.decisions.md", export_decisions(intel_result, result.transcript))
        _write(output_dir / f"{stem}.questions.md", export_questions(intel_result, result.transcript))
        _write(output_dir / f"{stem}.timeline.md", export_timeline(intel_result, result.transcript))
        _write(output_dir / f"{stem}.entities.json", export_entities(intel_result))

        metrics_json = "{}"
        try:
            metrics_json = export_metrics(
                result.transcript, result.elapsed_seconds, result.corrections,
                loaded_profile.name, intel_result, low_conf_threshold=conf_threshold,
            )
            _write(output_dir / f"{stem}.metrics.json", metrics_json)
        except Exception as exc:
            _log.warning("Failed to export metrics: %s", exc)

        try:
            _write(
                output_dir / f"{stem}.review.md",
                generate_review(result.transcript, result.corrections, low_conf_threshold=conf_threshold),
            )
        except Exception as exc:
            _log.warning("Failed to generate review: %s", exc)

        try:
            index_data = generate_index(
                result.transcript, intel_result, result.corrections, loaded_profile.name
            )
            _write(
                output_dir / f"{stem}.index.json",
                json.dumps(index_data, indent=2, ensure_ascii=False),
            )
        except Exception as exc:
            _log.warning("Failed to generate index: %s", exc)

        # Extract confidence from metrics
        try:
            metrics_dict = json.loads(metrics_json)
            _run_state["metrics"] = metrics_dict
            _run_state["avg_confidence"] = metrics_dict.get("avg_word_confidence")
        except Exception as exc:
            # Telemetry only — the transcript is already written, so a bad
            # metrics blob must not fail the job. Logged so it isn't invisible.
            _log.warning("Could not parse metrics JSON for job %s: %s", job_id, exc)

        push_event(
            {
                "type": "stage",
                "stage": "Creating Bundle",
                "fraction": 0.98,
                "message": "Creating bundle",
            }
        )
        tracker.transition("Creating Bundle")
        stage_ref["current"] = "Creating Bundle"

        bundle_path = output_dir / "transcript_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(output_dir.iterdir()):
                if (
                    f.suffix in {".md", ".json", ".txt", ".srt", ".vtt"}
                    and f != bundle_path
                ):
                    zf.write(f, f.name)

        # Validate outputs AFTER bundle is written so transcript_bundle.zip
        # is not falsely reported missing in processing_report.json.
        output_validation = _validate_outputs(output_dir)
        _run_state["output_validation"] = output_validation

        total_so_far = time.monotonic() - pipeline_start
        tracker.close()
        stage_report = tracker.report(total_so_far)

        _write_resource_report(output_dir, job_id, resource_samples)
        _write_processing_report(
            output_dir=output_dir,
            job_id=job_id,
            audio_filename=audio_path.name,
            audio_duration_s=_run_state["audio_duration_s"],
            started_at=started_at,
            completed_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            total_s=total_so_far,
            status="completed",
            upload_s=upload_seconds,
            stage_report=stage_report,
            model_info=_run_state["model_info"],
            output_validation=output_validation,
            temp_validation={
                "audio_deleted": False,
                "temp_dir_deleted": False,
            },  # updated in finally
            error=None,
        )
        _write_production_report(
            output_dir=output_dir,
            audio_filename=audio_path.name,
            started_at=started_at,
            total_s=total_so_far,
            status="completed",
            stage_report=stage_report,
            resource_samples=resource_samples,
            output_validation=output_validation,
            temp_validation={"audio_deleted": False, "temp_dir_deleted": False},
            pipeline_warnings=pipeline_warnings,
            error=None,
            word_count=_run_state["word_count"],
            speaker_count=_run_state["speaker_count"],
            corrections_count=_run_state["corrections_count"],
            avg_confidence=_run_state.get("avg_confidence"),
        )

        artifacts = sorted(
            f.name
            for f in output_dir.iterdir()
            if f.is_file()
            and f.suffix in {".md", ".json", ".zip", ".txt", ".srt", ".vtt"}
        )
        metrics = json.loads(metrics_json)

        job_manager.update(
            job_id,
            status=JobStatus.completed,
            completed_at=datetime.now(tz=UTC),
            artifacts=artifacts,
            metrics=metrics,
        )
        _run_state["status"] = "completed"
        push_event({"type": "completed", "fraction": 1.0})

    except Exception as exc:
        _stop_ticker()
        msg = _humanize_error(exc)

        # Full traceback and context for debugging
        elapsed = time.monotonic() - pipeline_start
        try:
            import psutil  # noqa: PLC0415

            rss_gb = psutil.Process().memory_info().rss / 1_073_741_824
            mem_note = f", RAM={rss_gb:.2f}GB"
        except Exception:
            mem_note = ""
        _log.error(
            "Pipeline failed for job %s at stage=%s elapsed=%.1fs%s:\n%s",
            job_id,
            tracker.current(),
            elapsed,
            mem_note,
            _traceback.format_exc(),
        )
        pipeline_warnings.append(
            f"Pipeline failed at stage '{tracker.current()}': {type(exc).__name__}: {exc}"
        )

        _run_state["error"] = msg
        _run_state["status"] = "failed"

        job_manager.update(
            job_id,
            status=JobStatus.failed,
            completed_at=datetime.now(tz=UTC),
            error=msg,
        )
        push_event({"type": "error", "message": msg})

        # Write failure report
        try:
            total_s = time.monotonic() - pipeline_start
            tracker.close()
            stage_report = tracker.report(total_s)
            _write_processing_report(
                output_dir=output_dir,
                job_id=job_id,
                audio_filename=audio_path.name,
                audio_duration_s=_run_state["audio_duration_s"],
                started_at=started_at,
                completed_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                total_s=total_s,
                status="failed",
                upload_s=upload_seconds,
                stage_report=stage_report,
                model_info=_run_state["model_info"],
                output_validation=_run_state["output_validation"],
                temp_validation={"audio_deleted": False, "temp_dir_deleted": False},
                error=msg,
            )
            _write_production_report(
                output_dir=output_dir,
                audio_filename=audio_path.name,
                started_at=started_at,
                total_s=total_s,
                status="failed",
                stage_report=stage_report,
                resource_samples=resource_samples,
                output_validation=_run_state["output_validation"],
                temp_validation={"audio_deleted": False, "temp_dir_deleted": False},
                pipeline_warnings=pipeline_warnings,
                error=msg,
                word_count=_run_state["word_count"],
                speaker_count=_run_state["speaker_count"],
                corrections_count=_run_state["corrections_count"],
                avg_confidence=_run_state.get("avg_confidence"),
            )
            _write_resource_report(output_dir, job_id, resource_samples)
        except Exception as report_exc:
            _log.warning("Could not write failure reports: %s", report_exc)

    finally:
        _stop_monitor.set()
        _stop_watchdog.set()
        _stop_ticker()

        # Delete uploaded audio, orphaned chunk files, and job temp dir (privacy + disk).
        audio_deleted = False
        temp_dir_deleted = False
        orphaned_chunks: list[str] = []
        try:
            audio_path.unlink(missing_ok=True)
            audio_deleted = not audio_path.exists()
            parent = audio_path.parent

            # Delete any orphaned chunk WAVs written by split_audio but not cleaned
            # up (e.g. if the pipeline crashed during parallel diarization).
            if parent.exists():
                chunks = list(parent.glob("_chunk_*.wav"))
                if chunks:
                    orphaned_chunks = [str(c) for c in chunks]
                    _log.warning(
                        "Deleting %d orphaned chunk file(s) in %s", len(chunks), parent
                    )
                    for chunk_file in chunks:
                        chunk_file.unlink(missing_ok=True)

            # Remove the job temp dir if it is now empty.
            if parent.name != "temp" and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            temp_dir_deleted = not parent.exists()
        except Exception as exc:
            # Never re-raise from the finally block — that would mask the real
            # pipeline exception. But do log it: a silent cleanup failure means
            # audio is left on disk, which is a privacy and disk-usage concern.
            _log.warning("Temp cleanup failed for job %s: %s", job_id, exc)

        _log.info(
            "Temp cleanup: audio_deleted=%s, temp_dir_deleted=%s, orphaned_chunks=%d",
            audio_deleted,
            temp_dir_deleted,
            len(orphaned_chunks),
        )
        if not audio_deleted:
            _log.warning("Audio file NOT deleted: %s", audio_path)

        _run_state["temp_validation"] = {
            "audio_deleted": audio_deleted,
            "temp_dir_deleted": temp_dir_deleted,
            "orphaned_chunks": len(orphaned_chunks),
        }


async def run_pipeline_async(
    job_id: str,
    audio_path: Path,
    profile_name: str,
    mode: str,
    output_dir: Path,
    timestamps: str,
    highlight_confidence: bool,
    progress_queue: asyncio.Queue[dict[str, Any]],
    upload_seconds: float = 0.0,
) -> None:
    """Schedule pipeline in thread pool, feeding events into the queue."""
    loop = asyncio.get_running_loop()

    def push(event: dict[str, Any]) -> None:
        # The suppress must wrap put_nowait, not call_soon_threadsafe.
        # call_soon_threadsafe schedules the callback and returns immediately —
        # QueueFull is raised later inside the event loop thread where the outer
        # suppress context is no longer active.  Moving suppress inside the
        # callback ensures it catches the exception where it actually fires.
        def _put() -> None:
            with contextlib.suppress(asyncio.QueueFull):
                progress_queue.put_nowait(event)
        loop.call_soon_threadsafe(_put)

    await loop.run_in_executor(
        _executor,
        _run_pipeline_sync,
        job_id,
        audio_path,
        profile_name,
        mode,
        output_dir,
        timestamps,
        highlight_confidence,
        push,
        upload_seconds,
    )
