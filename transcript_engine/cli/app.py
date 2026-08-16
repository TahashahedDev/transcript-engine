"""
Transcript Engine CLI

Thin Typer adapter over Pipeline. No business logic here.
All decisions are made by the pipeline and its components.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from transcript_engine.config.loader import load_settings
from transcript_engine.exporters.corrections import export_corrections
from transcript_engine.exporters.quality import export_quality
from transcript_engine.exporters.registry import available_formats, get_exporter
from transcript_engine.exporters.stats import export_stats
from transcript_engine.logging import configure_logging, get_logger
from transcript_engine.model_registry.registry import ModelRegistry
from transcript_engine.models.pipeline import PipelineResult
from transcript_engine.models.transcript import Transcript
from transcript_engine.pipeline.orchestrator import Pipeline
from transcript_engine.profiles.loader import list_profiles, load_profile

app = typer.Typer(
    name="transcript-engine",
    help="Production-quality local AI meeting transcription engine.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
logger = get_logger(__name__)


_STAGE_LABELS: dict[str, str] = {
    "Preprocessing audio": "Audio",
    "Audio ready": "Audio",
    "Loading Whisper model": "Loading Whisper",
    "Transcribing": "Transcribing",
    "Aligning timestamps": "Aligning",
    "Transcription complete": "Transcribing",
    "Loading diarization model": "Loading Diarizer",
    "Running speaker diarization": "Diarizing",
    "Diarization complete": "Diarizing",
    "Merging speakers and words": "Merging",
    "Running processors": "Processing",
    "Done": "Exporting",
}

_STAGE_ORDER = [
    "Audio",
    "Loading Whisper",
    "Transcribing",
    "Aligning",
    "Loading Diarizer",
    "Diarizing",
    "Merging",
    "Processing",
    "Exporting",
]


def _infer_label(message: str) -> str | None:
    """Map partial progress messages to stage labels by keyword."""
    msg = message.lower()
    if "audio" in msg or "preprocess" in msg:
        return "Audio"
    if "whisper" in msg or "loading" in msg and "diariz" not in msg:
        return "Loading Whisper"
    if "transcrib" in msg:
        return "Transcribing"
    if "align" in msg:
        return "Aligning"
    if "diariz" in msg and "load" in msg:
        return "Loading Diarizer"
    if "diariz" in msg or "speaker" in msg:
        return "Diarizing"
    if "merg" in msg:
        return "Merging"
    if "process" in msg or "vocab" in msg or "clean" in msg:
        return "Processing"
    if "export" in msg or "done" in msg:
        return "Exporting"
    return None


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description:<24}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


@app.command()
def transcribe(
    input_file: Annotated[
        Path, typer.Argument(help="Audio or video file to transcribe.")
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Comma-separated export formats: markdown,json,txt,srt,vtt",
        ),
    ] = "markdown,json",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-d", help="Directory for output files."),
    ] = Path("output"),
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            "-p",
            help="Profile name (e.g. banking, generic). Defaults to generic.",
        ),
    ] = "generic",
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code (e.g. 'en'). Auto-detected if omitted.",
        ),
    ] = None,
    speakers: Annotated[
        int | None,
        typer.Option(
            "--speakers",
            "-s",
            help="Expected number of speakers. Helps diarization accuracy.",
        ),
    ] = None,
    max_speakers: Annotated[
        int | None,
        typer.Option("--max-speakers", help="Maximum number of speakers."),
    ] = None,
    no_diarize: Annotated[
        bool,
        typer.Option(
            "--no-diarization", help="Skip diarization (single-speaker mode)."
        ),
    ] = False,
    timestamps: Annotated[
        str,
        typer.Option(
            "--timestamps",
            "-t",
            help="Timestamp mode: none, speaker, paragraph, minute.",
        ),
    ] = "none",
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence-threshold",
            help="Highlight words below this confidence (0.0–1.0). Requires --highlight-confidence.",
        ),
    ] = 0.65,
    highlight_confidence: Annotated[
        bool,
        typer.Option(
            "--highlight-confidence",
            help="Highlight low-confidence words as **[word?]** in markdown.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Enable debug logging."),
    ] = False,
) -> None:
    """Transcribe an audio or video file."""
    valid_ts_modes = ("none", "speaker", "paragraph", "minute")
    if timestamps not in valid_ts_modes:
        console.print(
            f"[red]Invalid --timestamps value:[/red] '{timestamps}'. Choose from: {', '.join(valid_ts_modes)}"
        )
        raise typer.Exit(1)

    if not input_file.exists():
        console.print(f"[red]File not found:[/red] {input_file}")
        raise typer.Exit(1)

    settings = load_settings()
    configure_logging("DEBUG" if verbose else settings.log_level, settings.log_file)

    # Load and validate the profile early so we fail fast before model loading
    profiles_dir = settings.pipeline.processing.profiles_dir
    try:
        loaded_profile = load_profile(profile, profiles_dir=profiles_dir)
    except FileNotFoundError:
        available = list_profiles(profiles_dir)
        console.print(
            f"[red]Profile not found:[/red] '{profile}'. "
            f"Available: {', '.join(available) or 'none'}"
        )
        raise typer.Exit(1) from None

    settings = settings.model_copy(
        update={
            "pipeline": settings.pipeline.model_copy(
                update={
                    "transcription": settings.pipeline.transcription.model_copy(
                        update={"language": language} if language else {}
                    ),
                    "diarization": settings.pipeline.diarization.model_copy(
                        update={
                            **(
                                {"min_speakers": speakers, "max_speakers": speakers}
                                if speakers
                                else {}
                            ),
                            **({"max_speakers": max_speakers} if max_speakers else {}),
                            "enabled": not no_diarize,
                        }
                    ),
                    "processing": settings.pipeline.processing.model_copy(
                        update={"profile": profile}
                    ),
                    "export": settings.pipeline.export.model_copy(
                        update={
                            "formats": [f.strip() for f in output.split(",")],
                            "output_dir": output_dir,
                            "timestamp_mode": timestamps,
                            "highlight_confidence": highlight_confidence,
                            "low_confidence_threshold": confidence_threshold,
                        }
                    ),
                }
            )
        }
    )

    registry = ModelRegistry(settings)
    pipeline = Pipeline(settings.pipeline, registry, profile=loaded_profile)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Input:[/bold] {input_file.name}\n"
            f"[bold]Profile:[/bold] {loaded_profile.name}\n"
            f"[bold]Formats:[/bold] {output}\n"
            f"[bold]Output:[/bold] {output_dir}",
            title="[bold green]Transcript Engine[/bold green]",
            expand=False,
        )
    )

    with _make_progress() as progress:
        stage_tasks = {
            label: progress.add_task(label, total=100, completed=0)
            for label in _STAGE_ORDER
        }
        current_stage: list[str] = [_STAGE_ORDER[0]]

        def on_progress(fraction: float, message: str) -> None:
            label = _STAGE_LABELS.get(message) or _infer_label(message)
            if label and label != current_stage[0]:
                if current_stage[0] in stage_tasks:
                    progress.update(stage_tasks[current_stage[0]], completed=100)
                current_stage[0] = label
            if label and label in stage_tasks:
                completed = min(99, int(fraction * 100))
                progress.update(
                    stage_tasks[label], completed=completed, description=label
                )

        try:
            result = pipeline.run(input_file, on_progress=on_progress)
        except Exception as exc:
            console.print(f"\n[red]Pipeline failed:[/red] {exc}")
            if verbose:
                import traceback

                console.print(traceback.format_exc())
            registry.release_all()
            raise typer.Exit(1) from exc
        finally:
            registry.release_all()
            for t in stage_tasks.values():
                progress.update(t, completed=100)

    import json  # noqa: PLC0415

    from transcript_engine.intelligence.ai_backend import AIBackend  # noqa: PLC0415
    from transcript_engine.intelligence.engine import (
        MeetingIntelligenceEngine,
    )  # noqa: PLC0415
    from transcript_engine.intelligence.exports import (  # noqa: PLC0415
        export_action_items,
        export_decisions,
        export_entities,
        export_metrics,
        export_questions,
        export_summary,
        export_timeline,
    )
    from transcript_engine.review.index import generate_index  # noqa: PLC0415
    from transcript_engine.review.report import generate_review  # noqa: PLC0415

    stem = input_file.stem
    generated: list[tuple[str, str]] = []  # (label, path_str)

    # ── Core transcript formats ────────────────────────────────────────────
    _export_result(result.transcript, settings.pipeline.export.formats, output_dir)
    for fmt in settings.pipeline.export.formats:
        try:
            exporter_cls = get_exporter(fmt)
            generated.append(
                (
                    f"Transcript ({fmt})",
                    str(output_dir / f"{stem}{exporter_cls.extension}"),
                )
            )
        except KeyError:
            pass

    # ── Corrections ────────────────────────────────────────────────────────
    if result.corrections:
        corrections_path = output_dir / f"{stem}.corrections.md"
        corrections_path.write_text(
            export_corrections(
                result.corrections, loaded_profile.name, input_file.name
            ),
            encoding="utf-8",
        )
        generated.append(
            (f"Corrections ({len(result.corrections)} applied)", str(corrections_path))
        )

    # ── Stats + Quality ────────────────────────────────────────────────────
    stats_path = output_dir / f"{stem}.stats.json"
    stats_path.write_text(
        export_stats(
            result.transcript,
            result.elapsed_seconds,
            result.corrections,
            loaded_profile.name,
            low_conf_threshold=confidence_threshold,
        ),
        encoding="utf-8",
    )
    quality_path = output_dir / f"{stem}.quality.md"
    quality_path.write_text(
        export_quality(
            result.transcript,
            result.corrections,
            loaded_profile.name,
            low_conf_threshold=confidence_threshold,
        ),
        encoding="utf-8",
    )

    # ── Meeting intelligence ───────────────────────────────────────────────
    ai_backend = AIBackend()
    intel_engine = MeetingIntelligenceEngine(
        ai_backend=ai_backend if ai_backend.configured else None
    )
    intel_result = intel_engine.analyze(
        result.transcript, profile_name=loaded_profile.name
    )

    (output_dir / f"{stem}.summary.md").write_text(
        export_summary(intel_result, result.transcript), encoding="utf-8"
    )
    generated.append(("Meeting summary", str(output_dir / f"{stem}.summary.md")))

    (output_dir / f"{stem}.action_items.md").write_text(
        export_action_items(intel_result, result.transcript), encoding="utf-8"
    )
    generated.append(
        (
            f"Action items ({len(intel_result.action_items)} found)",
            str(output_dir / f"{stem}.action_items.md"),
        )
    )

    (output_dir / f"{stem}.decisions.md").write_text(
        export_decisions(intel_result, result.transcript), encoding="utf-8"
    )
    generated.append(
        (
            f"Decisions ({len(intel_result.decisions)} found)",
            str(output_dir / f"{stem}.decisions.md"),
        )
    )

    (output_dir / f"{stem}.questions.md").write_text(
        export_questions(intel_result, result.transcript), encoding="utf-8"
    )
    generated.append(
        (
            f"Questions ({len(intel_result.questions)} open)",
            str(output_dir / f"{stem}.questions.md"),
        )
    )

    (output_dir / f"{stem}.timeline.md").write_text(
        export_timeline(intel_result, result.transcript), encoding="utf-8"
    )
    generated.append(("Timeline", str(output_dir / f"{stem}.timeline.md")))

    (output_dir / f"{stem}.entities.json").write_text(
        export_entities(intel_result), encoding="utf-8"
    )
    generated.append(("Entities", str(output_dir / f"{stem}.entities.json")))

    metrics_path = output_dir / f"{stem}.metrics.json"
    metrics_path.write_text(
        export_metrics(
            result.transcript,
            result.elapsed_seconds,
            result.corrections,
            loaded_profile.name,
            intel_result,
            low_conf_threshold=confidence_threshold,
        ),
        encoding="utf-8",
    )
    generated.append(("Metrics", str(metrics_path)))

    # ── Review report + index ──────────────────────────────────────────────
    review_path = output_dir / f"{stem}.review.md"
    review_path.write_text(
        generate_review(
            result.transcript,
            result.corrections,
            low_conf_threshold=confidence_threshold,
        ),
        encoding="utf-8",
    )
    generated.append(("Review report", str(review_path)))

    index_data = generate_index(
        result.transcript, intel_result, result.corrections, loaded_profile.name
    )
    index_path = output_dir / f"{stem}.index.json"
    index_path.write_text(
        json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generated.append(("Search index", str(index_path)))

    if result.warnings:
        for warning in result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

    _print_completion(result, output_dir, generated, intel_result)


@app.command()
def rename(
    transcript_json: Annotated[Path, typer.Argument(help="JSON transcript file.")],
    mappings: Annotated[
        list[str],
        typer.Argument(help="Speaker mappings: SPEAKER_00=John SPEAKER_01=Neel"),
    ],
    reexport: Annotated[
        bool,
        typer.Option("--reexport", "-r", help="Re-export all formats after renaming."),
    ] = False,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Formats to re-export (if --reexport)."),
    ] = "markdown,json",
) -> None:
    """Rename speakers in an existing transcript JSON and optionally re-export."""
    if not transcript_json.exists():
        console.print(f"[red]File not found:[/red] {transcript_json}")
        raise typer.Exit(1)

    speaker_map: dict[str, str] = {}
    for mapping in mappings:
        if "=" not in mapping:
            console.print(
                f"[red]Invalid mapping (expected SPEAKER_00=Name):[/red] {mapping}"
            )
            raise typer.Exit(1)
        speaker_id, display_name = mapping.split("=", 1)
        speaker_map[speaker_id.strip()] = display_name.strip()

    transcript = Transcript.model_validate_json(
        transcript_json.read_text(encoding="utf-8")
    )
    updated_speakers = {**transcript.speakers, **speaker_map}
    transcript = transcript.with_speakers(updated_speakers)

    updated_json = transcript.model_dump_json(indent=2)
    transcript_json.write_text(updated_json, encoding="utf-8")
    console.print(f"[green]Updated:[/green] {transcript_json}")

    for sid, name in speaker_map.items():
        console.print(f"  {sid} → {name}")

    if reexport:
        load_settings()
        output_dir = transcript_json.parent
        formats = [f.strip() for f in output.split(",")]
        _export_result(transcript, formats, output_dir)


@app.command()
def export(
    transcript_json: Annotated[Path, typer.Argument(help="JSON transcript file.")],
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Comma-separated formats."),
    ] = "markdown",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-d", help="Output directory."),
    ] = Path("output"),
) -> None:
    """Export an existing transcript JSON to additional formats."""
    if not transcript_json.exists():
        console.print(f"[red]File not found:[/red] {transcript_json}")
        raise typer.Exit(1)

    transcript = Transcript.model_validate_json(
        transcript_json.read_text(encoding="utf-8")
    )
    load_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in output.split(",")]
    _export_result(transcript, formats, output_dir)


@app.command("build-profile")
def build_profile_cmd(
    transcripts_dir: Annotated[
        Path, typer.Argument(help="Directory of transcript JSON files.")
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output suggested_vocabulary.yaml path."),
    ] = Path("suggested_vocabulary.yaml"),
    min_frequency: Annotated[
        int,
        typer.Option("--min-freq", help="Minimum occurrence count to include a term."),
    ] = 3,
) -> None:
    """Analyze existing transcripts and suggest vocabulary entries for a new profile."""
    if not transcripts_dir.exists():
        console.print(f"[red]Directory not found:[/red] {transcripts_dir}")
        raise typer.Exit(1)

    json_files = list(transcripts_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No .json files found in {transcripts_dir}[/yellow]")
        raise typer.Exit(0)

    word_counts: dict[str, int] = collections.Counter()
    total_words = 0

    for jf in json_files:
        try:
            t = Transcript.model_validate_json(jf.read_text(encoding="utf-8"))
            for seg in t.segments:
                for w in seg.words:
                    token = w.text.strip(".,!?;:")
                    if token:
                        word_counts[token] += 1
                        total_words += 1
        except Exception as exc:
            console.print(f"[yellow]Skipped {jf.name}:[/yellow] {exc}")

    # Candidates: all-caps words OR capitalized multi-char words that appear >= min_frequency
    _all_caps = re.compile(r"^[A-Z][A-Z0-9_]{1,}$")
    _capitalized = re.compile(r"^[A-Z][a-z]{2,}$")

    candidates = {
        word: count
        for word, count in word_counts.items()
        if count >= min_frequency
        and (_all_caps.match(word) or _capitalized.match(word))
    }

    if not candidates:
        console.print(
            "[yellow]No candidate terms found meeting the frequency threshold.[/yellow]"
        )
        raise typer.Exit(0)

    lines = [
        "# Suggested vocabulary — review and edit before using",
        f"# Analyzed {len(json_files)} transcript(s), {total_words} total words",
        f"# Showing {len(candidates)} terms appearing >= {min_frequency} times",
        "# For each entry: add aliases (how Whisper mis-transcribes the term)",
        "",
    ]
    for word, count in sorted(candidates.items(), key=lambda x: -x[1]):
        lines += [
            f"{word}:  # appeared {count}x",
            "  aliases: []",
            "  confidence: 0.90",
            "  context: []",
            "",
        ]

    output.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Suggested vocabulary written to:[/green] {output}")
    console.print(
        f"[dim]Found {len(candidates)} candidate terms. "
        "Review, add aliases, then save to profiles/<name>/vocabulary.yaml[/dim]"
    )


@app.command("profiles")
def list_profiles_cmd() -> None:
    """List available profiles."""
    settings = load_settings()
    available = list_profiles(settings.pipeline.processing.profiles_dir)
    if not available:
        console.print("[yellow]No profiles found.[/yellow]")
        return
    table = Table(title="Available Profiles", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Industry", style="dim")
    for name in available:
        try:
            p = load_profile(name, settings.pipeline.processing.profiles_dir)
            table.add_row(p.name, p.description, p.industry)
        except Exception:
            table.add_row(name, "—", "—")
    console.print(table)


@app.command("config-show")
def config_show() -> None:
    """Show the current resolved configuration."""
    settings = load_settings()
    table = Table(title="Transcript Engine Configuration", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    flat = _flatten(settings.model_dump(), prefix="")
    for key, value in sorted(flat.items()):
        table.add_row(key, str(value))

    console.print(table)


def _export_result(
    transcript: Transcript,
    formats: list[str],
    output_dir: Path,
) -> None:
    settings = load_settings()
    export_config = settings.pipeline.export

    for fmt in formats:
        try:
            exporter_cls = get_exporter(fmt)
            exporter = exporter_cls()
            content = exporter.export(transcript, export_config)

            stem = transcript.audio_path.stem
            output_path = output_dir / f"{stem}{exporter_cls.extension}"
            output_path.write_bytes(content)
            console.print(f"[green]Exported:[/green] {output_path}")
        except KeyError:
            console.print(
                f"[red]Unknown format:[/red] {fmt}. "
                f"Available: {', '.join(available_formats())}"
            )
        except Exception as exc:
            console.print(f"[red]Export failed ({fmt}):[/red] {exc}")


def _print_completion(
    result: PipelineResult,
    output_dir: Path,
    generated: list[tuple[str, str]],
    intel_result: Any,
) -> None:
    """Print a clean completion panel with ✓ checklist and stats."""
    rtf = (
        f"{result.audio.duration / result.elapsed_seconds:.1f}x"
        if result.elapsed_seconds > 0
        else "—"
    )
    dur_m, dur_s = divmod(int(result.audio.duration), 60)
    proc_m, proc_s = divmod(int(result.elapsed_seconds), 60)

    checks = "\n".join(f"  [green]✓[/green] {label}" for label, _ in generated)

    stats = (
        f"[bold]Audio:[/bold] {dur_m}m {dur_s}s  "
        f"[bold]Processing:[/bold] {proc_m}m {proc_s}s  "
        f"[bold]RTF:[/bold] {rtf}\n"
        f"[bold]Words:[/bold] {result.transcript.word_count}  "
        f"[bold]Speakers:[/bold] {len(result.transcript.speakers)}  "
        f"[bold]Segments:[/bold] {len(result.transcript.segments)}\n"
        f"[bold]Output:[/bold] {output_dir.resolve()}"
    )

    speakers = "  ".join(
        f"[cyan]{sid}[/cyan] → {name}"
        for sid, name in result.transcript.speakers.items()
    )

    body = f"{checks}\n\n{stats}"
    if speakers:
        body += f"\n[bold]Speakers:[/bold] {speakers}"

    console.print(Panel(body, title="[bold green]Complete[/bold green]", expand=False))


def _flatten(d: dict[str, Any], prefix: str) -> dict[str, object]:
    items: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items


@app.command()
def review(
    transcript_json: Annotated[Path, typer.Argument(help="Transcript JSON file.")],
    corrections_index: Annotated[
        Path | None,
        typer.Option(
            "--corrections", help="Optional index.json to load corrections from."
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-d",
            help="Output directory. Defaults to transcript's parent.",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold", help="Confidence threshold for flagging (0.0–1.0)."
        ),
    ] = 0.65,
) -> None:
    """Regenerate meeting.review.md from an existing transcript JSON."""
    import json  # noqa: PLC0415

    from transcript_engine.review.report import generate_review  # noqa: PLC0415

    if not transcript_json.exists():
        console.print(f"[red]File not found:[/red] {transcript_json}")
        raise typer.Exit(1)

    transcript = Transcript.model_validate_json(
        transcript_json.read_text(encoding="utf-8")
    )

    corrections = []
    if corrections_index is not None:
        if not corrections_index.exists():
            console.print(f"[red]Index not found:[/red] {corrections_index}")
            raise typer.Exit(1)
        idx = json.loads(corrections_index.read_text(encoding="utf-8"))
        from transcript_engine.models.correction import (
            CorrectionRecord,
        )  # noqa: PLC0415

        for rec in idx.get("corrections", []):
            corrections.append(
                CorrectionRecord(
                    original=rec["original"],
                    replacement=rec["replacement"],
                    reason=rec["reason"],
                    confidence=rec["confidence"],
                    profile=rec["profile"],
                    segment_id=rec["segment_id"],
                )
            )

    out_dir = output_dir or transcript_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{transcript_json.stem}.review.md"
    report_path.write_text(
        generate_review(transcript, corrections, low_conf_threshold=threshold),
        encoding="utf-8",
    )
    console.print(f"[green]Review report:[/green] {report_path}")


@app.command()
def search(
    index_json: Annotated[Path, typer.Argument(help="Meeting index JSON file.")],
    query: Annotated[str, typer.Argument(help="Search term or phrase.")],
    max_results: Annotated[
        int,
        typer.Option("--max", "-n", help="Maximum results to show."),
    ] = 20,
    context_words: Annotated[
        int,
        typer.Option("--context", "-c", help="Words of context to show around match."),
    ] = 8,
) -> None:
    """Search a transcript index for a word or phrase."""
    import json  # noqa: PLC0415

    from transcript_engine.review.search import search_index  # noqa: PLC0415

    if not index_json.exists():
        console.print(f"[red]File not found:[/red] {index_json}")
        raise typer.Exit(1)

    index_data = json.loads(index_json.read_text(encoding="utf-8"))
    results = search_index(index_data, query, context_words=context_words)

    if not results:
        console.print(f"No results found for '{query}'.")
        return

    results = results[:max_results]
    table = Table(title=f"Search results for '{query}'", show_header=True)
    table.add_column("Time", style="cyan", width=8)
    table.add_column("Type", style="dim", width=12)
    table.add_column("Speaker", style="bold", width=16)
    table.add_column("Context", style="white")

    def _ts(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"[{m:02d}:{s:02d}]"

    for r in results:
        table.add_row(
            _ts(r["timestamp"]),
            r["type"],
            r["speaker_name"],
            r["context"] or r["text"][:120],
        )

    console.print(table)
    console.print(f"[dim]{len(results)} result(s) shown.[/dim]")


@app.command()
def bundle(
    directory: Annotated[
        Path, typer.Argument(help="Directory containing meeting artifacts.")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output ZIP path. Defaults to {dir}/{stem}_bundle.zip.",
        ),
    ] = None,
) -> None:
    """Package all meeting artifacts into a deliverable ZIP bundle."""
    import zipfile  # noqa: PLC0415

    if not directory.exists() or not directory.is_dir():
        console.print(f"[red]Directory not found:[/red] {directory}")
        raise typer.Exit(1)

    extensions = {".md", ".json", ".txt", ".srt", ".vtt"}
    files = sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix in extensions and not f.name.endswith("_bundle.zip")
    )

    if not files:
        console.print("[yellow]No artifact files found in directory.[/yellow]")
        raise typer.Exit(0)

    stem = _find_bundle_stem(directory)
    zip_path = output or directory / f"{stem}_bundle.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    table = Table(title="Bundle Contents", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Size", style="dim")
    for f in files:
        size = f.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
        table.add_row(f.name, size_str)

    console.print(table)
    console.print(f"[green]Bundle:[/green] {zip_path} ({len(files)} files)")


def _find_bundle_stem(directory: Path) -> str:
    """Find the base stem for the bundle ZIP from the artifacts directory."""
    known_suffixes = {
        ".summary",
        ".action_items",
        ".decisions",
        ".questions",
        ".timeline",
        ".metrics",
        ".entities",
        ".stats",
        ".index",
        ".corrections",
        ".quality",
        ".review",
    }
    # Prefer a JSON file whose stem has no dots (clean transcript JSON)
    for f in sorted(directory.glob("*.json")):
        if not Path(f.stem).suffix:
            return f.stem
    # Fallback: strip known suffixes from first JSON file
    for f in sorted(directory.glob("*.json")):
        stem = f.stem
        for suf in known_suffixes:
            if stem.endswith(suf):
                base = stem[: -len(suf)]
                if base:
                    return base
    # Last resort: use directory name
    return directory.name


if __name__ == "__main__":
    app()
