from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from transcript_engine.config.paths import project_root, resolve_path


class TranscriptionMode(StrEnum):
    fast = "fast"
    balanced = "balanced"
    high_accuracy = "high_accuracy"
    archive = "archive"


@dataclass
class ModeSpec:
    model_id: str
    skip_alignment: bool
    beam_size: int
    best_of: int
    temperatures: list[float]
    processors: list[str]
    label: str
    description: str


PIPELINE_MODES: dict[str, ModeSpec] = {
    TranscriptionMode.fast: ModeSpec(
        model_id="Systran/faster-distil-whisper-large-v3",
        skip_alignment=True,
        beam_size=1,
        best_of=1,
        temperatures=[0.0],
        processors=["context_correction", "transcript_cleanup", "speaker_formatting"],
        label="Fast",
        description="Distil-Whisper · no forced alignment · fastest processing",
    ),
    TranscriptionMode.balanced: ModeSpec(
        model_id="large-v3-turbo",
        skip_alignment=False,
        beam_size=2,
        best_of=1,
        temperatures=[0.0],
        processors=[
            "vocabulary_correction",
            "context_correction",
            "transcript_cleanup",
            "transcript_intelligence",
            "speaker_formatting",
        ],
        label="Balanced",
        description="Whisper turbo · word timestamps · meeting intelligence",
    ),
    TranscriptionMode.high_accuracy: ModeSpec(
        model_id="large-v3",
        skip_alignment=False,
        beam_size=5,
        best_of=5,
        temperatures=[0.0],
        processors=[
            "vocabulary_correction",
            "context_correction",
            "transcript_cleanup",
            "transcript_intelligence",
            "speaker_formatting",
        ],
        label="High Accuracy",
        description="Whisper large-v3 · beam search · maximum transcription quality",
    ),
    TranscriptionMode.archive: ModeSpec(
        model_id="large-v3",
        skip_alignment=False,
        beam_size=5,
        best_of=5,
        temperatures=[0.0],
        processors=[
            "vocabulary_correction",
            "context_correction",
            "transcript_cleanup",
            "transcript_intelligence",
            "speaker_formatting",
            "ai_grammar",
        ],
        label="Archive",
        description="Whisper large-v3 · AI grammar correction · full quality suite",
    ),
}


class AudioConfig(BaseModel):
    ffmpeg_path: str = "ffmpeg"
    normalize_audio: bool = True
    denoise: bool = False
    target_sample_rate: int = 16_000


class ParakeetConfig(BaseModel):
    """
    Parakeet-TDT-specific inference settings.

    All values default to 0 / "auto" to trigger automatic hardware detection.
    Override via env vars:
      TE_PIPELINE__PARAKEET__CHUNK_SECONDS=600
      TE_PIPELINE__PARAKEET__OVERLAP_SECONDS=1.0
    """

    # 0 = auto-detect from available VRAM using the hardware.optimal_chunk_seconds table.
    # Set explicitly to override (e.g. 600 for 10-min chunks on a 12 GB GPU).
    # Maximum supported by Parakeet's full-attention architecture is ~1380 s (23 min).
    chunk_seconds: float = Field(default=0.0, ge=0.0, le=1380.0)

    # Overlap between adjacent chunks in seconds.  Words that straddle the
    # nominal chunk boundary are captured by the overlap and deduplication ensures
    # each word appears exactly once in the final transcript.
    # 0.5 s covers the longest English word at normal speaking pace.
    overlap_seconds: float = Field(default=0.5, ge=0.0, le=30.0)

    # Maximum retry attempts per chunk on CUDA OOM.  The engine halves the chunk
    # size on each retry.  0 = no retries (fail fast).
    max_chunk_retries: int = Field(default=2, ge=0, le=5)


class TranscriptionConfig(BaseModel):
    # When True, wav2vec2 alignment is always skipped regardless of the audio
    # analyzer's recommendation.  Set by fast mode to avoid the alignment step.
    force_skip_alignment: bool = False
    # "large-v3-turbo" uses CTranslate2 int8 (CPU) with VAD + streaming path.
    # MLX (Metal GPU) is disabled by default due to thermal throttling on M1 8 GB
    # that degrades ~8-min meetings from ~2 min to 20+ min.  Enable with TE_USE_MLX=1.
    model_id: str = "large-v3-turbo"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # "auto" delegates compute_type selection to ModelRegistry._resolve_compute_type
    # which picks float16/float32/int8 based on the detected device.
    compute_type: Literal["auto", "float16", "float32", "int8", "int8_float16"] = "auto"
    # batch_size=1 is fastest on M1 CPU: serial segment decoding keeps matrix
    # ops small enough to stay in L1/L2 cache. Batching 8+ segments saturates
    # memory bandwidth and slows CTranslate2 by 25-50% on Apple Silicon.
    batch_size: int = 1
    # Set to None to auto-detect language (adds ~5-7s per job).
    # "en" skips detection and is the right default for English business meetings.
    language: str = "en"
    chunk_length: int = 30
    # beam_size=2 is ~1.7x faster than beam_size=5 with < 0.5% WER increase on
    # clear English speech.  Raise to 5 for noisy audio or heavy accents via
    # TE_PIPELINE__TRANSCRIPTION__BEAM_SIZE=5.
    beam_size: int = 2
    word_timestamps: bool = True
    vad_filter: bool = True
    # CTranslate2 thread count for matrix operations.
    # 6 of 8 M1 cores — leaves 2 cores (1P + 1E) available for Chrome/VS Code/Cursor
    # at all times, reducing sustained TDP and thermal throttling on fanless hardware.
    # Combined with QOS_CLASS_UTILITY (see scheduling.py), transcription yields to
    # interactive apps immediately when they need CPU, then resumes at full speed.
    # Override with TE_PIPELINE__TRANSCRIPTION__CPU_THREADS=8 to maximize throughput
    # on a dedicated machine where system responsiveness is not a concern.
    cpu_threads: int = 6
    # temperatures=[0.0] skips the fallback retry loop (which tries 0.2, 0.4 ...
    # 1.0 on segments that fail compression/logprob checks).  For clean meeting
    # audio, the retry loop adds 5-15% overhead with no quality benefit.
    # Raise to [0.0, 0.2, 0.4] for noisy or low-quality recordings.
    temperatures: list[float] = Field(default_factory=lambda: [0.0])
    # best_of=1 is equivalent to greedy decoding when temperatures=[0.0].
    best_of: int = 1


class DiarizationConfig(BaseModel):
    model_id: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None
    enabled: bool = True
    # Diarization worker mode.
    # num_workers=1: diarizes the full audio in-process (as a thread alongside
    #   transcription).  No subprocess overhead, no duplicate model loads, uses
    #   the already-warm pyannote pipeline from the registry.  Benchmark on M1
    #   8 GB shows this finishes 76 s before whisper (total block 276 s vs 287 s
    #   for num_workers=2), saving the subprocess spawn + per-worker model-load
    #   cost (~50-60 s).  Recommended for all hardware with CT2 (CPU) backend.
    # num_workers>1: splits audio into N chunks, each diarized in a separate
    #   subprocess.  Useful when you want strict CPU isolation, e.g. debugging.
    #   Set to 2-4 on 16+ GB machines if you observe memory-pressure slowdowns.
    num_workers: int = Field(default=1, ge=1, le=8)
    # Overlap between adjacent chunks (seconds).  Used by the speaker
    # reconciler to match speaker IDs across chunk boundaries.
    chunk_overlap_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    # Pyannote segmentation step as a fraction of the 10-second window.
    # 0.1 = 1 s step (max quality, ~147 min/60 min, single worker)
    # 0.3 = 3 s step (~49 min/60 min, good quality for turns > 3 s)
    # 0.5 = 5 s step (~17 min/60 min, fine for business meetings with turns > 5 s)
    segmentation_step: float = Field(default=0.5, gt=0.0, le=1.0)


class ProcessingConfig(BaseModel):
    processors: list[str] = Field(
        default_factory=lambda: [
            "vocabulary_correction",
            "context_correction",
            "transcript_cleanup",
            "transcript_intelligence",
            "speaker_formatting",
        ]
    )
    profiles_dir: Path = Path("profiles")
    profile: str = "generic"
    intelligence_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class ExportConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["markdown", "json"])
    output_dir: Path = Path("output")
    timestamp_mode: Literal["none", "speaker", "paragraph", "minute"] = "none"
    highlight_confidence: bool = False
    low_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # Legacy field kept for backward compat
    timestamp_format: str = "[{mm}:{ss}]"
    include_timestamps: bool = False


class PipelineConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    parakeet: ParakeetConfig = Field(default_factory=ParakeetConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TE_",
        env_nested_delimiter="__",
        # Anchored to the project root so the same .env is read regardless of
        # which directory the process was started from.
        env_file=project_root() / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    hf_token: str | None = None
    models_cache_dir: Path = Path.home() / ".cache" / "transcript_engine"
    log_level: str = "INFO"
    log_file: Path | None = None

    @model_validator(mode="after")
    def _anchor_relative_paths(self) -> Settings:
        """
        Resolve relative data directories against the project root.

        These are plain relative defaults ("profiles", "output"); interpreting
        them against the CWD meant the engine looked for profiles in a
        directory that only exists when launched from the repo root.
        """
        object.__setattr__(
            self.pipeline.processing, "profiles_dir", resolve_path(self.pipeline.processing.profiles_dir)
        )
        object.__setattr__(
            self.pipeline.export, "output_dir", resolve_path(self.pipeline.export.output_dir)
        )
        return self
