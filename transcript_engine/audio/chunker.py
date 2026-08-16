"""
Audio chunker — splits a prepared audio file into N overlapping segments.

Used by the parallel diarization pipeline so each segment can be processed
by an independent worker process while transcription runs on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioChunk:
    """One segment of the original audio, ready for diarization."""

    path: Path  # path to the written WAV file for this chunk
    chunk_idx: int  # 0-based index
    global_start: float  # start time in the original audio (seconds)
    global_end: float  # end time in the original audio (seconds)
    overlap_seconds: float  # overlap with the PREVIOUS chunk (0 for chunk 0)

    @property
    def duration(self) -> float:
        return self.global_end - self.global_start


def split_audio(
    audio_path: Path,
    n_workers: int,
    overlap_seconds: float = 30.0,
    out_dir: Path | None = None,
) -> list[AudioChunk]:
    """
    Split *audio_path* into *n_workers* overlapping WAV files.

    Each chunk covers one equal slice of the audio plus *overlap_seconds* of
    the next chunk.  The overlap is used by the speaker reconciler to match
    speaker IDs across chunk boundaries.

    If *n_workers* == 1, returns a single chunk equal to the full audio so the
    caller does not need a special code-path.
    """
    import torchaudio  # noqa: PLC0415

    if out_dir is None:
        out_dir = audio_path.parent

    info = torchaudio.info(str(audio_path))
    sr = info.sample_rate
    total_samples = info.num_frames
    total_duration = total_samples / sr

    if n_workers <= 1 or total_duration <= overlap_seconds * 2:
        return [
            AudioChunk(
                path=audio_path,
                chunk_idx=0,
                global_start=0.0,
                global_end=total_duration,
                overlap_seconds=0.0,
            )
        ]

    waveform, _ = torchaudio.load(str(audio_path))

    chunk_duration = total_duration / n_workers
    chunks: list[AudioChunk] = []

    for i in range(n_workers):
        start_sec = i * chunk_duration
        # Extend each chunk by overlap_seconds into the NEXT chunk so the
        # diarizer sees context across boundaries.
        end_sec = min((i + 1) * chunk_duration + overlap_seconds, total_duration)

        start_sample = int(start_sec * sr)
        end_sample = min(int(math.ceil(end_sec * sr)), total_samples)

        chunk_wave = waveform[:, start_sample:end_sample]
        chunk_path = out_dir / f"_chunk_{i:02d}.wav"
        torchaudio.save(str(chunk_path), chunk_wave, sr)

        chunks.append(
            AudioChunk(
                path=chunk_path,
                chunk_idx=i,
                global_start=start_sec,
                global_end=float(end_sample) / sr,
                overlap_seconds=overlap_seconds if i > 0 else 0.0,
            )
        )

    return chunks


def cleanup_chunks(chunks: list[AudioChunk], original_path: Path) -> None:
    """Delete temporary chunk files (not the original audio)."""
    import contextlib  # noqa: PLC0415

    for chunk in chunks:
        if chunk.path != original_path:
            with contextlib.suppress(Exception):
                chunk.path.unlink(missing_ok=True)
