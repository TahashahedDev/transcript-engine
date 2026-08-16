"""
Speaker reconciler — aligns pyannote speaker IDs across audio chunks.

When the pipeline diarizes N chunks independently, each chunk assigns
arbitrary local IDs (SPEAKER_00, SPEAKER_01, …).  This module builds a
globally consistent mapping by comparing who speaks in the overlap regions
between adjacent chunks.

Algorithm
---------
For each pair of adjacent chunks (i, i+1):
  1. Find the overlap region [boundary - overlap, boundary] in global time.
  2. For each 1-second frame in the overlap, look up the active speaker
     in both chunk i (authoritative, already globally labelled) and
     chunk i+1 (local IDs).
  3. Vote: chunk_i+1_speaker → chunk_i_speaker (majority wins per speaker).
  4. Apply the mapping; speakers with no overlap match keep a new unique ID.

After reconciliation, deduplicate the overlap region by dropping the portion
from chunk i+1 that falls inside chunk i's time range.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from transcript_engine.models.pipeline import DiarizationResult, SpeakerSegment


@dataclass
class ChunkDiarization:
    """Diarization result for one audio chunk, with global timestamps."""

    chunk_idx: int
    global_start: float
    global_end: float
    overlap_seconds: float
    segments: list[SpeakerSegment]  # timestamps already offset to global time


def reconcile(chunks: list[ChunkDiarization]) -> DiarizationResult:
    """
    Merge per-chunk diarization into a single globally-consistent result.
    Returns a DiarizationResult covering the full audio.
    """
    if not chunks:
        return DiarizationResult(segments=(), num_speakers=0)

    if len(chunks) == 1:
        segs = chunks[0].segments
        return DiarizationResult(
            segments=tuple(segs),
            num_speakers=len({s.speaker_id for s in segs}),
        )

    # Sort by chunk index to process in order.
    ordered = sorted(chunks, key=lambda c: c.chunk_idx)

    # Start with chunk 0 — its local IDs become the global reference.
    global_segs: list[SpeakerSegment] = list(ordered[0].segments)
    global_id_counter = _max_speaker_idx(global_segs) + 1

    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        curr = ordered[i]

        # The authoritative boundary is where chunk i-1 ends WITHOUT overlap.
        # The overlap extends from curr.global_start to prev.global_end.
        overlap_start = curr.global_start
        overlap_end = prev.global_end
        has_overlap = overlap_end > overlap_start + 0.5

        if has_overlap:
            # Use the overlap region to vote on speaker correspondences.
            mapping = _vote_mapping(
                global_segs, curr.segments, overlap_start, overlap_end
            )
        else:
            mapping = {}

        # Assign global IDs to any unmapped local speakers.
        for local_sp in {s.speaker_id for s in curr.segments}:
            if local_sp not in mapping:
                mapping[local_sp] = f"SPEAKER_{global_id_counter:02d}"
                global_id_counter += 1

        # Apply mapping and drop the overlap portion from the new chunk.
        # The authoritative source for the overlap is global_segs (from chunk i-1).
        # We only add segments from curr that start at or after overlap_end.
        # For the seam itself, add curr segments clipped to [overlap_end, ...].
        for seg in curr.segments:
            mapped_id = mapping.get(seg.speaker_id, seg.speaker_id)
            if seg.end <= overlap_end:
                # Entirely inside overlap — already covered by global_segs.
                continue
            if seg.start < overlap_end:
                # Straddles the seam — clip to avoid duplication.
                global_segs.append(
                    SpeakerSegment(
                        speaker_id=mapped_id,
                        start=round(overlap_end, 3),
                        end=seg.end,
                    )
                )
            else:
                global_segs.append(
                    SpeakerSegment(speaker_id=mapped_id, start=seg.start, end=seg.end)
                )

    # Sort and filter zero-duration artefacts.
    final = sorted(
        (s for s in global_segs if s.end - s.start > 0.1),
        key=lambda s: s.start,
    )
    return DiarizationResult(
        segments=tuple(final),
        num_speakers=len({s.speaker_id for s in final}),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _speaker_at(segments: list[SpeakerSegment], t: float) -> str | None:
    """Return the speaker active at time *t*, or None."""
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg.speaker_id
    return None


def _vote_mapping(
    global_segs: list[SpeakerSegment],
    local_segs: list[SpeakerSegment],
    overlap_start: float,
    overlap_end: float,
) -> dict[str, str]:
    """
    For each local speaker, vote which global speaker they correspond to
    by comparing who speaks at each second inside the overlap region.
    """
    votes: dict[str, Counter[str]] = defaultdict(Counter)

    for t_int in range(int(overlap_start), int(overlap_end)):
        t = t_int + 0.5
        g_sp = _speaker_at(global_segs, t)
        l_sp = _speaker_at(local_segs, t)
        if g_sp and l_sp:
            votes[l_sp][g_sp] += 1

    mapping: dict[str, str] = {}
    # Greedy: assign highest-vote match, then remove from candidate pool.
    used_globals: set[str] = set()
    for local_sp, counter in sorted(votes.items(), key=lambda x: -sum(x[1].values())):
        for global_sp, _ in counter.most_common():
            if global_sp not in used_globals:
                mapping[local_sp] = global_sp
                used_globals.add(global_sp)
                break

    return mapping


def _max_speaker_idx(segments: list[SpeakerSegment]) -> int:
    """Return the highest numeric suffix in SPEAKER_XX labels, or -1."""
    import contextlib  # noqa: PLC0415

    idx = -1
    for seg in segments:
        if seg.speaker_id.startswith("SPEAKER_"):
            with contextlib.suppress(ValueError, IndexError):
                idx = max(idx, int(seg.speaker_id.split("_", 1)[1]))
    return idx
