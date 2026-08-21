# Speaker identification — architecture notes

Engineering reference for `transcript_engine/identity/`. Not a product doc —
see the manager report delivered alongside this code for that.

## What this solves

Diarization (`transcript_engine/diarization/pyannote_engine.py`) already
groups a recording's speech into `SPEAKER_00`/`SPEAKER_01`/... clusters.
This package turns those clusters into real names, using only data Echo
already has — no separate client enrollment recording is required or
assumed anywhere in this design.

## Pipeline

```
Transcript (segments already attributed to SPEAKER_00/01/...)
diarization segments (start/end per speaker, for embedding windows)
        │
        ├─ self_identification.py  → "I'm Neel" evidence, per segment
        │
        └─ embedding_extractor.py  → voice vector per qualifying window
                    │
                    ▼
              matcher.py — cosine similarity vs. known SpeakerProfile centroids
                    │
                    ▼
              pipeline.py (identify_speakers) — combines both signals,
              applies the confidence/enrollment policy below,
              writes Transcript.speakers
```

Integration point: `transcript_engine/pipeline/orchestrator.py`,
`Pipeline._apply_speaker_identification()`, called immediately after
`self._merger.merge(...)`. Opt-in via `TE_SPEAKER_ID_ENABLED=1`
(unset/default: the call never happens, zero behavior change to the working
pipeline). Wrapped in try/except — any failure degrades to "no
identification this run", never fails the transcription job.

## Model decision (made, not proposed)

`pyannote/wespeaker-voxceleb-resnet34-LM`. This is the same embedding model
`pyannote/speaker-diarization-3.1` already downloads and uses internally for
its own clustering — confirmed present in `~/.cache/torch/pyannote/` before
this code ever ran. Reusing it costs zero new downloads and zero new
dependencies. SpeechBrain's ECAPA model (also installed, as a pyannote
transitive dependency) was considered and rejected for the same reason: it
is not already part of this stack for anything Echo does today, and the
model that is already here is a standard VoxCeleb speaker-verification
model, appropriate for this task.

Verified working end-to-end on this development machine: loads from local
cache in ~1s (same HF token + the same PyTorch-2.6 `weights_only`
compatibility patch `ModelRegistry._register_torch_safe_globals()` already
uses for diarization), produces a 256-dim embedding per clip in <0.2s on
CPU. See `tests/unit/test_embedding_extractor_real_model.py` — real model,
synthetic (non-speech) audio, since no real multi-speaker recording exists
in this repository.

## Confidence / enrollment policy

| Signal | Outcome | Profile updated? |
|---|---|---|
| Voice similarity ≥ `HIGH_SIMILARITY_THRESHOLD` (0.80, PROPOSED) | Assign matched name, confidence HIGH | Yes |
| No strong voice match, but a HIGH-confidence self-introduction exists in that speaker's own segments | Bootstrap/attach a profile for that name, confidence HIGH | Yes |
| Voice similarity in `[MEDIUM, HIGH)` (0.65–0.80, PROPOSED) | Assign matched name, confidence MEDIUM | **No** — a medium match must never poison a profile |
| Neither | Unknown — `Transcript.speakers` left unset for that `speaker_id` | No |

Thresholds are explicitly not validated against real voices — no
multi-speaker recording exists locally to validate them against. They are
module constants (`transcript_engine/identity/matcher.py`) specifically so
real validation can override them without a code change.

## Why no re-clustering, no vector DB, no new persistence layer

- **No re-clustering**: diarization already did that; identification is a
  lookup against known centroids (`matcher.match_embedding`), not a second
  clustering pass.
- **No vector DB**: at today's actual scale — zero profiles in production,
  a handful of ~256-float embeddings per person if any existed — Pinecone/
  pgvector/etc. would be solving a problem that doesn't exist yet. NumPy +
  cosine similarity is O(profiles), trivial at any realistic count of known
  speakers (tens to low hundreds).
- **No Postgres `db/` module**: verified that module has no speaker/person
  table and, more importantly, is not imported by the pipeline that
  actually runs jobs (`api/pipeline_runner.py` never imports it — only the
  opt-in, explicitly-incomplete v2 API does). Wiring speaker data into it is
  a separate integration decision, out of this mission's scope.

`transcript_engine/identity/store.py` instead: one JSON file per profile
under `data/speaker_profiles/`, atomic write-then-rename.

## Windowing policy (embedding_extractor.py)

Diarization segments are already speech-only and speaker-homogeneous, so no
VAD step is needed. `WINDOW_MIN_S=2.0`, `WINDOW_TARGET_S=5.0`,
`WINDOW_MAX_S=10.0` — segments shorter than the minimum are dropped
(too likely to be "yeah"/"okay"); segments longer than the maximum are split
into target-sized windows so one long turn doesn't dominate a profile.
These are starting values from general speaker-verification practice, not
measurements — flagged PROPOSED, same as the similarity thresholds.

## Privacy (section 24 of the mission brief)

- **What is stored**: a display name string, float embedding vectors
  (256-dim, from the model above), the job IDs and evidence strings that
  justified each embedding (e.g. `"self_identification:high"`,
  `"voice_match:high:0.91"`), timestamps.
- **What is NOT stored**: no raw audio, no transcript text, in the profile
  store itself.
- **Where**: `data/speaker_profiles/*.json`, local disk only. No network
  calls beyond the one-time (cached) model download from HuggingFace.
- **Deletion**: `SpeakerProfileStore.delete(profile_id)` exists as the
  mechanical capability.
- **Retention / consent policy**: not decided here — this is a product
  policy question (when a profile should be created without a person's
  awareness, how long it persists, who can request deletion), explicitly
  out of scope for this engineering change. Flagged as an open question for
  the product owner before this ships to real users.
- **Voice embeddings are biometric data.** Treat this store with the same
  handling requirements as any other biometric data store before it holds
  anything derived from a real person's voice.

## Explicitly not done

- Overlap-aware quality filtering: current diarization output carries no
  overlap-probability field to filter on. Would require extending
  `SpeakerSegment` first.
- Threshold validation against real voices: no fixture available.
- Any UI for confirming/correcting a name.
- Wiring `db/` (Postgres) as a persistence alternative.
