# Transcript Engine — User Guide

A production-quality local AI meeting transcription engine that turns audio recordings into structured knowledge. All processing runs on your machine — no data leaves your environment.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Running a transcription](#running-a-transcription)
4. [Profiles](#profiles)
5. [Output files reference](#output-files-reference)
6. [Review workflow](#review-workflow)
7. [Search](#search)
8. [Bundle for delivery](#bundle-for-delivery)
9. [CLI reference](#cli-reference)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python 3.12 or newer
- FFmpeg (for audio conversion)
- ~5 GB disk space (for AI models, downloaded automatically on first use)
- A HuggingFace account (free) to enable speaker diarization

### Steps

```bash
git clone <repo-url>
cd transcript-engine
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .

# Verify
te --help
```

---

## Configuration

All settings can be overridden via environment variables or a `.env` file at the project root.

### Minimal `.env` for speaker diarization

```env
TE_HF_TOKEN=hf_your_token_here
```

Get a token at <https://huggingface.co/settings/tokens>, then accept the model terms at:
- <https://huggingface.co/pyannote/speaker-diarization-3.1>

### Full configuration reference

| Variable | Default | Description |
|---|---|---|
| `TE_HF_TOKEN` | — | HuggingFace token (required for diarization) |
| `TE_PIPELINE__TRANSCRIPTION__MODEL_ID` | `large-v3` | Whisper model to use |
| `TE_PIPELINE__TRANSCRIPTION__DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `TE_PIPELINE__TRANSCRIPTION__COMPUTE_TYPE` | `auto` | `int8`, `float16`, `float32` |
| `TE_PIPELINE__TRANSCRIPTION__LANGUAGE` | auto-detect | Force language: `en`, `es`, etc. |
| `TE_PIPELINE__DIARIZATION__MIN_SPEAKERS` | — | Minimum expected speakers |
| `TE_PIPELINE__DIARIZATION__MAX_SPEAKERS` | — | Maximum expected speakers |
| `TE_PIPELINE__PROCESSING__PROFILE` | `generic` | Default profile name |
| `TE_PIPELINE__EXPORT__OUTPUT_DIR` | `output/` | Default output directory |
| `TE_PIPELINE__EXPORT__TIMESTAMP_MODE` | `none` | `none`, `speaker`, `paragraph`, `minute` |
| `TE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` |
| `TE_AI_BASE_URL` | — | OpenAI-compatible endpoint for AI grammar pass |
| `TE_AI_MODEL` | — | Model name for AI grammar pass |

---

## Running a transcription

### Basic

```bash
te transcribe meeting.wav
```

### With speaker labels (requires HuggingFace token)

```bash
te transcribe meeting.wav --profile banking
```

### Common formats

```bash
# AAC (e.g. iPhone recordings)
te transcribe meeting.aac --profile banking

# MP3 podcast or interview
te transcribe interview.mp3 --profile generic

# MP4 screen recording
te transcribe demo.mp4 --no-diarization

# M4A (Apple Voice Memos)
te transcribe voicememo.m4a --profile banking

# WAV
te transcribe recording.wav --profile banking --timestamps speaker
```

### All options

```
--output, -o          Export formats: markdown,json,txt,srt,vtt (default: markdown,json)
--output-dir, -d      Output directory (default: output/)
--profile, -p         Vocabulary profile: generic, banking, or custom (default: generic)
--language, -l        Force language code (e.g. en). Auto-detects if omitted.
--speakers, -s        Expected number of speakers (improves diarization)
--max-speakers        Maximum number of speakers
--no-diarization      Skip speaker separation — single speaker mode
--timestamps, -t      Timestamp style: none, speaker, paragraph, minute (default: none)
--highlight-confidence    Flag uncertain words as **[word?]** in markdown
--confidence-threshold    Threshold for flagging (default: 0.65)
--verbose             Enable debug logging
```

---

## Profiles

Profiles scope vocabulary corrections to specific meeting types. Without a profile, only generic cleanup runs — no banking terms, no domain-specific corrections.

### Available profiles

```bash
te profiles
```

### Use a profile

```bash
te transcribe meeting.wav --profile banking
```

### Create a profile

1. Transcribe some representative meetings with the generic profile
2. Run the profile builder to identify domain terms:
   ```bash
   te build-profile output/ --output profiles/myteam/vocabulary.yaml
   ```
3. Create `profiles/myteam/profile.json`:
   ```json
   {
     "name": "myteam",
     "description": "My team meeting vocabulary",
     "industry": "technology",
     "processors": ["vocabulary_correction", "context_correction", "transcript_cleanup", "transcript_intelligence", "speaker_formatting"]
   }
   ```
4. Curate `profiles/myteam/vocabulary.yaml` — add `context` words to avoid false positives
5. Use it: `te transcribe meeting.wav --profile myteam`

---

## Output files reference

Every `te transcribe` run produces the following files in `--output-dir` (default: `output/`):

### Always generated

| File | Format | Description |
|---|---|---|
| `{name}.md` | Markdown | Human-readable transcript with speaker headers |
| `{name}.json` | JSON | Machine-readable full transcript with word-level timing and confidence |
| `{name}.summary.md` | Markdown | Executive summary (up to 10 bullet points) |
| `{name}.action_items.md` | Markdown | Extracted tasks with owner and due date |
| `{name}.decisions.md` | Markdown | Decisions with rationale and timestamp |
| `{name}.questions.md` | Markdown | Open questions flagged for follow-up |
| `{name}.timeline.md` | Markdown | Chronological meeting event log |
| `{name}.entities.json` | JSON | Named entities: people, organizations, acronyms, amounts, dates, IDs |
| `{name}.review.md` | Markdown | **Human review guide** — low-confidence words, speaker issues, codes/numbers |
| `{name}.quality.md` | Markdown | Confidence distribution and quality metrics |
| `{name}.stats.json` | JSON | Basic pipeline statistics |
| `{name}.metrics.json` | JSON | Full metrics including intelligence extraction counts |
| `{name}.index.json` | JSON | Search index — enables `te search` without re-running the pipeline |

### Generated only when corrections exist

| File | Format | Description |
|---|---|---|
| `{name}.corrections.md` | Markdown | Vocabulary corrections applied with context |

---

## Review workflow

The goal: a reviewer should verify a 60-minute transcript in 5–10 minutes.

### Step 1 — Start with the summary

Open `meeting.summary.md`. It shows up to 10 bullet points derived only from facts stated in the meeting.

### Step 2 — Check the review report

Open `meeting.review.md`. It contains six sections:

1. **Low-confidence words** — every word Whisper was uncertain about, with its context sentence
2. **Low-confidence segments** — entire passages below the confidence threshold
3. **Corrections applied** — vocabulary substitutions made by the active profile
4. **Potential proper nouns** — names or terms that appear only once and may be misspelled
5. **Numbers & codes** — all loan IDs, dollar amounts, percentages, dates, ticket numbers, URLs — verify these are transcribed correctly
6. **Possible speaker issues** — very short segments and rapid speaker switching that may indicate diarization errors

### Step 3 — Regenerate review with custom threshold

```bash
te review output/meeting.json --threshold 0.55
```

This regenerates only the review report without re-running Whisper.

### Step 4 — Rename speakers

```bash
te rename output/meeting.json SPEAKER_00=Alice SPEAKER_01=Bob --reexport
```

### Step 5 — Search for specific terms

```bash
te search output/meeting.index.json "LN12345"
te search output/meeting.index.json "action items"
```

---

## Search

The search index is generated automatically with every transcription.

```bash
# Find any word or phrase
te search output/meeting.index.json "APPL_ID"

# Special keywords — list extracted intelligence
te search output/meeting.index.json "action items"
te search output/meeting.index.json "decisions"
te search output/meeting.index.json "questions"

# Control results
te search output/meeting.index.json "loan" --max 10 --context 5
```

Results show: timestamp, speaker, matching sentence with the term highlighted in **bold**.

---

## Bundle for delivery

Package all meeting artifacts into a single ZIP for client delivery:

```bash
te bundle output/
```

This creates `output/meeting_bundle.zip` containing every `.md`, `.json`, `.txt`, `.srt`, and `.vtt` file — ready to send to a client or archive.

---

## CLI reference

```
te transcribe   Transcribe an audio/video file (main command)
te review       Regenerate review report from existing transcript JSON
te search       Search a meeting index for words, entities, or intelligence
te bundle       Package all artifacts into a ZIP bundle
te rename       Rename speaker IDs to real names in an existing transcript
te export       Re-export an existing transcript to additional formats
te profiles     List available vocabulary profiles
te build-profile  Analyze transcripts to suggest vocabulary for a new profile
te config-show  Show the current resolved configuration
```

---

## Troubleshooting

### Pipeline fails with "HuggingFace token required"

Add your token to `.env`:
```
TE_HF_TOKEN=hf_your_token_here
```
Then accept the model terms at the URL shown in the error message.

### All speech attributed to "Speaker 1" (no diarization)

Either `--no-diarization` was used, or the HuggingFace token is missing/invalid. Add the token and re-run.

### Slow on first run

The engine downloads Whisper large-v3 (~3 GB), alignment model (~360 MB), and diarization model (~300 MB) once. Subsequent runs reuse the local cache at `~/.cache/transcript_engine/`.

### Wrong vocabulary corrections

Corrections are profile-scoped. If corrections from a previous domain are appearing, you are using the wrong profile. Run `te profiles` to see what's available.

To audit what was corrected:
```bash
cat output/meeting.corrections.md
```

To disable all corrections, use `--profile generic`.

### Low confidence on specific terms

This is expected for proper nouns, acronyms, and technical terms that Whisper hasn't seen. Adding them to a vocabulary profile (`profiles/myprofile/vocabulary.yaml`) allows the engine to correct them post-transcription with high confidence.

### "Exported" shows wrong format

Only formats registered in the exporter registry are available: `markdown`, `json`, `txt`, `srt`, `vtt`. Use `--output markdown,json` (comma-separated, no spaces).

### macOS Apple Silicon — MPS device warning

The engine automatically routes Whisper to CPU on Apple Silicon (MPS is not supported by CTranslate2). Alignment and diarization use MPS when available. This is expected behavior, not an error.
