# Transcript Engine — Quick Start

Transcribe a meeting audio file to structured knowledge in under 5 minutes.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | `python3 --version` |
| FFmpeg | Any recent | `brew install ffmpeg` (macOS) or `apt install ffmpeg` |
| HuggingFace token | — | Required for speaker diarization (free account) |

### Get a HuggingFace token (for speaker labels)

1. Create a free account at <https://huggingface.co>
2. Generate a token at <https://huggingface.co/settings/tokens>
3. Accept the model terms at <https://huggingface.co/pyannote/speaker-diarization-3.1>

---

## Installation

```bash
git clone <repo>
cd transcript-engine
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Verify:

```bash
te --help
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Required for speaker diarization
TE_HF_TOKEN=hf_your_token_here

# Optional: override defaults
# TE_PIPELINE__TRANSCRIPTION__MODEL_ID=large-v3
# TE_PIPELINE__TRANSCRIPTION__DEVICE=auto
# TE_LOG_LEVEL=INFO
```

---

## First transcription

```bash
te transcribe meeting.wav
```

With speaker labels and banking vocabulary:

```bash
te transcribe meeting.wav --profile banking --timestamps speaker
```

Outputs are written to `output/` by default.

---

## Expected output

```
output/
├── meeting.md              ← Readable transcript (## Speaker headers)
├── meeting.json            ← Machine-readable transcript
├── meeting.summary.md      ← Executive summary (up to 10 bullets)
├── meeting.action_items.md ← Extracted tasks with owner and due date
├── meeting.decisions.md    ← Decisions with rationale
├── meeting.questions.md    ← Open questions
├── meeting.timeline.md     ← Chronological event log
├── meeting.entities.json   ← Named entities (people, orgs, acronyms, amounts)
├── meeting.corrections.md  ← Vocabulary corrections applied (if any)
├── meeting.review.md       ← Human review guide (low-confidence words, issues)
├── meeting.quality.md      ← Confidence distribution and quality metrics
├── meeting.stats.json      ← Basic statistics
├── meeting.metrics.json    ← Full metrics including intelligence counts
└── meeting.index.json      ← Search index for te search
```

---

## Rename speakers

After transcription, replace `SPEAKER_00` etc. with real names:

```bash
te rename output/meeting.json SPEAKER_00=Alice SPEAKER_01=Bob --reexport
```

---

## Review the transcript

Generate or regenerate the human review guide:

```bash
te review output/meeting.json
```

---

## Search

```bash
te search output/meeting.index.json "APPL_ID"
te search output/meeting.index.json "action items"   # lists all action items
te search output/meeting.index.json "decisions"       # lists all decisions
```

---

## Bundle for delivery

```bash
te bundle output/
# Creates: output/meeting_bundle.zip
```

---

## Troubleshooting

**`Pipeline failed: HuggingFace token required`**
→ Add `TE_HF_TOKEN=hf_...` to your `.env` file and accept the model terms at the URL shown.

**`Pipeline failed: unsupported device mps`**
→ Should not happen — the engine auto-routes to CPU for Whisper on Apple Silicon. If it does, set `TE_PIPELINE__TRANSCRIPTION__DEVICE=cpu` in `.env`.

**Very slow on first run**
→ Whisper large-v3 (~3GB) and alignment models are downloaded once. Subsequent runs reuse the cache.

**Single speaker label for whole transcript**
→ Diarization was skipped (`--no-diarization`) or the HuggingFace token is missing. Set `TE_HF_TOKEN` and re-run.

**Vocabulary not being corrected**
→ Corrections are profile-scoped. Use `--profile banking` (or your custom profile) — the `generic` profile intentionally applies no vocabulary corrections.
