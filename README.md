# Transcript Engine

Local AI meeting transcription. Audio in, speaker-labelled transcript and
meeting notes out. Everything runs on your own machine or your own GPU — no
audio is sent to a third-party service.

```
audio/video ─► preprocess ─► ASR ─────────┐
                            (Parakeet TDT) ├─► merge ─► processors ─► artifacts
                         diarization ──────┘
                          (pyannote)
```

---

## Quick start

**Prerequisites:** Python 3.12+, Node.js 20+, `ffmpeg`, and a
[HuggingFace token](https://huggingface.co/settings/tokens) (needed for speaker
diarization).

```bash
git clone <your-repo-url> transcript-engine
cd transcript-engine

cp .env.example .env
nano .env                 # set TE_HF_TOKEN=hf_...

bash setup_gpu.sh         # Linux/GPU  (macOS: make install)
bash start.sh             # starts backend + web UI
```

Then open **http://localhost:9098**.

`start.sh` is the only command you need — it installs and builds the web UI on
first run, starts both services, waits for them to become healthy, and prints
the URLs. `Ctrl+C` stops everything.

| Command | What it does |
|---|---|
| `bash start.sh` | Backend + web UI |
| `bash start.sh --backend-only` | API only, no web UI |
| `bash start.sh --skip-preflight` | Skip environment checks (faster restarts) |

---

## Accepting the model licenses

Diarization uses gated models. While logged in to HuggingFace, click **Agree** on
both, or speaker labels will be unavailable:

- <https://huggingface.co/pyannote/speaker-diarization-3.1>
- <https://huggingface.co/pyannote/segmentation-3.0>

---

## Running on a rented GPU (Vast.ai and similar)

The app is hardware-agnostic: it detects the GPU at runtime and sizes its work
to the VRAM actually available, so the same code runs on an 8 GB card or a
48 GB one without configuration.

```bash
bash setup_gpu.sh
bash start.sh
```

Expose ports **9097** (API) and **9098** (web UI) on the provider's firewall.
Open `http://<host-ip>:9098` — the frontend derives the API URL from the address
you loaded it on, so nothing needs editing for a remote host.

Two things worth knowing:

- **The web UI must reach the API directly from your browser.** Both ports need
  to be reachable, not just 9098.
- **PyTorch must support the card.** A very new GPU on an older PyTorch build
  reports a clear error on the setup page rather than failing mid-job.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full server guide.

---

## Output

Every completed job produces:

| File | Contents |
|---|---|
| `transcript.md` | Speaker-labelled transcript, formatted |
| `transcript.txt` | Plain text |
| `transcript.json` | Word-level timings and metadata (re-exportable) |
| `transcript.srt` / `.vtt` | Subtitle tracks |
| `transcript.summary.md` | Meeting summary |
| `transcript.action_items.md` | Action items |
| `transcript.decisions.md` | Decisions |
| `transcript.questions.md` | Questions raised |
| `transcript.timeline.md` | Topic timeline |
| `transcript_bundle.zip` | All of the above |

All are downloadable from the results page, individually or as the bundle.

---

## Configuration

Everything is set through `.env` (see `.env.example`). The most useful:

| Variable | Default | Purpose |
|---|---|---|
| `TE_HF_TOKEN` | — | HuggingFace token; required for diarization |
| `TE_ASR_BACKEND` | `whisper` | `parakeet` for GPU, `whisper` for CPU |
| `TE_API_PORT` / `TE_FRONTEND_PORT` | `9097` / `9098` | Service ports |
| `TE_PIPELINE__PARAKEET__CHUNK_SECONDS` | `0` (auto) | Override VRAM-based chunk sizing |
| `TE_API_STALL_TIMEOUT_MINUTES` | `45` | Fail a job that stops making progress |
| `TE_AI_BASE_URL` + `TE_AI_MODEL` | — | Enables the optional AI grammar pass |

Relative paths are resolved against the project root, so the app behaves the
same regardless of which directory you launch it from. Variables already set in
your environment take precedence over `.env`.

---

## Development

```bash
make install          # create .venv and install everything
make check            # lint + typecheck + tests
make test             # tests only

cd web && npm run dev # web UI with hot reload
```

| Document | What it covers |
|---|---|
| [PROJECT_ENGINEERING_GUIDE.md](PROJECT_ENGINEERING_GUIDE.md) | How the ML pipeline works, and why it is built this way |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Running on a rented GPU host, start to finish |
| [USER_GUIDE.md](USER_GUIDE.md) | The `te` command line interface |

---

## Current limitations

Stated plainly rather than buried:

- **Speaker labels are anonymous.** The system knows *that* speakers differ, not
  *who* they are. There is no voice enrolment or identification.
- **Jobs are held in memory.** Restarting the server loses in-flight job state.
  A PostgreSQL schema exists but is not yet wired to the live job store.
- **GPU performance is not yet benchmarked.** The engine logs real-time factor
  per job; no validated production number has been recorded, so none is quoted.
- **A hung job blocks the queue.** A watchdog marks it failed so the UI recovers,
  but the worker thread cannot be killed and holds the single-job executor until
  the process restarts.
- **Overlapping speech degrades both transcription and speaker attribution.**
- **The cloud API (`/api/v2`) is unfinished.** Its endpoints, PostgreSQL schema,
  and R2/Supabase wiring exist, but no worker consumes the queue — a job created
  there stays `queued`. It is off unless `TE_ENABLE_CLOUD_API=1`.
