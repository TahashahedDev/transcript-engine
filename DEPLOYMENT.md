# Deployment Guide

Transcript Engine on a clean Ubuntu Linux server.

Backend: port 9097  
Frontend: port 9098

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Ubuntu 22.04 or 24.04 | Tested; other distros require manual adaptation |
| Root / sudo access | Required for apt-get and NodeSource |
| NVIDIA GPU (optional) | Any CUDA GPU for Parakeet ASR — chunk size adapts to detected VRAM (8 GB+ recommended). CPU works with Whisper. The PyTorch build must support the card's compute capability; the setup page reports a mismatch explicitly. |
| HuggingFace token | Required for speaker diarization — free at huggingface.co |
| 20+ GB disk | Models: Parakeet ~2.5 GB, pyannote ~500 MB, Whisper ~3 GB |

---

## Deployment Steps

### 1. Clone the repository

```bash
git clone <repo-url> transcript-engine
cd transcript-engine
```

### 2. Configure the backend environment

```bash
cp .env.example .env
nano .env
```

Set at minimum:

```env
TE_HF_TOKEN=hf_your_token_here
TE_API_PORT=9097
TE_API_HOST=0.0.0.0
TE_ASR_BACKEND=parakeet          # or: whisper (for CPU-only servers)
TE_API_ALLOW_ALL_ORIGINS=1
TE_WARM_MODELS=1
```

### 3. Configure the frontend environment

**Normally nothing to do here.** The frontend derives the API URL from the
address you open the page on: browse to `http://<server-ip>:9098` and it calls
`http://<server-ip>:9097` automatically. This works unchanged on localhost and
on a rented GPU host, so the same build is portable between them.

Only create `web/.env.local` if the API lives on a *different* host or domain
than the UI (for example behind a reverse proxy):

```bash
cp web/.env.local.example web/.env.local
nano web/.env.local   # set NEXT_PUBLIC_API_URL
```

> **Do not point `NEXT_PUBLIC_API_URL` at `localhost` on a server.**
> `NEXT_PUBLIC_*` values are compiled into the browser bundle, so `localhost`
> resolves to each *visitor's own machine*, not the server — and editing the
> file after `npm run build` has no effect until you rebuild.

### 4. Accept HuggingFace model licenses

In a browser, while logged in with the HuggingFace account whose token you used:

- https://huggingface.co/pyannote/speaker-diarization-3.1 → click Accept
- https://huggingface.co/pyannote/segmentation-3.0 → click Accept

### 5. Run the setup script (as root or with sudo)

```bash
bash setup_server.sh
```

This installs Python 3.12, Node.js 20, PyTorch, all Python and frontend dependencies,
downloads models, builds the Next.js frontend, and runs preflight checks.

Expected duration: 10–30 min (first-time model download dominates).

### 6. Start both services

```bash
bash start_server.sh
```

Both services start in foreground. Press Ctrl+C to stop both.

### 7. Verify health

```bash
curl http://localhost:9097/health
```

Expected: `{"status":"ok","ready":true,...}`

### 8. Open the frontend

Open in a browser: `http://<server-ip>:9098`

---

## Running in the background (optional)

To keep both services running after you disconnect:

```bash
nohup bash start_server.sh > /var/log/transcript-engine.log 2>&1 &
```

Or use tmux:

```bash
tmux new -s te
bash start_server.sh
# Ctrl+B then D to detach
```

---

## Port reference

| Port | Service | Config |
|---|---|---|
| 9097 | FastAPI backend | `TE_API_PORT` in `.env` |
| 9098 | Next.js frontend | `--port 9098` in `web/package.json` |
| 9099 | Reserved | — |

---

## Packaging for upload

To create a clean `deployment.zip` from your development machine:

```bash
bash package_release.sh
```

Upload and deploy on the server:

```bash
scp deployment.zip user@server:/opt/
ssh user@server
cd /opt && unzip deployment.zip -d transcript-engine && cd transcript-engine
cp .env.example .env && nano .env
cp web/.env.local.example web/.env.local && nano web/.env.local
bash setup_server.sh
bash start_server.sh
```

---

## GPU vs CPU

| Mode | Backend | Expected time (90 min audio) |
|---|---|---|
| GPU | `TE_ASR_BACKEND=parakeet` | Target ~2–3 min on a 24 GB-class card — **not yet measured** |
| CPU | `TE_ASR_BACKEND=whisper` | ~45–90 min |

> The GPU figure is a design target, not a benchmark. No validated production
> measurement exists yet; the engine logs real-time factor (RTF) per job, so
> record actual numbers on your hardware before quoting them. Throughput scales
> with VRAM (larger chunks, fewer round trips), so a smaller card will be slower.

---

## Troubleshooting

**`python3.12: command not found`**  
Run: `apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get install -y python3.12 python3.12-venv`

**`torch.cuda.is_available()` returns False**  
PyTorch CUDA tag does not match driver version. Re-run setup (it auto-detects) or:  
`pip install torch --index-url https://download.pytorch.org/whl/cu124`

**Frontend shows "Cannot reach the transcription server"**  
Check in this order:
1. Is the backend running and listening on all interfaces? `curl http://<server-ip>:9097/health` from another machine. It must bind `0.0.0.0`, not `127.0.0.1`.
2. Is port 9097 open? On a rented GPU host both 9097 and 9098 must be exposed/forwarded, not just the frontend port.
3. Does `web/.env.local` exist and pin `NEXT_PUBLIC_API_URL` to `localhost`? That sends every visitor's browser to their own machine. Delete the line (auto-derivation handles it) and rebuild: `cd web && npm run build`.
4. CORS blocked? The API allows any host on the frontend port by default. If you changed the frontend port, set `TE_FRONTEND_PORT` to match, or list exact origins in `TE_API_CORS_ORIGINS`.

**`pyannote: 401 Unauthorized`**  
HuggingFace token is invalid or model terms not accepted. Accept at the URLs in Step 4.

**`ModuleNotFoundError: No module named 'nemo'`**  
NeMo is optional and only needed for Parakeet. Set `TE_ASR_BACKEND=whisper` to use the CPU-compatible backend, or install manually: `pip install 'nemo_toolkit[asr]>=1.23.0'`
