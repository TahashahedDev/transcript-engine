#!/bin/bash
# setup_server.sh — One-command setup for a clean Ubuntu Linux server.
#
# Supports: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS
# Installs: Python 3.12, Node.js 20, npm, ffmpeg, system libs,
#           PyTorch (auto-detects CUDA), all Python deps,
#           all frontend deps, builds the Next.js frontend.
#
# Usage (as root, or with sudo):
#   git clone <repo> transcript-engine && cd transcript-engine
#   cp .env.example .env
#   nano .env                       # set TE_HF_TOKEN=hf_...
#   cp web/.env.local.example web/.env.local
#   nano web/.env.local             # set NEXT_PUBLIC_API_URL if not localhost
#   bash setup_server.sh
#   bash start_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }
die()   { err "$*"; exit 1; }
step()  { echo -e "\n${BOLD}══ $* ══${NC}"; }
pass()  { echo -e "  ${GREEN}✓${NC}  $*"; }
fail()  { echo -e "  ${RED}✗${NC}  $*"; _FAILURES+=("$*"); }
warnc() { echo -e "  ${YELLOW}!${NC}  $*"; _WARNINGS+=("$*"); }

_FAILURES=()
_WARNINGS=()

echo ""
echo -e "${BOLD}Transcript Engine — Server Setup${NC}"
echo "=================================="
echo "Backend:  port 9097"
echo "Frontend: port 9098"
echo ""

# ── 1. OS detection ───────────────────────────────────────────────────────────
step "1/10  OS"

if [[ "$(uname -s)" != "Linux" ]]; then
  die "This script is for Linux only. On macOS: make install && cd web && npm ci && npm run build"
fi

if ! command -v apt-get &>/dev/null; then
  die "This script requires apt-get (Debian/Ubuntu). Adapt manually for other distros."
fi

# Parse /etc/os-release directly — lsb_release may not be installed on minimal images
UBUNTU_VER="unknown"
if [ -f /etc/os-release ]; then
  UBUNTU_VER=$(grep "^VERSION_ID=" /etc/os-release | cut -d'"' -f2 2>/dev/null || echo "unknown")
fi
pass "Linux / Ubuntu ${UBUNTU_VER}"

# ── 2. System packages ────────────────────────────────────────────────────────
step "2/10  System Packages"

export DEBIAN_FRONTEND=noninteractive
info "Running apt-get update..."
apt-get update -qq 2>&1 | tail -2

# Python 3.12:
#   Ubuntu 24.04 — ships Python 3.12 natively, no PPA needed
#   Ubuntu 22.04 — ships Python 3.10; deadsnakes PPA required for 3.12
if ! python3.12 --version &>/dev/null 2>&1; then
  info "Python 3.12 not found — adding deadsnakes PPA..."
  apt-get install -y -qq software-properties-common gnupg 2>&1 | tail -2
  add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -3
  apt-get update -qq 2>&1 | tail -2
fi

info "Installing system packages..."
# python3.12-distutils does NOT exist on Ubuntu 24.04 (distutils is bundled).
# pip + setuptools handles all build requirements without it.
apt-get install -y -qq \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  ffmpeg \
  git curl wget \
  build-essential \
  libsndfile1 \
  libgomp1 \
  ca-certificates \
  gnupg \
  psmisc \
  2>&1 | tail -5

# python3.12-distutils is Ubuntu 22.04 only (from deadsnakes PPA)
if [[ "${UBUNTU_VER}" == "22.04" ]]; then
  apt-get install -y -qq python3.12-distutils 2>&1 | tail -2 || \
    warn "python3.12-distutils not available (non-fatal, pip handles this)"
fi

if python3.12 --version &>/dev/null 2>&1; then
  pass "Python: $(python3.12 --version)"
else
  fail "python3.12 install failed"
  die "Cannot continue without Python 3.12."
fi

if command -v ffmpeg &>/dev/null; then
  FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)
  pass "ffmpeg: $FFMPEG_VER"
else
  fail "ffmpeg not found after install"
fi

# ── 3. Node.js 20 ─────────────────────────────────────────────────────────────
step "3/10  Node.js 20"

if ! node --version 2>/dev/null | grep -qE "^v2[0-9]"; then
  info "Installing Node.js 20 via NodeSource..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>&1 | tail -5
  apt-get install -y -qq nodejs 2>&1 | tail -3
fi

if node --version &>/dev/null 2>&1; then
  pass "Node.js: $(node --version)"
else
  fail "Node.js install failed"
  die "Cannot continue without Node.js."
fi

if npm --version &>/dev/null 2>&1; then
  pass "npm: $(npm --version)"
else
  fail "npm not found after Node.js install"
fi

# ── 4. Python virtual environment ─────────────────────────────────────────────
step "4/10  Python Virtual Environment"

if [ ! -d .venv ]; then
  info "Creating .venv with Python 3.12..."
  python3.12 -m venv .venv
  pass "Created .venv"
else
  pass ".venv already exists — skipping creation"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Verify venv Python is 3.12
PY_IN_VENV=$(python --version 2>&1)
if echo "$PY_IN_VENV" | grep -q "3.12"; then
  pass "Virtualenv: $PY_IN_VENV"
else
  die "Virtualenv Python is not 3.12: $PY_IN_VENV"
fi

pip install --upgrade pip setuptools wheel --quiet
pass "pip upgraded: $(pip --version | cut -d' ' -f1-2)"

# ── 5. CUDA detection + PyTorch ───────────────────────────────────────────────
step "5/10  PyTorch"

TORCH_INDEX="https://download.pytorch.org/whl/cpu"
CUDA_AVAILABLE=no

if command -v nvidia-smi &>/dev/null; then
  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+" || echo "")
  if [ -n "$CUDA_VER" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    CUDA_MINOR="${CUDA_MINOR:-0}"

    if [ "$CUDA_MAJOR" = "11" ]; then
      TORCH_CUDA_TAG="cu118"
    elif [ "$CUDA_MAJOR" = "12" ]; then
      if [ "${CUDA_MINOR}" -le 1 ] 2>/dev/null; then
        TORCH_CUDA_TAG="cu121"
      else
        TORCH_CUDA_TAG="cu124"
      fi
    else
      TORCH_CUDA_TAG="cu124"
    fi
    TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
    CUDA_AVAILABLE=yes
    info "GPU detected: CUDA ${CUDA_VER} — installing PyTorch ${TORCH_CUDA_TAG}"
  fi
fi

if [ "$CUDA_AVAILABLE" = "no" ]; then
  info "No GPU detected — installing CPU-only PyTorch"
fi

pip install torch torchaudio --index-url "$TORCH_INDEX" --quiet

CUDA_CHECK=$(python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'CUDA: {name} ({vram:.1f} GB VRAM)')
else:
    print('CPU only (no CUDA)')
" 2>&1)
pass "PyTorch: $CUDA_CHECK"

# ── 6. Python dependencies ────────────────────────────────────────────────────
step "6/10  Python Dependencies"

info "Installing transcript-engine base dependencies..."
pip install -e "." --quiet 2>&1 | tail -3
pass "Core Python deps installed"

if [ "$CUDA_AVAILABLE" = "yes" ]; then
  info "Installing NeMo (nemo_toolkit[asr]) — takes 5-10 min on first install..."
  if pip install "nemo_toolkit[asr]>=1.23.0" --quiet 2>&1 | tail -5; then
    NEMO_VER=$(python -c "import nemo; print(nemo.__version__)" 2>&1 || echo "unknown")
    pass "NeMo: $NEMO_VER"
  else
    warnc "NeMo install failed — Parakeet ASR unavailable, Whisper fallback still works"
    warnc "Retry: pip install 'nemo_toolkit[asr]>=1.23.0'"
  fi
else
  warnc "Skipping NeMo (no GPU detected) — using Whisper backend"
  warnc "To enable Parakeet: install a GPU instance and re-run setup_server.sh"
fi

# ── 7. Environment files ───────────────────────────────────────────────────────
step "7/10  Environment Files"

if [ ! -f .env ]; then
  cp .env.example .env
  warn ".env created from .env.example"
  warn "Set TE_HF_TOKEN=hf_... in .env before running model download"
else
  pass ".env exists"
fi

# Source .env so HF token check works
set -o allexport
# shellcheck disable=SC1091
source .env 2>/dev/null || true
set +o allexport

if [ -n "${TE_HF_TOKEN:-}" ] && [ "${TE_HF_TOKEN:-}" != "hf_your_token_here" ]; then
  pass "TE_HF_TOKEN is set (${TE_HF_TOKEN:0:8}...)"
  HF_TOKEN_SET=yes
else
  warnc "TE_HF_TOKEN not set — model download will be skipped"
  warnc "Set TE_HF_TOKEN in .env and run: python scripts/download_models.py"
  HF_TOKEN_SET=no
fi

# Intentionally does NOT create web/.env.local with a hardcoded URL.
# NEXT_PUBLIC_* values are compiled into the browser bundle, so a baked-in
# "localhost" would send the visitor's browser to their own machine instead of
# this server. With the variable unset, the frontend derives the API URL from
# the address the page was opened on (see web/lib/apiBase.ts), which is correct
# both locally and on a remote GPU host. Only set it when the API is on a
# different host/domain than the UI.
if [ -f web/.env.local ]; then
  if grep -qE '^NEXT_PUBLIC_API_URL=https?://(localhost|127\.0\.0\.1)' web/.env.local; then
    warn "web/.env.local pins NEXT_PUBLIC_API_URL to localhost."
    warn "Remote visitors will fail to reach the API. Remove that line unless"
    warn "the API really is on the same machine as the browser."
  else
    pass "web/.env.local exists"
  fi
else
  pass "web/.env.local not needed — API URL is derived from the page address"
fi

# ── 8. Frontend ───────────────────────────────────────────────────────────────
step "8/10  Frontend (Next.js)"

info "Installing frontend dependencies..."
cd web
npm ci --silent 2>&1 | tail -3
pass "npm packages installed: $(ls node_modules | wc -l | tr -d ' ') packages"

# NEXT_PUBLIC_API_URL is baked into the JS bundle at build time.
# If it is still localhost, remote browsers will call their own localhost
# instead of this server, and every API request will fail.
NEXT_PUBLIC_VALUE=$(grep "^NEXT_PUBLIC_API_URL=" "$SCRIPT_DIR/web/.env.local" 2>/dev/null | cut -d= -f2-)
if echo "${NEXT_PUBLIC_VALUE}" | grep -qE "localhost|127\.0\.0\.1"; then
  echo ""
  echo -e "${RED}${BOLD}  !! DEPLOYMENT WARNING !!${NC}"
  echo -e "${RED}  NEXT_PUBLIC_API_URL is set to: ${NEXT_PUBLIC_VALUE}${NC}"
  echo -e "${RED}  Remote browsers cannot reach this server via 'localhost'.${NC}"
  echo -e "${YELLOW}  Edit web/.env.local and set:${NC}"
  echo -e "${YELLOW}    NEXT_PUBLIC_API_URL=http://<YOUR_SERVER_IP>:9097${NC}"
  echo -e "${YELLOW}  Then re-run: cd web && npm run build${NC}"
  echo ""
  warnc "Building with localhost URL — OK for local testing only"
fi

info "Building Next.js frontend (port 9098)..."
npm run build 2>&1 | tail -10
pass "Frontend build complete (.next/ ready)"
cd "$SCRIPT_DIR"

# ── 9. Directories + model download ──────────────────────────────────────────
step "9/10  Directories + Models"

for d in temp outputs "${HOME}/.cache/transcript_engine" "${HOME}/.cache/huggingface"; do
  mkdir -p "$d"
  pass "Directory: $d"
done

if [ "$HF_TOKEN_SET" = "yes" ]; then
  info "Downloading models (Parakeet ~2.5 GB + pyannote ~500 MB)..."
  info "This is a one-time download. Subsequent runs use the cache."
  if python scripts/download_models.py; then
    pass "Models downloaded and verified"
  else
    fail "Model download failed — retry: python scripts/download_models.py"
    warnc "Server can still start; models download on first job request"
  fi
else
  warnc "Skipping model download — TE_HF_TOKEN not set"
fi

# ── 10. Final validation ──────────────────────────────────────────────────────
step "10/10  Validation"

info "Running preflight checks..."
python scripts/preflight_check.py
PREFLIGHT_EXIT=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo -e "${BOLD}Setup Summary${NC}"
echo "======================================"

if [ "${#_FAILURES[@]}" -gt 0 ]; then
  echo -e "${RED}FAILED — ${#_FAILURES[@]} blocker(s):${NC}"
  for f in "${_FAILURES[@]}"; do
    echo -e "  ${RED}✗${NC}  $f"
  done
  echo ""
fi

if [ "${#_WARNINGS[@]}" -gt 0 ]; then
  echo -e "${YELLOW}Warnings (non-blocking):${NC}"
  for w in "${_WARNINGS[@]}"; do
    echo -e "  ${YELLOW}!${NC}  $w"
  done
  echo ""
fi

if [ "${#_FAILURES[@]}" -eq 0 ] && [ "$PREFLIGHT_EXIT" -eq 0 ]; then
  echo -e "${GREEN}${BOLD}✓ SETUP COMPLETE${NC}"
  echo ""
  echo "Start the server:"
  echo "  bash start_server.sh"
  echo ""
  echo "Ports:"
  echo "  Backend:  http://0.0.0.0:9097"
  echo "  Frontend: http://0.0.0.0:9098"
  EXTERNAL_IP=$(curl -s --max-time 3 https://ifconfig.me 2>/dev/null || echo "")
  if [ -n "$EXTERNAL_IP" ]; then
    echo ""
    echo "  Backend  (public): http://${EXTERNAL_IP}:9097"
    echo "  Frontend (public): http://${EXTERNAL_IP}:9098"
  fi
else
  if [ "${#_FAILURES[@]}" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ SETUP COMPLETE (with warnings)${NC}"
    echo ""
    echo "  bash start_server.sh"
  else
    echo -e "${RED}Setup incomplete — fix blockers above then re-run.${NC}"
    exit 1
  fi
fi
