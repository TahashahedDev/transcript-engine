#!/bin/bash
# setup_gpu.sh — One-command setup for a fresh Vast.ai Ubuntu+CUDA GPU instance.
#
# Allowed manual inputs:
#   1. Vast.ai credentials (used to rent the instance)
#   2. HuggingFace token (set in .env)
#   3. git clone <repo>
#   4. bash setup_gpu.sh
#
# Everything else — Python 3.12, CUDA PyTorch, NeMo, ffmpeg, model downloads,
# environment validation — happens automatically.
#
# Usage:
#   git clone <repo> transcript-engine && cd transcript-engine
#   cp .env.example .env
#   nano .env   # set TE_HF_TOKEN=hf_...
#   bash setup_gpu.sh
#   bash start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Terminal colors ───────────────────────────────────────────────────────────
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

# ── PASS/FAIL tracker ─────────────────────────────────────────────────────────
_FAILURES=()
_WARNINGS=()

pass()  { echo -e "  ${GREEN}✓${NC}  $*"; }
fail()  { echo -e "  ${RED}✗${NC}  $*"; _FAILURES+=("$*"); }
warnc() { echo -e "  ${YELLOW}!${NC}  $*"; _WARNINGS+=("$*"); }

# ── 0. Banner ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Transcript Engine — GPU Bootstrap${NC}"
echo "======================================"
echo "Target: Vast.ai Ubuntu 22.04+ with NVIDIA CUDA"
echo "Time:   ~10-20 min (first-time model download)"
echo ""

# ── 1. NVIDIA driver ──────────────────────────────────────────────────────────
step "1/9  NVIDIA Driver"

if ! command -v nvidia-smi &>/dev/null; then
  die "nvidia-smi not found. This script requires a GPU instance with NVIDIA drivers pre-installed."
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+" || echo "")

if [ -z "$GPU_INFO" ] || [ -z "$CUDA_VER" ]; then
  die "nvidia-smi returned no GPU info. Check NVIDIA driver installation."
fi

pass "GPU: $GPU_INFO"
pass "CUDA Driver: $CUDA_VER"

# Parse CUDA major.minor for PyTorch index selection
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
CUDA_MINOR="${CUDA_MINOR:-0}"  # default to 0 if minor version is missing

# Select matching PyTorch CUDA wheel index
# cu124 works with CUDA 12.4+ and is backward-compatible with 12.2/12.3
if [ "$CUDA_MAJOR" = "11" ]; then
  TORCH_CUDA_TAG="cu118"
elif [ "$CUDA_MAJOR" = "12" ]; then
  if [ "$CUDA_MINOR" -le 1 ] 2>/dev/null; then
    TORCH_CUDA_TAG="cu121"
  else
    TORCH_CUDA_TAG="cu124"
  fi
else
  TORCH_CUDA_TAG="cu124"  # forward-compatible default for CUDA 13+
fi
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
info "Will install PyTorch for CUDA ${CUDA_VER} (index: ${TORCH_CUDA_TAG})"

# ── 2. System packages ────────────────────────────────────────────────────────
step "2/9  System Packages"

export DEBIAN_FRONTEND=noninteractive
info "Running apt-get update..."
apt-get update -qq 2>&1 | tail -2

# Python 3.12 is not in Ubuntu 22.04 default repos — add deadsnakes PPA
if ! python3.12 --version &>/dev/null 2>&1; then
  info "Adding deadsnakes PPA for Python 3.12..."
  apt-get install -y -qq software-properties-common 2>&1 | tail -2
  add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -3
  apt-get update -qq 2>&1 | tail -2
fi

info "Installing system packages..."
# python3.12-distutils does NOT exist on Ubuntu 24.04 (bundled in python3.12).
# On Ubuntu 22.04 with deadsnakes PPA it is available but not required — pip handles it.
apt-get install -y -qq \
  python3.12 python3.12-venv python3.12-dev \
  ffmpeg \
  git curl wget \
  build-essential libsndfile1 libgomp1 \
  ca-certificates gnupg psmisc \
  2>&1 | tail -5

# Verify installs
if python3.12 --version &>/dev/null 2>&1; then
  PY_VER=$(python3.12 --version)
  pass "Python: $PY_VER"
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

# ── 3. Virtual environment ────────────────────────────────────────────────────
step "3/9  Virtual Environment"

if [ ! -d .venv ]; then
  info "Creating .venv with Python 3.12..."
  python3.12 -m venv .venv
  pass "Created .venv"
else
  pass ".venv already exists — skipping creation"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python --version
pip install --upgrade pip setuptools wheel --quiet
pass "pip upgraded"

# ── 4. PyTorch with CUDA ──────────────────────────────────────────────────────
step "4/9  PyTorch + CUDA"

info "Installing PyTorch (CUDA ${CUDA_VER}, index=${TORCH_CUDA_TAG})..."
pip install torch torchaudio --index-url "$TORCH_INDEX" --quiet

# Verify CUDA is actually available
CUDA_CHECK=$(python -c "
import torch
ok = torch.cuda.is_available()
if ok:
    print(f'PASS: {torch.cuda.get_device_name(0)} — {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM')
else:
    print('FAIL: torch.cuda.is_available() returned False')
" 2>&1)

if echo "$CUDA_CHECK" | grep -q "^PASS"; then
  pass "PyTorch CUDA: $CUDA_CHECK"
else
  fail "PyTorch CUDA check failed: $CUDA_CHECK"
  err "Try: pip install torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
fi

# ── 5. Transcript Engine + NeMo ───────────────────────────────────────────────
step "5/9  Transcript Engine + NeMo"

info "Installing transcript-engine base dependencies..."
pip install -e "." --quiet 2>&1 | tail -3

info "Installing NeMo (nemo_toolkit[asr]) — this may take 5-10 min..."
# NeMo has known version sensitivity. Install separately with verbose output
# so failures are visible, and allow continued operation if NeMo fails
# (Whisper backend still works).
if pip install "nemo_toolkit[asr]>=1.23.0" --quiet 2>&1 | tail -5; then
  # Verify NeMo actually works
  NEMO_CHECK=$(python -c "
import nemo
from nemo.collections.asr.models import EncDecRNNTBPEModel
print(f'nemo_toolkit {nemo.__version__}')
" 2>&1)
  if echo "$NEMO_CHECK" | grep -q "nemo_toolkit"; then
    pass "NeMo: $NEMO_CHECK"
  else
    warnc "NeMo import check failed: $NEMO_CHECK"
    warnc "Parakeet ASR will not be available. Whisper backend still works."
  fi
else
  warnc "NeMo install failed (pip returned non-zero)"
  warnc "Try manually: pip install 'nemo_toolkit[asr]>=1.23.0'"
  warnc "Parakeet ASR will not be available. Whisper backend still works."
fi

# ── 6. Environment (.env) ─────────────────────────────────────────────────────
step "6/9  Environment"

if [ ! -f .env ]; then
  cp .env.example .env
  warn ".env created from .env.example — TE_HF_TOKEN not yet set."
  warn "Set it before running model download:"
  warn "  nano .env   # set TE_HF_TOKEN=hf_..."
else
  pass ".env exists"
fi

# Source .env into the current shell so downstream checks see it
set -o allexport
# shellcheck disable=SC1091
source .env 2>/dev/null || true
set +o allexport

# Check critical env vars
if [ -n "${TE_HF_TOKEN:-}" ] && [ "${TE_HF_TOKEN:-}" != "hf_your_token_here" ]; then
  pass "TE_HF_TOKEN is set (${TE_HF_TOKEN:0:8}...)"
  HF_TOKEN_SET=yes
else
  warnc "TE_HF_TOKEN not set — model download and diarization will be skipped"
  HF_TOKEN_SET=no
fi

if [ "${TE_ASR_BACKEND:-}" = "parakeet" ]; then
  pass "TE_ASR_BACKEND=parakeet"
else
  warnc "TE_ASR_BACKEND not set to 'parakeet' — check .env"
fi

# ── 7. Cache directories ──────────────────────────────────────────────────────
step "7/9  Cache Directories"

# Create all required directories so they exist before model downloads
DIRS=(
  "temp"
  "outputs"
  "${HOME}/.cache/transcript_engine"
  "${HOME}/.cache/huggingface"
)
for d in "${DIRS[@]}"; do
  mkdir -p "$d"
  pass "Directory: $d"
done

# ── 8. Model download ─────────────────────────────────────────────────────────
step "8/9  Model Download"

if [ "$HF_TOKEN_SET" = "yes" ]; then
  info "Downloading models (Parakeet ~2.5 GB + pyannote ~500 MB)..."
  info "This is a one-time download. Subsequent runs reuse the cache."
  if python scripts/download_models.py; then
    pass "Models downloaded successfully"
  else
    fail "Model download failed — check error above"
    err "You can retry manually: python scripts/download_models.py"
  fi
else
  warnc "Skipping model download — TE_HF_TOKEN not set"
  warnc "After setting TE_HF_TOKEN in .env, run: python scripts/download_models.py"
fi

# ── 9. Final validation ───────────────────────────────────────────────────────
step "9/9  Final Validation"

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
  echo -e "${GREEN}${BOLD}✓ SETUP COMPLETE — READY FOR GPU TRANSCRIPTION${NC}"
  echo ""
  echo "Start the API:"
  echo "  bash start.sh"
  echo ""
  echo "API will be available at:"
  echo "  http://0.0.0.0:${TE_API_PORT:-9097}"
  EXTERNAL_IP=$(curl -s --max-time 3 https://ifconfig.me 2>/dev/null || echo "")
  if [ -n "$EXTERNAL_IP" ]; then
    echo "  http://${EXTERNAL_IP}:${TE_API_PORT:-9097}  (Vast.ai public IP)"
  fi
else
  echo -e "${RED}Setup incomplete — address the blockers above before starting.${NC}"
  echo ""
  echo "After fixing, re-run:  bash setup_gpu.sh"
  exit 1
fi
