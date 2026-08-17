#!/usr/bin/env bash
# start.sh — Start Transcript Engine (backend + frontend) with one command.
#
#   bash start.sh                 # start everything
#   bash start.sh --skip-preflight   # skip the environment checks (faster restarts)
#   bash start.sh --backend-only     # API only, no web UI
#
# On a fresh clone this will install and build the frontend automatically, so
# the same command works on a laptop and on a freshly rented GPU box.
#
# Prerequisite: Python environment already installed.
#   Linux/GPU:  bash setup_gpu.sh
#   macOS:      make install
#
# Ctrl+C stops both services.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${GREEN}[start]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn] ${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

SKIP_PREFLIGHT=0
BACKEND_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --backend-only)   BACKEND_ONLY=1 ;;
    -h|--help)        sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown option: $arg (try --help)" ;;
  esac
done

# ── Load .env ─────────────────────────────────────────────────────────────────
# Variables already present in the environment win over the file. Sourcing .env
# unconditionally is the obvious implementation, but it silently overwrites what
# the caller exported — so `TE_ASR_BACKEND=whisper bash start.sh` would appear to
# do nothing, and a container platform's injected settings would be discarded in
# favour of a checked-in default. Every other dotenv loader works the other way
# round; match that.
#
# Written for bash 3.2 (what macOS ships): no associative arrays.
if [ -f .env ]; then
  _preserve=""
  # [:blank:] is space and tab only — [:space:] would also strip the newlines
  # that separate the keys, collapsing them into a single unusable token.
  for _key in $(grep -oE '^[[:blank:]]*[A-Za-z_][A-Za-z0-9_]*[[:blank:]]*=' .env \
                | tr -d '[:blank:]=' | sort -u); do
    if [ -n "${!_key+x}" ]; then
      _preserve="${_preserve}export ${_key}=$(printf '%q' "${!_key}")"$'\n'
    fi
  done

  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport

  if [ -n "$_preserve" ]; then
    eval "$_preserve"
  fi
  unset _preserve _key
fi

BACKEND_HOST="${TE_API_HOST:-0.0.0.0}"
BACKEND_PORT="${TE_API_PORT:-9097}"
FRONTEND_PORT="${TE_FRONTEND_PORT:-9098}"

# ── Python environment ────────────────────────────────────────────────────────
# A venv is preferred but not required: the deps may already be installed
# system-wide, which is common inside GPU container images.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif ! python3 -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  die "Python dependencies not found and no .venv present.
       Linux/GPU:  bash setup_gpu.sh
       macOS:      make install"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ── Preflight ─────────────────────────────────────────────────────────────────
# preflight_check.py validates a GPU production box: CUDA, NeMo, drivers. Those
# are the right blockers on a rented GPU host, where starting with a broken CUDA
# install just fails later and wastes paid time.
#
# On a machine with no NVIDIA GPU at all there is nothing to fix — the app runs
# on CPU via the Whisper backend, which is the normal local-development path. So
# the same failures are advisory there; treating them as fatal made the
# documented macOS quick start impossible without --skip-preflight.
if [ "$SKIP_PREFLIGHT" -eq 0 ] && [ -f scripts/preflight_check.py ]; then
  info "Running preflight checks..."
  if "$PYTHON_BIN" scripts/preflight_check.py; then
    :
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    die "Preflight failed on a GPU machine. Fix the items above, or re-run with --skip-preflight to start anyway."
  else
    echo ""
    warn "No NVIDIA GPU detected — the GPU checks above do not apply here."
    warn "Starting in CPU mode. Transcription will work but will be much slower,"
    warn "and the Parakeet backend is unavailable (it requires CUDA)."
    echo ""
  fi
fi

# ── Frontend build (auto) ─────────────────────────────────────────────────────
# Done here rather than failing with "not built" so a fresh clone needs exactly
# one command. Both steps are skipped when their outputs already exist.
if [ "$BACKEND_ONLY" -eq 0 ]; then
  command -v node >/dev/null 2>&1 || die "node not found. Install Node.js 20+ to serve the web UI, or use --backend-only."

  if [ ! -d web/node_modules ]; then
    info "Installing frontend dependencies (first run only, ~1 min)..."
    (cd web && npm ci --silent) || die "npm ci failed in web/"
  fi

  # Rebuild when the sources are newer than the last build, not only when
  # web/.next is missing. Otherwise `git pull && bash start.sh` silently serves
  # the previous build — the UI looks like the update simply had no effect,
  # which is a slow and confusing thing to diagnose.
  _needs_build=0
  if [ ! -f web/.next/BUILD_ID ]; then
    _needs_build=1
  elif [ -n "$(find web/app web/components web/hooks web/lib web/package.json \
                    web/next.config.ts -newer web/.next/BUILD_ID 2>/dev/null | head -1)" ]; then
    _needs_build=1
    info "Frontend sources changed since the last build — rebuilding."
  fi

  if [ "$_needs_build" -eq 1 ]; then
    info "Building frontend (~1 min)..."
    (cd web && npm run build) || die "Frontend build failed. Fix the errors above, then re-run."
  fi
fi

# ── Free ports ────────────────────────────────────────────────────────────────
# Returns the PIDs listening on a port, or nothing.
_port_pids() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$port" 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser "${port}/tcp" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true
  fi
}

# Frees a port and *confirms* it is free before returning.
#
# A plain SIGTERM plus a short sleep was not enough: `next start` supervises a
# child process that outlives the parent's TERM for a moment, so an immediate
# restart hit EADDRINUSE and the frontend died during startup. Escalate to
# SIGKILL only if the graceful stop does not take.
_free_port() {
  local port=$1 pids
  pids=$(_port_pids "$port")
  if [ -z "$pids" ]; then return 0; fi

  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 20); do            # up to 2s for a clean exit
    sleep 0.1
    if [ -z "$(_port_pids "$port")" ]; then return 0; fi
  done

  pids=$(_port_pids "$port")
  if [ -n "$pids" ]; then
    warn "Port ${port} still busy after SIGTERM — forcing."
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
      sleep 0.1
      if [ -z "$(_port_pids "$port")" ]; then return 0; fi
    done
    die "Could not free port ${port}. Something else is holding it."
  fi
}
_free_port "$BACKEND_PORT"
if [ "$BACKEND_ONLY" -eq 0 ]; then _free_port "$FRONTEND_PORT"; fi

# ── Start services ────────────────────────────────────────────────────────────
# Pre-loads models after startup so the first job doesn't pay cold-start.
export TE_WARM_MODELS="${TE_WARM_MODELS:-1}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  info "Shutting down..."
  if [ -n "$FRONTEND_PID" ]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
  if [ -n "$BACKEND_PID" ]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
  if [ -n "$FRONTEND_PID" ]; then wait "$FRONTEND_PID" 2>/dev/null || true; fi
  if [ -n "$BACKEND_PID" ]; then wait "$BACKEND_PID" 2>/dev/null || true; fi
  info "Stopped."
}
trap cleanup EXIT INT TERM

info "Starting backend on ${BACKEND_HOST}:${BACKEND_PORT}..."
uvicorn api.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --log-level info &
BACKEND_PID=$!

if [ "$BACKEND_ONLY" -eq 0 ]; then
  info "Starting frontend on 0.0.0.0:${FRONTEND_PORT}..."
  (cd web && node_modules/.bin/next start --port "$FRONTEND_PORT" --hostname 0.0.0.0) &
  FRONTEND_PID=$!
fi

# ── Wait for readiness ────────────────────────────────────────────────────────
info "Waiting for backend..."
BACKEND_READY=0
for _ in $(seq 1 240); do
  kill -0 "$BACKEND_PID" 2>/dev/null || die "Backend exited during startup. See the log above."
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${BACKEND_PORT}/health" 2>/dev/null || echo 000)" = "200" ]; then
    BACKEND_READY=1; break
  fi
  sleep 0.5
done
[ "$BACKEND_READY" -eq 1 ] || die "Backend did not respond within 120s."

if [ "$BACKEND_ONLY" -eq 0 ]; then
  info "Waiting for frontend..."
  for _ in $(seq 1 120); do
    kill -0 "$FRONTEND_PID" 2>/dev/null || die "Frontend exited during startup. See the log above."
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${FRONTEND_PORT}" 2>/dev/null || echo 000)
    case "$code" in 200|307|308) break ;; esac
    sleep 0.5
  done
fi

# ── Report GPU + URLs ─────────────────────────────────────────────────────────
read -r -d '' _GPU_SUMMARY_PY <<'PYEOF' || true
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit()
if d.get("cuda"):
    print("GPU:      {}  ({} GB total, {} GB free)".format(
        d.get("gpu"), d.get("vram_total_gb"), d.get("vram_free_gb")))
    if d.get("gpu_compatibility_error"):
        print("WARNING:  " + d["gpu_compatibility_error"])
else:
    print("GPU:      none detected - running on CPU (much slower)")
print("ASR:      {}".format(d.get("asr_backend")))
PYEOF

# Retried with a generous timeout: model warmup is running in a background
# thread at this point and is CPU-bound in Python, which can starve the event
# loop for several seconds. A single short request loses the race often enough
# that the banner would print an empty GPU line on exactly the machines where
# it matters most.
GPU_LINE=""
for _ in 1 2 3; do
  GPU_LINE=$(curl -s --max-time 15 "http://127.0.0.1:${BACKEND_PORT}/health" 2>/dev/null \
    | "$PYTHON_BIN" -c "$_GPU_SUMMARY_PY" 2>/dev/null || true)
  if [ -n "$GPU_LINE" ]; then break; fi
  sleep 2
done

# Public address matters on a rented GPU host, where the UI is opened from a
# laptop rather than the machine itself. Prefer IPv4: providers commonly expose
# forwarded IPv4 ports, and a bare IPv6 literal is not a valid URL host without
# brackets, so an unbracketed one would print a link that cannot be clicked.
PUBLIC_IP=$(curl -s4 --max-time 3 https://ifconfig.me 2>/dev/null \
         || curl -s4 --max-time 3 http://checkip.amazonaws.com 2>/dev/null || echo "")
PUBLIC_IP=$(printf '%s' "$PUBLIC_IP" | tr -d '[:space:]')
case "$PUBLIC_IP" in
  *:*) PUBLIC_IP="[${PUBLIC_IP}]" ;;   # bracket an IPv6 literal
esac

echo ""
echo -e "${GREEN}${BOLD}  Transcript Engine — READY${NC}"
echo    "  ─────────────────────────────────────────────"
if [ -n "$GPU_LINE" ]; then echo "$GPU_LINE" | sed 's/^/  /'; fi
echo    "  ─────────────────────────────────────────────"
if [ "$BACKEND_ONLY" -eq 0 ]; then
  echo -e "  Open:     ${BOLD}http://127.0.0.1:${FRONTEND_PORT}${NC}"
fi
echo    "  API:      http://127.0.0.1:${BACKEND_PORT}      (docs at /docs)"
if [ -n "$PUBLIC_IP" ]; then
  echo  ""
  echo  "  From another machine:"
  if [ "$BACKEND_ONLY" -eq 0 ]; then echo "    UI    http://${PUBLIC_IP}:${FRONTEND_PORT}"; fi
  echo  "    API   http://${PUBLIC_IP}:${BACKEND_PORT}"
  echo  "  (expose these ports on the host/provider firewall)"
fi
echo ""
echo "  Ctrl+C to stop."
echo ""

# ── Supervise ─────────────────────────────────────────────────────────────────
# `wait -n` needs bash 5.1+, so poll instead — portable across the older bash
# shipped on macOS and on minimal container images.
while true; do
  kill -0 "$BACKEND_PID" 2>/dev/null || die "Backend process exited."
  if [ "$BACKEND_ONLY" -eq 0 ]; then
    kill -0 "$FRONTEND_PID" 2>/dev/null || die "Frontend process exited."
  fi
  sleep 2
done
