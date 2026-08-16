#!/bin/bash
# package_release.sh — Create a clean deployment.zip ready to upload to a server.
#
# Removes all build artifacts, caches, and runtime files, then zips the repo.
# The zip is safe to extract on a fresh server and run: bash setup_server.sh
#
# Usage:
#   bash package_release.sh
#
# Output:
#   deployment.zip   (in the parent directory of this repo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[32m'
RED='\033[31m'
BOLD='\033[1m'
NC='\033[0m'

info() { echo -e "${GREEN}[package]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

REPO_NAME="$(basename "$SCRIPT_DIR")"
OUTPUT_ZIP="${SCRIPT_DIR}/../deployment.zip"
OUTPUT_ZIP="$(cd "$(dirname "$OUTPUT_ZIP")" && pwd)/$(basename "$OUTPUT_ZIP")"

echo ""
echo -e "${BOLD}Transcript Engine — Package Release${NC}"
echo "====================================="
echo "Source: $SCRIPT_DIR"
echo "Output: $OUTPUT_ZIP"
echo ""

# ── Safety: refuse to delete .env or .env.local (secrets) from the zip ────────
info "Note: .env and web/.env.local are in .gitignore and will NOT be included."
info "Set these on the server after upload."
echo ""

# ── Clean build artifacts ─────────────────────────────────────────────────────
info "Removing build artifacts..."

# Python
find . -type d -name "__pycache__"    -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache"  -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache"    -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info"     -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ruff_cache"    -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc"          -not -path "./.git/*" -delete 2>/dev/null || true
find . -type f -name "*.pyo"          -not -path "./.git/*" -delete 2>/dev/null || true

# Python virtualenv
rm -rf .venv

# Node.js
rm -rf web/node_modules web/.next

# Temporary files and outputs (runtime, not source)
rm -rf temp/*  2>/dev/null || true
rm -rf outputs/ 2>/dev/null || true
rm -rf output/ 2>/dev/null || true

# Logs
find . -name "*.log" -not -path "./.git/*" -delete 2>/dev/null || true

# macOS
find . -name ".DS_Store" -not -path "./.git/*" -delete 2>/dev/null || true

echo "  Cleaned."

# ── Create zip ────────────────────────────────────────────────────────────────
info "Creating deployment.zip..."

rm -f "$OUTPUT_ZIP"

# Exclude secrets, git history, archives, notebooks, and test fixtures
zip -r "$OUTPUT_ZIP" . \
  --exclude "*.git/*" \
  --exclude "*.env" \
  --exclude "*/.env.local" \
  --exclude "*/temp/*" \
  --exclude "*/outputs/*" \
  --exclude "*/output/*" \
  --exclude "*/archive/*" \
  --exclude "*/notebooks/*" \
  --exclude "*/tests/fixtures/audio/*" \
  --exclude "*/__pycache__/*" \
  --exclude "*/.venv/*" \
  --exclude "*/node_modules/*" \
  --exclude "*/.next/*" \
  --exclude "*/.pytest_cache/*" \
  --exclude "*/.mypy_cache/*" \
  --exclude "*/.ruff_cache/*" \
  --exclude "*.egg-info/*" \
  --exclude "*.pyc" \
  --exclude "*.pyo" \
  --exclude "*.log" \
  --exclude ".DS_Store" \
  --exclude "*.tsbuildinfo" \
  2>/dev/null

ZIP_SIZE=$(du -sh "$OUTPUT_ZIP" | cut -f1)
echo ""
echo -e "${GREEN}${BOLD}✓ deployment.zip created${NC}"
echo ""
echo "  Size:  ${ZIP_SIZE}"
echo "  Path:  ${OUTPUT_ZIP}"
echo ""
echo "Upload and deploy:"
echo "  scp ${OUTPUT_ZIP} user@server:/opt/"
echo "  ssh user@server"
echo "  cd /opt && unzip deployment.zip -d transcript-engine && cd transcript-engine"
echo "  cp .env.example .env && nano .env           # set TE_HF_TOKEN"
echo "  cp web/.env.local.example web/.env.local    # set NEXT_PUBLIC_API_URL"
echo "  bash setup_server.sh"
echo "  bash start_server.sh"
