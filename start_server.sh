#!/usr/bin/env bash
# start_server.sh — kept for compatibility; start.sh is now the single entry point.
#
# Both scripts previously started services with slightly different behaviour,
# which made "how do I run this?" ambiguous. start.sh now runs the backend and
# the web UI together and builds the frontend on first run, so this forwards to
# it rather than maintaining a second copy of the same logic.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[note] start_server.sh now forwards to start.sh — use 'bash start.sh' directly."
exec bash "$SCRIPT_DIR/start.sh" "$@"
