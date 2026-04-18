#!/usr/bin/env bash
# Browser dev loop: build the React UI bundle once, then trunk-serve
# the WASM/WebGPU build at http://localhost:8080. Counterpart to
# `scripts/dev.sh` (which starts vite + the native binary with the wry
# overlay). Requires: trunk, wasm32-unknown-unknown rustup target,
# Chrome / Edge / any WebGPU-enabled browser.
set -euo pipefail

PORT="${PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {
    if [[ -n "${TRUNK_PID:-}" ]]; then
        kill "$TRUNK_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 1. Build the React UI bundle once. Trunk's index.html copies
#    ui/dist/{ui.css,ui.js} into the served root, so it must exist.
if [[ ! -f "$ROOT/ui/dist/ui.js" ]]; then
    echo "Building React UI bundle (one-time)…"
    (cd "$ROOT/ui" && npm install --no-audit --no-fund && npm run build)
fi

# 2. Trunk's watch-ignore list references dirs that may not exist in
#    a fresh worktree. Create empties so the watcher's canonicalize
#    check doesn't trip.
mkdir -p "$ROOT/.claude" "$ROOT/external" "$ROOT/tests" "$ROOT/docs"

# 3. Trunk-serve. --no-autoreload prevents trunk from re-injecting on
#    every file save (we restart manually when iterating Rust).
echo "Starting trunk on http://localhost:$PORT"
cd "$ROOT"
trunk serve --port "$PORT" --no-autoreload &
TRUNK_PID=$!
wait "$TRUNK_PID"
