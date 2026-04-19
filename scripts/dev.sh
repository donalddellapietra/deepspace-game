#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    pkill -f "deepspace-game" 2>/dev/null || true
    pkill -f "node.*vite" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start Vite dev server in background
(cd ui && npx vite --port 5173) &
echo "Waiting for Vite..."
until curl -s http://localhost:5173 > /dev/null 2>&1; do sleep 0.2; done
echo "Vite ready"

# Start native game (wry WebView overlay loads from Vite)
# Any extra arguments to this script are forwarded to the game binary
# after `--`, so e.g. `scripts/dev.sh --menger-world --lod-pixels 2.0`
# launches with those flags active.
cargo run --bin deepspace-game -- "$@"
