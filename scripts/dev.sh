#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/dev.sh [--import]
#   --import   Load and stamp a .vox model on startup (debug_import feature)

FEATURES=""
for arg in "$@"; do
    case "$arg" in
        --import) FEATURES="debug_import" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

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
# Note: --features dev (dynamic linking) is incompatible with wry on macOS
# due to objc-sys symbol conflicts. Plain cargo run is fast enough with
# incremental builds (~3-5s).
if [ -n "$FEATURES" ]; then
    cargo run --features "$FEATURES"
else
    cargo run
fi
