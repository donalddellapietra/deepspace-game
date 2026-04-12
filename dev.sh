#!/usr/bin/env bash
set -euo pipefail

# Start Vite dev server in background
(cd ui && npm run dev -- --port 5173) &
VITE_PID=$!
trap "kill $VITE_PID 2>/dev/null" EXIT

# Wait for Vite to be ready
echo "Waiting for Vite..."
until curl -s http://localhost:5173 > /dev/null 2>&1; do sleep 0.2; done
echo "Vite ready"

# Start native game (WebView overlay connects to Vite)
cargo run --features dev
