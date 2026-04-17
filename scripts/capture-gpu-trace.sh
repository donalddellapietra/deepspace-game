#!/usr/bin/env bash
#
# Capture a Metal GPU Counters trace for the ray-march shader on macOS.
# Uses xctrace (ships with Xcode command-line tools) to record hardware
# counters — fragment occupancy, ALU utilization, buffer-read limiter,
# cache hit rates, TLB miss rate, etc. — while the harness runs a
# specific scenario. Produces a .trace bundle you can open in Instruments
# or parse via scripts/parse-metal-trace.py.
#
# Usage:
#   scripts/capture-gpu-trace.sh <label> -- <harness args>
#
#   # Example: capture the slow-soldier-at-zoom-4 scenario.
#   scripts/capture-gpu-trace.sh slow-soldier -- \
#     --render-harness --vox-model assets/vox/soldier_729.vxs \
#     --plain-layers 8 --spawn-xyz 1.15 1.1 1.04 --spawn-depth 5 \
#     --disable-overlay --harness-width 2560 --harness-height 1440 \
#     --exit-after-frames 300 --timeout-secs 15 --suppress-startup-logs
#
# Output: tmp/trace/<label>.trace (open in Instruments.app or feed to
# scripts/parse-metal-trace.py).
#
# Prereq: the binary is built via `cargo build --bin deepspace-game --release`.
# A release build matters — debug builds obscure shader-level hotspots.

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    sed -n '2,24p' "$0" | sed 's/^# \?//'
    exit 2
fi

LABEL="$1"
shift 2  # consume label + "--"

TRACE_DIR="tmp/trace"
TRACE_PATH="$TRACE_DIR/$LABEL.trace"

mkdir -p "$TRACE_DIR"
rm -rf "$TRACE_PATH"

BIN="$(pwd)/target/release/deepspace-game"
if [ ! -x "$BIN" ]; then
    echo "Error: $BIN not found. Run: cargo build --bin deepspace-game --release"
    exit 1
fi

echo "Capturing Metal GPU Counters → $TRACE_PATH"
echo "Scenario args: $*"
echo

xcrun xctrace record \
    --instrument "Metal GPU Counters" \
    --output "$TRACE_PATH" \
    --launch -- "$BIN" "$@"

echo
echo "Trace saved to $TRACE_PATH"
echo "Parse with: scripts/parse-metal-trace.py $TRACE_PATH"
echo "Or open in Instruments.app"
