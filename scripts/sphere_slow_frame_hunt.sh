#!/bin/bash
#
# Hunt for slow frames in the sphere world.
#
# The live event loop emits `frame_breakdown frame=... total_ms=...
# ... overlay_ms=... update_ms=... upload_ms=... pack_ms=... ...
# render_ms=... render_encode_ms=... render_submit_ms=...
# render_wait_ms=... gpu_pass_ms=... ...` on every frame ≥10 ms AND
# for 4 frames after any slow frame (so the recovery tail is
# visible). At 60 fps vsync every frame is ~16 ms, so this fires
# continuously — the signal is in the *outliers*, frames that blow
# past the display-refresh budget.
#
# This script routes multiple sphere scenarios through the live
# loop, captures every `frame_breakdown`, and prints:
#   - a phase-median summary across all frames (steady state)
#   - the top N slowest frames with their full breakdown
#
# No vsync-bypass needed — slow frames stand out in the percentile
# tail regardless of vsync pacing.

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | grep -E "^error" || true

RUN_SECS="${RUN_SECS:-6}"
TIMEOUT_SECS="${TIMEOUT_SECS:-15}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
TOP_SLOW="${TOP_SLOW:-5}"

SCENARIOS=(
    "plain-d8|--plain-world --plain-layers 20|--spawn-depth 8"
    "sphere-d5|--sphere-world|--spawn-on-surface --spawn-depth 5"
    "sphere-d8|--sphere-world|--spawn-on-surface --spawn-depth 8"
    "sphere-d12|--sphere-world|--spawn-on-surface --spawn-depth 12"
)

run_scenario() {
    local label=$1
    local world_args=$2
    local spawn_args=$3

    local log="tmp/perf/sphere_slow_hunt_${label}.log"
    mkdir -p "$(dirname "$log")"

    # shellcheck disable=SC2086
    timeout "${TIMEOUT_SECS}" ./target/debug/deepspace-game \
        $world_args \
        $spawn_args \
        --disable-overlay --disable-highlight \
        --harness-width "$WIDTH" --harness-height "$HEIGHT" \
        --run-for-secs "$RUN_SECS" --timeout-secs "$TIMEOUT_SECS" \
        --min-fps 1 --fps-warmup-frames 30 \
        --min-cadence-fps 1 --cadence-warmup-frames 30 \
        --max-frame-gap-ms 100000 --frame-gap-warmup-frames 30 \
        --suppress-startup-logs \
        2> "$log" || true

    echo "=== $label ==="
    python3 - "$log" "$TOP_SLOW" <<'PY'
import re, sys, statistics
from collections import defaultdict

log_path = sys.argv[1]
top_n = int(sys.argv[2])

# Parse `key=value` pairs out of frame_breakdown lines.
rows = []
with open(log_path) as f:
    for line in f:
        if not line.startswith("frame_breakdown"):
            continue
        kv = {}
        for m in re.finditer(r"(\w+)=([-\d.]+|true|false)", line):
            key, val = m.group(1), m.group(2)
            if val in ("true", "false"):
                kv[key] = val == "true"
            else:
                try:
                    kv[key] = float(val)
                except ValueError:
                    pass
        rows.append(kv)

if not rows:
    print("  (no frame_breakdown lines captured)")
    print()
    sys.exit(0)

phases = [
    "total_ms", "dt_ms", "overlay_ms", "update_ms", "upload_ms",
    "pack_ms", "ribbon_build_ms", "tree_write_ms", "bg_rebuild_ms",
    "ribbon_write_ms", "camera_write_ms", "highlight_ms",
    "render_ms", "render_encode_ms", "render_submit_ms",
    "render_wait_ms", "gpu_pass_ms",
]

# `render_ms - (encode + submit + wait) ≈ surface acquire + misc`.
# Derive it per-row so the "hidden" acquire cost shows up in the
# table.
for r in rows:
    r["acquire_approx_ms"] = r.get("render_ms", 0.0) \
        - r.get("render_encode_ms", 0.0) \
        - r.get("render_submit_ms", 0.0) \
        - r.get("render_wait_ms", 0.0)

phases_for_summary = phases + ["acquire_approx_ms"]

# Median + 95th per phase.
print(f"  steady-state medians ({len(rows)} frames):")
col_w = 18
def med(xs): return statistics.median(xs) if xs else 0.0
def p95(xs):
    if not xs: return 0.0
    s = sorted(xs)
    return s[int(len(s) * 0.95)]

sig_phases = [
    "total_ms", "render_ms", "acquire_approx_ms",
    "render_encode_ms", "render_submit_ms", "render_wait_ms",
    "gpu_pass_ms", "update_ms", "upload_ms", "pack_ms",
    "tree_write_ms", "bg_rebuild_ms",
]
header = "    " + "".join(f"{p:>{col_w}}" for p in sig_phases)
print(header)
med_row = "    " + "".join(f"{med([r.get(p, 0.0) for r in rows]):>{col_w}.2f}" for p in sig_phases)
p95_row = "    " + "".join(f"{p95([r.get(p, 0.0) for r in rows]):>{col_w}.2f}" for p in sig_phases)
print(f"  median:\n{med_row}")
print(f"  p95:\n{p95_row}")

print()
print(f"  top {top_n} slowest frames (by total_ms):")
slow = sorted(rows, key=lambda r: -r.get("total_ms", 0.0))[:top_n]
for r in slow:
    parts = [f"{p}={r.get(p, 0.0):.2f}" for p in sig_phases]
    print("    " + "  ".join(parts))
print()
PY
}

echo "=== sphere slow-frame hunt (${WIDTH}×${HEIGHT}, ${RUN_SECS}s per scenario) ==="
echo
for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r label world_args spawn_args <<< "$entry"
    run_scenario "$label" "$world_args" "$spawn_args"
done
