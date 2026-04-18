#!/bin/bash
#
# Live-path perf comparison: sphere world vs plain world vs vox model.
#
# Crucial detail: `--render-harness` uses `Renderer::render_offscreen`,
# which skips the CAMetalLayer swap-chain acquire/present path where
# vsync, compositor backpressure, and surface-texture contention
# live. The user's in-game frame cost is NOT captured by the
# offscreen harness — offscreen shows sphere 1.7× plain, but live
# shows sphere ~4× plain. So this script routes through the live
# event loop instead (`--run-for-secs` without `--screenshot` flips
# `prefers_live_loop()` to true).
#
# Output is `test_runner: perf summary ...` per scenario.

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | grep -E "^error" || true

RUN_SECS="${RUN_SECS:-4}"
TIMEOUT_SECS="${TIMEOUT_SECS:-10}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"

# Scenarios to compare. Each is `label|world-args|spawn-args`.
# Spawn-args should place the camera somewhere with real hits — empty
# sky views understate the ray-march cost, which is what we want to
# measure.
SCENARIOS=(
    "plain-d6|--plain-world --plain-layers 20|--spawn-depth 6"
    "plain-d8|--plain-world --plain-layers 20|--spawn-depth 8"
    "plain-d12|--plain-world --plain-layers 20|--spawn-depth 12"
    "sphere-d3|--sphere-world|--spawn-on-surface --spawn-depth 3"
    "sphere-d5|--sphere-world|--spawn-on-surface --spawn-depth 5"
    "sphere-d8|--sphere-world|--spawn-on-surface --spawn-depth 8"
)

run_scenario() {
    local label=$1
    local world_args=$2
    local spawn_args=$3

    # `--run-for-secs` + no `--screenshot` triggers the live loop
    # (see `TestConfig::prefers_live_loop`). `--live-sample-every 30`
    # emits `render_live_sample` every 30 rendered frames so we can
    # measure per-phase cost without waiting for a renderer_slow
    # event. `--min-fps 1 ...` are failsafes to suppress perf-exit.
    # shellcheck disable=SC2086
    out=$(timeout "${TIMEOUT_SECS}" ./target/debug/deepspace-game \
        $world_args \
        $spawn_args \
        --disable-overlay --disable-highlight \
        --harness-width "$WIDTH" --harness-height "$HEIGHT" \
        --run-for-secs "$RUN_SECS" --timeout-secs "$TIMEOUT_SECS" \
        --live-sample-every 30 \
        --min-fps 1 --fps-warmup-frames 30 \
        --min-cadence-fps 1 --cadence-warmup-frames 30 \
        --max-frame-gap-ms 100000 --frame-gap-warmup-frames 30 \
        --suppress-startup-logs 2>&1)

    summary=$(printf '%s\n' "$out" | grep 'test_runner: perf summary' | head -1)
    slow_count=$(printf '%s\n' "$out" | grep -c 'renderer_slow' || true)
    # Median of live_sample phase cost (skip early warmup samples).
    live_samples=$(printf '%s\n' "$out" | grep 'render_live_sample' | tail -n +3)
    acquire_med=$(printf '%s\n' "$live_samples" | awk -F'acquire_ms=' '/acquire_ms=/ {n=split($2, a, " "); print a[1]}' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)]}')
    encode_med=$(printf  '%s\n' "$live_samples" | awk -F'encode_ms='  '/encode_ms=/  {n=split($2, a, " "); print a[1]}' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)]}')
    submit_med=$(printf  '%s\n' "$live_samples" | awk -F'submit_ms='  '/submit_ms=/  {n=split($2, a, " "); print a[1]}' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)]}')
    present_med=$(printf '%s\n' "$live_samples" | awk -F'present_ms=' '/present_ms=/ {n=split($2, a, " "); print a[1]}' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)]}')
    total_med=$(printf   '%s\n' "$live_samples" | awk -F'total_ms='   '/total_ms=/   {n=split($2, a, " "); print a[1]}' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)]}')

    if [ -z "$summary" ]; then
        printf "  %-16s  (no perf summary — test did not complete)\n" "$label"
        return
    fi

    # Extract the numeric fields from the summary line.
    avg_frame_fps=$(printf '%s\n' "$summary" | sed -n 's/.*avg_frame_fps=\([0-9.]*\).*/\1/p')
    avg_cadence_fps=$(printf '%s\n' "$summary" | sed -n 's/.*avg_cadence_fps=\([0-9.]*\).*/\1/p')
    worst_frame_ms=$(printf '%s\n' "$summary" | sed -n 's/.*worst_frame_ms=\([0-9.]*\).*/\1/p')
    worst_dt_ms=$(printf '%s\n' "$summary" | sed -n 's/.*worst_dt_ms=\([0-9.]*\).*/\1/p')
    samples=$(printf '%s\n' "$summary" | sed -n 's/.*samples=\([0-9]*\).*/\1/p')

    printf "  %-16s  frame_fps=%6s  cadence_fps=%6s  worst_frame=%5sms  slow=%s  | medians acq=%5s enc=%5s sub=%5s pres=%5s total=%5s\n" \
        "$label" "$avg_frame_fps" "$avg_cadence_fps" "$worst_frame_ms" "$slow_count" \
        "$acquire_med" "$encode_med" "$submit_med" "$present_med" "$total_med"
}

echo "=== live-path perf (${WIDTH}×${HEIGHT}, ${RUN_SECS}s per scenario) ==="
echo
for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r label world_args spawn_args <<< "$entry"
    run_scenario "$label" "$world_args" "$spawn_args"
done
echo
echo "  Interpretation:"
echo "    - `frame_fps` = 1000 / avg frame cost in ms. The primary "slowness" metric."
echo "    - `cadence_fps` = 1000 / avg dt between frame starts. Closer to perceived fps if"
echo "      vsync is holding cadence but per-frame cost is variable."
echo "    - `slow_frames` = count of `renderer_slow` events (>= 30 ms total)."
