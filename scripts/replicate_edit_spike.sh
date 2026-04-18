#!/bin/bash
#
# Self-sufficient repro for the 40 ms first-frame-post-edit wall-clock
# spike. Uses `wall_ms` frame-to-frame DELTA, which is the only
# reliable harness signal for what the user actually feels — the
# Apple Silicon timestamp counters (`gpu_pass_ms`) coincidentally
# match the true cost on edit frames but `submitted_done_ms` does
# NOT (see draw.rs:527 comment re: non-monotonic timestamps).
#
# Scenario: 20-layer soldier world, camera landed on soldier via
# `fly_to_surface`, five alternating break/place edits spaced 5
# frames apart. Baseline is the four frames before each edit.
#
# Output:
#   - summary: per-edit wall-delta + baseline delta + overhead
#   - pass/fail: fails if mean edit-frame overhead exceeds budget
#
# Environment:
#   EDIT_SPIKE_BUDGET_MS (default 5) — max mean wall-delta overhead
#   before the script exits non-zero. Lets a future fix assert "we
#   got this under 5 ms" without manual inspection.

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | grep -E "(error|warning: unused)" || true

BUDGET_MS="${EDIT_SPIKE_BUDGET_MS:-5}"

./target/debug/deepspace-game --render-harness \
    --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 20 --vox-interior-depth 13 --lod-base-depth 20 --shader-stats \
    --spawn-xyz 0.5 1.5 0.12 --spawn-depth 10 --spawn-pitch -1.5 \
    --disable-overlay --harness-width 600 --harness-height 400 \
    --script "wait:10,fly_to_surface,wait:5,break,wait:5,place,wait:5,break,wait:5,place,wait:5,break,wait:5" \
    --perf-trace tmp/perf/edit-spike.csv --perf-trace-warmup 3 \
    --exit-after-frames 120 --timeout-secs 30 --suppress-startup-logs \
    2>&1 | grep -E "render_harness_(timing|workload)" | head -3

python3 - "$BUDGET_MS" <<'PY'
import csv, sys, statistics
budget = float(sys.argv[1])
rows = list(csv.DictReader(open('tmp/perf/edit-spike.csv')))
rows.sort(key=lambda r: int(r['frame']))

# Compute wall-delta (ms) between consecutive frames. This is the
# actual per-frame wall-clock cost the user feels; gpu_pass_ms and
# submitted_done_ms are not reliable on macOS under our setup.
walls = [(int(r['frame']), float(r['wall_ms'])) for r in rows]
deltas = {}
for (f1, w1), (f2, w2) in zip(walls, walls[1:]):
    if f2 == f1 + 1:
        deltas[f2] = w2 - w1  # wall time between sampling frame f1 and f2

# Edit frames in this scenario: fly_to_surface at t=40, break/place
# at t=45,50,55,60,65. Script command fires AFTER the sample for
# frame N=schedule_t-1, so the NEXT frame's wall-delta captures the
# full cost of (edit work + render of modified tree).
edit_schedule = [45, 50, 55, 60, 65]
edit_frame_deltas = {t: deltas[t] for t in edit_schedule if t in deltas}

# Baseline = the 4 frames immediately preceding each edit (same
# camera, same tree — pure rendering cost). Skip frames that are
# themselves edits or missing.
baseline_deltas = []
for e in edit_schedule:
    for k in range(e - 4, e):
        if k in deltas and k not in edit_schedule:
            baseline_deltas.append(deltas[k])

if not edit_frame_deltas or not baseline_deltas:
    print(f"  ERROR: could not identify edit/baseline frames in trace")
    sys.exit(2)

baseline_median = statistics.median(baseline_deltas)
edit_mean = statistics.mean(edit_frame_deltas.values())
edit_max = max(edit_frame_deltas.values())
overhead_mean = edit_mean - baseline_median
overhead_max = edit_max - baseline_median

print(f"=== Edit-frame wall-clock spike ===")
print(f"  baseline  (n={len(baseline_deltas)}, non-edit frames):  median={baseline_median:6.2f} ms")
print(f"  edit frames (n={len(edit_frame_deltas)}):")
for f, d in edit_frame_deltas.items():
    print(f"    frame={f:>3}  wall_delta={d:6.2f} ms  overhead={d - baseline_median:+6.2f} ms")
print(f"  edit overhead:  mean={overhead_mean:+6.2f} ms  max={overhead_max:+6.2f} ms")
print(f"  budget={budget:.1f} ms")

# Per-phase breakdown: compare edit-frame phase times to baseline.
# Anything that spikes on edit frames is a candidate for the cost.
phase_fields = [
    'update_ms', 'camera_write_ms',
    'upload_total_ms', 'pack_ms', 'ribbon_build_ms',
    'tree_write_ms', 'ribbon_write_ms', 'bind_group_rebuild_ms',
    'highlight_ms', 'highlight_raycast_ms', 'highlight_set_ms',
    'render_total_ms', 'render_encode_ms', 'render_submit_ms',
    'render_wait_ms', 'gpu_pass_ms', 'gpu_readback_ms',
    'submitted_done_ms',
]
by_frame = {int(r['frame']): r for r in rows}
print(f"\n=== Per-phase breakdown (edit vs baseline median) ===")
print(f"  {'phase':<25} {'baseline':>10} {'edit_mean':>10} {'edit_max':>10} {'delta_mean':>12}")
for phase in phase_fields:
    bl_vals = [float(by_frame[k][phase] or 0) for k in range(walls[0][0], walls[-1][0]+1) if k in by_frame and k not in edit_schedule]
    ed_vals = [float(by_frame[e][phase] or 0) for e in edit_schedule if e in by_frame]
    if not bl_vals or not ed_vals:
        continue
    bl_med = statistics.median(bl_vals)
    ed_mean = statistics.mean(ed_vals)
    ed_max = max(ed_vals)
    delta = ed_mean - bl_med
    flag = "  ***" if delta > 5.0 else ""
    print(f"  {phase:<25} {bl_med:>8.2f}   {ed_mean:>8.2f}   {ed_max:>8.2f}   {delta:>+10.2f}{flag}")

if overhead_mean > budget:
    print(f"\n  FAIL: mean overhead {overhead_mean:.2f} > budget {budget:.2f}")
    sys.exit(1)
print(f"\n  PASS")
PY
