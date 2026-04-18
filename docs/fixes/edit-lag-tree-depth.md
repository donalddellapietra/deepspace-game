# Edit-frame lag: `tree_depth()` O(library) DFS on every edit

## Symptom

User reported: breaking or placing a block caused a 40–50 ms frame
in the live dev build. FPS dropped from 60 to ~50 on each edit.

## One-line cause

`WorldState::tree_depth()` did a full recursive DFS over every
`NodeId` reachable from the root (~94k nodes on the soldier test
world) with a freshly-allocated `HashMap` as a memo. It was called
unconditionally from `App::upload_tree()` on every edit. That DFS
was the ~38 ms spike.

## Fix

One field on `Node`:

```rust
pub struct Node {
    pub children: Children,
    pub kind: NodeKind,
    pub ref_count: u32,
    pub representative_block: u8,
    pub uniform_type: u8,
    pub depth: u32,  // ← new, populated in insert_with_kind
}
```

`insert_with_kind` sets it from the already-cached depths of
`Child::Node` children — no recursion. `WorldState::tree_depth()`
becomes `library.get(root).map(|n| n.depth).unwrap_or(0)` — O(1).

Same caching pattern the struct already used for
`representative_block` and `uniform_type`.

## Result

Measured via `scripts/replicate_edit_spike.sh` (20-layer soldier
world, interior_depth=13, 5 alternating break/place edits):

| Metric                 | before   | after     |
| ---------------------- | -------- | --------- |
| Edit-frame wall delta  | 40.5 ms  | 1.5 ms    |
| Edit overhead (mean)   | +38.1 ms | +0.17 ms  |
| Edit overhead (max)    | +39.5 ms | +0.24 ms  |

Edit frames are now statistically indistinguishable from baseline.

## How I found it (systematic debugging)

Each iteration below was driven by a single log line or env var
added in 1–3 lines of code, run against a scripted harness, and
analyzed from the CSV trace. No user interaction.

### 1. Reproduce in the harness

The harness emits a per-frame CSV (`wall_ms, pack_ms, gpu_pass_ms,
submitted_done_ms, render_wait_ms, …`). I wrote
`scripts/replicate_edit_spike.sh`:

```bash
--script "wait:10,fly_to_surface,wait:5,break,wait:5,place,…"
```

where `fly_to_surface` was a new script command (10 lines) that
raycasts straight down and teleports the camera above whatever it
hits — removes the "camera isn't pointing at anything" problem
from testing.

The script parses `wall_ms` deltas between consecutive frames
(`delta[N] = wall_ms[N] - wall_ms[N-1]`) and separates edit frames
from baseline.

> **Key insight:** `wall_ms` is monotonically increasing total
> elapsed time since harness start. Only its frame-to-frame delta
> is meaningful as per-frame cost. `gpu_pass_ms` and
> `submitted_done_ms` both disagree with wall-delta on Apple
> Silicon — the timestamp counters return non-monotonic values
> (see `src/renderer/draw.rs:527` comment) and the
> `on_submitted_work_done` callback fires at queue-submission
> acknowledgement, not GPU execution completion.

Result: baseline 1.4 ms, edit frames 40 ms. Reproduced.

### 2. Which phase spikes?

The CSV has per-phase fields. I extended the Python analyzer to
dump every field's baseline-median vs. edit-mean. One log line per
phase:

```python
for phase in phase_fields:
    bl_med = median(baseline frames)
    ed_mean = mean(edit frames)
    print(f"{phase:<25} {bl_med:>8.2f} {ed_mean:>8.2f} {ed_mean-bl_med:+10.2f}")
```

Result: only `gpu_pass_ms` showed a big delta (+38 ms). Everything
else — `pack_ms`, `tree_write_ms`, `update_ms`, `render_encode_ms`,
`render_wait_ms` — was flat.

### 3. Is the GPU actually doing the work?

`gpu_pass_ms` said 38 ms. `submitted_done_ms` (CPU-observed work
completion) said 2.6 ms. `wall_delta` said 40 ms. Three signals,
two answers.

Hypothesis A: GPU really is slow for 38 ms.
Hypothesis B: GPU is fast; the 38 ms is CPU work somewhere outside
the instrumented phases.

**Test:** disable the GPU upload on edit (`DEEPSPACE_SKIP_ALL_UPLOADS=1`
env var, one `if` around `renderer.update_tree`). If the spike
disappears, it's the GPU upload. If it stays, it's CPU work.

Spike unchanged. **Hypothesis B.**

### 4. Where in the frame loop is the dark time?

Added one log line to the harness loop (`DEEPSPACE_TRACE_ITERS=1`
env gate):

```rust
eprintln!(
    "iter_trace frame={} post_sample_gap_ms={:.2} update_ms={:.2} \
     upload_ms={:.2} render_total_ms={:.2} script_ms={:.2} \
     script_had_work={} wall_ms={:.2}",
    ...
);
```

Two new timings: `post_sample_gap_ms` (instant between previous
iter's sample and this iter's start) and `script_ms` (time spent
in script command dispatch at the end of iter).

Result on the edit frame:

```
frame=44 ... script_ms=37.77 script_had_work=true
frame=45 post_sample_gap_ms=37.80 ... script_ms=0.00
```

**The 37.8 ms lives in script-command dispatch.** Not in the
render. The `do_break` / `do_place` handler itself is slow.

### 5. Which step of `do_break`?

Three log lines in `do_break`:

```rust
eprintln!("do_break_phase raycast_ms={:.2} ...", t_raycast);
eprintln!("do_break_phase edit_ms={:.2} upload_ms={:.2} ...",
          t_edit, t_upload);
```

Result:

```
do_break_phase raycast_ms=0.00 ...
do_break_phase edit_ms=0.05 upload_ms=38.06 changed=true
```

**`self.upload_tree()` takes 38 ms.** Raycast and edit itself are
negligible.

### 6. Which step of `upload_tree`?

`upload_tree` just calls `upload_tree_lod`, but does three lines of
setup first:

```rust
pub fn upload_tree(&mut self) {
    self.tree_depth = self.world.tree_depth();  // ← prime suspect
    self.highlight_epoch = self.highlight_epoch.wrapping_add(1);
    self.cached_highlight = None;
    self.upload_tree_lod();
}
```

I timed every sub-phase of `upload_tree_lod` (pack, update_tree,
ribbon build, all the uniform writes) with one `Instant::now()`
pair each. Sum was < 1 ms.

The 38 ms was in the three lines BEFORE `upload_tree_lod`. Only
one of them looks non-trivial: `self.world.tree_depth()`.

### 7. What does `tree_depth()` do?

```rust
pub fn tree_depth(&self) -> u32 {
    let mut cache = std::collections::HashMap::new();
    self.depth_of(self.root, &mut cache)   // full DFS
}
```

Full recursive DFS over the library (94k reachable nodes) with a
per-call memo. 38 ms.

And `upload_tree` calls it on every edit.

### 8. Fix + verify

Cached `depth` on `Node` (3 lines in struct, 8 lines in
`insert_with_kind` to populate, 1-line rewrite of `tree_depth`).
Re-ran `replicate_edit_spike.sh`:

```
before: edit overhead mean=+38.10 ms
after:  edit overhead mean= +0.17 ms
```

Confirmed. All 112 lib tests pass. All three motion repros
(`replicate_movement_lag.sh`, `replicate_superchunk_lag.sh`,
`replicate_movement_lag_deep.sh`) still at 0 repacks on motion.

## Tools

- **`scripts/replicate_edit_spike.sh`** — self-sufficient repro.
  Exits non-zero if mean edit-frame overhead > `EDIT_SPIKE_BUDGET_MS`
  (default 5 ms). Regression guard from now on.
- **CSV trace** (`--perf-trace`) — already present in the harness.
  Per-frame per-phase timings written to disk; analyzed in Python.
- **`DEEPSPACE_SKIP_ALL_UPLOADS=1`** — env var I added as a one-off
  A/B test, then removed. The debug scaffolding was always
  scoped-and-pulled: added to bisect, deleted after the bisection
  landed.

## Lessons

1. **`gpu_pass_ms` lies on Metal.** Apple Silicon's timestamp
   counters return non-monotonic values; the existing code even
   had an `abs()` workaround (`draw.rs:527`). `submitted_done_ms`
   fires on queue-submission acknowledgement, not GPU completion.
   **Only `wall_ms` frame deltas are trustworthy** — they're CPU
   wall-clock, and don't depend on any GPU instrumentation.

2. **Instrument the gaps, not just the phases.** Phase timings
   summed to 1 ms but wall-delta was 40 ms. The missing 39 ms was
   OUTSIDE any instrumented phase (script dispatch in the harness
   loop, initialization in `upload_tree`). You have to time the
   gaps between phases too, or you never find it.

3. **Named debug flags with narrow scope.** Each diagnostic (env
   var, eprintln) was added to answer one yes/no question, then
   removed. Don't let debug code accumulate.

4. **A test script that exits non-zero beats logs.** Once
   `replicate_edit_spike.sh` existed, every experimental change
   had an unambiguous pass/fail signal. No user in the loop, no
   subjective "feels faster" judgement.

## Commit

`27bb457 perf: O(1) tree_depth via cached Node::depth; kills 38ms edit spike`
