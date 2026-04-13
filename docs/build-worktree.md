# Building & Testing in Worktrees

## Native build

**Do NOT use `--features dev`** for native builds. The `dev` feature
enables `bevy/dynamic_linking` which conflicts with `wry`'s `objc-sys`
on macOS (undefined symbol `_rust_objc_sys_0_3_try_catch_exception`).
`scripts/dev.sh` handles this — it runs plain `cargo run`.

```bash
# Correct:
cargo run
cargo build

# WRONG — will fail with linker error:
cargo build --features dev
cargo run --features dev
```

### Shared target directory

The worktree's `target/` is symlinked to the main repo's target. This
mostly works, but has two gotchas:

1. **Stale fingerprints**: After editing a file in the worktree, cargo
   may consider it "unchanged" because the shared target's fingerprint
   cache was built against the main repo's mtimes. **Always `touch`
   modified files before building**:

   ```bash
   touch src/world/render.rs src/camera.rs
   cargo build --target wasm32-unknown-unknown
   ```

   Verify it actually recompiled: a real rebuild takes 5-15s. If it says
   "Finished in 0.2s", the touch didn't work or you touched the wrong file.

2. **Cross-branch incremental artifacts**: If native builds fail with
   linker errors after switching between branches, clear only the game
   crate's artifacts (NOT `cargo clean` — that's a 20-minute rebuild):

   ```bash
   find target/debug/incremental -name "deepspace_game-*" -type d -exec rm -rf {} +
   find target/debug/deps -name "deepspace_game-*" -delete
   find target/debug/deps -name "libdeepspace_game-*" -delete
   ```

## WASM build

Full rebuild cycle:

```bash
# 1. Touch modified files (required due to shared target fingerprints)
touch src/the_file_you_changed.rs

# 2. Compile to WASM
cargo build --target wasm32-unknown-unknown

# 3. Bundle (wasm-bindgen + copy assets to dist/)
#    If trunk fails with "No such file or directory", run: mkdir -p dist
trunk build

# 4. Serve (separate from build — trunk serve also rebuilds, but
#    the explicit build step above ensures the right code is compiled)
trunk serve --port 8082 --no-autoreload
```

`trunk build` sometimes fails with "error writing JS loader file" —
this is a race condition when dist/ was deleted. Fix: `mkdir -p dist`
then retry.

`trunk serve` dies when the sandbox kills the process. Use `nohup`:

```bash
nohup trunk serve --port 8082 --no-autoreload > /tmp/trunk-horizon.log 2>&1 &
```

Check it's alive: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8082/`
Takes ~5-8 seconds after launch to start serving.

## Playwright testing

### Setup

Tests are in `ui/tests/`. Playwright config uses real Chrome (not
bundled Chromium) with WebGPU flags:

```typescript
// playwright.config.ts
channel: "chrome",
launchOptions: {
  args: ["--enable-gpu", "--enable-unsafe-webgpu",
         "--enable-features=Vulkan", "--use-angle=metal"],
}
```

### Running tests

```bash
cd ui && TRUNK_PORT=8082 npx playwright test -g "horizon"
```

### Screenshot analysis

Screenshots land in `ui/test-results/`.

**WebGPU canvas returns all-black from JS `getImageData()`**. Do NOT
try to read pixels in the browser. Use Playwright's `page.screenshot()`
(which composites correctly via Chrome DevTools Protocol) and analyze
the PNG files with ImageMagick.

#### Vertical color profile (find where terrain/sky boundary is)

```bash
magick horizon-full.png -crop 1x1440+1280+0 -depth 8 txt:- | \
  awk -F'[(),: ]+' '/^[0-9]/{y=$2; for(i=1;i<=NF;i++){if($i~/#/){hex=$i}} \
  if(y%10==0 && y>=700 && y<=870) print "y="y" "hex}'
```

#### Horizontal color profile (check for block-level jaggedness)

```bash
magick horizon-full.png -crop 2560x1+0+820 -depth 8 txt:- | \
  awk -F'[(),: ]+' '/^[0-9]/{x=$1; for(i=1;i<=NF;i++){if($i~/#/){hex=$i}} \
  if(x%100==0) print "x="x" "hex}'
```

#### Compare left/center/right profiles

```bash
for x in 50 1280 2500; do
  echo "=== x=$x ==="
  magick horizon-full.png -crop 1x1440+${x}+0 -depth 8 txt:- | \
    awk -F'[(),: ]+' '/^[0-9]/{y=$2; for(i=1;i<=NF;i++){if($i~/#/){hex=$i}} \
    if(y%10==0 && y>=700 && y<=870) print "y="y" "hex}'
done
```

### Simulating elevated camera

Headless Chrome can't lock the cursor, so keyboard/mouse input to the
game doesn't work. To test elevated views, temporarily modify:

- `src/player.rs` `spawn_position()`: change `voxel: [mid, 2, mid]` to
  `voxel: [mid, 20, mid]` (20 blocks up)
- `src/camera.rs` `spawn_camera()`: change `pitch: 0.0` to
  `pitch: -0.15` (look slightly down)

Remember to revert these before committing.
