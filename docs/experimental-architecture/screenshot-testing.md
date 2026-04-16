# Screenshot-based testing

When iterating on rendering changes you want to verify visually
without alt-tabbing to the game window. macOS lets `screencapture`
target a specific window by ID, even if it's hidden behind other
windows.

## Prerequisites (one-time)

```bash
pip3 install --user pyobjc-framework-Quartz
```

PyObjC's Quartz binding is what gives Python access to
`CGWindowListCopyWindowInfo`, which is how we ask the OS for a
window's numeric ID.

## Capture command

With the game running (via `scripts/dev.sh`), run:

```bash
PID=$(pgrep -f "target/debug/deepspace-game$" | head -1) && \
WID=$(python3 -c "
import Quartz
ws = Quartz.CGWindowListCopyWindowInfo(
    Quartz.kCGWindowListOptionAll | Quartz.kCGWindowListExcludeDesktopElements,
    Quartz.kCGNullWindowID)
for w in ws:
    if w.get('kCGWindowOwnerPID') == $PID and w.get('kCGWindowName'):
        print(w['kCGWindowNumber'])
        break
") && \
screencapture -x -l "$WID" /tmp/ds.png && open /tmp/ds.png
```

Breakdown:
- `pgrep` resolves the game process's PID (filtered to the dev
  binary so other Cargo processes don't match).
- The Python snippet asks Quartz for every on-screen window owned by
  that PID and prints the first one with a non-empty title (skipping
  the hidden helper windows wgpu/winit creates).
- `screencapture -x -l <wid>` captures that window into the named
  file. `-x` suppresses the camera-shutter sound; `-l` specifies a
  window ID.
- `open` displays the PNG in Preview so you can eyeball it without
  having to bring the game window forward yourself.

The capture works even when the game window is occluded — it reads
the window's own backing store, not the screen. That's why this is
better than `screencapture -R x,y,w,h` (which captures whatever is
actually drawn at those screen coords, including overlapping
windows).


Only run pkill on your specific process.