#!/usr/bin/env bash
set -euo pipefail

# Native overlay test: starts game + Vite, checks for crashes, captures
# window screenshot, runs Playwright checks. Requires screen recording permission.

SCREENSHOT="/tmp/deepspace-native.png"
GAME_LOG="/tmp/deepspace-game.log"

cleanup() {
    pkill -f "deepspace-game" 2>/dev/null || true
    pkill -f "cargo run" 2>/dev/null || true
    pkill -f "vite.*5173" 2>/dev/null || true
    pkill -f "node.*vite" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PASS=0; FAIL=0
check() {
    if [ "$2" = "0" ]; then
        echo "  ✓ $1"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $1: $3"
        FAIL=$((FAIL + 1))
    fi
}

echo "==> Starting Vite..."
(cd ui && npx vite --port 5173) &>/dev/null &
until curl -s http://localhost:5173 >/dev/null 2>&1; do sleep 0.3; done
echo "    Vite ready"

echo "==> Starting game..."
cargo run 2>&1 | tee "$GAME_LOG" &
GAME_PID=$!
sleep 8

# ── Check 1: Game process still alive (no crash/panic) ──
echo "==> Checking game health..."
if kill -0 $GAME_PID 2>/dev/null; then
    check "Game process alive" 0
else
    check "Game process alive" 1 "process exited"
fi

# Check for panics in log
if grep -qi "panic" "$GAME_LOG"; then
    PANIC_MSG=$(grep -i "panic" "$GAME_LOG" | head -1)
    check "No panics in log" 1 "$PANIC_MSG"
else
    check "No panics in log" 0
fi

# Check webview was created
if grep -q "WebView created" "$GAME_LOG"; then
    check "WebView created" 0
else
    check "WebView created" 1 "no creation log found"
fi

# Check hitTest swizzle succeeded
if grep -q "hitTest.*swizzled" "$GAME_LOG"; then
    check "hitTest: swizzled" 0
else
    check "hitTest: swizzled" 1 "swizzle log not found"
fi

# ── Check 2: Window screenshot ──
echo "==> Capturing window screenshot..."
WID=$(swift -e '
import CoreGraphics
let opts: CGWindowListOption = [.optionAll]
guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { exit(1) }
var bestWid = 0; var bestArea = 0
for w in list {
    let owner = w[kCGWindowOwnerName as String] as? String ?? ""
    let layer = w[kCGWindowLayer as String] as? Int ?? -1
    if owner == "deepspace-game" && layer == 0 {
        let b = w[kCGWindowBounds as String] as? [String: CGFloat] ?? [:]
        let area = Int((b["Width"] ?? 0) * (b["Height"] ?? 0))
        if area > bestArea { bestArea = area; bestWid = w[kCGWindowNumber as String] as? Int ?? 0 }
    }
}
if bestWid > 0 { print(bestWid) }
' 2>/dev/null)

if [ -n "$WID" ]; then
    screencapture -l "$WID" -x "$SCREENSHOT" 2>/dev/null
    check "Window screenshot captured" 0
    echo "    → $SCREENSHOT (WID=$WID)"
else
    check "Window screenshot captured" 1 "could not find game window"
fi

# ── Check 3: Playwright UI tests ──
echo "==> Running Playwright checks..."
node << 'JSEOF'
const { chromium } = require('/Users/donalddellapietra/GitHub/deepspace-game/ui/node_modules/playwright');
(async () => {
    const r = { pass: 0, fail: 0 };
    function check(name, ok, detail) {
        console.log(ok ? `  ✓ ${name}` : `  ✗ ${name}: ${detail}`);
        ok ? r.pass++ : r.fail++;
    }

    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(2000);

    check('No JS errors', errors.length === 0, errors.join('; '));

    const bodyBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    check('Body bg transparent', bodyBg === 'rgba(0, 0, 0, 0)' || bodyBg === 'transparent', bodyBg);

    const rootOk = await page.evaluate(() => {
        const r = document.getElementById('root');
        if (!r) return null;
        const s = getComputedStyle(r);
        return { pos: s.position, pe: s.pointerEvents, bg: s.backgroundColor, kids: r.children.length };
    });
    check('#root exists', !!rootOk, 'missing');
    if (rootOk) {
        check('#root fixed', rootOk.pos === 'fixed', rootOk.pos);
        check('#root pointer-events:none', rootOk.pe === 'none', rootOk.pe);
        check('#root transparent', rootOk.bg === 'rgba(0, 0, 0, 0)' || rootOk.bg === 'transparent', rootOk.bg);
        check('React rendered', rootOk.kids > 0, `${rootOk.kids} children`);
    }

    // Note: Playwright loads the page in a regular browser, NOT inside the
    // wry webview. So wry-specific features (IPC, __stateBuffer, game state)
    // can't be tested here. Only CSS/layout/React rendering is testable.

    // Check hotbar element exists and is positioned correctly
    const hotbar = await page.evaluate(() => {
        const el = document.querySelector('[class*="hotbar"], [class*="Hotbar"]');
        if (!el) return null;
        const rect = el.getBoundingClientRect();
        return {
            exists: true,
            bottom: rect.bottom,
            viewportHeight: window.innerHeight,
            hasPointerEvents: getComputedStyle(el).pointerEvents === 'auto',
        };
    });
    check('Hotbar element exists', !!hotbar, 'not found');
    if (hotbar) {
        check('Hotbar near bottom', hotbar.bottom > hotbar.viewportHeight * 0.7,
            `bottom=${hotbar.bottom} viewport=${hotbar.viewportHeight}`);
        check('Hotbar has pointer-events:auto', hotbar.hasPointerEvents, 'not auto');
    }

    await page.screenshot({ path: '/tmp/deepspace-playwright.png' });
    console.log(`\n  Playwright: ${r.pass} passed, ${r.fail} failed`);
    await browser.close();
    process.exit(r.fail > 0 ? 1 : 0);
})();
JSEOF
PW_EXIT=$?

echo ""
echo "==> Summary"
echo "    Shell checks: $PASS passed, $FAIL failed"
echo "    Screenshots: $SCREENSHOT, /tmp/deepspace-playwright.png"

if [ "$FAIL" -gt 0 ] || [ "$PW_EXIT" -ne 0 ]; then
    echo "    RESULT: FAIL"
    exit 1
else
    echo "    RESULT: PASS"
fi
