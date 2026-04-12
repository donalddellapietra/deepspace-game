#!/usr/bin/env bash
set -euo pipefail

# Native overlay test: starts game + Vite, captures window screenshot,
# runs Playwright checks, cleans up. Requires screen recording permission.

SCREENSHOT="/tmp/deepspace-native.png"

cleanup() {
    pkill -f "deepspace-game" 2>/dev/null || true
    pkill -f "cargo run" 2>/dev/null || true
    pkill -f "vite.*5173" 2>/dev/null || true
    pkill -f "node.*vite" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting Vite..."
(cd ui && npx vite --port 5173) &>/dev/null &
until curl -s http://localhost:5173 >/dev/null 2>&1; do sleep 0.3; done
echo "    Vite ready"

echo "==> Starting game..."
cargo run 2>&1 | grep -E "overlay|ERROR|panic" &
GAME_PID=$!
sleep 12
echo "    Game running (PID $GAME_PID)"

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
    echo "    Saved to $SCREENSHOT (WID=$WID)"
else
    echo "    FAIL: could not find game window"
fi

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

    const htmlBg = await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor);
    check('HTML bg transparent', htmlBg === 'rgba(0, 0, 0, 0)' || htmlBg === 'transparent', htmlBg);

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

    // WebSocket test
    const wsOk = await page.evaluate(() => new Promise(res => {
        try {
            const ws = new WebSocket('ws://localhost:9000');
            ws.onopen = () => { ws.close(); res(true); };
            ws.onerror = () => res(false);
            setTimeout(() => res(false), 3000);
        } catch { res(false); }
    }));
    check('WebSocket connects', wsOk, 'connection failed');

    const wsData = await page.evaluate(() => new Promise(res => {
        try {
            const ws = new WebSocket('ws://localhost:9000');
            ws.onmessage = e => { ws.close(); res(e.data); };
            ws.onerror = () => res(null);
            setTimeout(() => { ws.close(); res(null); }, 5000);
        } catch { res(null); }
    }));
    if (wsData) {
        try { const p = JSON.parse(wsData); check('WS valid JSON', true, ''); check('WS has type', 'type' in p, Object.keys(p)); }
        catch { check('WS valid JSON', false, wsData.substring(0, 80)); }
    } else {
        check('WS receives data', false, 'no message in 5s');
    }

    await page.screenshot({ path: '/tmp/deepspace-playwright.png' });
    console.log(`\n==> ${r.pass} passed, ${r.fail} failed`);
    await browser.close();
    process.exit(r.fail > 0 ? 1 : 0);
})();
JSEOF

echo "==> Done. Screenshots:"
echo "    Native window: $SCREENSHOT"
echo "    Playwright:    /tmp/deepspace-playwright.png"
