#!/usr/bin/env bash
set -euo pipefail

# Automated test for the native wry overlay.
# Tests: Vite serves correctly, WebSocket works, React UI renders,
# transparency/layout is correct.

VITE_PORT=5173
WS_PORT=9000
SCREENSHOT="/tmp/deepspace-overlay-test.png"

cleanup() {
    kill $VITE_PID 2>/dev/null || true
    kill $GAME_PID 2>/dev/null || true
    wait $VITE_PID 2>/dev/null || true
    wait $GAME_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Starting Vite..."
(cd ui && npx vite --port $VITE_PORT) &
VITE_PID=$!
until curl -s http://localhost:$VITE_PORT > /dev/null 2>&1; do sleep 0.3; done
echo "    Vite ready on :$VITE_PORT"

echo "==> Building + starting game..."
cargo build 2>&1 | grep -E "Compiling deepspace|Finished|error" || true
cargo run &
GAME_PID=$!

# Wait for WebSocket server
echo "==> Waiting for WebSocket server on :$WS_PORT..."
for i in $(seq 1 30); do
    if nc -z 127.0.0.1 $WS_PORT 2>/dev/null; then
        echo "    WebSocket ready"
        break
    fi
    sleep 1
done

# Wait for game window to initialize
sleep 5

echo "==> Running Playwright tests..."
node << 'JSEOF'
const { chromium } = require('/Users/donalddellapietra/GitHub/deepspace-game/ui/node_modules/playwright');

(async () => {
    const results = { pass: 0, fail: 0, issues: [] };

    function check(name, condition, detail) {
        if (condition) {
            console.log(`  ✓ ${name}`);
            results.pass++;
        } else {
            console.log(`  ✗ ${name}: ${detail}`);
            results.fail++;
            results.issues.push({ name, detail });
        }
    }

    // Test 1: Load the React UI (same URL the webview loads)
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

    const errors = [];
    page.on('pageerror', e => errors.push(e.message));

    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(2000);

    check('Page loads without errors', errors.length === 0,
        errors.join('; '));

    // Test 2: Check CSS transparency
    const bodyBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    check('Body background is transparent',
        bodyBg === 'rgba(0, 0, 0, 0)' || bodyBg === 'transparent',
        `got "${bodyBg}"`);

    const htmlBg = await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor);
    check('HTML background is transparent',
        htmlBg === 'rgba(0, 0, 0, 0)' || htmlBg === 'transparent',
        `got "${htmlBg}"`);

    // Test 3: Check #root element
    const rootStyle = await page.evaluate(() => {
        const root = document.getElementById('root');
        if (!root) return null;
        const s = getComputedStyle(root);
        return {
            position: s.position,
            width: s.width,
            height: s.height,
            pointerEvents: s.pointerEvents,
            bg: s.backgroundColor,
        };
    });
    check('#root exists', rootStyle !== null, 'missing');
    if (rootStyle) {
        check('#root is fixed positioned', rootStyle.position === 'fixed',
            `got "${rootStyle.position}"`);
        check('#root has pointer-events: none', rootStyle.pointerEvents === 'none',
            `got "${rootStyle.pointerEvents}"`);
        check('#root background is transparent',
            rootStyle.bg === 'rgba(0, 0, 0, 0)' || rootStyle.bg === 'transparent',
            `got "${rootStyle.bg}"`);
    }

    // Test 4: Check React components rendered
    const childCount = await page.evaluate(() => {
        const root = document.getElementById('root');
        return root ? root.children.length : 0;
    });
    check('React components rendered', childCount > 0,
        `root has ${childCount} children`);

    // Test 5: Check hotbar is visible and positioned correctly
    const hotbar = await page.evaluate(() => {
        // Look for the hotbar container
        const els = document.querySelectorAll('[class*="hotbar"], [class*="Hotbar"]');
        if (els.length === 0) return null;
        const el = els[0];
        const rect = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return {
            bottom: rect.bottom,
            left: rect.left,
            width: rect.width,
            height: rect.height,
            viewportHeight: window.innerHeight,
            viewportWidth: window.innerWidth,
            pointerEvents: style.pointerEvents,
        };
    });
    if (hotbar) {
        check('Hotbar is near bottom of screen',
            hotbar.bottom > hotbar.viewportHeight * 0.7,
            `bottom=${hotbar.bottom} viewport=${hotbar.viewportHeight}`);
        check('Hotbar is horizontally centered',
            Math.abs((hotbar.left + hotbar.width / 2) - hotbar.viewportWidth / 2) < 100,
            `center=${hotbar.left + hotbar.width / 2} viewport=${hotbar.viewportWidth}`);
        check('Hotbar has pointer-events: auto',
            hotbar.pointerEvents === 'auto',
            `got "${hotbar.pointerEvents}"`);
    } else {
        check('Hotbar found', false, 'no hotbar element found');
    }

    // Test 6: WebSocket connectivity
    const wsConnected = await page.evaluate(() => {
        return new Promise((resolve) => {
            try {
                const ws = new WebSocket('ws://localhost:9000');
                ws.onopen = () => { ws.close(); resolve(true); };
                ws.onerror = () => resolve(false);
                setTimeout(() => resolve(false), 3000);
            } catch { resolve(false); }
        });
    });
    check('WebSocket connects to game', wsConnected, 'connection failed');

    // Test 7: WebSocket receives game state
    const wsState = await page.evaluate(() => {
        return new Promise((resolve) => {
            try {
                const ws = new WebSocket('ws://localhost:9000');
                ws.onmessage = (e) => { ws.close(); resolve(e.data); };
                ws.onerror = () => resolve(null);
                setTimeout(() => { ws.close(); resolve(null); }, 5000);
            } catch { resolve(null); }
        });
    });
    if (wsState) {
        try {
            const parsed = JSON.parse(wsState);
            check('WebSocket receives valid JSON state', true, '');
            check('State has expected type field', 'type' in parsed,
                `keys: ${Object.keys(parsed).join(',')}`);
        } catch {
            check('WebSocket state is valid JSON', false, `got: ${wsState.substring(0, 100)}`);
        }
    } else {
        check('WebSocket receives state', false, 'no message received in 5s');
    }

    // Take screenshot
    await page.screenshot({ path: '/tmp/deepspace-overlay-test.png', fullPage: true });

    // Summary
    console.log(`\n==> Results: ${results.pass} passed, ${results.fail} failed`);
    if (results.issues.length > 0) {
        console.log('Issues:');
        for (const i of results.issues) {
            console.log(`  - ${i.name}: ${i.detail}`);
        }
    }

    await browser.close();
    process.exit(results.fail > 0 ? 1 : 0);
})();
JSEOF

TEST_EXIT=$?
echo "==> Playwright exit code: $TEST_EXIT"

if [ -f "$SCREENSHOT" ]; then
    echo "==> Screenshot saved to $SCREENSHOT"
fi

exit $TEST_EXIT
