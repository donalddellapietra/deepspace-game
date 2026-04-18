import { test } from "@playwright/test";

test("WASM → JS state sync — verify overlay updates", async ({ page }) => {
  test.setTimeout(30_000);

  const errors: string[] = [];
  const stateUpdates: any[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("[state]")) console.log(text.slice(0, 300));
  });

  await page.setViewportSize({ width: 1024, height: 768 });

  // Install a probe BEFORE WASM loads, so we count every __onGameState call.
  await page.addInitScript(() => {
    (window as any).__stateCalls = 0;
    (window as any).__stateLog = [];
    let realHandler: ((d: any) => void) | null = null;
    Object.defineProperty(window, "__onGameState", {
      configurable: true,
      get() {
        return (data: any) => {
          (window as any).__stateCalls++;
          const parsed = typeof data === "string" ? JSON.parse(data) : data;
          (window as any).__stateLog.push(parsed);
          if (realHandler) realHandler(parsed);
        };
      },
      set(handler) {
        realHandler = handler;
      },
    });
  });

  await page.goto("/");
  await page.waitForTimeout(5_000);

  const result = await page.evaluate(() => {
    const calls = (window as any).__stateCalls;
    const log: any[] = (window as any).__stateLog ?? [];
    const kinds = new Set<string>();
    for (const u of log) {
      // Tagged enum: { Hotbar: {...} } / { ModeIndicator: {...} } etc.
      if (u && typeof u === "object") {
        for (const k of Object.keys(u)) kinds.add(k);
      }
    }
    return { calls, kinds: [...kinds], sample: log.slice(0, 3) };
  });
  console.log("state push calls:", result.calls);
  console.log("state kinds seen:", JSON.stringify(result.kinds));
  console.log("first 3 updates:", JSON.stringify(result.sample, null, 2));

  const realErrors = errors.filter(e => !e.includes("Using exceptions for control flow"));
  console.log("Real errors:", realErrors.length);
  realErrors.slice(0, 3).forEach(e => console.log("  ", e.split("\n")[0]));
});
