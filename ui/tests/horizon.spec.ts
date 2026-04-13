import { test, expect } from "@playwright/test";

/**
 * A/B horizon comparison. Takes screenshots at multiple time points
 * to capture the horizon after the scene has fully settled.
 */
test("horizon screenshots", async ({ page }) => {
  test.setTimeout(120_000);

  const errors: string[] = [];
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("panic")) errors.push(text.slice(0, 300));
  });
  page.on("pageerror", (err) => errors.push(err.message.slice(0, 300)));

  await page.goto("/");

  // Wait for game to start
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 60_000 },
  );

  // Take screenshots at increasing intervals to capture settled state
  for (const [label, waitMs] of [
    ["t1_2s", 2000],
    ["t2_5s", 3000],
    ["t3_10s", 5000],
    ["t4_15s", 5000],
  ] as const) {
    await page.waitForTimeout(waitMs);
    const perf = await page.evaluate(() => (window as any).__perfData);
    console.log(`${label}: ${perf.fps.toFixed(1)} FPS, ${perf.entityCount} entities`);
    await page.screenshot({
      path: `test-results/horizon-${label}.png`,
      fullPage: false,
    });
  }

  const panics = errors.filter((e) => e.includes("panic"));
  expect(panics).toHaveLength(0);
});
