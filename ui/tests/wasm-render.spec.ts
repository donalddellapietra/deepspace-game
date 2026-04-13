import { test, expect } from "@playwright/test";

test("WASM terrain renders", async ({ page }) => {
  test.setTimeout(60_000);

  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.text().includes("ERROR")) errors.push(msg.text().slice(0, 300));
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(3000);

  await page.screenshot({ path: "test-results/wasm-terrain.png" });

  console.log(`Errors: ${errors.length}`);
  for (const e of errors) console.log(`  ${e}`);

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`Perf: ${perf.fps.toFixed(1)} FPS, ${perf.entityCount} entities`);

  // Game is running and rendering terrain
  expect(perf.entityCount).toBeGreaterThan(10);

  // Visual regression check: screenshot saved to test-results/
  // for manual review. Playwright's toHaveScreenshot() can be
  // used for automated comparison once a baseline is established.
});
