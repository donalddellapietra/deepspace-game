import { test, expect } from "@playwright/test";

test("100K NPCs visual verification", async ({ page }) => {
  test.setTimeout(120_000);

  const panics: string[] = [];
  page.on("pageerror", (err) => {
    if (err.message.includes("pointer lock")) return;
    panics.push(err.message.slice(0, 200));
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(2000);

  // Spawn 100K NPCs
  await page.evaluate(() => { (window as any).__spawnNpcs = 100000; });
  await page.waitForTimeout(5000);

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`${perf.npcCount} NPCs, ${perf.fps.toFixed(1)} FPS, ${perf.frameTimeMs.toFixed(1)} ms`);

  // Take screenshot
  await page.screenshot({ path: "test-results/100k-npcs.png" });

  // Verify
  expect(panics.length).toBe(0);
  expect(perf.npcCount).toBeGreaterThanOrEqual(50000);
  expect(perf.fps).toBeGreaterThan(10);
});
