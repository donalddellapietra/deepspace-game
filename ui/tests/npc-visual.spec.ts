import { test, expect } from "@playwright/test";

test("NPCs on ground with heightmap physics", async ({ page }) => {
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
  await page.waitForTimeout(3000);

  // Spawn 1K NPCs — small enough to verify visually
  await page.evaluate(() => { (window as any).__spawnNpcs = 1000; });
  await page.waitForTimeout(3000);

  await page.screenshot({ path: "test-results/npcs-on-ground.png" });

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`${perf.npcCount} NPCs, ${perf.fps.toFixed(1)} FPS`);

  expect(panics.length).toBe(0);
  expect(perf.npcCount).toBeGreaterThan(0);
});

test("100K NPCs with physics", async ({ page }) => {
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
  await page.waitForTimeout(3000);

  await page.evaluate(() => { (window as any).__spawnNpcs = 100000; });
  await page.waitForTimeout(5000);

  await page.screenshot({ path: "test-results/100k-on-ground.png" });

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`${perf.npcCount} NPCs, ${perf.fps.toFixed(1)} FPS, ${perf.frameTimeMs.toFixed(1)} ms`);

  expect(panics.length).toBe(0);
  expect(perf.npcCount).toBeGreaterThanOrEqual(50000);
  expect(perf.fps).toBeGreaterThan(5);
});
