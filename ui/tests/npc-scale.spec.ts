import { test } from "@playwright/test";

test("NPC scale test via JS bridge", async ({ page }) => {
  test.setTimeout(120_000);

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(2000);

  // Spawn via JS bridge (bypasses keyboard focus issues)
  const counts = [10_000, 50_000, 100_000];
  for (const target of counts) {
    const current = await page.evaluate(() => (window as any).__perfData.npcCount);
    const toSpawn = target - current;
    if (toSpawn <= 0) continue;

    await page.evaluate((n) => { (window as any).__spawnNpcs = n; }, toSpawn);
    await page.waitForTimeout(5000);

    const perf = await page.evaluate(() => (window as any).__perfData);
    console.log(
      `${perf.npcCount} NPCs: ${perf.fps.toFixed(1)} FPS, ` +
      `${perf.frameTimeMs.toFixed(1)} ms, ${perf.entityCount} entities`
    );

    if (perf.fps < 2 && perf.npcCount > 0) {
      console.log("FPS too low, stopping");
      break;
    }
  }
});
