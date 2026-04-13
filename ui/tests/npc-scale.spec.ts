import { test } from "@playwright/test";

test("NPC scale test", async ({ page }) => {
  test.setTimeout(120_000);

  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message.slice(0, 300)));

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(2000);

  // Spawn 1000 via JS bridge
  await page.evaluate(() => { (window as any).__spawnNpcs = 1000; });
  await page.waitForTimeout(5000);

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`${perf.npcCount} NPCs, ${perf.fps.toFixed(1)} FPS`);

  // Spawn more
  await page.evaluate(() => { (window as any).__spawnNpcs = 9000; });
  await page.waitForTimeout(5000);

  const perf2 = await page.evaluate(() => (window as any).__perfData);
  console.log(`${perf2.npcCount} NPCs, ${perf2.fps.toFixed(1)} FPS`);

  console.log(`\nPage errors (${errors.length}):`);
  for (const e of errors) console.log(`  ${e}`);
});
