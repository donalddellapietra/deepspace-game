import { test, expect } from "@playwright/test";

test("NPC incremental scale", async ({ page }) => {
  test.setTimeout(120_000);

  const panics: string[] = [];
  page.on("pageerror", (err) => {
    // Filter out benign browser errors
    if (err.message.includes("pointer lock")) return;
    panics.push(err.message.slice(0, 200));
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(2000);

  const batches = [100000, 200000, 500000, 1000000];
  for (const count of batches) {
    await page.evaluate((n) => { (window as any).__spawnNpcs = n; }, count);
    await page.waitForTimeout(3000);

    const perf = await page.evaluate(() => (window as any).__perfData);
    console.log(`+${count}: ${perf.npcCount} total, ${perf.fps.toFixed(1)} FPS`);

    if (panics.length > 0) {
      console.log(`PANIC after +${count}: ${panics[panics.length - 1]}`);
      break;
    }
  }

  expect(panics.length).toBe(0);
});
