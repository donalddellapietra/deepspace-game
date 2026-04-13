import { test, expect } from "@playwright/test";

test("NPC scale with AI + animation", async ({ page }) => {
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

  const batches = [1000, 5000, 10000, 50000, 100000];
  for (const count of batches) {
    await page.evaluate((n) => { (window as any).__spawnNpcs = n; }, count);
    await page.waitForTimeout(3000);

    const perf = await page.evaluate(() => (window as any).__perfData);
    console.log(
      `+${count}: ${perf.npcCount} total, ${perf.fps.toFixed(1)} FPS, ` +
      `${perf.frameTimeMs.toFixed(1)} ms`
    );

    if (panics.length > 0) {
      console.log(`PANIC: ${panics[panics.length - 1]}`);
      break;
    }
    if (perf.fps < 5 && perf.npcCount > 0) {
      console.log("FPS too low, stopping");
      break;
    }
  }

  expect(panics.length).toBe(0);
});
