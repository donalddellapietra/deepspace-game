import { test } from "@playwright/test";

test("Debug spawn", async ({ page }) => {
  test.setTimeout(60_000);

  const errors: string[] = [];
  page.on("pageerror", (err) => {
    errors.push(err.message.slice(0, 400));
  });
  page.on("console", (msg) => {
    if (msg.text().includes("panicked") || msg.text().includes("ERROR") || msg.text().includes("Spawned") || msg.text().includes("bridge") || msg.text().includes("wgpu") || msg.text().includes("shader") || msg.text().includes("pipeline")) {
      errors.push(`[${msg.type()}] ${msg.text().slice(0, 500)}`);
    }
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(2000);

  // Try JS bridge spawn
  await page.evaluate(() => { (window as any).__spawnNpcs = 10000; });
  await page.waitForTimeout(2000);

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`After spawn: ${perf.npcCount} NPCs, ${perf.fps.toFixed(1)} FPS`);

  await page.screenshot({ path: "test-results/spawn-debug.png" });

  console.log(`Errors (${errors.length}):`);
  for (const e of errors) console.log(`  ${e}`);
});
