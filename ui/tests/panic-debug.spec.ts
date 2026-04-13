import { test } from "@playwright/test";

test("Find panic source", async ({ page }) => {
  test.setTimeout(120_000);

  const errors: string[] = [];
  page.on("pageerror", (err) => {
    if (err.message.includes("pointer lock")) return;
    errors.push(err.message.slice(0, 500));
  });
  page.on("console", (msg) => {
    if (msg.text().includes("panicked") || msg.text().includes("ERROR")) {
      errors.push(`[console] ${msg.text().slice(0, 500)}`);
    }
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(2000);

  // Spawn increasingly large batches
  const sizes = [50000, 100000];
  for (const size of sizes) {
    errors.length = 0;
    await page.evaluate((n) => { (window as any).__spawnNpcs = n; }, size);
    await page.waitForTimeout(3000);

    const perf = await page.evaluate(() => (window as any).__perfData);
    const hasUnreachable = errors.some(e => e.includes("unreachable"));
    console.log(`+${size} (${perf.npcCount} total): ${hasUnreachable ? "PANIC" : "OK"} - ${perf.fps.toFixed(1)} FPS`);
    if (hasUnreachable) {
      for (const e of errors.filter(e => e.includes("unreachable") || e.includes("panicked"))) {
        console.log(`  ${e.slice(0, 300)}`);
      }
      break;
    }
  }
});
