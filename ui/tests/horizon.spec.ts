import { test, expect } from "@playwright/test";

test("horizon debug", async ({ page }) => {
  test.setTimeout(60_000);

  const logs: string[] = [];
  page.on("console", (msg) => logs.push(msg.text().slice(0, 300)));

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(5000);

  // Print all logs that mention "Imposter" or "imposter"
  const imposterLogs = logs.filter((l) => l.toLowerCase().includes("imposter"));
  console.log(`Imposter logs (${imposterLogs.length}):`);
  for (const l of imposterLogs) console.log(`  ${l}`);

  // Also print any error logs
  const errorLogs = logs.filter((l) => l.includes("ERROR") || l.includes("panic"));
  if (errorLogs.length > 0) {
    console.log(`Error logs (${errorLogs.length}):`);
    for (const l of errorLogs) console.log(`  ${l}`);
  }

  await page.screenshot({ path: "test-results/horizon-debug.png" });

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`Perf: ${perf.fps.toFixed(1)} FPS, ${perf.entityCount} entities`);
});
