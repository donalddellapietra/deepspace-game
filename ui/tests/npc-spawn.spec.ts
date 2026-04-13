import { test, expect } from "@playwright/test";

test("NPC spawn via N key", async ({ page }) => {
  test.setTimeout(120_000);

  const logs: string[] = [];
  const errors: string[] = [];
  page.on("console", (msg) => {
    const text = msg.text();
    logs.push(`[${msg.type()}] ${text.slice(0, 300)}`);
  });
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/");

  // Wait for game to start rendering
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );

  // Wait extra for blueprint to load
  await page.waitForTimeout(3000);

  const baseline = await page.evaluate(() => (window as any).__perfData);
  console.log(`Baseline: ${baseline.fps.toFixed(1)} FPS, ${baseline.entityCount} entities, ${baseline.npcCount} NPCs`);

  // Check for blueprint-related logs
  const bpLogs = logs.filter(l => l.includes("blueprint") || l.includes("Blueprint") || l.includes("humanoid") || l.includes("NPC"));
  console.log(`Blueprint logs:`);
  for (const l of bpLogs) console.log(`  ${l}`);

  // Click canvas to give it focus, then try pressing N
  const canvas = page.locator("canvas");
  await canvas.click();
  await page.waitForTimeout(500);

  // Try pressing n multiple times with delays
  for (let attempt = 0; attempt < 5; attempt++) {
    await page.keyboard.press("n");
    await page.waitForTimeout(1000);
    const perf = await page.evaluate(() => (window as any).__perfData);
    console.log(`After press ${attempt + 1}: ${perf.npcCount} NPCs, ${perf.entityCount} entities`);
    if (perf.npcCount > 0) break;
  }

  const afterSpawn = await page.evaluate(() => (window as any).__perfData);

  // Check for panics
  const hasPanic = errors.some(e => e.includes("panicked"));
  if (hasPanic) {
    console.log(`Panics:`);
    for (const e of errors.filter(e => e.includes("panicked"))) {
      console.log(`  ${e.slice(0, 300)}`);
    }
  }

  expect(hasPanic).toBe(false);
  expect(afterSpawn.npcCount).toBeGreaterThan(0);
});
