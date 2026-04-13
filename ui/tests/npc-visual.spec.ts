import { test } from "@playwright/test";

test("NPC lifecycle tracking", async ({ page }) => {
  test.setTimeout(90_000);

  const logs: string[] = [];
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("NPC") || text.includes("npc") || text.includes("Spawned") || text.includes("despawn")) {
      logs.push(text.slice(0, 300));
    }
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(2000);

  await page.locator("canvas").click({ position: { x: 640, y: 360 } });
  await page.waitForTimeout(300);
  await page.locator("canvas").click({ position: { x: 640, y: 360 } });
  await page.waitForTimeout(300);

  // Spawn
  await page.keyboard.press("n");

  // Track NPC count over time
  for (let i = 0; i < 10; i++) {
    await page.waitForTimeout(200);
    const perf = await page.evaluate(() => (window as any).__perfData);
    console.log(`t=${(i+1)*200}ms: ${perf.npcCount} NPCs, ${perf.entityCount} entities, ${perf.fps.toFixed(1)} FPS`);
    if (i === 0) {
      await page.screenshot({ path: "test-results/npc-instanced.png" });
    }
  }

  console.log(`\nNPC-related logs:`);
  for (const l of logs) console.log(`  ${l}`);
});
