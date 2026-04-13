import { test } from "@playwright/test";

test("NPC visual check after spawn", async ({ page }) => {
  test.setTimeout(90_000);

  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => {
    if (msg.text().includes("ERROR") || msg.text().includes("panicked")) {
      errors.push(msg.text().slice(0, 400));
    }
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(3000);

  // Click and spawn
  await page.locator("canvas").click();
  await page.waitForTimeout(500);
  await page.keyboard.press("n");
  await page.waitForTimeout(3000);

  await page.screenshot({ path: "test-results/npc-instanced.png" });

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`Perf: ${perf.fps.toFixed(1)} FPS, ${perf.entityCount} entities, ${perf.npcCount} NPCs`);
  console.log(`Errors: ${errors.length}`);
  for (const e of errors.slice(0, 5)) console.log(`  ${e}`);
});
