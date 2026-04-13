import { test, expect } from "@playwright/test";

interface PerfData {
  fps: number;
  frameTimeMs: number;
  entityCount: number;
  npcCount: number;
}

test("NPC mass spawn performance", async ({ page }) => {
  test.setTimeout(300_000);

  const panics: string[] = [];
  page.on("pageerror", (err) => {
    if (err.message.includes("panicked")) panics.push(err.message);
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );

  const baseline = await page.evaluate<PerfData>(() => (window as any).__perfData);
  console.log(`Baseline: ${baseline.fps.toFixed(1)} FPS, ${baseline.entityCount} entities`);

  // Focus the canvas directly via JS (React overlay intercepts click events)
  await page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (canvas) canvas.focus();
  });
  await page.waitForTimeout(500);

  // Press M to mass-spawn NPCs (1000 at a time)
  for (let batch = 1; batch <= 10; batch++) {
    await page.keyboard.press("m");
    // Wait for entities to be created and FPS to settle
    await page.waitForTimeout(5000);

    const perf = await page.evaluate<PerfData>(() => (window as any).__perfData);
    console.log(
      `Batch ${batch}: ${perf.npcCount} NPCs, ${perf.entityCount} entities, ` +
      `${perf.fps.toFixed(1)} FPS, ${perf.frameTimeMs.toFixed(1)} ms/frame`
    );

    if (panics.length > 0) {
      console.log(`PANIC after batch ${batch}: ${panics[0].slice(0, 200)}`);
      break;
    }

    // If FPS drops too low, stop — we have our data
    if (perf.fps < 0.5 && perf.npcCount > 0) {
      console.log(`FPS too low, stopping at ${perf.npcCount} NPCs`);
      break;
    }
  }

  const final_ = await page.evaluate<PerfData>(() => (window as any).__perfData);
  console.log(`\n=== FINAL ===`);
  console.log(`NPCs: ${final_.npcCount}`);
  console.log(`Entities: ${final_.entityCount}`);
  console.log(`FPS: ${final_.fps.toFixed(1)}`);
  console.log(`Frame time: ${final_.frameTimeMs.toFixed(1)} ms`);

  expect(panics.length).toBe(0);
  expect(final_.npcCount).toBeGreaterThan(0);
});
