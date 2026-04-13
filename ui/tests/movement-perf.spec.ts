import { test, expect } from "@playwright/test";

test("Player movement FPS (no NPCs)", async ({ page }) => {
  test.setTimeout(60_000);

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

  // Baseline FPS with no movement
  const baseline = await page.evaluate(() => (window as any).__perfData);
  console.log(`Baseline (still): ${baseline.fps.toFixed(1)} FPS`);

  // Simulate movement by holding W key
  await page.evaluate(() => document.querySelector("canvas")?.focus());
  await page.waitForTimeout(200);

  // Hold W for 3 seconds
  await page.keyboard.down("w");
  await page.waitForTimeout(3000);
  await page.keyboard.up("w");

  const moving = await page.evaluate(() => (window as any).__perfData);
  console.log(`During movement: ${moving.fps.toFixed(1)} FPS`);

  // After stopping
  await page.waitForTimeout(1000);
  const after = await page.evaluate(() => (window as any).__perfData);
  console.log(`After stopping: ${after.fps.toFixed(1)} FPS`);

  // Movement should not cause massive FPS drop
  // (before fix: heightmap regen every frame = ~5 FPS during movement)
  expect(panics.length).toBe(0);
  console.log(`FPS ratio (moving/baseline): ${(moving.fps / baseline.fps * 100).toFixed(0)}%`);
});

test("Movement FPS with 10K NPCs", async ({ page }) => {
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

  // Spawn 10K NPCs
  await page.evaluate(() => { (window as any).__spawnNpcs = 10000; });
  await page.waitForTimeout(3000);

  const withNpcs = await page.evaluate(() => (window as any).__perfData);
  console.log(`10K NPCs (still): ${withNpcs.fps.toFixed(1)} FPS`);

  // Move with NPCs
  await page.evaluate(() => document.querySelector("canvas")?.focus());
  await page.waitForTimeout(200);
  await page.keyboard.down("w");
  await page.waitForTimeout(3000);
  await page.keyboard.up("w");

  const movingWithNpcs = await page.evaluate(() => (window as any).__perfData);
  console.log(`10K NPCs (moving): ${movingWithNpcs.fps.toFixed(1)} FPS`);

  expect(panics.length).toBe(0);
  console.log(`FPS ratio: ${(movingWithNpcs.fps / withNpcs.fps * 100).toFixed(0)}%`);
});
