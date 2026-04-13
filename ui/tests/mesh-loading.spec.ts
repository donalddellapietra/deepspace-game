import { test, expect } from "@playwright/test";

test("All visible chunks load at every zoom layer", async ({ page }) => {
  test.setTimeout(120_000);

  const errors: string[] = [];
  const logs: string[] = [];
  page.on("console", (msg) => {
    logs.push(`[${msg.type()}] ${msg.text().slice(0, 500)}`);
  });
  page.on("pageerror", (err) => {
    errors.push(err.message.slice(0, 500));
  });

  await page.goto("/");

  // Wait for game to start rendering.
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 60_000 },
  );
  await page.waitForTimeout(3000);

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(
    `Game running: ${perf.fps.toFixed(1)} FPS, ${perf.entityCount} entities, ` +
    `unbaked: ${perf.unbaked}, coldBakes: ${perf.coldBakes}`,
  );

  // Print errors.
  if (errors.length > 0) {
    console.log(`\nErrors:`);
    for (const e of errors) console.log(`  ${e}`);
  }

  // Print last logs with ENSURE_BAKED or MESH_CACHE.
  const meshLogs = logs.filter(l =>
    l.includes("ENSURE_BAKED") || l.includes("MESH_CACHE") || l.includes("SYNC_LOAD")
  );
  if (meshLogs.length > 0) {
    console.log(`\nMesh debug logs (${meshLogs.length}):`);
    for (const l of meshLogs.slice(0, 30)) console.log(`  ${l}`);
  }

  // No panics.
  const hasPanic = errors.some(
    (e) => e.includes("panicked") || e.includes("unreachable"),
  );
  expect(hasPanic).toBe(false);
  expect(perf.fps).toBeGreaterThan(0);
  expect(perf.entityCount).toBeGreaterThan(0);
  // In WASM with grassland, unbaked should be 0.
  expect(perf.unbaked).toBe(0);
});
