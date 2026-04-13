import { test, expect } from "@playwright/test";

test("WASM game loads and renders", async ({ page }) => {
  test.setTimeout(60_000);

  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`Game running: ${perf.fps.toFixed(1)} FPS, ${perf.entityCount} entities`);

  const hasPanic = errors.some(e => e.includes("panicked") || e.includes("unreachable"));
  expect(hasPanic).toBe(false);
  expect(perf.fps).toBeGreaterThan(0);
});
