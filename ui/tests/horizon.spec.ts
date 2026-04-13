import { test, expect } from "@playwright/test";

test("horizon elevated view", async ({ page }) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 2560, height: 1440 });

  const logs: string[] = [];
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("CLIP")) logs.push(text);
  });

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(8000);

  await page.screenshot({ path: "test-results/elevated-full.png" });

  console.log(`CLIP logs: ${logs.length}`);
  for (const l of logs) console.log(`  ${l}`);
});
