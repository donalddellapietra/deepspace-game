import { test } from "@playwright/test";

test("Debug WASM startup", async ({ page }) => {
  test.setTimeout(60_000);

  const errors: string[] = [];
  const logs: string[] = [];
  page.on("console", (msg) => {
    logs.push(`[${msg.type()}] ${msg.text().slice(0, 300)}`);
  });
  page.on("pageerror", (err) => {
    errors.push(err.message.slice(0, 300));
  });

  await page.goto("/");
  await page.waitForTimeout(15000);

  console.log(`Errors (${errors.length}):`);
  for (const e of errors) console.log(`  ${e}`);

  console.log(`\nLogs (last 20 of ${logs.length}):`);
  for (const l of logs.slice(-20)) console.log(`  ${l}`);

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`\nPerf: ${JSON.stringify(perf)}`);
});
