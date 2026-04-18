import { test } from "@playwright/test";

test("WASM screenshot — verify canvas renders pixels", async ({ page }) => {
  test.setTimeout(30_000);

  const errors: string[] = [];
  const logs: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));

  await page.setViewportSize({ width: 1024, height: 768 });
  await page.goto("/");
  // Wait for canvas resize + a few frames to render
  await page.waitForTimeout(5_000);

  const canvasInfo = await page.evaluate(() => {
    const all = Array.from(document.querySelectorAll("canvas")) as HTMLCanvasElement[];
    return all.map((c, i) => ({
      idx: i,
      width: c.width,
      height: c.height,
      cssWidth: c.clientWidth,
      cssHeight: c.clientHeight,
      style: c.getAttribute("style") ?? "",
      parent: c.parentElement?.tagName ?? "none",
    }));
  });
  console.log("Canvases:", JSON.stringify(canvasInfo, null, 2));

  // Filter out the expected winit "Using exceptions for control flow" pseudo-error
  const realErrors = errors.filter(e => !e.includes("Using exceptions for control flow"));
  console.log("Real errors:", realErrors.length);
  realErrors.forEach((e) => console.log("  ", e.split("\n")[0]));

  console.log("=== ALL CONSOLE ===");
  logs.forEach(l => console.log("  ", l.slice(0, 250)));

  await page.screenshot({ path: "../tmp/wasm-render.png", fullPage: false });
  console.log("Screenshot saved to tmp/wasm-render.png");
});
