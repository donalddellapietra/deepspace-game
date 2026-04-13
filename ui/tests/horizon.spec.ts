import { test, expect } from "@playwright/test";

test("horizon systematic debug", async ({ page }) => {
  test.setTimeout(90_000);

  const logs: string[] = [];
  page.on("console", (msg) => logs.push(msg.text().slice(0, 300)));

  await page.goto("/");
  await page.waitForFunction(
    () => (window as any).__perfData?.fps > 0,
    { timeout: 45_000 },
  );

  // Wait for scene to settle
  await page.waitForTimeout(3000);

  // Ground level screenshot
  await page.screenshot({ path: "test-results/horizon-A-ground.png" });

  // Try to elevate: click to lock, then press space repeatedly
  const canvas = page.locator("canvas");
  const box = await canvas.boundingBox();
  if (box) {
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(300);

    // Press space to jump (may not work without pointer lock)
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press("Space");
      await page.waitForTimeout(50);
    }
    await page.waitForTimeout(1000);
  }
  await page.screenshot({ path: "test-results/horizon-B-elevated.png" });

  // Try towering: place blocks below
  // Press number keys to select blocks, then click to place
  await page.keyboard.press("Digit1");
  await page.waitForTimeout(100);
  for (let i = 0; i < 10; i++) {
    // Right-click to place block
    if (box) {
      await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2 + 50, { button: "right" });
    }
    await page.waitForTimeout(200);
  }
  await page.waitForTimeout(1000);
  await page.screenshot({ path: "test-results/horizon-C-towered.png" });

  const perf = await page.evaluate(() => (window as any).__perfData);
  console.log(`Perf: ${perf.fps.toFixed(1)} FPS, ${perf.entityCount} entities`);

  // Extract horizon region pixel analysis
  // Crop a 1px tall strip across the middle of the screen
  const stripData = await page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return null;
    try {
      const w = canvas.width;
      const h = canvas.height;
      const offscreen = document.createElement("canvas");
      offscreen.width = w;
      offscreen.height = h;
      const ctx = offscreen.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(canvas, 0, 0);

      // Sample a vertical strip at center X, from top to bottom
      const samples: number[][] = [];
      for (let y = 0; y < h; y += 5) {
        const pixel = ctx.getImageData(w / 2, y, 1, 1).data;
        samples.push([y, pixel[0], pixel[1], pixel[2]]);
      }
      return samples;
    } catch {
      return null;
    }
  });

  if (stripData && stripData.length > 0 && stripData[0][1] !== 0) {
    console.log("Vertical color strip (y, r, g, b):");
    for (const [y, r, g, b] of stripData) {
      console.log(`  y=${y}: rgb(${r},${g},${b})`);
    }
  } else {
    console.log("Could not read canvas pixels (WebGPU limitation)");
  }

  const impLogs = logs.filter((l) => l.toLowerCase().includes("imposter")).slice(0, 2);
  for (const l of impLogs) console.log(l.replace(/%c[^%]*/g, "").trim());
});
