import { test, expect } from "@playwright/test";

test("zoom in repeatedly without buffer-size validation errors", async ({ page }) => {
  test.setTimeout(40_000);

  const validationErrors: string[] = [];
  page.on("console", (msg) => {
    const text = msg.text();
    // wgpu / WebGPU validation messages land as console.error.
    if (msg.type() === "error" && (
      text.includes("isn't a multiple of") ||
      text.includes("Invalid BindGroup") ||
      text.includes("Invalid CommandBuffer")
    )) {
      validationErrors.push(text);
    }
  });

  await page.setViewportSize({ width: 1024, height: 768 });
  await page.goto("/");
  await page.waitForTimeout(3_000);

  // Click canvas to focus + lock.
  await page.locator("canvas").first().click({ position: { x: 512, y: 384 } });
  await page.waitForTimeout(300);

  // Scroll-wheel down many times to zoom in (each step changes the
  // anchor depth, growing the packed tree). The original bug fired
  // when the 1.5x headroom calc produced a non-multiple-of-4 buffer
  // size on the recreate path.
  for (let i = 0; i < 30; i++) {
    await page.mouse.wheel(0, 120);
    await page.waitForTimeout(50);
  }
  await page.waitForTimeout(1_500);

  console.log("validation errors:", validationErrors.length);
  validationErrors.slice(0, 3).forEach(e => console.log("  ", e.split("\n")[0].slice(0, 200)));

  await page.screenshot({ path: "../tmp/wasm-zoom-buffer.png" });

  expect(validationErrors).toHaveLength(0);
});
