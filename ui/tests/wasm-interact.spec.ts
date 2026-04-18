import { test } from "@playwright/test";

test("WASM interactive — focus, press E, screenshot", async ({ page }) => {
  test.setTimeout(30_000);

  await page.setViewportSize({ width: 1024, height: 768 });
  await page.goto("/");
  await page.waitForTimeout(3_000);

  // Click on the canvas to give it focus + lock pointer (game uses cursor lock).
  const canvas = page.locator("canvas").first();
  await canvas.click({ position: { x: 512, y: 384 } });
  await page.waitForTimeout(500);

  await page.screenshot({ path: "../tmp/wasm-locked.png" });

  // Press E for inventory.
  await page.keyboard.press("KeyE");
  await page.waitForTimeout(800);
  await page.screenshot({ path: "../tmp/wasm-inventory.png" });

  // Inspect React state via DOM — inventory panel should be visible.
  const inventoryVisible = await page.evaluate(() => {
    return Array.from(document.querySelectorAll("*")).some((el) => {
      const txt = el.textContent ?? "";
      return txt.includes("Inventory") || txt.includes("Builtin") || txt.includes("Custom");
    });
  });
  console.log("inventory visible in DOM:", inventoryVisible);

  // Try Q to zoom layer.
  await page.keyboard.press("KeyQ");
  await page.waitForTimeout(500);
  await page.screenshot({ path: "../tmp/wasm-zoomed.png" });
});
