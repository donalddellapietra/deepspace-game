import { test, expect } from "@playwright/test";

// These tests run against a live trunk serve instance at localhost:8080.
// The Bevy WASM game must be loaded for the React overlay to render.

test.describe("React UI Overlay", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    // Wait for both the canvas (Bevy) and the React root to be present
    await page.waitForSelector("canvas", { timeout: 20_000 });
    await page.waitForSelector("#root", { timeout: 5_000 });
  });

  test("React root mounts over the canvas", async ({ page }) => {
    const root = page.locator("#root");
    await expect(root).toBeVisible();

    // Root should have pointer-events: none (click-through)
    const pointerEvents = await root.evaluate(
      (el) => getComputedStyle(el).pointerEvents
    );
    expect(pointerEvents).toBe("none");
  });

  test("Hotbar renders with 10 slots", async ({ page }) => {
    const hotbar = page.locator(".hotbar");
    await expect(hotbar).toBeVisible();

    const slots = page.locator(".hotbar-swatch");
    await expect(slots).toHaveCount(10);

    // Hotbar should have pointer-events: auto (interactive)
    const pointerEvents = await hotbar.evaluate(
      (el) => getComputedStyle(el).pointerEvents
    );
    expect(pointerEvents).toBe("auto");
  });

  test("Hotbar shows key hints 1-9 and 0", async ({ page }) => {
    const keys = page.locator(".hotbar-key");
    await expect(keys).toHaveCount(10);

    const labels = await keys.allTextContents();
    expect(labels).toEqual(["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]);
  });

  test("Mode indicator shows layer", async ({ page }) => {
    const indicator = page.locator(".mode-indicator");
    await expect(indicator).toBeVisible();

    const layerText = page.locator(".mode-layer");
    await expect(layerText).toContainText("Layer");
  });

  test("Inventory is hidden by default", async ({ page }) => {
    const inventory = page.locator(".inventory-panel");
    await expect(inventory).toHaveCount(0);
  });

  test("Color picker is hidden by default", async ({ page }) => {
    const picker = page.locator(".color-picker-panel");
    await expect(picker).toHaveCount(0);
  });

  test("Toast container exists but is empty", async ({ page }) => {
    const toasts = page.locator(".toast");
    await expect(toasts).toHaveCount(0);
  });

  test("CSS variables are defined on :root", async ({ page }) => {
    const accentColor = await page.evaluate(() =>
      getComputedStyle(document.documentElement)
        .getPropertyValue("--accent")
        .trim()
    );
    expect(accentColor).toBeTruthy();
  });

  test("game state bridge is wired up", async ({ page }) => {
    // The React app registers window.__onGameState and __pollUiCommands
    const hasOnGameState = await page.evaluate(
      () => typeof (window as any).__onGameState === "function"
    );
    expect(hasOnGameState).toBe(true);

    const hasPollCommands = await page.evaluate(
      () => typeof (window as any).__pollUiCommands === "function"
    );
    expect(hasPollCommands).toBe(true);
  });

  test("pushing hotbar state updates the UI", async ({ page }) => {
    // Simulate Rust pushing hotbar state
    await page.evaluate(() => {
      (window as any).__onGameState({
        type: "hotbar",
        data: {
          active: 2,
          slots: Array.from({ length: 10 }, (_, i) => ({
            kind: "block",
            index: i + 1,
            name: `Block ${i + 1}`,
            color: [i * 0.1, 0.5, 0.5, 1.0],
          })),
          layer: 3,
        },
      });
    });

    // Active slot (index 2) should have the active class
    const activeSlot = page.locator(".hotbar-swatch.active");
    await expect(activeSlot).toHaveCount(1);

    // Label should show the active block name
    const label = page.locator(".hotbar-label");
    await expect(label).toContainText("Block 3");
  });

  test("pushing inventory state opens the panel", async ({ page }) => {
    await page.evaluate(() => {
      (window as any).__onGameState({
        type: "inventory",
        data: {
          open: true,
          builtinBlocks: [
            { voxel: 1, name: "Stone", color: [0.5, 0.5, 0.5, 1.0] },
            { voxel: 2, name: "Dirt", color: [0.45, 0.3, 0.15, 1.0] },
          ],
          customBlocks: [],
          savedMeshes: [],
          layer: 2,
        },
      });
    });

    const panel = page.locator(".inventory-panel");
    await expect(panel).toBeVisible();

    // Should show 2 block tiles
    const tiles = page.locator(".inv-block-tile");
    await expect(tiles).toHaveCount(2);
  });

  test("pushing color picker state opens the picker", async ({ page }) => {
    await page.evaluate(() => {
      (window as any).__onGameState({
        type: "colorPicker",
        data: { open: true, r: 0.8, g: 0.2, b: 0.5 },
      });
    });

    const picker = page.locator(".color-picker-panel");
    await expect(picker).toBeVisible();

    // Hex display should reflect the RGB values
    const hex = page.locator(".cp-hex");
    await expect(hex).toContainText("#CC3380");
  });

  test("clicking inventory block sends assignBlockToSlot command", async ({
    page,
  }) => {
    // Set up command capture, push state, click, and check — all in one
    // evaluate to avoid DOM detach from game loop re-renders.
    const captured = await page.evaluate(async () => {
      const all: any[] = [];
      const orig = (window as any).__pollUiCommands;
      (window as any).__pollUiCommands = () => {
        const result = orig();
        all.push(...JSON.parse(result));
        return result;
      };

      (window as any).__onGameState({
        type: "inventory",
        data: {
          open: true,
          builtinBlocks: [
            { voxel: 3, name: "Grass", color: [0.3, 0.6, 0.2, 1.0] },
          ],
          customBlocks: [],
          savedMeshes: [],
          layer: 2,
        },
      });

      // Wait a tick for React to render
      await new Promise((r) => setTimeout(r, 100));

      const tile = document.querySelector(".inv-block-tile") as HTMLElement;
      tile?.click();

      // Wait for game loop to drain
      await new Promise((r) => setTimeout(r, 500));
      return all;
    });

    expect(captured).toContainEqual({
      cmd: "assignBlockToSlot",
      voxel: 3,
    });
  });

  test("color picker slider dispatches setColorPickerRgb command", async ({
    page,
  }) => {
    const captured = await page.evaluate(async () => {
      const all: any[] = [];
      const orig = (window as any).__pollUiCommands;
      (window as any).__pollUiCommands = () => {
        const result = orig();
        all.push(...JSON.parse(result));
        return result;
      };

      (window as any).__onGameState({
        type: "colorPicker",
        data: { open: true, r: 0.5, g: 0.5, b: 0.5 },
      });

      await new Promise((r) => setTimeout(r, 100));

      const slider = document.querySelector(
        '.cp-slider-row input[type="range"]'
      ) as HTMLInputElement;
      if (slider) {
        const nativeSetter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype,
          "value"
        )!.set!;
        nativeSetter.call(slider, "0.9");
        slider.dispatchEvent(new Event("input", { bubbles: true }));
        slider.dispatchEvent(new Event("change", { bubbles: true }));
      }

      await new Promise((r) => setTimeout(r, 500));
      return all;
    });

    const rgbCmd = captured.find(
      (c: any) => c.cmd === "setColorPickerRgb"
    );
    expect(rgbCmd).toBeDefined();
  });
});
