import { test } from "@playwright/test";

test("WASM diagnostic — capture console + errors", async ({ page }) => {
  test.setTimeout(30_000);

  const consoleMessages: string[] = [];
  const pageErrors: string[] = [];
  const requestFailures: string[] = [];

  page.on("console", (msg) => {
    consoleMessages.push(`[${msg.type()}] ${msg.text()}`);
  });
  page.on("pageerror", (err) => {
    pageErrors.push(`${err.message}\n${err.stack ?? ""}`);
  });
  page.on("requestfailed", (req) => {
    requestFailures.push(`${req.method()} ${req.url()} — ${req.failure()?.errorText}`);
  });

  await page.goto("/");
  // Give WASM time to load + start
  await page.waitForTimeout(8_000);

  const diag = await page.evaluate(async () => {
    const hasWebGPU = !!(navigator as any).gpu;
    let adapterOk = false;
    let adapterInfo: any = null;
    if (hasWebGPU) {
      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        adapterOk = !!adapter;
        if (adapter) {
          adapterInfo = {
            features: [...adapter.features].slice(0, 5),
            limits: { maxBindGroups: adapter.limits.maxBindGroups },
          };
        }
      } catch (e: any) {
        adapterInfo = { error: e.message };
      }
    }
    const canvas = document.querySelector("canvas");
    return {
      hasWebGPU,
      adapterOk,
      adapterInfo,
      canvasFound: !!canvas,
      canvasSize: canvas ? { w: canvas.width, h: canvas.height } : null,
      perfData: (window as any).__perfData,
    };
  });

  console.log("=== DIAGNOSTIC ===");
  console.log("WebGPU available:", diag.hasWebGPU);
  console.log("Adapter OK:", diag.adapterOk);
  console.log("Adapter info:", JSON.stringify(diag.adapterInfo));
  console.log("Canvas found:", diag.canvasFound, diag.canvasSize);
  console.log("Perf data:", JSON.stringify(diag.perfData));
  console.log("\n=== CONSOLE (" + consoleMessages.length + ") ===");
  consoleMessages.forEach((m) => console.log(m));
  console.log("\n=== PAGE ERRORS (" + pageErrors.length + ") ===");
  pageErrors.forEach((e) => console.log(e));
  console.log("\n=== REQUEST FAILURES (" + requestFailures.length + ") ===");
  requestFailures.forEach((r) => console.log(r));
});
