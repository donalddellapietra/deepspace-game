import { defineConfig } from "@playwright/test";

const port = process.env.TRUNK_PORT ?? "8080";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    headless: true,
    // Use real Chrome (not bundled Chromium) for GPU/WebGPU support
    channel: "chrome",
    launchOptions: {
      args: [
        "--enable-gpu",
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--use-angle=metal",
      ],
    },
  },
  // Don't start a server — assume trunk serve is already running
  webServer: undefined,
});
