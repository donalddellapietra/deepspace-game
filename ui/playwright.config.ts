import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  use: {
    // Point at the running trunk server
    baseURL: "http://127.0.0.1:8083",
    headless: true,
  },
  // Don't start a server — assume trunk serve is already running
  webServer: undefined,
});
