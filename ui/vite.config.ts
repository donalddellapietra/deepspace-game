import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    // Output to dist/ as a single-page app
    outDir: "dist",
    // Generate predictable filenames (no hash) for Trunk integration
    rollupOptions: {
      output: {
        entryFileNames: "ui.js",
        chunkFileNames: "ui-[name].js",
        assetFileNames: "ui.[ext]",
      },
    },
  },
});
