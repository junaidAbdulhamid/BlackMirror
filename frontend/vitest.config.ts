import { defineConfig } from "vitest/config";
import { resolve } from "node:path";

export default defineConfig({
  resolve: { alias: { "@": resolve(__dirname, ".") } },
  test: { environment: "node", include: ["**/__tests__/**/*.test.ts", "**/__bench__/**/*.test.ts"] },
});
