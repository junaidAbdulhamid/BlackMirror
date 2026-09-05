/** Required end-to-end Phase 6 workflow: real artifacts through API, UI, and browser. */
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const apiPort = 8766;
const webPort = 3766;
const root = fileURLToPath(new URL("../../", import.meta.url));
const frontend = fileURLToPath(new URL("../", import.meta.url));
const children = [];
const start = (command, args, options) => {
  const child = spawn(command, args, { stdio: "pipe", ...options });
  children.push(child);
  return child;
};
start(`${root}.venv/bin/uvicorn`, ["blackmirror.api.app:app", "--port", String(apiPort)], {
  cwd: root,
});
start("npm", ["run", "dev", "--", "-p", String(webPort)], {
  cwd: frontend,
  env: { ...process.env, NEXT_PUBLIC_API_ORIGIN: `http://127.0.0.1:${apiPort}` },
});

async function waitFor(url) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`server did not become ready: ${url}`);
}

let browser;
try {
  await Promise.all([
    waitFor(`http://127.0.0.1:${apiPort}/api/health`),
    waitFor(`http://127.0.0.1:${webPort}/scoring`),
  ]);
  browser = await chromium.launch({ channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const problems = [];
  page.on("pageerror", (error) => problems.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(message.text());
  });
  page.on("response", (response) => {
    if (response.status() >= 500) problems.push(`${response.status()} ${response.url()}`);
  });
  const started = performance.now();
  await page.goto(`http://127.0.0.1:${webPort}/scoring`, { waitUntil: "networkidle" });
  const variants = page.locator('input[type="checkbox"]');
  if ((await variants.count()) < 2) throw new Error("fewer than two real runs are available");
  await variants.nth(0).check();
  await variants.nth(1).check();
  await page.getByRole("button", { name: "Add objective" }).click();
  await page.getByRole("button", { name: "Score variants" }).click();
  await page.getByText("Variant leaderboard").waitFor({ timeout: 30_000 });
  await page.getByText("Why this variant ranked highest").waitFor();
  const report = {
    loadAndScoreMs: performance.now() - started,
    variants: 2,
    leaderboard: true,
    explanation: true,
    problems,
  };
  console.log(JSON.stringify(report, null, 2));
  if (problems.length) throw new Error("browser workflow reported errors");
} finally {
  await browser?.close();
  for (const child of children) child.kill("SIGTERM");
}
