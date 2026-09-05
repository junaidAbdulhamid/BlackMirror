/** Non-skipping end-to-end cortical difference benchmark (Next + FastAPI + WebGL). */
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const comparison = process.env.NEUROSPLIT_COMPARISON ?? "cmp-20260905T001441Z-18eac4e1";
const candidate = process.env.NEUROSPLIT_CANDIDATE ?? "20260904T213725Z-5271c5f9";
const apiPort = 8765;
const webPort = 3765;
const root = fileURLToPath(new URL("../../", import.meta.url));
const frontend = fileURLToPath(new URL("../", import.meta.url));
const children = [];
const start = (command, args, options) => {
  const child = spawn(command, args, { stdio: "pipe", ...options });
  children.push(child);
  return child;
};
const api = start(`${root}.venv/bin/uvicorn`, ["blackmirror.api.app:app", "--port", String(apiPort)], { cwd: root });
const web = start("npm", ["run", "dev", "--", "-p", String(webPort)], {
  cwd: frontend,
  env: { ...process.env, NEXT_PUBLIC_API_ORIGIN: `http://127.0.0.1:${apiPort}` },
});
const waitFor = async (url) => {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`server did not become ready: ${url}`);
};

let browser;
try {
  await Promise.all([
    waitFor(`http://127.0.0.1:${apiPort}/api/health`),
    waitFor(`http://127.0.0.1:${webPort}/comparisons/${comparison}`),
  ]);
  browser = await chromium.launch({ channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const problems = [];
  page.on("pageerror", (error) => problems.push(String(error)));
  page.on("console", (message) => { if (message.type() === "error") problems.push(message.text()); });
  page.on("response", (response) => { if (response.status() >= 400) problems.push(`${response.status()} ${response.url()}`); });
  const started = performance.now();
  await page.goto(`http://127.0.0.1:${webPort}/comparisons/${comparison}`, { waitUntil: "networkidle" });
  await page.getByText("Cortical difference map").waitFor();
  const canvas = page.locator("canvas").last();
  await canvas.waitFor();
  await page.waitForTimeout(1000);
  const loadMs = performance.now() - started;
  const pixels = async () => canvas.evaluate((element) => {
    const gl = element.getContext("webgl2") ?? element.getContext("webgl");
    if (!gl) throw new Error("WebGL context unavailable");
    const data = new Uint8Array(element.width * element.height * 4);
    gl.readPixels(0, 0, element.width, element.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    let left = 0, right = 0;
    for (let y = 0; y < element.height; y += 1) for (let x = 0; x < element.width; x += 1) {
      const index = (y * element.width + x) * 4;
      const painted = Math.abs(data[index] - 10) > 18 || Math.abs(data[index + 1] - 12) > 18 || Math.abs(data[index + 2] - 16) > 18;
      if (painted) (x < element.width / 2 ? left++ : right++);
    }
    return { left, right, total: element.width * element.height, sample: Array.from(data) };
  });
  const before = await pixels();
  const updateStarted = performance.now();
  await page.getByLabel("Difference timestamp").fill("1");
  await page.waitForTimeout(250);
  const updateMs = performance.now() - updateStarted;
  const after = await pixels();
  let changed = 0;
  for (let index = 0; index < before.sample.length; index += 1) if (Math.abs(before.sample[index] - after.sample[index]) > 5) changed += 1;
  const fps = await page.evaluate(() => new Promise((resolve) => {
    let frames = 0; const start = performance.now();
    const tick = () => { frames += 1; if (performance.now() - start < 1500) requestAnimationFrame(tick); else resolve(frames * 1000 / (performance.now() - start)); };
    requestAnimationFrame(tick);
  }));
  const report = { comparison, candidate, loadMs, updateMs, fps, leftPixels: before.left, rightPixels: before.right, changedPixels: changed, problems };
  console.log(JSON.stringify(report, null, 2));
  if (before.left < before.total * 0.005 || before.right < before.total * 0.005) throw new Error("both hemispheres were not painted");
  if (changed === 0) throw new Error("time switch did not repaint the difference cortex");
  if (updateMs > 1000 || fps < 30 || problems.length) throw new Error("browser benchmark threshold/error gate failed");
} finally {
  await browser?.close();
  for (const child of children) child.kill("SIGTERM");
}
