/**
 * End-to-end visual validation against a REAL Phase 1 run.
 *
 * Checks the things a unit test cannot: that WebGL actually painted cortex,
 * that both hemispheres are visible rather than one occluding the other, that
 * changing the timestep repaints the surface, and what the frame rate is.
 *
 * Requires the API and frontend running, plus a Chrome/Chromium Playwright can
 * drive. Usage:
 *   node scripts/validate-visualization.mjs [runId]
 */

import { chromium } from "playwright";

const RUN = process.argv[2] ?? "20260903T160907Z-63dfdbb8";
const ORIGIN = process.env.NEUROSPLIT_ORIGIN ?? "http://127.0.0.1:3000";
const SHOTS = process.env.NEUROSPLIT_SHOTS ?? "/tmp/shots";

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({
  viewport: { width: 1680, height: 1050 },
  deviceScaleFactor: 2,
});

/**
 * Resources a run may legitimately lack.
 *
 * Phase 3 analytics are derived on demand (`blackmirror analyze <run>`), so a
 * run without them 404s by design and the panel renders nothing. Treating that
 * as a failure would make this gate useless on un-analysed runs.
 */
const OPTIONAL_PATHS = [/\/analytics(\/|$)/];
const isOptional = (url) => OPTIONAL_PATHS.some((p) => p.test(url));

const problems = [];
const expected404s = [];

page.on("pageerror", (e) => problems.push(String(e)));
page.on("response", (r) => {
  if (r.status() < 400) return;
  if (r.status() === 404 && isOptional(r.url())) expected404s.push(r.url());
  else problems.push(`${r.status()} ${r.url()}`);
});
// Console errors are recorded, but a bare "failed to load" for an optional
// resource is the same event already classified above.
page.on("console", (m) => {
  if (m.type() !== "error") return;
  const text = m.text();
  if (/Failed to load resource/.test(text) && expected404s.length > 0) return;
  problems.push(text);
});

const started = Date.now();
await page.goto(`${ORIGIN}/experiments/${RUN}`, { waitUntil: "networkidle" });
await page.waitForSelector("canvas");
await page.waitForTimeout(2500);
const loadMs = Date.now() - started;

/** Fraction of the canvas covered by cortex, split either side of centre. */
const coverage = () =>
  page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    const { width: w, height: h } = canvas;
    const px = new Uint8Array(4 * w * h);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let left = 0;
    let right = 0;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const i = (y * w + x) * 4;
        const surface =
          Math.abs(px[i] - 10) > 18 ||
          Math.abs(px[i + 1] - 12) > 18 ||
          Math.abs(px[i + 2] - 16) > 18;
        if (surface) (x < w / 2 ? left++ : right++);
      }
    }
    return { left, right, total: w * h };
  });

const centrePixels = () =>
  page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    const w = 400;
    const h = 300;
    const px = new Uint8Array(4 * w * h);
    gl.readPixels(
      Math.floor(canvas.width / 2 - w / 2),
      Math.floor(canvas.height / 2 - h / 2),
      w, h, gl.RGBA, gl.UNSIGNED_BYTE, px,
    );
    return Array.from(px);
  });

await page.screenshot({ path: `${SHOTS}/default.png` });
const both = await coverage();

await page.getByRole("button", { name: "Lateral" }).click();
await page.waitForTimeout(800);
await page.screenshot({ path: `${SHOTS}/lateral.png` });
const lateral = await coverage();

const before = await centrePixels();
for (let i = 0; i < 3; i += 1) {
  await page.getByLabel("Next prediction sample").click();
  await page.waitForTimeout(350);
}
const after = await centrePixels();
let changed = 0;
for (let i = 0; i < before.length; i += 1) {
  if (Math.abs(before[i] - after[i]) > 6) changed += 1;
}
await page.screenshot({ path: `${SHOTS}/last-timestep.png` });

await page.getByRole("button", { name: "Restart" }).click();
await page.locator('button[aria-label="Play"]').click().catch(() => {});
const fps = await page.evaluate(
  () =>
    new Promise((resolve) => {
      let frames = 0;
      const start = performance.now();
      const tick = () => {
        frames += 1;
        if (performance.now() - start < 3000) requestAnimationFrame(tick);
        else resolve((frames * 1000) / (performance.now() - start));
      };
      requestAnimationFrame(tick);
    }),
);

const bothHemispheresVisible =
  both.left > both.total * 0.01 && both.right > both.total * 0.01;

// The developer overlay is the app's own account of alignment and coverage;
// reading it back cross-checks the renderer against the data it loaded.
await page.keyboard.press("d");
await page.waitForTimeout(400);
const debugText = await page
  .locator("text=Prediction index")
  .locator("..")
  .innerText()
  .catch(() => "");
const debug = Object.fromEntries(
  debugText
    .split("\n")
    .map((line) => line.split(":").map((part) => part.trim()))
    .filter((parts) => parts.length === 2),
);

const report = {
  run: RUN,
  loadMs,
  fps: Number(fps.toFixed(1)),
  coverage: { default: both, lateral },
  bothHemispheresVisible,
  pixelsChangedBetweenTimesteps: changed,
  aligned: debug.Aligned ?? "unknown",
  covered: debug.Covered ?? "unknown",
  nonFiniteVertices: debug["Non-finite"] ?? "unknown",
  optional404s: expected404s.length,
  problems,
};
console.log(JSON.stringify(report, null, 2));

await browser.close();
const failed =
  problems.length > 0 ||
  !bothHemispheresVisible ||
  changed === 0 ||
  fps < 30 ||
  report.aligned !== "yes";
process.exit(failed ? 1 : 0);
