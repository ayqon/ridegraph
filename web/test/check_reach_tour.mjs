// Cross-check the browser's travel-time maps, many-to-many and tour planning
// against Python.  node web/test/check_reach_tour.mjs <data-dir> <cases.json>

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";
import { Engine, viewArrays } from "../js/engine.js";
import { planTour } from "../js/tour.js";

const [dataDir, casesPath] = process.argv.slice(2);
const load = (file) => {
  const buf = gunzipSync(readFileSync(join(dataDir, file)));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const manifest = JSON.parse(readFileSync(join(dataDir, "manifest.json"), "utf8"));
const engine = new Engine(manifest, load(manifest.graph.file));
engine.fetchHierarchy = async (preset) => viewArrays(load(preset.file), preset.arrays);
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
await engine.usePreset(manifest.presets.find((p) => p.id === cases.preset));

const failures = [];
let checked = 0;
for (const r of cases.reach) {
  const { times } = engine.timesWithin(r.origin, r.budget, r.arriving);
  const reached = [];
  times.forEach((t, v) => Number.isFinite(t) && reached.push(v));
  checked++;
  if (reached.length !== Object.keys(r.times).length) failures.push(`reach ${r.origin}: ${reached.length} vs ${Object.keys(r.times).length}`);
  for (const [v, t] of Object.entries(r.times)) {
    if (Math.abs(times[Number(v)] - t) > 1e-6 * Math.max(1, t)) failures.push(`reach ${r.origin}->${v}: ${times[v]} vs ${t}`);
  }
}
for (const t of cases.tours) {
  const m = engine.manyToMany(t.stops, t.stops);
  checked++;
  m.forEach((row, i) =>
    row.forEach((v, j) => {
      if (Math.abs(v - t.matrix[i][j]) > 1e-3 * Math.max(1, t.matrix[i][j])) failures.push(`matrix ${i},${j}: ${v} vs ${t.matrix[i][j]}`);
    }),
  );
  // Plan on Python's exact matrix so orders are comparable.
  const plan = planTour(t.matrix, t.return_to_start);
  for (const [name, secs] of Object.entries(t.seconds)) {
    if (Math.abs(plan.seconds[name] - secs) > 1e-6 * secs) failures.push(`${name}: ${plan.seconds[name]} vs ${secs}`);
  }
}
console.log(JSON.stringify({ checked, failures: failures.slice(0, 20), failed: failures.length }));
process.exit(failures.length ? 1 : 0);
