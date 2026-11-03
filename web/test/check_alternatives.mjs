// Cross-check the browser's alternative routes and charge zone against Python.
// node web/test/check_alternatives.mjs <data-dir> <cases.json>

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";
import { Engine, viewArrays } from "../js/engine.js";
import { alternatives, zoneNodes } from "../js/alternatives.js";

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
const zone = zoneNodes(engine, manifest.charge);
const count = zone.reduce((a, b) => a + b, 0);
if (count !== cases.zone_count) failures.push(`zone nodes ${count} vs ${cases.zone_count}`);
for (const c of cases.cases) {
  const { routes } = alternatives(engine, c.s, c.t, { zone, avoidZone: c.avoid });
  const got = routes.map((r) => r.seconds);
  if (got.length !== c.seconds.length) failures.push(`${c.s}->${c.t}: ${got.length} routes vs ${c.seconds.length}`);
  got.forEach((s, i) => {
    if (Math.abs(s - c.seconds[i]) > 1e-6 * Math.max(1, s)) failures.push(`${c.s}->${c.t} #${i}: ${s} vs ${c.seconds[i]}`);
    if (routes[i].entersZone !== c.zone[i]) failures.push(`${c.s}->${c.t} #${i}: zone ${routes[i].entersZone}`);
  });
}
if (failures.length) {
  console.log(failures.slice(0, 20).join("\n"));
  process.exit(1);
}
console.log(JSON.stringify({ checked: cases.cases.length, zone: count }));
