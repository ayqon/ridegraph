// Cross-check the browser ETA model against Python.
//
//   node web/test/check_eta.mjs <data-dir> <cases.json>
//
// cases.json: [{"preset": "0800", "path": [..junctions..], "features": [...], "log_ratio": 0.12}, ...]

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";
import { Engine, viewArrays } from "../js/engine.js";
import { EtaModel } from "../js/eta.js";

const [dataDir, casesPath] = process.argv.slice(2);
const load = (file) => {
  const buf = gunzipSync(readFileSync(join(dataDir, file)));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const manifest = JSON.parse(readFileSync(join(dataDir, "manifest.json"), "utf8"));
const engine = new Engine(manifest, load(manifest.graph.file));
engine.fetchHierarchy = async (preset) => viewArrays(load(preset.file), preset.arrays);
const presets = Object.fromEntries(manifest.presets.map((p) => [p.id, p]));
const model = new EtaModel(JSON.parse(readFileSync(join(dataDir, manifest.eta.file), "utf8")), engine);

const failures = [];
let checked = 0;
for (const c of JSON.parse(readFileSync(casesPath, "utf8"))) {
  await engine.usePreset(presets[c.preset]);
  const p = model.predict(c.path);
  checked++;
  p.features.forEach((v, i) => {
    if (Math.abs(v - c.features[i]) > 1e-9 * Math.max(1, Math.abs(c.features[i]))) {
      failures.push(`case ${checked} feature ${manifest.eta ? i : i}: ${v} vs ${c.features[i]}`);
    }
  });
  if (Math.abs(Math.log(p.ratio) - c.log_ratio) > 1e-9) failures.push(`case ${checked}: ${Math.log(p.ratio)} vs ${c.log_ratio}`);
  const w = model.logWindow(p.features);
  if (Math.abs(w[0] - c.window[0]) > 1e-9 || Math.abs(w[1] - c.window[1]) > 1e-9) {
    failures.push(`case ${checked}: window ${w} vs ${c.window}`);
  }
  if (!(p.low <= p.seconds && p.seconds <= p.high)) failures.push(`case ${checked}: ETA outside its own window`);
  const slow = model.slowdownAt(engine.lat[c.path[0]], engine.lon[c.path[0]]);
  if (!(slow > 0 && Number.isFinite(slow))) failures.push(`case ${checked}: bad slowdown ${slow}`);
}
console.log(JSON.stringify({ checked, failures: failures.slice(0, 20), failed: failures.length }));
process.exit(failures.length ? 1 : 0);
