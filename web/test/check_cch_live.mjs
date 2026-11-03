// Cross-check the browser's Customizable CH and live-disruption matching against Python.
// node web/test/check_cch_live.mjs <data-dir> <cases.json>

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";
import { Engine, viewArrays } from "../js/engine.js";
import { parse, match } from "../js/live.js";

const [dataDir, casesPath] = process.argv.slice(2);
const load = (file) => {
  const buf = gunzipSync(readFileSync(join(dataDir, file)));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const manifest = JSON.parse(readFileSync(join(dataDir, "manifest.json"), "utf8"));
const engine = new Engine(manifest, load(manifest.graph.file));
engine.fetchHierarchy = async (preset) => viewArrays(load(preset.file), preset.arrays);
engine.fetchCCH = async (info) => viewArrays(load(info.file), info.arrays);
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
await engine.usePreset(manifest.presets.find((p) => p.id === cases.preset));
await engine.loadCCH();

const failures = [];
const same = (a, b) => (a === null ? !Number.isFinite(b) : Math.abs(a - b) <= 1e-6 * Math.max(1, a));

// Matching: the snapshot and the raw payload must give Python's multipliers.
const snapshot = JSON.parse(readFileSync(join(dataDir, manifest.live.snapshot), "utf8"));
for (const [label, payload] of [["payload", cases.payload], ["snapshot", snapshot]]) {
  const { factor, matched } = match(engine, parse(payload), manifest.live);
  if (matched.length !== 2) failures.push(`${label}: ${matched.length} stretches matched`);
  cases.factor.forEach((f, e) => {
    if (!same(f, factor[e])) failures.push(`${label} edge ${e}: ${factor[e]} vs ${f}`);
  });
}

for (const [name, live] of [["plain", null], ["live", match(engine, parse(cases.payload), manifest.live).factor]]) {
  engine.setLive(live);
  cases.pairs.forEach(([s, t], i) => {
    const r = engine.route("cch", s, t);
    const want = cases.expected[name][i];
    if (!same(want, r.seconds)) failures.push(`${name} ${s}->${t}: ${r.seconds} vs ${want}`);
    if (r.found) {
      // The unpacked path must cost what the query says, on current times.
      const d = engine.route("dijkstra", s, t);
      if (Math.abs(d.seconds - r.seconds) > 1e-6 * Math.max(1, r.seconds)) failures.push(`${name} path ${s}->${t}`);
    }
  });
}
if (failures.length) {
  console.log(failures.slice(0, 20).join("\n"));
  process.exit(1);
}
console.log(JSON.stringify({ checked: cases.pairs.length * 2 }));
