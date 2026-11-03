// Cross-check the JavaScript engine against expected results from Python.
//
//   node web/test/check.mjs <data-dir> <cases.json>
//
// cases.json: [{"preset": "0800", "turns": false, "s": 1, "t": 2, "seconds": 123.4 | null}, ...]
// Exits non-zero and lists failures if any algorithm disagrees.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";
import { Engine, viewArrays, ALGORITHMS } from "../js/engine.js";

const [dataDir, casesPath] = process.argv.slice(2);
const load = (file) => {
  const buf = gunzipSync(readFileSync(join(dataDir, file)));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const manifest = JSON.parse(readFileSync(join(dataDir, "manifest.json"), "utf8"));
const engine = new Engine(manifest, load(manifest.graph.file));
engine.fetchHierarchy = async (preset) => viewArrays(load(preset.file), preset.arrays);
engine.fetchCCH = async (info) => viewArrays(load(info.file), info.arrays);
await engine.loadCCH();
const presets = Object.fromEntries(manifest.presets.map((p) => [p.id, p]));

const failures = [];
let checked = 0;
for (const c of JSON.parse(readFileSync(casesPath, "utf8"))) {
  await engine.usePreset(presets[c.preset]);
  engine.setTurns(Boolean(c.turns));
  for (const algorithm of ALGORITHMS) {
    if (algorithm === "ch" && !engine.chAvailable) continue;
    if (algorithm === "cch" && !engine.cchAvailable) continue;
    const r = engine.route(algorithm, c.s, c.t);
    checked++;
    const where = `${algorithm} ${c.preset}${c.turns ? " turns" : ""} ${c.s}->${c.t}`;
    if (c.seconds === null) {
      if (r.found) failures.push(`${where}: found a route where Python found none`);
      continue;
    }
    if (!r.found) {
      failures.push(`${where}: no route, expected ${c.seconds}`);
      continue;
    }
    // CH weights ship as float32, so allow rounding there.
    const tol = (algorithm === "ch" ? 1e-5 : 1e-9) * Math.max(1, c.seconds);
    if (Math.abs(r.seconds - c.seconds) > tol) failures.push(`${where}: ${r.seconds} vs ${c.seconds}`);
    if (r.junctions[0] !== c.s || r.junctions[r.junctions.length - 1] !== c.t) failures.push(`${where}: wrong endpoints`);
    if (c.manoeuvres && r.manoeuvres) {
      for (const [k, v] of Object.entries(c.manoeuvres)) {
        if (r.manoeuvres[k] !== v) failures.push(`${where}: ${k} ${r.manoeuvres[k]} vs ${v}`);
      }
    }
    if (r.settled !== r.trace.nodes.length) failures.push(`${where}: trace length mismatch`);
  }
}
console.log(JSON.stringify({ checked, failures: failures.slice(0, 20), failed: failures.length }));
process.exit(failures.length ? 1 : 0);
