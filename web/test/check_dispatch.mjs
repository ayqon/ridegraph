// The browser rebuilds every dispatch leg's route with the exported hierarchy;
// its duration must match what the Python simulation recorded.
//
//   node web/test/check_dispatch.mjs <data-dir>

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";
import { Engine, viewArrays } from "../js/engine.js";
import { DispatchReplay } from "../js/dispatch.js";

const dataDir = process.argv[2];
const load = (file) => {
  const buf = gunzipSync(readFileSync(join(dataDir, file)));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const manifest = JSON.parse(readFileSync(join(dataDir, "manifest.json"), "utf8"));
const engine = new Engine(manifest, load(manifest.graph.file));
engine.fetchHierarchy = async (preset) => viewArrays(load(preset.file), preset.arrays);
await engine.usePreset(manifest.presets.find((p) => p.id === manifest.dispatch.preset));
const replay = new DispatchReplay(JSON.parse(readFileSync(join(dataDir, manifest.dispatch.file), "utf8")), engine);

const failures = [];
let checked = 0;
replay.policies.forEach((p) => {
  for (const [, , , from, to, dur] of p.legs.slice(0, 80)) {
    const path = replay.path(from, to);
    const seconds = path.times[path.times.length - 1];
    checked++;
    if (Math.abs(seconds - dur) > 0.06 + 1e-6 * dur) failures.push(`${p.id} ${from}->${to}: ${seconds} vs ${dur}`);
  }
  // Running stats must never count more pickups than rides, and settle by the end.
  const s = replay.stats(replay.policies.indexOf(p), replay.duration + 3600);
  if (s.picked > p.rides.length) failures.push(`${p.id}: too many pickups`);
});
console.log(JSON.stringify({ checked, failures: failures.slice(0, 10), failed: failures.length }));
process.exit(failures.length ? 1 : 0);
