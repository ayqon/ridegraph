// Learned ETA in the browser: the same features as fastroute/eta.py,
// fed through the same gradient-boosted trees exported from scikit-learn.
//
// The model predicts log(observed / graph ETA) for a route, so its output is
// a correction applied to the router's own estimate.

import { TURN } from "./engine.js";

const KM_PER_DEG_LAT = 111.32;
const EARTH_RADIUS_M = 6371008.8;
const RAD = Math.PI / 180;

function haversineM(lat1, lon1, lat2, lon2) {
  const p1 = lat1 * RAD;
  const p2 = lat2 * RAD;
  const dp = p2 - p1;
  const dl = (lon2 - lon1) * RAD;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(a));
}

export class EtaModel {
  constructor(model, engine) {
    this.model = model;
    this.engine = engine;
    const { n, m, first_out: first, head } = engine;
    this.mStart = new Int32Array(m + 1);
    for (let i = 0; i < m; i++) this.mStart[i + 1] = this.mStart[i] + (first[head[i] + 1] - first[head[i]]);
    let latMin = Infinity;
    let latMax = -Infinity;
    let lonMin = Infinity;
    let lonMax = -Infinity;
    for (let i = 0; i < n; i++) {
      latMin = Math.min(latMin, engine.lat[i]);
      latMax = Math.max(latMax, engine.lat[i]);
      lonMin = Math.min(lonMin, engine.lon[i]);
      lonMax = Math.max(lonMax, engine.lon[i]);
    }
    this.lat0 = (latMin + latMax) / 2;
    this.lon0 = (lonMin + lonMax) / 2;
    this.kmPerDegLon = KM_PER_DEG_LAT * Math.cos(this.lat0 * RAD);
  }

  group(roadClass) {
    return roadClass < 6 ? 0 : roadClass < 10 ? 1 : 2;
  }

  xyKm(node) {
    const e = this.engine;
    return [(e.lon[node] - this.lon0) * this.kmPerDegLon, (e.lat[node] - this.lat0) * KM_PER_DEG_LAT];
  }

  /** Feature vector for a route through junctions, at the engine's current time. */
  features(path) {
    const e = this.engine;
    const times = e.roads.weight;
    const edges = [];
    for (let i = 1; i < path.length; i++) edges.push(e.roadBetween(path[i - 1], path[i]));

    let eta = 0;
    let length = 0;
    const byGroup = [0, 0, 0];
    for (const edge of edges) {
      eta += times[edge];
      const seg = e.length_m[edge];
      length += seg;
      byGroup[this.group(e.road_class[edge])] += seg;
    }
    const a = path[0];
    const b = path[path.length - 1];
    const crow = haversineM(e.lat[a], e.lon[a], e.lat[b], e.lon[b]);

    const counts = { [TURN.LEFT]: 0, [TURN.RIGHT]: 0, [TURN.SHARP_LEFT]: 0, [TURN.SHARP_RIGHT]: 0, [TURN.U_TURN]: 0 };
    for (let i = 1; i < edges.length; i++) {
      const from = edges[i - 1];
      const kind = e.turn_kind[this.mStart[from] + (edges[i] - e.first_out[e.head[from]])];
      if (kind in counts) counts[kind]++;
    }
    let lights = 0;
    for (let i = 1; i < path.length - 1; i++) if (e.signals[path[i]]) lights++;

    const angle = (2 * Math.PI * e.hour) / 24;
    const [ox, oy] = this.xyKm(a);
    const [dx, dy] = this.xyKm(b);
    const [mx, my] = this.xyKm(path[Math.floor(path.length / 2)]);
    return [
      eta,
      Math.log(Math.max(eta, 1)),
      length / 1000,
      crow / 1000,
      length / Math.max(crow, 1),
      byGroup[0] / Math.max(length, 1),
      byGroup[1] / Math.max(length, 1),
      byGroup[2] / Math.max(length, 1),
      counts[TURN.LEFT],
      counts[TURN.RIGHT],
      counts[TURN.SHARP_LEFT] + counts[TURN.SHARP_RIGHT],
      counts[TURN.U_TURN],
      lights,
      edges.length,
      Math.sin(angle),
      Math.cos(angle),
      e.weekend ? 1 : 0,
      ox,
      oy,
      dx,
      dy,
      mx,
      my,
    ];
  }

  /** Walk every tree; the sum is the predicted log ratio. */
  logRatio(x) {
    let total = this.model.baseline;
    for (const t of this.model.trees) {
      let i = 0;
      while (!t.leaf[i]) i = x[t.feature[i]] <= t.threshold[i] ? t.left[i] : t.right[i];
      total += t.value[i];
    }
    return total;
  }

  /** Learned ETA for a route through junctions; needs at least one road segment. */
  predict(path) {
    if (path.length < 2) return null;
    const x = this.features(path);
    const ratio = Math.exp(this.logRatio(x));
    return { seconds: x[0] * ratio, graphSeconds: x[0], ratio, features: x };
  }
}
