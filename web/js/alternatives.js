// Alternative routes and the Congestion Charge zone, ported from
// fastroute/alternatives.py and fastroute/charges.py.
//
// Penalty method: find the quickest route, make the roads on it slower, search
// again, and keep candidates that are different enough, not much slower, and
// free of pointless detours. Searches here use A* on the custom weights (a
// penalty only ever raises a weight, so the straight-line estimate stays
// admissible); Python uses Dijkstra, and both find the same optimal routes.

const EARTH_RADIUS_M = 6371008.8;
const SAFETY = 0.9999;

/** Point-in-polygon by ray casting (charges.inside). */
export function insidePolygon(lat, lon, polygon) {
  let inside = false;
  for (let i = 0, k = polygon.length; i < k; i++) {
    const [y1, x1] = polygon[i];
    const [y2, x2] = polygon[(i + 1) % k];
    if (y1 > lat !== y2 > lat) {
      const xAt = x1 + ((lat - y1) * (x2 - x1)) / (y2 - y1);
      if (lon < xAt) inside = !inside;
    }
  }
  return inside;
}

/** Metres to the nearest polygon edge (charges.distance_to_boundary_m). */
export function distanceToBoundary(lat, lon, polygon) {
  const lat0 = polygon.reduce((a, p) => a + p[0], 0) / polygon.length;
  const kx = 111320 * Math.cos((lat0 * Math.PI) / 180);
  const ky = 110540;
  const px = lon * kx;
  const py = lat * ky;
  let best = Infinity;
  for (let i = 0, k = polygon.length; i < k; i++) {
    const ay = polygon[i][0] * ky;
    const ax = polygon[i][1] * kx;
    const by = polygon[(i + 1) % k][0] * ky;
    const bx = polygon[(i + 1) % k][1] * kx;
    const dx = bx - ax;
    const dy = by - ay;
    const t = Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
    best = Math.min(best, Math.hypot(px - (ax + t * dx), py - (ay + t * dy)));
  }
  return best;
}

/** Junctions inside the zone, clear of the boundary road (charges.zone_nodes). */
export function zoneNodes(engine, charge) {
  const zone = new Uint8Array(engine.n);
  for (let v = 0; v < engine.n; v++) {
    const lat = engine.lat[v];
    const lon = engine.lon[v];
    zone[v] = insidePolygon(lat, lon, charge.zone) && distanceToBoundary(lat, lon, charge.zone) > charge.margin_m ? 1 : 0;
  }
  return zone;
}

export function charging(charge, hour, weekend = false) {
  const [start, end] = weekend ? charge.weekend_hours : charge.weekday_hours;
  const h = ((hour % 24) + 24) % 24;
  return h >= start && h < end;
}

/** A* on the road graph with custom weights; returns { seconds, edges } (CSR edge ids). */
function search(engine, s, t, weight, side) {
  const g = engine.roads;
  const { first_out: first, head, lat, lon } = g;
  const scale = g.maxSpeed > 0 ? (SAFETY * 2 * EARTH_RADIUS_M) / g.maxSpeed : 0;
  const rad = Math.PI / 180;
  const tLat = lat[t] * rad;
  const tLon = lon[t] * rad;
  const cosT = Math.cos(tLat);
  const h = (v) => {
    const vLat = lat[v] * rad;
    const a = Math.sin((tLat - vLat) / 2) ** 2 + Math.cos(vLat) * cosT * Math.sin((tLon - lon[v] * rad) / 2) ** 2;
    return scale * Math.asin(Math.min(1, Math.sqrt(a)));
  };
  const S = side;
  S.reset();
  S.set(s, 0, -1);
  S.heap.push(h(s), s);
  while (S.heap.size) {
    const u = S.heap.pop();
    if (S.isDone(u)) continue;
    S.markDone(u);
    const d = S.get(u);
    if (u === t) {
      const edges = [];
      for (let v = t; v !== s; ) {
        const e = S.parent[v];
        edges.push(e);
        v = engine.edgeSrc[e];
      }
      return { seconds: d, edges: edges.reverse() };
    }
    for (let e = first[u]; e < first[u + 1]; e++) {
      const w = weight[e];
      if (w === Infinity) continue;
      const v = head[e];
      if (S.isDone(v)) continue;
      const nd = d + w;
      if (nd < S.get(v)) {
        S.set(v, nd, e); // parent holds the edge, so parallel roads stay distinct
        S.heap.push(nd + h(v), v);
      }
    }
  }
  return { seconds: Infinity, edges: [] };
}

function make(engine, s, edges, zone) {
  const g = engine.roads;
  const nodes = [s];
  let seconds = 0;
  let metres = 0;
  let entersZone = false;
  for (const e of edges) {
    nodes.push(g.head[e]);
    seconds += g.weight[e];
    metres += g.edgeLength[e];
  }
  if (zone) entersZone = nodes.some((v) => zone[v]);
  return { nodes, edges, seconds, metres, entersZone };
}

function roadKey(engine, e) {
  const a = engine.edgeSrc[e];
  const b = engine.roads.head[e];
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function overlap(engine, a, b) {
  const roads = new Set(b.edges.map((e) => roadKey(engine, e)));
  let shared = 0;
  for (const e of a.edges) if (roads.has(roadKey(engine, e))) shared += engine.roads.edgeLength[e];
  return a.metres > 0 ? shared / a.metres : 1;
}

function locallySensible(engine, alt, windowS, tolerance, weight, side, samples = 8) {
  const times = [0];
  for (const e of alt.edges) times.push(times[times.length - 1] + weight[e]);
  const total = times[times.length - 1];
  if (total <= windowS) return true;
  const searchRight = (x) => {
    let lo = 0;
    let hi = times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  const searchLeft = (x) => {
    let lo = 0;
    let hi = times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  for (let k = 0; k < samples; k++) {
    const start = ((total - windowS) * k) / Math.max(1, samples - 1);
    const i = searchRight(start) - 1;
    const j = Math.min(searchLeft(times[i] + windowS), alt.nodes.length - 1);
    if (j <= i + 1) continue;
    const best = search(engine, alt.nodes[i], alt.nodes[j], weight, side).seconds;
    if (times[j] - times[i] > tolerance * best + 1e-9) return false;
  }
  return true;
}

/**
 * Up to `count` routes between junctions a and b on the road graph, quickest
 * first. With `avoidZone`, every road into the zone is closed.
 */
export function alternatives(engine, a, b, opts = {}) {
  const { count = 3, penalty = 1.4, maxStretch = 1.3, maxOverlap = 0.65, attempts = 8, zone = null, avoidZone = false, weight = null } = opts;
  const g = engine.roads;
  const side = g.sides[1];
  const base = Float64Array.from(weight ?? g.weight);
  if (avoidZone && zone) {
    if (zone[a] || zone[b]) return { error: "inside" };
    for (let e = 0; e < base.length; e++) if (zone[g.head[e]]) base[e] = Infinity;
  }
  const first = search(engine, a, b, base, side);
  if (!Number.isFinite(first.seconds)) return { routes: [] };
  const found = [make(engine, a, first.edges, zone)];
  const used = new Map();
  const work = Float64Array.from(base);
  for (const e of first.edges) {
    used.set(e, penalty);
    work[e] = base[e] * penalty;
  }
  for (let k = 0; k < attempts && found.length < count; k++) {
    const { edges } = search(engine, a, b, work, side);
    if (!edges.length) break;
    for (const e of edges) {
      const u = (used.get(e) ?? 1) * penalty;
      used.set(e, u);
      work[e] = base[e] * u;
    }
    const cand = make(engine, a, edges, zone);
    if (cand.seconds > maxStretch * found[0].seconds) continue;
    if (found.some((f) => overlap(engine, cand, f) > maxOverlap)) continue;
    if (!locallySensible(engine, cand, 0.25 * found[0].seconds, 1.15, base, side)) continue;
    found.push(cand);
  }
  found.sort((x, y) => x.seconds - y.seconds);
  return { routes: found };
}
