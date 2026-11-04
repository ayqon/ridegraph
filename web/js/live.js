// Live road disruptions from Transport for London, ported from ridegraph/live.py.
//
// Fetches TfL's open street-disruption feed, matches each disrupted stretch
// onto the road graph (both ends and the middle of an edge within a few metres
// of the stretch) and turns it into a time multiplier per edge: Infinity for
// a closure, a slowdown by severity otherwise. If the live feed can't be
// reached, the snapshot taken when the map was built is used instead.

export function feedUrl(base, now = new Date(), hours = 1) {
  const fmt = (d) => d.toISOString().slice(0, 19);
  const end = new Date(now.getTime() + hours * 3600e3);
  return `${base}?startDate=${fmt(now)}&endDate=${fmt(end)}`;
}

function points(line) {
  let coords = line;
  if (!Array.isArray(line)) {
    const text = String(line).trim();
    if (/^(MULTI)?LINESTRING/i.test(text)) {
      coords = text
        .slice(text.indexOf("("))
        .replace(/[()]/g, " ")
        .split(",")
        .filter((p) => p.trim())
        .map((p) => p.trim().split(/\s+/).map(Number));
    } else {
      try {
        coords = JSON.parse(text);
      } catch {
        return [];
      }
    }
  }
  while (coords.length && Array.isArray(coords[0]) && Array.isArray(coords[0][0])) coords = coords.flat();
  return coords.filter((p) => Array.isArray(p) && p.length >= 2).map((p) => [Number(p[1]), Number(p[0])]);
}

/** Every disrupted stretch in a feed payload: { street, severity, closure, points: [[lat, lon]] }. */
export function parse(payload) {
  const found = [];
  const walk = (node) => {
    if (Array.isArray(node)) node.forEach(walk);
    else if (node && typeof node === "object") {
      if ("lineString" in node) {
        const pts = points(node.lineString);
        if (pts.length >= 2) {
          found.push({
            street: String(node.streetName || node.location || "a street"),
            severity: String(node.severity || ""),
            closure: String(node.closure || ""),
            points: pts,
          });
        }
        return;
      }
      Object.values(node).forEach(walk);
    }
  };
  walk(payload);
  return found;
}

export function isClosed(d) {
  const c = d.closure.toLowerCase();
  return (c.includes("clos") || c.includes("full")) && !c.includes("partial") && !c.includes("open");
}

export function factorOf(d, info) {
  if (isClosed(d)) return Infinity;
  const f = info.severity_factor[d.severity.toLowerCase()] ?? info.default_factor;
  return d.closure.toLowerCase().includes("partial") ? Math.max(f, 1.6) : f;
}

function distanceToPolyline(px, py, lx, ly) {
  let best = Infinity;
  for (let i = 0; i < lx.length - 1; i++) {
    const dx = lx[i + 1] - lx[i];
    const dy = ly[i + 1] - ly[i];
    const denom = dx * dx + dy * dy;
    const t = denom === 0 ? 0 : Math.min(1, Math.max(0, ((px - lx[i]) * dx + (py - ly[i]) * dy) / denom));
    best = Math.min(best, Math.hypot(px - (lx[i] + t * dx), py - (ly[i] + t * dy)));
  }
  return best;
}

/** Time multiplier per road edge (1 where clear) and the stretches that landed on this map. */
export function match(engine, disruptions, info) {
  const { n, m, lat, lon, head } = engine;
  let lat0 = 0;
  for (let v = 0; v < n; v++) lat0 += lat[v];
  lat0 /= n;
  const kx = 111320 * Math.cos((lat0 * Math.PI) / 180);
  const ky = 110540;
  const nx = new Float64Array(n);
  const ny = new Float64Array(n);
  for (let v = 0; v < n; v++) {
    nx[v] = lon[v] * kx;
    ny[v] = lat[v] * ky;
  }
  // Grid index over junctions.
  const cell = 100;
  const grid = new Map();
  for (let v = 0; v < n; v++) {
    const key = `${Math.floor(nx[v] / cell)},${Math.floor(ny[v] / cell)}`;
    let list = grid.get(key);
    if (!list) grid.set(key, (list = []));
    list.push(v);
  }
  const factor = new Float64Array(m).fill(1);
  const close = new Uint8Array(n);
  const reach = info.match_m;
  const matched = [];
  for (const d of disruptions) {
    const lx = d.points.map((p) => p[1] * kx);
    const ly = d.points.map((p) => p[0] * ky);
    const pad = reach + 1;
    const x0 = Math.floor((Math.min(...lx) - pad) / cell);
    const x1 = Math.floor((Math.max(...lx) + pad) / cell);
    const y0 = Math.floor((Math.min(...ly) - pad) / cell);
    const y1 = Math.floor((Math.max(...ly) + pad) / cell);
    const near = [];
    for (let cx = x0; cx <= x1; cx++) {
      for (let cy = y0; cy <= y1; cy++) {
        for (const v of grid.get(`${cx},${cy}`) ?? []) {
          if (distanceToPolyline(nx[v], ny[v], lx, ly) <= reach) {
            close[v] = 1;
            near.push(v);
          }
        }
      }
    }
    const f = factorOf(d, info);
    let hits = 0;
    for (const u of near) {
      for (let e = engine.first_out[u]; e < engine.first_out[u + 1]; e++) {
        const v = head[e];
        if (v === u || !close[v]) continue;
        if (distanceToPolyline((nx[u] + nx[v]) / 2, (ny[u] + ny[v]) / 2, lx, ly) > reach) continue;
        if (f > factor[e]) factor[e] = f;
        hits++;
      }
    }
    for (const v of near) close[v] = 0;
    if (hits) matched.push({ ...d, closed: isClosed(d), factor: f, edges: hits });
  }
  return { factor, matched };
}

/** Try the live feed, then the snapshot. Resolves to { payload, source, at }. */
export async function loadDisruptions(info, base = "data") {
  try {
    const res = await fetch(feedUrl(info.url), { signal: AbortSignal.timeout(8000) });
    if (res.ok) return { payload: await res.json(), source: "live", at: new Date() };
  } catch {
    // Network blocked or the feed is down; fall back to the snapshot.
  }
  if (!info.snapshot) return null;
  const res = await fetch(`${base}/${info.snapshot}`);
  if (!res.ok) return null;
  return { payload: await res.json(), source: "snapshot", at: info.fetched ? new Date(info.fetched) : null };
}
