// Replays a simulated morning of ride dispatch, exported by fastroute/dispatch.py.
//
// Only the decisions are stored (which driver drove from where to where, and
// when). Each leg's road route is rebuilt here with the same Contraction
// Hierarchy the simulation used, so the cars follow exactly the roads and
// timings the Python simulation priced.

export const PATIENCE_S = 600; // matches DispatchConfig.patience_s
const LEG = { PICKUP: 0, TRIP: 1, REPOSITION: 2, SHARED: 3 };
const STATE = ["pickup", "trip", "reposition", "shared"];

export class DispatchReplay {
  constructor(data, engine) {
    this.data = data;
    this.engine = engine;
    this.duration = data.hours * 3600;
    this.warmup = data.warmup_min * 60;
    this.policies = data.policies.map((p) => this.preparePolicy(p));
    this.paths = new Map();
  }

  preparePolicy(p) {
    const byDriver = Array.from({ length: this.data.drivers }, () => []);
    for (const leg of p.legs) {
      const [driver, kind, t0, from, to, dur, ride] = leg;
      byDriver[driver].push({ kind, t0, from, to, dur, ride });
    }
    for (const legs of byDriver) legs.sort((a, b) => a.t0 - b.t0);
    const rides = p.rides.map(([created, origin, dest, assigned, picked, status, shared = 0, multiplier = 1]) => ({
      created,
      origin,
      dest,
      assigned,
      picked,
      cancelled: status === 1,
      pricedOut: status === 2,
      shared: Boolean(shared),
      multiplier,
    }));
    const surge = (p.surge ?? []).map(([t, cells]) => ({ t, cells: new Map(cells) }));
    return { ...p, byDriver, rideList: rides, surgeFrames: surge };
  }

  /** Road route of a leg as junctions plus cumulative seconds, cached. */
  path(from, to) {
    const key = `${from}>${to}`;
    let p = this.paths.get(key);
    if (!p) {
      const r = from === to ? { path: [from] } : this.engine.ch_query(from, to);
      const nodes = r.path?.length ? r.path : [from, to];
      const times = [0];
      for (let i = 1; i < nodes.length; i++) {
        let w = 0;
        try {
          w = this.engine.roads.weight[this.engine.roadBetween(nodes[i - 1], nodes[i])];
        } catch {
          w = 0;
        }
        times.push(times[i - 1] + w);
      }
      p = { nodes, times };
      this.paths.set(key, p);
    }
    return p;
  }

  /** Where every driver is at time t, and what they are doing. */
  drivers(policyIndex, t, atlas) {
    const pol = this.policies[policyIndex];
    const out = [];
    for (let d = 0; d < pol.byDriver.length; d++) {
      const legs = pol.byDriver[d];
      let state = "idle";
      let node = pol.start_nodes[d];
      let xy = null;
      let ride = -1;
      for (const leg of legs) {
        if (leg.t0 > t) break;
        if (t < leg.t0 + leg.dur) {
          state = STATE[leg.kind] ?? "trip";
          ride = leg.ride;
          xy = this.along(this.path(leg.from, leg.to), t - leg.t0, leg.dur, atlas);
          break;
        }
        node = leg.to;
      }
      if (!xy) xy = [atlas.X[node], atlas.Y[node]];
      out.push({ state, xy, ride });
    }
    return out;
  }

  along(path, elapsed, duration, atlas) {
    const { nodes, times } = path;
    const total = times[times.length - 1] || duration || 1;
    // Scale to the leg's recorded duration (they agree; this guards rounding).
    const at = Math.min(total, (elapsed / (duration || 1)) * total);
    let lo = 0;
    let hi = times.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= at) lo = mid;
      else hi = mid;
    }
    const a = nodes[lo];
    const b = nodes[Math.min(lo + 1, nodes.length - 1)];
    const span = times[Math.min(lo + 1, times.length - 1)] - times[lo] || 1;
    const f = Math.max(0, Math.min(1, (at - times[lo]) / span));
    return [atlas.X[a] + (atlas.X[b] - atlas.X[a]) * f, atlas.Y[a] + (atlas.Y[b] - atlas.Y[a]) * f];
  }

  /** Riders waiting at time t: [{node, waited, assigned}]. */
  waiting(policyIndex, t) {
    const out = [];
    for (const r of this.policies[policyIndex].rideList) {
      if (r.created > t || r.pricedOut) continue;
      if (r.picked >= 0 && t >= r.picked) continue;
      if (r.picked < 0 && t >= r.created + PATIENCE_S) continue;
      out.push({ node: r.origin, waited: t - r.created, assigned: r.assigned >= 0 && t >= r.assigned });
    }
    return out;
  }

  /** Running results up to time t, counting rides after the warm-up. */
  stats(policyIndex, t) {
    let picked = 0;
    let waitSum = 0;
    let cancelled = 0;
    let decided = 0;
    let shared = 0;
    let pricedOut = 0;
    let requests = 0;
    for (const r of this.policies[policyIndex].rideList) {
      if (r.created < this.warmup || r.created > t) continue;
      requests++;
      if (r.pricedOut) {
        pricedOut++;
        continue;
      }
      if (r.picked >= 0 && r.picked <= t) {
        picked++;
        if (r.shared) shared++;
        waitSum += r.picked - r.created;
        decided++;
      } else if (r.picked < 0 && r.created + PATIENCE_S <= t) {
        cancelled++;
        decided++;
      }
    }
    return {
      picked,
      cancelled,
      meanWaitMin: picked ? waitSum / picked / 60 : null,
      cancelledPct: decided ? (100 * cancelled) / decided : null,
      sharedPct: picked ? (100 * shared) / picked : 0,
      pricedOutPct: requests ? (100 * pricedOut) / requests : 0,
    };
  }

  /** Surge multipliers in force at time t: Map of square -> multiplier. */
  surgeAt(policyIndex, t) {
    const frames = this.policies[policyIndex].surgeFrames;
    let current = null;
    for (const f of frames) {
      if (f.t > t) break;
      current = f.cells;
    }
    return current;
  }
}
