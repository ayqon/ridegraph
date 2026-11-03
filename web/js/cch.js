// Customizable Contraction Hierarchies in the browser, ported from fastroute/cch.py.
//
// The exported topology (junction ranks from nested dissection and the upward
// neighbours of every junction after contraction with all fill-in) never
// depends on travel times. `customize` loads the current time of every road,
// live closures included, and sweeps the lower triangles bottom-up, so new
// traffic takes tens of milliseconds instead of a fresh preprocessing run.

import { Side, Trace } from "./engine.js";

export class CustomizableCH {
  constructor(engine, arrays) {
    this.engine = engine;
    this.rank = arrays.rank;
    this.firstUp = arrays.first_up;
    this.upHead = arrays.up_head;
    const n = (this.n = this.rank.length);
    this.m = this.upHead.length;
    this.order = new Int32Array(n);
    for (let v = 0; v < n; v++) this.order[this.rank[v]] = v;
    this.up = new Float64Array(this.m);
    this.down = new Float64Array(this.m);
    this.sides = [new Side(n), new Side(n)];
    this.triC = null;
    this.customizeMs = 0;
  }

  /** CCH edge id of the pair lo-hi (rank[lo] < rank[hi]), or -1. */
  edgeId(lo, hi) {
    const { firstUp, upHead, rank } = this;
    let a = firstUp[lo];
    let b = firstUp[lo + 1] - 1;
    const r = rank[hi];
    while (a <= b) {
      const mid = (a + b) >> 1;
      const rm = rank[upHead[mid]];
      if (rm === r) return mid;
      if (rm < r) a = mid + 1;
      else b = mid - 1;
    }
    return -1;
  }

  /** Metric-independent work done once: triangle list, road-to-edge map, lower neighbours. */
  prepare() {
    if (this.triC) return;
    const { n, firstUp, upHead, order } = this;
    let count = 0;
    for (let v = 0; v < n; v++) {
      const d = firstUp[v + 1] - firstUp[v];
      count += (d * (d - 1)) / 2;
    }
    const triC = new Int32Array(count);
    let t = 0;
    for (let r = 0; r < n; r++) {
      const v = order[r];
      const s = firstUp[v];
      const e = firstUp[v + 1];
      for (let i = s; i < e; i++) {
        const u = upHead[i];
        for (let k = i + 1; k < e; k++) triC[t++] = this.edgeId(u, upHead[k]);
      }
    }
    this.triC = triC;

    const g = this.engine;
    this.roadEdge = new Int32Array(g.m);
    this.roadUp = new Uint8Array(g.m);
    for (let e = 0; e < g.m; e++) {
      const a = g.edgeSrc[e];
      const b = g.head[e];
      if (a === b) {
        this.roadEdge[e] = -1;
        continue;
      }
      const up = this.rank[a] < this.rank[b];
      this.roadEdge[e] = up ? this.edgeId(a, b) : this.edgeId(b, a);
      this.roadUp[e] = up ? 1 : 0;
    }

    // Lower neighbours of each junction, sorted by rank, for unpacking shortcuts.
    const firstDown = new Int32Array(n + 1);
    for (let i = 0; i < this.m; i++) firstDown[upHead[i] + 1]++;
    for (let v = 0; v < n; v++) firstDown[v + 1] += firstDown[v];
    const fill = firstDown.slice(0, n);
    const downTail = new Int32Array(this.m);
    const downEdge = new Int32Array(this.m);
    for (let r = 0; r < n; r++) {
      const v = order[r]; // ascending rank, so each list ends up sorted by rank
      for (let i = firstUp[v]; i < firstUp[v + 1]; i++) {
        const p = fill[upHead[i]]++;
        downTail[p] = v;
        downEdge[p] = i;
      }
    }
    this.firstDown = firstDown;
    this.downTail = downTail;
    this.downEdge = downEdge;
  }

  /** Load the engine's current road times and sweep the lower triangles. */
  customize() {
    this.prepare();
    const started = performance.now();
    const { up, down, triC, firstUp, order, n } = this;
    up.fill(Infinity);
    down.fill(Infinity);
    const w = this.engine.roads.weight;
    for (let e = 0; e < w.length; e++) {
      const id = this.roadEdge[e];
      if (id < 0) continue;
      if (this.roadUp[e]) {
        if (w[e] < up[id]) up[id] = w[e];
      } else if (w[e] < down[id]) down[id] = w[e];
    }
    let t = 0;
    for (let r = 0; r < n; r++) {
      const v = order[r];
      const s = firstUp[v];
      const e = firstUp[v + 1];
      for (let i = s; i < e; i++) {
        const upI = up[i];
        const downI = down[i];
        for (let k = i + 1; k < e; k++) {
          const c = triC[t++];
          const viaUp = downI + up[k]; // u -> v -> w
          if (viaUp < up[c]) up[c] = viaUp;
          const viaDown = down[k] + upI; // w -> v -> u
          if (viaDown < down[c]) down[c] = viaDown;
        }
      }
    }
    this.customizeMs = performance.now() - started;
  }

  cost(x, y) {
    return this.rank[x] < this.rank[y] ? this.up[this.edgeId(x, y)] : this.down[this.edgeId(y, x)];
  }

  directCost(x, y) {
    const g = this.engine.roads;
    let best = Infinity;
    for (let e = g.first_out[x]; e < g.first_out[x + 1]; e++) if (g.head[e] === y && g.weight[e] < best) best = g.weight[e];
    return best;
  }

  middle(x, y, target) {
    if (this.directCost(x, y) <= target * (1 + 1e-12) + 1e-9) return -1;
    const { firstDown, downTail, rank } = this;
    let i = firstDown[x];
    let j = firstDown[y];
    const ie = firstDown[x + 1];
    const je = firstDown[y + 1];
    while (i < ie && j < je) {
      const a = downTail[i];
      const b = downTail[j];
      if (a === b) {
        if (Math.abs(this.cost(x, a) + this.cost(a, y) - target) <= 1e-9 * Math.max(1, target)) return a;
        i++;
        j++;
      } else if (rank[a] < rank[b]) i++;
      else j++;
    }
    return -1;
  }

  unpack(x, y, out) {
    const stack = [[x, y]];
    while (stack.length) {
      const [a, b] = stack.pop();
      const mid = this.middle(a, b, this.cost(a, b));
      if (mid < 0) out.push(b);
      else stack.push([mid, b], [a, mid]);
    }
  }

  /** Quickest route between junctions; returns the engine's usual result object. */
  query(s, t) {
    const started = performance.now();
    const trace = new Trace();
    const engine = this.engine;
    if (s === t) {
      trace.push(s, 0);
      return engine.result("cch", [s], trace, started);
    }
    const [F, B] = this.sides;
    F.reset();
    B.reset();
    F.set(s, 0, -1);
    F.heap.push(0, s);
    B.set(t, 0, -1);
    B.heap.push(0, t);
    let best = Infinity;
    let meet = -1;
    const { firstUp, upHead } = this;
    while (F.heap.size || B.heap.size) {
      for (let side = 0; side < 2; side++) {
        const S = side === 0 ? F : B;
        const O = side === 0 ? B : F;
        if (!S.heap.size) continue;
        if (S.heap.minKey() >= best) {
          S.heap.clear();
          continue;
        }
        const v = S.heap.pop();
        if (S.isDone(v)) continue;
        S.markDone(v);
        trace.push(v, side);
        const d = S.get(v);
        const other = O.get(v);
        if (d + other < best) {
          best = d + other;
          meet = v;
        }
        const cost = side === 0 ? this.up : this.down;
        for (let e = firstUp[v]; e < firstUp[v + 1]; e++) {
          const nd = d + cost[e];
          const w = upHead[e];
          if (nd < S.get(w)) {
            S.set(w, nd, v);
            S.heap.push(nd, w);
          }
        }
      }
    }
    if (meet < 0) return engine.result("cch", null, trace, started);
    const fwd = [meet];
    while (fwd[fwd.length - 1] !== s) fwd.push(F.parent[fwd[fwd.length - 1]]);
    fwd.reverse();
    const bwd = [meet];
    while (bwd[bwd.length - 1] !== t) bwd.push(B.parent[bwd[bwd.length - 1]]);
    const hops = fwd.concat(bwd.slice(1));
    const path = [hops[0]];
    for (let i = 1; i < hops.length; i++) this.unpack(hops[i - 1], hops[i], path);
    return engine.result("cch", path, trace, started);
  }
}
