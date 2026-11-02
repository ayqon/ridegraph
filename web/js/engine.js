// Routing engine for the browser, a direct port of the Python package.
//
// The same module runs under Node for the test suite, which checks every
// algorithm here against the Python implementation on identical data.
//
// Searches run on a "graph" object: either the plain road graph (nodes are
// junctions) or the turn-expanded graph (nodes are road segments, edges are
// manoeuvres), which is rebuilt here from the exported turn kinds exactly as
// fastroute/turns.py builds it. Every search records the order in which it
// settles nodes, which is what the map animates.

const DTYPES = { float32: Float32Array, float64: Float64Array, int32: Int32Array, uint8: Uint8Array };
const EARTH_RADIUS_M = 6371008.8;
const ASTAR_SAFETY = 0.9999; // matches fastroute/astar.py
const NO_MIDDLE = -1;
export const TURN = { NONE: 0, STRAIGHT: 1, LEFT: 2, RIGHT: 3, SHARP_LEFT: 4, SHARP_RIGHT: 5, U_TURN: 6 };

export function viewArrays(buffer, layout) {
  const out = {};
  for (const a of layout) out[a.name] = new DTYPES[a.dtype](buffer, a.offset, a.length);
  return out;
}

async function fetchArrayBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status}).`);
  const raw = await res.arrayBuffer();
  const head = new Uint8Array(raw, 0, 2);
  // Some hosts decompress .gz transparently; only gunzip when the bytes are still gzip.
  if (head[0] !== 0x1f || head[1] !== 0x8b) return raw;
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export async function loadEngine(baseUrl) {
  const res = await fetch(`${baseUrl}/manifest.json`);
  if (!res.ok) throw new Error(`Could not load the map data (HTTP ${res.status}).`);
  const manifest = await res.json();
  const graph = await fetchArrayBuffer(`${baseUrl}/${manifest.graph.file}`);
  const engine = new Engine(manifest, graph);
  engine.fetchHierarchy = async (preset) =>
    viewArrays(await fetchArrayBuffer(`${baseUrl}/${preset.file}`), preset.arrays);
  return engine;
}

// Binary min-heap over (key, node) pairs in typed arrays, with lazy deletion.
class Heap {
  constructor(capacity = 1024) {
    this.keys = new Float64Array(capacity);
    this.vals = new Int32Array(capacity);
    this.size = 0;
    this.lastKey = 0;
  }
  clear() {
    this.size = 0;
  }
  minKey() {
    return this.keys[0];
  }
  push(key, val) {
    if (this.size === this.keys.length) {
      const keys = new Float64Array(this.size * 2);
      const vals = new Int32Array(this.size * 2);
      keys.set(this.keys);
      vals.set(this.vals);
      this.keys = keys;
      this.vals = vals;
    }
    const k = this.keys;
    const v = this.vals;
    let i = this.size++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p];
      v[i] = v[p];
      i = p;
    }
    k[i] = key;
    v[i] = val;
  }
  pop() {
    const k = this.keys;
    const v = this.vals;
    const top = v[0];
    this.lastKey = k[0];
    const n = --this.size;
    if (n > 0) {
      const key = k[n];
      const val = v[n];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && k[c + 1] < k[c]) c++;
        if (k[c] >= key) break;
        k[i] = k[c];
        v[i] = v[c];
        i = c;
      }
      k[i] = key;
      v[i] = val;
    }
    return top;
  }
}

// Per-direction search state, reset in O(1) between queries via generation stamps.
class Side {
  constructor(n) {
    this.dist = new Float64Array(n);
    this.parent = new Int32Array(n);
    this.seen = new Int32Array(n);
    this.done = new Int32Array(n);
    this.gen = 0;
    this.heap = new Heap();
  }
  reset() {
    this.gen++;
    this.heap.clear();
  }
  get(v) {
    return this.seen[v] === this.gen ? this.dist[v] : Infinity;
  }
  set(v, d, parent) {
    this.seen[v] = this.gen;
    this.dist[v] = d;
    this.parent[v] = parent;
  }
  isDone(v) {
    return this.done[v] === this.gen;
  }
  markDone(v) {
    this.done[v] = this.gen;
  }
}

// Growable record of settled nodes, in order, with the side that settled them.
class Trace {
  constructor() {
    this.nodes = new Int32Array(4096);
    this.sides = new Uint8Array(4096);
    this.length = 0;
  }
  push(node, side) {
    if (this.length === this.nodes.length) {
      const nodes = new Int32Array(this.length * 2);
      const sides = new Uint8Array(this.length * 2);
      nodes.set(this.nodes);
      sides.set(this.sides);
      this.nodes = nodes;
      this.sides = sides;
    }
    this.nodes[this.length] = node;
    this.sides[this.length++] = side;
  }
  done() {
    return { nodes: this.nodes.subarray(0, this.length), sides: this.sides.subarray(0, this.length) };
  }
}

/** Incoming-edge index for a CSR graph, for searches that run backwards. */
function addReverse(g) {
  const { n, first_out: first, head } = g;
  const m = head.length;
  const firstIn = new Int32Array(n + 1);
  for (let e = 0; e < m; e++) firstIn[head[e] + 1]++;
  for (let i = 0; i < n; i++) firstIn[i + 1] += firstIn[i];
  const fill = firstIn.slice(0, n);
  g.first_in = firstIn;
  g.tail = new Int32Array(m);
  g.inEdge = new Int32Array(m);
  for (let u = 0; u < n; u++) {
    for (let e = first[u]; e < first[u + 1]; e++) {
      const p = fill[head[e]]++;
      g.tail[p] = u;
      g.inEdge[p] = e;
    }
  }
  g.weight = new Float64Array(m);
  g.inWeight = new Float64Array(m);
  g.sides = [new Side(n), new Side(n)];
  g.hCache = new Float64Array(n);
  g.hSeen = new Int32Array(n);
  g.hGen = 0;
  return g;
}

export class Engine {
  constructor(manifest, graphBuffer) {
    this.manifest = manifest;
    Object.assign(this, viewArrays(graphBuffer, manifest.graph.arrays));
    const n = (this.n = manifest.nodes);
    const m = (this.m = manifest.edges);

    this.edgeSrc = new Int32Array(m);
    for (let u = 0; u < n; u++) for (let e = this.first_out[u]; e < this.first_out[u + 1]; e++) this.edgeSrc[e] = u;

    this.roads = addReverse({
      kind: "roads",
      n,
      first_out: this.first_out,
      head: this.head,
      lat: this.lat,
      lon: this.lon,
      edgeLength: this.length_m,
    });
    this.graph = this.roads;
    this.turnGraph = null;
    this.turns = false;
    this.hierarchies = new Map();
    this.ch = null;
    this.setTime(manifest.presets[0]?.hour ?? 3);
  }

  // ---------------------------------------------------------------- traffic

  congestion(hour, weekend = false) {
    const curve = weekend ? this.manifest.traffic.weekend : this.manifest.traffic.weekday;
    const h = ((hour % 24) + 24) % 24;
    const lower = Math.floor(h);
    const frac = h - lower;
    return curve[lower] * (1 - frac) + curve[(lower + 1) % 24] * frac;
  }

  /** Re-weight every edge for a time of day (same model as traffic.py). */
  setTime(hour, weekend = false) {
    const level = this.congestion(hour, weekend);
    const factor = this.manifest.traffic.sensitivity.map((s) => 1 - s * level);
    const w = this.roads.weight;
    for (let e = 0; e < this.m; e++) w[e] = this.freeflow_s[e] / factor[this.road_class[e]];
    this.hour = hour;
    this.weekend = weekend;
    this.finishWeights(this.roads);
    if (this.turnGraph) this.weighTurnGraph();
  }

  finishWeights(g) {
    let maxSpeed = 0;
    const { weight, edgeLength } = g;
    for (let e = 0; e < weight.length; e++) {
      if (weight[e] > 0) maxSpeed = Math.max(maxSpeed, edgeLength[e] / weight[e]);
    }
    for (let p = 0; p < weight.length; p++) g.inWeight[p] = weight[g.inEdge[p]];
    g.maxSpeed = maxSpeed;
  }

  /** Switch to a preset time, loading its Contraction Hierarchy on first use. */
  async usePreset(preset) {
    if (!this.hierarchies.has(preset.id)) {
      this.hierarchies.set(preset.id, await this.fetchHierarchy(preset));
    }
    this.setTime(preset.hour);
    this.ch = this.hierarchies.get(preset.id);
    this.preset = preset;
  }

  // ---------------------------------------------------------------- turns

  /**
   * Build the edge-expanded graph, numbered exactly like fastroute/turns.py:
   * segments 0..m-1 (CSR edge ids), start(x) = m + x, end(x) = m + n + x.
   */
  buildTurnGraph() {
    const { n, m, first_out: first, head, edgeSrc } = this;
    const N = m + 2 * n;
    const degree = (x) => first[x + 1] - first[x];
    // Manoeuvres out of segment i are numbered from mStart[i], in CSR order of head[i].
    const mStart = new Int32Array(m + 1);
    for (let i = 0; i < m; i++) mStart[i + 1] = mStart[i] + degree(head[i]);
    if (mStart[m] !== this.turn_kind.length) throw new Error("turn data does not match the graph");

    const outFirst = new Int32Array(N + 1);
    for (let i = 0; i < m; i++) outFirst[i + 1] = degree(head[i]) + 1;
    for (let x = 0; x < n; x++) outFirst[m + x + 1] = degree(x) + 1;
    for (let v = 0; v < N; v++) outFirst[v + 1] += outFirst[v];
    const E = outFirst[N];
    const to = new Int32Array(E);
    const target = new Int32Array(E); // road segment whose travel time the edge includes, or -1
    const penalty = new Float64Array(E);
    const kind = new Uint8Array(E);
    const { seconds, signals_s: signalWait } = this.manifest.turns;

    let e = 0;
    for (let i = 0; i < m; i++) {
      const via = head[i];
      for (let k = 0; k < degree(via); k++) {
        const j = first[via] + k;
        const turn = this.turn_kind[mStart[i] + k];
        to[e] = j;
        target[e] = j;
        kind[e] = turn;
        penalty[e] = seconds[turn] + (this.signals[via] ? signalWait : 0);
        e++;
      }
      to[e] = m + n + via; // arrive: segment -> end(head)
      target[e++] = -1;
    }
    for (let x = 0; x < n; x++) {
      for (let j = first[x]; j < first[x + 1]; j++) {
        to[e] = j; // depart: start(x) -> segment leaving x
        target[e++] = j;
      }
      to[e] = m + n + x; // trip to the same place
      target[e++] = -1;
    }

    const lat = new Float64Array(N);
    const lon = new Float64Array(N);
    for (let i = 0; i < m; i++) {
      lat[i] = this.lat[head[i]];
      lon[i] = this.lon[head[i]];
    }
    lat.set(this.lat, m);
    lon.set(this.lon, m);
    lat.set(this.lat, m + n);
    lon.set(this.lon, m + n);
    const edgeLength = new Float64Array(E);
    for (let k = 0; k < E; k++) edgeLength[k] = target[k] >= 0 ? this.length_m[target[k]] : 0;

    this.turnGraph = addReverse({
      kind: "turns",
      n: N,
      first_out: outFirst,
      head: to,
      lat,
      lon,
      edgeLength,
      target,
      penalty,
      turnKind: kind,
      mStart,
      edgeSrc,
    });
    this.weighTurnGraph();
  }

  weighTurnGraph() {
    const g = this.turnGraph;
    const w = this.roads.weight;
    for (let e = 0; e < g.weight.length; e++) {
      g.weight[e] = (g.target[e] >= 0 ? w[g.target[e]] : 0) + g.penalty[e];
    }
    this.finishWeights(g);
  }

  setTurns(on) {
    if (on && !this.turnGraph) this.buildTurnGraph();
    this.turns = on;
    this.graph = on ? this.turnGraph : this.roads;
  }

  /** Graph nodes to search between for two junctions. */
  endpoints(a, b) {
    return this.turns ? [this.m + a, this.m + this.n + b] : [a, b];
  }

  /** Junction a graph node stands for. */
  junction(v) {
    if (!this.turns) return v;
    if (v < this.m) return this.head[v];
    return (v - this.m) % this.n;
  }

  // ---------------------------------------------------------------- helpers

  nearest(lat, lon) {
    const c = Math.cos((lat * Math.PI) / 180);
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < this.n; i++) {
      const dx = (this.lon[i] - lon) * c;
      const dy = this.lat[i] - lat;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  static edgeBetween(g, u, v) {
    let best = -1;
    let bestW = Infinity;
    for (let e = g.first_out[u]; e < g.first_out[u + 1]; e++) {
      if (g.head[e] === v && g.weight[e] < bestW) {
        best = e;
        bestW = g.weight[e];
      }
    }
    if (best < 0) throw new Error(`no edge from ${u} to ${v}`);
    return best;
  }

  /** Cheapest road segment between two junctions at the current time. */
  roadBetween(u, v) {
    return Engine.edgeBetween(this.roads, u, v);
  }

  result(algorithm, path, trace, started) {
    const ms = performance.now() - started;
    const t = { ...trace.done(), turns: this.turns };
    const base = { algorithm, settled: t.nodes.length, ms, trace: t, turns: this.turns };
    if (!path) return { ...base, found: false, path: [], junctions: [], seconds: Infinity, metres: 0 };
    const g = this.graph;
    let seconds = 0;
    let metres = 0;
    const manoeuvres = { left: 0, right: 0, sharp: 0, uturn: 0, lights: 0 };
    for (let i = 1; i < path.length; i++) {
      const e = Engine.edgeBetween(g, path[i - 1], path[i]);
      seconds += g.weight[e];
      metres += g.edgeLength[e];
      if (this.turns && path[i - 1] < this.m && path[i] < this.m) {
        const k = g.turnKind[e];
        if (k === TURN.LEFT) manoeuvres.left++;
        else if (k === TURN.RIGHT) manoeuvres.right++;
        else if (k === TURN.SHARP_LEFT || k === TURN.SHARP_RIGHT) manoeuvres.sharp++;
        else if (k === TURN.U_TURN) manoeuvres.uturn++;
        if (this.signals[this.head[path[i - 1]]]) manoeuvres.lights++;
      }
    }
    const junctions = [];
    for (const v of path) {
      const x = this.junction(v);
      if (junctions[junctions.length - 1] !== x) junctions.push(x);
    }
    return { ...base, found: true, path, junctions, seconds, metres, manoeuvres: this.turns ? manoeuvres : null };
  }

  static unwind(parent, from, to) {
    const path = [to];
    let node = to;
    while (node !== from) {
      node = parent[node];
      path.push(node);
    }
    return path.reverse();
  }

  // ---------------------------------------------------------------- searches

  dijkstra(s, t) {
    const started = performance.now();
    const trace = new Trace();
    const g = this.graph;
    const S = g.sides[0];
    S.reset();
    const { first_out: first, head, weight } = g;
    S.set(s, 0, -1);
    S.heap.push(0, s);
    while (S.heap.size) {
      const u = S.heap.pop();
      const d = S.heap.lastKey;
      if (S.isDone(u)) continue;
      S.markDone(u);
      trace.push(u, 0);
      if (u === t) return this.result("dijkstra", Engine.unwind(S.parent, s, t), trace, started);
      for (let e = first[u]; e < first[u + 1]; e++) {
        const v = head[e];
        const nd = d + weight[e];
        if (nd < S.get(v)) {
          S.set(v, nd, u);
          S.heap.push(nd, v);
        }
      }
    }
    return this.result("dijkstra", null, trace, started);
  }

  bidirectional(s, t) {
    const started = performance.now();
    const trace = new Trace();
    if (s === t) {
      trace.push(s, 0);
      return this.result("bidirectional", [s], trace, started);
    }
    const g = this.graph;
    const [F, B] = g.sides;
    F.reset();
    B.reset();
    F.set(s, 0, -1);
    B.set(t, 0, -1);
    F.heap.push(0, s);
    B.heap.push(0, t);
    const adj = [
      [g.first_out, g.head, g.weight],
      [g.first_in, g.tail, g.inWeight],
    ];
    let best = Infinity;
    let meeting = -1;
    while (F.heap.size && B.heap.size) {
      if (F.heap.minKey() + B.heap.minKey() >= best) break;
      const side = F.heap.minKey() <= B.heap.minKey() ? 0 : 1;
      const me = side === 0 ? F : B;
      const them = side === 0 ? B : F;
      const u = me.heap.pop();
      const d = me.heap.lastKey;
      if (me.isDone(u)) continue;
      me.markDone(u);
      trace.push(u, side);
      const [first, nb, w] = adj[side];
      for (let e = first[u]; e < first[u + 1]; e++) {
        const v = nb[e];
        const nd = d + w[e];
        if (nd < me.get(v)) {
          me.set(v, nd, u);
          me.heap.push(nd, v);
        }
        const joined = me.get(v) + them.get(v);
        if (joined < best) {
          best = joined;
          meeting = v;
        }
      }
    }
    if (meeting < 0) return this.result("bidirectional", null, trace, started);
    return this.result("bidirectional", this.joinPaths(g, s, t, meeting), trace, started);
  }

  joinPaths(g, s, t, meeting) {
    const [F, B] = g.sides;
    const path = Engine.unwind(F.parent, s, meeting);
    let node = meeting;
    while (node !== t) {
      node = B.parent[node];
      path.push(node);
    }
    return path;
  }

  astar(s, t) {
    const started = performance.now();
    const trace = new Trace();
    const g = this.graph;
    const S = g.sides[0];
    S.reset();
    const gen = ++g.hGen;
    const { first_out: first, head, weight, lat, lon, hCache, hSeen } = g;
    const scale = g.maxSpeed > 0 ? (ASTAR_SAFETY * 2 * EARTH_RADIUS_M) / g.maxSpeed : 0;
    const rad = Math.PI / 180;
    const tLat = lat[t] * rad;
    const tLon = lon[t] * rad;
    const cosT = Math.cos(tLat);
    const h = (v) => {
      if (hSeen[v] === gen) return hCache[v];
      const vLat = lat[v] * rad;
      const a =
        Math.sin((tLat - vLat) / 2) ** 2 + Math.cos(vLat) * cosT * Math.sin((tLon - lon[v] * rad) / 2) ** 2;
      const value = scale * Math.asin(Math.min(1, Math.sqrt(a)));
      hSeen[v] = gen;
      hCache[v] = value;
      return value;
    };
    S.set(s, 0, -1);
    S.heap.push(h(s), s);
    while (S.heap.size) {
      const u = S.heap.pop();
      if (S.isDone(u)) continue;
      S.markDone(u);
      trace.push(u, 0);
      const d = S.get(u);
      if (u === t) return this.result("astar", Engine.unwind(S.parent, s, t), trace, started);
      for (let e = first[u]; e < first[u + 1]; e++) {
        const v = head[e];
        if (S.isDone(v)) continue;
        const nd = d + weight[e];
        if (nd < S.get(v)) {
          S.set(v, nd, u);
          S.heap.push(nd + h(v), v);
        }
      }
    }
    return this.result("astar", null, trace, started);
  }

  /** Whether Contraction Hierarchies can answer in the current mode. */
  get chAvailable() {
    return Boolean(this.ch) && !this.turns;
  }

  ch_query(s, t) {
    const started = performance.now();
    const trace = new Trace();
    const ch = this.ch;
    if (!this.chAvailable) throw new Error("Contraction Hierarchies are not available in this mode.");
    if (s === t) {
      trace.push(s, 0);
      return this.result("ch", [s], trace, started);
    }
    const g = this.roads;
    const [F, B] = g.sides;
    F.reset();
    B.reset();
    F.set(s, 0, -1);
    B.set(t, 0, -1);
    F.heap.push(0, s);
    B.heap.push(0, t);
    const up = [ch.up_first, ch.up_head, ch.up_weight];
    const down = [ch.down_first, ch.down_tail, ch.down_weight];
    const expand = [up, down];
    const stall = [down, up];
    let best = Infinity;
    let meeting = -1;
    for (;;) {
      // Each side stops on its own: upward cones do not shrink towards each other.
      let side = -1;
      if (F.heap.size && F.heap.minKey() < best) side = 0;
      if (B.heap.size && B.heap.minKey() < best && (side < 0 || B.heap.minKey() < F.heap.minKey())) side = 1;
      if (side < 0) break;
      const me = side === 0 ? F : B;
      const them = side === 0 ? B : F;
      const u = me.heap.pop();
      const d = me.heap.lastKey;
      if (me.isDone(u)) continue;
      me.markDone(u);
      trace.push(u, side);
      const joined = d + them.get(u);
      if (joined < best) {
        best = joined;
        meeting = u;
      }
      // Stall-on-demand: u was reached sub-optimally if a higher node offers a cheaper way in.
      const [sf, sa, sw] = stall[side];
      let stalled = false;
      for (let e = sf[u]; e < sf[u + 1]; e++) {
        if (me.get(sa[e]) + sw[e] < d) {
          stalled = true;
          break;
        }
      }
      if (stalled) continue;
      const [ef, ea, ew] = expand[side];
      for (let e = ef[u]; e < ef[u + 1]; e++) {
        const v = ea[e];
        const nd = d + ew[e];
        if (nd < me.get(v)) {
          me.set(v, nd, u);
          me.heap.push(nd, v);
        }
      }
    }
    if (meeting < 0) return this.result("ch", null, trace, started);
    return this.result("ch", this.unpack(this.joinPaths(g, s, t, meeting)), trace, started);
  }

  middle(a, b) {
    const ch = this.ch;
    const upward = ch.rank[a] < ch.rank[b];
    const first = upward ? ch.up_first : ch.down_first;
    const adj = upward ? ch.up_head : ch.down_tail;
    const w = upward ? ch.up_weight : ch.down_weight;
    const mid = upward ? ch.up_middle : ch.down_middle;
    const at = upward ? a : b;
    const other = upward ? b : a;
    let bestE = -1;
    let bestW = Infinity;
    for (let e = first[at]; e < first[at + 1]; e++) {
      if (adj[e] === other && w[e] < bestW) {
        bestE = e;
        bestW = w[e];
      }
    }
    if (bestE < 0) throw new Error(`no hierarchy edge ${a} -> ${b}`);
    return mid[bestE];
  }

  /** Expand a route over shortcut edges back into real road nodes. */
  unpack(hierarchyPath) {
    const path = [hierarchyPath[0]];
    for (let i = 1; i < hierarchyPath.length; i++) {
      const stack = [[hierarchyPath[i - 1], hierarchyPath[i]]];
      while (stack.length) {
        const [x, y] = stack.pop();
        const m = this.middle(x, y);
        if (m === NO_MIDDLE) path.push(y);
        else stack.push([m, y], [x, m]);
      }
    }
    return path;
  }

  /** Quickest route between two junctions, in the current mode. */
  route(algorithm, a, b) {
    const [s, t] = this.endpoints(a, b);
    switch (algorithm) {
      case "dijkstra":
        return this.dijkstra(s, t);
      case "bidirectional":
        return this.bidirectional(s, t);
      case "astar":
        return this.astar(s, t);
      case "ch":
        return this.ch_query(s, t);
      default:
        throw new Error(`unknown algorithm ${algorithm}`);
    }
  }
}

export const ALGORITHMS = ["dijkstra", "bidirectional", "astar", "ch"];
