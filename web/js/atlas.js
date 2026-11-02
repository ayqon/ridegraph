// A street-atlas renderer for the routing graph.
//
// The map is drawn from the same graph the algorithms search, styled after a
// London A-Z: roads coloured by class, a blue Thames, and a lettered grid so
// any point has a reference like "4G". No tile server is involved, so what
// you see is exactly the network being routed on.

const R = 6378137;
const DEG = Math.PI / 180;
export const GRID_METRES = 500;

// Road buckets, drawn bottom to top. Width is real-world metres, with a
// minimum on-screen width so the network stays legible when zoomed out.
const BUCKETS = [
  { name: "service", classes: [13], fill: "#ffffff", casing: "#d3d4cf", metres: 5, minPx: 0.35, minScale: 0.25 },
  { name: "minor", classes: [10, 11, 12], fill: "#ffffff", casing: "#b9bbb5", metres: 8, minPx: 0.55, minScale: 0 },
  { name: "tertiary", classes: [8, 9], fill: "#fbe58b", casing: "#b49d43", metres: 11, minPx: 1.0, minScale: 0 },
  { name: "secondary", classes: [6, 7], fill: "#f4a524", casing: "#a46705", metres: 13, minPx: 1.4, minScale: 0 },
  { name: "primary", classes: [4, 5], fill: "#e3404a", casing: "#9c1d27", metres: 16, minPx: 1.8, minScale: 0 },
  { name: "trunk", classes: [2, 3], fill: "#3a9d65", casing: "#1c623b", metres: 18, minPx: 2.1, minScale: 0 },
  { name: "motorway", classes: [0, 1], fill: "#3a78d4", casing: "#1d4b95", metres: 22, minPx: 2.4, minScale: 0 },
];
const RIVER = { fill: "#a3cdea", edge: "#76abd4", metres: 230 };
const LAND = "#e5e7e3";
const GRID_LINE = "rgba(17, 17, 17, 0.08)";
const SEARCH_COLOURS = ["#1b3a8c", "#a3268f"];

export function gridLabel(col, row) {
  let letters = "";
  let c = col;
  do {
    letters = String.fromCharCode(65 + (c % 26)) + letters;
    c = Math.floor(c / 26) - 1;
  } while (c >= 0);
  return { col: letters, row: String(row + 1) };
}

export class Atlas {
  constructor(root, engine, { rivers = [], landmarks = [] } = {}) {
    this.root = root;
    this.engine = engine;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);

    const [s, w, n, e] = engine.manifest.bbox;
    this.lat0 = (s + n) / 2;
    this.k = 1 / Math.cos(this.lat0 * DEG); // Mercator stretch at this latitude
    this.origin = this.mercator((s + n) / 2, (w + e) / 2);
    this.bounds = { min: this.local(s, w), max: this.local(n, e) };
    this.gridStep = GRID_METRES * this.k;

    this.X = new Float64Array(engine.n);
    this.Y = new Float64Array(engine.n);
    for (let i = 0; i < engine.n; i++) {
      const [x, y] = this.local(engine.lat[i], engine.lon[i]);
      this.X[i] = x;
      this.Y[i] = y;
    }
    this.buildPaths(rivers);
    this.landmarks = landmarks
      .filter((l) => l.lat > s && l.lat < n && l.lon > w && l.lon < e)
      .map((l) => ({ ...l, xy: this.local(l.lat, l.lon) }));

    this.base = this.canvas("atlas-base");
    this.overlay = this.canvas("atlas-overlay");
    this.cache = document.createElement("canvas");
    this.cachedView = null;
    this.view = { cx: 0, cy: 0, scale: 0.1 };
    this.trace = null;
    this.revealed = 0;
    this.route = null;
    this.listeners = { click: [], view: [] };

    this.resize();
    this.fit();
    this.bindEvents();
    new ResizeObserver(() => {
      this.resize();
      this.invalidate();
    }).observe(root);
  }

  // ------------------------------------------------------------- geometry

  mercator(lat, lon) {
    return [R * lon * DEG, R * Math.log(Math.tan(Math.PI / 4 + (lat * DEG) / 2))];
  }

  local(lat, lon) {
    const [x, y] = this.mercator(lat, lon);
    return [x - (this.origin ? this.origin[0] : 0), y - (this.origin ? this.origin[1] : 0)];
  }

  toLatLon(x, y) {
    const mx = x + this.origin[0];
    const my = y + this.origin[1];
    return { lat: (2 * Math.atan(Math.exp(my / R)) - Math.PI / 2) / DEG, lon: mx / R / DEG };
  }

  toScreen(x, y) {
    const v = this.view;
    return [(x - v.cx) * v.scale + this.width / 2, this.height / 2 - (y - v.cy) * v.scale];
  }

  toWorld(sx, sy) {
    const v = this.view;
    return [(sx - this.width / 2) / v.scale + v.cx, v.cy - (sy - this.height / 2) / v.scale];
  }

  nodeScreen(i) {
    return this.toScreen(this.X[i], this.Y[i]);
  }

  /** Grid reference ("4G") of a world point. */
  gridRef(x, y) {
    const col = Math.floor((x - this.bounds.min[0]) / this.gridStep);
    const row = Math.floor((this.bounds.max[1] - y) / this.gridStep);
    const label = gridLabel(Math.max(0, col), Math.max(0, row));
    return `${label.row}${label.col}`;
  }

  // ------------------------------------------------------------- setup

  canvas(cls) {
    const c = document.createElement("canvas");
    c.className = cls;
    c.setAttribute("aria-hidden", "true");
    this.root.appendChild(c);
    return c;
  }

  buildPaths(rivers) {
    const e = this.engine;
    const byClass = new Map();
    BUCKETS.forEach((b) => b.classes.forEach((c) => byClass.set(c, b.name)));
    this.paths = Object.fromEntries(BUCKETS.map((b) => [b.name, new Path2D()]));
    for (let u = 0; u < e.n; u++) {
      for (let j = e.first_out[u]; j < e.first_out[u + 1]; j++) {
        const v = e.head[j];
        // Two-way streets are two edges; draw each street once.
        if (v < u && this.hasEdge(v, u)) continue;
        const path = this.paths[byClass.get(e.road_class[j])];
        path.moveTo(this.X[u], this.Y[u]);
        path.lineTo(this.X[v], this.Y[v]);
      }
    }
    this.river = new Path2D();
    for (const line of rivers) {
      line.forEach(([lat, lon], i) => {
        const [x, y] = this.local(lat, lon);
        if (i === 0) this.river.moveTo(x, y);
        else this.river.lineTo(x, y);
      });
    }
  }

  hasEdge(u, v) {
    const e = this.engine;
    for (let j = e.first_out[u]; j < e.first_out[u + 1]; j++) if (e.head[j] === v) return true;
    return false;
  }

  resize() {
    const r = this.root.getBoundingClientRect();
    this.width = Math.max(1, r.width);
    this.height = Math.max(1, r.height);
    for (const c of [this.base, this.overlay, this.cache]) {
      c.width = Math.round(this.width * this.dpr);
      c.height = Math.round(this.height * this.dpr);
      if (c !== this.cache) {
        c.style.width = `${this.width}px`;
        c.style.height = `${this.height}px`;
      }
    }
    this.cachedView = null;
  }

  fit(pad = 24) {
    const { min, max } = this.bounds;
    const sx = (this.width - 2 * pad) / (max[0] - min[0]);
    const sy = (this.height - 2 * pad) / (max[1] - min[1]);
    this.view = { cx: (min[0] + max[0]) / 2, cy: (min[1] + max[1]) / 2, scale: Math.min(sx, sy) };
    this.minScale = this.view.scale * 0.8;
    this.maxScale = 12;
    this.invalidate();
  }

  /** Frame a set of junctions, e.g. a trip on a small screen. */
  fitNodes(nodes, pad = 48) {
    if (!nodes.length) return;
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const i of nodes) {
      x0 = Math.min(x0, this.X[i]);
      x1 = Math.max(x1, this.X[i]);
      y0 = Math.min(y0, this.Y[i]);
      y1 = Math.max(y1, this.Y[i]);
    }
    const scale = Math.min((this.width - 2 * pad) / Math.max(x1 - x0, 1), (this.height - 2 * pad) / Math.max(y1 - y0, 1));
    this.view = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: Math.min(this.maxScale, Math.max(this.minScale, scale)) };
    this.invalidate();
  }

  // ------------------------------------------------------------- interaction

  on(event, fn) {
    this.listeners[event].push(fn);
  }

  emit(event, ...args) {
    this.listeners[event].forEach((fn) => fn(...args));
  }

  zoomAt(factor, sx = this.width / 2, sy = this.height / 2) {
    const [wx, wy] = this.toWorld(sx, sy);
    const scale = Math.min(this.maxScale, Math.max(this.minScale, this.view.scale * factor));
    this.view.scale = scale;
    // Keep the world point under the cursor fixed.
    this.view.cx = wx - (sx - this.width / 2) / scale;
    this.view.cy = wy + (sy - this.height / 2) / scale;
    this.clampView();
    this.invalidate();
  }

  panBy(dx, dy) {
    this.view.cx -= dx / this.view.scale;
    this.view.cy += dy / this.view.scale;
    this.clampView();
    this.invalidate();
  }

  clampView() {
    const { min, max } = this.bounds;
    this.view.cx = Math.min(max[0], Math.max(min[0], this.view.cx));
    this.view.cy = Math.min(max[1], Math.max(min[1], this.view.cy));
  }

  bindEvents() {
    const el = this.overlay;
    const pointers = new Map();
    let down = null;
    let pinch = null;

    el.addEventListener("pointerdown", (ev) => {
      el.setPointerCapture(ev.pointerId);
      pointers.set(ev.pointerId, [ev.offsetX, ev.offsetY]);
      if (pointers.size === 1) down = { x: ev.offsetX, y: ev.offsetY, moved: false };
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), scale: this.view.scale };
        down = null;
      }
    });
    el.addEventListener("pointermove", (ev) => {
      if (!pointers.has(ev.pointerId)) return;
      const prev = pointers.get(ev.pointerId);
      pointers.set(ev.pointerId, [ev.offsetX, ev.offsetY]);
      if (pinch && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        const target = (pinch.scale * d) / pinch.d;
        this.zoomAt(target / this.view.scale, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
        return;
      }
      if (down) {
        if (Math.hypot(ev.offsetX - down.x, ev.offsetY - down.y) > 4) down.moved = true;
        if (down.moved) {
          el.classList.add("is-panning");
          this.panBy(ev.offsetX - prev[0], ev.offsetY - prev[1]);
        }
      }
    });
    const end = (ev) => {
      pointers.delete(ev.pointerId);
      el.classList.remove("is-panning");
      if (pointers.size < 2) pinch = null;
      if (down && !down.moved && ev.type === "pointerup") {
        const [x, y] = this.toWorld(ev.offsetX, ev.offsetY);
        this.emit("click", this.toLatLon(x, y));
      }
      down = null;
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        this.zoomAt(Math.exp(-ev.deltaY * (ev.deltaMode ? 0.05 : 0.0015)), ev.offsetX, ev.offsetY);
      },
      { passive: false },
    );
    el.addEventListener("dblclick", (ev) => this.zoomAt(2, ev.offsetX, ev.offsetY));
  }

  // ------------------------------------------------------------- drawing

  invalidate() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.draw();
    });
  }

  draw() {
    this.drawBase();
    this.drawOverlay();
    this.emit("view");
    clearTimeout(this.idle);
    // Interaction redraws the cached map transformed; re-render crisply once it settles.
    if (!this.sameView(this.cachedView, this.view)) this.idle = setTimeout(() => this.renderCache(true), 90);
  }

  sameView(a, b) {
    return a && a.cx === b.cx && a.cy === b.cy && a.scale === b.scale;
  }

  renderCache(redraw) {
    const ctx = this.cache.getContext("2d");
    const v = this.view;
    const s = v.scale * this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = LAND;
    ctx.fillRect(0, 0, this.cache.width, this.cache.height);
    ctx.setTransform(s, 0, 0, -s, this.dpr * (this.width / 2 - v.cx * v.scale), this.dpr * (this.height / 2 + v.cy * v.scale));
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Grid squares, as on an atlas page.
    const { min, max } = this.bounds;
    ctx.strokeStyle = GRID_LINE;
    ctx.lineWidth = 1 / v.scale;
    ctx.beginPath();
    for (let x = min[0]; x <= max[0] + 1; x += this.gridStep) {
      ctx.moveTo(x, min[1]);
      ctx.lineTo(x, max[1]);
    }
    for (let y = max[1]; y >= min[1] - 1; y -= this.gridStep) {
      ctx.moveTo(min[0], y);
      ctx.lineTo(max[0], y);
    }
    ctx.stroke();

    const px = (metres, minPx) => Math.max(minPx, metres * this.k * v.scale) / v.scale;
    ctx.strokeStyle = RIVER.edge;
    ctx.lineWidth = px(RIVER.metres, 6) + 2 / v.scale;
    ctx.stroke(this.river);
    ctx.strokeStyle = RIVER.fill;
    ctx.lineWidth = px(RIVER.metres, 6);
    ctx.stroke(this.river);

    const zoomedOut = v.scale < 0.35;
    for (const b of BUCKETS) {
      if (v.scale < b.minScale) continue;
      const w = px(b.metres, b.minPx);
      if (zoomedOut && (b.name === "minor" || b.name === "service")) {
        ctx.strokeStyle = b.casing;
        ctx.lineWidth = w;
        ctx.stroke(this.paths[b.name]);
        continue;
      }
      ctx.strokeStyle = b.casing;
      ctx.lineWidth = w + 1.6 / v.scale;
      ctx.stroke(this.paths[b.name]);
    }
    for (const b of BUCKETS) {
      if (v.scale < b.minScale || (zoomedOut && (b.name === "minor" || b.name === "service"))) continue;
      ctx.strokeStyle = b.fill;
      ctx.lineWidth = px(b.metres, b.minPx);
      ctx.stroke(this.paths[b.name]);
    }
    this.cachedView = { ...v };
    if (redraw) this.drawBase();
  }

  drawBase() {
    if (!this.cachedView) this.renderCache(false);
    const ctx = this.base.getContext("2d");
    const c = this.cachedView;
    const v = this.view;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = LAND;
    ctx.fillRect(0, 0, this.base.width, this.base.height);
    // Map the cached frame into the current view.
    const r = v.scale / c.scale;
    const dx = ((c.cx - v.cx) * v.scale + (this.width / 2) * (1 - r)) * this.dpr;
    const dy = (-(c.cy - v.cy) * v.scale + (this.height / 2) * (1 - r)) * this.dpr;
    ctx.setTransform(r, 0, 0, r, dx, dy);
    ctx.drawImage(this.cache, 0, 0);
  }

  setTrace(trace, revealed = trace ? trace.nodes.length : 0) {
    this.trace = trace;
    this.revealed = revealed;
    this.drawOverlay();
  }

  /** Grid squares of the map, in world coordinates, with their centres as lat/lon. */
  gridCells() {
    const { min, max } = this.bounds;
    const cells = [];
    for (let x = min[0]; x < max[0]; x += this.gridStep) {
      for (let y = max[1]; y > min[1]; y -= this.gridStep) {
        const x1 = Math.min(x + this.gridStep, max[0]);
        const y0 = Math.max(y - this.gridStep, min[1]);
        cells.push({ x0: x, y0, x1, y1: y, centre: this.toLatLon((x + x1) / 2, (y0 + y) / 2) });
      }
    }
    return cells;
  }

  /** Shade grid squares by value (null clears). Darker means slower. */
  setHeat(cells) {
    this.heat = cells;
    this.drawOverlay();
  }

  drawHeat(ctx) {
    if (!this.heat || !this.heat.length) return;
    const values = this.heat.map((c) => c.value);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    for (const c of this.heat) {
      const t = hi > lo ? (c.value - lo) / (hi - lo) : 0.5;
      const [ax, ay] = this.toScreen(c.x0, c.y1);
      const [bx, by] = this.toScreen(c.x1, c.y0);
      ctx.fillStyle = `rgba(17, 17, 17, ${(0.04 + 0.5 * t ** 1.4).toFixed(3)})`;
      ctx.fillRect(ax, ay, bx - ax, by - ay);
    }
  }

  setRoute(path) {
    this.route = path;
    this.drawOverlay();
  }

  /**
   * Paint the search onto its own layer. Large searches light up the streets
   * leaving each settled junction, so the search visibly floods the network;
   * small ones (Contraction Hierarchies) are shown as dots. Painting is
   * incremental while the view stays put, so animating 70,000 junctions stays cheap.
   */
  paintTrace() {
    const layer = (this.traceLayer ??= document.createElement("canvas"));
    const t = this.trace;
    const viewKey = `${this.view.cx},${this.view.cy},${this.view.scale},${this.width},${this.height}`;
    const fresh =
      layer.width !== this.overlay.width ||
      layer.height !== this.overlay.height ||
      this.painted?.trace !== t ||
      this.painted?.view !== viewKey ||
      this.painted.count > this.revealed;
    if (fresh) {
      layer.width = this.overlay.width;
      layer.height = this.overlay.height;
      this.painted = { trace: t, view: viewKey, count: 0 };
    }
    if (!t || this.painted.count >= this.revealed) return layer;

    const ctx = layer.getContext("2d");
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const { nodes, sides } = t;
    const from = this.painted.count;
    const to = this.revealed;
    const asDots = nodes.length < 2000 && !t.turns;
    const e = this.engine;
    for (let side = 0; side < 2; side++) {
      ctx.beginPath();
      for (let i = from; i < to; i++) {
        if (sides[i] !== side) continue;
        const u = nodes[i];
        if (t.turns) {
          // With turn costs the search settles road segments, so paint the segment.
          if (u >= e.m) continue;
          const [ax, ay] = this.nodeScreen(e.edgeSrc[u]);
          const [bx, by] = this.nodeScreen(e.head[u]);
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          continue;
        }
        const [x, y] = this.nodeScreen(u);
        if (asDots) {
          ctx.moveTo(x + 3.5, y);
          ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
          continue;
        }
        for (let j = e.first_out[u]; j < e.first_out[u + 1]; j++) {
          const [vx, vy] = this.nodeScreen(e.head[j]);
          ctx.moveTo(x, y);
          // Half the edge from each end: settled from both sides it joins up.
          ctx.lineTo((x + vx) / 2, (y + vy) / 2);
        }
      }
      if (asDots) {
        ctx.fillStyle = SEARCH_COLOURS[side];
        ctx.globalAlpha = 0.9;
        ctx.fill();
      } else {
        ctx.strokeStyle = SEARCH_COLOURS[side];
        ctx.globalAlpha = 0.85;
        ctx.lineCap = "round";
        ctx.lineWidth = Math.max(2.6, Math.min(6, 16 * this.k * this.view.scale));
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    this.painted.count = to;
    return layer;
  }

  drawOverlay() {
    const ctx = this.overlay.getContext("2d");
    const d = this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    ctx.setTransform(d, 0, 0, d, 0, 0);
    this.drawHeat(ctx);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.trace && this.revealed) ctx.drawImage(this.paintTrace(), 0, 0);
    ctx.setTransform(d, 0, 0, d, 0, 0);
    this.drawLandmarks(ctx);

    if (this.route && this.route.length > 1) {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      this.route.forEach((node, i) => {
        const [x, y] = this.nodeScreen(node);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 8;
      ctx.stroke();
      ctx.strokeStyle = "#ffd21f";
      ctx.lineWidth = 3.5;
      ctx.stroke();
    }
    if (this.extras) this.extras(ctx);
  }

  drawLandmarks(ctx) {
    if (this.view.scale < 0.03) return;
    const size = this.view.scale < 0.09 ? 10.5 : 12;
    ctx.font = `600 ${size}px 'Barlow Condensed', 'Arial Narrow', sans-serif`;
    ctx.textBaseline = "middle";
    for (const l of this.landmarks) {
      const [x, y] = this.toScreen(...l.xy);
      if (x < 0 || y < 0 || x > this.width || y > this.height) continue;
      const text = l.name.toUpperCase();
      ctx.fillStyle = "#111111";
      ctx.fillRect(x - 2.5, y - 2.5, 5, 5);
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.strokeText(text, x + 6, y);
      ctx.fillText(text, x + 6, y);
    }
  }

  /** Grid ticks visible in the current view, for the rulers. */
  gridTicks() {
    const { min, max } = this.bounds;
    const cols = [];
    const rows = [];
    for (let c = 0, x = min[0]; x < max[0]; c++, x += this.gridStep) {
      const a = this.toScreen(x, 0)[0];
      const b = this.toScreen(Math.min(x + this.gridStep, max[0]), 0)[0];
      if (b > 0 && a < this.width) cols.push({ label: gridLabel(c, 0).col, at: (a + b) / 2, from: a, to: b });
    }
    for (let r = 0, y = max[1]; y > min[1]; r++, y -= this.gridStep) {
      const a = this.toScreen(0, y)[1];
      const b = this.toScreen(0, Math.max(y - this.gridStep, min[1]))[1];
      if (b > 0 && a < this.height) rows.push({ label: String(r + 1), at: (a + b) / 2, from: a, to: b });
    }
    return { cols, rows };
  }
}
