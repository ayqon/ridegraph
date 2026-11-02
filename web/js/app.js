import { loadEngine, ALGORITHMS } from "./engine.js";
import { Atlas } from "./atlas.js";
import { EtaModel } from "./eta.js";
import { DispatchReplay, PATIENCE_S } from "./dispatch.js";

const FEATURE_NAMES = {
  graph_eta_s: "the router's estimate",
  log_graph_eta: "the router's estimate",
  length_km: "route length",
  crow_km: "straight-line distance",
  detour: "how indirect the route is",
  share_major: "time on main roads",
  share_mid: "time on B roads",
  share_minor: "time on side streets",
  n_left: "left turns",
  n_right: "right turns",
  n_sharp: "sharp turns",
  n_uturn: "U-turns",
  n_lights: "traffic lights",
  n_segments: "number of road segments",
  hour_sin: "time of day",
  hour_cos: "time of day",
  weekend: "weekday or weekend",
  origin_x_km: "where it starts",
  origin_y_km: "where it starts",
  dest_x_km: "where it ends",
  dest_y_km: "where it ends",
  mid_x_km: "where it passes through",
  mid_y_km: "where it passes through",
};

const METHODS = {
  dijkstra: {
    name: "Dijkstra",
    note: "Expands outwards in every direction until it reaches B. Every other method has to find exactly the route it finds.",
  },
  bidirectional: {
    name: "Bidirectional",
    note: "Searches from A and from B at the same time, and stops once neither side can find anything quicker.",
  },
  astar: {
    name: "A*",
    note: "Leans towards B, guided by straight-line distance. In a city that guide is weak: no road gets near a straight line at top speed.",
  },
  ch: {
    name: "Contraction Hierarchies",
    note: "Climbs from each end towards more important junctions, using shortcuts worked out in advance. Production routing engines rely on this.",
  },
};

const LANDMARKS = [
  ["Paddington", 51.5154, -0.1755],
  ["Marble Arch", 51.5131, -0.1589],
  ["Oxford Circus", 51.5152, -0.1419],
  ["Euston", 51.5282, -0.1337],
  ["King's Cross", 51.5308, -0.1238],
  ["Angel", 51.5322, -0.1058],
  ["Holborn", 51.5174, -0.12],
  ["Piccadilly Circus", 51.5101, -0.134],
  ["Trafalgar Square", 51.508, -0.1281],
  ["Westminster", 51.5007, -0.1246],
  ["Waterloo", 51.5033, -0.1145],
  ["Bank", 51.5133, -0.089],
  ["Liverpool Street", 51.5178, -0.0823],
  ["London Bridge", 51.5055, -0.0865],
  ["Elephant & Castle", 51.4946, -0.1005],
  ["Victoria", 51.4952, -0.1441],
  ["Hyde Park Corner", 51.5027, -0.1527],
  ["South Kensington", 51.4941, -0.1738],
  ["Notting Hill Gate", 51.5094, -0.1967],
  ["Vauxhall", 51.4861, -0.1253],
  ["Aldgate", 51.5143, -0.0755],
].map(([name, lat, lon]) => ({ name, lat, lon }));

const DEFAULT_TRIP = { a: [51.5152, -0.1419], b: [51.5033, -0.1145] }; // Oxford Circus to Waterloo
const DEFAULT_PRESET = "0800";

const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const number = new Intl.NumberFormat("en-GB");

const state = { a: null, b: null, method: "ch", preset: null, turns: false, results: null, animation: 0 };
let engine;
let atlas;
let etaModel = null;

function formatEta(seconds) {
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = String(total % 60).padStart(2, "0");
  return m ? `${m}<small>min</small>${s}<small>s</small>` : `${total % 60}<small>s</small>`;
}

function formatMs(ms) {
  if (ms < 0.1) return "under 0.1 ms";
  return `${ms < 10 ? ms.toFixed(2) : ms.toFixed(0)} ms`;
}

function inBounds([lat, lon]) {
  const [s, w, n, e] = engine.manifest.bbox;
  return lat > s && lat < n && lon > w && lon < e;
}

function defaultEnds() {
  if (inBounds(DEFAULT_TRIP.a) && inBounds(DEFAULT_TRIP.b)) {
    return [engine.nearest(...DEFAULT_TRIP.a), engine.nearest(...DEFAULT_TRIP.b)];
  }
  const [s, w, n, e] = engine.manifest.bbox;
  return [
    engine.nearest(s + (n - s) * 0.65, w + (e - w) * 0.25),
    engine.nearest(s + (n - s) * 0.3, w + (e - w) * 0.72),
  ];
}

// ------------------------------------------------------------------ controls

function buildControls() {
  const presets = $("presets");
  for (const p of engine.manifest.presets) {
    const h = Math.floor(p.hour);
    const m = Math.round((p.hour - h) * 60);
    const label = document.createElement("label");
    label.className = "preset";
    label.innerHTML = `<input type="radio" name="preset" value="${p.id}"><span>${p.label}<time>${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}</time></span>`;
    presets.appendChild(label);
  }
  presets.addEventListener("change", async (ev) => {
    await selectPreset(ev.target.value);
    updateHeat();
    run({ animate: true });
  });

  const methods = $("methods");
  for (const id of ALGORITHMS) {
    const label = document.createElement("label");
    label.className = "method";
    label.innerHTML = `<input type="radio" name="method" value="${id}">
      <span class="method-row">
        <span class="method-name">${METHODS[id].name}</span>
        <span class="method-count" data-count="${id}">–</span>
        <span class="method-bar"><i data-bar="${id}" style="width:0"></i></span>
        <span class="method-time" data-time="${id}">&nbsp;</span>
      </span>`;
    methods.appendChild(label);
  }
  methods.addEventListener("change", (ev) => {
    state.method = ev.target.value;
    showMethodNote();
    play();
  });
  document.querySelector(`input[name=method][value=${state.method}]`).checked = true;
  showMethodNote();

  $("replay").addEventListener("click", play);
  $("turns").addEventListener("change", (ev) => {
    state.turns = ev.target.checked;
    engine.setTurns(state.turns);
    if (!state.turns && state.method === "astar" && state.switchedFromCh) {
      state.method = "ch";
      document.querySelector("input[name=method][value=ch]").checked = true;
      showMethodNote();
    }
    state.switchedFromCh = false;
    run({ animate: true });
  });
  $("zoom-in").addEventListener("click", () => atlas.zoomAt(1.6));
  $("zoom-out").addEventListener("click", () => atlas.zoomAt(1 / 1.6));
}

async function selectPreset(id) {
  const preset = engine.manifest.presets.find((p) => p.id === id) ?? engine.manifest.presets[0];
  $("eta").textContent = "Loading…";
  await engine.usePreset(preset);
  state.preset = preset;
  document.querySelector(`input[name=preset][value="${preset.id}"]`).checked = true;
}

function showMethodNote() {
  $("method-note").textContent = METHODS[state.method].note;
}

// ------------------------------------------------------------------ routing

function run({ animate }) {
  if (state.a === null || state.b === null) return;
  const results = {};
  const available = ALGORITHMS.filter((id) => id !== "ch" || engine.chAvailable);
  for (const id of available) results[id] = engine.route(id, state.a, state.b);
  state.results = results;

  const most = Math.max(...available.map((id) => results[id].settled));
  // With turn costs, searches settle road segments rather than junctions.
  $("effort-label").textContent = state.turns ? "road segments examined" : "junctions examined";
  for (const id of ALGORITHMS) {
    const r = results[id];
    const row = document.querySelector(`input[name=method][value=${id}]`);
    row.disabled = !r;
    row.closest(".method").classList.toggle("is-unavailable", !r);
    document.querySelector(`[data-count="${id}"]`).textContent = r ? number.format(r.settled) : "–";
    document.querySelector(`[data-bar="${id}"]`).style.width = r ? `${(100 * r.settled) / most}%` : "0";
    document.querySelector(`[data-time="${id}"]`).textContent = r
      ? `found in ${formatMs(r.ms)}`
      : "needs its own preprocessing with turn costs; try it in the Python tool";
  }

  if (!results[state.method]) {
    // Contraction Hierarchies are off with turn costs; fall back, and return once they are back.
    state.switchedFromCh = true;
    state.method = "astar";
    document.querySelector(`input[name=method][value=astar]`).checked = true;
    showMethodNote();
  }
  const r = results[state.method];
  if (!r.found) {
    $("eta").textContent = "No route";
    $("trip-meta").textContent = "These two points are not connected by road.";
  } else {
    $("eta").innerHTML = formatEta(r.seconds);
    const from = atlas.gridRef(atlas.X[state.a], atlas.Y[state.a]);
    const to = atlas.gridRef(atlas.X[state.b], atlas.Y[state.b]);
    const parts = [`${(r.metres / 1000).toFixed(2)} km`, `${from} to ${to}`, `${state.preset.label.toLowerCase()} traffic`];
    if (r.manoeuvres) parts.push(describeManoeuvres(r.manoeuvres));
    $("trip-meta").textContent = parts.join(" · ");
  }
  showLearnedEta(r);
  $("replay").disabled = false;
  if (animate) play();
  else {
    cancelAnimationFrame(state.animation);
    atlas.setTrace(r.trace);
    atlas.setRoute(r.junctions);
  }
}

/** Animate the chosen method's search, node by node, then draw the route. */
function play() {
  if (!state.results) return;
  const r = state.results[state.method];
  cancelAnimationFrame(state.animation);
  const total = r.trace.nodes.length;
  if (reducedMotion.matches || total === 0) {
    atlas.setTrace(r.trace);
    atlas.setRoute(r.junctions);
    return;
  }
  atlas.setRoute(null);
  const duration = Math.min(2400, Math.max(650, 500 + total * 0.045));
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - (1 - t) ** 2;
    atlas.setTrace(r.trace, Math.max(1, Math.round(total * eased)));
    if (t < 1) state.animation = requestAnimationFrame(step);
    else atlas.setRoute(r.junctions);
  };
  state.animation = requestAnimationFrame(step);
}

function showLearnedEta(r) {
  const learned = etaModel && r.found ? etaModel.predict(r.junctions) : null;
  $("learned").hidden = !learned;
  if (!learned) return;
  $("learned-eta").innerHTML = formatEta(learned.seconds).replaceAll("<small>", " ").replaceAll("</small>", " ").trim();
  const change = Math.round((learned.seconds / r.seconds - 1) * 100);
  const parts = [`${change >= 0 ? "+" : ""}${change}% on the router's estimate`];
  if (learned.low !== null) {
    const lo = Math.max(1, Math.round(learned.low / 60));
    const hi = Math.max(lo, Math.round(learned.high / 60));
    parts.unshift(lo === hi ? `usually about ${lo} min` : `usually ${lo} to ${hi} min`);
  }
  $("learned-delta").textContent = parts.join(" · ");
}

// ------------------------------------------------------------------ learned slowdown layer

function updateHeat() {
  if (!state.heat || !etaModel) {
    atlas.setHeat(null);
    $("key-heat").hidden = true;
    return;
  }
  const cells = atlas.gridCells().map((c) => ({ ...c, value: etaModel.slowdownAt(c.centre.lat, c.centre.lon) }));
  atlas.setHeat(cells);
  const values = cells.map((c) => c.value);
  const pct = (v) => `${v >= 1 ? "+" : ""}${Math.round((v - 1) * 100)}%`;
  $("heat-range").textContent = `Learned slowdown at ${state.preset.label.toLowerCase()}: ${pct(Math.min(...values))} to ${pct(Math.max(...values))} on the router`;
  $("key-heat").hidden = false;
}

function buildModelCard() {
  const info = engine.manifest.eta;
  if (!info) return;
  $("model-card").hidden = false;
  $("model-trips").textContent = number.format(info.trips);
  const rows = Object.entries(info.metrics).map(([name, m]) => {
    const best = name === "Gradient-boosted trees";
    const label = name.replace("Graph ETA (router only)", "Router alone").replace("Graph ETA x global factor", "Router × one factor");
    return `<tr class="${best ? "is-best" : ""}"><td>${label}</td><td>${m.mape_pct.toFixed(1)}%</td><td>${Math.round(m.within_10_pct)}%</td></tr>`;
  });
  $("model-metrics").innerHTML = rows.join("");
  const names = [];
  for (const [feature] of info.importance) {
    const name = FEATURE_NAMES[feature] ?? feature;
    if (!names.includes(name)) names.push(name);
    if (names.length === 4) break;
  }
  $("model-features").textContent = names.join(", ");
}

function describeManoeuvres(m) {
  const count = (n, one, many) => (n ? `${n} ${n === 1 ? one : many}` : null);
  const parts = [
    count(m.right, "right turn", "right turns"),
    count(m.left, "left turn", "left turns"),
    count(m.sharp, "sharp turn", "sharp turns"),
    count(m.uturn, "U-turn", "U-turns"),
    count(m.lights, "set of lights", "sets of lights"),
  ].filter(Boolean);
  return parts.length ? parts.join(", ") : "no turns or lights";
}

function clearTrip(message) {
  cancelAnimationFrame(state.animation);
  state.results = null;
  atlas.setTrace(null);
  atlas.setRoute(null);
  $("eta").textContent = message;
  $("trip-meta").innerHTML = "&nbsp;";
  $("learned").hidden = true;
  $("replay").disabled = true;
  for (const id of ALGORITHMS) {
    document.querySelector(`[data-count="${id}"]`).textContent = "–";
    document.querySelector(`[data-bar="${id}"]`).style.width = "0";
    document.querySelector(`[data-time="${id}"]`).innerHTML = "&nbsp;";
  }
}

// ------------------------------------------------------------------ pins

function placePins() {
  for (const [key, el] of [
    ["a", $("pin-a")],
    ["b", $("pin-b")],
  ]) {
    const node = state[key];
    el.hidden = node === null;
    if (node === null) continue;
    const [x, y] = atlas.nodeScreen(node);
    el.style.transform = `translate(${x}px, ${y}px)`;
  }
}

function bindPins() {
  const sheet = $("sheet");
  for (const key of ["a", "b"]) {
    const el = $(`pin-${key}`);
    let dragging = false;
    let pending = null;
    const moveTo = (sx, sy) => {
      const [x, y] = atlas.toWorld(sx, sy);
      const { lat, lon } = atlas.toLatLon(x, y);
      const node = engine.nearest(lat, lon);
      if (node === state[key]) return;
      state[key] = node;
      placePins();
      run({ animate: false });
    };
    el.addEventListener("pointerdown", (ev) => {
      dragging = true;
      el.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    });
    el.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const box = sheet.getBoundingClientRect();
      // The pin's stem sits below its head; aim the stem, not the cursor.
      pending = [ev.clientX - box.left, ev.clientY - box.top + 8];
      if (!el.dataset.frame) {
        el.dataset.frame = requestAnimationFrame(() => {
          delete el.dataset.frame;
          if (pending) moveTo(...pending);
        });
      }
    });
    const stop = () => {
      dragging = false;
    };
    el.addEventListener("pointerup", stop);
    el.addEventListener("pointercancel", stop);
    el.addEventListener("keydown", (ev) => {
      const step = ev.shiftKey ? 60 : 20;
      const delta = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[ev.key];
      if (!delta || state[key] === null) return;
      ev.preventDefault();
      const [x, y] = atlas.nodeScreen(state[key]);
      moveTo(x + delta[0], y + delta[1]);
    });
  }
}

function onMapClick({ lat, lon }) {
  if (state.mode === "dispatch") return;
  const node = engine.nearest(lat, lon);
  if (state.a !== null && state.b === null) {
    state.b = node;
    placePins();
    run({ animate: true });
    return;
  }
  // Start a new trip: the click becomes A.
  state.a = node;
  state.b = null;
  placePins();
  clearTrip("Now place B");
}

// ------------------------------------------------------------------ rulers

function drawRulers() {
  const { cols, rows } = atlas.gridTicks();
  const dpr = atlas.dpr;
  for (const [id, ticks, horizontal] of [
    ["ruler-top", cols, true],
    ["ruler-left", rows, false],
  ]) {
    const c = $(id);
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#111111";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#ffd21f";
    ctx.strokeStyle = "rgba(255, 210, 31, 0.35)";
    ctx.lineWidth = 1;
    ctx.font = "600 13px 'Barlow Condensed', 'Arial Narrow', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const span = ticks.length ? Math.abs(ticks[0].to - ticks[0].from) : 0;
    const every = span < 18 ? Math.ceil(18 / Math.max(span, 1)) : 1; // thin labels when squares are tiny
    ticks.forEach((t, i) => {
      ctx.beginPath();
      if (horizontal) {
        ctx.moveTo(Math.round(t.from) + 0.5, h * 0.55);
        ctx.lineTo(Math.round(t.from) + 0.5, h);
      } else {
        ctx.moveTo(w * 0.55, Math.round(t.from) + 0.5);
        ctx.lineTo(w, Math.round(t.from) + 0.5);
      }
      ctx.stroke();
      if (i % every) return;
      if (horizontal) ctx.fillText(t.label, t.at, h / 2 + 1);
      else ctx.fillText(t.label, w / 2, t.at + 1);
    });
  }
}


// ------------------------------------------------------------------ dispatch mode

const POLICY_NOTES = {
  nearest: "Riders, oldest first, each take the closest free driver.",
  optimal:
    "Every 30 seconds the Hungarian algorithm assigns the whole batch at once, minimising total pickup time and putting riders who have waited longest first.",
  "optimal+history":
    "Optimal matching, and idle drivers move towards neighbourhoods where riders usually appear at this time, by the historical average.",
  "optimal+forecast":
    "Optimal matching, and idle drivers move to where the ML forecast expects riders in the next 15 minutes, reading what has happened so far this morning.",
};
const SPEEDS = [30, 60, 180];
const CAR_COLOURS = { idle: "#ffffff", pickup: "#1b3a8c", trip: "#ffd21f", reposition: "#9a9a90" };

const sim = { replay: null, policy: 3, t: 0, speed: 60, playing: true, frame: 0, last: 0 };

async function enterDispatch() {
  const info = engine.manifest.dispatch;
  if (!info) return;
  state.mode = "dispatch";
  setModeTabs();
  cancelAnimationFrame(state.animation);
  if (!sim.replay) {
    $("clock").textContent = "Loading…";
    const res = await fetch(`data/${info.file}`);
    if (!res.ok) {
      $("clock").textContent = "Replay didn't load";
      $("dispatch-meta").textContent = "Refresh the page to try again.";
      return;
    }
    const preset = engine.manifest.presets.find((p) => p.id === info.preset);
    await engine.usePreset(preset);
    sim.replay = new DispatchReplay(await res.json(), engine);
    sim.playing = !reducedMotion.matches;
    buildDispatchControls(info);
    updatePlayButton();
  }
  const preset = engine.manifest.presets.find((p) => p.id === info.preset);
  await engine.usePreset(preset);
  engine.setTurns(false);
  atlas.setTrace(null);
  atlas.setRoute(null);
  atlas.setHeat(null);
  atlas.extras = drawDispatch;
  for (const id of ["pin-a", "pin-b", "heat-toggle", "key-route"]) $(id).hidden = true;
  $("key-dispatch").hidden = false;
  $("hint").textContent = "A simulated morning of ride requests. Pick a dispatch method to watch it.";
  sim.last = performance.now();
  sim.frame = requestAnimationFrame(tick);
}

async function enterRoute() {
  state.mode = "route";
  setModeTabs();
  cancelAnimationFrame(sim.frame);
  atlas.extras = null;
  $("key-dispatch").hidden = true;
  $("key-route").hidden = false;
  $("heat-toggle").hidden = !etaModel?.model.reference;
  $("hint").textContent = "Click to place A, then B. Drag a pin to move it.";
  await engine.usePreset(state.preset);
  engine.setTurns(state.turns);
  placePins();
  updateHeat();
  run({ animate: false });
}

function setModeTabs() {
  const dispatch = state.mode === "dispatch";
  $("mode-route").setAttribute("aria-selected", String(!dispatch));
  $("mode-dispatch").setAttribute("aria-selected", String(dispatch));
  $("route-panel").hidden = dispatch;
  $("dispatch-panel").hidden = !dispatch;
}

function buildDispatchControls(info) {
  const r = sim.replay;
  $("scrub").max = String(r.duration);
  $("dispatch-meta").textContent = `Morning peak · ${r.data.drivers} drivers · the same ride requests for every method`;
  const speeds = $("speeds");
  for (const s of SPEEDS) {
    const label = document.createElement("label");
    const each = s >= 60 ? `${s / 60} simulated minute${s > 60 ? "s" : ""}` : `${s} simulated seconds`;
    label.innerHTML = `<input type="radio" name="speed" value="${s}" aria-label="${each} per second" ${s === sim.speed ? "checked" : ""}><span>×${s}</span>`;
    speeds.appendChild(label);
  }
  speeds.addEventListener("change", (ev) => {
    sim.speed = Number(ev.target.value);
  });
  $("play").addEventListener("click", () => {
    if (sim.t >= r.duration) sim.t = 0;
    sim.playing = !sim.playing;
    updatePlayButton();
  });
  $("scrub").addEventListener("input", (ev) => {
    sim.t = Number(ev.target.value);
    drawFrame();
  });

  const box = $("policies");
  r.policies.forEach((p, i) => {
    const label = document.createElement("label");
    label.className = "method";
    label.innerHTML = `<input type="radio" name="policy" value="${i}" ${i === sim.policy ? "checked" : ""}>
      <span class="method-row">
        <span class="method-name">${p.name}</span>
        <span class="method-count" data-wait="${i}">–</span>
        <span class="method-bar"><i data-wait-bar="${i}" style="width:0"></i></span>
        <span class="method-time" data-detail="${i}">&nbsp;</span>
      </span>`;
    box.appendChild(label);
  });
  box.addEventListener("change", (ev) => {
    sim.policy = Number(ev.target.value);
    $("policy-note").textContent = POLICY_NOTES[r.policies[sim.policy].id] ?? "";
    drawFrame();
  });
  $("policy-note").textContent = POLICY_NOTES[r.policies[sim.policy].id] ?? "";

  $("dispatch-days").textContent = String(info.days);
  $("dispatch-summary").innerHTML = r.policies
    .map((p) => {
      const m = info.summary[p.id];
      const best = p.id === "optimal+forecast";
      return `<tr class="${best ? "is-best" : ""}"><td>${p.name}</td><td>${m.mean_wait_min.toFixed(1)} min</td><td>${m.cancelled_pct.toFixed(1)}%</td><td>${m.repositioning_min_per_driver.toFixed(0)} min</td></tr>`;
    })
    .join("");
  $("forecast-summary").innerHTML = Object.entries(info.forecast)
    .map(([name, m]) => `<tr class="${name.startsWith("Gradient") ? "is-best" : ""}"><td>${name}</td><td>${m.deviance.toFixed(2)}</td></tr>`)
    .join("");
}

function updatePlayButton() {
  $("play").textContent = sim.playing ? "Pause" : sim.t >= sim.replay.duration ? "Replay" : "Play";
  $("play").setAttribute("aria-label", $("play").textContent);
}

function tick(now) {
  const dt = Math.min(0.1, (now - sim.last) / 1000);
  sim.last = now;
  const r = sim.replay;
  if (sim.playing && !reducedMotion.matches) {
    sim.t = Math.min(r.duration, sim.t + dt * sim.speed);
    if (sim.t >= r.duration) {
      sim.playing = false;
      updatePlayButton();
    }
  }
  drawFrame();
  if (state.mode === "dispatch") sim.frame = requestAnimationFrame(tick);
}

function drawFrame() {
  const r = sim.replay;
  const clock = r.data.start_hour * 60 + sim.t / 60;
  $("clock").textContent = `${String(Math.floor(clock / 60) % 24).padStart(2, "0")}:${String(Math.floor(clock % 60)).padStart(2, "0")}`;
  $("scrub").value = String(Math.round(sim.t));
  const stats = r.policies.map((_, i) => r.stats(i, sim.t));
  const worst = Math.max(1, ...stats.map((s) => s.meanWaitMin ?? 0));
  stats.forEach((s, i) => {
    document.querySelector(`[data-wait="${i}"]`).textContent = s.meanWaitMin === null ? "–" : `${s.meanWaitMin.toFixed(1)} min`;
    document.querySelector(`[data-wait-bar="${i}"]`).style.width = `${(100 * (s.meanWaitMin ?? 0)) / worst}%`;
    document.querySelector(`[data-detail="${i}"]`).textContent =
      s.meanWaitMin === null
        ? "warming up: counting starts 20 minutes in"
        : `${number.format(s.picked)} picked up · ${s.cancelledPct.toFixed(1)}% gave up`;
  });
  atlas.drawOverlay();
}

function drawDispatch(ctx) {
  const r = sim.replay;
  if (!r) return;
  const riders = r.waiting(sim.policy, sim.t);
  const cars = r.drivers(sim.policy, sim.t, atlas);
  const size = Math.max(5, Math.min(10, atlas.view.scale * 60));

  // Fade the street map a little so the fleet reads first.
  ctx.fillStyle = "rgba(229, 231, 227, 0.55)";
  ctx.fillRect(0, 0, atlas.width, atlas.height);

  // Lines from cars on their way to the rider they will collect.
  ctx.strokeStyle = "rgba(27, 58, 140, 0.55)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  const rides = r.policies[sim.policy].rideList;
  for (const c of cars) {
    if (c.state !== "pickup" || c.ride < 0) continue;
    const [x, y] = atlas.toScreen(...c.xy);
    const [rx, ry] = atlas.nodeScreen(rides[c.ride].origin);
    ctx.moveTo(x, y);
    ctx.lineTo(rx, ry);
  }
  ctx.stroke();

  for (const w of riders) {
    const [x, y] = atlas.nodeScreen(w.node);
    const urgency = Math.min(1, w.waited / PATIENCE_S);
    ctx.beginPath();
    ctx.arc(x, y, 3 + urgency * 4, 0, 2 * Math.PI);
    ctx.fillStyle = `rgba(163, 38, 143, ${(0.15 + 0.25 * urgency).toFixed(2)})`;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    if (w.assigned) {
      ctx.strokeStyle = "#a3268f";
      ctx.lineWidth = 1.5;
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.fillStyle = "#a3268f";
      ctx.fill();
    }
  }

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#111111";
  for (const c of cars) {
    const [x, y] = atlas.toScreen(...c.xy);
    ctx.fillStyle = CAR_COLOURS[c.state];
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
    ctx.strokeRect(x - size / 2, y - size / 2, size, size);
  }
}

// ------------------------------------------------------------------ start

async function main() {
  try {
    engine = await loadEngine("data");
  } catch (err) {
    console.error(err);
    $("eta").textContent = "Map didn't load";
    $("trip-meta").textContent = "The road data couldn't be downloaded. Refresh the page to try again.";
    return;
  }
  $("place").textContent = engine.manifest.name;
  document.title = `fastroute · Quickest routes across ${engine.manifest.name.toLowerCase()}`;

  atlas = new Atlas($("sheet"), engine, { rivers: engine.manifest.rivers ?? [], landmarks: LANDMARKS });
  atlas.on("view", () => {
    placePins();
    drawRulers();
  });
  atlas.on("click", onMapClick);
  buildControls();
  bindPins();
  if (engine.manifest.eta) {
    try {
      const res = await fetch(`data/${engine.manifest.eta.file}`);
      if (res.ok) etaModel = new EtaModel(await res.json(), engine);
      buildModelCard();
      if (etaModel?.model.reference) {
        const toggle = $("heat-toggle");
        toggle.hidden = false;
        toggle.addEventListener("click", () => {
          state.heat = !state.heat;
          toggle.setAttribute("aria-pressed", String(state.heat));
          updateHeat();
        });
      }
    } catch (err) {
      console.warn("The learned ETA model didn't load; showing router ETAs only.", err);
    }
  }

  if (engine.manifest.dispatch) {
    $("mode-dispatch").hidden = false;
    $("mode-dispatch").addEventListener("click", () => state.mode !== "dispatch" && enterDispatch());
    $("mode-route").addEventListener("click", () => state.mode !== "route" && enterRoute());
  }
  state.mode = "route";

  await selectPreset(DEFAULT_PRESET);
  [state.a, state.b] = defaultEnds();
  placePins();
  drawRulers();
  run({ animate: true });
  // On a phone the whole city is too small to read; open on the trip instead.
  if (atlas.width < 700 && state.results) {
    const r = state.results[state.method];
    if (r?.found) atlas.fitNodes(r.junctions, 56);
  }
  document.body.dataset.ready = "true";
  window.fastroute = { engine, atlas, state, sim }; // for debugging and automated checks
}

main();
