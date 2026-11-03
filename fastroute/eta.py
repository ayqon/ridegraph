"""Learned ETA: correcting the router's estimate from how trips actually go.

Production ride-hailing systems do not trust the routing engine's ETA as is.
The graph knows road speeds, but not the queue that always forms at a
particular junction, how long lights really hold you at 8:30, or that the
evening peak arrives later than the timetable says. The approach popularised
by Uber's DeepETA keeps the routing engine and trains a model to predict the
*residual*: the gap between the engine's ETA and the trip time actually
observed, from features of the route and the moment.

Real trip logs for London are not public, so this module also provides the
observations: ``World`` is a hidden ground truth, a city that differs from the
router's model in ways a real one would. It has congestion hotspots, a later
and stronger peak, extra slowdown on main roads, waits at turns and lights,
occasional incidents and noise. Trips are routed with the router, timed by
the world, and models learn the correction from route features alone. They
are judged on trips they never saw.

Everything a feature uses is something a production system would know at
request time: the route, its geometry and road mix, its manoeuvres and the
time. Nothing reads the hidden world.

The model runs in the browser too: the gradient-boosted trees export to JSON
and ``web/js/eta.js`` computes the same features and walks the same trees.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .ch import ContractionHierarchy
from .graph import Graph, RoadNetwork, csr_sorted, haversine_m
from .router import Router, Scenario
from .traffic import DEFAULT_PROFILE
from .turns import Turn, TurnGraph

FEATURES: tuple[str, ...] = (
    "graph_eta_s",
    "log_graph_eta",
    "length_km",
    "crow_km",
    "detour",
    "share_major",
    "share_mid",
    "share_minor",
    "n_left",
    "n_right",
    "n_sharp",
    "n_uturn",
    "n_lights",
    "n_segments",
    "hour_sin",
    "hour_cos",
    "weekend",
    "origin_x_km",
    "origin_y_km",
    "dest_x_km",
    "dest_y_km",
    "mid_x_km",
    "mid_y_km",
)

# Road-class groups by index into ROAD_CLASSES: motorway..primary_link,
# secondary..tertiary_link, everything smaller.
MAJOR, MID = range(0, 6), range(6, 10)
KM_PER_DEG_LAT = 111.32
LIGHT_SET_GAP_M = 60.0


def class_group(road_class: np.ndarray) -> np.ndarray:
    """0 for major roads, 1 for secondary/tertiary, 2 for minor roads."""
    return np.where(road_class < 6, 0, np.where(road_class < 10, 1, 2))


class RouteContext:
    """Everything needed to describe a route, built once per network.

    The network is put in CSR order so edge ids match ``Graph`` and the turn
    graph, exactly as in the browser export.
    """

    def __init__(self, network: RoadNetwork) -> None:
        self.network = network = csr_sorted(network)
        self.graph = Graph(network)
        self.freeflow_s = np.array(self.graph.weight)
        self.road_class = network.road_class.astype(np.int64)
        self.group = class_group(self.road_class)
        self.length_m = network.length_m.astype(np.float64)
        turns = TurnGraph(network)
        self.turn_kind = turns.turn_kind[: turns.num_manoeuvres]
        degree = np.diff(np.array(self.graph.first_out))
        self.m_start = np.concatenate([[0], np.cumsum(degree[network.dst])])
        self.signals = np.asarray(network.traffic_signals, dtype=bool)
        self.lat0 = float((network.lat.min() + network.lat.max()) / 2)
        self.lon0 = float((network.lon.min() + network.lon.max()) / 2)
        self.km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(self.lat0))

    def edge_times(self, hour: float, weekend: bool) -> np.ndarray:
        """Router's seconds per edge at a time of day (same model as ``traffic.py``)."""
        factors = DEFAULT_PROFILE.speed_factors(hour, weekend=weekend)
        return self.freeflow_s / factors[self.road_class]

    def path_edges(self, path: list[int], times: np.ndarray) -> list[int]:
        """Cheapest edge between each consecutive pair of junctions."""
        first, head = self.graph.first_out, self.graph.head
        edges = []
        for u, v in pairwise(path):
            best, best_t = -1, math.inf
            for e in range(first[u], first[u + 1]):
                if head[e] == v and times[e] < best_t:
                    best, best_t = e, times[e]
            if best < 0:
                raise KeyError(f"no edge {u} -> {v}")
            edges.append(best)
        return edges

    def xy_km(self, node: int) -> tuple[float, float]:
        net = self.network
        return (
            (float(net.lon[node]) - self.lon0) * self.km_per_deg_lon,
            (float(net.lat[node]) - self.lat0) * KM_PER_DEG_LAT,
        )

    def turns_along(self, edges: list[int]) -> list[int]:
        """Turn kind of each manoeuvre between consecutive edges of a route."""
        first, head = self.graph.first_out, self.graph.head
        return [
            int(self.turn_kind[self.m_start[a] + (b - first[head[a]])]) for a, b in pairwise(edges)
        ]

    def light_sets(self, path: list[int], edges: list[int]) -> int:
        """Sets of traffic lights a route passes.

        OpenStreetMap often tags lights on each approach to a junction and on
        nearby crossings, so signal nodes closer than ``LIGHT_SET_GAP_M`` along
        the route count as one set.
        """
        count = 0
        since = math.inf
        for i in range(1, len(path) - 1):
            since += float(self.length_m[edges[i - 1]])
            if self.signals[path[i]] and since >= LIGHT_SET_GAP_M:
                count += 1
                since = 0.0
        return count

    def features(self, path: list[int], hour: float, weekend: bool) -> np.ndarray:
        times = self.edge_times(hour, weekend)
        edges = self.path_edges(path, times)
        return self.features_from_edges(path, edges, times, hour, weekend)

    def features_from_edges(
        self, path: list[int], edges: list[int], times: np.ndarray, hour: float, weekend: bool
    ) -> np.ndarray:
        # Plain left-to-right sums, so the browser reproduces these bit for bit.
        eta = 0.0
        length = 0.0
        by_group = [0.0, 0.0, 0.0]
        for edge in edges:
            eta += float(times[edge])
            seg = float(self.length_m[edge])
            length += seg
            by_group[int(self.group[edge])] += seg
        net = self.network
        a, b = path[0], path[-1]
        crow = haversine_m(
            float(net.lat[a]), float(net.lon[a]), float(net.lat[b]), float(net.lon[b])
        )
        kinds = self.turns_along(edges)
        lights = self.light_sets(path, edges)
        angle = 2 * math.pi * hour / 24
        ox, oy = self.xy_km(a)
        dx, dy = self.xy_km(b)
        mx, my = self.xy_km(path[len(path) // 2])
        return np.array(
            [
                eta,
                math.log(max(eta, 1.0)),
                length / 1000,
                crow / 1000,
                length / max(crow, 1.0),
                by_group[0] / max(length, 1.0),
                by_group[1] / max(length, 1.0),
                by_group[2] / max(length, 1.0),
                kinds.count(Turn.LEFT),
                kinds.count(Turn.RIGHT),
                kinds.count(Turn.SHARP_LEFT) + kinds.count(Turn.SHARP_RIGHT),
                kinds.count(Turn.U_TURN),
                lights,
                len(edges),
                math.sin(angle),
                math.cos(angle),
                float(weekend),
                ox,
                oy,
                dx,
                dy,
                mx,
                my,
            ],
            dtype=np.float64,
        )


@dataclass
class World:
    """The hidden ground truth that trip times are drawn from.

    None of this is visible to the router or to the features; the models must
    infer its effects from routes and outcomes.
    """

    ctx: RouteContext
    seed: int = 0
    hotspots: int = 7
    peak_shift_h: float = 0.5  # real peaks arrive this much later than the router assumes
    peak_gain: float = 1.1
    group_extra: tuple[float, float, float] = (0.10, 0.05, 0.0)  # extra peak slowdown per group
    turn_wait_s: dict[int, float] = field(
        default_factory=lambda: {
            Turn.LEFT: 2.0,
            Turn.RIGHT: 6.0,
            Turn.SHARP_LEFT: 6.0,
            Turn.SHARP_RIGHT: 10.0,
            Turn.U_TURN: 25.0,
        }
    )
    incident_rate: float = 0.03
    noise_sigma: float = 0.06

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        net = self.ctx.network
        major = np.flatnonzero(self.ctx.group == 0)
        pool = major if len(major) >= self.hotspots else np.arange(net.num_edges)
        centres = net.src[rng.choice(pool, size=self.hotspots, replace=False)]
        # Edge midpoints in km, and a smooth congestion field around each hotspot.
        mid_lat = (net.lat[net.src] + net.lat[net.dst]) / 2
        mid_lon = (net.lon[net.src] + net.lon[net.dst]) / 2
        ex = (mid_lon - self.ctx.lon0) * self.ctx.km_per_deg_lon
        ey = (mid_lat - self.ctx.lat0) * KM_PER_DEG_LAT
        field_ = np.zeros(net.num_edges)
        self.spots: list[tuple[float, float, float, float]] = []
        for c in centres:
            cx = (float(net.lon[c]) - self.ctx.lon0) * self.ctx.km_per_deg_lon
            cy = (float(net.lat[c]) - self.ctx.lat0) * KM_PER_DEG_LAT
            radius = rng.uniform(0.4, 1.1)
            strength = rng.uniform(0.15, 0.45)
            self.spots.append((cx, cy, radius, strength))
            field_ += strength * np.exp(-((ex - cx) ** 2 + (ey - cy) ** 2) / (2 * radius**2))
        self.hotspot_field = field_
        self.group_extra_arr = np.asarray(self.group_extra)[self.ctx.group]

    def field_at(self, x_km: float, y_km: float) -> float:
        """Hidden hotspot intensity at a point (km from the network's centre)."""
        return sum(
            s * math.exp(-((x_km - cx) ** 2 + (y_km - cy) ** 2) / (2 * r**2))
            for cx, cy, r, s in self.spots
        )

    def level(self, hour: float, weekend: bool) -> float:
        return min(
            1.2,
            self.peak_gain * DEFAULT_PROFILE.congestion(hour - self.peak_shift_h, weekend=weekend),
        )

    def trip_seconds(
        self,
        path: list[int],
        edges: list[int],
        router_times: np.ndarray,
        hour: float,
        weekend: bool,
        rng: np.random.Generator,
    ) -> float:
        e = np.asarray(edges, dtype=np.int64)
        level = self.level(hour, weekend)
        driving = router_times[e] * (1 + (self.hotspot_field[e] + self.group_extra_arr[e]) * level)
        total = float(driving.sum())
        for kind in self.ctx.turns_along(edges):
            wait = self.turn_wait_s.get(kind, 0.0)
            if wait:
                total += wait * rng.uniform(0.5, 1.5)
        # Each set of lights: stopped more often at peak, and for longer.
        sets = self.ctx.light_sets(path, edges)
        if sets:
            stopped = rng.random(sets) < 0.25 + 0.3 * level
            waits = rng.uniform(5, 20 + 15 * level, size=sets)
            total += float(waits[stopped].sum())
        if rng.random() < self.incident_rate:
            total *= rng.uniform(1.2, 1.8)
        return total * float(np.exp(rng.normal(0, self.noise_sigma)))


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray  # log(observed / graph ETA)
    graph_eta: np.ndarray
    observed: np.ndarray
    # Per trip, for the graph neural network: its road segments and their
    # router seconds, turn and light counts, and the time of day.
    edges: list[np.ndarray] = field(default_factory=list)
    seconds: list[np.ndarray] = field(default_factory=list)
    counts: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    tau: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))


def nearest_preset(hour: float, presets: dict[float, ContractionHierarchy]) -> ContractionHierarchy:
    def gap(h: float) -> float:
        d = abs(hour - h) % 24
        return min(d, 24 - d)

    return presets[min(presets, key=gap)]


def simulate_trips(
    ctx: RouteContext,
    world: World,
    hierarchies: dict[float, ContractionHierarchy],
    count: int,
    *,
    seed: int = 1,
    min_crow_km: float = 0.4,
    log: Callable[[str], None] = lambda msg: None,
) -> Dataset:
    """Route random trips with the router and time them in the hidden world.

    Trips are routed with the hierarchy of the nearest preset time, like an app
    whose routes are refreshed a few times a day, then timed at their own hour.
    """
    rng = np.random.default_rng(seed)
    net = ctx.network
    n = net.num_nodes
    rows: list[np.ndarray] = []
    ys: list[float] = []
    etas: list[float] = []
    observed: list[float] = []
    all_edges, all_secs, counts, hours, weekends = [], [], [], [], []
    started = time.perf_counter()
    attempts = 0
    while len(rows) < count and attempts < count * 5:
        attempts += 1
        a, b = int(rng.integers(n)), int(rng.integers(n))
        crow = haversine_m(
            float(net.lat[a]), float(net.lon[a]), float(net.lat[b]), float(net.lon[b])
        )
        if crow < min_crow_km * 1000:
            continue
        hour = float(rng.uniform(6, 22) if rng.random() < 0.75 else rng.uniform(0, 24))
        weekend = bool(rng.random() < 2 / 7)
        route = nearest_preset(hour, hierarchies).query(a, b)
        if not route.found or len(route.path) < 2:
            continue
        times = ctx.edge_times(hour, weekend)
        edges = ctx.path_edges(route.path, times)
        x = ctx.features_from_edges(route.path, edges, times, hour, weekend)
        truth = world.trip_seconds(route.path, edges, times, hour, weekend, rng)
        rows.append(x)
        e = np.asarray(edges, dtype=np.int64)
        all_edges.append(e)
        all_secs.append(times[e])
        counts.append([x[8], x[9], x[10], x[11], x[12]])
        hours.append(hour)
        weekends.append(weekend)
        etas.append(x[0])
        observed.append(truth)
        ys.append(math.log(truth / x[0]))
        if len(rows) % 5000 == 0:
            log(f"    {len(rows):,} trips simulated ({time.perf_counter() - started:.0f}s)")
    from .gnn import time_features

    hours_a, weekend_a = np.array(hours), np.array(weekends)
    level = np.array(
        [DEFAULT_PROFILE.congestion(h, weekend=w) for h, w in zip(hours, weekends, strict=True)]
    )
    return Dataset(
        np.array(rows),
        np.array(ys),
        np.array(etas),
        np.array(observed),
        all_edges,
        all_secs,
        np.array(counts, dtype=np.float64).reshape(-1, 5),
        time_features(hours_a, weekend_a, level),
    )


def error_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    err = predicted - observed
    pct = np.abs(err) / observed
    return {
        "mae_s": float(np.mean(np.abs(err))),
        "mape_pct": float(100 * np.mean(pct)),
        "p90_pct": float(100 * np.quantile(pct, 0.9)),
        "within_10_pct": float(100 * np.mean(pct <= 0.10)),
    }


@dataclass
class EtaResult:
    metrics: dict[str, dict[str, float]]
    importance: list[tuple[str, float]]
    model: object
    trips: int
    model_json: dict[str, object]
    interval: dict[str, float] = field(default_factory=dict)
    quantile_models: dict[str, object] = field(default_factory=dict)


def export_trees(model: object, features: tuple[str, ...] = FEATURES) -> dict[str, object]:
    """Gradient-boosted trees as compact arrays the browser can walk."""
    trees = []
    for (predictor,) in model._predictors:  # type: ignore[attr-defined]
        nodes = predictor.nodes
        trees.append(
            {
                "feature": nodes["feature_idx"].astype(int).tolist(),
                "threshold": [float(t) for t in nodes["num_threshold"]],
                "left": nodes["left"].astype(int).tolist(),
                "right": nodes["right"].astype(int).tolist(),
                "leaf": nodes["is_leaf"].astype(int).tolist(),
                "value": [float(v) for v in nodes["value"]],
            }
        )
    return {
        "kind": "gradient-boosted trees predicting log(observed / graph ETA)",
        "features": list(features),
        "baseline": float(np.ravel(model._baseline_prediction)[0]),  # type: ignore[attr-defined]
        "trees": trees,
    }


def predict_from_json(model_json: dict[str, object], x: np.ndarray) -> float:
    """Reference implementation of the browser's tree walk, used in tests."""
    total = float(model_json["baseline"])  # type: ignore[arg-type]
    trees = model_json["trees"]
    assert isinstance(trees, list)
    for tree in trees:
        i = 0
        while not tree["leaf"][i]:
            i = (
                tree["left"][i]
                if x[tree["feature"][i]] <= tree["threshold"][i]
                else tree["right"][i]
            )
        total += tree["value"][i]
    return total


def train(
    network: RoadNetwork,
    *,
    trips: int = 20_000,
    hierarchies: dict[float, ContractionHierarchy] | None = None,
    seed: int = 0,
    gnn_epochs: int = 25,
    log: Callable[[str], None] = lambda msg: None,
) -> EtaResult:
    """Simulate trips, train and compare ETA models, return the best for export."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ctx = RouteContext(network)
    if hierarchies is None:
        from .web_export import PRESETS

        log("  contracting the network for each preset time")
        hierarchies = {
            p.hour: ContractionHierarchy.build(Router(ctx.network, Scenario(hour=p.hour)).graph)
            for p in PRESETS
        }
    world = World(ctx, seed=seed)
    log(f"  simulating {trips:,} trips")
    data = simulate_trips(ctx, world, hierarchies, trips, seed=seed + 1, log=log)
    idx_train, idx_test = train_test_split(np.arange(len(data.y)), test_size=0.2, random_state=seed)
    Xtr, Xte = data.X[idx_train], data.X[idx_test]
    ytr = data.y[idx_train]
    eta_te, obs_te = data.graph_eta[idx_test], data.observed[idx_test]

    def to_seconds(pred_log_ratio: np.ndarray) -> np.ndarray:
        return eta_te * np.exp(pred_log_ratio)

    metrics: dict[str, dict[str, float]] = {}
    metrics["Graph ETA (router only)"] = error_metrics(eta_te, obs_te)
    metrics["Graph ETA x global factor"] = error_metrics(
        to_seconds(np.full(len(idx_test), ytr.mean())), obs_te
    )

    log("  training ridge regression")
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr)
    metrics["Ridge regression"] = error_metrics(to_seconds(ridge.predict(Xte)), obs_te)

    log("  training neural network")
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64),
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    ).fit(Xtr, ytr)
    metrics["Neural network (MLP)"] = error_metrics(to_seconds(mlp.predict(Xte)), obs_te)

    log("  training gradient-boosted trees")
    gbdt = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    ).fit(Xtr, ytr)
    metrics["Gradient-boosted trees"] = error_metrics(to_seconds(gbdt.predict(Xte)), obs_te)

    gnn_r = math.nan
    if gnn_epochs:
        log("  training the graph neural network")
        gnn_model = _train_gnn(ctx, data, idx_train, gnn_epochs, seed, log)
        from .gnn import batch as gnn_batch

        metrics["Graph neural network"] = error_metrics(
            gnn_model.predict(gnn_batch(data, idx_test)), obs_te
        )
        gnn_r = gnn_hotspot_correlation(gnn_model, ctx, world)

    sample = np.arange(min(len(idx_test), 3000))
    imp = permutation_importance(
        gbdt, Xte[sample], data.y[idx_test][sample], n_repeats=3, random_state=seed
    )
    order = np.argsort(imp.importances_mean)[::-1]
    importance = [(FEATURES[i], float(imp.importances_mean[i])) for i in order]

    # Arrival window: separate models for the 10th and 90th percentile of the
    # correction, so a trip gets a range a rider can plan around.
    # Quantile models are fitted on part of the training set; the rest calibrates
    # them (conformalized quantile regression), which makes the window's 80%
    # coverage hold on new trips instead of being optimistic.
    log("  training 10th and 90th percentile models for the arrival window")
    fit_idx, cal_idx = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=seed)
    quantile_models = {}
    for name, q in (("p10", 0.1), ("p90", 0.9)):
        quantile_models[name] = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            random_state=seed,
        ).fit(Xtr[fit_idx], ytr[fit_idx])
    margin = conformal_margin(
        quantile_models["p10"].predict(Xtr[cal_idx]),
        quantile_models["p90"].predict(Xtr[cal_idx]),
        ytr[cal_idx],
        coverage=0.8,
    )
    middle = to_seconds(gbdt.predict(Xte))
    raw_low = to_seconds(quantile_models["p10"].predict(Xte))
    raw_high = to_seconds(quantile_models["p90"].predict(Xte))
    low = raw_low * np.exp(-margin)
    high = raw_high * np.exp(margin)
    interval = interval_metrics(low, high, middle, obs_te)
    uncalibrated = interval_metrics(raw_low, raw_high, middle, obs_te)
    interval["uncalibrated_coverage_pct"] = uncalibrated["coverage_pct"]
    interval["margin_log"] = margin

    model_json = export_trees(gbdt)
    model_json["quantiles"] = {name: export_trees(m) for name, m in quantile_models.items()}
    model_json["conformal_margin"] = margin
    # A typical trip, used by the map to ask "how slow would a trip through here be?"
    model_json["reference"] = [float(v) for v in np.median(Xtr, axis=0)]
    interval["learned_hotspot_r"] = learned_hotspot_correlation(model_json, ctx, world)
    if math.isfinite(gnn_r):
        interval["gnn_hotspot_r"] = gnn_r
    return EtaResult(metrics, importance, gbdt, len(data.y), model_json, interval, quantile_models)


def _train_gnn(ctx: RouteContext, data: Dataset, idx_train, epochs: int, seed: int, log):
    from .gnn import line_graph, propagate, segment_features, train_gnn

    net = ctx.network
    x_km = (net.lon - ctx.lon0) * ctx.km_per_deg_lon
    y_km = (net.lat - ctx.lat0) * KM_PER_DEG_LAT
    own = segment_features(net, ctx.graph, ctx.group, x_km, y_km)
    H = propagate(own, line_graph(ctx.graph)).astype(np.float32)
    return train_gnn(H, data, idx_train, epochs=epochs, seed=seed, log=log)


def gnn_hotspot_correlation(model, ctx: RouteContext, world: World) -> float:
    """Correlation of the GNN's per-road slowdown at 08:30 with the hidden truth."""
    from .gnn import time_features

    hour, weekend = 8.5, False
    level = DEFAULT_PROFILE.congestion(hour, weekend=weekend)
    tau = time_features(np.array([hour]), np.array([weekend]), np.array([level]))[0]
    learned = model.segment_log_slowdown(np.arange(ctx.network.num_edges), tau)
    truth = np.log1p((world.hotspot_field + world.group_extra_arr) * world.level(hour, weekend))
    return float(np.corrcoef(learned, truth)[0, 1])


def slowdown_at(
    model_json: dict[str, object], x_km: float, y_km: float, hour: float, weekend: bool
) -> float:
    """Model's expected slowdown for a typical trip placed at a point (as in the map)."""
    features = model_json["features"]
    assert isinstance(features, list)
    names = list(features)
    x = np.array(model_json["reference"], dtype=np.float64)
    angle = 2 * math.pi * hour / 24
    values = {
        "origin_x_km": x_km,
        "origin_y_km": y_km,
        "dest_x_km": x_km,
        "dest_y_km": y_km,
        "mid_x_km": x_km,
        "mid_y_km": y_km,
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "weekend": float(weekend),
    }
    for name, value in values.items():
        x[names.index(name)] = value
    return math.exp(predict_from_json(model_json, x))


def learned_hotspot_correlation(
    model_json: dict[str, object], ctx: RouteContext, world: World, *, hour: float = 8.5
) -> float:
    """Pearson correlation between the model's slowdown map and the hidden hotspots.

    The model never sees the hotspots; a strong correlation means it recovered
    where they are from trip outcomes alone.
    """
    net = ctx.network
    xs = (net.lon - ctx.lon0) * ctx.km_per_deg_lon
    ys = (net.lat - ctx.lat0) * KM_PER_DEG_LAT
    learned, hidden = [], []
    for gx in np.arange(xs.min() + 0.25, xs.max(), 0.5):
        for gy in np.arange(ys.min() + 0.25, ys.max(), 0.5):
            learned.append(math.log(slowdown_at(model_json, gx, gy, hour, False)))
            hidden.append(world.field_at(gx, gy))
    return float(np.corrcoef(learned, hidden)[0, 1])


def conformal_margin(
    low: np.ndarray, high: np.ndarray, y: np.ndarray, *, coverage: float = 0.8
) -> float:
    """Widening (in log space) that gives ``coverage`` on new data.

    Conformalized quantile regression (Romano et al., 2019): score each
    calibration trip by how far it falls outside the predicted range (negative
    when inside), and widen both ends by the right quantile of those scores.
    """
    scores = np.maximum(low - y, y - high)
    n = len(scores)
    level = min(1.0, coverage * (1 + 1 / n))
    return float(np.quantile(scores, level, method="higher"))


def interval_metrics(
    low: np.ndarray, high: np.ndarray, middle: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    """How often observed times fall inside the window, and how wide it is."""
    return {
        "target_pct": 80.0,
        "coverage_pct": float(100 * np.mean((observed >= low) & (observed <= high))),
        "below_pct": float(100 * np.mean(observed < low)),
        "above_pct": float(100 * np.mean(observed > high)),
        "mean_width_pct": float(100 * np.mean((high - low) / middle)),
    }


def metrics_markdown(metrics: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Model | Mean abs. error | Mean abs. % error | 90th pct. % error | Within 10% |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m['mae_s']:.0f} s | {m['mape_pct']:.1f}% | {m['p90_pct']:.1f}% "
            f"| {m['within_10_pct']:.0f}% |"
        )
    return "\n".join(lines)
