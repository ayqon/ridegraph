from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fastroute.eta import (
    FEATURES,
    RouteContext,
    World,
    error_metrics,
    export_trees,
    predict_from_json,
    train,
)
from fastroute.router import Router, Scenario
from fastroute.synthetic import grid_network
from fastroute.web_export import PRESETS, export

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def network():
    return grid_network(30, 30, seed=3)


@pytest.fixture(scope="module")
def trained(network):
    return train(network, trips=2500, seed=1)


def test_features_describe_the_route(network) -> None:
    ctx = RouteContext(network)
    router = Router(ctx.network, Scenario(hour=8))
    trip = router.trip(31, 870, "dijkstra")
    x = dict(zip(FEATURES, ctx.features(trip.intersections, 8, False), strict=True))
    assert x["graph_eta_s"] == pytest.approx(trip.seconds)
    assert x["length_km"] == pytest.approx(trip.metres / 1000)
    assert x["share_major"] + x["share_mid"] + x["share_minor"] == pytest.approx(1)
    assert x["detour"] >= 1
    assert x["n_segments"] == len(trip.intersections) - 1
    assert x["n_left"] + x["n_right"] >= 1  # crossing a grid diagonally must turn


def test_world_is_reproducible_and_slower_at_peak(network) -> None:
    ctx = RouteContext(network)
    w1, w2 = World(ctx, seed=4), World(ctx, seed=4)
    assert np.array_equal(w1.hotspot_field, w2.hotspot_field)
    assert w1.level(8.5, False) > w1.level(3, False)
    path = Router(ctx.network).trip(31, 870, "dijkstra").intersections
    times = ctx.edge_times(8.5, False)
    edges = ctx.path_edges(path, times)
    rng = np.random.default_rng(0)
    observed = [w1.trip_seconds(path, edges, times, 8.5, False, rng) for _ in range(50)]
    assert np.median(observed) > times[edges].sum()  # the real city is slower than the model


def test_exported_trees_match_sklearn(trained) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURES))) * 3
    expected = trained.model.predict(X)
    got = [predict_from_json(trained.model_json, x) for x in X]
    assert np.allclose(got, expected, atol=1e-10)
    assert export_trees(trained.model)["features"] == list(FEATURES)


def test_learned_model_beats_the_router(trained) -> None:
    m = trained.metrics
    graph = m["Graph ETA (router only)"]["mape_pct"]
    scaled = m["Graph ETA x global factor"]["mape_pct"]
    gbdt = m["Gradient-boosted trees"]["mape_pct"]
    assert gbdt < scaled < graph
    assert gbdt < 0.6 * graph
    assert trained.importance[0][1] > 0


def test_arrival_window_is_calibrated(trained) -> None:
    iv = trained.interval
    # 80% target; a few thousand held-out trips leave some sampling noise.
    assert 70 <= iv["coverage_pct"] <= 90
    assert iv["below_pct"] < 20 and iv["above_pct"] < 20
    assert iv["mean_width_pct"] > 0
    assert set(trained.model_json["quantiles"]) == {"p10", "p90"}
    assert len(trained.model_json["reference"]) == len(FEATURES)


def test_conformal_margin_reaches_target_coverage() -> None:
    from fastroute.eta import conformal_margin

    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    # Deliberately overconfident window: +-0.5 covers ~38% of a standard normal.
    low, high = np.full_like(y, -0.5), np.full_like(y, 0.5)
    margin = conformal_margin(low[:2500], high[:2500], y[:2500], coverage=0.8)
    inside = np.mean((y[2500:] >= low[2500:] - margin) & (y[2500:] <= high[2500:] + margin))
    assert 0.77 <= inside <= 0.83


def test_error_metrics() -> None:
    m = error_metrics(np.array([110.0, 90.0]), np.array([100.0, 100.0]))
    assert m["mae_s"] == pytest.approx(10)
    assert m["mape_pct"] == pytest.approx(10)
    assert m["within_10_pct"] == pytest.approx(100)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_eta_matches_python(tmp_path, network) -> None:
    manifest = export(network, tmp_path, eta_trips=1500, log=lambda m: None)
    model = json.loads((tmp_path / manifest["eta"]["file"]).read_text())
    ctx = RouteContext(network)
    rng = random.Random(2)
    cases = []
    for preset in PRESETS:
        router = Router(ctx.network, Scenario(hour=preset.hour))
        for _ in range(12):
            s, t = rng.randrange(network.num_nodes), rng.randrange(network.num_nodes)
            path = router.trip(s, t, "dijkstra").intersections
            if len(path) < 2:
                continue
            x = ctx.features(path, preset.hour, False)
            cases.append(
                {
                    "preset": preset.id,
                    "path": path,
                    "features": x.tolist(),
                    "log_ratio": predict_from_json(model, x),
                    "window": [
                        predict_from_json(model["quantiles"]["p10"], x) - model["conformal_margin"],
                        predict_from_json(model["quantiles"]["p90"], x) + model["conformal_margin"],
                    ],
                }
            )
    cases_path = tmp_path / "eta-cases.json"
    cases_path.write_text(json.dumps(cases))
    proc = subprocess.run(
        [NODE, str(ROOT / "web/test/check_eta.mjs"), str(tmp_path), str(cases_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["checked"] == len(cases)
