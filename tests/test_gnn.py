from __future__ import annotations

import numpy as np
import pytest

from fastroute.ch import ContractionHierarchy
from fastroute.eta import RouteContext, World, simulate_trips, train
from fastroute.gnn import SegmentGNN, batch, line_graph, propagate, segment_features
from fastroute.graph import Graph
from fastroute.router import Router, Scenario
from fastroute.synthetic import grid_network

pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def setup():
    ctx = RouteContext(grid_network(14, 14, seed=2))
    ch = ContractionHierarchy.build(Router(ctx.network, Scenario(hour=8.0)).graph)
    data = simulate_trips(ctx, World(ctx, seed=1), {8.0: ch}, 60, seed=2, min_crow_km=0.2)
    net = ctx.network
    x = (net.lon - ctx.lon0) * ctx.km_per_deg_lon
    y = (net.lat - ctx.lat0) * 111.32
    H = propagate(segment_features(net, ctx.graph, ctx.group, x, y), line_graph(ctx.graph))
    return ctx, data, H


def test_line_graph_is_symmetric_and_normalised(setup) -> None:
    ctx, _, _ = setup
    a = line_graph(ctx.graph)
    assert abs(a - a.T).max() < 1e-12
    # A segment's successors share its head junction.
    g: Graph = ctx.graph
    e = 5
    succ = range(g.first_out[g.head[e]], g.first_out[g.head[e] + 1])
    assert all(a[e, f] > 0 for f in succ)


def analytic_gradients(model: SegmentGNN, b) -> dict[str, np.ndarray]:
    """Gradients from one optimiser step with a zero learning rate.

    After one Adam step the first moment is (1 - beta1) x gradient, and a zero
    rate leaves the parameters where they were.
    """
    before = {k: v.copy() for k, v in model.params.items()}
    opt = {"t": 0}
    model.step(b, opt, 0.0)
    grads = {}
    for k in model.params:
        g = opt[("m", k)] / (1 - 0.9)
        if k in ("W1", "W2"):
            g = g - 1e-5 * before[k]  # the step adds a small weight decay
        grads[k] = g
        assert np.array_equal(model.params[k], before[k])
    return grads


def test_gradients_match_finite_differences(setup) -> None:
    _, data, H = setup
    model = SegmentGNN.create(H, data.tau.shape[1], data.counts.shape[1], seed=3)
    rng = np.random.default_rng(0)
    model.params["w3"] = rng.normal(0, 0.1, model.params["w3"].shape)  # exercise every path
    b = batch(data, np.arange(12))

    def loss() -> float:
        return float(np.mean((np.log(model.predict(b)) - np.log(b.observed)) ** 2))

    grads = analytic_gradients(model, b)
    for name in ("w3", "b3", "W2", "b2", "W1", "b1", "C"):
        p = model.params[name]
        for _ in range(3):
            idx = tuple(int(rng.integers(s)) for s in p.shape)
            eps = 1e-6
            p[idx] += eps
            up = loss()
            p[idx] -= 2 * eps
            down = loss()
            p[idx] += eps
            numeric = (up - down) / (2 * eps)
            assert grads[name][idx] == pytest.approx(numeric, rel=1e-3, abs=1e-9), name


def test_gnn_is_reported_and_beats_the_router() -> None:
    result = train(grid_network(24, 24, seed=3), trips=1500, seed=2, gnn_epochs=10)
    m = result.metrics
    assert "Graph neural network" in m
    assert m["Graph neural network"]["mape_pct"] < m["Graph ETA (router only)"]["mape_pct"]
    assert -1 <= result.interval["gnn_hotspot_r"] <= 1
