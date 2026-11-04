from __future__ import annotations

import numpy as np
import pytest

from ridegraph.ch import ContractionHierarchy
from ridegraph.demand import SLOTS_PER_DAY, DemandWorld, forecast_features, train_forecaster
from ridegraph.dispatch import POLICIES, DispatchConfig, compare, generate_rides, simulate
from ridegraph.graph import Graph, csr_sorted
from ridegraph.synthetic import grid_network

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def city():
    network = csr_sorted(grid_network(26, 26, seed=4))
    ch = ContractionHierarchy.build(Graph(network))
    world = DemandWorld(network, peak_per_min=2.0, seed=1)
    return network, ch, world


@pytest.fixture(scope="module")
def forecaster(city):
    _, _, world = city
    return train_forecaster(world, train_days=10, test_days=3)


def test_demand_follows_its_daily_rhythm(city) -> None:
    _, _, world = city
    day = world.day(0)
    morning = sum(day.expected_counts(s).sum() for s in range(28, 40))  # 07:00-10:00
    night = sum(day.expected_counts(s).sum() for s in range(12, 24))  # 03:00-06:00
    assert morning > 3 * night
    w = day.weights(8.0, "origin")
    assert w.min() >= 0 and w.sum() == pytest.approx(1)


def test_days_differ_but_are_reproducible(city) -> None:
    _, _, world = city
    a, b, a2 = world.day(3), world.day(4), world.day(3)
    assert np.allclose(a.surprise, a2.surprise) and a.level == a2.level
    assert not np.allclose(a.surprise, b.surprise)


def test_forecast_features_only_look_backwards(city) -> None:
    _, _, world = city
    counts = np.zeros((SLOTS_PER_DAY, world.grid.num_cells))
    counts[40] += 5  # the slot being predicted, and later ones, must not leak
    counts[41:] += 9
    x = forecast_features(world.grid, counts, 40, False, world.grid.active)
    assert np.all(x[:, 5:] == 0)


def test_forecaster_beats_the_baselines(forecaster) -> None:
    m = forecaster.metrics
    model = m["Gradient-boosted forecast"]["deviance"]
    assert model < m["Historical average for the time"]["deviance"]
    assert model < m["Same as the last 15 minutes"]["deviance"]


def test_rides_are_reproducible_and_valid(city) -> None:
    network, ch, world = city
    cfg = DispatchConfig(drivers=30, hours=1.0, warmup_min=10)
    rides = generate_rides(world.day(5), ch, cfg, np.random.default_rng(1))
    again = generate_rides(world.day(5), ch, cfg, np.random.default_rng(1))
    assert [(r.origin, r.dest) for r in rides] == [(r.origin, r.dest) for r in again]
    assert all(0 <= r.created < 3600 and r.trip_s > 0 for r in rides)
    assert all(r.origin != r.dest for r in rides)


def test_policies_run_on_identical_requests(city, forecaster) -> None:
    _, ch, world = city
    cfg = DispatchConfig(drivers=35, start_hour=7.5, hours=1.0, warmup_min=10)
    results = compare(world, 20, ch, cfg, forecaster.model_json, history=forecaster.historical_mean)
    assert [r.policy for r in results] == list(POLICIES)
    requests = {r.metrics["requests"] for r in results}
    assert len(requests) == 1 and requests.pop() > 20
    for r in results:
        m = r.metrics
        assert m["served_pct"] + m["cancelled_pct"] <= 100.0001
        for ride in r.rides:
            if ride.assigned:
                assert ride.pickup_s <= cfg.max_pickup_s
                assert ride.picked_up >= ride.created
    by = {r.policy: r.metrics for r in results}
    assert by["nearest"]["repositioning_min_per_driver"] == 0
    assert by["optimal"]["repositioning_min_per_driver"] == 0
    assert by["optimal+forecast"]["repositioning_min_per_driver"] >= 0


def test_optimal_matching_never_has_a_longer_total_pickup_per_batch(city) -> None:
    from ridegraph.matching import greedy_matches, optimal_matches

    _, ch, world = city
    rng = np.random.default_rng(0)
    n = world.network.num_nodes
    drivers = [int(x) for x in rng.choice(n, 15)]
    riders = [int(x) for x in rng.choice(n, 10)]
    cost = ch.many_to_many(drivers, riders)
    limit = 1e9
    g = greedy_matches(cost, limit)
    o = optimal_matches(cost, limit)
    assert len(o) == len(g) == 10
    assert sum(cost[i, j] for i, j in o) <= sum(cost[i, j] for i, j in g) + 1e-9


def test_unknown_policy_rejected(city) -> None:
    _, ch, world = city
    with pytest.raises(ValueError):
        simulate(world, world.day(0), [], ch, "teleport", DispatchConfig())


def test_export_writes_a_consistent_replay(tmp_path) -> None:
    import json

    from ridegraph.web_export import export

    network = grid_network(20, 20, seed=6)
    manifest = export(network, tmp_path, dispatch_days=1, log=lambda m: None)
    info = manifest["dispatch"]
    replay = json.loads((tmp_path / info["file"]).read_text())
    assert [p["id"] for p in replay["policies"]] == list(POLICIES)
    assert set(info["summary"]) == set(POLICIES)

    # The browser rebuilds every leg with the exported morning hierarchy, so
    # each recorded duration must be what that hierarchy says.
    preset = next(p for p in manifest["presets"] if p["id"] == info["preset"])
    from ridegraph.router import Router, Scenario

    ch = ContractionHierarchy.build(
        Router(csr_sorted(network), Scenario(hour=preset["hour"])).graph
    )
    n = manifest["nodes"]
    for policy in replay["policies"]:
        assert len(policy["start_nodes"]) == replay["drivers"]
        for driver, kind, _t0, u, v, dur, ride in policy["legs"][:60]:
            assert 0 <= driver < replay["drivers"] and 0 <= u < n and 0 <= v < n
            assert dur == pytest.approx(ch.query(u, v).seconds, abs=0.06)
            if kind != 2:
                assert 0 <= ride < len(policy["rides"])

    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node:
        script = Path(__file__).resolve().parent.parent / "web/test/check_dispatch.mjs"
        proc = subprocess.run(
            [node, str(script), str(tmp_path)], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert json.loads(proc.stdout)["checked"] > 0
