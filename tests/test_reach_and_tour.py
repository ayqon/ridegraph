from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from fastroute.ch import ContractionHierarchy
from fastroute.dijkstra import one_to_all
from fastroute.graph import Graph
from fastroute.reach import reach, times_within
from fastroute.synthetic import grid_network
from fastroute.tour import (
    held_karp,
    improve,
    nearest_neighbour,
    plan,
    tour_cost,
)
from fastroute.traffic import traffic_cost


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph(grid_network(25, 25, seed=7, oneway_share=0.3))


def test_times_within_matches_dijkstra(graph) -> None:
    full = one_to_all(graph, 300)
    budget = 600.0
    got = times_within(graph, 300, budget)
    expected = {v: d for v, d in enumerate(full) if d <= budget}
    assert got.keys() == expected.keys()
    for v, d in got.items():
        assert d == pytest.approx(expected[v])


def test_arriving_uses_journeys_into_the_point(graph) -> None:
    into = times_within(graph, 300, 600, arriving=True)
    for v, d in list(into.items())[:40]:
        assert one_to_all(graph, v)[300] == pytest.approx(d)


def test_reach_grows_with_budget_and_shrinks_at_rush_hour() -> None:
    network = grid_network(25, 25, seed=7)
    night = reach(Graph(network, traffic_cost(3)), 312, (2, 4))
    peak = reach(Graph(network, traffic_cost(8)), 312, (2, 4))
    assert night.area_km2[0] <= night.area_km2[1]
    assert peak.area_km2[1] < night.area_km2[1]
    assert peak.junctions[1] < night.junctions[1]


def brute_force(matrix: np.ndarray, *, return_to_start: bool) -> float:
    n = len(matrix)
    return min(
        tour_cost(matrix, [0, *p], return_to_start=return_to_start)
        for p in itertools.permutations(range(1, n))
    )


@pytest.mark.parametrize("return_to_start", [True, False])
def test_held_karp_is_optimal_on_asymmetric_matrices(return_to_start) -> None:
    rng = np.random.default_rng(1)
    for n in (2, 3, 5, 7):
        for _ in range(5):
            m = rng.uniform(10, 100, size=(n, n))
            np.fill_diagonal(m, 0)
            order = held_karp(m, return_to_start=return_to_start)
            assert sorted(order) == list(range(n)) and order[0] == 0
            assert tour_cost(m, order, return_to_start=return_to_start) == pytest.approx(
                brute_force(m, return_to_start=return_to_start)
            )


def test_improvement_never_makes_a_tour_worse() -> None:
    rng = np.random.default_rng(2)
    for _ in range(10):
        m = rng.uniform(10, 100, size=(9, 9))
        start = nearest_neighbour(m)
        better = improve(m, start, return_to_start=True)
        assert better[0] == 0 and sorted(better) == list(range(9))
        assert tour_cost(m, better, return_to_start=True) <= tour_cost(
            m, start, return_to_start=True
        )


def test_plan_on_a_road_network(graph) -> None:
    ch = ContractionHierarchy.build(graph)
    rng = random.Random(4)
    stops = rng.sample(range(graph.num_nodes), 8)
    result = plan(ch, stops)
    best = result.seconds["Optimal (Held-Karp)"]
    for name, secs in result.seconds.items():
        assert secs >= best - 1e-6, name
    assert result.seconds["Improved (2-opt + Or-opt)"] <= result.seconds["Nearest next stop"] + 1e-9
    # The matrix is real travel time: check a few entries against Dijkstra.
    for i, j in [(0, 1), (3, 5), (7, 2)]:
        assert result.matrix[i, j] == pytest.approx(one_to_all(graph, stops[i])[stops[j]])


def test_browser_matches_python(tmp_path) -> None:
    import json
    import shutil
    import subprocess
    from pathlib import Path

    from fastroute.graph import csr_sorted
    from fastroute.router import Router, Scenario
    from fastroute.web_export import export

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    network = grid_network(16, 16, seed=8, oneway_share=0.3)
    export(network, tmp_path, log=lambda m: None)
    router = Router(csr_sorted(network), Scenario(hour=8.0))
    ch = ContractionHierarchy.build(router.graph)
    rng = random.Random(9)
    reach_cases = []
    for arriving in (False, True):
        for _ in range(4):
            o = rng.randrange(network.num_nodes)
            t = times_within(router.graph, o, 240, arriving=arriving)
            reach_cases.append({"origin": o, "budget": 240, "arriving": arriving, "times": t})
    tours = []
    for return_to_start in (True, False):
        stops = rng.sample(range(network.num_nodes), 7)
        p = plan(ch, stops, return_to_start=return_to_start)
        tours.append(
            {
                "stops": stops,
                "return_to_start": return_to_start,
                "matrix": p.matrix.tolist(),
                "seconds": p.seconds,
            }
        )
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"preset": "0800", "reach": reach_cases, "tours": tours}))
    script = Path(__file__).resolve().parent.parent / "web/test/check_reach_tour.mjs"
    proc = subprocess.run(
        [node, str(script), str(tmp_path), str(cases)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
