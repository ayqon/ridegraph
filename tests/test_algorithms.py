"""Every fast algorithm must return exactly what Dijkstra returns."""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from fastroute.astar import astar
from fastroute.bidirectional import bidirectional_dijkstra
from fastroute.dijkstra import dijkstra
from fastroute.graph import Graph
from fastroute.result import Route

from .conftest import grid_network, make_network

Algorithm = Callable[[Graph, int, int], Route]
ALGORITHMS: dict[str, Algorithm] = {
    "astar": astar,
    "bidirectional": bidirectional_dijkstra,
}


@pytest.fixture(params=list(ALGORITHMS))
def algorithm(request: pytest.FixtureRequest) -> Algorithm:
    return ALGORITHMS[request.param]


def assert_same_as_dijkstra(graph: Graph, algorithm: Algorithm, s: int, t: int) -> Route:
    expected = dijkstra(graph, s, t)
    route = algorithm(graph, s, t)
    assert route.found == expected.found
    if expected.found:
        assert route.seconds == pytest.approx(expected.seconds, rel=1e-9)
        assert route.path[0] == s and route.path[-1] == t
        assert graph.path_time_s(route.path) == pytest.approx(route.seconds, rel=1e-9)
    return route


@pytest.mark.parametrize("seed", range(4))
def test_random_queries(algorithm: Algorithm, seed: int) -> None:
    graph = Graph(grid_network(30, 30, seed=seed, oneway_share=0.3))
    rng = random.Random(100 + seed)
    for _ in range(50):
        assert_same_as_dijkstra(
            graph, algorithm, rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        )


def test_trivial_cases(algorithm: Algorithm) -> None:
    graph = Graph(make_network([(0, 1, 100), (1, 2, 100), (2, 0, 100)], num_nodes=4))
    assert algorithm(graph, 1, 1).path == [1]
    assert algorithm(graph, 1, 0).path == [1, 2, 0]
    assert not algorithm(graph, 0, 3).found


def test_astar_explores_less_than_dijkstra() -> None:
    graph = Graph(grid_network(40, 40, seed=3, oneway_share=0.0))
    rng = random.Random(9)
    ratios = []
    for _ in range(30):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        d, a = dijkstra(graph, s, t), astar(graph, s, t)
        if d.settled > 50:
            ratios.append(a.settled / d.settled)
    assert sum(ratios) / len(ratios) < 0.8


def test_bidirectional_explores_less_than_dijkstra() -> None:
    graph = Graph(grid_network(40, 40, seed=3, oneway_share=0.0))
    rng = random.Random(9)
    total_d = total_b = 0
    for _ in range(30):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        total_d += dijkstra(graph, s, t).settled
        total_b += bidirectional_dijkstra(graph, s, t).settled
    assert total_b < total_d
