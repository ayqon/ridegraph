from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from fastroute.dijkstra import dijkstra, one_to_all
from fastroute.graph import Graph

from .conftest import grid_network, make_network


def to_networkx(graph: Graph) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(graph.num_nodes))
    for u in range(graph.num_nodes):
        for v, w in graph.out_edges(u):
            if not g.has_edge(u, v) or g[u][v]["weight"] > w:
                g.add_edge(u, v, weight=w)
    return g


def test_picks_faster_road_over_shorter_one() -> None:
    # 0 -> 1 -> 3 is 600 m of living street; 0 -> 2 -> 3 is 900 m of primary road.
    graph = Graph(
        make_network(
            [
                (0, 1, 300, "living_street"),
                (1, 3, 300, "living_street"),
                (0, 2, 450, "primary"),
                (2, 3, 450, "primary"),
            ]
        )
    )
    route = dijkstra(graph, 0, 3)
    assert graph.path_length_m([0, 1, 3]) < graph.path_length_m(route.path)
    assert route.path == [0, 2, 3]
    assert graph.path_length_m(route.path) == pytest.approx(900)


def test_same_source_and_target() -> None:
    graph = Graph(make_network([(0, 1, 100)]))
    route = dijkstra(graph, 0, 0)
    assert route.path == [0]
    assert route.seconds == 0


def test_unreachable_target() -> None:
    graph = Graph(make_network([(0, 1, 100)], num_nodes=3))
    route = dijkstra(graph, 0, 2)
    assert not route.found
    assert route.path == []


def test_respects_one_way_streets() -> None:
    graph = Graph(make_network([(0, 1, 100), (1, 2, 100), (2, 0, 100)]))
    # Going against the cycle means travelling almost all the way round.
    assert dijkstra(graph, 1, 0).path == [1, 2, 0]


def test_parallel_edges_use_cheapest() -> None:
    graph = Graph(make_network([(0, 1, 500), (0, 1, 100)]))
    assert dijkstra(graph, 0, 1).seconds == pytest.approx(graph.weight[graph.edge_between(0, 1)])


@pytest.mark.parametrize("seed", range(5))
def test_matches_networkx_on_random_queries(seed: int) -> None:
    graph = Graph(grid_network(25, 25, seed=seed, oneway_share=0.3))
    reference = to_networkx(graph)
    rng = random.Random(seed)
    for _ in range(40):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        route = dijkstra(graph, s, t)
        try:
            expected = nx.dijkstra_path_length(reference, s, t)
        except nx.NetworkXNoPath:
            assert not route.found
            continue
        assert route.seconds == pytest.approx(expected)
        # The returned path must be a real path whose cost is the reported cost.
        assert route.path[0] == s and route.path[-1] == t
        assert graph.path_time_s(route.path) == pytest.approx(route.seconds)


def test_one_to_all_agrees_with_point_queries(grid) -> None:
    graph = Graph(grid)
    dist = one_to_all(graph, 0)
    for t in range(0, graph.num_nodes, 37):
        route = dijkstra(graph, 0, t)
        assert (route.seconds if route.found else math.inf) == pytest.approx(dist[t])


def test_settles_fewer_nodes_for_nearby_targets(grid) -> None:
    graph = Graph(grid)
    centre = graph.num_nodes // 2
    near = dijkstra(graph, centre, centre + 1)
    far = dijkstra(graph, 0, graph.num_nodes - 1)
    assert near.settled < far.settled
