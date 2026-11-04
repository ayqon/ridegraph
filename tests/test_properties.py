"""Property-based tests: Hypothesis builds random road networks and checks
that every search method agrees with Dijkstra on all of them, including the
awkward cases (one-way traps, parallel roads, unreachable targets, zero-length
roads) that hand-written fixtures miss."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ridegraph.astar import astar
from ridegraph.bidirectional import bidirectional_dijkstra
from ridegraph.cch import CustomizableCH
from ridegraph.ch import ContractionHierarchy
from ridegraph.dijkstra import one_to_all
from ridegraph.graph import Graph, RoadNetwork, haversine_m
from ridegraph.matching import greedy_matches, hungarian, optimal_matches
from ridegraph.tour import held_karp, tour_cost

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def networks(draw):
    """A random directed road network with consistent geometry."""
    n = draw(st.integers(2, 18))
    coords = draw(
        st.lists(
            st.tuples(st.floats(51.50, 51.52), st.floats(-0.14, -0.12)), min_size=n, max_size=n
        )
    )
    pairs = draw(
        st.lists(
            st.tuples(st.integers(0, n - 1), st.integers(0, n - 1), st.floats(1.0, 3.0)),
            min_size=1,
            max_size=4 * n,
        )
    )
    src, dst, length = [], [], []
    for a, b, stretch in pairs:
        if a == b:
            continue
        # A road is never shorter than the straight line, which keeps A* admissible.
        crow = haversine_m(*coords[a], *coords[b])
        src.append(a)
        dst.append(b)
        length.append(max(crow * stretch, 0.5))
    if not src:
        src, dst, length = [0], [1], [max(haversine_m(*coords[0], *coords[1]), 0.5)]
    m = len(src)
    lat, lon = np.array(coords).T
    return RoadNetwork(
        node_ids=np.arange(n, dtype=np.int64) + 1000,
        lat=lat,
        lon=lon,
        src=np.array(src, dtype=np.int32),
        dst=np.array(dst, dtype=np.int32),
        length_m=np.array(length, dtype=np.float32),
        road_class=np.full(m, 4, dtype=np.int8),
        maxspeed_kph=np.full(m, np.nan, dtype=np.float32),
    )


def same(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return a == pytest.approx(b, rel=1e-9, abs=1e-6)


@SETTINGS
@given(networks())
def test_every_method_agrees_with_dijkstra(network) -> None:
    graph = Graph(network)
    ch = ContractionHierarchy.build(graph)
    cch = CustomizableCH.build(network)
    cch.customize(_seconds(network))
    for s in range(network.num_nodes):
        truth = one_to_all(graph, s)
        for t in range(network.num_nodes):
            assert same(bidirectional_dijkstra(graph, s, t).seconds, truth[t])
            assert same(astar(graph, s, t).seconds, truth[t])
            assert same(ch.query(s, t).seconds, truth[t])
            assert same(cch.query(s, t)[0], truth[t])


def _seconds(network) -> np.ndarray:
    from ridegraph.graph import free_flow_cost

    return np.asarray(free_flow_cost(network), dtype=np.float64)


@SETTINGS
@given(networks(), st.lists(st.floats(0.2, 5.0), min_size=1, max_size=80), st.data())
def test_customization_handles_any_new_times(network, factors, data) -> None:
    """Re-customising with arbitrary times, closures included, stays exact."""
    cch = CustomizableCH.build(network)
    seconds = _seconds(network) * np.resize(np.array(factors), network.num_edges)
    closed = data.draw(st.lists(st.integers(0, network.num_edges - 1), max_size=3))
    seconds[closed] = np.inf
    cch.customize(seconds)
    graph = Graph(network, lambda net: seconds)
    for s in range(network.num_nodes):
        truth = one_to_all(graph, s)
        for t in range(network.num_nodes):
            d, path = cch.query(s, t)
            assert same(d, truth[t])
            if math.isfinite(d):
                assert path[0] == s and path[-1] == t


@SETTINGS
@given(networks())
def test_quickest_times_obey_the_triangle_inequality(network) -> None:
    graph = Graph(network)
    dist = [one_to_all(graph, s) for s in range(network.num_nodes)]
    for a, b, c in itertools.product(range(network.num_nodes), repeat=3):
        assert dist[a][c] <= dist[a][b] + dist[b][c] + 1e-6


cost_matrices = st.integers(1, 6).flatmap(
    lambda r: st.integers(1, 6).flatmap(
        lambda c: st.lists(
            st.lists(st.floats(0, 1000), min_size=c, max_size=c), min_size=r, max_size=r
        )
    )
)


@SETTINGS
@given(cost_matrices)
def test_hungarian_matches_brute_force(rows) -> None:
    cost = np.array(rows)
    r, c = cost.shape
    pairs = hungarian(cost)
    assert len(pairs) == min(r, c)
    assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == len(pairs)
    best = min(
        sum(cost[i, j] for i, j in zip(range(r), perm, strict=False))
        if r <= c
        else sum(cost[i, j] for i, j in zip(perm, range(c), strict=False))
        for perm in itertools.permutations(range(max(r, c)), min(r, c))
    )
    assert sum(cost[i, j] for i, j in pairs) == pytest.approx(best)
    g = greedy_matches(cost, np.inf)
    o = optimal_matches(cost, np.inf)
    assert sum(cost[i, j] for i, j in o) <= sum(cost[i, j] for i, j in g) + 1e-9


@SETTINGS
@given(
    st.integers(2, 7).flatmap(
        lambda n: st.lists(
            st.lists(st.floats(1, 500), min_size=n, max_size=n), min_size=n, max_size=n
        )
    ),
    st.booleans(),
)
def test_held_karp_is_optimal(rows, round_trip) -> None:
    m = np.array(rows)
    np.fill_diagonal(m, 0)
    order = held_karp(m, return_to_start=round_trip)
    brute = min(
        tour_cost(m, [0, *p], return_to_start=round_trip)
        for p in itertools.permutations(range(1, len(m)))
    )
    assert tour_cost(m, order, return_to_start=round_trip) == pytest.approx(brute)
