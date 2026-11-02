from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from fastroute.ch import ContractionHierarchy
from fastroute.dijkstra import one_to_all
from fastroute.graph import Graph
from fastroute.matching import greedy_matches, hungarian, optimal_matches
from fastroute.synthetic import grid_network


def brute_force(cost: np.ndarray) -> float:
    n, m = cost.shape
    if n <= m:
        return min(
            sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n)
        )
    return brute_force(cost.T)


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (4, 6), (6, 4), (5, 5)])
def test_hungarian_is_optimal(shape) -> None:
    rng = np.random.default_rng(sum(shape))
    for _ in range(15):
        cost = rng.integers(1, 100, size=shape).astype(float)
        pairs = hungarian(cost)
        assert len(pairs) == min(shape)
        assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == len(pairs)
        assert sum(cost[i, j] for i, j in pairs) == pytest.approx(brute_force(cost))


def test_hungarian_empty() -> None:
    assert hungarian(np.zeros((0, 3))) == []


def test_limit_excludes_far_pairs() -> None:
    # Only driver 0 -> rider 0 and driver 1 -> rider 0 are within the limit.
    cost = np.array([[1.0, 500.0], [400.0, 600.0]])
    assert optimal_matches(cost, limit=450) == [(0, 0)]
    assert greedy_matches(cost, limit=450) == [(0, 0)]


def test_greedy_can_be_worse_than_optimal() -> None:
    # Rider 0 grabs driver 0, leaving rider 1 a long wait; optimal swaps them.
    cost = np.array([[1.0, 2.0], [3.0, 100.0]])
    greedy = greedy_matches(cost, limit=1e6)
    best = optimal_matches(cost, limit=1e6)
    assert sum(cost[i, j] for i, j in best) < sum(cost[i, j] for i, j in greedy)


def test_many_to_many_matches_dijkstra() -> None:
    graph = Graph(grid_network(18, 18, seed=2, oneway_share=0.3))
    ch = ContractionHierarchy.build(graph)
    rng = random.Random(3)
    sources = [rng.randrange(graph.num_nodes) for _ in range(12)]
    targets = [rng.randrange(graph.num_nodes) for _ in range(9)]
    table = ch.many_to_many(sources, targets)
    for i, s in enumerate(sources):
        dist = one_to_all(graph, s)
        for j, t in enumerate(targets):
            assert table[i, j] == pytest.approx(dist[t])
    # Reusing target buckets gives the same table.
    again = ch.many_to_many(sources, targets, ch.target_buckets(targets))
    assert np.allclose(table, again)
