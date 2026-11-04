from __future__ import annotations

import random

import pytest

from ridegraph.ch import NO_MIDDLE, ContractionHierarchy
from ridegraph.dijkstra import dijkstra
from ridegraph.graph import Graph

from .conftest import grid_network, make_network


@pytest.fixture(scope="module", params=[0, 1, 2])
def contracted(request: pytest.FixtureRequest) -> tuple[Graph, ContractionHierarchy]:
    graph = Graph(grid_network(25, 25, seed=request.param, oneway_share=0.3))
    return graph, ContractionHierarchy.build(graph)


def test_matches_dijkstra(contracted: tuple[Graph, ContractionHierarchy]) -> None:
    graph, ch = contracted
    rng = random.Random(42)
    for _ in range(150):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        expected = dijkstra(graph, s, t)
        route = ch.query(s, t)
        assert route.found == expected.found
        if expected.found:
            assert route.seconds == pytest.approx(expected.seconds, rel=1e-9)
            # Unpacked shortcuts must form a real road path with the reported cost.
            assert route.path[0] == s and route.path[-1] == t
            assert graph.path_time_s(route.path) == pytest.approx(route.seconds, rel=1e-9)


def test_ranks_are_a_permutation(contracted: tuple[Graph, ContractionHierarchy]) -> None:
    graph, ch = contracted
    assert sorted(ch.rank) == list(range(graph.num_nodes))


def test_edges_respect_ranking(contracted: tuple[Graph, ContractionHierarchy]) -> None:
    _, ch = contracted
    for u in range(ch.num_nodes):
        for e in range(ch.up_first[u], ch.up_first[u + 1]):
            assert ch.rank[ch.up_head[e]] > ch.rank[u]
        for e in range(ch.down_first[u], ch.down_first[u + 1]):
            assert ch.rank[ch.down_tail[e]] > ch.rank[u]


def test_adds_shortcuts_and_searches_less(contracted: tuple[Graph, ContractionHierarchy]) -> None:
    graph, ch = contracted
    assert ch.num_shortcuts > 0
    rng = random.Random(7)
    total_d = total_ch = 0
    for _ in range(40):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        total_d += dijkstra(graph, s, t).settled
        total_ch += ch.query(s, t).settled
    assert total_ch * 3 < total_d


def test_shortcut_needed_through_middle_of_a_line() -> None:
    # A one-way chain: the only route from 0 to 2 goes through 1, so whichever
    # inner node is contracted first must leave a shortcut behind.
    graph = Graph(make_network([(0, 1, 100), (1, 2, 100), (2, 3, 100)]))
    ch = ContractionHierarchy.build(graph)
    assert ch.query(0, 3).path == [0, 1, 2, 3]
    assert not ch.query(3, 0).found


def test_witness_avoids_unneeded_shortcut() -> None:
    # 0 -> 1 -> 2 costs the same as the direct 0 -> 2, so contracting 1 needs
    # no shortcut: the direct road is a witness.
    graph = Graph(make_network([(0, 1, 100), (1, 2, 100), (0, 2, 200)]))
    ch = ContractionHierarchy.build(graph)
    assert ch.query(0, 2).seconds == pytest.approx(dijkstra(graph, 0, 2).seconds)
    middles = [m for m in ch.up_middle + ch.down_middle if m != NO_MIDDLE]
    assert middles == []


def test_save_and_load_round_trip(tmp_path, contracted) -> None:
    graph, ch = contracted
    path = tmp_path / "ch.npz"
    ch.save(path)
    loaded = ContractionHierarchy.load(path)
    rng = random.Random(1)
    for _ in range(20):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        assert loaded.query(s, t).seconds == pytest.approx(ch.query(s, t).seconds)
