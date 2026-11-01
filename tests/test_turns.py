from __future__ import annotations

import random

import numpy as np
import pytest

from fastroute.astar import astar
from fastroute.bidirectional import bidirectional_dijkstra
from fastroute.ch import ContractionHierarchy
from fastroute.dijkstra import dijkstra
from fastroute.graph import Graph, free_flow_cost
from fastroute.synthetic import grid_network
from fastroute.turns import NO_PENALTIES, Turn, TurnGraph, TurnPenalties

from .conftest import make_network

D = 0.001  # about 111 m of latitude
CENTRE, NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3, 4


def crossroads(signals: bool = False):
    """A + shaped junction around node 0, all arms two-way."""
    coords = [(51.5, -0.1), (51.5 + D, -0.1), (51.5, -0.1 + 1.6 * D), (51.5 - D, -0.1),
              (51.5, -0.1 - 1.6 * D)]  # fmt: skip
    edges = []
    for arm in (NORTH, EAST, SOUTH, WEST):
        edges += [(CENTRE, arm, 111), (arm, CENTRE, 111)]
    network = make_network(edges, coords=coords)
    if signals:
        lights = np.zeros(5, dtype=bool)
        lights[CENTRE] = True
        network = type(network)(**{**network.__dict__, "traffic_signals": lights})
    return network


def trip_seconds(tg: TurnGraph, a: int, b: int) -> float:
    return dijkstra(tg.graph(), tg.start(a), tg.end(b)).seconds


def test_right_turns_cost_more_than_left_in_the_uk() -> None:
    network = crossroads()
    tg = TurnGraph(network)
    base = 2 * float(free_flow_cost(network)[0])
    p = TurnPenalties()
    # Heading north from SOUTH: EAST is a right turn, WEST a left, NORTH straight on.
    assert trip_seconds(tg, SOUTH, NORTH) == pytest.approx(base)
    assert trip_seconds(tg, SOUTH, WEST) == pytest.approx(base + p.near_side_s)
    assert trip_seconds(tg, SOUTH, EAST) == pytest.approx(base + p.far_side_s)


def test_driving_on_the_right_swaps_the_costs() -> None:
    network = crossroads()
    p = TurnPenalties(drive_on_left=False)
    tg = TurnGraph(network, p)
    base = 2 * float(free_flow_cost(network)[0])
    assert trip_seconds(tg, SOUTH, EAST) == pytest.approx(base + p.near_side_s)
    assert trip_seconds(tg, SOUTH, WEST) == pytest.approx(base + p.far_side_s)


def test_traffic_lights_add_a_wait() -> None:
    network = crossroads(signals=True)
    tg = TurnGraph(network)
    base = 2 * float(free_flow_cost(network)[0])
    assert trip_seconds(tg, SOUTH, NORTH) == pytest.approx(base + TurnPenalties().traffic_signals_s)
    route = dijkstra(tg.graph(), tg.start(SOUTH), tg.end(EAST))
    assert tg.manoeuvres(route.path) == {"right": 1, "traffic lights": 1}


def test_bend_in_a_road_is_free() -> None:
    # 0 -> 1 heads north, 1 -> 2 heads east: a 90 degree bend, but only one road.
    coords = [(51.5, -0.1), (51.5 + D, -0.1), (51.5 + D, -0.1 + 1.6 * D)]
    network = make_network([(0, 1, 111), (1, 2, 111), (2, 1, 111), (1, 0, 111)], coords=coords)
    tg = TurnGraph(network)
    assert trip_seconds(tg, 0, 2) == pytest.approx(float(free_flow_cost(network)[:2].sum()))


def test_u_turn_is_classified_and_charged() -> None:
    network = crossroads()
    tg = TurnGraph(network)
    u_turns = tg.turn_kind[: tg.num_manoeuvres] == Turn.U_TURN
    assert u_turns.sum() == 8  # one per arm, at each end of every two-way segment
    assert np.all(tg.penalty[: tg.num_manoeuvres][u_turns] == TurnPenalties().u_turn_s)


def test_trip_to_same_place_is_free() -> None:
    tg = TurnGraph(crossroads())
    route = dijkstra(tg.graph(), tg.start(SOUTH), tg.end(SOUTH))
    assert route.seconds == 0
    assert tg.intersections(route.path) == [SOUTH]


def test_without_penalties_matches_plain_graph() -> None:
    network = grid_network(15, 15, seed=5, oneway_share=0.3)
    plain = Graph(network)
    tg = TurnGraph(network, NO_PENALTIES)
    expanded = tg.graph()
    rng = random.Random(3)
    for _ in range(60):
        s, t = rng.randrange(plain.num_nodes), rng.randrange(plain.num_nodes)
        a = dijkstra(plain, s, t)
        b = dijkstra(expanded, tg.start(s), tg.end(t))
        assert a.found == b.found
        if a.found:
            assert b.seconds == pytest.approx(a.seconds)
            path = tg.intersections(b.path)
            assert path[0] == s and path[-1] == t
            assert plain.path_time_s(path) == pytest.approx(a.seconds)


def test_penalties_only_ever_add_time() -> None:
    network = grid_network(15, 15, seed=6)
    plain = Graph(network)
    turned = TurnGraph(network).graph()
    tg = TurnGraph(network)
    rng = random.Random(4)
    for _ in range(40):
        s, t = rng.randrange(plain.num_nodes), rng.randrange(plain.num_nodes)
        assert dijkstra(turned, tg.start(s), tg.end(t)).seconds >= dijkstra(plain, s, t).seconds


def test_all_algorithms_agree_on_turn_graph() -> None:
    network = grid_network(14, 14, seed=8, oneway_share=0.25)
    tg = TurnGraph(network)
    graph = tg.graph()
    ch = ContractionHierarchy.build(graph)
    rng = random.Random(11)
    for _ in range(50):
        s, t = tg.start(rng.randrange(network.num_nodes)), tg.end(rng.randrange(network.num_nodes))
        expected = dijkstra(graph, s, t)
        for route in (astar(graph, s, t), bidirectional_dijkstra(graph, s, t), ch.query(s, t)):
            assert route.found == expected.found
            if expected.found:
                assert route.seconds == pytest.approx(expected.seconds, rel=1e-9)
                assert graph.path_time_s(route.path) == pytest.approx(route.seconds, rel=1e-9)
