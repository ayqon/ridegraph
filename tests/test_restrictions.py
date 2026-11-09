"""Turn restrictions: storing them on the network, and obeying them."""

from __future__ import annotations

import numpy as np
import pytest

from ridegraph.graph import RoadNetwork

from .conftest import make_network


def with_bans(network: RoadNetwork, bans: list[tuple[int, int, int]]) -> RoadNetwork:
    return RoadNetwork(**{**network.__dict__, "banned_turns": np.array(bans, dtype=np.int32)})


def line() -> RoadNetwork:
    return make_network([(0, 1, 100), (1, 2, 100), (2, 1, 100), (1, 0, 100)])


def test_no_bans_by_default():
    banned = line().banned_turns
    assert banned is not None and banned.shape == (0, 3)


def test_bans_survive_save_and_load(tmp_path):
    network = with_bans(line(), [(0, 1, 2)])
    network.save(tmp_path / "net.npz")
    loaded = RoadNetwork.load(tmp_path / "net.npz")
    assert loaded.banned_turns is not None
    assert loaded.banned_turns.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("bans", [[(0, 1, 9)], [(-1, 1, 2)]])
def test_bans_must_point_at_real_nodes(bans):
    with pytest.raises(ValueError):
        with_bans(line(), bans)


def test_bans_must_be_triples():
    with pytest.raises(ValueError):
        RoadNetwork(**{**line().__dict__, "banned_turns": np.zeros((1, 2), dtype=np.int32)})


# -- reading restrictions from OpenStreetMap ---------------------------------

D = 0.001
# A crossroads: junction 1 with arms north (2), east (3), south (4), west (5).
NODES = {1: (51.5, -0.1), 2: (51.5 + D, -0.1), 3: (51.5, -0.1 + D), 4: (51.5 - D, -0.1),
         5: (51.5, -0.1 - D)}  # fmt: skip
WAYS = {10: [2, 1], 11: [1, 3], 12: [4, 1], 13: [1, 5]}  # north, east, south, west


def payload(*relations: dict) -> dict:
    elements: list[dict] = [
        {"type": "node", "id": i, "lat": la, "lon": lo} for i, (la, lo) in NODES.items()
    ]
    elements += [
        {"type": "way", "id": w, "nodes": nodes, "tags": {"highway": "residential"}}
        for w, nodes in WAYS.items()
    ]
    return {"elements": elements + list(relations)}


def restriction(kind: str, from_way: int, to_way: int, via: int = 1, **tags: str) -> dict:
    return {
        "type": "relation",
        "id": 99,
        "tags": {"type": "restriction", "restriction": kind, **tags},
        "members": [
            {"type": "way", "ref": from_way, "role": "from"},
            {"type": "node", "ref": via, "role": "via"},
            {"type": "way", "ref": to_way, "role": "to"},
        ],
    }


def bans_as_osm_ids(network: RoadNetwork) -> set[tuple[int, int, int]]:
    assert network.banned_turns is not None
    ids = network.node_ids
    return {tuple(int(ids[i]) for i in row) for row in network.banned_turns}  # type: ignore[misc]


def test_no_turn_bans_exactly_that_turn():
    from ridegraph.osm import network_from_overpass

    network = network_from_overpass(payload(restriction("no_left_turn", 12, 13)))
    assert bans_as_osm_ids(network) == {(4, 1, 5)}


def test_only_straight_on_bans_every_other_way_out():
    from ridegraph.osm import network_from_overpass

    network = network_from_overpass(payload(restriction("only_straight_on", 12, 10)))
    # From the south arm, east, west and turning back south are all banned.
    assert bans_as_osm_ids(network) == {(4, 1, 3), (4, 1, 5), (4, 1, 4)}


def test_restrictions_that_exempt_cars_are_ignored():
    from ridegraph.osm import network_from_overpass

    rel = restriction("no_right_turn", 12, 11, **{"except": "bus;motorcar"})
    assert bans_as_osm_ids(network_from_overpass(payload(rel))) == set()


def test_restrictions_through_a_stretch_of_road_are_skipped():
    from ridegraph.osm import network_from_overpass

    rel = restriction("no_right_turn", 12, 11)
    rel["members"][1] = {"type": "way", "ref": 10, "role": "via"}
    assert bans_as_osm_ids(network_from_overpass(payload(rel))) == set()


# -- obeying them --------------------------------------------------------------


def test_turn_graph_never_takes_a_banned_turn():
    from ridegraph.dijkstra import dijkstra
    from ridegraph.osm import network_from_overpass
    from ridegraph.turns import NO_PENALTIES, TurnGraph

    free = network_from_overpass(payload())
    banned = network_from_overpass(payload(restriction("no_left_turn", 12, 13)))
    idx = {int(osm): i for i, osm in enumerate(free.node_ids)}
    south, west = idx[4], idx[5]

    direct = TurnGraph(free, NO_PENALTIES)
    detour = TurnGraph(banned, NO_PENALTIES)
    allowed = dijkstra(direct.graph(), direct.start(south), direct.end(west))
    forced = dijkstra(detour.graph(), detour.start(south), detour.end(west))
    assert allowed.found and forced.found
    # Without the left turn the only legal way is to go round another arm and back.
    assert forced.seconds > allowed.seconds
    assert detour.num_manoeuvres == direct.num_manoeuvres - 1


def test_bans_leave_other_turns_alone():
    from ridegraph.dijkstra import dijkstra
    from ridegraph.osm import network_from_overpass
    from ridegraph.turns import NO_PENALTIES, TurnGraph

    free = TurnGraph(network_from_overpass(payload()), NO_PENALTIES)
    banned = TurnGraph(
        network_from_overpass(payload(restriction("no_left_turn", 12, 13))), NO_PENALTIES
    )
    for a in range(5):
        for b in range(5):
            if a == b:
                continue
            x = dijkstra(free.graph(), free.start(a), free.end(b)).seconds
            y = dijkstra(banned.graph(), banned.start(a), banned.end(b)).seconds
            assert y >= x


def test_eta_features_work_on_a_network_with_bans():
    import random

    from ridegraph.ch import ContractionHierarchy
    from ridegraph.eta import RouteContext, World, simulate_trips
    from ridegraph.router import Router, Scenario
    from ridegraph.synthetic import grid_network

    network = grid_network(14, 14, seed=3)
    out: dict[int, list[int]] = {}
    for u, v in zip(network.src.tolist(), network.dst.tolist(), strict=True):
        out.setdefault(u, []).append(v)
    turns = [(u, v, w) for u, v in zip(network.src.tolist(), network.dst.tolist(), strict=True)
             for w in out.get(v, [])]  # fmt: skip
    bans = np.array(random.Random(1).sample(turns, 40), dtype=np.int32)
    ctx = RouteContext(RoadNetwork(**{**network.__dict__, "banned_turns": bans}))
    hierarchies = {8.0: ContractionHierarchy.build(Router(ctx.network, Scenario(hour=8.0)).graph)}
    data = simulate_trips(ctx, World(ctx, seed=0), hierarchies, 200, seed=2)
    assert len(data.y) == 200
