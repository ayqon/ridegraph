from __future__ import annotations

import numpy as np

from ridegraph.graph import Graph
from ridegraph.osm import directions, is_drivable, network_from_overpass
from ridegraph.speeds import CLASS_INDEX


def node(i: int, lat: float, lon: float) -> dict:
    return {"type": "node", "id": i, "lat": lat, "lon": lon}


def way(i: int, nodes: list[int], **tags: str) -> dict:
    return {"type": "way", "id": i, "nodes": nodes, "tags": tags}


NODES = [node(i, 51.5 + i * 0.001, -0.1) for i in range(1, 6)]


def test_directions() -> None:
    assert directions({"highway": "residential"}) == (True, True)
    assert directions({"highway": "residential", "oneway": "yes"}) == (True, False)
    assert directions({"highway": "residential", "oneway": "-1"}) == (False, True)
    assert directions({"highway": "primary", "junction": "roundabout"}) == (True, False)
    assert directions({"highway": "motorway"}) == (True, False)
    assert directions({"highway": "motorway", "oneway": "no"}) == (True, True)


def test_filters_undrivable_ways() -> None:
    assert is_drivable({"highway": "residential"})
    assert not is_drivable({"highway": "footway"})
    assert not is_drivable({"highway": "residential", "access": "private"})
    assert not is_drivable({"highway": "service", "service": "parking_aisle"})
    assert not is_drivable({"highway": "pedestrian"})
    assert not is_drivable({"highway": "residential", "oneway": "reversible"})


def test_builds_edges_per_segment_and_direction() -> None:
    payload = {
        "elements": NODES
        + [
            way(10, [1, 2, 3], highway="residential"),  # two-way: 4 edges
            way(11, [3, 4], highway="primary", oneway="yes", maxspeed="20 mph"),
            way(12, [4, 1], highway="primary"),  # closes the loop: 2 edges
            way(13, [4, 5], highway="footway"),  # ignored
        ]
    }
    network = network_from_overpass(payload)
    assert network.num_nodes == 4
    assert network.num_edges == 4 + 1 + 2
    assert 5 not in network.node_ids
    primary = network.road_class == CLASS_INDEX["primary"]
    assert np.nanmin(network.maxspeed_kph[primary]) < 33  # 20 mph
    # Segment lengths come from coordinates: 0.001 deg latitude is ~111 m.
    assert 105 < float(network.length_m[0]) < 117
    Graph(network)  # builds without error


def test_drops_nodes_outside_largest_strong_component() -> None:
    payload = {
        "elements": NODES
        + [
            way(10, [1, 2, 3], highway="residential"),
            # A one-way stub out of the network that never comes back.
            way(11, [3, 4, 5], highway="residential", oneway="yes"),
        ]
    }
    network = network_from_overpass(payload)
    assert sorted(network.node_ids.tolist()) == [1, 2, 3]
    unfiltered = network_from_overpass(payload, largest_component=False)
    assert unfiltered.num_nodes == 5
