from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ridegraph import charges
from ridegraph.alternatives import alternatives, shortest
from ridegraph.dijkstra import one_to_all
from ridegraph.graph import Graph, csr_sorted
from ridegraph.synthetic import grid_network
from ridegraph.traffic import traffic_cost


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph(csr_sorted(grid_network(25, 25, seed=7, oneway_share=0.3)), traffic_cost(8.0))


def test_known_places_in_and_out_of_the_zone() -> None:
    inside = [
        (51.5117, -0.124),
        (51.5033, -0.1145),
        (51.5138, -0.0984),
    ]  # Covent Garden, Waterloo, Bank
    outside = [
        (51.5154, -0.1755),
        (51.5390, -0.1426),
        (51.4613, -0.1156),
    ]  # Paddington, Camden, Brixton
    lat, lon = np.array(inside + outside).T
    assert charges.inside(lat, lon).tolist() == [True] * 3 + [False] * 3


def test_charging_hours() -> None:
    assert charges.charging(8.0) and charges.charging(17.9)
    assert not charges.charging(3.0) and not charges.charging(18.0) and not charges.charging(None)
    assert not charges.charging(10.0, weekend=True) and charges.charging(13.0, weekend=True)


def test_boundary_roads_are_free() -> None:
    # A point on Euston Road (the northern boundary) is inside the polygon
    # line's tolerance, so it does not count as entering the zone.
    lat, lon = np.array([51.5262]), np.array([-0.1330])
    assert charges.distance_to_boundary_m(lat, lon)[0] < charges.BOUNDARY_MARGIN_M


def test_first_alternative_is_the_quickest_route(graph) -> None:
    rng = random.Random(1)
    for _ in range(5):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        alts = alternatives(graph, s, t)
        assert alts[0].seconds == pytest.approx(one_to_all(graph, s)[t])


def test_alternatives_are_distinct_and_not_much_slower(graph) -> None:
    alts = alternatives(graph, 0, graph.num_nodes - 1, count=3)
    assert len(alts) >= 2
    best = alts[0].seconds
    for a in alts:
        assert a.seconds <= 1.3 * best + 1e-9
        assert a.nodes[0] == 0 and a.nodes[-1] == graph.num_nodes - 1
        # Edges join up into a path.
        for e, (u, v) in zip(a.edges, zip(a.nodes, a.nodes[1:], strict=False), strict=True):
            assert graph.head[e] == v and graph.first_out[u] <= e < graph.first_out[u + 1]
    assert [a.seconds for a in alts] == sorted(a.seconds for a in alts)
    assert len({tuple(a.nodes) for a in alts}) == len(alts)


def test_avoiding_an_area_keeps_every_route_out(graph) -> None:
    zone = np.zeros(graph.num_nodes, dtype=bool)
    lat, lon = graph.network.lat, graph.network.lon
    zone[(lat > 51.522) & (lat < 51.530) & (lon > -0.108) & (lon < -0.097)] = True
    s = int(np.argmin(np.hypot(lat - 51.526, lon + 0.118)))
    t = int(np.argmin(np.hypot(lat - 51.526, lon + 0.087)))
    plain = alternatives(graph, s, t, zone=zone)
    avoid = alternatives(graph, s, t, zone=zone, avoid_zone=True)
    assert plain[0].enters_zone
    assert avoid and not any(a.enters_zone for a in avoid)
    assert avoid[0].seconds >= plain[0].seconds
    with pytest.raises(ValueError):
        alternatives(graph, int(np.flatnonzero(zone)[0]), t, zone=zone, avoid_zone=True)


def test_shortest_handles_closed_roads(graph) -> None:
    closed = [float("inf")] * graph.num_edges
    secs, edges = shortest(graph, 0, 5, closed)
    assert secs == float("inf") and edges == []


def test_browser_matches_python(tmp_path) -> None:
    from ridegraph.web_export import export

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    network = grid_network(18, 18, seed=8, oneway_share=0.3)
    manifest = export(network, tmp_path, log=lambda m: None)
    preset = next(p for p in manifest["presets"] if p["id"] == "0800")
    graph = Graph(csr_sorted(network), traffic_cost(preset["hour"]))
    zone = charges.zone_nodes(graph.network)
    rng = random.Random(3)
    cases = []
    for _ in range(6):
        s, t = rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)
        avoid = bool(not zone[s] and not zone[t] and zone.any())
        alts = alternatives(graph, s, t, zone=zone, avoid_zone=avoid)
        cases.append(
            {
                "s": s,
                "t": t,
                "avoid": avoid,
                "seconds": [a.seconds for a in alts],
                "zone": [a.enters_zone for a in alts],
            }
        )
    data = {"preset": "0800", "zone_count": int(zone.sum()), "cases": cases}
    (tmp_path / "cases.json").write_text(json.dumps(data))
    script = Path(__file__).resolve().parent.parent / "web/test/check_alternatives.mjs"
    proc = subprocess.run(
        [node, str(script), str(tmp_path), str(tmp_path / "cases.json")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
