from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from ridegraph.ch import ContractionHierarchy
from ridegraph.graph import Graph, free_flow_cost
from ridegraph.osm import network_from_overpass
from ridegraph.osrm import compare, markdown, write_osm_xml, write_segment_speeds

PAYLOAD = {
    "elements": [
        {"type": "node", "id": 1, "lat": 51.5000, "lon": -0.1000},
        {
            "type": "node",
            "id": 2,
            "lat": 51.5010,
            "lon": -0.1000,
            "tags": {"highway": "traffic_signals"},
        },
        {"type": "node", "id": 3, "lat": 51.5010, "lon": -0.0985},
        {"type": "node", "id": 4, "lat": 51.5000, "lon": -0.0985},
        {"type": "node", "id": 9, "lat": 51.6, "lon": -0.2},
        {
            "type": "way",
            "id": 10,
            "nodes": [1, 2, 3],
            "tags": {"highway": "primary", "name": "A & B"},
        },
        {
            "type": "way",
            "id": 11,
            "nodes": [3, 4, 1],
            "tags": {"highway": "residential", "oneway": "yes"},
        },
        {"type": "way", "id": 12, "nodes": [1, 9], "tags": {"highway": "footway"}},
    ]
}


def test_osm_xml_keeps_drivable_ways_and_their_nodes(tmp_path) -> None:
    nodes, ways = write_osm_xml(PAYLOAD, tmp_path / "map.osm")
    assert (nodes, ways) == (4, 2)
    root = ET.parse(tmp_path / "map.osm").getroot()
    assert {n.get("id") for n in root.iter("node")} == {"1", "2", "3", "4"}
    names = [t.get("v") for t in root.iter("tag") if t.get("k") == "name"]
    assert names == ["A & B"]  # escaped and read back intact


def test_segment_speeds_reproduce_free_flow_times(tmp_path) -> None:
    network = network_from_overpass(PAYLOAD)
    write_segment_speeds(network, tmp_path / "speeds.csv")
    rows = [line.split(",") for line in (tmp_path / "speeds.csv").read_text().splitlines()]
    assert len(rows) == network.num_edges
    seconds = free_flow_cost(network)
    for (a, b, kmh), e in zip(rows, range(network.num_edges), strict=True):
        assert (
            int(a) == network.node_ids[network.src[e]]
            and int(b) == network.node_ids[network.dst[e]]
        )
        assert network.length_m[e] / (float(kmh) / 3.6) == pytest.approx(seconds[e], rel=1e-4)


def test_compare_summarises_agreement() -> None:
    network = network_from_overpass(PAYLOAD)
    ch = ContractionHierarchy.build(Graph(network))
    ids = network.node_ids

    def route(s, t):
        r = ch.query(s, t)
        return r.seconds, r.path

    def fake_osrm(a, b):
        s = int(
            min(
                range(network.num_nodes),
                key=lambda i: (network.lat[i] - a[0]) ** 2 + (network.lon[i] - a[1]) ** 2,
            )
        )
        t = int(
            min(
                range(network.num_nodes),
                key=lambda i: (network.lat[i] - b[0]) ** 2 + (network.lon[i] - b[1]) ** 2,
            )
        )
        r = ch.query(s, t)
        return {
            "code": "Ok",
            "routes": [
                {
                    "duration": r.seconds * 1.005,
                    "distance": 1.0,
                    "legs": [{"annotation": {"nodes": [int(ids[v]) for v in r.path]}}],
                }
            ],
        }

    summary = compare(network, route, fake_osrm, queries=20, min_km=0.0)
    assert summary["both_found"] == summary["queries"] == 20
    assert summary["time_within_1pct"] == 100.0 and summary["same_route_pct"] == 100.0
    assert "within 1%" in markdown(summary)
    json.dumps(summary)
