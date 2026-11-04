from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ridegraph import live
from ridegraph.cch import CustomizableCH, nested_dissection_order
from ridegraph.dijkstra import one_to_all
from ridegraph.graph import Graph, csr_sorted
from ridegraph.synthetic import grid_network
from ridegraph.traffic import traffic_cost


@pytest.fixture(scope="module")
def network():
    return csr_sorted(grid_network(22, 22, seed=5, oneway_share=0.3))


@pytest.fixture(scope="module")
def cch(network):
    return CustomizableCH.build(network)


def _check_against_dijkstra(network, cch, seconds, pairs) -> None:
    graph = Graph(network, lambda net: seconds)
    for s, t in pairs:
        d, path = cch.query(s, t)
        ref = one_to_all(graph, s)[t]
        if not math.isfinite(ref):
            assert not math.isfinite(d) and path == []
            continue
        assert d == pytest.approx(ref)
        assert path[0] == s and path[-1] == t
        total = 0.0
        for a, b in zip(path, path[1:], strict=False):
            edges = np.flatnonzero((network.src == a) & (network.dst == b))
            total += float(seconds[edges].min())
        assert total == pytest.approx(ref)


def test_order_is_a_permutation_with_separators_last(network) -> None:
    rank = nested_dissection_order(network)
    assert sorted(rank.tolist()) == list(range(network.num_nodes))


def test_customization_gives_exact_routes_for_any_times(network, cch) -> None:
    rng = random.Random(0)
    pairs = [
        (rng.randrange(network.num_nodes), rng.randrange(network.num_nodes)) for _ in range(25)
    ]
    for hour in (3.0, 8.0):
        seconds = np.asarray(traffic_cost(hour)(network), dtype=np.float64)
        cch.customize(seconds)
        _check_against_dijkstra(network, cch, seconds, pairs)
    # Arbitrary new times, including closed roads, reuse the same topology.
    noisy = seconds * np.random.default_rng(1).uniform(0.5, 3.0, network.num_edges)
    noisy[np.random.default_rng(2).random(network.num_edges) < 0.05] = np.inf
    cch.customize(noisy)
    _check_against_dijkstra(network, cch, noisy, pairs)


def _payload(network):
    lat, lon = network.lat, network.lon
    row = np.flatnonzero(np.isclose(lat, np.unique(lat)[10]))
    row = row[np.argsort(lon[row])][3:9]
    line = [[float(lon[v]), float(lat[v])] for v in row]
    return {
        "$type": "response",
        "segments": [
            {
                "streetName": "Test Street",
                "closure": "Closed",
                "severity": "Serious",
                "lineString": json.dumps(line),
            },
            {
                "streetName": "Slow Road",
                "closure": "Open",
                "severity": "Moderate",
                "lineString": "LINESTRING (" + ", ".join(f"{x} {y}" for x, y in line[:2]) + ")",
            },
            {
                "streetName": "Elsewhere",
                "closure": "Closed",
                "severity": "Serious",
                "lineString": "[[0.5,52.0],[0.51,52.0]]",
            },
        ],
    }, row


def test_parse_and_match(network) -> None:
    payload, row = _payload(network)
    stretches = live.parse(payload)
    assert [d.street for d in stretches] == ["Test Street", "Slow Road", "Elsewhere"]
    assert stretches[0].closed and not stretches[1].closed
    factor, hits = live.match(network, stretches)
    assert hits[0] > 0 and hits[2] == 0
    on_street = {(int(a), int(b)) for a, b in zip(row, row[1:], strict=False)}
    closed = {(int(network.src[e]), int(network.dst[e])) for e in np.flatnonzero(np.isinf(factor))}
    assert on_street & closed  # the closed street's own roads are closed
    assert live.summary(stretches, hits) == {
        "stretches": 3,
        "on_this_map": 2,
        "closed": 1,
        "slowed": 1,
    }


def test_closures_reroute_through_customization(network, cch) -> None:
    payload, row = _payload(network)
    factor, _ = live.match(network, live.parse(payload))
    seconds = np.asarray(traffic_cost(13.0)(network), dtype=np.float64)
    cch.customize(seconds * factor)
    rng = random.Random(4)
    pairs = [(int(row[0]), int(row[-1]))] + [
        (rng.randrange(network.num_nodes), rng.randrange(network.num_nodes)) for _ in range(10)
    ]
    _check_against_dijkstra(network, cch, seconds * factor, pairs)
    _, path = cch.query(int(row[0]), int(row[-1]))
    closed = np.isinf(factor)
    for a, b in zip(path, path[1:], strict=False):
        edges = np.flatnonzero((network.src == a) & (network.dst == b))
        assert not closed[edges].all()


def test_browser_matches_python(tmp_path, network) -> None:
    from ridegraph.web_export import export

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    payload, _ = _payload(network)
    manifest = export(
        network, tmp_path, tfl=payload, tfl_fetched="2026-11-03T00:00:00+00:00", log=lambda m: None
    )
    assert manifest["live"]["snapshot"] == "tfl.json"
    preset = next(p for p in manifest["presets"] if p["id"] == "1300")
    factor, _ = live.match(network, live.parse(payload))
    seconds = np.asarray(traffic_cost(preset["hour"])(network), dtype=np.float64)
    cch = CustomizableCH.build(network)
    rng = random.Random(7)
    pairs = [
        (rng.randrange(network.num_nodes), rng.randrange(network.num_nodes)) for _ in range(20)
    ]
    expected = {}
    for name, w in (("plain", seconds), ("live", seconds * factor)):
        cch.customize(w)
        expected[name] = [cch.query(s, t)[0] for s, t in pairs]
    cases = {
        "preset": "1300",
        "pairs": pairs,
        "expected": {k: [x if math.isfinite(x) else None for x in v] for k, v in expected.items()},
        "factor": [x if math.isfinite(x) else None for x in factor.tolist()],
        "payload": payload,
    }
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    script = Path(__file__).resolve().parent.parent / "web/test/check_cch_live.mjs"
    proc = subprocess.run(
        [node, str(script), str(tmp_path), str(tmp_path / "cases.json")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
