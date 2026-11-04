"""The JavaScript engine must agree with the Python one on exported data."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from ridegraph.router import Router, Scenario
from ridegraph.synthetic import grid_network
from ridegraph.web_export import PRESETS, export

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


def test_export_layout(tmp_path) -> None:
    network = grid_network(8, 8, seed=1)
    manifest = export(network, tmp_path, log=lambda m: None)
    assert manifest["nodes"] == network.num_nodes
    assert [p["id"] for p in manifest["presets"]] == [p.id for p in PRESETS]
    for entry in [manifest["graph"], *manifest["presets"]]:
        assert (tmp_path / entry["file"]).exists()
        offsets = [a["offset"] for a in entry["arrays"]]
        assert all(o % 8 == 0 for o in offsets) and offsets == sorted(offsets)
    assert json.loads((tmp_path / "manifest.json").read_text())["version"] == 1


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_javascript_engine_matches_python(tmp_path) -> None:
    network = grid_network(18, 18, seed=9, oneway_share=0.3)
    export(network, tmp_path, log=lambda m: None)
    rng = random.Random(5)
    cases = []
    for preset in PRESETS:
        for turns in (False, True):
            router = Router(network, Scenario(hour=preset.hour, turns=turns))
            for _ in range(15):
                s, t = rng.randrange(network.num_nodes), rng.randrange(network.num_nodes)
                trip = router.trip(s, t, "dijkstra")
                case = {
                    "preset": preset.id,
                    "turns": turns,
                    "s": s,
                    "t": t,
                    "seconds": trip.seconds if trip.found else None,
                }
                if turns and trip.found:
                    m = trip.manoeuvres
                    case["manoeuvres"] = {
                        "left": m["left"],
                        "right": m["right"],
                        "sharp": m["sharp left"] + m["sharp right"],
                        "uturn": m["u turn"],
                        "lights": m["traffic lights"],
                    }
                cases.append(case)
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    proc = subprocess.run(
        [NODE, str(ROOT / "web/test/check.mjs"), str(tmp_path), str(cases_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(proc.stdout)
    plain = sum(1 for c in cases if not c["turns"])
    # Plain cases run all five methods; with turn costs, both hierarchies sit out.
    assert summary["checked"] == plain * 5 + (len(cases) - plain) * 3


def test_river_lines_are_extracted() -> None:
    from ridegraph.web_export import river_lines

    payload = {
        "elements": [
            {"type": "node", "id": 1, "lat": 51.50, "lon": -0.12},
            {"type": "node", "id": 2, "lat": 51.51, "lon": -0.11},
            {"type": "way", "id": 9, "nodes": [1, 2], "tags": {"waterway": "river"}},
            {"type": "way", "id": 8, "nodes": [1, 2], "tags": {"highway": "primary"}},
        ]
    }
    assert river_lines(payload) == [[[51.5, -0.12], [51.51, -0.11]]]
