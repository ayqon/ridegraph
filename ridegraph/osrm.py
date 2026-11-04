"""A head-to-head with OSRM, the open-source routing engine used in production.

Agreeing with yourself proves little: every algorithm here is tested against
Dijkstra, but all of them share this package's map parsing and speed model.
OSRM (Open Source Routing Machine) shares neither. It parses OpenStreetMap
with its own C++ code and routes with Multi-Level Dijkstra, the engine
behind many commercial map products.

To compare like with like, OSRM is given the same roads and the same speeds:

* ``write_osm_xml`` writes the ways this package treats as drivable, with
  their tags, as OSM XML for ``osrm-extract``;
* ``write_segment_speeds`` writes this package's free-flow speed for every
  road segment as an OSRM *segment speed file*, which ``osrm-customize``
  loads in place of OSRM's own speed guesses;
* the CI workflow switches off OSRM's turn and traffic-light penalties, since
  the free-flow router here has none.

``compare`` then asks both engines the same random questions and reports how
often the travel times agree, how much of the two routes coincide, and how
long each takes to answer. Differences that remain come from the parts that
are genuinely independent: OSM parsing, which ways each engine accepts, how
coordinates snap to roads, and OSRM storing times in tenths of a second.
"""

from __future__ import annotations

import json
import math
import random
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import numpy as np

from .graph import RoadNetwork, free_flow_cost
from .osm import is_drivable


def write_osm_xml(payload: Mapping[str, Any], path: Path) -> tuple[int, int]:
    """Drivable ways and the nodes they use, as OSM XML; returns (nodes, ways)."""
    elements = payload.get("elements", [])
    ways = [el for el in elements if el.get("type") == "way" and is_drivable(el.get("tags", {}))]
    used = {int(r) for w in ways for r in w.get("nodes", [])}
    nodes = [el for el in elements if el.get("type") == "node" and int(el["id"]) in used]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<osm version="0.6" generator="ridegraph">\n')
        for n in nodes:
            tags = n.get("tags", {})
            attrs = f'id="{int(n["id"])}" version="1" lat="{n["lat"]:.7f}" lon="{n["lon"]:.7f}"'
            if not tags:
                fh.write(f"  <node {attrs}/>\n")
                continue
            fh.write(f"  <node {attrs}>\n")
            for k, v in tags.items():
                fh.write(f"    <tag k={quoteattr(str(k))} v={quoteattr(str(v))}/>\n")
            fh.write("  </node>\n")
        for w in ways:
            fh.write(f'  <way id="{int(w["id"])}" version="1">\n')
            for r in w.get("nodes", []):
                fh.write(f'    <nd ref="{int(r)}"/>\n')
            for k, v in w.get("tags", {}).items():
                fh.write(f"    <tag k={quoteattr(str(k))} v={quoteattr(str(v))}/>\n")
            fh.write("  </way>\n")
        fh.write("</osm>\n")
    return len(nodes), len(ways)


def write_segment_speeds(network: RoadNetwork, path: Path) -> int:
    """``from_osm_id,to_osm_id,speed_kmh`` for every directed road segment."""
    seconds = np.asarray(free_flow_cost(network), dtype=np.float64)
    kmh = network.length_m.astype(np.float64) / np.maximum(seconds, 1e-9) * 3.6
    ids = network.node_ids
    with open(path, "w", encoding="utf-8") as fh:
        for a, b, s in zip(
            ids[network.src].tolist(), ids[network.dst].tolist(), kmh.tolist(), strict=True
        ):
            fh.write(f"{a},{b},{s:.4f}\n")
    return network.num_edges


def osrm_route(base_url: str, a: tuple[float, float], b: tuple[float, float]) -> dict[str, Any]:
    """One OSRM route request (coordinates are (lat, lon))."""
    url = (
        f"{base_url.rstrip('/')}/route/v1/driving/{a[1]:.7f},{a[0]:.7f};{b[1]:.7f},{b[0]:.7f}"
        "?overview=false&annotations=nodes&steps=false"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - local OSRM
        return json.loads(resp.read())


def compare(
    network: RoadNetwork,
    route: Callable[[int, int], tuple[float, list[int]]],
    ask_osrm: Callable[[tuple[float, float], tuple[float, float]], dict[str, Any]],
    *,
    queries: int = 1000,
    seed: int = 7,
    min_km: float = 1.0,
) -> dict[str, Any]:
    """Ask both engines the same questions and summarise how they agree."""
    rng = random.Random(seed)
    lat, lon, ids = network.lat, network.lon, network.node_ids
    rows: list[dict[str, Any]] = []
    ours_ms, theirs_ms = [], []
    attempts = 0
    while len(rows) < queries and attempts < queries * 5:
        attempts += 1
        s, t = rng.randrange(network.num_nodes), rng.randrange(network.num_nodes)
        if s == t:
            continue
        dy = (lat[s] - lat[t]) * 111.32
        dx = (lon[s] - lon[t]) * 111.32 * math.cos(math.radians(lat[s]))
        if math.hypot(dx, dy) < min_km:
            continue
        started = time.perf_counter()
        secs, path = route(s, t)
        ours_ms.append((time.perf_counter() - started) * 1000)
        a, b = (float(lat[s]), float(lon[s])), (float(lat[t]), float(lon[t]))
        started = time.perf_counter()
        answer = ask_osrm(a, b)
        theirs_ms.append((time.perf_counter() - started) * 1000)
        if answer.get("code") != "Ok" or not math.isfinite(secs):
            rows.append({"ok": False})
            continue
        r = answer["routes"][0]
        osrm_nodes = {n for leg in r["legs"] for n in leg["annotation"]["nodes"]}
        our_nodes = {int(ids[v]) for v in path}
        union = our_nodes | osrm_nodes
        rows.append(
            {
                "ok": True,
                "ours_s": secs,
                "osrm_s": float(r["duration"]),
                "osrm_m": float(r["distance"]),
                "overlap": len(our_nodes & osrm_nodes) / len(union) if union else 1.0,
            }
        )
    return summarise(rows, ours_ms, theirs_ms)


def summarise(rows: list[dict[str, Any]], ours_ms: list[float], theirs_ms: list[float]) -> dict:
    ok = [r for r in rows if r["ok"]]
    diff = np.array([(r["osrm_s"] - r["ours_s"]) / r["ours_s"] for r in ok]) if ok else np.zeros(0)
    overlap = np.array([r["overlap"] for r in ok]) if ok else np.zeros(0)

    def pct(mask) -> float:
        return float(100 * np.mean(mask)) if len(diff) else 0.0

    return {
        "queries": len(rows),
        "both_found": len(ok),
        "time_within_1pct": pct(np.abs(diff) <= 0.01),
        "time_within_5pct": pct(np.abs(diff) <= 0.05),
        "median_abs_time_diff_pct": float(100 * np.median(np.abs(diff))) if len(diff) else 0.0,
        "mean_time_diff_pct": float(100 * np.mean(diff)) if len(diff) else 0.0,
        "same_route_pct": pct(overlap >= 0.95),
        "median_overlap_pct": float(100 * np.median(overlap)) if len(overlap) else 0.0,
        "ours_median_ms": float(np.median(ours_ms)) if ours_ms else 0.0,
        "osrm_median_ms": float(np.median(theirs_ms)) if theirs_ms else 0.0,
    }


def markdown(summary: dict[str, Any]) -> str:
    s = summary
    return "\n".join(
        [
            f"| ridegraph vs OSRM, {s['both_found']:,} random central London trips | |",
            "|---|---:|",
            f"| Travel time within 1% of OSRM's | {s['time_within_1pct']:.1f}% |",
            f"| Travel time within 5% of OSRM's | {s['time_within_5pct']:.1f}% |",
            f"| Median time difference | {s['median_abs_time_diff_pct']:.2f}% |",
            f"| Same route (95%+ of junctions shared) | {s['same_route_pct']:.1f}% |",
            f"| Median share of junctions in common | {s['median_overlap_pct']:.1f}% |",
            f"| Median query, ridegraph CH in Python | {s['ours_median_ms']:.2f} ms |",
            f"| Median query, OSRM over HTTP | {s['osrm_median_ms']:.2f} ms |",
        ]
    )
