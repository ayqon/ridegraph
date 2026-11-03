"""Export a road network and its hierarchies for the in-browser demo.

The browser gets exactly what the Python router uses, in a compact form:

* ``graph.bin.gz``: node coordinates and the CSR edge arrays, with free-flow
  seconds per edge. The JavaScript engine derives rush-hour weights itself
  from the traffic profile in the manifest, using the same formula as
  ``traffic.py``, so Dijkstra, bidirectional search and A* work at any time.
* ``ch-HHMM.bin.gz``: one Contraction Hierarchy per time-of-day preset, since
  a hierarchy is built for one set of weights. They load on demand.
* ``manifest.json``: array layouts, the traffic profile and the presets.

Arrays are little-endian and aligned to 8 bytes so the browser can view them
in place as typed arrays without copying.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ch import ContractionHierarchy
from .graph import Graph, RoadNetwork, csr_sorted
from .router import Router, Scenario
from .speeds import ROAD_CLASSES
from .traffic import DEFAULT_PROFILE
from .turns import UK_PENALTIES, Turn, TurnGraph, TurnPenalties, penalty_seconds

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    hour: float


PRESETS: tuple[Preset, ...] = (
    Preset("0300", "Night", 3.0),
    Preset("0800", "Morning peak", 8.0),
    Preset("1300", "Midday", 13.0),
    Preset("1730", "Evening peak", 17.5),
)


def _pack(arrays: dict[str, np.ndarray]) -> tuple[bytes, list[dict[str, object]]]:
    """Concatenate arrays into one 8-byte-aligned little-endian blob."""
    chunks: list[bytes] = []
    layout: list[dict[str, object]] = []
    offset = 0
    for name, array in arrays.items():
        data = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<"), copy=False)
        raw = data.tobytes()
        layout.append(
            {"name": name, "dtype": data.dtype.name, "offset": offset, "length": int(data.size)}
        )
        chunks.append(raw)
        offset += len(raw)
        pad = (-offset) % 8
        if pad:
            chunks.append(b"\0" * pad)
            offset += pad
    return b"".join(chunks), layout


def _write(path: Path, blob: bytes) -> int:
    path.write_bytes(gzip.compress(blob, compresslevel=9, mtime=0))
    return path.stat().st_size


def graph_arrays(network: RoadNetwork) -> dict[str, np.ndarray]:
    """Graph arrays for a network already in CSR order (see ``csr_sorted``)."""
    graph = Graph(network)
    turns = TurnGraph(network)
    order = np.arange(network.num_edges)
    return {
        # float64: A*'s heuristic must not overshoot by coordinate rounding.
        "lat": network.lat.astype(np.float64),
        "lon": network.lon.astype(np.float64),
        "first_out": np.array(graph.first_out, dtype=np.int32),
        "head": np.array(graph.head, dtype=np.int32),
        "freeflow_s": np.array(graph.weight, dtype=np.float64),
        "length_m": network.length_m[order].astype(np.float32),
        "road_class": network.road_class[order].astype(np.uint8),
        # Turn kind of every manoeuvre (segment followed by a segment leaving its
        # head), in TurnGraph order; the browser rebuilds the turn graph from it.
        "turn_kind": turns.turn_kind[: turns.num_manoeuvres].astype(np.uint8),
        "signals": np.asarray(network.traffic_signals, dtype=np.uint8),
    }


def ch_arrays(ch: ContractionHierarchy) -> dict[str, np.ndarray]:
    return {
        "rank": np.array(ch.rank, dtype=np.int32),
        "up_first": np.array(ch.up_first, dtype=np.int32),
        "up_head": np.array(ch.up_head, dtype=np.int32),
        "up_weight": np.array(ch.up_weight, dtype=np.float32),
        "up_middle": np.array(ch.up_middle, dtype=np.int32),
        "down_first": np.array(ch.down_first, dtype=np.int32),
        "down_tail": np.array(ch.down_tail, dtype=np.int32),
        "down_weight": np.array(ch.down_weight, dtype=np.float32),
        "down_middle": np.array(ch.down_middle, dtype=np.int32),
    }


def turn_penalty_table(penalties: TurnPenalties = UK_PENALTIES) -> dict[str, object]:
    """Seconds per turn kind (indexed by ``Turn``) and per set of traffic lights."""
    kinds = np.arange(len(Turn), dtype=np.int8)
    return {
        "kinds": [t.name.lower() for t in Turn],
        "seconds": penalty_seconds(kinds, penalties).tolist(),
        "signals_s": penalties.traffic_signals_s,
    }


def river_lines(payload: dict) -> list[list[list[float]]]:
    """River centrelines from an Overpass response, as ``[[lat, lon], ...]`` polylines."""
    coords: dict[int, tuple[float, float]] = {}
    rivers = []
    for el in payload.get("elements", []):
        if el.get("type") == "node":
            coords[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
        elif el.get("type") == "way" and el.get("tags", {}).get("waterway") == "river":
            rivers.append(el)
    lines = []
    for way in rivers:
        line = [
            [round(coords[n][0], 6), round(coords[n][1], 6)]
            for n in way.get("nodes", [])
            if n in coords
        ]
        if len(line) >= 2:
            lines.append(line)
    return lines


def export(
    network: RoadNetwork,
    out_dir: Path,
    *,
    raw: dict | None = None,
    name: str = "Central London",
    eta_trips: int = 0,
    dispatch_days: int = 0,
    presets: tuple[Preset, ...] = PRESETS,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    network = csr_sorted(network)
    blob, layout = _pack(graph_arrays(network))
    size = _write(out_dir / "graph.bin.gz", blob)
    log(f"  graph: {network.num_nodes:,} nodes, {network.num_edges:,} edges, {size / 1e6:.2f} MB")

    preset_entries = []
    hierarchies: dict[float, ContractionHierarchy] = {}
    for preset in presets:
        started = time.perf_counter()
        router = Router(network, Scenario(hour=preset.hour))
        ch = ContractionHierarchy.build(router.graph)
        hierarchies[preset.hour] = ch
        ch_blob, ch_layout = _pack(ch_arrays(ch))
        filename = f"ch-{preset.id}.bin.gz"
        size = _write(out_dir / filename, ch_blob)
        log(
            f"  {preset.label} ({preset.id}): {ch.num_shortcuts:,} shortcuts, "
            f"{size / 1e6:.2f} MB, {time.perf_counter() - started:.1f}s"
        )
        preset_entries.append(
            {
                "id": preset.id,
                "label": preset.label,
                "hour": preset.hour,
                "file": filename,
                "arrays": ch_layout,
                "shortcuts": ch.num_shortcuts,
            }
        )

    manifest: dict[str, object] = {
        "version": FORMAT_VERSION,
        "name": name,
        "nodes": network.num_nodes,
        "edges": network.num_edges,
        "bbox": [
            float(network.lat.min()),
            float(network.lon.min()),
            float(network.lat.max()),
            float(network.lon.max()),
        ],
        "classes": list(ROAD_CLASSES),
        "traffic": {
            "weekday": list(DEFAULT_PROFILE.weekday),
            "weekend": list(DEFAULT_PROFILE.weekend),
            "sensitivity": list(DEFAULT_PROFILE.sensitivity),
        },
        "graph": {"file": "graph.bin.gz", "arrays": layout},
        "turns": turn_penalty_table(),
        "rivers": river_lines(raw) if raw else [],
        "presets": preset_entries,
        "charge": charge_info(),
    }
    if eta_trips:
        from .eta import train

        log(f"  training the ETA model on {eta_trips:,} simulated trips")
        started = time.perf_counter()
        result = train(network, trips=eta_trips, hierarchies=hierarchies, log=log)
        (out_dir / "eta-model.json").write_text(
            json.dumps(result.model_json, separators=(",", ":"))
        )
        best = result.metrics["Gradient-boosted trees"]
        log(
            f"  ETA model: {len(result.model_json['trees'])} trees, "
            f"{best['mape_pct']:.1f}% mean error on held-out trips, "
            f"{time.perf_counter() - started:.0f}s"
        )
        from .eta import metrics_markdown

        log(metrics_markdown(result.metrics))
        log(
            "  most useful features: " + ", ".join(f"{n} {v:.3f}" for n, v in result.importance[:8])
        )
        iv = result.interval
        log(
            f"  arrival window: {iv['coverage_pct']:.1f}% of held-out trips inside the "
            f"10th-90th percentile range (target 80%; {iv['uncalibrated_coverage_pct']:.1f}% "
            f"before calibration), mean width {iv['mean_width_pct']:.0f}%"
        )
        log(
            f"  learned slowdown map vs hidden hotspots at 08:30: r = {iv['learned_hotspot_r']:.2f}"
        )
        manifest["eta"] = {
            "file": "eta-model.json",
            "trips": result.trips,
            "metrics": result.metrics,
            "interval": result.interval,
            "importance": result.importance[:8],
        }
    if dispatch_days:
        manifest["dispatch"] = export_dispatch(network, hierarchies, out_dir, dispatch_days, log)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def charge_info() -> dict[str, object]:
    from . import charges

    return {
        "name": "Congestion Charge",
        "daily_gbp": charges.DAILY_CHARGE_GBP,
        "weekday_hours": list(charges.WEEKDAY_HOURS),
        "weekend_hours": list(charges.WEEKEND_HOURS),
        "margin_m": charges.BOUNDARY_MARGIN_M,
        "zone": [list(p) for p in charges.ZONE],
    }


def export_dispatch(
    network: RoadNetwork,
    hierarchies: dict[float, ContractionHierarchy],
    out_dir: Path,
    days: int,
    log: Callable[[str], None],
) -> dict[str, object]:
    """Simulate ride dispatch on the morning peak and save a replay for the map."""
    from .demand import DemandWorld, forecast_markdown, train_forecaster
    from .dispatch import DispatchConfig, average, compare, replay_json, results_markdown

    started = time.perf_counter()
    config = DispatchConfig()
    ch = hierarchies[8.0]  # the morning-peak hierarchy the map also uses
    world = DemandWorld(network)
    log("  training the demand forecast")
    forecast = train_forecaster(world)
    log(forecast_markdown(forecast.metrics))
    results = [
        compare(world, 1000 + k, ch, config, forecast.model_json, history=forecast.historical_mean)
        for k in range(days)
    ]
    summary = average(results)
    log(f"  dispatch over {days} simulated mornings, {config.drivers} drivers:")
    log(results_markdown(summary))
    log(
        "  repositioning minutes per driver: "
        + ", ".join(f"{p} {m['repositioning_min_per_driver']:.1f}" for p, m in summary.items())
    )
    (out_dir / "dispatch.json").write_text(
        json.dumps(replay_json(results[0], config), separators=(",", ":"))
    )
    log(f"  dispatch export {time.perf_counter() - started:.0f}s")
    return {
        "file": "dispatch.json",
        "preset": "0800",
        "days": days,
        "summary": summary,
        "forecast": forecast.metrics,
        "centres": [
            {"name": c.name, "kind": c.kind, "x_km": c.x_km, "y_km": c.y_km} for c in world.centres
        ],
    }
