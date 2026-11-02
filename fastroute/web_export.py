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
from .graph import Graph, RoadNetwork
from .router import Router, Scenario
from .speeds import ROAD_CLASSES
from .traffic import DEFAULT_PROFILE

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
    graph = Graph(network)
    order = np.argsort(network.src, kind="stable")
    return {
        # float64: A*'s heuristic must not overshoot by coordinate rounding.
        "lat": network.lat.astype(np.float64),
        "lon": network.lon.astype(np.float64),
        "first_out": np.array(graph.first_out, dtype=np.int32),
        "head": np.array(graph.head, dtype=np.int32),
        "freeflow_s": np.array(graph.weight, dtype=np.float64),
        "length_m": network.length_m[order].astype(np.float32),
        "road_class": network.road_class[order].astype(np.uint8),
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


def export(
    network: RoadNetwork,
    out_dir: Path,
    *,
    name: str = "Central London",
    presets: tuple[Preset, ...] = PRESETS,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob, layout = _pack(graph_arrays(network))
    size = _write(out_dir / "graph.bin.gz", blob)
    log(f"  graph: {network.num_nodes:,} nodes, {network.num_edges:,} edges, {size / 1e6:.2f} MB")

    preset_entries = []
    for preset in presets:
        started = time.perf_counter()
        router = Router(network, Scenario(hour=preset.hour))
        ch = ContractionHierarchy.build(router.graph)
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
        "presets": preset_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
