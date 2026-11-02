"""Road network storage and the weighted routing graph built from it.

Two layers, kept separate on purpose:

* ``RoadNetwork`` is the raw map: where nodes are, which segments join them,
  how long each segment is and what kind of road it is. It is what the data
  pipeline produces and what gets saved to disk.
* ``Graph`` is what algorithms run on: a compact directed graph whose edge
  weights are travel times. Changing the cost model (traffic, time of day)
  means building a new ``Graph`` from the same ``RoadNetwork``.

Edges are stored in CSR (compressed sparse row) form: all outgoing edges of
node ``u`` live in ``head[first_out[u]:first_out[u + 1]]``. It is the layout
production routing engines use, being cache friendly and trivially serialised.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .speeds import DEFAULT_SPEED_KPH, LIMIT_REALISATION, ROAD_CLASSES

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class RoadNetwork:
    """Raw directed road segments, before any cost model is applied.

    Two-way streets appear as two edges, one per direction.
    ``maxspeed_kph`` is NaN where OSM has no usable limit. ``traffic_signals``
    marks nodes with lights; networks saved before it existed load as all-False.
    """

    node_ids: np.ndarray  # int64, OSM node ids
    lat: np.ndarray  # float64
    lon: np.ndarray  # float64
    src: np.ndarray  # int32, index into node arrays
    dst: np.ndarray  # int32
    length_m: np.ndarray  # float32
    road_class: np.ndarray  # int8, index into ROAD_CLASSES
    maxspeed_kph: np.ndarray  # float32
    traffic_signals: np.ndarray | None = None  # bool per node

    def __post_init__(self) -> None:
        n, m = len(self.node_ids), len(self.src)
        if self.traffic_signals is None:
            object.__setattr__(self, "traffic_signals", np.zeros(n, dtype=bool))
        elif len(self.traffic_signals) != n:
            raise ValueError("traffic_signals must have one entry per node")
        if not (len(self.lat) == len(self.lon) == n):
            raise ValueError("node arrays must have equal length")
        for name in ("dst", "length_m", "road_class", "maxspeed_kph"):
            if len(getattr(self, name)) != m:
                raise ValueError(f"edge array {name!r} has wrong length")
        if m and (
            min(self.src.min(), self.dst.min()) < 0 or max(self.src.max(), self.dst.max()) >= n
        ):
            raise ValueError("edge endpoint out of range")

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.src)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            node_ids=self.node_ids,
            lat=self.lat,
            lon=self.lon,
            src=self.src,
            dst=self.dst,
            length_m=self.length_m,
            road_class=self.road_class,
            maxspeed_kph=self.maxspeed_kph,
            traffic_signals=self.traffic_signals,
        )

    @classmethod
    def load(cls, path: str | Path) -> RoadNetwork:
        with np.load(path) as data:
            return cls(**{key: data[key] for key in data.files})


def csr_sorted(network: RoadNetwork) -> RoadNetwork:
    """The same network with edges stored in CSR order (stably sorted by source).

    After this, an edge's index in the network equals its index in ``Graph``,
    so edge-indexed data (turn graphs, exports) lines up without remapping.
    """
    order = np.argsort(network.src, kind="stable")
    return RoadNetwork(
        node_ids=network.node_ids,
        lat=network.lat,
        lon=network.lon,
        src=network.src[order],
        dst=network.dst[order],
        length_m=network.length_m[order],
        road_class=network.road_class[order],
        maxspeed_kph=network.maxspeed_kph[order],
        traffic_signals=network.traffic_signals,
    )


EdgeCost = Callable[[RoadNetwork], np.ndarray]


def free_flow_speeds_kph(network: RoadNetwork) -> np.ndarray:
    """Free-flow speed per edge, vectorised (same rule as ``speeds.free_flow_speed_kph``)."""
    defaults = np.array([DEFAULT_SPEED_KPH[c] for c in ROAD_CLASSES], dtype=np.float64)
    limits = network.maxspeed_kph.astype(np.float64)
    from_limit = np.maximum(5.0, limits * LIMIT_REALISATION)
    return np.where(np.isnan(limits), defaults[network.road_class], from_limit)


def free_flow_cost(network: RoadNetwork) -> np.ndarray:
    """Seconds per edge at free-flow speed (no traffic)."""
    return network.length_m.astype(np.float64) / (free_flow_speeds_kph(network) / 3.6)


class Graph:
    """Directed graph in CSR form with travel-time weights in seconds."""

    __slots__ = (
        "network",
        "first_out",
        "head",
        "weight",
        "length_m",
        "first_in",
        "tail",
        "in_weight",
        "max_speed_mps",
    )

    def __init__(self, network: RoadNetwork, cost: EdgeCost = free_flow_cost) -> None:
        weights = np.asarray(cost(network), dtype=np.float64)
        if weights.shape != (network.num_edges,):
            raise ValueError("cost function must return one weight per edge")
        if network.num_edges and weights.min() < 0:
            raise ValueError("edge weights must be non-negative")

        order = np.argsort(network.src, kind="stable")
        counts = np.bincount(network.src, minlength=network.num_nodes)
        first_out = np.zeros(network.num_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=first_out[1:])

        self.network = network
        # Hot loops index these per edge; plain Python lists are several times
        # faster than numpy scalar access, so the arrays are converted once here.
        self.first_out: list[int] = first_out.tolist()
        self.head: list[int] = network.dst[order].tolist()
        self.weight: list[float] = weights[order].tolist()
        self.length_m: list[float] = network.length_m[order].astype(np.float64).tolist()

        # The same edges indexed by their head, for searches that run backwards
        # from the target: incoming edges of v are tail[first_in[v]:first_in[v + 1]].
        rorder = np.argsort(network.dst, kind="stable")
        first_in = np.zeros(network.num_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(network.dst, minlength=network.num_nodes), out=first_in[1:])
        self.first_in: list[int] = first_in.tolist()
        self.tail: list[int] = network.src[rorder].tolist()
        self.in_weight: list[float] = weights[rorder].tolist()

        # Fastest speed anywhere in the graph, which is what keeps A*'s
        # straight-line heuristic admissible.
        lengths = network.length_m.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            speeds = np.where(weights > 0, lengths / weights, 0.0)
        self.max_speed_mps: float = float(speeds.max()) if len(speeds) else 0.0

    @property
    def num_nodes(self) -> int:
        return len(self.first_out) - 1

    @property
    def num_edges(self) -> int:
        return len(self.head)

    def out_edges(self, u: int) -> Iterator[tuple[int, float]]:
        """Yield ``(neighbour, seconds)`` for each edge leaving ``u``."""
        for e in range(self.first_out[u], self.first_out[u + 1]):
            yield self.head[e], self.weight[e]

    def coords(self, u: int) -> tuple[float, float]:
        return float(self.network.lat[u]), float(self.network.lon[u])

    def nearest_node(self, lat: float, lon: float) -> int:
        """Index of the node closest to a coordinate (equirectangular, fine at city scale)."""
        net = self.network
        dx = np.radians(net.lon - lon) * math.cos(math.radians(lat))
        dy = np.radians(net.lat - lat)
        return int(np.argmin(dx * dx + dy * dy))

    def edge_between(self, u: int, v: int) -> int:
        """Index of the cheapest edge ``u -> v``; raises if none exists."""
        best, best_w = -1, math.inf
        for e in range(self.first_out[u], self.first_out[u + 1]):
            if self.head[e] == v and self.weight[e] < best_w:
                best, best_w = e, self.weight[e]
        if best < 0:
            raise KeyError(f"no edge {u} -> {v}")
        return best

    def path_length_m(self, path: list[int]) -> float:
        return sum(self.length_m[self.edge_between(u, v)] for u, v in pairwise(path))

    def path_time_s(self, path: list[int]) -> float:
        return sum(self.weight[self.edge_between(u, v)] for u, v in pairwise(path))
