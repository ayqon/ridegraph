"""Travel-time maps: everywhere reachable within a time budget.

An isochrone answers "where can I get to in 20 minutes from here?" (or, for
someone choosing where to live, "from where can I get *to* my office in 20
minutes?"). It is a single-source search that settles every junction in time
order and stops at the budget, so it touches only the reachable part of the
city. Run on the reverse graph, the same search measures journeys *into* a
point.

Reachable area is reported by counting 250 m squares that contain a reached
junction, which tracks what a map reader sees better than a convex hull
would around a river or a park.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .graph import Graph

AREA_CELL_KM = 0.25
KM_PER_DEG_LAT = 111.32


def times_within(
    graph: Graph, origin: int, budget_s: float, *, arriving: bool = False
) -> dict[int, float]:
    """Seconds from ``origin`` (or to it, if ``arriving``) for every junction within budget."""
    if arriving:
        first, adj, weight = graph.first_in, graph.tail, graph.in_weight
    else:
        first, adj, weight = graph.first_out, graph.head, graph.weight
    dist = {origin: 0.0}
    done: dict[int, float] = {}
    heap = [(0.0, origin)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        if d > budget_s:
            break
        done[u] = d
        for e in range(first[u], first[u + 1]):
            v = adj[e]
            nd = d + weight[e]
            if nd <= budget_s and nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return done


@dataclass(frozen=True)
class Reach:
    bands_min: tuple[int, ...]
    area_km2: tuple[float, ...]  # cumulative: within each band's limit
    junctions: tuple[int, ...]


def reach(
    graph: Graph,
    origin: int,
    bands_min: tuple[int, ...] = (10, 20, 30),
    *,
    arriving: bool = False,
) -> Reach:
    times = times_within(graph, origin, max(bands_min) * 60, arriving=arriving)
    lat, lon = graph.network.lat, graph.network.lon
    lat0 = float(lat.mean())
    km_lon = KM_PER_DEG_LAT * math.cos(math.radians(lat0))
    nodes = np.fromiter(times.keys(), dtype=np.int64, count=len(times))
    secs = np.fromiter(times.values(), dtype=np.float64, count=len(times))
    cx = np.floor(lon[nodes] * km_lon / AREA_CELL_KM).astype(np.int64)
    cy = np.floor(lat[nodes] * KM_PER_DEG_LAT / AREA_CELL_KM).astype(np.int64)
    areas, counts = [], []
    for band in bands_min:
        inside = secs <= band * 60
        cells = set(zip(cx[inside].tolist(), cy[inside].tolist(), strict=True))
        areas.append(len(cells) * AREA_CELL_KM**2)
        counts.append(int(inside.sum()))
    return Reach(tuple(bands_min), tuple(areas), tuple(counts))
