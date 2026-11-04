"""A* search: Dijkstra that knows which way the destination is.

Dijkstra orders nodes by time-so-far. A* orders them by time-so-far plus an
estimate of time-remaining, so the search ball is stretched towards the target
instead of growing evenly in every direction.

The estimate here is the straight-line distance to the target divided by the
fastest speed anywhere on the network. No road can beat a straight line driven
at top speed, so the estimate never overshoots (it is *admissible*), and since
it also obeys the triangle inequality (*consistent*), the first time the target
is settled its time is optimal, exactly as with Dijkstra.

On city graphs the gain is real but bounded. The heuristic assumes motorway
speed in a straight line, which in central London is wildly optimistic, so the
search still explores a fat ellipse. Contraction Hierarchies (``ch.py``) are
how production engines go further.
"""

from __future__ import annotations

import heapq
import math

from .graph import EARTH_RADIUS_M, Graph
from .result import Route, unpack_path

# Shaves a hair off the estimate so float32 edge lengths can never make it
# overshoot by rounding.
_SAFETY = 0.9999


def astar(graph: Graph, source: int, target: int) -> Route:
    """Quickest route from ``source`` to ``target`` using A*."""
    first_out, head, weight = graph.first_out, graph.head, graph.weight
    lat = graph.network.lat
    lon = graph.network.lon
    max_speed = graph.max_speed_mps
    scale = _SAFETY * 2 * EARTH_RADIUS_M / max_speed if max_speed > 0 else 0.0

    t_lat = math.radians(float(lat[target]))
    t_lon = math.radians(float(lon[target]))
    cos_t = math.cos(t_lat)
    sin, cos, asin, sqrt, radians = math.sin, math.cos, math.asin, math.sqrt, math.radians
    estimates: dict[int, float] = {}

    def h(v: int) -> float:
        cached = estimates.get(v)
        if cached is not None:
            return cached
        v_lat = radians(float(lat[v]))
        a = (
            sin((t_lat - v_lat) / 2) ** 2
            + cos(v_lat) * cos_t * sin((t_lon - radians(float(lon[v]))) / 2) ** 2
        )
        value = scale * asin(min(1.0, sqrt(a)))
        estimates[v] = value
        return value

    dist: dict[int, float] = {source: 0.0}
    parent: dict[int, int] = {}
    settled: set[int] = set()
    heap: list[tuple[float, int]] = [(h(source), source)]

    while heap:
        _, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        d = dist[u]
        if u == target:
            return Route(
                source, target, unpack_path(parent, source, target), d, len(settled), "astar"
            )
        for e in range(first_out[u], first_out[u + 1]):
            v = head[e]
            if v in settled:
                continue
            nd = d + weight[e]
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd + h(v), v))

    return Route(source, target, settled=len(settled), algorithm="astar")
