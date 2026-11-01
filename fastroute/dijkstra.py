"""Dijkstra's algorithm: the correctness baseline every faster method is tested against.

It grows a ball of settled nodes around the source in order of travel time,
so by the time the target is settled its time is provably optimal. The cost
is that the ball covers everything closer than the target in every direction,
which is exactly the waste A* and Contraction Hierarchies remove.

Implementation notes:

* Binary heap with lazy deletion. Instead of a decrease-key operation, a
  better distance is simply pushed again and stale entries are skipped when
  popped. In Python this beats any hand-rolled indexed heap.
* Distances live in a dict touched only for reached nodes, so a short query on
  a big graph does not pay to initialise an array of every node.
"""

from __future__ import annotations

import heapq
import math

from .graph import Graph
from .result import Route, unpack_path


def dijkstra(graph: Graph, source: int, target: int) -> Route:
    """Quickest route from ``source`` to ``target`` (node indices)."""
    first_out, head, weight = graph.first_out, graph.head, graph.weight
    dist: dict[int, float] = {source: 0.0}
    parent: dict[int, int] = {}
    settled: set[int] = set()
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue  # stale entry, a shorter one was already processed
        settled.add(u)
        if u == target:
            return Route(
                source,
                target,
                unpack_path(parent, source, target),
                d,
                len(settled),
                "dijkstra",
            )
        for e in range(first_out[u], first_out[u + 1]):
            v = head[e]
            nd = d + weight[e]
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))

    return Route(source, target, settled=len(settled), algorithm="dijkstra")


def one_to_all(graph: Graph, source: int) -> list[float]:
    """Travel time from ``source`` to every node (``inf`` if unreachable)."""
    first_out, head, weight = graph.first_out, graph.head, graph.weight
    dist = [math.inf] * graph.num_nodes
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for e in range(first_out[u], first_out[u + 1]):
            v = head[e]
            nd = d + weight[e]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
