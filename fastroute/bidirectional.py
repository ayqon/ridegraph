"""Bidirectional Dijkstra: search from both ends and meet in the middle.

Two balls of radius r/2 cover far less ground than one ball of radius r (on a
road network, roughly half). The forward search follows edges out of the
source; the backward search follows edges *into* the target.

The subtle part is when to stop. The first node settled by both searches is
not necessarily on the best route. Instead, every time a search relaxes an
edge into territory the other side has reached, it records the length of that
joined route as a candidate ``best``. Once the two queue minimums add up to at
least ``best``, no undiscovered route can be shorter and the search ends.

This is also the query skeleton Contraction Hierarchies reuse, which is the
main reason it is worth having as its own module.
"""

from __future__ import annotations

import heapq
import math

from .graph import Graph
from .result import Route


def bidirectional_dijkstra(graph: Graph, source: int, target: int) -> Route:
    """Quickest route from ``source`` to ``target``, searching from both ends."""
    if source == target:
        return Route(source, target, [source], 0.0, 1, "bidirectional")

    fwd = (graph.first_out, graph.head, graph.weight)
    bwd = (graph.first_in, graph.tail, graph.in_weight)
    dist = ({source: 0.0}, {target: 0.0})
    parent: tuple[dict[int, int], dict[int, int]] = ({}, {})
    done: tuple[set[int], set[int]] = (set(), set())
    heaps: tuple[list[tuple[float, int]], list[tuple[float, int]]] = (
        [(0.0, source)],
        [(0.0, target)],
    )
    best = math.inf
    meeting = -1

    while heaps[0] and heaps[1]:
        if heaps[0][0][0] + heaps[1][0][0] >= best:
            break
        # Advance whichever frontier is currently smaller: keeps the balls balanced.
        side = 0 if heaps[0][0][0] <= heaps[1][0][0] else 1
        other = 1 - side
        d, u = heapq.heappop(heaps[side])
        if u in done[side]:
            continue
        done[side].add(u)
        first, adj, w = (fwd, bwd)[side]
        my_dist, their_dist = dist[side], dist[other]
        for e in range(first[u], first[u + 1]):
            v = adj[e]
            nd = d + w[e]
            if nd < my_dist.get(v, math.inf):
                my_dist[v] = nd
                parent[side][v] = u
                heapq.heappush(heaps[side], (nd, v))
            joined = my_dist.get(v, math.inf) + their_dist.get(v, math.inf)
            if joined < best:
                best, meeting = joined, v

    settled = len(done[0]) + len(done[1])
    if meeting < 0:
        return Route(source, target, settled=settled, algorithm="bidirectional")

    path = [meeting]
    node = meeting
    while node != source:
        node = parent[0][node]
        path.append(node)
    path.reverse()
    node = meeting
    while node != target:
        node = parent[1][node]
        path.append(node)
    return Route(source, target, path, best, settled, "bidirectional")
