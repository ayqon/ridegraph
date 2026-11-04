"""Alternative routes: a few genuinely different ways to make the same trip.

A map app that shows only the quickest route hides useful choices: a route a
minute slower that avoids a charge, or stays on main roads. Good
alternatives must be (Abraham et al., "Alternative routes in road networks"):

* **different**: share at most ``max_overlap`` of their length with any route
  already offered;
* **not much slower**: at most ``max_stretch`` times the quickest;
* **sensible**: no pointless detours.

This module uses the *penalty method*: find the quickest route, make every road
on it a little slower, search again, and repeat. Each search is pushed off the
roads already used, so it finds the next most natural corridor. Candidates
are judged on the real travel times, not the penalised ones. A detour check
then rejects candidates that leave a road only to rejoin it: every short
stretch of the route must be close to the quickest way between its ends.

The same machinery avoids an area: roads into it cost infinity, so every
route found keeps out. That is how the Congestion Charge zone is avoided.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .graph import Graph


@dataclass(frozen=True)
class Alternative:
    nodes: list[int]
    edges: list[int]  # CSR edge ids
    seconds: float
    metres: float
    enters_zone: bool = False


def shortest(
    graph: Graph, source: int, target: int, weight: list[float]
) -> tuple[float, list[int]]:
    """Dijkstra on custom weights, returning (seconds, CSR edge ids)."""
    first_out, head = graph.first_out, graph.head
    dist = {source: 0.0}
    via: dict[int, int] = {}
    done: set[int] = set()
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == target:
            break
        for e in range(first_out[u], first_out[u + 1]):
            w = weight[e]
            if w == math.inf:
                continue
            v = head[e]
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                via[v] = e
                heapq.heappush(heap, (nd, v))
    if target not in done:
        return math.inf, []
    edges = []
    v = target
    while v != source:
        e = via[v]
        edges.append(e)
        v = _tail(graph, e)
    edges.reverse()
    return dist[target], edges


def _tail(graph: Graph, e: int) -> int:
    lo, hi = 0, len(graph.first_out) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if graph.first_out[mid] <= e:
            lo = mid
        else:
            hi = mid
    return lo


def _make(graph: Graph, source: int, edges: list[int], zone: np.ndarray | None) -> Alternative:
    nodes = [source] + [graph.head[e] for e in edges]
    seconds = sum(graph.weight[e] for e in edges)
    metres = sum(graph.length_m[e] for e in edges)
    enters = bool(zone is not None and any(zone[v] for v in nodes))
    return Alternative(nodes, edges, seconds, metres, enters)


def zone_weights(graph: Graph, zone: np.ndarray) -> list[float]:
    """Travel times with every road into the zone closed."""
    return [math.inf if zone[v] else w for w, v in zip(graph.weight, graph.head, strict=True)]


def is_locally_sensible(
    graph: Graph,
    alt: Alternative,
    window_s: float,
    tolerance: float,
    weight: list[float],
    samples: int = 8,
) -> bool:
    """Stretches of about ``window_s`` are within ``tolerance`` of their quickest way.

    Checks ``samples`` stretches spread along the route, which catches the
    detours the penalty method produces (leaving a road only to rejoin it)
    without a search from every junction.
    """
    times = np.concatenate([[0.0], np.cumsum([weight[e] for e in alt.edges])])
    total = float(times[-1])
    if total <= window_s:
        return True
    for k in range(samples):
        start = (total - window_s) * k / max(1, samples - 1)
        i = int(np.searchsorted(times, start, side="right")) - 1
        j = int(np.searchsorted(times, times[i] + window_s, side="left"))
        j = min(j, len(alt.nodes) - 1)
        if j <= i + 1:
            continue
        best, _ = shortest(graph, alt.nodes[i], alt.nodes[j], weight)
        if times[j] - times[i] > tolerance * best + 1e-9:
            return False
    return True


def alternatives(
    graph: Graph,
    source: int,
    target: int,
    *,
    count: int = 3,
    penalty: float = 1.4,
    max_stretch: float = 1.3,
    max_overlap: float = 0.65,
    attempts: int = 8,
    zone: np.ndarray | None = None,
    avoid_zone: bool = False,
    local_check: bool = True,
) -> list[Alternative]:
    """Up to ``count`` routes, quickest first; ``avoid_zone`` keeps them out of ``zone``."""
    base = list(graph.weight)
    if avoid_zone and zone is not None:
        if zone[source] or zone[target]:
            raise ValueError("the trip starts or ends inside the zone, so it cannot avoid it")
        base = zone_weights(graph, zone)
    best_s, first = shortest(graph, source, target, base)
    if not math.isfinite(best_s):
        return []
    found = [_make(graph, source, first, zone)]
    used = {e: penalty for e in first}
    work = list(base)
    for e in first:
        work[e] = base[e] * penalty
    for _ in range(attempts):
        if len(found) >= count:
            break
        _, edges = shortest(graph, source, target, work)
        if not edges:
            break
        for e in edges:
            used[e] = used.get(e, 1.0) * penalty
            work[e] = base[e] * used[e]
        cand = _make(graph, source, edges, zone)
        if cand.seconds > max_stretch * found[0].seconds:
            continue
        if any(_overlap(graph, cand, f) > max_overlap for f in found):
            continue
        window = 0.25 * found[0].seconds
        if local_check and not is_locally_sensible(graph, cand, window, 1.15, base):
            continue
        found.append(cand)
    return sorted(found, key=lambda a: a.seconds)


def _overlap(graph: Graph, a: Alternative, b: Alternative) -> float:
    """Share of ``a``'s length that is also on ``b`` (either direction of a road)."""
    roads_b = {frozenset((_tail(graph, e), graph.head[e])) for e in b.edges}
    shared = sum(
        graph.length_m[e] for e in a.edges if frozenset((_tail(graph, e), graph.head[e])) in roads_b
    )
    return shared / a.metres if a.metres > 0 else 1.0
