"""Customizable Contraction Hierarchies: preprocessing once, new traffic in milliseconds.

Ordinary Contraction Hierarchies (``ch.py``) choose which shortcuts to add by
running witness searches with the current travel times. Change the times
(a closure, a jam) and the whole hierarchy is stale. Customizable CH
(Dibbelt, Strasser and Wagner, 2016) splits the work in three:

1. **Metric-independent preprocessing** (slow, once per map). Order the
   junctions by *nested dissection*: cut the city in two along a thin
   separator, order each half recursively, and put the separator last. Then
   contract in that order *without* witness searches: every pair of higher
   neighbours gets a shortcut. The result is a fixed "chordal" graph whose
   size does not depend on travel times.
2. **Customization** (fast, whenever times change). Give every original road
   its current time, then sweep the *lower triangles* bottom-up: for a
   shortcut u-w over a lower junction v, ``u->w = min(u->w, u->v + v->w)``.
   Every shortcut now holds the true quickest time between its ends using
   only lower junctions, which is exactly what the CH query needs.
3. **Queries** as in CH: two upward searches meeting at the top.

Because step 1 never looks at travel times, the separators come from the
road layout alone: a geometric cut picks the direction (north-south,
east-west or a diagonal) that crosses the fewest roads, the way rivers and
parks make natural cuts in a real city.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .graph import RoadNetwork

LEAF_SIZE = 32
DIRECTIONS = ((1.0, 0.0), (0.0, 1.0), (0.7071, 0.7071), (0.7071, -0.7071))


def _undirected(network: RoadNetwork) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(network.num_nodes)]
    for u, v in zip(network.src.tolist(), network.dst.tolist(), strict=True):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def nested_dissection_order(network: RoadNetwork, leaf_size: int = LEAF_SIZE) -> np.ndarray:
    """Rank of every junction; separators are ranked above the parts they split."""
    adj = _undirected(network)
    lat0 = float(np.mean(network.lat)) if network.num_nodes else 0.0
    x = np.asarray(network.lon, dtype=np.float64) * math.cos(math.radians(lat0))
    y = np.asarray(network.lat, dtype=np.float64)
    order: list[int] = []

    # Explicit stack to avoid deep recursion: ("part", nodes) expands into its
    # two halves and its separator; ("emit", nodes) appends to the order.
    stack: list[tuple[str, np.ndarray]] = [("part", np.arange(network.num_nodes))]
    while stack:
        kind, nodes = stack.pop()
        if kind == "emit":
            order.extend(nodes.tolist())
            continue
        if len(nodes) <= leaf_size:
            # Low degree first inside a leaf keeps fill-in small.
            inside = set(nodes.tolist())
            deg = [sum(1 for w in adj[v] if w in inside) for v in nodes.tolist()]
            order.extend(nodes[np.argsort(deg, kind="stable")].tolist())
            continue
        left, right, sep = _bisect(nodes, adj, x, y)
        # Processed in reverse: left part first, then right, then the separator.
        stack.append(("emit", sep))
        stack.append(("part", right))
        stack.append(("part", left))
    rank = np.empty(network.num_nodes, dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
    return rank


def _bisect(nodes: np.ndarray, adj, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    member = set(nodes.tolist())
    best = None
    for dx, dy in DIRECTIONS:
        proj = x[nodes] * dx + y[nodes] * dy
        cut = np.median(proj)
        is_left = {int(v): bool(p <= cut) for v, p in zip(nodes, proj, strict=True)}
        # Separator: the left-hand ends of roads that cross the cut (a vertex cover).
        sep_l, sep_r = set(), set()
        for v in nodes.tolist():
            for w in adj[v]:
                if w in member and is_left[v] and not is_left[w]:
                    sep_l.add(v)
                    sep_r.add(w)
        sep = sep_l if len(sep_l) <= len(sep_r) else sep_r
        n_left = sum(1 for v in is_left if is_left[v] and v not in sep)
        balance = min(n_left, len(nodes) - len(sep) - n_left)
        if balance < len(nodes) // 5:
            continue
        if best is None or len(sep) < len(best[2]):
            best = (is_left, sep, sep)
    if best is None:  # degenerate shape: split by index
        half = len(nodes) // 2
        return nodes[:half], nodes[half:], np.array([], dtype=np.int64)
    is_left, sep, _ = best
    left = np.array([v for v in nodes.tolist() if is_left[v] and v not in sep], dtype=np.int64)
    right = np.array([v for v in nodes.tolist() if not is_left[v] and v not in sep], dtype=np.int64)
    return left, right, np.array(sorted(sep), dtype=np.int64)


@dataclass
class CustomizableCH:
    """Chordal graph from contracting in a fixed order, with per-metric weights."""

    rank: np.ndarray
    first_up: np.ndarray  # CSR over nodes: upward neighbours, sorted by rank
    up_head: np.ndarray
    # Lower triangles grouped into levels that can be customised together.
    tri_a: np.ndarray  # edge (v, u)
    tri_b: np.ndarray  # edge (v, w)
    tri_c: np.ndarray  # edge (u, w)
    level_start: np.ndarray
    # Where each original road's time goes: CCH edge id and direction.
    road_edge: np.ndarray
    road_up: np.ndarray  # True if the road runs from lower to higher rank
    road_src: np.ndarray
    road_dst: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.zeros(0))  # cost lower -> higher
    down: np.ndarray = field(default_factory=lambda: np.zeros(0))  # cost higher -> lower
    build_seconds: float = 0.0
    _direct: dict[tuple[int, int], float] | None = field(default=None, init=False, repr=False)
    # Plain-list views for the query loop (see ``_lists``).
    _topology: tuple[list[int], list[int], list[int]] | None = field(
        default=None, init=False, repr=False
    )
    _edge_ids: dict[tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _list_cache: tuple[np.ndarray, np.ndarray, list[float], list[float]] | None = field(
        default=None, init=False, repr=False
    )
    customize_seconds: float = 0.0

    @property
    def num_nodes(self) -> int:
        return len(self.first_up) - 1

    @property
    def num_edges(self) -> int:
        return len(self.up_head)

    @property
    def num_triangles(self) -> int:
        return len(self.tri_a)

    @classmethod
    def build(cls, network: RoadNetwork, rank: np.ndarray | None = None) -> CustomizableCH:
        started = time.perf_counter()
        if rank is None:
            rank = nested_dissection_order(network)
        n = network.num_nodes
        adj = _undirected(network)
        by_rank = np.argsort(rank)
        up_sets: list[set[int]] = [set() for _ in range(n)]
        for v in range(n):
            rv = rank[v]
            up_sets[v] = {w for w in adj[v] if rank[w] > rv}
        # Contract in order; the elimination tree passes each node's higher
        # neighbourhood to its lowest higher neighbour, which adds the fill-in.
        for v in by_rank.tolist():
            ups = up_sets[v]
            if len(ups) > 1:
                parent = min(ups, key=lambda w: rank[w])
                up_sets[parent] |= ups - {parent}
        first_up = np.zeros(n + 1, dtype=np.int64)
        heads: list[int] = []
        for v in range(n):
            ordered = sorted(up_sets[v], key=lambda w: rank[w])
            heads.extend(ordered)
            first_up[v + 1] = len(heads)
        up_head = np.asarray(heads, dtype=np.int64)

        ids = {
            (lo, hi): k
            for lo in range(n)
            for k, hi in zip(
                range(int(first_up[lo]), int(first_up[lo + 1])),
                heads[first_up[lo] : first_up[lo + 1]],
                strict=True,
            )
        }
        edge_id = ids.__getitem__

        # Lower triangles: for v, every pair u < w of its upward neighbours.
        ta, tb, tc, tl = [], [], [], []
        level = np.zeros(n, dtype=np.int64)
        for v in by_rank.tolist():
            s, e = int(first_up[v]), int(first_up[v + 1])
            ups = up_head[s:e].tolist()
            for i, u in enumerate(ups):
                if level[u] < level[v] + 1:
                    level[u] = level[v] + 1
                for k in range(i + 1, len(ups)):
                    ta.append(s + i)
                    tb.append(s + k)
                    tc.append(edge_id((u, ups[k])))
                    tl.append(level[v])
        tri_a = np.asarray(ta, dtype=np.int64)
        tri_b = np.asarray(tb, dtype=np.int64)
        tri_c = np.asarray(tc, dtype=np.int64)
        tri_l = np.asarray(tl, dtype=np.int64)
        order = np.argsort(tri_l, kind="stable")
        tri_a, tri_b, tri_c, tri_l = tri_a[order], tri_b[order], tri_c[order], tri_l[order]
        levels = int(tri_l.max()) + 1 if len(tri_l) else 0
        level_start = np.searchsorted(tri_l, np.arange(levels + 1))

        src, dst = network.src, network.dst
        road_edge = np.full(network.num_edges, -1, dtype=np.int64)
        road_up = np.zeros(network.num_edges, dtype=bool)
        for e, (a, b) in enumerate(zip(src.tolist(), dst.tolist(), strict=True)):
            if a == b:
                continue
            lo, hi = (a, b) if rank[a] < rank[b] else (b, a)
            road_edge[e] = edge_id((lo, hi))
            road_up[e] = lo == a
        cch = cls(
            rank,
            first_up,
            up_head,
            tri_a,
            tri_b,
            tri_c,
            level_start,
            road_edge,
            road_up,
            np.asarray(src),
            np.asarray(dst),
        )
        cch.build_seconds = time.perf_counter() - started
        return cch

    def customize(self, road_seconds: np.ndarray) -> None:
        """Load travel times for every original road and sweep the lower triangles."""
        started = time.perf_counter()
        m = self.num_edges
        up = np.full(m, np.inf)
        down = np.full(m, np.inf)
        ok = self.road_edge >= 0
        w = np.asarray(road_seconds, dtype=np.float64)[ok]
        ids = self.road_edge[ok]
        is_up = self.road_up[ok]
        np.minimum.at(up, ids[is_up], w[is_up])
        np.minimum.at(down, ids[~is_up], w[~is_up])
        a, b, c = self.tri_a, self.tri_b, self.tri_c
        for lvl in range(len(self.level_start) - 1):
            s, e = self.level_start[lvl], self.level_start[lvl + 1]
            if s == e:
                continue
            aa, bb, cc = a[s:e], b[s:e], c[s:e]
            # u -> w via v = (u -> v) + (v -> w); w -> u via v = (w -> v) + (v -> u).
            np.minimum.at(up, cc, down[aa] + up[bb])
            np.minimum.at(down, cc, down[bb] + up[aa])
        self.up, self.down = up, down
        self._road_seconds = np.asarray(road_seconds, dtype=np.float64)
        self._direct = None
        self.customize_seconds = time.perf_counter() - started

    def query(self, source: int, target: int) -> tuple[float, list[int]]:
        """Quickest time and junction path; ``customize`` must have run."""
        if source == target:
            return 0.0, [source]
        dist = ({source: 0.0}, {target: 0.0})
        parent: tuple[dict[int, int], dict[int, int]] = ({}, {})
        heaps = ([(0.0, source)], [(0.0, target)])
        done: tuple[set[int], set[int]] = (set(), set())
        best, meet = math.inf, -1
        first_up, up_head, up, down = self._lists()
        while heaps[0] or heaps[1]:
            for side in (0, 1):
                heap = heaps[side]
                if not heap:
                    continue
                d, v = heapq.heappop(heap)
                if v in done[side] or d > dist[side].get(v, math.inf):
                    continue
                if d >= best:
                    heap.clear()
                    continue
                done[side].add(v)
                other = dist[1 - side].get(v)
                if other is not None and d + other < best:
                    best, meet = d + other, v
                cost = up if side == 0 else down
                for e in range(first_up[v], first_up[v + 1]):
                    nd = d + cost[e]
                    w = up_head[e]
                    if nd < dist[side].get(w, math.inf):
                        dist[side][w] = nd
                        parent[side][w] = v
                        heapq.heappush(heap, (nd, w))
        if meet < 0:
            return math.inf, []
        fwd = [meet]
        while fwd[-1] != source:
            fwd.append(parent[0][fwd[-1]])
        bwd = [meet]
        while bwd[-1] != target:
            bwd.append(parent[1][bwd[-1]])
        hops = fwd[::-1] + bwd[1:]
        path = [hops[0]]
        for a, b in zip(hops, hops[1:], strict=False):
            path.extend(self._unpack(a, b)[1:])
        return best, path

    def _lists(self) -> tuple[list[int], list[int], list[float], list[float]]:
        """Plain-list views for the query loop, rebuilt when new times are loaded.

        Python lists index several times faster than numpy arrays one element at
        a time, which is all the search does.
        """
        if self._topology is None:
            first, heads = self.first_up.tolist(), self.up_head.tolist()
            self._topology = (first, heads, self.rank.tolist())
            self._edge_ids = {
                (lo, heads[e]): e
                for lo in range(self.num_nodes)
                for e in range(first[lo], first[lo + 1])
            }
        cache = self._list_cache
        if cache is None or cache[0] is not self.up or cache[1] is not self.down:
            cache = (self.up, self.down, self.up.tolist(), self.down.tolist())
            self._list_cache = cache
        return self._topology[0], self._topology[1], cache[2], cache[3]

    def _edge(self, lo: int, hi: int) -> int:
        return self._edge_ids[(lo, hi)]

    def _cost(self, a: int, b: int) -> float:
        _, _, up, down = self._lists()
        assert self._topology is not None
        rank = self._topology[2]
        if rank[a] < rank[b]:
            return up[self._edge(a, b)]
        return down[self._edge(b, a)]

    def _unpack(self, a: int, b: int) -> list[int]:
        """Junctions along the CCH edge a -> b, expanding shortcuts over lower junctions."""
        out = [a]
        stack = [(a, b)]
        result: list[int] = []
        # Iterative in-order expansion.
        while stack:
            x, y = stack.pop()
            target = self._cost(x, y)
            mid = self._middle(x, y, target)
            if mid is None:
                result.append(y)
            else:
                stack.append((mid, y))
                stack.append((x, mid))
        out.extend(result)
        return out

    def _middle(self, x: int, y: int, target: float) -> int | None:
        if self._direct_cost(x, y) <= target * (1 + 1e-12) + 1e-9:
            return None
        self._lists()
        assert self._topology is not None
        rank = self._topology[2]
        lo = x if rank[x] < rank[y] else y
        hi = y if lo == x else x
        for v in self._lower(lo) & self._lower(hi):
            if abs(self._cost(x, v) + self._cost(v, y) - target) <= 1e-9 * max(1.0, target):
                return v
        return None

    def _direct_cost(self, a: int, b: int) -> float:
        if self._direct is None:
            direct: dict[tuple[int, int], float] = {}
            for x, y, w in zip(
                self.road_src.tolist(),
                self.road_dst.tolist(),
                self._road_seconds.tolist(),
                strict=True,
            ):
                if w < direct.get((x, y), math.inf):
                    direct[(x, y)] = w
            self._direct = direct
        return self._direct.get((a, b), math.inf)

    def _lower(self, v: int) -> set[int]:
        if not hasattr(self, "_down_sets"):
            sets: list[set[int]] = [set() for _ in range(self.num_nodes)]
            for lo in range(self.num_nodes):
                for e in range(self.first_up[lo], self.first_up[lo + 1]):
                    sets[int(self.up_head[e])].add(lo)
            self._down_sets = sets
        return self._down_sets[v]
