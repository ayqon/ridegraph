"""Contraction Hierarchies: the preprocessing trick behind production routing engines.

Idea
----
Rank every node by "importance", then remove (*contract*) them one at a time,
least important first. Removing node ``v`` can break quickest routes that pass
through it, so for each neighbour pair ``u -> v -> w`` we check whether some
other route from ``u`` to ``w`` is at least as quick (a *witness*). If not, we
add a **shortcut** edge ``u -> w`` with the combined cost and remember it skips
``v``.

After all nodes are contracted, every quickest route in the original graph has
an equivalent route in the augmented graph that only goes *up* the ranking
from the source and then only *down* the ranking into the target. A query is
therefore a bidirectional Dijkstra in which each side may only climb. Each
side sees a tiny "upward cone" of the graph instead of a ball of the city,
which is where the orders-of-magnitude speedup comes from.

Implementation choices
----------------------
* **Node order** by lazy priority updates: priority is the *edge difference*
  (shortcuts added minus edges removed) plus the number of already-contracted
  neighbours, which spreads contraction evenly across the map. A node's
  priority is recomputed when it reaches the top of the queue; if it is no
  longer the minimum it goes back in.
* **Witness search** is a local Dijkstra capped by cost and settled count. If
  the cap is hit we add the shortcut anyway: an unneeded shortcut costs a
  little space, never correctness.
* **Stall-on-demand** at query time: if a node can be reached more cheaply
  through a higher-ranked node the search has already seen, its upward
  distance is provably not optimal and its edges are not expanded.
* **Path unpacking**: each shortcut stores its middle node, so a route of
  shortcuts expands recursively back into real road segments.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .graph import Graph
from .result import Route

NO_MIDDLE = -1


@dataclass
class _Edge:
    cost: float
    middle: int  # contracted node this shortcut skips, or NO_MIDDLE for a real edge


class _Contractor:
    """Mutable working state while building the hierarchy."""

    def __init__(self, graph: Graph, settle_limit: int) -> None:
        n = graph.num_nodes
        self.n = n
        self.settle_limit = settle_limit
        self.out: list[dict[int, _Edge]] = [{} for _ in range(n)]
        self.inc: list[dict[int, float]] = [{} for _ in range(n)]
        for u in range(n):
            for e in range(graph.first_out[u], graph.first_out[u + 1]):
                v, w = graph.head[e], graph.weight[e]
                if u != v:
                    self._add_edge(u, v, w, NO_MIDDLE)
        self.deleted_neighbours = [0] * n

    def _add_edge(self, u: int, v: int, cost: float, middle: int) -> bool:
        existing = self.out[u].get(v)
        if existing is not None and existing.cost <= cost:
            return False
        self.out[u][v] = _Edge(cost, middle)
        self.inc[v][u] = cost
        return True

    def _witness_distances(
        self, source: int, skip: int, targets: dict[int, float], max_cost: float
    ) -> dict[int, float]:
        """Local Dijkstra from ``source`` that never passes through ``skip``."""
        dist = {source: 0.0}
        heap = [(0.0, source)]
        done: set[int] = set()
        remaining = len(targets)
        out = self.out
        while heap and len(done) < self.settle_limit:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            if d > max_cost:
                break
            done.add(u)
            if u in targets:
                remaining -= 1
                if remaining == 0:
                    break
            for v, edge in out[u].items():
                if v == skip:
                    continue
                nd = d + edge.cost
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def shortcuts_needed(self, v: int) -> list[tuple[int, int, float]]:
        """Shortcuts that contracting ``v`` right now would require."""
        needed: list[tuple[int, int, float]] = []
        outgoing = self.out[v]
        for u, cost_uv in self.inc[v].items():
            targets = {w: cost_uv + e.cost for w, e in outgoing.items() if w != u}
            if not targets:
                continue
            dist = self._witness_distances(u, v, targets, max(targets.values()))
            for w, via_v in targets.items():
                if dist.get(w, math.inf) > via_v:
                    needed.append((u, w, via_v))
        return needed

    def priority(self, v: int, shortcuts: list[tuple[int, int, float]]) -> float:
        edge_difference = len(shortcuts) - len(self.inc[v]) - len(self.out[v])
        return edge_difference + self.deleted_neighbours[v]


class ContractionHierarchy:
    """A contracted road graph that answers quickest-route queries fast.

    ``up_*`` arrays hold, for each node ``u``, edges ``u -> v`` to higher-ranked
    nodes. ``down_*`` arrays hold, for each node ``v``, edges ``u -> v`` *from*
    higher-ranked nodes, indexed at ``v`` so the backward search can climb
    from the target.
    """

    def __init__(
        self,
        rank: list[int],
        up: tuple[list[int], list[int], list[float], list[int]],
        down: tuple[list[int], list[int], list[float], list[int]],
        num_original_edges: int,
        build_seconds: float = 0.0,
    ) -> None:
        self.rank = rank
        self.up_first, self.up_head, self.up_weight, self.up_middle = up
        self.down_first, self.down_tail, self.down_weight, self.down_middle = down
        self.num_original_edges = num_original_edges
        self.build_seconds = build_seconds

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        graph: Graph,
        *,
        settle_limit: int = 500,
        progress: Callable[[int, int], None] | None = None,
    ) -> ContractionHierarchy:
        started = time.perf_counter()
        c = _Contractor(graph, settle_limit)
        n = c.n

        heap = []
        for v in range(n):
            heap.append((c.priority(v, c.shortcuts_needed(v)), v))
        heapq.heapify(heap)

        rank = [-1] * n
        up_edges: list[list[tuple[int, float, int]]] = [[] for _ in range(n)]
        down_edges: list[list[tuple[int, float, int]]] = [[] for _ in range(n)]
        next_rank = 0

        while heap:
            _, v = heapq.heappop(heap)
            if rank[v] >= 0:
                continue
            shortcuts = c.shortcuts_needed(v)
            current = c.priority(v, shortcuts)
            if heap and current > heap[0][0]:
                heapq.heappush(heap, (current, v))
                continue

            rank[v] = next_rank
            next_rank += 1
            # Every remaining neighbour will be contracted later, so outranks v.
            for w, edge in c.out[v].items():
                up_edges[v].append((w, edge.cost, edge.middle))
            for u in c.inc[v]:
                edge = c.out[u][v]
                down_edges[v].append((u, edge.cost, edge.middle))

            neighbours = set(c.out[v]) | set(c.inc[v])
            for w in c.out[v]:
                del c.inc[w][v]
            for u in c.inc[v]:
                del c.out[u][v]
            c.out[v].clear()
            c.inc[v].clear()
            for u, w, cost in shortcuts:
                c._add_edge(u, w, cost, v)
            for x in neighbours:
                c.deleted_neighbours[x] += 1

            if progress is not None and (next_rank % 1000 == 0 or next_rank == n):
                progress(next_rank, n)

        return cls(
            rank,
            _to_csr(up_edges),
            _to_csr(down_edges),
            graph.num_edges,
            time.perf_counter() - started,
        )

    # ------------------------------------------------------------------ stats

    @property
    def num_nodes(self) -> int:
        return len(self.rank)

    @property
    def num_edges(self) -> int:
        return len(self.up_head) + len(self.down_tail)

    @property
    def num_shortcuts(self) -> int:
        return sum(1 for m in self.up_middle if m != NO_MIDDLE) + sum(
            1 for m in self.down_middle if m != NO_MIDDLE
        )

    # ------------------------------------------------------------------ query

    def query(self, source: int, target: int) -> Route:
        """Quickest route from ``source`` to ``target``."""
        if source == target:
            return Route(source, target, [source], 0.0, 1, "ch")

        up = (self.up_first, self.up_head, self.up_weight)
        down = (self.down_first, self.down_tail, self.down_weight)
        dist: tuple[dict[int, float], dict[int, float]] = ({source: 0.0}, {target: 0.0})
        parent: tuple[dict[int, int], dict[int, int]] = ({}, {})
        heaps: tuple[list[tuple[float, int]], list[tuple[float, int]]] = (
            [(0.0, source)],
            [(0.0, target)],
        )
        done: tuple[set[int], set[int]] = (set(), set())
        best = math.inf
        meeting = -1

        while heaps[0] or heaps[1]:
            # Unlike plain bidirectional search, each side may only stop on its
            # own: the upward cones do not shrink towards each other.
            side = -1
            for s in (0, 1):
                if (
                    heaps[s]
                    and heaps[s][0][0] < best
                    and (side < 0 or heaps[s][0][0] < heaps[side][0][0])
                ):
                    side = s
            if side < 0:
                break
            other = 1 - side
            d, u = heapq.heappop(heaps[side])
            if u in done[side]:
                continue
            done[side].add(u)

            my_dist, their_dist = dist[side], dist[other]
            joined = d + their_dist.get(u, math.inf)
            if joined < best:
                best, meeting = joined, u

            # Stall-on-demand: edges that come *down* into u (forward side) or
            # leave u upwards (backward side) prove whether u was reached
            # optimally from above.
            s_first, s_adj, s_w = (down, up)[side]
            stalled = False
            for e in range(s_first[u], s_first[u + 1]):
                if my_dist.get(s_adj[e], math.inf) + s_w[e] < d:
                    stalled = True
                    break
            if stalled:
                continue

            first, adj, w = (up, down)[side]
            for e in range(first[u], first[u + 1]):
                v = adj[e]
                nd = d + w[e]
                if nd < my_dist.get(v, math.inf):
                    my_dist[v] = nd
                    parent[side][v] = u
                    heapq.heappush(heaps[side], (nd, v))

        settled = len(done[0]) + len(done[1])
        if meeting < 0:
            return Route(source, target, settled=settled, algorithm="ch")

        hierarchy_path = [meeting]
        node = meeting
        while node != source:
            node = parent[0][node]
            hierarchy_path.append(node)
        hierarchy_path.reverse()
        node = meeting
        while node != target:
            node = parent[1][node]
            hierarchy_path.append(node)
        return Route(source, target, self.unpack(hierarchy_path), best, settled, "ch")

    # ------------------------------------------------------------ many-to-many

    def upward_search(self, start: int, *, backward: bool = False) -> dict[int, float]:
        """Exhaustive search up the hierarchy from ``start``, with stall-on-demand.

        Returns the settled, unstalled nodes and their distances. Every quickest
        route meets at a node that both ends' upward searches settle correctly,
        which is what makes bucket-based many-to-many queries work.
        """
        up = (self.up_first, self.up_head, self.up_weight)
        down = (self.down_first, self.down_tail, self.down_weight)
        first, adj, w = down if backward else up
        s_first, s_adj, s_w = up if backward else down
        dist = {start: 0.0}
        heap = [(0.0, start)]
        done: dict[int, float] = {}
        seen: set[int] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            if any(
                dist.get(s_adj[e], math.inf) + s_w[e] < d for e in range(s_first[u], s_first[u + 1])
            ):
                continue  # stalled: reached sub-optimally, cannot be a meeting point
            done[u] = d
            for e in range(first[u], first[u + 1]):
                v = adj[e]
                nd = d + w[e]
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return done

    def target_buckets(self, targets: list[int]) -> dict[int, list[tuple[int, float]]]:
        """Backward upward searches from each target, filed by the node they reach."""
        buckets: dict[int, list[tuple[int, float]]] = {}
        for j, t in enumerate(targets):
            for v, d in self.upward_search(t, backward=True).items():
                buckets.setdefault(v, []).append((j, d))
        return buckets

    def many_to_many(
        self,
        sources: list[int],
        targets: list[int],
        buckets: dict[int, list[tuple[int, float]]] | None = None,
    ) -> np.ndarray:
        """Travel time from every source to every target (``inf`` if unreachable).

        Bucket-based many-to-many (Knopp et al., 2007): one backward upward
        search per target files its distances in buckets at the nodes it
        reaches; one forward upward search per source then reads the buckets
        of the nodes it reaches. Cost grows with sources + targets, not their
        product, which is what lets a dispatcher price every driver against
        every waiting rider at once. Reuse ``buckets`` when targets repeat.
        """
        if buckets is None:
            buckets = self.target_buckets(targets)
        out = np.full((len(sources), len(targets)), math.inf)
        for i, s in enumerate(sources):
            row = out[i]
            for v, d in self.upward_search(s).items():
                for j, dt in buckets.get(v, ()):
                    if d + dt < row[j]:
                        row[j] = d + dt
        return out

    # ------------------------------------------------------------ unpacking

    def _middle(self, a: int, b: int) -> int:
        """Middle node of the hierarchy edge ``a -> b`` (``NO_MIDDLE`` if real)."""
        if self.rank[a] < self.rank[b]:
            first, adj, weight, middle, at, other = (
                self.up_first,
                self.up_head,
                self.up_weight,
                self.up_middle,
                a,
                b,
            )
        else:
            first, adj, weight, middle, at, other = (
                self.down_first,
                self.down_tail,
                self.down_weight,
                self.down_middle,
                b,
                a,
            )
        best_e, best_w = -1, math.inf
        for e in range(first[at], first[at + 1]):
            if adj[e] == other and weight[e] < best_w:
                best_e, best_w = e, weight[e]
        if best_e < 0:
            raise KeyError(f"no hierarchy edge {a} -> {b}")
        return middle[best_e]

    def unpack(self, hierarchy_path: list[int]) -> list[int]:
        """Expand a route over shortcut edges into the original road nodes."""
        path = [hierarchy_path[0]]
        for a, b in pairwise(hierarchy_path):
            stack = [(a, b)]
            while stack:
                x, y = stack.pop()
                mid = self._middle(x, y)
                if mid == NO_MIDDLE:
                    path.append(y)
                else:
                    # Push the second half first so the first half is expanded first.
                    stack.append((mid, y))
                    stack.append((x, mid))
        return path

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            rank=np.array(self.rank, dtype=np.int32),
            up_first=np.array(self.up_first, dtype=np.int64),
            up_head=np.array(self.up_head, dtype=np.int32),
            up_weight=np.array(self.up_weight, dtype=np.float64),
            up_middle=np.array(self.up_middle, dtype=np.int32),
            down_first=np.array(self.down_first, dtype=np.int64),
            down_tail=np.array(self.down_tail, dtype=np.int32),
            down_weight=np.array(self.down_weight, dtype=np.float64),
            down_middle=np.array(self.down_middle, dtype=np.int32),
            meta=np.array([self.num_original_edges, self.build_seconds], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> ContractionHierarchy:
        with np.load(path) as d:
            meta = d["meta"]
            return cls(
                d["rank"].tolist(),
                (
                    d["up_first"].tolist(),
                    d["up_head"].tolist(),
                    d["up_weight"].tolist(),
                    d["up_middle"].tolist(),
                ),
                (
                    d["down_first"].tolist(),
                    d["down_tail"].tolist(),
                    d["down_weight"].tolist(),
                    d["down_middle"].tolist(),
                ),
                int(meta[0]),
                float(meta[1]),
            )


def _to_csr(
    adjacency: list[list[tuple[int, float, int]]],
) -> tuple[list[int], list[int], list[float], list[int]]:
    first = [0]
    nodes: list[int] = []
    weights: list[float] = []
    middles: list[int] = []
    for edges in adjacency:
        for v, w, m in edges:
            nodes.append(v)
            weights.append(w)
            middles.append(m)
        first.append(len(nodes))
    return first, nodes, weights, middles
