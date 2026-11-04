"""Multi-stop routes: the best order to visit a list of addresses.

A courier with a van and ten drops wants the order that gets them all done
soonest: the travelling salesman problem, on real one-way streets, so the
time from A to B is not the time from B to A. The pipeline:

1. **One matrix of travel times.** A many-to-many Contraction Hierarchies
   query prices every stop against every other stop at the chosen time of
   day.
2. **Exact for small jobs.** Held-Karp dynamic programming finds the
   provably best order in O(n^2 2^n) time, which is instant up to about a
   dozen stops.
3. **Heuristics for comparison and for larger jobs.** Nearest-next-stop
   construction, then 2-opt and Or-opt improvement (re-evaluating whole
   tours, since reversing a stretch of an asymmetric tour changes its cost).

The first stop is the depot. Tours either return to it or end at the last
drop.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .ch import ContractionHierarchy

MAX_EXACT_STOPS = 13  # including the depot


def tour_cost(matrix: np.ndarray, order: list[int], *, return_to_start: bool) -> float:
    total = sum(float(matrix[a, b]) for a, b in itertools.pairwise(order))
    if return_to_start and len(order) > 1:
        total += float(matrix[order[-1], order[0]])
    return total


def as_entered(n: int) -> list[int]:
    return list(range(n))


def nearest_neighbour(matrix: np.ndarray) -> list[int]:
    n = len(matrix)
    order, left = [0], set(range(1, n))
    while left:
        here = order[-1]
        nxt = min(left, key=lambda j: matrix[here, j])
        order.append(nxt)
        left.remove(nxt)
    return order


def improve(matrix: np.ndarray, order: list[int], *, return_to_start: bool) -> list[int]:
    """2-opt and Or-opt moves until no move shortens the tour (depot stays first)."""
    best = list(order)
    best_cost = tour_cost(matrix, best, return_to_start=return_to_start)
    n = len(best)
    improved = True
    while improved:
        improved = False
        # 2-opt: reverse a stretch.
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                c = tour_cost(matrix, cand, return_to_start=return_to_start)
                if c < best_cost - 1e-9:
                    best, best_cost, improved = cand, c, True
        # Or-opt: move a run of 1-3 stops elsewhere.
        for length in (1, 2, 3):
            for i in range(1, n - length + 1):
                run = best[i : i + length]
                rest = best[:i] + best[i + length :]
                for j in range(1, len(rest) + 1):
                    cand = rest[:j] + run + rest[j:]
                    c = tour_cost(matrix, cand, return_to_start=return_to_start)
                    if c < best_cost - 1e-9:
                        best, best_cost, improved = cand, c, True
    return best


def held_karp(matrix: np.ndarray, *, return_to_start: bool) -> list[int]:
    """Provably optimal order starting at stop 0 (Held-Karp dynamic programming)."""
    n = len(matrix)
    if n <= 2:
        return list(range(n))
    if n > MAX_EXACT_STOPS:
        raise ValueError(f"exact search is limited to {MAX_EXACT_STOPS} stops")
    m = n - 1  # stops other than the depot, as bits
    full = (1 << m) - 1
    cost = np.full((1 << m, m), math.inf)
    parent = np.full((1 << m, m), -1, dtype=np.int64)
    for j in range(m):
        cost[1 << j, j] = matrix[0, j + 1]
    for subset in range(1, full + 1):
        for j in range(m):
            if not subset & (1 << j) or not math.isfinite(cost[subset, j]):
                continue
            base = cost[subset, j]
            for k in range(m):
                if subset & (1 << k):
                    continue
                nxt = subset | (1 << k)
                c = base + matrix[j + 1, k + 1]
                if c < cost[nxt, k]:
                    cost[nxt, k] = c
                    parent[nxt, k] = j
    ends = cost[full] + (matrix[1:, 0] if return_to_start else 0)
    j = int(np.argmin(ends))
    order, subset = [], full
    while j >= 0:
        order.append(j + 1)
        prev = int(parent[subset, j])
        subset &= ~(1 << j)
        j = prev
    return [0, *order[::-1]]


@dataclass(frozen=True)
class TourPlan:
    stops: list[int]  # junctions, depot first, as entered
    matrix: np.ndarray
    orders: dict[str, list[int]]  # indices into stops
    seconds: dict[str, float]
    return_to_start: bool


def plan_matrix(matrix: np.ndarray, *, return_to_start: bool = True) -> TourPlan:
    """Compare visiting orders for a travel-time matrix (stop 0 is the depot)."""
    n = len(matrix)
    orders = {
        "As entered": as_entered(n),
        "Nearest next stop": nearest_neighbour(matrix),
    }
    orders["Improved (2-opt + Or-opt)"] = improve(
        matrix, orders["Nearest next stop"], return_to_start=return_to_start
    )
    if n <= MAX_EXACT_STOPS:
        orders["Optimal (Held-Karp)"] = held_karp(matrix, return_to_start=return_to_start)
    seconds = {
        name: tour_cost(matrix, order, return_to_start=return_to_start)
        for name, order in orders.items()
    }
    return TourPlan(list(range(n)), matrix, orders, seconds, return_to_start)


def plan(ch: ContractionHierarchy, stops: list[int], *, return_to_start: bool = True) -> TourPlan:
    """Compare visiting orders for ``stops`` (depot first)."""
    result = plan_matrix(ch.many_to_many(stops, stops), return_to_start=return_to_start)
    return TourPlan(stops, result.matrix, result.orders, result.seconds, return_to_start)
