"""Assigning drivers to riders.

Given a matrix of pickup times (driver x rider), find the assignment that
minimises the total. This is the linear assignment problem; the Hungarian
algorithm solves it exactly in O(n^2 m) using dual potentials, here in the
compact shortest-augmenting-path form.

Rectangular inputs are fine: with more drivers than riders every rider is
matched; with more riders than drivers every driver is. Pairs that must not
be matched (pickup too far away) carry ``FORBIDDEN`` and are dropped from the
result afterwards.
"""

from __future__ import annotations

import math

import numpy as np

FORBIDDEN = 1e9


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment as ``(row, col)`` pairs, one per row or column."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    c = cost.T if transposed else cost
    n, m = c.shape
    c = np.where(np.isfinite(c), c, FORBIDDEN).tolist()

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    owner = [0] * (m + 1)  # owner[j]: row (1-based) assigned to column j
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = owner[j0]
            row = c[i0 - 1]
            delta, j1 = math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1

    pairs = [(owner[j] - 1, j - 1) for j in range(1, m + 1) if owner[j]]
    if transposed:
        pairs = [(b, a) for a, b in pairs]
    return sorted(pairs)


def optimal_matches(cost: np.ndarray, limit: float) -> list[tuple[int, int]]:
    """Minimum total cost matching that never uses a pair above ``limit``.

    Pairs over the limit (or infinite) cost ``FORBIDDEN``, so the assignment
    first matches as many pairs as it can, then minimises their total.
    """
    cost = np.asarray(cost, dtype=np.float64)
    ok = np.isfinite(cost) & (cost <= limit)
    allowed = np.where(ok, cost, FORBIDDEN)
    return [(i, j) for i, j in hungarian(allowed) if ok[i, j]]


def greedy_matches(cost: np.ndarray, limit: float) -> list[tuple[int, int]]:
    """Riders in arrival order (columns) each take the nearest free driver (rows)."""
    cost = np.asarray(cost, dtype=np.float64)
    free = set(range(cost.shape[0]))
    pairs = []
    for j in range(cost.shape[1]):
        if not free:
            break
        best = min(free, key=lambda i: cost[i, j])
        if cost[best, j] <= limit:
            pairs.append((best, j))
            free.remove(best)
    return pairs
