"""Benchmark every algorithm on the same random queries.

Two numbers matter per algorithm:

* **Nodes settled**, the work the algorithm does, independent of language. This
  is the fair comparison.
* **Query time**, which in pure Python mostly measures interpreter overhead per
  node; useful for relative comparison, not as an absolute figure.

Every query's answer is cross-checked against Dijkstra, so a benchmark run
doubles as a correctness check on whatever network it is given.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from .astar import astar
from .bidirectional import bidirectional_dijkstra
from .ch import ContractionHierarchy
from .dijkstra import dijkstra
from .graph import Graph
from .result import Route


@dataclass(frozen=True)
class BenchRow:
    algorithm: str
    mean_settled: float
    median_ms: float
    mean_ms: float
    speedup_settled: float  # vs Dijkstra
    speedup_time: float


def run(
    graph: Graph,
    ch: ContractionHierarchy | None = None,
    *,
    queries: int = 200,
    seed: int = 0,
    pairs: list[tuple[int, int]] | None = None,
) -> list[BenchRow]:
    """Compare algorithms on ``pairs`` of graph nodes (random pairs by default)."""
    algorithms: dict[str, Callable[[int, int], Route]] = {
        "Dijkstra": lambda s, t: dijkstra(graph, s, t),
        "Bidirectional Dijkstra": lambda s, t: bidirectional_dijkstra(graph, s, t),
        "A*": lambda s, t: astar(graph, s, t),
    }
    if ch is not None:
        algorithms["Contraction Hierarchies"] = ch.query

    if pairs is None:
        rng = random.Random(seed)
        pairs = [
            (rng.randrange(graph.num_nodes), rng.randrange(graph.num_nodes)) for _ in range(queries)
        ]

    settled: dict[str, list[int]] = {name: [] for name in algorithms}
    timings: dict[str, list[float]] = {name: [] for name in algorithms}
    for s, t in pairs:
        reference: Route | None = None
        for name, fn in algorithms.items():
            started = time.perf_counter()
            route = fn(s, t)
            timings[name].append((time.perf_counter() - started) * 1000)
            settled[name].append(route.settled)
            if reference is None:
                reference = route
            elif abs(route.seconds - reference.seconds) > 1e-6 * max(1.0, reference.seconds):
                raise AssertionError(
                    f"{name} disagrees with Dijkstra on {s} -> {t}: "
                    f"{route.seconds} vs {reference.seconds}"
                )

    base_settled = statistics.fmean(settled["Dijkstra"])
    base_ms = statistics.fmean(timings["Dijkstra"])
    rows = []
    for name in algorithms:
        mean_settled = statistics.fmean(settled[name])
        mean_ms = statistics.fmean(timings[name])
        rows.append(
            BenchRow(
                name,
                mean_settled,
                statistics.median(timings[name]),
                mean_ms,
                base_settled / mean_settled,
                base_ms / mean_ms,
            )
        )
    return rows


def to_markdown(rows: list[BenchRow]) -> str:
    lines = [
        "| Algorithm | Nodes settled (mean) | vs Dijkstra | Query time (median) | vs Dijkstra |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.algorithm} | {r.mean_settled:,.0f} | {r.speedup_settled:.1f}x "
            f"| {r.median_ms:.2f} ms | {r.speedup_time:.1f}x |"
        )
    return "\n".join(lines)
