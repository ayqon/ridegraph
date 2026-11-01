"""Common result type returned by every routing algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Route:
    """Outcome of a point-to-point query.

    ``settled`` counts nodes whose final distance the algorithm fixed. It is the
    honest measure of search effort, and the number the benchmarks compare,
    since wall-clock time in Python mostly measures the interpreter.
    """

    source: int
    target: int
    path: list[int] = field(default_factory=list)
    seconds: float = math.inf
    settled: int = 0
    algorithm: str = ""

    @property
    def found(self) -> bool:
        return math.isfinite(self.seconds)


def unpack_path(parent: dict[int, int] | list[int], source: int, target: int) -> list[int]:
    """Walk parent pointers back from ``target`` to ``source``."""
    path = [target]
    node = target
    while node != source:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path
