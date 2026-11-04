"""Shared fixtures: synthetic road networks with known structure."""

from __future__ import annotations

import numpy as np
import pytest

from ridegraph.graph import RoadNetwork
from ridegraph.speeds import CLASS_INDEX
from ridegraph.synthetic import grid_network

__all__ = ["grid_network", "make_network"]


def make_network(
    edges: list[tuple[int, int, float]] | list[tuple[int, int, float, str]],
    num_nodes: int | None = None,
    coords: list[tuple[float, float]] | None = None,
    road_class: str = "residential",
) -> RoadNetwork:
    """Network from ``(u, v, length_m[, road_class])`` directed edges."""
    n = num_nodes if num_nodes is not None else 1 + max(max(e[0], e[1]) for e in edges)
    if coords is None:
        coords = [(51.5 + i * 1e-4, -0.1) for i in range(n)]
    return RoadNetwork(
        node_ids=np.arange(n, dtype=np.int64),
        lat=np.array([c[0] for c in coords], dtype=np.float64),
        lon=np.array([c[1] for c in coords], dtype=np.float64),
        src=np.array([e[0] for e in edges], dtype=np.int32),
        dst=np.array([e[1] for e in edges], dtype=np.int32),
        length_m=np.array([e[2] for e in edges], dtype=np.float32),
        road_class=np.array(
            [CLASS_INDEX[e[3] if len(e) > 3 else road_class] for e in edges], dtype=np.int8
        ),
        maxspeed_kph=np.full(len(edges), np.nan, dtype=np.float32),
    )


@pytest.fixture
def grid() -> RoadNetwork:
    return grid_network(30, 30, seed=7)
