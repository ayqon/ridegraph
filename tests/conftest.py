"""Shared fixtures: synthetic road networks with known structure."""

from __future__ import annotations

import random

import numpy as np
import pytest

from fastroute.graph import RoadNetwork
from fastroute.speeds import CLASS_INDEX


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


def grid_network(rows: int, cols: int, seed: int = 0, oneway_share: float = 0.2) -> RoadNetwork:
    """A street grid ~100 m per block with mixed road types and some one-way streets.

    Coordinates are real (around Holborn) so geometric heuristics behave as they
    would on the actual map.
    """
    rng = random.Random(seed)
    lat0, lon0 = 51.515, -0.12
    dlat = 100 / 111_320
    dlon = 100 / (111_320 * np.cos(np.radians(lat0)))
    coords = [(lat0 + r * dlat, lon0 + c * dlon) for r in range(rows) for c in range(cols)]
    classes = ["residential", "tertiary", "secondary", "primary"]
    src, dst, length, cls = [], [], [], []

    def add(u: int, v: int) -> None:
        seg = 100 * rng.uniform(0.8, 1.3)
        kind = CLASS_INDEX[rng.choice(classes)]
        directions = [(u, v), (v, u)]
        if rng.random() < oneway_share:
            directions = [rng.choice(directions)]
        for a, b in directions:
            src.append(a)
            dst.append(b)
            length.append(seg)
            cls.append(kind)

    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                add(u, u + 1)
            if r + 1 < rows:
                add(u, u + cols)

    return RoadNetwork(
        node_ids=np.arange(rows * cols, dtype=np.int64),
        lat=np.array([p[0] for p in coords]),
        lon=np.array([p[1] for p in coords]),
        src=np.array(src, dtype=np.int32),
        dst=np.array(dst, dtype=np.int32),
        length_m=np.array(length, dtype=np.float32),
        road_class=np.array(cls, dtype=np.int8),
        maxspeed_kph=np.full(len(src), np.nan, dtype=np.float32),
    )


@pytest.fixture
def grid() -> RoadNetwork:
    return grid_network(30, 30, seed=7)
