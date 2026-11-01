"""Synthetic road networks for tests and benchmarks without map data."""

from __future__ import annotations

import random

import numpy as np

from .graph import RoadNetwork, haversine_m
from .speeds import CLASS_INDEX


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
        # Roads bend, so a segment is at least as long as the straight line
        # between its ends; never shorter, or geometric heuristics break.
        seg = haversine_m(*coords[u], *coords[v]) * rng.uniform(1.0, 1.3)
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
