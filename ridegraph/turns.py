"""Turn costs via an edge-expanded graph.

Why a different graph?
----------------------
In the plain road graph a node is an intersection, so the search cannot tell
*how* it arrived there: turning right across oncoming traffic and carrying
straight on look identical. The standard fix is to expand the graph so each
**node is a directed road segment** ("I have just driven u -> v") and each
**edge is a manoeuvre** at v ("then take v -> w"). A manoeuvre's cost is the
time to drive the next segment plus a penalty that depends on the turn.

Every algorithm in this package runs on the expanded graph unchanged, since
it is just another weighted directed graph. Contraction Hierarchies included.

Two extra node kinds make queries between intersections work: ``start(x)``
has edges to every segment leaving ``x`` (no turn, you are already facing
that way) and ``end(x)`` is reached free from every segment arriving at
``x``. A zero-cost edge ``start(x) -> end(x)`` handles trips that go nowhere.

Penalties
---------
Defaults are for the UK, where traffic drives on the left: a left turn is the
easy, near-side manoeuvre and a right turn crosses oncoming traffic. Only real
junctions (three or more distinct neighbouring nodes) charge for turning,
since a bend in a single road is not a decision. Traffic lights add an
average wait to every movement through them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import pairwise

import numpy as np

from .graph import EdgeCost, Graph, RoadNetwork, free_flow_cost


class Turn(IntEnum):
    NONE = 0  # start/end connectors and plain bends
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3
    SHARP_LEFT = 4
    SHARP_RIGHT = 5
    U_TURN = 6


@dataclass(frozen=True)
class TurnPenalties:
    """Seconds added per manoeuvre, on top of driving the next segment."""

    straight_s: float = 0.0
    near_side_s: float = 4.0  # left in the UK: no oncoming traffic to cross
    far_side_s: float = 10.0  # right in the UK: wait for a gap in oncoming traffic
    sharp_near_s: float = 8.0
    sharp_far_s: float = 15.0
    u_turn_s: float = 30.0
    traffic_signals_s: float = 12.0  # average wait at lights
    straight_within_deg: float = 30.0
    sharp_beyond_deg: float = 120.0
    drive_on_left: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.straight_within_deg < self.sharp_beyond_deg < 180:
            raise ValueError("need 0 < straight_within_deg < sharp_beyond_deg < 180")


UK_PENALTIES = TurnPenalties()

NO_PENALTIES = TurnPenalties(
    near_side_s=0.0,
    far_side_s=0.0,
    sharp_near_s=0.0,
    sharp_far_s=0.0,
    u_turn_s=0.0,
    traffic_signals_s=0.0,
)


def segment_bearings(network: RoadNetwork) -> np.ndarray:
    """Compass bearing of each directed segment in degrees, clockwise from north."""
    lat = np.radians(network.lat)
    lon = np.radians(network.lon)
    u, v = network.src, network.dst
    dx = (lon[v] - lon[u]) * np.cos((lat[u] + lat[v]) / 2)
    dy = lat[v] - lat[u]
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def junction_mask(network: RoadNetwork) -> np.ndarray:
    """True for nodes where three or more distinct roads meet."""
    a = np.minimum(network.src, network.dst).astype(np.int64)
    b = np.maximum(network.src, network.dst).astype(np.int64)
    pairs = np.unique(a * network.num_nodes + b)
    ends = np.concatenate([pairs // network.num_nodes, pairs % network.num_nodes])
    return np.bincount(ends, minlength=network.num_nodes) >= 3


def classify_turns(
    angle_deg: np.ndarray, is_reverse: np.ndarray, at_junction: np.ndarray, p: TurnPenalties
) -> np.ndarray:
    """Turn kind per manoeuvre. Positive angles turn clockwise (to the right)."""
    mag = np.abs(angle_deg)
    right = angle_deg > 0
    kind = np.full(len(angle_deg), Turn.STRAIGHT, dtype=np.int8)
    kind[(mag >= p.straight_within_deg) & right] = Turn.RIGHT
    kind[(mag >= p.straight_within_deg) & ~right] = Turn.LEFT
    kind[(mag >= p.sharp_beyond_deg) & right] = Turn.SHARP_RIGHT
    kind[(mag >= p.sharp_beyond_deg) & ~right] = Turn.SHARP_LEFT
    kind[~at_junction] = Turn.NONE  # a bend in the road, not a decision
    kind[is_reverse] = Turn.U_TURN
    return kind


def penalty_seconds(kind: np.ndarray, p: TurnPenalties) -> np.ndarray:
    near_left = p.drive_on_left
    table = np.zeros(len(Turn), dtype=np.float64)
    table[Turn.STRAIGHT] = p.straight_s
    table[Turn.LEFT] = p.near_side_s if near_left else p.far_side_s
    table[Turn.RIGHT] = p.far_side_s if near_left else p.near_side_s
    table[Turn.SHARP_LEFT] = p.sharp_near_s if near_left else p.sharp_far_s
    table[Turn.SHARP_RIGHT] = p.sharp_far_s if near_left else p.sharp_near_s
    table[Turn.U_TURN] = p.u_turn_s
    return table[kind]


class TurnGraph:
    """Edge-expanded graph of a ``RoadNetwork`` with turn and signal costs.

    Node ids: ``0..M-1`` are road segments (located at their end node),
    ``M + x`` is ``start(x)`` and ``M + N + x`` is ``end(x)``.
    """

    def __init__(self, network: RoadNetwork, penalties: TurnPenalties = UK_PENALTIES):
        self.network = network
        self.penalties = penalties
        n, m = network.num_nodes, network.num_edges
        src = network.src.astype(np.int64)
        dst = network.dst.astype(np.int64)

        # Segments leaving each node, via CSR over segments sorted by source.
        order = np.argsort(src, kind="stable")
        out_count = np.bincount(src, minlength=n)
        out_first = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(out_count, out=out_first[1:])

        # Manoeuvres: every segment i = (u -> v) followed by every segment j leaving v.
        per_seg = out_count[dst]
        total = int(per_seg.sum())
        from_seg = np.repeat(np.arange(m, dtype=np.int64), per_seg)
        offset_in_group = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(per_seg) - per_seg, per_seg
        )
        to_seg = order[np.repeat(out_first[dst], per_seg) + offset_in_group]

        # Turns the law forbids are simply never created.
        bans = network.banned_turns
        if bans is not None and len(bans):
            key = (src[from_seg] * n + dst[from_seg]) * n + dst[to_seg]
            banned_key = (bans[:, 0].astype(np.int64) * n + bans[:, 1]) * n + bans[:, 2]
            allowed = ~np.isin(key, banned_key)
            from_seg, to_seg = from_seg[allowed], to_seg[allowed]
            total = len(from_seg)

        bearing = segment_bearings(network)
        angle = (bearing[to_seg] - bearing[from_seg] + 180.0) % 360.0 - 180.0
        via = dst[from_seg]
        kind = classify_turns(
            angle, dst[to_seg] == src[from_seg], junction_mask(network)[via], penalties
        )
        penalty = penalty_seconds(kind, penalties)
        signals = network.traffic_signals
        assert signals is not None
        penalty = penalty + np.where(signals[via], penalties.traffic_signals_s, 0.0)

        start_node = m + np.arange(n, dtype=np.int64)
        end_node = m + n + np.arange(n, dtype=np.int64)
        seg_ids = np.arange(m, dtype=np.int64)

        # Edge groups: manoeuvres, start(x) -> segments leaving x,
        # segments -> end(their head), start(x) -> end(x).
        self.exp_src = np.concatenate([from_seg, start_node[src], seg_ids, start_node])
        self.exp_dst = np.concatenate([to_seg, seg_ids, end_node[dst], end_node])
        self.target_seg = np.concatenate([to_seg, seg_ids, np.full(m + n, -1, dtype=np.int64)])
        self.penalty = np.concatenate([penalty, np.zeros(2 * m + n)])
        self.turn_kind = np.concatenate([kind, np.zeros(2 * m + n, dtype=np.int8)])
        self.signal_at = np.concatenate([signals[via], np.zeros(2 * m + n, dtype=bool)])
        self.num_manoeuvres = total

        has_seg = self.target_seg >= 0
        length = np.where(has_seg, network.length_m[np.maximum(self.target_seg, 0)], 0.0)
        cls = network.road_class[np.maximum(self.target_seg, 0)]
        self.expanded = RoadNetwork(
            node_ids=np.arange(m + 2 * n, dtype=np.int64),
            lat=np.concatenate([network.lat[dst], network.lat, network.lat]),
            lon=np.concatenate([network.lon[dst], network.lon, network.lon]),
            src=self.exp_src.astype(np.int32),
            dst=self.exp_dst.astype(np.int32),
            length_m=length.astype(np.float32),
            road_class=cls.astype(np.int8),
            maxspeed_kph=np.full(len(self.exp_src), np.nan, dtype=np.float32),
        )

    @property
    def num_segments(self) -> int:
        return self.network.num_edges

    def start(self, intersection: int) -> int:
        return self.num_segments + intersection

    def end(self, intersection: int) -> int:
        return self.num_segments + self.network.num_nodes + intersection

    def graph(self, cost: EdgeCost = free_flow_cost) -> Graph:
        """Routing graph whose weights are segment times (under ``cost``) plus turn penalties."""
        seg_time = np.asarray(cost(self.network), dtype=np.float64)
        weights = (
            np.where(self.target_seg >= 0, seg_time[np.maximum(self.target_seg, 0)], 0.0)
            + self.penalty
        )
        return Graph(self.expanded, cost=lambda _net: weights)

    def intersections(self, path: list[int]) -> list[int]:
        """Map a path through the expanded graph back to intersections."""
        m, n = self.num_segments, self.network.num_nodes
        out: list[int] = []
        for node in path:
            x = int(self.network.dst[node]) if node < m else (node - m) % n
            if not out or out[-1] != x:
                out.append(x)
        return out

    def manoeuvres(self, path: list[int]) -> Counter[str]:
        """Count turns and traffic lights along a route through the expanded graph."""
        total = self.expanded.num_nodes
        keys = self.exp_src * total + self.exp_dst
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        counts: Counter[str] = Counter()
        for a, b in pairwise(path):
            pos = int(np.searchsorted(sorted_keys, a * total + b))
            if pos >= len(sorted_keys) or sorted_keys[pos] != a * total + b:
                raise KeyError(f"no manoeuvre {a} -> {b}")
            e = int(order[pos])
            kind = Turn(int(self.turn_kind[e]))
            if kind not in (Turn.NONE, Turn.STRAIGHT):
                counts[kind.name.lower().replace("_", " ")] += 1
            if self.signal_at[e]:
                counts["traffic lights"] += 1
        return counts
