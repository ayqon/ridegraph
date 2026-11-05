"""Turn OpenStreetMap data into a ``RoadNetwork``.

The pipeline is small but each step matters for route quality:

1. Keep only ways a car may drive on (no footpaths, private roads, car park
   aisles or driveways).
2. Respect direction: ``oneway`` tags, roundabouts and motorways.
3. Split every way into one edge per consecutive node pair, measuring each.
4. Keep only the largest strongly connected component. Clipping a city to a
   bounding box leaves stubs you can drive into but not out of; routing from
   or to them would fail, so they are dropped.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

import numpy as np

from .graph import RoadNetwork, haversine_m
from .speeds import CLASS_INDEX, parse_maxspeed

_NO_ACCESS = {"no", "private"}
_EXCLUDED_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}
_ONEWAY_FORWARD = {"yes", "true", "1"}
_ONEWAY_BACKWARD = {"-1", "reverse"}
_ONEWAY_SKIP = {"reversible", "alternating"}
_IMPLIED_ONEWAY_CLASSES = {"motorway", "motorway_link"}
_ROUNDABOUT = {"roundabout", "circular"}


def is_drivable(tags: Mapping[str, str]) -> bool:
    if tags.get("highway") not in CLASS_INDEX or tags.get("area") == "yes":
        return False
    if tags.get("access") in _NO_ACCESS or tags.get("motor_vehicle") in _NO_ACCESS:
        return False
    if tags.get("motorcar") in _NO_ACCESS:
        return False
    if tags.get("highway") == "service" and tags.get("service") in _EXCLUDED_SERVICE:
        return False
    return tags.get("oneway") not in _ONEWAY_SKIP


def has_traffic_signals(tags: Mapping[str, str]) -> bool:
    """Lights a car must stop at: junction signals or signalled pedestrian crossings."""
    return tags.get("highway") == "traffic_signals" or tags.get("crossing") == "traffic_signals"


def directions(tags: Mapping[str, str]) -> tuple[bool, bool]:
    """``(forward, backward)``: which ways along the node list a car may travel."""
    oneway = tags.get("oneway", "").lower()
    if oneway in _ONEWAY_BACKWARD:
        return False, True
    if oneway in _ONEWAY_FORWARD:
        return True, False
    if oneway == "no":
        return True, True
    if tags.get("junction") in _ROUNDABOUT or tags.get("highway") in _IMPLIED_ONEWAY_CLASSES:
        return True, False
    return True, True


def network_from_overpass(
    payload: Mapping[str, Any], *, largest_component: bool = True
) -> RoadNetwork:
    """Build a ``RoadNetwork`` from an Overpass JSON response."""
    elements: Iterable[Mapping[str, Any]] = payload.get("elements", [])
    coords: dict[int, tuple[float, float]] = {}
    signals: set[int] = set()
    ways: list[Mapping[str, Any]] = []
    restrictions: list[Mapping[str, Any]] = []
    for el in elements:
        if el.get("type") == "node":
            coords[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
            if has_traffic_signals(el.get("tags", {})):
                signals.add(int(el["id"]))
        elif el.get("type") == "way" and is_drivable(el.get("tags", {})):
            ways.append(el)
        elif el.get("type") == "relation" and el.get("tags", {}).get("type") == "restriction":
            restrictions.append(el)

    index: dict[int, int] = {}
    src: list[int] = []
    dst: list[int] = []
    length: list[float] = []
    road_class: list[int] = []
    maxspeed: list[float] = []

    def idx(osm_id: int) -> int:
        i = index.get(osm_id)
        if i is None:
            i = index[osm_id] = len(index)
        return i

    for way in ways:
        tags = way.get("tags", {})
        refs = [int(r) for r in way.get("nodes", []) if int(r) in coords]
        if len(refs) < 2:
            continue
        forward, backward = directions(tags)
        cls = CLASS_INDEX[tags["highway"]]
        limit = parse_maxspeed(tags.get("maxspeed"))
        limit_value = math.nan if limit is None else limit
        for a, b in pairwise(refs):
            if a == b:
                continue
            seg = haversine_m(*coords[a], *coords[b])
            ia, ib = idx(a), idx(b)
            for u, v, ok in ((ia, ib, forward), (ib, ia, backward)):
                if ok:
                    src.append(u)
                    dst.append(v)
                    length.append(seg)
                    road_class.append(cls)
                    maxspeed.append(limit_value)

    way_nodes = {
        int(w["id"]): [int(r) for r in w.get("nodes", []) if int(r) in coords] for w in ways
    }
    banned = banned_turns(restrictions, way_nodes, index, src, dst)
    ids = np.fromiter(index.keys(), dtype=np.int64, count=len(index))
    keys = list(index)
    lat = np.array([coords[i][0] for i in keys], dtype=np.float64)
    lon = np.array([coords[i][1] for i in keys], dtype=np.float64)
    network = RoadNetwork(
        node_ids=ids,
        lat=lat,
        lon=lon,
        src=np.array(src, dtype=np.int32),
        dst=np.array(dst, dtype=np.int32),
        length_m=np.array(length, dtype=np.float32),
        road_class=np.array(road_class, dtype=np.int8),
        maxspeed_kph=np.array(maxspeed, dtype=np.float32),
        traffic_signals=np.fromiter(
            (i in signals for i in ids.tolist()), dtype=bool, count=len(ids)
        ),
        banned_turns=banned,
    )
    return keep_largest_scc(network) if largest_component else network


def restriction_kind(tags: Mapping[str, str]) -> str | None:
    """The restriction that applies to cars, e.g. ``"no_right_turn"``, or None.

    Restrictions that exempt cars, or apply only to other vehicles or only at
    certain times, are left out.
    """
    if "motorcar" in tags.get("except", "").split(";"):
        return None
    kind = tags.get("restriction:motorcar") or tags.get("restriction")
    if not kind or not kind.startswith(("no_", "only_")):
        return None
    return kind


def banned_turns(
    relations: Iterable[Mapping[str, Any]],
    way_nodes: Mapping[int, list[int]],
    index: Mapping[int, int],
    src: list[int],
    dst: list[int],
) -> np.ndarray:
    """Banned (from, via, to) node-index triples from OSM restriction relations.

    Only restrictions whose ``via`` is a single junction are read; those that
    run through a stretch of road ("via ways") are rare and skipped. The
    ``from`` and ``to`` ways are reduced to the node next to the junction on
    each, which is how the turn-aware graph identifies a manoeuvre.
    """
    out: dict[int, set[int]] = {}
    for u, v in zip(src, dst, strict=True):
        out.setdefault(u, set()).add(v)

    def beside(way: int, via: int) -> list[int]:
        nodes = way_nodes.get(way, [])
        found = []
        for i, node in enumerate(nodes):
            if node != via:
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(nodes) and nodes[j] in index and nodes[j] != via:
                    found.append(index[nodes[j]])
        return found

    rows: set[tuple[int, int, int]] = set()
    for rel in relations:
        kind = restriction_kind(rel.get("tags", {}))
        if kind is None:
            continue
        members = rel.get("members", [])
        froms = [
            int(m["ref"]) for m in members if m.get("role") == "from" and m.get("type") == "way"
        ]
        tos = [int(m["ref"]) for m in members if m.get("role") == "to" and m.get("type") == "way"]
        vias = [m for m in members if m.get("role") == "via"]
        if len(froms) != 1 or len(tos) != 1 or len(vias) != 1 or vias[0].get("type") != "node":
            continue
        via_osm = int(vias[0]["ref"])
        if via_osm not in index:
            continue
        via = index[via_osm]
        sources = beside(froms[0], via_osm)
        targets = set(beside(tos[0], via_osm))
        if not sources or not targets:
            continue
        # "no_*" bans the named turn; "only_*" bans every other way out.
        banned_to = targets if kind.startswith("no_") else out.get(via, set()) - targets
        for a in sources:
            for b in banned_to:
                rows.add((a, via, b))
    if not rows:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(sorted(rows), dtype=np.int32)


def strongly_connected_components(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Component label per node (iterative Kosaraju, safe for large graphs)."""

    def csr(a: np.ndarray, b: np.ndarray) -> tuple[list[int], list[int]]:
        order = np.argsort(a, kind="stable")
        first = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(a, minlength=num_nodes), out=first[1:])
        return first.tolist(), b[order].tolist()

    f_first, f_head = csr(src, dst)
    r_first, r_head = csr(dst, src)

    # Pass 1: order nodes by DFS finish time on the forward graph.
    visited = [False] * num_nodes
    finish: list[int] = []
    for root in range(num_nodes):
        if visited[root]:
            continue
        visited[root] = True
        stack = [(root, f_first[root])]
        while stack:
            u, e = stack[-1]
            if e < f_first[u + 1]:
                stack[-1] = (u, e + 1)
                v = f_head[e]
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, f_first[v]))
            else:
                stack.pop()
                finish.append(u)

    # Pass 2: sweep the reverse graph in decreasing finish time.
    label = [-1] * num_nodes
    comp = 0
    for root in reversed(finish):
        if label[root] >= 0:
            continue
        label[root] = comp
        stack2 = [root]
        while stack2:
            u = stack2.pop()
            for e in range(r_first[u], r_first[u + 1]):
                v = r_head[e]
                if label[v] < 0:
                    label[v] = comp
                    stack2.append(v)
        comp += 1
    return np.array(label, dtype=np.int64)


def keep_largest_scc(network: RoadNetwork) -> RoadNetwork:
    """Restrict the network to its largest strongly connected component."""
    if network.num_nodes == 0:
        return network
    labels = strongly_connected_components(network.num_nodes, network.src, network.dst)
    biggest = int(np.bincount(labels).argmax())
    keep = labels == biggest
    remap = np.full(network.num_nodes, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    edge_keep = keep[network.src] & keep[network.dst]
    return RoadNetwork(
        node_ids=network.node_ids[keep],
        lat=network.lat[keep],
        lon=network.lon[keep],
        src=remap[network.src[edge_keep]].astype(np.int32),
        dst=remap[network.dst[edge_keep]].astype(np.int32),
        length_m=network.length_m[edge_keep],
        road_class=network.road_class[edge_keep],
        maxspeed_kph=network.maxspeed_kph[edge_keep],
        traffic_signals=None if network.traffic_signals is None else network.traffic_signals[keep],
        banned_turns=_remap_bans(network.banned_turns, keep, remap),
    )


def _remap_bans(bans: np.ndarray | None, keep: np.ndarray, remap: np.ndarray) -> np.ndarray | None:
    """Carry banned turns over to a smaller network, dropping any that leave it."""
    if bans is None or not len(bans):
        return bans
    inside = keep[bans].all(axis=1)
    return remap[bans[inside]].astype(np.int32)
