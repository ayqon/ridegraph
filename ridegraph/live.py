"""Live road disruptions from Transport for London, matched onto the graph.

TfL's open Unified API lists current roadworks, closures and incidents on
London's streets, each as a polyline with a severity and a closure state
(``/Road/all/Street/Disruption``). This module

1. fetches and parses that feed (standard library only, no key needed for
   light use),
2. *map-matches* each disrupted stretch onto graph edges: an edge is on the
   stretch when both its ends and its midpoint lie within ``MATCH_M`` metres
   of the polyline, and
3. turns the result into a time multiplier per edge: infinity for a closed
   road, a slowdown for the rest by severity.

The multipliers are applied to the time-of-day weights, and a Customizable
Contraction Hierarchy (``cch.py``) re-customises in milliseconds, so the
quickest-route queries stay instant with live closures in force.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .graph import RoadNetwork

TFL_STREET_DISRUPTIONS = "https://api.tfl.gov.uk/Road/all/Street/Disruption"
MATCH_M = 18.0

# Time multipliers by TfL severity, for stretches that stay open.
SEVERITY_FACTOR = {
    "severe": 3.0,
    "serious": 2.2,
    "moderate": 1.6,
    "minimal": 1.15,
}
DEFAULT_FACTOR = 1.4
CLOSED = math.inf


@dataclass(frozen=True)
class Disruption:
    street: str
    severity: str
    closure: str
    points: list[tuple[float, float]]  # (lat, lon)

    @property
    def closed(self) -> bool:
        c = self.closure.lower()
        return ("clos" in c or "full" in c) and "partial" not in c and "open" not in c

    @property
    def factor(self) -> float:
        if self.closed:
            return CLOSED
        f = SEVERITY_FACTOR.get(self.severity.lower(), DEFAULT_FACTOR)
        return max(f, 1.6) if "partial" in self.closure.lower() else f


def feed_url(now: datetime | None = None, hours: float = 1.0) -> str:
    now = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    fmt = "%Y-%m-%dT%H:%M:%S"
    end = now + timedelta(hours=hours)
    return f"{TFL_STREET_DISRUPTIONS}?startDate={now.strftime(fmt)}&endDate={end.strftime(fmt)}"


def fetch(timeout: float = 30.0) -> object:
    """Current street disruptions as parsed JSON."""
    req = urllib.request.Request(feed_url(), headers={"User-Agent": "ridegraph"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read())


def _points(line: object) -> list[tuple[float, float]]:
    """Polyline from a TfL lineString: JSON "[[lon,lat],...]" or WKT LINESTRING."""
    if isinstance(line, list):
        coords = line
    else:
        text = str(line).strip()
        if text.upper().startswith(("LINESTRING", "MULTILINESTRING")):
            body = text[text.index("(") :].replace("(", " ").replace(")", " ")
            coords = [[float(v) for v in pair.split()] for pair in body.split(",") if pair.strip()]
        else:
            try:
                coords = json.loads(text)
            except ValueError:
                return []
    while coords and isinstance(coords[0], list) and coords[0] and isinstance(coords[0][0], list):
        coords = [p for part in coords for p in part]  # flatten multi-lines
    out = []
    for p in coords:
        if isinstance(p, list) and len(p) >= 2:
            lon, lat = float(p[0]), float(p[1])
            out.append((lat, lon))
    return out


def parse(payload: object) -> list[Disruption]:
    """Every disrupted stretch in the feed, wherever it sits in the JSON."""
    found: list[Disruption] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "lineString" in node:
                pts = _points(node["lineString"])
                if len(pts) >= 2:
                    found.append(
                        Disruption(
                            street=str(
                                node.get("streetName") or node.get("location") or "a street"
                            ),
                            severity=str(node.get("severity") or ""),
                            closure=str(node.get("closure") or ""),
                            points=pts,
                        )
                    )
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _to_xy(lat, lon, lat0: float):
    kx = 111_320.0 * math.cos(math.radians(lat0))
    return np.asarray(lon) * kx, np.asarray(lat) * 110_540.0


def _distance_to_polyline(px, py, line_x, line_y):
    best = np.full(np.shape(px), np.inf)
    for i in range(len(line_x) - 1):
        ax, ay, bx, by = line_x[i], line_y[i], line_x[i + 1], line_y[i + 1]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        t = 0.0 if denom == 0 else np.clip(((px - ax) * dx + (py - ay) * dy) / denom, 0, 1)
        best = np.minimum(best, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def match(network: RoadNetwork, disruptions: list[Disruption]) -> tuple[np.ndarray, list[int]]:
    """Time multiplier per edge (1 where clear), and how many edges each stretch hit."""
    lat0 = float(np.mean(network.lat))
    nx, ny = _to_xy(network.lat, network.lon, lat0)
    src, dst = network.src, network.dst
    factor = np.ones(network.num_edges)
    hits = []
    for d in disruptions:
        lx, ly = _to_xy([p[0] for p in d.points], [p[1] for p in d.points], lat0)
        pad = MATCH_M + 1
        near = np.flatnonzero(
            (nx >= lx.min() - pad)
            & (nx <= lx.max() + pad)
            & (ny >= ly.min() - pad)
            & (ny <= ly.max() + pad)
        )
        close = np.zeros(network.num_nodes, dtype=bool)
        if len(near):
            close[near] = _distance_to_polyline(nx[near], ny[near], lx, ly) <= MATCH_M
        cand = np.flatnonzero(close[src] & close[dst] & (src != dst))
        if len(cand):
            mx = (nx[src[cand]] + nx[dst[cand]]) / 2
            my = (ny[src[cand]] + ny[dst[cand]]) / 2
            cand = cand[_distance_to_polyline(mx, my, lx, ly) <= MATCH_M]
            factor[cand] = np.maximum(factor[cand], d.factor)
        hits.append(len(cand))
    return factor, hits


def summary(disruptions: list[Disruption], hits: list[int]) -> dict[str, int]:
    matched = [d for d, h in zip(disruptions, hits, strict=True) if h]
    return {
        "stretches": len(disruptions),
        "on_this_map": len(matched),
        "closed": sum(1 for d in matched if d.closed),
        "slowed": sum(1 for d in matched if not d.closed),
    }
