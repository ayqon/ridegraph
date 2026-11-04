"""How realistic is the simulated London? A check against published speeds.

The learned ETA is trained and scored on trips timed in a hidden simulated
city (``eta.World``), because real trip data is not available to a personal
project. That makes one question unavoidable: are its speeds plausible for
central London? This module times random trips in the simulated city at
several times of day and sets the results beside independent published
figures:

* **Department for Transport**, *Travel time measures for local 'A' roads,
  2022* (tables CGN0503b and CGN0503d): average speed on local A roads in
  the City of London 7.7 mph and Camden 8.1 mph, the two slowest local
  authorities in England, and 14.4 mph across Greater London.
* **TomTom Traffic Index 2023**: London's city centre averaged about 10 mph,
  falling to 8.5 mph at peak hours.

The definitions differ (DfT measures A roads only, TomTom whole trips in its
own city-centre area), so the comparison is about the range, not a single
target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ch import ContractionHierarchy
from .graph import RoadNetwork, haversine_m
from .router import Router, Scenario

MPH = 1.609344  # km/h per mph


@dataclass(frozen=True)
class Reference:
    source: str
    what: str
    kmh: float


REFERENCES = (
    Reference("DfT 2022, CGN0503d", "City of London, local A roads", 7.7 * MPH),
    Reference("DfT 2022, CGN0503d", "Camden, local A roads", 8.1 * MPH),
    Reference("DfT 2022, CGN0503b", "Greater London, local A roads", 14.4 * MPH),
    Reference("TomTom Traffic Index 2023", "London city centre, all day", 10.0 * MPH),
    Reference("TomTom Traffic Index 2023", "London city centre, peak hours", 8.5 * MPH),
)


@dataclass(frozen=True)
class SpeedRow:
    hour: float
    simulated_kmh: float  # trips timed in the hidden world (what the ETA learns from)
    router_kmh: float  # the router's own estimate for the same trips
    mean_trip_km: float


def simulated_speeds(
    network: RoadNetwork,
    hours: tuple[float, ...] = (3.0, 8.0, 13.0, 17.5),
    trips: int = 300,
    *,
    min_km: float = 2.0,
    seed: int = 3,
) -> list[SpeedRow]:
    """Distance-weighted average speed of random trips at each hour (weekday)."""
    from .eta import RouteContext, World

    ctx = RouteContext(network)
    world = World(ctx, seed=0)
    rng = np.random.default_rng(seed)
    net = ctx.network
    rows = []
    for hour in hours:
        ch = ContractionHierarchy.build(Router(net, Scenario(hour=hour)).graph)
        times = ctx.edge_times(hour, False)
        observed, routed = 0.0, 0.0
        metres: list[float] = []
        attempts = 0
        while len(metres) < trips and attempts < trips * 20:
            attempts += 1
            a, b = int(rng.integers(net.num_nodes)), int(rng.integers(net.num_nodes))
            crow = haversine_m(net.lat[a], net.lon[a], net.lat[b], net.lon[b])
            if crow < min_km * 1000:
                continue
            route = ch.query(a, b)
            if not route.found:
                continue
            edges = ctx.path_edges(route.path, times)
            metres.append(float(ctx.length_m[edges].sum()))
            routed += float(times[edges].sum())
            observed += world.trip_seconds(route.path, edges, times, hour, False, rng)
        total = sum(metres)
        rows.append(
            SpeedRow(
                hour,
                3.6 * total / observed if observed else float("nan"),
                3.6 * total / routed if routed else float("nan"),
                total / max(len(metres), 1) / 1000,
            )
        )
    return rows


def markdown(rows: list[SpeedRow]) -> str:
    lines = [
        "| Weekday time | Simulated trips | Router's estimate | Mean trip |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        h, m = divmod(round(r.hour * 60), 60)
        lines.append(
            f"| {h:02d}:{m:02d} | {r.simulated_kmh:.1f} km/h | {r.router_kmh:.1f} km/h "
            f"| {r.mean_trip_km:.1f} km |"
        )
    lines += ["", "| Published figure | Source | Speed |", "|---|---|---:|"]
    for ref in REFERENCES:
        lines.append(f"| {ref.what} | {ref.source} | {ref.kmh:.1f} km/h |")
    return "\n".join(lines)
