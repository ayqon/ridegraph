"""Time-of-day traffic: the same road, a different speed at 3am and 8am.

Production systems estimate live speeds per road segment from the GPS traces
of vehicles currently driving on them, then fall back to historical speeds
for that segment at that time of week. This module models the historical
layer with two ingredients:

* a **congestion curve** per day type, giving how congested the city is at
  each hour (0 = empty roads, 1 = the worst peak), and
* a **sensitivity** per road class, giving how much that kind of road slows
  down at the worst peak. Arterials clog badly; residential streets less so,
  since most through-traffic avoids them.

    speed(edge, hour) = free_flow_speed(edge) * (1 - sensitivity(class) * congestion(hour))

The curves are illustrative, shaped after the familiar London pattern
(sharp weekday peaks around 8:00 and 17:30, a busy middle of the day, a
flatter midday hump at weekends), not fitted to measured data. Swapping in
real per-segment speed data would replace this module without touching the
routing code, which only ever sees edge weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .graph import RoadNetwork, free_flow_speeds_kph
from .speeds import ROAD_CLASSES

# Congestion at each whole hour 00:00..23:00; interpolated linearly between hours.
WEEKDAY_CONGESTION: tuple[float, ...] = (
    0.05, 0.03, 0.02, 0.02, 0.03, 0.08,  # 00-05
    0.25, 0.70, 1.00, 0.80, 0.55, 0.55,  # 06-11
    0.60, 0.60, 0.60, 0.70, 0.85, 1.00,  # 12-17
    0.85, 0.55, 0.35, 0.25, 0.15, 0.08,  # 18-23
)  # fmt: skip
WEEKEND_CONGESTION: tuple[float, ...] = (
    0.10, 0.08, 0.05, 0.03, 0.03, 0.03,  # 00-05
    0.05, 0.10, 0.20, 0.35, 0.50, 0.60,  # 06-11
    0.65, 0.65, 0.60, 0.55, 0.55, 0.50,  # 12-17
    0.45, 0.40, 0.30, 0.25, 0.20, 0.15,  # 18-23
)  # fmt: skip

# Fraction of free-flow speed lost at the worst peak, per road class.
PEAK_SENSITIVITY: dict[str, float] = {
    "motorway": 0.60,
    "motorway_link": 0.55,
    "trunk": 0.55,
    "trunk_link": 0.50,
    "primary": 0.50,
    "primary_link": 0.45,
    "secondary": 0.45,
    "secondary_link": 0.40,
    "tertiary": 0.35,
    "tertiary_link": 0.30,
    "unclassified": 0.25,
    "residential": 0.20,
    "living_street": 0.10,
    "service": 0.10,
}


@dataclass(frozen=True)
class TrafficProfile:
    weekday: tuple[float, ...] = WEEKDAY_CONGESTION
    weekend: tuple[float, ...] = WEEKEND_CONGESTION
    sensitivity: tuple[float, ...] = tuple(PEAK_SENSITIVITY[c] for c in ROAD_CLASSES)

    def __post_init__(self) -> None:
        for curve in (self.weekday, self.weekend):
            if len(curve) != 24 or not all(0.0 <= c <= 1.0 for c in curve):
                raise ValueError("congestion curves need 24 hourly values in [0, 1]")
        if len(self.sensitivity) != len(ROAD_CLASSES) or not all(
            0.0 <= s < 1.0 for s in self.sensitivity
        ):
            raise ValueError("need one sensitivity in [0, 1) per road class")

    def congestion(self, hour: float, *, weekend: bool = False) -> float:
        """City-wide congestion in [0, 1] at a (fractional) hour, wrapping at midnight."""
        curve = self.weekend if weekend else self.weekday
        hour %= 24.0
        lower = int(hour)
        frac = hour - lower
        return curve[lower] * (1 - frac) + curve[(lower + 1) % 24] * frac

    def speed_factors(self, hour: float, *, weekend: bool = False) -> np.ndarray:
        """Multiplier on free-flow speed for each road class (indexed like ``ROAD_CLASSES``)."""
        level = self.congestion(hour, weekend=weekend)
        return 1.0 - np.asarray(self.sensitivity) * level


DEFAULT_PROFILE = TrafficProfile()


def traffic_cost(
    hour: float, *, weekend: bool = False, profile: TrafficProfile = DEFAULT_PROFILE
) -> Callable[[RoadNetwork], np.ndarray]:
    """Edge-cost function for ``Graph``: seconds per edge at a given time of day."""

    def cost(network: RoadNetwork) -> np.ndarray:
        factors = profile.speed_factors(hour, weekend=weekend)[network.road_class]
        speeds_mps = free_flow_speeds_kph(network) * factors / 3.6
        return network.length_m.astype(np.float64) / speeds_mps

    return cost


def parse_time(text: str) -> float:
    """``"8"``, ``"8.5"`` or ``"08:30"`` -> fractional hour."""
    text = text.strip()
    if ":" in text:
        hh, mm = text.split(":", 1)
        hour = int(hh) + int(mm) / 60
    else:
        hour = float(text)
    if not 0 <= hour < 24:
        raise ValueError(f"time must be within 00:00-23:59, got {text!r}")
    return hour
