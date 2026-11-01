"""Travel-time model: turns raw road attributes into seconds.

Routing for "quickest" rather than "shortest" hinges on one idea: an edge's
weight is how long it takes to drive, not how long it is. This module owns
that conversion so the routing code never has to know about road types.

Free-flow speeds below are deliberately conservative urban values for London.
A posted limit is an upper bound, not a typical speed, so when OSM provides
``maxspeed`` we scale it down rather than taking it at face value.
"""

from __future__ import annotations

import re

# Road classes we route on, in a fixed order so they can be stored as small ints.
ROAD_CLASSES: tuple[str, ...] = (
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
)
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(ROAD_CLASSES)}

# Typical free-flow speed (km/h) per class when no usable limit is tagged.
DEFAULT_SPEED_KPH: dict[str, float] = {
    "motorway": 90.0,
    "motorway_link": 55.0,
    "trunk": 50.0,
    "trunk_link": 35.0,
    "primary": 32.0,
    "primary_link": 25.0,
    "secondary": 28.0,
    "secondary_link": 22.0,
    "tertiary": 25.0,
    "tertiary_link": 20.0,
    "unclassified": 22.0,
    "residential": 20.0,
    "living_street": 10.0,
    "service": 12.0,
}

# Share of the posted limit drivers actually sustain between junctions in a city.
LIMIT_REALISATION = 0.8

MPH_TO_KPH = 1.609344
_SPEED_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?\s*$", re.IGNORECASE)
# Named UK limits that sometimes appear instead of a number.
_NAMED_LIMITS_KPH: dict[str, float] = {
    "gb:nsl_single": 60 * MPH_TO_KPH,
    "gb:nsl_dual": 70 * MPH_TO_KPH,
    "gb:motorway": 70 * MPH_TO_KPH,
    "uk:nsl_single": 60 * MPH_TO_KPH,
    "uk:nsl_dual": 70 * MPH_TO_KPH,
    "uk:motorway": 70 * MPH_TO_KPH,
}


def parse_maxspeed(raw: str | None) -> float | None:
    """Parse an OSM ``maxspeed`` value into km/h.

    Bare numbers are km/h per the OSM convention. Multi-valued tags like
    ``"30 mph; 20 mph"`` take the lowest value, the safe choice for routing.
    Returns ``None`` for anything unparseable (``"signals"``, ``"none"``...).
    """
    if not raw:
        return None
    values: list[float] = []
    for part in raw.split(";"):
        part = part.strip()
        named = _NAMED_LIMITS_KPH.get(part.lower())
        if named is not None:
            values.append(named)
            continue
        match = _SPEED_RE.match(part)
        if not match:
            continue
        number = float(match.group(1))
        unit = (match.group(2) or "km/h").lower()
        values.append(number * MPH_TO_KPH if unit == "mph" else number)
    values = [v for v in values if v > 0]
    return min(values) if values else None


def free_flow_speed_kph(road_class: str, maxspeed_kph: float | None) -> float:
    """Expected speed on a road segment with no traffic."""
    default = DEFAULT_SPEED_KPH[road_class]
    if maxspeed_kph is None:
        return default
    # A tagged limit tells us about the road, but a 30 mph sign in Soho does not
    # mean anyone does 30: take the share of the limit drivers actually realise.
    return max(5.0, maxspeed_kph * LIMIT_REALISATION)


def travel_time_s(length_m: float, speed_kph: float) -> float:
    """Seconds to cover ``length_m`` at ``speed_kph``."""
    return length_m / (speed_kph / 3.6)
