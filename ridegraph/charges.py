"""The London Congestion Charge: where it applies, when, and what it costs.

Driving inside central London's Congestion Charge zone during charging hours
costs a flat daily fee, however briefly you are inside. The boundary roads
themselves are free. The polygon below traces the boundary (Marylebone and
Euston Road, Pentonville Road, City Road, Old Street, Commercial Street, Tower
Bridge, New Kent Road, Kennington Lane, Vauxhall Bridge Road, Park Lane and
Edgware Road) closely enough for a city-scale demo, not for enforcement.

Figures from Transport for London, as of January 2026: £18 a day, charged
07:00 to 18:00 on weekdays and 12:00 to 18:00 at weekends and on bank holidays.
"""

from __future__ import annotations

import numpy as np

DAILY_CHARGE_GBP = 18.0
WEEKDAY_HOURS = (7.0, 18.0)
WEEKEND_HOURS = (12.0, 18.0)

# (lat, lon), clockwise from the Marylebone flyover.
ZONE: tuple[tuple[float, float], ...] = (
    (51.5203, -0.1695),  # Edgware Road / Marylebone Road
    (51.5226, -0.1571),  # Baker Street
    (51.5238, -0.1438),  # Great Portland Street
    (51.5257, -0.1358),  # Euston Square
    (51.5299, -0.1240),  # King's Cross
    (51.5318, -0.1058),  # Angel
    (51.5262, -0.0880),  # Old Street roundabout
    (51.5247, -0.0775),  # Shoreditch
    (51.5155, -0.0735),  # Commercial Street / Aldgate East
    (51.5105, -0.0730),  # Mansell Street / Tower Hill
    (51.5055, -0.0752),  # Tower Bridge
    (51.4955, -0.0830),  # Tower Bridge Road / Bricklayers Arms
    (51.4946, -0.1000),  # Elephant & Castle
    (51.4870, -0.1225),  # Kennington Lane / Vauxhall
    (51.4890, -0.1270),  # Vauxhall Bridge
    (51.4962, -0.1420),  # Vauxhall Bridge Road / Victoria
    (51.5027, -0.1527),  # Hyde Park Corner
    (51.5131, -0.1589),  # Marble Arch
)


def charging(hour: float | None, *, weekend: bool = False) -> bool:
    """Whether the charge applies at this time of day."""
    if hour is None:
        return False
    start, end = WEEKEND_HOURS if weekend else WEEKDAY_HOURS
    return start <= hour % 24 < end


def inside(lat: np.ndarray, lon: np.ndarray, polygon=ZONE) -> np.ndarray:
    """Point-in-polygon by ray casting, vectorised over points."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    result = np.zeros(lat.shape, dtype=bool)
    k = len(polygon)
    for i in range(k):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % k]
        crosses = (y1 > lat) != (y2 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        result ^= crosses & (lon < x_at)
    return result


def distance_to_boundary_m(lat: np.ndarray, lon: np.ndarray, polygon=ZONE) -> np.ndarray:
    """Metres from each point to the nearest edge of the polygon."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat0 = float(np.mean([p[0] for p in polygon]))
    kx = 111_320.0 * np.cos(np.radians(lat0))
    ky = 110_540.0
    px, py = lon * kx, lat * ky
    best = np.full(lat.shape, np.inf)
    k = len(polygon)
    for i in range(k):
        ay, ax = polygon[i][0] * ky, polygon[i][1] * kx
        by, bx = polygon[(i + 1) % k][0] * ky, polygon[(i + 1) % k][1] * kx
        dx, dy = bx - ax, by - ay
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        best = np.minimum(best, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


# Junctions this close to the line are treated as on the (free) boundary road.
BOUNDARY_MARGIN_M = 60.0


def zone_nodes(network) -> np.ndarray:
    """Boolean per junction: inside the zone, clear of the boundary road."""
    lat, lon = network.lat, network.lon
    return inside(lat, lon) & (distance_to_boundary_m(lat, lon) > BOUNDARY_MARGIN_M)
